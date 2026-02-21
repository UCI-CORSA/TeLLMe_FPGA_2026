/*
 * BSD 3-Clause License
 * 
 * Copyright (c) 2024-2026, Zhiheng Chen
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef RMS_NORM_HPP
#define RMS_NORM_HPP
#include "../config/config.hpp" 
#include "../config/macro.hpp"

using namespace hls;

qkv_t max_bram_block(qkv_block_t src, qkv_t cur_max){
    #pragma HLS INLINE
    qkv_t m1 = 0;
    qkv_t m2 = 0;

    for(int i = 0; i < HIDDEN_BLOCK_SIZE; i+=2){
        #pragma HLS UNROLL
        qkv_t tmp1 = hls::abs(src[i]);
        qkv_t tmp2 = hls::abs(src[i + 1]);
        m1 = (m1 < tmp1) ? tmp1 : m1;
        m2 = (m2 < tmp2) ? tmp2 : m2;
    }
    m1 = (m1 < cur_max) ? cur_max : m1;
    return (m1 < m2) ? m2 : m1 ;
}

void read_qkv_block(
    qkv_block_t qkv_in[],
    stream<qkv_block_t> &qkv_stream,
    int in_dim_block,
    int token_length,
    int kv_cache_length
){ 
    for(int i = 0; i < token_length * in_dim_block ; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1536 max= 1536 * 128
        #pragma HLS PIPELINE
        qkv_stream.write(qkv_in[i + kv_cache_length * NUM_HIDDEN_BLOCKS]);
    }
}

void write_qkv_block(
    qkv_block_t qkv_out[],
    stream<qkv_block_t> &qkv_stream,
    int out_dim_block,
    int token_length,
    int kv_cache_length
){ 
    for(int i = 0; i < token_length * out_dim_block; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1536 max=1536 * 128
        #pragma HLS PIPELINE
        qkv_out[i + kv_cache_length * NUM_HIDDEN_BLOCKS] = qkv_stream.read();
    }
}

void rmsnorm_accumulate(unsigned int token_length, unsigned int block_length, stream<qkv_block_t> & in_stream,  stream<qkv_block_t>  & x_stream, stream<l2norm_t> & l2norm_stream)
{
    #pragma HLS inline off

    unsigned int hidden_size;
    l2norm_t hidden_size_inv;

    if(block_length==256) 
    {
        hidden_size = 4096;
        hidden_size_inv = (l2norm_t)0.000244140625;
    }
    else 
    {
        hidden_size = 1536;
        hidden_size_inv = (l2norm_t)0.000651041667;
    }

    for(int t = 0; t < token_length; t++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=128 

        l2norm_t l2norm_[2] = {0};

        accumulate_loop_block:for(unsigned int i = 0; i < NUM_HIDDEN_BLOCKS_FFN; i++)
        {
            #pragma HLS pipeline
            if(i < block_length)
            {
                qkv_block_t in1_block;
                in1_block = in_stream.read();
                x_stream.write(in1_block);

                l2norm_t partial_sum_sq = 0.0;
                #pragma HLS BIND_OP variable=partial_sum_sq op=fadd impl=fulldsp

                accumulate_loop_vector:for(unsigned int j = 0; j < HIDDEN_BLOCK_SIZE; j++ )
                {
                    l2norm_t square = (l2norm_t)(in1_block[j]) * (l2norm_t)(in1_block[j]);
                    partial_sum_sq += square;
                }
                l2norm_[i % 2] = l2norm_[i % 2] + partial_sum_sq;
            } else {
                break;
            }
        }
        l2norm_[1] = (l2norm_[1] + l2norm_[0]) * hidden_size_inv;
        l2norm_[1] += norm_eps;
        #pragma HLS BIND_OP variable=l2norm_ op=frsqrt impl=fulldsp

        l2norm_t l2norm = hls::rsqrt(l2norm_[1]);
        l2norm_stream.write(l2norm);
    }
}

void rmsnorm_output(
    unsigned int token_length,
    unsigned int block_length,
    hls::stream<qkv_block_t>& max_stream,
    hls::stream<qkv_block_t>& x_stream,
    hls::stream<l2norm_t> & l2norm_stream,
    wt_norm_t rms_weight[NUM_HIDDEN_BLOCKS * 3 * HIDDEN_BLOCK_SIZE]
)
{

    for(int t = 0; t < token_length; t++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=128
        l2norm_t l2norm = l2norm_stream.read();
        qkv_t l2norm_inv  = (qkv_t)(l2norm);

        output_loop_block:for(unsigned int i = 0; i < NUM_HIDDEN_BLOCKS_FFN; i++)
        {
            #pragma HLS ALLOCATION operation instances=hmul limit=24
            #pragma HLS pipeline
            if(i < block_length)
            {
                qkv_block_t x_block = x_stream.read();
                qkv_block_t x_block_tmp = x_block * l2norm_inv;

                wt_norm_block_t weight_block;
                #pragma HLS ARRAY_PARTITION variable=weight_block complete dim=1
                for (int j = 0; j < HIDDEN_BLOCK_SIZE; j++) {
                    #pragma HLS UNROLL
                    weight_block[j] = rms_weight[i * HIDDEN_BLOCK_SIZE + j];
                }

                qkv_block_t x_block_tmp1 = x_block_tmp * weight_block;

                max_stream << x_block_tmp1;
            }
            else {
                break;
            }
        }
    }
}

void find_max_block_stream(stream<qkv_block_t> & input_stream, stream<qkv_block_t> & output_stream, unsigned int token_length, unsigned int block_length, qkv_t max[]){
    qkv_block_t tmp;
    #pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

    for(int i = 0; i < token_length; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=128 
        qkv_t cur_max = 0;

        for(int j = 0; j < NUM_HIDDEN_BLOCKS_FFN; j++){ 
            if(j < block_length){
                tmp = input_stream.read();
                output_stream.write(tmp);
                cur_max = max_bram_block(tmp, cur_max);
            } else {
                break;
            }
        }
        max[i] = cur_max;
    }
}

void compute_norm(
    qkv_block_t* in1, 
    qkv_block_t* out,
    wt_norm_t rms_weight[NUM_HIDDEN_BLOCKS * 3 * HIDDEN_BLOCK_SIZE],
    qkv_t max_array[2048],
    int token_length,
    int block_length,
    int kv_cache_length
)
{
    #pragma HLS dataflow

    #pragma HLS ARRAY_PARTITION variable=rms_weight cyclic factor=16 dim=1
    #pragma HLS interface bram port=max_array

    hls::stream<qkv_block_t> max_stream; 
    hls::stream<qkv_block_t> dram_stream; 
    hls::stream<qkv_block_t> in_stream; 
    hls::stream<qkv_block_t> x_stream; 
    #pragma HLS STREAM variable=x_stream depth=NUM_HIDDEN_BLOCKS_FFN

    hls::stream<l2norm_t> l2norm_stream; 

    read_qkv_block(in1, in_stream, block_length, token_length, kv_cache_length);
    rmsnorm_accumulate(token_length, block_length, in_stream,  x_stream, l2norm_stream);
    rmsnorm_output(token_length, block_length, max_stream, x_stream, l2norm_stream, rms_weight);
    find_max_block_stream(max_stream, dram_stream, token_length, block_length, max_array);
    write_qkv_block(out, dram_stream, block_length, token_length, kv_cache_length);
}

#endif  