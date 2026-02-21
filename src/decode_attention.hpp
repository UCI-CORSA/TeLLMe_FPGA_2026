/*
 * BSD 3-Clause License
 * 
 * Copyright (c) 2024-2026, Yifan Zhang
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

#ifndef DECODING_ATTENTION_V5_HPP
#define DECODING_ATTENTION_V5_HPP
#include <cassert>
#include "../config/config.hpp"
#include "../config/macro.hpp"

void read_q(
    hls::stream<qkv_block_t>& q_stream,
    qkv_block_t on_chip_src[],
    int kv_cache_length
){

    for(int dim = 0; dim < HIDDEN_SIZE; dim += HIDDEN_BLOCK_SIZE){

            #pragma HLS pipeline
            int dim_block =  dim / HIDDEN_BLOCK_SIZE;
                q_stream << on_chip_src[dim_block + kv_cache_length * NUM_HIDDEN_BLOCKS];

    }

} 

void read_kv(
    hls::stream<qkv_block_t>& kv_stream,
    qkv_block_t src[],  
    int token_length
){

        read_kv_token:for(int j = 0; j < token_length; j++){ 
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TOKEN_LENGTH 
        read_kv_blk:for(int dim = 0; dim < HIDDEN_SIZE; dim += HIDDEN_BLOCK_SIZE){
            #pragma HLS PIPELINE

                int dim_block =  dim / HIDDEN_BLOCK_SIZE;
                kv_stream << src[j * NUM_HIDDEN_BLOCKS + dim_block];

        }
    }
}

void write_score(

    hls::stream<head_block_t> & score_stream,
    head_block_t dst[],   
    qkv_t softmax_recip[NUM_ATT_HEADS],
    qkv_t softmax_bias[NUM_ATT_HEADS],
    int token_length
){
    for(int i = 0; i < token_length; i++){
        head_block_t tmp = score_stream.read();
        head_block_t tmp2;
        for(int head = 0; head < NUM_ATT_HEADS; head++){
            tmp2[head] = hls::exp(tmp[head]-softmax_bias[head])*softmax_recip[head];
        }
        dst[i] = tmp2;
    }
}

qkv_block_t zero_qkv_vec = {0, 0, 0 ,0 ,0 , 0 , 0 , 0 ,0 , 0, 0 , 0, 0, 0, 0, 0};
void compute_output_sub(
    hls::stream<head_block_t> & attn_score_stream,
    hls::stream<qkv_block_t> &v_stream,
    qkv_block_t output[],
    qkv_t softmax_recip[NUM_ATT_HEADS],
    qkv_t softmax_bias[NUM_ATT_HEADS],
    int token_length,
    int kv_cache_length
){

    qkv_block_t output_tmp_vecs[2][NUM_HIDDEN_BLOCKS];
    #pragma HLS ARRAY_PARTITION variable=output_tmp_vecs complete dim=1

    #pragma HLS bind_storage variable=output_tmp_vecs type=ram_1wnr impl=bram

    for(int i = 0; i < token_length; i++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=TOKEN_LENGTH

        head_block_t attn_score_tmp = attn_score_stream.read();
        head_block_t attn_score_sfmx;

        compute_score: for(int head_dim = 0; head_dim < NUM_ATT_HEADS; head_dim++){
            attn_score_sfmx[head_dim] = hls::exp(attn_score_tmp[head_dim]-softmax_bias[head_dim])*softmax_recip[head_dim];
        }

        compute_output: for(int dim_block = 0; dim_block < HIDDEN_SIZE/HIDDEN_BLOCK_SIZE; dim_block++){
            #pragma HLS pipeline
            int head = dim_block / BLOCK_PER_HEAD;

           qkv_block_t v_vec = v_stream.read();

            qkv_block_t output_tmp;

        for(int j = 0; j < HIDDEN_BLOCK_SIZE; j++){
                #pragma HLS UNROLL
                output_tmp[j] = attn_score_sfmx[head] * v_vec[j]; 
             }

            output_tmp_vecs[dim_block % 2][dim_block] += output_tmp;
        }
    }

    for(int dim_block = 0; dim_block < HIDDEN_SIZE/HIDDEN_BLOCK_SIZE; dim_block++){
        #pragma HLS pipeline
        output[dim_block + kv_cache_length * NUM_HIDDEN_BLOCKS] = 
            output_tmp_vecs[dim_block % 2][dim_block];
        output_tmp_vecs[dim_block % 2][dim_block] = zero_qkv_vec;
    }

}

void compute_output(
    hls::stream<head_block_t> & attn_score_stream,
    qkv_block_t src_v[],
    qkv_block_t output[],
    qkv_t softmax_recip[NUM_ATT_HEADS],
    qkv_t softmax_bias[NUM_ATT_HEADS],
    int token_length,
    int kv_cache_length
)
{
    stream<qkv_block_t> v_stream;
    #pragma HLS dataflow

    read_kv(
    v_stream,
    src_v,
    token_length);

    compute_output_sub(
        attn_score_stream,
        v_stream,
        output,
        softmax_recip,
        softmax_bias,
        token_length,
        kv_cache_length
    );

}

void compute_mv_fused_softmax(
    stream<qkv_block_t> & q_stream,  
    stream<qkv_block_t> & k_stream,  
    stream<head_block_t> & attn_score_stream,
    qkv_t softmax_recip[NUM_ATT_HEADS],
    qkv_t softmax_bias[NUM_ATT_HEADS],
    int token_length
)

{
    qkv_block_t q_vecs[NUM_HIDDEN_BLOCKS];  
    qkv_block_t qk_output_vecs[NUM_HIDDEN_BLOCKS]; 

    qkv_t sum[NUM_ATT_HEADS];
    qkv_t bias[NUM_ATT_HEADS];

    qkv_t curr_softmax_sum[NUM_ATT_HEADS] = {0};
    qkv_t curr_softmax_bias[NUM_ATT_HEADS] = {HALF_MIN};
    qkv_t curr_softmax_sum_recip[NUM_ATT_HEADS] = {0};

    readq:for(int dim_block = 0; dim_block < HIDDEN_SIZE/HIDDEN_BLOCK_SIZE; dim_block++){

        q_stream >> q_vecs[dim_block];

    }

    atten_score_token:for(int i = 0; i < token_length; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TOKEN_LENGTH

    head_block_t attn_score;  
  qkv_t qk_accum_output[NUM_HIDDEN_BLOCKS];

        atten_score_qk_mul:for(int dim_block = 0; dim_block < HIDDEN_SIZE/HIDDEN_BLOCK_SIZE; dim_block++){
            #pragma HLS pipeline

            int head = dim_block / BLOCK_PER_HEAD;

            qkv_block_t q_vec = q_vecs[dim_block];

            qkv_block_t k_vec;
            qkv_block_t att_vec;

            k_stream  >> k_vec;
             qkv_t attn_score_tmp;

             for(int j = 0; j < HIDDEN_BLOCK_SIZE; j++){
                #pragma HLS UNROLL
                att_vec[j] = q_vec[j] *k_vec[j];
             }
            attn_score_tmp = att_vec.reduce_add();
            qk_accum_output[dim_block] = attn_score_tmp;
        }

         for(int head = 0; head < NUM_ATT_HEADS; head++){
                #pragma HLS UNROLL
                for(int j = 0; j < BLOCK_PER_HEAD; j++){
                #pragma HLS UNROLL

                    attn_score[head] +=  qk_accum_output[head * BLOCK_PER_HEAD + j];

                }
            }
        attn_score = attn_score * SQRT_HEAD_DIM_REP;
        attn_score_stream << attn_score;

        oneline_softmax:for(int head = 0; head < NUM_ATT_HEADS; head++){
            #pragma HLS pipeline
            if(attn_score[head]>curr_softmax_bias[head])
            {
                curr_softmax_sum[head] = curr_softmax_sum[head] * hls::exp(curr_softmax_bias[head]-attn_score[head]) + 1;
                curr_softmax_bias[head] = attn_score[head];
            }
            else 
            {
                curr_softmax_sum[head] = curr_softmax_sum[head] + hls::exp(attn_score[head] - curr_softmax_bias[head]);
            }
        }
    }

    for(int head = 0; head < NUM_ATT_HEADS; head++){
        softmax_bias[head] = curr_softmax_bias[head];
        softmax_recip[head] = hls::recip(curr_softmax_sum[head]);
    }

}

void compute_decoding_attn_top(
    qkv_block_t src_q[],
    qkv_block_t src_k[],
    qkv_block_t src_v[],
    qkv_block_t final_output[],
    int token_length,
    int kv_cache_length
){

    stream<qkv_block_t> q_stream;  

    stream<qkv_block_t> k_stream;  

    qkv_t softmax_recip[NUM_ATT_HEADS];
    qkv_t softmax_bias[NUM_ATT_HEADS];

    stream<head_block_t> attn_score_stream; 
#pragma HLS STREAM depth=1024 variable=attn_score_stream
#pragma HLS bind_storage variable=attn_score_stream type=fifo impl=uram 

#pragma HLS dataflow

read_q(
    q_stream,
    src_q,
    kv_cache_length
);

read_kv(
    k_stream,
    src_k,
    token_length
);

compute_mv_fused_softmax(
    q_stream,  
    k_stream,  
    attn_score_stream,
    softmax_recip,
    softmax_bias,
    token_length
);

compute_output(

    attn_score_stream,
    src_v,  
    final_output,  
    softmax_recip,
    softmax_bias,
    token_length,
    kv_cache_length
);

}
#endif  