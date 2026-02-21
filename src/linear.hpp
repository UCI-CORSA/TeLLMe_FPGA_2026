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

#ifndef LINEAR_HPP
#define LINEAR_HPP
#include <cassert>
#include "../config/config.hpp"
#include "utils_quant.hpp"
#include "elem_wise.hpp"

qkv_block_t zero_vec_qkv = {0, 0, 0 ,0 ,0 , 0 , 0 , 0 ,0 , 0, 0 , 0, 0, 0, 0, 0};
act_block_t zero_vec_act = {0, 0, 0 ,0 ,0 , 0 , 0 , 0 ,0 , 0, 0 , 0, 0, 0, 0, 0};

void read_qkv_block(
    qkv_block_t qkv_in[],
    stream<qkv_block_t> &qkv_stream,
    int in_dim_block,
    unsigned int token_length,
    int kv_cache_length
){ 
    int total_blocks = token_length * in_dim_block;

    for(int i = 0; i < total_blocks ; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1536 max=2048 * 4096
        #pragma HLS PIPELINE
            qkv_stream.write(qkv_in[i + kv_cache_length * NUM_HIDDEN_BLOCKS]);
    }
}

void read_elem_wise_block(
    qkv_block_t elem_in[],
    stream<qkv_block_t> &elem_stream,
    int in_dim_block,
    unsigned int token_length,
    int elem_mode,
    int kv_cache_length    
){ 
    int total_blocks = token_length * in_dim_block;

    if(elem_mode != NONE) {  
        for(int i = 0; i < total_blocks ; i++){
            #pragma HLS LOOP_TRIPCOUNT min=1536 max=2048 * 4096
            #pragma HLS PIPELINE
            elem_stream.write(elem_in[i + kv_cache_length * NUM_HIDDEN_BLOCKS]);   
        }
    }
}

void generate_table_simd (
    linear_in_t A_BLOCK,
    table_t TMAC_LUT[TMAC_TABLE_NUM][2*TMAC_TABLE_SIZE+1]
){
    #pragma HLS inline

    TABLE_LOOP:
    for(int t = 0; t < TMAC_TABLE_NUM; t++){
        #pragma HLS UNROLL

        act_t val1 = A_BLOCK[t * TMAC_GROUP_SIZE + 0];
        act_t val2 = A_BLOCK[t * TMAC_GROUP_SIZE + 1];
        act_t val3 = A_BLOCK[t * TMAC_GROUP_SIZE + 2];

        ap_int<10> v1 = val1;
        ap_int<10> v2 = val2;
        ap_int<10> v3 = val3;
        ap_int<10> nv1 = -v1;
        ap_int<10> nv2 = -v2;
        ap_int<10> nv3 = -v3;

        ap_int<48> src_a_l2 = 0;
        src_a_l2.range(9, 0)   = v1;
        src_a_l2.range(21, 12) = v1;
        src_a_l2.range(33, 24) = v1;
        src_a_l2.range(45, 36) = nv1;

        ap_int<48> src_b_l2 = 0;
        src_b_l2.range(9, 0)   = v2;
        src_b_l2.range(21, 12) = v2;
        src_b_l2.range(33, 24) = nv2;
        src_b_l2.range(45, 36) = v2;

        ap_int<48> src_c_l2 = 0;
        src_c_l2.range(9, 0)   = v3;
        src_c_l2.range(21, 12) = nv3;
        src_c_l2.range(33, 24) = v3;
        src_c_l2.range(45, 36) = v3;

        ap_int<48> res_l2 = src_a_l2 + src_b_l2 + src_c_l2;
        #ifdef USE_DSP_ADD
        #pragma HLS BIND_OP variable=res_l2 op=add impl=dsp
        #endif

        table_t sum123          = (ap_int<10>)res_l2.range(9, 0);
        table_t sum12_m_v3      = (ap_int<10>)res_l2.range(21, 12);
        table_t diff12_p_v3     = (ap_int<10>)res_l2.range(33, 24);
        table_t neg_v1_p_sum23  = (ap_int<10>)res_l2.range(45, 36);

        ap_int<48> src_a_l1a = 0;
        src_a_l1a.range(9, 0)   = v1;
        src_a_l1a.range(21, 12) = v1;
        src_a_l1a.range(33, 24) = v1;
        src_a_l1a.range(45, 36) = v1;

        ap_int<48> src_b_l1a = 0;
        src_b_l1a.range(9, 0)   = v2;
        src_b_l1a.range(21, 12) = nv2;
        src_b_l1a.range(33, 24) = v3;
        src_b_l1a.range(45, 36) = nv3;

        ap_int<48> res_l1a = src_a_l1a + src_b_l1a;
        #ifdef USE_DSP_ADD
        #pragma HLS BIND_OP variable=res_l1a op=add impl=dsp
        #endif

        table_t sum12  = (ap_int<10>)res_l1a.range(9, 0);
        table_t diff12 = (ap_int<10>)res_l1a.range(21, 12);
        table_t sum13  = (ap_int<10>)res_l1a.range(33, 24);
        table_t diff13 = (ap_int<10>)res_l1a.range(45, 36);

        ap_int<36> src_a_l1b = 0;
        src_a_l1b.range(9, 0)   = v2;
        src_a_l1b.range(21, 12) = v2;

        ap_int<36> src_b_l1b = 0;
        src_b_l1b.range(9, 0)   = v3;
        src_b_l1b.range(21, 12) = nv3;

        ap_int<36> res_l1b = src_a_l1b + src_b_l1b;
        #ifdef USE_DSP_ADD
        #pragma HLS BIND_OP variable=res_l1b op=add impl=dsp
        #endif

        table_t sum23  = (ap_int<10>)res_l1b.range(9, 0);
        table_t diff23 = (ap_int<10>)res_l1b.range(21, 12);

        TMAC_LUT[t][0] = 0;
        TMAC_LUT[t][1] = val1;
        TMAC_LUT[t][2] = val2;
        TMAC_LUT[t][3] = val3;
        TMAC_LUT[t][4] = sum12;
        TMAC_LUT[t][5] = diff12;
        TMAC_LUT[t][6] = sum13;
        TMAC_LUT[t][7] = diff13;
        TMAC_LUT[t][8] = sum23;
        TMAC_LUT[t][9] = diff23;
        TMAC_LUT[t][10] = sum123;
        TMAC_LUT[t][11] = sum12_m_v3;
        TMAC_LUT[t][12] = diff12_p_v3;
        TMAC_LUT[t][13] = neg_v1_p_sum23;
        TMAC_LUT[t][14] = -val1;
        TMAC_LUT[t][15] = -val2;
        TMAC_LUT[t][16] = -val3;
        TMAC_LUT[t][17] = -sum12;
        TMAC_LUT[t][18] = -diff12;
        TMAC_LUT[t][19] = -sum13;
        TMAC_LUT[t][20] = -diff13;
        TMAC_LUT[t][21] = -sum23;
        TMAC_LUT[t][22] = -diff23;
        TMAC_LUT[t][23] = -sum123;
        TMAC_LUT[t][24] = -sum12_m_v3;
        TMAC_LUT[t][25] = -diff12_p_v3;
        TMAC_LUT[t][26] = -neg_v1_p_sum23;
    }
}

void compute_linear_on_stream_block(
    hls::stream<act_block_t>& out_stream,
    hls::stream<act_t>& in_stream,
    wt_index_vec_t weights_buffer [3][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][HIDDEN2_SIZE],
    int select_index,
    unsigned int token_length,
    int mode

){                             
    table_t TMAC_LUT[TMAC_TABLE_NUM][2*TMAC_TABLE_SIZE+1];  
    #pragma HLS ARRAY_PARTITION variable=TMAC_LUT complete dim=1
     #pragma HLS BIND_STORAGE variable=TMAC_LUT type=ram_1wnr impl=lutram latency=3

    linear_in_t A_BLOCK;
    act_block_t C_BLOCK[FFN_NUM_HIDDEN_BLOCKS];

    for(int i = 0; i < FFN_NUM_HIDDEN_BLOCKS; i++){
        #pragma HLS pipeline
        C_BLOCK[i] =zero_vec_act;
    }

    TOKEN_LOOP:
    for(int i = 0 ; i < token_length; i++){ 
    #pragma HLS LOOP_TRIPCOUNT min=1 max=2048 

    HIDDEN1_BLOCK_LOOP:
    for(int j = 0; j < FFN_HIDDEN1_SIZE; j += TMAC_TABLE_NUM * TMAC_GROUP_SIZE){

        if(j == (HIDDEN1_SIZE) && (mode == 0 || mode == 1)){

             break;
        }

         READ_A:
         for(int p = 0; p < TMAC_TABLE_NUM * TMAC_GROUP_SIZE; p++){
              #pragma HLS PIPELINE
             A_BLOCK[p] = in_stream.read();
         }

        generate_table_simd(A_BLOCK, TMAC_LUT);

    LUT_HIDDEN_LOOP:
    for(int m = 0; m < FFN_HIDDEN2_SIZE; m += HIDDEN2_BLOCK){  
    #pragma HLS pipeline
    if(m == (HIDDEN2_SIZE) &&  (mode == 0 || mode == 2)) {

        break;
    }

    HIDDEN2_BLOCK_LOOP:
    for(int n = 0; n < HIDDEN2_BLOCK; n++){
        #pragma HLS UNROLL
        int x_index_mode2 = (j / (TMAC_TABLE_NUM * TMAC_GROUP_SIZE)) / (TMAC_INDEX_SIZE / TMAC_TABLE_NUM);
        int y_index =  (j / (TMAC_TABLE_NUM * TMAC_GROUP_SIZE)) % (TMAC_INDEX_SIZE / TMAC_TABLE_NUM);  
        int x_index_mode1 = (m + n) / (HIDDEN2_SIZE);
        int z_index = (m + n) % (HIDDEN2_SIZE);  
        int x_final = (mode == 0) ? select_index :
                      (mode == 1) ? x_index_mode1 :
                                    x_index_mode2 ;  

        wt_index_vec_t index_vec = weights_buffer[x_final][y_index][z_index];  

    LOOK_UP_LOOP:
        for(int t = 0; t < TMAC_TABLE_NUM; t++){  
        #pragma HLS UNROLL

                wt_index_t lut_index = index_vec[t]; 
                C_BLOCK[m / HIDDEN_BLOCK_SIZE][n] += TMAC_LUT[t][lut_index];  

            }

    }

    }

    }

WRITE_C:
    for(int p = 0; p < FFN_NUM_HIDDEN_BLOCKS ; p++){
     #pragma HLS PIPELINE
       if(p == (NUM_HIDDEN_BLOCKS) && (mode == 0|| mode == 2)) break;

        out_stream.write(C_BLOCK[p]);
        C_BLOCK[p] = zero_vec_act;

    }

    }

}

void write_qkv_block_bitnet(
    qkv_block_t qkv_out[],
    stream<qkv_block_t> &qkv_stream,
    int out_dim_block,
    int token_length,
    int kv_cache_length
){ 
    int total_blocks = token_length * out_dim_block;

    for(int i = 0; i < total_blocks; i++){
#pragma HLS LOOP_TRIPCOUNT min=1536 max=2048 * 256

            #pragma HLS PIPELINE
            qkv_out[i + kv_cache_length * NUM_HIDDEN_BLOCKS] = qkv_stream.read();

    }
}

void compute_linear_bitnet_block_fuse_elemwise(
    qkv_block_t * qkv_in,
    qkv_block_t * qkv_out,  
    qkv_block_t * elem_in,
    wt_index_vec_t weights_buffer[3][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][HIDDEN2_SIZE],
    qkv_t max[],
    qkv_t w_quant,
    unsigned int token_length,
    int kv_cache_length,
    int mode,
    int select_index,
    int in_dim_block,
    int out_dim_block,
    int elem_mode

){

    hls::stream<act_t> in_stream;
    hls::stream<act_block_t> out_stream;
    hls::stream<qkv_block_t> qkv_stream_in;
    hls::stream<qkv_block_t> qkv_stream_out;
    hls::stream<qkv_block_t> qkv_stream_elem_out;
    hls::stream<qkv_block_t> elem_stream_out;
#pragma HLS STREAM variable=in_stream depth=16 type=fifo
#pragma HLS STREAM variable=out_stream depth=2 type=fifo
#pragma HLS STREAM variable=elem_stream_out depth=4 type=fifo
#pragma HLS STREAM variable=qkv_stream_in depth=4 type=fifo
#pragma HLS STREAM variable=qkv_stream_out depth=4 type=fifo
#pragma HLS STREAM variable=qkv_stream_elem_out depth=2 type=fifo

    #pragma HLS dataflow

    read_elem_wise_block(elem_in, elem_stream_out, out_dim_block, token_length, elem_mode, kv_cache_length);

    read_qkv_block(qkv_in, qkv_stream_in, in_dim_block, token_length, kv_cache_length);

    quant_qkv_t2act_t(qkv_stream_in, in_stream, token_length,in_dim_block, max);

    compute_linear_on_stream_block(out_stream, in_stream, weights_buffer, select_index, token_length, mode);

    dequant_act_t2qkv_t_block(out_stream, qkv_stream_out, token_length, out_dim_block,max, w_quant);

    apply_elem_wise(qkv_stream_out, qkv_stream_elem_out, elem_stream_out, token_length, kv_cache_length, out_dim_block, elem_mode);

    write_qkv_block_bitnet(qkv_out, qkv_stream_elem_out, out_dim_block, token_length, kv_cache_length);

}

#endif