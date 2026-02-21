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

#ifndef __UTILS_QUANT_HPP__
#define __UTILS_QUANT_HPP__

#include "../config/config.hpp"

void quant_qkv_t2act_t(
    stream<qkv_block_t>& attn_stream, 
    stream<act_t>& proj_stream, 
    int token_length, 
    int in_dim_block,
    qkv_t max[]
){

    for(int i = 0; i < token_length; i++){ 
         #pragma HLS LOOP_TRIPCOUNT min=1 max=2048 

        for(int j = 0; j < in_dim_block; j++){
        #pragma HLS LOOP_TRIPCOUNT min= 96 max= 256
            qkv_block_t attn_block = attn_stream.read();
            for(int k = 0; k < HIDDEN_BLOCK_SIZE; k++){
            #pragma HLS PIPELINE

                   act_t act_val = (act_t)(((qkv_t)attn_block[k] *  ((qkv_t)QUANT_MAX * hls::recip((qkv_t)max[i]))));

                 proj_stream.write(act_val);
            }

            if(j == in_dim_block - 1 && in_dim_block == NUM_HIDDEN_BLOCKS){
            #pragma HLS occurrence cycle=HIDDEN_BLOCK_SIZE
                for(int k = 0; k < HIDDEN1_SIZE - HIDDEN_SIZE; k++){
                    act_t act_val = (act_t)(0);
                    proj_stream.write(act_val);
                }
            }

            if(j == in_dim_block - 1 && in_dim_block == FFN_NUM_HIDDEN_BLOCKS){
                #pragma HLS occurrence cycle=HIDDEN_BLOCK_SIZE
                for(int k = 0; k < FFN_HIDDEN1_SIZE - FFN_HIDDEN2_SIZE; k++){
                    act_t act_val = (act_t)(0);
                    proj_stream.write(act_val);
                }
            }
        }
    }

}

void dequant_act_t2qkv_t_block(stream<act_block_t>& proj_stream, stream<qkv_block_t>& attn_stream, int token_length, int out_dim_block,qkv_t max[], qkv_t w_quant){ 

    for(int i = 0; i < token_length; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2048

        for(int j = 0; j < out_dim_block; j++){
            #pragma HLS LOOP_TRIPCOUNT min=96 max=256
            #pragma HLS PIPELINE II=1

            qkv_block_t attn_block;
            act_block_t act_val = proj_stream.read();

            for(int k = 0; k < HIDDEN_BLOCK_SIZE; k++){
                #pragma HLS UNROLL

                qkv_t attn_val = (qkv_t)max[i] * (qkv_t)w_quant * (qkv_t)((float)QUANT_MAX_RECIP * (float)act_val[k]);

            attn_block[k] = attn_val;
        }
        attn_stream.write(attn_block);
    }
    }

}

#endif 