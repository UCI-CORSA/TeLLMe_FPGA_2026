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

#ifndef TOP_BITNET_HPP
#define TOP_BITNET_HPP

#include "../config/config.hpp"
#include "../config/macro.hpp"
#include "load_weight.hpp"
#include "linear.hpp"
#include "rms_norm.hpp"
#include "reverse_attention.hpp"
#include "decode_attention.hpp"
#include "elem_wise.hpp"
#include "utils_quant.hpp"

void top_bitnet_fuse_ele_multi_weight_port(  

    qkv_block_t tmp1[MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],  
    qkv_block_t tmp2[MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],  
    qkv_block_t tmp3[MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],  
    qkv_block_t tmp4[MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],  
    qkv_block_t rope_freq[MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],   
    qkv_block_t out[MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],   
    qkv_block_t k_cache[NUM_LAYER][MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],  
    qkv_block_t v_cache[NUM_LAYER][MAX_TOKEN_LENGTH * NUM_HIDDEN_BLOCKS],  

    wt_index_vec_widen_type_t weight_dram1[NUM_LAYER][NUM_ATTN_LINEAR + 3 * NUM_GATE_LINEAR][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][WEIGHT_DRAM_DIM],
    wt_index_vec_widen_type_t weight_dram2[NUM_LAYER][NUM_ATTN_LINEAR + 3 * NUM_GATE_LINEAR][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][WEIGHT_DRAM_DIM],
    wt_index_vec_widen_type_t weight_dram3[NUM_LAYER][NUM_ATTN_LINEAR + 3 * NUM_GATE_LINEAR][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][WEIGHT_DRAM_DIM],

    int token_length,
    int kv_cache_length

){

    #pragma HLS interface m_axi port=tmp1 offset=slave bundle=inout1 depth=2048*256   max_widen_bitwidth=AXI_XFER_BIT_WIDTH  
    #pragma HLS interface m_axi port=tmp2 offset=slave bundle=inout2 depth=2048*256   max_widen_bitwidth=AXI_XFER_BIT_WIDTH  
    #pragma HLS interface m_axi port=tmp3 offset=slave bundle=inout3 depth=2048*256   max_widen_bitwidth=AXI_XFER_BIT_WIDTH   
    #pragma HLS interface m_axi port=tmp4 offset=slave bundle=inout4 depth=2048*256   max_widen_bitwidth=AXI_XFER_BIT_WIDTH      
    #pragma HLS interface m_axi port=rope_freq offset=slave bundle=inout4 depth=2048*256  max_widen_bitwidth=AXI_XFER_BIT_WIDTH    
    #pragma HLS interface m_axi port=out offset=slave bundle=inout1 depth=2048*256  max_widen_bitwidth=AXI_XFER_BIT_WIDTH    
    #pragma HLS interface m_axi port=k_cache offset=slave bundle=inout2 depth=2048*256  max_widen_bitwidth=AXI_XFER_BIT_WIDTH   
    #pragma HLS interface m_axi port=v_cache offset=slave bundle=inout4 depth=2048*256  max_widen_bitwidth=AXI_XFER_BIT_WIDTH    

    #pragma HLS interface m_axi depth=24*13*19*512 port=weight_dram1 offset=slave bundle=inout1  max_widen_bitwidth=AXI_XFER_BIT_WIDTH   
    #pragma HLS interface m_axi depth=24*13*19*512 port=weight_dram2 offset=slave bundle=inout2  max_widen_bitwidth=AXI_XFER_BIT_WIDTH  
    #pragma HLS interface m_axi depth=24*13*19*512 port=weight_dram3 offset=slave bundle=inout3  max_widen_bitwidth=AXI_XFER_BIT_WIDTH  

    #pragma HLS interface s_axilite port=tmp1 bundle=control
    #pragma HLS interface s_axilite port=tmp2 bundle=control
    #pragma HLS interface s_axilite port=tmp3 bundle=control
    #pragma HLS interface s_axilite port=tmp4 bundle=control
    #pragma HLS interface s_axilite port=rope_freq bundle=control
    #pragma HLS interface s_axilite port=out bundle=control
    #pragma HLS interface s_axilite port=k_cache bundle=control
    #pragma HLS interface s_axilite port=v_cache bundle=control
    #pragma HLS interface s_axilite port=weight_dram1 bundle=control
    #pragma HLS interface s_axilite port=weight_dram2 bundle=control
    #pragma HLS interface s_axilite port=weight_dram3 bundle=control

    #pragma HLS interface s_axilite port=token_length bundle=control
    #pragma HLS interface s_axilite port=kv_cache_length bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    wt_index_vec_t weights_buffer[3][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][HIDDEN2_SIZE];

    #pragma HLS ARRAY_RESHAPE variable=weights_buffer complete dim=4
    #pragma HLS ARRAY_PARTITION variable=weights_buffer cyclic factor = 8 dim=3
    #pragma HLS BIND_STORAGE variable=weights_buffer type=ram_1wnr impl=uram latency=3

    wt_norm_t rms_weight[NUM_HIDDEN_BLOCKS * 3 * HIDDEN_BLOCK_SIZE];

    #pragma HLS ARRAY_PARTITION variable=rms_weight cyclic factor=16 dim=1

    qkv_t max[MAX_TOKEN_LENGTH];

    qkv_t w_quant_qkv[NUM_LAYER][NUM_ATTN_LINEAR]
    #include "../config/w_quant_data.txt"
    ;

    qkv_t w_quant_gate[NUM_LAYER][NUM_GATE_LINEAR]
    #include "../config/w_quant_gate_data.txt"
    ;

    for(int layer = 0; layer < NUM_LAYER; layer++){
        #pragma HLS pipeline off
        #pragma HLS allocation function instances=compute_decoding_attn_top limit=1

        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, ATTN_Q);

        if(layer == 0){

            compute_norm(tmp2, tmp1, rms_weight, max, token_length, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, kv_cache_length);
        }

        compute_linear_bitnet_block_fuse_elemwise(tmp1, tmp3, rope_freq, weights_buffer, max, w_quant_qkv[layer][ATTN_Q], token_length, kv_cache_length, 0, 0, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, ROPE);

        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, ATTN_K);

        compute_linear_bitnet_block_fuse_elemwise(tmp1, k_cache[layer], rope_freq, weights_buffer, max, w_quant_qkv[layer][ATTN_K], token_length, kv_cache_length, 0, 0, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, ROPE);

        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, ATTN_V);

        compute_linear_bitnet_block_fuse_elemwise(tmp1, v_cache[layer], tmp3, weights_buffer, max, w_quant_qkv[layer][ATTN_V], token_length, kv_cache_length, 0, 0, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, NONE);

        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, ATTN_PROJ);

        if(kv_cache_length == 0){
            compute_qkv_block_top_no_max(tmp3, k_cache[layer], v_cache[layer], out, token_length);
        } else {
            compute_decoding_attn_top(tmp3, k_cache[layer], v_cache[layer], out, kv_cache_length + token_length, kv_cache_length);
        }

        compute_norm(out, tmp4, rms_weight, max, token_length, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, kv_cache_length);

        compute_linear_bitnet_block_fuse_elemwise(tmp4, tmp1, tmp2, weights_buffer, max, w_quant_qkv[layer][ATTN_PROJ], token_length, kv_cache_length, 0, 0, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, ADD);
        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, NUM_ATTN_LINEAR + Gate_UP);

        compute_norm(tmp1, tmp3, rms_weight, max, token_length, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, kv_cache_length);

        compute_linear_bitnet_block_fuse_elemwise(tmp3, tmp2, rope_freq, weights_buffer, max, w_quant_gate[layer][Gate_UP], token_length, kv_cache_length, 1, 0, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, FFN_HIDDEN2_SIZE/HIDDEN_BLOCK_SIZE, NONE);

        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, NUM_ATTN_LINEAR + Gate_PROJ);
        compute_linear_bitnet_block_fuse_elemwise(tmp3, out, tmp2, weights_buffer, max, w_quant_gate[layer][Gate_PROJ], token_length, kv_cache_length, 1, 0, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, FFN_HIDDEN2_SIZE/HIDDEN_BLOCK_SIZE, MULT_SILU);

        compute_norm(out, tmp4, rms_weight, max, token_length, FFN_HIDDEN2_SIZE/HIDDEN_BLOCK_SIZE, kv_cache_length);

        load_weight_unified_no_vector_combined(
            weight_dram1, weight_dram2, weight_dram3, 
            weights_buffer, rms_weight, layer, NUM_ATTN_LINEAR + Gate_DOWN);
        compute_linear_bitnet_block_fuse_elemwise(tmp4, tmp2, tmp1, weights_buffer, max, w_quant_gate[layer][Gate_DOWN], token_length, kv_cache_length, 2, 0, FFN_HIDDEN2_SIZE/HIDDEN_BLOCK_SIZE, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, ADD);

        compute_norm(tmp2, tmp1, rms_weight, max, token_length, HIDDEN_SIZE/HIDDEN_BLOCK_SIZE, kv_cache_length);
    }
}

#endif  