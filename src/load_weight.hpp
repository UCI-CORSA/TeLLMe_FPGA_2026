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

#ifndef LOAD_WEIGHT_HPP
#define LOAD_WEIGHT_HPP
#include <cassert>
#include "../config/config.hpp"
#include "hls_half.h"
#include "utils/x_hls_utils.h"

constexpr int ITERATIONS_PER_I = (HIDDEN2_SIZE + 4) / 5;   
constexpr int RMS_ARRAY_SIZE = NUM_HIDDEN_BLOCKS * 3 * HIDDEN_BLOCK_SIZE;   

void unpack_bus_to_vector(
    wide_bus_t& wide_bus_in,
    wt_index_vec_t& wt_vec_out) 
{
    #pragma HLS INLINE  

    UNPACK_LOOP:
    for (int i = 0; i < TMAC_TABLE_NUM; ++i) {
        #pragma HLS UNROLL

        int high = (i + 1) * TMAC_INDEX_WIDTH - 1;
        int low = i * TMAC_INDEX_WIDTH;
        wt_vec_out[i] = wide_bus_in.range(high, low);
    }
}

void load_weight_unified_no_vector_combined(
    wt_index_vec_widen_type_t weight_dram1[NUM_LAYER][NUM_ATTN_LINEAR + 3 * NUM_GATE_LINEAR][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][WEIGHT_DRAM_DIM],
    wt_index_vec_widen_type_t weight_dram2[NUM_LAYER][NUM_ATTN_LINEAR + 3 * NUM_GATE_LINEAR][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][WEIGHT_DRAM_DIM],
    wt_index_vec_widen_type_t weight_dram3[NUM_LAYER][NUM_ATTN_LINEAR + 3 * NUM_GATE_LINEAR][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][WEIGHT_DRAM_DIM],
    wt_index_vec_t weights_buffer[3][TMAC_INDEX_SIZE/TMAC_TABLE_NUM][HIDDEN2_SIZE],
    wt_norm_t rms_weight[NUM_HIDDEN_BLOCKS * 3 * HIDDEN_BLOCK_SIZE],
    int layer,
    int linear_idx   
){

    bool is_attn = (linear_idx < NUM_ATTN_LINEAR);
    int num_regions = is_attn ? 1 : 3;

    int start_region;
    if (is_attn) {

        start_region = linear_idx;
    } else {

        int gate_idx = linear_idx - NUM_ATTN_LINEAR;   
        start_region = NUM_ATTN_LINEAR + gate_idx * 3;   
    }

    int rms_write_idx = 0;

    for(int k = 0; k < num_regions; k++){
        int region_idx = start_region + k;

        for(int i = 0; i < TMAC_INDEX_SIZE/TMAC_TABLE_NUM; i++){

            for(int j = 0; j < (HIDDEN2_SIZE + 4)/5 ; j++){
                #pragma HLS PIPELINE II=1

                ap_uint<512> two_index = (weight_dram2[layer][region_idx][i][j].index, weight_dram1[layer][region_idx][i][j].index);
                ap_uint<768> three_index = (weight_dram3[layer][region_idx][i][j].index, two_index);

                wt_index_vec_t index_1;
                wt_index_vec_t index_2;
                wt_index_vec_t index_3;
                wt_index_vec_t index_4;
                wt_index_vec_t index_5;

                #pragma HLS aggregate variable=index_1 compact=bit
                #pragma HLS aggregate variable=index_2 compact=bit
                #pragma HLS aggregate variable=index_3 compact=bit
                #pragma HLS aggregate variable=index_4 compact=bit
                #pragma HLS aggregate variable=index_5 compact=bit

                wide_bus_t index_vector1 = three_index.range(TOTAL_WIDTH - 1, 0);
                wide_bus_t index_vector2 = three_index.range(2 * TOTAL_WIDTH - 1, TOTAL_WIDTH);
                wide_bus_t index_vector3 = three_index.range(3 * TOTAL_WIDTH - 1, 2 * TOTAL_WIDTH);
                wide_bus_t index_vector4 = three_index.range(4 * TOTAL_WIDTH - 1, 3 * TOTAL_WIDTH);
                wide_bus_t index_vector5 = three_index.range(5 * TOTAL_WIDTH - 1, 4 * TOTAL_WIDTH);

                unpack_bus_to_vector(index_vector1, index_1);
                unpack_bus_to_vector(index_vector2, index_2);
                unpack_bus_to_vector(index_vector3, index_3);
                unpack_bus_to_vector(index_vector4, index_4);
                unpack_bus_to_vector(index_vector5, index_5);

                int base_idx = 5 * j;
                weights_buffer[k][i][base_idx] = index_1;
                if(base_idx + 1 < HIDDEN2_SIZE) weights_buffer[k][i][base_idx + 1] = index_2;
                if(base_idx + 2 < HIDDEN2_SIZE) weights_buffer[k][i][base_idx + 2] = index_3;
                if(base_idx + 3 < HIDDEN2_SIZE) weights_buffer[k][i][base_idx + 3] = index_4;
                if(base_idx + 4 < HIDDEN2_SIZE) weights_buffer[k][i][base_idx + 4] = index_5;

                ap_uint<16> temp_rms = three_index.range(715, 700);
                fp_struct<half> half_val(temp_rms);

                if (rms_write_idx < RMS_ARRAY_SIZE) {
                    rms_weight[rms_write_idx] = half_val.to_half();
                }

                rms_write_idx++;

            }
        }
    }

}

#endif  