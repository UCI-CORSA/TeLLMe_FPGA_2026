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

#ifndef ELEM_WISE_HPP
#define ELEM_WISE_HPP
#include "../config/config.hpp"

#define MIN_VAL -8.0f      
#define MAX_VAL 8.0f       
#define ENTRIES 64 
const qkv_t g_scale = (qkv_t)(ENTRIES - 1) / (MAX_VAL - MIN_VAL);

qkv_t g_silu_lut[ENTRIES] = {
        -0.0026836395, -0.0033493042, -0.0041732788, -0.0051994324, -0.0064659119, -0.0080261230, -0.0099563599, -0.0123291016,
        -0.0152282715, -0.0187835693, -0.0231170654, -0.0283813477, -0.0347595215, -0.0424194336, -0.0515747070, -0.0625000000,
        -0.0753784180, -0.0903930664, -0.1077270508, -0.1274414062, -0.1494140625, -0.1732177734, -0.1983642578, -0.2235107422,
        -0.2468261719, -0.2658691406, -0.2770996094, -0.2763671875, -0.2590332031, -0.2199707031, -0.1546630859, -0.0594787598,
        0.0675048828, 0.2263183594, 0.4150390625, 0.6298828125, 0.8666992188, 1.1201171875, 1.3847656250, 1.6582031250,
        1.9355468750, 2.2148437500, 2.4941406250, 2.7714843750, 3.0468750000, 3.3203125000, 3.5917968750, 3.8613281250,
        4.1289062500, 4.3945312500, 4.6562500000, 4.9179687500, 5.1796875000, 5.4375000000, 5.6953125000, 5.9531250000,
        6.2109375000, 6.4648437500, 6.7226562500, 6.9765625000, 7.2343750000, 7.4882812500, 7.7421875000, 7.9960937500
    };
typedef ap_fixed<16, 8, AP_RND_CONV, AP_SAT> fixed16_t;
const qkv_t MIN_VAL_ADJUSTED = (qkv_t)((float)MIN_VAL - (float)0.5 / (float)g_scale);  
qkv_t silu_correction_lut(qkv_t x) {

    #pragma HLS INLINE

    qkv_t x_clamped = x;

    qkv_t index_float = (x_clamped - MIN_VAL_ADJUSTED) * (qkv_t)g_scale;

    fp_struct<half> fp(index_float);
    int e = fp.expv();

    ap_uint<6> index;
    if (e < 0) {
        index = 0;
    } else if (e >= 6) {
        index = ENTRIES - 1;
    } else {

        ap_uint<11> m_bits = fp.mantissa().range(10, 0);
        index = m_bits >> (10 - e);
    }

    return g_silu_lut[index];
}

void apply_elem_wise(
    stream<qkv_block_t>& input_stream,
    stream<qkv_block_t>& output_stream,
    stream<qkv_block_t>& elem_stream,
    int token_length, 
    int kv_cache_length,
    int out_dim_block,
    int elem_mode  
) 
{
    int total_blocks = token_length * out_dim_block;

    process_embeddings:
    for (int pos = 0; pos < token_length; pos++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2048

        apply_rope_transform:
        for (int dim = 0; dim < out_dim_block * HIDDEN_BLOCK_SIZE ; dim += HIDDEN_BLOCK_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=96 max=256

            qkv_block_t qkv_block = input_stream.read();

            qkv_block_t output_block;
            qkv_block_t elem_block;

            if(elem_mode == ROPE){  
                elem_block = elem_stream.read();

                rope_pairs:
                for(int j = 0; j < HIDDEN_BLOCK_SIZE; j += 2) {
                    #pragma HLS UNROLL

                    qkv_t x_i = qkv_block[j];
                    qkv_t x_i_plus_1 = qkv_block[j + 1];

                    qkv_t cos_theta = elem_block[j];
                    qkv_t sin_theta = elem_block[j + 1];
                    output_block[j] = x_i * cos_theta - x_i_plus_1 * sin_theta;
                    output_block[j + 1] = x_i * sin_theta + x_i_plus_1 * cos_theta;

                }
                output_stream.write(output_block);
            }
            else if(elem_mode == ADD) {  
                elem_block = elem_stream.read();
                output_block = qkv_block + elem_block;  
                output_stream.write(output_block);
            }
            else if(elem_mode == MULT_SILU) {  
                elem_block = elem_stream.read();
                SILU_MULT:
                for(int j = 0; j < HIDDEN_BLOCK_SIZE; j ++){
                #pragma HLS UNROLL 
                    qkv_block[j] = silu_correction_lut(qkv_block[j]);
                }
                output_block = qkv_block * elem_block;  
                output_stream.write(output_block);
            }
            else if(elem_mode == NONE) {
                output_stream.write(qkv_block);
            }
        }
    }
}

#endif
