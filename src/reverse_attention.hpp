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

#ifndef REVERSE_ATTENTION_HPP
#define REVERSE_ATTENTION_HPP
#include "../config/config.hpp" 
#include "../config/macro.hpp" 
qkv_block_t zero_vec = {0, 0, 0 ,0 ,0 , 0 , 0 , 0 ,0 , 0, 0 , 0, 0, 0, 0, 0};
float_block_t zero_vec_float = {0, 0, 0 ,0 ,0 , 0 , 0 , 0 ,0 , 0, 0 , 0, 0, 0, 0, 0};

void read_q_block(
    hls::stream<qkv_block_t>& q_stream,
    qkv_block_t src[],  
    int token_length
){
  for(int i = 0; i < token_length; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512 
        for(int dim = 0; dim < NUM_HIDDEN_BLOCKS; dim ++){

                #pragma HLS pipeline
                q_stream << src[i * NUM_HIDDEN_BLOCKS + dim];

        }
    }

} 

void read_kv_block(
    hls::stream<qkv_block_t>& kv_stream,
    qkv_block_t src[],  
    int token_length
){

    for(int i = 0; i < token_length; i += ATTN_MATMUL_PARALLEL){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512 
        for(int j = i; j < token_length; j++){ 
            #pragma HLS LOOP_TRIPCOUNT min=1 max=512 
        for(int dim = 0; dim < NUM_HIDDEN_BLOCKS; dim ++){
            #pragma HLS PIPELINE

                kv_stream << src[j * NUM_HIDDEN_BLOCKS + dim];

        }
    }
    }
}

void write_attn_block(
    hls::stream<qkv_block_t> & attn_stream,
    qkv_block_t dst[],
    int token_length
){
    for(int i = 0; i < token_length; i++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512 
        for(int dim = 0; dim < NUM_HIDDEN_BLOCKS; dim ++){

            attn_stream >> dst[i * NUM_HIDDEN_BLOCKS + dim];

        }
    }
}

void fuse_qkv_pe_bundle_block_stream(
    stream<qkv_block_t> & q_stream,  
    stream<qkv_block_t> & k_stream,  
    stream<qkv_block_t> & v_stream,  
    stream<qkv_block_t> & attn_stream,
    int token_length
){
    qkv_blocks_t q_vecs [ATTN_MATMUL_PARALLEL];
    #pragma HLS ARRAY_RESHAPE variable=q_vecs complete dim = 3
    #pragma HLS ARRAY_PARTITION variable=q_vecs complete dim = 1
    #pragma HLS BIND_STORAGE variable=q_vecs type=ram_1wnr impl=bram 

    qkv_blocks_t v_vecs ;
     #pragma HLS BIND_STORAGE variable=v_vecs type=ram_1wnr impl=uram latency=3

    qkv_blocks_t qkv_vecs [ATTN_MATMUL_PARALLEL];
    #ifdef REVERSE_ATTENTION_OPT 
    #pragma HLS ARRAY_PARTITION variable=qkv_vecs cyclic factor=BLOCK_PER_HEAD dim = 2
    #endif

    qkv_t attn_dim_block [ATTN_MATMUL_PARALLEL][NUM_HIDDEN_BLOCKS] = {0};
    #ifndef REVERSE_ATTENTION_OPT
    #pragma HLS ARRAY_PARTITION variable=attn_dim_block complete dim = 1 
    #else
    #pragma HLS ARRAY_PARTITION variable=attn_dim_block cyclic factor=BLOCK_PER_HEAD dim = 2
    #endif

    qkv_t attn_block [ATTN_MATMUL_PARALLEL][NUM_ATT_HEADS] = {0};
    #pragma HLS ARRAY_PARTITION variable=attn_block complete dim = 1 

    qkv_t last_max [ATTN_MATMUL_PARALLEL][NUM_ATT_HEADS] = {HALF_MIN};  
    #pragma HLS ARRAY_PARTITION variable=last_max complete dim = 1
    qkv_t denom[ATTN_MATMUL_PARALLEL][NUM_ATT_HEADS] = {0};
    #pragma HLS ARRAY_PARTITION variable=denom complete dim = 1

   Q_DIM:
   for(int q_dim = 0; q_dim < token_length; q_dim += ATTN_MATMUL_PARALLEL){

   #pragma HLS LOOP_TRIPCOUNT min=1 max=512 
    KV_DIM:
        for(int kv_dim =  q_dim; kv_dim < token_length ; kv_dim++){

        #pragma HLS LOOP_TRIPCOUNT min=4 max=512         
            VEC_COMPUTE:
            for(int dim = 0; dim < HIDDEN_SIZE; dim += HIDDEN_BLOCK_SIZE){
                #pragma HLS pipeline

                int dim_block =  dim / HIDDEN_BLOCK_SIZE;

                if(kv_dim - q_dim < ATTN_MATMUL_PARALLEL){

                    q_stream >> q_vecs[kv_dim - q_dim][dim_block];
                    WATCH2(dim, kv_dim - q_dim, dim_block);
                }

                qkv_block_t k_vec;
                k_stream >> k_vec;

                v_stream >> v_vecs[dim_block];

                #pragma HLS allocation operation instances=hadd limit=32

                 for(int i = 0; i < ATTN_MATMUL_PARALLEL; i++){
                    #pragma HLS UNROLL
                    qkv_block_t q_vec = q_vecs[i][dim_block];
                    qkv_t attn_addend = 0;
                    for(int j = 0; j < HIDDEN_BLOCK_SIZE; j++){
                    #pragma HLS UNROLL
                        attn_addend += q_vec[j] * k_vec[j];
                        WATCH2(q_vec[j], k_vec[j],j);

                    }
                    attn_addend = attn_addend * SQRT_HEAD_DIM_REP;  

                    attn_dim_block[i][dim_block] += attn_addend;

                }

            }

            VEC_REDUCE:
            for(int head = 0; head < NUM_ATT_HEADS; head++){
            for(int i = 0; i < ATTN_MATMUL_PARALLEL; i++){
                #pragma HLS UNROLL
                for(int j = 0; j < BLOCK_PER_HEAD; j++){
                #pragma HLS UNROLL
                    attn_block[i][head] +=  attn_dim_block[i][head * BLOCK_PER_HEAD + j];

                }
            }
            }
            PARTIAL_SOFT_MAX:  
            for(int i = 0; i < ATTN_MATMUL_PARALLEL; i++){  
                for(int head = 0; head < NUM_ATT_HEADS; head++){
                    #pragma HLS LOOP_FLATTEN
                    #pragma HLS PIPELINE
                    qkv_t head_sum = attn_block[i][head];
                    attn_block[i][head] = 0;
                    qkv_t last_head_max =  (kv_dim ==  q_dim) ? head_sum : last_max[i][head];
                    qkv_t cur_max = fmax(head_sum, last_head_max);
                    qkv_t rescale_factor =  (i > (kv_dim - q_dim) || kv_dim == q_dim) ? (qkv_t)0 : exp((last_head_max - cur_max));
                    qkv_t exp_head_sum = (i > (kv_dim - q_dim)) ? (qkv_t)0 : exp((head_sum - cur_max));
                    WATCH(head_sum);
                    WATCH(last_head_max);
                    WATCH(cur_max);
                    WATCH(rescale_factor);
                    WATCH(exp_head_sum);
                    WATCH2(denom[i][head], i, head);
                    denom[i][head] =   rescale_factor * denom[i][head] + exp_head_sum;
                    WATCH2(denom[i][head], i, head);
                    last_max[i][head] = cur_max;
                    WATCH2(last_max[i][head], i, head);

                    for(int j = 0; j < HEAD_DIM; j++){
                        #pragma HLS unroll
                        int block_index = head * BLOCK_PER_HEAD + j / HIDDEN_BLOCK_SIZE;
                        int entry_index = j % HIDDEN_BLOCK_SIZE;

                        WATCH2(v_vecs[block_index][entry_index], block_index, entry_index);
                        WATCH3(qkv_vecs[i][block_index][entry_index], i ,block_index, entry_index);

                        WATCH(rescale_factor);
                        qkv_vecs[i][block_index][entry_index] =  qkv_vecs[i][block_index][entry_index] * rescale_factor + 
                                                    v_vecs[block_index][entry_index] * exp_head_sum;
                        WATCH3(qkv_vecs[i][block_index][entry_index], i ,block_index, entry_index);
                        attn_dim_block[i][block_index] = 0;
                    }
                }
            }

        }
        OUTPUT: 
        for(int i = 0; i < ATTN_MATMUL_PARALLEL; i++){
            for(int dim_block = 0; dim_block < NUM_HIDDEN_BLOCKS; dim_block++){
            #pragma HLS pipeline II=1
                int head = dim_block / BLOCK_PER_HEAD;

                qkv_block_t attn_block;
                for(int n = 0; n < HIDDEN_BLOCK_SIZE; n++){
                #pragma HLS UNROLL          
                     attn_block[n] = (qkv_t)((qkv_t)qkv_vecs[i][dim_block][n] *(qkv_t)hls::recip ((denom[i][head])));
                }
                attn_stream << attn_block;

                WATCH2(qkv_vecs[i][dim_block][15], i, dim_block);
                WATCH2(denom[i][head], i, head);

                qkv_vecs[i][dim_block] = zero_vec;
                q_vecs[i][dim_block]  =  zero_vec;

                last_max[i][head] = HALF_MIN;

            } 
        }

   }

}

void compute_qkv_block_top_no_max(
    qkv_block_t src_q[],
    qkv_block_t src_k[],
    qkv_block_t src_v[],
    qkv_block_t dst_attn[],
    int token_length
){

    #pragma HLS dataflow
    stream<qkv_block_t> q_stream;  

    stream<qkv_block_t> k_stream;  
    stream<qkv_block_t> v_stream;  
    stream<qkv_block_t> attn_stream;

    read_q_block(q_stream, src_q, token_length);
    read_kv_block(k_stream, src_k, token_length);
    read_kv_block(v_stream, src_v, token_length);

    fuse_qkv_pe_bundle_block_stream(
        q_stream,  
        k_stream,  
        v_stream,  
        attn_stream,
        token_length
    );

    write_attn_block(attn_stream, dst_attn, token_length);

}

#endif