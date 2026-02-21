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

#ifndef __DATATYPES_HPP__
#define __DATATYPES_HPP__

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <ap_float.h>

using namespace hls;

#define PINGPONG_WHILE 1
#define USE_DSP_ADD 1

enum AttentionLinear {
    ATTN_Q = 0,      
    ATTN_K = 1,      
    ATTN_V = 2,      
    ATTN_PROJ = 3,   
    NUM_ATTN_LINEAR  
};

enum GateLinear {
    Gate_UP = 0,     
    Gate_PROJ = 1,   
    Gate_DOWN = 2,   
    NUM_GATE_LINEAR  
};

enum ElementWise {
    NONE = 0,        
    ROPE = 1,        
    ADD = 2,         
    MULT_SILU = 3    
};

const int NUM_LAYER = 24;                   
const int HIDDEN_SIZE = 1536;               
const int INTER_SIZE = 4096;                
const int NUM_ATT_HEADS = 16;               
const int LM_HEAD_LENGTH = 32016;           
constexpr int HEAD_DIM = HIDDEN_SIZE / NUM_ATT_HEADS;  

const int FFN_HIDDEN1_SIZE = 4096 + 4116 - 4096;  
const int FFN_HIDDEN2_SIZE = 4096;

const int HIDDEN1_SIZE = 1536 + 1596 - 1536;  
const int HIDDEN2_SIZE = 1536;

constexpr int WEIGHT_DRAM_DIM = (HIDDEN2_SIZE + 4) / 5;   

const int WEIGHT_STATE = 3;             
const int ACT_DATA_WIDTH = 8;           

typedef ap_int<2> wt_t;                 
typedef ap_fixed<8, 8, AP_RND_CONV, AP_SAT_SYM> act_t;  
#ifndef AP_FLOAT_HLS
typedef half qkv_t;                     
#else
typedef ap_float<8, 4> qkv_t;                     
#endif
typedef half wt_norm_t;                 
typedef float l2norm_t;                 
typedef ap_fixed<32, 10> fixed_t;       

constexpr float QUANT_MAX = 127.0f;
constexpr float QUANT_MAX_RECIP = 1.0f / QUANT_MAX;
const qkv_t QUANT_MIN = -128.0;

const int MAX_TOKEN_LENGTH = 3000;          
const int REAL_MAX_TOKEN_LENGTH = 1024;     

const int CUR_TOKEN_LENGTH_PF = 16;        
constexpr int TOKEN_LENGTH = CUR_TOKEN_LENGTH_PF;
const int KV_CACHE_SIZE = 512;              

constexpr int AXI_XFER_BIT_WIDTH = 256;     

const int TMAC_GROUP_SIZE = 3;              
const int TMAC_TABLE_SIZE = (3 * 3 * 3 - 1) / 2;  
const int TMAC_INDEX_WIDTH = 5;             
const int TMAC_TABLE_NUM = 28;              
const int TMAC_INDEX_SIZE = HIDDEN1_SIZE / WEIGHT_STATE;
const int TOTAL_WIDTH = TMAC_INDEX_WIDTH * TMAC_TABLE_NUM;  

typedef ap_int<ACT_DATA_WIDTH + 2> table_t;           
typedef ap_uint<TMAC_INDEX_WIDTH> wt_index_t;          
typedef ap_uint<TOTAL_WIDTH> wide_bus_t;               
typedef ap_uint<256> wt_split_t;                       

typedef hls::vector<wt_index_t, TMAC_TABLE_NUM> wt_index_vec_t;
typedef hls::vector<table_t, TMAC_TABLE_NUM> table_vec_t;
typedef hls::vector<act_t, TMAC_GROUP_SIZE * TMAC_TABLE_NUM> linear_in_t;  

typedef struct __attribute__((packed)) {
    wt_split_t index;  
} wt_index_vec_widen_type_t;

const int FLASH_ATT_BLOCK_SIZE = 16;
const qkv_t HALF_MIN = (qkv_t)-65504;      
const qkv_t LN2_REP = 1.4426950408889634;              
const qkv_t SQRT_HEAD_DIM_REP = (qkv_t)(1.0f / 9.79795897f);  

const qkv_t ROPE_BASE = 10000.0f;

const qkv_t norm_eps = 1e-5;               

const int HIDDEN2_BLOCK = 16;
#define BLOCK_SIZE_J HIDDEN2_BLOCK
#define BLOCK_SIZE_K (TMAC_GROUP_SIZE * TMAC_TABLE_NUM)

constexpr int HIDDEN_BLOCK_SIZE = (AXI_XFER_BIT_WIDTH / 16);  
constexpr int NUM_HIDDEN_BLOCKS = HIDDEN_SIZE / HIDDEN_BLOCK_SIZE;  
constexpr int FFN_NUM_HIDDEN_BLOCKS = FFN_HIDDEN2_SIZE / HIDDEN_BLOCK_SIZE;  
constexpr int NUM_HIDDEN_BLOCKS_FFN = FFN_NUM_HIDDEN_BLOCKS;
constexpr int BLOCK_PER_HEAD = HEAD_DIM / HIDDEN_BLOCK_SIZE;  

constexpr int ATTN_MATMUL_PARALLEL = 8;

typedef ap_int<ACT_DATA_WIDTH * 3> act_out_t;  

typedef hls::vector<act_t, HIDDEN2_BLOCK> linear_out_t;
typedef hls::vector<act_out_t, HIDDEN_BLOCK_SIZE> act_block_t;
typedef hls::vector<float, HIDDEN_BLOCK_SIZE> float_block_t;
typedef hls::vector<qkv_t, HIDDEN_BLOCK_SIZE> qkv_block_t;
typedef hls::vector<qkv_t, NUM_ATT_HEADS> head_block_t;
typedef hls::vector<wt_norm_t, HIDDEN_BLOCK_SIZE> wt_norm_block_t;

typedef qkv_block_t qkv_blocks_t[NUM_HIDDEN_BLOCKS];
typedef qkv_blocks_t* token_blocks_t;
typedef ap_uint<AXI_XFER_BIT_WIDTH> wt_dram_t;

template <typename T>
static constexpr T ap_fixed_min() {
    return T(-(1 << (T::iwidth - 1)));
}

#endif  