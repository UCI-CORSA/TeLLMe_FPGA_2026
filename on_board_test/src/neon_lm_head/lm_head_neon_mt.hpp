/**
 * BSD 3-Clause License
 * 
 * Copyright (c) 2025, Yifan Zhang
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

/**
 * @file lm_head_neon_mt.hpp
 * @brief Multi-threaded NEON lm_head implementation using pthreads
 * 
 * This implementation leverages all 4 ARM Cortex-A53 cores on KV260
 * for parallel computation of the lm_head projection.
 * 
 * The workload is divided evenly across threads, with each thread
 * processing a contiguous range of output vocabulary indices.
 */

#ifndef LM_HEAD_NEON_MT_HPP
#define LM_HEAD_NEON_MT_HPP

#include "lm_head_neon.hpp"
#include <pthread.h>
#include <cstring>

// ============================================================================
// Multi-threading Configuration
// ============================================================================

#ifndef NUM_THREADS
#define NUM_THREADS 4  // 4 ARM Cortex-A53 cores on KV260
#endif

// ============================================================================
// Static Buffers (avoid dynamic allocation in hot path)
// ============================================================================

// Pre-allocated buffers for lm_head computation (thread-local would be even better)
static int8_t g_quant_input_buffer[HIDDEN_SIZE] __attribute__((aligned(16)));
static float g_input_fp32_buffer[HIDDEN_SIZE] __attribute__((aligned(16)));
static float g_output_fp32_buffer[VOCAB_SIZE] __attribute__((aligned(16)));

// ============================================================================
// Thread Work Structure
// ============================================================================

struct LMHeadThreadWork {
    const int8_t* quant_input;
    const int8_t* weight_int8;
    const float* weight_scale;
    float* output_fp32;
    int hidden_size;
    int start_row;
    int end_row;
    float input_dequant_scale;
};

// ============================================================================
// Thread Worker Function
// ============================================================================

void* lm_head_thread_worker(void* arg) {
    LMHeadThreadWork* work = static_cast<LMHeadThreadWork*>(arg);
    
    lm_head_neon_rows(
        work->quant_input,
        work->weight_int8,
        work->weight_scale,
        work->output_fp32,
        work->hidden_size,
        work->start_row,
        work->end_row,
        work->input_dequant_scale
    );
    
    return nullptr;
}

// ============================================================================
// Multi-threaded LM Head Implementation
// ============================================================================

/**
 * @brief Multi-threaded NEON-optimized lm_head projection
 * 
 * Distributes the computation across NUM_THREADS threads,
 * utilizing all available ARM Cortex-A53 cores.
 * 
 * @param input_fp32 Input hidden state (FP32, shape: [hidden_size])
 * @param weight_int8 Quantized weight matrix (INT8, shape: [vocab_size, hidden_size])
 * @param weight_scale Per-output-channel scales (FP32, shape: [vocab_size])
 * @param output_fp32 Output logits (FP32, shape: [vocab_size])
 * @param hidden_size Hidden dimension
 * @param vocab_size Vocabulary size
 * @param num_threads Number of threads to use (default: NUM_THREADS)
 */
void lm_head_neon_mt(
    const float* input_fp32,
    const int8_t* weight_int8,
    const float* weight_scale,
    float* output_fp32,
    int hidden_size,
    int vocab_size,
    int num_threads = NUM_THREADS
) {
    // Step 1: Find absmax and compute scales (single-threaded, small computation)
    float input_absmax = neon_absmax_fp32(input_fp32, hidden_size);
    float input_quant_scale = QUANT_SCALE / input_absmax;
    float input_dequant_scale = input_absmax / QUANT_SCALE;
    
    // Step 2: Quantize input using static buffer (avoid malloc)
    int8_t* quant_input = g_quant_input_buffer;
    neon_quantize_fp32_to_int8(input_fp32, quant_input, hidden_size, input_quant_scale);
    
    // Step 3: Parallel matrix-vector multiplication
    pthread_t threads[NUM_THREADS];
    LMHeadThreadWork work[NUM_THREADS];
    
    // Pre-compute thread work distribution
    int rows_per_thread = vocab_size / num_threads;
    int remaining_rows = vocab_size % num_threads;
    
    int current_row = 0;
    for (int t = 0; t < num_threads; t++) {
        int thread_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        
        work[t].quant_input = quant_input;
        work[t].weight_int8 = weight_int8;
        work[t].weight_scale = weight_scale;
        work[t].output_fp32 = output_fp32;
        work[t].hidden_size = hidden_size;
        work[t].start_row = current_row;
        work[t].end_row = current_row + thread_rows;
        work[t].input_dequant_scale = input_dequant_scale;
        
        current_row += thread_rows;
    }
    
    // Create all threads first (batch creation for better scheduling)
    for (int t = 0; t < num_threads; t++) {
        pthread_create(&threads[t], nullptr, lm_head_thread_worker, &work[t]);
    }
    
    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
}

/**
 * @brief Multi-threaded NEON-optimized lm_head projection with FP16 I/O
 */
void lm_head_neon_mt_fp16(
    const uint16_t* input_fp16,
    const int8_t* weight_int8,
    const float* weight_scale,
    uint16_t* output_fp16,
    int hidden_size,
    int vocab_size,
    int num_threads = NUM_THREADS
) {
    // Use static buffers instead of std::vector
    float* input_fp32 = g_input_fp32_buffer;
    float* output_fp32 = g_output_fp32_buffer;
    
    // Convert FP16 input to FP32 using NEON (8 elements at a time)
    int i = 0;
    for (; i + 8 <= hidden_size; i += 8) {
        float16x8_t fp16_vec = vld1q_f16(reinterpret_cast<const __fp16*>(input_fp16 + i));
        float32x4_t fp32_low = vcvt_f32_f16(vget_low_f16(fp16_vec));
        float32x4_t fp32_high = vcvt_f32_f16(vget_high_f16(fp16_vec));
        vst1q_f32(input_fp32 + i, fp32_low);
        vst1q_f32(input_fp32 + i + 4, fp32_high);
    }
    // Handle remaining
    for (; i < hidden_size; i++) {
        __fp16 fp16_val = *reinterpret_cast<const __fp16*>(&input_fp16[i]);
        input_fp32[i] = static_cast<float>(fp16_val);
    }
    
    // Compute in FP32 (using static output buffer)
    lm_head_neon_mt(
        input_fp32,
        weight_int8,
        weight_scale,
        output_fp32,
        hidden_size,
        vocab_size,
        num_threads
    );
    
    // Convert FP32 output to FP16 using NEON (8 elements at a time)
    i = 0;
    for (; i + 8 <= vocab_size; i += 8) {
        float32x4_t fp32_low = vld1q_f32(output_fp32 + i);
        float32x4_t fp32_high = vld1q_f32(output_fp32 + i + 4);
        float16x4_t fp16_low = vcvt_f16_f32(fp32_low);
        float16x4_t fp16_high = vcvt_f16_f32(fp32_high);
        float16x8_t fp16_vec = vcombine_f16(fp16_low, fp16_high);
        vst1q_f16(reinterpret_cast<__fp16*>(output_fp16 + i), fp16_vec);
    }
    // Handle remaining
    for (; i < vocab_size; i++) {
        __fp16 fp16_val = static_cast<__fp16>(output_fp32[i]);
        output_fp16[i] = *reinterpret_cast<uint16_t*>(&fp16_val);
    }
}

// ============================================================================
// Convenience Wrapper with Default Parameters
// ============================================================================

/**
 * @brief Compute lm_head with BitNet model default dimensions
 * 
 * Uses HIDDEN_SIZE=1536 and VOCAB_SIZE=32016 by default
 */
void compute_lm_head_quant_neon(
    const float* last_hidden_state,
    const int8_t* quant_weight,
    const float* weight_scale,
    float* logits
) {
    lm_head_neon_mt(
        last_hidden_state,
        quant_weight,
        weight_scale,
        logits,
        HIDDEN_SIZE,
        VOCAB_SIZE,
        NUM_THREADS
    );
}

#endif // LM_HEAD_NEON_MT_HPP
