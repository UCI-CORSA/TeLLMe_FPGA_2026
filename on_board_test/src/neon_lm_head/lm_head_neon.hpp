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
 * @file lm_head_neon.hpp
 * @brief NEON SIMD optimized INT8 quantized lm_head implementation for ARM Cortex-A53
 * 
 * This implementation provides:
 * - Per-token absmax quantization of input activations
 * - INT8 matrix-vector multiplication using NEON SIMD
 * - Per-channel dequantization with weight scales
 * 
 * Optimized for ARM Cortex-A53 on Xilinx KV260 platform
 * 
 * Reference: case_top_bitnet_decode_v2.cpp::compute_lm_head_quant_cpu()
 */

#ifndef LM_HEAD_NEON_HPP
#define LM_HEAD_NEON_HPP

#include <arm_neon.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

// ============================================================================
// Configuration Constants
// ============================================================================

// Model dimensions (from config.hpp)
constexpr int HIDDEN_SIZE = 1536;
constexpr int VOCAB_SIZE = 32002;

// Quantization constants
constexpr float QUANT_SCALE = 127.0f;
constexpr float QUANT_SCALE_RECIP = 1.0f / 127.0f;

// NEON optimization parameters
constexpr int NEON_INT8_LANES = 16;   // 128-bit NEON register holds 16 x int8
constexpr int NEON_INT16_LANES = 8;   // 128-bit NEON register holds 8 x int16
constexpr int NEON_INT32_LANES = 4;   // 128-bit NEON register holds 4 x int32

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get current time in milliseconds for profiling
 */
inline double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

// ============================================================================
// NEON-Optimized Quantization Functions
// ============================================================================

/**
 * @brief Find absolute maximum value in FP16 array using NEON
 * 
 * Uses NEON to parallelize the absmax computation
 * 
 * @param input Pointer to FP16 input array
 * @param size Number of elements
 * @return Maximum absolute value as float
 */
inline float neon_absmax_fp16(const uint16_t* input, int size) {
    float max_val = 0.0f;
    
    // Process 8 FP16 values at a time using NEON
    int i = 0;
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    
    for (; i + 8 <= size; i += 8) {
        // Load 8 FP16 values and convert to FP32
        float16x8_t fp16_vec = vld1q_f16(reinterpret_cast<const __fp16*>(input + i));
        float32x4_t fp32_low = vcvt_f32_f16(vget_low_f16(fp16_vec));
        float32x4_t fp32_high = vcvt_f32_f16(vget_high_f16(fp16_vec));
        
        // Compute absolute values
        fp32_low = vabsq_f32(fp32_low);
        fp32_high = vabsq_f32(fp32_high);
        
        // Update maximum
        max_vec = vmaxq_f32(max_vec, fp32_low);
        max_vec = vmaxq_f32(max_vec, fp32_high);
    }
    
    // Reduce max_vec to scalar
    float32x2_t max2 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    max2 = vpmax_f32(max2, max2);
    max_val = vget_lane_f32(max2, 0);
    
    // Handle remaining elements
    for (; i < size; i++) {
        __fp16 fp16_val = *reinterpret_cast<const __fp16*>(input + i);
        float val = std::abs(static_cast<float>(fp16_val));
        if (val > max_val) max_val = val;
    }
    
    return std::max(max_val, 1e-5f);  // Avoid division by zero
}

/**
 * @brief Find absolute maximum value in FP32 array using NEON
 */
inline float neon_absmax_fp32(const float* input, int size) {
    float max_val = 0.0f;
    
    int i = 0;
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t vec = vld1q_f32(input + i);
        vec = vabsq_f32(vec);
        max_vec = vmaxq_f32(max_vec, vec);
    }
    
    // Reduce
    float32x2_t max2 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    max2 = vpmax_f32(max2, max2);
    max_val = vget_lane_f32(max2, 0);
    
    // Remaining
    for (; i < size; i++) {
        float val = std::abs(input[i]);
        if (val > max_val) max_val = val;
    }
    
    return std::max(max_val, 1e-5f);
}

/**
 * @brief Quantize FP16 input to INT8 using NEON (per-token absmax quantization)
 * 
 * Implements: q = round(clamp(x * scale, -128, 127))
 * 
 * @param input Pointer to FP16 input array
 * @param output Pointer to INT8 output array
 * @param size Number of elements
 * @param scale Quantization scale (127.0 / absmax)
 */
inline void neon_quantize_fp16_to_int8(
    const uint16_t* input,
    int8_t* output,
    int size,
    float scale
) {
    int i = 0;
    float32x4_t scale_vec = vdupq_n_f32(scale);
    
    // Process 8 elements at a time
    for (; i + 8 <= size; i += 8) {
        // Load 8 FP16 values
        float16x8_t fp16_vec = vld1q_f16(reinterpret_cast<const __fp16*>(input + i));
        
        // Convert to FP32
        float32x4_t fp32_low = vcvt_f32_f16(vget_low_f16(fp16_vec));
        float32x4_t fp32_high = vcvt_f32_f16(vget_high_f16(fp16_vec));
        
        // Scale
        fp32_low = vmulq_f32(fp32_low, scale_vec);
        fp32_high = vmulq_f32(fp32_high, scale_vec);
        
        // Round to nearest (NEON uses round-to-nearest-even by default with vcvtnq)
        int32x4_t int32_low = vcvtnq_s32_f32(fp32_low);
        int32x4_t int32_high = vcvtnq_s32_f32(fp32_high);
        
        // Saturate to INT16
        int16x4_t int16_low = vqmovn_s32(int32_low);
        int16x4_t int16_high = vqmovn_s32(int32_high);
        int16x8_t int16_vec = vcombine_s16(int16_low, int16_high);
        
        // Saturate to INT8
        int8x8_t int8_vec = vqmovn_s16(int16_vec);
        
        // Store
        vst1_s8(output + i, int8_vec);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        __fp16 fp16_val = *reinterpret_cast<const __fp16*>(input + i);
        float val = static_cast<float>(fp16_val) * scale;
        val = std::round(val);
        val = std::max(-128.0f, std::min(127.0f, val));
        output[i] = static_cast<int8_t>(val);
    }
}

/**
 * @brief Quantize FP32 input to INT8 using NEON
 */
inline void neon_quantize_fp32_to_int8(
    const float* input,
    int8_t* output,
    int size,
    float scale
) {
    int i = 0;
    float32x4_t scale_vec = vdupq_n_f32(scale);
    
    // Process 8 elements at a time (need 2 loads of 4 FP32)
    for (; i + 8 <= size; i += 8) {
        // Load 8 FP32 values
        float32x4_t fp32_low = vld1q_f32(input + i);
        float32x4_t fp32_high = vld1q_f32(input + i + 4);
        
        // Scale
        fp32_low = vmulq_f32(fp32_low, scale_vec);
        fp32_high = vmulq_f32(fp32_high, scale_vec);
        
        // Round and convert to INT32
        int32x4_t int32_low = vcvtnq_s32_f32(fp32_low);
        int32x4_t int32_high = vcvtnq_s32_f32(fp32_high);
        
        // Saturate to INT16
        int16x4_t int16_low = vqmovn_s32(int32_low);
        int16x4_t int16_high = vqmovn_s32(int32_high);
        int16x8_t int16_vec = vcombine_s16(int16_low, int16_high);
        
        // Saturate to INT8
        int8x8_t int8_vec = vqmovn_s16(int16_vec);
        
        // Store
        vst1_s8(output + i, int8_vec);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        float val = input[i] * scale;
        val = std::round(val);
        val = std::max(-128.0f, std::min(127.0f, val));
        output[i] = static_cast<int8_t>(val);
    }
}

// ============================================================================
// NEON-Optimized Matrix-Vector Multiplication (INT8)
// ============================================================================

/**
 * @brief Compute INT8 dot product of two vectors using NEON (4x unrolled)
 * 
 * Uses NEON SIMD with 4x loop unrolling for better instruction-level parallelism.
 * Processes 64 elements per iteration to maximize pipeline utilization on A53.
 * 
 * @param a Pointer to first INT8 vector (should be 16-byte aligned)
 * @param b Pointer to second INT8 vector (should be 16-byte aligned)
 * @param size Vector length
 * @return INT32 dot product result
 */
inline int32_t neon_dot_product_int8(
    const int8_t* __restrict a,
    const int8_t* __restrict b,
    int size
) {
    int i = 0;
    
    // Use 4 accumulator pairs for better pipelining (8 total accumulators)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    int32x4_t acc4 = vdupq_n_s32(0);
    int32x4_t acc5 = vdupq_n_s32(0);
    int32x4_t acc6 = vdupq_n_s32(0);
    int32x4_t acc7 = vdupq_n_s32(0);
    
    // Process 64 elements at a time (4x16) for better pipelining
    for (; i + 64 <= size; i += 64) {
        // Load 4 sets of 16 INT8 values
        int8x16_t va0 = vld1q_s8(a + i);
        int8x16_t vb0 = vld1q_s8(b + i);
        int8x16_t va1 = vld1q_s8(a + i + 16);
        int8x16_t vb1 = vld1q_s8(b + i + 16);
        int8x16_t va2 = vld1q_s8(a + i + 32);
        int8x16_t vb2 = vld1q_s8(b + i + 32);
        int8x16_t va3 = vld1q_s8(a + i + 48);
        int8x16_t vb3 = vld1q_s8(b + i + 48);
        
        // Process first 16 elements
        int16x8_t prod0_low = vmull_s8(vget_low_s8(va0), vget_low_s8(vb0));
        int16x8_t prod0_high = vmull_s8(vget_high_s8(va0), vget_high_s8(vb0));
        acc0 = vpadalq_s16(acc0, prod0_low);
        acc1 = vpadalq_s16(acc1, prod0_high);
        
        // Process second 16 elements
        int16x8_t prod1_low = vmull_s8(vget_low_s8(va1), vget_low_s8(vb1));
        int16x8_t prod1_high = vmull_s8(vget_high_s8(va1), vget_high_s8(vb1));
        acc2 = vpadalq_s16(acc2, prod1_low);
        acc3 = vpadalq_s16(acc3, prod1_high);
        
        // Process third 16 elements
        int16x8_t prod2_low = vmull_s8(vget_low_s8(va2), vget_low_s8(vb2));
        int16x8_t prod2_high = vmull_s8(vget_high_s8(va2), vget_high_s8(vb2));
        acc4 = vpadalq_s16(acc4, prod2_low);
        acc5 = vpadalq_s16(acc5, prod2_high);
        
        // Process fourth 16 elements
        int16x8_t prod3_low = vmull_s8(vget_low_s8(va3), vget_low_s8(vb3));
        int16x8_t prod3_high = vmull_s8(vget_high_s8(va3), vget_high_s8(vb3));
        acc6 = vpadalq_s16(acc6, prod3_low);
        acc7 = vpadalq_s16(acc7, prod3_high);
    }
    
    // Process remaining 16-element chunks
    for (; i + 16 <= size; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        
        int16x8_t prod_low = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t prod_high = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        
        acc0 = vpadalq_s16(acc0, prod_low);
        acc1 = vpadalq_s16(acc1, prod_high);
    }
    
    // Merge all accumulators
    acc0 = vaddq_s32(acc0, acc1);
    acc2 = vaddq_s32(acc2, acc3);
    acc4 = vaddq_s32(acc4, acc5);
    acc6 = vaddq_s32(acc6, acc7);
    acc0 = vaddq_s32(acc0, acc2);
    acc4 = vaddq_s32(acc4, acc6);
    acc0 = vaddq_s32(acc0, acc4);
    
    // Reduce to scalar
    int32x2_t sum2 = vadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
    sum2 = vpadd_s32(sum2, sum2);
    int32_t sum = vget_lane_s32(sum2, 0);
    
    // Handle remaining elements (< 16)
    for (; i < size; i++) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    
    return sum;
}

/**
 * @brief Compute INT8 matrix-vector product for a single output element
 * 
 * Computes: output[v] = sum(weight[v, :] * input[:])
 * 
 * @param weight Pointer to weight row (INT8, length = hidden_size)
 * @param input Pointer to quantized input vector (INT8, length = hidden_size)
 * @param hidden_size Vector length
 * @return INT32 accumulation result
 */
inline int32_t neon_matvec_row_int8(
    const int8_t* weight,
    const int8_t* input,
    int hidden_size
) {
    return neon_dot_product_int8(weight, input, hidden_size);
}

// ============================================================================
// Complete LM Head Computation
// ============================================================================

/**
 * @brief NEON-optimized quantized lm_head projection
 * 
 * Implements the complete lm_head computation with INT8 quantization:
 * 1. Quantize input activations (per-token absmax)
 * 2. INT8 matrix-vector multiplication
 * 3. Dequantize output with per-channel weight scales
 * 
 * @param input_fp16 Input hidden state (FP16, shape: [hidden_size])
 * @param weight_int8 Quantized weight matrix (INT8, shape: [vocab_size, hidden_size])
 * @param weight_scale Per-output-channel scales (FP32, shape: [vocab_size])
 * @param output_fp16 Output logits (FP16, shape: [vocab_size])
 * @param hidden_size Hidden dimension
 * @param vocab_size Vocabulary size
 */
void lm_head_neon_fp16(
    const uint16_t* input_fp16,
    const int8_t* weight_int8,
    const float* weight_scale,
    uint16_t* output_fp16,
    int hidden_size,
    int vocab_size
) {
    // Step 1: Find absmax of input
    float input_absmax = neon_absmax_fp16(input_fp16, hidden_size);
    float input_quant_scale = QUANT_SCALE / input_absmax;
    float input_dequant_scale = input_absmax / QUANT_SCALE;
    
    // Step 2: Quantize input to INT8
    std::vector<int8_t> quant_input(hidden_size);
    neon_quantize_fp16_to_int8(input_fp16, quant_input.data(), hidden_size, input_quant_scale);
    
    // Step 3: Matrix-vector multiplication and dequantization
    for (int v = 0; v < vocab_size; v++) {
        // INT8 dot product
        int32_t int_sum = neon_matvec_row_int8(
            weight_int8 + v * hidden_size,
            quant_input.data(),
            hidden_size
        );
        
        // Dequantize: output = int_sum * weight_scale[v] * input_dequant_scale
        float output_fp32 = static_cast<float>(int_sum) * weight_scale[v] * input_dequant_scale;
        
        // Convert to FP16 and store
        __fp16 fp16_out = static_cast<__fp16>(output_fp32);
        output_fp16[v] = *reinterpret_cast<uint16_t*>(&fp16_out);
    }
}

/**
 * @brief NEON-optimized quantized lm_head projection (FP32 I/O version)
 * 
 * Same as above but with FP32 input/output for easier testing
 */
void lm_head_neon_fp32(
    const float* input_fp32,
    const int8_t* weight_int8,
    const float* weight_scale,
    float* output_fp32,
    int hidden_size,
    int vocab_size
) {
    // Step 1: Find absmax of input
    float input_absmax = neon_absmax_fp32(input_fp32, hidden_size);
    float input_quant_scale = QUANT_SCALE / input_absmax;
    float input_dequant_scale = input_absmax / QUANT_SCALE;
    
    // Step 2: Quantize input to INT8
    std::vector<int8_t> quant_input(hidden_size);
    neon_quantize_fp32_to_int8(input_fp32, quant_input.data(), hidden_size, input_quant_scale);
    
    // Step 3: Matrix-vector multiplication and dequantization
    for (int v = 0; v < vocab_size; v++) {
        // INT8 dot product
        int32_t int_sum = neon_matvec_row_int8(
            weight_int8 + v * hidden_size,
            quant_input.data(),
            hidden_size
        );
        
        // Dequantize
        output_fp32[v] = static_cast<float>(int_sum) * weight_scale[v] * input_dequant_scale;
    }
}

// ============================================================================
// Batch Processing (Multi-threaded support)
// ============================================================================

/**
 * @brief Process a range of output rows (for multi-threading)
 * 
 * This function can be called from multiple threads to parallelize
 * the computation across the 4 ARM Cortex-A53 cores.
 * 
 * @param quant_input Pre-quantized input (INT8)
 * @param weight_int8 Weight matrix (INT8)
 * @param weight_scale Weight scales (FP32)
 * @param output_fp32 Output array (FP32)
 * @param hidden_size Hidden dimension
 * @param start_row Starting output index
 * @param end_row Ending output index (exclusive)
 * @param input_dequant_scale Dequantization scale for input
 */
void lm_head_neon_rows(
    const int8_t* quant_input,
    const int8_t* weight_int8,
    const float* weight_scale,
    float* output_fp32,
    int hidden_size,
    int start_row,
    int end_row,
    float input_dequant_scale
) {
    for (int v = start_row; v < end_row; v++) {
        int32_t int_sum = neon_matvec_row_int8(
            weight_int8 + v * hidden_size,
            quant_input,
            hidden_size
        );
        output_fp32[v] = static_cast<float>(int_sum) * weight_scale[v] * input_dequant_scale;
    }
}

#endif // LM_HEAD_NEON_HPP
