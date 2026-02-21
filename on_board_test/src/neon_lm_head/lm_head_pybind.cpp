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
 * @file lm_head_pybind.cpp
 * @brief PyBind11 wrapper for NEON-optimized LM head implementation
 * 
 * Fixed-size buffer implementation for BitNet b1.58-large:
 * - HIDDEN_SIZE = 1536
 * - VOCAB_SIZE = 32002
 * 
 * Data is loaded in Python (numpy), C++ NEON only computes.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>

#include "lm_head_neon.hpp"
#include "lm_head_neon_mt.hpp"

namespace py = pybind11;

// ============================================================================
// Fixed-size LM Head Processor
// ============================================================================

class LMHeadNEON {
private:
    // Fixed-size buffers (pre-allocated)
    static constexpr int HIDDEN = HIDDEN_SIZE;   // 1536
    static constexpr int VOCAB = VOCAB_SIZE;     // 32002
    static constexpr size_t WEIGHT_TOTAL = static_cast<size_t>(VOCAB) * HIDDEN;
    
    // Weight buffer [VOCAB_SIZE, HIDDEN_SIZE] - INT8
    int8_t weight_int8_[VOCAB * HIDDEN];
    
    // Scale buffer [VOCAB_SIZE] - FP32
    float weight_scale_[VOCAB];
    
    // Output buffer [VOCAB_SIZE] - FP32
    float output_fp32_[VOCAB];
    
    bool weights_loaded_;
    
public:
    LMHeadNEON() : weights_loaded_(false) {
        // Zero-initialize all buffers
        std::memset(weight_int8_, 0, sizeof(weight_int8_));
        std::memset(weight_scale_, 0, sizeof(weight_scale_));
        std::memset(output_fp32_, 0, sizeof(output_fp32_));
    }
    
    /**
     * @brief Load weights from numpy arrays (called once at init)
     * 
     * @param weights INT8 weights as int32 array [VOCAB_SIZE, HIDDEN_SIZE]
     *                (stored as int32 but values are in [-128, 127])
     * @param scales FP32 per-channel scales [VOCAB_SIZE]
     */
    void load_weights(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> scales
    ) {
        auto w_info = weights.request();
        auto s_info = scales.request();
        
        // Verify sizes
        size_t w_size = 1;
        for (ssize_t i = 0; i < w_info.ndim; i++) {
            w_size *= w_info.shape[i];
        }
        
        if (w_size != WEIGHT_TOTAL) {
            throw std::runtime_error("Weight size mismatch");
        }
        if (static_cast<size_t>(s_info.shape[0]) != static_cast<size_t>(VOCAB)) {
            throw std::runtime_error("Scale size mismatch");
        }
        
        // Copy weights (int32 -> int8)
        const int32_t* w_ptr = static_cast<const int32_t*>(w_info.ptr);
        for (size_t i = 0; i < WEIGHT_TOTAL; i++) {
            weight_int8_[i] = static_cast<int8_t>(w_ptr[i]);
        }
        
        // Copy scales
        std::memcpy(weight_scale_, s_info.ptr, VOCAB * sizeof(float));
        
        weights_loaded_ = true;
    }
    
    /**
     * @brief Compute LM head projection using NEON (single-threaded, FP16)
     * 
     * @param hidden_state FP16 input [HIDDEN_SIZE] (as uint16)
     * @return FP16 output [VOCAB_SIZE] (as uint16)
     */
    py::array_t<uint16_t> forward_fp16(
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> hidden_state
    ) {
        if (!weights_loaded_) {
            throw std::runtime_error("Weights not loaded");
        }
        
        auto info = hidden_state.request();
        if (static_cast<int>(info.shape[0]) != HIDDEN) {
            throw std::runtime_error("Input size mismatch");
        }
        
        // Allocate output
        auto result = py::array_t<uint16_t>(VOCAB);
        auto result_info = result.request();
        
        // Call NEON FP16 implementation
        lm_head_neon_fp16(
            static_cast<const uint16_t*>(info.ptr),
            weight_int8_,
            weight_scale_,
            static_cast<uint16_t*>(result_info.ptr),
            HIDDEN,
            VOCAB
        );
        
        return result;
    }
    
    /**
     * @brief Compute LM head projection using NEON (multi-threaded, FP16)
     * 
     * @param hidden_state FP16 input [HIDDEN_SIZE] (as uint16)
     * @return FP16 output [VOCAB_SIZE] (as uint16)
     */
    py::array_t<uint16_t> forward_fp16_mt(
        py::array_t<uint16_t, py::array::c_style | py::array::forcecast> hidden_state
    ) {
        if (!weights_loaded_) {
            throw std::runtime_error("Weights not loaded");
        }
        
        auto info = hidden_state.request();
        if (static_cast<int>(info.shape[0]) != HIDDEN) {
            throw std::runtime_error("Input size mismatch");
        }
        
        // Allocate output
        auto result = py::array_t<uint16_t>(VOCAB);
        auto result_info = result.request();
        
        // Call NEON FP16 multi-threaded implementation
        lm_head_neon_mt_fp16(
            static_cast<const uint16_t*>(info.ptr),
            weight_int8_,
            weight_scale_,
            static_cast<uint16_t*>(result_info.ptr),
            HIDDEN,
            VOCAB,
            NUM_THREADS
        );
        
        return result;
    }
    
    /**
     * @brief Compute LM head projection using NEON (single-threaded, FP32)
     * 
     * @param hidden_state FP32 input [HIDDEN_SIZE]
     * @return FP32 output [VOCAB_SIZE]
     */
    py::array_t<float> forward_fp32(
        py::array_t<float, py::array::c_style | py::array::forcecast> hidden_state
    ) {
        if (!weights_loaded_) {
            throw std::runtime_error("Weights not loaded");
        }
        
        auto info = hidden_state.request();
        if (static_cast<int>(info.shape[0]) != HIDDEN) {
            throw std::runtime_error("Input size mismatch");
        }
        
        // Allocate output
        auto result = py::array_t<float>(VOCAB);
        auto result_info = result.request();
        
        // Call NEON FP32 implementation
        lm_head_neon_fp32(
            static_cast<const float*>(info.ptr),
            weight_int8_,
            weight_scale_,
            static_cast<float*>(result_info.ptr),
            HIDDEN,
            VOCAB
        );
        
        return result;
    }
    
    /**
     * @brief Compute LM head projection using NEON (multi-threaded, FP32)
     * 
     * @param hidden_state FP32 input [HIDDEN_SIZE]
     * @return FP32 output [VOCAB_SIZE]
     */
    py::array_t<float> forward_fp32_mt(
        py::array_t<float, py::array::c_style | py::array::forcecast> hidden_state
    ) {
        if (!weights_loaded_) {
            throw std::runtime_error("Weights not loaded");
        }
        
        auto info = hidden_state.request();
        if (static_cast<int>(info.shape[0]) != HIDDEN) {
            throw std::runtime_error("Input size mismatch");
        }
        
        // Allocate output
        auto result = py::array_t<float>(VOCAB);
        auto result_info = result.request();
        
        // Call NEON FP32 multi-threaded implementation
        lm_head_neon_mt(
            static_cast<const float*>(info.ptr),
            weight_int8_,
            weight_scale_,
            static_cast<float*>(result_info.ptr),
            HIDDEN,
            VOCAB,
            NUM_THREADS
        );
        
        return result;
    }
    
    // Getters
    static constexpr int hidden_size() { return HIDDEN; }
    static constexpr int vocab_size() { return VOCAB; }
    bool is_loaded() const { return weights_loaded_; }
};

// ============================================================================
// PyBind11 Module Definition
// ============================================================================

PYBIND11_MODULE(lm_head_neon, m) {
    m.doc() = R"pbdoc(
        NEON-optimized LM Head for BitNet b1.58-large
        
        Fixed-size implementation:
        - HIDDEN_SIZE = 1536
        - VOCAB_SIZE = 32002
        - NUM_THREADS = 4
        
        Usage:
            import lm_head_neon
            lm = lm_head_neon.LMHeadNEON()
            lm.load_weights(weight_int32, scale_fp32)
            logits = lm.forward_fp16(hidden_state.view(np.uint16))
    )pbdoc";
    
    // Export constants
    m.attr("HIDDEN_SIZE") = HIDDEN_SIZE;
    m.attr("VOCAB_SIZE") = VOCAB_SIZE;
    m.attr("NUM_THREADS") = NUM_THREADS;
    
    // LM Head processor class
    py::class_<LMHeadNEON>(m, "LMHeadNEON")
        .def(py::init<>())
        .def("load_weights", &LMHeadNEON::load_weights,
             py::arg("weights"), py::arg("scales"),
             "Load weights: weights[VOCAB, HIDDEN] as int32, scales[VOCAB] as float32")
        .def("forward_fp16", &LMHeadNEON::forward_fp16,
             py::arg("hidden_state"),
             "Forward pass with FP16 I/O (single-threaded)")
        .def("forward_fp16_mt", &LMHeadNEON::forward_fp16_mt,
             py::arg("hidden_state"),
             "Forward pass with FP16 I/O (multi-threaded)")
        .def("forward_fp32", &LMHeadNEON::forward_fp32,
             py::arg("hidden_state"),
             "Forward pass with FP32 I/O (single-threaded)")
        .def("forward_fp32_mt", &LMHeadNEON::forward_fp32_mt,
             py::arg("hidden_state"),
             "Forward pass with FP32 I/O (multi-threaded)")
        .def_property_readonly_static("hidden_size", 
             [](py::object) { return LMHeadNEON::hidden_size(); })
        .def_property_readonly_static("vocab_size",
             [](py::object) { return LMHeadNEON::vocab_size(); })
        .def_property_readonly("is_loaded", &LMHeadNEON::is_loaded);
}
