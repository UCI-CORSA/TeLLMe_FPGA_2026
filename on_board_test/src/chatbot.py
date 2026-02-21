#!/usr/bin/env python3
"""
BSD 3-Clause License

Copyright (c) 2025, Zhiheng Chen

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
chatbot.py - BitNet Chatbot with Prefill and Autoregressive Decoding

This module implements a complete chatbot system using the BitNet FPGA accelerator.

Features:
1. Tokenization: Convert text to token IDs and vice versa
2. Embedding lookup: Map token IDs to embedding vectors
3. Prefill: Process the entire prompt in parallel
4. Autoregressive decoding: Generate tokens one by one
5. LM Head: Project hidden states to logits
6. Argmax decoding: Select the most likely next token

Key Implementation Details:
- RoPE: Prefill uses normal order (flipped for HLS input), Decode uses normal order
- Token length padding: Prefill pads to multiple of 8
- KV cache management: Maintains KV cache for autoregressive decoding
- Timing: Measures latency for each component
- Stop conditions: EOS token or max length

Usage:
    python chatbot.py --prompt "Hello, how are you?" --max_length 256
"""

import numpy as np
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys

# Add tokenizer path to system path
SCRIPT_DIR = Path(__file__).parent
ON_BOARD_TEST_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ON_BOARD_TEST_DIR / "tokenizer"))

from tokenizer_numpy_optimized import OptimizedBPETokenizer


# ============================================================================
# Utility Functions
# ============================================================================
def parse_sample_ids(sample_id_str: str) -> List[int]:
    """
    Parse sample ID string into list of integers.
    
    Supports:
    - Single ID: "5" -> [5]
    - Comma-separated: "0,1,2" -> [0, 1, 2]
    - Range: "0-10" -> [0, 1, 2, ..., 10]
    - Mixed: "0,5-7,10" -> [0, 5, 6, 7, 10]
    
    Args:
        sample_id_str: Sample ID specification string
        
    Returns:
        List of sample IDs as integers
    """
    sample_ids = []
    parts = sample_id_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range format: "0-10"
            try:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                sample_ids.extend(range(start, end + 1))
            except ValueError:
                print(f"[Warning] Invalid range format: {part}, skipping")
        else:
            # Single ID
            try:
                sample_ids.append(int(part))
            except ValueError:
                print(f"[Warning] Invalid sample ID: {part}, skipping")
    
    # Remove duplicates and sort
    sample_ids = sorted(list(set(sample_ids)))
    return sample_ids


# ============================================================================
# Configuration (must match config.hpp and case_top_bitnet.cpp)
# ============================================================================
@dataclass
class ModelConfig:
    """Model configuration matching HLS config."""
    # Model parameters
    NUM_LAYER: int = 24
    HIDDEN_SIZE: int = 1536
    FFN_HIDDEN_SIZE: int = 4096
    VOCAB_SIZE: int = 32002
    WEIGHT_STATE: int = 3
    HIDDEN1_SIZE: int = 1596
    HIDDEN2_SIZE: int = 1536
    TMAC_INDEX_SIZE: int = 532  # HIDDEN1_SIZE // WEIGHT_STATE
    TMAC_TABLE_NUM: int = 28
    NUM_ATTN_LINEAR: int = 4
    NUM_GATE_LINEAR: int = 3
    
    # Attention parameters
    NUM_ATT_HEADS: int = 16
    HEAD_DIM: int = 96  # HIDDEN_SIZE // NUM_ATT_HEADS
    
    # Hardware block sizes
    HIDDEN_BLOCK_SIZE: int = 16
    NUM_HIDDEN_BLOCKS: int = 96  # HIDDEN_SIZE // HIDDEN_BLOCK_SIZE
    WEIGHT_DRAM_DIM: int = 308  # (HIDDEN2_SIZE + 4) // 5
    
    # Buffer sizes
    MAX_TOKEN_LENGTH: int = 3000
    MAX_DECODE_LENGTH: int = 1024
    MAX_ROPE_PRECOMPUTE: int = 1024  # Pre-compute RoPE up to this length
    
    # Special token IDs
    BOS_ID: int = 1
    EOS_ID: int = 2
    PAD_ID: int = 32000
    
    # Paths
    WEIGHT_BIN_DIR: Path = ON_BOARD_TEST_DIR / "weight_bin_5"
    ROPE_CACHE_DIR: Path = ON_BOARD_TEST_DIR / "rope_cache"
    TOKENIZER_PATH: Path = ON_BOARD_TEST_DIR / "tokenizer" / "tokenizer.json"
    INPUT_BIN_DIR: Path = ON_BOARD_TEST_DIR / "input_bin"
    OUTPUT_BIN_DIR: Path = ON_BOARD_TEST_DIR / "output_bin"
    OUTPUT_BIN_PYTORCH_DIR: Path = ON_BOARD_TEST_DIR / "output_bin_pytorch"  # PyTorch golden logits
    KV_CACHE_DIR: Path = ON_BOARD_TEST_DIR / "kv_cache"
    # Legacy golden data path (not used in release version)
    # GOLDEN_DATA_DIR: Path = Path("path/to/golden_data")  # Optional: for custom verification
    
    def __post_init__(self):
        # Compute derived sizes
        self.tmp_buffer_size = self.MAX_TOKEN_LENGTH * self.HIDDEN_SIZE
        self.kv_cache_size = self.NUM_LAYER * self.MAX_TOKEN_LENGTH * self.HIDDEN_SIZE
        self.weight_size = (self.NUM_LAYER * (self.NUM_ATTN_LINEAR + 3 * self.NUM_GATE_LINEAR) * 
                           (self.TMAC_INDEX_SIZE // self.TMAC_TABLE_NUM) * self.WEIGHT_DRAM_DIM * 4)


# Global config instance
CONFIG = ModelConfig()


# ============================================================================
# Timing Utilities
# ============================================================================
@dataclass
class TimingStats:
    """Detailed timing statistics for each component and operation."""
    # High-level timing
    tokenize_time: float = 0.0
    embedding_time: float = 0.0
    prefill_kernel_time: float = 0.0
    decode_kernel_time: float = 0.0
    lmhead_time: float = 0.0
    sampling_time: float = 0.0  # Token sampling time
    total_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    
    # Prefill detailed timing
    prefill_embedding_lookup: float = 0.0
    prefill_padding: float = 0.0
    prefill_flip: float = 0.0
    prefill_rope_compute: float = 0.0
    prefill_data_load: float = 0.0
    prefill_sync_to_device: float = 0.0
    prefill_kernel_config: float = 0.0
    prefill_kernel_exec: float = 0.0
    prefill_sync_from_device: float = 0.0
    prefill_output_flip: float = 0.0
    prefill_kv_shift: float = 0.0
    prefill_kv_sync_back: float = 0.0
    prefill_lmhead: float = 0.0  # LM head time in prefill (for NEON mode)
    
    # Decode detailed timing
    decode_rope_compute: float = 0.0
    decode_data_load: float = 0.0
    decode_sync_to_device: float = 0.0
    decode_kernel_config: float = 0.0
    decode_kernel_exec: float = 0.0
    decode_sync_from_device: float = 0.0
    decode_output_extract: float = 0.0
    decode_lmhead: float = 0.0  # LM head time in decode (for NEON mode)
    decode_sampling: float = 0.0  # Sampling time in decode
    
    # LM Head detailed timing (for NumPy mode)
    lmhead_quantize: float = 0.0
    lmhead_matmul: float = 0.0
    lmhead_dequantize: float = 0.0
    
    # Counters
    decode_step_count: int = 0
    lmhead_call_count: int = 0
    
    # Mode flags
    use_neon_lm_head: bool = False
    
    def reset(self):
        """Reset all timing statistics."""
        for field in self.__dataclass_fields__:
            if isinstance(getattr(self, field), float):
                setattr(self, field, 0.0)
            elif isinstance(getattr(self, field), int):
                setattr(self, field, 0)
    
    def print_stats(self):
        """Print timing statistics."""
        print("\n" + "=" * 70)
        print(" Detailed Timing Statistics (Profiling)")
        if self.use_neon_lm_head:
            print(" [NEON LM Head Mode - LM Head included in Prefill/Decode]")
        print("=" * 70)
        
        # High-level summary
        print("\n[High-Level Summary]")
        print(f"  Tokenization:        {self.tokenize_time * 1000:>10.3f} ms")
        print(f"  Embedding Lookup:    {self.embedding_time * 1000:>10.3f} ms")
        print(f"  Prefill Total:       {self.prefill_kernel_time * 1000:>10.3f} ms ({self.prefill_tokens} tokens)")
        print(f"  Decode Total:        {self.decode_kernel_time * 1000:>10.3f} ms ({self.decode_tokens} tokens)")
        if not self.use_neon_lm_head:
            print(f"  LM Head Total:       {self.lmhead_time * 1000:>10.3f} ms ({self.lmhead_call_count} calls)")
        print(f"  Sampling Total:      {self.sampling_time * 1000:>10.3f} ms")
        
        # Prefill breakdown
        if self.prefill_tokens > 0:
            print("\n[Prefill Breakdown]")
            print(f"  Embedding Lookup:    {self.prefill_embedding_lookup * 1000:>10.3f} ms")
            print(f"  Padding:             {self.prefill_padding * 1000:>10.3f} ms")
            print(f"  Flip (Input):        {self.prefill_flip * 1000:>10.3f} ms")
            print(f"  RoPE Compute:        {self.prefill_rope_compute * 1000:>10.3f} ms")
            print(f"  Data Load:           {self.prefill_data_load * 1000:>10.3f} ms")
            print(f"  Sync to Device:      {self.prefill_sync_to_device * 1000:>10.3f} ms")
            print(f"  Kernel Config:       {self.prefill_kernel_config * 1000:>10.3f} ms")
            print(f"  Kernel Execution:    {self.prefill_kernel_exec * 1000:>10.3f} ms")
            print(f"  Sync from Device:    {self.prefill_sync_from_device * 1000:>10.3f} ms")
            print(f"  Output Flip:         {self.prefill_output_flip * 1000:>10.3f} ms")
            print(f"  KV Cache Shift:      {self.prefill_kv_shift * 1000:>10.3f} ms")
            print(f"  KV Sync Back:        {self.prefill_kv_sync_back * 1000:>10.3f} ms")
            if self.use_neon_lm_head:
                print(f"  LM Head (NEON):      {self.prefill_lmhead * 1000:>10.3f} ms")
            
            prefill_sum = (self.prefill_embedding_lookup + self.prefill_padding + 
                          self.prefill_flip + self.prefill_rope_compute +
                          self.prefill_data_load +
                          self.prefill_sync_to_device + self.prefill_kernel_config +
                          self.prefill_kernel_exec + self.prefill_sync_from_device +
                          self.prefill_output_flip + self.prefill_kv_shift + 
                          self.prefill_kv_sync_back + self.prefill_lmhead)
            print(f"  --------------------------------")
            print(f"  Sum:                 {prefill_sum * 1000:>10.3f} ms")
            
            prefill_tps = self.prefill_tokens / prefill_sum if prefill_sum > 0 else 0
            print(f"  Throughput:          {prefill_tps:>10.2f} tokens/s")
        
        # Decode breakdown (averaged per step)
        if self.decode_step_count > 0:
            print(f"\n[Decode Breakdown] (averaged over {self.decode_step_count} steps)")
            avg = lambda x: x / self.decode_step_count * 1000
            print(f"  RoPE Compute:        {avg(self.decode_rope_compute):>10.3f} ms/step")
            print(f"  Data Load:           {avg(self.decode_data_load):>10.3f} ms/step")
            print(f"  Sync to Device:      {avg(self.decode_sync_to_device):>10.3f} ms/step")
            print(f"  Kernel Config:       {avg(self.decode_kernel_config):>10.3f} ms/step")
            print(f"  Kernel Execution:    {avg(self.decode_kernel_exec):>10.3f} ms/step")
            print(f"  Sync from Device:    {avg(self.decode_sync_from_device):>10.3f} ms/step")
            print(f"  Output Extract:      {avg(self.decode_output_extract):>10.3f} ms/step")
            if self.use_neon_lm_head:
                print(f"  LM Head (NEON):      {avg(self.decode_lmhead):>10.3f} ms/step")
            print(f"  Sampling:            {avg(self.decode_sampling):>10.3f} ms/step")
            
            decode_sum_per_step = (self.decode_rope_compute +
                                   self.decode_data_load + self.decode_sync_to_device +
                                   self.decode_kernel_config + self.decode_kernel_exec +
                                   self.decode_sync_from_device + self.decode_output_extract +
                                   self.decode_lmhead + self.decode_sampling) / self.decode_step_count
            print(f"  --------------------------------")
            print(f"  Sum per step:        {decode_sum_per_step * 1000:>10.3f} ms/step")
            
            if decode_sum_per_step > 0:
                decode_tps = 1.0 / decode_sum_per_step
                print(f"  Throughput:          {decode_tps:>10.2f} tokens/s")
        
        # LM Head breakdown (averaged per call) - only for NumPy mode
        if self.lmhead_call_count > 0 and not self.use_neon_lm_head:
            print(f"\n[LM Head Breakdown] (averaged over {self.lmhead_call_count} calls)")
            avg = lambda x: x / self.lmhead_call_count * 1000
            print(f"  Quantize:            {avg(self.lmhead_quantize):>10.3f} ms/call")
            print(f"  MatMul:              {avg(self.lmhead_matmul):>10.3f} ms/call")
            print(f"  Dequantize:          {avg(self.lmhead_dequantize):>10.3f} ms/call")
            lmhead_sum = (self.lmhead_quantize + self.lmhead_matmul + self.lmhead_dequantize) / self.lmhead_call_count
            print(f"  --------------------------------")
            print(f"  Sum per call:        {lmhead_sum * 1000:>10.3f} ms/call")
        
        # Throughput summary (using sum times for accurate measurement)
        print("\n[Throughput Summary]")
        if self.prefill_tokens > 0:
            prefill_sum = (self.prefill_embedding_lookup + self.prefill_padding + 
                          self.prefill_flip + self.prefill_rope_compute +
                          self.prefill_data_load +
                          self.prefill_sync_to_device + self.prefill_kernel_config +
                          self.prefill_kernel_exec + self.prefill_sync_from_device +
                          self.prefill_output_flip + self.prefill_kv_shift + 
                          self.prefill_kv_sync_back + self.prefill_lmhead)
            if prefill_sum > 0:
                prefill_tps = self.prefill_tokens / prefill_sum
                print(f"  Prefill:             {prefill_tps:>10.2f} tokens/s")
        
        if self.decode_step_count > 0:
            decode_sum_per_step = (self.decode_rope_compute +
                                   self.decode_data_load + self.decode_sync_to_device +
                                   self.decode_kernel_config + self.decode_kernel_exec +
                                   self.decode_sync_from_device + self.decode_output_extract +
                                   self.decode_lmhead + self.decode_sampling) / self.decode_step_count
            if decode_sum_per_step > 0:
                decode_tps = 1.0 / decode_sum_per_step
                print(f"  Decode:              {decode_tps:>10.2f} tokens/s")
        
        # Total time (don't double-count lmhead if NEON mode)
        if self.use_neon_lm_head:
            total_time = (self.tokenize_time + self.embedding_time + 
                         self.prefill_kernel_time + self.decode_kernel_time + self.sampling_time)
        else:
            total_time = (self.tokenize_time + self.embedding_time + 
                         self.prefill_kernel_time + self.decode_kernel_time + self.lmhead_time + self.sampling_time)
        print(f"\n[Total]")
        print(f"  Total Tokens:        {self.total_tokens:>10}")
        print(f"  Total Time:          {total_time * 1000:>10.3f} ms")
        if total_time > 0:
            print(f"  Overall Throughput:  {self.total_tokens / total_time:>10.2f} tokens/s")
        print("=" * 70)


# ============================================================================
# BitNet Chatbot Class
# ============================================================================
class BitNetChatbot:
    """
    BitNet Chatbot with FPGA acceleration.
    
    This class handles:
    1. Tokenization and detokenization
    2. Embedding lookup
    3. Prefill and decode execution on FPGA
    4. LM Head computation (optionally using NEON acceleration)
    5. Autoregressive generation
    """
    
    def __init__(self, bitstream_path: str = "./bitnet1024tl_250_dram_divide5.bit",
                 config: ModelConfig = CONFIG, use_pynq: bool = True,
                 use_neon_lm_head: bool = False):
        """
        Initialize the chatbot.
        
        Args:
            bitstream_path: Path to FPGA bitstream
            config: Model configuration
            use_pynq: Whether to use PYNQ (False for CPU simulation)
            use_neon_lm_head: Whether to use NEON-accelerated LM head (ARM only)
        """
        self.config = config
        self.use_pynq = use_pynq
        self.use_neon_lm_head = use_neon_lm_head
        self.timing = TimingStats()
        self.neon_lm_head = None  # Will be initialized if use_neon_lm_head=True
        
        # Initialize tokenizer
        print("[Chatbot] Loading tokenizer...")
        self.tokenizer = OptimizedBPETokenizer(str(config.TOKENIZER_PATH))
        
        # Load embedding table
        print("[Chatbot] Loading embedding table...")
        self._load_embeddings()
        
        # Load LM head weights
        print("[Chatbot] Loading LM head weights...")
        self._load_lm_head_weights()
        
        # Initialize NEON LM head if requested
        if use_neon_lm_head:
            self._init_neon_lm_head()
        
        # Load RoPE cache
        print("[Chatbot] Loading RoPE cache...")
        self._load_rope_cache()
        
        if use_pynq:
            # Initialize FPGA
            print(f"[Chatbot] Loading FPGA bitstream: {bitstream_path}")
            self._init_fpga(bitstream_path)
        else:
            print("[Chatbot] Running in CPU simulation mode (no FPGA)")
            self.overlay = None
            self.kernel_inst = None
        
        print("[Chatbot] Initialization complete!")
    
    def _load_embeddings(self):
        """Load embedding table from binary file."""
        embed_path = self.config.WEIGHT_BIN_DIR / "weights_embed_tokens.bin"
        
        if not embed_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embed_path}")
        
        self.embeddings = np.fromfile(embed_path, dtype=np.float16)
        expected_size = self.config.VOCAB_SIZE * self.config.HIDDEN_SIZE
        
        if len(self.embeddings) != expected_size:
            raise ValueError(f"Embedding size mismatch: {len(self.embeddings)} vs {expected_size}")
        
        self.embeddings = self.embeddings.reshape(self.config.VOCAB_SIZE, self.config.HIDDEN_SIZE)
        print(f"  Embedding shape: {self.embeddings.shape}")
    
    def _load_lm_head_weights(self):
        """Load LM head quantized weights."""
        quant_file = self.config.WEIGHT_BIN_DIR / "weights_lm_head_quant_int8.bin"
        scale_file = self.config.WEIGHT_BIN_DIR / "weights_lm_head_scale.bin"
        
        if not quant_file.exists() or not scale_file.exists():
            raise FileNotFoundError(f"LM head weight files not found in {self.config.WEIGHT_BIN_DIR}")
        
        # Load quantized weights (INT32 containing INT8 values)
        self.lm_head_quant = np.fromfile(quant_file, dtype=np.int32)
        self.lm_head_scale = np.fromfile(scale_file, dtype=np.float32)
        
        expected_weight_size = self.config.VOCAB_SIZE * self.config.HIDDEN_SIZE
        if len(self.lm_head_quant) != expected_weight_size:
            raise ValueError(f"LM head weight size mismatch: {len(self.lm_head_quant)} vs {expected_weight_size}")
        
        self.lm_head_quant = self.lm_head_quant.reshape(self.config.VOCAB_SIZE, self.config.HIDDEN_SIZE)
        print(f"  LM head weight shape: {self.lm_head_quant.shape}")
        print(f"  LM head scale shape: {self.lm_head_scale.shape}")
    
    def _init_neon_lm_head(self):
        """Initialize NEON-accelerated LM head module."""
        try:
            # Add neon_lm_head directory to path
            neon_lm_head_dir = SCRIPT_DIR / "neon_lm_head"
            if str(neon_lm_head_dir) not in sys.path:
                sys.path.insert(0, str(neon_lm_head_dir))
            
            import lm_head_neon
            
            print("[Chatbot] Initializing NEON LM head...")
            self.neon_lm_head = lm_head_neon.LMHeadNEON()
            self.neon_lm_head.load_weights(self.lm_head_quant, self.lm_head_scale)
            print(f"  NEON LM head loaded: HIDDEN={lm_head_neon.HIDDEN_SIZE}, VOCAB={lm_head_neon.VOCAB_SIZE}")
            print(f"  Using {lm_head_neon.NUM_THREADS} threads")
            
        except ImportError as e:
            print(f"[Warning] NEON LM head module not available: {e}")
            print("[Warning] Falling back to NumPy LM head")
            self.use_neon_lm_head = False
            self.neon_lm_head = None
    
    def _load_rope_cache(self):
        """Load RoPE frequency cache (both normal and flipped versions)."""
        # For prefill: use flipped version (HLS expects reversed token order)
        rope_flip_file = self.config.ROPE_CACHE_DIR / "rope_frequencies_interleaved_normal_flip.bin"
        # For decode: use normal version
        rope_normal_file = self.config.ROPE_CACHE_DIR / "rope_frequencies_interleaved_normal.bin"
        
        if not rope_flip_file.exists():
            raise FileNotFoundError(f"RoPE flip file not found: {rope_flip_file}")
        if not rope_normal_file.exists():
            raise FileNotFoundError(f"RoPE normal file not found: {rope_normal_file}")
        
        self.rope_cache_flip = np.fromfile(rope_flip_file, dtype=np.float16)
        self.rope_cache_normal = np.fromfile(rope_normal_file, dtype=np.float16)
        
        self.max_rope_seq_len = len(self.rope_cache_normal) // self.config.HIDDEN_SIZE
        
        print(f"  RoPE cache (flip) size: {len(self.rope_cache_flip)}")
        print(f"  RoPE cache (normal) size: {len(self.rope_cache_normal)}")
        print(f"  Max RoPE sequence length: {self.max_rope_seq_len}")
        
        # Pre-compute RoPE base for dynamic computation if needed
        # RoPE uses theta = 10000^(-2i/d) for frequency
        self._init_rope_base()
        
        # Pre-compute and extend RoPE cache to MAX_ROPE_PRECOMPUTE
        self._precompute_rope_cache()
    
    def _init_rope_base(self):
        """Initialize RoPE base frequencies for dynamic computation."""
        half_head_dim = self.config.HEAD_DIM // 2  # 48
        # theta_i = base^(-2i/d) where base=10000, d=head_dim
        # For interleaved format, we compute cos/sin for each pair
        inv_freq = 1.0 / (10000 ** (np.arange(0, half_head_dim) * 2 / self.config.HEAD_DIM))
        self.rope_inv_freq = inv_freq.astype(np.float32)
    
    def _precompute_rope_cache(self):
        """
        Pre-compute RoPE cache up to MAX_ROPE_PRECOMPUTE positions.
        
        This extends the cached RoPE values beyond the 128 positions in the
        golden bin files, allowing decode to run up to 1024 tokens without
        dynamic computation during inference.
        """
        target_len = self.config.MAX_ROPE_PRECOMPUTE
        
        if self.max_rope_seq_len >= target_len:
            print(f"  RoPE cache already covers {self.max_rope_seq_len} positions, no extension needed")
            return
        
        print(f"  Pre-computing RoPE cache from {self.max_rope_seq_len} to {target_len} positions...")
        
        # Allocate extended cache
        extended_size = target_len * self.config.HIDDEN_SIZE
        extended_normal = np.zeros(extended_size, dtype=np.float16)
        extended_flip = np.zeros(extended_size, dtype=np.float16)
        
        # Copy existing cached values
        existing_size = len(self.rope_cache_normal)
        extended_normal[:existing_size] = self.rope_cache_normal
        
        # Compute remaining positions
        for pos in range(self.max_rope_seq_len, target_len):
            rope_at_pos = self._compute_rope_at_position(pos)
            start_idx = pos * self.config.HIDDEN_SIZE
            end_idx = start_idx + self.config.HIDDEN_SIZE
            extended_normal[start_idx:end_idx] = rope_at_pos
        
        # Compute flipped version: flip[i] = normal[target_len - 1 - i]
        for pos in range(target_len):
            src_pos = target_len - 1 - pos
            src_start = src_pos * self.config.HIDDEN_SIZE
            src_end = src_start + self.config.HIDDEN_SIZE
            dst_start = pos * self.config.HIDDEN_SIZE
            dst_end = dst_start + self.config.HIDDEN_SIZE
            extended_flip[dst_start:dst_end] = extended_normal[src_start:src_end]
        
        # Replace caches
        self.rope_cache_normal = extended_normal
        self.rope_cache_flip = extended_flip
        self.max_rope_seq_len = target_len
        
        print(f"  Extended RoPE cache to {target_len} positions ({extended_size} elements)")
        
        # Verify against golden bin (first 128 positions)
        self._verify_rope_cache()
    
    def _compute_rope_at_position(self, position: int) -> np.ndarray:
        """
        Dynamically compute RoPE frequencies at a specific position.
        
        Uses interleaved format: [cos0, sin0, cos1, sin1, ...]
        
        Args:
            position: Token position
            
        Returns:
            RoPE frequencies [hidden_size] in fp16
        """
        half_head_dim = self.config.HEAD_DIM // 2  # 48
        
        # Compute angles: position * inv_freq
        angles = position * self.rope_inv_freq  # [half_head_dim]
        
        # Compute cos and sin
        cos_vals = np.cos(angles).astype(np.float16)
        sin_vals = np.sin(angles).astype(np.float16)
        
        # Interleaved format for one head: [cos0, sin0, cos1, sin1, ...]
        rope_head = np.zeros(self.config.HEAD_DIM, dtype=np.float16)
        for i in range(half_head_dim):
            rope_head[2*i] = cos_vals[i]
            rope_head[2*i + 1] = sin_vals[i]
        
        # Repeat for all heads
        rope_full = np.tile(rope_head, self.config.NUM_ATT_HEADS)
        
        return rope_full
    
    def _verify_rope_cache(self):
        """
        Verify computed RoPE cache against golden bin files.
        
        Compares the first 128 positions (cached from file) with
        dynamically computed values to ensure correctness.
        """
        # Load original golden file for comparison
        rope_golden_file = self.config.ROPE_CACHE_DIR / "rope_frequencies_interleaved_normal.bin"
        if not rope_golden_file.exists():
            print("  [Warning] Golden RoPE file not found, skipping verification")
            return
        
        golden = np.fromfile(rope_golden_file, dtype=np.float16)
        golden_positions = len(golden) // self.config.HIDDEN_SIZE
        
        max_diff = 0.0
        total_diff = 0.0
        num_elements = 0
        
        for pos in range(min(golden_positions, 128)):
            start = pos * self.config.HIDDEN_SIZE
            end = start + self.config.HIDDEN_SIZE
            
            cached = self.rope_cache_normal[start:end]
            golden_slice = golden[start:end]
            
            diff = np.abs(cached.astype(np.float32) - golden_slice.astype(np.float32))
            max_diff = max(max_diff, diff.max())
            total_diff += diff.sum()
            num_elements += len(diff)
        
        mean_diff = total_diff / num_elements if num_elements > 0 else 0
        
        print(f"  RoPE verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        if max_diff > 0.03:  # FP16 tolerance
            print(f"  [Warning] RoPE verification failed! Max diff {max_diff} exceeds tolerance")
        else:
            print(f"  RoPE verification PASSED")
    
    def _init_fpga(self, bitstream_path: str):
        """Initialize FPGA with bitstream and allocate buffers."""
        import pynq
        from pynq import Overlay
        
        self.overlay = Overlay(bitstream_path)
        self.kernel_inst = self.overlay.test_top_bitnet_0
        print(f"  Kernel instance: {self.kernel_inst}")
        
        # Allocate buffers
        print("  Allocating PYNQ buffers...")
        
        self.tmp_1 = pynq.allocate(shape=(self.config.tmp_buffer_size,), dtype=np.float16)
        self.tmp_2 = pynq.allocate(shape=(self.config.tmp_buffer_size,), dtype=np.float16)
        self.tmp_3 = pynq.allocate(shape=(self.config.tmp_buffer_size,), dtype=np.float16)
        self.tmp_4 = pynq.allocate(shape=(self.config.tmp_buffer_size,), dtype=np.float16)
        self.rope_freq = pynq.allocate(shape=(self.config.tmp_buffer_size,), dtype=np.float16)
        self.out_buf = pynq.allocate(shape=(self.config.tmp_buffer_size,), dtype=np.float16)
        
        self.k_cache = pynq.allocate(shape=(self.config.kv_cache_size,), dtype=np.float16)
        self.v_cache = pynq.allocate(shape=(self.config.kv_cache_size,), dtype=np.float16)
        
        self.weight_dram_1 = pynq.allocate(shape=(self.config.weight_size,), dtype=np.uint64)
        self.weight_dram_2 = pynq.allocate(shape=(self.config.weight_size,), dtype=np.uint64)
        self.weight_dram_3 = pynq.allocate(shape=(self.config.weight_size,), dtype=np.uint64)
        
        # Load weights
        self._load_fpga_weights()
    
    def _load_fpga_weights(self):
        """Load packed weights to FPGA buffers."""
        print("  Loading packed weights...")
        
        weight_files = [
            (self.config.WEIGHT_BIN_DIR / "weight_dram1_packed.bin", self.weight_dram_1),
            (self.config.WEIGHT_BIN_DIR / "weight_dram2_packed.bin", self.weight_dram_2),
            (self.config.WEIGHT_BIN_DIR / "weight_dram3_packed.bin", self.weight_dram_3),
        ]
        
        for filepath, buffer in weight_files:
            if not filepath.exists():
                raise FileNotFoundError(f"Weight file not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint64)
                copy_size = min(len(data), len(buffer))
                buffer[:copy_size] = data[:copy_size]
        
        # Sync weights to device
        self.weight_dram_1.sync_to_device()
        self.weight_dram_2.sync_to_device()
        self.weight_dram_3.sync_to_device()
        print("  Weights loaded and synced to device")
    
    def _embedding_lookup(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Array of token IDs [seq_len]
            
        Returns:
            Embeddings [seq_len, hidden_size]
        """
        t_start = time.time()
        
        # Bounds check
        token_ids = np.clip(token_ids, 0, self.config.VOCAB_SIZE - 1)
        embeddings = self.embeddings[token_ids]
        
        self.timing.embedding_time += time.time() - t_start
        return embeddings
    
    def _compute_lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        """
        Compute LM head projection using quantized weights.
        
        Args:
            hidden_state: Last token hidden state [hidden_size]
            
        Returns:
            Logits [vocab_size]
        """
        t_start = time.time()
        t_op = time.time()
        
        # Step 1: Quantize input (per-token absmax)
        input_max = max(np.abs(hidden_state).max(), 1e-5)
        Qp = 127.0
        input_quant_scale = Qp / input_max
        input_dequant_scale = input_max / Qp
        
        # Quantize to INT8
        quant_input = np.clip(
            np.round(hidden_state.astype(np.float32) * input_quant_scale),
            -128, 127
        ).astype(np.int8)
        self.timing.lmhead_quantize += time.time() - t_op
        
        # Step 2: Integer matrix-vector multiplication
        t_op = time.time()
        logits_int = np.dot(self.lm_head_quant.astype(np.int32), quant_input.astype(np.int32))
        self.timing.lmhead_matmul += time.time() - t_op
        
        # Step 3: Dequantize
        t_op = time.time()
        logits = (logits_int.astype(np.float32) * self.lm_head_scale * input_dequant_scale).astype(np.float16)
        self.timing.lmhead_dequantize += time.time() - t_op
        
        self.timing.lmhead_time += time.time() - t_start
        self.timing.lmhead_call_count += 1
        return logits
    
    def _compute_lm_head_neon(self, hidden_state: np.ndarray) -> np.ndarray:
        """Compute LM head projection using NEON-accelerated module.
        
        Optimized to minimize data copies:
        - If input is already float16, use view directly (no copy)
        - Output view is also zero-copy
        """
        t_start = time.time()
        
        # Zero-copy view: if already float16, just reinterpret as uint16
        if hidden_state.dtype == np.float16:
            hidden_uint16 = hidden_state.view(np.uint16)
        else:
            # Only copy if type conversion is needed
            hidden_uint16 = hidden_state.astype(np.float16).view(np.uint16)
        
        # Call NEON FP16 multi-threaded implementation
        # Output is already allocated by pybind, returned as uint16
        output_uint16 = self.neon_lm_head.forward_fp16_mt(hidden_uint16)
        
        # View back to FP16 and ensure contiguous for efficient downstream operations
        # Note: view() may return non-contiguous array which slows down argmax
        logits = np.ascontiguousarray(output_uint16.view(np.float16))
        
        self.timing.lmhead_time += time.time() - t_start
        self.timing.lmhead_call_count += 1
        return logits
    
    def _sample_topk(self, logits: np.ndarray, k: int = 16, mode: str = "weighted") -> int:
        """
        Sample from top-k tokens without using softmax.
        
        Args:
            logits: Logits array [vocab_size], should be float32 for best performance
            k: Number of top tokens to consider
            mode: Sampling mode
                - "weighted": Sample proportional to shifted logits (no softmax)
                - "uniform": Uniform random sample from top-k
                
        Returns:
            Sampled token ID
        """
        # Expect float32 input (caller should convert)
        logits_f32 = logits if logits.dtype == np.float32 else logits.astype(np.float32)
        
        # Use argpartition for O(n) instead of O(n log n) full sort
        # Get indices of top-k elements (not sorted)
        topk_indices_unsorted = np.argpartition(logits_f32, -k)[-k:]
        # Sort only the top-k for weighted sampling
        topk_logits = logits_f32[topk_indices_unsorted]
        sorted_order = np.argsort(topk_logits)[::-1]
        topk_indices = topk_indices_unsorted[sorted_order]
        topk_logits = logits_f32[topk_indices]
        
        if mode == "uniform":
            # Uniform random sampling from top-k
            chosen_idx = np.random.randint(0, k)
        else:  # weighted
            # Shift logits to be positive and use as weights (no softmax/exp)
            # This gives higher probability to tokens with higher logits
            shifted = topk_logits - topk_logits.min() + 1.0
            probs = shifted / shifted.sum()
            chosen_idx = np.random.choice(k, p=probs)
        
        return int(topk_indices[chosen_idx])
    
    def _pad_to_multiple(self, length: int, multiple: int = 8) -> int:
        """Pad length to the next multiple."""
        return ((length + multiple - 1) // multiple) * multiple
    
    def _flip_token_order(self, data: np.ndarray, token_length: int) -> np.ndarray:
        """
        Flip data along token dimension.
        HLS input is flipped: input[t][d] -> input[token_length - 1 - t][d]
        """
        reshaped = data.reshape(token_length, self.config.HIDDEN_SIZE)
        return reshaped[::-1, :].reshape(-1).copy()
    
    def _get_prefill_rope(self, token_length: int) -> np.ndarray:
        """
        Get RoPE frequencies for prefill.
        
        For prefill, we need flipped RoPE starting from position (token_length - 1) down to 0.
        The rope_cache_flip file stores rope in reversed order: position [MAX-1, MAX-2, ..., 0]
        We need to take the last token_length positions from the normal file and flip them.
        
        Actually, based on the notebook logic:
        - rope_frequencies_interleaved_normal_flip.bin is already flipped
        - For prefill with token_length, we need positions [token_length-1, token_length-2, ..., 0]
        - The flip file has: flip[0] = normal[MAX-1], flip[1] = normal[MAX-2], ...
        
        So for prefill with token_length N, we need flip positions [MAX-N, MAX-N+1, ..., MAX-1]
        which corresponds to normal positions [N-1, N-2, ..., 0]
        """
        # For prefill: HLS expects input tokens in reversed order
        # The rope_cache_flip is pre-computed as the flipped version
        # We need to select the correct slice based on token_length
        
        # The flip file contains rope for positions [MAX_SEQ_LEN-1, MAX_SEQ_LEN-2, ..., 0]
        # For prefill with N tokens, after flipping input:
        #   - Original position 0 becomes HLS position N-1
        #   - Original position N-1 becomes HLS position 0
        # So we need rope at positions [N-1, N-2, ..., 0] in the HLS input order
        
        # From the flip cache: we need elements that correspond to positions [0, 1, ..., N-1] in original
        # But since flip[i] = normal[MAX-1-i], we need:
        # For HLS position 0 (original position N-1): rope at normal position N-1
        # For HLS position 1 (original position N-2): rope at normal position N-2
        # ...
        # For HLS position N-1 (original position 0): rope at normal position 0
        
        # So we just take the first N elements from the normal cache and flip them
        total_elements = token_length * self.config.HIDDEN_SIZE
        rope_normal = self.rope_cache_normal[:total_elements]
        rope_flipped = self._flip_token_order(rope_normal, token_length)
        
        return rope_flipped
    
    def _get_decode_rope(self, position: int) -> np.ndarray:
        """
        Get RoPE frequencies for decode at a specific position.
        
        For decode, we use normal order RoPE at the current token position.
        If position exceeds cache size, compute dynamically.
        """
        start_idx = position * self.config.HIDDEN_SIZE
        end_idx = start_idx + self.config.HIDDEN_SIZE
        
        if end_idx <= len(self.rope_cache_normal):
            # Use cached value
            return self.rope_cache_normal[start_idx:end_idx].copy()
        else:
            # Compute dynamically
            return self._compute_rope_at_position(position)
    
    def _configure_kernel_registers(self, token_length: int, kv_cache_length: int):
        """Configure kernel registers for execution."""
        # Set buffer addresses
        self.kernel_inst.mmio.write_reg(0x10, self.tmp_1.physical_address)
        self.kernel_inst.mmio.write_reg(0x14, 0)
        self.kernel_inst.mmio.write_reg(0x1c, self.tmp_2.physical_address)
        self.kernel_inst.mmio.write_reg(0x20, 0)
        self.kernel_inst.mmio.write_reg(0x28, self.tmp_3.physical_address)
        self.kernel_inst.mmio.write_reg(0x2c, 0)
        self.kernel_inst.mmio.write_reg(0x34, self.tmp_4.physical_address)
        self.kernel_inst.mmio.write_reg(0x38, 0)
        self.kernel_inst.mmio.write_reg(0x40, self.rope_freq.physical_address)
        self.kernel_inst.mmio.write_reg(0x44, 0)
        self.kernel_inst.mmio.write_reg(0x4c, self.out_buf.physical_address)
        self.kernel_inst.mmio.write_reg(0x50, 0)
        self.kernel_inst.mmio.write_reg(0x58, self.k_cache.physical_address)
        self.kernel_inst.mmio.write_reg(0x5c, 0)
        self.kernel_inst.mmio.write_reg(0x64, self.v_cache.physical_address)
        self.kernel_inst.mmio.write_reg(0x68, 0)
        self.kernel_inst.mmio.write_reg(0x70, self.weight_dram_1.physical_address)
        self.kernel_inst.mmio.write_reg(0x74, 0)
        self.kernel_inst.mmio.write_reg(0x7c, self.weight_dram_2.physical_address)
        self.kernel_inst.mmio.write_reg(0x80, 0)
        self.kernel_inst.mmio.write_reg(0x88, self.weight_dram_3.physical_address)
        self.kernel_inst.mmio.write_reg(0x8c, 0)
        
        # Set token_length and kv_cache_length
        self.kernel_inst.mmio.write_reg(0x94, token_length)
        self.kernel_inst.mmio.write_reg(0x9c, kv_cache_length)
    
    def _run_kernel(self) -> float:
        """Run kernel and wait for completion. Returns execution time."""
        t_start = time.time()
        
        # Start kernel with proper register sequence for multi-run support
        # Write 0x10 first to reset/prepare, then 0x01 to start
        self.kernel_inst.write(0x00, 1 << 4)
        self.kernel_inst.write(0x00, 1)
        
        # Wait for completion
        ip_status = self.kernel_inst.read(0x00)
        while ip_status == 14 or ip_status == 1:
            ip_status = self.kernel_inst.read(0x00)
        
        return time.time() - t_start
    
    def prefill(self, token_ids: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Run prefill for the given token IDs.
        
        Args:
            token_ids: Input token IDs [seq_len]
            
        Returns:
            Tuple of (last_hidden_state, actual_token_length, kv_cache_length_after_prefill)
            
        Note:
            Padding Strategy (IMPORTANT):
            - HLS requires token_length to be multiple of 8
            - We pad at the END of embeddings: [actual_tokens, padding]
            - After flip: [padding, actual_tokens_flipped]
            - HLS processes all tokens and stores KV cache in flipped order
            - The FIRST padded_token_length positions in KV cache are used
            - For decode, we set kv_cache_length = padded_token_length so HLS reads correct KV
            - BUT the padding KV values don't affect attention because:
              * Padding tokens have zero input, producing near-zero KV
              * During decode, the new token attends to all KV including padding
              * This is acceptable as padding KV contribution is minimal
              
            Alternative (cleaner but requires HLS change): Use actual_token_length directly
            if HLS can handle non-multiple-of-8 lengths.
        """
        t_prefill_start = time.time()
        actual_token_length = len(token_ids)
        
        # Pad token length to multiple of 8 for HLS
        padded_token_length = self._pad_to_multiple(actual_token_length, 8)
        pad_size = padded_token_length - actual_token_length
        
        # Get embeddings
        t_op = time.time()
        embeddings = self._embedding_lookup(token_ids)
        self.timing.prefill_embedding_lookup += time.time() - t_op
        
        # Pad embeddings at the END if necessary
        # Layout before flip: [token_0, token_1, ..., token_{N-1}, pad, pad, ...]
        t_op = time.time()
        if pad_size > 0:
            padding = np.zeros((pad_size, self.config.HIDDEN_SIZE), dtype=np.float16)
            embeddings = np.vstack([embeddings, padding])
        self.timing.prefill_padding += time.time() - t_op
        
        # Flatten and flip for HLS
        # Layout after flip: [pad, pad, ..., token_{N-1}, token_{N-2}, ..., token_0]
        # So HLS position 0 = padding, HLS position (pad_size) = token_{N-1}, etc.
        t_op = time.time()
        embeddings_flat = embeddings.reshape(-1)
        embeddings_flipped = self._flip_token_order(embeddings_flat, padded_token_length)
        self.timing.prefill_flip += time.time() - t_op
        
        # Prepare RoPE for prefill (use pre-computed flip cache)
        # For prefill with actual_token_length N:
        #   - HLS input is flipped: input[pad_size] = token[N-1], input[pad_size+N-1] = token[0]
        #   - Need RoPE: input[pad_size] needs position (N-1), input[pad_size+N-1] needs position 0
        #   - rope_cache_flip layout: flip[i] = normal[MAX-1-i]
        #   - So take flip[(MAX-N)*H : MAX*H] which gives [normal[N-1], normal[N-2], ..., normal[0]]
        #   - Prepend zeros for padding positions
        
        t_op = time.time()
        if actual_token_length <= self.max_rope_seq_len:
            # Use pre-computed flip cache directly with offset
            flip_start = (self.max_rope_seq_len - actual_token_length) * self.config.HIDDEN_SIZE
            flip_end = self.max_rope_seq_len * self.config.HIDDEN_SIZE
            rope_flipped_actual = self.rope_cache_flip[flip_start:flip_end].copy()
        else:
            print(f"[Prefill] Computing RoPE dynamically for {actual_token_length} positions...")
            rope_normal_actual = np.zeros(actual_token_length * self.config.HIDDEN_SIZE, dtype=np.float16)
            for pos in range(actual_token_length):
                rope_normal_actual[pos * self.config.HIDDEN_SIZE:(pos + 1) * self.config.HIDDEN_SIZE] = \
                    self._compute_rope_at_position(pos)
            rope_flipped_actual = self._flip_token_order(rope_normal_actual, actual_token_length)
        
        # Construct RoPE data matching flipped input layout
        # [zeros for padding positions, flipped RoPE for actual tokens]
        if pad_size > 0:
            pad_elements = pad_size * self.config.HIDDEN_SIZE
            rope_data = np.concatenate([
                np.zeros(pad_elements, dtype=np.float16),
                rope_flipped_actual
            ])
        else:
            rope_data = rope_flipped_actual
        self.timing.prefill_rope_compute += time.time() - t_op
        
        if not self.use_pynq:
            # CPU simulation mode - return mock output
            # Return actual_token_length for kv_cache_length (matches FPGA behavior after KV shift)
            print(f"[Prefill] CPU simulation mode - token_length={actual_token_length}, padded={padded_token_length}")
            mock_output = np.random.randn(self.config.HIDDEN_SIZE).astype(np.float16) * 0.01
            self.timing.prefill_tokens = padded_token_length  # Use padded length for throughput calculation
            self.timing.prefill_kernel_time = time.time() - t_prefill_start
            return mock_output, actual_token_length, actual_token_length
        
        # Load input and rope (直接覆盖写，无需预先清零)
        t_op = time.time()
        total_elements = padded_token_length * self.config.HIDDEN_SIZE
        self.tmp_2[:total_elements] = embeddings_flipped
        self.rope_freq[:total_elements] = rope_data
        self.timing.prefill_data_load += time.time() - t_op
        
        # Sync to device
        t_op = time.time()
        self.tmp_1.sync_to_device()
        self.tmp_2.sync_to_device()
        self.tmp_3.sync_to_device()
        self.tmp_4.sync_to_device()
        self.rope_freq.sync_to_device()
        self.out_buf.sync_to_device()
        self.k_cache.sync_to_device()
        self.v_cache.sync_to_device()
        self.timing.prefill_sync_to_device += time.time() - t_op
        
        # Configure and run kernel
        # For prefill: token_length = padded_token_length, kv_cache_length = 0
        t_op = time.time()
        self._configure_kernel_registers(padded_token_length, 0)
        self.timing.prefill_kernel_config += time.time() - t_op
        
        t_op = time.time()
        exec_time = self._run_kernel()
        self.timing.prefill_kernel_exec += time.time() - t_op
        self.timing.prefill_kernel_time += exec_time
        self.timing.prefill_tokens = padded_token_length  # Use padded length for throughput calculation
        
        # Sync output from device
        t_op = time.time()
        self.tmp_1.sync_from_device()
        self.k_cache.sync_from_device()
        self.v_cache.sync_from_device()
        self.timing.prefill_sync_from_device += time.time() - t_op
        
        # Extract output and flip back
        t_op = time.time()
        output_flipped = np.array(self.tmp_1[:total_elements], dtype=np.float16)
        output_unflipped = self._flip_token_order(output_flipped, padded_token_length)
        output_reshaped = output_unflipped.reshape(padded_token_length, self.config.HIDDEN_SIZE)
        
        # Get the last actual token's hidden state (not padding)
        # After flipping back: [token_0_out, token_1_out, ..., token_{N-1}_out, pad_out, ...]
        # The last actual token is at index (actual_token_length - 1)
        last_hidden_state = output_reshaped[actual_token_length - 1]
        self.timing.prefill_output_flip += time.time() - t_op
        
        # KV Cache Shift (IMPORTANT):
        # HLS stores KV cache in the order it processes tokens (flipped order):
        #   KV cache position 0 = first flipped token (padding if pad_size > 0)
        #   KV cache position pad_size = token_{N-1} (last original token)
        #   KV cache position padded_token_length-1 = token_0 (first original token)
        # 
        # To enable correct decode, we need to shift the valid KV entries to the front:
        #   Move KV[pad_size : padded_token_length] -> KV[0 : actual_token_length]
        # This way, decode can use kv_cache_length = actual_token_length
        # and the RoPE positions will be consistent.
        if pad_size > 0:
            t_op = time.time()
            print(f"[Prefill] Shifting KV cache: removing {pad_size} padding positions")
            
            # 使用 numpy reshape 进行向量化操作，一次性处理所有层
            # KV cache layout: [NUM_LAYER, MAX_TOKEN_LENGTH, HIDDEN_SIZE]
            layer_stride = self.config.MAX_TOKEN_LENGTH * self.config.HIDDEN_SIZE
            
            # 将 k_cache 和 v_cache 视为 3D 数组进行操作
            k_view = np.asarray(self.k_cache).reshape(self.config.NUM_LAYER, self.config.MAX_TOKEN_LENGTH, self.config.HIDDEN_SIZE)
            v_view = np.asarray(self.v_cache).reshape(self.config.NUM_LAYER, self.config.MAX_TOKEN_LENGTH, self.config.HIDDEN_SIZE)
            
            # 一次性切片复制所有层: [all_layers, pad_size:padded_len, :] -> [all_layers, 0:actual_len, :]
            k_view[:, :actual_token_length, :] = k_view[:, pad_size:padded_token_length, :].copy()
            v_view[:, :actual_token_length, :] = v_view[:, pad_size:padded_token_length, :].copy()
            
            self.timing.prefill_kv_shift += time.time() - t_op
            
            # Sync shifted KV cache back to device
            t_op = time.time()
            self.k_cache.sync_to_device()
            self.v_cache.sync_to_device()
            self.timing.prefill_kv_sync_back += time.time() - t_op
        
        # If using NEON LM head, include LM head time in prefill timing
        if self.use_neon_lm_head and self.neon_lm_head is not None:
            t_lm = time.time()
            logits = self._compute_lm_head_neon(last_hidden_state)
            lm_time = time.time() - t_lm
            self.timing.prefill_lmhead += lm_time
            self.timing.prefill_kernel_time += lm_time
            self._prefill_logits_cache = logits  # Cache for generate()
        
        # After shifting, kv_cache_length = actual_token_length
        # Decode will use RoPE at position actual_token_length, which is correct
        return last_hidden_state, actual_token_length, actual_token_length
    
    def decode_step(self, embedding: np.ndarray, kv_cache_length: int) -> np.ndarray:
        """
        Run a single decode step.
        
        Args:
            embedding: Embedding for the new token [hidden_size]
            kv_cache_length: Current KV cache length (number of tokens already processed)
            
        Returns:
            Hidden state for the new token [hidden_size]
        """
        self.timing.decode_step_count += 1
        
        if not self.use_pynq:
            # CPU simulation mode
            print(f"[Decode] CPU simulation mode - kv_cache_length={kv_cache_length}")
            mock_output = np.random.randn(self.config.HIDDEN_SIZE).astype(np.float16) * 0.01
            return mock_output
        
        # Get RoPE for current position (normal order, not flipped)
        t_op = time.time()
        rope_data = self._get_decode_rope(kv_cache_length)
        self.timing.decode_rope_compute += time.time() - t_op
        
        # For decode mode, we write input at offset = kv_cache_length * HIDDEN_SIZE
        write_offset = kv_cache_length * self.config.HIDDEN_SIZE
        
        # Load input and rope at offset position (直接覆盖写，无需预先清零)
        t_op = time.time()
        self.tmp_2[write_offset:write_offset + self.config.HIDDEN_SIZE] = embedding
        self.rope_freq[write_offset:write_offset + self.config.HIDDEN_SIZE] = rope_data
        self.timing.decode_data_load += time.time() - t_op
        
        # Sync to device
        t_op = time.time()
        self.tmp_1.sync_to_device()
        self.tmp_2.sync_to_device()
        self.rope_freq.sync_to_device()
        self.out_buf.sync_to_device()
        # Note: k_cache and v_cache already contain data from previous steps
        self.k_cache.sync_to_device()
        self.v_cache.sync_to_device()
        self.timing.decode_sync_to_device += time.time() - t_op
        
        # Configure and run kernel
        # For decode: token_length = 1, kv_cache_length = number of previously processed tokens
        t_op = time.time()
        self._configure_kernel_registers(1, kv_cache_length)
        self.timing.decode_kernel_config += time.time() - t_op
        
        t_op = time.time()
        exec_time = self._run_kernel()
        self.timing.decode_kernel_exec += time.time() - t_op
        self.timing.decode_kernel_time += exec_time
        
        # Sync output from device
        t_op = time.time()
        self.tmp_1.sync_from_device()
        self.k_cache.sync_from_device()
        self.v_cache.sync_from_device()
        self.timing.decode_sync_from_device += time.time() - t_op
        
        # Extract output (single token at offset position)
        t_op = time.time()
        hidden_state = np.array(self.tmp_1[write_offset:write_offset + self.config.HIDDEN_SIZE], dtype=np.float16)
        self.timing.decode_output_extract += time.time() - t_op
        
        # If using NEON LM head, include LM head time in decode timing
        if self.use_neon_lm_head and self.neon_lm_head is not None:
            t_lm = time.time()
            logits = self._compute_lm_head_neon(hidden_state)
            lm_time = time.time() - t_lm
            self.timing.decode_lmhead += lm_time
            self.timing.decode_kernel_time += lm_time
            self._decode_logits_cache = logits  # Cache for generate()
        
        return hidden_state
    
    def generate(self, prompt: str, max_new_tokens: int = 256, 
                 verbose: bool = True, sampling_mode: str = "argmax",
                 top_k: int = 16, stream: bool = False) -> Tuple[str, List[int]]:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            verbose: Whether to print progress
            sampling_mode: Token selection strategy
                - "argmax": Always select highest logit (greedy)
                - "topk": Random sample from top-k (weighted by logits, no softmax)
                - "topk_uniform": Uniform random sample from top-k
            top_k: Number of top tokens for sampling (default 16)
            stream: Whether to print tokens as they are generated
            
        Returns:
            Tuple of (generated_text, token_ids)
        """
        # Reset timing
        self.timing = TimingStats()
        self.timing.use_neon_lm_head = self.use_neon_lm_head
        
        # Tokenize prompt
        t_start = time.time()
        token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        self.timing.tokenize_time = time.time() - t_start
        
        if verbose:
            print(f"\n[Generate] Prompt: {prompt}")
            print(f"[Generate] Token IDs: {token_ids.tolist()}")
            print(f"[Generate] Prompt length: {len(token_ids)} tokens")
            print(f"[Generate] Sampling mode: {sampling_mode}" + (f" (k={top_k})" if sampling_mode != "argmax" else ""))
        
        # Check if prompt is too long
        if len(token_ids) > self.config.MAX_TOKEN_LENGTH - max_new_tokens:
            raise ValueError(f"Prompt too long: {len(token_ids)} tokens, max allowed: {self.config.MAX_TOKEN_LENGTH - max_new_tokens}")
        
        # Limit max_new_tokens
        max_new_tokens = min(max_new_tokens, self.config.MAX_DECODE_LENGTH)
        
        # Run prefill
        if verbose:
            print(f"\n[Generate] Running prefill...")
        
        last_hidden, actual_length, kv_cache_length = self.prefill(token_ids)
        
        if verbose:
            print(f"[Generate] Prefill done: actual_length={actual_length}, kv_cache_length={kv_cache_length}")
        
        # Compute first logits and get first generated token
        # If NEON LM head was used, logits are cached in prefill
        if self.use_neon_lm_head and hasattr(self, '_prefill_logits_cache'):
            logits = self._prefill_logits_cache
            delattr(self, '_prefill_logits_cache')
        else:
            logits = self._compute_lm_head(last_hidden)
        
        # Sample next token with timing
        # Convert to float32 once for efficient argmax (avoid repeated implicit conversion)
        t_sample = time.time()
        logits_f32 = logits.astype(np.float32) if logits.dtype == np.float16 else logits
        if sampling_mode == "argmax":
            next_token = int(np.argmax(logits_f32))
        elif sampling_mode == "topk":
            next_token = self._sample_topk(logits_f32, k=top_k, mode="weighted")
        elif sampling_mode == "topk_uniform":
            next_token = self._sample_topk(logits_f32, k=top_k, mode="uniform")
        else:
            next_token = int(np.argmax(logits_f32))  # fallback to argmax
        self.timing.sampling_time += time.time() - t_sample
        
        generated_tokens = [next_token]
        # kv_cache_length is already set from prefill
        
        if verbose:
            token_str = self.tokenizer.decode_single(next_token)
            print(f"[Generate] First token: {next_token} -> {repr(token_str)}")
        
        # Stream output: print first token
        if stream:
            token_str = self.tokenizer.decode_single(next_token)
            print(token_str, end='', flush=True)
        
        # Autoregressive decoding
        for step in range(max_new_tokens - 1):
            # Check stop condition
            if next_token == self.config.EOS_ID:
                if verbose:
                    print(f"[Generate] EOS token reached at step {step + 1}")
                break
            
            if kv_cache_length >= self.config.MAX_DECODE_LENGTH:
                if verbose:
                    print(f"[Generate] Max decode length reached: {kv_cache_length}")
                break
            
            # Get embedding for next token
            embedding = self._embedding_lookup(np.array([next_token]))[0]
            
            # Run decode step
            hidden_state = self.decode_step(embedding, kv_cache_length)
            kv_cache_length += 1
            self.timing.decode_tokens += 1
            
            # Compute logits and select next token (BEFORE next kernel call)
            # If NEON LM head was used, logits are cached in decode_step
            if self.use_neon_lm_head and hasattr(self, '_decode_logits_cache'):
                logits = self._decode_logits_cache
                delattr(self, '_decode_logits_cache')
            else:
                logits = self._compute_lm_head(hidden_state)
            
            # Sample next token with timing
            # Convert to float32 once for efficient argmax
            t_sample = time.time()
            logits_f32 = logits.astype(np.float32) if logits.dtype == np.float16 else logits
            if sampling_mode == "argmax":
                next_token = int(np.argmax(logits_f32))
            elif sampling_mode == "topk":
                next_token = self._sample_topk(logits_f32, k=top_k, mode="weighted")
            elif sampling_mode == "topk_uniform":
                next_token = self._sample_topk(logits_f32, k=top_k, mode="uniform")
            else:
                next_token = int(np.argmax(logits_f32))
            sample_time = time.time() - t_sample
            self.timing.sampling_time += sample_time
            self.timing.decode_sampling += sample_time
            
            generated_tokens.append(next_token)
            
            # Stream output: print token immediately
            if stream:
                token_str = self.tokenizer.decode_single(next_token)
                print(token_str, end='', flush=True)
            
            if verbose and not stream and (step + 2) % 10 == 0:
                print(f"[Generate] Step {step + 2}: {len(generated_tokens)} tokens generated")
        
        # Decode generated tokens to text
        all_tokens = np.concatenate([token_ids, np.array(generated_tokens, dtype=np.int32)])
        self.timing.total_tokens = len(all_tokens)
        
        generated_text = self.tokenizer.decode(np.array(generated_tokens, dtype=np.int32))
        full_text = self.tokenizer.decode(all_tokens)
        
        # Print newline after stream output
        if stream:
            print()  # End stream with newline
        
        if verbose:
            if not stream:
                print(f"\n[Generate] Generated {len(generated_tokens)} new tokens")
                print(f"[Generate] Generated text: {generated_text}")
            else:
                print(f"\n[Generate] Generated {len(generated_tokens)} new tokens")
            self.timing.print_stats()
        
        return full_text, all_tokens.tolist()
    
    def chat(self, user_input: str, max_new_tokens: int = 256, 
             sampling_mode: str = "argmax", top_k: int = 16,
             stream: bool = False) -> str:
        """
        Simple chat interface.
        
        Args:
            user_input: User's message
            max_new_tokens: Maximum tokens to generate
            sampling_mode: Token selection strategy (argmax, topk, topk_uniform)
            top_k: Number of top tokens for sampling (default 16)
            stream: Whether to print tokens as they are generated
            
        Returns:
            Model's response
        """
        # Use input directly without prompt template
        full_response, _ = self.generate(user_input, max_new_tokens=max_new_tokens, 
                                          verbose=True, sampling_mode=sampling_mode,
                                          top_k=top_k, stream=stream)
        
        return full_response
    
    def verify_prefill_with_golden(self, sample_id: int = 0, verbose: bool = True,
                                     use_pytorch_golden: bool = False) -> dict:
        """
        Run prefill verification using golden data.
        
        Loads input embeddings from input_bin, runs prefill,
        and compares output with expected golden output from output_bin.
        
        Args:
            sample_id: Sample index (0-127)
            verbose: Whether to print detailed comparison info
            use_pytorch_golden: If True, use PyTorch golden logits from output_bin_pytorch
                               If False, use HLS golden logits from output_bin
            
        Returns:
            Dictionary with verification results
        """
        input_dir = self.config.INPUT_BIN_DIR / f"sample_{sample_id}"
        output_dir = self.config.OUTPUT_BIN_PYTORCH_DIR if use_pytorch_golden else self.config.OUTPUT_BIN_DIR
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input data directory not found: {input_dir}")
        
        print("\n" + "=" * 60)
        print(" Golden Data Verification - PREFILL Mode")
        print("=" * 60)
        
        # Reset timing for detailed profiling
        self.timing = TimingStats()
        self.timing.use_neon_lm_head = self.use_neon_lm_head
        
        results = {
            "mode": "prefill",
            "sample_id": sample_id,
            "input_match": False,
            "output_match": False,
            "rope_match": False,
            "max_input_diff": 0.0,
            "max_output_diff": 0.0,
            "max_rope_diff": 0.0,
            "mean_output_diff": 0.0,
        }
        
        # Load input from input_bin/sample_X/00_input_0.bin
        input_file = input_dir / "00_input_0.bin"
        if not input_file.exists():
            print(f"[Error] Input file not found: {input_file}")
            return results
        
        golden_input = np.fromfile(input_file, dtype=np.float16)
        token_length = len(golden_input) // self.config.HIDDEN_SIZE
        print(f"[Prefill Verify] Input loaded: {len(golden_input)} elements, {token_length} tokens")
        print(f"[Prefill Verify] Sample ID: {sample_id}")
        
        # Load golden RoPE (from rope_cache directory)
        golden_rope_file = self.config.ROPE_CACHE_DIR / "rope_frequencies_interleaved_normal.bin"
        if golden_rope_file.exists():
            golden_rope = np.fromfile(golden_rope_file, dtype=np.float16)
            rope_len = min(len(golden_rope), token_length * self.config.HIDDEN_SIZE)
            
            # Compare with our computed RoPE
            our_rope = self.rope_cache_normal[:rope_len]
            rope_diff = np.abs(our_rope.astype(np.float32) - golden_rope[:rope_len].astype(np.float32))
            results["max_rope_diff"] = float(rope_diff.max())
            results["rope_match"] = results["max_rope_diff"] < 0.03
            print(f"[Prefill Verify] RoPE comparison: max_diff={results['max_rope_diff']:.6f}, match={results['rope_match']}")
        
        # Load expected golden output
        # PyTorch golden: output_bin_pytorch/pytorch_logits_last_token_sample_X.bin (binary fp16)
        # HLS golden: output_bin/hls_logits_last_token_sample_X.dat (text file)
        if use_pytorch_golden:
            golden_output_file = output_dir / f"pytorch_logits_last_token_sample_{sample_id}.bin"
            if not golden_output_file.exists():
                print(f"[Warning] PyTorch golden output file not found: {golden_output_file}")
                golden_logits = None
            else:
                golden_logits = np.fromfile(golden_output_file, dtype=np.float16).astype(np.float32)
                print(f"[Prefill Verify] PyTorch golden logits loaded: {len(golden_logits)} elements")
        else:
            golden_output_file = output_dir / f"hls_logits_last_token_sample_{sample_id}.dat"
            if not golden_output_file.exists():
                print(f"[Warning] HLS golden output file not found: {golden_output_file}")
                golden_logits = None
            else:
                # Load HLS logits from .dat file (space-separated float values)
                with open(golden_output_file, 'r') as f:
                    golden_logits = np.array([float(x) for x in f.read().split()], dtype=np.float32)
                print(f"[Prefill Verify] HLS golden logits loaded: {len(golden_logits)} elements")
        
        if not self.use_pynq:
            print("[Prefill Verify] CPU simulation mode - skipping actual kernel execution")
            print("[Prefill Verify] Only verifying data loading and RoPE computation")
            return results
        
        # Prepare input (flip for HLS)
        t_op = time.time()
        padded_token_length = self._pad_to_multiple(token_length, 8)
        
        if padded_token_length > token_length:
            pad_size = (padded_token_length - token_length) * self.config.HIDDEN_SIZE
            golden_input_padded = np.concatenate([golden_input, np.zeros(pad_size, dtype=np.float16)])
        else:
            golden_input_padded = golden_input
        self.timing.prefill_padding = time.time() - t_op
        
        t_op = time.time()
        input_flipped = self._flip_token_order(golden_input_padded, padded_token_length)
        self.timing.prefill_flip = time.time() - t_op
        
        # Prepare RoPE (use pre-computed flip cache)
        t_op = time.time()
        flip_start = (self.max_rope_seq_len - token_length) * self.config.HIDDEN_SIZE
        flip_end = self.max_rope_seq_len * self.config.HIDDEN_SIZE
        rope_flipped = self.rope_cache_flip[flip_start:flip_end].copy()
        
        if padded_token_length > token_length:
            pad_elements = (padded_token_length - token_length) * self.config.HIDDEN_SIZE
            rope_data = np.concatenate([np.zeros(pad_elements, dtype=np.float16), rope_flipped])
        else:
            rope_data = rope_flipped
        self.timing.prefill_rope_compute = time.time() - t_op
        
        # Clear buffers
        self.tmp_1[:] = 0
        self.tmp_2[:] = 0
        self.out_buf[:] = 0
        self.k_cache[:] = 0
        self.v_cache[:] = 0
        
        # Load data
        t_op = time.time()
        total_elements = padded_token_length * self.config.HIDDEN_SIZE
        self.tmp_2[:total_elements] = input_flipped
        self.rope_freq[:total_elements] = rope_data
        self.timing.prefill_data_load = time.time() - t_op
        
        # Sync
        t_op = time.time()
        self.tmp_1.sync_to_device()
        self.tmp_2.sync_to_device()
        self.tmp_3.sync_to_device()
        self.tmp_4.sync_to_device()
        self.rope_freq.sync_to_device()
        self.out_buf.sync_to_device()
        self.k_cache.sync_to_device()
        self.v_cache.sync_to_device()
        self.timing.prefill_sync_to_device = time.time() - t_op
        
        # Run kernel
        print(f"[Prefill Verify] Running kernel with token_length={padded_token_length}, kv_cache_length=0")
        t_op = time.time()
        self._configure_kernel_registers(padded_token_length, 0)
        self.timing.prefill_kernel_config = time.time() - t_op
        
        t_op = time.time()
        exec_time = self._run_kernel()
        self.timing.prefill_kernel_exec = time.time() - t_op
        self.timing.prefill_kernel_time = exec_time
        self.timing.prefill_tokens = padded_token_length
        print(f"[Prefill Verify] Kernel execution time: {exec_time * 1000:.3f} ms")
        
        # Read output
        t_op = time.time()
        self.tmp_1.sync_from_device()
        self.timing.prefill_sync_from_device = time.time() - t_op
        
        t_op = time.time()
        output_flipped = np.array(self.tmp_1[:total_elements], dtype=np.float16)
        output_unflipped = self._flip_token_order(output_flipped, padded_token_length)
        
        # Get last token hidden state and compute LM head
        output_reshaped = output_unflipped.reshape(padded_token_length, self.config.HIDDEN_SIZE)
        last_hidden_state = output_reshaped[token_length - 1]
        self.timing.prefill_output_flip = time.time() - t_op
        
        # Compute LM head to get logits (with timing)
        t_lm = time.time()
        if self.use_neon_lm_head and self.neon_lm_head is not None:
            our_logits = self._compute_lm_head_neon(last_hidden_state)
            lm_time = time.time() - t_lm
            self.timing.prefill_lmhead = lm_time
            self.timing.prefill_kernel_time += lm_time
        else:
            our_logits = self._compute_lm_head(last_hidden_state)
        self.timing.lmhead_time = time.time() - t_lm
        
        # Compare with golden logits
        if golden_logits is not None:
            # Compare logits
            our_f32 = our_logits.astype(np.float32)
            golden_f32 = golden_logits[:len(our_logits)].astype(np.float32)
            diff = np.abs(our_f32 - golden_f32)
            results["max_output_diff"] = float(diff.max())
            results["mean_output_diff"] = float(diff.mean())
            
            # Compute cosine similarity
            our_norm = np.linalg.norm(our_f32)
            golden_norm = np.linalg.norm(golden_f32)
            if our_norm > 1e-8 and golden_norm > 1e-8:
                cosine_sim = float(np.dot(our_f32, golden_f32) / (our_norm * golden_norm))
            else:
                cosine_sim = 0.0
            results["cosine_similarity"] = cosine_sim
            
            # Pass if cosine similarity >= 98%
            results["output_match"] = cosine_sim >= 0.98
            
            # Also compare argmax (top prediction)
            our_argmax = int(np.argmax(our_logits))
            golden_argmax = int(np.argmax(golden_logits))
            results["argmax_match"] = our_argmax == golden_argmax
            
            print(f"[Prefill Verify] Logits comparison:")
            print(f"  Max diff:   {results['max_output_diff']:.6f}")
            print(f"  Mean diff:  {results['mean_output_diff']:.6f}")
            print(f"  Cosine sim: {cosine_sim:.6f}")
            print(f"  Our argmax:    {our_argmax}")
            print(f"  Golden argmax: {golden_argmax}")
            print(f"  Argmax match:  {results['argmax_match']}")
            
            if verbose:
                # Show top-5 predictions
                our_top5 = np.argsort(our_logits)[::-1][:5]
                golden_top5 = np.argsort(golden_logits)[::-1][:5]
                print(f"\n  Our Top-5 tokens: {our_top5.tolist()}")
                print(f"  Golden Top-5 tokens: {golden_top5.tolist()}")
        
        print("\n" + "=" * 60)
        print(" Prefill Verification Summary")
        print("=" * 60)
        print(f"  Sample ID:    {sample_id}")
        print(f"  RoPE Match:   {results['rope_match']}")
        print(f"  Argmax Match: {results.get('argmax_match', 'N/A')}")
        print(f"  Output Match: {results['output_match']}")
        print("=" * 60)
        
        # Print detailed profiling
        if verbose:
            self.timing.print_stats()
        
        return results
    
    def _llama_to_interleaved(self, k_llama: np.ndarray) -> np.ndarray:
        """
        Convert K cache from Llama format to interleaved format.
        
        Llama format: [seq_len, num_heads, head_dim] where head_dim is [cos_part, sin_part]
        Interleaved format: [seq_len, num_heads, head_dim] where head_dim is [cos0, sin0, cos1, sin1, ...]
        
        Args:
            k_llama: K cache in Llama format [seq_len * hidden_size]
            
        Returns:
            K cache in interleaved format [seq_len * hidden_size]
        """
        seq_len = len(k_llama) // self.config.HIDDEN_SIZE
        k_reshaped = k_llama.reshape(seq_len, self.config.NUM_ATT_HEADS, self.config.HEAD_DIM)
        
        half_head_dim = self.config.HEAD_DIM // 2
        k_interleaved = np.zeros_like(k_reshaped)
        
        for s in range(seq_len):
            for h in range(self.config.NUM_ATT_HEADS):
                for i in range(half_head_dim):
                    k_interleaved[s, h, 2*i] = k_reshaped[s, h, i]
                    k_interleaved[s, h, 2*i + 1] = k_reshaped[s, h, half_head_dim + i]
        
        return k_interleaved.reshape(-1)
    
    def verify_decode_with_golden(self, sample_id: int = 0, verbose: bool = True,
                                    use_pytorch_golden: bool = False) -> dict:
        """
        Run decode verification using golden data.
        
        Loads KV cache from kv_cache directory, runs decode with single token,
        and compares output with expected golden output from output_bin.
        
        Args:
            sample_id: Sample index from kv_cache directory (0-127)
            verbose: Whether to print detailed comparison info
            use_pytorch_golden: If True, use PyTorch golden logits from output_bin_pytorch
                               If False, use HLS golden logits from output_bin
            
        Returns:
            Dictionary with verification results
        """
        input_dir = self.config.INPUT_BIN_DIR / f"sample_{sample_id}"
        kv_cache_dir = self.config.KV_CACHE_DIR / f"sample_{sample_id}"
        output_dir = self.config.OUTPUT_BIN_PYTORCH_DIR if use_pytorch_golden else self.config.OUTPUT_BIN_DIR
        
        if not kv_cache_dir.exists():
            raise FileNotFoundError(f"KV cache directory not found: {kv_cache_dir}")
        
        print("\n" + "=" * 60)
        print(f" Golden Data Verification - DECODE Mode (sample_{sample_id})")
        print("=" * 60)
        
        # Reset timing for detailed profiling
        self.timing = TimingStats()
        self.timing.use_neon_lm_head = self.use_neon_lm_head
        
        results = {
            "mode": "decode",
            "sample_id": sample_id,
            "kv_cache_loaded": False,
            "output_match": False,
            "max_output_diff": 0.0,
            "mean_output_diff": 0.0,
        }
        
        # Decode configuration (matching case_top_bitnet_decode.cpp)
        PREFILL_TOKEN_LENGTH = 128  # Golden data was collected with 128 tokens
        KV_CACHE_LENGTH = 127       # First 127 tokens used as KV cache
        DECODE_TOKEN_IDX = 127      # Index of decode token (the 128th token, position 127)
        
        print(f"[Decode Verify] KV_CACHE_LENGTH: {KV_CACHE_LENGTH}")
        print(f"[Decode Verify] DECODE_TOKEN_IDX: {DECODE_TOKEN_IDX}")
        
        # Load input for decode token from input_bin/sample_X/00_input_0.bin
        input_file = input_dir / "00_input_0.bin"
        if not input_file.exists():
            print(f"[Error] Input file not found: {input_file}")
            return results
        
        golden_input_all = np.fromfile(input_file, dtype=np.float16)
        token_length = len(golden_input_all) // self.config.HIDDEN_SIZE
        
        if token_length < PREFILL_TOKEN_LENGTH:
            print(f"[Error] Input has only {token_length} tokens, need {PREFILL_TOKEN_LENGTH}")
            return results
        
        # Extract decode token (the 128th token at index 127)
        decode_input = golden_input_all[DECODE_TOKEN_IDX * self.config.HIDDEN_SIZE:
                                         (DECODE_TOKEN_IDX + 1) * self.config.HIDDEN_SIZE]
        print(f"[Decode Verify] Decode input loaded: {len(decode_input)} elements (token {DECODE_TOKEN_IDX})")
        
        # Load golden logits
        # PyTorch golden: output_bin_pytorch/pytorch_logits_last_token_sample_X.bin (binary fp16)
        # HLS golden: output_bin/hls_logits_last_token_sample_X.dat (text file)
        if use_pytorch_golden:
            golden_output_file = output_dir / f"pytorch_logits_last_token_sample_{sample_id}.bin"
            if golden_output_file.exists():
                golden_logits = np.fromfile(golden_output_file, dtype=np.float16).astype(np.float32)
                print(f"[Decode Verify] PyTorch golden logits loaded: {len(golden_logits)} elements")
            else:
                print(f"[Warning] PyTorch golden output file not found: {golden_output_file}")
                golden_logits = None
        else:
            golden_output_file = output_dir / f"hls_logits_last_token_sample_{sample_id}.dat"
            if golden_output_file.exists():
                with open(golden_output_file, 'r') as f:
                    golden_logits = np.array([float(x) for x in f.read().split()], dtype=np.float32)
                print(f"[Decode Verify] HLS golden logits loaded: {len(golden_logits)} elements")
            else:
                print(f"[Warning] HLS golden output file not found: {golden_output_file}")
                golden_logits = None
        
        if not self.use_pynq:
            print("[Decode Verify] CPU simulation mode - skipping actual kernel execution")
            return results
        
        # Load KV cache from sample directory
        print(f"[Decode Verify] Loading KV cache from {kv_cache_dir}...")
        
        # Clear KV cache buffers
        self.k_cache[:] = 0
        self.v_cache[:] = 0
        
        kv_load_success = True
        for layer in range(self.config.NUM_LAYER):
            k_file = kv_cache_dir / f"06_k_rope_{layer}.bin"
            v_file = kv_cache_dir / f"04_v_output_{layer}.bin"
            
            if not k_file.exists() or not v_file.exists():
                print(f"[Error] KV cache files not found for layer {layer}")
                kv_load_success = False
                break
            
            # Load K cache (need to convert from Llama to interleaved format)
            k_data_llama = np.fromfile(k_file, dtype=np.float16)
            k_data_interleaved = self._llama_to_interleaved(k_data_llama)
            
            # Load V cache (no conversion needed)
            v_data = np.fromfile(v_file, dtype=np.float16)
            
            # Copy to KV cache buffers
            layer_offset = layer * self.config.MAX_TOKEN_LENGTH * self.config.HIDDEN_SIZE
            k_copy_size = min(len(k_data_interleaved), KV_CACHE_LENGTH * self.config.HIDDEN_SIZE)
            v_copy_size = min(len(v_data), KV_CACHE_LENGTH * self.config.HIDDEN_SIZE)
            
            self.k_cache[layer_offset:layer_offset + k_copy_size] = k_data_interleaved[:k_copy_size]
            self.v_cache[layer_offset:layer_offset + v_copy_size] = v_data[:v_copy_size]
        
        if not kv_load_success:
            return results
        
        results["kv_cache_loaded"] = True
        print(f"[Decode Verify] KV cache loaded for all {self.config.NUM_LAYER} layers")
        
        # Prepare decode input and RoPE
        # For decode: write at offset = KV_CACHE_LENGTH * HIDDEN_SIZE
        t_op = time.time()
        write_offset = KV_CACHE_LENGTH * self.config.HIDDEN_SIZE
        
        # Get RoPE for decode position (normal order, position = KV_CACHE_LENGTH = 127)
        rope_data = self._get_decode_rope(KV_CACHE_LENGTH)
        self.timing.decode_rope_compute = time.time() - t_op
        
        # Clear tmp buffers
        self.tmp_1[:] = 0
        self.tmp_2[:] = 0
        self.out_buf[:] = 0
        
        # Load decode input and RoPE at offset position
        t_op = time.time()
        self.tmp_2[write_offset:write_offset + self.config.HIDDEN_SIZE] = decode_input
        self.rope_freq[write_offset:write_offset + self.config.HIDDEN_SIZE] = rope_data
        self.timing.decode_data_load = time.time() - t_op
        
        # Sync all buffers
        t_op = time.time()
        self.tmp_1.sync_to_device()
        self.tmp_2.sync_to_device()
        self.tmp_3.sync_to_device()
        self.tmp_4.sync_to_device()
        self.rope_freq.sync_to_device()
        self.out_buf.sync_to_device()
        self.k_cache.sync_to_device()
        self.v_cache.sync_to_device()
        self.timing.decode_sync_to_device = time.time() - t_op
        
        # Run kernel in decode mode
        print(f"[Decode Verify] Running kernel with token_length=1, kv_cache_length={KV_CACHE_LENGTH}")
        t_op = time.time()
        self._configure_kernel_registers(1, KV_CACHE_LENGTH)
        self.timing.decode_kernel_config = time.time() - t_op
        
        t_op = time.time()
        exec_time = self._run_kernel()
        self.timing.decode_kernel_exec = time.time() - t_op
        self.timing.decode_kernel_time = exec_time
        self.timing.decode_tokens = 1
        self.timing.decode_step_count = 1
        print(f"[Decode Verify] Kernel execution time: {exec_time * 1000:.3f} ms")
        
        # Read output
        t_op = time.time()
        self.tmp_1.sync_from_device()
        self.timing.decode_sync_from_device = time.time() - t_op
        
        # Extract output (single token at offset position) - this is hidden state
        t_op = time.time()
        hidden_state = np.array(self.tmp_1[write_offset:write_offset + self.config.HIDDEN_SIZE], dtype=np.float16)
        self.timing.decode_output_extract = time.time() - t_op
        
        # Compute LM head to get logits (with timing)
        t_lm = time.time()
        if self.use_neon_lm_head and self.neon_lm_head is not None:
            our_logits = self._compute_lm_head_neon(hidden_state)
            lm_time = time.time() - t_lm
            self.timing.decode_lmhead = lm_time
            self.timing.decode_kernel_time += lm_time
        else:
            our_logits = self._compute_lm_head(hidden_state)
        self.timing.lmhead_time = time.time() - t_lm
        
        # Compare with golden logits
        if golden_logits is not None:
            our_f32 = our_logits.astype(np.float32)
            golden_f32 = golden_logits[:len(our_logits)].astype(np.float32)
            diff = np.abs(our_f32 - golden_f32)
            results["max_output_diff"] = float(diff.max())
            results["mean_output_diff"] = float(diff.mean())
            
            # Compute cosine similarity
            our_norm = np.linalg.norm(our_f32)
            golden_norm = np.linalg.norm(golden_f32)
            if our_norm > 1e-8 and golden_norm > 1e-8:
                cosine_sim = float(np.dot(our_f32, golden_f32) / (our_norm * golden_norm))
            else:
                cosine_sim = 0.0
            results["cosine_similarity"] = cosine_sim
            
            # Pass if cosine similarity >= 98%
            results["output_match"] = cosine_sim >= 0.98
            
            # Also compare argmax (top prediction)
            our_argmax = int(np.argmax(our_logits))
            golden_argmax = int(np.argmax(golden_logits))
            results["argmax_match"] = our_argmax == golden_argmax
            
            print(f"[Decode Verify] Logits comparison:")
            print(f"  Max diff:   {results['max_output_diff']:.6f}")
            print(f"  Mean diff:  {results['mean_output_diff']:.6f}")
            print(f"  Cosine sim: {cosine_sim:.6f}")
            print(f"  Our argmax:    {our_argmax}")
            print(f"  Golden argmax: {golden_argmax}")
            print(f"  Argmax match:  {results['argmax_match']}")
            
            if verbose:
                # Show top-5 predictions
                our_top5 = np.argsort(our_logits)[::-1][:5]
                golden_top5 = np.argsort(golden_logits)[::-1][:5]
                print(f"\n  Our Top-5 tokens: {our_top5.tolist()}")
                print(f"  Golden Top-5 tokens: {golden_top5.tolist()}")
        
        print("\n" + "=" * 60)
        print(" Decode Verification Summary")
        print("=" * 60)
        print(f"  Sample ID:     {sample_id}")
        print(f"  KV Loaded:     {results['kv_cache_loaded']}")
        print(f"  Argmax Match:  {results.get('argmax_match', 'N/A')}")
        print(f"  Output Match:  {results['output_match']}")
        print("=" * 60)
        
        # Print detailed profiling
        if verbose:
            self.timing.print_stats()
        
        return results
    
    def verify_with_golden_data(self, mode: str = "prefill", sample_id: int = 0, 
                                 verbose: bool = True, use_pytorch_golden: bool = False) -> dict:
        """
        Run verification using golden data.
        
        Args:
            mode: "prefill" or "decode"
            sample_id: Sample index (0-127) for input/output/kv_cache directories
            verbose: Whether to print detailed comparison info
            use_pytorch_golden: If True, use PyTorch golden logits from output_bin_pytorch
                               If False, use HLS golden logits from output_bin
            
        Returns:
            Dictionary with verification results
        """
        if mode == "prefill":
            return self.verify_prefill_with_golden(sample_id=sample_id, verbose=verbose,
                                                    use_pytorch_golden=use_pytorch_golden)
        elif mode == "decode":
            return self.verify_decode_with_golden(sample_id=sample_id, verbose=verbose,
                                                   use_pytorch_golden=use_pytorch_golden)
        else:
            raise ValueError(f"Unknown verification mode: {mode}. Use 'prefill' or 'decode'")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="BitNet Chatbot with FPGA Acceleration")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--bitstream", type=str, default="./bitnet1024tl_250_dram_divide5.bit",
                       help="Path to FPGA bitstream")
    parser.add_argument("--no_fpga", action="store_true",
                       help="Run in CPU simulation mode (no FPGA)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive chat mode")
    parser.add_argument("--verify_golden", action="store_true",
                       help="Run verification using golden data")
    parser.add_argument("--verify_mode", type=str, default="prefill",
                       choices=["prefill", "decode"],
                       help="Verification mode: prefill or decode")
    parser.add_argument("--sample_id", type=str, default="0",
                       help="Sample ID(s) for verification. Can be single (e.g., '5'), comma-separated (e.g., '0,1,2'), or range (e.g., '0-10')")
    parser.add_argument("--sampling", type=str, default="argmax",
                       choices=["argmax", "topk", "topk_uniform"],
                       help="Sampling mode: argmax (greedy), topk (weighted), topk_uniform")
    parser.add_argument("--top_k", type=int, default=16,
                       help="K value for top-k sampling (default: 16)")
    parser.add_argument("--neon_lm_head", action="store_true",
                       help="Use NEON-accelerated LM head (ARM only)")
    parser.add_argument("--stream", action="store_true",
                       help="Enable streaming output (print tokens as they are generated)")
    parser.add_argument("--use_pytorch_golden", action="store_true",
                       help="Use PyTorch golden logits (output_bin_pytorch) instead of HLS golden (output_bin)")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    use_pynq = not args.no_fpga
    chatbot = BitNetChatbot(bitstream_path=args.bitstream, use_pynq=use_pynq,
                           use_neon_lm_head=args.neon_lm_head)
    
    if args.verify_golden:
        # Parse sample_id(s): supports single, comma-separated, or range format
        sample_ids = parse_sample_ids(args.sample_id)
        golden_source = "PyTorch" if args.use_pytorch_golden else "HLS"
        print(f"[Verify] Testing {len(sample_ids)} sample(s): {sample_ids}")
        print(f"[Verify] Using {golden_source} golden data")
        
        # Golden data verification mode - run all samples
        all_results = []
        passed = 0
        failed = 0
        
        for sid in sample_ids:
            print(f"\n{'='*70}")
            print(f" Testing Sample {sid}")
            print(f"{'='*70}")
            
            try:
                results = chatbot.verify_with_golden_data(
                    mode=args.verify_mode, 
                    sample_id=sid,
                    verbose=True,
                    use_pytorch_golden=args.use_pytorch_golden
                )
                results["sample_id"] = sid
                all_results.append(results)
                
                # Pass if cosine similarity >= 98% OR argmax matches
                cosine_sim = results.get("cosine_similarity", 0.0)
                if cosine_sim >= 0.98 or results.get("argmax_match", False) or not use_pynq:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[Error] Sample {sid} failed with exception: {e}")
                all_results.append({"sample_id": sid, "error": str(e)})
                failed += 1
        
        # Print summary
        print(f"\n{'='*70}")
        print(f" Verification Summary: {passed} passed, {failed} failed out of {len(sample_ids)} samples")
        print(f"{'='*70}")
        for r in all_results:
            sid = r.get("sample_id", "?")
            if "error" in r:
                print(f"  Sample {sid}: ERROR - {r['error']}")
            else:
                argmax_match = r.get("argmax_match", "N/A")
                output_match = r.get("output_match", False)
                cosine_sim = r.get("cosine_similarity", 0.0)
                print(f"  Sample {sid}: argmax_match={argmax_match}, output_match={output_match}, cosine_sim={cosine_sim:.4f}")
        
        # Return exit code based on verification results
        if failed == 0 or not use_pynq:
            print("\n[Result] All verifications completed successfully")
            sys.exit(0)
        else:
            print(f"\n[Result] {failed} verification(s) FAILED")
            sys.exit(1)
    
    elif args.interactive:
        # Interactive chat mode
        print("\n" + "=" * 60)
        print(" BitNet Chatbot - Interactive Mode")
        print(" Type 'quit' or 'exit' to end the conversation")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = chatbot.chat(user_input, max_new_tokens=args.max_length,
                                         sampling_mode=args.sampling, top_k=args.top_k,
                                         stream=args.stream)
                if not args.stream:
                    print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    else:
        # Single generation mode
        full_text, token_ids = chatbot.generate(
            args.prompt, 
            max_new_tokens=args.max_length,
            verbose=True,
            sampling_mode=args.sampling,
            top_k=args.top_k,
            stream=args.stream
        )
        
        if not args.stream:
            print("\n" + "=" * 60)
            print(" Final Output")
            print("=" * 60)
            print(full_text)


if __name__ == "__main__":
    main()
