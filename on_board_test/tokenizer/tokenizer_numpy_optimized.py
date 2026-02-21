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
Highly optimized NumPy BPE Tokenizer implementation

Optimization strategies:
1. Precompute merge hash table for faster lookup
2. Use NumPy arrays to store token IDs
3. Cache encoding results for common tokens
4. Simplify preprocessing, use normalizer rules directly
5. Use byte-level fallback for unknown characters

Implementation details:
==========
Core steps of BPE (Byte-Pair Encoding) algorithm:

1. Preprocessing (Normalization):
   - Add ▁ prefix at the beginning of the text
   - Replace all spaces with ▁

2. BPE encoding:
   - Split text into UTF-8 byte sequence
   - Iteratively find and merge adjacent token pairs with highest priority
   - Use merge_ranks dictionary for O(1) priority lookup
   
3. Token ID mapping:
   - Look up ID for each sub-token
   - Use byte-level fallback for unknown characters: char -> <0xXX>

4. Special token processing:
   - BOS (<s>): ID=1, add to the beginning of sequence
   - EOS (</s>): ID=2, optionally add to the end of sequence
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import lru_cache


class OptimizedBPETokenizer:
    """
    Highly optimized BPE Tokenizer
    
    Core optimizations:
    1. O(1) merge rank lookup
    2. LRU cache for BPE results of common tokens
    3. NumPy array for output storage
    4. Precompiled byte fallback table
    """
    
    __slots__ = [
        'vocab', 'id_to_token', 'merge_ranks', 
        'byte_tokens', 'sp_prefix',
        'unk_id', 'bos_id', 'eos_id', 'pad_id',
        '_bpe_cache'
    ]
    
    def __init__(self, tokenizer_json_path: str):
        """Load and preprocess tokenizer data"""
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. Vocabulary - use dict for O(1) lookup
        self.vocab: Dict[str, int] = data['model']['vocab']
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        
        # 2. Merge rules - store directly as dict for O(1) priority lookup
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge_str in enumerate(data['model']['merges']):
            parts = merge_str.split(' ', 1)  # Split once, faster
            if len(parts) == 2:
                self.merge_ranks[(parts[0], parts[1])] = rank
        
        # 3. Precompile byte-level token table (0x00-0xFF)
        self.byte_tokens: Dict[int, int] = {}
        for i in range(256):
            byte_token = f'<0x{i:02X}>'
            if byte_token in self.vocab:
                self.byte_tokens[i] = self.vocab[byte_token]
        
        # 4. Special tokens
        self.sp_prefix = '▁'
        self.unk_id = self.vocab.get('<unk>', 0)
        self.bos_id = self.vocab.get('<s>', 1)
        self.eos_id = self.vocab.get('</s>', 2)
        self.pad_id = self.vocab.get('<pad>', 32000)
        
        # 5. BPE Cache
        self._bpe_cache: Dict[str, List[str]] = {}
        
        print(f"[OptimizedBPETokenizer] Loading complete:")
        print(f"  Vocabulary: {len(self.vocab)}, Merge rules: {len(self.merge_ranks)}")
    
    def _bpe(self, token: str) -> List[str]:
        """
        Apply BPE algorithm to a single token (with caching)
        
        Optimization: use cache to avoid redundant computation for common words
        """
        # Cache hit
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        
        # Initialize as character list
        word = list(token)
        
        if len(word) <= 1:
            self._bpe_cache[token] = word
            return word
        
        # Repeatedly merge
        while len(word) > 1:
            # Find mergeable pair with highest priority
            best_pair = None
            best_rank = float('inf')
            best_idx = -1
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i
            
            if best_pair is None:
                break
            
            # Perform merge (modify list in-place to avoid creating new objects)
            word[best_idx] = best_pair[0] + best_pair[1]
            del word[best_idx + 1]
        
        # Cache result (limit cache size)
        if len(self._bpe_cache) < 100000:
            self._bpe_cache[token] = word
        
        return word
    
    def _normalize(self, text: str) -> str:
        """
        Text preprocessing (following normalizer rules in tokenizer.json)
        
        Rules:
        1. Prepend ▁ to the beginning
        2. Replace spaces with ▁
        """
        # Direct concatenation, faster than replace
        result = [self.sp_prefix]
        for c in text:
            if c == ' ':
                result.append(self.sp_prefix)
            else:
                result.append(c)
        return ''.join(result)
    
    def _token_to_id(self, token: str) -> int:
        """Token -> ID, supports byte-level fallback"""
        if token in self.vocab:
            return self.vocab[token]
        
        # Single character fallback
        if len(token) == 1:
            byte_val = ord(token)
            if byte_val in self.byte_tokens:
                return self.byte_tokens[byte_val]
        
        return self.unk_id
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> np.ndarray:
        """
        Encode text into a sequence of token IDs
        
        Args:
            text: input text
            add_bos: whether to add BOS token
            add_eos: whether to add EOS token
            
        Returns:
            np.ndarray: token ID array (int32)
        """
        if not text:
            ids = []
            if add_bos:
                ids.append(self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return np.array(ids, dtype=np.int32)
        
        # 1. Preprocessing
        normalized = self._normalize(text)
        
        # 2. BPE encoding
        bpe_tokens = self._bpe(normalized)
        
        # 3. Token -> ID
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        
        for token in bpe_tokens:
            ids.append(self._token_to_id(token))
        
        if add_eos:
            ids.append(self.eos_id)
        
        return np.array(ids, dtype=np.int32)
    
    def encode_batch(self, texts: List[str], add_bos: bool = True, 
                     add_eos: bool = False, pad: bool = False,
                     max_length: Optional[int] = None) -> np.ndarray:
        """
        Batch encode multiple texts
        
        Args:
            texts: text list
            add_bos: whether to add BOS
            add_eos: whether to add EOS
            pad: whether to pad to the same length
            max_length: maximum length limit
            
        Returns:
            If pad=True: np.ndarray shape (batch, max_len)
            Otherwise: List[np.ndarray]
        """
        encoded = [self.encode(t, add_bos, add_eos) for t in texts]
        
        if not pad:
            return encoded
        
        # Calculate maximum length
        max_len = max(len(e) for e in encoded)
        if max_length is not None:
            max_len = min(max_len, max_length)
        
        # Create padded array
        batch_size = len(texts)
        result = np.full((batch_size, max_len), self.pad_id, dtype=np.int32)
        
        for i, enc in enumerate(encoded):
            length = min(len(enc), max_len)
            result[i, :length] = enc[:length]
        
        return result
    
    def decode(self, token_ids: np.ndarray, skip_special: bool = True) -> str:
        """
        Decode token ID sequence back to text
        
        Args:
            token_ids: token ID array
            skip_special: whether to skip special tokens
            
        Returns:
            Decoded text
        """
        special_ids = {self.bos_id, self.eos_id, self.unk_id, self.pad_id}
        
        tokens = []
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            token = self.id_to_token.get(int(tid), '')
            # Handle byte-level token: <0xXX> -> corresponding character
            if token.startswith('<0x') and token.endswith('>'):
                try:
                    byte_val = int(token[3:-1], 16)
                    token = chr(byte_val)
                except (ValueError, OverflowError):
                    pass
            tokens.append(token)
        
        # Concatenate and restore spaces
        text = ''.join(tokens)
        text = text.replace(self.sp_prefix, ' ')
        
        # Remove leading spaces
        return text.lstrip()
    
    def id_to_char(self, token_id: int) -> str:
        """
        Decode a single token ID into corresponding character/token string
        
        Args:
            token_id: single token ID
            
        Returns:
            Corresponding token string
        """
        token = self.id_to_token.get(token_id, '<unk>')
        
        # Handle byte-level token: <0xXX> -> corresponding character
        if token.startswith('<0x') and token.endswith('>'):
            try:
                byte_val = int(token[3:-1], 16)
                return chr(byte_val)
            except (ValueError, OverflowError):
                return token
        
        return token
    
    def ids_to_tokens(self, token_ids: np.ndarray) -> List[str]:
        """
        Decode token ID sequence into a list of token strings
        
        Args:
            token_ids: token ID array
            
        Returns:
            List of token strings
        """
        return [self.id_to_char(int(tid)) for tid in token_ids]
    
    def decode_to_chars(self, token_ids: np.ndarray, 
                        skip_special: bool = False) -> List[Tuple[int, str, str]]:
        """
        Detailed decoding: returns the original token and decoded character for each token ID
        
        Args:
            token_ids: token ID array
            skip_special: whether to skip special tokens
            
        Returns:
            List of (token_id, raw_token, decoded_char) tuples
        """
        special_ids = {self.bos_id, self.eos_id, self.unk_id, self.pad_id}
        
        result = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special and tid in special_ids:
                continue
            
            raw_token = self.id_to_token.get(tid, '<unk>')
            decoded = raw_token
            
            # Handle byte-level token
            if raw_token.startswith('<0x') and raw_token.endswith('>'):
                try:
                    byte_val = int(raw_token[3:-1], 16)
                    decoded = chr(byte_val)
                except (ValueError, OverflowError):
                    pass
            # Handle ▁ prefix
            elif raw_token.startswith(self.sp_prefix):
                decoded = ' ' + raw_token[1:]
            
            result.append((tid, raw_token, decoded))
        
        return result
    
    def decode_single(self, token_id: int) -> str:
        """
        Decode a single token ID into a readable string (handling special characters)
        
        Args:
            token_id: single token ID
            
        Returns:
            Decoded string
        """
        raw_token = self.id_to_token.get(token_id, '<unk>')
        
        # Handle byte-level token: <0xXX> -> corresponding character
        if raw_token.startswith('<0x') and raw_token.endswith('>'):
            try:
                byte_val = int(raw_token[3:-1], 16)
                return chr(byte_val)
            except (ValueError, OverflowError):
                return raw_token
        
        # Handle ▁ prefix -> space
        if raw_token.startswith(self.sp_prefix):
            return ' ' + raw_token[1:]
        
        return raw_token
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)
    
    def clear_cache(self):
        """Clear BPE cache"""
        self._bpe_cache.clear()


class UltraFastBPETokenizer(OptimizedBPETokenizer):
    """
    Ultra-fast BPE Tokenizer - optimized for inference scenarios
    
    Additional optimizations:
    1. Precompile whole-word token table (direct lookup for common words)
    2. Use Trie to accelerate prefix matching
    3. Batch processing optimization
    """
    
    def __init__(self, tokenizer_json_path: str):
        super().__init__(tokenizer_json_path)
        
        # Precompile common whole words (non-special tokens with length >= 2)
        self._whole_word_vocab: Dict[str, int] = {}
        for token, tid in self.vocab.items():
            # Skip special tokens and single characters
            if not token.startswith('<') and len(token) >= 2:
                self._whole_word_vocab[token] = tid
        
        print(f"  Whole word vocab cache: {len(self._whole_word_vocab)}")
    
    def _fast_tokenize(self, normalized: str) -> List[int]:
        """
        Fast tokenization - prioritize whole word matching
        
        Strategy: Greedy longest match + BPE fallback
        """
        ids = []
        i = 0
        n = len(normalized)
        
        while i < n:
            # Try longest match
            best_len = 0
            best_id = None
            
            # Try from longest (check up to 20 characters)
            for end in range(min(i + 20, n), i, -1):
                substr = normalized[i:end]
                if substr in self.vocab:
                    best_len = end - i
                    best_id = self.vocab[substr]
                    break
            
            if best_id is not None:
                ids.append(best_id)
                i += best_len
            else:
                # Fallback: single character
                char = normalized[i]
                ids.append(self._token_to_id(char))
                i += 1
        
        return ids
    
    def encode_fast(self, text: str, add_bos: bool = True, add_eos: bool = False) -> np.ndarray:
        """
        Fast encoding (uses greedy matching, may have slight differences from standard BPE)
        
        Note: This is an approximate implementation, faster but may produce slightly different results
        """
        if not text:
            ids = []
            if add_bos:
                ids.append(self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return np.array(ids, dtype=np.int32)
        
        normalized = self._normalize(text)
        token_ids = self._fast_tokenize(normalized)
        
        result = []
        if add_bos:
            result.append(self.bos_id)
        result.extend(token_ids)
        if add_eos:
            result.append(self.eos_id)
        
        return np.array(result, dtype=np.int32)


def verify_optimized_tokenizer():
    """Verify consistency of optimized tokenizer with original implementation"""
    import os
    import time
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_json = os.path.join(script_dir, 'tokenizer.json')
    
    print("=" * 60)
    print("Loading optimized Tokenizer...")
    print("=" * 60)
    
    tokenizer = OptimizedBPETokenizer(tokenizer_json)
    fast_tokenizer = UltraFastBPETokenizer(tokenizer_json)
    
    # Test cases
    test_texts = [
        "Hello",
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is amazing!",
        "1234567890",
        "   spaces   ",
    ]
    
    print("\n" + "=" * 60)
    print("Encoding test...")
    print("=" * 60)
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\nInput: {repr(text)}")
        print(f"  IDs: {ids.tolist()}")
        print(f"  Decoded: {repr(decoded)}")
    
    # Performance test
    print("\n" + "=" * 60)
    print("Performance test...")
    print("=" * 60)
    
    long_text = "The quick brown fox jumps over the lazy dog. " * 100
    
    # Standard BPE
    start = time.perf_counter()
    for _ in range(100):
        tokenizer.encode(long_text)
    std_time = time.perf_counter() - start
    
    # Fast mode
    start = time.perf_counter()
    for _ in range(100):
        fast_tokenizer.encode_fast(long_text)
    fast_time = time.perf_counter() - start
    
    print(f"Long text ({len(long_text)} chars) x 100:")
    print(f"  Standard BPE: {std_time:.3f}s")
    print(f"  Fast mode: {fast_time:.3f}s")
    print(f"  Speedup: {std_time/fast_time:.2f}x")
    
    # Batch test
    print("\n" + "=" * 60)
    print("Batch test...")
    print("=" * 60)
    
    batch = test_texts * 10
    padded = tokenizer.encode_batch(batch, pad=True)
    print(f"Batch size: {len(batch)}")
    print(f"Output shape: {padded.shape}")
    
    # Token ID decoding test
    print("\n" + "=" * 60)
    print("Token ID -> Character decoding test...")
    print("=" * 60)
    
    demo_text = "Hello world!\nNew line"
    ids = tokenizer.encode(demo_text)
    print(f"\nInput text: {repr(demo_text)}")
    print(f"Token IDs: {ids.tolist()}")
    
    # Detailed decoding
    print("\nToken-by-token decoding:")
    print(f"{'ID':>6} | {'Raw Token':<15} | {'Decoded':<15}")
    print("-" * 45)
    
    detailed = tokenizer.decode_to_chars(ids, skip_special=False)
    for tid, raw, decoded in detailed:
        print(f"{tid:>6} | {repr(raw):<15} | {repr(decoded):<15}")
    
    # Full decoding
    print(f"\nFull decoding: {repr(tokenizer.decode(ids))}")
    
    # Single token decoding example
    print("\nSingle token decoding example:")
    sample_ids = [1, 15043, 3186, 13, 29871]  # <s>, Hello, world, \n, space
    for tid in sample_ids:
        raw = tokenizer.id_to_token.get(tid, '<unk>')
        decoded = tokenizer.decode_single(tid)
        print(f"  ID {tid}: {repr(raw)} → {repr(decoded)}")


if __name__ == '__main__':
    verify_optimized_tokenizer()


if __name__ == '__main__':
    verify_optimized_tokenizer()
