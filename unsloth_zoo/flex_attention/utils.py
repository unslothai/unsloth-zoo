# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "HAS_FLEX_ATTENTION",
    "FLEX_ATTENTION_BLOCK_SIZE",
    "flex_attention",
    "create_block_mask_cached",
    "causal_mask",
    "generate_sliding_window",
    "FlexAttentionCache",
]

import torch
import functools
from ..temporary_patches.common import torch_compile
FLEX_ATTENTION_KV_INCREMENT = 512

try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True
    from torch.nn.attention.flex_attention import _score_mod_signature, _mask_mod_signature
    flex_attention = torch_compile(flex_attention)

    @functools.lru_cache
    def create_block_mask_cached(mask_mod, M, N, device = "cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return create_block_mask(mask_mod, None, None, M, N, device = device)

    def causal_mask(batch, head, q_idx, kv_idx):
        """Causal mask for Flex Attention"""
        return q_idx >= kv_idx

    @functools.lru_cache
    def generate_sliding_window(window_size: int):
        """Sliding window mask for Flex Attention"""
        def sliding_window(batch, head, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            """
            [NOTE] Official PyTorch uses <= window_size which is (window_size+1) tokens
            Ie if window_size = 3, HuggingFace does the below:
            X
            X X
            X X X
              X X X
                X X X
            Whilst official PyTorch does (ie attends to the last 3 and itself)
            X
            X X
            X X X
            X X X X
              X X X X
            """
            windowed_mask = q_idx - kv_idx < window_size
            # Official PyTorch attends to 1 extra token
            # windowed_mask = q_idx - kv_idx <= window_size
            return causal_mask & windowed_mask
        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}"
        return sliding_window

    # For inference see https://pytorch.org/blog/flexattention-for-inference
    def get_score_mod_w_offset(score_mod: _score_mod_signature, _offset: torch.tensor):
        def _score_mod(score, b, h, q, kv):
            return score_mod(score, b, h, q + _offset, kv)
        return _score_mod

    def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: torch.tensor):
        def _mask_mod(b, h, q, kv):
            return mask_mod(b, h, q + _offset, kv)
        return _mask_mod

    # Used for every attention layer
    class FlexAttentionCache:
        __slots__ = "offset", "offset_tensor", "block_mask", "mask_mod", "max_length", "block_size", "sliding_window"

        def __init__(self, key, mask_mod, sliding_window):
            bsz, heads_KV, qlen_KV, dim = key.shape
            if sliding_window is None:
                """
                Normal causal mask:
                    k0 k1 k2 k3 k4
                q0   X
                q1   X  X
                q2   X  X  X
                q3   X  X  X  X
                q4   X  X  X  X  X
                During decoding:
                    k0 k1 k2 k3 k4
                q0   
                q1   
                q2   
                q3   
                q4   X  X  X  X  X
                But q_index = 0, so we need an offset = 4
                If no offset, we get:
                   k0 k1 k2 k3 k4
                q0   
                q1   
                q2   
                q3   
                q4   X
                Which is wrong.
                Note it's offset==index since it's indexed from 0 as seen in
                https://pytorch.org/blog/flexattention-for-inference/
                https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L91
                """
                # Get next multiple of FLEX_ATTENTION_KV_INCREMENT
                div, mod = divmod(qlen_KV, FLEX_ATTENTION_KV_INCREMENT)
                n = FLEX_ATTENTION_KV_INCREMENT*div + (FLEX_ATTENTION_KV_INCREMENT if mod != 0 else 0)
                self.offset = qlen_KV - 2 # Minus two since we need the block mask to use the saved offset_tensor
                if self.offset <= -2:
                    # Minimum is -1
                    self.offset = -1
                    # During decoding we do self.offset += 1, so self.offset = 0
                self.sliding_window = None
            else:
                """
                Sliding window of 2 + causal mask:
                    k0 k1 k2 k3 k4
                q0   X
                q1   X  X
                q2      X  X
                q3         X  X
                q4            X  X
                During decoding:
                    k0 k1 k2 k3 k4
                q0   
                q1   
                q2   
                q3   
                q4            X  X
                If we set cache_implementation = "static" which we assume, we don't use an offset
                since the K is a rolling matrix of the past window size.
                Ie if sliding_window = 2, K is shape (2, dim). So in actuality, we get:
                    -- -- -- k3 k4
                q0   
                q1   
                q2   
                q3   
                q4            X  X
                But since we use a rolling matrix, offset = sliding_window-1 always ie 2-1 = 1
                    -- -- -- k0 k1
                q0   
                q1   
                q2   
                q3(0)   
                q4(1)         X  X
                Note it's offset==index since it's indexed from 0 as seen in
                https://pytorch.org/blog/flexattention-for-inference/
                https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L91
                For sliding window, it's always sliding_window - 1
                since 128 means index 127
                """
                n = sliding_window
                self.offset = min(sliding_window, qlen_KV) - 2 # Minus 2 since block mask is indexing
                if self.offset <= -2:
                    # Minimum is -1
                    self.offset = -1
                    # During decoding we do self.offset += 1, so self.offset = 0
                self.sliding_window = sliding_window - 1 # Minus 1 since token 128 means index 127
            self.offset_tensor = torch.tensor(self.offset, device = key.device, dtype = torch.int32)
            self.block_mask = create_block_mask_cached(mask_mod, n, n)
            self.mask_mod = mask_mod
            self.max_length = n
            self.block_size = self.block_mask.BLOCK_SIZE[0]

        def __call__(self, key):
            # We increment beforehand to get the correct index since offset_tensor is used
            #                                    Assume sliding_window=128-1 = 127
            #                                    offset=126, so offset+1 = 127
            if (self.sliding_window is None) or (self.offset < self.sliding_window):
                self.offset += 1
                self.offset_tensor.add_(1)
            if self.offset >= self.max_length:
                # Must be >= since offset=127, max_length=128 means size=127+1=128
                # since we do zero indexing
                self.max_length += FLEX_ATTENTION_KV_INCREMENT
                self.block_mask = create_block_mask_cached(self.mask_mod, self.max_length, self.max_length)
                self.block_size = self.block_mask.BLOCK_SIZE[0]
            bsz, heads_KV, qlen_KV, dim = key.shape
            block_offset = self.offset // self.block_size
            block_mask_slice = self.block_mask[:, :, block_offset]
            block_mask_slice.mask_mod = get_mask_mod_w_offset(self.mask_mod, self.offset_tensor)
            # Must set seq_lengths as seen in
            # https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L80
            block_mask_slice.seq_lengths = (1, qlen_KV)
            return block_mask_slice
    pass

except:
    HAS_FLEX_ATTENTION = False
    FLEX_ATTENTION_BLOCK_SIZE = None
    flex_attention = None
    create_block_mask_cached = None
    causal_mask = None
    generate_sliding_window = None
    FlexAttentionCache = None
pass
