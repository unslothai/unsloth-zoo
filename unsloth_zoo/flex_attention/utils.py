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
]

import torch
import functools
from ..temporary_patches.common import torch_compile

try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True
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
except:
    HAS_FLEX_ATTENTION = False
    FLEX_ATTENTION_BLOCK_SIZE = None
    flex_attention = None
    create_block_mask_cached = None
    causal_mask = None
    generate_sliding_window = None
pass
