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
    "flex_attention_with_sink",
]

import torch
import functools
from .utils import (
    create_block_mask_cached,
    flex_attention,
    generate_sliding_window,
    causal_mask,
)
from torch.nn.attention.flex_attention import flex_attention as uncompiled_flex_attention

def causal_mask_with_sink(batch, head, q_idx, kv_idx):
    """
      0 1 2 3     0 1 2 3
    0 X X       1   X
    1 X X X     2   X X
    2 X X X X   3   X X X
    """
    # We add (q_idx + 1) since first column is sink token
    causal_mask = (q_idx + 1) >= kv_idx
    sink_first_column = kv_idx == 0
    return causal_mask | sink_first_column
pass

@functools.lru_cache
def generate_sliding_window_with_sink(window_size: int):
    def sliding_window(batch, head, q_idx, kv_idx):
        causal_mask = (q_idx + 1) >= kv_idx
        # Official PyTorch attends to 1 extra token
        # windowed_mask = q_idx - kv_idx <= window_size
        # HuggingFace and official GPT OSS attends to only 128 tokens not (128+1)
        windowed_mask = (q_idx + 1) - kv_idx < window_size
        sink_first_column = kv_idx == 0
        return (causal_mask & windowed_mask) | sink_first_column
    sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}_sink"
    return sliding_window
pass

@functools.lru_cache
def generate_sink_score_mod(sink_weights : torch.Tensor):
    def sink_score_mod(score, batch, head, q_idx, kv_idx):
        # Sink token is at the first location
        return torch.where(
            kv_idx == 0,
            sink_weights[head].to(score.dtype) + 0.0, # Add +0 to allow gradients
            score,
        )
    return sink_score_mod
pass


def flex_attention_with_sink(
    self_attn,
    query,
    key,
    value,
    scale = None,
    sliding_window = None,
    compile = True,
):
    """
    Allows one sink token to be attended to for full/sliding window attention
    Similar to Efficient Streaming Language Models with Attention Sinks
    Primarily for GPT-OSS 2025
    """
    assert getattr(self_attn, "sinks", None) is not None, "Unsloth: self_attn must have sinks"
    sink_weights = self_attn.sinks
    enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
    scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

    bsz, heads_Q, qlen_Q, dim = query.shape
    _, heads_KV, qlen_KV, _ = key.shape

    # Add K and V with a row of 0s to allow sinks to be placed there
    key_padded   = torch.cat([key  .new_zeros(bsz, heads_KV, 1, dim), key],   dim = 2)
    value_padded = torch.cat([value.new_zeros(bsz, heads_KV, 1, dim), value], dim = 2)

    # Check for sliding window
    sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
    mask_mod = \
        generate_sliding_window_with_sink(sliding_window) \
        if type(sliding_window) is int and sliding_window != 0 else \
        causal_mask_with_sink
    score_mod = generate_sink_score_mod(sink_weights)
    block_mask = create_block_mask_cached(mask_mod, qlen_Q, qlen_KV+1) # Add 1 since we padded
    attn_output = (flex_attention if compile else uncompiled_flex_attention)(
        query,
        key_padded,
        value_padded,
        block_mask = block_mask,
        score_mod = score_mod,
        enable_gqa = enable_gqa,
        scale = scale,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output
pass


def new_flex_attention_with_sink(
    self_attn,
    query,
    key,
    value,
    scale = None,
    sliding_window = None,
    compile = True,
):
    """
    Allows one sink token to be attended to for full/sliding window attention
    Similar to Efficient Streaming Language Models with Attention Sinks
    Primarily for GPT-OSS 2025

    [WARNING] has higher error than old_flex_attention_with_sink
    """
    assert getattr(self_attn, "sinks", None) is not None, "Unsloth: self_attn must have sinks"
    sink_weights = self_attn.sinks
    enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
    scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

    bsz, heads_Q, qlen_Q, dim = query.shape
    _, heads_KV, qlen_KV, _ = key.shape

    # Check for sliding window
    sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
    mask_mod = \
        generate_sliding_window(sliding_window) \
        if type(sliding_window) is int and sliding_window != 0 else \
        causal_mask
    block_mask = create_block_mask_cached(mask_mod, qlen_Q, qlen_KV)
    attn_output, logsumexp = (flex_attention if compile else uncompiled_flex_attention)(
        query,
        key,
        value,
        block_mask = block_mask,
        score_mod = None, # None needed
        enable_gqa = enable_gqa,
        scale = scale,
        return_lse = True, # log(sum(exp(xi)))
    )

    #### 3 versions to add sink tokens ####
    #### Version 1: Basic reciprocal denominator removal
    # softmax_sum = torch.exp(logsumexp)
    # new_denominator = softmax_sum / (softmax_sum + torch.exp(self_attn.sinks.unsqueeze(1)))
    # scale = new_denominator
    
    ### Version 2: logaddexp does log(exp(x1) + exp(x2))
    # logsumexp_new = torch.logaddexp(logsumexp, self_attn.sinks.unsqueeze(1))
    # scale = torch.exp(logsumexp - logsumexp_new)

    ### Version 3: Most simple uses sigmoid and scale
    scale = torch.sigmoid(logsumexp - self_attn.sinks.unsqueeze(1))

    # All 3 versions scale the original attn_output!
    attn_output = attn_output * scale.unsqueeze(-1).to(attn_output.dtype)
    # To reduce error, one should do attn_output.to(torch.float32)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output
pass
