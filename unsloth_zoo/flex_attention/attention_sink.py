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
    "old_flex_attention_with_sink",
    "is_flex_attention_decoding",
    "flex_attention_with_sink_decoding",
    "flex_attention_add_sinks",
    "flash_attention_left_padded",
]

import torch
import functools
import math
import torch.nn.functional as F
from .utils import (
    create_block_mask_cached,
    create_block_mask,
    compiled_create_block_mask,
    _flex_attention as uncompiled_flex_attention,
    flex_attention,
    FlexAttentionCache,

    causal_mask,
    generate_causal_mask_with_padding,
    generate_decoding_causal_mask_with_padding,

    generate_sliding_window_mask,
    generate_sliding_window_mask_with_padding,
    generate_decoding_sliding_window_mask_with_padding,
)

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


def old_flex_attention_with_sink(
    self_attn,
    query,
    key,
    value,
    attention_mask = None,
    scale = None,
    sliding_window = None,
    compile = True,
):
    """
    Allows one sink token to be attended to for full/sliding window attention
    Similar to Efficient Streaming Language Models with Attention Sinks
    Primarily for GPT-OSS 2025

    [WARNING] This only works for training. Inference fails since KV cache's
    absolute positioning will fail.
    """
    if not self_attn.training:
        raise NotImplementedError("Unsloth: This version of flex attention only works for training")
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
    block_mask = compiled_create_block_mask(mask_mod, qlen_Q, qlen_KV+1, device = key.device) # Add 1 since we padded
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


def is_flex_attention_decoding(self_attn, query):
    if query.dim() == 4:
        bsz, heads_Q, qlen_Q, dim = query.shape
    else:
        bsz, qlen_Q, dim = query.shape
    is_training = self_attn.training
    has_flex_cache = hasattr(self_attn, "_flex_attention_cache")
    if is_training or (
        not is_training and (not has_flex_cache or qlen_Q != 1)
    ):
        return False
    return True
pass


def flex_attention_with_sink(
    self_attn,
    query,
    key,
    value,
    attention_mask = None,
    scale = None,
    sliding_window = None,
    compile = True,
    has_static_cache = True,
):
    """
    Allows one sink token to be attended to for full/sliding window attention
    Similar to Efficient Streaming Language Models with Attention Sinks
    Primarily for GPT-OSS 2025

    [WARNING] has higher error than old_flex_attention_with_sink, but works for inference
    """
    assert getattr(self_attn, "sinks", None) is not None, "Unsloth: self_attn must have sinks"
    sink_weights = self_attn.sinks
    enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
    scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

    bsz, heads_Q, qlen_Q, dim = query.shape
    _, heads_KV, qlen_KV, _ = key.shape

    # Check for sliding window
    sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
    is_training = self_attn.training
    mask_mod = None
    block_mask = None
    has_flex_cache = hasattr(self_attn, "_flex_attention_cache")
    # Handle inference and training
    if attention_mask is not None and has_static_cache:
        if is_training or (
            not is_training and (not has_flex_cache or qlen_Q != 1)
        ):
            if is_training:
                if has_flex_cache:
                    del self_attn._flex_attention_cache
            else:
                # Consider left padding as well for prefill
                assert attention_mask is not None
                assert attention_mask.dim() == 2, f"Unsloth: Attention_mask has dim = {attention_mask.dim()}"
                # We must account for left padding
                padding_start_idx = attention_mask.argmax(1).to(query.device)
                do_padding = torch.arange(max(qlen_Q, qlen_KV), device = query.device).repeat((bsz, 1)) < padding_start_idx.unsqueeze(0).T
                # We also make all padded tokens Q=1, K=-inf
                # Note if Q=0, K=0, Q*K = 0, but exp(0) = 1, so that's wrong
                # Only exp(-inf) = 0. So Q=1, K=-inf, Q*K = -inf
                query.transpose(2, 1)[do_padding[:, :qlen_Q ]] = 1
                key  .transpose(2, 1)[do_padding[:, :qlen_KV]] = -torch.inf
                value.transpose(2, 1)[do_padding[:, :qlen_KV]] = 0
                # Use special padded mask creators
                mask_mod = prefill_mask_mod = \
                    generate_sliding_window_mask_with_padding(sliding_window, padding_start_idx) \
                    if type(sliding_window) is int and sliding_window != 0 else \
                    generate_causal_mask_with_padding(padding_start_idx)
                decoding_mask_mod = \
                    generate_decoding_sliding_window_mask_with_padding(sliding_window, padding_start_idx) \
                    if type(sliding_window) is int and sliding_window != 0 else \
                    generate_decoding_causal_mask_with_padding(padding_start_idx)
                self_attn._flex_attention_cache = FlexAttentionCache(key, decoding_mask_mod, sliding_window)
        else:
            block_mask = self_attn._flex_attention_cache(key)
        pass
    pass
    # Create mask_mod on training and decoding steps
    if mask_mod is None:
        mask_mod = \
            generate_sliding_window_mask(sliding_window) \
            if type(sliding_window) is int and sliding_window != 0 else \
            causal_mask
    if block_mask is None:
        block_mask = compiled_create_block_mask(mask_mod, bsz, heads_Q, qlen_Q, qlen_KV, device = key.device)

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
    # sink_scale = new_denominator
    
    ### Version 2: logaddexp does log(exp(x1) + exp(x2))
    # logsumexp_new = torch.logaddexp(logsumexp, self_attn.sinks.unsqueeze(1))
    # sink_scale = torch.exp(logsumexp - logsumexp_new)

    ### Version 3: Most simple uses sigmoid and scale
    sink_scale = torch.sigmoid(logsumexp - self_attn.sinks.unsqueeze(1))

    # All 3 versions scale the original attn_output!
    attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)
    # To reduce error, one should do attn_output.to(torch.float32)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output
pass

# Fails sometimes
# @torch.compile(dynamic = True, fullgraph = False, mode = "reduce-overhead")
def flex_attention_with_sink_decoding(
    self_attn,
    query,
    key,
    value,
    scale = None,
):
    assert getattr(self_attn, "sinks", None) is not None, "Unsloth: self_attn must have sinks"
    enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
    scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale
    block_mask = self_attn._flex_attention_cache(key)
    attn_output, logsumexp = flex_attention(
        query,
        key,
        value,
        block_mask = block_mask,
        score_mod = None, # None needed
        enable_gqa = enable_gqa,
        scale = scale,
        return_lse = True, # log(sum(exp(xi)))
    )
    return attn_output, logsumexp
pass

def flex_attention_add_sinks(
    self_attn,
    attn_output,
    logsumexp,
):
    logsumexp -= self_attn.sinks.unsqueeze(1)
    sink_scale = torch.sigmoid(logsumexp, out = logsumexp)
    attn_output *= sink_scale.unsqueeze(-1).to(attn_output.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output
pass

def flash_attention_left_padded(
    self_attn,
    query_states,
    key_states,
    value_states,
    attention_mask,
    is_causal = True,
    window_size_left = None,
    dropout_p = 0.0,
    scale = None,
):
    assert attention_mask.dtype in (torch.int32, torch.int64, torch.bool)
    device = query_states.device

    bsz, qlen = attention_mask.shape
    n_heads = self_attn.config.num_attention_heads
    n_kv_heads = getattr(self_attn.config, "num_key_value_heads", n_heads)
    head_dim = self_attn.head_dim

    bsz, heads_Q, qlen_Q, dim = query_states.shape
    _, heads_KV, qlen_KV, _ = key_states.shape

    Q = query_states.transpose(1, 2)
    K = key_states.transpose(1, 2)
    V = value_states.transpose(1, 2)

    # ---- lengths & cumulative starts (int32 on CUDA) ----
    seqlens = attention_mask.to(dtype=torch.int32, device=device).sum(dim=1)
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())

    # ---- unpad/pack ----
    flat_mask = attention_mask.reshape(-1).to(device=device)
    keep = flat_mask.nonzero(as_tuple=False).squeeze(-1)

    Q_flat = Q.reshape(bsz * qlen_Q,  n_heads,    head_dim)
    K_flat = K.reshape(bsz * qlen_KV, n_kv_heads, head_dim)
    V_flat = V.reshape(bsz * qlen_KV, n_kv_heads, head_dim)

    Q_unpad = Q_flat.index_select(0, keep).contiguous()
    K_unpad = K_flat.index_select(0, keep).contiguous()
    V_unpad = V_flat.index_select(0, keep).contiguous()

    # ---- call flash attention ----
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    kwargs = dict( scale = scale,)
    # Only pass window sizes if you actually want sliding window
    if window_size_left is not None:
        kwargs["window_size_left"] = int(window_size_left)
        kwargs["window_size_right"] = 0  # causal right window

    """
    aten::_flash_attention_forward(
        Tensor query, Tensor key, Tensor value, Tensor?
        cum_seq_q, Tensor? cum_seq_k,
        SymInt max_q, SymInt max_k,
        float dropout_p, bool is_causal, bool return_debug_mask, *,
        float? scale=None,
        SymInt? window_size_left=None, SymInt? window_size_right=None,
        Tensor? seqused_k=None, Tensor? alibi_slopes=None) -> 
    (Tensor output, Tensor softmax_logsumexp, Tensor rng_state, Tensor unused, Tensor debug_attn_mask)
    """
    attn_output, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        query = Q_unpad,
        key = K_unpad,
        value = V_unpad,
        cum_seq_q = cu_seqlens,
        cum_seq_k = cu_seqlens, 
        max_q = max_seqlen,
        max_k = max_seqlen,
        dropout_p = float(dropout_p),
        is_causal = bool(is_causal),
        return_debug_mask = False,
        **kwargs
    )
    # All 3 versions scale the original attn_output!
    sink_scale = torch.sigmoid(logsumexp - self_attn.sinks.unsqueeze(1))
    attn_output = attn_output * sink_scale.unsqueeze(-1).transpose(0, 1).to(attn_output.dtype)

    out_flat = Q_flat.new_zeros((bsz * qlen_Q, n_heads, head_dim))
    out_flat[keep] = attn_output
    attn_output = out_flat.view(bsz, qlen_Q, n_heads, head_dim)

    attn_output = attn_output.contiguous()
    return attn_output
pass
