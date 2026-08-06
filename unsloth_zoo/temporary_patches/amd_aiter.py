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
"""
AMD aiter Flash Attention registered as a transformers attention backend.

The source rewrite in compiler.py cannot reach modern models: on transformers
4.50+ the attention forward has no literal SDPA call, it dispatches through
ALL_ATTENTION_FUNCTIONS and the real call lives in
transformers/integrations/sdpa_attention.py, which the compiler never reads.
A census of 4.51.3 / 4.57.6 / 5.5.0 finds 0 of 106 SDPA call sites in a shape
any rewrite could match.

Registering a backend reaches every model that uses the attention interface,
needs no codegen, and cannot emit invalid Python.

Layout: the interface passes q/k/v as (B, H, S, D) and expects (B, S, H, D)
back, which is aiter's native layout, so one transpose in and none out.

Anything not provably safe falls through to the stock SDPA backend, so this is
a kernel swap and never a behaviour change.
"""

import os

import torch

from .common import TEMPORARY_PATCHES
from .utils import raise_error

__all__ = ["patch_amd_aiter_attention", "aiter_attention_forward"]

_DISABLE_ENV = "UNSLOTH_DISABLE_AITER"

# flash_attn_func applies 1/sqrt(head_dim) itself and takes no softmax_scale,
# so a model asking for a different scale must not take this path.
_SCALE_RTOL = 1e-6

# Delivered by the interface but harmless here: RoPE is already applied to q/k
# and cache bookkeeping happens outside the kernel. Measured on transformers
# 4.57.6, 5.0.0 and 5.14.1.
_BENIGN_KWARGS = frozenset((
    "position_ids", "use_cache", "cache_position", "past_key_value",
    "past_key_values", "layer_idx", "layer_head_mask", "output_attentions",
    "head_mask", "sliding_window", "position_bias",
))

# These change the result and plain flash_attn_func cannot express them:
# sliding_window is windowed attention, position_bias becomes an additive mask
# on transformers >= 5.14, softcap and s_aux are soft capping and sinks.
_SEMANTIC_KWARGS = (
    "sliding_window", "position_bias", "softcap", "logit_softcapping",
    "s_aux", "attention_chunk_size", "head_mask",
)


def aiter_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout = 0.0,
    scaling = None,
    is_causal = None,
    **kwargs,
):
    """Use aiter when provably safe, else the stock SDPA backend.

    Signature matches transformers' sdpa_attention_forward on 4.51 through 5.x.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def _fallback():
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout = dropout, scaling = scaling, is_causal = is_causal, **kwargs,
        )

    if os.environ.get(_DISABLE_ENV, "0") == "1":
        return _fallback()

    # Registration is AMD gated, but this is exported, directly callable, and the
    # gate behind it is cached at import. Re-check per call: `is_hip` is uncached
    # and cheap, and ROCm tensors report device.type "cuda", so both are needed.
    from ..device_type import is_hip
    if not is_hip() or not query.is_cuda:
        return _fallback()

    # Needs the probability matrix, which a fused kernel never materialises.
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        return _fallback()

    for name in _SEMANTIC_KWARGS:
        if kwargs.get(name) is not None:
            return _fallback()

    # An unknown keyword means a newer transformers grew an argument this
    # backend has never seen, so fall back rather than ignore its semantics.
    # Do not name the loop variable `value`, it would shadow the tensor.
    for kwarg_name, kwarg_value in kwargs.items():
        if (kwarg_name not in _BENIGN_KWARGS
                and kwarg_value is not None and kwarg_value is not False):
            return _fallback()

    from ..device_type import get_amd_flash_attn_func
    flash_attn_func = get_amd_flash_attn_func()
    if flash_attn_func is None:
        return _fallback()

    # aiter aligns causal masks bottom-right, SDPA top-left, so they agree only
    # with no explicit mask and equal q/k lengths.
    if attention_mask is not None or query.shape[-2] != key.shape[-2]:
        return _fallback()
    if dropout:
        return _fallback()
    if scaling is not None and abs(scaling - query.shape[-1] ** -0.5) > _SCALE_RTOL:
        return _fallback()
    if query.dtype not in (torch.float16, torch.bfloat16):
        return _fallback()

    # aiter asserts return_lse when any input requires grad and we omit it, so
    # the fast path is inference only.
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return _fallback()

    resolved_is_causal = is_causal
    if resolved_is_causal is None:
        resolved_is_causal = query.shape[2] > 1 and getattr(module, "is_causal", True)
    if resolved_is_causal is not True:
        return _fallback()

    # aiter broadcasts kv heads itself, but only when the head counts divide.
    groups = getattr(module, "num_key_value_groups", 1) or 1
    if groups != 1 and query.shape[1] % key.shape[1] != 0:
        return _fallback()

    try:
        attn_output = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            causal = True,
        )
    except Exception:
        # Unsupported head dim, arch mismatch or a JIT build failure degrades to
        # SDPA instead of killing the forward pass.
        return _fallback()
    if attn_output is None:
        return _fallback()
    return attn_output.contiguous(), None


def patch_amd_aiter_attention():
    """Register `aiter` as an attention backend when the hardware supports it."""
    try:
        from ..device_type import get_amd_attention_implementation
        if get_amd_attention_implementation() != "amd_aiter":
            return  # NVIDIA, Intel, CPU, MLX, ROCm < 7, or an unsupported arch
        from transformers.modeling_utils import AttentionInterface
    except Exception as e:
        return raise_error("amd_aiter_attention", e)

    try:
        AttentionInterface.register("aiter", aiter_attention_forward)
    except Exception as e:
        return raise_error("amd_aiter_attention_register", e)
pass

TEMPORARY_PATCHES.append(patch_amd_aiter_attention)
