# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Quantized-KV attention through `mx.fast.scaled_dot_product_attention` over a dequantized copy,
once the `[B, HQ, L, S]` scores the runtimes materialize would cost more than the copy."""
import functools

import mlx.core as mx


def _result_dtype(queries, q_keys, q_values):
    # `mx.quantized_matmul` widens to float32 when either cache disagrees with the queries.
    same = queries.dtype == q_keys[1].dtype == q_values[1].dtype
    return queries.dtype if same else mx.float32


def _row_bytes(q_cache, group_size, dtype):
    # Dequantized plus packed: a prefix view is compacted first, and contiguity is not visible
    # from the arrays, so the compaction is charged always.
    packed = sum(part.shape[-1] * part.dtype.size for part in q_cache)
    return q_cache[1].shape[-1] * group_size * dtype.size + packed


def dequantizing_is_smaller(queries, q_keys, q_values, group_size):
    """Scores cost `HQ * L` per cached token, the copy `HKV` rows; the depth cancels."""
    dtype = _result_dtype(queries, q_keys, q_values)
    HQ, L = queries.shape[-3], queries.shape[-2]
    HKV = q_keys[0].shape[-3]
    scores = HQ * L * dtype.size
    copy = HKV * (_row_bytes(q_keys, group_size, dtype) + _row_bytes(q_values, group_size, dtype))
    return scores > copy


# Elsewhere mlx falls back and builds the scores anyway, so the copy would cost extra rather
# than instead. ScaledDotProductAttention::use_fallback, mlx 0.32.1.
_FUSED_FULL_HEAD_DIMS = frozenset((64, 72, 80, 96, 128))
_FUSED_VECTOR_HEAD_DIMS = frozenset((64, 96, 128, 256))


def fused_kernel_exists(queries, q_keys, q_values, group_size, mask=None):
    """Whether mlx will really fuse this call. Reachable today at `head_dim` 256 (Qwen3-Next)."""
    if isinstance(mask, mx.array) and mx.issubdtype(mask.dtype, mx.floating):
        # The fused kernel rejects such a mask; the runtime widens the scores instead.
        dtype = _result_dtype(queries, q_keys, q_values)
        if mx.result_type(mask.dtype, dtype) != dtype:
            return False
    head_dim = q_keys[1].shape[-1] * group_size
    value_dim = q_values[1].shape[-1] * group_size
    L = queries.shape[-2]
    if L > 8:
        return head_dim == value_dim and head_dim in _FUSED_FULL_HEAD_DIMS
    return (L * (queries.shape[-3] // q_keys[0].shape[-3]) <= 32
            and ((head_dim == value_dim and head_dim in _FUSED_VECTOR_HEAD_DIMS)
                 or (head_dim == 192 and value_dim == 128)))


def dequantized_sdpa(queries, q_keys, q_values, scale, mask=None, group_size=64, bits=8):
    """The fused kernel over one dequantized copy. An emptied row averages, as the unfused path
    does: mlx masks with a finite minimum, not -inf."""
    dtype = _result_dtype(queries, q_keys, q_values)
    keys = mx.dequantize(*q_keys, group_size=group_size, bits=bits).astype(dtype)
    values = mx.dequantize(*q_values, group_size=group_size, bits=bits).astype(dtype)
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)


def quantized_sdpa_over(unfused):
    """Wrap a runtime's `quantized_scaled_dot_product_attention`; it still answers below the tie."""
    @functools.wraps(unfused)
    def quantized_sdpa(queries, q_keys, q_values, scale, mask=None, group_size=64, bits=8,
                       **kwargs):
        # An argument this wrapper predates has no fused equivalent: delegate, never drop it.
        if (not kwargs
                and fused_kernel_exists(queries, q_keys, q_values, group_size, mask)
                and dequantizing_is_smaller(queries, q_keys, q_values, group_size)):
            return dequantized_sdpa(queries, q_keys, q_values, scale, mask,
                                    group_size=group_size, bits=bits)
        return unfused(queries, q_keys, q_values, scale, mask, group_size=group_size, bits=bits,
                       **kwargs)
    return quantized_sdpa


_PATCH_FLAG = "_unsloth_quantized_attention_patch"

_PATCH_TARGETS = ("mlx_lm.models.base", "mlx_vlm.models.base")


def install_quantized_attention():
    """Wrap the imported runtimes' quantized attention, idempotently; returns the paths patched."""
    import sys

    if hasattr(mx.fast, "quantized_scaled_dot_product_attention"):
        return ()

    patched = []
    for module_path in _PATCH_TARGETS:
        module = sys.modules.get(module_path)
        if module is None or getattr(module, _PATCH_FLAG, False):
            continue
        unfused = getattr(module, "quantized_scaled_dot_product_attention", None)
        if unfused is None:
            continue
        module.quantized_scaled_dot_product_attention = quantized_sdpa_over(unfused)
        setattr(module, _PATCH_FLAG, True)
        patched.append(module_path)
    return tuple(patched)
