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
once the `[B, HQ, L, S]` scores the runtimes materialize would cost more than the copy, and Metal kernels for
Qwen Sparse Attention (QSA) in mlx-vlm's qwen4_exp."""
import functools
import hashlib
import math
import os
import re

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


# The calls mlx fuses (ScaledDotProductAttention::has_fused_kernel and use_fallback, mlx 0.32.3).
# Elsewhere mlx builds the scores anyway, so the copy would cost extra rather than instead.
_FUSED_FULL_HEAD_DIMS = frozenset((64, 72, 80, 96, 128))
_FUSED_VECTOR_HEAD_DIMS = frozenset((64, 96, 128, 256))
_SPLIT_MIN_QUERIES = 1024
_D512_MIN_QUERY_BLOCKS = 1024
# mlx also fuses square causal head_dim 256 prefill this long; on NAX the split covers it.
_SQUARE_D256_MIN_QUERIES = 2048


def _atoi(value):
    match = re.match(r"\s*[+-]?\d+", value)
    return int(match.group()) if match else 0


@functools.cache
def _tf32_enabled():
    # mlx reads MLX_ENABLE_TF32 once, through atoi.
    return _atoi(os.environ.get("MLX_ENABLE_TF32", "1")) != 0


def _full_kernel_exists(queries, q_keys, head_dim, value_dim, dtype, mask):
    from .nax import nax_available

    B, HQ, L = queries.shape[0], queries.shape[-3], queries.shape[-2]
    S = q_keys[0].shape[-2]
    causal = isinstance(mask, str) and mask == "causal"
    if causal and L > S:
        return False
    split = L >= _SPLIT_MIN_QUERIES and (_tf32_enabled() or dtype != mx.float32) and nax_available()
    if head_dim == value_dim == 256:
        if split and (causal or isinstance(mask, mx.array)):
            return True
        return dtype != mx.float32 and causal and L >= _SQUARE_D256_MIN_QUERIES and L == S
    if head_dim == value_dim == 512:
        return split and causal and B * HQ * -(-L // 32) >= _D512_MIN_QUERY_BLOCKS
    return ((head_dim == value_dim and head_dim in _FUSED_FULL_HEAD_DIMS)
            or (head_dim, value_dim) == (96, 64))


def _vector_kernel_exists(queries, q_keys, head_dim, value_dim, mask):
    L, S = queries.shape[-2], q_keys[0].shape[-2]
    gqa = queries.shape[-3] // q_keys[0].shape[-3]
    if L > S or L * gqa > 32:
        return False
    if head_dim == value_dim == 512:
        # mlx reads MLX_SDPA_D512_MIN_KL on every call; 0 also lifts the other defaults.
        min_keys = _atoi(os.environ.get("MLX_SDPA_D512_MIN_KL", "1024"))
        return S >= min_keys and (min_keys == 0
                                  or (L == 1 and gqa == 8 and not isinstance(mask, mx.array)))
    return ((head_dim == value_dim and head_dim in _FUSED_VECTOR_HEAD_DIMS)
            or (head_dim, value_dim) in ((192, 128), (96, 64)))


def fused_kernel_exists(queries, q_keys, q_values, group_size, mask=None):
    """Whether mlx will really fuse this call."""
    dtype = _result_dtype(queries, q_keys, q_values)
    if isinstance(mask, mx.array) and mx.issubdtype(mask.dtype, mx.floating):
        # The fused kernel rejects such a mask; the runtime widens the scores instead.
        if mx.result_type(mask.dtype, dtype) != dtype:
            return False
    head_dim = q_keys[1].shape[-1] * group_size
    value_dim = q_values[1].shape[-1] * group_size
    if queries.shape[-2] > 8:
        return _full_kernel_exists(queries, q_keys, head_dim, value_dim, dtype, mask)
    return _vector_kernel_exists(queries, q_keys, head_dim, value_dim, mask)


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


def _indexed_attention_training_over(original):
    @functools.wraps(original)
    def attention(*args, **kwargs):
        from .utils import mlx_training_patches_active
        if mlx_training_patches_active():
            return None
        return original(*args, **kwargs)
    return attention


def _qsa_attention_training_over(original):
    import inspect
    backend = original.__globals__
    signature = inspect.signature(original)

    # *args pass-through: the wrapper outlives training, so newer mlx-vlm kwargs must reach `original`.
    @functools.wraps(original)
    def attention(*args, **kwargs):
        from .utils import mlx_training_patches_active
        if mlx_training_patches_active():
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            a = call.arguments
            q, k, v = a["queries"], a["keys"], a["values"]
            plan = backend["select_qsa_execution_plan"](
                q, k, v, a["block_indices"], a["query_ends"],
                block_size=a["block_size"], causal=a["causal"],
                allow_sparse_decode=a.get("allow_sparse_decode", False))
            if plan is backend["QSAExecutionPlan"].INDEXED_SPARSE_PREFILL:
                # The fused path ignores `mask`; its selection owns the sparse mask.
                sparse_mask = a["mask_factory"]()
                output = backend["scaled_dot_product_attention"](
                    q, k, v, cache=a["cache"], scale=a["scale"], mask=sparse_mask)
                return mx.where(mx.any(sparse_mask, axis=-1, keepdims=True), output, 0)
        return original(*args, **kwargs)
    return attention


def install_sparse_attention_training():
    """Route loaded custom attention kernels through differentiable training paths."""
    import sys

    targets = (
        ("mlx_vlm.models.sparse_attention", "indexed_sparse_attention",
         _indexed_attention_training_over),
        ("mlx_vlm.models.qwen4_exp.qsa_kernel", "dispatch_qsa_attention",
         _qsa_attention_training_over),
    )
    for module_path, name, wrap in targets:
        module = sys.modules.get(module_path)
        original = getattr(module, name, None)
        if original is None:
            continue
        if getattr(original, "_unsloth_sparse_attention_original", None) is not None:
            wrapped = original
            original = wrapped._unsloth_sparse_attention_original
        else:
            wrapped = wrap(original)
            wrapped._unsloth_sparse_attention_original = original
        for path, consumer in list(sys.modules.items()):
            if path.startswith("mlx_vlm.") and consumer is not None:
                for alias, value in list(vars(consumer).items()):
                    if value is original:
                        setattr(consumer, alias, wrapped)


# QSA block selection.
_SELECT_HEADER = """
#include <metal_simdgroup>
using namespace metal;

// mlx-vlm's sum(maximum(head_scores, 0), heads) / sqrt(d): maximum keeps NaN, the column sum adds
// head by head, the divide is IEEE.
inline float qsa_block_score(const device float* scores, size_t head_stride, uint heads, float divisor) {
    float total = 0.0f;
    for (uint h = 0; h < heads; ++h) {
        const float v = scores[h * head_stride];
        total = (metal::isnan(v) ? v : (v > 0.0f ? v : 0.0f)) + total;
    }
    return metal::precise::divide(total, divisor);
}

// Block scores are +0, positive or NaN; MLX's sort puts NaN above +inf.
inline uint qsa_order_key(float score) {
    return metal::isnan(score) ? 0xFFFFFFFFu : as_type<uint>(score);
}
"""

# One threadgroup per query row. A radix select over the order keys finds the top-k threshold; the
# blocks above it and the highest-index blocks tied with it are argpartition's set (the last k of
# MLX's stable ascending argsort), written in increasing order after the -1 of missing picks.
_SELECT_SOURCE = """
    constexpr uint TG = 1024;
    constexpr uint NSG = TG / 32;
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nb = uint(dims[0]);
    const uint L = uint(dims[1]);
    const uint b = row / L;
    const uint c = metal::min(uint(metal::max(complete[row], 0)), nb);
    device int* picks = out + size_t(row) * TOPK;

    if (c <= TOPK) {
        for (uint i = tid; i < TOPK; i += TG) {
            picks[i] = i < TOPK - c ? -1 : int(i - (TOPK - c));
        }
        return;
    }

    threadgroup atomic_uint hist[256];
    threadgroup uint decided[2];
    threadgroup uint sg_counts[NSG];
    device uint* keys = scratch + size_t(row) * nb;
    const device float* row_scores = scores + (size_t(b) * H * L + (row - b * L)) * nb;
    const size_t head_stride = size_t(L) * nb;
    const float div = divisor[0];

    uint prefix = 0;
    uint remaining = TOPK;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24 - 8 * pass;
        if (tid < 256) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < c; e += TG) {
            uint key;
            if (pass == 0) {
                key = qsa_order_key(qsa_block_score(row_scores + e, head_stride, H, div));
                keys[e] = key;
            } else {
                key = keys[e];
            }
            if (pass == 0 || (key >> (shift + 8)) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            // Lane l owns digits 8l..8l+7: find the digit where the count above reaches `remaining`.
            uint counts[8];
            uint owned = 0;
            for (uint j = 0; j < 8; ++j) {
                counts[j] = atomic_load_explicit(&hist[8 * lane + j], memory_order_relaxed);
                owned += counts[j];
            }
            const uint above = simd_sum(owned) - simd_prefix_inclusive_sum(owned);
            if (above < remaining && remaining <= above + owned) {
                uint acc = above;
                for (int j = 7; j >= 0; --j) {
                    if (acc + counts[j] >= remaining) {
                        decided[0] = (prefix << 8) | (8 * lane + uint(j));
                        decided[1] = remaining - acc;
                        break;
                    }
                    acc += counts[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = decided[0];
        remaining = decided[1];
    }

    const uint threshold = prefix;
    const uint skip = atomic_load_explicit(&hist[threshold & 255u], memory_order_relaxed) - remaining;
    uint tie_carry = 0;
    uint pick_carry = 0;
    for (uint base = 0; base < c; base += TG) {
        const uint e = base + tid;
        const uint key = e < c ? keys[e] : 0u;
        const uint tie = (e < c && key == threshold) ? 1u : 0u;
        uint tie_before = 0, tie_chunk = 0;
        if (skip) {
            const uint ties = simd_sum(tie);
            if (lane == 0) {
                sg_counts[sg] = ties;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = 0; s < NSG; ++s) {
                tie_before += s < sg ? sg_counts[s] : 0u;
                tie_chunk += sg_counts[s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tie_before += simd_prefix_exclusive_sum(tie);
        }
        const uint pick = (e < c && (key > threshold || (tie && tie_carry + tie_before >= skip))) ? 1u : 0u;
        const uint simd_picks = simd_sum(pick);
        if (lane == 0) {
            sg_counts[sg] = simd_picks;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint pick_before = 0, pick_chunk = 0;
        for (uint s = 0; s < NSG; ++s) {
            pick_before += s < sg ? sg_counts[s] : 0u;
            pick_chunk += sg_counts[s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pick) {
            picks[pick_carry + pick_before + simd_prefix_exclusive_sum(pick)] = int(e);
        }
        tie_carry += tie_chunk;
        pick_carry += pick_chunk;
    }
"""


@functools.cache
def _select_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name = "unsloth_qsa_select_blocks", input_names = ["scores", "complete", "divisor", "dims"],
        output_names = ["out", "scratch"], header = _SELECT_HEADER, source = _SELECT_SOURCE,
    )


@functools.cache
def _divisor(head_dim):
    # The float32 MLX makes of the Python float in `scores / math.sqrt(head_dim)`.
    return mx.array([math.sqrt(head_dim)], dtype = mx.float32)


def select_blocks(scores, complete, topk, head_dim):
    """mlx-vlm's QSA block selection from float32 per-head scores [B, H, L, blocks], in one launch.

    Returns [B, L, topk] int32: the set `argpartition` picks among each row's first `complete`
    blocks, ascending, after one -1 per missing pick, which is the order the indexed attention
    kernel sorts any selection into.
    """
    B, H, L, nb = scores.shape
    out, _ = _select_kernel()(
        inputs = [mx.contiguous(scores), mx.contiguous(complete.astype(mx.int32)), _divisor(head_dim),
                  mx.array([nb, L], dtype = mx.int32)],
        template = [("H", H), ("TOPK", topk)],
        grid = (1024, B * L, 1), threadgroup = (1024, 1, 1),
        output_shapes = [(B, L, topk), (B * L, max(nb, 1))], output_dtypes = [mx.int32, mx.uint32],
    )
    return out


# Tensor-unit QSA prefill attention over the selected blocks (inexact: fp32 sums in another order).
_PREFILL_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define UNROLL _Pragma("clang loop unroll(full)")

// MLX's 16x16 tensor-unit fragment and its 16x32x16 matmul
// (mlx/backend/metal/kernels/steel/attn/nax.h, Copyright © 2025 Apple Inc., MIT).
template <typename C, typename A, typename B, bool TB>
inline void nax_mma(thread vec<C, 8>& c0, thread vec<C, 8>& c1, const thread vec<A, 8>& a,
                    const thread vec<B, 8>& b0, const thread vec<B, 8>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto cb = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), C>();
  UNROLL for (short i = 0; i < 8; i++) {
    ca[i] = a[i];
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  UNROLL for (short i = 0; i < 8; i++) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

template <typename T>
inline T row_max(T x) {
  x = metal::max(x, simd_shuffle_xor(x, ushort(1)));
  return metal::max(x, simd_shuffle_xor(x, ushort(8)));
}

template <typename T>
inline T row_sum(T x) {
  x += simd_shuffle_xor(x, ushort(1));
  return x + simd_shuffle_xor(x, ushort(8));
}
"""

# MLX's NAX flash attention (steel_attention_nax.h) over gathered keys. A threadgroup is one (query row, KV head);
# fragment rows are that KV head's query heads. Simdgroup sg scores half of D, the halves meet in threadgroup
# memory, and sg accumulates P V for its half of D. Keys come 32 at a time from 8 units: the row's sorted picks
# (-1 first), then its partial tail block.
_PREFILL_SOURCE = """
    constexpr int D = 256;
    const ushort lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tq = threadgroup_position_in_grid.x;
    const int kvh = threadgroup_position_in_grid.y;
    const int kL = params[0];
    const int L = params[1];
    const float scale2 = scale[0] * 1.44269504089f;
    threadgroup float partial[2 * 2 * 32 * 16];

    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    const int qe = query_ends[tq];
    const int complete = qe >> 2;
    const int nsel = min(TOPK, complete);
    const int ntail = qe - (complete << 2);
    const int U = nsel + (ntail > 0 ? 1 : 0);
    const device int* blk = sel + size_t(tq) * TOPK + (TOPK - nsel);
    // mlx-vlm sends no pick outside [0, complete); skipping them keeps any input from addressing past the keys.
    auto unit_block = [&](int u) -> int {
      if (u < nsel) {
        const int b = blk[u];
        return uint(b) < uint(complete) ? b : -1;
      }
      return u < U ? complete : -1;
    };

    bool row_ok[2];
    const device bfloat* qrow[2];
    UNROLL for (short i = 0; i < 2; i++) {
      row_ok[i] = fm + 8 * i < GQA;
      qrow[i] = (const device bfloat*)q + (size_t(kvh * GQA + (row_ok[i] ? fm + 8 * i : 0)) * L + tq) * D + fn;
    }
    const device bfloat* kh = (const device bfloat*)k + size_t(kvh) * kL * D + fn;
    const device bfloat* vh = (const device bfloat*)v + size_t(kvh) * kL * D + sg * 128 + fn;

    vec<float, 8> o[8];
    UNROLL for (short d = 0; d < 8; d++) o[d] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};

    for (int u0 = 0; u0 < U; u0 += 8) {
      // Key rows this lane loads (fragment f, element row i) and visible tokens of its score columns.
      uint krow[2][2];
      int nvis[2];
      UNROLL for (short f = 0; f < 2; f++) {
        UNROLL for (short i = 0; i < 2; i++) {
          const short slot = 16 * f + fm + 8 * i;
          const int b = unit_block(u0 + (slot >> 2));
          krow[f][i] = b < 0 ? 0u : uint(min(4 * b + (slot & 3), kL - 1));
        }
        const int u = u0 + ((16 * f + fn) >> 2);
        const int b = unit_block(u);
        nvis[f] = b < 0 ? 0 : (u < nsel ? 4 : ntail);
      }

      vec<float, 8> s[2];
      s[0] = 0.0f;
      s[1] = 0.0f;
      UNROLL for (short d = sg * 8; d < sg * 8 + 8; d++) {
        vec<bfloat, 8> qa, k0, k1;
        UNROLL for (short i = 0; i < 2; i++) {
          const vec<bfloat, 4> qv = row_ok[i] ? *(const device vec<bfloat, 4>*)(qrow[i] + 16 * d) : vec<bfloat, 4>(0);
          const vec<bfloat, 4> a = *(const device vec<bfloat, 4>*)(kh + krow[0][i] * D + 16 * d);
          const vec<bfloat, 4> c = *(const device vec<bfloat, 4>*)(kh + krow[1][i] * D + 16 * d);
          UNROLL for (short j = 0; j < 4; j++) {
            qa[4 * i + j] = qv[j];
            k0[4 * i + j] = a[j];
            k1[4 * i + j] = c[j];
          }
        }
        nax_mma<float, bfloat, bfloat, true>(s[0], s[1], qa, k0, k1);
      }
      threadgroup float* buf = partial + ((u0 >> 3) & 1) * 1024 + lane * 16;
      UNROLL for (short e = 0; e < 8; e++) {
        buf[sg * 512 + e] = s[0][e];
        buf[sg * 512 + 8 + e] = s[1][e];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNROLL for (short e = 0; e < 8; e++) {
        s[0][e] += buf[(1 - sg) * 512 + e];
        s[1][e] += buf[(1 - sg) * 512 + 8 + e];
      }

      UNROLL for (short i = 0; i < 2; i++) {
        float smax = -INFINITY;
        UNROLL for (short f = 0; f < 2; f++) {
          UNROLL for (short j = 0; j < 4; j++) {
            const float x = j < nvis[f] ? s[f][4 * i + j] * scale2 : -INFINITY;
            s[f][4 * i + j] = x;
            smax = metal::max(smax, x);
          }
        }
        const float mnew = metal::max(m[i], row_max(smax));
        const float mref = mnew == -INFINITY ? 0.0f : mnew;
        const float factor = fast::exp2(m[i] - mref);
        m[i] = mnew;
        float psum = 0.0f;
        UNROLL for (short f = 0; f < 2; f++) {
          UNROLL for (short j = 0; j < 4; j++) {
            const float p = fast::exp2(s[f][4 * i + j] - mref);
            s[f][4 * i + j] = p;
            psum += p;
          }
        }
        l[i] = l[i] * factor + row_sum(psum);
        UNROLL for (short d = 0; d < 8; d++) {
          UNROLL for (short j = 0; j < 4; j++) o[d][4 * i + j] *= factor;
        }
      }

      // The tensor units read float operands at 10 mantissa bits; a half part plus its remainder keeps ~22 bits of P.
      vec<float, 8> hi[2], lo[2];
      UNROLL for (short f = 0; f < 2; f++) {
        UNROLL for (short e = 0; e < 8; e++) {
          hi[f][e] = float(half(s[f][e]));
          lo[f][e] = s[f][e] - hi[f][e];
        }
      }
      UNROLL for (short d = 0; d < 8; d += 2) {
        UNROLL for (short f = 0; f < 2; f++) {
          vec<bfloat, 8> v0, v1;
          UNROLL for (short i = 0; i < 2; i++) {
            const vec<bfloat, 4> a = *(const device vec<bfloat, 4>*)(vh + krow[f][i] * D + 16 * d);
            const vec<bfloat, 4> c = *(const device vec<bfloat, 4>*)(vh + krow[f][i] * D + 16 * d + 16);
            UNROLL for (short j = 0; j < 4; j++) {
              v0[4 * i + j] = a[j];
              v1[4 * i + j] = c[j];
            }
          }
          nax_mma<float, float, bfloat, false>(o[d], o[d + 1], hi[f], v0, v1);
          nax_mma<float, float, bfloat, false>(o[d], o[d + 1], lo[f], v0, v1);
        }
      }
    }

    UNROLL for (short i = 0; i < 2; i++) {
      if (!row_ok[i]) continue;
      const float rr = l[i] == 0.0f ? 0.0f : 1.0f / l[i];
      device bfloat* orow = (device bfloat*)out + (size_t(kvh * GQA + fm + 8 * i) * L + tq) * D + sg * 128 + fn;
      UNROLL for (short d = 0; d < 8; d++) {
        vec<bfloat, 4> w;
        UNROLL for (short j = 0; j < 4; j++) w[j] = bfloat(o[d][4 * i + j] * rr);
        *(device vec<bfloat, 4>*)(orow + 16 * d) = w;
      }
    }"""

PREFILL_PROBE_KEY = "qsa_prefill:" + hashlib.sha256((_PREFILL_HEADER + _PREFILL_SOURCE).encode()).hexdigest()[:12]
PREFILL_MIN_ROWS = 64  # fewer rows run faster on the native kernel at long contexts


@functools.cache
def _prefill_kernel():
    return mx.fast.metal_kernel(
        name = "unsloth_qsa_prefill", input_names = ["q", "k", "v", "sel", "query_ends", "params", "scale"],
        output_names = ["out"], header = _PREFILL_HEADER, source = _PREFILL_SOURCE,
    )


def fast_prefill_supported(queries, keys, block_indices, block_size):
    """Whether `fast_prefill_attention` covers this indexed prefill call: the measured Qwen3.8 layout."""
    B, H, L, D = queries.shape
    return (B == 1 and H == 24 and keys.shape[1] == 2 and D == 256 and block_size == 4
            and block_indices.shape[-1] == 512 and L >= PREFILL_MIN_ROWS and queries.dtype == mx.bfloat16)


def fast_prefill_attention(queries, keys, values, block_indices, query_ends, scale):
    """mlx-vlm's indexed QSA prefill attention on the tensor units, for `fast_prefill_supported` calls."""
    _, H, L, D = queries.shape
    return _prefill_kernel()(
        inputs = [queries, keys, values, mx.sort(block_indices.astype(mx.int32), axis = -1),
                  query_ends.astype(mx.int32), mx.array([keys.shape[2], L], dtype = mx.int32),
                  mx.array([scale], dtype = mx.float32)],
        template = [("GQA", H // keys.shape[1]), ("TOPK", block_indices.shape[-1])],
        grid = (64 * L, keys.shape[1], 1), threadgroup = (64, 1, 1),
        output_shapes = [(1, H, L, D)], output_dtypes = [queries.dtype],
    )[0]


def probe_fast_prefill():
    """Compare with the native kernel on shuffled picks, -1 picks, tail blocks and rows with no visible key."""
    from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention
    for L, N in ((64, 4101), (96, 1501), (64, 63)):
        q = mx.random.normal((1, 24, L, 256), key = mx.random.key(N + L)).astype(mx.bfloat16)
        k, v = (mx.random.normal((1, 2, N, 256), key = mx.random.key(N + i)).astype(mx.bfloat16) for i in (0, 1))
        ends = mx.arange(N - L + 1, N + 1, dtype = mx.int32)[None]
        complete = ends // 4
        blocks = max(N // 4, 512)
        scores = mx.random.uniform(shape = (1, L, blocks), key = mx.random.key(3))
        scores = mx.where(mx.arange(blocks)[None, None] < complete[..., None], scores, -mx.inf)
        picks = mx.argpartition(scores, kth = -512, axis = -1)[..., -512:].astype(mx.int32)
        drop = (mx.random.uniform(shape = picks.shape, key = mx.random.key(5)) < 0.05) | (ends[..., None] == 4)
        picks = mx.where((picks < complete[..., None]) & ~drop, picks, -1)
        shuffle = mx.argsort(mx.random.uniform(shape = picks.shape, key = mx.random.key(4)), axis = -1)
        picks = mx.take_along_axis(picks, shuffle, axis = -1)
        native = qsa_sparse_attention(q, k, v, picks, ends, scale = 256 ** -0.5, block_size = 4)
        got = fast_prefill_attention(q, k, v, picks, ends, 256 ** -0.5)
        error = mx.abs(got.astype(mx.float32) - native.astype(mx.float32)).max().item()
        assert error <= 0.02 * mx.abs(native.astype(mx.float32)).max().item(), (L, N, error)
        # Only rounding differs (fp32 summation order, P's last bits), so nearly every bf16 output matches.
        assert mx.mean(got == native).item() >= 0.99, (L, N)
