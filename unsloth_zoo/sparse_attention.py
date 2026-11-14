# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Trainable index-gather sparse attention with an attention sink (DeepSeek-V3.2 / V4 / V4.1 style).

`sparse_attn` has exactly the semantics of deepseek-ai's reference `inference/kernel.py::sparse_attn`:
one KV tensor shared by every head serves as both key and value, each query attends to the KV rows listed in
its `topk_idxs` row (-1 = unused slot), logits are `softmax_scale * q.kv`, and the per-head sink logit (not
scaled) only joins the softmax denominator. A row with no valid index returns zero. The reference kernel is
forward only; this one is differentiable in q, kv and attn_sink (indices carry no gradient).

`sparse_attn_window` is the same attention when the selection is "causal sliding window over this chunk plus
selected compressed rows": `kv = cat(window_kv, compress_kv)`, window rows `max(0, t - window + 1) .. t`,
compressed rows `compress_idxs + seq_len`. It lets the gather kernel read the window without index lists.

Backends, first eligible wins (UNSLOTH_SPARSE_ATTN_BACKEND=gather|flex|torch forces one):
  gather  attention-gym gather_attn Triton fwd + bwd (vendored, unsloth_zoo/_vendored/attn_gym), its shared-KV
          schedule only: capability >= 10, bf16, head_dim <= 128 or 512, heads padded to 16 (not ROCm).
  triton  Unsloth's index-gather kernel (unsloth_zoo/sparse_mqa_kernel.py): every other CUDA part, fp16 / bf16 (bf16 on
          sm80+), head_dim a power of two up to 512; tiled gather, online softmax, fp32 atomics for dKV.
  flex    torch flex_attention over a membership block mask, sink applied exactly from the logsumexp; eager only,
          and only while the [b, s, n] membership table stays under 2^28 entries. A membership table counts a
          repeated index once, the reference and gather count every listed slot: rows must not repeat an index
          (DeepSeek's top-k and QSA's block lists never do).
  torch   gather + softmax in fp32, chunked over queries, each chunk recomputed in backward (one [chunk, k, d]
          slab live at a time); also the CPU path and the oracle. A fallback, not a fast path.
A backend whose launch fails is latched off for that shape class (eager only) and the next one runs.
"""

__all__ = [
    "sparse_attn",
    "sparse_attn_window",
    "sparse_attn_reference",
    "window_topk_idxs",
    "sparse_gqa_attn",
    "qsa_select_tokens",
    "SPARSE_ATTN_STATS",
]

import importlib.util
import math
import os

import torch

SPARSE_ATTN_STATS = {"gather": 0, "triton": 0, "flex": 0, "torch": 0, "fallback": 0}

_BACKENDS = ("gather", "triton", "flex", "torch")
_FAILED = {}
_HAS_TRITON = importlib.util.find_spec("triton") is not None
_GATHER = None
_FLEX = None
_BLOCK = 128
# Query rows per chunk of the torch path: bounds the [chunk, k, d] gathered slab.
_TORCH_CHUNK_ELEMS = 1 << 24
# head_dim 512 (DeepSeek): 64-wide flex tiles fault with a misaligned address, see deepseek_v4_sparse_flex.py.
_FLEX_KERNEL_OPTIONS_D512 = {
    "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32,
    "num_warps": 8, "num_stages": 1, "FORCE_USE_FLEX_ATTENTION": True,
}


def _count(key):
    if not torch.compiler.is_compiling():
        SPARSE_ATTN_STATS[key] += 1


def _forced_backend():
    value = os.environ.get("UNSLOTH_SPARSE_ATTN_BACKEND", "").strip().lower()
    return value if value in _BACKENDS else None


def _gather_fn():
    global _GATHER
    if _GATHER is None:
        from unsloth_zoo._vendored.attn_gym.sparse.gather_attn import gather_attn
        _GATHER = gather_attn
    return _GATHER


def _gather_eligible(q, window = 0):
    """gather_attn's shared-KV schedule only: k-tiled tl.dot over the gathered rows, every head sharing one KV
    load. Its other Triton kernels unroll tl.static_range(TOPK) one row at a time (TOPK a compile key; a TOPK of
    515 compiled for over 6 minutes on B200), so those never run here. Mirrors attn-gym's
    can_use_shared_kv_schedule: capability >= 10 (read from q's device), bf16, head_dim <= 128 or 512, window <= 2048,
    heads padded up to 16 (at most 128)."""
    if not (q.is_cuda and _HAS_TRITON and torch.version.hip is None and q.dtype == torch.bfloat16):
        return False
    d, h = q.shape[-1], q.shape[2]
    if d % 16 or not (d <= 128 or d == 512) or h > 128 or window > 2048:
        return False
    return torch.cuda.get_device_capability(q.device)[0] >= 10


# The flex fallback reads a [b, s, n] membership table (S x S for QSA); above this many entries the chunked torch
# path, which never holds more than one [chunk, k, d] slab, runs instead.
_FLEX_MAX_TABLE = 1 << 28


# head_dim 512 flex (32 x 32 tiles, 1 stage) needs about 160 KB of shared memory per block: T4 (64 KB), L4 / sm86 /
# sm89 / sm120 consumer parts (~100 KB) cannot launch it, so they take the torch path without a failing compile.
_FLEX_D512_MIN_SHARED_MEMORY = 160 * 1024
_SHARED_MEMORY = {}


def _shared_memory(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SHARED_MEMORY:
        try:
            _SHARED_MEMORY[index] = getattr(torch.cuda.get_device_properties(index), "shared_memory_per_block_optin", 0)
        except Exception:
            _SHARED_MEMORY[index] = 0
    return _SHARED_MEMORY[index]


def _triton_eligible(q):
    from .sparse_mqa_kernel import sparse_mqa_supported
    return _HAS_TRITON and sparse_mqa_supported(q, q)


def _flex_eligible(q, kv_len = None):
    if not (q.is_cuda and _HAS_TRITON and q.dtype in (torch.float16, torch.bfloat16)):
        return False
    if q.shape[-1] == 512 and _shared_memory(q.device) < _FLEX_D512_MIN_SHARED_MEMORY:
        return False
    return kv_len is None or q.shape[0] * q.shape[1] * kv_len <= _FLEX_MAX_TABLE


def _failure_key(backend, q, k):
    return (backend, q.device.type, q.device.index, q.dtype, q.shape[2], q.shape[3], k)


def _order(q, k, backend = None, kv_len = None):
    forced = backend if backend in _BACKENDS else _forced_backend()
    if forced is not None:
        return (forced,)
    order = []
    if _gather_eligible(q):
        order.append("gather")
    if _triton_eligible(q):
        order.append("triton")
    # Flex is an eager fallback: it compiles flex_attention itself, which does not nest inside a traced region.
    if _flex_eligible(q, kv_len) and not torch.compiler.is_compiling():
        order.append("flex")
    order.append("torch")
    if torch.compiler.is_compiling():
        return tuple(order)
    return tuple(b for b in order if b == "torch" or _failure_key(b, q, k) not in _FAILED)


def window_topk_idxs(seq_len, window_size, batch, device):
    """deepseek-ai `get_window_topk_idxs` for a prefill chunk: row t lists max(0, t-window+1) .. t, -1 padded."""
    width = min(seq_len, window_size)
    end = torch.arange(seq_len, device = device).unsqueeze(1)
    idxs = (end - window_size + 1).clamp(min = 0) + torch.arange(width, device = device)
    idxs = torch.where(idxs > end, -1, idxs)
    return idxs.to(torch.int32).unsqueeze(0).expand(batch, -1, -1)


def sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather + fp32 (fp64 for fp64 inputs) softmax with sink; differentiable; chunked over queries."""
    b, s, h, d = q.shape
    k = topk_idxs.shape[-1]
    acc = torch.promote_types(q.dtype, torch.float32)
    sink = attn_sink.to(acc).view(1, 1, h)
    out = q.new_zeros((b, s, h, d)) if k == 0 else None
    if out is not None:
        return out
    n = kv.shape[1]
    # A zero sentinel row keeps -1 / out-of-range slots indexable.
    kv_pad = torch.cat([kv, kv.new_zeros((b, 1, d))], dim = 1)
    valid = (topk_idxs >= 0) & (topk_idxs < n)
    safe = torch.where(valid, topk_idxs, n).long()
    chunk = max(1, min(s, _TORCH_CHUNK_ELEMS // max(1, k * d * b)))
    batch_index = torch.arange(b, device = q.device).view(b, 1, 1)

    def block(q_chunk, kv_pad, sink, safe_chunk, valid_chunk):
        slots = kv_pad[batch_index, safe_chunk].to(acc)                            # [b, c, k, d]
        logits = torch.einsum("bchd,bckd->bchk", q_chunk.to(acc), slots) * softmax_scale
        row_valid = valid_chunk.unsqueeze(2)                                       # [b, c, 1, k]
        logits = logits.masked_fill(~row_valid, float("-inf"))
        # The sink joins the denominator only; a row with no valid slot returns zero.
        full = torch.cat([logits, sink.expand(b, q_chunk.shape[1], h).unsqueeze(-1)], dim = -1)
        empty = ~row_valid.any(dim = -1, keepdim = True)
        probs = torch.softmax(torch.where(empty, 0, full), dim = -1)[..., :-1]
        probs = torch.where(empty, 0, probs)
        return torch.einsum("bchk,bckd->bchd", probs, slots).to(q.dtype)

    # Under autograd each chunk is recomputed in backward, so only one gathered slab is ever live.
    recompute = torch.is_grad_enabled() and any(t.requires_grad for t in (q, kv, attn_sink)) and s > chunk
    outs = []
    for start in range(0, s, chunk):
        stop = min(s, start + chunk)
        args = (q[:, start:stop], kv_pad, sink, safe[:, start:stop], valid[:, start:stop])
        if recompute:
            from torch.utils.checkpoint import checkpoint
            outs.append(checkpoint(block, *args, use_reentrant = False))
        else:
            outs.append(block(*args))
    return torch.cat(outs, dim = 1)


_MIN_SHARED_HEADS = 16


def _gather_call(q, local_kv, sparse_kv, idxs, attn_sink, softmax_scale, window):
    """q [b, s, h, d]; KV [b, n, d] expanded over heads with stride 0 (the shared schedule's layout); fewer than 16
    heads are padded with zero queries and a -inf sink (their outputs are dropped, their gradients are zero)."""
    b, s, h, d = q.shape
    sink = attn_sink.float()
    heads = max(h, _MIN_SHARED_HEADS)
    if heads != h:
        q = torch.cat([q, q.new_zeros((b, s, heads - h, d))], dim = 2)
        sink = torch.cat([sink, sink.new_full((heads - h,), float("-inf"))])
    expand = lambda kv: None if kv is None else kv.unsqueeze(1).expand(b, heads, kv.shape[1], d)
    out = _gather_fn()(
        q.transpose(1, 2), expand(local_kv), expand(sparse_kv), idxs, sink,
        sliding_window_size = int(window), scale = float(softmax_scale),
    )
    return out.transpose(1, 2)[:, :, :h]


def _sparse_attn_gather(q, kv, attn_sink, topk_idxs, softmax_scale):
    return _gather_call(q, None, kv, topk_idxs, attn_sink, softmax_scale, 0)


def _sparse_attn_window_gather(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size):
    return _gather_call(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)


def _compiled_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention)
    return _FLEX


# 128 < head_dim <= 256 on sm8x: default flex backward tiles were ~10x slower than 32 x 32, 1 stage (A100), and the
# torch 2.10 default forward needed 152 KB of shared memory (L4 has 99 KB).
_FLEX_KERNEL_OPTIONS_SMALL_TILES = {
    "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 1,
    "FORCE_USE_FLEX_ATTENTION": True,
}


def _flex_kernel_options(q):
    d = q.shape[-1]
    if d == 512:
        return _FLEX_KERNEL_OPTIONS_D512
    if d > 128 and torch.cuda.get_device_capability(q.device)[0] < 9:
        return _FLEX_KERNEL_OPTIONS_SMALL_TILES
    # Training / prefill only: below 128 queries Inductor otherwise picks flex_decoding, which rejects this
    # block mask's tiles ("Q and KV block sizes must be divisible") and is forward only.
    return {"FORCE_USE_FLEX_ATTENTION": True}


def _pack(selected, num_blocks):
    count = selected.sum(dim = -1, dtype = torch.int32)
    order = torch.arange(num_blocks, device = selected.device)
    indices = torch.where(selected, order, num_blocks).argsort(dim = -1, stable = True).to(torch.int32)
    return count.unsqueeze(1), indices.unsqueeze(1)


def _membership_block_mask(topk_idxs, kv_len):
    """BlockMask whose mask_mod reads a [b, s, n] membership table built from the index lists."""
    from torch.nn.attention.flex_attention import BlockMask

    b, s, k = topk_idxs.shape
    nq, nk = -(-s // _BLOCK), -(-kv_len // _BLOCK)
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    # Invalid slots scatter into a spare column that is cut off below.
    cols = torch.where(valid, topk_idxs, nk * _BLOCK).long()
    table = torch.zeros((b, nq * _BLOCK, nk * _BLOCK + 1), dtype = torch.bool, device = topk_idxs.device)
    table[:, :s].scatter_(-1, cols, True)
    table = table[..., : nk * _BLOCK]
    tiles = table.view(b, nq, _BLOCK, nk, _BLOCK)
    any_visible = tiles.any(dim = 4).any(dim = 2)
    all_visible = tiles.all(dim = 4).all(dim = 2)
    partial_count, partial_index = _pack(any_visible & ~all_visible, nk)
    full_count, full_index = _pack(all_visible, nk)

    def mask_mod(b_idx, h_idx, q_idx, kv_idx):
        return table[b_idx, q_idx, kv_idx]

    return BlockMask.from_kv_blocks(
        partial_count, partial_index, full_count, full_index,
        BLOCK_SIZE = _BLOCK, mask_mod = mask_mod, seq_lengths = (s, kv_len),
    )


def _sparse_attn_flex(q, kv, attn_sink, topk_idxs, softmax_scale):
    from unsloth_zoo.flex_attention.utils import FLEX_LSE_KWARGS, flex_output_and_lse

    b, s, h, d = q.shape
    block_mask = _membership_block_mask(topk_idxs, kv.shape[1])
    # Duplicated slots: the membership table counts each row once; the reference counts every listed slot.
    query = q.transpose(1, 2)
    kv4 = kv.unsqueeze(1).expand(b, h, kv.shape[1], d)
    out, lse = flex_output_and_lse(_compiled_flex()(
        query, kv4, kv4, block_mask = block_mask, scale = float(softmax_scale),
        kernel_options = _flex_kernel_options(q), **FLEX_LSE_KWARGS,
    ))
    # softmax over keys + sink = flex softmax * exp(lse) / (exp(lse) + exp(sink)); empty rows: lse = -inf -> 0.
    sink = attn_sink.float().view(1, h, 1)
    sink_scale = torch.where(torch.isneginf(sink), 1.0, torch.sigmoid(lse.float() - sink))
    out = (out.float() * sink_scale.unsqueeze(-1)).to(q.dtype)
    return out.transpose(1, 2)


def _run(backend, q, kv, attn_sink, topk_idxs, softmax_scale):
    if backend == "gather":
        return _sparse_attn_gather(q, kv, attn_sink, topk_idxs, softmax_scale)
    if backend == "triton":
        from .sparse_mqa_kernel import sparse_mqa_attention
        return sparse_mqa_attention(q, kv, attn_sink, topk_idxs, softmax_scale)
    if backend == "flex":
        return _sparse_attn_flex(q, kv, attn_sink, topk_idxs, softmax_scale)
    return sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale)


def _latch(backend, q, k, error):
    # Never swallow trace-time, capture or memory failures: the shape class is not at fault.
    if isinstance(error, torch.OutOfMemoryError) or type(error).__module__.startswith("torch._dynamo"):
        raise error
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise error
    _FAILED[_failure_key(backend, q, k)] = repr(error)[:500]
    _count("fallback")
    from .temporary_patches.utils import logger
    logger.warning(f"Unsloth: sparse attention backend '{backend}' failed, falling back: {error}")


def _check(q, kv, attn_sink, topk_idxs):
    if q.dim() != 4 or kv.dim() != 3 or topk_idxs.dim() != 3:
        raise ValueError("sparse_attn expects q [b, s, h, d], kv [b, n, d], topk_idxs [b, s, k]")
    if kv.shape[0] != q.shape[0] or kv.shape[-1] != q.shape[-1] or topk_idxs.shape[:2] != q.shape[:2]:
        raise ValueError(f"sparse_attn shape mismatch: q {tuple(q.shape)}, kv {tuple(kv.shape)}, idxs {tuple(topk_idxs.shape)}")
    if attn_sink.shape != (q.shape[2],):
        raise ValueError(f"attn_sink must be [{q.shape[2]}], got {tuple(attn_sink.shape)}")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, *, backend = None):
    """q [b, s, h, d], kv [b, n, d], attn_sink [h], topk_idxs [b, s, k] int (-1 = unused) -> [b, s, h, d].
    `backend` ("gather" / "flex" / "torch") pins one, like UNSLOTH_SPARSE_ATTN_BACKEND."""
    _check(q, kv, attn_sink, topk_idxs)
    k = topk_idxs.shape[-1]
    if k == 0:
        _count("torch")
        return q.new_zeros(q.shape) + 0 * q
    order = _order(q, k, backend, kv.shape[1])
    for i, backend in enumerate(order):
        if torch.compiler.is_compiling() or i == len(order) - 1:
            out = _run(backend, q, kv, attn_sink, topk_idxs, softmax_scale)
        else:
            try:
                out = _run(backend, q, kv, attn_sink, topk_idxs, softmax_scale)
            except Exception as error:
                _latch(backend, q, k, error)
                continue
        _count(backend)
        return out


def sparse_attn_window(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size, *, backend = None):
    """Window of this chunk's KV + selected compressed rows; equal to sparse_attn over their concatenation.

    q [b, s, h, d], window_kv [b, s, d] (this chunk, already rotated), compress_kv [b, nc, d],
    compress_idxs [b, s, k] into compress_kv (-1 = unused, no offset), window_size > 0.
    """
    b, s, h, d = q.shape
    if window_kv.shape[:2] != (b, s):
        raise ValueError("sparse_attn_window: window_kv must cover the query chunk [b, s, d]")
    forced = backend if backend in _BACKENDS else _forced_backend()
    use_gather = forced == "gather" or (forced is None and _gather_eligible(q, int(window_size)))
    if use_gather:
        key = ("window", int(window_size), compress_idxs.shape[-1])
        if torch.compiler.is_compiling():
            _count("gather")
            return _sparse_attn_window_gather(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)
        try:
            out = _sparse_attn_window_gather(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)
            _count("gather")
            return out
        except Exception as error:
            _latch("gather", q, key, error)
    kv = torch.cat([window_kv, compress_kv], dim = 1)
    window = window_topk_idxs(s, int(window_size), b, q.device)
    compressed = torch.where(compress_idxs >= 0, compress_idxs + s, -1).to(torch.int32)
    return sparse_attn(q, kv, attn_sink, torch.cat([window, compressed], dim = -1), softmax_scale, backend = forced)


def sparse_gqa_attn(query, key, value, topk_idxs, softmax_scale, *, backend = None):
    """Softmax attention of each query over its listed keys only (no sink), K != V, grouped-query heads.

    query [b, hq, s, dk], key [b, hk, n, dk], value [b, hk, n, dv], topk_idxs [b, s, K] shared by every head
    (-1 = unused) -> [b, s, hq, dv]. Head j reads kv head j // (hq // hk), as transformers' repeat_kv does.
    The shared-KV kernels take one tensor as both key and value, so this runs them on [K | V] with the query
    zero-padded over V: the logits are exactly q.K and the second half of the output is the V-weighted sum.
    """
    b, hq, s, dk = query.shape
    hk, dv = key.shape[1], value.shape[-1]
    groups = hq // hk
    forced = backend if backend in _BACKENDS else _forced_backend()
    folded = query.new_empty((1, 1, groups, dk + dv))
    kernel_fold = _gather_eligible(folded) or _triton_eligible(folded)
    if forced == "flex" or (forced is None and not kernel_fold and _flex_eligible(query.transpose(1, 2), key.shape[2])
                            and not torch.compiler.is_compiling()):
        key_ = ("gqa_flex", dk, dv)
        if forced == "flex" or _failure_key("flex", query.transpose(1, 2), key_) not in _FAILED:
            try:
                out = _sparse_gqa_flex(query, key, value, topk_idxs, softmax_scale)
                _count("flex")
                return out
            except Exception as error:
                if forced == "flex":
                    raise
                _latch("flex", query.transpose(1, 2), key_, error)
        if forced is None:
            backend = "torch"
    q = query.reshape(b, hk, groups, s, dk).reshape(b * hk, groups, s, dk).transpose(1, 2)
    q = torch.cat([q, q.new_zeros((b * hk, s, groups, dv))], dim = -1)
    kv = torch.cat([key, value], dim = -1).reshape(b * hk, key.shape[2], dk + dv)
    idxs = topk_idxs.repeat_interleave(hk, dim = 0) if hk > 1 else topk_idxs
    sink = torch.full((groups,), float("-inf"), device = query.device, dtype = torch.float32)
    out = sparse_attn(q, kv, sink, idxs, softmax_scale, backend = backend)[..., dk:]    # [b*hk, s, g, dv]
    return out.reshape(b, hk, s, groups, dv).permute(0, 2, 1, 3, 4).reshape(b, s, hq, dv)


def _sparse_gqa_flex(query, key, value, topk_idxs, softmax_scale):
    """flex_attention with GQA and K != V over the listed keys (membership block mask); no sink."""
    block_mask = _membership_block_mask(topk_idxs, key.shape[2])
    out = _compiled_flex()(query, key, value, block_mask = block_mask, scale = float(softmax_scale),
                           enable_gqa = key.shape[1] != query.shape[1], kernel_options = _flex_kernel_options(query))
    return out.transpose(1, 2)


def _contiguous_rows(visible):
    """(lo, hi) per query row when every row's visible keys form one interval [lo, hi] and every key in it has
    the same first key (causal documents, right padding); None otherwise (sliding windows, empty rows)."""
    b, s, n = visible.shape
    count = visible.sum(-1)
    lo = visible.to(torch.int8).argmax(-1)
    hi = n - 1 - visible.flip(-1).to(torch.int8).argmax(-1)
    ok = (count > 0) & (count == hi - lo + 1)
    if n == s:
        ok = ok & (lo.gather(1, lo) == lo) & (lo.gather(1, hi) == lo)
    else:
        ok = ok & (lo == 0)
    if not bool(ok.all()):
        return None
    return lo, hi


def _qsa_geometry(attention_mask, ratio):
    """(lo, hi, block starts per batch row) for an interval-shaped mask, else None. The scan and the start lists
    need host syncs, so the result is cached on the mask tensor (every QSA layer of a forward, and its gradient
    checkpoint recompute, gets the same mask object), keyed by its version; inference tensors are rescanned."""
    use_cache = not attention_mask.is_inference()
    key = (tuple(attention_mask.shape), attention_mask._version, ratio)
    cached = getattr(attention_mask, "_unsloth_qsa_geometry", None) if use_cache else None
    if cached is not None and cached[0] == key:
        return cached[1]
    visible = attention_mask[:, 0] if attention_mask.dtype == torch.bool else attention_mask[:, 0] == 0
    seq = visible.shape[-1]
    result = None
    bounds = _contiguous_rows(visible)
    if bounds is not None:
        lo, hi = bounds
        positions = torch.arange(seq, device = visible.device)
        starts = []
        for b in range(lo.shape[0]):
            # Block starts: every ratio-th key from its row's first visible key, whole block inside the sequence.
            is_start = ((positions - lo[b]) % ratio == 0) & (positions + ratio - 1 < seq)
            starts.append(torch.nonzero(is_start).flatten())
        result = (lo, hi, starts)
    if use_cache:
        try:
            attention_mask._unsloth_qsa_geometry = (key, result)
        except Exception:
            pass
    return result


@torch.compiler.disable
def qsa_select_tokens(indexer, hidden_states, position_embeddings, attention_mask, apply_rotary_pos_emb,
                      query_chunk = 2048, past_key_values = None):
    """Vectorised transformers Qwen4ExpTextQSAIndexer selection (no cache): [b, s, K] token indices, -1 padded,
    the same set the reference's per-query loop keeps; None when the mask is not interval-shaped.

    Per query: its visible keys split into compress_ratio blocks from its first visible key; complete blocks are
    mean-pooled (fp32), k_layernorm-ed and rotated at their first token; scored by sum over index heads of
    relu(q . k) / sqrt(d); the top budget // ratio blocks plus the incomplete tail are kept.
    `past_key_values` (an empty cache: prefill) receives the raw index keys as the reference indexer stores them,
    only once the mask was accepted, so a None return never leaves the cache updated.
    """
    batch, seq, _ = hidden_states.shape
    ratio, top_blocks, dim = indexer.compress_ratio, indexer.block_topk, indexer.index_head_dim
    if attention_mask.shape[-1] != seq:
        return None
    geometry = _qsa_geometry(attention_mask, ratio)
    if geometry is None:
        return None
    lo, hi, row_starts = geometry
    full_cos, full_sin = position_embeddings
    cos, sin = full_cos[:, -seq:, :], full_sin[:, -seq:, :]
    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [indexer.index_n_heads * dim, indexer.index_kv_heads * dim], dim = -1)
    q = indexer.q_layernorm(q.reshape(batch, seq, -1, dim))
    q = apply_rotary_pos_emb(q, cos = cos, sin = sin, unsqueeze_dim = 2)
    raw_keys = token_k.reshape(batch, seq, -1, dim).squeeze(2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, indexer.layer_idx)

    positions = torch.arange(seq, device = hidden_states.device)
    arange_r = torch.arange(ratio, device = hidden_states.device)
    width = top_blocks * ratio + ratio - 1
    out = torch.full((batch, seq, width), -1, dtype = torch.int32, device = hidden_states.device)
    for b in range(batch):
        # Block starts: every ratio-th key from its row's first visible key, whole block inside the sequence.
        lo_b, hi_b, starts = lo[b], hi[b], row_starts[b]
        if starts.numel() == 0:
            block_keys = None
        else:
            groups = raw_keys[b].index_select(0, (starts[:, None] + arange_r).flatten()).view(-1, ratio, dim)
            pooled = indexer.k_layernorm(groups.float().mean(dim = 1).to(raw_keys.dtype))
            block_keys = apply_rotary_pos_emb(
                pooled.unsqueeze(1), cos = full_cos[b].index_select(0, starts), sin = full_sin[b].index_select(0, starts),
            ).squeeze(1).float()
        tail_start = lo_b + ((hi_b - lo_b + 1) // ratio) * ratio
        tail = tail_start[:, None] + arange_r[: ratio - 1]
        tail = torch.where(tail <= hi_b[:, None], tail, -1)
        out[b, :, top_blocks * ratio :] = tail.to(torch.int32)
        if block_keys is None:
            continue
        for t0 in range(0, seq, query_chunk):
            t1 = min(seq, t0 + query_chunk)
            scores = torch.matmul(q[b, t0:t1].float(), block_keys.transpose(-1, -2)).transpose(-1, -2)
            scores = torch.relu(scores).sum(dim = -1) / math.sqrt(dim)                          # [c, C]
            # Complete blocks of this query's interval only (the same document by construction).
            valid = (starts[None, :] >= lo_b[t0:t1, None]) & (starts[None, :] + ratio - 1 <= hi_b[t0:t1, None])
            scores = scores.masked_fill(~valid, float("-inf"))
            k = min(top_blocks, starts.numel())
            # torch.topk leaves the order of equal scores unspecified (relu makes exact-zero ties common), so
            # ties are broken deterministically: a stable descending sort over the blocks reversed keeps the
            # later block, which is what the reference's CPU topk was observed to do.
            order = scores.flip(-1).sort(dim = -1, descending = True, stable = True)
            top_values = order.values[..., :k]
            top_indices = scores.shape[-1] - 1 - order.indices[..., :k]
            tokens = starts[top_indices][..., None] + arange_r                               # [c, k, ratio]
            tokens = torch.where(torch.isfinite(top_values)[..., None], tokens, -1)
            out[b, t0:t1, : k * ratio] = tokens.flatten(1).to(torch.int32)
    return out
