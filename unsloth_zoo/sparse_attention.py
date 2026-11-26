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
          and only while the bit-packed [b, s, n] membership table stays under 2^31 entries. A membership table counts a
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
from typing import Tuple

import torch

SPARSE_ATTN_STATS = {"gather": 0, "triton": 0, "flex": 0, "torch": 0, "hd512": 0, "fallback": 0}
# QSA selection: query chunks scanned for top-k boundary ties, and rows recomputed the reference's way.
QSA_TIE_STATS = {"chunks": 0, "tied_rows": 0}


@torch.compiler.assume_constant_result
def _env(name, default):
    """os.environ lookup that traces on every supported torch: torch 2.4 cannot inline os.environ.get under
    fullgraph ("Mapping.get" in skipfiles). Traced calls bake the value read at trace time, as 2.6+ already do."""
    return os.environ.get(name, default)


def _exact_ties():
    """UNSLOTH_QWEN4_EXP_EXACT_TIES=1 recomputes tied rows as transformers indexer does; off by
    default (host sync per query chunk, one topk per tied row: L4 medium Qwen 4k 756 vs 641 ms
    per step); without it a tied row may keep another equally-scored block"""
    return _env("UNSLOTH_QWEN4_EXP_EXACT_TIES", "0") == "1"

_BACKENDS = ("gather", "triton", "flex", "torch")
_FAILED = {}
_HAS_TRITON = importlib.util.find_spec("triton") is not None
# flex_attention first ships in torch 2.5; without it the flex backend is never eligible (not tried, failed, latched).
_HAS_FLEX = importlib.util.find_spec("torch.nn.attention.flex_attention") is not None
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


def _forced_backend(window = False):
    """UNSLOTH_SPARSE_ATTN_BACKEND; "hd512" only where it exists (the window-only form)."""
    value = _env("UNSLOTH_SPARSE_ATTN_BACKEND", "").strip().lower()
    return value if value in (_BACKENDS + ("hd512",) if window else _BACKENDS) else None


def _gather_fn():
    global _GATHER
    if not _GATHER:   # None (not tried) or False (failed: re-raise the real import error for a forced call)
        from unsloth_zoo._vendored.attn_gym.sparse.gather_attn import gather_attn
        # Import kernels now: Triton < 3.5 lacks constexpr_function; fail here, not inside a compiled step.
        import unsloth_zoo._vendored.attn_gym.sparse.gather_attn.impl.triton.backward  # noqa: F401
        import unsloth_zoo._vendored.attn_gym.sparse.gather_attn.impl.triton.forward  # noqa: F401
        _GATHER = gather_attn
    return _GATHER


def _gather_importable():
    """The vendored kernels need triton >= 3.5 (triton.constexpr_function); never select gather without them."""
    global _GATHER
    if _GATHER is None and not torch.compiler.is_compiling():
        try:
            _gather_fn()
        except Exception:
            _GATHER = False
    return bool(_GATHER)


def _resolve_gather():
    """Once at import, so traced dispatch only reads _GATHER (a cold trace would skip gather)."""
    global _GATHER
    try:
        _gather_fn()
    except Exception:
        _GATHER = False


def _gather_eligible(q, window = 0):
    """gather_attn's shared-KV schedule only: k-tiled tl.dot over the gathered rows, every head sharing one KV
    load. Its other Triton kernels unroll tl.static_range(TOPK) one row at a time (TOPK a compile key; a TOPK of
    515 compiled for over 6 minutes on B200), so those never run here. Mirrors attn-gym's
    can_use_shared_kv_schedule (capability >= 10, read from q's device; bf16, head_dim <= 128 or 512, window <= 2048,
    heads padded up to 16, at most 128), narrowed to datacenter Blackwell (capability 10.x / 11.x, the parts it was
    written and validated for). sm120 (RTX PRO 6000 / G4 / RTX 50xx: mma.sync, no tcgen05, ~100 KB shared memory)
    and everything below sm100 (T4 / A100 / L4) take the Unsloth Triton kernel instead (sparse_mqa_kernel: one
    compiled variant per head_dim, configs sized to the part's shared memory), never the torch path."""
    if not (q.is_cuda and _HAS_TRITON and torch.version.hip is None and q.dtype == torch.bfloat16):
        return False
    d, h = q.shape[-1], q.shape[2]
    if d % 16 or not (d <= 128 or d == 512) or h > 128 or window > 2048:
        return False
    return torch.cuda.get_device_capability(q.device)[0] in (10, 11) and _gather_importable()


# Max bit-packed membership entries for flex (256 MB, s = 46k at b = 1); above it the fold / torch path runs.
_FLEX_MAX_TABLE = 1 << 31


# head_dim 512 flex (32 x 32 tiles, 1 stage) needs about 160 KB of shared memory per block: T4 (64 KB), L4 / sm86 /
# sm89 / sm120 consumer parts (~100 KB) cannot launch it, so they take the torch path without a failing compile.
_FLEX_D512_MIN_SHARED_MEMORY = 160 * 1024
# head_dim 256 flex failed to compile on T4 (64 KB: "No valid triton configs", 65536 required) and compiled on G4
# (~100 KB); 129..256 therefore needs more than 64 KB.
_FLEX_D256_MIN_SHARED_MEMORY = 96 * 1024
_SHARED_MEMORY = {}


def _query_shared_memory(index):
    try:
        return getattr(torch.cuda.get_device_properties(index), "shared_memory_per_block_optin", 0)
    except Exception:
        return 0


def _shared_memory(device):
    if device.type != "cuda":
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    # Traced code skips the cache: a dict first filled during a trace becomes a guard and recompiles the next call.
    if torch.compiler.is_compiling():
        return _query_shared_memory(index)
    if index not in _SHARED_MEMORY:
        _SHARED_MEMORY[index] = _query_shared_memory(index)
    return _SHARED_MEMORY[index]


def _tensor_core_triton(q):
    """Triton 3.x lowers tl.dot to tensor-core mma.sync only on sm80+; on sm75 (T4) every dot is scalar fp32 FMA
    (0 mma.sync in the sm75 PTX of these kernels), and the measured T4 step was 2.5-3.6x slower than the torch
    path. Automatic dispatch therefore skips the Triton kernels there; UNSLOTH_SPARSE_ATTN_BACKEND=triton (or
    backend = "triton") still forces them."""
    return q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 8


def _triton_eligible(q, forced = False, kv = None):
    if not (_HAS_TRITON and q.is_cuda and q.dim() == 4):
        return False
    from .sparse_mqa_kernel import sparse_mqa_supported
    kv = q.new_empty((q.shape[0], 0, q.shape[-1])) if kv is None else kv
    return sparse_mqa_supported(q, kv) and (forced or _tensor_core_triton(q))


def _flex_eligible(q, kv_len = None):
    if not (q.is_cuda and _HAS_TRITON and _HAS_FLEX and q.dtype in (torch.float16, torch.bfloat16)):
        return False
    # sm75 (T4): flex_attention failed to compile at head_dim 64 and 256 ("No valid triton configs") and the flex
    # training arm failed; sm80+ only.
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        return False
    if q.shape[-1] > 256 and _shared_memory(q.device) < _FLEX_D512_MIN_SHARED_MEMORY:
        return False
    if q.shape[-1] > 128 and _shared_memory(q.device) < _FLEX_D256_MIN_SHARED_MEMORY:
        return False
    return kv_len is None or q.shape[0] * q.shape[1] * kv_len <= _FLEX_MAX_TABLE


# The sink is applied through flex's log-sum-exp (out * sigmoid(lse - sink)); its gradients need flex to propagate
# LSE gradients. Numerically verified on torch 2.14 (B200 oracle, Colab T4 / L4 / G4); CPU flex has no backward,
# so older releases could not be checked here and keep the reference for differentiable calls with a sink.
_FLEX_SINK_VERIFIED_TORCH = (2, 14)


def _torch_version():
    try:
        return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return (0, 0)


def _flex_sink_ok(q, kv, attn_sink):
    if attn_sink is None or _torch_version() >= _FLEX_SINK_VERIFIED_TORCH:
        return True
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, kv, attn_sink)))


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


def _connected_zeros(q, *others):
    """Zero output that stays in every input's autograd graph (zero gradients, not None): empty-slice sums connect
    without reading values (a disabled -inf sink would make `x * 0` NaN)."""
    out = q.new_zeros(q.shape)
    for t in (q,) + others:
        if t is not None and t.is_floating_point():
            out = out + t.reshape(-1)[:0].sum().to(out.dtype)
    return out


def _reference_rows(kv, topk_idxs):
    """kv with a zero sentinel row appended (keeps -1 / out-of-range slots indexable), safe indices, validity."""
    b, n, d = kv.shape
    kv_pad = torch.cat([kv, kv.new_zeros((b, 1, d))], dim = 1)
    valid = (topk_idxs >= 0) & (topk_idxs < n)
    return kv_pad, torch.where(valid, topk_idxs, n).long(), valid


def _reference_block(q_chunk, kv_pad, sink, safe_chunk, valid_chunk, softmax_scale):
    """Rows q_chunk [b, c, h, d] of the reference; sink [1, 1, h] in the accumulation dtype."""
    b, c, h, _ = q_chunk.shape
    acc = sink.dtype
    batch_index = torch.arange(b, device = q_chunk.device).view(b, 1, 1)
    slots = kv_pad[batch_index, safe_chunk].to(acc)                                # [b, c, k, d]
    logits = torch.einsum("bchd,bckd->bchk", q_chunk.to(acc), slots) * softmax_scale
    row_valid = valid_chunk.unsqueeze(2)                                           # [b, c, 1, k]
    logits = logits.masked_fill(~row_valid, float("-inf"))
    # The sink joins the denominator only; a row with no valid slot returns zero.
    full = torch.cat([logits, sink.expand(b, c, h).unsqueeze(-1)], dim = -1)
    empty = ~row_valid.any(dim = -1, keepdim = True)
    probs = torch.softmax(torch.where(empty, 0, full), dim = -1)[..., :-1]
    probs = torch.where(empty, 0, probs)
    return torch.einsum("bchk,bckd->bchd", probs, slots).to(q_chunk.dtype)


def _reference_chunk_rows(q, k):
    b, s, _, d = q.shape
    return max(1, min(s, _TORCH_CHUNK_ELEMS // max(1, k * d * b)))


# Compiled code runs the chunked reference as one opaque op; unrolled, the first step never
# finished.
_REF_V = "v1"


@torch.library.custom_op(f"unsloth_zoo::sparse_attn_chunked_fwd_{_REF_V}", mutates_args = ())
def _chunked_fwd_op(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, topk_idxs: torch.Tensor,
                    softmax_scale: float) -> torch.Tensor:
    s = q.shape[1]
    chunk = _reference_chunk_rows(q, topk_idxs.shape[-1])
    kv_pad, safe, valid = _reference_rows(kv, topk_idxs)
    return torch.cat([
        _reference_block(q[:, a:a + chunk], kv_pad, sink, safe[:, a:a + chunk], valid[:, a:a + chunk], softmax_scale)
        for a in range(0, s, chunk)
    ], dim = 1)


@_chunked_fwd_op.register_fake
def _(q, kv, sink, topk_idxs, softmax_scale):
    return q.new_empty(q.shape)


def _reference_block_backward(grad_out, q_chunk, kv_pad, sink, safe_chunk, valid_chunk, softmax_scale, dkv_pad):
    """Gradients of _reference_block (recomputed forward): returns dq_chunk and dsink [h] (accumulation dtype),
    adds the kv gradient into dkv_pad [b, n + 1, d] (accumulation dtype)."""
    b, c, h, _ = q_chunk.shape
    acc = sink.dtype
    batch_index = torch.arange(b, device = q_chunk.device).view(b, 1, 1).expand_as(safe_chunk)
    q_acc = q_chunk.to(acc)
    slots = kv_pad[batch_index, safe_chunk].to(acc)                                # [b, c, k, d]
    logits = torch.einsum("bchd,bckd->bchk", q_acc, slots) * softmax_scale
    row_valid = valid_chunk.unsqueeze(2)
    logits = logits.masked_fill(~row_valid, float("-inf"))
    full = torch.cat([logits, sink.expand(b, c, h).unsqueeze(-1)], dim = -1)
    empty = ~row_valid.any(dim = -1, keepdim = True)
    probs = torch.where(empty, 0, torch.softmax(torch.where(empty, 0, full), dim = -1))
    p_slots, p_sink = probs[..., :-1], probs[..., -1:]
    go = grad_out.to(acc)
    dp = torch.einsum("bchd,bckd->bchk", go, slots)
    # Softmax backward over [slots | sink]; the sink carries no value, so its dp is 0.
    delta = (p_slots * dp).sum(dim = -1, keepdim = True)
    dlogits = p_slots * (dp - delta)
    dsink = -(p_sink * delta).sum(dim = (0, 1)).squeeze(-1)
    dq = torch.einsum("bchk,bckd->bchd", dlogits, slots) * softmax_scale
    dslots = torch.einsum("bchk,bchd->bckd", p_slots, go) + torch.einsum("bchk,bchd->bckd", dlogits, q_acc) * softmax_scale
    dkv_pad.index_put_((batch_index, safe_chunk), dslots, accumulate = True)
    return dq.to(q_chunk.dtype), dsink


@torch.library.custom_op(f"unsloth_zoo::sparse_attn_chunked_bwd_{_REF_V}", mutates_args = ())
def _chunked_bwd_op(grad_out: torch.Tensor, q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
                    topk_idxs: torch.Tensor, softmax_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s, n = q.shape[1], kv.shape[1]
    chunk = _reference_chunk_rows(q, topk_idxs.shape[-1])
    kv_pad, safe, valid = _reference_rows(kv, topk_idxs)
    dq = q.new_empty(q.shape)
    dkv_pad = torch.zeros(kv_pad.shape, device = kv.device, dtype = sink.dtype)
    dsink = torch.zeros(sink.shape[-1], device = sink.device, dtype = sink.dtype)
    for a in range(0, s, chunk):
        dq[:, a:a + chunk], dsink_chunk = _reference_block_backward(
            grad_out[:, a:a + chunk], q[:, a:a + chunk], kv_pad, sink, safe[:, a:a + chunk], valid[:, a:a + chunk],
            softmax_scale, dkv_pad,
        )
        dsink += dsink_chunk
    return dq, dkv_pad[:, :n].to(kv.dtype).contiguous(), dsink.view(sink.shape)


@_chunked_bwd_op.register_fake
def _(grad_out, q, kv, sink, topk_idxs, softmax_scale):
    return q.new_empty(q.shape), kv.new_empty(kv.shape), sink.new_empty(sink.shape)


def _chunked_setup_context(ctx, inputs, output):
    q, kv, sink, topk_idxs, softmax_scale = inputs
    ctx.save_for_backward(q, kv, sink, topk_idxs)
    ctx.softmax_scale = softmax_scale


def _chunked_backward(ctx, grad_out):
    q, kv, sink, topk_idxs = ctx.saved_tensors
    dq, dkv, dsink = _chunked_bwd_op(grad_out.contiguous(), q, kv, sink, topk_idxs, ctx.softmax_scale)
    need = ctx.needs_input_grad
    return dq if need[0] else None, dkv if need[1] else None, dsink if need[2] else None, None, None


_chunked_fwd_op.register_autograd(_chunked_backward, setup_context = _chunked_setup_context)


def sparse_attn_reference(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather + fp32 (fp64 for fp64 inputs) softmax with sink; differentiable; chunked over queries."""
    b, s, h, d = q.shape
    k = topk_idxs.shape[-1]
    acc = torch.promote_types(q.dtype, torch.float32)
    if attn_sink is None:
        sink = torch.full((1, 1, h), float("-inf"), dtype = acc, device = q.device)
    else:
        sink = attn_sink.to(acc).view(1, 1, h)
    if k == 0:
        return _connected_zeros(q, kv, attn_sink)
    chunk = _reference_chunk_rows(q, k)
    if s > chunk and torch.compiler.is_compiling():
        return _chunked_fwd_op(q, kv, sink, topk_idxs, float(softmax_scale))
    kv_pad, safe, valid = _reference_rows(kv, topk_idxs)

    def block(q_chunk, kv_pad, sink, safe_chunk, valid_chunk):
        return _reference_block(q_chunk, kv_pad, sink, safe_chunk, valid_chunk, softmax_scale)

    if s <= chunk:
        # One chunk: a Python loop over the length recompiles per length even under dynamic=True.
        return block(q, kv_pad, sink, safe, valid)
    # Under autograd each chunk is recomputed in backward, so only one gathered slab is ever live.
    recompute = torch.is_grad_enabled() and any(t.requires_grad for t in (q, kv, attn_sink) if t is not None)
    outs = []
    for start in range(0, s, chunk):
        stop = min(s, start + chunk)
        args = (q[:, start:stop], kv_pad, sink, safe[:, start:stop], valid[:, start:stop])
        if recompute:
            from .gradient_checkpointing import _torch_checkpoint
            outs.append(_torch_checkpoint()(block, *args, use_reentrant = False))
        else:
            outs.append(block(*args))
    return torch.cat(outs, dim = 1)


_MIN_SHARED_HEADS = 16


def _single_kv_head(kv):
    return None if kv is None else kv.unsqueeze(1)


def _gather_call(q, local_kv, sparse_kv, idxs, attn_sink, softmax_scale, window):
    """q [b, s, h, d]; KV [b, n, d] expanded over heads with stride 0 (the shared schedule's layout); fewer than 16
    heads are padded with zero queries and a -inf sink (their outputs are dropped, their gradients are zero)."""
    b, s, h, d = q.shape
    if attn_sink is None:
        sink = torch.full((h,), float("-inf"), device = q.device, dtype = torch.float32)
    else:
        sink = attn_sink.float()
    heads = max(h, _MIN_SHARED_HEADS)
    if heads != h:
        q = torch.cat([q, q.new_zeros((b, s, heads - h, d))], dim = 2)
        sink = torch.cat([sink, sink.new_full((heads - h,), float("-inf"))])
    # TOPK is a constexpr + tuner key and DeepSeek's width varies with length: pad with -1 slots to a power of two
    # (>= 16). Eager only: traced, bit_length would guard on every exact width.
    k = idxs.shape[-1]
    if isinstance(k, int) and k and not torch.compiler.is_compiling():
        width = max(16, 1 << (k - 1).bit_length())
        if width != k:
            idxs = torch.cat([idxs, idxs.new_full((*idxs.shape[:-1], width - k), -1)], dim = -1)
    # One KV head ([b, 1, n, d]): gather_attn records share_kv = True, so the backward keeps the shared schedule
    # (a pre-expanded [b, heads, n, d] view is treated as per-head KV and allocates per-head gradients).
    out = _gather_fn()(
        q.transpose(1, 2), _single_kv_head(local_kv), _single_kv_head(sparse_kv), idxs, sink,
        sliding_window_size = int(window), scale = float(softmax_scale),
    )
    return out.transpose(1, 2)[:, :, :h].contiguous()


def _sparse_attn_gather(q, kv, attn_sink, topk_idxs, softmax_scale):
    return _gather_call(q, None, kv, topk_idxs, attn_sink, softmax_scale, 0)


def _sparse_attn_window_gather(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size):
    return _gather_call(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)


def _compiled_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        from unsloth_zoo.rocm_flex_static import rocm_static_flex

        def rocm_static_entry(query, key, value, *args, **kwargs):
            return flex_attention(query, key, value, *args, **kwargs)
        # ROCm: calls whose backward will run get a static compile (symbolic lengths gave NaN grads on gfx1151).
        _FLEX = rocm_static_flex(torch.compile(flex_attention), rocm_static_entry)
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
    """BlockMask reading a bit-packed [b, s, n / 32] int32 membership table (8x smaller than bool, kept for backward)."""
    from torch.nn.attention.flex_attention import BlockMask

    b, s, k = topk_idxs.shape
    nq, nk = -(-s // _BLOCK), -(-kv_len // _BLOCK)
    words = nk * _BLOCK // 32
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    # Invalid slots land in a spare word / tile that is cut off below.
    safe = torch.where(valid, topk_idxs, nk * _BLOCK).long()
    rows = (b, nq * _BLOCK)
    # Rows never repeat an index (see the module docstring), so adding distinct bits is an OR; int32 wraps bit 31.
    bit = torch.where(valid, torch.ones((), dtype = torch.int32, device = safe.device) << (safe & 31).int(), 0)
    bits = torch.zeros((*rows, words + 1), dtype = torch.int32, device = safe.device)
    bits[:, :s].scatter_add_(-1, safe >> 5, bit)
    bits = bits[..., :words]
    # Tile visible if any row sees a key in it, full if every row sees all its keys.
    counts = torch.zeros((*rows, nk + 1), dtype = torch.int32, device = safe.device)
    counts[:, :s].scatter_add_(-1, safe // _BLOCK, valid.int())
    tiles = counts[..., :nk].view(b, nq, _BLOCK, nk)
    any_visible = (tiles > 0).any(dim = 2)
    all_visible = (tiles == _BLOCK).all(dim = 2)
    partial_count, partial_index = _pack(any_visible & ~all_visible, nk)
    full_count, full_index = _pack(all_visible, nk)

    def mask_mod(b_idx, h_idx, q_idx, kv_idx):
        return ((bits[b_idx, q_idx, kv_idx >> 5] >> (kv_idx & 31)) & 1) != 0

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
    # Infinities are replaced before the subtraction: -inf - (-inf) would make the sink gradient NaN.
    if attn_sink is not None:
        sink = attn_sink.float().view(1, h, 1)
        no_sink, empty = torch.isneginf(sink), torch.isneginf(lse)
        diff = torch.where(empty, 0.0, lse.float()) - torch.where(no_sink, 0.0, sink)
        sink_scale = torch.where(no_sink, 1.0, torch.where(empty, 0.0, torch.sigmoid(diff)))
        out = (out.float() * sink_scale.unsqueeze(-1)).to(q.dtype)
    return out.transpose(1, 2).contiguous()


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
    # Re-raise trace, capture and memory failures; latch a backend's own compile failure (torch 2.6) otherwise.
    from torch._dynamo.exc import BackendCompilerFailed
    inner = getattr(error, "inner_exception", error) if isinstance(error, BackendCompilerFailed) else error
    if isinstance(inner, torch.OutOfMemoryError) or (inner is error and type(error).__module__.startswith("torch._dynamo")):
        raise error
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise error
    _FAILED[_failure_key(backend, q, k)] = repr(error)[:500]
    _count("fallback")
    from unsloth_zoo.temporary_patches.utils import logger
    logger.warning(f"Unsloth: sparse attention backend '{backend}' failed, falling back: {error}")


def _check(q, kv, attn_sink, topk_idxs):
    if q.dim() != 4 or kv.dim() != 3 or topk_idxs.dim() != 3:
        raise ValueError("sparse_attn expects q [b, s, h, d], kv [b, n, d], topk_idxs [b, s, k]")
    if kv.shape[0] != q.shape[0] or kv.shape[-1] != q.shape[-1] or topk_idxs.shape[:2] != q.shape[:2]:
        raise ValueError(f"sparse_attn shape mismatch: q {tuple(q.shape)}, kv {tuple(kv.shape)}, idxs {tuple(topk_idxs.shape)}")
    if attn_sink is not None and tuple(attn_sink.shape) != (q.shape[2],):
        raise ValueError(f"attn_sink must be [{q.shape[2]}], got {tuple(attn_sink.shape)}")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, *, backend = None):
    """q [b, s, h, d], kv [b, n, d], attn_sink [h], topk_idxs [b, s, k] int (-1 = unused) -> [b, s, h, d].
    `backend` ("gather" / "flex" / "torch") pins one, like UNSLOTH_SPARSE_ATTN_BACKEND."""
    _check(q, kv, attn_sink, topk_idxs)
    k = topk_idxs.shape[-1]
    if k == 0:
        _count("torch")
        return _connected_zeros(q, kv, attn_sink)
    order = _order(q, k, backend, kv.shape[1])
    if "flex" in order and not _flex_sink_ok(q, kv, attn_sink):
        order = tuple(b for b in order if b != "flex") or ("torch",)
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
    """Equals sparse_attn over this chunk KV plus selected compressed rows (-1 = unused, no offset);
    q [b,s,h,d], window_kv [b,s,d] already rotated, compress_kv [b,nc,d], compress_idxs [b,s,k],
    window_size > 0"""
    _check(q, compress_kv, attn_sink, compress_idxs)
    b, s, h, d = q.shape
    if window_kv.dim() != 3 or tuple(window_kv.shape) != (b, s, d):
        raise ValueError(f"sparse_attn_window: window_kv must be [b, s, d] = {(b, s, d)}, got {tuple(window_kv.shape)}")
    if int(window_size) <= 0:
        raise ValueError(f"sparse_attn_window: window_size must be positive, got {window_size}")
    forced = backend if backend in _BACKENDS + ("hd512",) else _forced_backend(window = True)
    # Window-only (DeepSeek compress_ratio 0 or no compressed slot): head_dim 512 kernel with sink
    # exact; L4 4.8/9.9 ms at 4k/8k vs 9.5/19.5 ms for window kernel; half the peak memory
    if (compress_idxs.shape[-1] == 0 or compress_kv.shape[1] == 0) and forced in (None, "hd512") and _HAS_TRITON:
        from .sparse_mqa_kernel import window_hd512_attention, window_hd512_supported
        key = ("hd512_window", int(window_size))
        if window_hd512_supported(q, window_kv) and (forced == "hd512" or _failure_key("hd512", q, key) not in _FAILED):
            if torch.compiler.is_compiling() or forced == "hd512":
                out = window_hd512_attention(q, window_kv, attn_sink, softmax_scale, window_size)
                _count("hd512")
                return out
            try:
                out = window_hd512_attention(q, window_kv, attn_sink, softmax_scale, window_size)
                _count("hd512")
                return out
            except Exception as error:
                _latch("hd512", q, key, error)
        if forced == "hd512":
            forced = None
    key = ("window", int(window_size), compress_idxs.shape[-1])
    use_gather = forced == "gather" or (
        forced is None and _gather_eligible(q, int(window_size)) and _failure_key("gather", q, key) not in _FAILED)
    if use_gather:
        if torch.compiler.is_compiling() or forced == "gather":
            out = _sparse_attn_window_gather(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)
            _count("gather")
            return out
        try:
            out = _sparse_attn_window_gather(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)
            _count("gather")
            return out
        except Exception as error:
            _latch("gather", q, key, error)
    if forced in (None, "triton") and _HAS_TRITON:
        from .sparse_mqa_kernel import sparse_mqa_window_attention, sparse_mqa_window_supported
        key = ("window", compress_idxs.shape[-1])
        if sparse_mqa_window_supported(q, window_kv) and (forced == "triton" or (
                _tensor_core_triton(q) and _failure_key("triton", q, key) not in _FAILED)):
            # Banded window kernel over the contiguous window rows + the index kernel over the compressed slots.
            if torch.compiler.is_compiling() or forced == "triton":
                out = sparse_mqa_window_attention(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)
                _count("triton")
                return out
            try:
                out = sparse_mqa_window_attention(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size)
                _count("triton")
                return out
            except Exception as error:
                _latch("triton", q, key, error)
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
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("sparse_gqa_attn expects query [b, hq, s, dk], key [b, hk, n, dk], value [b, hk, n, dv]")
    b, hq, s, dk = query.shape
    hk, dv = key.shape[1], value.shape[-1]
    if hk <= 0 or hq % hk:
        raise ValueError(f"sparse_gqa_attn: {hq} query heads are not a multiple of {hk} KV heads")
    if key.shape[0] != b or value.shape[:3] != key.shape[:3] or key.shape[-1] != dk:
        raise ValueError(f"sparse_gqa_attn shape mismatch: query {tuple(query.shape)}, key {tuple(key.shape)}, "
                         f"value {tuple(value.shape)}")
    if topk_idxs.dim() != 3 or tuple(topk_idxs.shape[:2]) != (b, s):
        raise ValueError(f"sparse_gqa_attn: topk_idxs must be [b, s, K] = [{b}, {s}, K], got {tuple(topk_idxs.shape)}")
    groups = hq // hk
    forced = backend if backend in _BACKENDS else _forced_backend()
    folded = query.new_empty((1, 1, groups, dk + dv))
    kernel_fold = _gather_eligible(folded) or _triton_eligible(folded)
    # Flex first where it compiles at the native head dims (QSA's 8 / 2 heads x 256 on G4: 4.1 ms and 0.17 GB vs
    # 36.2 ms and 0.44 GB for the [K | V] fold through the Triton kernel at 4k), the fold otherwise; flex stays out
    # of traced regions (it compiles flex_attention itself).
    if forced == "flex" or (forced is None and _flex_eligible(query.transpose(1, 2), key.shape[2])
                            and _flex_eligible(value.transpose(1, 2)) and not torch.compiler.is_compiling()):
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
        if forced is None and not kernel_fold:
            backend = "torch"
    q = query.reshape(b, hk, groups, s, dk).reshape(b * hk, groups, s, dk).transpose(1, 2)
    q = torch.cat([q, q.new_zeros((b * hk, s, groups, dv))], dim = -1)
    kv = torch.cat([key, value], dim = -1).reshape(b * hk, key.shape[2], dk + dv)
    sink = torch.full((groups,), float("-inf"), device = query.device, dtype = torch.float32)
    if hk > 1 and backend != "torch" and forced in (None, "triton") and _triton_eligible(q, forced == "triton") and (
            forced == "triton" or _failure_key("triton", q, topk_idxs.shape[-1]) not in _FAILED):
        # The Triton kernel reads one index row per hk folded batches: no [b * hk, s, K] repeated copy kept for
        # backward (128 MB per layer at 4k tokens with K = 2051, hk = 2).
        from .sparse_mqa_kernel import sparse_mqa_attention
        out = None
        if forced == "triton" or torch.compiler.is_compiling():
            out = sparse_mqa_attention(q, kv, sink, topk_idxs, softmax_scale)
        else:
            try:
                out = sparse_mqa_attention(q, kv, sink, topk_idxs, softmax_scale)
            except Exception as error:
                _latch("triton", q, topk_idxs.shape[-1], error)
        if out is not None:
            _count("triton")
            return out[..., dk:].reshape(b, hk, s, groups, dv).permute(0, 2, 1, 3, 4).reshape(b, s, hq, dv)
    idxs = topk_idxs.repeat_interleave(hk, dim = 0) if hk > 1 else topk_idxs
    out = sparse_attn(q, kv, sink, idxs, softmax_scale, backend = backend)[..., dk:]    # [b*hk, s, g, dv]
    return out.reshape(b, hk, s, groups, dv).permute(0, 2, 1, 3, 4).reshape(b, s, hq, dv)


def sparse_gqa_fast_path(query, key, value):
    """True when sparse_gqa_attn has a kernel for these operands (flex, or the [K | V] fold on gather / Triton), or
    a backend is forced. False (T4 / sm75, CPU) means only the chunked torch path is left, which is slower than the
    model's own dense attention there: callers keep their upstream path."""
    if _forced_backend() is not None:
        return True
    groups = query.shape[1] // key.shape[1]
    folded = query.new_empty((1, 1, groups, key.shape[-1] + value.shape[-1]))
    if _gather_eligible(folded) or _triton_eligible(folded):
        return True
    # Same policy as sparse_gqa_attn: flex only outside traced regions.
    return (not torch.compiler.is_compiling() and _flex_eligible(query.transpose(1, 2), key.shape[2])
            and _flex_eligible(value.transpose(1, 2)))


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
        # Every key inside [lo, hi] must start at the same first key, not only the endpoints: rows with equal lo
        # must form one run (segment id equal at both ends of the interval).
        seg = torch.cat([lo.new_zeros(b, 1), (lo[:, 1:] != lo[:, :-1]).long()], 1).cumsum(1)
        ok = ok & (lo.gather(1, lo) == lo) & (seg.gather(1, lo) == seg.gather(1, hi))
    else:
        ok = ok & (lo == 0)
    if not bool(ok.all()):
        return None
    return lo, hi


def _qsa_geometry(attention_mask, ratio):
    """(lo, hi, block starts per batch row) for an interval-shaped mask, else None. The scan and the start lists
    need host syncs, so the result is cached on the mask tensor (every QSA layer of a forward, and its gradient
    checkpoint recompute, gets the same mask object), keyed by its version; inference tensors are rescanned."""
    # One shared selection per batch row: per-head masks ([b, h > 1, s, n]) cannot be represented.
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 4 or attention_mask.shape[1] != 1:
        return None
    # Inference tensors have no version counter: never read it (or cache on them).
    use_cache = not attention_mask.is_inference()
    key = (tuple(attention_mask.shape), attention_mask._version, ratio) if use_cache else None
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


def _repair_boundary_ties(out, b, t0, scores, valid, k, q_rows, block_keys, starts, arange_r, top_blocks, ratio, dim):
    """torch.topk tie order is unspecified; queries whose k-th and (k+1)-th blocks tie or nearly tie
    (batched scores differ in last bits) are recomputed as the transformers indexer does."""
    best = scores.topk(k + 1, dim = -1).values
    kth, nxt = best[:, k - 1], best[:, k]
    tied = torch.isfinite(nxt) & ((kth - nxt).abs() <= 1e-5 * kth.abs().clamp(min = 1.0))
    rows = tied.nonzero().flatten().tolist()
    QSA_TIE_STATS["chunks"] += 1
    QSA_TIE_STATS["tied_rows"] += len(rows)
    for r in rows:
        candidates = valid[r].nonzero().flatten()
        row = torch.matmul(q_rows[r].float(), block_keys.index_select(0, candidates).transpose(-1, -2)).transpose(-1, -2)
        row = torch.relu(row).sum(dim = -1) / math.sqrt(dim)
        chosen = row.topk(min(top_blocks, candidates.numel()), dim = 0).indices
        tokens = (starts[candidates[chosen]][:, None] + arange_r).flatten().to(torch.int32)
        out[b, t0 + r, : top_blocks * ratio] = -1
        out[b, t0 + r, : tokens.numel()] = tokens


@torch.compiler.disable
def qsa_select_tokens(indexer, hidden_states, position_embeddings, attention_mask, apply_rotary_pos_emb,
                      query_chunk = 2048, past_key_values = None, exact_ties = None):
    """Returns [b, s, K] token indices, -1 padded; None if the mask is not interval-shaped (cache
    untouched); UNSLOTH_QWEN4_EXP_EXACT_TIES=1 keeps the reference tie set, else a k-th tie may
    keep another equal block."""
    batch, seq, _ = hidden_states.shape
    ratio, top_blocks, dim = indexer.compress_ratio, indexer.block_topk, indexer.index_head_dim
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 4 or attention_mask.shape[1] != 1
            or attention_mask.shape[-1] != seq):
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
            order = scores.topk(k, dim = -1)
            top_values, top_indices = order.values, order.indices
            tokens = starts[top_indices][..., None] + arange_r                               # [c, k, ratio]
            tokens = torch.where(torch.isfinite(top_values)[..., None], tokens, -1)
            out[b, t0:t1, : k * ratio] = tokens.flatten(1).to(torch.int32)
            if k < starts.numel() and (_exact_ties() if exact_ties is None else exact_ties):
                _repair_boundary_ties(out, b, t0, scores, valid, k, q[b, t0:t1], block_keys, starts, arange_r,
                                      top_blocks, ratio, dim)
    return out


if _HAS_TRITON:
    _resolve_gather()
