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

"""Stable on-device counting sort of MoE routing ids, bit-identical to scatter_add count +
stable argsort + int32 cumsum, sync free and CUDA graph capturable. No atomics, so deterministic.
Out of range ids are not counted and go to the tail. UNSLOTH_MOE_TRITON_ROUTING=0 kills it, =1
forces it (e.g. ROCm); the first call per (device, E, launch shape) is verified against torch.
"""

__all__ = [
    "moe_routing_sort",
    "torch_moe_routing_sort",
    "triton_routing_available",
    "ROUTING_CALLS",
]

import contextlib
import logging
import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

_ENV = os.environ.get("UNSLOTH_MOE_TRITON_ROUTING", "auto").strip().lower()
_KILLED = _ENV in ("0", "false", "off", "no")
_FORCED = _ENV in ("1", "true", "on", "yes")
# Composite sort key expert * BLOCK + position fits int32; single programs hold pow2(E + 2) bins.
_MAX_EXPERTS = 2048
_MAX_ROWS = (1 << 31) - (1 << 16)
_SINGLE_BLOCK_MAX = 512
_MULTI_BLOCK = 256
_SCAN_EXPERTS = 8
_SCAN_BLOCKS = 256
_WARPS = (4, 4, 4)   # hist, scan, scatter

ROUTING_CALLS = {"triton": 0, "torch": 0, "fallback": 0}

_DISABLED_REASON = None
_VERIFIED = {}


if triton is not None:

    @triton.jit
    def _load_keys(ids_ptr, idx, N, K_COLS, stride_row, stride_col, E):
        # idx: int64 flat positions. Invalid ids -> E (sentinel row), padding -> E + 1.
        in_range = idx < N
        row = idx // K_COLS
        col = idx - row * K_COLS
        key = tl.load(ids_ptr + row * stride_row + col * stride_col, mask = in_range, other = 0)
        valid = (key >= 0) & (key < E)
        key = tl.where(valid, key, E)
        key = tl.where(in_range, key, E + 1)
        return key.to(tl.int32)

    @triton.jit(do_not_specialize = ["N"])
    def _route_single_kernel(
        ids_ptr, N, K_COLS: tl.constexpr, stride_row: tl.constexpr, stride_col, E: tl.constexpr,  # top-k, top-k, n_experts
        counts_ptr, offsets_ptr, out_ptr,
        BLOCK: tl.constexpr, NUM_BINS: tl.constexpr,
    ):
        pos = tl.arange(0, BLOCK)
        key = _load_keys(ids_ptr, pos.to(tl.int64), N, K_COLS, stride_row, stride_col, E)
        hist = tl.histogram(key, NUM_BINS)
        bins = tl.arange(0, NUM_BINS)
        tl.store(counts_ptr + bins, hist.to(tl.int32), mask = bins < E)
        tl.store(offsets_ptr + bins, tl.cumsum(hist, 0).to(tl.int32), mask = bins < E)
        order = tl.sort(key * BLOCK + pos)
        src = order % BLOCK
        tl.store(out_ptr + pos, src.to(tl.int64), mask = pos < N)

    @triton.jit(do_not_specialize = ["N"])
    def _route_hist_kernel(
        ids_ptr, N, K_COLS: tl.constexpr, stride_row: tl.constexpr, stride_col, E: tl.constexpr,  # top-k, top-k, n_experts
        hist_ptr, below_ptr,
        BLOCK: tl.constexpr, NUM_BINS: tl.constexpr,
    ):
        # Block-major [NB, E + 1]: hist = rows of each expert in this block; below = rows of smaller experts in this block (block-local run start).
        b = tl.program_id(0)
        idx = b.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        key = _load_keys(ids_ptr, idx, N, K_COLS, stride_row, stride_col, E)
        hist = tl.histogram(key, NUM_BINS).to(tl.int32)
        bins = tl.arange(0, NUM_BINS)
        row = b.to(tl.int64) * (E + 1) + bins
        tl.store(hist_ptr + row, hist, mask = bins <= E)
        tl.store(below_ptr + row, tl.cumsum(hist, 0) - hist, mask = bins <= E)

    @triton.jit(do_not_specialize = ["NB"])
    def _route_scan_kernel(
        hist_ptr, below_ptr, counts_ptr, NB, E: tl.constexpr,  # n_experts
        EC: tl.constexpr, BB: tl.constexpr,
    ):
        # One program per EC experts carrying over blocks; rewrites below[b, e] in place to (rows of e in earlier
        # blocks) - (rows of experts < e in block b); counts[e] = total.
        e = tl.program_id(0) * EC + tl.arange(0, EC)
        cols = e[None, :]
        carry = tl.zeros([EC], dtype = tl.int32)
        for b0 in range(0, NB, BB):
            rows = (b0 + tl.arange(0, BB))[:, None]
            m = (rows < NB) & (cols <= E)
            at = rows.to(tl.int64) * (E + 1) + cols
            h = tl.load(hist_ptr + at, mask = m, other = 0)
            below = tl.load(below_ptr + at, mask = m, other = 0)
            tl.store(below_ptr + at, tl.cumsum(h, 0) - h + carry[None, :] - below, mask = m)
            carry += tl.sum(h, 0)
        tl.store(counts_ptr + e, carry, mask = e < E)

    @triton.jit
    def _route_starts_kernel(counts_ptr, starts_ptr, offsets_ptr, E: tl.constexpr, EP: tl.constexpr):
        e = tl.arange(0, EP)
        c = tl.load(counts_ptr + e, mask = e < E, other = 0)
        ends = tl.cumsum(c, 0)
        tl.store(offsets_ptr + e, ends, mask = e < E)
        tl.store(starts_ptr + e, ends - c, mask = e <= E)   # starts[E] = valid rows (sentinel run)

    @triton.jit(do_not_specialize = ["N"])
    def _route_scatter_kernel(
        ids_ptr, N, K_COLS: tl.constexpr, stride_row: tl.constexpr, stride_col, E: tl.constexpr,  # top-k, top-k, n_experts
        rank_base_ptr, starts_ptr, out_ptr,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        pos = tl.arange(0, BLOCK)
        key = _load_keys(ids_ptr, b.to(tl.int64) * BLOCK + pos, N, K_COLS, stride_row, stride_col, E)
        order = tl.sort(key * BLOCK + pos)
        e = order // BLOCK
        src = order % BLOCK
        real = e <= E
        dest = (
            tl.load(starts_ptr + e, mask = real, other = 0)
            + tl.load(rank_base_ptr + b.to(tl.int64) * (E + 1) + e, mask = real, other = 0)
            + pos
        )
        tl.store(out_ptr + dest, b.to(tl.int64) * BLOCK + src, mask = real)


def _pow2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


def _as_2d(top_k_index):
    # [T, k] is read through its strides (a transposed / sliced top-k needs no copy).
    if top_k_index.dim() == 2:
        return top_k_index
    return top_k_index.reshape(-1).unsqueeze(1)


def torch_moe_routing_sort(top_k_index, num_experts):
    """The reference path: scatter_add count, stable argsort, int32 cumsum. Sync free."""
    flat = top_k_index.reshape(-1)
    index = flat if flat.dtype == torch.int64 else flat.long()
    counts = torch.zeros(num_experts, dtype = torch.int32, device = flat.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype = torch.int32))
    sorted_indices = flat.argsort(stable = True)
    offsets = torch.cumsum(counts, dim = 0, dtype = torch.int32)
    return counts, sorted_indices, offsets


def _launch(top_k_index, num_experts, single_block_max = None, block = None):
    ids = _as_2d(top_k_index)
    N = ids.numel()
    E = int(num_experts)
    device = ids.device
    counts = torch.empty(E, dtype = torch.int32, device = device)
    offsets = torch.empty(E, dtype = torch.int32, device = device)
    out = torch.empty(N, dtype = torch.int64, device = device)
    K_COLS = max(ids.shape[1], 1)
    s_row, s_col = ids.stride(0), ids.stride(1)
    single_block_max = _SINGLE_BLOCK_MAX if single_block_max is None else single_block_max
    on_device = device.index == torch.cuda.current_device()
    with contextlib.nullcontext() if on_device else torch.cuda.device(device):
        if N <= single_block_max:
            BLOCK = max(_pow2(N), 16)
            _route_single_kernel[(1,)](
                ids, N, K_COLS, s_row, s_col, E, counts, offsets, out,
                BLOCK = BLOCK, NUM_BINS = _pow2(E + 2),
                num_warps = 4 if BLOCK <= 1024 else 8, enable_fp_fusion = True,
            )
            return counts, out, offsets
        BLOCK = _MULTI_BLOCK if block is None else block
        NB = (N + BLOCK - 1) // BLOCK
        size = NB * (E + 1)
        scratch = torch.empty(2 * size + E + 1, dtype = torch.int32, device = device)
        hist, rank_base, starts = scratch[:size], scratch[size: 2 * size], scratch[2 * size:]
        _route_hist_kernel[(NB,)](
            ids, N, K_COLS, s_row, s_col, E, hist, rank_base,
            BLOCK = BLOCK, NUM_BINS = _pow2(E + 2), num_warps = _WARPS[0], enable_fp_fusion = True,
        )
        EC = min(_SCAN_EXPERTS, _pow2((E + 32) // 32))
        _route_scan_kernel[((E + EC) // EC,)](
            hist, rank_base, counts, NB, E,
            EC = EC, BB = _SCAN_BLOCKS, num_warps = _WARPS[1],   # fixed tile: one variant for every NB
            enable_fp_fusion = True,
        )
        _route_starts_kernel[(1,)](
            counts, starts, offsets, E, EP = _pow2(E + 1), num_warps = 4, enable_fp_fusion = True,
        )
        _route_scatter_kernel[(NB,)](
            ids, N, K_COLS, s_row, s_col, E, rank_base, starts, out,
            BLOCK = BLOCK, num_warps = _WARPS[2], enable_fp_fusion = True,
        )
    return counts, out, offsets


def _disable(exc):
    global _DISABLED_REASON
    _DISABLED_REASON = f"{type(exc).__name__}: {exc}"
    logging.getLogger(__name__).warning(
        "Unsloth: the Triton MoE routing sort failed and is disabled for this process; the torch "
        f"argsort path is used instead. Reason: {_DISABLED_REASON}"
    )


def _is_capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def triton_routing_available(device) -> bool:
    """Static gate: Triton present, not killed or disabled, a CUDA device (ROCm only when forced)."""
    return not (
        triton is None
        or _KILLED
        or _DISABLED_REASON is not None
        or getattr(device, "type", None) != "cuda"
        or (torch.version.hip is not None and not _FORCED)
        or os.environ.get("TRITON_INTERPRET", "0") == "1"
    )


def _verified_launch(top_k_index, num_experts, key):
    """First call per (device, E, launch shape): run both paths on the real input and compare (one
    host sync, sync debug mode suspended). A mismatch disables the kernel for the process."""
    got = _launch(top_k_index, num_experts)
    ref = torch_moe_routing_sort(top_k_index, num_experts)
    mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(0)
        same = all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(got, ref))
    finally:
        torch.cuda.set_sync_debug_mode(mode)
    if not same:
        _disable(RuntimeError(
            f"result differs from the torch path (E={num_experts}, rows={top_k_index.numel()})"
        ))
        return None, ref
    _VERIFIED[key] = True
    return got, ref


def _routing_impl(top_k_index, num_experts):
    N = top_k_index.numel()
    if N <= _MAX_ROWS and 0 < num_experts <= _MAX_EXPERTS and triton_routing_available(top_k_index.device):
        key = (top_k_index.device.index, num_experts, N <= _SINGLE_BLOCK_MAX)
        try:
            if key in _VERIFIED:
                result = _launch(top_k_index, num_experts)
            elif _is_capturing():
                result = None   # cannot verify inside a capture: record the torch path this time
            else:
                result, ref = _verified_launch(top_k_index, num_experts, key)
                if result is None:
                    ROUTING_CALLS["fallback"] += 1
                    ROUTING_CALLS["torch"] += 1
                    return ref
            if result is not None:
                ROUTING_CALLS["triton"] += 1
                return result
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            _disable(exc)
            ROUTING_CALLS["fallback"] += 1
    ROUTING_CALLS["torch"] += 1
    return torch_moe_routing_sort(top_k_index, num_experts)


def _register_op():
    if not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
        return None
    try:
        return torch.ops.unsloth_zoo.moe_routing_sort.default
    except (AttributeError, RuntimeError):
        pass
    try:
        @torch.library.custom_op("unsloth_zoo::moe_routing_sort", mutates_args = ())
        def moe_routing_sort_op(
            top_k_index: torch.Tensor, num_experts: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _routing_impl(top_k_index, num_experts)

        @moe_routing_sort_op.register_fake
        def _moe_routing_sort_fake(top_k_index, num_experts):
            return (
                top_k_index.new_empty((num_experts,), dtype = torch.int32),
                top_k_index.new_empty((top_k_index.numel(),), dtype = torch.int64),
                top_k_index.new_empty((num_experts,), dtype = torch.int32),
            )

        return torch.ops.unsloth_zoo.moe_routing_sort.default
    except Exception:
        return None


_OP = _register_op() if triton is not None else None


def moe_routing_sort(top_k_index, num_experts):
    """(counts int32 [E], argsort(flat, stable) int64 [N], cumsum(counts) int32 [E]), sync free.

    The gate reads only device, constants and shape, so it never guards on data.
    """
    num_experts = int(num_experts)
    if (
        _OP is not None
        and not _KILLED
        and top_k_index.is_cuda
        and (torch.version.hip is None or _FORCED)
        and not top_k_index.is_floating_point()
    ):
        if torch.compiler.is_compiling():
            return _OP(top_k_index, num_experts)
        return _routing_impl(top_k_index, num_experts)
    return torch_moe_routing_sort(top_k_index, num_experts)
