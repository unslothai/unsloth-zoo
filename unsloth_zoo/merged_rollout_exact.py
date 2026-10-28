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

"""Exact in-place LoRA fold/unfold, no pristine base copy: d = s * B @ A in fp32 must be bit-identical (one kernel,
explicit fma in a fixed order, no fp fusion); unfold restores the original bits of inexact elements from host stores;
a weight-bit checksum raises on mismatch."""

__all__ = [
    "exact_merge_available",
    "ExactFoldTarget",
    "exact_fold",
    "exact_unfold",
]

import os
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

_COUNT, _FOLD, _UNFOLD = 0, 1, 2
_BLOCK_N, _BLOCK_K = 32, 128


_SELF_CHECK = {}


def exact_merge_available(device, dtype = torch.bfloat16):
    """Triton + CUDA, and a one-time fold / unfold self-check on this device and dtype passed."""
    # All Unsloth Zoo code licensed under LGPLv3
    if triton is None or getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return False
    key = (torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device(), dtype)
    if key not in _SELF_CHECK:
        try:
            _SELF_CHECK[key] = _self_check(torch.device("cuda", key[0]), dtype)
        except Exception as error:
            _SELF_CHECK[key] = repr(error)
    return _SELF_CHECK[key] is True
pass


def exact_merge_failure(device, dtype = torch.bfloat16):
    # All Unsloth Zoo code licensed under LGPLv3
    index = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    result = _SELF_CHECK.get((index, dtype), None)
    return None if result is True else result
pass


def _self_check(device, dtype):
    # Bit-exact restore (incl. -0.0 under a zero delta row) and a fold within 1 ulp of torch.
    g = torch.Generator(device = device).manual_seed(0)
    W = (torch.randn(80, 300, device = device, generator = g) * 0.02).to(dtype)
    A = torch.randn(1, 4, 300, device = device, generator = g).to(dtype)
    B = torch.randn(1, 80, 4, device = device, generator = g).to(dtype)
    B[0, 0] = 0
    W[0, :16] = -0.0
    W0 = W.clone()
    t = ExactFoldTarget(W.unsqueeze(0), A, B, 0.37)
    with torch.cuda.device(device):
        exact_fold([t])
        want = (W0.float() + (B[0].double() @ A[0].double() * 0.37).float()).to(dtype)
        ulps = (W.view(torch.int16).int() - want.view(torch.int16).int()).abs().max().item()
        exact_unfold([t])
    if ulps > 1: return f"fold is {ulps} ulp from the reference"
    if not torch.equal(W.view(torch.int16), W0.view(torch.int16)): return "unfold is not bit exact"
    return True
pass


if triton is not None:
    @triton.jit
    def _exact_merge_kernel(
        W, A, B, counts, offsets, idx_out, val_out, checksum,
        N, K,
        sWe, sWn, sWk, sAe, sAr, sAk, sBe, sBn, sBr,
        scale, mode,
        R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        e = tl.program_id(2)
        pid = (e * tl.num_programs(0) + pid_n) * tl.num_programs(1) + pid_k
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mn = rn < N
        mk = rk < K
        mask = mn[:, None] & mk[None, :]

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype = tl.float32)
        for j in tl.static_range(R):
            b = tl.load(B + e * sBe + rn * sBn + j * sBr, mask = mn, other = 0.0).to(tl.float32)
            a = tl.load(A + e * sAe + j * sAr + rk * sAk, mask = mk, other = 0.0).to(tl.float32)
            acc = tl.fma(b[:, None], a[None, :], acc)
        d = acc * scale

        ptrs = W + e * sWe + rn[:, None] * sWn + rk[None, :] * sWk
        w = tl.load(ptrs, mask = mask, other = 0.0)
        if mode == 2:
            back = (w.to(tl.float32) - d).to(w.dtype)
            tl.store(ptrs, back, mask = mask)
            bits = tl.where(mask, back.to(tl.int16, bitcast = True).to(tl.int64), 0)
            tl.atomic_add(checksum, tl.sum(tl.sum(bits, 1), 0))
        else:
            new = (w.to(tl.float32) + d).to(w.dtype)
            back = (new.to(tl.float32) - d).to(w.dtype)
            w_bits = w.to(tl.int16, bitcast = True)
            bad = (back.to(tl.int16, bitcast = True) != w_bits) & mask
            if mode == 0:
                tl.store(counts + pid, tl.sum(tl.sum(bad.to(tl.int32), 1), 0))
                bits = tl.where(mask, w_bits.to(tl.int64), 0)
                tl.atomic_add(checksum, tl.sum(tl.sum(bits, 1), 0))
            else:
                flat_bad = tl.reshape(bad.to(tl.int32), (BLOCK_N * BLOCK_K,))
                pos = tl.load(offsets + pid) + tl.cumsum(flat_bad, 0) - 1
                flat_idx = tl.reshape(
                    (e.to(tl.int64) * N + rn[:, None].to(tl.int64)) * K + rk[None, :].to(tl.int64),
                    (BLOCK_N * BLOCK_K,),
                )
                keep = flat_bad != 0
                tl.store(idx_out + pos, flat_idx, mask = keep)
                tl.store(val_out + pos, tl.reshape(w, (BLOCK_N * BLOCK_K,)), mask = keep)
                tl.store(ptrs, new, mask = mask)
pass


class ExactFoldTarget:
    """One weight to fold, W viewed as [E, N, K]: w_view is its strided base view (E=1 for dense); a [E, R, K] and b [E,
    N, R] take any strides; scale is the LoRA scaling."""
    # All Unsloth Zoo code licensed under LGPLv3
    __slots__ = ("w", "a", "b", "scale", "counts", "checksum", "idx", "val", "host_idx", "host_val", "folded", "ready")

    def __init__(self, w_view, a, b, scale):
        assert w_view.dim() == 3 and a.dim() == 3 and b.dim() == 3
        E, N, K = w_view.shape
        assert a.shape == (E, a.shape[1], K) and b.shape == (E, N, a.shape[1]), (w_view.shape, a.shape, b.shape)
        assert w_view.dtype in (torch.bfloat16, torch.float16)
        self.w, self.a, self.b, self.scale = w_view, a, b, float(scale)
        self.counts = self.checksum = self.idx = self.val = self.host_idx = self.host_val = None
        self.folded = False
        self.ready = None
    pass

    def grid(self):
        E, N, K = self.w.shape
        return (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K), E)
    pass

    def launch(self, mode, offsets = None, idx = None, val = None):
        E, N, K = self.w.shape
        w, a, b = self.w, self.a, self.b
        dummy = self.counts
        _exact_merge_kernel[self.grid()](
            w, a, b, self.counts, offsets if offsets is not None else dummy,
            idx if idx is not None else dummy, val if val is not None else dummy, self.checksum,
            N, K,
            w.stride(0), w.stride(1), w.stride(2),
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            self.scale, mode,
            R = a.shape[1], BLOCK_N = _BLOCK_N, BLOCK_K = _BLOCK_K,
            num_warps = 4, enable_fp_fusion = False,
        )
    pass
pass


def _device_ctx(t):
    return torch.cuda.device(t.device)
pass


def exact_fold(targets, stream = None):
    """Fold targets in place and return {"bad": n, "elements": m}; each target's fix-up store (index, original bits) stays in pinned host memory, copied on stream."""
    # All Unsloth Zoo code licensed under LGPLv3
    total_bad, total_elements = 0, 0
    for t in targets:
        with _device_ctx(t.w):
            n_prog = 1
            for g in t.grid(): n_prog *= g
            t.counts = torch.empty(n_prog, dtype = torch.int32, device = t.w.device)
            t.checksum = torch.zeros(1, dtype = torch.int64, device = t.w.device)
            t.launch(_COUNT)
    sizes = torch.stack([t.counts.sum(dtype = torch.int64) for t in targets]).tolist()
    # Host stores before any weight changes: an allocation failure leaves W untouched.
    _host_buffers(targets, sizes)
    for t, n_bad in zip(targets, sizes):
        with _device_ctx(t.w):
            offsets = torch.cumsum(t.counts, 0, dtype = torch.int64) - t.counts
            t.idx = torch.empty(n_bad, dtype = t.host_idx.dtype, device = t.w.device)
            t.val = torch.empty(n_bad, dtype = t.w.dtype, device = t.w.device)
            t.launch(_FOLD, offsets = offsets, idx = t.idx, val = t.val)
            t.folded = True
            total_bad += n_bad
            total_elements += t.w.numel()
    _offload(targets, stream)
    return {"bad": total_bad, "elements": total_elements}
pass


def _host_buffers(targets, sizes):
    # Pageable when pinning is disabled, over budget or refused (WSL2): still exact, only synchronous.
    from .block_swap import _pin_budget
    idx_types = [torch.int32 if t.w.numel() < 2**31 else torch.int64 for t in targets]
    nbytes = sum(n * (d.itemsize + t.w.element_size()) for t, n, d in zip(targets, sizes, idx_types))
    pin = os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") != "1" and nbytes <= _pin_budget()
    for t, n, d in zip(targets, sizes, idx_types):
        if pin:
            try:
                t.host_idx = torch.empty(n, dtype = d, pin_memory = True)
                t.host_val = torch.empty(n, dtype = t.w.dtype, pin_memory = True)
                continue
            except RuntimeError:
                pin = False
        t.host_idx = torch.empty(n, dtype = d)
        t.host_val = torch.empty(n, dtype = t.w.dtype)
pass


def _offload(targets, stream):
    # Fix-up stores to host on a side stream, so they hold no GPU memory during generation.
    if stream is None:
        for t in targets:
            t.host_idx.copy_(t.idx); t.host_val.copy_(t.val)
            t.idx = t.val = None
        return
    stream.wait_stream(torch.cuda.current_stream(targets[0].w.device))
    with torch.cuda.stream(stream):
        for t in targets:
            t.host_idx.copy_(t.idx, non_blocking = True)
            t.host_val.copy_(t.val, non_blocking = True)
            t.idx.record_stream(stream); t.val.record_stream(stream)
            t.idx = t.val = None
        # The unfold may run on another stream: it waits on this before reading the host stores.
        ready = torch.cuda.Event()
        ready.record(stream)
        for t in targets: t.ready = ready
pass


def exact_unfold(targets, stream = None):
    """Undo exact_fold bit for bit; raises if any checksum disagrees."""
    # All Unsloth Zoo code licensed under LGPLv3
    # A fold that failed part way: undo only what was folded, from wherever its store is.
    targets = [t for t in targets if t.folded]
    if not targets: return
    current = torch.cuda.current_stream(targets[0].w.device)
    for ready in {id(t.ready): t.ready for t in targets if t.ready is not None}.values():
        (stream or current).wait_event(ready)
    if stream is not None:
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for t in targets:
                if t.idx is None:
                    t.idx = t.host_idx.to(t.w.device, non_blocking = True)
                    t.val = t.host_val.to(t.w.device, non_blocking = True)
    else:
        for t in targets:
            if t.idx is None: t.idx = t.host_idx.to(t.w.device); t.val = t.host_val.to(t.w.device)
    folded_sums = []
    for t in targets:
        with _device_ctx(t.w):
            folded_sums.append(t.checksum.clone())
            t.checksum.zero_()
            t.launch(_UNFOLD)
    if stream is not None:
        current.wait_stream(stream)
        for t in targets:
            t.idx.record_stream(current); t.val.record_stream(current)
    checks = []
    for t, before in zip(targets, folded_sums):
        with _device_ctx(t.w):
            E, N, K = t.w.shape
            flat = t.idx.to(torch.int64)
            e, rem = flat // (N * K), flat % (N * K)
            rows, cols = rem // K, rem % K
            wrong = t.w[e, rows, cols]
            t.w[e, rows, cols] = t.val
            after = t.checksum - wrong.view(torch.int16).to(torch.int64).sum() + t.val.view(torch.int16).to(torch.int64).sum()
            checks.append(torch.stack([before[0], after[0]]))
    pairs = torch.stack(checks).tolist()
    for t in targets:
        t.idx = t.val = t.host_idx = t.host_val = t.counts = t.ready = None
        t.folded = False
    bad = [i for i, (x, y) in enumerate(pairs) if x != y]
    if bad:
        raise RuntimeError(
            f"Unsloth: exact LoRA unfold checksum mismatch on {len(bad)} of {len(targets)} weights; "
            "the base weights may be corrupted."
        )
pass
