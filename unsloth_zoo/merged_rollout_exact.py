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
explicit fma in a fixed order, no fp fusion); unfold restores the original bits from a compact host store; a weight-bit
checksum raises on mismatch.

The store needs no positions. Unfold sees the folded value new = round(W + d) and recomputes d, so it knows
back = round(new - d) and the exponent gap r = exp(new) - exp(back): when r <= 0 the map W -> round(W + d) is one to
one there and back == W (no bits stored), and when r >= 1 about 2^r originals round to the same new, so the original
lies within 2^(r - 1) steps of back in the monotone order of the float bit patterns. Each element with 1 <= r <= 7
stores c = ord(W) - ord(back) in r + 1 bits, packed per tile in a bit stream whose layout both sides compute from
(new, d) alone. Anything else that does not round-trip (ties, |c| out of range, inf / NaN) is an escape: flat index and
original bits, applied after the stream decode."""

__all__ = [
    "exact_merge_available",
    "ExactFoldTarget",
    "exact_fold",
    "exact_unfold",
]

import contextlib
import os
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

_COUNT, _FOLD, _UNFOLD = 0, 1, 2
_BLOCK_N, _BLOCK_K = 32, 128
# Widest code: r + 1 bits for an exponent gap r <= 7, so a code spans at most two 32-bit words.
_MAX_GAP = 7


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
    def _ord(b):
        # int16 bit pattern (as int32) -> position in the real order (-0.0 just below +0.0); its own inverse.
        return tl.where(b < 0, -b - 32769, b)

    @triton.jit
    def _width(nb, bb, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr):
        # Code bits from the folded value and back only (unfold knows both): r + 1 for an exponent gap 1 <= r <= MAX_GAP.
        en = tl.maximum((nb >> MANT) & EXPM, 1)
        eb = tl.maximum((bb >> MANT) & EXPM, 1)
        r = en - eb
        return tl.where((r >= 1) & (r <= MAX_GAP), r + 1, 0)

    # mode (0 / 1 / 2) is a runtime branch: one compiled variant for every mode, not a separate mode == 1 one.
    @triton.jit(do_not_specialize = ["mode", "n_words"])
    def _exact_merge_kernel(
        W, A, B, counts, woff, words, idx_out, val_out, eoff, checksum, n_words,
        N, K,
        sWe, sWn, sWk, sAe, sAr, sAk, sBe, sBn, sBr,
        scale, mode,
        R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  # LoRA rank / tile over out rows / tile over in cols
        MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,  # mantissa bits / exponent mask of W's dtype
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
        # Each mode lives in its own helper: a runtime branch must not leave differently typed names behind.
        if mode == 2:
            _unfold_tile(w, d, ptrs, mask, pid, woff, words, checksum, n_words, BLOCK_N, BLOCK_K, MANT, EXPM, MAX_GAP)
        elif mode == 0:
            _count_tile(w, d, mask, pid, counts, checksum, MANT, EXPM, MAX_GAP)
        else:
            _fold_tile(
                w, d, ptrs, mask, pid, e, rn, rk, N, K, woff, words, idx_out, val_out, eoff,
                BLOCK_N, BLOCK_K, MANT, EXPM, MAX_GAP,
            )

    @triton.jit
    def _classify(w, d, mask, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr):
        # (folded value, original bits, code width, correction, inexact, escape) of each element.
        new = (w.to(tl.float32) + d).to(w.dtype)
        back = (new.to(tl.float32) - d).to(w.dtype)
        wb = w.to(tl.int16, bitcast = True).to(tl.int32)
        nb = new.to(tl.int16, bitcast = True).to(tl.int32)
        bb = back.to(tl.int16, bitcast = True).to(tl.int32)
        bad = (bb != wb) & mask
        k = tl.where(mask, _width(nb, bb, MANT, EXPM, MAX_GAP), 0)
        c = _ord(wb) - _ord(bb)
        half = 1 << (tl.maximum(k, 1) - 1)
        esc = bad & ((k == 0) | (c <= -half) | (c >= half))
        return new, wb, k, tl.where(esc, -half, c), bad, esc

    @triton.jit
    def _count_tile(w, d, mask, pid, counts, checksum, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr):
        new, wb, k, c, bad, esc = _classify(w, d, mask, MANT, EXPM, MAX_GAP)
        tl.store(counts + pid * 3 + 0, tl.sum(tl.sum(bad.to(tl.int32), 1), 0))
        tl.store(counts + pid * 3 + 1, tl.sum(tl.sum(k, 1), 0))
        tl.store(counts + pid * 3 + 2, tl.sum(tl.sum(esc.to(tl.int32), 1), 0))
        # A tile's bit-pattern sum fits int32 (|sum| <= 32 * 128 * 32768 = 2^27): 64-bit math once per tile.
        tl.atomic_add(checksum, tl.sum(tl.sum(tl.where(mask, wb, 0), 1), 0).to(tl.int64))

    @triton.jit
    def _fold_tile(
        w, d, ptrs, mask, pid, e, rn, rk, N, K, woff, words, idx_out, val_out, eoff,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,
    ):
        TILE: tl.constexpr = BLOCK_N * BLOCK_K
        new, wb, k, c, bad, esc = _classify(w, d, mask, MANT, EXPM, MAX_GAP)
        fk = tl.reshape(k, (TILE,))
        # 32-bit math: a tile's stream is at most 32 * 128 * (MAX_GAP + 1) bits, its codes at most 8 bits wide.
        code = (tl.reshape(c, (TILE,)) & ((1 << tl.maximum(fk, 1)) - 1)).to(tl.uint32)
        local = tl.cumsum(fk, 0) - fk
        base = words + tl.load(woff + pid)
        word = local >> 5
        shift = local & 31
        put = (fk > 0) & (code != 0)
        lo = (code << shift.to(tl.uint32)).to(tl.int32, bitcast = True)
        hi = tl.where(shift > 0, code >> ((32 - shift) & 31).to(tl.uint32), 0).to(tl.int32, bitcast = True)
        tl.atomic_or(base + word, lo, mask = put, sem = "relaxed")
        tl.atomic_or(base + word + 1, hi, mask = put & (shift + fk > 32), sem = "relaxed")
        flat_esc = tl.reshape(esc.to(tl.int32), (TILE,))
        pos = tl.load(eoff + pid) + tl.cumsum(flat_esc, 0) - 1
        flat_idx = tl.reshape(
            (e.to(tl.int64) * N + rn[:, None].to(tl.int64)) * K + rk[None, :].to(tl.int64),
            (TILE,),
        )
        keep = flat_esc != 0
        tl.store(idx_out + pos, flat_idx, mask = keep)
        tl.store(val_out + pos, tl.reshape(w, (TILE,)), mask = keep)
        tl.store(ptrs, new, mask = mask)

    @triton.jit
    def _unfold_tile(
        w, d, ptrs, mask, pid, woff, words, checksum, n_words,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,
    ):
        # w is the folded value: back = round(w - d), plus the stored correction where the stream has one.
        TILE: tl.constexpr = BLOCK_N * BLOCK_K
        back = (w.to(tl.float32) - d).to(w.dtype)
        nb = w.to(tl.int16, bitcast = True).to(tl.int32)
        bb = back.to(tl.int16, bitcast = True).to(tl.int32)
        k = tl.reshape(tl.where(mask, _width(nb, bb, MANT, EXPM, MAX_GAP), 0), (TILE,))
        # 32-bit math (as the fold): bit offsets inside this tile's stream; reads stay inside the stream.
        start = tl.load(woff + pid)
        limit = tl.minimum(n_words - start, 1 << 30).to(tl.int32)
        base = words + start
        local = tl.cumsum(k, 0) - k
        word = local >> 5
        shift = local & 31
        has = k > 0
        lo = tl.load(base + word, mask = has & (word < limit), other = 0).to(tl.uint32, bitcast = True)
        hi = tl.load(base + word + 1, mask = has & (shift + k > 32) & (word + 1 < limit), other = 0)
        hi = hi.to(tl.uint32, bitcast = True)
        kk = tl.maximum(k, 1)
        joined = (lo >> shift.to(tl.uint32)) | tl.where(shift > 0, hi << ((32 - shift) & 31).to(tl.uint32), 0)
        code = (joined & ((1 << kk) - 1).to(tl.uint32)).to(tl.int32)
        half = 1 << (kk - 1)
        c = tl.where(code >= half, code - (1 << kk), code)
        # The escape code (-half) keeps back: the escape list overwrites it.
        ok = has & (c != -half)
        fb = tl.reshape(bb, (TILE,))
        out = tl.where(ok, _ord(_ord(fb) + c), fb)
        out = tl.reshape(out, (BLOCK_N, BLOCK_K))
        tl.store(ptrs, out.to(tl.int16).to(w.dtype, bitcast = True), mask = mask)
        tl.atomic_add(checksum, tl.sum(tl.sum(tl.where(mask, out, 0), 1), 0).to(tl.int64))
pass


_STORE = ("words", "wcnt", "idx", "val")


class ExactFoldTarget:
    """One weight to fold, W viewed as [E, N, K]: w_view is its strided base view (E=1 for dense); a [E, R, K] and b [E,
    N, R] take any strides; scale is the LoRA scaling. Between fold and unfold the store is host_words (int32 bit
    stream), host_wcnt (int16 words of each tile), host_idx / host_val (escapes: flat index, original bits)."""
    # All Unsloth Zoo code licensed under LGPLv3
    __slots__ = (
        "w", "a", "b", "scale", "counts", "checksum", "words", "wcnt", "idx", "val",
        "host_words", "host_wcnt", "host_idx", "host_val", "folded", "ready",
    )

    def __init__(self, w_view, a, b, scale):
        assert w_view.dim() == 3 and a.dim() == 3 and b.dim() == 3
        E, N, K = w_view.shape
        assert a.shape == (E, a.shape[1], K) and b.shape == (E, N, a.shape[1]), (w_view.shape, a.shape, b.shape)
        assert w_view.dtype in (torch.bfloat16, torch.float16)
        self.w, self.a, self.b, self.scale = w_view, a, b, float(scale)
        self.counts = self.checksum = None
        for name in _STORE:
            setattr(self, name, None)
            setattr(self, "host_" + name, None)
        self.folded = False
        self.ready = None
    pass

    def grid(self):
        E, N, K = self.w.shape
        return (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K), E)
    pass

    def launch(self, mode, woff = None, words = None, idx = None, val = None, eoff = None):
        E, N, K = self.w.shape
        w, a, b = self.w, self.a, self.b
        # Unused buffers get stand-ins of the same dtypes: all three modes share one compiled specialization.
        idx_type = torch.int32 if w.numel() < 2**31 else torch.int64
        fp16 = w.dtype == torch.float16
        _exact_merge_kernel[self.grid()](
            w, a, b,
            self.counts if self.counts is not None else _dummy(w.device, torch.int32),
            woff if woff is not None else _dummy(w.device, torch.int64),
            words if words is not None else _dummy(w.device, torch.int32),
            idx if idx is not None else _dummy(w.device, idx_type),
            val if val is not None else _dummy(w.device, w.dtype),
            eoff if eoff is not None else _dummy(w.device, torch.int64),
            self.checksum,
            words.numel() if words is not None else 0,
            N, K,
            w.stride(0), w.stride(1), w.stride(2),
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            self.scale, mode,
            R = a.shape[1], BLOCK_N = _BLOCK_N, BLOCK_K = _BLOCK_K,
            MANT = 10 if fp16 else 7, EXPM = 31 if fp16 else 255, MAX_GAP = _MAX_GAP,
            num_warps = 4, enable_fp_fusion = False,
        )
pass


_DUMMIES = {}


def _dummy(device, dtype):
    key = (str(device), dtype)
    if key not in _DUMMIES: _DUMMIES[key] = torch.zeros(1, dtype = dtype, device = device)
    return _DUMMIES[key]
pass


def _device_ctx(t):
    # CPU tensors only run under TRITON_INTERPRET=1 (the reconstruction tests).
    return torch.cuda.device(t.device) if t.device.type == "cuda" else contextlib.nullcontext()
pass


def exact_fold(targets, stream = None):
    """Fold targets in place and return {"bad": n, "elements": m, "store_bytes": b}; each target's store stays in
    pinned host memory, copied on stream."""
    # All Unsloth Zoo code licensed under LGPLv3
    total_bad, total_elements = 0, 0
    for t in targets:
        with _device_ctx(t.w):
            n_prog = 1
            for g in t.grid(): n_prog *= g
            t.counts = torch.empty(n_prog, 3, dtype = torch.int32, device = t.w.device)
            t.checksum = torch.zeros(1, dtype = torch.int64, device = t.w.device)
            t.launch(_COUNT)
    tile_words = [(t.counts[:, 1].to(torch.int64) + 31) // 32 for t in targets]
    sums = torch.stack([
        torch.stack([t.counts[:, 0].sum(dtype = torch.int64), tw.sum(), t.counts[:, 2].sum(dtype = torch.int64)])
        for t, tw in zip(targets, tile_words)
    ]).tolist()
    # Host stores before any weight changes: an allocation failure leaves W untouched.
    sizes = [(n_words, n_esc) for _, n_words, n_esc in sums]
    _host_buffers(targets, sizes)
    for t, tw, (n_bad, n_words, n_esc) in zip(targets, tile_words, sums):
        with _device_ctx(t.w):
            # At most 32 * 128 * 8 / 32 = 1024 words per tile: int16 counts, the offsets are rebuilt for the unfold.
            woff = torch.cumsum(tw, 0) - tw
            eoff = torch.cumsum(t.counts[:, 2], 0, dtype = torch.int64) - t.counts[:, 2]
            t.counts = None
            t.wcnt = tw.to(torch.int16)
            t.words = torch.zeros(n_words, dtype = torch.int32, device = t.w.device)
            t.idx = torch.empty(n_esc, dtype = t.host_idx.dtype, device = t.w.device)
            t.val = torch.empty(n_esc, dtype = t.w.dtype, device = t.w.device)
            t.launch(_FOLD, woff = woff, words = t.words, idx = t.idx, val = t.val, eoff = eoff)
            del woff, eoff
            t.folded = True
            total_bad += n_bad
            total_elements += t.w.numel()
    store = sum(getattr(t, "host_" + name).nbytes for t in targets for name in _STORE)
    _offload(targets, stream)
    return {"bad": total_bad, "elements": total_elements, "store_bytes": store}
pass


def _host_buffers(targets, sizes):
    # Pageable when pinning is disabled, over budget or refused (WSL2): still exact, only synchronous.
    idx_types = [torch.int32 if t.w.numel() < 2**31 else torch.int64 for t in targets]
    n_tiles = []
    for t in targets:
        n = 1
        for g in t.grid(): n *= g
        n_tiles.append(n)
    nbytes = sum(
        n_words * 4 + m * 2 + n_esc * (d.itemsize + t.w.element_size())
        for t, (n_words, n_esc), d, m in zip(targets, sizes, idx_types, n_tiles)
    )
    from .block_swap import _pin_budget
    pin = (
        targets[0].w.is_cuda and os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") != "1" and nbytes <= _pin_budget()
    )
    for t, (n_words, n_esc), d, m in zip(targets, sizes, idx_types, n_tiles):
        shapes = ((n_words, torch.int32), (m, torch.int16), (n_esc, d), (n_esc, t.w.dtype))
        if pin:
            try:
                for name, (n, dt) in zip(_STORE, shapes):
                    setattr(t, "host_" + name, torch.empty(n, dtype = dt, pin_memory = True))
                continue
            except RuntimeError:
                pin = False
        for name, (n, dt) in zip(_STORE, shapes):
            setattr(t, "host_" + name, torch.empty(n, dtype = dt))
pass


def _offload(targets, stream):
    # Stores to host on a side stream, so they hold no GPU memory during generation.
    if stream is None:
        for t in targets:
            for name in _STORE:
                getattr(t, "host_" + name).copy_(getattr(t, name))
                setattr(t, name, None)
        return
    stream.wait_stream(torch.cuda.current_stream(targets[0].w.device))
    with torch.cuda.stream(stream):
        for t in targets:
            for name in _STORE:
                buf = getattr(t, name)
                getattr(t, "host_" + name).copy_(buf, non_blocking = True)
                buf.record_stream(stream)
                setattr(t, name, None)
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
    if not targets[0].w.is_cuda:
        stream = None  # TRITON_INTERPRET=1 on CPU tensors: no streams
    current = torch.cuda.current_stream(targets[0].w.device) if targets[0].w.is_cuda else None
    for ready in {id(t.ready): t.ready for t in targets if t.ready is not None}.values():
        (stream or current).wait_event(ready)
    if stream is not None:
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for t in targets:
                for name in _STORE:
                    if getattr(t, name) is None:
                        setattr(t, name, getattr(t, "host_" + name).to(t.w.device, non_blocking = True))
    else:
        for t in targets:
            for name in _STORE:
                if getattr(t, name) is None: setattr(t, name, getattr(t, "host_" + name).to(t.w.device))
    if stream is not None:
        # The unfold pass reads the uploaded stream: it runs on the current stream after the uploads land.
        current.wait_stream(stream)
        for t in targets:
            for name in _STORE: getattr(t, name).record_stream(current)
    folded_sums = []
    for t in targets:
        with _device_ctx(t.w):
            folded_sums.append(t.checksum.clone())
            t.checksum.zero_()
            counts = t.wcnt.to(torch.int64)
            t.launch(_UNFOLD, woff = torch.cumsum(counts, 0) - counts, words = t.words)
            del counts
    checks = []
    for t, before in zip(targets, folded_sums):
        with _device_ctx(t.w):
            after = t.checksum
            if t.idx.numel():
                E, N, K = t.w.shape
                flat = t.idx.to(torch.int64)
                if t.w.is_contiguous():
                    # The stored index (e * N + n) * K + k is the flat offset: no per-axis index tensors.
                    w_flat = t.w.view(-1)
                    wrong = w_flat[flat]
                    w_flat[flat] = t.val
                else:
                    e, rem = flat // (N * K), flat % (N * K)
                    rows, cols = rem // K, rem % K
                    wrong = t.w[e, rows, cols]
                    t.w[e, rows, cols] = t.val
                after = after - wrong.view(torch.int16).to(torch.int64).sum() + t.val.view(torch.int16).to(torch.int64).sum()
            checks.append(torch.stack([before[0], after[0]]))
    pairs = torch.stack(checks).tolist()
    for t in targets:
        for name in _STORE:
            setattr(t, name, None)
            setattr(t, "host_" + name, None)
        t.counts = t.ready = None
        t.folded = False
    bad = [i for i, (x, y) in enumerate(pairs) if x != y]
    if bad:
        raise RuntimeError(
            f"Unsloth: exact LoRA unfold checksum mismatch on {len(bad)} of {len(targets)} weights; "
            "the base weights may be corrupted."
        )

