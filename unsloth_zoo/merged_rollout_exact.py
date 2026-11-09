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

"""Exact in-place LoRA fold/unfold, no pristine base copy: d = s * B @ A in fp32 must be bit-identical (one kernel
source, explicit fma in a fixed order, no fp fusion); unfold restores the original bits from a compact host store; a weight-bit
checksum raises on mismatch.

The store needs no positions. Unfold sees the folded value new = round(W + d) and recomputes d, so it knows
back = round(new - d) and the exponent gap r = exp(new) - exp(back): when r <= 0 the map W -> round(W + d) is one to
one there and back == W (no bits stored), and when r >= 1 about 2^r originals round to the same new, so the original
lies within 2^(r - 1) steps of back in the monotone order of the float bit patterns. Each element with 1 <= r <= 7
stores c = ord(W) - ord(back) in r + 1 bits, packed per tile in a bit stream whose layout both sides compute from
(new, d) alone. Anything else that does not round-trip (ties, |c| out of range, inf / NaN) is an escape: flat index and
original bits, applied after the stream decode.

The stream costs GPU time in every tile with code words, so each weight picks its form from its own count pass: few
inexact elements per coded tile (small adapters, GRPO steps) keep every inexact element as an escape instead (the
indexed form, no stream), many keep the stream. Every weight of one exact_fold call shares one store: one copy each way,
one readback per fold and per unfold, one launch for all escapes."""

__all__ = [
    "exact_merge_available",
    "ExactFoldTarget",
    "ExactFoldRefused",
    "exact_fold",
    "exact_unfold",
    "fixup_report",
    "fold_weight_",
    "unfold_weight_",
    "host_arena_stats",
]

import contextlib
import os
import threading
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

_COUNT, _FOLD, _UNFOLD = 0, 1, 2
_BLOCK_N, _BLOCK_K = 32, 128
# 8 warps: 16 elements of each tile tensor per thread; 4 warps spill registers in the fold / unfold passes on sm75.
_NUM_WARPS = 8
# Per weight, after the count pass: the bit stream costs GPU time in every tile that has code words (a scan and the
# packing, about 0.3 us per tile on a T4, both passes), the indexed form (every inexact element kept as flat index +
# original bits, the pre-stream store) costs about 1 ns per inexact element (copies both ways and the scatter). A weight
# with at most this many inexact elements per coded tile takes the indexed form: small adapters (GRPO steps) restore at
# the indexed speed, trained ones (hundreds per tile) keep the small stream. -1 forces the stream everywhere.
_INDEXED_PER_TILE = 128
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

    # mode (0 / 1 / 2) is a compile-time constant: each pass gets only its own registers (one variant with every mode
    # spilled on sm75). Every target of one exact_fold call shares one store: tile0 is this target's first tile in it,
    # slot its checksum. indexed (fold only): every inexact element is an escape and the target has no stream.
    @triton.jit(do_not_specialize = ["tile0", "slot", "indexed"])
    def _exact_merge_kernel(
        W, A, B, counts, woff, wcnt, words, idx_out, val_out, eoff, checksum, tile0, slot, indexed,
        N, K,
        sWe, sWn, sWk, sAe, sAr, sAk, sBe, sBn, sBr,
        scale, mode: tl.constexpr,
        R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  # LoRA rank / tile over out rows / tile over in cols
        MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,  # mantissa bits / exponent mask of W's dtype
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        e = tl.program_id(2)
        # This tile's index in the shared store.
        g = tile0 + (e * tl.num_programs(0) + pid_n) * tl.num_programs(1) + pid_k
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
            _unfold_tile(w, d, ptrs, mask, g, woff, wcnt, words, checksum + slot, BLOCK_N, BLOCK_K, MANT, EXPM, MAX_GAP)
        elif mode == 0:
            _count_tile(w, d, mask, g, counts, checksum + slot, MANT, EXPM, MAX_GAP)
        else:
            _fold_tile(
                w, d, ptrs, mask, g, e, rn, rk, N, K, counts, woff, wcnt, words, idx_out, val_out, eoff, indexed,
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
    def _count_tile(w, d, mask, g, counts, checksum, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr):
        new, wb, k, c, bad, esc = _classify(w, d, mask, MANT, EXPM, MAX_GAP)
        tl.store(counts + g * 3 + 0, tl.sum(tl.sum(bad.to(tl.int32), 1), 0))
        tl.store(counts + g * 3 + 1, tl.sum(tl.sum(k, 1), 0))
        tl.store(counts + g * 3 + 2, tl.sum(tl.sum(esc.to(tl.int32), 1), 0))
        # A tile's bit-pattern sum fits int32 (|sum| <= 32 * 128 * 32768 = 2^27): 64-bit math once per tile.
        tl.atomic_add(checksum, tl.sum(tl.sum(tl.where(mask, wb, 0), 1), 0).to(tl.int64))

    @triton.jit
    def _prefix(x):
        # Exclusive prefix sum of a [BLOCK_N, BLOCK_K] tile in row-major order (the stream order), kept 2D: row offsets
        # plus a scan along each row, no flattening of the tile.
        rows = tl.sum(x, 1)
        return (tl.cumsum(rows, 0) - rows)[:, None] + tl.cumsum(x, 1) - x

    @triton.jit
    def _fold_tile(
        w, d, ptrs, mask, g, e, rn, rk, N, K, counts, woff, wcnt, words, idx_out, val_out, eoff, indexed,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,
    ):
        new, wb, k, c, bad, esc = _classify(w, d, mask, MANT, EXPM, MAX_GAP)
        # An indexed target keeps every inexact element as an escape (its tiles have no stream words).
        esc = tl.where(indexed != 0, bad, esc)
        tl.store(ptrs, new, mask = mask)
        # A tile with no codes (every exponent gap is 0, common for small adapters) writes no stream: skip the scan and
        # the atomics.
        if tl.load(wcnt + g).to(tl.int32) > 0:
            # 32-bit math: a tile's stream is at most 32 * 128 * (MAX_GAP + 1) bits, its codes at most 8 bits wide.
            code = (c & ((1 << tl.maximum(k, 1)) - 1)).to(tl.uint32)
            local = _prefix(k)
            base = words + tl.load(woff + g)
            word = local >> 5
            shift = local & 31
            put = (k > 0) & (code != 0)
            lo = (code << shift.to(tl.uint32)).to(tl.int32, bitcast = True)
            hi = tl.where(shift > 0, code >> ((32 - shift) & 31).to(tl.uint32), 0).to(tl.int32, bitcast = True)
            tl.atomic_or(base + word, lo, mask = put, sem = "relaxed")
            tl.atomic_or(base + word + 1, hi, mask = put & (shift + k > 32), sem = "relaxed")
        # Escapes can sit in a tile without codes (gap 0 or > MAX_GAP): the per-tile escape count from the count pass
        # (counts is that int64 column here) says whether this tile has any, no reduction over the tile needed.
        if tl.load(counts + g) > 0:
            pos = tl.load(eoff + g) + _prefix(esc.to(tl.int32))
            flat_idx = (e.to(tl.int64) * N + rn[:, None].to(tl.int64)) * K + rk[None, :].to(tl.int64)
            tl.store(idx_out + pos, flat_idx, mask = esc)
            # Original bits as int16: one store holds fp16 and bf16 weights alike.
            tl.store(val_out + pos, w.to(tl.int16, bitcast = True), mask = esc)

    @triton.jit
    def _unfold_tile(
        w, d, ptrs, mask, g, woff, wcnt, words, checksum,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,
    ):
        # w is the folded value: back = round(w - d), plus the stored correction where the stream has one.
        back = (w.to(tl.float32) - d).to(w.dtype)
        bb = back.to(tl.int16, bitcast = True).to(tl.int32)
        n_words = tl.load(wcnt + g).to(tl.int32)
        if n_words == 0:
            # No codes in this tile: back is the original everywhere (escapes are applied after the kernel).
            tl.store(ptrs, back, mask = mask)
            tl.atomic_add(checksum, tl.sum(tl.sum(tl.where(mask, bb, 0), 1), 0).to(tl.int64))
        else:
            _unfold_stream(w, bb, ptrs, mask, g, woff, n_words, words, checksum, MANT, EXPM, MAX_GAP)

    @triton.jit
    def _unfold_stream(
        w, bb, ptrs, mask, g, woff, n_words, words, checksum,
        MANT: tl.constexpr, EXPM: tl.constexpr, MAX_GAP: tl.constexpr,
    ):
        nb = w.to(tl.int16, bitcast = True).to(tl.int32)
        k = tl.where(mask, _width(nb, bb, MANT, EXPM, MAX_GAP), 0)
        # 32-bit math (as the fold): bit offsets inside this tile's stream; reads stay inside its n_words words.
        base = words + tl.load(woff + g)
        local = _prefix(k)
        word = local >> 5
        shift = local & 31
        has = k > 0
        lo = tl.load(base + word, mask = has & (word < n_words), other = 0).to(tl.uint32, bitcast = True)
        hi = tl.load(base + word + 1, mask = has & (shift + k > 32) & (word + 1 < n_words), other = 0)
        hi = hi.to(tl.uint32, bitcast = True)
        kk = tl.maximum(k, 1)
        joined = (lo >> shift.to(tl.uint32)) | tl.where(shift > 0, hi << ((32 - shift) & 31).to(tl.uint32), 0)
        code = (joined & ((1 << kk) - 1).to(tl.uint32)).to(tl.int32)
        half = 1 << (kk - 1)
        c = tl.where(code >= half, code - (1 << kk), code)
        # The escape code (-half) keeps back: the escape list overwrites it.
        ok = has & (c != -half)
        out = tl.where(ok, _ord(_ord(bb) + c), bb)
        tl.store(ptrs, out.to(tl.int16).to(w.dtype, bitcast = True), mask = mask)
        tl.atomic_add(checksum, tl.sum(tl.sum(tl.where(mask, out, 0), 1), 0).to(tl.int64))

    @triton.jit(do_not_specialize = ["n"])
    def _escape_kernel(base, table, slots, idx, val, wrong, n, BLOCK: tl.constexpr):
        # Every escape of a store in one launch: escape i belongs to target slots[i], whose weight starts table[s, 0]
        # int16 elements from base with strides table[s, 1:4] and shape (N, K) = table[s, 4:6]. The overwritten bits go
        # to wrong (the checksum correction).
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        row = table + tl.load(slots + i, mask = m, other = 0) * 6
        off = tl.load(row + 0, mask = m, other = 0)
        sWe = tl.load(row + 1, mask = m, other = 0)
        sWn = tl.load(row + 2, mask = m, other = 0)
        sWk = tl.load(row + 3, mask = m, other = 0)
        N = tl.load(row + 4, mask = m, other = 1)
        K = tl.load(row + 5, mask = m, other = 1)
        flat = tl.load(idx + i, mask = m, other = 0).to(tl.int64)
        e = flat // (N * K)
        rem = flat - e * (N * K)
        r = rem // K
        p = base + off + e * sWe + r * sWn + (rem - r * K) * sWk
        tl.store(wrong + i, tl.load(p, mask = m, other = 0), mask = m)
        tl.store(p, tl.load(val + i, mask = m, other = 0), mask = m)
pass


class ExactFoldRefused(RuntimeError):
    """exact_fold refused before changing any weight (inexact fraction over max_fixup)."""
pass


_STORE = ("words", "wcnt", "idx", "val")


class ExactFoldTarget:
    """One weight to fold, W viewed as [E, N, K]: w_view is its strided base view (E=1 for dense); a [E, R, K] and b [E,
    N, R] take any strides; scale is the LoRA scaling. Between fold and unfold its store is a slice of the store shared
    by every target of the exact_fold call: host_words (int32 bit stream), host_wcnt (int16 words of each tile),
    host_idx / host_val (escapes: flat index, original bits)."""
    # All Unsloth Zoo code licensed under LGPLv3
    __slots__ = ("w", "a", "b", "scale", "store", "slot", "folded", "bad")

    def __init__(self, w_view, a, b, scale):
        assert w_view.dim() == 3 and a.dim() == 3 and b.dim() == 3
        E, N, K = w_view.shape
        assert a.shape == (E, a.shape[1], K) and b.shape == (E, N, a.shape[1]), (w_view.shape, a.shape, b.shape)
        assert w_view.dtype in (torch.bfloat16, torch.float16)
        self.w, self.a, self.b, self.scale = w_view, a, b, float(scale)
        self.store = None
        self.slot = 0
        self.folded = False
        self.bad = 0
    pass

    def grid(self):
        E, N, K = self.w.shape
        return (triton.cdiv(N, _BLOCK_N), triton.cdiv(K, _BLOCK_K), E)
    pass

    def n_tiles(self):
        n = 1
        for g in self.grid(): n *= g
        return n
    pass

    def _host(self, name):
        s = self.store
        return None if s is None else s.host_view(name, self.slot, self.w.dtype)

    host_words = property(lambda self: self._host("words"))
    host_wcnt = property(lambda self: self._host("wcnt"))
    host_idx = property(lambda self: self._host("idx"))
    host_val = property(lambda self: self._host("val"))

    def host_store_bytes(self):
        s = self.store
        return 0 if s is None or s.host is None else s.store_bytes(self.slot)
    pass
pass


class _Store:
    """The store of one exact_fold call on one device: every target's bit stream, tile word counts and escapes laid end
    to end in one buffer (one copy to host at the fold, one back at the unfold), one checksum slot per target and one
    host lease, given back when the last of its targets is unfolded. Each target launches its own kernels; everything
    else (offsets, sizes, copies, the checksum readback) runs once per call, not once per weight."""
    __slots__ = ("device", "targets", "tile0", "tiles", "words", "escs", "indexed", "idx_dtype", "shapes", "dev", "host",
                 "lease", "checksum", "counts", "live", "ready")

    def __init__(self, device, targets):
        self.device, self.targets = device, targets
        self.tile0, self.tiles = [], []
        at = 0
        for i, t in enumerate(targets):
            t.store, t.slot = self, i
            n = t.n_tiles()
            self.tile0.append(at)
            self.tiles.append(n)
            at += n
        self.idx_dtype = torch.int32 if all(t.w.numel() < 2**31 for t in targets) else torch.int64
        self.words = self.escs = self.shapes = self.indexed = None
        self.dev = self.host = self.lease = self.checksum = self.counts = self.ready = None
        self.live = 0
    pass

    def count(self):
        # Count pass of every target, then this call's tile offsets and per-target sizes, still on the device.
        n_tiles = self.tile0[-1] + self.tiles[-1]
        self.checksum = torch.zeros(len(self.targets), dtype = torch.int64, device = self.device)
        counts = torch.empty(n_tiles, 3, dtype = torch.int32, device = self.device)
        for t in self.targets:
            if self.tiles[t.slot]: _launch(t, _COUNT, counts = counts)
        # Per tile: inexact elements, code words (at most 32 * 128 * 8 / 32 = 1024), escapes, and whether it has words.
        x = counts.to(torch.int64)
        x[:, 1] = (x[:, 1] + 31) // 32
        x = torch.cat([x, (x[:, 1:2] > 0).to(torch.int64)], 1)
        self.counts = x
        # Inclusive sums at every target boundary (a zero row first): one row per target plus one.
        cs0 = torch.cat([torch.zeros(1, 4, dtype = torch.int64, device = self.device), torch.cumsum(x, 0)])
        return cs0[self.tile0 + [n_tiles]]
    pass

    def set_sizes(self, bounds):
        # bounds: the boundary rows read back ([inexact, words, escapes, coded tiles] summed up to each target's first
        # tile). Each target picks its format here, from its own counts.
        n = len(self.targets)
        rows = [[bounds[i + 1][j] - bounds[i][j] for j in range(4)] for i in range(n)]
        self.indexed = [_indexed(bad, coded) for bad, _, _, coded in rows]
        self.words, self.escs = [], []
        at_w = at_e = 0
        for (bad, words, escs, _), indexed in zip(rows, self.indexed):
            if indexed: words, escs = 0, bad
            self.words.append((at_w, words))
            self.escs.append((at_e, escs))
            at_w += words
            at_e += escs
        self.shapes = (
            (at_w, torch.int32), (self.tile0[-1] + self.tiles[-1], torch.int16), (at_e, self.idx_dtype), (at_e, torch.int16),
        )
        return [row[0] for row in rows]
    pass

    def fold(self):
        x = self.counts
        self.counts = None
        if any(self.indexed):
            # Indexed targets: no words in any of their tiles, every inexact element an escape.
            flag = torch.repeat_interleave(
                torch.tensor(self.indexed, device = self.device), torch.tensor(self.tiles, device = self.device),
                output_size = x.shape[0],
            )
            x[:, 1].masked_fill_(flag, 0)
            x[:, 2] = torch.where(flag, x[:, 0], x[:, 2])
        cs = torch.cumsum(x[:, 1:3], 0)
        woff = cs[:, 0] - x[:, 1]
        eoff = cs[:, 1] - x[:, 2]
        n_esc = x[:, 2].contiguous()
        self.dev = torch.empty(_carve_bytes(self.shapes), dtype = torch.uint8, device = self.device)
        words, wcnt, idx, val = _carve(self.dev, self.shapes)
        words.zero_()
        wcnt.copy_(x[:, 1])
        del x, cs
        for t in self.targets:
            # The per-tile escape counts (int64) tell each tile whether it has escapes.
            if self.tiles[t.slot]:
                _launch(t, _FOLD, counts = n_esc, woff = woff, wcnt = wcnt, words = words, idx = idx, val = val,
                        eoff = eoff, indexed = int(self.indexed[t.slot]))
            t.folded = True
            self.live += 1
    pass

    def store_bytes(self, slot):
        n_esc = self.escs[slot][1]
        return self.words[slot][1] * 4 + self.tiles[slot] * 2 + n_esc * (self.idx_dtype.itemsize + 2)
    pass

    def host_view(self, name, slot, dtype):
        if self.host is None:
            return None
        i = _STORE.index(name)
        start, n = (self.words[slot], (self.tile0[slot], self.tiles[slot]), self.escs[slot], self.escs[slot])[i]
        buf = _carve(self.host, self.shapes)[i][start:start + n]
        return buf.view(dtype) if name == "val" else buf
    pass

    def release(self):
        self.dev = self.host = self.checksum = self.counts = self.ready = None
        if self.lease is not None:
            self.lease.release()
            self.lease = None
    pass
pass


def _indexed(n_bad, coded_tiles):
    # Indexed form for this weight (see _INDEXED_PER_TILE); a weight without code words has no stream cost to save.
    return coded_tiles > 0 and n_bad <= _INDEXED_PER_TILE * coded_tiles
pass


def _launch(t, mode, counts = None, woff = None, wcnt = None, words = None, idx = None, val = None, eoff = None,
            checksum = None, indexed = 0):
    E, N, K = t.w.shape
    w, a, b, s = t.w, t.a, t.b, t.store
    dev = w.device
    fp16 = w.dtype == torch.float16
    # Unused buffers get stand-ins of the same dtypes, so each mode compiles once per weight dtype.
    _exact_merge_kernel[t.grid()](
        w, a, b,
        counts if counts is not None else _dummy(dev, torch.int32),
        woff if woff is not None else _dummy(dev, torch.int64),
        wcnt if wcnt is not None else _dummy(dev, torch.int16),
        words if words is not None else _dummy(dev, torch.int32),
        idx if idx is not None else _dummy(dev, s.idx_dtype),
        val if val is not None else _dummy(dev, torch.int16),
        eoff if eoff is not None else _dummy(dev, torch.int64),
        checksum if checksum is not None else s.checksum,
        s.tile0[t.slot], t.slot, indexed,
        N, K,
        w.stride(0), w.stride(1), w.stride(2),
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        t.scale, mode,
        R = a.shape[1], BLOCK_N = _BLOCK_N, BLOCK_K = _BLOCK_K,
        MANT = 10 if fp16 else 7, EXPM = 31 if fp16 else 255, MAX_GAP = _MAX_GAP,
        num_warps = _NUM_WARPS, enable_fp_fusion = False,
    )
pass


_DUMMIES = {}


def _dummy(device, dtype):
    key = (str(device), dtype)
    if key not in _DUMMIES: _DUMMIES[key] = torch.zeros(1, dtype = dtype, device = device)
    return _DUMMIES[key]
pass


def _device_ctx(device):
    # CPU tensors only run under TRITON_INTERPRET=1 (the reconstruction tests).
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
pass


def _stores_of(targets):
    # Each distinct store of these targets once, in order.
    out = {}
    for t in targets:
        if t.store is not None: out.setdefault(id(t.store), t.store)
    return list(out.values())
pass


def exact_fold(targets, stream = None, max_fixup = None):
    """Fold targets in place and return {"bad": n, "elements": m, "store_bytes": b}; the store stays in pinned host
    memory, copied on stream. max_fixup: raise ExactFoldRefused, with no weight changed, when more than this fraction of
    the elements does not round-trip by itself; None never refuses."""
    # All Unsloth Zoo code licensed under LGPLv3
    targets = list(targets)
    if not targets:
        return {"bad": 0, "elements": 0, "store_bytes": 0}
    by_device = {}
    for t in targets:
        by_device.setdefault(t.w.device, []).append(t)
    stores = [_Store(device, group) for device, group in by_device.items()]
    try:
        bounds = []
        for s in stores:
            with _device_ctx(s.device):
                bounds.append(s.count())
        # The one host sync of the fold: every target's sizes at once.
        bounds = [b.tolist() for b in bounds] if len(stores) > 1 else [bounds[0].tolist()]
        for s, b in zip(stores, bounds):
            for t, n_bad in zip(s.targets, s.set_sizes(b)):
                t.bad = n_bad
        elements = sum(t.w.numel() for t in targets)
        total_bad = sum(t.bad for t in targets)
        if max_fixup is not None and total_bad > max_fixup * elements:
            for s in stores: s.release()
            for t in targets: t.store = None
            raise ExactFoldRefused(
                f"Unsloth: the exact fold would leave {total_bad:,} of {elements:,} weights "
                f"({100 * total_bad / max(elements, 1):.1f}%) not round-tripping, over the {100 * max_fixup:.1f}% limit"
            )
        # Host stores before any weight changes: an allocation failure leaves W untouched.
        _host_buffers(targets, stores)
        try:
            for s in stores:
                with _device_ctx(s.device):
                    s.fold()
        except BaseException:
            # Stores with no folded target go back to the arena now; the others go back at their unfold.
            for s in stores:
                if not s.live: s.release()
            raise
        store = sum(s.store_bytes(t.slot) for s in stores for t in s.targets)
        _offload(targets, stream)
        return {"bad": total_bad, "elements": elements, "store_bytes": store}
    finally:
        # The store keeps no reference to its targets past the call (targets reference the store), so dropping
        # every folded target frees its host lease at once instead of at the next cycle collection.
        for s in stores: s.targets = None
pass


def _host_buffers(targets, stores):
    # Pageable when pinning is disabled, over budget or refused (WSL2): still exact, only synchronous. The buffers are
    # carved from a reused host arena, so a steady fold / unfold loop allocates no host memory after warm-up.
    nbytes = sum(_carve_bytes(s.shapes) for s in stores)
    from .block_swap import _pin_budget
    pin = targets[0].w.is_cuda and os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") != "1"
    if pin and nbytes > _ARENAS[True].spare() and nbytes > _pin_budget():
        pin = False
    for s in stores:
        n = _carve_bytes(s.shapes)
        lease = None
        if pin:
            try:
                lease = _ARENAS[True].take(n)
            except RuntimeError:
                pin = False
        if lease is None:
            lease = _ARENAS[False].take(n)
        s.lease = lease
        s.host = lease.buffer[:n]
pass


_ALIGN = 64


def _carve_bytes(shapes):
    return sum((n * dt.itemsize + _ALIGN - 1) // _ALIGN * _ALIGN for n, dt in shapes)
pass


def _carve(buffer, shapes):
    out, at = [], 0
    for n, dt in shapes:
        size = n * dt.itemsize
        out.append(buffer[at:at + size].view(dt))
        at += (size + _ALIGN - 1) // _ALIGN * _ALIGN
    return out
pass


class _Lease:
    """One fold's slice of a host arena, given back at unfold (or when its target is dropped)."""
    __slots__ = ("arena", "buffer", "done")

    def __init__(self, arena, buffer):
        self.arena, self.buffer, self.done = arena, buffer, False

    def release(self):
        if not self.done:
            self.done = True
            self.buffer = None
            self.arena._give()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
pass


class _HostArena:
    """Host memory for fold stores, reused across calls. Chunks are carved front to back while folds are live; when the
    last lease returns, the cursor rewinds and several chunks are merged into one of their combined size, so a workload
    whose stores stop growing allocates nothing after warm-up. A fold that does not fit adds a chunk of at least half the
    current capacity (capacity grows by 1.5x or more); live stores never move."""

    def __init__(self, pinned):
        self.pinned = pinned
        self.chunks, self.used = [], []
        self.live = 0
        self.allocations = 0
        # Re-entrant: a lease dropped by garbage collection inside take() gives back under the same lock.
        self.lock = threading.RLock()

    def _new(self, nbytes):
        self.allocations += 1
        return torch.empty(nbytes, dtype = torch.uint8, pin_memory = self.pinned)

    def capacity(self):
        return sum(c.numel() for c in self.chunks)

    def spare(self):
        return sum(c.numel() - u for c, u in zip(self.chunks, self.used))

    def take(self, nbytes):
        nbytes = max(_ALIGN, (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN)
        with self.lock:
            for i, chunk in enumerate(self.chunks):
                if self.used[i] + nbytes <= chunk.numel():
                    buffer = chunk[self.used[i]:self.used[i] + nbytes]
                    self.used[i] += nbytes
                    self.live += 1
                    return _Lease(self, buffer)
            chunk = self._new(max(nbytes, self.capacity() // 2, _ARENA_MIN_BYTES))
            self.chunks.append(chunk)
            self.used.append(nbytes)
            self.live += 1
            return _Lease(self, chunk[:nbytes])

    def _give(self):
        with self.lock:
            self.live -= 1
            if self.live:
                return
            self.used = [0] * len(self.chunks)
            if len(self.chunks) > 1:
                total = self.capacity()
                self.chunks = []
                self.chunks = [self._new(total)]
                self.used = [0]

    def clear(self):
        # Frees the memory once nothing is live (tests, or a caller done with folding).
        with self.lock:
            if not self.live:
                self.chunks, self.used = [], []
pass


_ARENA_MIN_BYTES = 1 << 20
_ARENAS = {True: _HostArena(True), False: _HostArena(False)}


def host_arena_stats():
    """Host arena state: allocations so far, capacity and live leases, for pinned and pageable memory."""
    # All Unsloth Zoo code licensed under LGPLv3
    return {
        ("pinned" if pinned else "pageable"): {
            "allocations": a.allocations, "capacity_bytes": a.capacity(), "live": a.live, "chunks": len(a.chunks),
        }
        for pinned, a in _ARENAS.items()
    }
pass


def _offload(targets, stream):
    # Stores to host (one copy per call) on a side stream, so they hold no GPU memory during generation.
    stores = [s for s in _stores_of(targets) if s.dev is not None]
    if not stores:
        return
    if stream is None:
        for s in stores:
            s.host.copy_(s.dev)
            s.dev = None
        return
    stream.wait_stream(torch.cuda.current_stream(stores[0].device))
    with torch.cuda.stream(stream):
        for s in stores:
            s.host.copy_(s.dev, non_blocking = True)
            s.dev.record_stream(stream)
        # The unfold may run on another stream: it waits on this before reading the host stores.
        ready = torch.cuda.Event()
        ready.record(stream)
    for s in stores:
        s.ready = ready
        s.dev = None
pass


def _apply_escapes(t, idx, val, after):
    # Escapes overwrite what the kernel restored there; the checksum takes the difference.
    E, N, K = t.w.shape
    flat = idx.to(torch.int64)
    val = val.view(t.w.dtype)
    if t.w.is_contiguous():
        # The stored index (e * N + n) * K + k is the flat offset: no per-axis index tensors.
        w_flat = t.w.view(-1)
        wrong = w_flat[flat]
        w_flat[flat] = val
    else:
        e, rem = flat // (N * K), flat % (N * K)
        rows, cols = rem // K, rem % K
        wrong = t.w[e, rows, cols]
        t.w[e, rows, cols] = val
    after[t.slot] += val.view(torch.int16).to(torch.int64).sum() - wrong.view(torch.int16).to(torch.int64).sum()
pass


def _restore_escapes(s, group, idx, val, after):
    # Every escape of the store at once (all of its targets are being unfolded): one launch, one checksum correction.
    n_esc = s.shapes[2][0]
    if not n_esc:
        return
    group = sorted(group, key = lambda t: t.slot)
    base = group[0].w.view(torch.int16)
    table = []
    for t in group:
        E, N, K = t.w.shape
        delta = t.w.data_ptr() - base.data_ptr()
        table.append((delta // 2, t.w.stride(0), t.w.stride(1), t.w.stride(2), N, K))
    table = torch.tensor(table, dtype = torch.int64).to(s.device, non_blocking = True)
    counts = torch.tensor([n for _, n in s.escs], dtype = torch.int64).to(s.device, non_blocking = True)
    slots = torch.repeat_interleave(torch.arange(len(group), device = s.device), counts, output_size = n_esc)
    wrong = torch.empty(n_esc, dtype = torch.int16, device = s.device)
    block = 1024
    _escape_kernel[(triton.cdiv(n_esc, block),)](base, table, slots, idx, val, wrong, n_esc, BLOCK = block, num_warps = 4)
    after.index_add_(0, slots, val.to(torch.int64) - wrong.to(torch.int64))
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
    stores = _stores_of(targets)
    for ready in {id(s.ready): s.ready for s in stores if s.ready is not None}.values():
        (stream or current).wait_event(ready)
    # One upload per store; a store whose offload failed is still on the device.
    uploads = [s for s in stores if s.dev is None]
    if stream is not None and uploads:
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for s in uploads: s.dev = s.host.to(s.device, non_blocking = True)
        # The unfold pass reads the uploaded stream: it runs on the current stream after the uploads land.
        current.wait_stream(stream)
        for s in uploads: s.dev.record_stream(current)
    else:
        for s in uploads: s.dev = s.host.to(s.device)
    members = {id(s): [] for s in stores}
    for t in targets: members[id(t.store)].append(t)
    checks = []
    for s in stores:
        with _device_ctx(s.device):
            words, wcnt, idx, val = _carve(s.dev, s.shapes)
            counts = wcnt.to(torch.int64)
            woff = torch.cumsum(counts, 0) - counts
            after = torch.zeros_like(s.checksum)
            for t in members[id(s)]:
                if s.tiles[t.slot]: _launch(t, _UNFOLD, woff = woff, wcnt = wcnt, words = words, checksum = after)
            group = members[id(s)]
            if len(group) == len(s.tiles) and len(group) > 1:
                _restore_escapes(s, group, idx, val, after)
            else:
                # One target (one op per weight) or part of a store: the escapes of these targets only.
                for t in group:
                    start, n = s.escs[t.slot]
                    if n: _apply_escapes(t, idx[start:start + n], val[start:start + n], after)
            checks.append(torch.stack([s.checksum, after], 1))
            del counts, woff
    # The one host sync of the unfold.
    pairs = [c.tolist() for c in checks]
    bad = []
    for s, p in zip(stores, pairs):
        for t in members[id(s)]:
            if p[t.slot][0] != p[t.slot][1]: bad.append(t)
            t.folded = False
            t.store = None
            s.live -= 1
        # The checksum readback above synchronised the device: every copy out of the store has landed.
        if s.live <= 0: s.release()
    if bad:
        raise RuntimeError(
            f"Unsloth: exact LoRA unfold checksum mismatch on {len(bad)} of {len(targets)} weights; "
            "the base weights may be corrupted."
        )


def fixup_report(targets, info = None):
    """Memory of a fold (call between exact_fold and exact_unfold): inexact fraction, host store bytes (pinned or not)
    and bytes per weight element, and the peak extra device bytes of the fold (bit stream, tile offsets and escapes
    before the offload) and of one exact_unfold call over these targets (every store uploaded, plus one target's int64
    escape index and displaced values)."""
    # All Unsloth Zoo code licensed under LGPLv3
    bad = elements = host = escapes = 0
    pinned = True
    device_fold = scatter = 0
    for t in targets:
        n = t.host_store_bytes()
        host += n
        for name in _STORE:
            buf = getattr(t, "host_" + name)
            if isinstance(buf, torch.Tensor) and buf.numel():
                pinned = pinned and buf.is_pinned()
        n_esc = t.host_idx.numel() if isinstance(t.host_idx, torch.Tensor) else 0
        escapes += n_esc
        bad += t.bad
        elements += t.w.numel()
        device_fold += n
        scatter = max(scatter, n_esc * (8 + t.w.element_size()))
    if info is not None:
        bad, elements = info.get("bad", bad), info.get("elements", elements)
    return {
        "bad": bad, "elements": elements, "fraction": bad / max(elements, 1), "escapes": escapes,
        "host_store_bytes": host, "host_store_pinned": pinned if host else None,
        "bytes_per_element": host / max(elements, 1),
        "device_fold_bytes": device_fold, "device_unfold_peak_bytes": device_fold + scatter,
    }
pass


# torch.compile entry points: one custom op per weight, opaque to Dynamo, so a compiled function that folds, runs and
# unfolds has no graph break. The count readback, the store and the checksum live inside the op (README 13.18); each
# op returns a small status tensor the caller can thread on (Inductor drops side-effect-only ops, README 13.14). A
# folded weight's store is kept here under the caller's key until its unfold: not under the weight's address, which
# functionalization may hand the op as a clone (aot_eager clones a mutated input, then copies back).
_OP_LIVE = {}
_OP_LOCK = threading.Lock()


def _as_3d(w, a, b):
    if w.dim() == 2:
        return w.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0)
    return w, a, b
pass


def _fold_weight_impl(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, key: int) -> torch.Tensor:
    slot = (key, tuple(w.shape), w.dtype)
    with _OP_LOCK:
        if slot in _OP_LIVE:
            raise RuntimeError(f"Unsloth: weight key {key} is already folded; unfold it first.")
        _OP_LIVE[slot] = None
    try:
        t = ExactFoldTarget(*_as_3d(w, a, b), scale)
        info = exact_fold([t])
    except BaseException:
        with _OP_LOCK:
            _OP_LIVE.pop(slot, None)
        raise
    with _OP_LOCK:
        _OP_LIVE[slot] = t
    return torch.tensor([info["bad"], info["store_bytes"]], dtype = torch.int64).to(w.device)
pass


def _unfold_weight_impl(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, key: int) -> torch.Tensor:
    with _OP_LOCK:
        t = _OP_LIVE.pop((key, tuple(w.shape), w.dtype), None)
    if t is None:
        raise RuntimeError(f"Unsloth: weight key {key} was not folded by fold_weight_.")
    # The folded bits are read from w as given now (a functionalized call may pass a copy of the folded weight), and
    # the delta is recomputed from the adapter given here, which must be the one folded.
    t.w, t.a, t.b = _as_3d(w, a, b)
    t.scale = float(scale)
    bad = t.bad
    exact_unfold([t])
    return torch.tensor([bad, 0], dtype = torch.int64).to(w.device)
pass


def _status_fake(w, a, b, scale, key):
    return w.new_empty((2,), dtype = torch.int64)
pass


def _register_weight_ops():
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None, None
    import inspect
    params = frozenset(inspect.signature(custom_op).parameters)
    # Host reads inside: keep the ops out of CUDA graphs (reduce-overhead) where the tag exists.
    tag = getattr(torch.Tag, "cudagraph_unsafe", None)
    kwargs = dict(tags = (tag,)) if tag is not None and "tags" in params else {}
    ops = []
    for name, impl in (("exact_fold_weight_v1", _fold_weight_impl), ("exact_unfold_weight_v1", _unfold_weight_impl)):
        # A reloaded module copy may own the name already; registering twice segfaults on torch 2.10.
        if hasattr(torch.ops.unsloth_zoo, name):
            ops.append(getattr(torch.ops.unsloth_zoo, name).default)
            continue
        try:
            op = custom_op(f"unsloth_zoo::{name}", mutates_args = ("w",), **kwargs)(impl)
            op.register_fake(_status_fake)
            ops.append(getattr(torch.ops.unsloth_zoo, name).default)
        except Exception:
            return None, None
    return ops[0], ops[1]
pass


_FOLD_OP, _UNFOLD_OP = _register_weight_ops()


def fold_weight_(w, a, b, scale, key):
    """Fold scale * b @ a into w in place, exactly reversible by unfold_weight_ with the same key and adapter. w is
    [N, K] (a [R, K], b [N, R]) or [E, N, K]; key is an int naming this weight (a layer index), unique among weights
    folded at the same time. torch.compile traces it as one opaque op: no graph break, one compiled variant per key
    set. Returns an int64 [2] tensor (elements that need the store, store bytes)."""
    # All Unsloth Zoo code licensed under LGPLv3
    if _FOLD_OP is None:
        return _fold_weight_impl(w, a, b, float(scale), int(key))
    return _FOLD_OP(w, a, b, float(scale), int(key))
pass


def unfold_weight_(w, a, b, scale, key):
    """Undo fold_weight_ bit for bit (raises on a checksum mismatch). Returns an int64 [2] tensor."""
    # All Unsloth Zoo code licensed under LGPLv3
    if _UNFOLD_OP is None:
        return _unfold_weight_impl(w, a, b, float(scale), int(key))
    return _UNFOLD_OP(w, a, b, float(scale), int(key))
pass
