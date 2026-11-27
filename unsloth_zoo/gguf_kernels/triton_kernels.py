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

"""Triton GGUF kernels: per-element block decoders shared by a dequantize kernel, a fused GEMV (one row)
and a fused tile GEMM (tl.dot, dense or grouped by expert, forward ``x @ W^T`` or backward ``g @ W``).

The decoders mirror ``reference.py`` operation for operation in float32; launched with fp fusion off they
are bitwise equal to it."""

import triton
import triton.language as tl

from .formats import GGMLType as _T, TABLE_OFFSETS as _OFF

__all__ = ["dequantize_kernel", "gemv_kernel", "mm_kernel"]

_Q4_0 = tl.constexpr(_T.Q4_0)
_Q4_1 = tl.constexpr(_T.Q4_1)
_Q5_0 = tl.constexpr(_T.Q5_0)
_Q5_1 = tl.constexpr(_T.Q5_1)
_Q8_0 = tl.constexpr(_T.Q8_0)
_Q2_K = tl.constexpr(_T.Q2_K)
_Q3_K = tl.constexpr(_T.Q3_K)
_Q4_K = tl.constexpr(_T.Q4_K)
_Q5_K = tl.constexpr(_T.Q5_K)
_Q6_K = tl.constexpr(_T.Q6_K)
_IQ2_XXS = tl.constexpr(_T.IQ2_XXS)
_IQ2_XS = tl.constexpr(_T.IQ2_XS)
_IQ2_S = tl.constexpr(_T.IQ2_S)
_IQ3_XXS = tl.constexpr(_T.IQ3_XXS)
_IQ3_S = tl.constexpr(_T.IQ3_S)
_IQ1_S = tl.constexpr(_T.IQ1_S)
_IQ1_M = tl.constexpr(_T.IQ1_M)
_IQ4_NL = tl.constexpr(_T.IQ4_NL)
_IQ4_XS = tl.constexpr(_T.IQ4_XS)
_TQ1_0 = tl.constexpr(_T.TQ1_0)
_TQ2_0 = tl.constexpr(_T.TQ2_0)
_MXFP4 = tl.constexpr(_T.MXFP4)
_NVFP4 = tl.constexpr(_T.NVFP4)
_Q1_0 = tl.constexpr(_T.Q1_0)
_Q2_0 = tl.constexpr(_T.Q2_0)
_F8_E4M3 = tl.constexpr(_T.F8_E4M3)
_F32 = tl.constexpr(_T.F32)
_F16 = tl.constexpr(_T.F16)
_BF16 = tl.constexpr(_T.BF16)

_KSIGNS = tl.constexpr(_OFF["ksigns"])
_IQ4NL_T = tl.constexpr(_OFF["iq4nl"])
_FP4_T = tl.constexpr(_OFF["fp4"])
_IQ2XXS_T = tl.constexpr(_OFF["iq2xxs"])
_IQ2XS_T = tl.constexpr(_OFF["iq2xs"])
_IQ2S_T = tl.constexpr(_OFF["iq2s"])
_IQ3XXS_T = tl.constexpr(_OFF["iq3xxs"])
_IQ3S_T = tl.constexpr(_OFF["iq3s"])
_IQ1S_T = tl.constexpr(_OFF["iq1s"])


@triton.jit
def _u8(p, m):
    return tl.load(p, mask = m, other = 0).to(tl.int32)


@triton.jit
def _i8(p, m):
    return tl.load(p, mask = m, other = 0).to(tl.int8, bitcast = True).to(tl.int32)


@triton.jit
def _u16(p, m):
    return _u8(p, m) | (_u8(p + 1, m) << 8)


@triton.jit
def _u32(p, m):
    return (_u16(p, m) | (_u16(p + 2, m) << 16)).to(tl.uint32, bitcast = True)


@triton.jit
def _h2f(bits):
    return bits.to(tl.uint16).to(tl.float16, bitcast = True).to(tl.float32)


@triton.jit
def _f16(p, m):
    return _h2f(_u16(p, m))


@triton.jit
def _bits2f(bits):
    return bits.to(tl.int32).to(tl.float32, bitcast = True)


@triton.jit
def _lut(tab, idx, m):
    return tl.load(tab + idx, mask = m, other = 0).to(tl.int32)


@triton.jit
def _neg(v):
    # sign-bit flip: exact IEEE negation, -0.0 for +0.0 (Triton 3.2 lowers -v to 0 - v, which keeps +0.0)
    return (v.to(tl.uint32, bitcast = True) ^ 0x80000000).to(tl.float32, bitcast = True)


@triton.jit
def _signed(v, s):
    return tl.where(s != 0, _neg(v), v)


@triton.jit
def _decode(
    b, j, jh, m, tab,
    TYPE: tl.constexpr,  # ggml type id (GGMLType), selects the block decoder
):
    """Values of element ``j`` (int32, 0 <= j < block size) of the blocks starting at byte pointer ``b``.
    ``jh`` is ``j`` rounded down to a multiple of 8 (same shape or broadcastable): every field that depends on
    ``j // 8`` or coarser is indexed by it, so it loads once per 8 values."""
    if TYPE == _Q8_0:
        v = _i8(b + 2 + j, m).to(tl.float32) * _f16(b, m)
    elif TYPE == _Q4_0:
        q = (_u8(b + 2 + j % 16, m) >> (4 * (jh // 16))) & 0xF
        v = _f16(b, m) * (q - 8).to(tl.float32)
    elif TYPE == _Q4_1:
        q = (_u8(b + 4 + j % 16, m) >> (4 * (jh // 16))) & 0xF
        v = _f16(b, m) * q.to(tl.float32) + _f16(b + 2, m)
    elif TYPE == _Q5_0:
        ql = (_u8(b + 6 + j % 16, m) >> (4 * (jh // 16))) & 0xF
        qh = ((_u8(b + 2 + jh // 8, m) >> (j % 8)) & 1)
        v = _f16(b, m) * ((ql | (qh << 4)) - 16).to(tl.float32)
    elif TYPE == _Q5_1:
        ql = (_u8(b + 8 + j % 16, m) >> (4 * (jh // 16))) & 0xF
        qh = ((_u8(b + 4 + jh // 8, m) >> (j % 8)) & 1)
        v = _f16(b, m) * (ql | (qh << 4)).to(tl.float32) + _f16(b + 2, m)
    elif TYPE == _Q2_K:
        sc = _u8(b + jh // 16, m)
        q = (_u8(b + 16 + (jh // 128) * 32 + j % 32, m) >> (2 * ((jh % 128) // 32))) & 3
        dl = _f16(b + 80, m) * (sc & 0xF).to(tl.float32)
        ml = _f16(b + 82, m) * (sc >> 4).to(tl.float32)
        v = dl * q.to(tl.float32) - ml
    elif TYPE == _Q3_K:
        s = jh // 16
        lo = (_u8(b + 96 + s % 8, m) >> (4 * (s // 8))) & 0xF
        hi = (_u8(b + 104 + s % 4, m) >> (2 * (s // 4))) & 3
        dl = _f16(b + 108, m) * ((lo | (hi << 4)) - 32).to(tl.float32)
        ql = (_u8(b + 32 + (jh // 128) * 32 + j % 32, m) >> (2 * ((jh % 128) // 32))) & 3
        h = (_u8(b + j % 32, m) >> (jh // 32)) & 1
        v = dl * (ql - ((h ^ 1) << 2)).to(tl.float32)
    elif (TYPE == _Q4_K) or (TYPE == _Q5_K):
        s = jh // 32
        s4 = s % 4
        a = _u8(b + 4 + s4, m)
        mm = _u8(b + 8 + s4, m)
        md = _u8(b + 12 + s4, m)
        low = s < 4
        sc = tl.where(low, a & 0x3F, (md & 0xF) | ((a >> 2) & 0x30))
        mn = tl.where(low, mm & 0x3F, (md >> 4) | ((mm >> 2) & 0x30))
        if TYPE == _Q4_K:
            q = (_u8(b + 16 + (jh // 64) * 32 + j % 32, m) >> (4 * ((jh % 64) // 32))) & 0xF
        else:
            ql = (_u8(b + 48 + (jh // 64) * 32 + j % 32, m) >> (4 * ((jh % 64) // 32))) & 0xF
            q = ql | (((_u8(b + 16 + j % 32, m) >> (jh // 32)) & 1) << 4)
        v = (_f16(b, m) * sc.to(tl.float32)) * q.to(tl.float32) - (_f16(b + 2, m) * mn.to(tl.float32))
    elif TYPE == _Q6_K:
        lq = (_u8(b + (jh // 128) * 64 + j % 64, m) >> (4 * ((jh % 128) // 64))) & 0xF
        hq = (_u8(b + 128 + (jh // 128) * 32 + j % 32, m) >> (2 * ((jh % 128) // 32))) & 3
        q = (lq | (hq << 4)) - 32
        v = (_f16(b + 208, m) * _i8(b + 192 + jh // 16, m).to(tl.float32)) * q.to(tl.float32)
    elif TYPE == _TQ1_0:
        r1 = tl.maximum(j - 160, 0)
        r2 = tl.maximum(j - 240, 0)
        off = tl.where(j < 160, j % 32, tl.where(j < 240, 32 + r1 % 16, 48 + r2 % 4))
        pw = tl.where(j < 160, j // 32, tl.where(j < 240, r1 // 16, r2 // 4))
        mul = tl.where(pw == 0, 1, tl.where(pw == 1, 3, tl.where(pw == 2, 9, tl.where(pw == 3, 27, 81))))
        q = (_u8(b + off, m) * mul) & 0xFF
        v = _f16(b + 52, m) * (((q * 3) >> 8) - 1).to(tl.float32)
    elif TYPE == _TQ2_0:
        q = (_u8(b + (jh // 128) * 32 + j % 32, m) >> (2 * ((jh % 128) // 32))) & 3
        v = _f16(b + 64, m) * (q - 1).to(tl.float32)
    elif TYPE == _Q1_0:
        bit = (_u8(b + 2 + jh // 8, m) >> (j % 8)) & 1
        d = _f16(b, m)
        v = tl.where(bit != 0, d, _neg(d))
    elif TYPE == _Q2_0:
        q = (_u8(b + 2 + j // 4, m) >> (2 * (j % 4))) & 3
        v = _f16(b, m) * (q - 1).to(tl.float32)
    elif TYPE == _MXFP4:
        e = _u8(b, m)
        d = _bits2f(tl.where(e < 2, 0x00200000 << e, (e - 1) << 23))
        q = (_u8(b + 1 + j % 16, m) >> (4 * (jh // 16))) & 0xF
        v = d * _lut(tab + _FP4_T, q, m).to(tl.float32)
    elif TYPE == _NVFP4:
        s = jh // 16
        r = j % 16
        x = _u8(b + s, m)
        e = (x >> 3) & 0xF
        man = x & 7
        d = tl.where(e == 0, man.to(tl.float32) * 0.0009765625, _bits2f(((e + 119) << 23) | (man << 20)))
        d = tl.where((x == 0) | (x == 0x7F), 0.0, d)
        q = (_u8(b + 4 + s * 8 + r % 8, m) >> (4 * ((jh % 16) // 8))) & 0xF
        v = d * _lut(tab + _FP4_T, q, m).to(tl.float32)
    elif TYPE == _IQ4_NL:
        q = (_u8(b + 2 + j % 16, m) >> (4 * (jh // 16))) & 0xF
        v = _f16(b, m) * _lut(tab + _IQ4NL_T, q, m).to(tl.float32)
    elif TYPE == _IQ4_XS:
        ib = jh // 32
        w = j % 32
        ls = (_u8(b + 4 + ib // 2, m) >> (4 * (ib % 2))) & 0xF
        hs = (_u16(b + 2, m) >> (2 * ib)) & 3
        dl = _f16(b, m) * ((ls | (hs << 4)) - 32).to(tl.float32)
        q = (_u8(b + 8 + ib * 16 + w % 16, m) >> (4 * ((jh % 32) // 16))) & 0xF
        v = dl * _lut(tab + _IQ4NL_T, q, m).to(tl.float32)
    elif TYPE == _IQ2_XXS:
        ib = jh // 32
        l = (jh % 32) // 8
        k = j % 8
        aux = _u32(b + 6 + 8 * ib, m)
        db = (_f16(b, m) * (0.5 + (aux >> 28).to(tl.float32))) * 0.25
        grid = _lut(tab + _IQ2XXS_T, _u8(b + 2 + 8 * ib + l, m) * 8 + k, m).to(tl.float32)
        sign = (_lut(tab + _KSIGNS, ((aux >> (7 * l).to(tl.uint32)) & 0x7F).to(tl.int32), m) >> k) & 1
        v = _signed(db * grid, sign)
    elif (TYPE == _IQ2_XS) or (TYPE == _IQ2_S):
        e = jh // 8
        k = j % 8
        s = jh // 16
        if TYPE == _IQ2_XS:
            q16 = _u16(b + 2 + 2 * e, m)
            grid = _lut(tab + _IQ2XS_T, (q16 & 511) * 8 + k, m)
            sign = (_lut(tab + _KSIGNS, q16 >> 9, m) >> k) & 1
            ls = (_u8(b + 66 + s // 2, m) >> (4 * (s % 2))) & 0xF
        else:
            gidx = _u8(b + 2 + e, m) | (((_u8(b + 66 + e // 4, m) >> (2 * (e % 4))) & 3) << 8)
            grid = _lut(tab + _IQ2S_T, gidx * 8 + k, m)
            sign = (_u8(b + 34 + e, m) >> k) & 1
            ls = (_u8(b + 74 + s // 2, m) >> (4 * (s % 2))) & 0xF
        db = (_f16(b, m) * (0.5 + ls.to(tl.float32))) * 0.25
        v = _signed(db * grid.to(tl.float32), sign)
    elif TYPE == _IQ3_XXS:
        ib = jh // 32
        l = (jh % 32) // 8
        k = j % 8
        aux = _u32(b + 66 + 4 * ib, m)
        db = (_f16(b, m) * (0.5 + (aux >> 28).to(tl.float32))) * 0.5
        grid = _lut(tab + _IQ3XXS_T, _u8(b + 2 + j // 4, m) * 4 + j % 4, m).to(tl.float32)
        sign = (_lut(tab + _KSIGNS, ((aux >> (7 * l).to(tl.uint32)) & 0x7F).to(tl.int32), m) >> k) & 1
        v = _signed(db * grid, sign)
    elif TYPE == _IQ3_S:
        s = jh // 32
        e = j // 4
        sc = (_u8(b + 106 + s // 2, m) >> (4 * (s % 2))) & 0xF
        db = _f16(b, m) * (1 + 2 * sc).to(tl.float32)
        gidx = _u8(b + 2 + e, m) | (((_u8(b + 66 + e // 8, m) >> (e % 8)) & 1) << 8)
        grid = _lut(tab + _IQ3S_T, gidx * 4 + j % 4, m).to(tl.float32)
        sign = (_u8(b + 74 + jh // 8, m) >> (j % 8)) & 1
        v = _signed(db * grid, sign)
    elif TYPE == _IQ1_S:
        e = jh // 8
        qh = _u16(b + 34 + 2 * (jh // 32), m)
        dl = _f16(b, m) * (2 * ((qh >> 12) & 7) + 1).to(tl.float32)
        delta = tl.where((qh & 0x8000) != 0, -0.125, 0.125)
        gidx = _u8(b + 2 + e, m) | (((qh >> (3 * (e % 4))) & 7) << 8)
        v = dl * (_lut(tab + _IQ1S_T, gidx * 8 + j % 8, m).to(tl.float32) + delta)
    elif TYPE == _IQ1_M:
        s0 = _u16(b + 48, m)
        s1 = _u16(b + 50, m)
        s2 = _u16(b + 52, m)
        s3 = _u16(b + 54, m)
        d = _h2f(((s0 & 0xF000) >> 12) | ((s1 & 0xF000) >> 8) | ((s2 & 0xF000) >> 4) | (s3 & 0xF000))
        si = jh // 16
        e = jh // 8
        sc = (_u16(b + 48 + 2 * (si // 4), m) >> (3 * (si % 4))) & 7
        dl = d * (2 * sc + 1).to(tl.float32)
        hb = _u8(b + 32 + e // 2, m) >> (4 * (e % 2))
        gidx = _u8(b + e, m) | ((hb & 7) << 8)
        delta = tl.where((hb & 8) != 0, -0.125, 0.125)
        v = dl * (_lut(tab + _IQ1S_T, gidx * 8 + j % 8, m).to(tl.float32) + delta)
    elif TYPE == _F8_E4M3:
        x = _u8(b, m)
        mag = x & 0x7F
        e = mag >> 3
        man = mag & 7
        v = tl.where(e == 0, man.to(tl.float32) * 0.001953125, _bits2f(((e + 120) << 23) | (man << 20)))
        v = tl.where((x & 0x80) != 0, _neg(v), v)
        v = tl.where(mag == 0x7F, float("nan"), v)
    elif TYPE == _F16:
        v = _f16(b, m)
    elif TYPE == _BF16:
        v = _bits2f(_u16(b, m) << 16)
    else:  # F32
        v = _bits2f((_u16(b, m) | (_u16(b + 2, m) << 16)))
    return v


@triton.jit
def _tile3(
    P, rows, blk0, rmask, row_bytes, nblk, tab,
    TYPE: tl.constexpr,  # ggml type id (GGMLType), selects the block decoder
    QK: tl.constexpr,  # values per ggml block
    BS: tl.constexpr,  # bytes per ggml block
    NB: tl.constexpr,  # ggml blocks per tile along the value axis
    SPLIT: tl.constexpr,  # decode in 8-value groups (group fields load once per 8 values), else one group per block
):
    """Decoded float32 tile (rows, NB blocks, G groups, QK // G values) of a packed matrix starting at block
    ``blk0``. SPLIT: 8-value groups, so group fields load once per 8 values; else one group of the whole block on
    the last axis (cheaper to reshape into a dot operand; group fields then load per value). Block fields load once
    per (row, block); index math runs on the in-block index only."""
    G: tl.constexpr = QK // 8 if (QK >= 8 and SPLIT) else 1  # groups per block
    W: tl.constexpr = QK // G  # values per group, on the last (contiguous) axis
    blk = blk0 + tl.arange(0, NB)
    b = P + rows.to(tl.int64)[:, None, None, None] * row_bytes + (blk * BS)[None, :, None, None]
    m = rmask[:, None, None, None] & (blk < nblk)[None, :, None, None]
    if G > 1:
        jh = (tl.arange(0, G) * W)[None, None, :, None]
        j = jh + tl.arange(0, W)[None, None, None, :]
    else:
        j = tl.arange(0, W)[None, None, None, :]
        jh = j
    return _decode(b, j, jh, m, tab, TYPE)


@triton.jit
def _cols(
    blk0,
    QK: tl.constexpr,  # values per ggml block
    NB: tl.constexpr,  # ggml blocks per tile along the value axis
    SPLIT: tl.constexpr,  # decode in 8-value groups (group fields load once per 8 values), else one group per block
):
    """Column index of every value of a ``_tile3`` tile, shape (1, NB, G, QK // G)."""
    G: tl.constexpr = QK // 8 if (QK >= 8 and SPLIT) else 1
    W: tl.constexpr = QK // G
    jh = (tl.arange(0, G) * W)[None, None, :, None]
    return (blk0 + tl.arange(0, NB))[None, :, None, None] * QK + jh + tl.arange(0, W)[None, None, None, :]


@triton.jit
def dequantize_kernel(
    P, OUT, tab, ROWS, K, row_bytes,
    TYPE: tl.constexpr,  # ggml type id (GGMLType), selects the block decoder
    QK: tl.constexpr,  # values per ggml block
    BS: tl.constexpr,  # bytes per ggml block
    BLOCK_R: tl.constexpr,  # packed rows per program
    NB: tl.constexpr,  # ggml blocks per tile along the value axis
    SPLIT: tl.constexpr,  # decode in 8-value groups (group fields load once per 8 values), else one group per block
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    blk0 = tl.program_id(1) * NB
    nblk = K // QK
    rmask = rows < ROWS
    w = _tile3(P, rows, blk0, rmask, row_bytes, nblk, tab, TYPE, QK, BS, NB, SPLIT)
    cols = _cols(blk0, QK, NB, SPLIT)
    out = OUT + rows[:, None, None, None] * K + cols
    tl.store(out, w.to(OUT.dtype.element_ty), mask = rmask[:, None, None, None] & (cols < K))


@triton.jit
def gemv_kernel(
    X, P, OUT, BIAS, tab, N, K, row_bytes,
    TYPE: tl.constexpr,  # ggml type id (GGMLType), selects the block decoder
    QK: tl.constexpr,  # values per ggml block
    BS: tl.constexpr,  # bytes per ggml block
    HAS_BIAS: tl.constexpr,  # add BIAS to the output
    BLOCK_N: tl.constexpr,  # output columns per program
    NB: tl.constexpr,  # ggml blocks per tile along the value axis
    SPLIT: tl.constexpr,  # decode in 8-value groups (group fields load once per 8 values), else one group per block
):
    """out[n] = sum_k x[k] * W[n, k] (+ bias[n]) for one row of x."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    nblk = K // QK
    G: tl.constexpr = QK // 8 if (QK >= 8 and SPLIT) else 1
    acc = tl.zeros((BLOCK_N, NB, G, QK // G), dtype = tl.float32)
    for blk0 in range(0, nblk, NB):
        cols = _cols(blk0, QK, NB, SPLIT)
        x = tl.load(X + cols, mask = cols < K, other = 0.0).to(tl.float32)
        w = _tile3(P, rows, blk0, rmask, row_bytes, nblk, tab, TYPE, QK, BS, NB, SPLIT)
        acc += w * x
    # one reduction over a 2-D view: Triton 3.2 maps nested 4-D reductions back to the wrong rows (duplicated pairs)
    out = tl.sum(tl.reshape(acc, (BLOCK_N, NB * QK)), axis = 1)
    if HAS_BIAS:
        out += tl.load(BIAS + rows, mask = rmask, other = 0.0).to(tl.float32)
    tl.store(OUT + rows, out.to(OUT.dtype.element_ty), mask = rmask)


@triton.jit
def mm_kernel(
    X, P, OUT, BIAS, COUNTS, SCHED, tab,
    M, N, K, E, R, row_bytes, stride_xm, stride_om,
    TYPE: tl.constexpr,  # ggml type id (GGMLType), selects the block decoder
    QK: tl.constexpr,  # values per ggml block
    BS: tl.constexpr,  # bytes per ggml block
    TRANS_B: tl.constexpr,  # out = x @ W[e]^T (forward) instead of x @ W[e] (backward)
    GROUPED: tl.constexpr,  # rows of x sorted by expert, COUNTS[e] rows each
    E_POW2: tl.constexpr,  # E rounded up to a power of two, for the per-expert aranges
    HAS_BIAS: tl.constexpr,  # add BIAS to the output
    HOST_SCHED: tl.constexpr,  # tile / row prefix sums come precomputed in SCHED (Triton < 3.3)
    BLOCK_M: tl.constexpr,  # rows of x per program
    BLOCK_N: tl.constexpr,  # output columns per program
    BLOCK_K: tl.constexpr,  # reduction elements per loop step
    SPLIT: tl.constexpr,  # decode in 8-value groups (group fields load once per 8 values), else one group per block
):
    """Packed weight ``W[e]`` is (R, C) with C values per packed row (R rows per expert).
    TRANS_B: out (M, N = R) = x (M, K = C) @ W[e]^T.  Else: out (M, N = C) = x (M, K = R) @ W[e].
    GROUPED: rows of x sorted by expert, ``COUNTS[e]`` rows each. HOST_SCHED: the tile / row prefix sums come
    precomputed in ``SCHED`` (2, E) instead of a tl.cumsum (Triton < 3.3 scans a replicated layout wrongly). The packed-row axis of a tile is
    BLOCK_N (TRANS_B) or BLOCK_K wide; the value axis is the other one, a whole number of blocks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if GROUPED:
        e_offs = tl.arange(0, E_POW2)
        counts = tl.load(COUNTS + e_offs, mask = e_offs < E, other = 0)
        tiles = (counts + BLOCK_M - 1) // BLOCK_M
        if HOST_SCHED:
            tile_end = tl.load(SCHED + e_offs, mask = e_offs < E, other = 1 << 30)
            row_end_all = tl.load(SCHED + E + e_offs, mask = e_offs < E, other = 0)
        else:
            tile_end = tl.cumsum(tiles, 0)
            row_end_all = tl.cumsum(counts, 0)
        e = tl.sum((tile_end <= pid_m).to(tl.int32), 0)
        if e >= E:
            return
        row_end = tl.sum(tl.where(e_offs == e, row_end_all, 0), 0)
        row_start = row_end - tl.sum(tl.where(e_offs == e, counts, 0), 0)
        tile_start = tl.sum(tl.where(e_offs == e, tile_end - tiles, 0), 0)
        m0 = row_start + (pid_m - tile_start) * BLOCK_M
        row0 = e.to(tl.int64) * R
    else:
        m0 = pid_m * BLOCK_M
        row_end = M
        row0 = tl.full((), 0, tl.int64)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < row_end
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    x_rows = X + offs_m.to(tl.int64)[:, None] * stride_xm
    Pe = P + row0 * row_bytes
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    if TRANS_B:
        NBK: tl.constexpr = BLOCK_K // QK
        nblk = K // QK
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(x_rows + offs_k[None, :], mask = mask_m[:, None] & (offs_k < K)[None, :], other = 0.0)
            w = _tile3(Pe, offs_n, k0 // QK, mask_n, row_bytes, nblk, tab, TYPE, QK, BS, NBK, SPLIT)
            w = tl.reshape(w, (BLOCK_N, BLOCK_K))
            acc = tl.dot(a, tl.trans(w.to(a.dtype)), acc)
    else:
        NBN: tl.constexpr = BLOCK_N // QK
        nblk = N // QK
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(x_rows + offs_k[None, :], mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
            w = _tile3(Pe, offs_k, (pid_n * BLOCK_N) // QK, mask_k, row_bytes, nblk, tab, TYPE, QK, BS, NBN, SPLIT)
            w = tl.reshape(w, (BLOCK_K, BLOCK_N))
            acc = tl.dot(a, w.to(a.dtype), acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask = mask_n, other = 0.0).to(tl.float32)[None, :]
    out = OUT + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
    tl.store(out, acc.to(OUT.dtype.element_ty), mask = mask_m[:, None] & mask_n[None, :])
