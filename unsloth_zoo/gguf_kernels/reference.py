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

"""Pure-torch GGUF dequantization: the CPU path and the oracle for the Triton kernels.

Every format is decoded per element from its block bytes with the same float32 operation order as
ggml's reference (gguf-py), so the result is bitwise identical to it in float32."""

import torch

from .formats import GGML_BLOCK, GGMLType as T, TABLE_OFFSETS, ggml_packed_bytes, ggml_type_id, table_values

__all__ = ["dequantize_reference"]

_TABLES = {}


def _table(device):
    key = str(device)
    tab = _TABLES.get(key)
    if tab is None:
        tab = torch.tensor(table_values(), dtype = torch.int32, device = device)
        _TABLES[key] = tab
    return tab


class _Blocks:
    """Byte gathers over a (num_blocks, block_bytes) int32 view."""

    def __init__(self, blocks):
        self.b = blocks

    def u8(self, off):
        if isinstance(off, int):
            return self.b[:, off:off + 1]
        return self.b[:, off]

    def i8(self, off):
        v = self.u8(off)
        return v - ((v >> 7) << 8)

    def u16(self, off):
        return self.u8(off) | (self.u8(off + 1) << 8)

    def u32(self, off):
        return (self.u16(off).to(torch.int64) | (self.u16(off + 2).to(torch.int64) << 16))

    def f16(self, off):
        return _f16_bits(self.u16(off))


def _f16_bits(v):
    v = v - ((v >> 15) << 16)  # to signed 16-bit range
    return v.to(torch.int16).view(torch.float16).to(torch.float32)


def _f32_bits(bits):
    bits = bits.to(torch.int64)
    bits = bits - ((bits >> 31) << 32)
    return bits.to(torch.int32).view(torch.float32)


def _lut(tab, name, idx):
    return tab[TABLE_OFFSETS[name] + idx]


def _signed(value, negative):
    return torch.where(negative != 0, -value, value)


def _decode(ggml_type, B, j, tab):
    f = torch.float32
    if ggml_type == T.Q8_0:
        return B.i8(2 + j).to(f) * B.f16(0)
    if ggml_type == T.Q4_0:
        q = (B.u8(2 + j % 16) >> (4 * (j // 16))) & 0xF
        return B.f16(0) * (q - 8).to(f)
    if ggml_type == T.Q4_1:
        q = (B.u8(4 + j % 16) >> (4 * (j // 16))) & 0xF
        return B.f16(0) * q.to(f) + B.f16(2)
    if ggml_type == T.Q5_0:
        ql = (B.u8(6 + j % 16) >> (4 * (j // 16))) & 0xF
        qh = (B.u32(2) >> j) & 1
        return B.f16(0) * ((ql | (qh << 4)) - 16).to(f)
    if ggml_type == T.Q5_1:
        ql = (B.u8(8 + j % 16) >> (4 * (j // 16))) & 0xF
        qh = (B.u32(4) >> j) & 1
        return B.f16(0) * (ql | (qh << 4)).to(f) + B.f16(2)
    if ggml_type == T.Q2_K:
        sc = B.u8(j // 16)
        q = (B.u8(16 + (j // 128) * 32 + j % 32) >> (2 * ((j % 128) // 32))) & 3
        dl = B.f16(80) * (sc & 0xF).to(f)
        ml = B.f16(82) * (sc >> 4).to(f)
        return dl * q.to(f) - ml
    if ggml_type == T.Q3_K:
        s = j // 16
        lo = (B.u8(96 + s % 8) >> (4 * (s // 8))) & 0xF
        hi = (B.u8(104 + s % 4) >> (2 * (s // 4))) & 3
        dl = B.f16(108) * ((lo | (hi << 4)) - 32).to(f)
        ql = (B.u8(32 + (j // 128) * 32 + j % 32) >> (2 * ((j % 128) // 32))) & 3
        h = (B.u8(j % 32) >> (j // 32)) & 1
        return dl * (ql - ((h ^ 1) << 2)).to(f)
    if ggml_type in (T.Q4_K, T.Q5_K):
        s = j // 32
        s4 = s % 4
        low = s < 4
        a, m, md = B.u8(4 + s4), B.u8(8 + s4), B.u8(12 + s4)
        sc = torch.where(low, a & 0x3F, (md & 0xF) | ((a >> 2) & 0x30))
        mn = torch.where(low, m & 0x3F, (md >> 4) | ((m >> 2) & 0x30))
        if ggml_type == T.Q4_K:
            q = (B.u8(16 + (j // 64) * 32 + j % 32) >> (4 * ((j % 64) // 32))) & 0xF
        else:
            ql = (B.u8(48 + (j // 64) * 32 + j % 32) >> (4 * ((j % 64) // 32))) & 0xF
            q = ql | (((B.u8(16 + j % 32) >> (j // 32)) & 1) << 4)
        return (B.f16(0) * sc.to(f)) * q.to(f) - (B.f16(2) * mn.to(f))
    if ggml_type == T.Q6_K:
        lq = (B.u8((j // 128) * 64 + j % 64) >> (4 * ((j % 128) // 64))) & 0xF
        hq = (B.u8(128 + (j // 128) * 32 + j % 32) >> (2 * ((j % 128) // 32))) & 3
        q = (lq | (hq << 4)) - 32
        return (B.f16(208) * B.i8(192 + j // 16).to(f)) * q.to(f)
    if ggml_type == T.TQ1_0:
        r1, r2 = j - 160, j - 240
        byte = torch.where(j < 160, B.u8(j % 32), torch.where(j < 240, B.u8(32 + r1.clamp(min = 0) % 16), B.u8(48 + r2.clamp(min = 0) % 4)))
        power = torch.where(j < 160, j // 32, torch.where(j < 240, r1 // 16, r2 // 4))
        q = (byte * (3 ** power)) & 0xFF
        return B.f16(52) * (((q * 3) >> 8) - 1).to(f)
    if ggml_type == T.TQ2_0:
        q = (B.u8((j // 128) * 32 + j % 32) >> (2 * ((j % 128) // 32))) & 3
        return B.f16(64) * (q - 1).to(f)
    if ggml_type == T.Q1_0:
        bit = (B.u8(2 + j // 8) >> (j % 8)) & 1
        d = B.f16(0)
        return torch.where(bit != 0, d, -d)
    if ggml_type == T.Q2_0:
        q = (B.u8(2 + j // 4) >> (2 * (j % 4))) & 3
        return B.f16(0) * (q - 1).to(f)
    if ggml_type == T.MXFP4:
        e = B.u8(0)
        d = _f32_bits(torch.where(e < 2, 0x00200000 << e, (e - 1) << 23))
        q = (B.u8(1 + j % 16) >> (4 * (j // 16))) & 0xF
        return d * _lut(tab, "fp4", q).to(f)
    if ggml_type == T.NVFP4:
        s, r = j // 16, j % 16
        x = B.u8(s)
        e, man = (x >> 3) & 0xF, x & 7
        # value * 0.5: subnormal man * 2^-10, normal (1 + man / 8) * 2^(e - 8); codes 0 and 0x7F decode to 0.
        normal = _f32_bits(((e + 119) << 23) | (man << 20))
        d = torch.where(e == 0, man.to(f) * (2.0 ** -10), normal)
        d = torch.where((x == 0) | (x == 0x7F), torch.zeros_like(d), d)
        q = (B.u8(4 + s * 8 + r % 8) >> (4 * (r // 8))) & 0xF
        return d * _lut(tab, "fp4", q).to(f)
    if ggml_type == T.IQ4_NL:
        q = (B.u8(2 + j % 16) >> (4 * (j // 16))) & 0xF
        return B.f16(0) * _lut(tab, "iq4nl", q).to(f)
    if ggml_type == T.IQ4_XS:
        ib, w = j // 32, j % 32
        ls = (B.u8(4 + ib // 2) >> (4 * (ib % 2))) & 0xF
        hs = (B.u16(2) >> (2 * ib)) & 3
        dl = B.f16(0) * ((ls | (hs << 4)) - 32).to(f)
        q = (B.u8(8 + ib * 16 + w % 16) >> (4 * (w // 16))) & 0xF
        return dl * _lut(tab, "iq4nl", q).to(f)
    if ggml_type == T.IQ2_XXS:
        ib, l, k = j // 32, (j % 32) // 8, j % 8
        aux = B.u32(6 + 8 * ib)
        db = (B.f16(0) * (0.5 + (aux >> 28).to(f))) * 0.25
        grid = _lut(tab, "iq2xxs", B.u8(2 + 8 * ib + l) * 8 + k).to(f)
        sign = (_lut(tab, "ksigns", (aux >> (7 * l)) & 0x7F) >> k) & 1
        return _signed(db * grid, sign)
    if ggml_type in (T.IQ2_XS, T.IQ2_S):
        e, k, s = j // 8, j % 8, j // 16
        if ggml_type == T.IQ2_XS:
            q16 = B.u16(2 + 2 * e)
            grid = _lut(tab, "iq2xs", (q16 & 511) * 8 + k)
            sign = (_lut(tab, "ksigns", q16 >> 9) >> k) & 1
            sc_off = 66
        else:
            gidx = B.u8(2 + e) | (((B.u8(66 + e // 4) >> (2 * (e % 4))) & 3) << 8)
            grid = _lut(tab, "iq2s", gidx * 8 + k)
            sign = (B.u8(34 + e) >> k) & 1
            sc_off = 74
        ls = (B.u8(sc_off + s // 2) >> (4 * (s % 2))) & 0xF
        db = (B.f16(0) * (0.5 + ls.to(f))) * 0.25
        return _signed(db * grid.to(f), sign)
    if ggml_type == T.IQ3_XXS:
        ib, l, k = j // 32, (j % 32) // 8, j % 8
        aux = B.u32(66 + 4 * ib)
        db = (B.f16(0) * (0.5 + (aux >> 28).to(f))) * 0.5
        grid = _lut(tab, "iq3xxs", B.u8(2 + j // 4) * 4 + j % 4).to(f)
        sign = (_lut(tab, "ksigns", (aux >> (7 * l)) & 0x7F) >> k) & 1
        return _signed(db * grid, sign)
    if ggml_type == T.IQ3_S:
        s, e = j // 32, j // 4
        sc = (B.u8(106 + s // 2) >> (4 * (s % 2))) & 0xF
        db = B.f16(0) * (1 + 2 * sc).to(f)
        gidx = B.u8(2 + e) | (((B.u8(66 + e // 8) >> (e % 8)) & 1) << 8)
        grid = _lut(tab, "iq3s", gidx * 4 + j % 4).to(f)
        sign = (B.u8(74 + j // 8) >> (j % 8)) & 1
        return _signed(db * grid, sign)
    if ggml_type == T.IQ1_S:
        ib, e = j // 32, j // 8
        qh = B.u16(34 + 2 * ib)
        dl = B.f16(0) * (2 * ((qh >> 12) & 7) + 1).to(f)
        delta = torch.where((qh & 0x8000) != 0, -0.125, 0.125)
        gidx = B.u8(2 + e) | (((qh >> (3 * (e % 4))) & 7) << 8)
        return dl * (_lut(tab, "iq1s", gidx * 8 + j % 8).to(f) + delta)
    if ggml_type == T.IQ1_M:
        s0, s1, s2, s3 = B.u16(48), B.u16(50), B.u16(52), B.u16(54)
        d = _f16_bits(((s0 & 0xF000) >> 12) | ((s1 & 0xF000) >> 8) | ((s2 & 0xF000) >> 4) | (s3 & 0xF000))
        si, e = j // 16, j // 8
        sc = (B.u16(48 + 2 * (si // 4)) >> (3 * (si % 4))) & 7
        dl = d * (2 * sc + 1).to(f)
        hb = B.u8(32 + e // 2) >> (4 * (e % 2))
        gidx = B.u8(e) | ((hb & 7) << 8)
        delta = torch.where((hb & 8) != 0, -0.125, 0.125)
        return dl * (_lut(tab, "iq1s", gidx * 8 + j % 8).to(f) + delta)
    if ggml_type == T.F8_E4M3:
        x = B.u8(0)
        mag = x & 0x7F
        e, man = mag >> 3, mag & 7
        value = torch.where(e == 0, man.to(f) * (2.0 ** -9), _f32_bits(((e + 120) << 23) | (man << 20)))
        value = torch.where((x & 0x80) != 0, -value, value)
        return torch.where(mag == 0x7F, torch.full_like(value, float("nan")), value)
    raise NotImplementedError(f"Unsloth: no reference decoder for ggml type {ggml_type}")


def _check(packed, ggml_type, shape):
    if packed.dtype != torch.uint8:
        raise TypeError(f"Unsloth: packed GGUF data must be torch.uint8, got {packed.dtype}")
    need = ggml_packed_bytes(ggml_type, shape)
    if packed.numel() != need:
        raise ValueError(f"Unsloth: packed GGUF tensor has {packed.numel()} bytes, shape {tuple(shape)} needs {need}")


def dequantize_reference(packed, ggml_type, shape, dtype = torch.float32, chunk_blocks = 1 << 16):
    """Dequantize ``packed`` (uint8, any shape with ``prod(shape[:-1]) * row_bytes`` bytes) to ``shape``."""
    ggml_type = ggml_type_id(ggml_type)
    shape = tuple(int(s) for s in shape)
    _check(packed, ggml_type, shape)
    flat = packed.contiguous().reshape(-1)
    if ggml_type == T.F32:
        return flat.view(torch.float32).reshape(shape).to(dtype)
    if ggml_type == T.F16:
        return flat.view(torch.float16).reshape(shape).to(torch.float32).to(dtype)
    if ggml_type == T.BF16:
        bits = flat.view(torch.int16).to(torch.int32) << 16
        return bits.view(torch.float32).reshape(shape).to(dtype)
    qk, bs = GGML_BLOCK[ggml_type]
    blocks = flat.reshape(-1, bs)
    tab = _table(flat.device)
    j = torch.arange(qk, device = flat.device)
    out = torch.empty((blocks.shape[0], qk), dtype = dtype, device = flat.device)
    for start in range(0, blocks.shape[0], chunk_blocks):
        chunk = blocks[start:start + chunk_blocks].to(torch.int32)
        out[start:start + chunk_blocks] = _decode(ggml_type, _Blocks(chunk), j, tab).to(dtype)
    return out.reshape(shape)
