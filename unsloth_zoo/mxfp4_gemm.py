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

"""Grouped GEMM that reads MXFP4 expert stacks directly: the e2m1 nibbles are decoded to bf16 in the
GEMM main loop, so no dense bf16 expert stack is ever materialised.

The packed stack ``P`` is ``blocks (E, R, G, 16)`` + ``scales (E, R, G)``, one row of ``C = G * 32``
values per ``r``. Rows of ``x`` are sorted by expert (``counts[e]`` rows each, like torch._grouped_mm's
``offs``). ``transpose_b=True`` computes ``x @ P[e]^T`` (GPT-OSS forward: logical weight ``(E, C, R)``),
``transpose_b=False`` computes ``x @ P[e]`` (its dX backward). ``mxfp4_matmul`` runs the same kernel on one weight
(no expert schedule), e.g. a packed MXFP4 Linear."""

import torch

__all__ = [
    "mxfp4_grouped_mm",
    "mxfp4_grouped_mm_available",
    "Mxfp4GroupedMM",
    "mxfp4_expert_grouped_mm",
    "mxfp4_grouped_matmul",
    "mxfp4_gemm_available",
    "mxfp4_matmul",
]

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _decode_mxfp4_tile(packed, scale, ROWS: tl.constexpr, COLS: tl.constexpr):
        """(ROWS, COLS // 2) bytes + (ROWS, COLS // 32) e8m0 scales -> (ROWS, COLS) bf16, low nibble first.
        The e2m1 bits go straight into the bf16 exponent / mantissa, giving value * 2^-126 for every
        code (0.5 lands on the bf16 subnormal 2^-127); the per-group factor 2^(scale - 1) restores it.
        Powers of two only, so each value is exact (checked bit for bit against mxfp4_dequantize)."""
        x0 = ((packed & 0x07) << 6) | ((packed & 0x08) << 12)
        x1 = ((packed & 0x70) << 2) | ((packed & 0x80) << 8)
        lo = x0.to(tl.uint16).to(tl.bfloat16, bitcast = True)
        hi = x1.to(tl.uint16).to(tl.bfloat16, bitcast = True)
        vals = tl.reshape(tl.join(lo, hi), (ROWS, COLS // 32, 32))
        # 2^(scale - 1) as bf16 bits (always normal); above scale 128 it overflows, so split off a 2^127.
        big = scale > 128
        f1 = tl.where(big, scale - 1, scale + 126) << 7
        f1 = f1.to(tl.uint16).to(tl.bfloat16, bitcast = True)
        vals = vals * f1[:, :, None]
        vals = vals * tl.where(big, 1.7014118346046923e38, 1.0).to(tl.bfloat16)[:, :, None]
        return tl.reshape(vals, (ROWS, COLS))

    @triton.jit
    def _decode_mxfp4_tile_asm(packed, scale):
        # packed (R, C // 2) uint8, scale (R, C // 32) uint8 -> (R, C) bf16, bit-identical to mxfp4_dequant.
        # Each nibble becomes bf16 bits sign << 15 | e2m1 << 6 (= value * 2^-126, subnormal for e = 0), then two
        # exact bf16x2 fmas by 2^(min(s, 128) - 1) and 2^(max(s, 128) - 128): every e8m0 incl. 255 = 2^128.
        R: tl.constexpr = packed.shape[0]
        CB: tl.constexpr = packed.shape[1]
        CG: tl.constexpr = scale.shape[1]
        sc = tl.reshape(tl.broadcast_to(scale[:, :, None], (R, CG, 16)), (R, CB))
        lo, hi = tl.inline_asm_elementwise(
            asm = """
            {
            .reg .b32 t, u, s, f, g, z;
            and.b32 s, $5, 0xFF;
            min.u32 f, s, 128;
            add.u32 f, f, 126;
            shl.b32 f, f, 7;
            prmt.b32 f, f, 0, 0x1010;
            max.u32 g, s, 128;
            sub.u32 g, g, 1;
            shl.b32 g, g, 7;
            prmt.b32 g, g, 0, 0x1010;
            mov.b32 z, 0x80008000;
            prmt.b32 t, $4, 0, 0x4140;
            and.b32 u, t, 0x00070007;
            shl.b32 u, u, 6;
            and.b32 $0, t, 0x00080008;
            shl.b32 $0, $0, 12;
            or.b32 $0, $0, u;
            fma.rn.bf16x2 $0, $0, f, z;
            fma.rn.bf16x2 $0, $0, g, z;
            and.b32 u, t, 0x00700070;
            shl.b32 u, u, 2;
            and.b32 $2, t, 0x00800080;
            shl.b32 $2, $2, 8;
            or.b32 $2, $2, u;
            fma.rn.bf16x2 $2, $2, f, z;
            fma.rn.bf16x2 $2, $2, g, z;
            prmt.b32 t, $4, 0, 0x4342;
            and.b32 u, t, 0x00070007;
            shl.b32 u, u, 6;
            and.b32 $1, t, 0x00080008;
            shl.b32 $1, $1, 12;
            or.b32 $1, $1, u;
            fma.rn.bf16x2 $1, $1, f, z;
            fma.rn.bf16x2 $1, $1, g, z;
            and.b32 u, t, 0x00700070;
            shl.b32 u, u, 2;
            and.b32 $3, t, 0x00800080;
            shl.b32 $3, $3, 8;
            or.b32 $3, $3, u;
            fma.rn.bf16x2 $3, $3, f, z;
            fma.rn.bf16x2 $3, $3, g, z;
            }
            """,
            constraints = "=r,=r,=r,=r,r,r",
            args = [packed, sc], dtype = (tl.bfloat16, tl.bfloat16), is_pure = True, pack = 4,
        )
        return tl.reshape(tl.join(lo, hi), (R, CB * 2))

    @triton.jit
    def _mxfp4_scale_factors(scale):
        """e8m0 -> (2^(scale - 1) or 2^(scale - 128), 1 or 2^127) as bf16; their product restores value * 2^-126."""
        big = scale > 128
        f1 = (tl.where(big, scale - 1, scale + 126) << 7).to(tl.uint16).to(tl.bfloat16, bitcast = True)
        f2 = tl.where(big, 1.7014118346046923e38, 1.0).to(tl.bfloat16)
        return f1, f2

    @triton.jit
    def _decode_mxfp4_halves(packed, scale, ROWS: tl.constexpr, BYTES: tl.constexpr):
        """(ROWS, BYTES) bytes + (ROWS, BYTES // 16) scales -> low-nibble and high-nibble (ROWS, BYTES) bf16
        tiles, no interleave (the caller pairs them with even / odd columns)."""
        lo = (((packed & 0x07) << 6) | ((packed & 0x08) << 12)).to(tl.uint16).to(tl.bfloat16, bitcast = True)
        hi = (((packed & 0x70) << 2) | ((packed & 0x80) << 8)).to(tl.uint16).to(tl.bfloat16, bitcast = True)
        f1, f2 = _mxfp4_scale_factors(scale)
        f1 = f1[:, :, None]
        f2 = f2[:, :, None]
        lo = tl.reshape(tl.reshape(lo, (ROWS, BYTES // 16, 16)) * f1 * f2, (ROWS, BYTES))
        hi = tl.reshape(tl.reshape(hi, (ROWS, BYTES // 16, 16)) * f1 * f2, (ROWS, BYTES))
        return lo, hi

    @triton.jit
    def _mxfp4_grouped_mm_kernel(
        x_ptr, blocks_ptr, scales_ptr, out_ptr, counts_ptr, bias_ptr,
        E, R, G, K, N, M,
        stride_xm, stride_om,
        TRANS_B: tl.constexpr, E_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT: tl.constexpr, GROUPED: tl.constexpr, HAS_BIAS: tl.constexpr, ASM: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if GROUPED:
            # Tile pid_m -> (expert, tile within expert); tiles are laid out expert after expert.
            e_offs = tl.arange(0, E_POW2)
            counts = tl.load(counts_ptr + e_offs, mask = e_offs < E, other = 0)
            tiles = (counts + BLOCK_M - 1) // BLOCK_M
            tile_end = tl.cumsum(tiles, 0)
            e = tl.sum((tile_end <= pid_m).to(tl.int32), 0)
            if e >= E:
                return
            row_end_all = tl.cumsum(counts, 0)
            row_end = tl.sum(tl.where(e_offs == e, row_end_all, 0), 0)
            row_start = row_end - tl.sum(tl.where(e_offs == e, counts, 0), 0)
            tile_start = tl.sum(tl.where(e_offs == e, tile_end - tiles, 0), 0)
            m0 = row_start + (pid_m - tile_start) * BLOCK_M
            e64 = e.to(tl.int64)
        else:
            m0 = pid_m * BLOCK_M
            row_end = M
            e64 = tl.full((), 0, tl.int64)

        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < row_end
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
        C = G * 32
        if TRANS_B:
            # out[m, n] = sum_k x[m, k] * P[e, n, k]; K = C, N = R. Packed rows are n.
            prow = e64 * R + offs_n
            mask_n = offs_n < R
            if SPLIT:
                # x columns arrive permuted per BLOCK_K block as [even | odd], matching lo / hi nibbles.
                for k0 in range(0, K, BLOCK_K):
                    offs_h = k0 + tl.arange(0, BLOCK_K // 2)
                    a_lo = tl.load(x_rows + offs_h[None, :], mask = mask_m[:, None], other = 0.0)
                    a_hi = tl.load(x_rows + (offs_h + BLOCK_K // 2)[None, :], mask = mask_m[:, None], other = 0.0)
                    offs_kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                    packed = tl.load(blocks_ptr + prow[:, None] * (G * 16) + offs_kb[None, :], mask = mask_n[:, None], other = 0).to(tl.int32)
                    offs_kg = k0 // 32 + tl.arange(0, BLOCK_K // 32)
                    scale = tl.load(scales_ptr + prow[:, None] * G + offs_kg[None, :], mask = mask_n[:, None], other = 127).to(tl.int32)
                    lo, hi = _decode_mxfp4_halves(packed, scale, BLOCK_N, BLOCK_K // 2)
                    acc = tl.dot(a_lo, tl.trans(lo.to(a_lo.dtype)), acc)
                    acc = tl.dot(a_hi, tl.trans(hi.to(a_hi.dtype)), acc)
            else:
                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    a = tl.load(x_rows + offs_k[None, :], mask = mask_m[:, None] & (offs_k[None, :] < K), other = 0.0)
                    offs_kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                    packed = tl.load(
                        blocks_ptr + prow[:, None] * (G * 16) + offs_kb[None, :],
                        mask = mask_n[:, None] & (offs_kb[None, :] < G * 16), other = 0,
                    )
                    offs_kg = k0 // 32 + tl.arange(0, BLOCK_K // 32)
                    scale = tl.load(
                        scales_ptr + prow[:, None] * G + offs_kg[None, :],
                        mask = mask_n[:, None] & (offs_kg[None, :] < G), other = 127,
                    )
                    if ASM:
                        w = _decode_mxfp4_tile_asm(packed, scale)
                    else:
                        w = _decode_mxfp4_tile(packed.to(tl.int32), scale.to(tl.int32), BLOCK_N, BLOCK_K)
                    acc = tl.dot(a, tl.trans(w.to(a.dtype)), acc)
        else:
            # out[m, n] = sum_k x[m, k] * P[e, k, n]; K = R, N = C. Packed rows are k.
            mask_n = offs_n < C
            if SPLIT:
                # Low / high nibbles are even / odd output columns: two half-width dots, interleaved at the store.
                offs_nb = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
                offs_ng = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
                acc_lo = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype = tl.float32)
                acc_hi = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype = tl.float32)
                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < K
                    a = tl.load(x_rows + offs_k[None, :], mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
                    prow = e64 * R + offs_k
                    packed = tl.load(
                        blocks_ptr + prow[:, None] * (G * 16) + offs_nb[None, :],
                        mask = mask_k[:, None] & (offs_nb[None, :] < G * 16), other = 0,
                    ).to(tl.int32)
                    scale = tl.load(
                        scales_ptr + prow[:, None] * G + offs_ng[None, :],
                        mask = mask_k[:, None] & (offs_ng[None, :] < G), other = 127,
                    ).to(tl.int32)
                    lo, hi = _decode_mxfp4_halves(packed, scale, BLOCK_K, BLOCK_N // 2)
                    acc_lo = tl.dot(a, lo.to(a.dtype), acc_lo)
                    acc_hi = tl.dot(a, hi.to(a.dtype), acc_hi)
                acc = tl.reshape(tl.join(acc_lo, acc_hi), (BLOCK_M, BLOCK_N))
            else:
                offs_nb = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
                offs_ng = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < K
                    a = tl.load(x_rows + offs_k[None, :], mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
                    prow = e64 * R + offs_k
                    packed = tl.load(
                        blocks_ptr + prow[:, None] * (G * 16) + offs_nb[None, :],
                        mask = mask_k[:, None] & (offs_nb[None, :] < G * 16), other = 0,
                    )
                    scale = tl.load(
                        scales_ptr + prow[:, None] * G + offs_ng[None, :],
                        mask = mask_k[:, None] & (offs_ng[None, :] < G), other = 127,
                    )
                    if ASM:
                        w = _decode_mxfp4_tile_asm(packed, scale)
                    else:
                        w = _decode_mxfp4_tile(packed.to(tl.int32), scale.to(tl.int32), BLOCK_K, BLOCK_N)
                    acc = tl.dot(a, w.to(a.dtype), acc)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask = mask_n, other = 0.0).to(tl.float32)[None, :]
        out = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :]
        tl.store(out, acc.to(out_ptr.dtype.element_ty), mask = mask_m[:, None] & mask_n[None, :])


if _HAS_TRITON:

    @triton.jit
    def _decode_only_kernel(blocks_ptr, scales_ptr, out_ptr, RG, ROWS: tl.constexpr, COLS: tl.constexpr):
        # Test hook: decode (rows, G) groups with the GEMM's decode, row-major (E * R, C) output.
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols_b = tl.program_id(1) * (COLS // 2) + tl.arange(0, COLS // 2)
        cols_g = tl.program_id(1) * (COLS // 32) + tl.arange(0, COLS // 32)
        G16 = tl.num_programs(1) * (COLS // 2)
        G = tl.num_programs(1) * (COLS // 32)
        m = rows[:, None] < RG
        packed = tl.load(blocks_ptr + rows[:, None].to(tl.int64) * G16 + cols_b[None, :], mask = m, other = 0).to(tl.int32)
        scale = tl.load(scales_ptr + rows[:, None].to(tl.int64) * G + cols_g[None, :], mask = m, other = 127).to(tl.int32)
        vals = _decode_mxfp4_tile(packed, scale, ROWS, COLS)
        cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
        tl.store(out_ptr + rows[:, None].to(tl.int64) * (G * 32) + cols[None, :], vals, mask = m)


def _decode_for_test(blocks, scales):
    """(E, R, G, 16) -> (E, R, G * 32) bf16 through the GEMM's in-register decode."""
    *prefix, G, _ = blocks.shape
    rows = blocks.numel() // (G * 16)
    out = torch.empty((*prefix, G * 32), dtype = torch.bfloat16, device = blocks.device)
    _decode_only_kernel[(triton.cdiv(rows, 32), G)](blocks.contiguous(), scales.contiguous(), out, rows, ROWS = 32, COLS = 32)
    return out


def _pick_config(M, E, N, K, transpose_b):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split). Swept on B200 over gpt-oss-20b shapes with
    uniform (decode) and real, skewed (prefill / training, hottest expert 3-6x the mean) routing."""
    per_expert = M / max(E, 1)
    if transpose_b:
        if per_expert <= 1:
            return (16, 64, 128, 4, 4, False)
        if per_expert <= 16:
            return (16, 128, 64, 4, 4, False)
        if per_expert <= 48:
            return (64, 256, 64, 8, 3, False)
        if per_expert <= 192:
            return (128, 256, 64, 4, 3, False)
        return (256, 256, 64, 8, 4, False)
    if per_expert <= 1:
        return (16, 64, 256, 4, 2, True)
    if per_expert <= 16:
        return (16, 64, 128, 4, 2, True)
    if per_expert <= 96:
        return (64, 128, 64, 4, 3, True)
    return (128, 128, 64, 8, 3, True)


def _split_permute(x, block_k):
    """Columns of each block_k block reordered [even | odd] for the split forward."""
    M, K = x.shape
    return x.view(M, K // block_k, block_k // 2, 2).transpose(-1, -2).reshape(M, K)


def _pick_dense_config(M, N, K, transpose_b):
    """Single weight (no expert schedule), non-split tiles: narrow N tiles keep every SM busy for skinny inputs.
    B200 sweep over Llama-3-8B Linear shapes, M <= 64; wider inputs reuse the grouped table."""
    if M > 64:
        return _pick_config(M, 1, N, K, transpose_b)
    if transpose_b:
        if M <= 16:
            return (16, 32, 256, 4, 3, False) if N <= 4096 else (16, 64, 256, 4, 3, False)
        if M <= 32:
            return (16, 64, 256, 4, 3, False) if N <= 4096 else (32, 64, 256, 4, 2, False)
        return (64, 32, 128, 4, 3, False) if N <= 4096 else (64, 64, 128, 4, 4, False)
    if M <= 16:
        return (16, 32, 256, 4, 3, False) if N <= 4096 else (16, 64, 256, 4, 3, False)
    if M <= 32:
        return (32, 32, 128, 4, 4, False) if N <= 4096 else (32, 64, 256, 4, 3, False)
    return (64, 32, 256, 4, 3, False) if N <= 4096 else (64, 64, 128, 4, 4, False)


_ASM_OK = {}


def _asm_ok(device):
    """Inline-PTX bf16x2 fma decode (Kimi-K3 port): sm_80+ CUDA; UNSLOTH_MXFP4_ASM=0 turns it off."""
    ok = _ASM_OK.get(device.index)
    if ok is None:
        import os
        index = device.index if device.index is not None else torch.cuda.current_device()
        ok = _ASM_OK[device.index] = (
            torch.version.hip is None
            and torch.cuda.get_device_capability(index) >= (8, 0)
            and os.environ.get("UNSLOTH_MXFP4_ASM", "1") != "0"
        )
    return ok


def _launch(x, blocks, scales, counts, out, transpose_b, config, launcher, bias = None):
    E, R, G, _ = blocks.shape
    M, K = x.shape
    N = out.shape[1]
    grouped = counts is not None
    if config is None:
        config = _pick_config(M, E, N, K, transpose_b) if grouped else _pick_dense_config(M, N, K, transpose_b)
    BM, BN, BK, warps, stages, split = config[:6]
    asm = config[6] if len(config) > 6 else (not grouped and _asm_ok(x.device))
    split = split and (K % BK == 0 if transpose_b else BN >= 32)
    if split and transpose_b:
        x = _split_permute(x, BK)
    # Plain ints: triton.cdiv / next_power_of_2 cost microseconds per call on the decode path.
    grid = (-(-M // BM) + (E if grouped else 0), -(-N // BN))
    launcher(_mxfp4_grouped_mm_kernel)[grid](
        x, blocks, scales, out, x if counts is None else counts, x if bias is None else bias,
        E, R, G, K, N, M,
        x.stride(0), out.stride(0),
        TRANS_B = transpose_b, E_POW2 = 1 << (E - 1).bit_length(),
        BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK, SPLIT = split,
        GROUPED = grouped, HAS_BIAS = bias is not None, ASM = asm,
        num_warps = warps, num_stages = stages,
    )
    return out


def _out_shape(x, blocks, transpose_b):
    E, R, G, _ = blocks.shape
    M, K = x.shape
    if transpose_b and K != G * 32:
        raise ValueError(f"Unsloth: x has {K} columns, packed rows hold {G * 32}")
    if not transpose_b and K != R:
        raise ValueError(f"Unsloth: x has {K} columns, packed stack has {R} rows")
    return (M, R if transpose_b else G * 32)


def mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = True, out = None, config = None):
    """Grouped ``x @ P[e]^T`` (``transpose_b``) or ``x @ P[e]`` over rows sorted by expert.
    ``counts`` (E,) int32 on device: rows per expert, sum == x.shape[0]. Returns (M, N) in x.dtype."""
    shape = _out_shape(x, blocks, transpose_b)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if out is None:
        out = torch.empty(shape, dtype = x.dtype, device = x.device)
    if x.shape[0] == 0:
        return out
    with torch.cuda.device(x.device):
        return _launch(x, blocks, scales, counts, out, transpose_b, config, lambda kernel: kernel)


# Custom op so torch.compile (and CUDA graphs) can capture the kernel inside a fused MoE region.
mxfp4_grouped_mm_op = None
if _HAS_TRITON:
    try:
        from torch.library import triton_op as _triton_op, wrap_triton as _wrap_triton

        @_triton_op("unsloth_zoo::mxfp4_grouped_mm", mutates_args = ())
        def mxfp4_grouped_mm_op(
            x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor, counts: torch.Tensor, transpose_b: bool,
        ) -> torch.Tensor:
            x = x.contiguous()
            out = torch.empty(_out_shape(x, blocks, transpose_b), dtype = x.dtype, device = x.device)
            return _launch(x, blocks, scales, counts, out, transpose_b, None, _wrap_triton)
    except Exception:
        mxfp4_grouped_mm_op = None


class Mxfp4GroupedMM(torch.autograd.Function):
    """Grouped GEMM against a frozen packed stack; backward returns dX only (the transposed product)."""

    @staticmethod
    def forward(ctx, x, counts, blocks, scales, transpose_b):
        ctx.save_for_backward(counts, blocks, scales)
        ctx.transpose_b = transpose_b
        return mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = transpose_b)

    @staticmethod
    def backward(ctx, grad_out):
        counts, blocks, scales = ctx.saved_tensors
        grad_x = mxfp4_grouped_mm(grad_out.contiguous(), blocks, scales, counts, transpose_b = not ctx.transpose_b)
        return grad_x, None, None, None, None


def mxfp4_expert_grouped_mm(x, blocks, scales, counts, transpose_b = True):
    """Autograd-aware ``mxfp4_grouped_mm`` (dX only; the packed stack is frozen)."""
    if torch.is_grad_enabled() and x.requires_grad:
        return Mxfp4GroupedMM.apply(x, counts, blocks, scales, transpose_b)
    return mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = transpose_b)


_AVAILABLE = {}


def mxfp4_grouped_mm_available(device = None) -> bool:
    """Whether the fused kernel compiles on ``device`` and matches dequantize + matmul on a probe."""
    if not _HAS_TRITON or not torch.cuda.is_available():
        return False
    device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return False
    key = str(device)
    if key in _AVAILABLE:
        return _AVAILABLE[key]
    ok = False
    try:
        with torch.autocast(device.type, enabled = False), torch.no_grad():
            ok = _probe(device)
    except Exception as exception:
        import warnings
        warnings.warn(f"Unsloth: fused MXFP4 grouped GEMM unavailable ({exception}); dequantizing instead.")
        ok = False
    _AVAILABLE[key] = ok
    return ok


def _probe(device):
    if True:
        from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch
        gen = torch.Generator(device = device).manual_seed(0)
        E, R, G = 3, 96, 4
        blocks = torch.randint(0, 256, (E, R, G, 16), dtype = torch.uint8, device = device, generator = gen)
        scales = torch.randint(118, 136, (E, R, G), dtype = torch.uint8, device = device, generator = gen)
        counts = torch.tensor([5, 0, 19], dtype = torch.int32, device = device)
        x = torch.randn(24, G * 32, dtype = torch.bfloat16, device = device, generator = gen)
        dense = mxfp4_dequantize_torch(blocks, scales, dtype = torch.bfloat16)  # (E, R, C)
        got = mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = True).float()
        rows = torch.repeat_interleave(torch.arange(E, device = device), counts.long())
        want = torch.bmm(x.float().unsqueeze(1), dense.float()[rows].transpose(1, 2)).squeeze(1)
        gy = torch.randn(24, R, dtype = torch.bfloat16, device = device, generator = gen)
        got_t = mxfp4_grouped_mm(gy, blocks, scales, counts, transpose_b = False).float()
        want_t = torch.bmm(gy.float().unsqueeze(1), dense.float()[rows]).squeeze(1)
        return bool(
            torch.allclose(got, want, rtol = 2e-2, atol = 2e-2 * want.abs().max().item())
            and torch.allclose(got_t, want_t, rtol = 2e-2, atol = 2e-2 * want_t.abs().max().item())
        )


def mxfp4_grouped_matmul(x, blocks, scales, counts, trans = False):
    """Stacked-experts calling convention: ``trans=False`` is ``x @ W^T`` (reduce over the packed axis),
    ``trans=True`` its dX ``x @ W``; no autograd. Same kernel as ``mxfp4_grouped_mm``."""
    counts = counts.to(device = x.device, dtype = torch.int32)
    if scales.device != blocks.device:
        scales = scales.to(blocks.device)
    return mxfp4_grouped_mm(x, blocks.contiguous(), scales.contiguous(), counts.contiguous(), transpose_b = not trans)


def mxfp4_matmul(x, blocks, scales, trans = False, out = None, bias = None):
    """``x @ W^T (+ bias)`` (``trans=False``) or ``x @ W`` for one MXFP4 weight W = decode(blocks ``(N, G, 16)`` or
    ``(N, G * 16)``, scales ``(N, G)``). bias is added to the fp32 accumulator; ``out`` (contiguous, x's dtype, right
    size) is written in place, else a new tensor is returned."""
    N, G = scales.shape[-2], scales.shape[-1]
    blocks = blocks.contiguous().view(1, N, G, 16)
    if scales.device != blocks.device:
        scales = scales.to(blocks.device)
    scales = scales.contiguous().view(1, N, G)
    lead = x.shape[:-1]
    x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) < x2.shape[1]):
        x2 = x2.contiguous()
    M = x2.shape[0]
    n_out = G * 32 if trans else N
    if x2.shape[1] != (N if trans else G * 32):
        raise ValueError(f"Unsloth: MXFP4 matmul expects {N if trans else G * 32} input features, got {x2.shape[1]}")
    if out is None or out.dtype != x.dtype or not out.is_contiguous() or out.numel() != M * n_out:
        out = torch.empty((M, n_out), dtype = x.dtype, device = x.device)
    else:
        out = out.view(M, n_out)
    if bias is not None:
        bias = bias.to(device = x.device).contiguous()
    if M > 0:
        # Decode-sized calls are CPU bound: switch devices only when needed.
        index = x.device.index
        prior = torch.cuda.current_device()
        switch = index is not None and index != prior
        if switch:
            torch.cuda.set_device(index)
        try:
            _launch(x2, blocks, scales, None, out, not trans, None, lambda kernel: kernel, bias = bias)
        finally:
            if switch:
                torch.cuda.set_device(prior)
    return out if x.dim() == 2 else out.view(*lead, n_out)


_AVAILABLE_DTYPE = {}


def _probe_dense(device, dtype):
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch
    gen = torch.Generator(device = device).manual_seed(0)
    N, G = 96, 5
    blocks = torch.randint(0, 256, (N, G, 16), dtype = torch.uint8, device = device, generator = gen)
    scales = torch.randint(118, 130, (N, G), dtype = torch.uint8, device = device, generator = gen)
    w = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)
    bias = torch.randn(N, device = device, generator = gen).to(dtype)
    for rows in (1, 37):
        x = torch.randn(rows, G * 32, device = device, generator = gen).to(dtype)
        got = mxfp4_matmul(x, blocks, scales, bias = bias).float()
        want = x.float() @ w.t() + bias.float()
        gy = torch.randn(rows, N, device = device, generator = gen).to(dtype)
        got_t = mxfp4_matmul(gy, blocks, scales, trans = True).float()
        want_t = gy.float() @ w
        for g, t in ((got, want), (got_t, want_t)):
            if not torch.allclose(g, t, rtol = 2e-2, atol = 2e-2 * t.abs().max().item()):
                return False
    return True


def mxfp4_gemm_available(device = None, dtype = torch.bfloat16) -> bool:
    """Fused kernel usable for ``dtype`` activations on ``device``: self-checked once per (device, dtype), grouped and
    single-weight. bf16 and fp16 (the decode is exact in bf16, then cast); ``UNSLOTH_MXFP4_FUSED_GEMM=0`` turns it off."""
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    import os
    if os.environ.get("UNSLOTH_MXFP4_FUSED_GEMM", "auto") in ("0", "dequant"):
        return False
    key = (None if device is None else str(device), dtype)
    ok = _AVAILABLE_DTYPE.get(key)
    if ok is not None:
        return ok
    ok = mxfp4_grouped_mm_available(device)
    if ok:
        device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        try:
            with torch.autocast(device.type, enabled = False), torch.no_grad():
                ok = _probe_dense(device, dtype)
        except Exception as exception:
            import warnings
            warnings.warn(f"Unsloth: fused MXFP4 GEMM unavailable for {dtype} ({exception}); dequantizing instead.")
            ok = False
    _AVAILABLE_DTYPE[key] = ok
    return ok
