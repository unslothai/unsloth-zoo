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

"""bf16 / fp16 activations x MXFP4 weights (blocks ``(E, N, G, 16)`` + e8m0 scales ``(E, N, G)``) without a 16-bit weight.

The weight tile is decoded in registers (same bits as ``mxfp4_dequant``) and fed to ``tl.dot``, so only the packed
bytes are read. ``trans = False`` is ``x @ W^T`` (reduce over the packed K), ``trans = True`` is ``x @ W`` (reduce over
N, the dX of a frozen base). Grouped mode runs expert-sorted rows; experts with no rows are never read.
"""

import os
import torch

__all__ = [
    "mxfp4_matmul",
    "mxfp4_grouped_matmul",
    "mxfp4_gemm_available",
]

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_FAILED = False
_VERIFIED = {}


if _HAS_TRITON:

    @triton.jit
    def _decode_tile_asm(packed, scale):
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
    def _decode_tile2d(packed, scale, FAST_CVT: tl.constexpr, HAS_TOP: tl.constexpr):
        # packed (R, C // 2) uint8, scale (R, C // 32) uint8 -> (R, C) bf16, bits as mxfp4_dequant.
        R: tl.constexpr = packed.shape[0]
        CB: tl.constexpr = packed.shape[1]
        CG: tl.constexpr = scale.shape[1]
        if FAST_CVT:
            pair = tl.inline_asm_elementwise(
                asm = """
                {
                .reg .b8 in_8;
                .reg .f16x2 out;
                cvt.u8.u32 in_8, $1;
                cvt.rn.f16x2.e2m1x2 out, in_8;
                mov.b32 $0, out;
                }
                """,
                constraints = "=r,r", args = [packed.to(tl.uint32)], dtype = tl.uint32, is_pure = True, pack = 1,
            )
            lo = (pair & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast = True).to(tl.bfloat16)
            hi = (pair >> 16).to(tl.uint16).to(tl.float16, bitcast = True).to(tl.bfloat16)
        else:
            p = packed.to(tl.int32)
            em0 = p & 0x07
            em1 = p & 0x70
            x0 = (em0 << 6) | ((p & 0x08) << 12)
            x1 = (em1 << 2) | ((p & 0x80) << 8)
            x0 = tl.where((em0 & 0x06) != 0, x0 + (126 << 7), x0)
            x1 = tl.where((em1 & 0x60) != 0, x1 + (126 << 7), x1)
            x0 = tl.where(em0 == 0x01, 16128 | (x0 & 0x8000), x0)
            x1 = tl.where(em1 == 0x10, 16128 | (x1 & 0x8000), x1)
            lo = x0.to(tl.uint16).to(tl.bfloat16, bitcast = True)
            hi = x1.to(tl.uint16).to(tl.bfloat16, bitcast = True)
        vals = tl.reshape(tl.join(lo, hi), (R, CG, 32))
        s = scale.to(tl.int32)
        if HAS_TOP:
            top = s == 255
            s = tl.where(top, 254, s)
        s16 = tl.maximum(s << 7, 0x0040).to(tl.uint16).to(tl.bfloat16, bitcast = True)
        vals = vals * s16[:, :, None]
        if HAS_TOP:
            vals = vals * tl.where(top, 2.0, 1.0).to(tl.bfloat16)[:, :, None]
        return tl.reshape(vals, (R, CG * 32))

    @triton.jit
    def _mxfp4_gemm_kernel(
        x_ptr, blocks_ptr, scales_ptr, out_ptr, tile_cum_ptr, row_end_ptr, bias_ptr,
        M, N, G, NUM_E, stride_xm, stride_om,
        TRANS: tl.constexpr, GROUPED: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        FAST_CVT: tl.constexpr, HAS_TOP: tl.constexpr, ASM: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_n = tl.program_id(1)
        if GROUPED:
            total = tl.load(tile_cum_ptr + NUM_E - 1)
            if pid_t >= total:
                return
            # First expert whose inclusive tile cumsum passes pid_t.
            lo = 0
            hi = NUM_E - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if tl.load(tile_cum_ptr + mid) > pid_t:
                    hi = mid
                else:
                    lo = mid + 1
            e = lo
            tile_start = tl.where(e > 0, tl.load(tile_cum_ptr + e - 1, mask = e > 0, other = 0), 0)
            row_start = tl.where(e > 0, tl.load(row_end_ptr + e - 1, mask = e > 0, other = 0), 0)
            m0 = row_start + (pid_t - tile_start) * BLOCK_M
            m_end = tl.load(row_end_ptr + e)
        else:
            e = 0
            m0 = pid_t * BLOCK_M
            m_end = M
        K = G * 32
        e64 = e.to(tl.int64)
        w_base = blocks_ptr + e64 * N * G * 16
        s_base = scales_ptr + e64 * N * G
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_end
        x_rows = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
        red_hi = N if TRANS else K
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        GB = G * 16
        if TRANS:
            # out (M, K) = x (M, N) @ W (N, K): tile rows = reduction over N, cols = output over K.
            offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_cb = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
            offs_cg = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
            mask_cb = offs_cb < GB
            mask_cg = offs_cg < G
            for r0 in range(0, red_hi, BLOCK_K):
                offs_r = r0 + tl.arange(0, BLOCK_K)
                mask_r = offs_r < red_hi
                a = tl.load(x_rows + offs_r[None, :], mask = mask_m[:, None] & mask_r[None, :], other = 0.0)
                rows64 = offs_r.to(tl.int64)
                packed = tl.load(w_base + rows64[:, None] * GB + offs_cb[None, :], mask = mask_r[:, None] & mask_cb[None, :], other = 0)
                scale = tl.load(s_base + rows64[:, None] * G + offs_cg[None, :], mask = mask_r[:, None] & mask_cg[None, :], other = 127)
                if ASM:
                    w = _decode_tile_asm(packed, scale).to(a.dtype)
                else:
                    w = _decode_tile2d(packed, scale, FAST_CVT, HAS_TOP).to(a.dtype)
                acc = tl.dot(a, w, acc)
            out_cols = offs_c
            mask_out = mask_m[:, None] & (offs_c[None, :] < K)
        else:
            # out (M, N) = x (M, K) @ W^T: tile rows = output over N, cols = reduction over packed K.
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            rows64 = offs_n.to(tl.int64)
            for k0 in range(0, red_hi, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                offs_kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                offs_g = k0 // 32 + tl.arange(0, BLOCK_K // 32)
                a = tl.load(x_rows + offs_k[None, :], mask = mask_m[:, None] & (offs_k[None, :] < red_hi), other = 0.0)
                packed = tl.load(w_base + rows64[:, None] * GB + offs_kb[None, :], mask = mask_n[:, None] & (offs_kb[None, :] < red_hi // 2), other = 0)
                scale = tl.load(s_base + rows64[:, None] * G + offs_g[None, :], mask = mask_n[:, None] & (offs_g[None, :] < red_hi // 32), other = 127)
                if ASM:
                    w = _decode_tile_asm(packed, scale).to(a.dtype)
                else:
                    w = _decode_tile2d(packed, scale, FAST_CVT, HAS_TOP).to(a.dtype)
                acc = tl.dot(a, tl.trans(w), acc)
            out_cols = offs_n
            mask_out = mask_m[:, None] & mask_n[None, :]
        if HAS_BIAS:
            acc += tl.load(bias_ptr + out_cols, mask = out_cols < (K if TRANS else N), other = 0.0).to(tl.float32)[None, :]
        out_ptrs = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + out_cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask = mask_out)


# Test / tuning hook: a fixed (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for every launch.
CONFIG_OVERRIDE = tuple(int(v) for v in os.environ.get("UNSLOTH_MXFP4_GEMM_CONFIG", "").split(",") if v) or None
ENABLED = os.environ.get("UNSLOTH_MXFP4_FUSED_GEMM", "1") != "0"


def _pick_config(M_rows, n_out, red, trans, grouped, num_sms):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) from B200 sweeps; M_rows = rows per group."""
    if CONFIG_OVERRIDE is not None:
        return CONFIG_OVERRIDE
    if not grouped:
        # One weight and few rows: narrow tiles keep every SM busy.
        # The transposed tile needs BLOCK_N >= 64 (Triton rejects the 16-byte-wide shared layout).
        if M_rows <= 16:
            return (16, 32, 256, 4, 3) if n_out <= 4096 and not trans else (16, 64, 256, 4, 3)
        if M_rows <= 32:
            return (32, 64 if trans else 32, 256, 4, 3)
        if M_rows <= 64:
            return (64, 64 if trans else 32, 128, 4, 3)
    elif M_rows <= 16:
        return (16, 128, 128, 4, 3)
    elif M_rows <= 32:
        return (64, 128, 128, 4, 3) if trans else (32, 128, 64, 4, 3)
    elif M_rows <= 64:
        return (64, 128, 128, 4, 3)
    return (128, 128, 64, 4, 3) if trans else (128, 128, 64, 8, 3)


_DEVICE_INFO = {}


def _device_info(device):
    """(num_sms, fast_cvt, asm_ok) per device; cvt.rn.f16x2.e2m1x2 needs sm_100+, the fma decode sm_80+."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    info = _DEVICE_INFO.get(index)
    if info is None:
        props = torch.cuda.get_device_properties(index)
        cuda = torch.version.hip is None
        info = _DEVICE_INFO[index] = (
            props.multi_processor_count,
            cuda and props.major >= 10 and os.environ.get("UNSLOTH_MXFP4_FAST_CVT", "1") != "0",
            cuda and (props.major, props.minor) >= (8, 0) and os.environ.get("UNSLOTH_MXFP4_ASM", "1") != "0",
        )
    return info


def _cdiv(a, b):
    return -(-a // b)


def _launch(x, blocks, scales, trans, grouped, counts, out_dtype, M_rows_hint, out = None, bias = None):
    # Decode-sized calls are CPU bound: plain Python ints, no split-K reduction kernels, one launch.
    E = blocks.shape[0] if grouped else 1
    N, G = blocks.shape[-3], blocks.shape[-2]
    K = G * 32
    M = x.shape[0]
    n_out, red = (K, N) if trans else (N, K)
    if x.shape[1] != red:
        raise ValueError(f"Unsloth: MXFP4 matmul expects {red} input features, got {x.shape[1]}")
    num_sms, fast_cvt, asm_ok = _device_info(x.device)
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _pick_config(M_rows_hint, n_out, red, trans, grouped, num_sms)
    if grouped:
        tiles = torch.div(counts + (BLOCK_M - 1), BLOCK_M, rounding_mode = "floor")
        tile_cum = torch.cumsum(tiles, 0, dtype = torch.int32)
        row_end = torch.cumsum(counts, 0, dtype = torch.int32)
        # Host bound without a sync: every non-empty expert wastes at most one partial tile.
        n_tiles = _cdiv(M, BLOCK_M) + min(E, M)
    else:
        tile_cum = row_end = x
        n_tiles = _cdiv(M, BLOCK_M)
    if out is None or out.dtype != out_dtype or not out.is_contiguous() or out.numel() != M * n_out:
        out = torch.empty((M, n_out), dtype = out_dtype, device = x.device)
    else:
        out = out.view(M, n_out)
    if M == 0:
        return out
    switch = x.device.index is not None and x.device.index != torch.cuda.current_device()
    prior = torch.cuda.current_device() if switch else None
    if switch:
        torch.cuda.set_device(x.device)
    try:
        _mxfp4_gemm_kernel[(n_tiles, _cdiv(n_out, BLOCK_N))](
            x, blocks, scales, out, tile_cum, row_end, x if bias is None else bias,
            M, N, G, E, x.stride(0), n_out,
            TRANS = trans, GROUPED = grouped,
            FAST_CVT = fast_cvt, HAS_TOP = True, ASM = asm_ok,
            HAS_BIAS = bias is not None,
            BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N, BLOCK_K = BLOCK_K,
            num_warps = num_warps, num_stages = num_stages,
        )
    finally:
        if switch:
            torch.cuda.set_device(prior)
    return out


def _prepare(x, blocks, scales):
    if x.stride(-1) != 1 or (x.dim() == 2 and x.shape[0] > 1 and x.stride(0) < x.shape[1]):
        x = x.contiguous()
    if not blocks.is_contiguous():
        blocks = blocks.contiguous()
    if scales.device != blocks.device:
        scales = scales.to(blocks.device)
    if not scales.is_contiguous():
        scales = scales.contiguous()
    return x, blocks, scales


def mxfp4_matmul(x, blocks, scales, trans = False, out = None, bias = None):
    """``x @ W^T (+ bias)`` (``trans=False``) or ``x @ W`` for W = decode(blocks ``(N, G, 16)`` or ``(N, G*16)``, scales
    ``(N, G)``); bias is added to the fp32 accumulator. ``out`` (contiguous, x's dtype, right size) is written in place."""
    N, G = scales.shape[-2], scales.shape[-1]
    blocks = blocks.reshape(N, G, 16)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    x2, blocks, scales = _prepare(x2, blocks, scales)
    if bias is not None:
        bias = bias.to(device = x2.device).contiguous()
    y = _launch(x2, blocks, scales.reshape(N, G), trans, False, None, x.dtype, x2.shape[0], out, bias)
    return y.view(*lead, y.shape[-1])


def mxfp4_grouped_matmul(x, blocks, scales, counts, trans = False):
    """Expert-sorted rows ``x`` (rows of expert e contiguous, ``counts[e]`` of them) times each expert's MXFP4 W."""
    x, blocks, scales = _prepare(x, blocks, scales)
    counts = counts.to(device = x.device, dtype = torch.int32)
    E = blocks.shape[0]
    hint = max(1, x.shape[0] // max(1, min(E, x.shape[0])))
    return _launch(x, blocks, scales, trans, True, counts, x.dtype, hint)


def _reference(x, blocks, scales, trans, counts):
    from .mxfp4_dequant import mxfp4_dequantize_torch
    w = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)  # (E, N, K)
    xf = x.float()
    if counts is None:
        return xf @ (w[0] if trans else w[0].t())
    outs, start = [], 0
    for e, c in enumerate(counts.tolist()):
        outs.append(xf[start:start + c] @ (w[e] if trans else w[e].t()))
        start += c
    return torch.cat(outs, 0)


def _self_check(device, dtype):
    """Decode bits vs mxfp4_dequant's reference (identity input: every product is exact), both decoders; then a
    grouped + transposed GEMM vs an fp32 reference. Returns (ok, asm_ok)."""
    from .mxfp4_dequant import mxfp4_dequantize_torch

    gen = torch.Generator(device = device).manual_seed(0)

    def decode_exact(high):
        N, G = 96, 5
        blocks = torch.randint(0, 256, (N, G, 16), dtype = torch.uint8, device = device, generator = gen)
        scales = torch.randint(0, high + 1, (N, G), dtype = torch.uint8, device = device, generator = gen)
        edges = [0, 1, 127, 128] if high <= 128 else [0, 1, 127, 254, 255]
        scales[: len(edges), 0] = torch.tensor(edges, dtype = torch.uint8, device = device)
        if high > 128:
            # Rows scaled by >= 2^127 keep only |v| <= 0.5 (nibbles 0, 1, 8, 9) so every value stays finite.
            small = torch.tensor([0, 1, 8, 9], dtype = torch.uint8, device = device)
            pick = torch.randint(0, 4, (2, N, G, 16), device = device, generator = gen)
            blocks = torch.where(scales[:, :, None] > 126, small[pick[0]] | (small[pick[1]] << 4), blocks)
        want = mxfp4_dequantize_torch(blocks, scales, dtype = dtype).float()
        eye_k = torch.eye(G * 32, device = device, dtype = dtype)
        eye_n = torch.eye(N, device = device, dtype = dtype)
        return torch.equal(mxfp4_matmul(eye_k, blocks, scales).float().t(), want) and torch.equal(
            mxfp4_matmul(eye_n, blocks, scales, trans = True).float(), want
        )

    def decode_ok():
        # fp16 cannot hold 2^127-scaled values, so only bf16 checks the top of the e8m0 range.
        return decode_exact(128) and (dtype != torch.bfloat16 or decode_exact(255))

    asm_ok = decode_ok()
    if not asm_ok and _device_info(device)[2]:
        index = device.index if device.index is not None else torch.cuda.current_device()
        num_sms, fast_cvt, _ = _DEVICE_INFO[index]
        _DEVICE_INFO[index] = (num_sms, fast_cvt, False)
        if not decode_ok():
            return False, False
    elif not asm_ok:
        return False, False
    counts = [5, 0, 37]
    blocks = torch.randint(0, 256, (3, 64, 3, 16), dtype = torch.uint8, device = device, generator = gen)
    scales = torch.randint(120, 128, (3, 64, 3), dtype = torch.uint8, device = device, generator = gen)
    for trans in (False, True):
        x = torch.randn(sum(counts), 64 if trans else 96, device = device, generator = gen).to(dtype)
        got = mxfp4_grouped_matmul(x, blocks, scales, torch.tensor(counts, device = device), trans = trans)
        want = _reference(x, blocks, scales, trans, torch.tensor(counts))
        if (got.float() - want).abs().max().item() > 2e-2 * max(1.0, want.abs().max().item()):
            return False, asm_ok
    return True, asm_ok


def mxfp4_gemm_available(device = None, dtype = torch.bfloat16) -> bool:
    """Triton path usable here, self-checked once per (device, dtype); ``UNSLOTH_MXFP4_FUSED_GEMM=0`` turns it off."""
    global _FAILED
    if not ENABLED or _FAILED or not _HAS_TRITON:
        return False
    index = getattr(device, "index", None) if device is not None else None
    ok = _VERIFIED.get((index, dtype))
    if ok is not None:
        return ok
    if dtype not in (torch.bfloat16, torch.float16) or not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if device.type != "cuda":
        return False
    try:
        ok, _ = _self_check(device, dtype)
        if not ok:
            import warnings
            warnings.warn("Unsloth: fused MXFP4 GEMM failed its self-check here; decoding the weight instead.")
    except Exception as exception:
        import warnings
        warnings.warn(f"Unsloth: fused MXFP4 GEMM unavailable ({exception}); decoding the weight instead.")
        _FAILED = True
        ok = False
    _VERIFIED[(device.index, dtype)] = ok
    if index is None:
        _VERIFIED[(None, dtype)] = ok
    return ok
