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
"""Grouped GEMM over expert-sorted rows decoding bitsandbytes NF4 expert stacks in the K loop.

Decode matches dequantize_4bit bitwise (same fp32 rounding order); backward is dX only via swapped
strides on the packed tensors, so no dense expert stack is ever allocated.
UNSLOTH_MOE_NF4_FUSED=1 (CUDA sm80+, not ROCm), =auto (sm80+, at most UNSLOTH_MOE_NF4_AUTO_MAX_ROWS,
default 128, rows per expert: slower than dequant from 256 on L4), =force (any CUDA; sm75 is 60x+
slower). UNSLOTH_MOE_NF4_AUTOTUNE=1 tunes tiles per shape (bits then follow the pick)."""

__all__ = [
    "NF4_FUSED_STATS",
    "nf4_fused_mode",
    "nf4_auto_fused",
    "nf4_grouped_mm",
    "nf4_grouped_spec",
    "maybe_nf4_grouped_mm",
]

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

# calls / fallbacks: eager dispatch decisions; launches: kernel runs (forward and dX), counted
# inside the op so a compiled caller is counted too.
NF4_FUSED_STATS = {"calls": 0, "fallbacks": 0, "launches": 0}
_DISABLED_REASON = None   # set by hand / tests: every stack takes the dequant path
FAILED_SHAPES = {}        # (inner, outer, s_i, s_j) -> reason: a build or launch failed for that GEMM
_OP_NS = "unsloth_zoo"


def nf4_fused_mode() -> str:
    """'0' off (default), '1' sm80+ CUDA, 'auto' sm80+ CUDA at or below AUTO_MAX_ROWS rows per expert,
    'force' any CUDA device."""
    mode = os.environ.get("UNSLOTH_MOE_NF4_FUSED", "0").strip().lower()
    if mode in ("1", "true", "on", "yes"):
        return "1"
    if mode in ("auto", "force"):
        return mode
    return "0"


# Average rows per expert (tokens * top_k / E) at or below which "auto" fuses: the dequant pass is
# then most of the current path's time. Measured crossover on L4 bf16.
AUTO_MAX_ROWS = 128


def nf4_auto_max_rows() -> int:
    try:
        return int(os.environ.get("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", AUTO_MAX_ROWS))
    except ValueError:
        return AUTO_MAX_ROWS


def nf4_auto_fused(rows, num_experts) -> bool:
    """Shapes only (no host sync); under torch.compile this guards on the row count."""
    return rows <= nf4_auto_max_rows() * num_experts


if triton is not None:

    def _configs():
        c = triton.Config
        out = []
        for bm in (16, 32, 64, 128):
            for bn in (64, 128):
                for warps in (4, 8):
                    for lut in (0, 1):
                        if warps == 8 and bm * bn < 64 * 64:
                            continue
                        out.append(c({"BM": bm, "BN": bn, "BK": 64, "LUT": lut, "MSUB": 1}, num_warps = warps, num_stages = 3))
        return out

    def _prune(configs, named_args, **kwargs):
        # Shared memory and the pair layout need BK, BN >= 16 and even; small dims drop huge tiles.
        args = {**named_args, **kwargs}
        I, J, rows = args["I"], args["J"], args["M_HINT"]
        # BM by average rows per expert: a tile mostly past an expert's last row decodes for nothing.
        bms = (16, 32) if rows <= 16 else (32, 64) if rows <= 64 else (64, 128)
        keep = [
            cfg for cfg in configs
            if cfg.kwargs["BM"] in bms
            # A [64+, 32] pair tile transposed for the dot trips TritonGPUOptimizeDotOperands on sm100.
            and not (args["PAIR"] == 2 and cfg.kwargs["BK"] < 64)
            and cfg.kwargs["BN"] <= max(16, triton.next_power_of_2(J))
            and cfg.kwargs["BK"] <= max(16, triton.next_power_of_2(I))
        ]
        return keep or configs[:1]

    @triton.jit
    def _nf4_select(nib, CODE):
        # code[nib] as a select tree on the nibble bits (bitsandbytes' dDequantizeNF4 shape): no gather.
        b0 = (nib & 1) != 0
        b1 = (nib & 2) != 0
        b2 = (nib & 4) != 0
        b3 = (nib & 8) != 0
        v0 = tl.where(b0, tl.load(CODE + 1), tl.load(CODE + 0))
        v1 = tl.where(b0, tl.load(CODE + 3), tl.load(CODE + 2))
        v2 = tl.where(b0, tl.load(CODE + 5), tl.load(CODE + 4))
        v3 = tl.where(b0, tl.load(CODE + 7), tl.load(CODE + 6))
        v4 = tl.where(b0, tl.load(CODE + 9), tl.load(CODE + 8))
        v5 = tl.where(b0, tl.load(CODE + 11), tl.load(CODE + 10))
        v6 = tl.where(b0, tl.load(CODE + 13), tl.load(CODE + 12))
        v7 = tl.where(b0, tl.load(CODE + 15), tl.load(CODE + 14))
        w0 = tl.where(b1, v1, v0)
        w1 = tl.where(b1, v3, v2)
        w2 = tl.where(b1, v5, v4)
        w3 = tl.where(b1, v7, v6)
        x0 = tl.where(b2, w1, w0)
        x1 = tl.where(b2, w3, w2)
        return tl.where(b3, x1, x0)

    @triton.jit
    def _code(nib, CODE, LUT: tl.constexpr):
        if LUT == 1:
            return _nf4_select(nib, CODE)
        return tl.load(CODE + nib)

    @triton.jit
    def _decode_pairs(q, scale, CODE, W_DT: tl.constexpr, OUT_DT: tl.constexpr, LUT: tl.constexpr):
        hi = _code((q >> 4).to(tl.int32), CODE, LUT)
        lo = _code((q & 15).to(tl.int32), CODE, LUT)
        hi = (hi * scale).to(W_DT).to(OUT_DT)
        lo = (lo * scale).to(W_DT).to(OUT_DT)
        return hi, lo

    @triton.jit
    def _scale(blk, mask, ABSMAX, CODE2, ABSMAX2, OFFSET, BS2: tl.constexpr, NESTED: tl.constexpr):
        if NESTED:
            aq = tl.load(ABSMAX + blk, mask = mask, other = 0).to(tl.int32)
            am2 = tl.load(ABSMAX2 + blk // BS2, mask = mask, other = 0.0).to(tl.float32)
            # Multiply then add, two roundings, as bitsandbytes does.
            return tl.load(CODE2 + aq) * am2 + tl.load(OFFSET).to(tl.float32)
        return tl.load(ABSMAX + blk, mask = mask, other = 0.0).to(tl.float32)

    @triton.autotune(configs = _configs(), key = ["I", "J", "M_HINT", "PAIR"], prune_configs_by = {"early_config_prune": _prune})
    @triton.jit
    def _nf4_grouped_kernel(
        A, Q, ABSMAX, CODE2, ABSMAX2, OFFSET, CODE, ENDS, C,
        M, M_HINT,
        NUM_E: tl.constexpr, E_POW2: tl.constexpr,
        I: tl.constexpr, J: tl.constexpr, S_I: tl.constexpr, S_J: tl.constexpr,
        PAIR: tl.constexpr,                 # 1: bytes pair along J, 2: along I, 0: per element
        BS: tl.constexpr, BS2: tl.constexpr, NESTED: tl.constexpr, W_DT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, LUT: tl.constexpr,
        MSUB: tl.constexpr,                 # M sub-tiles sharing one decoded weight tile
    ):
        pid = tl.program_id(0)
        pid_n = tl.program_id(1)
        # Tile -> expert without a host sync: per-expert tile counts from the row ends, prefix-summed.
        e_idx = tl.arange(0, E_POW2)
        e_ok = e_idx < NUM_E
        ends = tl.load(ENDS + e_idx, mask = e_ok, other = 0).to(tl.int32)
        starts = tl.load(ENDS + e_idx - 1, mask = e_ok & (e_idx > 0), other = 0).to(tl.int32)
        tiles = tl.where(e_ok, (ends - starts + BM * MSUB - 1) // (BM * MSUB), 0)
        cum = tl.cumsum(tiles, 0)
        if pid >= tl.sum(tiles, 0):
            return
        e = tl.sum((cum <= pid).to(tl.int32), 0)
        sel = e_idx == e
        row_lo = tl.sum(tl.where(sel, starts, 0), 0)
        row_hi = tl.sum(tl.where(sel, ends, 0), 0)
        local = pid - tl.sum(tl.where(sel, cum - tiles, 0), 0)

        offs_m = row_lo + local * (BM * MSUB) + tl.arange(0, BM)
        mask_m = offs_m < row_hi
        offs_m = offs_m.to(tl.int64)
        offs_m1 = offs_m + BM
        mask_m1 = offs_m1 < row_hi
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask_n = offs_n < J
        w_base = e.to(tl.int64) * (I * J)       # even whenever PAIR != 0
        out_dt = C.dtype.element_ty

        acc = tl.zeros((BM, BN), dtype = tl.float32)
        acc1 = tl.zeros((BM, BN), dtype = tl.float32)
        for k0 in range(0, I, BK):
            offs_k = k0 + tl.arange(0, BK)
            mask_k = offs_k < I
            a = tl.load(A + offs_m[:, None] * I + offs_k[None, :], mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
            if PAIR == 1:
                # W[k, n] at w_base + k * J + n: a byte holds columns 2p (high nibble) and 2p + 1.
                offs_p = pid_n * (BN // 2) + tl.arange(0, BN // 2)
                bmask = mask_k[:, None] & (offs_p < J // 2)[None, :]
                byte = w_base // 2 + offs_k[:, None].to(tl.int64) * (J // 2) + offs_p[None, :]
                q = tl.load(Q + byte, mask = bmask, other = 0)
                if (J % BS == 0) and (BS % BN == 0):
                    # The tile's columns sit in one absmax block per row: one scale per k.
                    blk = (w_base + offs_k.to(tl.int64) * J + pid_n * BN) // BS
                    s = _scale(blk, mask_k, ABSMAX, CODE2, ABSMAX2, OFFSET, BS2, NESTED)[:, None]
                else:
                    s = _scale(byte // (BS // 2), bmask, ABSMAX, CODE2, ABSMAX2, OFFSET, BS2, NESTED)
                hi, lo = _decode_pairs(q, s, CODE, W_DT, out_dt, LUT)
                b = tl.reshape(tl.join(hi, lo), (BK, BN))
            elif PAIR == 2:
                # W[k, n] at w_base + n * I + k: a byte holds rows 2p and 2p + 1 of one column.
                offs_p = k0 // 2 + tl.arange(0, BK // 2)
                bmask = mask_n[:, None] & (offs_p < I // 2)[None, :]
                byte = w_base // 2 + offs_n[:, None].to(tl.int64) * (I // 2) + offs_p[None, :]
                q = tl.load(Q + byte, mask = bmask, other = 0)
                if (I % BS == 0) and (BS % BK == 0):
                    blk = (w_base + offs_n.to(tl.int64) * I + k0) // BS
                    s = _scale(blk, mask_n, ABSMAX, CODE2, ABSMAX2, OFFSET, BS2, NESTED)[:, None]
                else:
                    s = _scale(byte // (BS // 2), bmask, ABSMAX, CODE2, ABSMAX2, OFFSET, BS2, NESTED)
                hi, lo = _decode_pairs(q, s, CODE, W_DT, out_dt, LUT)
                b = tl.trans(tl.reshape(tl.join(hi, lo), (BN, BK)))
            else:
                bmask = mask_k[:, None] & mask_n[None, :]
                flat = w_base + offs_k[:, None].to(tl.int64) * S_I + offs_n[None, :].to(tl.int64) * S_J
                q = tl.load(Q + flat // 2, mask = bmask, other = 0)
                nib = tl.where((flat % 2) == 0, q >> 4, q & 15).to(tl.int32)
                s = _scale(flat // BS, bmask, ABSMAX, CODE2, ABSMAX2, OFFSET, BS2, NESTED)
                b = (_code(nib, CODE, LUT) * s).to(W_DT).to(out_dt)
            # Masked weights decode finite against zero A / unstored columns: no select needed.
            acc = tl.dot(a, b, acc)
            if MSUB == 2:
                a1 = tl.load(A + offs_m1[:, None] * I + offs_k[None, :], mask = mask_m1[:, None] & mask_k[None, :], other = 0.0)
                acc1 = tl.dot(a1, b, acc1)
        tl.store(C + offs_m[:, None] * J + offs_n[None, :], acc.to(out_dt), mask = mask_m[:, None] & mask_n[None, :])
        if MSUB == 2:
            tl.store(C + offs_m1[:, None] * J + offs_n[None, :], acc1.to(out_dt), mask = mask_m1[:, None] & mask_n[None, :])

    _TL_DTYPES = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


def _launch(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j, blocksize, blocksize2, nested, w_dtype):
    M = a.shape[0]
    out = a.new_empty((M, outer))
    if M == 0:
        return out
    E = ends.shape[0]
    if s_j == 1 and outer % 2 == 0:
        pair = 1
    elif s_i == 1 and inner % 2 == 0:
        pair = 2
    else:
        pair = 0
    m_hint = triton.next_power_of_2(max(1, M // max(E, 1)))
    grid = lambda meta: (triton.cdiv(M, meta["BM"] * meta["MSUB"]) + E, triton.cdiv(outer, meta["BN"]))
    key = (inner, outer, s_i, s_j)
    if _DISABLED_REASON is not None or key in FAILED_SHAPES:
        return _launch_eager(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j,
                             blocksize, blocksize2, nested, w_dtype)
    NF4_FUSED_STATS["launches"] += 1
    try:
        _run_kernel(a, out, grid, packed, absmax, code2, absmax2, offset, code, ends, M, m_hint, E,
                    inner, outer, s_i, s_j, pair, blocksize, blocksize2, nested, w_dtype)
    except Exception as exc:
        # A Triton build or launch failure: decode in torch for this call; the stack takes the dequant path after.
        NF4_FUSED_STATS["launches"] -= 1
        FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"[:800]
        logger.warning(f"Unsloth: the NF4 fused grouped GEMM failed for {key}, using the dequant path: {FAILED_SHAPES[key]}")
        return _launch_eager(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j,
                             blocksize, blocksize2, nested, w_dtype)
    return out


# Tiles by average rows per expert. Fixed, not autotuned by default: configs that lower tl.dot to
# different MMA instructions round differently, so a timing-picked config would make the bits
# depend on the run. UNSLOTH_MOE_NF4_AUTOTUNE=1 tunes per shape instead.
# The select-tree codebook (LUT 1) beat the gather 3-6x in every tuned shape; pairs along N take
# BN 64 so a tile row stays in one 64-element absmax block (one scale per row); per-element decode
# (odd contiguous dim) takes small tiles, its int64 index tiles are register heavy.
# From 256 rows per expert two BM 64 row blocks share each decoded tile (MSUB 2): 3-7% faster on
# L4 than BM 128 / 256 there; BM 256 and MSUB 2 at BM 128 were slower everywhere.
def _fixed_config(m_hint, inner, outer, pair):
    bm = 16 if m_hint <= 16 else 32 if m_hint <= 32 else 64 if m_hint <= 64 else 128
    msub = 1
    if m_hint >= 256:
        bm, msub = 64, 2
    bn = 128 if pair == 2 else 64
    bn = min(bn, max(16, triton.next_power_of_2(outer)))
    bk = 64 if pair == 2 else 32 if pair == 0 else min(64, max(16, triton.next_power_of_2(inner)))
    warps = 8 if bm * bn >= 128 * 128 else 4
    return {"BM": bm, "BN": bn, "BK": bk, "LUT": 1, "MSUB": msub}, warps, 3


CONFIGS_USED = {}


def _run_kernel(a, out, grid, packed, absmax, code2, absmax2, offset, code, ends, M, m_hint, E,
                inner, outer, s_i, s_j, pair, blocksize, blocksize2, nested, w_dtype):
    args = (a, packed, absmax, code2, absmax2, offset, code, ends, out, M, m_hint)
    kwargs = dict(
        NUM_E = E, E_POW2 = max(16, triton.next_power_of_2(E)),
        I = inner, J = outer, S_I = s_i, S_J = s_j, PAIR = pair,
        BS = blocksize, BS2 = blocksize2, NESTED = nested, W_DT = _TL_DTYPES[w_dtype],
        enable_fp_fusion = False,
    )
    with torch.cuda.device(a.device):
        if os.environ.get("UNSLOTH_MOE_NF4_AUTOTUNE", "0") == "1":
            _nf4_grouped_kernel[grid](*args, **kwargs)
            key = ("autotune", inner, outer, m_hint, pair)
            if key not in CONFIGS_USED and getattr(_nf4_grouped_kernel, "best_config", None) is not None:
                CONFIGS_USED[key] = str(_nf4_grouped_kernel.best_config)
            return
        cfg, warps, stages = _fixed_config(m_hint, inner, outer, pair)
        CONFIGS_USED.setdefault((inner, outer, m_hint, pair), f"{cfg} warps {warps} stages {stages}")
        _nf4_grouped_kernel.fn[grid(cfg)](*args, **kwargs, **cfg, num_warps = warps, num_stages = stages)


def _decode_torch(packed, absmax, code2, absmax2, offset, code, numel, blocksize, blocksize2, nested, w_dtype):
    """The same decode in torch ops: flat [numel] stack in w_dtype."""
    q = packed.reshape(-1)
    nib = torch.stack(((q >> 4), (q & 15)), dim = -1).reshape(-1).long()
    if nested:
        scale = code2[absmax.long()] * absmax2.float().repeat_interleave(blocksize2)[: absmax.numel()] + offset.float()
    else:
        scale = absmax.float()
    scale = scale.repeat_interleave(blocksize)[:numel]
    return (code[nib][:numel] * scale).to(w_dtype)


def _launch_eager(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j,
                  blocksize, blocksize2, nested, w_dtype):
    E = ends.shape[0]
    flat = _decode_torch(packed, absmax, code2, absmax2, offset, code, E * inner * outer,
                         blocksize, blocksize2, nested, w_dtype)
    w = flat.as_strided((E, inner, outer), (inner * outer, s_i, s_j)).to(a.dtype)
    out = a.new_zeros((a.shape[0], outer))
    lo = 0
    for e, hi in enumerate(ends.tolist()):
        if hi > lo:
            out[lo:hi] = (a[lo:hi].float() @ w[e].float()).to(a.dtype)
        lo = hi
    return out


_W_DTYPE_CODE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}
_W_DTYPE_ID = {v: k for k, v in _W_DTYPE_CODE.items()}


def _strides(kn_layout, K, N):
    return (N, 1) if kn_layout else (1, K)


def _register_ops():
    if not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
        return None
    try:
        return torch.ops.unsloth_zoo.nf4_grouped_mm.default
    except (AttributeError, RuntimeError):
        pass

    @torch.library.custom_op(f"{_OP_NS}::nf4_grouped_mm", mutates_args = ())
    def nf4_grouped_mm_op(
        x: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor,
        absmax2: torch.Tensor, code2: torch.Tensor, offset: torch.Tensor, offsets: torch.Tensor,
        K: int, N: int, kn_layout: bool, blocksize: int, blocksize2: int, nested: bool, w_dtype: int,
    ) -> torch.Tensor:
        s_i, s_j = _strides(kn_layout, K, N)
        return _launch(x.contiguous(), packed, absmax, code2, absmax2, offset, code, offsets,
                       K, N, s_i, s_j, blocksize, blocksize2, nested, _W_DTYPE_CODE[w_dtype])

    @nf4_grouped_mm_op.register_fake
    def _(x, packed, absmax, code, absmax2, code2, offset, offsets, K, N, kn_layout, blocksize, blocksize2, nested, w_dtype):
        return x.new_empty((x.shape[0], N))

    @torch.library.custom_op(f"{_OP_NS}::nf4_grouped_mm_dx", mutates_args = ())
    def nf4_grouped_mm_dx_op(
        dy: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor,
        absmax2: torch.Tensor, code2: torch.Tensor, offset: torch.Tensor, offsets: torch.Tensor,
        K: int, N: int, kn_layout: bool, blocksize: int, blocksize2: int, nested: bool, w_dtype: int,
    ) -> torch.Tensor:
        # dX[m, k] = sum_n dY[m, n] W[k, n]: inner N, outer K, same packed elements.
        s_k, s_n = _strides(kn_layout, K, N)
        return _launch(dy.contiguous(), packed, absmax, code2, absmax2, offset, code, offsets,
                       N, K, s_n, s_k, blocksize, blocksize2, nested, _W_DTYPE_CODE[w_dtype])

    @nf4_grouped_mm_dx_op.register_fake
    def _(dy, packed, absmax, code, absmax2, code2, offset, offsets, K, N, kn_layout, blocksize, blocksize2, nested, w_dtype):
        return dy.new_empty((dy.shape[0], K))

    def _setup_context(ctx, inputs, output):
        x, packed, absmax, code, absmax2, code2, offset, offsets, K, N, kn_layout, bs, bs2, nested, w_dtype = inputs
        ctx.save_for_backward(packed, absmax, code, absmax2, code2, offset, offsets)
        ctx.meta = (K, N, kn_layout, bs, bs2, nested, w_dtype)

    def _backward(ctx, grad):
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.ops.unsloth_zoo.nf4_grouped_mm_dx(grad, *ctx.saved_tensors, *ctx.meta)
        return (dx,) + (None,) * 14

    nf4_grouped_mm_op.register_autograd(_backward, setup_context = _setup_context)
    return torch.ops.unsloth_zoo.nf4_grouped_mm.default


_OP = _register_ops() if triton is not None else None


def nf4_grouped_mm(x, offsets, spec):
    """x [M, K] sorted by expert, offsets int32 [E] inclusive row ends (offsets[-1] == M) -> [M, N]."""
    return _OP(
        x, spec["packed"], spec["absmax"], spec["code"], spec["absmax2"], spec["code2"], spec["offset"],
        offsets, spec["K"], spec["N"], spec["kn_layout"], spec["blocksize"], spec["blocksize2"],
        spec["nested"], spec["w_dtype"],
    )


def _unwrap(param):
    while hasattr(param, "base_layer"):
        param = param.base_layer
    return param


def _device_ok(device, mode) -> bool:
    if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return False
    if mode == "force":
        return True
    return torch.cuda.get_device_capability(device)[0] >= 8


def _layout_from_preprocess(shape, preprocess):
    """True: logical [K, N] == stored; False: logical is the stored transpose; None: anything else."""
    probe = torch.empty(shape, device = "meta")
    out = preprocess(probe)
    if out is probe:
        return True, tuple(shape[1:])
    if (
        out.dim() == 3 and out.shape == (shape[0], shape[2], shape[1])
        and out.stride() == (probe.stride(0), 1, shape[2])
    ):
        return False, (shape[2], shape[1])
    return None, None


def nf4_grouped_spec(param, preprocess, compute_dtype, mode = None):
    """Tensors and flags for nf4_grouped_mm, or None when this stack must take the dequant path.
    preprocess maps a [E, a, b] tensor to the grouped_mm [E, K, N] weight (layout probe on meta)."""
    mode = nf4_fused_mode() if mode is None else mode
    if mode == "0" or _OP is None or _DISABLED_REASON is not None:
        return None
    try:
        from bitsandbytes.nn import Params4bit
    except Exception:
        return None
    param = _unwrap(param)
    if not isinstance(param, Params4bit) or param.requires_grad:
        return None
    qs = getattr(param, "quant_state", None)
    if qs is None or getattr(qs, "quant_type", None) != "nf4" or compute_dtype not in (torch.float16, torch.bfloat16):
        return None
    if not _device_ok(param.device, mode):
        return None
    shape = getattr(param, "_original_shape", None) or qs.shape
    if shape is None or len(shape) != 3:
        return None
    shape = tuple(int(s) for s in shape)
    numel = math.prod(shape)
    bs = int(qs.blocksize)
    if bs < 2 or bs % 2 or qs.dtype not in _W_DTYPE_ID or qs.code.numel() != 16:
        return None
    data = param.data
    if not data.is_contiguous():
        return None
    packed = data.reshape(-1)
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)   # FSDP float quant storage holds the same bytes
    if packed.numel() * 2 != numel or qs.absmax.numel() != -(-numel // bs):
        return None
    kn_layout, kn = _layout_from_preprocess(shape, preprocess)
    if kn_layout is None:
        return None
    device = packed.device
    code = qs.code if qs.code.dtype == torch.float32 else qs.code.float()
    nested = bool(getattr(qs, "nested", False))
    if nested:
        s2 = qs.state2
        offset = qs.offset
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(float(offset), dtype = torch.float32, device = device)
        absmax, absmax2 = qs.absmax, s2.absmax
        code2 = s2.code if s2.code.dtype == torch.float32 else s2.code.float()
        bs2 = int(s2.blocksize)
        if absmax.dtype != torch.uint8:
            return None
    else:
        absmax = qs.absmax
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
        absmax2 = code2 = offset = absmax   # unused, valid pointers
        bs2 = 1
    tensors = (code, absmax, absmax2, code2, offset.reshape(-1))
    if any(t.device != device for t in tensors):
        return None
    return {
        "packed": packed, "absmax": absmax, "code": code, "absmax2": absmax2, "code2": code2,
        "offset": offset.reshape(-1), "K": int(kn[0]), "N": int(kn[1]), "kn_layout": kn_layout,
        "blocksize": bs, "blocksize2": bs2, "nested": nested, "w_dtype": _W_DTYPE_ID[qs.dtype],
        "num_experts": shape[0],
    }


def maybe_nf4_grouped_mm(inputs, offsets, source, preprocess):
    """Fused NF4 base GEMM for a frozen 4-bit expert stack, or None to keep the current path."""
    if nf4_fused_mode() == "auto" and not nf4_auto_fused(inputs.shape[0], offsets.shape[0]):
        return None
    try:
        spec = nf4_grouped_spec(source, preprocess, inputs.dtype)
    except Exception:
        spec = None
    if spec is None:
        if nf4_fused_mode() != "0" and not torch.compiler.is_compiling():
            NF4_FUSED_STATS["fallbacks"] += 1
        return None
    if (
        inputs.dim() != 2 or inputs.shape[1] != spec["K"] or offsets.shape[0] != spec["num_experts"]
        or inputs.device != spec["packed"].device
    ):
        return None
    if FAILED_SHAPES:
        K, N = spec["K"], spec["N"]
        s_k, s_n = _strides(spec["kn_layout"], K, N)
        if (K, N, s_k, s_n) in FAILED_SHAPES or (N, K, s_n, s_k) in FAILED_SHAPES:
            return None
    offsets = offsets if offsets.dtype == torch.int32 else offsets.to(torch.int32)
    out = nf4_grouped_mm(inputs, offsets, spec)
    if not torch.compiler.is_compiling():
        NF4_FUSED_STATS["calls"] += 1
    return out
