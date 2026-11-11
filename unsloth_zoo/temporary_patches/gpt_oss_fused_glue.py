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
"""Fused epilogues for the gpt-oss bf16 grouped QLoRA forward (gpt_oss_grouped_qlora.grouped_qlora_forward).

gate_up_epilogue: gate_up + bias[e] (+ lora * scaling) then the clamped SwiGLU, one pass; saves only the summed
gate_up. down_epilogue: fp32 down + bias[e] (+ lora * scaling). Each rounds where the eager chain does (bf16 adds and
the LoRA scale round to bf16, the SwiGLU computes in fp32 and rounds once), so they are bit-exact against the eager
(uncompiled) chain, which the per-device self-check verifies. They are not bit-exact against the shipped default:
its SwiGLU is Inductor-compiled, whose fp32 sigmoid can differ by an fp32 ulp before the bf16 rounding.
UNSLOTH_GPTOSS_FUSED_GLUE=0 keeps the eager chain. CUDA only (inline PTX rcp.rn); a failed launch or self-check
falls back to the eager chain, as do inputs outside the kernels' contract and create_graph=True backward."""

__all__ = ["fused_glue_enabled", "gate_up_epilogue", "down_epilogue", "FUSED_GLUE_CALLS"]

import os
import warnings

import torch

try:
    import triton
    import triton.language as tl
    from .moe_fused_swiglu import _clamp_grad_boundaries, _clamp_gptoss, _sigmoid
except Exception:  # pragma: no cover - no Triton, no fused glue
    triton = None

FUSED_GLUE_CALLS = {"gate_up": 0, "gate_up_bwd": 0, "down": 0, "fallback": 0}
_STATE = {"failed": False}
_CHECKED = {}


def fused_glue_enabled(t) -> bool:
    """Kill switch, device and state gate; read per call (the caller is an eager island)."""
    return (
        triton is not None
        and not _STATE["failed"]
        and os.environ.get("UNSLOTH_GPTOSS_FUSED_GLUE", "1") != "0"
        and t.is_cuda
        and torch.version.hip is None
        and type(t) is torch.Tensor
    )


if triton is not None:

    @triton.jit
    def _bf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _gu_sum(
        GU, B, D, rows, eb, col, mask, s_gu, s_d, s_b, SCALING,
        ACC_FP32: tl.constexpr, HAS_D: tl.constexpr,  # fp32 accumulate / LoRA term
    ):
        # gate_up.to(acc) + bias[e].to(acc), then + (lora * scaling).to(acc): bf16 adds round, fp32 adds do not.
        v = tl.load(GU + rows[:, None] * s_gu + col, mask = mask, other = 0.0).to(tl.float32)
        b = tl.load(B + eb[:, None] * s_b + col, mask = mask, other = 0.0).to(tl.float32)
        v = v + b
        if not ACC_FP32:
            v = _bf16(v)
        if HAS_D:
            d = _bf16(tl.load(D + rows[:, None] * s_d + col, mask = mask, other = 0.0).to(tl.float32) * SCALING)
            v = v + d
            if not ACC_FP32:
                v = _bf16(v)
        return v

    # M (routed rows) is runtime: no recompile per token count.
    @triton.jit(do_not_specialize = ["M"])
    def _gate_up_fwd_kernel(
        GU, B, EID, D, T2, OUT, M, N, s_gu, s_d, s_b, s_t2, s_o, SCALING, ALPHA, LIMIT,
        ACC_FP32: tl.constexpr, HAS_D: tl.constexpr, WRITE_T2: tl.constexpr,  # fp32 accumulate / LoRA term / save sum
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  # tile rows / tile intermediate cols
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        eb = tl.load(EID + rows, mask = rows < M, other = 0).to(tl.int64)
        g_col = (cols.to(tl.int64) * 2)[None, :]
        u_col = g_col + 1
        g = _gu_sum(GU, B, D, rows, eb, g_col, mask, s_gu, s_d, s_b, SCALING, ACC_FP32, HAS_D)
        u = _gu_sum(GU, B, D, rows, eb, u_col, mask, s_gu, s_d, s_b, SCALING, ACC_FP32, HAS_D)
        if WRITE_T2:
            DT2: tl.constexpr = T2.dtype.element_ty
            tl.store(T2 + rows[:, None] * s_t2 + g_col, g.to(DT2), mask = mask)
            tl.store(T2 + rows[:, None] * s_t2 + u_col, u.to(DT2), mask = mask)
        # swiglu_torch_forward in fp32, rounded once to the output dtype.
        gc, uc = _clamp_gptoss(g, u, LIMIT, False, tl.float32)
        sg = _sigmoid(gc * ALPHA)
        y = (uc + 1.0) * (gc * sg)
        tl.store(OUT + rows[:, None] * s_o + cols[None, :], y.to(OUT.dtype.element_ty), mask = mask)

    @triton.jit(do_not_specialize = ["M"])
    def _gate_up_bwd_kernel(
        DY, T2, DGU, DD, M, N, s_dy, s_t2, s_g, s_d, SCALING, ALPHA, LIMIT,
        HAS_D: tl.constexpr, ACC_FP32: tl.constexpr,  # LoRA term / fp32 accumulate
        G_MAX_INC: tl.constexpr, U_MAX_INC: tl.constexpr, U_MIN_INC: tl.constexpr,  # clamp bounds inclusive
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  # tile rows / tile intermediate cols
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        g_col = (cols.to(tl.int64) * 2)[None, :]
        u_col = g_col + 1
        g = tl.load(T2 + rows[:, None] * s_t2 + g_col, mask = mask, other = 0.0).to(tl.float32)
        u = tl.load(T2 + rows[:, None] * s_t2 + u_col, mask = mask, other = 0.0).to(tl.float32)
        dy = tl.load(DY + rows[:, None] * s_dy + cols[None, :], mask = mask, other = 0.0).to(tl.float32)
        # Autograd of the fp32 chain (moe_fused_swiglu._swiglu_bwd_kernel with ROUND off).
        gc, uc = _clamp_gptoss(g, u, LIMIT, False, tl.float32)
        sg = _sigmoid(gc * ALPHA)
        glu = gc * sg
        u1 = uc + 1.0
        d_u1 = dy * glu
        d_glu = dy * u1
        d_gc = d_glu * sg + ((d_glu * gc) * (1.0 - sg)) * sg * ALPHA
        if G_MAX_INC:
            g_in = g <= LIMIT
        else:
            g_in = g < LIMIT
        if U_MAX_INC:
            u_in = u <= LIMIT
        else:
            u_in = u < LIMIT
        if U_MIN_INC:
            u_in = u_in & (u >= -LIMIT)
        else:
            u_in = u_in & (u > -LIMIT)
        # Eager rounds to the accumulate dtype (the .to(float32) backward), sums two slice_backward buffers in it (-0
        # comes out +0), then rounds to bf16 (the GEMM output's cast back).
        d_g = tl.where(g_in, d_gc, 0.0)
        d_u = tl.where(u_in, d_u1, 0.0)
        if ACC_FP32:
            d_g = _bf16(tl.where(d_g == 0, 0.0, d_g))
            d_u = _bf16(tl.where(d_u == 0, 0.0, d_u))
        else:
            d_g = _bf16(d_g)
            d_u = _bf16(d_u)
            d_g = tl.where(d_g == 0, 0.0, d_g)
            d_u = tl.where(d_u == 0, 0.0, d_u)
        tl.store(DGU + rows[:, None] * s_g + g_col, d_g.to(DGU.dtype.element_ty), mask = mask)
        tl.store(DGU + rows[:, None] * s_g + u_col, d_u.to(DGU.dtype.element_ty), mask = mask)
        if HAS_D:
            # (lora * scaling) backward: the bf16 grad times scaling, rounded to bf16.
            tl.store(DD + rows[:, None] * s_d + g_col, (d_g * SCALING).to(DD.dtype.element_ty), mask = mask)
            tl.store(DD + rows[:, None] * s_d + u_col, (d_u * SCALING).to(DD.dtype.element_ty), mask = mask)

    @triton.jit(do_not_specialize = ["M"])
    def _down_fwd_kernel(
        Y, B, EID, D, OUT, M, H, s_y, s_d, s_b, s_o, SCALING,
        HAS_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  # LoRA term / tile rows / tile cols
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < H)
        eb = tl.load(EID + rows, mask = rows < M, other = 0).to(tl.int64)
        y = tl.load(Y + rows[:, None] * s_y + cols[None, :], mask = mask, other = 0.0).to(tl.float32)
        y = y + tl.load(B + eb[:, None] * s_b + cols[None, :], mask = mask, other = 0.0).to(tl.float32)
        if HAS_D:
            y = y + _bf16(tl.load(D + rows[:, None] * s_d + cols[None, :], mask = mask, other = 0.0).to(tl.float32) * SCALING)
        tl.store(OUT + rows[:, None] * s_o + cols[None, :], y, mask = mask)


def _grid(M, N, block_n = 256):
    block_m = max(1, 2048 // block_n)
    return (triton.cdiv(M, block_m), triton.cdiv(N, block_n)), block_m, block_n


def _launch_gate_up(gu, bias, eid, d, scaling, alpha, limit, acc_dtype, write_t2):
    M, two_i = gu.shape
    N = two_i // 2
    out = torch.empty((M, N), dtype = torch.bfloat16, device = gu.device)
    t2 = torch.empty((M, two_i), dtype = acc_dtype, device = gu.device) if write_t2 else None
    if M == 0:
        return out, t2
    grid, bm, bn = _grid(M, N)
    dd = d if d is not None else gu
    with torch.cuda.device(gu.device):
        _gate_up_fwd_kernel[grid](
            gu, bias, eid, dd, t2 if t2 is not None else gu, out, M, N,
            gu.stride(0), dd.stride(0), bias.stride(0), t2.stride(0) if t2 is not None else 0, out.stride(0),
            float(scaling), float(alpha), float(limit),
            ACC_FP32 = acc_dtype == torch.float32, HAS_D = d is not None, WRITE_T2 = bool(write_t2),
            BLOCK_M = bm, BLOCK_N = bn, num_warps = 4, enable_fp_fusion = False,
        )
    return out, t2


def _launch_gate_up_bwd(dy, t2, scaling, alpha, limit, has_d):
    M, two_i = t2.shape
    N = two_i // 2
    dgu = torch.empty((M, two_i), dtype = torch.bfloat16, device = t2.device)
    dd = torch.empty((M, two_i), dtype = torch.bfloat16, device = t2.device) if has_d else None
    if M == 0:
        return dgu, dd
    dy = dy.contiguous()
    g_max, u_max, u_min = _clamp_grad_boundaries()
    grid, bm, bn = _grid(M, N)
    with torch.cuda.device(t2.device):
        _gate_up_bwd_kernel[grid](
            dy, t2, dgu, dd if dd is not None else dgu, M, N,
            dy.stride(0), t2.stride(0), dgu.stride(0), (dd if dd is not None else dgu).stride(0),
            float(scaling), float(alpha), float(limit),
            HAS_D = bool(has_d), ACC_FP32 = t2.dtype == torch.float32,
            G_MAX_INC = g_max, U_MAX_INC = u_max, U_MIN_INC = u_min,
            BLOCK_M = bm, BLOCK_N = bn, num_warps = 4, enable_fp_fusion = False,
        )
    return dgu, dd


def _swiglu_eager(x, alpha, limit):
    a = x[..., ::2].to(torch.float32).clamp(max = limit)
    b = x[..., 1::2].to(torch.float32).clamp(min = -limit, max = limit)
    return ((a * torch.sigmoid(alpha * a)) * (b + 1)).to(torch.bfloat16)


class _GateUpEpilogue(torch.autograd.Function):
    """Returns (gated, gate_up sum). The sum is a differentiable output saved for backward, so a create_graph=True
    backward replays the eager SwiGLU on it and its second derivative reaches gu / d through this node."""

    @staticmethod
    def forward(ctx, gu, d, bias, eid, scaling, alpha, limit, acc_dtype):
        out, t2 = _launch_gate_up(gu, bias, eid, d, scaling, alpha, limit, acc_dtype, True)
        ctx.save_for_backward(t2)
        ctx.cfg = (scaling, alpha, limit, d is not None)
        ctx.set_materialize_grads(False)
        FUSED_GLUE_CALLS["gate_up"] += 1
        return out, t2

    @staticmethod
    def backward(ctx, dy, dt2):
        (t2,) = ctx.saved_tensors
        scaling, alpha, limit, has_d = ctx.cfg
        create = torch.is_grad_enabled()
        if dy is not None and dt2 is None and not create and not _STATE["failed"]:
            try:
                dgu, dd = _launch_gate_up_bwd(dy, t2, scaling, alpha, limit, has_d)
                FUSED_GLUE_CALLS["gate_up_bwd"] += 1
                return dgu, dd, None, None, None, None, None, None
            except Exception as err:
                if not _failed(err):
                    raise
        # Eager autograd of the same chain: the fallback, and the differentiable replay under create_graph=True.
        dx = None
        if dy is not None:
            with torch.enable_grad():
                x = t2 if create and t2.requires_grad else t2.detach().requires_grad_()
                (dx,) = torch.autograd.grad(_swiglu_eager(x, alpha, limit), x, dy, create_graph = create)
        if dt2 is not None:
            dx = dt2 if dx is None else dx + dt2
        if dx is None:
            return None, None, None, None, None, None, None, None
        # gu.to(acc) and (d * scaling).to(acc) backward: round to bf16, then the bf16 grad times scaling. Separate casts,
        # as eager has, so a double backward sums the two branches in the accumulate dtype.
        dd = dx.to(torch.bfloat16) * scaling if has_d else None
        return dx.to(torch.bfloat16), dd, None, None, None, None, None, None


class _DownEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, d, bias, eid, scaling):
        M, H = y.shape
        out = torch.empty((M, H), dtype = torch.float32, device = y.device)
        if M:
            grid, bm, bn = _grid(M, H)
            dd = d if d is not None else y
            with torch.cuda.device(y.device):
                _down_fwd_kernel[grid](
                    y, bias, eid, dd, out, M, H, y.stride(0), dd.stride(0), bias.stride(0), out.stride(0), float(scaling),
                    HAS_D = d is not None, BLOCK_M = bm, BLOCK_N = bn, num_warps = 4, enable_fp_fusion = False,
                )
        ctx.scaling = scaling
        ctx.dtypes = (y.dtype, None if d is None else d.dtype)
        FUSED_GLUE_CALLS["down"] += 1
        return out

    @staticmethod
    def backward(ctx, g):
        # The .float() casts' backward rounds to the GEMM dtype; (lora * scaling) multiplies that bf16 grad.
        gy = g.to(ctx.dtypes[0])
        gd = None if ctx.dtypes[1] is None else gy.to(ctx.dtypes[1]) * ctx.scaling
        return gy, gd, None, None, None


def _eager_gate_up(gu, d, bias, eid, scaling, alpha, limit, acc_dtype):
    """The chain grouped_qlora_forward shipped (with the SwiGLU uncompiled): the fused kernels' reference."""
    x = gu.to(acc_dtype) + bias[eid].to(acc_dtype)
    if d is not None:
        x = x + (d * scaling).to(acc_dtype)
    return _swiglu_eager(x, alpha, limit)


def _eager_down(y, d, bias, eid, scaling):
    out = y.float() + bias[eid].float()
    if d is not None:
        out = out + (d * scaling).float()
    return out


def _bits(a, b):
    iv = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def _self_check(device, acc_dtype, alpha, limit) -> bool:
    """Cached bitwise probe of both epilogues, forward and backward, against the eager chain on this device."""
    key = (str(device), acc_dtype, float(alpha), float(limit))
    hit = _CHECKED.get(key)
    if hit is not None:
        return hit
    # Temporary contexts decline without caching: the next eager call probes again.
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return False
    try:
        # Identity hooks keep the probe out of an enclosing checkpoint frame (innermost wins).
        hooks = torch.autograd.graph.saved_tensors_hooks(_identity, _identity)
        hooks.__enter__()
    except Exception:
        return False  # hooks disabled here (e.g. a functorch transform)
    ok = False
    calls = dict(FUSED_GLUE_CALLS)
    try:
        with torch.inference_mode(False), torch.enable_grad(), torch.cuda.device(device):
            gen = torch.Generator(device = device).manual_seed(1234)
            E, M, N, H = 3, 37, 96, 64
            eid = torch.randint(0, E, (M,), device = device, generator = gen).sort().values
            gu = (torch.randn(M, 2 * N, device = device, generator = gen) * 4).to(torch.bfloat16)
            special = torch.tensor([limit, -limit, 0.0, -0.0, 1.0, -1.0, 2 * limit, -2 * limit], device = device)
            gu[0, : special.numel()] = special.to(torch.bfloat16)
            gu[1, -special.numel():] = special.to(torch.bfloat16)
            d = torch.randn(M, 2 * N, device = device, generator = gen).to(torch.bfloat16)
            bias = (torch.randn(E, 2 * N, device = device, generator = gen) * 0.5).to(torch.bfloat16)
            dy = torch.randn(M, N, device = device, generator = gen).to(torch.bfloat16)
            leaves = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
            ref = _eager_gate_up(leaves[0], leaves[1], bias, eid, 0.5, alpha, limit, acc_dtype)
            r_grads = torch.autograd.grad(ref, leaves, dy)
            mine = [gu.clone().requires_grad_(), d.clone().requires_grad_()]
            out = _GateUpEpilogue.apply(mine[0], mine[1], bias, eid, 0.5, alpha, limit, acc_dtype)[0]
            m_grads = torch.autograd.grad(out, mine, dy)
            ok = _bits(out, ref) and all(_bits(a, b) for a, b in zip(m_grads, r_grads))
            y = torch.randn(M, H, device = device, generator = gen).to(torch.bfloat16).requires_grad_()
            d2 = torch.randn(M, H, device = device, generator = gen).to(torch.bfloat16).requires_grad_()
            b2 = torch.randn(E, H, device = device, generator = gen).to(torch.bfloat16)
            g2 = torch.randn(M, H, device = device, generator = gen)
            ref2 = _eager_down(y, d2, b2, eid, 0.5)
            mine2 = _DownEpilogue.apply(y, d2, b2, eid, 0.5)
            ok = ok and _bits(mine2, ref2) and all(
                _bits(a, b) for a, b in zip(torch.autograd.grad(mine2, (y, d2), g2), torch.autograd.grad(ref2, (y, d2), g2)))
        ok = ok and not _STATE["failed"]  # a backward launch that failed fell back to eager inside the probe
        if not ok:
            warnings.warn("Unsloth: fused gpt-oss MoE glue does not reproduce this torch's eager bits; keeping the eager chain.")
    except Exception as err:
        if _passthrough(err):
            raise
        ok = False
        warnings.warn(f"Unsloth: fused gpt-oss MoE glue self-check failed ({type(err).__name__}: {err}); "
                      "keeping the eager chain.")
    finally:
        hooks.__exit__(None, None, None)
        FUSED_GLUE_CALLS.update(calls)  # the probe's own launches are not real calls
    _CHECKED[key] = ok
    return ok


def _identity(t):
    return t


def _passthrough(err):
    """OOM, Dynamo and checkpoint control flow (_StopRecomputationError, CheckpointError) are not kernel faults:
    re-raise them; swallowing one disables the kernels process-wide or trips checkpoint's early-stop assertion."""
    from .moe_triton_kernels import _autograd_control_flow_errors
    return (
        isinstance(err, (torch.cuda.OutOfMemoryError,) + _autograd_control_flow_errors())
        or type(err).__module__.startswith("torch._dynamo")
    )


def _failed(err):
    if _passthrough(err):
        return False
    if not _STATE["failed"]:
        _STATE["failed"] = True
        warnings.warn(f"Unsloth: fused gpt-oss MoE glue failed ({type(err).__name__}: {err}); using the eager chain.")
    return True


def usable(t, acc_dtype, alpha, limit) -> bool:
    return (acc_dtype in (torch.bfloat16, torch.float32) and fused_glue_enabled(t)
            and _self_check(t.device, acc_dtype, alpha, limit))


def _row_major(t):
    # The kernels index rows by stride(0) and assume unit column / element stride.
    return t if t.stride(-1) == 1 else t.contiguous()


def _contract(x, d, bias, eid, width):
    """Normalized (x, d, bias, eid) when the kernels apply, else None (the eager chain handles the rest)."""
    tensors = [x, bias, eid] + ([d] if d is not None else [])
    if any(type(t) is not torch.Tensor or t.device != x.device for t in tensors):
        return None
    M = x.shape[0] if x.dim() == 2 else -1
    if (
        x.dim() != 2 or x.dtype is not torch.bfloat16 or x.shape[1] != width
        or (d is not None and (d.dtype is not torch.bfloat16 or d.shape != x.shape))
        or bias.dim() != 2 or bias.shape[1] != width or bias.dtype not in (torch.bfloat16, torch.float32)
        or eid.dim() != 1 or eid.shape[0] != M or eid.dtype.is_floating_point or eid.dtype is torch.bool
        # A trainable bias would get no gradient from the kernels.
        or (bias.requires_grad and torch.is_grad_enabled())
    ):
        return None
    return _row_major(x), (None if d is None else _row_major(d)), _row_major(bias), eid.contiguous()


def gate_up_epilogue(gu, d, bias, eid, scaling, alpha, limit, acc_dtype):
    """bf16 gated [M, I] = clamped SwiGLU of gu.to(acc) + bias[eid] (+ (d * scaling).to(acc)); None: use eager."""
    width = gu.shape[-1] if gu.dim() == 2 and gu.shape[-1] % 2 == 0 else -1  # odd widths: eager
    # The kernel adds the bias unrounded: eager's bias.to(acc) only matches for a bf16 bias or fp32 accumulate.
    ok_acc = acc_dtype is torch.float32 or (acc_dtype is torch.bfloat16 and bias.dtype is torch.bfloat16)
    args = _contract(gu, d, bias, eid, width) if ok_acc else None
    if args is None:
        FUSED_GLUE_CALLS["fallback"] += 1
        return None
    gu, d, bias, eid = args
    try:
        if torch.is_grad_enabled() and (gu.requires_grad or (d is not None and d.requires_grad)):
            return _GateUpEpilogue.apply(gu, d, bias, eid, scaling, alpha, limit, acc_dtype)[0]
        out, _ = _launch_gate_up(gu, bias, eid, d, scaling, alpha, limit, acc_dtype, False)
        FUSED_GLUE_CALLS["gate_up"] += 1
        return out
    except Exception as err:
        if not _failed(err):
            raise
    FUSED_GLUE_CALLS["fallback"] += 1
    return None


def down_epilogue(y, d, bias, eid, scaling):
    """fp32 [M, H] = y.float() + bias[eid].float() (+ (d * scaling).float()); None: use eager."""
    args = _contract(y, d, bias, eid, y.shape[-1] if y.dim() == 2 else -1)
    if args is None:
        FUSED_GLUE_CALLS["fallback"] += 1
        return None
    y, d, bias, eid = args
    try:
        return _DownEpilogue.apply(y, d, bias, eid, scaling)
    except Exception as err:
        if not _failed(err):
            raise
    FUSED_GLUE_CALLS["fallback"] += 1
    return None
