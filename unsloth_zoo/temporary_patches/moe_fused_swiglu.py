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
"""Fused MoE SwiGLU / gpt-oss clamped SwiGLU over gate_up that saves only gate_up for backward.

eager_rounding=True replays the eager ATen op chain (bit-identical fwd + bwd); False rounds once.
UNSLOTH_MOE_FUSED_SWIGLU: 0 off, 1 eager frames only (default), 2 also compiled frames. CUDA only
(no ROCm: div.rn / fma pinned through PTX / libdevice); a failed launch falls back to the eager chain."""

__all__ = [
    "ACT_SILU",
    "ACT_GPTOSS",
    "moe_fused_swiglu",
    "moe_swiglu_reference",
    "moe_swiglu_backward_reference",
    "fused_swiglu_enabled",
    "refresh_fused_swiglu_enabled",
    "fused_swiglu_supported",
    "fused_swiglu_self_check",
    "FUSED_SWIGLU_CALLS",
]

import functools
import os
import warnings

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - no Triton / libdevice, no fused kernel
    triton = None

ACT_SILU = 0
ACT_GPTOSS = 1

# [forward, backward] launches, for engagement tests.
FUSED_SWIGLU_CALLS = [0, 0]
_STATE = {"failed": False, "enabled": False, "compiled": False}
# Eager-only cache: compiled frames never read it, so new keys cannot invalidate their guards.
_CACHE = {}
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def refresh_fused_swiglu_enabled() -> bool:
    """Re-read UNSLOTH_MOE_FUSED_SWIGLU into _STATE so compiled frames never trace os.environ."""
    mode = os.environ.get("UNSLOTH_MOE_FUSED_SWIGLU", "1").strip().lower()
    _STATE["enabled"] = triton is not None and mode not in ("0", "false", "off")
    _STATE["compiled"] = _STATE["enabled"] and mode in ("2", "compile", "compiled", "always")
    return _STATE["enabled"]


def fused_swiglu_enabled() -> bool:
    return _STATE["enabled"]


refresh_fused_swiglu_enabled()


def _bf16_capable(gate_up) -> bool:
    # sm75 has no bf16 conversion instructions.
    if torch.compiler.is_compiling():
        return True
    # Per operand device: in a mixed system an older visible card must not disable fusion on a capable one.
    index = gate_up.device.index
    if index is None:
        index = torch.cuda.current_device()
    key = ("bf16_ok", index)
    ok = _CACHE.get(key)
    if ok is None:
        ok = _CACHE[key] = torch.cuda.get_device_capability(index)[0] >= 8
    return ok


def fused_swiglu_supported(gate_up: torch.Tensor) -> bool:
    return (
        _STATE["enabled"]
        and not _STATE["failed"]
        and (_STATE["compiled"] or not torch.compiler.is_compiling())
        and gate_up.is_cuda
        and (torch.compiler.is_compiling() or type(gate_up) is torch.Tensor)  # no DTensor / subclasses
        and torch.version.hip is None
        and gate_up.dtype in _DTYPES
        and (gate_up.dtype != torch.bfloat16 or _bf16_capable(gate_up))
        and gate_up.dim() >= 1
        and gate_up.shape[-1] % 2 == 0
    )


@functools.lru_cache(maxsize = 1)
def _clamp_grad_boundaries():
    """clamp's gradient at x == bound passes up to torch 2.13 and is zeroed from 2.14: probe it."""
    try:
        with torch.inference_mode(False), torch.enable_grad():
            a = torch.tensor([2.0], requires_grad = True)
            a.clamp(min = None, max = 2.0).sum().backward()
            b = torch.tensor([2.0, -2.0], requires_grad = True)
            b.clamp(min = -2.0, max = 2.0).sum().backward()
        return bool(a.grad[0] != 0), bool(b.grad[0] != 0), bool(b.grad[1] != 0)
    except Exception:
        return True, True, True


_clamp_grad_boundaries()  # probe at import, outside any compile / capture / fake mode


def _le(x, bound, inclusive):
    return x <= bound if inclusive else x < bound


def _split(gate_up, interleaved):
    if interleaved:
        return gate_up[..., 0::2], gate_up[..., 1::2]
    return gate_up.chunk(2, dim = -1)


def moe_swiglu_reference(gate_up, act = ACT_SILU, interleaved = False, alpha = 1.702, limit = 7.0, eager_rounding = True):
    dtype = gate_up.dtype
    if not eager_rounding:
        gate_up = gate_up.float()
    gate, up = _split(gate_up, interleaved)
    if act == ACT_GPTOSS:
        gate = gate.clamp(min = None, max = limit)
        up = up.clamp(min = -limit, max = limit)
        glu = gate * torch.sigmoid(gate * alpha)
        out = (up + 1.0) * glu
    else:
        out = F.silu(gate) * up
    return out.to(dtype)


def moe_swiglu_backward_reference(grad, gate_up, act = ACT_SILU, interleaved = False, alpha = 1.702, limit = 7.0, eager_rounding = True):
    dtype = gate_up.dtype
    if not eager_rounding:
        gate_up, grad = gate_up.float(), grad.float()
    gate, up = _split(gate_up, interleaved)
    if act == ACT_GPTOSS:
        gc = gate.clamp(min = None, max = limit)
        uc = up.clamp(min = -limit, max = limit)
        t = gc * alpha
        sg = torch.sigmoid(t)
        glu = gc * sg
        u1 = uc + 1.0
        d_u1 = grad * glu
        d_glu = grad * u1
        d_gc = d_glu * sg
        d_sg = d_glu * gc
        d_t = torch.ops.aten.sigmoid_backward(d_sg, sg)
        d_gc = d_gc + d_t * alpha
        zero = torch.zeros((), dtype = d_gc.dtype, device = d_gc.device)
        g_max, u_max, u_min = _clamp_grad_boundaries()
        d_gate = torch.where(_le(gate, limit, g_max), d_gc, zero)
        d_up = torch.where(_le(-limit, up, u_min) & _le(up, limit, u_max), d_u1, zero)
    else:
        s = F.silu(gate)
        d_up = grad * s
        if torch.is_grad_enabled() and (gate_up.requires_grad or grad.requires_grad):
            # create_graph=True: aten.silu_backward has no derivative, spell it out.
            sig = torch.sigmoid(gate)
            d_gate = (grad * up) * sig * (1.0 + gate * (1.0 - sig))
        else:
            d_gate = torch.ops.aten.silu_backward(grad * up, gate)
    if interleaved:
        # Eager sums two slice_backward buffers (zeros + slice): -0 + +0 = +0.
        d = torch.stack((d_gate, d_up), dim = -1).flatten(-2)
        d = d + torch.zeros((), dtype = d.dtype, device = d.device)
    else:
        d = torch.cat((d_gate, d_up), dim = -1)
    return d.to(dtype)


if triton is not None:

    @triton.jit
    def _rnd(x, ROUND: tl.constexpr, DT: tl.constexpr):
        if ROUND:
            return x.to(DT).to(tl.float32)
        return x

    @triton.jit
    def _load_gate_up(GU, rows, cols, mask, s_m, s_c, N, INTERLEAVED: tl.constexpr):
        base = GU + rows[:, None] * s_m
        if INTERLEAVED:
            g_col = cols.to(tl.int64) * 2
            u_col = g_col + 1
        else:
            g_col = cols.to(tl.int64)
            u_col = g_col + N
        g = tl.load(base + g_col[None, :] * s_c, mask = mask, other = 0.0).to(tl.float32)
        u = tl.load(base + u_col[None, :] * s_c, mask = mask, other = 0.0).to(tl.float32)
        return g, u, g_col, u_col

    @triton.jit
    def _clamp_gptoss(g, u, LIMIT, ROUND: tl.constexpr, DT: tl.constexpr):
        gc = _rnd(tl.where(g > LIMIT, LIMIT, g), ROUND, DT)
        uc = _rnd(tl.where(u > LIMIT, LIMIT, tl.where(u < -LIMIT, -LIMIT, u)), ROUND, DT)
        return gc, uc

    @triton.jit
    def _div_rn(a, b):
        # Not libdevice.div_rn: it links as div.rn.ftz.f32 and flushes subnormals ATen keeps.
        return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype = tl.float32, is_pure = True, pack = 1)

    @triton.jit
    def _rcp_rn(a):
        return tl.inline_asm_elementwise("rcp.rn.f32 $0, $1;", "=f,f", [a], dtype = tl.float32, is_pure = True, pack = 1)

    @triton.jit
    def _sigmoid(x):
        return _rcp_rn(1.0 + libdevice.exp(-x))

    @triton.jit
    def _swiglu_fwd_kernel(
        GU, OUT, M, N, s_gm, s_gc, s_om, ALPHA, LIMIT,
        ACT: tl.constexpr, INTERLEAVED: tl.constexpr, ROUND: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        DT: tl.constexpr = OUT.dtype.element_ty
        g, u, g_col, u_col = _load_gate_up(GU, rows, cols, mask, s_gm, s_gc, N, INTERLEAVED)
        if ACT == 1:
            gc, uc = _clamp_gptoss(g, u, LIMIT, ROUND, DT)
            t = _rnd(gc * ALPHA, ROUND, DT)
            sg = _rnd(_sigmoid(t), ROUND, DT)
            glu = _rnd(gc * sg, ROUND, DT)
            u1 = _rnd(uc + 1.0, ROUND, DT)
            y = u1 * glu
        else:
            s = _rnd(_div_rn(g, 1.0 + libdevice.exp(-g)), ROUND, DT)
            y = s * u
        tl.store(OUT + rows[:, None] * s_om + cols[None, :], y.to(DT), mask = mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        DY, GU, DGU, M, N, s_dm, s_dc, s_gm, s_gc, s_om, ALPHA, LIMIT,
        ACT: tl.constexpr, INTERLEAVED: tl.constexpr, ROUND: tl.constexpr,
        G_MAX_INC: tl.constexpr, U_MAX_INC: tl.constexpr, U_MIN_INC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        DT: tl.constexpr = DGU.dtype.element_ty
        g, u, g_col, u_col = _load_gate_up(GU, rows, cols, mask, s_gm, s_gc, N, INTERLEAVED)
        dy = tl.load(DY + rows[:, None] * s_dm + cols[None, :].to(tl.int64) * s_dc, mask = mask, other = 0.0).to(tl.float32)
        if ACT == 1:
            gc, uc = _clamp_gptoss(g, u, LIMIT, ROUND, DT)
            t = _rnd(gc * ALPHA, ROUND, DT)
            sg = _rnd(_sigmoid(t), ROUND, DT)
            glu = _rnd(gc * sg, ROUND, DT)
            u1 = _rnd(uc + 1.0, ROUND, DT)
            d_u1 = _rnd(dy * glu, ROUND, DT)
            d_glu = _rnd(dy * u1, ROUND, DT)
            d_gc_a = _rnd(d_glu * sg, ROUND, DT)
            d_sg = _rnd(d_glu * gc, ROUND, DT)
            d_t = _rnd(_rnd(d_sg * _rnd(1.0 - sg, ROUND, DT), ROUND, DT) * sg, ROUND, DT)
            d_gc_b = _rnd(d_t * ALPHA, ROUND, DT)
            d_gc = d_gc_a + d_gc_b
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
            d_gate = tl.where(g_in, d_gc, 0.0)
            d_up = tl.where(u_in, d_u1, 0.0)
        else:
            sig = _sigmoid(g)
            s = _rnd(_div_rn(g, 1.0 + libdevice.exp(-g)), ROUND, DT)
            d_up = dy * s
            d_s = _rnd(dy * u, ROUND, DT)
            # nvcc contracts silu_backward's inner mul-add to an fma.
            d_gate = (d_s * sig) * libdevice.fma(g, 1.0 - sig, 1.0)
        d_gate = d_gate.to(DT)
        d_up = d_up.to(DT)
        if INTERLEAVED:
            # Eager sums two slice_backward buffers: -0 comes out +0.
            d_gate = tl.where(d_gate == 0, 0.0, d_gate).to(DT)
            d_up = tl.where(d_up == 0, 0.0, d_up).to(DT)
        out_row = DGU + rows[:, None] * s_om
        tl.store(out_row + g_col[None, :], d_gate, mask = mask)
        tl.store(out_row + u_col[None, :], d_up, mask = mask)


def _blocks(N):
    # Least-padding column tile (>= 128, must save > N/8), ~2048 elements per program.
    N = max(int(N), 1)
    bn = min(1024, triton.next_power_of_2(N))
    best_bn, best_waste = bn, triton.cdiv(N, bn) * bn - N
    while bn > 128:
        bn //= 2
        waste = triton.cdiv(N, bn) * bn - N
        if waste < best_waste - N // 8:
            best_bn, best_waste = bn, waste
    return max(1, 2048 // best_bn), best_bn


def _as_2d(t, width):
    if t.dim() == 2:
        return t
    return t.reshape(-1, width)


def _launch_fwd(gate_up, act, interleaved, alpha, limit, eager_rounding):
    two_i = gate_up.shape[-1]
    N = two_i // 2
    out = torch.empty((*gate_up.shape[:-1], N), dtype = gate_up.dtype, device = gate_up.device)
    gu = _as_2d(gate_up, two_i)
    M = gu.shape[0]
    if M == 0 or N == 0:
        return out
    o2 = out.view(M, N)
    block_m, block_n = _blocks(N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    with torch.cuda.device(gate_up.device):
        _swiglu_fwd_kernel[grid](
            gu, o2, M, N, gu.stride(0), gu.stride(1), o2.stride(0), float(alpha), float(limit),
            ACT = int(act), INTERLEAVED = bool(interleaved), ROUND = bool(eager_rounding) and gate_up.dtype != torch.float32,
            BLOCK_M = block_m, BLOCK_N = block_n, num_warps = 4, enable_fp_fusion = False,
        )
    return out


def _launch_bwd(grad, gate_up, act, interleaved, alpha, limit, eager_rounding):
    two_i = gate_up.shape[-1]
    N = two_i // 2
    d_gate_up = torch.empty(gate_up.shape, dtype = gate_up.dtype, device = gate_up.device)
    gu = _as_2d(gate_up, two_i)
    M = gu.shape[0]
    if M == 0 or N == 0:
        return d_gate_up
    if grad.dtype != gate_up.dtype:
        grad = grad.to(gate_up.dtype)
    dy = _as_2d(grad, N)
    dgu = d_gate_up.view(M, two_i)
    g_max, u_max, u_min = _clamp_grad_boundaries()
    block_m, block_n = _blocks(N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    with torch.cuda.device(gate_up.device):
        _swiglu_bwd_kernel[grid](
            dy, gu, dgu, M, N, dy.stride(0), dy.stride(1), gu.stride(0), gu.stride(1), dgu.stride(0),
            float(alpha), float(limit),
            ACT = int(act), INTERLEAVED = bool(interleaved), ROUND = bool(eager_rounding) and gate_up.dtype != torch.float32,
            G_MAX_INC = g_max, U_MAX_INC = u_max, U_MIN_INC = u_min,
            BLOCK_M = block_m, BLOCK_N = block_n, num_warps = 4, enable_fp_fusion = False,
        )
    return d_gate_up


def _launch_failed(err):
    # Never swallow OOM / Dynamo errors: that would disable the kernel process-wide.
    if isinstance(err, torch.cuda.OutOfMemoryError) or type(err).__module__.startswith("torch._dynamo"):
        return False
    if not _STATE["failed"]:
        _STATE["failed"] = True
        warnings.warn(f"Unsloth: fused MoE SwiGLU kernel failed ({type(err).__name__}: {err}); using the eager activation.")
    return True


def _fwd_impl(gate_up, act, interleaved, alpha, limit, eager_rounding):
    if not _STATE["failed"]:
        try:
            out = _launch_fwd(gate_up, act, interleaved, alpha, limit, eager_rounding)
            FUSED_SWIGLU_CALLS[0] += 1
            return out
        except Exception as err:
            if not _launch_failed(err):
                raise
    with torch.no_grad():
        return moe_swiglu_reference(gate_up, act, interleaved, alpha, limit, eager_rounding).contiguous()


def _bwd_impl(grad, gate_up, act, interleaved, alpha, limit, eager_rounding):
    if not _STATE["failed"]:
        try:
            out = _launch_bwd(grad, gate_up, act, interleaved, alpha, limit, eager_rounding)
            FUSED_SWIGLU_CALLS[1] += 1
            return out
        except Exception as err:
            if not _launch_failed(err):
                raise
    with torch.no_grad():
        return moe_swiglu_backward_reference(grad, gate_up, act, interleaved, alpha, limit, eager_rounding).contiguous()


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, act, interleaved, alpha, limit, eager_rounding):
        ctx.cfg = (act, interleaved, alpha, limit, eager_rounding)
        ctx.save_for_backward(gate_up)
        return _fwd_impl(gate_up, act, interleaved, alpha, limit, eager_rounding)

    @staticmethod
    def backward(ctx, grad):
        (gate_up,) = ctx.saved_tensors
        if torch.is_grad_enabled():  # create_graph=True: differentiable replay
            d = moe_swiglu_backward_reference(grad, gate_up, *ctx.cfg)
        else:
            d = _bwd_impl(grad, gate_up, *ctx.cfg)
        return d, None, None, None, None, None


def _register_ops():
    if not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
        return None
    try:
        return torch.ops.unsloth_zoo.moe_fused_swiglu.default
    except (AttributeError, RuntimeError):
        pass
    try:
        @torch.library.custom_op("unsloth_zoo::moe_fused_swiglu", mutates_args = ())
        def moe_fused_swiglu_op(
            gate_up: torch.Tensor, act: int, interleaved: bool, alpha: float, limit: float, eager_rounding: bool,
        ) -> torch.Tensor:
            return _fwd_impl(gate_up, act, interleaved, alpha, limit, eager_rounding)

        @moe_fused_swiglu_op.register_fake
        def _moe_fused_swiglu_fake(gate_up, act, interleaved, alpha, limit, eager_rounding):
            return gate_up.new_empty((*gate_up.shape[:-1], gate_up.shape[-1] // 2))

        @torch.library.custom_op("unsloth_zoo::moe_fused_swiglu_bwd", mutates_args = ())
        def moe_fused_swiglu_bwd_op(
            grad: torch.Tensor, gate_up: torch.Tensor, act: int, interleaved: bool, alpha: float, limit: float,
            eager_rounding: bool,
        ) -> torch.Tensor:
            return _bwd_impl(grad, gate_up, act, interleaved, alpha, limit, eager_rounding)

        @moe_fused_swiglu_bwd_op.register_fake
        def _moe_fused_swiglu_bwd_fake(grad, gate_up, act, interleaved, alpha, limit, eager_rounding):
            return gate_up.new_empty(gate_up.shape)

        def _setup_context(ctx, inputs, output):
            gate_up, act, interleaved, alpha, limit, eager_rounding = inputs
            ctx.cfg = (act, interleaved, alpha, limit, eager_rounding)
            ctx.save_for_backward(gate_up)

        def _backward(ctx, grad):
            (gate_up,) = ctx.saved_tensors
            if torch.is_grad_enabled():
                d = moe_swiglu_backward_reference(grad, gate_up, *ctx.cfg)
            else:
                d = torch.ops.unsloth_zoo.moe_fused_swiglu_bwd(grad, gate_up, *ctx.cfg)
            return d, None, None, None, None, None

        moe_fused_swiglu_op.register_autograd(_backward, setup_context = _setup_context)
        return torch.ops.unsloth_zoo.moe_fused_swiglu.default
    except Exception:
        return None


_FUSED_SWIGLU_OP = _register_ops() if triton is not None else None


def _bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def _identity(t):
    return t


def fused_swiglu_self_check(device, dtype, act, interleaved, alpha = 1.702, limit = 7.0) -> bool:
    """Cached bitwise probe of fused fwd + bwd vs eager autograd; eager ATen bits change across torch releases."""
    key = ("self_check", str(device), dtype, int(act), bool(interleaved), float(alpha), float(limit))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return False
    ok = False
    # First call may sit in a non-reentrant checkpoint region; identity saved_tensors_hooks (innermost wins) keep the
    # probe self-contained, since outer hooks would count its tensors (CheckpointError) and its grad would trigger a
    # nested recompute.
    try:
        hooks = torch.autograd.graph.saved_tensors_hooks(_identity, _identity)
        hooks.__enter__()
    except Exception:
        return False  # hooks disabled here (e.g. a functorch transform): eager now, probe again later
    try:
        with torch.inference_mode(False), torch.enable_grad(), torch.cuda.device(device):
            gen = torch.Generator(device = device).manual_seed(1234)
            M, N = 8, 96
            gu = torch.randn(M, 2 * N, device = device, generator = gen, dtype = torch.float32) * 4
            special = torch.tensor([limit, -limit, 0.0, -0.0, 1.0, -1.0, limit * 2, -limit * 2], device = device)
            gu[0, : special.numel()] = special
            gu[1, -special.numel():] = special
            gu = gu.to(dtype)
            grad = (torch.randn(M, N, device = device, generator = gen) * 2).to(dtype)
            grad[2, :4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device = device).to(dtype)
            x_e = gu.clone().requires_grad_()
            y_e = moe_swiglu_reference(x_e, act, interleaved, alpha, limit, True)
            (g_e,) = torch.autograd.grad(y_e, x_e, grad)
            y_f = _launch_fwd(gu, act, interleaved, alpha, limit, True)
            g_f = _launch_bwd(grad, gu, act, interleaved, alpha, limit, True)
            ok = _bits_equal(y_f, y_e) and _bits_equal(g_f, g_e)
        if not ok:
            warnings.warn(
                f"Unsloth: fused MoE SwiGLU does not reproduce this torch's eager bits for {dtype} act={act} "
                f"interleaved={interleaved}; keeping the eager activation there."
            )
    except Exception as err:
        if isinstance(err, torch.cuda.OutOfMemoryError):
            raise
        ok = False
    finally:
        hooks.__exit__(None, None, None)
    _CACHE[key] = ok
    return ok


def moe_fused_swiglu(gate_up, act = ACT_SILU, interleaved = False, alpha = 1.702, limit = 7.0, eager_rounding = True):
    """Callers check fused_swiglu_supported(gate_up) first."""
    act, interleaved, alpha, limit, eager_rounding = int(act), bool(interleaved), float(alpha), float(limit), bool(eager_rounding)
    if torch.compiler.is_compiling():
        if _FUSED_SWIGLU_OP is None:
            return moe_swiglu_reference(gate_up, act, interleaved, alpha, limit, eager_rounding)
        return _FUSED_SWIGLU_OP(gate_up, act, interleaved, alpha, limit, eager_rounding)
    return _FusedSwiGLU.apply(gate_up, act, interleaved, alpha, limit, eager_rounding)
