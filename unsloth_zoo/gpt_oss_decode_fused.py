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
"""One-launch replacements for the small eager op chains in gpt-oss decode.

Decode at batch 1 is host bound (the GPU is busy 16-45% of a step): every aten op costs a launch.
  add_rms_norm      (x + residual) -> new residual, RMSNorm(x + residual): replaces the residual
                    clone + bf16 add + 9-op fp32 RMSNorm (rms_layernorm_forward) with one kernel.
  sink_scale_output attention * sigmoid(lse - sink), transposed to [B, Q, H, D]: replaces 6 ops.
Same math and rounding points as the eager code (the residual sum is rounded to the input dtype
before the norm, the sink scale is rounded to the output dtype before the multiply); only the
fp32 reduction order and exp / rsqrt may differ by an ulp. Inference only: callers check
fused_decode_ok() (CUDA / HIP tensors, Triton present, no autograd, not tracing).
Kill switch: UNSLOTH_GPT_OSS_FUSED_DECODE=0, read at import.
"""

__all__ = [
    "fused_decode_ok",
    "add_rms_norm",
    "sink_scale_output",
    "FUSED_DECODE_CALLS",
]

import contextlib
import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

_ENABLED = HAS_TRITON and os.environ.get("UNSLOTH_GPT_OSS_FUSED_DECODE", "1") != "0"
# Engagement counters (tests and benchmarks read them).
FUSED_DECODE_CALLS = {"add_rms_norm": 0, "sink_scale_output": 0}
_MAX_COLS = 32768


def _on_device(t):
    # Launch on the tensor's card; entering torch.cuda.device costs host time, so only when needed.
    if t.is_cuda and t.device.index != torch.cuda.current_device():
        return torch.cuda.device(t.device)
    return contextlib.nullcontext()


def fused_decode_ok(x):
    """True when the fused kernels may replace the eager ops for this tensor."""
    return (
        _ENABLED
        and x.is_cuda
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


if HAS_TRITON:
    @triton.jit
    def _add_rms_norm_kernel(
        X, R, W, Y, N, stride_x, stride_r, stride_y, eps,
        HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < N
        x = tl.load(X + row * stride_x + cols, mask = ok, other = 0.0)
        if HAS_RES:
            r = tl.load(R + row * stride_r + cols, mask = ok, other = 0.0)
            # Eager `hidden_states += residual`: one rounding to the activation dtype.
            x = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
            tl.store(X + row * stride_x + cols, x, mask = ok)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, axis = 0) / N + eps
        y = xf * tl.rsqrt(var)
        w = tl.load(W + cols, mask = ok, other = 0.0).to(tl.float32)
        tl.store(Y + row * stride_y + cols, (y * w).to(Y.dtype.element_ty), mask = ok)

    @triton.jit
    def _sink_scale_kernel(
        A, LSE, SINK, OUT, H, Q, D,
        sa_b, sa_h, sa_q, sl_b, sl_h, sl_q, so_b, so_q, so_h,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q = pid % Q
        bh = pid // Q
        h = bh % H
        b = (bh // H).to(tl.int64)
        lse = tl.load(LSE + b * sl_b + h * sl_h + q * sl_q).to(tl.float32)
        sink = tl.load(SINK + h).to(tl.float32)
        # Eager: sigmoid in fp32, then the scale is cast to the output dtype before the multiply.
        scale = 1.0 / (1.0 + tl.exp(sink - lse))
        scale = scale.to(OUT.dtype.element_ty).to(tl.float32)
        cols = tl.arange(0, BLOCK_D)
        ok = cols < D
        a = tl.load(A + b * sa_b + h * sa_h + q * sa_q + cols, mask = ok, other = 0.0).to(tl.float32)
        tl.store(OUT + b * so_b + q * so_q + h * so_h + cols, (a * scale).to(OUT.dtype.element_ty), mask = ok)


def add_rms_norm(x, residual, weight, eps):
    """RMSNorm(x + residual) with x overwritten by x + residual (the next residual); residual None
    skips the add and leaves x untouched. x [..., N] contiguous; returns the normed tensor, or None
    when the shapes or devices are not supported (callers then run the eager ops)."""
    N = x.shape[-1]
    if (
        N > _MAX_COLS or not x.is_contiguous() or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.shape != (N,) or weight.device != x.device
    ):
        return None
    if residual is not None and (
        residual.shape != x.shape or residual.dtype != x.dtype or not residual.is_contiguous()
        or residual.device != x.device
    ):
        return None
    rows = x.numel() // N
    y = torch.empty_like(x)
    if rows == 0:
        return y
    BLOCK = triton.next_power_of_2(N)
    with _on_device(x):
        _add_rms_norm_kernel[(rows,)](
            x, x if residual is None else residual, weight, y, N, N, N, N, float(eps),
            HAS_RES = residual is not None, BLOCK = BLOCK, num_warps = 8 if BLOCK >= 4096 else 4,
        )
    FUSED_DECODE_CALLS["add_rms_norm"] += 1
    return y


def sink_scale_output(attn_output, lse, sinks):
    """(attn_output * sigmoid(lse - sinks)).transpose(1, 2).contiguous() in one launch.
    attn_output [B, H, Q, D], lse [B, H, Q] (fp32), sinks [H]; None when unsupported."""
    if attn_output.dim() != 4 or lse.dim() != 3 or sinks.dim() != 1:
        return None
    B, H, Q, D = attn_output.shape
    if (
        tuple(lse.shape) != (B, H, Q) or sinks.shape[0] != H or attn_output.stride(-1) != 1
        or sinks.device != attn_output.device or lse.device != attn_output.device or D > 1024
    ):
        return None
    out = torch.empty((B, Q, H, D), dtype = attn_output.dtype, device = attn_output.device)
    n = B * H * Q
    if n == 0:
        return out
    if not sinks.is_contiguous():
        sinks = sinks.contiguous()
    with _on_device(attn_output):
        _sink_scale_kernel[(n,)](
            attn_output, lse, sinks, out, H, Q, D,
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            lse.stride(0), lse.stride(1), lse.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D = triton.next_power_of_2(D), num_warps = 1 if D <= 128 else 4,
        )
    FUSED_DECODE_CALLS["sink_scale_output"] += 1
    return out
