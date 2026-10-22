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

"""Triton causal depthwise conv (+ SiLU) for GatedDeltaNet and Mamba2 when `causal_conv1d` is missing.

Replaces the transformers `F.conv1d` fallback (Qwen3.5/3.6, Qwen3-Next; on transformers >= 5.15
also the Mamba2 mixers of Nemotron-H, Falcon-H1, Granite-4.0-H, Bamba, Zamba2), reading and
writing channel-last directly to skip the transpose copies. Layout follows fla's
`causal_conv1d` (fla-org/flash-linear-attention, MIT), without initial state; `seq_idx`
(B, T) keeps taps inside one packed sequence.
Numerics match torch: fp32 accumulate, pre-activation and SiLU grad rounded to input dtype.
Cached decode gets a Triton `causal_conv1d_update` too (in-place conv state, one launch instead
of torch's cat / copy / conv / SiLU chain), so these models need no causal-conv1d binary.
``UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D=1`` keeps the torch fallback; a usable
`causal_conv1d` package always wins.
"""

__all__ = [
    "patch_gdn_causal_conv1d",
]

import os
import sys
import inspect
import functools
import importlib
import importlib.util

import torch

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
    logger,
)

_KILL_SWITCH = "UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D"
_MARK = "_unsloth_triton_causal_conv1d"
_HUB_MARK = "_unsloth_triton_causal_conv1d_hub"

_GDN_MODELING = ("qwen3_5", "qwen3_5_moe", "qwen3_next")
# Mamba2 mixers call the same depthwise causal conv (x (B, D, T), weight (D, W), SiLU) through the
# same hub decorator, so they share the dispatch on transformers >= 5.15. Older releases gate the
# whole Mamba2 fast path on the causal_conv1d package itself, so there is nothing to rebind there.
_MAMBA2_MODELING = ("nemotron_h", "falcon_h1", "granitemoehybrid", "bamba", "zamba2", "mamba2")
_GDN_LEAVES = frozenset(
    f"{prefix}_{package}" for package in _GDN_MODELING + _MAMBA2_MODELING
    for prefix in ("modeling", "unsloth_compiled_module")
)

GDN_CAUSAL_CONV1D_STATS = {
    "triton_fwd": 0, "triton_bwd": 0, "triton_seq_idx": 0, "fallback": 0,
    "triton_update": 0, "update_fallback": 0, "mamba_fused_fwd": 0, "mamba_fused_bwd": 0,
}

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_WIDTH = 8
_ACTIVATIONS = (None, "silu", "swish")

# Decode steps feed one token, speculative / chunked ones a few; longer runs use torch.
_MAX_UPDATE_T = 64

_kernels = None
_update_kernel = None
_broken = False
_update_broken = False


def _disabled():
    return os.environ.get(_KILL_SWITCH, "0") == "1"


def _real_causal_conv1d_available():
    # Usable, not just installed: misc.py's CUDA probe may set causal_conv1d_fn = None.
    try:
        if importlib.util.find_spec("causal_conv1d") is None:
            return False
        import causal_conv1d
        return callable(getattr(causal_conv1d, "causal_conv1d_fn", None))
    except Exception:
        return False


def _is_marked(fn):
    return getattr(inspect.unwrap(fn, stop = lambda f: getattr(f, _MARK, False)), _MARK, False)


def _build_kernels():
    # Triton 3.3 resolves nested kernels' `tl.constexpr` via globals ("tl is not defined").
    global _kernels, tl
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _causal_conv1d_fwd_kernel(
        x, weight, bias, y, seq_idx,
        T, D,
        stride_x_b, stride_x_t, stride_x_d,
        W: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
        HAS_SEQ_IDX: tl.constexpr,
        PAD_NEG: tl.constexpr,
    ):
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_t = i_t * BT + tl.arange(0, BT)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        # int64: channel-first x / dy has stride_d = T, and (D - 1) * T can pass 2**31.
        o_d64 = o_d.to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        if HAS_SEQ_IDX:
            p_s = seq_idx + tl.cast(i_b, tl.int64) * T
            b_s = tl.load(p_s + o_t, mask = o_t < T, other = -1)
        acc = tl.zeros((BT, BD), dtype = tl.float32)
        for i_w in tl.static_range(W):
            # Output t sees x[t - (W - 1) + i_w] (left zero padding of W - 1).
            s_t = o_t - (W - 1) + i_w
            m_t = (s_t >= 0) & (s_t < T)
            if HAS_SEQ_IDX:
                m_t = m_t & (tl.load(p_s + s_t, mask = m_t, other = -2) == b_s)
            m = m_t[:, None] & m_d[None, :]
            b_x = tl.load(
                p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d64[None, :] * stride_x_d,
                mask = m, other = 0,
            ).to(tl.float32)
            b_w = tl.load(weight + o_d * W + i_w, mask = m_d, other = 0).to(tl.float32)
            acc += b_x * b_w[None, :]
        if HAS_BIAS:
            acc += tl.load(bias + o_d, mask = m_d, other = 0).to(tl.float32)[None, :]
        if ACTIVATION:
            # The torch path stores the conv output in the input dtype, then SiLU.
            acc = acc.to(y.dtype.element_ty).to(tl.float32)
            acc = acc * tl.sigmoid(acc)
        if PAD_NEG:
            # causal_conv1d's rule: padding (seq_idx < 0) outputs 0; it never feeds a real position.
            acc = tl.where((b_s >= 0)[:, None], acc, 0.0)
        p_y = y + tl.cast(i_b, tl.int64) * T * D + o_t.to(tl.int64)[:, None] * D + o_d[None, :]
        tl.store(p_y, acc.to(y.dtype.element_ty), mask = (o_t < T)[:, None] & m_d[None, :])

    @triton.jit
    def _causal_conv1d_bwd_g_kernel(
        x, weight, bias, dy, g, seq_idx,
        T, D,
        stride_x_b, stride_x_t, stride_x_d,
        stride_dy_b, stride_dy_t, stride_dy_d,
        W: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_SEQ_IDX: tl.constexpr,
    ):
        # Pass 1: g = d(loss)/d(pre-activation) through SiLU, once per position (the pre-activation
        # is recomputed exactly as the forward does), stored channel-last in the input dtype.
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_t = i_t * BT + tl.arange(0, BT)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        o_d64 = o_d.to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        if HAS_SEQ_IDX:
            p_s = seq_idx + tl.cast(i_b, tl.int64) * T
            b_s = tl.load(p_s + o_t, mask = o_t < T, other = -1)
        acc = tl.zeros((BT, BD), dtype = tl.float32)
        for i_w in tl.static_range(W):
            s_t = o_t - (W - 1) + i_w
            m_t = (s_t >= 0) & (s_t < T)
            if HAS_SEQ_IDX:
                m_t = m_t & (tl.load(p_s + s_t, mask = m_t, other = -2) == b_s)
            b_x = tl.load(
                p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d64[None, :] * stride_x_d,
                mask = m_t[:, None] & m_d[None, :], other = 0,
            ).to(tl.float32)
            acc += b_x * tl.load(weight + o_d * W + i_w, mask = m_d, other = 0).to(tl.float32)[None, :]
        if HAS_BIAS:
            acc += tl.load(bias + o_d, mask = m_d, other = 0).to(tl.float32)[None, :]
        m = (o_t < T)[:, None] & m_d[None, :]
        b_dy = tl.load(
            dy + tl.cast(i_b, tl.int64) * stride_dy_b + o_t.to(tl.int64)[:, None] * stride_dy_t + o_d64[None, :] * stride_dy_d,
            mask = m, other = 0,
        ).to(tl.float32)
        pre = acc.to(g.dtype.element_ty).to(tl.float32)
        sig = tl.sigmoid(pre)
        # torch's silu_backward stores the gradient in the input dtype.
        b_g = b_dy * sig * (1 + pre * (1 - sig))
        p_g = g + tl.cast(i_b, tl.int64) * T * D + o_t.to(tl.int64)[:, None] * D + o_d[None, :]
        tl.store(p_g, b_g.to(g.dtype.element_ty), mask = m)

    @triton.jit
    def _causal_conv1d_bwd_dxdw_kernel(
        x, weight, g, dx, dw, db, seq_idx,
        T, D, NT,
        stride_x_b, stride_x_t, stride_x_d,
        stride_g_b, stride_g_t, stride_g_d,
        W: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        NEED_DX: tl.constexpr,
        NEED_DW: tl.constexpr,
        NEED_DB: tl.constexpr,
        HAS_SEQ_IDX: tl.constexpr,
        PAD_NEG: tl.constexpr,
    ):
        # Pass 2: dx[s] = sum_k w[W - 1 - k] g[s + k]; per-tile dw[j] = sum_t g[t] x[t - (W - 1) + j]
        # and db = sum_t g[t]; taps never cross a seq_idx boundary.
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_t = i_t * BT + tl.arange(0, BT)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        o_d64 = o_d.to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        p_g = g + tl.cast(i_b, tl.int64) * stride_g_b
        if HAS_SEQ_IDX:
            p_s = seq_idx + tl.cast(i_b, tl.int64) * T
            b_s = tl.load(p_s + o_t, mask = o_t < T, other = -1)
        if NEED_DX:
            b_dx = tl.zeros((BT, BD), dtype = tl.float32)
            for k in tl.static_range(W):
                r_t = o_t + k
                m_r = r_t < T
                if HAS_SEQ_IDX:
                    m_r = m_r & (tl.load(p_s + r_t, mask = m_r, other = -1) == b_s)
                if PAD_NEG:
                    # A padding output is the constant 0: no gradient flows back through it.
                    m_r = m_r & (b_s >= 0)
                b_g = tl.load(
                    p_g + r_t.to(tl.int64)[:, None] * stride_g_t + o_d64[None, :] * stride_g_d,
                    mask = m_r[:, None] & m_d[None, :], other = 0,
                ).to(tl.float32)
                b_dx += b_g * tl.load(weight + o_d * W + (W - 1 - k), mask = m_d, other = 0).to(tl.float32)[None, :]
            p_dx = dx + tl.cast(i_b, tl.int64) * T * D + o_t.to(tl.int64)[:, None] * D + o_d[None, :]
            tl.store(p_dx, b_dx.to(dx.dtype.element_ty), mask = (o_t < T)[:, None] & m_d[None, :])
        if NEED_DW or NEED_DB:
            m_g0 = o_t < T
            if PAD_NEG:
                m_g0 = m_g0 & (b_s >= 0)
            b_g0 = tl.load(
                p_g + o_t.to(tl.int64)[:, None] * stride_g_t + o_d64[None, :] * stride_g_d,
                mask = m_g0[:, None] & m_d[None, :], other = 0,
            ).to(tl.float32)
            row = (tl.cast(i_b, tl.int64) * NT + i_t) * D + o_d
            if NEED_DB:
                tl.store(db + row, tl.sum(b_g0, 0), mask = m_d)
            if NEED_DW:
                for j in tl.static_range(W):
                    s_t = o_t - (W - 1) + j
                    m_t = (s_t >= 0) & (s_t < T)
                    if HAS_SEQ_IDX:
                        m_t = m_t & (tl.load(p_s + s_t, mask = m_t, other = -2) == b_s)
                    b_xs = tl.load(
                        p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d64[None, :] * stride_x_d,
                        mask = m_t[:, None] & m_d[None, :], other = 0,
                    ).to(tl.float32)
                    tl.store(dw + row * W + j, tl.sum(b_g0 * b_xs, 0), mask = m_d)

    _kernels = (_causal_conv1d_fwd_kernel, (_causal_conv1d_bwd_g_kernel, _causal_conv1d_bwd_dxdw_kernel))
    return _kernels


# (BT, BD, num_warps). The backward is two forward-sized passes (g, then dx / dw / db): one
# W-tap recompute per position instead of W per tap, 2.3-3.3x faster than the single-pass kernel on a T4.
_FWD_CONFIG = (32, 64, 2)
_BWD_CONFIG = (32, 64, 2)
# CUDA caps grid dims 1 and 2 at 65535: T tiles ride dim 1, batch rides dim 2.
_MAX_GRID_YZ = 65535


def _launch_fwd(x_btd, weight, bias, activation, seq_idx = None, pad_negative = False):
    fwd, _ = _build_kernels()
    _BT, _BD, _NUM_WARPS = _FWD_CONFIG
    B, T, D = x_btd.shape
    W = weight.shape[1]
    y = torch.empty((B, T, D), dtype = x_btd.dtype, device = x_btd.device)
    grid = ((D + _BD - 1) // _BD, (T + _BT - 1) // _BT, B)
    ptrs = (x_btd, weight, bias if bias is not None else weight, y, seq_idx if seq_idx is not None else weight)
    ints = (T, D, *x_btd.stride())
    consts = (W, _BT, _BD, bias is not None, activation is not None, seq_idx is not None, pad_negative and seq_idx is not None)
    _conv_launch("fwd", fwd, grid, ptrs, ints, consts, _NUM_WARPS, (y,))
    return y


def _launch_bwd(x_btd, weight, bias, dy_btd, activation, need_dx, need_dw, need_db, seq_idx = None, pad_negative = False):
    _, (bwd_g, bwd_dxdw) = _build_kernels()
    _BT, _BD, _NUM_WARPS = _BWD_CONFIG
    B, T, D = x_btd.shape
    W = weight.shape[1]
    NT = (T + _BT - 1) // _BT
    grid = ((D + _BD - 1) // _BD, NT, B)
    seq_ptr = seq_idx if seq_idx is not None else weight
    if activation is not None:
        g = torch.empty((B, T, D), dtype = x_btd.dtype, device = x_btd.device)
        _conv_launch(
            "bwd_g", bwd_g, grid,
            (x_btd, weight, bias if bias is not None else weight, dy_btd, g, seq_ptr),
            (T, D, *x_btd.stride(), *dy_btd.stride()),
            (W, _BT, _BD, bias is not None, seq_idx is not None), _NUM_WARPS, (g,),
        )
    else:
        g = dy_btd  # no activation: the pre-activation gradient is dy itself
    dx = torch.empty((B, T, D), dtype = x_btd.dtype, device = x_btd.device) if need_dx else None
    dw = torch.empty((B * NT, D, W), dtype = torch.float32, device = x_btd.device) if need_dw else None
    db = torch.empty((B * NT, D), dtype = torch.float32, device = x_btd.device) if need_db else None
    if need_dx or need_dw or need_db:
        _conv_launch(
            "bwd_dxdw", bwd_dxdw, grid,
            (x_btd, weight, g, dx if dx is not None else x_btd, dw if dw is not None else weight,
             db if db is not None else weight, seq_ptr),
            (T, D, NT, *x_btd.stride(), *g.stride()),
            (W, _BT, _BD, need_dx, need_dw, need_db, seq_idx is not None, pad_negative and seq_idx is not None), _NUM_WARPS,
            tuple(t for t in (dx, dw, db) if t is not None),
        )
    if dw is not None:
        dw = dw.sum(0).to(weight.dtype)
    if db is not None:
        db = db.sum(0).to(bias.dtype)
    return dx, dw, db


# Training and prefill call the conv once per layer per step with the same shapes, so the Triton
# launch (specialization + argument binding, ~20-30 us of host time) is cached per shape like the
# decode update below: key -> (grid, compiled kernel). Same gates: Triton 3.6+ launcher convention,
# a bitwise self-check per kernel on first use, full launches under profiler hooks, for unaligned
# pointers (Triton specializes on 16-byte alignment) and off the current device.
# T is in the key: cap it (FIFO) so variable lengths do not grow it unbounded.
_conv_plans = {}
_CONV_PLANS_MAX = 256
_conv_fast_ok = {}


def _conv_full_launch(kernel, grid, ptrs, ints, consts, num_warps):
    args = ptrs + ints + consts
    if ptrs[0].get_device() != torch._C._cuda_getDevice():
        with torch.cuda.device(ptrs[0].device):
            return kernel[grid](*args, num_warps = num_warps, enable_fp_fusion = True)
    return kernel[grid](*args, num_warps = num_warps, enable_fp_fusion = True)


def _conv_fast_launch(plan, ptrs, ints, consts):
    grid, compiled = plan
    compiled.run(
        grid[0], grid[1], grid[2], torch._C._cuda_getCurrentRawStream(ptrs[0].get_device()),
        compiled.function, compiled.packed_metadata, None, None, None,
        *ptrs, *ints, *consts,
    )


def _conv_launch(name, kernel, grid, ptrs, ints, consts, num_warps, outs):
    """One conv kernel launch, through the cached launcher when it is safe. `outs` are the tensors
    the kernel writes (and only writes), compared bitwise on the first cached launch."""
    aligned = not (functools.reduce(lambda acc, t: acc | t.data_ptr(), ptrs, 0) & 15)
    key = (name, grid, ints, consts, tuple(t.dtype for t in ptrs), ptrs[0].get_device())
    plan = _conv_plans.get(key) if aligned else None
    if (
        plan is None or _conv_fast_ok.get(name) is False or _launch_hooks_set()
        or ptrs[0].get_device() != torch._C._cuda_getDevice()
    ):
        compiled = _conv_full_launch(kernel, grid, ptrs, ints, consts, num_warps)
        if aligned and plan is None and compiled is not None and _triton_launcher_verified():
            if len(_conv_plans) >= _CONV_PLANS_MAX:
                _conv_plans.pop(next(iter(_conv_plans)), None)
            _conv_plans[key] = (grid, compiled)
        return
    if name not in _conv_fast_ok:
        _conv_full_launch(kernel, grid, ptrs, ints, consts, num_warps)
        refs = [t.clone() for t in outs]
        try:
            _conv_fast_launch(plan, ptrs, ints, consts)
            ok = all(torch.equal(t, r) for t, r in zip(outs, refs))
        except Exception:
            ok = False
        _conv_fast_ok[name] = ok
        if not ok:
            for t, r in zip(outs, refs):
                t.copy_(r)
        return
    _conv_fast_launch(plan, ptrs, ints, consts)


class _CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bdt, weight, bias, activation, seq_idx = None):
        y_btd = _launch_fwd(x_bdt.transpose(1, 2), weight, bias, activation, seq_idx)
        ctx.save_for_backward(x_bdt, weight, bias, seq_idx)
        ctx.activation = activation
        GDN_CAUSAL_CONV1D_STATS["triton_fwd"] += 1
        if seq_idx is not None:
            GDN_CAUSAL_CONV1D_STATS["triton_seq_idx"] += 1
        return y_btd.transpose(1, 2)

    @staticmethod
    def backward(ctx, dy_bdt):
        x_bdt, weight, bias, seq_idx = ctx.saved_tensors
        dx, dw, db = _launch_bwd(
            x_bdt.transpose(1, 2), weight, bias, dy_bdt.transpose(1, 2),
            ctx.activation, *ctx.needs_input_grad[:3], seq_idx = seq_idx,
        )
        GDN_CAUSAL_CONV1D_STATS["triton_bwd"] += 1
        return (dx.transpose(1, 2) if dx is not None else None), dw, db, None, None


def _build_update_kernel():
    global _update_kernel, tl
    if _update_kernel is not None:
        return _update_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _causal_conv1d_update_kernel(
        x, state, weight, bias, y,
        D, T,
        stride_x_b, stride_x_d, stride_x_t,
        stride_s_b, stride_s_d, stride_s_l,
        stride_y_b, stride_y_d, stride_y_t,
        L: tl.constexpr,
        LP: tl.constexpr,
        W: tl.constexpr,
        BD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        # Taps index c = [state (L) | x (T)]: output t sees c[L + t - (W - 1) + i_w], the new
        # state is c[T:T + L]. Same as the torch path's cat + copy_ + conv1d(padding = 0).
        i_d, i_b = tl.program_id(0), tl.program_id(1)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        o_d64 = o_d.to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b + o_d64 * stride_x_d
        p_s = state + tl.cast(i_b, tl.int64) * stride_s_b + o_d64 * stride_s_d
        p_y = y + tl.cast(i_b, tl.int64) * stride_y_b + o_d64 * stride_y_d
        if HAS_BIAS:
            b_bias = tl.load(bias + o_d, mask = m_d, other = 0).to(tl.float32)
        for t in range(T):
            acc = tl.zeros((BD,), dtype = tl.float32)
            for i_w in tl.static_range(W):
                c = L + t - (W - 1) + i_w
                b_c = tl.load(p_s + c * stride_s_l, mask = m_d & (c < L), other = 0).to(tl.float32)
                b_c += tl.load(p_x + (c - L) * stride_x_t, mask = m_d & (c >= L), other = 0).to(tl.float32)
                acc += b_c * tl.load(weight + o_d * W + i_w, mask = m_d, other = 0).to(tl.float32)
            if HAS_BIAS:
                acc += b_bias
            if ACTIVATION:
                # The torch path stores the conv output in the input dtype, then SiLU.
                acc = acc.to(y.dtype.element_ty).to(tl.float32)
                acc = acc * tl.sigmoid(acc)
            tl.store(p_y + t * stride_y_t, acc.to(y.dtype.element_ty), mask = m_d)
        o_l = tl.arange(0, LP)
        c = T + o_l
        m_l = (o_l < L)[:, None] & m_d[None, :]
        b_new = tl.load(
            p_s[None, :] + (tl.minimum(c, L - 1) * stride_s_l)[:, None],
            mask = m_l & (c < L)[:, None], other = 0,
        )
        b_new += tl.load(
            p_x[None, :] + (tl.maximum(c - L, 0) * stride_x_t)[:, None],
            mask = m_l & (c >= L)[:, None], other = 0,
        )
        # The state is updated in place, and other threads may still be reading the old values.
        tl.debug_barrier()
        tl.store(p_s[None, :] + (o_l * stride_s_l)[:, None], b_new, mask = m_l)

    _update_kernel = _causal_conv1d_update_kernel
    return _update_kernel


# (BD, num_warps): a decode step is a few KB, so this is launch bound.
_UPDATE_CONFIG = (128, 4)


def triton_next_power_of_2(n):
    return 1 << max(0, int(n) - 1).bit_length()


_runtime_hooks = None


def _launch_hooks_set():
    global _runtime_hooks
    if _runtime_hooks is None:
        try:
            from triton import knobs
            _runtime_hooks = knobs.runtime
        except Exception:
            from triton.compiler.compiler import CompiledKernel
            _runtime_hooks = CompiledKernel
    return _hook_active(_runtime_hooks.launch_enter_hook) or _hook_active(_runtime_hooks.launch_exit_hook)


def _hook_active(hook):
    # Triton 3.6+ keeps an always-present HookChain; empty means no profiler hooked in.
    return hook is not None and bool(getattr(hook, "calls", True))


def triton_causal_conv1d_update(x, conv_state, weight, bias = None, activation = None):
    """x (B, D, T), conv_state (B, D, L) updated in place, weight (D, W) -> (B, D, T) in x's layout.
    Always the full Triton launch (specialization + argument binding); the decode path caches it."""
    return _launch_update(x, conv_state, weight, bias, activation)[0]


def _launch_update(x, conv_state, weight, bias, activation, y = None):
    kernel = _build_update_kernel()
    _BD, _NUM_WARPS = _UPDATE_CONFIG
    B, D, T = x.shape
    L, W = conv_state.shape[-1], weight.shape[1]
    if y is None:
        # Keeps x's strides (the modeling code hands a transposed channel-last view): no extra copy.
        y = torch.empty_like(x)
    grid = ((D + _BD - 1) // _BD, B)
    ints = (D, T, *x.stride(), *conv_state.stride(), *y.stride())
    tail = (L, triton_next_power_of_2(L), W, _BD, bias is not None, activation is not None)
    args = (x, conv_state, weight, bias if bias is not None else weight, y) + ints + tail
    if x.get_device() != torch._C._cuda_getDevice():
        with torch.cuda.device(x.device):
            compiled = kernel[grid](*args, num_warps = _NUM_WARPS, enable_fp_fusion = True)
    else:
        compiled = kernel[grid](*args, num_warps = _NUM_WARPS, enable_fp_fusion = True)
    return y, (grid, compiled, ints, ints + tail)


# key -> (grid, compiled kernel, runtime ints, runtime ints + constexprs) for repeated decode
# steps; None = ineligible.
_update_plans = {}
# Which argument list the cached launcher takes on this Triton: None = not checked yet,
# "all" (every parameter, Triton 3.6+), False = full launches only.
_fast_launch_mode = None


def _update_key(x, conv_state, weight, bias, activation):
    # Everything the eligibility check and Triton's specialization read: shapes, strides, dtypes,
    # devices and the integer values (==1, %16). Pointer alignment is checked by the caller.
    return (
        x.shape, x.stride(), x.dtype, x.get_device(),
        conv_state.shape, conv_state.stride(), conv_state.dtype, conv_state.get_device(),
        weight.shape, weight.stride(), weight.dtype, weight.get_device(),
        None if bias is None else (bias.shape, bias.stride(), bias.dtype, bias.get_device()),
        activation,
    )


def _update_eligible(x, conv_state, weight, bias, activation):
    return (
        isinstance(x, torch.Tensor) and isinstance(conv_state, torch.Tensor)
        and isinstance(weight, torch.Tensor) and (bias is None or isinstance(bias, torch.Tensor))
        and x.is_cuda and x.dim() == 3 and x.numel() > 0 and x.dtype in _SUPPORTED_DTYPES
        and 1 <= x.shape[2] <= _MAX_UPDATE_T and x.shape[0] <= _MAX_GRID_YZ
        and conv_state.dim() == 3
        and conv_state.dtype == x.dtype and conv_state.device == x.device
        and conv_state.shape[:2] == x.shape[:2]
        and weight.dim() == 2 and weight.dtype == x.dtype and weight.device == x.device
        and weight.is_contiguous() and weight.shape[0] == x.shape[1]
        and 1 <= weight.shape[1] <= _MAX_WIDTH
        # Fewer than W - 1 cached taps: the torch path returns a shorter output.
        and weight.shape[1] - 1 <= conv_state.shape[2] <= 4 * _MAX_WIDTH
        and activation in _ACTIVATIONS
        and (bias is None or (
            bias.dtype == x.dtype and bias.device == x.device
            and bias.shape == (x.shape[1],) and bias.is_contiguous()
        ))
    )


def _fast_launch(mode, plan, x, conv_state, weight, bias, y):
    grid, compiled, ints, all_ints = plan
    compiled.run(
        grid[0], grid[1], 1, torch._C._cuda_getCurrentRawStream(x.get_device()),
        compiled.function, compiled.packed_metadata, None, None, None,
        x, conv_state, weight, bias if bias is not None else weight, y,
        *(all_ints if mode == "all" else ints),
    )


def _triton_launcher_verified():
    """Triton 3.6+ (torch 2.10+): the launcher convention _fast_launch uses was checked there."""
    try:
        import triton
        major, minor = (int(part) for part in triton.__version__.split(".")[:2])
    except Exception:
        return False
    return (major, minor) >= (3, 6)


def _check_fast_launch(plan, x, conv_state, weight, bias, activation):
    """Once per process: launch through the cached launcher and compare with a full launch on
    copies; pick the argument convention this Triton takes, or keep full launches."""
    global _fast_launch_mode
    if not _triton_launcher_verified():
        # A wrong argument list can fault on the device instead of raising, which no fallback
        # survives, so only the launcher convention checked on real hardware is tried.
        _fast_launch_mode = False
        return _launch_update(x, conv_state, weight, bias, activation)[0]
    state_before = conv_state.clone()
    for mode in ("all",):
        ref_state = state_before.clone()
        y_ref = _launch_update(x, ref_state, weight, bias, activation)[0]
        y = torch.empty_like(x)
        try:
            _fast_launch(mode, plan, x, conv_state, weight, bias, y)
            ok = torch.equal(y, y_ref) and torch.equal(conv_state, ref_state)
        except Exception:
            ok = False
        if ok:
            _fast_launch_mode = mode
            return y
        conv_state.copy_(state_before)
    _fast_launch_mode = False
    return _launch_update(x, conv_state, weight, bias, activation)[0]


def _run_update(x, conv_state, weight, bias, activation):
    """Triton update for repeated decode steps: one dict lookup on the cached plan, then a direct
    launcher call (no re-specialization or argument binding). None = use the torch fallback."""
    try:
        key = _update_key(x, conv_state, weight, bias, activation)
    except (AttributeError, TypeError):
        return None  # not tensors: the torch path decides
    plan = _update_plans.get(key, 0)
    if plan is None:
        return None
    if plan == 0:
        if not _update_eligible(x, conv_state, weight, bias, activation):
            _update_plans[key] = None
            return None
        y, plan = _launch_update(x, conv_state, weight, bias, activation)
        # Triton specializes pointers on 16-byte alignment: cache only the all-aligned plan.
        if not _unaligned(x, conv_state, weight, bias):
            _update_plans[key] = plan
        return y
    if (
        _fast_launch_mode is False or _unaligned(x, conv_state, weight, bias) or _launch_hooks_set()
        or x.get_device() != torch._C._cuda_getDevice()
    ):
        return _launch_update(x, conv_state, weight, bias, activation)[0]
    if _fast_launch_mode is None:
        return _check_fast_launch(plan, x, conv_state, weight, bias, activation)
    y = torch.empty_like(x)
    _fast_launch(_fast_launch_mode, plan, x, conv_state, weight, bias, y)
    return y


def _unaligned(x, conv_state, weight, bias):
    return (x.data_ptr() | conv_state.data_ptr() | weight.data_ptr() | (0 if bias is None else bias.data_ptr())) & 15


# Read once: the decode path runs per token per layer. `_refresh_kill_switch` re-reads it.
_update_disabled = _disabled()


def _refresh_kill_switch():
    global _update_disabled
    _update_disabled = _disabled()


def _try_fast_update(x, conv_state, weight, bias, activation):
    """Run the Triton update, or return None to use the torch fallback."""
    global _update_broken
    if _update_broken or _update_disabled or torch.compiler.is_compiling():
        return None
    # The raw launch has no backward: anything autograd would track takes the torch path.
    if torch.is_grad_enabled() and (
        getattr(x, "requires_grad", False) or getattr(conv_state, "requires_grad", False)
        or getattr(weight, "requires_grad", False) or getattr(bias, "requires_grad", False)
    ):
        return None
    try:
        out = _run_update(x, conv_state, weight, bias, activation)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:
        _update_broken = True
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Triton causal_conv1d_update failed, using torch: {e}")
        return None
    if out is not None:
        GDN_CAUSAL_CONV1D_STATS["triton_update"] += 1
    return out


def causal_conv1d_update_reference(hidden_states, conv_state, weight, bias = None, activation = None):
    """The transformers 5.x torch fallback: conv_state (B, D, L) is updated in place."""
    import torch.nn.functional as F
    hidden_size, seq_len = hidden_states.shape[1], hidden_states.shape[2]
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim = -1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding = 0, groups = hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = _act(activation)(out)
    return out.to(hidden_states.dtype)


def _make_update_dispatch(torch_function):
    """Wrap a torch `causal_conv1d_update` fallback with the Triton update."""
    @functools.wraps(torch_function)
    def causal_conv1d_update(hidden_states, conv_state, weight, bias = None, activation = None):
        out = _try_fast_update(hidden_states, conv_state, weight, bias, activation)
        if out is not None:
            return out
        if not torch.compiler.is_compiling():
            GDN_CAUSAL_CONV1D_STATS["update_fallback"] += 1
        return torch_function(hidden_states, conv_state, weight, bias, activation)

    setattr(causal_conv1d_update, _MARK, True)
    return _own_signature(causal_conv1d_update)


def _act(activation):
    from transformers.activations import ACT2FN
    return ACT2FN[activation]


def causal_conv1d_reference(x, weight, bias = None, activation = None):
    """The transformers torch fallback: x (B, D, T), weight (D, W)."""
    import torch.nn.functional as F
    D, T = x.shape[1], x.shape[2]
    out = F.conv1d(
        x.to(weight.dtype), weight.unsqueeze(1), bias,
        padding = weight.shape[-1] - 1, groups = D,
    )[:, :, :T]
    if activation is not None:
        out = _act(activation)(out)
    return out.to(x.dtype)


def _check_seq_idx(x, seq_idx):
    """Validate seq_idx (B, T); raise rather than silently drop a mis-shaped one."""
    if seq_idx is None:
        return None
    if (
        not isinstance(seq_idx, torch.Tensor) or seq_idx.dim() != 2
        or tuple(seq_idx.shape) != (x.shape[0], x.shape[-1])
        or seq_idx.dtype.is_floating_point or seq_idx.dtype.is_complex or seq_idx.dtype == torch.bool
    ):
        raise ValueError(
            "Unsloth: causal_conv1d seq_idx must be an integer tensor of shape (batch, seqlen) = "
            f"{(x.shape[0], x.shape[-1])}, got {getattr(seq_idx, 'dtype', type(seq_idx))} "
            f"{getattr(seq_idx, 'shape', '')}."
        )
    # Collators may leave it on CPU; a host to device copy can be async, the reverse cannot.
    return seq_idx.to(device = x.device, non_blocking = x.device.type != "cpu")


def _eligible(x, weight, bias, activation, seq_idx = None, mixed_dtypes = False):
    # mixed_dtypes: weight / bias may differ from x (one dtype between them), as causal_conv1d's
    # entry points allow under autocast; the kernels load every operand in its own dtype.
    wdtype = weight.dtype if mixed_dtypes else x.dtype
    if not (
        x.is_cuda and x.dim() == 3 and x.numel() > 0 and x.dtype in _SUPPORTED_DTYPES
        and weight.dim() == 2 and weight.dtype == wdtype and wdtype in _SUPPORTED_DTYPES
        and weight.device == x.device
        and weight.is_contiguous() and weight.shape[0] == x.shape[1]
        and 1 <= weight.shape[1] <= _MAX_WIDTH and activation in _ACTIVATIONS
        and (bias is None or (
            bias.dtype == wdtype and bias.device == x.device
            and bias.shape == (x.shape[1],) and bias.is_contiguous()
        ))
        and (seq_idx is None or seq_idx.device == x.device)
    ):
        return False
    # Both launches must fit the grid: the backward tiles T finer and has no fallback.
    min_bt = min(_FWD_CONFIG[0], _BWD_CONFIG[0])
    return x.shape[0] <= _MAX_GRID_YZ and (x.shape[2] + min_bt - 1) // min_bt <= _MAX_GRID_YZ


def triton_causal_conv1d(x, weight, bias = None, activation = None, seq_idx = None):
    """x (B, D, T), weight (D, W), seq_idx (B, T) -> (B, D, T) stored channel-last."""
    if seq_idx is not None:
        seq_idx = _check_seq_idx(x, seq_idx)
        # Kernels only compare IDs for equality in the loaded dtype: never narrow (0 == 2**32 in
        # int32); widen the rest (int8/16, uint*) to int64, a bijection for uint64.
        if seq_idx.dtype not in (torch.int32, torch.int64):
            seq_idx = seq_idx.to(torch.int64)
        seq_idx = seq_idx.contiguous()
    return _CausalConv1dFunction.apply(x, weight, bias, activation, seq_idx)


def _try_fast(x, weight, bias, activation, seq_idx = None):
    """Run the Triton kernel, or return None to use the torch fallback."""
    global _broken
    # Under torch.compile (decode regions) trace the torch path, never the launcher.
    if _broken or torch.compiler.is_compiling() or _disabled():
        return None
    if not _eligible(x, weight, bias, activation, seq_idx):
        return None
    try:
        return triton_causal_conv1d(x, weight, bias, activation, seq_idx)
    except torch.cuda.OutOfMemoryError:
        # Not a kernel fault: a caught OOM (batch size finders) must not disable it for good.
        raise
    except Exception as e:
        _broken = True
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Triton causal_conv1d failed, using torch: {e}")
        return None


def _count_fallback():
    # Not while tracing: a global counter read inside a compiled region becomes a guard.
    if not torch.compiler.is_compiling():
        GDN_CAUSAL_CONV1D_STATS["fallback"] += 1


def _causal_conv1d_any(x, weight, bias = None, activation = None, seq_idx = None):
    seq_idx = _check_seq_idx(x, seq_idx)
    out = _try_fast(x, weight, bias, activation, seq_idx)
    if out is not None:
        return out
    _count_fallback()
    return _causal_conv1d_reference_seq_idx(x, weight, bias, activation, seq_idx)


def _own_signature(fn):
    # Else signature follows __wrapped__ (or a __signature__ functools.wraps copied) and the
    # hub wrapper drops `seq_idx`.
    fn.__dict__.pop("__signature__", None)
    fn.__signature__ = inspect.signature(fn, follow_wrapped = False)
    return fn


def _make_hub_dispatch(torch_function):
    """Wrap the transformers torch fallback; names `seq_idx` for the hub wrapper and packing gate."""
    @functools.wraps(torch_function)
    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, seq_idx = None, **kwargs):
        if seq_idx is not None:
            return _causal_conv1d_any(hidden_states, weight, bias, activation, seq_idx)
        out = _try_fast(hidden_states, weight, bias, activation)
        if out is not None:
            return out
        _count_fallback()
        return torch_function(hidden_states, weight, bias, activation, **kwargs)

    setattr(causal_conv1d_fn, _MARK, True)
    return _own_signature(causal_conv1d_fn)


def _causal_conv1d_reference_seq_idx(x, weight, bias, activation, seq_idx, pad_negative = False):
    """Torch causal conv whose taps do not cross `seq_idx` (B, T) boundaries. *pad_negative*:
    causal_conv1d's rule that seq_idx < 0 marks padding, whose output is 0."""
    import torch.nn.functional as F
    seq_idx = _check_seq_idx(x, seq_idx)
    if seq_idx is None:
        return causal_conv1d_reference(x, weight, bias, activation)
    W, T = weight.shape[-1], x.shape[-1]
    acc = torch.promote_types(weight.dtype, torch.float32)
    xp = F.pad(x.to(acc), (W - 1, 0))
    # int64 on both sides: torch has no promotion for uint16/32/64.
    sp = F.pad(seq_idx.to(torch.int64), (W - 1, 0), value = -1)
    w = weight.to(acc)
    out = torch.zeros_like(xp[:, :, :T])
    for j in range(W):
        same = (sp[:, j:j + T] == sp[:, W - 1:]).unsqueeze(1)
        out = out + xp[:, :, j:j + T] * w[:, j].view(1, -1, 1) * same
    if bias is not None:
        out = out + bias.to(acc).view(1, -1, 1)
    out = out.to(weight.dtype)
    if activation is not None:
        out = _act(activation)(out)
    if pad_negative:
        out = out * (seq_idx.to(torch.int64) >= 0).unsqueeze(1)
    return out.to(x.dtype)


def _legacy_causal_conv1d_fn(x, weight, bias = None, activation = None, seq_idx = None, **kwargs):
    """`causal_conv1d_fn` stand-in for transformers < 5.15; names `seq_idx` for the packing gate."""
    unsupported = sorted(k for k, v in kwargs.items() if v is not None and v is not False)
    if unsupported:
        raise NotImplementedError(
            f"Unsloth: the torch causal_conv1d fallback does not support {unsupported}."
        )
    return _causal_conv1d_any(x, weight, bias, activation, seq_idx)


setattr(_legacy_causal_conv1d_fn, _MARK, True)


def _is_gdn_function(torch_function):
    module = getattr(torch_function, "__module__", None) or ""
    return module.rsplit(".", 1)[-1] in _GDN_LEAVES


def _hub_gdn_function(decorator, torch_function):
    """Hub-decorate the GDN dispatch. transformers >= 5.20 refuses `seq_idx` on its torch path
    when causal-conv1d is missing, but this dispatch keeps packed rows apart, so packed calls
    go to it directly; everything else still goes through the hub wrapper."""
    dispatch = _make_hub_dispatch(torch_function)
    hub_fn = decorator(dispatch)
    # The hub wrapper passes only the implementation's named parameters on.
    names = frozenset(inspect.signature(dispatch).parameters)

    if isinstance(hub_fn, torch.nn.Module):
        # With `kernels` installed the hub decorator turns the function into a layer instance whose
        # class carries `kernel_layer_name`, which `use_kernelized_func` checks and `kernelize` reads
        # from `type(fn)`. A plain function wrapper loses both, so route on the instance instead:
        # the layer stays what transformers registered, and `kernelize` can still swap its forward.
        hub_forward = hub_fn.forward

        @functools.wraps(hub_forward)
        def forward(*args, **kwargs):
            if kwargs.get("seq_idx") is not None:
                return dispatch(*args, **{k: v for k, v in kwargs.items() if k in names})
            return hub_forward(*args, **kwargs)

        hub_fn.forward = forward
        # Signature checks (Unsloth's packing gate looks for `seq_idx`) see the dispatch, not (*args, **kwargs).
        hub_fn.__signature__ = inspect.signature(dispatch)
        setattr(hub_fn, _MARK, True)
        return hub_fn

    @functools.wraps(hub_fn)
    def causal_conv1d_fn(*args, **kwargs):
        if kwargs.get("seq_idx") is not None:
            return dispatch(*args, **{k: v for k, v in kwargs.items() if k in names})
        return hub_fn(*args, **kwargs)

    setattr(causal_conv1d_fn, _MARK, True)
    return causal_conv1d_fn


def _hub_gdn_update(decorator, torch_function):
    """Hub-decorate the GDN update dispatch. With `kernels` installed the result is a layer
    instance (its class keeps `kernel_layer_name`), which does not inherit the dispatch's mark."""
    hub_fn = decorator(_make_update_dispatch(torch_function))
    if isinstance(hub_fn, torch.nn.Module):
        setattr(hub_fn, _MARK, True)
    return hub_fn


def _patch_hub_decorator():
    """transformers >= 5.15: hook the hub decorator so later-decorated GDN convs
    (incl. compiled copies) get the Triton dispatch."""
    try:
        import transformers.integrations as integrations
        from transformers.integrations import hub_kernels
        original = hub_kernels.use_kernel_func_from_hub_with_fallback
    except Exception:
        return None
    if getattr(original, _HUB_MARK, False):
        return original

    @functools.wraps(original)
    def use_kernel_func_from_hub_with_fallback(func_name, package, *args, **kwargs):
        decorator = original(func_name, package, *args, **kwargs)
        if package != "causal_conv1d":
            return decorator
        # A usable package still wins: the hub wrapper only calls this torch fallback without it.
        if func_name == "causal_conv1d_fn":
            return lambda fn: _hub_gdn_function(decorator, fn) if _is_gdn_function(fn) else decorator(fn)
        if func_name == "causal_conv1d_update":
            return lambda fn: _hub_gdn_update(decorator, fn) if _is_gdn_function(fn) else decorator(fn)
        return decorator

    setattr(use_kernel_func_from_hub_with_fallback, _HUB_MARK, True)
    hub_kernels.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    if getattr(integrations, "use_kernel_func_from_hub_with_fallback", None) is original:
        integrations.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    return use_kernel_func_from_hub_with_fallback


def _legacy_update(module):
    """`causal_conv1d_update` stand-in for transformers < 5.15, falling back to the module's own
    torch update (4.57's `torch_causal_conv1d_update` always applies SiLU)."""
    torch_function = vars(module).get("torch_causal_conv1d_update") or causal_conv1d_update_reference
    return _make_update_dispatch(torch_function)


def _rebind_name(module, name, legacy, patched_decorator):
    current = vars(module)[name]
    if current is None:
        # transformers < 5.15: None unless the package imported; copied in layer __init__.
        setattr(module, name, legacy())
        return True
    if _is_marked(current):
        return True
    torch_function = getattr(current, "__wrapped__", None)
    if torch_function is None or patched_decorator is None:
        return False
    setattr(module, name, patched_decorator(name, "causal_conv1d")(torch_function))
    return _is_marked(getattr(module, name))


def _rebind_module(module, patched_decorator):
    """Rebind an imported module; True if its conv now dispatches to Triton. Re-decorating
    also replaces a wrapper that captured a since-disabled package kernel."""
    names = vars(module)
    if patched_decorator is not None and "use_kernel_func_from_hub_with_fallback" in names:
        module.use_kernel_func_from_hub_with_fallback = patched_decorator
    if "causal_conv1d_update" in names:
        _rebind_name(module, "causal_conv1d_update", lambda: _legacy_update(module), patched_decorator)
    if "causal_conv1d_fn" not in names:
        return False
    return _rebind_name(module, "causal_conv1d_fn", lambda: _legacy_causal_conv1d_fn, patched_decorator)


# mamba_ssm's fused split path (`mamba_split_conv1d_scan_combined`, the training fast path of every
# Mamba2 mixer) calls causal_conv1d's C++ entry points directly. These keep their signatures so the
# fused path, with its recompute-in-backward memory savings, runs on the Triton conv instead.
_MAMBA_FUSED_MARK = "_unsloth_triton_mamba_fused"


def _mamba_conv_operands(x_bds, weight, bias):
    # Kept in their own dtype (an fp32 parameter under bf16 autocast): the kernels accumulate in
    # fp32 and dW / db come back in the parameter dtype, as causal_conv1d's entry points do.
    return weight.contiguous(), (bias.contiguous() if bias is not None else None)


def _autograd_tracks(*tensors):
    return torch.is_grad_enabled() and any(getattr(t, "requires_grad", False) for t in tensors)


def mamba_causal_conv1d_fwd_function(x, weight, bias, seq_idx, initial_states, final_states_out, silu_activation):
    if initial_states is not None or final_states_out is not None:
        raise NotImplementedError("Unsloth: Triton causal conv has no initial / final states")
    activation = "silu" if silu_activation else None
    w, b = _mamba_conv_operands(x, weight, bias)
    if seq_idx is not None and seq_idx.dtype not in (torch.int32, torch.int64):
        seq_idx = seq_idx.to(torch.int64)
    # mamba_ssm calls this inside its autograd.Function (grad mode off); a raw launch has no
    # backward, so anything autograd would track takes the differentiable torch reference.
    if _autograd_tracks(x, weight, bias) or not _eligible(x, w, b, activation, seq_idx, mixed_dtypes = True):
        return _causal_conv1d_reference_seq_idx(x, w, b, activation, seq_idx, pad_negative = True)
    GDN_CAUSAL_CONV1D_STATS["mamba_fused_fwd"] += 1
    return _launch_fwd(x.transpose(1, 2), w, b, activation, seq_idx, pad_negative = True).transpose(1, 2)


def mamba_causal_conv1d_bwd_function(x, weight, bias, dout, seq_idx, initial_states, dfinal_states, dx, return_dinitial_states, silu_activation):
    if initial_states is not None or dfinal_states is not None or return_dinitial_states:
        raise NotImplementedError("Unsloth: Triton causal conv has no initial / final states")
    activation = "silu" if silu_activation else None
    w, b = _mamba_conv_operands(x, weight, bias)
    if seq_idx is not None and seq_idx.dtype not in (torch.int32, torch.int64):
        seq_idx = seq_idx.to(torch.int64)
    if not _autograd_tracks(x, weight, bias, dout) and _eligible(x, w, b, activation, seq_idx, mixed_dtypes = True):
        GDN_CAUSAL_CONV1D_STATS["mamba_fused_bwd"] += 1
        dx_btd, dw, db = _launch_bwd(
            x.transpose(1, 2), w, b, dout.transpose(1, 2), activation,
            True, True, b is not None, seq_idx = seq_idx, pad_negative = True,
        )
        dx_new = dx_btd.transpose(1, 2)
    else:
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            wr = w.detach().requires_grad_(True)
            br = b.detach().requires_grad_(True) if b is not None else None
            y = _causal_conv1d_reference_seq_idx(xr, wr, br, activation, seq_idx, pad_negative = True)
            grads = torch.autograd.grad(y, [t for t in (xr, wr, br) if t is not None], dout)
        dx_new, dw = grads[0], grads[1]
        db = grads[2] if b is not None else None
    if dx is not None:
        dx.copy_(dx_new)
        dx_new = dx
    dw = dw.to(weight.dtype)
    db = db.to(bias.dtype) if db is not None else None
    return dx_new, dw, db, None


setattr(mamba_causal_conv1d_fwd_function, _MAMBA_FUSED_MARK, True)
setattr(mamba_causal_conv1d_bwd_function, _MAMBA_FUSED_MARK, True)


def mamba_fused_uses_triton_conv():
    module = sys.modules.get("mamba_ssm.ops.triton.ssd_combined")
    return module is not None and getattr(vars(module).get("causal_conv1d_fwd_function"), _MAMBA_FUSED_MARK, False)


def install_mamba_ssm_triton_conv():
    """Point mamba_ssm's fused split path at the Triton conv when causal_conv1d is unusable.
    True when installed (or already installed); False leaves the caller's fallback in charge."""
    if _disabled() or _real_causal_conv1d_available():
        return False
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import triton  # noqa: F401
        if importlib.util.find_spec("mamba_ssm") is None:
            return False
        ssd_combined = importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
    except Exception:
        return False
    names = vars(ssd_combined)
    if getattr(names.get("causal_conv1d_fwd_function"), _MAMBA_FUSED_MARK, False):
        return True
    if names.get("causal_conv1d_fwd_function", None) is not None or names.get("causal_conv1d_bwd_function", None) is not None:
        return False  # a real (or other) implementation is bound: leave it
    if "causal_conv1d_fwd_function" not in names:
        return False  # not the mamba_ssm layout this was written against
    ssd_combined.causal_conv1d_fwd_function = mamba_causal_conv1d_fwd_function
    ssd_combined.causal_conv1d_bwd_function = mamba_causal_conv1d_bwd_function
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: mamba_ssm fused split path uses the Triton causal conv")
    return True


def patch_gdn_causal_conv1d():
    _refresh_kill_switch()
    if _disabled() or _real_causal_conv1d_available():
        return
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return
    try:
        import triton  # noqa: F401
    except Exception:
        return

    patched_decorator = _patch_hub_decorator()
    names = [f"transformers.models.{p}.modeling_{p}" for p in _GDN_MODELING]
    if patched_decorator is not None:
        names += [f"transformers.models.{p}.modeling_{p}" for p in _MAMBA2_MODELING]
    if patched_decorator is None:
        # transformers < 5.15 has no decorator to hook: import modeling modules now.
        for name in names:
            try:
                importlib.import_module(name)
            except Exception:
                pass
    # On >= 5.15 modeling modules not imported yet are covered by the patched decorator.
    names += sorted(
        name for name in list(sys.modules)
        if name.rsplit(".", 1)[-1].startswith("unsloth_compiled_module_")
        and name.rsplit(".", 1)[-1] in _GDN_LEAVES
    )
    rebound = [
        name for name in names
        if sys.modules.get(name) is not None and _rebind_module(sys.modules[name], patched_decorator)
    ]
    if rebound and UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: Triton causal_conv1d for the GatedDeltaNet short conv on {', '.join(rebound)}")


TEMPORARY_PATCHES.append(patch_gdn_causal_conv1d)
