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

"""Triton causal depthwise conv (+ SiLU) for GatedDeltaNet when `causal_conv1d` is missing.

Replaces the transformers `F.conv1d` fallback (Qwen3.5/3.6, Qwen3-Next), reading and
writing channel-last directly to skip the transpose copies. Layout follows fla's
`causal_conv1d` (fla-org/flash-linear-attention, MIT), without varlen / initial state.
Numerics match torch: fp32 accumulate, pre-activation and SiLU grad rounded to input dtype.
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
_GDN_LEAVES = frozenset(
    f"{prefix}_{package}" for package in _GDN_MODELING
    for prefix in ("modeling", "unsloth_compiled_module")
)

GDN_CAUSAL_CONV1D_STATS = {"triton_fwd": 0, "triton_bwd": 0, "fallback": 0}

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_WIDTH = 8
_ACTIVATIONS = (None, "silu", "swish")

_kernels = None
_broken = False


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
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _causal_conv1d_fwd_kernel(
        x, weight, bias, y,
        T, D,
        stride_x_b, stride_x_t, stride_x_d,
        W: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_t = i_t * BT + tl.arange(0, BT)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        # int64: channel-first x / dy has stride_d = T, and (D - 1) * T can pass 2**31.
        o_d64 = o_d.to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        acc = tl.zeros((BT, BD), dtype = tl.float32)
        for i_w in tl.static_range(W):
            # Output t sees x[t - (W - 1) + i_w] (left zero padding of W - 1).
            s_t = o_t - (W - 1) + i_w
            m = ((s_t >= 0) & (s_t < T))[:, None] & m_d[None, :]
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
        p_y = y + tl.cast(i_b, tl.int64) * T * D + o_t.to(tl.int64)[:, None] * D + o_d[None, :]
        tl.store(p_y, acc.to(y.dtype.element_ty), mask = (o_t < T)[:, None] & m_d[None, :])

    @triton.jit
    def _causal_conv1d_bwd_kernel(
        x, weight, bias, dy, dx, dw, db,
        T, D, NT,
        stride_x_b, stride_x_t, stride_x_d,
        stride_dy_b, stride_dy_t, stride_dy_d,
        W: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
        NEED_DX: tl.constexpr,
        NEED_DW: tl.constexpr,
        NEED_DB: tl.constexpr,
    ):
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_t = i_t * BT + tl.arange(0, BT)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        o_d64 = o_d.to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_b
        if HAS_BIAS:
            b_bias = tl.load(bias + o_d, mask = m_d, other = 0).to(tl.float32)

        b_dx = tl.zeros((BT, BD), dtype = tl.float32)
        # dx[s] += w[W - 1 - k] * g[s + k], g = d(loss)/d(pre-activation).
        for k in tl.static_range(W):
            r_t = o_t + k
            m_r = r_t < T
            b_g = tl.load(
                p_dy + r_t.to(tl.int64)[:, None] * stride_dy_t + o_d64[None, :] * stride_dy_d,
                mask = m_r[:, None] & m_d[None, :], other = 0,
            ).to(tl.float32)
            if ACTIVATION:
                b_pre = tl.zeros((BT, BD), dtype = tl.float32)
                for j in tl.static_range(W):
                    s_t = r_t - (W - 1) + j
                    m = ((s_t >= 0) & (s_t < T))[:, None] & m_d[None, :]
                    b_xs = tl.load(
                        p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d64[None, :] * stride_x_d,
                        mask = m, other = 0,
                    ).to(tl.float32)
                    b_wj = tl.load(weight + o_d * W + j, mask = m_d, other = 0).to(tl.float32)
                    b_pre += b_xs * b_wj[None, :]
                if HAS_BIAS:
                    b_pre += b_bias[None, :]
            if ACTIVATION:
                b_pre = b_pre.to(dx.dtype.element_ty).to(tl.float32)
                b_sig = tl.sigmoid(b_pre)
                b_g = b_g * b_sig * (1 + b_pre * (1 - b_sig))
                # torch's silu_backward stores the gradient in the input dtype.
                b_g = b_g.to(dx.dtype.element_ty).to(tl.float32)
            if k == 0:
                if NEED_DB:
                    tl.store(db + (tl.cast(i_b, tl.int64) * NT + i_t) * D + o_d, tl.sum(b_g, 0), mask = m_d)
                if NEED_DW:
                    # dw[j] = sum_t g[t] * x[t - (W - 1) + j] over this tile's rows.
                    for j in tl.static_range(W):
                        s_t = o_t - (W - 1) + j
                        m = ((s_t >= 0) & (s_t < T))[:, None] & m_d[None, :]
                        b_xs = tl.load(
                            p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d64[None, :] * stride_x_d,
                            mask = m, other = 0,
                        ).to(tl.float32)
                        tl.store(
                            dw + ((tl.cast(i_b, tl.int64) * NT + i_t) * D + o_d) * W + j,
                            tl.sum(b_g * b_xs, 0), mask = m_d,
                        )
            if NEED_DX:
                b_wk = tl.load(weight + o_d * W + (W - 1 - k), mask = m_d, other = 0).to(tl.float32)
                b_dx += b_g * b_wk[None, :]
        if NEED_DX:
            p_dx = dx + tl.cast(i_b, tl.int64) * T * D + o_t.to(tl.int64)[:, None] * D + o_d[None, :]
            tl.store(p_dx, b_dx.to(dx.dtype.element_ty), mask = (o_t < T)[:, None] & m_d[None, :])

    _kernels = (_causal_conv1d_fwd_kernel, _causal_conv1d_bwd_kernel)
    return _kernels


# (BT, BD, num_warps). Small: backward recomputes W pre-activations per tile and spills.
_FWD_CONFIG = (32, 64, 2)
_BWD_CONFIG = (16, 64, 2)
# CUDA caps grid dims 1 and 2 at 65535: T tiles ride dim 1, batch rides dim 2.
_MAX_GRID_YZ = 65535


def _launch_fwd(x_btd, weight, bias, activation):
    fwd, _ = _build_kernels()
    _BT, _BD, _NUM_WARPS = _FWD_CONFIG
    B, T, D = x_btd.shape
    W = weight.shape[1]
    y = torch.empty((B, T, D), dtype = x_btd.dtype, device = x_btd.device)
    grid = ((D + _BD - 1) // _BD, (T + _BT - 1) // _BT, B)
    with torch.cuda.device(x_btd.device):
        fwd[grid](
            x_btd, weight, bias if bias is not None else weight, y,
            T, D,
            x_btd.stride(0), x_btd.stride(1), x_btd.stride(2),
            W = W, BT = _BT, BD = _BD,
            HAS_BIAS = bias is not None,
            ACTIVATION = activation is not None,
            num_warps = _NUM_WARPS,
        )
    return y


def _launch_bwd(x_btd, weight, bias, dy_btd, activation, need_dx, need_dw, need_db):
    _, bwd = _build_kernels()
    _BT, _BD, _NUM_WARPS = _BWD_CONFIG
    B, T, D = x_btd.shape
    W = weight.shape[1]
    NT = (T + _BT - 1) // _BT
    dx = torch.empty((B, T, D), dtype = x_btd.dtype, device = x_btd.device) if need_dx else None
    dw = torch.empty((B * NT, D, W), dtype = torch.float32, device = x_btd.device) if need_dw else None
    db = torch.empty((B * NT, D), dtype = torch.float32, device = x_btd.device) if need_db else None
    grid = ((D + _BD - 1) // _BD, NT, B)
    with torch.cuda.device(x_btd.device):
        bwd[grid](
            x_btd, weight, bias if bias is not None else weight, dy_btd,
            dx if dx is not None else x_btd,
            dw if dw is not None else weight,
            db if db is not None else weight,
            T, D, NT,
            x_btd.stride(0), x_btd.stride(1), x_btd.stride(2),
            dy_btd.stride(0), dy_btd.stride(1), dy_btd.stride(2),
            W = W, BT = _BT, BD = _BD,
            HAS_BIAS = bias is not None,
            ACTIVATION = activation is not None,
            NEED_DX = need_dx, NEED_DW = need_dw, NEED_DB = need_db,
            num_warps = _NUM_WARPS,
        )
    if dw is not None:
        dw = dw.sum(0).to(weight.dtype)
    if db is not None:
        db = db.sum(0).to(bias.dtype)
    return dx, dw, db


class _CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bdt, weight, bias, activation):
        y_btd = _launch_fwd(x_bdt.transpose(1, 2), weight, bias, activation)
        ctx.save_for_backward(x_bdt, weight, bias)
        ctx.activation = activation
        GDN_CAUSAL_CONV1D_STATS["triton_fwd"] += 1
        return y_btd.transpose(1, 2)

    @staticmethod
    def backward(ctx, dy_bdt):
        x_bdt, weight, bias = ctx.saved_tensors
        dx, dw, db = _launch_bwd(
            x_bdt.transpose(1, 2), weight, bias, dy_bdt.transpose(1, 2),
            ctx.activation, *ctx.needs_input_grad[:3],
        )
        GDN_CAUSAL_CONV1D_STATS["triton_bwd"] += 1
        return (dx.transpose(1, 2) if dx is not None else None), dw, db, None


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


def _eligible(x, weight, bias, activation):
    if not (
        x.is_cuda and x.dim() == 3 and x.numel() > 0 and x.dtype in _SUPPORTED_DTYPES
        and weight.dim() == 2 and weight.dtype == x.dtype and weight.device == x.device
        and weight.is_contiguous() and weight.shape[0] == x.shape[1]
        and 1 <= weight.shape[1] <= _MAX_WIDTH and activation in _ACTIVATIONS
        and (bias is None or (
            bias.dtype == x.dtype and bias.device == x.device
            and bias.shape == (x.shape[1],) and bias.is_contiguous()
        ))
    ):
        return False
    # Both launches must fit the grid: the backward tiles T finer and has no fallback.
    min_bt = min(_FWD_CONFIG[0], _BWD_CONFIG[0])
    return x.shape[0] <= _MAX_GRID_YZ and (x.shape[2] + min_bt - 1) // min_bt <= _MAX_GRID_YZ


def triton_causal_conv1d(x, weight, bias = None, activation = None):
    """x (B, D, T), weight (D, W) -> (B, D, T) stored channel-last."""
    return _CausalConv1dFunction.apply(x, weight, bias, activation)


def _try_fast(x, weight, bias, activation):
    """Run the Triton kernel, or return None to use the torch fallback."""
    global _broken
    # Under torch.compile (decode regions) trace the torch path, never the launcher.
    if _broken or torch.compiler.is_compiling() or _disabled():
        return None
    if not _eligible(x, weight, bias, activation):
        return None
    try:
        return triton_causal_conv1d(x, weight, bias, activation)
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


def _make_hub_dispatch(torch_function):
    """Wrap the transformers torch fallback (which ignores its kwargs)."""
    @functools.wraps(torch_function)
    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        out = _try_fast(hidden_states, weight, bias, activation)
        if out is not None:
            return out
        _count_fallback()
        return torch_function(hidden_states, weight, bias, activation, **kwargs)

    setattr(causal_conv1d_fn, _MARK, True)
    return causal_conv1d_fn


def _causal_conv1d_reference_seq_idx(x, weight, bias, activation, seq_idx):
    """Torch causal conv whose taps do not cross `seq_idx` (B, T) boundaries."""
    import torch.nn.functional as F
    W, T = weight.shape[-1], x.shape[-1]
    acc = torch.promote_types(weight.dtype, torch.float32)
    xp = F.pad(x.to(acc), (W - 1, 0))
    sp = F.pad(seq_idx.to(torch.int64), (W - 1, 0), value = -1)
    w = weight.to(acc)
    out = torch.zeros_like(xp[:, :, :T])
    for j in range(W):
        same = (sp[:, j:j + T] == seq_idx).unsqueeze(1)
        out = out + xp[:, :, j:j + T] * w[:, j].view(1, -1, 1) * same
    if bias is not None:
        out = out + bias.to(acc).view(1, -1, 1)
    out = out.to(weight.dtype)
    if activation is not None:
        out = _act(activation)(out)
    return out.to(x.dtype)


def _legacy_causal_conv1d_fn(x, weight, bias = None, activation = None, **kwargs):
    """`causal_conv1d_fn` stand-in for transformers < 5.15 (used only when not None).
    `seq_idx` is deliberately not named: the hybrid packing gate enables packing when the
    signature names it. If passed (5.9 to 5.14), a per-segment torch path honours it."""
    seq_idx = kwargs.pop("seq_idx", None)
    unsupported = sorted(k for k, v in kwargs.items() if v is not None and v is not False)
    if unsupported:
        raise NotImplementedError(
            f"Unsloth: the torch causal_conv1d fallback does not support {unsupported}."
        )
    if seq_idx is None:
        out = _try_fast(x, weight, bias, activation)
        if out is not None:
            return out
    _count_fallback()
    if seq_idx is not None:
        return _causal_conv1d_reference_seq_idx(x, weight, bias, activation, seq_idx)
    return causal_conv1d_reference(x, weight, bias, activation)


setattr(_legacy_causal_conv1d_fn, _MARK, True)


def _is_gdn_function(torch_function):
    module = getattr(torch_function, "__module__", None) or ""
    return module.rsplit(".", 1)[-1] in _GDN_LEAVES


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
        if func_name != "causal_conv1d_fn" or package != "causal_conv1d":
            return decorator
        # A usable package still wins: the hub wrapper only calls this torch fallback without it.
        return lambda fn: decorator(_make_hub_dispatch(fn) if _is_gdn_function(fn) else fn)

    setattr(use_kernel_func_from_hub_with_fallback, _HUB_MARK, True)
    hub_kernels.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    if getattr(integrations, "use_kernel_func_from_hub_with_fallback", None) is original:
        integrations.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    return use_kernel_func_from_hub_with_fallback


def _rebind_module(module, patched_decorator):
    """Rebind an imported module; True if it now dispatches to Triton. Re-decorating
    also replaces a wrapper that captured a since-disabled package kernel."""
    names = vars(module)
    if patched_decorator is not None and "use_kernel_func_from_hub_with_fallback" in names:
        module.use_kernel_func_from_hub_with_fallback = patched_decorator
    if "causal_conv1d_fn" not in names:
        return False
    current = module.causal_conv1d_fn
    if current is None:
        # transformers < 5.15: None unless the package imported; copied in layer __init__.
        module.causal_conv1d_fn = _legacy_causal_conv1d_fn
        return True
    if _is_marked(current):
        return True
    torch_function = getattr(current, "__wrapped__", None)
    if torch_function is None or patched_decorator is None:
        return False
    module.causal_conv1d_fn = patched_decorator("causal_conv1d_fn", "causal_conv1d")(torch_function)
    return _is_marked(module.causal_conv1d_fn)


def patch_gdn_causal_conv1d():
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
