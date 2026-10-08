# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Triton short causal conv for the GatedDeltaNet mixer when `causal_conv1d` is missing.

Qwen3.5 / Qwen3.6 (dense and MoE) and Qwen3-Next run `in_proj_qkv` and then a
depthwise causal conv (kernel 4) + SiLU over the channel dim. Without the
`causal-conv1d` wheel transformers falls back to `F.conv1d` on a channel-first
`(B, D, T)` view of the channel-last `(B, T, D)` projection output. The torch
depthwise conv wants a channel-first contiguous input, so every call pays a
transposing `.contiguous()` on the input, another on `grad_output` in backward,
runs the slow `conv_depthwise2d` kernels, and hands a channel-first output to
`chunk_gated_delta_rule`, whose input guard transposes it back.

This module replaces only that torch fallback with a Triton kernel that reads
the channel-last input through its strides and writes a channel-last output, so
the copies go away and the conv itself becomes a single memory-bound pass. The
kernel layout follows fla's `causal_conv1d` (fla-org/flash-linear-attention,
MIT); it is reimplemented here self-contained without varlen / initial state
support because the call site never passes them.

Numerics follow the torch path: fp32 accumulation, the pre-activation is
rounded to the input dtype before SiLU, and in backward the SiLU gradient is
rounded to the input dtype before the conv transpose, as the separate torch ops
do.

Env: ``UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D=1`` keeps the torch fallback (read
per call). The real `causal_conv1d` package always wins when importable.
"""

__all__ = [
    "patch_gdn_causal_conv1d",
    "triton_causal_conv1d",
    "causal_conv1d_reference",
    "GDN_CAUSAL_CONV1D_STATS",
]

import os
import sys
import functools
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

# Modeling packages whose GatedDeltaNet uses the depthwise causal conv fallback.
_GDN_MODELING = ("qwen3_5", "qwen3_5_moe", "qwen3_next")

# Engagement counters (Triton forward calls / backward calls / torch fallbacks).
GDN_CAUSAL_CONV1D_STATS = {"triton_fwd": 0, "triton_bwd": 0, "fallback": 0}

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_WIDTH = 8
_ACTIVATIONS = (None, "silu", "swish")

_kernels = None


def _disabled():
    return os.environ.get(_KILL_SWITCH, "0") == "1"


def _real_causal_conv1d_available():
    try:
        return importlib.util.find_spec("causal_conv1d") is not None
    except Exception:
        return False


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
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        acc = tl.zeros((BT, BD), dtype = tl.float32)
        for i_w in tl.static_range(W):
            # Output t sees x[t - (W - 1) + i_w] (left zero padding of W - 1).
            s_t = o_t - (W - 1) + i_w
            m = ((s_t >= 0) & (s_t < T))[:, None] & m_d[None, :]
            b_x = tl.load(
                p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
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
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_b
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_b
        if HAS_BIAS:
            b_bias = tl.load(bias + o_d, mask = m_d, other = 0).to(tl.float32)

        b_dx = tl.zeros((BT, BD), dtype = tl.float32)
        # k = how far ahead the output row is: dx[s] += w[W - 1 - k] * g[s + k],
        # where g = d(loss)/d(pre-activation).
        for k in tl.static_range(W):
            r_t = o_t + k
            m_r = r_t < T
            b_g = tl.load(
                p_dy + r_t.to(tl.int64)[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
                mask = m_r[:, None] & m_d[None, :], other = 0,
            ).to(tl.float32)
            if ACTIVATION:
                # Recompute the pre-activation for rows r_t.
                b_pre = tl.zeros((BT, BD), dtype = tl.float32)
                for j in tl.static_range(W):
                    s_t = r_t - (W - 1) + j
                    m = ((s_t >= 0) & (s_t < T))[:, None] & m_d[None, :]
                    b_xs = tl.load(
                        p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
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
                    tl.store(db + tl.cast(i_b * NT + i_t, tl.int64) * D + o_d, tl.sum(b_g, 0), mask = m_d)
                if NEED_DW:
                    # dw[j] = sum_t g[t] * x[t - (W - 1) + j] over this tile's rows.
                    for j in tl.static_range(W):
                        s_t = o_t - (W - 1) + j
                        m = ((s_t >= 0) & (s_t < T))[:, None] & m_d[None, :]
                        b_xs = tl.load(
                            p_x + s_t.to(tl.int64)[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
                            mask = m, other = 0,
                        ).to(tl.float32)
                        tl.store(
                            dw + (tl.cast(i_b * NT + i_t, tl.int64) * D + o_d) * W + j,
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


# (BT, BD, num_warps). Small tiles: the backward recomputes W shifted pre-activations
# per tile and spills with larger ones. Picked on B200 (bf16, D 6144, T 4096); the
# forward is memory bound and flat across tile sizes.
_FWD_CONFIG = (32, 64, 2)
_BWD_CONFIG = (16, 64, 2)


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
        # x_bdt is (B, D, T); usually a transposed view of a (B, T, D) tensor.
        x_btd = x_bdt.transpose(1, 2)
        y_btd = _launch_fwd(x_btd, weight, bias, activation)
        ctx.save_for_backward(x_bdt, weight, bias)
        ctx.activation = activation
        GDN_CAUSAL_CONV1D_STATS["triton_fwd"] += 1
        return y_btd.transpose(1, 2)

    @staticmethod
    def backward(ctx, dy_bdt):
        x_bdt, weight, bias = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        need_db = need_db and bias is not None
        if not (need_dx or need_dw or need_db):
            return None, None, None, None
        dx, dw, db = _launch_bwd(
            x_bdt.transpose(1, 2), weight, bias, dy_bdt.transpose(1, 2),
            ctx.activation, need_dx, need_dw, need_db,
        )
        GDN_CAUSAL_CONV1D_STATS["triton_bwd"] += 1
        return (
            dx.transpose(1, 2) if dx is not None else None,
            dw, db, None,
        )


def causal_conv1d_reference(x, weight, bias = None, activation = None):
    """The transformers torch fallback: x (B, D, T), weight (D, W)."""
    import torch.nn.functional as F
    D, T = x.shape[1], x.shape[2]
    out = F.conv1d(
        x.to(weight.dtype), weight.unsqueeze(1), bias,
        padding = weight.shape[-1] - 1, groups = D,
    )[:, :, :T]
    if activation is not None:
        out = F.silu(out)
    return out.to(x.dtype)


def _eligible(x, weight, bias, activation):
    if not (isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor)):
        return False
    if x.device.type != "cuda" or weight.device != x.device:
        return False
    if getattr(torch.version, "hip", None) is not None:
        # Not validated on ROCm yet.
        return False
    if x.dim() != 3 or weight.dim() != 2:
        return False
    if x.dtype not in _SUPPORTED_DTYPES or weight.dtype != x.dtype:
        return False
    if weight.shape[0] != x.shape[1] or not (1 <= weight.shape[1] <= _MAX_WIDTH):
        return False
    if x.shape[2] == 0 or x.shape[0] == 0:
        return False
    if not weight.is_contiguous():
        return False
    if bias is not None and (
        not isinstance(bias, torch.Tensor) or bias.dtype != x.dtype
        or bias.device != x.device or bias.shape != (x.shape[1],) or not bias.is_contiguous()
    ):
        return False
    if activation not in _ACTIVATIONS:
        return False
    return True


def triton_causal_conv1d(x, weight, bias = None, activation = None):
    """Causal depthwise conv over x (B, D, T) with weight (D, W); returns (B, D, T)
    stored channel-last. Same semantics as the transformers torch fallback."""
    return _CausalConv1dFunction.apply(x, weight, bias, activation)


def _try_fast(x, weight, bias, activation, extra):
    """Run the Triton kernel, or return None to use the torch fallback."""
    # Under torch.compile (decode regions) trace the torch path, never the launcher.
    try:
        if torch.compiler.is_compiling():
            return None
    except Exception:
        return None
    if _disabled():
        return None
    # Anything beyond the plain call (seq_idx, initial_states, ...) keeps the fallback.
    for value in extra.values():
        if value is not None and value is not False:
            return None
    if not _eligible(x, weight, bias, activation):
        return None
    try:
        return triton_causal_conv1d(x, weight, bias, activation)
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Triton causal_conv1d failed, using torch: {e}")
        return None


def _make_hub_dispatch(torch_function):
    """Wrap the transformers torch fallback `causal_conv1d_fn(hidden_states, weight,
    bias=None, activation=None, **kwargs)`."""
    if getattr(torch_function, _MARK, False):
        return torch_function

    @functools.wraps(torch_function)
    def causal_conv1d_fn(hidden_states, weight, bias = None, activation = None, **kwargs):
        out = _try_fast(hidden_states, weight, bias, activation, {})
        if out is not None:
            return out
        GDN_CAUSAL_CONV1D_STATS["fallback"] += 1
        return torch_function(hidden_states, weight, bias, activation, **kwargs)

    setattr(causal_conv1d_fn, _MARK, True)
    return causal_conv1d_fn


def _legacy_causal_conv1d_fn(
    x, weight, bias = None, seq_idx = None, initial_states = None,
    return_final_states = False, final_states_out = None, activation = None,
):
    """Stands in for `causal_conv1d.causal_conv1d_fn` on transformers < 5.16, where
    the GatedDeltaNet calls `self.causal_conv1d_fn(x=, weight=, bias=, activation=,
    seq_idx=None)` only when the package imported."""
    extra = dict(
        seq_idx = seq_idx, initial_states = initial_states,
        return_final_states = return_final_states, final_states_out = final_states_out,
    )
    out = _try_fast(x, weight, bias, activation, extra)
    if out is not None:
        return out
    GDN_CAUSAL_CONV1D_STATS["fallback"] += 1
    if any(v is not None and v is not False for v in extra.values()):
        raise NotImplementedError(
            "Unsloth: the torch causal_conv1d fallback does not support "
            "seq_idx / initial_states / final states."
        )
    return causal_conv1d_reference(x, weight, bias, activation)


setattr(_legacy_causal_conv1d_fn, _MARK, True)


def _is_gdn_function(torch_function):
    module = getattr(torch_function, "__module__", "") or ""
    for package in _GDN_MODELING:
        if module.endswith(f"modeling_{package}") or module.endswith(f"unsloth_compiled_module_{package}"):
            return True
    return False


def _patch_hub_decorator():
    """Wrap `use_kernel_func_from_hub_with_fallback` so a GatedDeltaNet
    `causal_conv1d_fn` decorated later (including Unsloth's compiled copies, which
    import the decorator from the modeling module) gets the Triton dispatch."""
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return None
    original = getattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", None)
    if original is None:
        return None
    if getattr(original, _HUB_MARK, False):
        return original

    @functools.wraps(original)
    def use_kernel_func_from_hub_with_fallback(func_name, package, *args, **kwargs):
        decorator = original(func_name, package, *args, **kwargs)
        if func_name != "causal_conv1d_fn" or package != "causal_conv1d":
            return decorator

        def wrap(torch_function):
            if _real_causal_conv1d_available() or not _is_gdn_function(torch_function):
                return decorator(torch_function)
            return decorator(_make_hub_dispatch(torch_function))

        return wrap

    setattr(use_kernel_func_from_hub_with_fallback, _HUB_MARK, True)
    hub_kernels.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    try:
        import transformers.integrations as integrations
        if getattr(integrations, "use_kernel_func_from_hub_with_fallback", None) is original:
            integrations.use_kernel_func_from_hub_with_fallback = use_kernel_func_from_hub_with_fallback
    except Exception:
        pass
    return use_kernel_func_from_hub_with_fallback


def _rebind_module(module, patched_decorator):
    """Rebind one already-imported modeling (or compiled) module. Returns True if
    its `causal_conv1d_fn` now dispatches to Triton."""
    if module is None:
        return False
    if patched_decorator is not None and "use_kernel_func_from_hub_with_fallback" in vars(module):
        current = module.use_kernel_func_from_hub_with_fallback
        if not getattr(current, _HUB_MARK, False):
            module.use_kernel_func_from_hub_with_fallback = patched_decorator

    if "causal_conv1d_fn" not in vars(module):
        return False
    current = module.causal_conv1d_fn
    if current is None:
        # transformers < 5.16: None unless the package imported; the layer copies
        # this global onto `self.causal_conv1d_fn` in __init__.
        module.causal_conv1d_fn = _legacy_causal_conv1d_fn
        return True
    if getattr(current, _MARK, False):
        return True
    torch_function = getattr(current, "__wrapped__", None)
    if torch_function is None or patched_decorator is None:
        return False
    # Only replace a wrapper that resolved to its own torch fallback.
    implementation = None
    code = getattr(current, "__code__", None)
    for name, cell in zip(getattr(code, "co_freevars", ()), getattr(current, "__closure__", None) or ()):
        if name == "implementation":
            try:
                implementation = cell.cell_contents
            except ValueError:
                implementation = None
    if implementation is not None and implementation is not torch_function:
        return False
    if implementation is None and _real_causal_conv1d_available():
        return False
    try:
        module.causal_conv1d_fn = patched_decorator("causal_conv1d_fn", "causal_conv1d")(torch_function)
    except Exception:
        return False
    return getattr(module.causal_conv1d_fn, _MARK, False) or _wraps_marked(module.causal_conv1d_fn)


def _wraps_marked(fn):
    seen = 0
    while fn is not None and seen < 5:
        if getattr(fn, _MARK, False):
            return True
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and getattr(value, _MARK, False):
                return True
        fn = getattr(fn, "__wrapped__", None)
        seen += 1
    return False


def patch_gdn_causal_conv1d():
    if _disabled():
        return False
    if _real_causal_conv1d_available():
        return False
    try:
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
            return False
        import triton  # noqa: F401
    except Exception:
        return False

    patched_decorator = _patch_hub_decorator()
    rebound = []
    for package in _GDN_MODELING:
        name = f"transformers.models.{package}.modeling_{package}"
        # Not imported yet: the patched decorator covers its first import.
        if _rebind_module(sys.modules.get(name), patched_decorator):
            rebound.append(name)
    for name in sorted(list(sys.modules)):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in tuple(f"unsloth_compiled_module_{p}" for p in _GDN_MODELING):
            continue
        if _rebind_module(sys.modules.get(name), patched_decorator):
            rebound.append(name)
    if rebound and UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: Triton causal_conv1d for the GatedDeltaNet short conv on {', '.join(rebound)}")
    return bool(rebound) or patched_decorator is not None


TEMPORARY_PATCHES.append(patch_gdn_causal_conv1d)
