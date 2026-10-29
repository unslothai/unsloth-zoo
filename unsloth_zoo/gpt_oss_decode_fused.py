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
  rope_decode       one-token rotary embedding of q and k in one launch. Bit-identical to the compiled
                    region: bf16 / fp16 products are exact in fp32, so each output rounds once either way.
  static_cache_update  one-token update of transformers' StaticLayer / StaticSlidingWindowLayer: the
                    sliding roll + host index tensor (a pageable copy, so a stream sync per layer) +
                    index_put + copy becomes cat + copy; the full layer's arange + add drop out.
                    Pure data movement, bit-identical; torch ops, so CUDA, ROCm and CPU.
Same math and rounding points as the eager code (the residual sum is rounded to the input dtype
before the norm, the sink scale is rounded to the output dtype before the multiply). The sink
epilogue is bit-identical on CUDA; the norm differs only in the order of its fp32 sum. Inference only: callers check
fused_decode_ok() (NVIDIA CUDA tensors, Triton present, no autograd, not tracing).
Kill switch: UNSLOTH_GPT_OSS_FUSED_DECODE=0, read at import.
"""

__all__ = [
    "fused_decode_ok",
    "add_rms_norm",
    "sink_scale_output",
    "static_cache_update",
    "rope_decode",
    "FUSED_DECODE_CALLS",
]

import os

import torch

from .log import logger
from .triton_fast_launch import launch as _fast_launch

try:
    import torch._dynamo.eval_frame as _eval_frame
except Exception:
    _eval_frame = None

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
# The device math library's exp / rsqrt (what aten's CUDA kernels call), not Triton's approximations:
# with an IEEE division the sink scale is bit-identical to torch.sigmoid.
_libdevice = None
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"  # CPU tests: no libdevice there
if HAS_TRITON and not _INTERPRET:
    try:
        from triton.language.extra import libdevice as _libdevice
    except Exception:
        _libdevice = None

# NVIDIA only for now: the libdevice exp / rsqrt and rounding were verified on sm75 / sm80 / sm120, not on ROCm.
_ENABLED = HAS_TRITON and torch.version.hip is None and os.environ.get("UNSLOTH_GPT_OSS_FUSED_DECODE", "1") != "0"
# Engagement counters (tests and benchmarks read them).
FUSED_DECODE_CALLS = {"add_rms_norm": 0, "sink_scale_output": 0, "static_cache_update": 0, "rope_decode": 0}
_MAX_COLS = 32768
# aten rounds every multiply before the add that follows it; contracting them into FMAs would not.
_NO_FMA = {} if _INTERPRET else {"enable_fp_fusion": False}


def _disable(error):
    """A Triton build that cannot compile or launch these kernels: eager ops from now on (the launch
    failed before writing anything, so the caller's eager path still sees the original inputs)."""
    global _ENABLED
    if _ENABLED:
        _ENABLED = False
        logger.warning(f"Unsloth: gpt-oss fused decode kernels disabled, using the eager ops ({type(error).__name__}: {error})")
    return None


def fused_decode_ok(x):
    """True when the fused kernels may replace the eager ops for this tensor."""
    return (
        _ENABLED
        and x.is_cuda
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and not torch.jit.is_tracing()
    )


if HAS_TRITON:
    @triton.jit
    def _add_rms_norm_kernel(
        X, R, W, Y, S, N, stride_x, stride_r, stride_y, stride_s, eps, inv_n,
        HAS_RES: tl.constexpr, BLOCK: tl.constexpr, EXACT: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < N
        x = tl.load(X + row * stride_x + cols, mask = ok, other = 0.0)
        if HAS_RES:
            r = tl.load(R + row * stride_r + cols, mask = ok, other = 0.0)
            # Eager `hidden_states += residual`: one rounding to the activation dtype.
            x = (x.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
            tl.store(S + row * stride_s + cols, x, mask = ok)
        xf = x.to(tl.float32)
        # aten's mean is sum * (1 / N); only the order of the fp32 sum differs from it.
        var = tl.sum(xf * xf, axis = 0) * inv_n + eps
        if EXACT:
            y = xf * _libdevice.rsqrt(var)
        else:
            y = xf * tl.rsqrt(var)
        w = tl.load(W + cols, mask = ok, other = 0.0).to(tl.float32)
        tl.store(Y + row * stride_y + cols, (y * w).to(Y.dtype.element_ty), mask = ok)

    @triton.jit
    def _sink_scale_kernel(
        A, LSE, SINK, OUT, H, Q, D,
        sa_b, sa_h, sa_q, sl_b, sl_h, sl_q, so_b, so_q, so_h,
        BLOCK_D: tl.constexpr, EXACT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q = pid % Q
        bh = pid // Q
        h = bh % H
        b = (bh // H).to(tl.int64)
        lse = tl.load(LSE + b * sl_b + h * sl_h + q * sl_q).to(tl.float32)
        sink = tl.load(SINK + h).to(tl.float32)
        # Eager: sigmoid in fp32, then the scale is cast to the output dtype before the multiply.
        if EXACT:
            scale = tl.math.div_rn(1.0, 1.0 + _libdevice.exp(sink - lse))
        else:
            scale = 1.0 / (1.0 + tl.exp(sink - lse))
        scale = scale.to(OUT.dtype.element_ty).to(tl.float32)
        cols = tl.arange(0, BLOCK_D)
        ok = cols < D
        a = tl.load(A + b * sa_b + h * sa_h + q * sa_q + cols, mask = ok, other = 0.0).to(tl.float32)
        tl.store(OUT + b * so_b + q * so_q + h * so_h + cols, (a * scale).to(OUT.dtype.element_ty), mask = ok)

    @triton.jit
    def _rope_decode_kernel(
        Q, K, COS, SIN, QO, KO, HQ, HK, HALF,
        sq_b, sq_h, sk_b, sk_h, sc_b, ss_b, sqo_b, sqo_h, sko_b, sko_h,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = (pid // (HQ + HK)).to(tl.int64)
        h = (pid % (HQ + HK)).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < HALF
        c = tl.load(COS + b * sc_b + cols, mask = ok, other = 0.0).to(tl.float32)
        s = tl.load(SIN + b * ss_b + cols, mask = ok, other = 0.0).to(tl.float32)
        if h < HQ:
            src = Q + b * sq_b + h * sq_h
            dst = QO + b * sqo_b + h * sqo_h
        else:
            src = K + b * sk_b + (h - HQ) * sk_h
            dst = KO + b * sko_b + (h - HQ) * sko_h
        x1 = tl.load(src + cols, mask = ok, other = 0.0).to(tl.float32)
        x2 = tl.load(src + HALF + cols, mask = ok, other = 0.0).to(tl.float32)
        tl.store(dst + cols, (x1 * c - x2 * s).to(dst.dtype.element_ty), mask = ok)
        tl.store(dst + HALF + cols, (x2 * c + x1 * s).to(dst.dtype.element_ty), mask = ok)


def add_rms_norm(x, residual, weight, eps):
    """(RMSNorm(x + residual), x + residual) in one launch, the sum in a new tensor (x is never written,
    so an exception after the launch leaves the caller's eager fallback correct); residual None returns
    (RMSNorm(x), x). x [..., N] contiguous; None when disabled or the shapes or devices are not supported
    (callers then run the eager ops)."""
    if not _ENABLED or torch.jit.is_tracing():
        return None
    N = x.shape[-1]
    if (
        N > _MAX_COLS or not x.is_contiguous() or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.shape != (N,) or weight.device != x.device
    ):
        return None
    if not weight.is_contiguous():  # the kernel reads it with unit stride
        weight = weight.contiguous()
    if residual is not None and (
        residual.shape != x.shape or residual.dtype != x.dtype or not residual.is_contiguous()
        or residual.device != x.device
    ):
        return None
    rows = x.numel() // N
    y = torch.empty_like(x)
    s = x if residual is None else torch.empty_like(x)
    if rows == 0:
        return y, s
    BLOCK = triton.next_power_of_2(N)
    try:
        args = (x, x if residual is None else residual, weight, y, s, N, N, N, N, N, float(eps), 1.0 / N)
        consts = dict(HAS_RES = residual is not None, BLOCK = BLOCK, EXACT = _libdevice is not None)
        if _INTERPRET:
            _add_rms_norm_kernel[(rows,)](*args, **consts, num_warps = 8 if BLOCK >= 4096 else 4)
        else:
            _fast_launch(_add_rms_norm_kernel, (rows,), args, consts, x.get_device(),
                         (y,) if residual is None else (y, s), num_warps = 8 if BLOCK >= 4096 else 4, **_NO_FMA)
    except Exception as e:
        return _disable(e)
    FUSED_DECODE_CALLS["add_rms_norm"] += 1
    return y, s


def sink_scale_output(attn_output, lse, sinks):
    """(attn_output * sigmoid(lse - sinks)).transpose(1, 2).contiguous() in one launch.
    attn_output [B, H, Q, D], lse [B, H, Q] (fp32), sinks [H]; None when disabled or unsupported."""
    if not _ENABLED or torch.jit.is_tracing():
        return None
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
    try:
        args = (attn_output, lse, sinks, out, H, Q, D,
                attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
                lse.stride(0), lse.stride(1), lse.stride(2),
                out.stride(0), out.stride(1), out.stride(2))
        consts = dict(BLOCK_D = triton.next_power_of_2(D), EXACT = _libdevice is not None)
        if _INTERPRET:
            _sink_scale_kernel[(n,)](*args, **consts, num_warps = 1 if D <= 128 else 4)
        else:
            _fast_launch(_sink_scale_kernel, (n,), args, consts, attn_output.get_device(), (out,),
                         num_warps = 1 if D <= 128 else 4, **_NO_FMA)
    except Exception as e:
        return _disable(e)
    FUSED_DECODE_CALLS["sink_scale_output"] += 1
    return out


# static_cache_update replays transformers' own decode branch with fewer ops, so it runs only where that
# code is the one verified here (transformers 5.8 - 5.21): sha256 of each method's stripped source lines. 5.20 / 5.21
# change Cache.update only in its offloading branch, which this path declines.
_CACHE_ENABLED = (
    os.environ.get("UNSLOTH_GPT_OSS_FUSED_DECODE", "1") != "0"
    and os.environ.get("UNSLOTH_GPT_OSS_FAST_CACHE_UPDATE", "1") != "0"
)
_VERIFIED_UPDATE = {
    "Cache": ("035177fec467f485", "f5c3ba02bbf4c5f4"),
    "StaticLayer": ("6a9b8de193e7b639",),
    "StaticSlidingWindowLayer": ("c114eb5d4cfb9a71",),
}
_STATIC_CLASSES = None


def _source_digest(func):
    import hashlib
    import inspect
    lines = (line.strip() for line in inspect.getsource(func).splitlines())
    return hashlib.sha256("\n".join(line for line in lines if line).encode()).hexdigest()[:16]


def _static_classes():
    """(Cache, StaticLayer, StaticSlidingWindowLayer) when their update methods are the verified ones, else ()."""
    global _STATIC_CLASSES
    if _STATIC_CLASSES is None:
        _STATIC_CLASSES = ()
        try:
            from transformers import cache_utils
            classes = (cache_utils.Cache, cache_utils.StaticLayer, cache_utils.StaticSlidingWindowLayer)
            if all(_source_digest(c.update) in _VERIFIED_UPDATE[c.__name__] for c in classes):
                _STATIC_CLASSES = classes
        except Exception:
            pass
    return _STATIC_CLASSES


def static_cache_update(cache, key_states, value_states, layer_idx):
    """cache.update(key_states, value_states, layer_idx) for one new token on an initialized StaticLayer or a
    full StaticSlidingWindowLayer, with the same results and state; None when the call is not covered (callers
    then run cache.update)."""
    if (
        not _CACHE_ENABLED or key_states.shape[-2] != 1 or key_states.device.type not in ("cuda", "cpu")
        or torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.jit.is_tracing()
    ):
        return None
    classes = _static_classes()
    if not classes:
        return None
    base, static, sliding = classes
    if type(cache).update is not base.update or "update" in cache.__dict__ or getattr(cache, "offloading", False):
        return None
    layers = getattr(cache, "layers", None)
    if layers is None or not (0 <= layer_idx < len(layers)):
        return None
    layer = layers[layer_idx]
    kind = type(layer)
    if (
        (kind is not static and kind is not sliding) or "update" in layer.__dict__
        or not getattr(layer, "is_initialized", False)
    ):
        return None
    keys, values = layer.keys, layer.values
    if (
        key_states.dtype != keys.dtype or value_states.dtype != values.dtype
        or key_states.device != keys.device or value_states.device != values.device
        or key_states.shape[:2] != keys.shape[:2] or key_states.shape[-1] != keys.shape[-1]
        or value_states.shape[:2] != values.shape[:2] or value_states.shape[-1] != values.shape[-1]
    ):
        return None
    if kind is sliding:
        # transformers: is_full -> roll(-1), write the last slot through a host-built index, copy back.
        if layer.cumulative_length_int < layer.max_cache_len:
            return None
        layer.cumulative_length_int += 1
        keys.copy_(torch.cat((keys[:, :, 1:], key_states), dim = -2))
        values.copy_(torch.cat((values[:, :, 1:], value_states), dim = -2))
    else:
        # transformers: index_copy_ at arange(1) + cumulative_length, then cumulative_length += 1. The copies
        # read the counter before the add on the same stream.
        position = layer.cumulative_length
        if position.dim() != 0 or position.device != keys.device:
            return None
        position = position.view(1)
        keys.index_copy_(2, position, key_states)
        values.index_copy_(2, position, value_states)
        layer.cumulative_length.add_(1)
    FUSED_DECODE_CALLS["static_cache_update"] += 1
    return keys, values


_ROPE_ENABLED = os.environ.get("UNSLOTH_GPT_OSS_FUSED_ROPE", "1") != "0"
_ROPE_CHECKED = {}  # (region, device, dtype, B, Hq, Hk, D) -> bool
_ROPE_CHECKED_MAX = 256
_ROPE_GEN = {}  # device -> generator for the calibration inputs (the global RNG is never touched)


def _rope_launch(q, k, cos, sin):
    B, HQ, _, D = q.shape
    HK = k.shape[1]
    half = D // 2
    # Contiguous, like the region's Inductor output, so later guards see the same strides.
    qo = torch.empty((B, HQ, 1, D), dtype = q.dtype, device = q.device)
    ko = torch.empty((B, HK, 1, D), dtype = k.dtype, device = k.device)
    args = (q, k, cos, sin, qo, ko, HQ, HK, half,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), cos.stride(0), sin.stride(0),
            qo.stride(0), qo.stride(1), ko.stride(0), ko.stride(1))
    consts = dict(BLOCK = triton.next_power_of_2(half))
    grid = (B * (HQ + HK),)
    if _INTERPRET:
        _rope_decode_kernel[grid](*args, **consts, num_warps = 1)
    else:
        _fast_launch(_rope_decode_kernel, grid, args, consts, q.get_device(), (qo, ko), num_warps = 1, **_NO_FMA)
    return qo, ko


def _same(a, b):
    return a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype and torch.equal(
        a.view(torch.int16), b.view(torch.int16))


def _region_eager():
    # A non-default stance (force_eager) runs the region eagerly even where it checked compiled. Dynamo
    # disabled or recompiles falling back to eager only affect shapes not compiled yet, which
    # _check_region sees on their first call.
    try:
        return _eval_frame._stance.stance != "default"
    except Exception:
        return False


def _capturing(device):
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):  # the operands' device, not the caller's current one
        return torch.cuda.is_current_stream_capturing()


def _calibrated(t, gen, lo = None):
    c = torch.empty_strided(t.shape, t.stride(), dtype = t.dtype, device = t.device)
    return c.uniform_(-1.0, 1.0, generator = gen) if lo else c.normal_(generator = gen)


def _check_region(q, k, cos, sin, region):
    """Kernel vs region, bitwise and layout, on random inputs of this exact shape and layout: a real first
    token (position 0: sin = 0, cos = 1) cannot tell one rounding from per-op rounding."""
    dev = q.device
    gen = _ROPE_GEN.get(dev)
    if gen is None:
        gen = _ROPE_GEN[dev] = torch.Generator(device = dev)
    gen.manual_seed(0)
    qc, kc = _calibrated(q, gen), _calibrated(k, gen)
    cc, sc = _calibrated(cos, gen, True), _calibrated(sin, gen, True)
    out = _rope_launch(qc, kc, cc, sc)
    ref = region(qc, kc, cc, sc)
    return _same(out[0], ref[0]) and _same(out[1], ref[1])


def rope_decode(q, k, cos, sin, region):
    """One-token apply_rotary_pos_emb, or None (caller runs region). region must be torch.compile'd; the
    first call per region, device, dtype and shape checks the kernel against it bitwise on random
    inputs (_check_region)."""
    if not (_ENABLED and _ROPE_ENABLED) or torch.jit.is_tracing():
        return None
    if getattr(region, "_torchdynamo_orig_callable", None) is None or _region_eager():
        return None
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.dim() != 3:
        return None
    B, HQ, S, D = q.shape
    half = D // 2
    dtype = q.dtype
    if (
        S != 1 or k.shape[2] != 1 or k.shape[0] != B or k.shape[3] != D or D != 2 * half or half > 1024
        or dtype not in (torch.bfloat16, torch.float16) or k.dtype != dtype or cos.dtype != dtype
        or sin.dtype != dtype or tuple(cos.shape) != (B, 1, half) or tuple(sin.shape) != (B, 1, half)
        or q.stride(-1) != 1 or k.stride(-1) != 1 or cos.stride(-1) != 1 or sin.stride(-1) != 1
        or k.device != q.device or cos.device != q.device or sin.device != q.device
        or not (q.is_cuda or _INTERPRET)
    ):
        return None
    # Strides follow from these shapes at this call site (view + transpose of fresh projections).
    key = (region, q.get_device(), dtype, B, HQ, k.shape[1], D)
    checked = _ROPE_CHECKED.get(key)
    if checked is False:
        return None
    if checked is None:
        if _capturing(q.device):
            return None  # the check syncs the host: leave it to an eager call
        try:
            ok = _check_region(q, k, cos, sin, region)
        except Exception as e:
            return _disable(e)
        if len(_ROPE_CHECKED) >= _ROPE_CHECKED_MAX:
            _ROPE_CHECKED.clear()
        _ROPE_CHECKED[key] = ok
        if not ok:
            logger.warning("Unsloth: gpt-oss fused rotary embedding differs from the compiled region here, using the region")
            return None
    try:
        out = _rope_launch(q, k, cos, sin)
    except Exception as e:
        return _disable(e)
    FUSED_DECODE_CALLS["rope_decode"] += 1
    return out
