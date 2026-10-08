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

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import ast
import contextlib
import functools
import os
import stat
import threading
import weakref
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools
import inspect
from .common import (
    TEMPORARY_PATCHES,
    torch_compile,
    _torch_compile,
    get_torch_compile_options,
    UNSLOTH_ENABLE_LOGGING,
    UNSLOTH_COMPILE_DISABLE,
    logger,
)
from importlib.metadata import version as importlib_version
from unsloth_zoo.utils import Version
from unsloth_zoo.mxfp4_dequant import is_mxfp4_expert_param
from .gpt_oss_routed import ROUTED_MAX_SLOTS, prepare_routed_experts, routed_experts_forward, routed_mlp_forward
transformers_version = Version(importlib_version("transformers"))
has_static_cache = transformers_version >= Version("4.56.0.dev0")
from .utils import (
    patch_function,
    patch_function_past_key_values,
    dedent,
    KWARGS_TYPE,
    raise_error,
    logger,
    Cache,
    process_return,
)
from unsloth_zoo.hf_utils import dtype_from_config
torch_cuda_device = torch.cuda.device

# UNSLOTH_MXFP4_NO_DEQUANTIZE=1 keeps MXFP4 quantized (needs triton_kernels); else dequantized to bf16 for LoRA.
UNSLOTH_MXFP4_NO_DEQUANTIZE = os.environ.get("UNSLOTH_MXFP4_NO_DEQUANTIZE", "0") == "1"

# Set when patch_GptOssAttention installs the forward that routes training through
# flex_attention_with_sink, which builds its own BlockMask and ignores these masks.
_GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED = False
_TRAINING_FLAG_ATTR = "_unsloth_gpt_oss_model_training"


def _gpt_oss_layer_attention_type(decoder_layer, config, layer_idx):
    # 4.x sets decoder_layer.attention_type; 5.x dropped it and stock indexes config.layer_types.
    attention_type = getattr(decoder_layer, "attention_type", None)
    if isinstance(attention_type, str):
        return attention_type
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and 0 <= layer_idx < len(layer_types):
        return layer_types[layer_idx]
    self_attn = getattr(decoder_layer, "self_attn", None)
    return "sliding_attention" if getattr(self_attn, "sliding_window", None) is not None else "full_attention"


def _gpt_oss_select_mask(attention_mask, attention_type):
    # Key presence decides, never tensor truthiness: a None value is a valid causal fast path.
    if not isinstance(attention_mask, dict):
        return attention_mask
    if attention_type in attention_mask:
        return attention_mask[attention_type]
    if "full_attention" in attention_mask:
        return attention_mask["full_attention"]
    return next(iter(attention_mask.values()), None)


def _check_triton_kernels_available():
    """Is OpenAI's triton_kernels package available for MXFP4."""
    try:
        from triton_kernels import matmul_ogs, swiglu

        return True
    except ImportError:
        return False


_TRITON_KERNELS_AVAILABLE = None


def is_triton_kernels_available():
    """Cached check for triton_kernels availability."""
    global _TRITON_KERNELS_AVAILABLE
    if _TRITON_KERNELS_AVAILABLE is None:
        _TRITON_KERNELS_AVAILABLE = _check_triton_kernels_available()
    return _TRITON_KERNELS_AVAILABLE


# Newer triton_kernels returns a layout instance, not (class, kwargs): unpacking blind
# raises TypeError there.
def _mxfp4_layout_selection_is_class_contract(selection):
    return (
        isinstance(selection, tuple)
        and len(selection) == 2
        and isinstance(selection[1], dict)
        and isinstance(selection[0], type)
    )


def _normalize_mxfp4_value_layout(selection):
    """(layout_arg, ctor_kwargs) for either contract, ready for convert_layout."""
    if _mxfp4_layout_selection_is_class_contract(selection):
        return selection[0], selection[1]
    return selection, {}


_HOPPER_ONLY_VALUE_ASSERT = ast.dump(
    ast.parse(
        'SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None',
        mode = "eval",
    ).body,
    include_attributes = False,
)


@functools.lru_cache(maxsize = 8)
def _source_rejects_blackwell_value_swizzle(source):
    """Does this kernel source refuse any value swizzle other than Hopper's?

    Matched as an AST node, so the same words in a comment or an error string do not
    count. Absence is not proof Blackwell works, only that there is nothing to fix.
    """
    try:
        tree = ast.parse(dedent(source))
    except Exception:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "static_assert"
            and isinstance(func.value, ast.Name)
            and func.value.id == "tl"
        ):
            continue
        if ast.dump(node.args[0], include_attributes = False) == _HOPPER_ONLY_VALUE_ASSERT:
            return True
    return False


def _blackwell_value_swizzle_unsupported():
    """Read the matmul kernel this process would actually launch."""
    try:
        from triton_kernels.matmul_ogs_details import _matmul_ogs as _kernel_module
    except Exception:
        return False
    found = False
    # Every kernel: newer builds add a persistent variant the dispatcher may pick.
    for name in dir(_kernel_module):
        kernel = getattr(_kernel_module, name, None)
        source = getattr(kernel, "src", None)
        if not isinstance(source, str):
            fn = getattr(kernel, "fn", None)
            if fn is None:
                continue
            try:
                source = inspect.getsource(fn)
            except Exception:
                continue
        if not isinstance(source, str):
            continue
        if _source_rejects_blackwell_value_swizzle(source):
            found = True
    return found


_MXFP4_STRIDED_VALUES_WARNED = False


def _mxfp4_layout_arguments(layout_module, w):
    """(layout_arg, ctor_kwargs, strided_arg) for this weight. strided_arg is a class or
    an instance to match the installed convert_layout, and is reused for the scales."""
    selection = layout_module.make_default_matmul_mxfp4_w_layout(mx_axis = 1)
    class_contract = _mxfp4_layout_selection_is_class_contract(selection)
    value_layout, value_layout_opts = _normalize_mxfp4_value_layout(selection)
    StridedLayout = layout_module.StridedLayout
    strided_argument = StridedLayout if class_contract else StridedLayout()

    if _force_strided_mxfp4_values(value_layout, layout_module, w):
        # Otherwise generate() dies in matmul_ogs: "Only Hopper swizzling is supported
        # for values". Unswizzled values pair with the strided scales, still MXFP4.
        global _MXFP4_STRIDED_VALUES_WARNED
        value_layout, value_layout_opts = strided_argument, {}
        if not _MXFP4_STRIDED_VALUES_WARNED:
            _MXFP4_STRIDED_VALUES_WARNED = True
            logger.info(
                "Unsloth: This triton_kernels build cannot run Blackwell MXFP4 value "
                "swizzling, so gpt-oss weights stay unswizzled (still MXFP4). Set "
                "UNSLOTH_MXFP4_VALUE_LAYOUT=default to opt out."
            )
    return value_layout, value_layout_opts, strided_argument


def _force_strided_mxfp4_values(value_layout, layout_module, w):
    """Should this weight skip Blackwell value swizzling? UNSLOTH_MXFP4_VALUE_LAYOUT
    = auto | strided | default, read per call so it can be set after import."""
    override = os.environ.get("UNSLOTH_MXFP4_VALUE_LAYOUT", "auto").strip().lower()
    if override == "default":
        return False
    try:
        # The weight's own device: a process can hold Blackwell and non-Blackwell cards.
        if not (hasattr(w, "is_cuda") and w.is_cuda):
            return False
        if override == "strided":
            return True
        if torch.cuda.get_device_capability(w.device)[0] < 10:
            return False
        blackwell = getattr(layout_module, "BlackwellMXValueLayout", None)
        if blackwell is None:
            return False
        selected_is_blackwell = (
            value_layout is blackwell
            or (isinstance(value_layout, type) and issubclass(value_layout, blackwell))
            or isinstance(value_layout, blackwell)
        )
        if not selected_is_blackwell:
            return False
        return _blackwell_value_swizzle_unsupported()
    except Exception:
        return False


@torch_compile(dynamic = True, fullgraph = True)
def swiglu_torch_forward(a, alpha, limit, dtype = None):
    a_gelu = a[..., ::2].to(torch.float32)
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2].to(torch.float32)
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    out = out_gelu * (a_linear + 1)
    return out.to(a.dtype if dtype is None else dtype)
pass


@torch_compile(dynamic = True, fullgraph = True)
def swiglu_torch_backward(pre_act, alpha, limit, g1):
    g, l = pre_act[..., ::2].to(torch.float32), pre_act[..., 1::2].to(torch.float32)

    if limit is not None:
        mask_g = g <= limit
        mask_l = l.abs() <= limit
        ḡ = torch.where(mask_g, g, limit)
        l̄ = torch.where(mask_l, l, l.sign() * limit)
    else:                                            # no clipping
        mask_g = mask_l = torch.ones_like(g, dtype=bool)
        ḡ, l̄ = g, l

    σ   = torch.sigmoid(alpha * ḡ)
    dg  = (σ + alpha * ḡ * σ * (1 - σ)) * (l̄ + 1)
    dl  = ḡ * σ
    dg  = torch.where(mask_g, dg, 0.)                # clamp-grad
    dl  = torch.where(mask_l, dl, 0.)

    grad = torch.empty_like(pre_act)
    grad[..., ::2], grad[..., 1::2] = dg, dl
    return g1 * grad.to(g1.dtype)
pass

_MXFP4_E2M1_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _mxfp4_dequantize_experts_torch(blocks, scales, dtype = torch.bfloat16):
    """Reference MXFP4 decode to (E, in, out), one expert at a time; low nibble first."""
    if blocks.dim() != 4 or blocks.shape[-1] != 16 or tuple(blocks.shape[:-1]) != tuple(scales.shape):
        raise ValueError(
            f"Unsloth: MXFP4 blocks {tuple(blocks.shape)} do not match scales {tuple(scales.shape)}"
        )
    E, R, G, B = blocks.shape
    out = torch.empty((E, G * B * 2, R), dtype = dtype, device = blocks.device)
    lut = torch.tensor(_MXFP4_E2M1_VALUES, dtype = torch.float32, device = blocks.device)
    # torch.ldexp on CUDA launches on the current device, not the operand's.
    guard = torch.cuda.device(blocks.device) if blocks.is_cuda else contextlib.nullcontext()
    with guard:
        for e in range(E):
            blk = blocks[e]
            vals = torch.empty((R, G, B * 2), dtype = torch.float32, device = blocks.device)
            vals[..., 0::2] = lut[(blk & 0x0F).long()]
            vals[..., 1::2] = lut[(blk >> 4).long()]
            vals = torch.ldexp(vals, (scales[e].to(torch.int32) - 127).unsqueeze(-1))
            out[e].copy_(vals.reshape(R, G * B * 2).t())
            del vals, blk
    return out


_CONVERT_TRANSPOSES = {}


def _convert_moe_packed_tensors_transposes(convert):
    """(E, in, out) (stock >= 4.56) vs (E, out, in) (Unsloth's patch); down_proj is square, so probe non-square."""
    key = id(convert)
    if key not in _CONVERT_TRANSPOSES:
        probe = convert(
            torch.zeros((1, 2, 1, 16), dtype = torch.uint8),
            torch.full((1, 2, 1), 127, dtype = torch.uint8),
        )
        shape = tuple(probe.shape)
        if shape == (1, 32, 2):
            _CONVERT_TRANSPOSES[key] = True
        elif shape == (1, 2, 32):
            _CONVERT_TRANSPOSES[key] = False
        else:
            raise RuntimeError(f"Unsloth: convert_moe_packed_tensors returned an unexpected shape {shape}")
    return _CONVERT_TRANSPOSES[key]


def _dequantize_mxfp4_experts(blocks, scales, dtype = torch.bfloat16):
    """Dense (E, in, out) stack on any transformers version."""
    try:
        import transformers.integrations.mxfp4 as mxfp4_integration
    except Exception:
        mxfp4_integration = None
    expected = (blocks.shape[0], blocks.shape[2] * blocks.shape[3] * 2, blocks.shape[1])

    if blocks.is_cuda and blocks.dtype == torch.uint8:
        try:
            from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize
            return mxfp4_dequantize(blocks, scales, dtype = dtype, transpose = True)
        except Exception:
            pass

    dequantize = getattr(mxfp4_integration, "dequantize", None)
    if dequantize is not None:
        try:
            parameters = inspect.signature(dequantize).parameters
            takes_blocks_scales = tuple(parameters)[:2] == ("blocks", "scales")
        except (TypeError, ValueError):
            takes_blocks_scales = False
        if takes_blocks_scales:
            try:
                out = dequantize(blocks, scales)
                if tuple(out.shape) == expected:
                    return out.to(dtype).contiguous()
            except Exception:
                pass

    convert = getattr(mxfp4_integration, "convert_moe_packed_tensors", None)
    if convert is not None:
        try:
            transposes = _convert_moe_packed_tensors_transposes(convert)
            out = convert(blocks, scales, dtype = dtype)
            if not transposes:
                out = out.transpose(1, 2)
            if tuple(out.shape) == expected:
                return out.to(device = blocks.device, dtype = dtype).contiguous()
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: convert_moe_packed_tensors failed ({e}); using the local MXFP4 decode.")

    return _mxfp4_dequantize_experts_torch(blocks, scales, dtype = dtype)

def _mxfp4_hub_kernel_unreachable():
    """True when transformers loads MXFP4 kernels via the `kernels` hub and it is unusable."""
    try:
        import inspect
        import transformers.integrations.mxfp4 as mxfp4_integration
        source = inspect.getsource(mxfp4_integration.replace_with_mxfp4_linear)
    except Exception:
        return False
    if "get_kernel" not in source:
        return False
    if hasattr(mxfp4_integration, "_replace_with_mxfp4_linear"):
        return False
    try:
        from transformers.utils import is_kernels_available as _real_is_kernels_available
        return not _real_is_kernels_available()
    except Exception:
        return True
pass


def topk_to_routing_tensors(router_indices, routing_weights, n_expts_tot):
    """triton_kernels.routing.routing_torch from its sort step on, fed the router's already-softmaxed weights.

    routing_weights is dense (n_tokens, n_experts) or aligned with router_indices (n_tokens, top_k).
    Returns (gate_scal, expt_hist, combine_indx, dispatch_indx), expert-major.
    """
    n_tokens, top_k = router_indices.shape
    expt_indx = router_indices.to(torch.int64)
    if tuple(routing_weights.shape) == (n_tokens, n_expts_tot):
        expt_scal = torch.gather(routing_weights, 1, expt_indx)
    elif tuple(routing_weights.shape) == (n_tokens, top_k):
        expt_scal = routing_weights
    else:
        raise ValueError(
            f"Unsloth: routing_weights must be {(n_tokens, n_expts_tot)} or {(n_tokens, top_k)}, "
            f"got {tuple(routing_weights.shape)}"
        )
    expt_indx, order = torch.sort(expt_indx, dim = 1)
    expt_scal = torch.gather(expt_scal, 1, order).reshape(-1)
    expt_indx = expt_indx.reshape(-1)
    combine_indx = torch.argsort(expt_indx, stable = True)
    dispatch_indx = torch.argsort(combine_indx, stable = True)
    # scatter_add, not bincount: CUDA bincount syncs to size its output.
    expt_hist = torch.zeros(n_expts_tot, dtype = torch.int32, device = expt_indx.device).scatter_add_(
        0, expt_indx, torch.ones_like(expt_indx, dtype = torch.int32),
    )
    return expt_scal[combine_indx], expt_hist, combine_indx.to(torch.int32), dispatch_indx.to(torch.int32)
pass


def _triton_kernels_root(module):
    # Same triton_kernels copy as the weights (other copies reject its Tensor), else the resolved one.
    weight = module.__dict__.get("_gate_up_proj", module.__dict__.get("gate_up_proj"))
    if weight is not None and not isinstance(weight, torch.Tensor):
        return type(weight).__module__.rsplit(".tensor", 1)[0]
    from unsloth_zoo.triton_kernels_compat import get_triton_kernels
    tk = get_triton_kernels()
    return tk.__name__ if tk is not None else "triton_kernels"
pass


def expt_data_from_hist(hist, n_expts_tot, n_gates, block_ms):
    """triton_kernels.routing.compute_expt_data_torch, batched over block_ms and sync-free.

    Returns (token_offs_raw, {block_m: token_offs_pad}, {block_m: block_pid_map}).
    """
    device = hist.device
    zero = torch.zeros(1, dtype = torch.int32, device = device)
    token_offs_raw = torch.cat((zero, torch.cumsum(hist, 0, dtype = torch.int32)))
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(block_ms))
    bm = torch.tensor(block_ms, dtype = torch.int32, device = device)[:, None]
    n_tiles = (hist[None, :] + bm - 1) // bm
    token_offs_pad = torch.cat((zero.expand(len(block_ms), 1), torch.cumsum(n_tiles, 1, dtype = torch.int32)), 1)
    col = torch.arange(max_n_tiles, dtype = torch.int32, device = device)
    vals = torch.arange(n_expts_tot, dtype = torch.int32, device = device)[:, None] + (col << 16)[None, :]
    # Tiles past each expert's count land in a spare trailing slot instead of a boolean-mask gather (host sync).
    idxs = torch.where(
        col[None, None, :] < n_tiles[:, :, None],
        token_offs_pad[:, :-1, None] + col[None, None, :],
        max_n_tiles,
    ) + torch.arange(len(block_ms), dtype = torch.int32, device = device)[:, None, None] * (max_n_tiles + 1)
    block_pid_map = torch.full((len(block_ms) * (max_n_tiles + 1),), -1, dtype = torch.int32, device = device)
    block_pid_map.scatter_(0, idxs.reshape(-1).long(), vals.expand(len(block_ms), -1, -1).reshape(-1))
    block_pid_map = block_pid_map.view(len(block_ms), max_n_tiles + 1)[:, :max_n_tiles]
    return (
        token_offs_raw,
        {b: token_offs_pad[i] for i, b in enumerate(block_ms)},
        {b: block_pid_map[i] for i, b in enumerate(block_ms)},
    )
pass


@torch.compiler.disable
def _mxfp4_routing_from_topk(module, router_indices, routing_weights):
    import importlib
    tk_routing = importlib.import_module(_triton_kernels_root(module) + ".routing")
    n_expts_tot = module.num_experts
    with torch_cuda_device(router_indices.device):
        gate_scal, expt_hist, combine_indx, dispatch_indx = topk_to_routing_tensors(
            router_indices, routing_weights, n_expts_tot,
        )
        expt_data = None
        if hasattr(tk_routing, "ExptData"):
            block_ms = [16, 32, 64, 128] + ([256] if getattr(tk_routing, "is_hip", lambda: False)() else [])
            expt_data = tk_routing.ExptData(
                expt_hist, *expt_data_from_hist(expt_hist, n_expts_tot, router_indices.numel(), block_ms),
            )
    return (
        tk_routing.RoutingData(gate_scal, expt_hist, n_expts_tot, router_indices.shape[1], expt_data),
        tk_routing.GatherIndx(src_indx = combine_indx, dst_indx = dispatch_indx),
        tk_routing.ScatterIndx(src_indx = dispatch_indx, dst_indx = combine_indx),
    )
pass


def _mxfp4_experts_routing(module, hidden_states, routing_data, gather_idx, scatter_idx, router_indices, routing_weights):
    """(hidden_states, routing_data, gather_idx, scatter_idx, leading_shape) from either experts() call form.

    mlp_forward passes triton_kernels routing objects; stock GptOssMLP.forward passes the router's
    (indices, weights), by keyword on transformers 4.x and positionally on 5.x (unsloth-zoo#385).
    """
    if isinstance(routing_data, torch.Tensor):
        router_indices, routing_weights, routing_data, gather_idx = routing_data, gather_idx, None, None
    if routing_data is not None:
        return hidden_states, routing_data, gather_idx, scatter_idx, None
    if router_indices is None or routing_weights is None:
        raise TypeError(
            "Unsloth: Mxfp4GptOssExperts.forward needs (routing_data, gather_idx, scatter_idx) "
            "or (router_indices, routing_weights)"
        )
    leading_shape = hidden_states.shape[:-1]
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    n_tokens = hidden_states.shape[0]
    routing_data, gather_idx, scatter_idx = _mxfp4_routing_from_topk(
        module, router_indices.reshape(n_tokens, -1), routing_weights.reshape(n_tokens, -1),
    )
    return hidden_states, routing_data, gather_idx, scatter_idx, leading_shape
pass


def patch_gpt_oss():
    try:
        import triton_kernels

        HAS_TRITON_KERNELS = True
    except Exception as e:
        HAS_TRITON_KERNELS = False
        # return raise_error("Please install triton_kernels", e)
    try:
        import transformers.quantizers.quantizer_mxfp4

        # Always allow LoRA training (works with dequantized bf16 weights too)
        transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer.is_trainable = lambda *args, **kwargs: True
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer", e)

    if HAS_TRITON_KERNELS and _mxfp4_hub_kernel_unreachable():
        # Claiming kernels skips the bf16 fallback, then the hub load raises ImportError (vLLM triton_kernels).
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                "Unsloth: triton_kernels is importable but transformers cannot load the MXFP4 "
                "hub kernels, so MXFP4 GPT OSS weights will be dequantized to bf16."
            )
        return
    elif HAS_TRITON_KERNELS:
        # Only override is_kernels_available when triton_kernels IS available
        try:
            def is_kernels_available(): return True

            transformers.quantizers.quantizer_mxfp4.is_kernels_available = is_kernels_available
        except Exception as e:
            return raise_error("transformers.quantizers.quantizer_mxfp4.is_kernels_available", e)

        if hasattr(transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer, "_lazy_import_kernels"):
            transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer._lazy_import_kernels = lambda *args, **kwargs: triton_kernels

        try:
            from triton_kernels import matmul_ogs, swiglu

            FnSpecs, FusedActivation, matmul_ogs = (
                matmul_ogs.FnSpecs,
                matmul_ogs.FusedActivation,
                matmul_ogs.matmul_ogs,
            )
            swiglu_fn = swiglu.swiglu_fn
        except Exception as e:
            return raise_error("triton_kernels", e)
    else:
        # Leave is_kernels_available intact so transformers' validate_environment()
        # correctly sets dequantize=True, enabling bf16 fallback.
        return

    try:
        import transformers.integrations.mxfp4
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4", e)

    def swizzle_mxfp4(w, w_scale, *args, **kwargs):
        from triton_kernels import tensor, tensor_details
        FP4, convert_layout, wrap_torch_tensor = (
            tensor.FP4,
            tensor.convert_layout,
            tensor.wrap_torch_tensor,
        )
        layout = tensor_details.layout

        value_layout, value_layout_opts, strided_argument = _mxfp4_layout_arguments(layout, w)
        w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
        # TODO : add that when we are actually sure that it works on B200
        # if torch.cuda.get_device_capability()[0] == 10:
        #     constraints = {
        #         "is_persistent": True,
        #         "epilogue_subtile": 1,
        #     }
        #     opt_flags.update_opt_flags_constraints(constraints)
        # # transpose the tensor so that the quantization axis is on dim1

        # TODO: there is still an issue with the scales on hopper
        # scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=8)
        # w_scale = convert_layout(wrap_torch_tensor(w_scale), scale_layout, **scale_layout_opts)
        w_scale = convert_layout(wrap_torch_tensor(w_scale), strided_argument)
        return w, w_scale
    patch_function(transformers.integrations.mxfp4, "swizzle_mxfp4", swizzle_mxfp4, match_level = "relaxed")

    class Mxfp4GptOssExperts(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.num_experts = config.num_local_experts
            self.intermediate_size = config.intermediate_size
            self.hidden_size = config.hidden_size

            # MXFP4 quantized format (blocks + scales)
            self.gate_up_proj_blocks = nn.Parameter(
                torch.zeros(
                    self.num_experts,
                    2 * self.intermediate_size,
                    self.hidden_size // 32,
                    16,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_scales = nn.Parameter(
                torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, dtype=torch.uint8),
                requires_grad=False,
            )
            self.gate_up_proj_bias = nn.Parameter(
                torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False,
            )

            self.down_proj_blocks = nn.Parameter(
                torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8), requires_grad=False
            )
            self.down_proj_scales = nn.Parameter(
                torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size // 32, dtype=torch.uint8), requires_grad=False
            )
            self.down_proj_bias = nn.Parameter(
                torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32), requires_grad=False
            )

            self.alpha = 1.702
            self.limit = getattr(config, "swiglu_limit", 7.0)
            self.gate_up_proj_precision_config = None
            self.down_proj_precision_config = None

        @property
        def gate_up_proj(self):
            """gate_up_proj tensor, from blocks/scales or stored directly."""
            # Already set from checkpoint loading or previous dequantization
            if "_gate_up_proj" in self.__dict__:
                return self.__dict__["_gate_up_proj"]

            # MXFP4 weights present when blocks/scales are not all zeros
            blocks_valid = self.__dict__.get("_gate_up_proj_blocks_valid", False) or (
                self.gate_up_proj_blocks.device.type != "meta"
                and self.gate_up_proj_blocks.numel() > 0
                and bool(self.gate_up_proj_blocks.any())
            )
            self.__dict__["_gate_up_proj_blocks_valid"] = blocks_valid

            if not blocks_valid:
                raise AttributeError(
                    f"Mxfp4GptOssExperts.gate_up_proj: No weights loaded. "
                    f"Try 'openai/gpt-oss-20b' with load_in_4bit=True instead."
                )

            # Still packed (transformers 5 skips load_and_swizzle_mxfp4): decode per call, uncached.
            try:
                return _dequantize_mxfp4_experts(self.gate_up_proj_blocks, self.gate_up_proj_scales)
            except Exception as e:
                raise RuntimeError(f"Failed to dequantize MXFP4 gate_up_proj: {e}") from e

        @gate_up_proj.setter
        def gate_up_proj(self, value):
            """Set gate_up_proj tensor (during checkpoint loading)."""
            self.__dict__["_gate_up_proj"] = value

        @property
        def down_proj(self):
            """down_proj tensor, from blocks/scales or stored directly."""
            if "_down_proj" in self.__dict__:
                return self.__dict__["_down_proj"]

            blocks_valid = self.__dict__.get("_down_proj_blocks_valid", False) or (
                self.down_proj_blocks.device.type != "meta"
                and self.down_proj_blocks.numel() > 0
                and bool(self.down_proj_blocks.any())
            )
            self.__dict__["_down_proj_blocks_valid"] = blocks_valid

            if not blocks_valid:
                raise AttributeError(
                    f"Mxfp4GptOssExperts.down_proj: No weights loaded."
                )

            # Still packed (transformers 5 skips load_and_swizzle_mxfp4): decode per call, uncached.
            try:
                return _dequantize_mxfp4_experts(self.down_proj_blocks, self.down_proj_scales)
            except Exception as e:
                raise RuntimeError(f"Failed to dequantize MXFP4 down_proj: {e}") from e

        @down_proj.setter
        def down_proj(self, value):
            """Set down_proj tensor (during checkpoint loading)."""
            self.__dict__["_down_proj"] = value

        def forward(
            self, hidden_states: torch.Tensor, routing_data = None, gather_idx = None, scatter_idx = None,
            router_indices = None, routing_weights = None,
        ) -> torch.Tensor:
            hidden_states, routing_data, gather_idx, scatter_idx, leading_shape = _mxfp4_experts_routing(
                self, hidden_states, routing_data, gather_idx, scatter_idx, router_indices, routing_weights,
            )
            with torch_cuda_device(hidden_states.device):
                if not hasattr(self, "act"):
                    self.act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, self.limit), 2)
                if not (torch.is_grad_enabled() and hidden_states.requires_grad):
                    intermediate_cache1 = matmul_ogs(
                        hidden_states.to(torch.bfloat16),  # tl.dot_scaled upcasts to BF16 for old hardware
                        self.gate_up_proj,
                        self.gate_up_proj_bias,
                        routing_data,
                        gather_indx=gather_idx,
                        precision_config=self.gate_up_proj_precision_config,
                        gammas=None,
                        fused_activation=self.act,
                    )
                    intermediate_cache3 = matmul_ogs(
                        intermediate_cache1,
                        self.down_proj,
                        self.down_proj_bias,
                        routing_data,
                        scatter_indx=scatter_idx,
                        precision_config=self.down_proj_precision_config,
                        gammas=routing_data.gate_scal if routing_data else None,
                    )
                else:
                    intermediate_cache3 = mxfp4_ogs_experts_forward(
                        self, hidden_states, routing_data, gather_idx, scatter_idx,
                    )
            if leading_shape is not None:
                intermediate_cache3 = intermediate_cache3.reshape(*leading_shape, intermediate_cache3.shape[-1])
            return intermediate_cache3

        pass

    patch_function(transformers.integrations.mxfp4, "Mxfp4GptOssExperts", Mxfp4GptOssExperts)

    if HAS_TRITON_KERNELS:
        try:
            routing = triton_kernels.routing.routing
            routing = torch.compiler.disable(routing)
        except Exception as e:
            return raise_error("triton_kernels.routing.routing", e)

        def mlp_forward(self, hidden_states):
            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
            router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)

            with torch_cuda_device(router_logits.device):
                routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k)

            routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx)
            routed_out = routed_out.reshape(batch_size, -1, self.router.hidden_dim)
            return routed_out, router_logits

        patch_function(transformers.integrations.mxfp4, "mlp_forward", mlp_forward)

    if HAS_TRITON_KERNELS:
        try:
            PrecisionConfig, FlexCtx, InFlexData = (
                triton_kernels.matmul_ogs.PrecisionConfig,
                triton_kernels.matmul_ogs.FlexCtx,
                triton_kernels.matmul_ogs.InFlexData,
            )
        except Exception as e:
            return raise_error("triton_kernels.matmul_ogs", e)

    # Legacy per-parameter TP loader hook. transformers 5.16.0 (upstream PR #47579,
    # the DTensor tensor parallel rewrite) reduced it to a tombstone that raises when
    # called, and dropped load_and_swizzle_mxfp4, so the patch below is inert there.
    # The import itself still succeeds; skipping it is defensive.
    if transformers_version < Version("5.16.0"):
        try:
            from transformers.integrations.tensor_parallel import shard_and_distribute_module
        except Exception as e:
            return raise_error("transformers.integrations.tensor_parallel.shard_and_distribute_module", e)
    else:
        shard_and_distribute_module = None

    def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, *args, **kwargs):
        model = kwargs.get("model", None)
        empty_param = kwargs.get("empty_param", None)
        casting_dtype = kwargs.get("casting_dtype", None)
        to_contiguous = kwargs.get("to_contiguous", None)
        rank = kwargs.get("rank", None)
        device_mesh = kwargs.get("device_mesh", None)

        for proj in ["gate_up_proj", "down_proj"]:
            if proj in param_name:
                if device_mesh is not None:
                    if shard_and_distribute_module is None:
                        raise RuntimeError(
                            "Unsloth: tensor parallel MXFP4 loading needs transformers < 5.16.0, "
                            "which still provides shard_and_distribute_module. On 5.16.0 and newer "
                            "load with from_pretrained(..., tp_plan=...) instead."
                        )
                    shard_and_distribute_module(model, param_value, empty_param, param_name, casting_dtype, to_contiguous, rank, device_mesh)
                else:
                    setattr(module, param_name.rsplit(".", 1)[1], torch.nn.Parameter(param_value, requires_grad=False))
                blocks_attr = f"{proj}_blocks"
                scales_attr = f"{proj}_scales"
                blocks = getattr(module, blocks_attr)
                scales = getattr(module, scales_attr)
                # Valid = blocks/scales off meta AND non-zero (all-zeros means init, not checkpoint)
                blocks_valid = (
                    blocks.device.type != "meta"
                    and scales.device.type != "meta"
                    and blocks.numel() > 0
                    and blocks.any()  # At least some non-zero values
                )
                if blocks_valid:
                    # need it for ep
                    local_experts = blocks.size(0)
                    if proj == "gate_up_proj":
                        blocks = blocks.view(local_experts, module.intermediate_size * 2, -1)
                    else:
                        blocks = blocks.view(local_experts, -1, module.intermediate_size // 2)
                    # TODO: we need to have the weights on cuda, refactor later
                    if getattr(target_device, "type", target_device) == "cpu":
                        target_device = "cuda"
                    # TODO: check why we still do move the tensors despite the context manager
                    blocks = blocks.to(target_device)
                    scales = scales.to(target_device)
                    with torch.cuda.device(target_device):
                        triton_weight_tensor, weight_scale = swizzle_mxfp4(
                            blocks.transpose(-2, -1), scales.transpose(-2, -1)
                        )

                    # need to overwrite the shapes for the kernels
                    if proj == "gate_up_proj":
                        triton_weight_tensor.shape = torch.Size(
                            [local_experts, module.hidden_size, module.intermediate_size * 2]
                        )
                    else:
                        triton_weight_tensor.shape = torch.Size(
                            [local_experts, module.intermediate_size, module.hidden_size]
                        )

                    # triton_weight_tensor is what needs to be passed in oai kernels. It stores the data, the shapes and any more objects. It is like a subtensor
                    setattr(module, proj, triton_weight_tensor)
                    setattr(
                        module,
                        f"{proj}_precision_config",
                        PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
                    )

                    # delete blocks and scales
                    delattr(module, scales_attr)
                    delattr(module, blocks_attr)
                    # setattr(module, blocks_attr, torch.nn.Parameter(triton_weight_tensor.storage.data, requires_grad=False))
                    del blocks

    pass
    patch_function(transformers.integrations.mxfp4, "load_and_swizzle_mxfp4", load_and_swizzle_mxfp4, match_level = "relaxed")

    try:
        from transformers.integrations.mxfp4 import _replace_with_mxfp4_linear
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4._replace_with_mxfp4_linear", e)

    def replace_with_mxfp4_linear(
        model,
        modules_to_not_convert=None,
        current_key_name=None,
        quantization_config=None,
        config=None,
    ):
        if quantization_config.dequantize: return model
        modules_to_not_convert = (["lm_head"] if modules_to_not_convert is None else modules_to_not_convert)
        if quantization_config.modules_to_not_convert is not None:
            modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
        modules_to_not_convert = list(set(modules_to_not_convert))
        model, has_been_replaced = _replace_with_mxfp4_linear(model, modules_to_not_convert, current_key_name, quantization_config, config=config)
        if not has_been_replaced:
            logger.warning_once(
                "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is"
                " a bug."
            )

        return model

    patch_function(transformers.integrations.mxfp4, "replace_with_mxfp4_linear", replace_with_mxfp4_linear)
pass
TEMPORARY_PATCHES.append(patch_gpt_oss)


class ParameterModule(nn.Linear):
    """
    Wraps a parameter as an nn.Linear for PEFT, managing 3D <-> 2D weight conversion.
    Unsloth grouped_mm needs 3D weights (gate_up: (E, H, 2I); down: (E, I, H)); PEFT
    Linear needs 2D (Out, In). Stored 2D for PEFT, reshaped for Unsloth via get_param().
    """

    def __init__(
        self, in_features, out_features, shape_3d, permute_to_2d, permute_to_3d
    ):
        super().__init__(in_features, out_features, bias=False)
        self.shape_3d = shape_3d
        self.permute_to_2d = permute_to_2d
        self.permute_to_3d = permute_to_3d
        # Caller must overwrite the randomly initialized weight.

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, shape_3d={self.shape_3d}"

    def get_param(self):
        """Restore the 3D weight for Unsloth computation."""
        # 2D (Out, In) -> view to (E, 2I, H) [shape_3d permuted by permute_to_2d] -> permute -> 3D
        unflattened_shape = [self.shape_3d[i] for i in self.permute_to_2d]
        return self.weight.view(*unflattened_shape).permute(*self.permute_to_3d).contiguous()

    def set_weight_from_3d(self, weight_3d):
        """Set the 2D weight from a 3D tensor."""
        weight_2d = weight_3d.permute(*self.permute_to_2d).reshape(self.out_features, self.in_features)
        self.weight.data.copy_(weight_2d)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoint 'gate_up_proj' is 3D; load into 'gate_up_proj.weight' (2D)
        key = prefix[:-1]

        if key in state_dict:
            val = state_dict[key]
            val_2d = val.permute(*self.permute_to_2d).reshape(self.out_features, self.in_features)
            state_dict[prefix + "weight"] = val_2d
            del state_dict[key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def patch_gpt_oss_compiler_exports():
    model_name = os.environ.get("UNSLOTH_MODEL_NAME", "").replace("-", "_")
    if "gpt_oss" not in model_name:
        return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
    except Exception as e:
        raise_error("transformers.models.gpt_oss.modeling_gpt_oss", e)
        return

    # Export helpers so compiler-generated GPT-OSS modules can resolve symbols.
    m = transformers.models.gpt_oss.modeling_gpt_oss
    m.ParameterModule = ParameterModule
    m.swiglu_torch_forward = swiglu_torch_forward
    m.dtype_from_config = dtype_from_config
    m.transformers_version = transformers_version
    m.Version = Version
TEMPORARY_PATCHES.append(patch_gpt_oss_compiler_exports)


class GptOssExperts(nn.Module):
    """
    GPT OSS MoE Experts layer with 3D stacked parameters; supports grouped_mm with split LoRA.
    Same structure as transformers GptOssExperts:
    - gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
    - gate_up_proj_bias: (num_experts, 2 * expert_dim)
    - down_proj: (num_experts, expert_dim, hidden_size)
    - down_proj_bias: (num_experts, hidden_size)
    """

    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.intermediate_size = config.intermediate_size  # Alias for compatibility
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.dtype = dtype_from_config(config)

        # gate_up_proj: 3D (E, H, 2I) -> 2D (E*2I, H); permute (0,2,1), reverse (0,2,1)
        self.gate_up_proj = ParameterModule(
            in_features=self.hidden_size,
            out_features=self.num_experts * 2 * self.expert_dim,
            shape_3d=(self.num_experts, self.hidden_size, 2 * self.expert_dim),
            permute_to_2d=(0, 2, 1),
            permute_to_3d=(0, 2, 1),
        )
        self.gate_up_proj.set_weight_from_3d(
            torch.zeros(
                self.num_experts,
                self.hidden_size,
                2 * self.expert_dim,
                dtype=self.dtype,
            )
        )

        self.gate_up_proj_bias = nn.Parameter(torch.zeros(self.num_experts, 2 * self.expert_dim, dtype=self.dtype))

        # down_proj: 3D (E, I, H) -> 2D (H, E*I); permute (2,0,1), reverse (1,2,0)
        self.down_proj = ParameterModule(
            in_features=self.num_experts * self.expert_dim,
            out_features=self.hidden_size,
            shape_3d=(self.num_experts, self.expert_dim, self.hidden_size),
            permute_to_2d=(2, 0, 1),
            permute_to_3d=(1, 2, 0),
        )
        self.down_proj.set_weight_from_3d(
            torch.empty(
                self.num_experts, self.expert_dim, self.hidden_size, dtype=self.dtype
            )
        )

        self.down_proj_bias = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, dtype=self.dtype))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Convert checkpoint 3D tensors (gate_up_proj, down_proj nn.Parameter) to the 2D
        .weight format that ParameterModule (nn.Linear subclass) expects.
        """
        gate_up_key = prefix + "gate_up_proj"
        gate_up_weight_key = prefix + "gate_up_proj.weight"
        if gate_up_key in state_dict and gate_up_weight_key not in state_dict:
            val_3d = state_dict.pop(gate_up_key)
            # 3D (E, H, 2I) -> permute (0,2,1) -> (E, 2I, H) -> reshape (E*2I, H)
            val_2d = val_3d.permute(0, 2, 1).reshape(self.num_experts * 2 * self.expert_dim, self.hidden_size)
            state_dict[gate_up_weight_key] = val_2d

        down_key = prefix + "down_proj"
        down_weight_key = prefix + "down_proj.weight"
        if down_key in state_dict and down_weight_key not in state_dict:
            val_3d = state_dict.pop(down_key)
            # 3D (E, I, H) -> permute (2,0,1) -> (H, E, I) -> reshape (H, E*I)
            val_2d = val_3d.permute(2, 0, 1).reshape(self.hidden_size, self.num_experts * self.expert_dim)
            state_dict[down_weight_key] = val_2d

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        """Forward using grouped_mm or loop fallback with LoRA support."""
        if _check_torch_grouped_mm_supported():
            return forward_native_grouped_mm(self, hidden_states, router_indices, routing_weights)
        return torch_native_forward(self, hidden_states, router_indices, routing_weights)


pass


class _RouterLinearParams(nn.Module):
    """
    weight/bias container like nn.Linear but NOT nn.Linear, so BitsAndBytes will NOT quantize it.
    State dict keys linear.weight/linear.bias match BnB 4-bit checkpoints where the router was
    saved via an nn.Linear submodule.
    """
    def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class GptOssTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        # _RouterLinearParams (not nn.Linear) avoids BnB 4-bit quantization; keys
        # router.linear.weight/bias match the BnB 4-bit checkpoint format.
        self.linear = _RouterLinearParams(self.hidden_dim, self.num_experts, dtype=dtype_from_config(config))

    # transformers' _init_weights expects .weight and .bias
    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        self.linear.bias = value

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.linear(hidden_states.to(self.linear.weight.dtype))  # (batch_size * seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits, dtype=router_logits.dtype).scatter_(1, router_indices, router_top_value)
        if transformers_version >= Version("5.0.0"):
            return router_logits, router_scores, router_indices
        else:
            return router_scores, router_indices

pass


# BitsAndBytes 4bit compatible classes for loading pre-quantized models
class GptOssExpertsBnb4bit(nn.Module):
    """
    GPT OSS MoE Experts using ModuleLists of nn.Linear (gate_up_projs, down_projs)
    so BitsAndBytes can quantize them.
    """

    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.intermediate_size = config.intermediate_size
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.dtype = dtype_from_config(config)

        self.gate_up_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])

        # Empty buffers for transformers _init_weights compat; avoid large bf16 alloc in 4-bit mode.
        self.register_buffer(
            "gate_up_proj", torch.empty(0, dtype=self.dtype), persistent=False
        )
        self.register_buffer(
            "gate_up_proj_bias", torch.empty(0, dtype=self.dtype), persistent=False
        )
        self.register_buffer(
            "down_proj", torch.empty(0, dtype=self.dtype), persistent=False
        )
        self.register_buffer(
            "down_proj_bias", torch.empty(0, dtype=self.dtype), persistent=False
        )

    def _grouped_bnb4bit_ready(self):
        """True when every expert is a bnb Linear4bit with a populated quant_state,
        either plain or wrapped by one supported PEFT LoRA adapter (see
        gpt_oss_grouped_qlora.expert_lora_state), so the grouped torch._grouped_mm
        path applies. Stores the LoRA state on self._unsloth_grouped_lora.

        Self-contained (local imports, no module globals): the compiler can copy
        this class's source into the standalone compiled cache, whose module
        namespace lacks this file's globals."""
        import os
        if os.environ.get("UNSLOTH_GPTOSS_GROUPED", "1") == "0":
            return False
        # The grouped path materializes torch._grouped_mm stacks meant to run under the
        # compiled cache; when the user disables compilation they have opted into the plain
        # eager per-expert loop, so honor that and fall back.
        if os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1":
            return False
        # The full check costs ~0.6 ms per layer; reuse it while ready_signature is unchanged,
        # and skip even that while no expert state changed (moe_ready_epoch).
        from unsloth_zoo.temporary_patches.gpt_oss_grouped_qlora import (
            cached_ready, ready_record, ready_signature,
        )
        hit = cached_ready(self)
        if hit is not None:
            self._unsloth_grouped_lora = hit[1]
            return hit[0]
        sig = ready_signature(self)
        cached = getattr(self, "_unsloth_grouped_ready", None)
        if sig is not None and cached is not None and cached[0] == sig:
            self._unsloth_grouped_lora = cached[2]
            self._unsloth_grouped_ready = cached[:3] + (ready_record(self),)
            return cached[1]
        def _uncached():
            def _fail(reason):
                self._unsloth_grouped_lora = None
                if (
                    os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
                    and not getattr(self, "_unsloth_grouped_logged", False)
                ):
                    self._unsloth_grouped_logged = True
                    import logging
                    logging.getLogger("unsloth_zoo.temporary_patches").info(
                        f"Unsloth: gpt-oss grouped path disabled: {reason}"
                    )
                return False
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Params4bit
                from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported
                if not _check_torch_grouped_mm_supported():
                    # fp16 experts run moe_grouped_fp16's Triton GEMMs, which need no torch._grouped_mm.
                    from unsloth_zoo.temporary_patches.moe_grouped_fp16 import fp16_grouped_available
                    first = getattr(self.gate_up_projs[0], "base_layer", self.gate_up_projs[0])
                    if not fp16_grouped_available(first.weight.device):
                        return _fail("torch._grouped_mm unsupported")
                from unsloth_zoo.temporary_patches.gpt_oss_grouped_qlora import expert_lora_state
                lora = expert_lora_state(self)
                if isinstance(lora, str):
                    return _fail(f"LoRA-wrapped experts: {lora}")
                blocksize = fmt = None
                proj_dtype = {}
                for which, lin in [(0, m) for m in self.gate_up_projs] + [(1, m) for m in self.down_projs]:
                    lin = getattr(lin, "base_layer", lin)
                    w = getattr(lin, "weight", None)
                    if not (isinstance(w, Params4bit) and getattr(w, "quant_state", None) is not None):
                        return _fail(f"expert weight {type(w).__name__} without quant_state")
                    if w.requires_grad:
                        return _fail("expert weight requires_grad")
                    qs = w.quant_state
                    # Concatenating experts is exact only with whole blocks of one shared blocksize.
                    numel = 1
                    for s in qs.shape:
                        numel *= int(s)
                    if not qs.blocksize or numel % int(qs.blocksize) != 0:
                        return _fail(f"expert numel {numel} not a multiple of blocksize {qs.blocksize}")
                    if blocksize is None:
                        blocksize, fmt = int(qs.blocksize), qs
                    elif int(qs.blocksize) != blocksize:
                        return _fail(f"mixed blocksizes {blocksize} vs {qs.blocksize}")
                    # The bitsandbytes fallback decodes every expert with the first one's format.
                    elif (
                        qs.quant_type != fmt.quant_type
                        or bool(getattr(qs, "nested", False)) != bool(getattr(fmt, "nested", False))
                        or not (qs.code is fmt.code or torch.equal(qs.code, fmt.code))
                    ):
                        return _fail("mixed quantization formats across experts")
                    # One dequant dtype per projection: on float16 the loader keeps down in fp32.
                    if proj_dtype.setdefault(which, qs.dtype) != qs.dtype:
                        return _fail("mixed quantization formats across experts")
                    b = getattr(lin, "bias", None)
                    # The grouped path stacks per-expert biases.
                    if b is None:
                        return _fail("expert bias is None")
                    if b.requires_grad:
                        return _fail("expert bias requires_grad")
            except Exception as e:
                return _fail(f"{type(e).__name__}: {e}")
            self._unsloth_grouped_lora = lora
            return True

        verdict = _uncached()
        if sig is not None:
            lora = getattr(self, "_unsloth_grouped_lora", None)
            self._unsloth_grouped_ready = (sig, verdict, lora, ready_record(self))
        return verdict

    def _forward_grouped_bnb4bit(self, hidden_states, router_indices, routing_weights,
                                 batch_size, num_tokens, num_experts, top_k):
        """Grouped equivalent of the per-expert loop (gpt_oss_grouped_qlora): one gather,
        one stacked NF4 dequant + torch._grouped_mm per projection (rebuilt in backward
        per _moe_recompute_default), the LoRA adapter as grouped_mm over the stacked
        per-expert A / B, fp32 index_add combine. fp16 experts (fp32 down under the
        loader's rule) take Triton grouped GEMMs with the loop's per-projection dtypes.
        None (the caller keeps the per-expert loop) when no grouped path matches.

        Self-contained (local imports, no module globals) for the standalone
        compiled cache; see _grouped_bnb4bit_ready."""
        from unsloth_zoo.temporary_patches.gpt_oss_grouped_qlora import grouped_qlora_forward
        lora = getattr(self, "_unsloth_grouped_lora", None)
        if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return None
        return grouped_qlora_forward(
            self, hidden_states, router_indices, routing_weights,
            batch_size, num_tokens, num_experts, top_k, lora = lora,
        )

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]
        num_experts = routing_weights.shape[1]
        top_k = router_indices.shape[1]

        # fp16 experts take the grouped path too: it keeps the loop's fp32 swiglu and
        # fp32 down output (gpt_oss_grouped_qlora.compute_mode decides per call).
        if (
            self.training
            and self._grouped_bnb4bit_ready()
        ):
            try:
                grouped = self._forward_grouped_bnb4bit(
                    hidden_states, router_indices, routing_weights,
                    batch_size, num_tokens, num_experts, top_k,
                )
                if grouped is not None:
                    return grouped
            except Exception as exc:
                # Checkpoint early-stop is control flow; an OOM should surface, not retry the loop.
                from torch.utils import checkpoint as _ckpt
                control = tuple(
                    c for c in (getattr(_ckpt, "_StopRecomputationError", None), getattr(_ckpt, "CheckpointError", None))
                    if c is not None
                )
                if isinstance(exc, control) or isinstance(exc, torch.OutOfMemoryError):
                    raise
                import os as _os
                if _os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
                    import traceback; traceback.print_exc()
                # fall through to the per-expert loop

        if self.training:
            with torch.no_grad():
                flat_experts = router_indices.flatten()  # [tokens * topk]
                token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)
                
                sorted_idx = flat_experts.argsort(stable=True)
                sorted_tokens = token_ids[sorted_idx]
                
                # bincount on purpose: the .tolist() below already syncs.
                counts = torch.bincount(flat_experts, minlength=num_experts).tolist()
            
            next_states = torch.zeros_like(hidden_states, dtype=torch.float32, device=hidden_states.device)
            offset = 0
            
            for expert_idx in range(num_experts):
                count = counts[expert_idx]
                if count == 0:
                    continue
                # Use pre-computed indices (no torch.where needed)
                token_idx = sorted_tokens[offset:offset + count]
                current_state = hidden_states[token_idx]

                gate_up = self.gate_up_projs[expert_idx](current_state)
                gated_output = swiglu_torch_forward(gate_up, self.alpha, self.limit)
                # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                # gate = gate.clamp(min=None, max=self.limit)
                # up = up.clamp(min=-self.limit, max=self.limit)
                # glu = gate * torch.sigmoid(gate * self.alpha)
                # gated_output = (up + 1) * glu
                out = self.down_projs[expert_idx](gated_output)
                
                weighted_output = out * routing_weights[token_idx, expert_idx, None].to(torch.float32)
                next_states.index_add_(0, token_idx, weighted_output)
                
                offset += count
            
            next_states = next_states.view(batch_size, -1, self.hidden_size)
            return next_states.to(hidden_states.dtype)
        else:
            X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
            gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
            gate_up = torch.stack(gate_up_list, dim=0)
            fused = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = X_rep.dtype)
            # gate = gate_up[..., ::2]
            # up_h = gate_up[..., 1::2]
            # gate = gate.clamp(max=self.limit)
            # up_h = up_h.clamp(min=-self.limit, max=self.limit)
            # glu = gate * torch.sigmoid(gate * self.alpha)
            # fused = (up_h + 1) * glu
            out_list = [down_l(fused[e]) for e, down_l in enumerate(self.down_projs)]
            outs = torch.stack(out_list, dim=0)
            rw = routing_weights.transpose(0, 1).unsqueeze(-1)
            mixed = (outs.to(torch.float32) * rw.to(torch.float32)).sum(dim=0)
            return mixed.view(batch_size, -1, self.hidden_size).to(hidden_states.dtype)

pass


class GptOssTopKRouterBnb4bit(nn.Module):
    """
    GPT OSS Router with weight/bias as direct nn.Parameter (not nested Linear) for BnB 4bit.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.dtype = dtype_from_config(config)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, dtype=self.dtype))
        self.bias = nn.Parameter(torch.zeros(self.num_experts, dtype=self.dtype))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Accept checkpoints that used a Linear router (linear.weight/bias)
        linear_weight_key = prefix + "linear.weight"
        linear_bias_key = prefix + "linear.bias"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        moved_weight = False
        moved_bias = False
        if linear_weight_key in state_dict and weight_key not in state_dict:
            state_dict[weight_key] = state_dict.pop(linear_weight_key)
            moved_weight = True
        if linear_bias_key in state_dict and bias_key not in state_dict:
            state_dict[bias_key] = state_dict.pop(linear_bias_key)
            moved_bias = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if moved_weight:
            if linear_weight_key in unexpected_keys:
                unexpected_keys.remove(linear_weight_key)
            if weight_key in missing_keys:
                missing_keys.remove(weight_key)
        if moved_bias:
            if linear_bias_key in unexpected_keys:
                unexpected_keys.remove(linear_bias_key)
            if bias_key in missing_keys:
                missing_keys.remove(bias_key)
        if self.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            self.weight.data = self.weight.data.to(self.dtype)
        if self.bias is not None and self.bias.dtype not in (
            torch.float16, torch.bfloat16, torch.float32
        ):
            self.bias.data = self.bias.data.to(self.dtype)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states.to(self.weight.dtype), self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        dtype = torch.float32 if router_logits.dtype == torch.float16 else router_logits.dtype
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=torch.float32).to(dtype)
        router_scores = torch.zeros_like(router_logits, dtype=dtype).scatter_(1, router_indices, router_top_value)
        if transformers_version >= Version("5.0.0"):
            return router_logits, router_scores, router_indices
        else:
            return router_scores, router_indices


pass


def patch_gpt_oss_bnb4bit():
    """
    Patch transformers to use BnB 4bit compatible classes for pre-quantized models.
    Call before loading models saved with BitsAndBytes (linear-based expert structure).

    Usage:
        from unsloth_zoo.temporary_patches.gpt_oss import patch_gpt_oss_bnb4bit
        patch_gpt_oss_bnb4bit()  # Call before loading the model
        model = FastLanguageModel.from_pretrained(...)
    """
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss", e)

    # Store original classes for restoration
    if not hasattr(transformers.models.gpt_oss.modeling_gpt_oss, '_original_GptOssExperts'):
        transformers.models.gpt_oss.modeling_gpt_oss._original_GptOssExperts = \
            transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts
        transformers.models.gpt_oss.modeling_gpt_oss._original_GptOssTopKRouter = \
            transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter

    # Replace with BnB 4bit versions; preserve original symbol names for compiler-generated modules.
    GptOssExpertsBnb4bit.__name__ = "GptOssExperts"
    GptOssExpertsBnb4bit.__qualname__ = "GptOssExperts"

    transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts = GptOssExpertsBnb4bit
    # Use unsloth GptOssTopKRouter (self.linear = nn.Linear): BnB 4-bit checkpoints store
    # router.linear.weight/bias. GptOssTopKRouterBnb4bit remapped keys via _load_from_state_dict,
    # but transformers v5 bypasses it (accelerate's set_module_tensor_to_device), so weights
    # stayed randomly initialized -> high loss (~4-5).
    transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter = GptOssTopKRouter

    logger.info("Unsloth: Patched GPT OSS with BitsAndBytes 4bit compatible classes")
    os.environ["UNSLOTH_GPT_OSS_BNB4BIT_PATCHED"] = "1"

    # Inject BnB helpers so compiler-generated modules can import them.
    m = transformers.models.gpt_oss.modeling_gpt_oss
    m._RouterLinearParams  = _RouterLinearParams
    m.swiglu_torch_forward = swiglu_torch_forward
    m.dtype_from_config    = dtype_from_config
    m.transformers_version = transformers_version
    m.Version              = Version

    _rebind_gpt_oss_compiled_classes()
    return True


pass


def _gpt_oss_class_is_bnb4bit(cls):
    # A BnB router (compiled or not) builds `self.linear`; stock builds `self.weight`.
    if cls is GptOssExpertsBnb4bit:
        return True
    init = getattr(cls, "__init__", None)
    return "linear" in getattr(getattr(init, "__code__", None), "co_names", ())


def _rebind_gpt_oss_compiled_classes():
    # The compiler runs once per process, so its module keeps the first load's flavor of these classes.
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss as modeling
    except Exception:
        return
    want_bnb = modeling.GptOssExperts is GptOssExpertsBnb4bit
    seen = set()
    for cls in list(vars(modeling).values()):
        if not isinstance(cls, type):
            continue
        for fn in vars(cls).values():
            g = getattr(fn, "__globals__", None)
            if g is None or id(g) in seen:
                continue
            seen.add(id(g))
            if not str(g.get("__name__", "")).startswith("unsloth_compiled_module_gpt_oss"):
                continue
            for name in ("GptOssExperts", "GptOssTopKRouter"):
                if isinstance(g.get(name), type) and _gpt_oss_class_is_bnb4bit(g[name]) != want_bnb:
                    g[name] = getattr(modeling, name)


def restore_gpt_oss_original():
    """
    Restore original GPT-OSS classes (undo BnB 4bit patch).
    """
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        if hasattr(transformers.models.gpt_oss.modeling_gpt_oss, '_original_GptOssExperts'):
            transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts = \
                transformers.models.gpt_oss.modeling_gpt_oss._original_GptOssExperts
            transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter = \
                transformers.models.gpt_oss.modeling_gpt_oss._original_GptOssTopKRouter
            logger.info("Unsloth: Restored original GPT OSS classes")
            _rebind_gpt_oss_compiled_classes()
            return True
    except Exception:
        pass
    return False

def _normalized_unsloth_model_name() -> str:
    return os.environ.get("UNSLOTH_MODEL_NAME", "").replace("-", "_")


def _should_use_gpt_oss_bnb4bit() -> bool:
    """
    Use BnB-compatible 4-bit experts (default when load_in_4bit active).
    UNSLOTH_GPT_OSS_BNB4BIT_DISABLE=1 forces the BF16 path.
    """
    if "gpt_oss" not in _normalized_unsloth_model_name():
        return False
    if "_load_in_4bit_" not in _normalized_unsloth_model_name():
        return False
    return os.environ.get("UNSLOTH_GPT_OSS_BNB4BIT_DISABLE", "0") != "1"


def _is_gpt_oss_4bit_load() -> bool:
    return "_load_in_4bit_" in _normalized_unsloth_model_name()


def _is_transformers_v5() -> bool:
    return transformers_version >= Version("5.0.0.dev0")


_GPT_OSS_COMPILED_MODULE = "unsloth_compiled_module_gpt_oss"
_GPT_OSS_FLAVOR_MARKER    = ".unsloth_gpt_oss_compiled_flavor"


def _gpt_oss_cache_locations():
    """Dirs that may hold the compiled gpt_oss module: the configured location and the
    temp-fallback used when it is not writable. Resolved without the compiler so it never
    triggers distributed coordination at patch time."""
    locs = []
    loc = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
    if loc:
        locs.append(loc)
        try:
            import tempfile
            locs.append(os.path.join(tempfile.gettempdir(), os.path.basename(loc)))
        except Exception:
            pass
    # De-dup, preserve order (configured first = primary).
    seen, out = set(), []
    for _l in locs:
        if _l and _l not in seen:
            seen.add(_l); out.append(_l)
    return out


def _gpt_oss_cache_location_is_trusted(loc):
    """Whether this process may write the flavor marker into `loc`.

    Deliberately weaker than `compile_cache._is_trusted_directory`, which gates
    LOADING executable artifacts and so walks every ancestor and refuses any group
    write: this only writes a flavor string through an O_NOFOLLOW descriptor, beside
    a compiled module the library itself writes 0644 into the same directory.
    Ownership alone is the wrong question, since a shared cache belongs to whoever
    built it first. The real one is whether everyone who can create an entry here is
    already trusted by the sharing group: ours, or group-write to a group we are in.
    """
    try:
        directory_stat = os.lstat(loc)
    except FileNotFoundError:
        return True   # a location we are about to create ourselves
    except Exception:
        return False
    try:
        if not stat.S_ISDIR(directory_stat.st_mode):
            return False
        if os.name != "posix":
            return True
        if directory_stat.st_mode & 0o002:
            return False
        if directory_stat.st_uid == os.geteuid():
            return True
        # A group member can already replace the 0644 module beside the marker.
        if not (directory_stat.st_mode & 0o020):
            return False
        try:
            groups = set(os.getgroups()) | {os.getgid(), os.getegid()}
        except Exception:
            return False
        return directory_stat.st_gid in groups
    except Exception:
        return False
pass


def _gpt_oss_marker_mode(loc):
    """0664 in a cache shared with our group, 0600 in one only we can reach.

    The marker records a flavor, not a secret, and in a shared cache every member has
    to be able to read AND rewrite it. Owner-only there means the member who switched
    flavor deletes the stale module but cannot record the new one, leaving the two
    disagreeing.
    """
    try:
        if os.name != "posix":
            return 0o600
        return 0o664 if (os.lstat(loc).st_mode & 0o020) else 0o600
    except Exception:
        return 0o600
pass


def _gpt_oss_replace_marker(marker_path, desired_flavor, mode = 0o600):
    """Land the marker by replacing the name: mkstemp then os.replace.

    Two things at once. A link sitting at `marker_path` is replaced rather than
    written through, and the update needs only directory write, so a group member can
    record a flavor into a marker file owned by whoever built the cache first.
    """
    import tempfile
    directory = os.path.dirname(marker_path) or "."
    descriptor, temporary_path = tempfile.mkstemp(
        prefix = f".{os.path.basename(marker_path)}.", suffix = ".tmp", dir = directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding = "utf-8") as f:
            descriptor = None
            f.write(desired_flavor)
        os.chmod(temporary_path, mode)      # mkstemp is 0600; a shared cache needs more
        os.replace(temporary_path, marker_path)
    except BaseException:
        if descriptor is not None:
            try: os.close(descriptor)
            except OSError: pass
        try: os.remove(temporary_path)
        except OSError: pass
        raise
pass


def _gpt_oss_write_marker(loc, desired_flavor):
    """Write the flavor marker without following a link out of `loc`.

    Same flags as `compiler._write_bytes_durably`: O_NONBLOCK because a planted FIFO
    would otherwise block the load forever, S_ISREG because one with a reader
    attached opens fine and would swallow the marker instead.
    """
    marker_path = os.path.join(loc, _GPT_OSS_FLAVOR_MARKER)
    mode = _gpt_oss_marker_mode(loc)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow or mode != 0o600:
        # Shared: the existing marker belongs to whoever built the cache, so only a
        # replacement can update it. Also the no-atomic-no-follow-open case, where an
        # lstat first would be a time of check.
        return _gpt_oss_replace_marker(marker_path, desired_flavor, mode)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= no_follow
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(marker_path, flags, mode)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"Unsloth: refusing to write the gpt-oss flavor marker in `{loc}`: not a regular file.")
        with os.fdopen(descriptor, "w", encoding = "utf-8") as f:
            descriptor = None
            f.write(desired_flavor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
pass


def _invalidate_gpt_oss_compiled_module(locations = None):
    """Drop the cached compiled gpt_oss module (sys.modules + on-disk .py/.pyc) so it is
    rebuilt against the CURRENT router/experts classes. The single per-model-type file
    hardcodes the BnB or stock layout, so a stale one survives a 4bit<->16bit switch with the
    wrong classes. Cleans every candidate location, or only `locations` when given: a
    location that is stale says nothing about the others, and sweeping them all let one
    planted candidate delete a perfectly good cache on every load."""
    try:
        import sys as _sys
        import importlib, importlib.util
        _sys.modules.pop(_GPT_OSS_COMPILED_MODULE, None)
        for loc in (_gpt_oss_cache_locations() if locations is None else locations):
            _f = os.path.join(loc, _GPT_OSS_COMPILED_MODULE + ".py")
            if os.path.isfile(_f):
                try:
                    os.remove(_f)
                except OSError:
                    pass
                # Drop the .pyc too so a stale module isn't re-imported from __pycache__.
                try:
                    _pyc = importlib.util.cache_from_source(_f)
                    if os.path.isfile(_pyc):
                        os.remove(_pyc)
                except Exception:
                    pass
        # Forget any cached finder/directory state for these paths.
        try:
            importlib.invalidate_caches()
        except Exception:
            pass
    except Exception:
        pass  # best-effort: cache invalidation must never break loading


def _sync_gpt_oss_compiled_flavor(desired_flavor):
    """Invalidate the on-disk compiled gpt_oss module when it was built for a DIFFERENT flavor
    ("bnb4bit" vs "stock") than this load needs, independently of the in-process flag (unset in
    a fresh process, so a fresh 16bit load after a 4bit one would import the stale BnB classes
    and hit "weights not initialized"). A marker file records the built flavor; on a mismatch or
    missing marker the stale module is dropped for the compiler to regenerate."""
    try:
        locations = _gpt_oss_cache_locations()
        # Per location, never global: marking all of them stale because one is lets
        # anyone who can create the predictable temp candidate delete a valid primary
        # cache on every load.
        stale = []
        for loc in locations:
            module_path = os.path.join(loc, _GPT_OSS_COMPILED_MODULE + ".py")
            if not os.path.isfile(module_path):
                continue
            if not _gpt_oss_cache_location_is_trusted(loc):
                # Any marker here is not ours, and the compiler imports the module
                # beside it with no such gate, so do not trust what it claims.
                stale.append(loc)
                continue
            on_disk = None
            marker_path = os.path.join(loc, _GPT_OSS_FLAVOR_MARKER)
            if os.path.isfile(marker_path):
                try:
                    with open(marker_path, "r", encoding = "utf-8") as f:
                        on_disk = f.read().strip()
                except Exception:
                    on_disk = None
            if on_disk != desired_flavor:
                stale.append(loc)
        if stale:
            _invalidate_gpt_oss_compiled_module(stale)
        # Record this load's flavor; always at the primary location, the temp fallback only if used.
        for idx, loc in enumerate(locations):
            if idx != 0 and not os.path.isdir(loc):
                continue
            # Never write into a directory the sharing group does not already trust.
            if not _gpt_oss_cache_location_is_trusted(loc):
                continue
            try:
                os.makedirs(loc, exist_ok = True)
                _gpt_oss_write_marker(loc, desired_flavor)
            except Exception:
                pass
    except Exception:
        pass


def patch_gpt_oss_bnb4bit_auto():
    """
    Auto-patch GPT-OSS for BnB 4-bit when load_in_4bit is active.
    Set UNSLOTH_GPT_OSS_BNB4BIT_DISABLE=1 to opt out.
    """
    # Cross-process safety: invalidate a stale on-disk module built for the other flavor (the
    # in-process flag is unset in a fresh process). Sole cache invalidator: drops the file only
    # on a real mismatch, so a matching cache is reused. Gated to gpt-oss loads.
    if "gpt_oss" in _normalized_unsloth_model_name():
        _sync_gpt_oss_compiled_flavor("bnb4bit" if _should_use_gpt_oss_bnb4bit() else "stock")

    if not _should_use_gpt_oss_bnb4bit():
        # Check the installed class too: the env flag can be cleared or inherited independently of it.
        try:
            import transformers.models.gpt_oss.modeling_gpt_oss as _modeling
            _installed = _modeling.GptOssExperts is GptOssExpertsBnb4bit
        except Exception:
            _installed = False
        if _installed or os.environ.get("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "0") == "1":
            restore_gpt_oss_original()
            os.environ["UNSLOTH_GPT_OSS_BNB4BIT_PATCHED"] = "0"
        return
    # patch_gpt_oss_bnb4bit() injects BnB helpers so the compiler resolves all symbols. A stale
    # stock module is handled by _sync above; a matching bnb module is reused, not recompiled.
    patch_gpt_oss_bnb4bit()
    # Inference path avoids torch.compile for 4-bit
    try:
        global moe_forward_inference
        moe_forward_inference = torch.compiler.disable(moe_forward_inference)
    except Exception:
        pass


TEMPORARY_PATCHES.append(patch_gpt_oss_bnb4bit_auto)


# Combo kernels uses too much VRAM for low memory GPUs
from unsloth_zoo.device_type import DEVICE_TYPE

# UNSLOTH_ALLOW_CPU=1 keeps DEVICE_TYPE="cuda" on GPU-less hosts, so guard
# with is_available() like device_synchronize() does.
if DEVICE_TYPE == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
    # Only total capacity is needed. mem_get_info() can create a device context at
    # import time, leaving otherwise idle processes with persistent device memory.
    device_memory = torch.xpu.get_device_properties(0).total_memory
elif DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
    # Integrated NVIDIA parts may report only the carve-out, so cuda_total_memory raises it to
    # the driver total when that is larger. Discrete cards and HIP are unchanged.
    from unsloth_zoo.integrated_device import cuda_total_memory
    device_memory = cuda_total_memory(0)
else:
    device_memory = 0
use_combo_kernels = False if device_memory/1024/1024/1024 <= 40 else True

# coordinate_descent_tuning used to be driven by use_combo_kernels too, so asking for combo
# kernels also bought the tuning. Measured apart on T4/L4/A100/B200: combo kernels are free
# (decode compile 7.36s vs 7.48s off), the tuning costs +28% on A100 and +32% on L4 decode
# compile for no measurable gain anywhere. Default it off, opt in to re-measure.
#
# The 40GB test above is in GiB, so an A100-SXM4-40GB reports 39.49 and falls BELOW it.
# Left as-is: combo kernels measured as no-effect on both sides of the boundary.
use_coordinate_descent = os.environ.get("UNSLOTH_COORDINATE_DESCENT_TUNING", "0") == "1"

fused_torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False, # Too slow
    shape_padding = True,
    cudagraphs = True,
    coordinate_descent_tuning = use_coordinate_descent, # Very slow, and no measured gain
    combo_kernels = use_combo_kernels,
    memory_planning = True,
    multi_kernel = False, # Fails on torch 2.10 nightly
    use_block_ptr = True,
    logging = UNSLOTH_ENABLE_LOGGING,
)
no_combo_fused_torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False, # Too slow
    shape_padding = True,
    cudagraphs = True,
    coordinate_descent_tuning = use_coordinate_descent, # Very slow, and no measured gain
    combo_kernels = False, # Breaks on attention
    memory_planning = True,
    multi_kernel = False, # Fails on torch 2.10 nightly
    use_block_ptr = True,
    logging = UNSLOTH_ENABLE_LOGGING,
)


@_torch_compile(dynamic=None, fullgraph=True, options=fused_torch_compile_options)
def moe_forward_inference(self, hidden_states):
    """Torch compile for forward inference path only with CUDAGraphs"""
    # Router
    router_out = self.router(hidden_states)
    if isinstance(router_out, tuple) and len(router_out) == 3:
        _, router_scores, router_indices = router_out
    else:
        router_scores, router_indices = router_out
    routing_weights = router_scores
    moe = self.experts
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, moe.hidden_size)

    num_experts = routing_weights.shape[1]
    X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)

    # ModuleList (old style) vs 3D parameters (new style)
    if hasattr(moe, "gate_up_projs"):
        gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(moe.gate_up_projs)]
        gate_up = torch.stack(gate_up_list, dim=0)
        dtype = torch.float32 if hidden_states.dtype != torch.bfloat16 else hidden_states.dtype
        fused = swiglu_torch_forward(gate_up, moe.alpha, moe.limit, dtype=dtype)

        fused = fused.to(dtype)
        device_type = fused.device.type if isinstance(fused.device.type, str) and fused.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            out_list = [down_l(fused[e].to(dtype)) for e, down_l in enumerate(moe.down_projs)]
        outs = torch.stack(out_list, dim=0)
    else:
        # 3D parameter style: gate_up_proj (E, H, 2I); bmm (E, N, H) @ (E, H, 2I) -> (E, N, 2I)
        gate_up = torch.bmm(X_rep, moe.gate_up_proj) + moe.gate_up_proj_bias[..., None, :]
        dtype = torch.float32 if hidden_states.dtype != torch.bfloat16 else hidden_states.dtype
        fused = swiglu_torch_forward(gate_up, moe.alpha, moe.limit, dtype=dtype)

        fused = fused.to(dtype)
        device_type = fused.device.type if isinstance(fused.device.type, str) and fused.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # down_proj (E, I, H); bmm (E, N, I) @ (E, I, H) -> (E, N, H)
            outs = torch.bmm(fused.to(dtype), moe.down_proj) + moe.down_proj_bias[..., None, :]

    rw = routing_weights.to(dtype).transpose(0, 1).unsqueeze(-1)
    mixed = (outs * rw).sum(dim=0)
    return mixed.view(batch_size, -1, moe.hidden_size).to(hidden_states.dtype)


pass


@torch_compile(dynamic=True, fullgraph=True)
def moe_router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, self.bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    dtype = torch.float32 if router_logits.dtype == torch.float16 else router_logits.dtype
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=torch.float32).to(dtype)
    router_scores = torch.zeros_like(router_logits, dtype = dtype).scatter_(1, router_indices, router_top_value)
    return router_scores, router_indices


pass


# Combo Kernels errors with InductorError: AttributeError: 'NullKernelHandler' object has no attribute 'index_to_str'
@_torch_compile(
    dynamic=None, fullgraph=True, options=no_combo_fused_torch_compile_options
)
def _moe_forward_inference_bf16_kernel(
    hidden_states, routing_weights, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, limit, alpha, hidden_size
):
    """Inner compiled kernel for BF16 MoE inference - works with raw tensors only."""
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, hidden_size)
    num_experts = routing_weights.shape[1]
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_size)

    gate_up = (
        torch.bmm(hidden_states, gate_up_proj) + gate_up_proj_bias[..., None, :]
    )
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate.to(torch.float32) * alpha).to(gate.dtype)
    next_states = torch.bmm(((up + 1) * glu), down_proj)
    next_states = next_states + down_proj_bias[..., None, :]
    next_states = next_states.view(num_experts, batch_size, -1, hidden_size)
    next_states = (
        next_states
        * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    )
    next_states = next_states.sum(dim=0)
    return next_states


def _unwrap_peft_experts(module):
    """Unwrap PEFT ParamWrapper chain to get the actual experts module."""
    while hasattr(module, 'base_layer'):
        module = module.base_layer
    return module


# Weak values: freed with the last packed parameter (and so model) that holds the slot.
_MXFP4_DECODE_SLOTS = weakref.WeakValueDictionary()


class _Mxfp4DecodeSlot:
    """Shared decode buffer; `lock` spans decode to enqueue, `event` = last reader so other streams wait."""

    __slots__ = ("stack", "lock", "event", "__weakref__")

    def __init__(self, shape, dtype, device):
        # A Parameter so the CUDA-graphed kernel reads it in place; other tensors are copied every call.
        self.stack = nn.Parameter(torch.zeros(shape, dtype = dtype, device = device), requires_grad = False)
        self.lock = threading.Lock()
        self.event = None


def _device_stream_api(device):
    device_type = getattr(device, "type", None)
    if device_type not in ("cuda", "xpu"):
        return None
    api = getattr(torch, device_type, None)
    if api is None or not hasattr(api, "Event") or not hasattr(api, "current_stream"):
        return None
    return api


def _mxfp4_decode_slot(param, dtype, role = ""):
    key = (role, tuple(param._original_shape), dtype, param.device)
    slot = _MXFP4_DECODE_SLOTS.get(key)
    if slot is None:
        slot = _Mxfp4DecodeSlot(param._original_shape, dtype, param.device)
        _MXFP4_DECODE_SLOTS[key] = slot
    held = getattr(param, "_unsloth_decode_stacks", None)
    if held is None:
        held = param._unsloth_decode_stacks = {}
    held[key] = slot
    return slot


def _mxfp4_decode_stack(param, dtype, token_counts, role = "", slot = None):
    """Only routed experts are rewritten; the kernel weighs the others by 0, so stale slices are harmless."""
    slot = slot or _mxfp4_decode_slot(param, dtype, role)
    api = _device_stream_api(param.device) if slot.event is not None else None
    if api is not None:
        api.current_stream(param.device).wait_event(slot.event)
    param.dequantize(dtype, token_counts = token_counts, out = slot.stack.data)
    return slot.stack


@_torch_compile(dynamic=None, fullgraph=True, options=no_combo_fused_torch_compile_options)
def _moe_forward_inference_mxfp4_kernel(
    hidden_states, routing_weights, router_indices,
    gu_blocks, gu_scales, gate_up_proj_bias, gu_trans,
    dn_blocks, dn_scales, down_proj_bias, dn_trans,
    limit, alpha, hidden_size,
):
    """Decode-time MoE on packed MXFP4 stacks: sort the routed (token, expert) rows, two fused grouped GEMMs."""
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm_compiled
    batch_size = hidden_states.shape[0]
    x = hidden_states.reshape(-1, hidden_size)
    num_experts = routing_weights.shape[1]
    top_k = router_indices.shape[1]
    # Each token's experts ascending, so the fixed-order sum below adds them in the dense path's expert order.
    flat = router_indices.sort(dim = -1).values.reshape(-1)
    counts = (flat[:, None] == torch.arange(num_experts, device = flat.device)).sum(0, dtype = torch.int32)
    order = torch.argsort(flat, stable = True)
    token = order // top_k
    expert = flat[order]
    # Biases may be float32 (the dense path promotes too); the GEMMs take the bf16 activations, as autocast would.
    gate_up = mxfp4_grouped_mm_compiled(x[token], gu_blocks, gu_scales, counts, gu_trans) + gate_up_proj_bias[expert]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate.to(torch.float32) * alpha).to(gate.dtype)
    inter = ((up + 1) * glu).to(x.dtype)
    down = mxfp4_grouped_mm_compiled(inter, dn_blocks, dn_scales, counts, dn_trans) + down_proj_bias[expert]
    weighted = down * routing_weights[token, expert][:, None]
    # Fixed-order fp32 sum: index_add_'s atomics made greedy decode nondeterministic.
    rows = torch.empty(weighted.shape, dtype = torch.float32, device = x.device)
    rows = rows.index_copy_(0, order, weighted.to(torch.float32))
    out = rows.view(x.shape[0], top_k, hidden_size).sum(1)
    return out.to(weighted.dtype).view(batch_size, -1, hidden_size)


def _mxfp4_static_operands(param, experts_module, proj_type):
    """(blocks, scales, transpose_b) as long-lived plain tensors, so compiled code sees stable inputs."""
    cached = getattr(param, "_unsloth_fused_operands", None)
    if cached is not None and cached[0].data_ptr() == param.data_ptr() and cached[1] is param.mxfp4_scales:
        return cached[0], cached[2], cached[3]
    from .moe_utils import _mxfp4_expert_layout
    layout = _mxfp4_expert_layout(param, proj_type, experts_module.hidden_size, getattr(experts_module, "_unsloth_model_type", None), experts_module)
    transpose_b = bool(param.mxfp4_transposed) if layout is None else layout[1]
    scales = param.mxfp4_scales
    if scales.device != param.device:
        scales = param.mxfp4_scales = scales.to(param.device)
    blocks = param.data
    scales = scales.contiguous()
    # Fixed addresses: CUDA graph trees then replay one graph per layer instead of copying the stacks in.
    torch._dynamo.mark_static_address(blocks)
    torch._dynamo.mark_static_address(scales)
    param._unsloth_fused_operands = (blocks, param.mxfp4_scales, scales, transpose_b)
    return blocks, scales, transpose_b


def _mxfp4_fused_decode_enabled(param, dtype):
    from .moe_utils import _mxfp4_fused_enabled
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm_op
    if os.environ.get("UNSLOTH_MXFP4_FUSED_GEMM") == "dequant":
        return False
    return mxfp4_grouped_mm_op is not None and _mxfp4_fused_enabled(param, dtype)


# matmul_ogs decode: bf16 activations x the packed MXFP4 stacks, routed rows only, no dequantized copy.
_OGS_WEIGHTS = {}
_OGS_FAILED = set()


@functools.lru_cache(maxsize = 1)
def _ogs_modules():
    from unsloth_zoo.triton_kernels_compat import get_triton_kernels
    import importlib
    tk = get_triton_kernels()  # top-level install or vLLM's vendored copy, never a lasting alias
    if tk is None:
        return None
    try:
        sub = lambda name: importlib.import_module(f"{tk.__name__}.{name}")
        return tk, sub("matmul_ogs"), sub("routing"), sub("swiglu"), sub("tensor"), sub("tensor_details.layout")
    except Exception:
        return None


def _ogs_kernels():
    """UNSLOTH_MXFP4_OGS=0 keeps decode on the exact packed path; read per call so tests can flip it."""
    if os.environ.get("UNSLOTH_MXFP4_OGS", "1") == "0":
        return None
    return _ogs_modules()


def _ogs_track(param, blocks, scales):
    """Tie the cached matmul_ogs view (which holds the packed storage) to `param`: dropped on rebind or free."""
    key = (blocks.data_ptr(), scales.data_ptr(), tuple(blocks.shape))
    old = getattr(param, "_unsloth_ogs_key", None)
    if old == key:
        return
    if old is not None:
        _OGS_WEIGHTS.pop(old, None)
    param._unsloth_ogs_key = key
    weakref.finalize(param, _OGS_WEIGHTS.pop, key, None)


def _ogs_weight(blocks, scales, in_dim):
    """Triton view of a packed (E, out, G, 16) stack; strided layouts share the packed bytes."""
    key = (blocks.data_ptr(), scales.data_ptr(), tuple(blocks.shape))
    cached = _OGS_WEIGHTS.get(key)
    if cached is not None:
        return cached
    _, mo, _, _, T, L = _ogs_kernels()
    E, out_dim = blocks.shape[:2]
    rows = blocks.reshape(E, out_dim, -1)
    value_layout, value_opts, strided = _mxfp4_layout_arguments(L, rows)
    w = T.convert_layout(T.wrap_torch_tensor(rows.transpose(-2, -1), dtype = T.FP4), value_layout, **value_opts)
    w.shape = torch.Size([E, in_dim, out_dim])
    w_scale = T.convert_layout(T.wrap_torch_tensor(scales.reshape(E, out_dim, -1).transpose(-2, -1)), strided)
    precision = mo.PrecisionConfig(weight_scale = w_scale, flex_ctx = mo.FlexCtx(rhs_data = mo.InFlexData()))
    _OGS_WEIGHTS[key] = (w, precision)
    return w, precision


@torch.library.custom_op("unsloth_zoo::mxfp4_ogs_moe", mutates_args = ())
def mxfp4_ogs_moe(
    x: torch.Tensor, router_logits: torch.Tensor, top_k: int,
    gu_blocks: torch.Tensor, gu_scales: torch.Tensor, gu_bias: torch.Tensor,
    dn_blocks: torch.Tensor, dn_scales: torch.Tensor, dn_bias: torch.Tensor,
    alpha: float, limit: float,
) -> torch.Tensor:
    tk, mo, rt, sw, _, _ = _ogs_kernels()
    hidden = x.shape[-1]
    intermediate = dn_blocks.shape[2] * 32
    w_gu, p_gu = _ogs_weight(gu_blocks, gu_scales, hidden)
    w_dn, p_dn = _ogs_weight(dn_blocks, dn_scales, intermediate)
    # Top-k + softmax over the top-k, as the router does; only an exact bf16 tie at k can pick another expert.
    # expt_indx (the router's own indices) is not used: it mis-builds the expert histogram in vLLM 0.30's copy.
    routing_data, gather, scatter = rt.routing(router_logits, top_k)
    act = mo.FusedActivation(mo.FnSpecs("swiglu", sw.swiglu_fn, ("alpha", "limit")), (alpha, limit), 2)
    inter = mo.matmul_ogs(
        x, w_gu, gu_bias, routing_data, gather_indx = gather, precision_config = p_gu, fused_activation = act,
    )
    return mo.matmul_ogs(
        inter, w_dn, dn_bias, routing_data, scatter_indx = scatter, precision_config = p_dn,
        gammas = routing_data.gate_scal,
    )


@mxfp4_ogs_moe.register_fake
def _(x, router_logits, top_k, gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias, alpha, limit):
    return torch.empty_like(x)


@_torch_compile(dynamic=None, fullgraph=True, options=no_combo_fused_torch_compile_options)
def _moe_forward_inference_ogs_kernel(
    hidden_states, router_weight, router_bias, top_k,
    gu_blocks, gu_scales, gate_up_proj_bias, dn_blocks, dn_scales, down_proj_bias, alpha, limit, hidden_size,
):
    batch_size = hidden_states.shape[0]
    x = hidden_states.reshape(-1, hidden_size)
    logits = F.linear(x.to(router_weight.dtype), router_weight, router_bias)
    out = torch.ops.unsloth_zoo.mxfp4_ogs_moe(
        x.to(torch.bfloat16), logits, top_k,
        gu_blocks, gu_scales, gate_up_proj_bias.to(torch.float32),
        dn_blocks, dn_scales, down_proj_bias.to(torch.float32), alpha, limit,
    )
    return out.to(hidden_states.dtype).view(batch_size, -1, hidden_size)


def _mxfp4_ogs_decode(self, hidden_states):
    """None unless both stacks are packed, nothing needs gradients, and triton_kernels runs them."""
    if torch.is_grad_enabled() or _ogs_kernels() is None:
        return None
    moe = _unwrap_peft_experts(self.experts)
    gate_up, down = moe.gate_up_proj, moe.down_proj
    if not (is_mxfp4_expert_param(gate_up) and is_mxfp4_expert_param(down)):
        return None
    if not (gate_up.mxfp4_transposed and down.mxfp4_transposed) or gate_up.device in _OGS_FAILED:
        return None
    from unsloth_zoo.triton_kernels_compat import matmul_ogs_available
    if not matmul_ogs_available(gate_up.device):  # once per device: import, layout, launch and accuracy
        _OGS_FAILED.add(gate_up.device)
        return None
    gu_blocks, gu_scales, _ = _mxfp4_static_operands(gate_up, moe, "gate_up")
    dn_blocks, dn_scales, _ = _mxfp4_static_operands(down, moe, "down")
    _ogs_track(gate_up, gu_blocks, gu_scales)
    _ogs_track(down, dn_blocks, dn_scales)
    try:
        return _moe_forward_inference_ogs_kernel(
            hidden_states, self.router.weight, self.router.bias, self.router.top_k,
            gu_blocks, gu_scales, moe.gate_up_proj_bias, dn_blocks, dn_scales, moe.down_proj_bias,
            float(moe.alpha), float(moe.limit), moe.hidden_size,
        )
    except Exception as error:
        # One failure per device drops to the exact packed path for the rest of the process.
        _OGS_FAILED.add(gate_up.device)
        logger.warning(f"Unsloth: matmul_ogs MXFP4 decode failed ({error}); using the exact packed path.")
        return None


def _has_active_expert_lora(experts):
    """True when a PEFT adapter on the expert parameters would change the output."""
    m = experts
    while hasattr(m, "base_layer"):
        if hasattr(m, "lora_A") and len(m.lora_A) and not getattr(m, "merged", False) \
                and not getattr(m, "disable_adapters", False):
            return True
        m = m.base_layer
    return False


def moe_forward_inference_bf16(self, hidden_states):
    """Wrapper that extracts weights from ParameterModule before calling the compiled kernel."""
    out = _mxfp4_ogs_decode(self, hidden_states)
    if out is not None:
        return out
    if _has_active_expert_lora(self.experts):
        # The fused kernel below reads the unwrapped base expert weights, so it would
        # silently drop expert LoRA; the module forward applies it.
        out = self(hidden_states)
        return out[0] if isinstance(out, tuple) else out
    router_scores, router_indices = moe_router_forward(self.router, hidden_states)
    routing_weights = router_scores

    moe = _unwrap_peft_experts(self.experts)

    # Extract weights (ParameterModule wrapping nn.Linear vs direct 3D tensor) before compiled region
    gate_up_proj = moe.gate_up_proj
    if hasattr(gate_up_proj, "get_param"):
        gate_up_proj = gate_up_proj.get_param()
    elif hasattr(gate_up_proj, "weight"):
        gate_up_proj = gate_up_proj.weight

    down_proj = moe.down_proj
    if hasattr(down_proj, "get_param"):
        down_proj = down_proj.get_param()
    elif hasattr(down_proj, "weight"):
        down_proj = down_proj.weight

    if (
        is_mxfp4_expert_param(gate_up_proj) and is_mxfp4_expert_param(down_proj)
        and _mxfp4_fused_decode_enabled(gate_up_proj, hidden_states.dtype)
    ):
        gu_blocks, gu_scales, gu_trans = _mxfp4_static_operands(gate_up_proj, moe, "gate_up")
        dn_blocks, dn_scales, dn_trans = _mxfp4_static_operands(down_proj, moe, "down")
        return _moe_forward_inference_mxfp4_kernel(
            hidden_states, routing_weights, router_indices,
            gu_blocks, gu_scales, moe.gate_up_proj_bias, gu_trans,
            dn_blocks, dn_scales, moe.down_proj_bias, dn_trans,
            moe.limit, moe.alpha, moe.hidden_size,
        )

    slots = []
    if is_mxfp4_expert_param(gate_up_proj) or is_mxfp4_expert_param(down_proj):
        from .moe_utils import count_tokens_per_expert
        counts = count_tokens_per_expert(router_indices.reshape(-1), routing_weights.shape[1], torch.int32)
        if is_mxfp4_expert_param(gate_up_proj):
            slots.append(_mxfp4_decode_slot(gate_up_proj, hidden_states.dtype, "gate_up"))
        if is_mxfp4_expert_param(down_proj):
            slots.append(_mxfp4_decode_slot(down_proj, hidden_states.dtype, "down"))

    with contextlib.ExitStack() as held:
        # Fixed order, so two threads never wait on each other's second lock.
        for slot in sorted(slots, key = id):
            held.enter_context(slot.lock)
        if is_mxfp4_expert_param(gate_up_proj):
            gate_up_proj = _mxfp4_decode_stack(gate_up_proj, hidden_states.dtype, counts, slot = slots[0])
        if is_mxfp4_expert_param(down_proj):
            down_proj = _mxfp4_decode_stack(down_proj, hidden_states.dtype, counts, slot = slots[-1])
        out = _moe_forward_inference_bf16_kernel(
            hidden_states,
            routing_weights,
            gate_up_proj,
            moe.gate_up_proj_bias,
            down_proj,
            moe.down_proj_bias,
            moe.limit,
            moe.alpha,
            moe.hidden_size,
        )
        api = _device_stream_api(hidden_states.device) if slots else None
        if api is not None:
            stream = api.current_stream(hidden_states.device)
            for slot in slots:
                if slot.event is None:
                    slot.event = api.Event()
                slot.event.record(stream)
    return out




class GptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        bsz, qlen, hd = hidden_states.shape
        if qlen == 1 and not self.training:
            return moe_forward_inference(self, hidden_states), None
        router_out = self.router(hidden_states)
        if isinstance(router_out, tuple) and len(router_out) == 3:
            _, router_scores, router_indices = router_out
        else:
            router_scores, router_indices = router_out
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores


pass


# ============================================================================
# GPT OSS MoE LoRA Support using grouped GEMM kernels
# ============================================================================

# IMPORTS FROM MOE UTILS
from .moe_utils import (
    _check_grouped_gemm_available,
    _TORCH_GROUPED_MM_AVAILABLE,
    _check_torch_grouped_mm_supported,
    native_moe_grouped_mm,
    _get_moe_lora_weights,
    _apply_lora_grouped_mm,
    _get_lora_wrapper_for_param,
    select_moe_backend,
    patch_param_wrapper_for_moe,
    forward_native_grouped_mm,
    forward_native_moe_loop,
    # torch_native_forward,
)


def patch_gpt_oss_moe_for_lora():
    """
    Patch GptOssExperts (3D parameter tensors) forward to use grouped GEMM kernels with LoRA.
    Only patches forward, not the class itself, so original structure loads weights correctly.
    """
    if "gpt_oss" not in _normalized_unsloth_model_name():
        return
    if _is_gpt_oss_4bit_load() or _should_use_gpt_oss_bnb4bit():
        # 4-bit loads should keep quantized weights and use default PEFT LoRA.
        return
    if not _is_transformers_v5():
        # Split-LoRA grouped_mm path is only needed for transformers v5+
        return
    patch_param_wrapper_for_moe()

    try:
        import transformers.models.gpt_oss.modeling_gpt_oss

        # Get the ORIGINAL class - don't replace it!
        GptOssExpertsClass = transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch GPT OSS MoE for LoRA: {e}")
        return

    if hasattr(GptOssExpertsClass, "_unsloth_lora_patched"):
        return

    backend = select_moe_backend()

    if backend == "grouped_mm":
        forward = forward_native_grouped_mm
    else:
        # torch_native_forward expects the bnb4bit ModuleList layout; the stock
        # 3D GptOssExperts needs the generic loop, which handles gpt-oss
        # activation, biases, and the separated LoRA stash.
        forward = forward_native_moe_loop

    # Store original forward and patch - but DON'T replace the class!
    GptOssExpertsClass._original_forward = GptOssExpertsClass.forward
    GptOssExpertsClass.forward = forward
    GptOssExpertsClass._unsloth_lora_patched = True

    if UNSLOTH_ENABLE_LOGGING:
        backend_desc = {
            "grouped_mm": "torch._grouped_mm (batched, fastest)",
            "unsloth_triton": "Triton kernels",
            "native_torch": "loop fallback (slower)",
        }.get(backend, backend)
        logger.info(
            f"Unsloth: Patched GPT OSS MoE for LoRA training using {backend_desc}"
        )


TEMPORARY_PATCHES.append(patch_gpt_oss_moe_for_lora)


# ============================================================================
# MXFP4 (4-bit) GPT OSS MoE LoRA Support
# ============================================================================

_MXFP4_LORA_PATH_LOGGED = False
_OGS_DEFAULT_CHUNK_BYTES = 2 << 30


def _ogs_chunk_bytes():
    try:
        return max(1, int(os.environ.get("UNSLOTH_MXFP4_EXPERT_CHUNK_MB", "")) << 20)
    except ValueError:
        return _OGS_DEFAULT_CHUNK_BYTES


def _ogs_dense_chunks(weight, precision_config, dtype):
    """Yield (first, last, dense (E_c, K, N)) exact decodes per expert chunk; scales group along K, so W^T needs a decode."""
    if isinstance(weight, torch.Tensor):
        yield 0, weight.shape[0], weight.to(dtype)
        return
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize

    E, K, N = (int(x) for x in weight.shape)
    data = weight.storage.layout.unswizzle_data(weight.storage.data)
    scale = precision_config.weight_scale
    scale = scale.storage.layout.unswizzle_data(scale.storage.data)
    # Back to the checkpoint layout (E, N, K / 32, 16) / (E, N, K / 32) the exact decoder reads.
    blocks, scales = data.transpose(-2, -1), scale.transpose(-2, -1)
    if tuple(blocks.shape) != (E, N, K // 2) or tuple(scales.shape) != (E, N, K // 32):
        raise RuntimeError(
            f"Unsloth: unexpected MXFP4 layout {tuple(blocks.shape)} / {tuple(scales.shape)} for {(E, K, N)}"
        )
    step = max(1, min(E, _ogs_chunk_bytes() // (K * N * torch.empty((), dtype = dtype).element_size())))
    for first in range(0, E, step):
        last = min(E, first + step)
        yield first, last, mxfp4_dequantize(
            blocks[first:last].contiguous().view(last - first, N, K // 32, 16),
            scales[first:last].contiguous(), dtype = dtype, transpose = True,
        )


def _ogs_expert_offsets(routing_data):
    # One host sync per backward: row ranges of each expert in the expert-sorted order.
    counts = routing_data.expt_hist.tolist()
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + int(count))
    return offsets


def _matmul_ogs_for(weight, routing_data = None):
    """matmul_ogs from the triton_kernels copy owning `weight` (other copies reject its Tensor); dense weights use `routing_data`'s."""
    import importlib
    if isinstance(weight, torch.Tensor):
        owner = type(routing_data).__module__ if routing_data is not None else ""
        if owner.endswith(".routing"):
            root = owner[: -len(".routing")]
        else:
            from unsloth_zoo.triton_kernels_compat import get_triton_kernels
            tk = get_triton_kernels()
            if tk is None:
                raise RuntimeError("Unsloth: triton_kernels is required for native MXFP4 GPT OSS training.")
            root = tk.__name__
    else:
        root = type(weight).__module__.rsplit(".tensor", 1)[0]
    return importlib.import_module(root + ".matmul_ogs").matmul_ogs


class _OgsGateUp(torch.autograd.Function):
    """Expert-sorted rows ``x[src // k] @ W[e] + b[e]`` via matmul_ogs; frozen MXFP4 ``W``."""

    @staticmethod
    def forward(ctx, x, bias, module, routing_data, gather_idx):
        # One read: on the still-packed path each property access decodes the whole stack.
        weight = module.gate_up_proj
        matmul_ogs = _matmul_ogs_for(weight, routing_data)

        out = matmul_ogs(
            x.to(torch.bfloat16), weight, bias, routing_data,
            gather_indx = gather_idx, precision_config = module.gate_up_proj_precision_config,
        )
        ctx.module, ctx.routing_data, ctx.gather_idx = module, routing_data, gather_idx
        ctx.x_shape, ctx.x_dtype = x.shape, x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        module, routing_data = ctx.module, ctx.routing_data
        k = routing_data.n_expts_act
        rows = ctx.gather_idx.src_indx.long()
        need_x, need_bias = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_out = grad_out.to(torch.bfloat16)
        grad_rows = grad_out.new_empty((grad_out.shape[0], ctx.x_shape[-1])) if need_x else None
        grad_bias = None
        if need_bias:
            grad_bias = torch.zeros_like(module.gate_up_proj_bias, dtype = torch.float32)
        offsets = _ogs_expert_offsets(routing_data)
        for first, last, dense in _ogs_dense_chunks(
            module.gate_up_proj, module.gate_up_proj_precision_config, torch.bfloat16,
        ):
            for e in range(first, last):
                lo, hi = offsets[e], offsets[e + 1]
                if hi == lo:
                    continue
                if need_x:
                    grad_rows[lo:hi] = grad_out[lo:hi] @ dense[e - first].t()
                if need_bias:
                    grad_bias[e] = grad_out[lo:hi].float().sum(0)
            del dense
        grad_x = None
        if need_x:
            grad_x = torch.zeros(ctx.x_shape, dtype = torch.float32, device = grad_out.device)
            grad_x.index_add_(0, rows // k, grad_rows.float())
            grad_x = grad_x.to(ctx.x_dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(module.gate_up_proj_bias.dtype)
        return grad_x, grad_bias, None, None, None


class _OgsDown(torch.autograd.Function):
    """``out[dst // k] += gamma * (h @ W[e] + b[e])`` via matmul_ogs; frozen MXFP4 ``W``."""

    @staticmethod
    def forward(ctx, h, gammas, bias, module, routing_data, scatter_idx):
        weight = module.down_proj
        matmul_ogs = _matmul_ogs_for(weight, routing_data)

        out = matmul_ogs(
            h, weight, bias, routing_data, scatter_indx = scatter_idx,
            precision_config = module.down_proj_precision_config,
            gammas = None if gammas is None else gammas.detach(),
        )
        ctx.module, ctx.routing_data, ctx.scatter_idx = module, routing_data, scatter_idx
        ctx.h_dtype, ctx.bias = h.dtype, bias
        # h only feeds the routing-weight gradient.
        ctx.save_for_backward(h if ctx.needs_input_grad[1] else None, gammas)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        h, gammas = ctx.saved_tensors
        module, routing_data = ctx.module, ctx.routing_data
        k = routing_data.n_expts_act
        dst = ctx.scatter_idx.dst_indx.long()
        valid = (dst >= 0).unsqueeze(-1)
        grad_rows = grad_out.to(torch.bfloat16)[(dst // k).clamp_min(0)] * valid
        scaled = grad_rows if gammas is None else grad_rows * gammas.to(grad_rows.dtype).unsqueeze(-1)
        need_h, need_gamma, need_bias = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        bias = ctx.bias
        grad_h = scaled.new_empty((scaled.shape[0], int(module.down_proj.shape[1]))) if need_h else None
        grad_gamma = torch.zeros(dst.shape[0], dtype = torch.float32, device = dst.device) if need_gamma else None
        grad_bias = torch.zeros_like(bias, dtype = torch.float32) if need_bias else None
        offsets = _ogs_expert_offsets(routing_data)
        for first, last, dense in _ogs_dense_chunks(
            module.down_proj, module.down_proj_precision_config, torch.bfloat16,
        ):
            for e in range(first, last):
                lo, hi = offsets[e], offsets[e + 1]
                if hi == lo:
                    continue
                weight = dense[e - first]
                if need_h:
                    grad_h[lo:hi] = scaled[lo:hi] @ weight.t()
                if need_gamma:
                    y = (h[lo:hi] @ weight).float()
                    if bias is not None:
                        y = y + bias[e].float()
                    grad_gamma[lo:hi] = (grad_rows[lo:hi].float() * y).sum(-1)
                if need_bias:
                    grad_bias[e] = scaled[lo:hi].float().sum(0)
            del dense
        if grad_h is not None:
            grad_h = grad_h.to(ctx.h_dtype)
        if grad_gamma is not None:
            grad_gamma = grad_gamma.to(gammas.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)
        return grad_h, grad_gamma, grad_bias, None, None, None


class _OgsSwiglu(torch.autograd.Function):
    """gpt-oss clamped SwiGLU on interleaved gate/up; saves only the bf16 pre-activation."""

    @staticmethod
    def forward(ctx, pre_activation, alpha, limit):
        ctx.alpha, ctx.limit = alpha, limit
        ctx.save_for_backward(pre_activation)
        return swiglu_torch_forward(pre_activation, alpha, limit)

    @staticmethod
    def backward(ctx, grad_out):
        (pre_activation,) = ctx.saved_tensors
        alpha, limit = ctx.alpha, ctx.limit
        gate, linear = pre_activation[..., ::2].float(), pre_activation[..., 1::2].float()
        grad = grad_out.float()
        if limit is not None:
            gate_kept, linear_kept = gate <= limit, linear.abs() <= limit
            gate, linear = gate.clamp(max = limit), linear.clamp(min = -limit, max = limit)
        sigmoid = torch.sigmoid(alpha * gate)
        grad_gate = grad * (sigmoid + alpha * gate * sigmoid * (1 - sigmoid)) * (linear + 1)
        grad_linear = grad * gate * sigmoid
        if limit is not None:
            grad_gate, grad_linear = grad_gate * gate_kept, grad_linear * linear_kept
        grad_pre = torch.empty_like(pre_activation)
        grad_pre[..., ::2], grad_pre[..., 1::2] = grad_gate, grad_linear
        return grad_pre, None, None


def _ogs_lora_offsets(routing_data):
    return torch.cumsum(routing_data.expt_hist, dim = 0, dtype = torch.int32)


def mxfp4_ogs_experts_forward(self, hidden_states, routing_data, gather_idx, scatter_idx, gate_up_lora = None, down_lora = None):
    """Differentiable native MXFP4 experts: matmul_ogs forward, exact chunked dequant in backward."""
    k = routing_data.n_expts_act
    alpha = getattr(self, "alpha", 1.702)
    limit = getattr(self, "limit", 7.0)
    # triton_kernels' routing() backpropagates through gate_scal into the router logits.
    gammas = routing_data.gate_scal
    pre_activation = _OgsGateUp.apply(hidden_states, self.gate_up_proj_bias, self, routing_data, gather_idx)
    if gate_up_lora is not None:
        first, second, scaling, _ = gate_up_lora
        permuted = hidden_states[gather_idx.src_indx.long() // k].to(torch.bfloat16)
        pre_activation = pre_activation + _apply_lora_grouped_mm(
            permuted, first, second, _ogs_lora_offsets(routing_data), scaling,
            grouped_mm_func = native_moe_grouped_mm,
        ).to(pre_activation.dtype)
    swiglu_output = _OgsSwiglu.apply(pre_activation, alpha, limit)
    out = _OgsDown.apply(swiglu_output, gammas, self.down_proj_bias, self, routing_data, scatter_idx)
    if down_lora is not None:
        first, second, scaling, _ = down_lora
        delta = _apply_lora_grouped_mm(
            swiglu_output, first, second, _ogs_lora_offsets(routing_data), scaling,
            grouped_mm_func = native_moe_grouped_mm,
        )
        dst = scatter_idx.dst_indx.long()
        valid = dst >= 0
        delta = (delta * gammas.unsqueeze(-1).to(delta.dtype))[valid]
        out = out.index_add(0, dst[valid] // k, delta.to(out.dtype))
    return out


@torch.compiler.disable
def forward_mxfp4_gpt_oss_with_lora(
    self,
    hidden_states: torch.Tensor,
    routing_data = None,
    gather_idx = None,
    scatter_idx = None,
    router_indices = None,
    routing_weights = None,
) -> torch.Tensor:
    """Native MXFP4 GPT OSS experts (matmul_ogs) with optional expert LoRA; exact dequant in backward."""
    if not is_triton_kernels_available():
        raise RuntimeError(
            "triton_kernels is required for native MXFP4 GPT OSS forward pass. "
            "Either:\n"
            "  1. Install triton_kernels from OpenAI, OR\n"
            "  2. Load model with dequantization: Mxfp4Config(dequantize=True), OR\n"
            "  3. Use the BF16 model: 'unsloth/gpt-oss-20b-BF16'\n"
            "Set UNSLOTH_MXFP4_NO_DEQUANTIZE=0 (default) to auto-dequantize."
        )
    gate_up_wrapper = _get_lora_wrapper_for_param(self, "gate_up_proj")
    down_wrapper = _get_lora_wrapper_for_param(self, "down_proj")
    gate_up_lora = _get_moe_lora_weights(gate_up_wrapper) if gate_up_wrapper is not None else None
    down_lora = _get_moe_lora_weights(down_wrapper) if down_wrapper is not None else None

    global _MXFP4_LORA_PATH_LOGGED
    if not _MXFP4_LORA_PATH_LOGGED:
        _MXFP4_LORA_PATH_LOGGED = True
        logger.warning_once(
            f"Unsloth: GPT-OSS MoE path: MXFP4 + triton_kernels. "
            f"LoRA={gate_up_lora is not None or down_lora is not None}, experts={self.num_experts}."
        )

    hidden_states, routing_data, gather_idx, scatter_idx, leading_shape = _mxfp4_experts_routing(
        self, hidden_states, routing_data, gather_idx, scatter_idx, router_indices, routing_weights,
    )
    with torch_cuda_device(hidden_states.device):
        if (
            gate_up_lora is None and down_lora is None
            and not (torch.is_grad_enabled() and hidden_states.requires_grad)
        ):
            out = self._original_forward(hidden_states, routing_data, gather_idx, scatter_idx)
        else:
            out = mxfp4_ogs_experts_forward(
                self, hidden_states, routing_data, gather_idx, scatter_idx,
                gate_up_lora = gate_up_lora, down_lora = down_lora,
            )
    if leading_shape is not None:
        out = out.reshape(*leading_shape, out.shape[-1])
    return out


def patch_mxfp4_gpt_oss_for_lora():
    """
    Patch MXFP4 GPT OSS experts for LoRA training (unsloth/gpt-oss-20b).
    Requires triton_kernels for native MXFP4 matmul; else use Mxfp4Config(dequantize=True)
    or the BF16 model 'unsloth/gpt-oss-20b-BF16'.
    """
    # ParamWrapper for MoE separated LoRA (v5 only)
    if _is_transformers_v5():
        patch_param_wrapper_for_moe()

    try:
        import transformers.integrations.mxfp4

        Mxfp4GptOssExpertsClass = getattr(
            transformers.integrations.mxfp4, "Mxfp4GptOssExperts", None
        )
        if Mxfp4GptOssExpertsClass is None:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(
                    "Unsloth: Mxfp4GptOssExperts not found in transformers.integrations.mxfp4"
                )
            return
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch MXFP4 GPT OSS for LoRA: {e}")
        return

    if hasattr(Mxfp4GptOssExpertsClass, "_unsloth_mxfp4_lora_patched"):
        return

    # Without triton_kernels, MXFP4 weights cannot be used directly (need dequant or BF16 model)
    if is_triton_kernels_available():
        # Native MXFP4 + LoRA, keeps weights quantized
        Mxfp4GptOssExpertsClass._original_forward = Mxfp4GptOssExpertsClass.forward
        Mxfp4GptOssExpertsClass.forward = forward_mxfp4_gpt_oss_with_lora
        Mxfp4GptOssExpertsClass._unsloth_mxfp4_lora_patched = True
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Patched MXFP4 GPT OSS MoE for LoRA training")
    else:
        # No triton_kernels: don't patch; model errors helpfully if MXFP4 used without dequant
        Mxfp4GptOssExpertsClass._unsloth_mxfp4_lora_patched = True
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                "Unsloth: triton_kernels is not installed. MXFP4 GPT OSS will NOT be patched for LoRA.\n"
                "To train GPT OSS with LoRA, either:\n"
                "  1. Install triton_kernels from OpenAI (for native MXFP4), OR\n"
                "  2. Use Mxfp4Config(dequantize=True) when loading (dequantizes to bf16), OR\n"
                "  3. Use the BF16 model: 'unsloth/gpt-oss-20b-BF16'"
            )


TEMPORARY_PATCHES.append(patch_mxfp4_gpt_oss_for_lora)


_MXFP4_DEQUANT_WARNED = False


def should_dequantize_mxfp4():
    """
    Whether MXFP4 should be dequantized to bf16. False only when
    UNSLOTH_MXFP4_NO_DEQUANTIZE="1" AND triton_kernels is available.
    Memory for GPT-OSS 20B: ~10GB quantized vs ~40GB dequantized to bf16.
    """
    global _MXFP4_DEQUANT_WARNED

    if not UNSLOTH_MXFP4_NO_DEQUANTIZE:
        # Default: dequantize to bf16
        if UNSLOTH_ENABLE_LOGGING and not _MXFP4_DEQUANT_WARNED:
            _MXFP4_DEQUANT_WARNED = True
            logger.warning(
                "Unsloth: MXFP4 will be dequantized to bf16 (~4x memory increase). "
                "To keep 4-bit: set UNSLOTH_MXFP4_NO_DEQUANTIZE=1 and install triton_kernels."
            )
        return True

    if not is_triton_kernels_available():
        if UNSLOTH_ENABLE_LOGGING and not _MXFP4_DEQUANT_WARNED:
            _MXFP4_DEQUANT_WARNED = True
            logger.warning(
                "Unsloth: UNSLOTH_MXFP4_NO_DEQUANTIZE=1 but triton_kernels not available. "
                "Will dequantize MXFP4 to bf16 (~4x memory increase). "
                "Install triton_kernels to keep 4-bit quantized weights."
            )
        return True  # triton_kernels required for native MXFP4

    if UNSLOTH_ENABLE_LOGGING and not _MXFP4_DEQUANT_WARNED:
        _MXFP4_DEQUANT_WARNED = True
        logger.info("Unsloth: Keeping MXFP4 quantized (triton_kernels available)")
    return False  # Keep MXFP4 quantized


@torch.compiler.disable
def _try_grouped_bnb4bit(self, hidden_states, router_indices, routing_weights,
                         batch_size, num_tokens, num_experts, top_k):
    """The grouped bnb-4bit training forward, or None for the per-expert loop. One opaque
    call, so a compiled decoder layer (and its gradient-checkpoint replay) takes a single
    graph break here instead of tracing the readiness checks."""
    if not self._grouped_bnb4bit_ready():
        return None
    try:
        return self._forward_grouped_bnb4bit(
            hidden_states, router_indices, routing_weights,
            batch_size, num_tokens, num_experts, top_k,
        )
    except Exception as exc:
        # Checkpoint early-stop is control flow; an OOM should surface, not retry the loop.
        from torch.utils import checkpoint as _ckpt
        control = tuple(
            c for c in (getattr(_ckpt, "_StopRecomputationError", None), getattr(_ckpt, "CheckpointError", None))
            if c is not None
        )
        if isinstance(exc, control) or isinstance(exc, torch.OutOfMemoryError):
            raise
        if UNSLOTH_ENABLE_LOGGING:
            import traceback; traceback.print_exc()
        return None  # fall through to the per-expert loop


def torch_native_forward(
    self,
    hidden_states: torch.Tensor,
    router_indices = None,
    routing_weights = None
) -> torch.Tensor:

    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    num_tokens = hidden_states.shape[0]
    num_experts = routing_weights.shape[1]
    top_k = router_indices.shape[1]

    # Grouped bnb-4bit fast path. The class dispatches this module-level
    # function (forward is rebound below), so the gate must live here too.
    # fp16 experts keep the loop's fp32 swiglu and fp32 down output there.
    if (
        self.training
        and hasattr(self, "_grouped_bnb4bit_ready")
    ):
        grouped = _try_grouped_bnb4bit(
            self, hidden_states, router_indices, routing_weights,
            batch_size, num_tokens, num_experts, top_k,
        )
        if grouped is not None:
            return grouped

    if self.training:
        with torch.no_grad():
            flat_experts = router_indices.flatten()  # [tokens * topk]
            token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)
            
            sorted_idx = flat_experts.argsort(stable=True)
            sorted_tokens = token_ids[sorted_idx]
            
            # bincount on purpose: the .tolist() below already syncs.
            counts = torch.bincount(flat_experts, minlength=num_experts).tolist()
        
        next_states = torch.zeros_like(hidden_states, dtype=torch.float32, device=hidden_states.device)
        offset = 0
        
        for expert_idx in range(num_experts):
            count = counts[expert_idx]
            if count == 0:
                continue
            
            # Use pre-computed indices (no torch.where needed)
            token_idx = sorted_tokens[offset:offset + count]
            current_state = hidden_states[token_idx]
            
            gate_up = self.gate_up_projs[expert_idx](current_state)
            down_proj = self.down_projs[expert_idx]
            gated_output = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = torch.float32)

            gated_output = gated_output.to(torch.float32)
            device_type = gated_output.device.type if isinstance(gated_output.device.type, str) and gated_output.device.type != "mps" else "cpu"
            # Three separate things keep this float32, none of them redundant. On the
            # forced-float32 path only, a quantized down_proj computes in float32 via
            # _pre_set_compute_dtype, set in unsloth's loader; without that rule the layer
            # takes the run's ordinary compute dtype, bfloat16 included. The float32
            # gated_output is what Linear4bit restores the output to, since it captures
            # inp_dtype and casts back. And autocast off is what protects the unquantized
            # fallback, which takes F.linear and ignores compute_dtype. It does not keep the
            # adapter matmuls in float32 -- the forced-float32 LoRA path casts those to
            # float16 itself.
            with torch.autocast(device_type=device_type, enabled=False):
                out = down_proj(gated_output)
            
            weighted_output = out.to(torch.float32) * routing_weights[token_idx, expert_idx, None].to(torch.float32)
            next_states.index_add_(0, token_idx, weighted_output)
            
            offset += count
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states.to(torch.float32)
    elif (
        num_tokens * top_k <= ROUTED_MAX_SLOTS
        and (routed := routed_experts_forward(self, hidden_states, router_indices, routing_weights)) is not None
    ):
        # Decode-sized eval calls read only the routed experts, with no host sync.
        return routed.view(batch_size, -1, self.hidden_size)
    else:
        if not torch.is_grad_enabled() and not torch.compiler.is_compiling():
            # Eager prefill builds the routed tables a compiled decode step then reads.
            prepare_routed_experts(self)
        X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
        gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
        gate_up = torch.stack(gate_up_list, dim=0)
        dtype = torch.float32 if hidden_states.dtype != torch.bfloat16 else hidden_states.dtype
        fused = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = dtype)
        # gate = gate_up[..., ::2]
        # up_h = gate_up[..., 1::2]
        # gate = gate.clamp(max=self.limit)
        # up_h = up_h.clamp(min=-self.limit, max=self.limit)
        # glu = gate * torch.sigmoid(gate * self.alpha)
        # fused = (up_h + 1) * glu

        # Autocast off for the down projection only. As above, it is the unquantized
        # fallback that needs this; on the forced-float32 path a quantized layer gets
        # float32 from _pre_set_compute_dtype instead. This branch also runs in bfloat16,
        # where no float32 rule is registered and the layer stays in bfloat16.
        device_type = fused.device.type if isinstance(fused.device.type, str) and fused.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            out_list = [
                down_l(fused[e].to(dtype))
                for e, down_l in enumerate(self.down_projs)
            ]
        outs = torch.stack(out_list, dim=0)
        rw = routing_weights.transpose(0, 1).unsqueeze(-1)
        mixed = (outs.to(dtype) * rw.to(dtype)).sum(dim=0)
        return mixed.view(batch_size, -1, self.hidden_size).to(hidden_states.dtype)
    pass
pass

# torch_native_forward protects the down projection three ways in float16 training: swiglu and
# gated_output in float32, which is also the dtype Linear4bit restores its output to;
# _pre_set_compute_dtype (registered by unsloth's loader on the forced-float32 path only) for
# the quantized compute; and autocast disabled for the unquantized fallback, which ignores
# compute_dtype. A bfloat16 run registers no float32 rule and keeps the layer in bfloat16.
GptOssExpertsBnb4bit.forward = torch_native_forward

def patch_gpt_oss_linearized():
    """
    Patch GPT OSS for 4bit loading with grouped_mm support.
    Only patches GptOssExperts.forward; keeps original classes for proper weight loading.
    """
    if "gpt_oss" not in _normalized_unsloth_model_name(): return
    if "_load_in_4bit_" not in _normalized_unsloth_model_name(): return
    if _should_use_gpt_oss_bnb4bit(): return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss", e)

    # Patch only GptOssExperts.forward (not the class) to keep 4-bit weight loading working
    backend = select_moe_backend()

    if backend == "grouped_mm":

        def experts_forward(
            self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
        ) -> torch.Tensor:
            return forward_native_grouped_mm(self, hidden_states, router_indices, routing_weights)
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts.forward = experts_forward
    else:

        def experts_forward(
            self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
        ) -> torch.Tensor:
            return torch_native_forward(self, hidden_states, router_indices, routing_weights)

        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts.forward = experts_forward

    if UNSLOTH_ENABLE_LOGGING: logger.info(f"Unsloth: Patched GPT OSS MoE for 4bit loading (backend: {backend})")
    return


pass
TEMPORARY_PATCHES.append(patch_gpt_oss_linearized)


def patch_GptOssAttention():
    if os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0": return
    # Uncompiled flex_attention backward has a dtype bug in PyTorch
    # (sdpa_dense_backward: expected Float got BFloat16). The inplace eager
    # fallback also uses out= matmul which is incompatible with autograd.
    # Skip the patch and let stock transformers eager attention handle sinks.
    if UNSLOTH_COMPILE_DISABLE: return
    if "gpt_oss" not in _normalized_unsloth_model_name(): return
    try:
        from unsloth_zoo.flex_attention import (
            flex_attention_with_sink,
            is_flex_attention_decoding,
            flex_attention_with_sink_decoding,
            flex_attention_add_sinks,
        )

        assert flex_attention_with_sink is not None
    except Exception as e:
        return raise_error("flex_attention_with_sink", e)
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention", e)

    torch._dynamo.config.cache_size_limit = 256

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep): (batch,
        num_key_value_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim).
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    F_softmax = torch.nn.functional.softmax
    F_dropout = nn.functional.dropout
    matmul = torch.matmul
    def _align_kv_to_mask(key_states, value_states, attention_mask):
        # Eager `attn_weights += attention_mask` requires KV length == mask key dim. On some
        # transformers/torch combos (e.g. transformers 5.x on torch < 2.11) the KV cache returns
        # more positions than the mask covers (pre-allocated slots), crashing full-attention layers.
        # Surplus positions are masked out anyway, so trim KV (and mask) to the shorter length.
        if attention_mask is None or not hasattr(attention_mask, "shape"):
            return key_states, value_states, attention_mask
        kvlen = key_states.shape[-2]
        masklen = attention_mask.shape[-1]
        if masklen < kvlen:
            key_states = key_states[:, :, :masklen, :]
            value_states = value_states[:, :, :masklen, :]
        elif masklen > kvlen:
            attention_mask = attention_mask[:, :, :, :kvlen]
        return key_states, value_states, attention_mask

    def inplace_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        key_states, value_states, attention_mask = _align_kv_to_mask(
            key_states, value_states, attention_mask
        )

        bsz, n_heads, qlen, _  = query.shape
        bsz, n_heads, kvlen, _ = key_states.shape
        # promote_types, not result_type: result_type returns a dtype, which graph-breaks Dynamo.
        out_dtype = torch.promote_types(query.dtype, key_states.dtype)
        combined_logits = key_states.new_empty((bsz, n_heads, qlen, kvlen + 1), dtype=out_dtype)

        if torch.compiler.is_compiling():
            # Dynamo cannot trace out= into a non-contiguous slice; Inductor fuses the copy.
            combined_logits[:, :, :, :kvlen] = matmul(query, key_states.transpose(2, 3))
            attn_weights = combined_logits[:, :, :, :kvlen]
        else:
            attn_weights = matmul(query, key_states.transpose(2, 3), out = combined_logits[:,:,:,:kvlen])
        attn_weights *= scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        # combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits[:, :, :, -1] = module.sinks.reshape(1, -1, 1)

        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        combined_logits[:] = F_softmax(combined_logits, dim=-1, dtype=torch.float32)
        probs = combined_logits
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = F_dropout(scores, p=dropout, training=module.training, inplace=True)
        attn_weights = attn_weights.to(value_states.dtype)
        attn_output = matmul(attn_weights, value_states, out = query)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    pass

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        key_states, value_states, attention_mask = _align_kv_to_mask(
            key_states, value_states, attention_mask
        )
        attn_weights = matmul(query, key_states.transpose(2, 3))
        attn_weights *= scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)

        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        combined_logits[:] = F_softmax(combined_logits, dim=-1, dtype=torch.float32)
        probs = combined_logits
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = F_dropout(scores, p=dropout, training=module.training, inplace=True)
        attn_weights = attn_weights.to(value_states.dtype)
        attn_output = matmul(attn_weights, value_states, out = query)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    pass

    apply_rotary_pos_emb = torch_compile(apply_rotary_pos_emb)
    if False:  # Version(torch.__version__) >= Version("2.10.0"):
        eager_attention_forward = torch_compile(eager_attention_forward, dynamic=None, fullgraph=True)
    else:
        # Too many recompilation failures on 2.8.0, 2.9.0
        eager_attention_forward = inplace_eager_attention_forward

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_dtype = getattr(past_key_value, "dtype", None)
            if cache_dtype is None and hasattr(past_key_value, "layers"):
                try:
                    cache_layer = past_key_value.layers[self.layer_idx]
                    if hasattr(cache_layer, "keys") and cache_layer.keys is not None:
                        cache_dtype = cache_layer.keys.dtype
                except Exception:
                    cache_dtype = None
            if cache_dtype is not None and key_states.dtype != cache_dtype:
                key_states = key_states.to(cache_dtype)
                value_states = value_states.to(cache_dtype)
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        if key_states.dtype != query_states.dtype or value_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)
            value_states = value_states.to(query_states.dtype)

        # flex_attention_with_sink only works for training since KV cache is wrong
        # switch to flex_attention_with_sink which allows all to work
        # if is_flex_attention_decoding(self, query_states) and has_static_cache:
        #     attn_output, logsumexp = flex_attention_with_sink_decoding(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #     )
        #     attn_output = flex_attention_add_sinks(
        #         self,
        #         attn_output,
        #         logsumexp,
        #     )
        # else:
        #     attn_output = flex_attention_with_sink(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask,
        #         has_static_cache = has_static_cache,
        #     )
        # attn_weights = None
        if self.training:
            attn_output = flex_attention_with_sink(
                self,
                query_states,
                key_states,
                value_states,
            )
            attn_weights = None
        else:
            # Weirdly for inference, flex attention returns gibberish
            # Most likely due to left padding
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,  # diff with Llama
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    pass

    functions = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)

    functions.append(forward)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)

    functions.append(forward)

    # Transformers >= 5.0 dropped `cache_position` from GptOssAttention.forward's
    # signature, so the variants above fail the strict signature match and the
    # attention patch silently does not apply (leaving stock attention, which
    # then mismatches the sliding-window mask the model patch builds). Add a
    # `past_key_values` variant without `cache_position` (it still arrives via
    # **kwargs) so the patch installs on transformers 5.x too. Only the
    # `past_key_values` spelling is needed: the singular `past_key_value` naming
    # predates transformers dropping `cache_position`, so a singular + no
    # `cache_position` signature does not exist in any transformers release.
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        cache_position = kwargs.pop("cache_position", None)
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)

    functions.append(forward)
    patch_function_past_key_values(transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention, "forward", functions)
    # forward_function picks flex_attention_with_sink from self.training alone, so
    # once it is live, training ignores whatever mask the factories build.
    global _GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED
    _GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED = (
        getattr(transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention.forward, "__module__", "")
        == forward_function.__module__
    )
    # Set env variable for padding purposes
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "1"
pass
TEMPORARY_PATCHES.append(patch_GptOssAttention)


# A module global so tests can wrap it.
from unsloth_zoo.offloaded_embedding import offloaded_embedding as _offloaded_embedding


def patch_GptOssModel():
    if os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0": return
    if UNSLOTH_COMPILE_DISABLE: return
    if "gpt_oss" not in _normalized_unsloth_model_name(): return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel
        from transformers.models.gpt_oss.modeling_gpt_oss import MoeModelOutputWithPast
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel", e)
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import DynamicCache
    except Exception as e:
        raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel", e)
        DynamicCache = lambda *args, **kwargs: None

    torch._dynamo.config.cache_size_limit = 256

    # Disable mask creations since we don't need them for GPT-OSS
    import transformers.masking_utils
    import transformers.generation.utils
    def _find_config(args, kwargs):
        config = kwargs.get("config", None)
        if config is None:
            for arg in args:
                if hasattr(arg, "_attn_implementation"):
                    config = arg
                    break
        return config

    # Records self.training on the config so the mask wrapper can key on the state
    # that selects the attention backend instead of guessing from requires_grad.
    def _record_training_state(GptOssModel):
        original = getattr(GptOssModel, "forward", None)
        if original is None or getattr(original, _TRAINING_FLAG_ATTR, False):
            return
        @functools.wraps(original)
        def forward(self, *args, **kwargs):
            config = getattr(self, "config", None)
            previous = getattr(config, _TRAINING_FLAG_ATTR, None)
            if config is not None:
                setattr(config, _TRAINING_FLAG_ATTR, bool(self.training))
            try:
                return original(self, *args, **kwargs)
            finally:
                if config is not None:
                    setattr(config, _TRAINING_FLAG_ATTR, previous)
        setattr(forward, _TRAINING_FLAG_ATTR, True)
        GptOssModel.forward = forward

    def wrap(f):
        def return_attention_mask(*args, **kwargs):
            input_embeds = kwargs.get("input_embeds", None)
            if input_embeds is None:
                input_embeds = kwargs.get("inputs_embeds", None)
            if input_embeds is None:
                for arg in args:
                    if type(arg) is torch.Tensor and arg.is_floating_point():
                        input_embeds = arg
                        break

            # Skipping is only safe when flex_attention_with_sink takes over and
            # ignores this mask, which it picks on self.training, not on
            # _attn_implementation. `requires_grad` is not a training signal:
            # enable_input_require_grads() fires on every LoRA-capable load, so
            # skipping there hands eager attention no causal mask at all.
            _config = _find_config(args, kwargs)
            _is_flex = getattr(_config, "_attn_implementation", None) == "flex_attention"
            _training = getattr(_config, _TRAINING_FLAG_ATTR, None)
            if _training is None:
                # Unusual caller, no model forward recorded it: at least exclude
                # no_grad inference.
                _training = bool(
                    torch.is_grad_enabled()
                    and input_embeds is not None
                    and input_embeds.requires_grad
                )
            # Only zoo's patched forward builds its own BlockMask; stock flex takes
            # causality from whatever this returns, so a flex config is not enough.
            if _training and _GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED:
                if "attention_mask" in kwargs:
                    return kwargs["attention_mask"]
                for arg in args:
                    if (
                        type(arg) is torch.Tensor and
                        arg.dtype in (torch.int32, torch.int64, torch.bool)
                    ):
                        return arg
                return f(*args, **kwargs)
            else:
                # Eager inference path. The config may still have
                # _attn_implementation="flex_attention" (set for training), in
                # which case the underlying mask factory returns a BlockMask.
                # Unsloth uses eager attention for inference (flex with KV
                # cache returns gibberish, see forward_function above), and
                # the eager forward cannot index a BlockMask, raising
                #   TypeError: unsupported operand type(s) for +=:
                #       'Tensor' and 'BlockMask'
                # Temporarily swap to eager so the factory returns a dense
                # 4D float mask (0 / -inf) the eager path can consume.
                if _is_flex:
                    original_impl = _config._attn_implementation
                    _config._attn_implementation = "eager"
                    try:
                        return f(*args, **kwargs)
                    finally:
                        _config._attn_implementation = original_impl
                return f(*args, **kwargs)
            pass
        return return_attention_mask
    pass
    create_causal_mask = getattr(
        transformers.masking_utils,
        "_old_create_causal_mask",
        getattr(transformers.masking_utils, "create_causal_mask", None),
    )
    create_sliding_window_causal_mask = getattr(
        transformers.masking_utils,
        "_old_create_sliding_window_causal_mask",
        getattr(transformers.masking_utils, "create_sliding_window_causal_mask", None),
    )
    if create_causal_mask is None:
        return raise_error("transformers.masking_utils.create_causal_mask")
    if create_sliding_window_causal_mask is None:
        return raise_error("transformers.masking_utils.create_sliding_window_causal_mask")
    if not hasattr(transformers.masking_utils, "__patched_causal_mask__"):
        transformers.masking_utils._old_create_causal_mask = _torch_compile(transformers.masking_utils.create_causal_mask, fullgraph = False, dynamic = True)
        transformers.masking_utils._old_create_sliding_window_causal_mask = _torch_compile(transformers.masking_utils.create_sliding_window_causal_mask, fullgraph = False, dynamic = True)
        transformers.masking_utils.create_causal_mask = wrap(create_causal_mask)
        transformers.masking_utils.create_sliding_window_causal_mask = wrap(create_sliding_window_causal_mask)
        transformers.models.gpt_oss.modeling_gpt_oss.create_causal_mask = transformers.masking_utils.create_causal_mask
        transformers.models.gpt_oss.modeling_gpt_oss.create_sliding_window_causal_mask = transformers.masking_utils.create_sliding_window_causal_mask
        transformers.masking_utils.create_masks_for_generate = wrap(transformers.masking_utils.create_masks_for_generate)
        transformers.generation.utils.create_masks_for_generate = wrap(transformers.generation.utils.create_masks_for_generate)
        transformers.masking_utils.__patched_causal_mask__ = True
    pass

    # transformers 4.x uses `input_embeds` and accepts `cache_position`;
    # transformers 5.x renamed it to `inputs_embeds` and later dropped
    # `cache_position` (deprecated then removed by 5.13). Inspect the mask factory
    # signatures once and pass only the kwargs they actually accept so this patch
    # works across 4.x and the whole 5.x line.
    import inspect as _inspect
    def _mask_factory_params(fn, name):
        # The factory may be wrapped (see misc.py) or torch-compiled, which
        # collapses inspect.signature() to (*args, **kwargs) and would make the
        # kwargs filter below drop every mask argument -> create_causal_mask()
        # called with no args -> "missing required positional arguments". Recover
        # the true parameter names from the pristine reference misc.py saves, then
        # __wrapped__, then the object itself, then a version-aware fallback.
        for cand in (
            getattr(transformers.masking_utils, "_unsloth_original_" + name, None),
            getattr(fn, "__wrapped__", None),
            fn,
        ):
            if cand is None: continue
            try:
                params = set(_inspect.signature(cand).parameters)
            except (TypeError, ValueError):
                continue
            if not params <= {"args", "kwargs", "self"}:
                return params
        # Introspection failed on every candidate (should not happen: misc.py saves
        # the pristine factory and torch.compile exposes __wrapped__). Fall back to
        # the parameter set for the running transformers, choosing the kwargs by
        # major version so a reached fallback never passes an unexpected keyword to
        # the real factory: 4.x uses `input_embeds` and accepts `cache_position`,
        # while 5.x uses `inputs_embeds` and dropped `cache_position` (optional and
        # defaulting to None on early 5.x, removed entirely by 5.13), so omitting it
        # on 5.x is safe across the whole line.
        try:
            _tf_major = int(str(transformers.__version__).split(".", 1)[0])
        except (ValueError, AttributeError, IndexError):
            _tf_major = 5
        if _tf_major < 5:
            return {"config", "input_embeds", "attention_mask",
                    "cache_position", "past_key_values", "position_ids"}
        return {"config", "inputs_embeds", "attention_mask",
                "past_key_values", "position_ids"}
    _ccm_params = _mask_factory_params(create_causal_mask, "create_causal_mask")
    _cswc_params = _mask_factory_params(create_sliding_window_causal_mask, "create_sliding_window_causal_mask")
    _mask_params = _ccm_params | _cswc_params

    def _build_mask_kwargs(config, inputs_embeds, attention_mask, cache_position, past_key_values):
        mk = {
            "config": config,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        if "inputs_embeds" in _mask_params:
            mk["inputs_embeds"] = inputs_embeds
        if "input_embeds" in _mask_params:
            mk["input_embeds"] = inputs_embeds
        if "cache_position" in _mask_params:
            mk["cache_position"] = cache_position
        return mk

    from unsloth_zoo.flex_attention import (
        is_flex_attention_decoding,
        flex_attention_with_sink_decoding,
        flex_attention_add_sinks,
    )

    apply_rotary_pos_emb = torch_compile(apply_rotary_pos_emb)
    try:
        from transformers.integrations.mxfp4 import mlp_forward
    except:
        mlp_forward = None

    def pre_attention_decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        return query_states, key_states, value_states, input_shape
    pass

    # Do flex_attention_with_sink_decoding with cannot be compiled
    # attn_output, logsumexp = flex_attention_with_sink_decoding(
    #     self,
    #     query_states,
    #     key_states,
    #     value_states,
    # )
    def post_attention_decoding(self_attn, attn_output, logsumexp, input_shape):
        attn_output = flex_attention_add_sinks(self_attn, attn_output, logsumexp)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self_attn.o_proj(attn_output)
        return attn_output

    pass

    # RMSNorm forward
    def rms_layernorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        variance += self.variance_epsilon
        hidden_states *= torch.rsqrt_(variance)
        hidden_states *= self.weight.to(hidden_states.device).to(torch.float32)
        return hidden_states.to(input_dtype)  # main diff with Llama
    pass

    # Re-compiling for each new sequence length which is NOT ideal
    @_torch_compile(dynamic = True, fullgraph = False, mode = "reduce-overhead")
    def pre_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        hidden_states = rms_layernorm_forward(self.input_layernorm, hidden_states)
        # Self Attention
        query_states, key_states, value_states, input_shape = pre_attention_decoding(
            self=self.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return query_states, key_states, value_states, input_shape
    pass
    fused_torch_compile_options = get_torch_compile_options(
        epilogue_fusion = True,
        max_autotune = False, # Too slow
        shape_padding = True,
        cudagraphs = True,
        coordinate_descent_tuning = False,
        combo_kernels = False,
        memory_planning = True,
        multi_kernel = False, # Fails on torch 2.10 nightly
        use_block_ptr = True,
        logging = UNSLOTH_ENABLE_LOGGING,
    )

    @_torch_compile(dynamic = None, fullgraph = True, options = fused_torch_compile_options)
    def post_forward(
        self,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
        logsumexp: torch.Tensor,
        input_shape,
    ):
        hidden_states = post_attention_decoding(self.self_attn, attn_output, logsumexp, input_shape)
        hidden_states += residual

        # Fully Connected
        residual = hidden_states.clone()
        hidden_states = rms_layernorm_forward(self.post_attention_layernorm, hidden_states)
        return hidden_states, residual
    pass

    def inference_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        **kwargs,
    ):
        residual = hidden_states.clone()
        hidden_states = rms_layernorm_forward(self.input_layernorm, hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states += residual.to(hidden_states.device)

        # Fully Connected
        residual = hidden_states.clone()
        hidden_states = rms_layernorm_forward(self.post_attention_layernorm, hidden_states)
        return hidden_states, residual
    pass
    # if has_static_cache and Version(torch.__version__) >= Version("2.10.0"):
    #     # torch 2.9.0 has excessive compilations
    #     inference_forward = _torch_compile(inference_forward, dynamic = None, fullgraph = True, options = fused_torch_compile_options)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: KWARGS_TYPE,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            # Account for CPU offloaded embed_tokens
            embed_device = self.embed_tokens.weight.device
            # Never non_blocking into the CPU: the lookup can read the ids before the copy lands
            # (generate does not sync between steps, so it embedded the previous token).
            if (
                torch.compiler.is_compiling()
                and _offloaded_embedding is not None
                and embed_device != input_ids.device
                and type(self.embed_tokens) is nn.Embedding
                # Inference only: the op has no backward, and it would skip embed_tokens' forward
                # hooks (enable_input_require_grads) that frozen-embedding training relies on.
                and not torch.is_grad_enabled()
            ):
                inputs_embeds = _offloaded_embedding(input_ids, self.embed_tokens.weight)
            else:
                inputs_embeds = self.embed_tokens(
                    input_ids.to(embed_device, non_blocking = embed_device.type != "cpu")
                ).to(input_ids.device)
        if not self.training and inputs_embeds.requires_grad:
            # detach, not requires_grad_(False): the embeddings are a non-leaf when an input
            # requires-grad hook or an offloaded copy produced them, and that raises.
            inputs_embeds = inputs_embeds.detach()

        cache_position = kwargs.pop("cache_position", None)
        if cache_position is None:
            past_seen_tokens = (past_key_values.get_seq_length() if past_key_values is not None else 0)
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Shape hints for the compiled callees; they cannot be traced, so skip them when this
        # whole forward is being compiled.
        if not torch.compiler.is_compiling():
            try:
                torch._dynamo.mark_static (hidden_states, 0)
                torch._dynamo.mark_dynamic(hidden_states, 1)
                torch._dynamo.mark_static (hidden_states, 2)
            except:
                pass

        # flex_attention_with_sink training windows its own BlockMask; all else needs the per-type mapping.
        _flex_sink_training = self.training and _GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED
        if not _flex_sink_training and not isinstance(attention_mask, dict):
            # Inference uses eager attention. If the config still has
            # _attn_implementation="flex_attention" (set for training), the
            # mask factory returns a BlockMask which eager cannot consume.
            # Temporarily swap to "eager" so a dense 4D float mask is built.
            _orig_attn_impl = getattr(self.config, "_attn_implementation", None)
            _swap_attn_impl = (not self.training) and _orig_attn_impl == "flex_attention"
            if _swap_attn_impl:
                self.config._attn_implementation = "eager"
            try:
                mask_kwargs = _build_mask_kwargs(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                )
                attention_mask = {
                    "full_attention": create_causal_mask(**{
                        k: v for k, v in mask_kwargs.items() if k in _ccm_params
                    }),
                    "sliding_attention": create_sliding_window_causal_mask(**{
                        k: v for k, v in mask_kwargs.items() if k in _cswc_params
                    }),
                }
            finally:
                if _swap_attn_impl:
                    self.config._attn_implementation = _orig_attn_impl

        # is_decoding = is_flex_attention_decoding(self.layers[0].self_attn, hidden_states)
        bsz, qlen, hd = hidden_states.shape
        block_swap = getattr(self.layers, "_unsloth_block_swap", None)
        # Across cards (embedding included) only the swapper's hooks move inputs: take the hooked path.
        _swap_device = getattr(block_swap, "layer_device", None)
        _cross_card = getattr(block_swap, "spans_devices", False) or (
            _swap_device is not None and hidden_states.device != _swap_device
        )
        if not self.training and qlen == 1 and isinstance(attention_mask, dict) and not _cross_card:
            # Add hack since residuals need to clone outside of the torch.compile region??
            # This forces it to free past residuals
            if not torch.compiler.is_compiling():
                torch.compiler.cudagraph_mark_step_begin()
            # Initialize for common return path
            all_hidden_states = None
            all_router_logits = None
            # This loop calls the layer's parts directly, so the block swapper's forward hooks never fire.
            for layer_idx, decoder_layer in enumerate(self.layers):
                if block_swap is not None:
                    block_swap.enter(layer_idx)
                mask = _gpt_oss_select_mask(
                    attention_mask,
                    _gpt_oss_layer_attention_type(decoder_layer, self.config, layer_idx),
                )
                hidden_states, residual = inference_forward(
                    decoder_layer,
                    hidden_states,
                    mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **kwargs,
                )
                _actual_experts = _unwrap_peft_experts(decoder_layer.mlp.experts)
                routed = routed_mlp_forward(decoder_layer.mlp, hidden_states)
                if routed is not None:
                    hidden_states = routed
                elif hasattr(_actual_experts, "gate_up_projs"):
                    hidden_states = moe_forward_inference(
                        decoder_layer.mlp, hidden_states
                    )
                elif (
                    _actual_experts.__class__.__name__ == "Mxfp4GptOssExperts"
                ):
                    if mlp_forward is None:
                        raise RuntimeError("Unsloth: MXFP4 forward is not found")
                    hidden_states, _ = mlp_forward(decoder_layer.mlp, hidden_states)
                else:
                    hidden_states = moe_forward_inference_bf16(decoder_layer.mlp, hidden_states)
                hidden_states += residual
                if block_swap is not None:
                    block_swap.leave(layer_idx)
            pass
            hidden_states = rms_layernorm_forward(self.norm, hidden_states)
        else:
            # flex_attention_with_sink builds its own BlockMask, so dropping the dense
            # one avoids an O(seq_len^2) allocation that OOMs at long context. Only that
            # forward ignores it: without it, stock attention loses causality entirely.
            if self.training and _GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED:
                attention_mask = None

            # Accumulate hidden states if requested
            output_hidden_states = kwargs.get(
                "output_hidden_states", self.config.output_hidden_states
            )
            all_hidden_states = () if output_hidden_states else None

            # Replaces stock @capture_outputs: without router_logits, aux_loss.to() fails (TRL >= 1.7 MoE).
            all_router_logits = None
            router_hooks = []
            if kwargs.get("output_router_logits", getattr(self.config, "output_router_logits", False)):
                all_router_logits = []
                def _record_router_logits(module, args, output):
                    all_router_logits.append(output[0] if isinstance(output, tuple) else output)
                for decoder_layer in self.layers:
                    router = getattr(getattr(decoder_layer, "mlp", None), "router", None)
                    if router is not None:
                        router_hooks.append(router.register_forward_hook(_record_router_logits))

            try:
                for layer_idx, decoder_layer in enumerate(self.layers):
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    mask = _gpt_oss_select_mask(
                        attention_mask,
                        _gpt_oss_layer_attention_type(decoder_layer, self.config, layer_idx),
                    )
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                pass
            finally:
                for hook in router_hooks:
                    hook.remove()
            if all_router_logits is not None:
                all_router_logits = tuple(all_router_logits)
            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        # Fix float16 / float32 mismatching
        hidden_states = hidden_states.to(inputs_embeds.dtype)
        return process_return(MoeModelOutputWithPast, {
                "last_hidden_state": hidden_states,
                "past_key_values": past_key_values,
                "hidden_states": all_hidden_states,
                "router_logits": all_router_logits,
            })

    patch_function(transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel, "forward", forward, match_level = "relaxed")
    # After the replacement, so the recorder wraps whichever forward ended up live:
    # zoo's own, or stock when the signature did not match.
    _record_training_state(transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel)
pass
TEMPORARY_PATCHES.append(patch_GptOssModel)

encoding = None

_HARMONY_SYMBOLS = (
    "Author",
    "Conversation",
    "DeveloperContent",
    "HarmonyEncodingName",
    "Message",
    "Role",
    "SystemContent",
    "ToolDescription",
    "load_harmony_encoding",
    "ReasoningEffort",
)

# Best-effort eager import; when openai_harmony is installed after this module was
# imported, _ensure_harmony rebinds the symbols lazily at call time.
# See https://github.com/unslothai/unsloth/issues/3361
try:
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        ToolDescription,
        load_harmony_encoding,
        ReasoningEffort
    )
except Exception:
    pass


def _ensure_harmony():
    g = globals()
    if all(g.get(name) is not None for name in _HARMONY_SYMBOLS):
        return
    try:
        import openai_harmony
    except ModuleNotFoundError as e:
        if not e.name or e.name == "openai_harmony":
            raise ImportError("Please install openai_harmony via `pip install openai_harmony`") from e
        if e.name.startswith("openai_harmony."):
            raise ImportError(f"Unsloth: failed to import openai_harmony: {e}") from e
        raise ImportError(f"Unsloth: failed to import openai_harmony; its dependency `{e.name}` is missing: {e}") from e
    except Exception as e:
        raise ImportError(f"Unsloth: failed to import openai_harmony: {e}") from e

    missing = [name for name in _HARMONY_SYMBOLS if getattr(openai_harmony, name, None) is None]
    if missing:
        raise ImportError(
            f"Unsloth: openai_harmony is installed but is missing required symbols ({', '.join(missing)}); "
            f"please upgrade via `pip install --upgrade openai_harmony`"
        )
    for name in _HARMONY_SYMBOLS:
        g[name] = getattr(openai_harmony, name)
pass


def _get_gpt_oss_harmony_encoding():
    global encoding
    _ensure_harmony()

    if encoding is None:
        try:
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        except Exception as e:
            raise RuntimeError(f"Unsloth: failed to load the gpt-oss harmony encoding: {e}") from e
    return encoding
pass


def encode_conversations_with_harmony(
    messages,
    reasoning_effort = "medium",
    add_generation_prompt = True,
    tool_calls = None,
    developer_instructions = None,
    model_identity = "You are ChatGPT, a large language model trained by OpenAI.",
):
    harmony_encoding = _get_gpt_oss_harmony_encoding()

    assert reasoning_effort in ("low", "medium", "high")

    # No else: an unmatched value must leave harmony_reasoning unbound exactly as
    # `match` did, which is what happens under -O when the assert above is stripped.
    if   reasoning_effort == "low":    harmony_reasoning = ReasoningEffort.LOW
    elif reasoning_effort == "medium": harmony_reasoning = ReasoningEffort.MEDIUM
    elif reasoning_effort == "high":   harmony_reasoning = ReasoningEffort.HIGH

    convos = []

    # Create system message
    import datetime

    today = datetime.datetime.today().strftime("%Y-%m-%d")
    system = Message.from_role_and_content(Role.SYSTEM,
        SystemContent.new()
            .with_model_identity(model_identity)
            .with_reasoning_effort(harmony_reasoning)
            .with_conversation_start_date(today)
            .with_knowledge_cutoff("2024-06")
            .with_required_channels(["analysis", "commentary", "final"]),
    )
    convos.append(system)

    # Developer message and tool calling
    dev = DeveloperContent.new()
    if developer_instructions is not None: dev = dev.with_instructions(developer_instructions)
    if tool_calls is not None:
        new_tools = []
        for function in tool_calls:
            function = function["function"]
            name = function["name"]
            description = function["description"]
            parameters = function["parameters"]
            tool = ToolDescription.new(name, description, parameters)
            new_tools.append(tool)
        dev = dev.with_function_tools(new_tools)
    pass
    if developer_instructions is not None or tool_calls is not None:
        dev = Message.from_role_and_content(Role.DEVELOPER, dev)
        convos.append(dev)

    for message in messages:
        if message["role"] == "user":
            convos.append(Message.from_role_and_content(Role.USER, message["content"]))
        elif message["role"] == "assistant":
            # An assistant turn can independently carry reasoning, a tool call, and/or a
            # final answer. Treat the parts separately so none silently suppresses another.
            # Guard on truthiness (not key membership) so a nullable "thinking" column
            # materialized as None or "" does not get passed into Harmony, which rejects it.
            if message.get("thinking"):
                x = Message.from_role_and_content(Role.ASSISTANT, message["thinking"])
                x = x.with_channel("analysis")
                convos.append(x)
            if message.get("tool_calls"):
                x = Message.from_role_and_content(Role.ASSISTANT, message["tool_calls"][0]["arguments"])
                x = x.with_channel("commentary").with_recipient(f"functions.{message['tool_calls'][0]['name']}").with_content_type("json")
                convos.append(x)
            elif message.get("content"):
                # A tool-call turn does not also emit a final answer (the answer comes
                # after the tool result), so tool_calls and content stay mutually exclusive.
                x = Message.from_role_and_content(Role.ASSISTANT, message["content"])
                x = x.with_channel("final")
                convos.append(x)
            continue
        elif message["role"] == "tool":
            x = Message.from_author_and_content(Author.new(Role.TOOL, f"functions.{message['name']}"), message["content"])
            x = x.with_recipient("assistant").with_channel("commentary")
            convos.append(x)
    pass

    # Create Harmony conversations
    convos = Conversation.from_messages(convos)
    if add_generation_prompt:
        harmony_input_ids = harmony_encoding.render_conversation_for_completion(convos, Role.ASSISTANT)
    else:
        harmony_input_ids = harmony_encoding.render_conversation(convos)
    harmony_decoded_text = harmony_encoding.decode(harmony_input_ids)
    return harmony_decoded_text, harmony_input_ids
pass


# Fix https://github.com/huggingface/transformers/pull/40474
# RuntimeError: Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.
# AutoConfig error: 'GptOssConfig' object has no attribute 'max_position_embeddings'
try:
    from transformers.configuration_utils import layer_type_validation

    try:
        from transformers.configuration_utils import PreTrainedConfig

        PretrainedConfig = PreTrainedConfig
    except:
        from transformers.configuration_utils import PretrainedConfig

    from transformers.modeling_rope_utils import rope_config_validation

    class Old_GptOssConfig(PretrainedConfig):
        r"""
        This will yield a configuration to that of the BERT
        [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

        """

        model_type = "gpt_oss"
        base_model_pp_plan = {
            "embed_tokens": (["input_ids"], ["inputs_embeds"]),
            "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
            "norm": (["hidden_states"], ["hidden_states"]),
        }
        base_model_tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.sinks": "local_rowwise",
            "layers.*.mlp.experts": "gather",
            "layers.*.mlp.router": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
        }

        def __init__(
            self,
            num_hidden_layers: int = 36,
            num_local_experts: int = 128,
            vocab_size: int = 201088,
            hidden_size: int = 2880,
            intermediate_size: int = 2880,
            head_dim: int = 64,
            num_attention_heads: int = 64,
            num_key_value_heads: int = 8,
            sliding_window: int = 128,
            rope_theta: float = 150000.0,
            tie_word_embeddings=False,
            hidden_act: str = "silu",
            initializer_range: float = 0.02,
            max_position_embeddings=131072,
            rms_norm_eps: float = 1e-5,
            rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
            attention_dropout: float = 0.0,
            num_experts_per_tok=4,
            router_aux_loss_coef: float = 0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_local_experts = num_local_experts
            self.sliding_window = sliding_window
            self.num_experts_per_tok = num_experts_per_tok
            # for backward compatibility
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.attention_dropout = attention_dropout
            self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
            self.layer_types = layer_types
            if self.layer_types is None:
                self.layer_types = ["sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)]
            layer_type_validation(self.layer_types)

            # Validate the correctness of rotary position embeddings parameters
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            if self.rope_scaling is not None and "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
            rope_config_validation(self)

            self.attention_bias = True
            self.max_position_embeddings = max_position_embeddings
            self.router_aux_loss_coef = router_aux_loss_coef
            self.output_router_logits = output_router_logits
            self.use_cache = use_cache
            super().__init__(
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )

    class GptOssConfig(PretrainedConfig):
        r"""
        This will yield a configuration to that of the BERT
        [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

        """

        model_type = "gpt_oss"
        base_model_pp_plan = {
            "embed_tokens": (["input_ids"], ["inputs_embeds"]),
            "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
            "norm": (["hidden_states"], ["hidden_states"]),
        }
        base_model_tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.sinks": "local_rowwise",
            "layers.*.mlp.experts": "gather",
            "layers.*.mlp.router": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
        }

        def __init__(
            self,
            num_hidden_layers: int = 36,
            num_local_experts: int = 128,
            vocab_size: int = 201088,
            hidden_size: int = 2880,
            intermediate_size: int = 2880,
            head_dim: int = 64,
            num_attention_heads: int = 64,
            num_key_value_heads: int = 8,
            sliding_window: int = 128,
            rope_theta: float = 150000.0,
            tie_word_embeddings=False,
            hidden_act: str = "silu",
            initializer_range: float = 0.02,
            max_position_embeddings=131072,
            rms_norm_eps: float = 1e-5,
            rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
            attention_dropout: float = 0.0,
            num_experts_per_tok=4,
            router_aux_loss_coef: float = 0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_local_experts = num_local_experts
            self.sliding_window = sliding_window
            self.num_experts_per_tok = num_experts_per_tok
            # for backward compatibility
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.attention_dropout = attention_dropout
            self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
            self.layer_types = layer_types
            if self.layer_types is None:
                self.layer_types = [
                    "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
                ]
            layer_type_validation(self.layer_types)
            self.attention_bias = True
            self.max_position_embeddings = max_position_embeddings
            self.router_aux_loss_coef = router_aux_loss_coef
            self.output_router_logits = output_router_logits
            self.use_cache = use_cache

            # Validate the correctness of rotary position embeddings parameters
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            if self.rope_scaling is not None and "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
            rope_config_validation(self)

            self.attention_bias = True
            self.max_position_embeddings = max_position_embeddings
            self.router_aux_loss_coef = router_aux_loss_coef
            self.output_router_logits = output_router_logits
            self.use_cache = use_cache
            super().__init__(
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )

    def patch_gpt_oss_config():
        try:
            import transformers.models.gpt_oss.configuration_gpt_oss

            transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig
        except Exception as e:
            return raise_error("transformers.models.gpt_oss.configuration_gpt_oss", e)

        try:
            current_class = dedent(inspect.getsource(transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig))
            new_class = dedent(inspect.getsource(Old_GptOssConfig))
            new_class = new_class.replace("Old_GptOssConfig", "GptOssConfig")
            if new_class == current_class:
                logger.info("Unsloth: Updating GPT OSS Config to fix missing `max_position_embeddings`")
                patch_function(transformers.models.gpt_oss.configuration_gpt_oss, "GptOssConfig", GptOssConfig)
        except Exception as e:
            return raise_error("transformers.models.gpt_oss.configuration_gpt_oss", e)

    pass
    TEMPORARY_PATCHES.append(patch_gpt_oss_config)
except Exception as e:
    raise_error("transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig", e)


def patch_gpt_oss_init_weights_modulelist_fix():
    if "gpt_oss" not in _normalized_unsloth_model_name():
        return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss", e)

    GptOssPreTrainedModel = (
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssPreTrainedModel
    )
    GptOssExperts = transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts
    GptOssTopKRouter = transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter
    if getattr(GptOssPreTrainedModel, "_unsloth_init_weights_fixed", False):
        return
    _original_init_weights = GptOssPreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        if isinstance(module, GptOssExperts) and not hasattr(module, "gate_up_proj"):
            std = self.config.initializer_range
            for up in getattr(module, "gate_up_projs", []):
                init.normal_(up.weight, mean=0.0, std=std)
                if up.bias is not None:
                    init.zeros_(up.bias)
            for down in getattr(module, "down_projs", []):
                init.normal_(down.weight, mean=0.0, std=std)
                if down.bias is not None:
                    init.zeros_(down.bias)
            return
        if isinstance(module, GptOssTopKRouter):
            # Router weight/bias live under .weight (stock) or .linear (Unsloth BnB-4bit).
            # Resolve whichever exists so stock _init_weights' module.weight access can't
            # raise "GptOssTopKRouter object has no attribute 'weight'" (#3119).
            std = self.config.initializer_range
            weight = getattr(module, "weight", None)
            if weight is None:
                weight = getattr(getattr(module, "linear", None), "weight", None)
            bias = getattr(module, "bias", None)
            if bias is None:
                bias = getattr(getattr(module, "linear", None), "bias", None)
            if weight is not None:
                init.normal_(weight, mean=0.0, std=std)
            if bias is not None:
                init.normal_(bias, mean=0.0, std=std)
            return
        _original_init_weights(self, module)

    patch_function(GptOssPreTrainedModel, "_init_weights", _patched_init_weights)
    GptOssPreTrainedModel._unsloth_init_weights_fixed = True
pass
TEMPORARY_PATCHES.append(patch_gpt_oss_init_weights_modulelist_fix)


# ============================================================================
# Patch GptOssForCausalLM.forward for GRPO training
# When UNSLOTH_RETURN_HIDDEN_STATES=1, return hidden_states instead of logits
# ============================================================================
def patch_gpt_oss_for_grpo(phase="post_compile"):
    """
    Patch GptOssForCausalLM.forward for GRPO: when UNSLOTH_RETURN_HIDDEN_STATES=1, return
    hidden_states instead of logits (fixes the GRPO matmul dimension mismatch).
    Runs post-compile so the compiler can pattern-match cross-entropy and fuse loss (avoids OOM).
    """
    if phase != "post_compile":
        return

    if "gpt_oss" not in _normalized_unsloth_model_name():
        return

    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssForCausalLM,
            MoeCausalLMOutputWithPast,
        )

        if hasattr(GptOssForCausalLM, '_unsloth_grpo_patched'):
            return

        _original_causal_lm_forward = GptOssForCausalLM.forward

        def _patched_causal_lm_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            cache_position=None,
            logits_to_keep=0,
            **kwargs,
        ):
            # This Unsloth Zoo code section is licensed under AGPL3

            # Generation passes a per-type mask mapping load_balancing_loss_func cannot read, and no labels.
            if isinstance(attention_mask, dict) and labels is None:
                kwargs["output_router_logits"] = False

            RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"

            if not RETURN_HIDDEN_STATES:
                # Normal forward pass
                return _original_causal_lm_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            # RETURN_HIDDEN_STATES mode - return hidden_states instead of logits
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state

            # Apply slice_indices to hidden_states (same indexing as for logits)
            if logits_to_keep != 0:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                hidden_states = hidden_states[:, slice_indices, :]

            # Return hidden_states as "logits" for GRPO to use
            return MoeCausalLMOutputWithPast(
                loss=None,
                aux_loss=getattr(outputs, 'aux_loss', None),
                logits=hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=getattr(outputs, 'router_logits', None),
            )

        # Preserve __qualname__ so _unsloth_get_batch_samples can detect
        # this is a CausalLM forward and compute num_items_in_batch properly.
        _patched_causal_lm_forward.__qualname__ = _original_causal_lm_forward.__qualname__
        GptOssForCausalLM.forward = _patched_causal_lm_forward
        GptOssForCausalLM._unsloth_grpo_patched = True
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Patched GptOssForCausalLM.forward for GRPO hidden states.")

    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch GptOssForCausalLM.forward: {e}")
pass
TEMPORARY_PATCHES.append(patch_gpt_oss_for_grpo)
