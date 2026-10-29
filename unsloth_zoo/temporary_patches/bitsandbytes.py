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

import os
import torch
import torch.nn as nn
import inspect
import importlib
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from .common import TEMPORARY_PATCHES, torch_compile, RESCOPE_PATCH_FLAG, WRAPPER_INNER_ATTR
from .utils import (
    patch_function,
    process_output_options,
    process_return,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    Unpack,
    _get_unique_storage_name,
)
from textwrap import dedent
import re


# transformers releases that drop the bnb quant_state sidecars of some composite checkpoints.
# `.dev0` bounds so a prerelease sorts with its own release line: without it
# `Version("5.6.0.dev0") < Version("5.6.0")` puts a 5.6.0 rc inside the window.
# Compared with `packaging.version.Version`, NOT `unsloth_zoo.utils.Version`: that one answers
# `Version('5.6.0.1')` for "5.6.0.dev0", which sorts after the release it must sort before.
_QUANT_STATE_BROKEN_TRANSFORMERS = ("5.4.0.dev0", "5.6.0.dev0")


def _transformers_drops_prequantized_quant_state():
    """True when the installed transformers is inside the quant_state defect window.

    transformers PR #44300 (5.4.0) recursed the conversion mapping into `PreTrainedModel`
    submodules, pulling the text model's `^model.language_model` -> `model` renaming into the
    composite mapping, where it rewrites the packed weight and its five sidecars off the map.
    The weight survives because the loader retries the ORIGINAL key when that key is a model
    parameter; a sidecar's never is. PR #45567 scoped it in 5.6.0.

    Version only, so necessary and not sufficient: it also needs a text submodel mapping that
    STRIPS the composite prefix (`qwen3_5_text`, `gemma3n_text`), not one that adds one.
    """
    installed = _installed_transformers_version()
    if installed == "unknown": return False
    try:
        from packaging.version import Version as _Version
        parsed = _Version(installed)
        low, high = _QUANT_STATE_BROKEN_TRANSFORMERS
        return _Version(low) <= parsed < _Version(high)
    except Exception:
        return False


def _installed_transformers_version():
    """The installed transformers version, or "unknown".

    Falls back to an already-imported `transformers.__version__`: a source checkout or frozen
    bundle has no `.dist-info` to read. Reads `sys.modules` only, so it never triggers an import.
    """
    try:
        from importlib.metadata import version as _version
        return _version("transformers")
    except Exception:
        pass
    try:
        import sys
        return getattr(sys.modules.get("transformers"), "__version__", "unknown")
    except Exception:
        return "unknown"


def _composite_renaming_repair_installed():
    """Is the runtime repair for that defect live in THIS process?

    Either package's mark counts (zoo's `conversion_mapping_rescope.py`, unsloth's
    `fix_transformers_composite_prefix_renaming`), anywhere down the chain, since they compose
    in either order with `moe_utils_bnb4bit.py`'s wrapper. Follow `WRAPPER_INNER_ATTR` as well
    as `__wrapped__`: a wrapper that must survive a rescope install cannot publish
    `__wrapped__`, which the rescope unwraps to choose what to wrap.

    Asked because with the repair live, a module reaching this guard has a checkpoint whose
    sidecars really are absent, the one case where re-quantizing IS the answer and the version
    branch says the opposite.
    """
    try:
        from transformers import conversion_mapping
    except Exception:
        return False
    function = getattr(conversion_mapping, "get_model_conversion_mapping", None)
    seen = 0
    while function is not None and seen < 8:
        if getattr(function, RESCOPE_PATCH_FLAG, False):
            return True
        if getattr(function, "_unsloth_patched_composite_prefix_renaming", False):
            return True
        function = getattr(function, "__wrapped__", None) or getattr(function, WRAPPER_INNER_ATTR, None)
        seen += 1
    return False


def _packed_weight_without_quant_state_error(module):
    """Message for a Linear4bit whose packed weight arrived with no quant_state."""
    transformers_version = _installed_transformers_version()
    shape = tuple(module.weight.shape)
    # No cause here: this inspects the module, never the checkpoint, and a genuinely
    # stateless checkpoint reaches this same line. The branches below qualify it.
    head = (
        f"Unsloth: a bitsandbytes Linear4bit holds what looks like its PACKED 4-bit "
        f"weight (shape {shape}, dtype {module.weight.dtype}, out_features "
        f"{getattr(module, 'out_features', 'unknown')}) but has no quant_state, so it "
        f"cannot be dequantized. Its quantization metadata is missing."
    )
    if _transformers_drops_prequantized_quant_state() and _composite_renaming_repair_installed():
        # Inside the window, but the repair is live, so the renaming never fired and the
        # version is not what went wrong here.
        return (
            f"{head}\nInstalled transformers=={transformers_version}, which is inside the "
            f"window that discards quant_state sidecars -- but Unsloth's runtime repair for "
            f"that is installed in this process, so it is not the explanation here. Check "
            f"that the checkpoint's `weight.absmax`, `weight.quant_map` and "
            f"`weight.quant_state.bitsandbytes__nf4` tensors are present and were not "
            f"reported as unexpected keys during loading."
        )
    if _transformers_drops_prequantized_quant_state():
        return (
            f"{head} The most likely reason is that it was lost while LOADING, not "
            f"while saving.\n"
            f"This is transformers=={transformers_version}: releases 5.4.0 and "
            f"5.5.0 to 5.5.4 discard the quant_state sidecar tensors of pre-quantized "
            f"composite (multimodal) checkpoints whose text submodel strips the "
            f"composite prefix, which covers Qwen3.5 and Gemma 3n. Introduced by "
            f"transformers PR #44300, fixed by PR #45567 in 5.6.0. Install "
            f"transformers>=5.6.0, or fall back to 5.3.0, and try "
            f"again before regenerating anything: if that is what happened here the "
            f"checkpoint is intact and re-quantizing it will not help. If a supported "
            f"transformers still reports this, the sidecar tensors really are missing "
            f"from the files and the checkpoint does need rebuilding."
        )
    return (
        f"{head}\nInstalled transformers=={transformers_version}. Check that the "
        f"checkpoint's `weight.absmax`, `weight.quant_map` and "
        f"`weight.quant_state.bitsandbytes__nf4` tensors are present and were not "
        f"reported as unexpected keys during loading."
    )


# Inference 4-bit matmuls straight into bitsandbytes' fused kernel. bitsandbytes >= 0.50 reaches
# lib.cgemm_4bit_* through matmul_4bit -> torch.ops.bitsandbytes.gemm_4bit -> _gemm_4bit_kernel_impl, about
# 50 us of Python per call on a Colab host (a decode step makes one per projection); the kernel and its
# arguments are the same here, so the output is too. The first call per dtype and layout is also run
# through bitsandbytes and must match bit for bit. UNSLOTH_BNB_4BIT_DIRECT=0 turns it off.
BNB_4BIT_DIRECT_CALLS = {"direct": 0}
_BNB_DIRECT = {"on": os.environ.get("UNSLOTH_BNB_4BIT_DIRECT", "1") != "0" and torch.version.hip is None,
               "ops": None, "checked": {}, "fn": {}}
_BNB_DIRECT_SIGNATURE = ("A", "B", "shapeB", "absmax", "blocksize", "quant_type", "bias", "absmax_8bit",
                         "absmax_code", "absmax_offset")
_BNB_DIRECT_FN = {torch.bfloat16: "cgemm_4bit_bf16", torch.float16: "cgemm_4bit_fp16",
                  torch.float32: "cgemm_4bit_fp32"}
_cuda_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_cuda_get_device = getattr(torch._C, "_cuda_getDevice", None)


def _bnb_direct_ops():
    ops = _BNB_DIRECT["ops"]
    if ops is None:
        ops = False
        try:
            from bitsandbytes.backends.cuda import ops as bops
            impl = bops._gemm_4bit_kernel_impl
            if (
                tuple(inspect.signature(impl).parameters) == _BNB_DIRECT_SIGNATURE
                and getattr(bops.lib, "compiled_with_cuda", False)
                and callable(bops._gemm_4bit_use_custom_fn)
                and isinstance(bops._gemm_4bit_custom_max_m, int)
                and _cuda_raw_stream is not None and _cuda_get_device is not None
            ):
                ops = bops
        except Exception:
            pass
        _BNB_DIRECT["ops"] = ops
    return ops


def _on_current_cuda(x):
    # bitsandbytes switches to the tensor's device first (multi-GPU device_map): left to it.
    return x.is_cuda and x.get_device() == _cuda_get_device()


def _bnb_direct_tensors(qs):
    if qs.nested:
        return [qs.state2.absmax, qs.absmax, qs.state2.code, qs.offset]
    return [qs.absmax]


_PTRS_AT = getattr(torch._C, "_tensors_data_ptrs_at_indices_equal", None)


_PTRS_IDX = {1: [0], 4: [0, 1, 2, 3]}


def _ptrs_equal(tensors, ptrs):
    try:
        if _PTRS_AT is not None:
            return _PTRS_AT(tensors, ptrs, _PTRS_IDX[len(tensors)])
        return all(t.data_ptr() == p for t, p in zip(tensors, ptrs))
    except Exception:
        return False  # a quant tensor set to None: rebuilt, and declined


def _bnb_direct_plan(weight, quant_state):
    """The weight-side arguments of _gemm_4bit_kernel_impl for this layer, or None."""
    qs = quant_state
    if len(qs.shape) != 2 or weight.dtype is not torch.uint8:
        return None
    if qs.nested and (qs.state2 is None or qs.state2.blocksize != 256 or qs.offset is None):
        return None
    tensors = _bnb_direct_tensors(qs)
    if tensors[0].dtype is not torch.float32:
        return None
    device = weight.get_device()
    if any(t.get_device() != device for t in tensors):
        return None  # the kernel reads every pointer on the activations' device
    if qs.nested:
        # matmul_4bit passes state2.absmax as absmax and the quantized absmax as absmax_8bit.
        offset = qs.offset.to(dtype = torch.float32)
        ptrs = (tensors[0].data_ptr(), tensors[1].data_ptr(), tensors[2].data_ptr(), offset.data_ptr())
    else:
        offset = None
        ptrs = (tensors[0].data_ptr(), None, None, None)
    return (qs, weight.data_ptr(), tensors, [t.data_ptr() for t in tensors], ptrs, offset,
            qs.shape[0], qs.shape[1], qs.blocksize, 1 if qs.quant_type == "fp4" else 2, {}, device)


def _bnb_4bit_direct(self, x, quant_state):
    """bitsandbytes.matmul_4bit(x, weight, bias, quant_state) for inference, or None (the caller runs it)."""
    if torch.compiler.is_compiling() or not _BNB_DIRECT["on"]:
        return None
    ops = _BNB_DIRECT["ops"]
    if ops is None:
        ops = _bnb_direct_ops()
    if ops is False:
        return None
    weight = self.weight
    bias = self.bias
    dtype = x.dtype
    if (
        dtype is not self.compute_dtype or not _on_current_cuda(x)
        or (torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad)))
    ):
        return None
    plan = self.__dict__.get("_unsloth_bnb_direct")
    # Rebuilt whenever the layer is requantized, moved, or a quant tensor is replaced or recast.
    if plan is None or plan[0] is not quant_state or plan[1] is not None and (
        plan[1] != weight.data_ptr() or not _ptrs_equal(_bnb_direct_tensors(quant_state), plan[3])
    ):
        try:
            plan = _bnb_direct_plan(weight, quant_state)
        except Exception:
            plan = None
        if plan is None:
            plan = (quant_state, None)
        self.__dict__["_unsloth_bnb_direct"] = plan
    if plan[1] is None:
        return None
    _, _, _, _, ptrs, _, N, K, blocksize, quant_type, picks, device = plan
    if x.shape[-1] != K or device != x.get_device() or (bias is not None and (
        bias.dim() != 1 or bias.get_device() != device
    )):
        return None
    M = x.numel() // K
    use = picks.get((M, dtype))
    if use is None:
        # torch.ops.bitsandbytes.gemm_4bit's own choice between its kernel and dequantize + F.linear.
        use = 0 < M <= ops._gemm_4bit_custom_max_m and K % blocksize == 0 and bool(
            ops._gemm_4bit_use_custom_fn(x.device.index, dtype, M, N, K))
        picks[(M, dtype)] = use
    if not use:
        return None
    fn = _BNB_DIRECT["fn"].get(dtype)
    if fn is None:
        fn = getattr(ops.lib, _BNB_DIRECT_FN.get(dtype, ""), None)
        # The 14 arguments passed below: a library built with another kernel ABI is left to bitsandbytes.
        if fn is None or len(getattr(fn, "argtypes", None) or ()) != 14:
            fn = False
        _BNB_DIRECT["fn"][dtype] = fn
    if fn is False:
        return None
    key = (dtype, ptrs[1] is not None, bias is not None)
    checked = _BNB_DIRECT["checked"].get(key)
    if checked is False or checked is None and torch.cuda.is_current_stream_capturing():
        return None  # the check reads back on the host: leave it to an eager call
    if bias is not None and bias.dtype is not dtype:
        bias = bias.to(dtype)  # as the forward below casts it
    A = x.contiguous()
    out = torch.empty((*A.shape[:-1], N), dtype = dtype, device = A.device)
    fn(A.data_ptr(), weight.data_ptr(), ptrs[0], ptrs[1], ptrs[2], ptrs[3], out.data_ptr(),
       bias.data_ptr() if bias is not None else None, M, N, K, blocksize, quant_type, _cuda_raw_stream(A.device.index))
    if checked is None:
        import bitsandbytes
        ref = bitsandbytes.matmul_4bit(x, weight.data.t(), bias = bias, quant_state = quant_state)
        ok = ref.shape == out.shape and ref.dtype == out.dtype and torch.equal(
            ref.view(torch.uint8), out.view(torch.uint8))
        _BNB_DIRECT["checked"][key] = ok
        if not ok:
            return ref
    BNB_4BIT_DIRECT_CALLS["direct"] += 1
    return out


def patch_bitsandbytes_linear4bit_forward():
    # Fixes torch.compile complaining about multiple things
    try:
        import bitsandbytes
        bitsandbytes.nn.modules.Linear4bit
        Params4bit = bitsandbytes.nn.modules.Params4bit
        fix_4bit_weight_quant_state_from_module = bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module
    except Exception as e:
        return raise_error("bitsandbytes.Linear4bit", e)

    # Fix Params4bit.__torch_function__ infinite recursion under torch.compile.
    # bnb >= 0.46's Params4bit.__torch_function__ delegates to super() (except
    # chunk/split), which re-dispatches back to it since Params4bit is still in
    # the types tuple. Eager mode blocks this via _disabled_torch_function_impl,
    # but torch.compile's AOT autograd runtime does not (seen on T4 with torch
    # 2.8.0 + bnb 0.49.2). Removing it falls back to Parameter/Tensor C-level
    # dispatch that cannot recurse; bnb only added chunk/split handling anyway.
    if hasattr(Params4bit, "__torch_function__") and \
       "__torch_function__" in Params4bit.__dict__:
        delattr(Params4bit, "__torch_function__")
    pass

    def forward(self, x: torch.Tensor):
        # In transformers 5.0+, weights may not be in packed format yet during init
        if self.weight.shape[-1] == 1:
            fix_4bit_weight_quant_state_from_module(self)

        # Some layers may not be quantized (no quant_state) - fall back to regular matmul
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is None:
            bias = None if self.bias is None else self.bias
            weight = self.weight
            # A packed 4-bit buffer is
            # (out_features * in_features // (2 * quant_storage.itemsize), 1) -- [N, 1] for
            # every quant_storage (bitsandbytes _ops.py quantize_4bit). F.linear on one only
            # ever produces "mat1 and mat2 shapes cannot be multiplied", which reads like a
            # corrupt checkpoint (unsloth #9867, #10010, #10017, #10276). The recovery above
            # cannot help: it copies module.quant_state, which is itself None here.
            #
            # The only legitimately unquantized [N, 1] weight is a one-input layer, where N is
            # out_features, so ask that rather than comparing row counts: a bare
            # shape[0] != out_features also excused packed blobs at
            # in_features == 2 * quant_storage.itemsize (2 uint8, 4 fp16/bf16, 8 fp32).
            # The float clause covers in_features == out_features == 1, which packs to (1, 1).
            # Dtype NARROWS the exemption and must not gate the raise: an uint8-gated raise
            # would disarm the guard for float-quant_storage checkpoints.
            in_features  = getattr(self, "in_features",  None)
            out_features = getattr(self, "out_features", None)
            if (
                weight.dim() == 2
                and weight.shape[-1] == 1
                and not (
                    in_features == 1
                    and weight.shape[0] == out_features
                    and weight.is_floating_point()
                )
            ):
                raise RuntimeError(_packed_weight_without_quant_state_error(self))
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            return torch.nn.functional.linear(x, weight, bias)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually

        # ** Errors out in torch.compile so remove it
        # if self.bias is not None and self.bias.dtype != x.dtype:
        #     self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        out = _bnb_4bit_direct(self, x, quant_state)
        if out is not None:
            return out

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        # ** Errors out in torch.compile
        # weight = self.weight.t() if self.weight.dim() == 2 else self.weight

        weight = self.weight.data.t()

        return bitsandbytes.matmul_4bit(x, weight, bias=bias, quant_state=quant_state).to(inp_dtype)

    patch_function(bitsandbytes.nn.modules.Linear4bit, "forward", forward)
    try:
        patch_function(bitsandbytes.nn.Linear4bit, "forward", forward)
    except:
        pass
pass
TEMPORARY_PATCHES.append(patch_bitsandbytes_linear4bit_forward)
