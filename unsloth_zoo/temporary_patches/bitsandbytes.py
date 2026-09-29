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
            f"transformers>=5.6.0, or fall back to 5.3.0 (on Apple Silicon Unsloth "
            f"caps transformers at 5.5.0, so 5.3.0 is the option there), and try "
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
