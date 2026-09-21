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
from .common import TEMPORARY_PATCHES, torch_compile
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


# transformers 5.4.0 and 5.5.x drop the bnb quant_state sidecars of composite checkpoints
_QUANT_STATE_BROKEN_TRANSFORMERS = ("5.4.0", "5.6.0")


def _transformers_drops_prequantized_quant_state():
    """True when the installed transformers is inside the quant_state defect window.

    transformers 5.4.0 (PR #44300) made the conversion mapping recurse into
    `PreTrainedModel` submodules, which pulled the text model's
    `^model.language_model.` -> `^model.` `WeightRenaming` into the composite
    model's mapping. Renamings run before the bitsandbytes converter, so
    `weight.absmax`, `weight.quant_map`, `weight.nested_absmax`,
    `weight.nested_quant_map` and `weight.quant_state.bitsandbytes__nf4` all
    renamed to keys the model does not have and were discarded as unexpected,
    while the packed `weight` was loaded raw. transformers PR #45567 fixed it in
    5.6.0, so the window is exactly 5.4.0 and 5.5.0 through 5.5.4.
    """
    try:
        from packaging.version import Version as _Version
        from importlib.metadata import version as _version
        parsed = _Version(_version("transformers"))
    except Exception:
        return False
    low, high = _QUANT_STATE_BROKEN_TRANSFORMERS
    try:
        return _Version(low) <= parsed < _Version(high)
    except Exception:
        return False


def _packed_weight_without_quant_state_error(module):
    """Message for a Linear4bit whose packed weight arrived with no quant_state."""
    try:
        from importlib.metadata import version as _version
        transformers_version = _version("transformers")
    except Exception:
        transformers_version = "unknown"
    shape = tuple(module.weight.shape)
    head = (
        f"Unsloth: a bitsandbytes Linear4bit still holds its PACKED 4-bit weight "
        f"(shape {shape}, dtype {module.weight.dtype}) but has no quant_state, so it "
        f"cannot be dequantized. The quantization metadata was lost while loading, not "
        f"while saving."
    )
    if _transformers_drops_prequantized_quant_state():
        return (
            f"{head}\nThis is transformers=={transformers_version}: releases 5.4.0 and "
            f"5.5.0 to 5.5.4 discard the quant_state sidecar tensors of pre-quantized "
            f"composite (multimodal) checkpoints. Introduced by transformers PR #44300, "
            f"fixed by PR #45567 in 5.6.0. Install transformers>=5.6.0, or fall back to "
            f"5.3.0 or 4.57.6, and try again before regenerating anything: if that is "
            f"what happened here the checkpoint is intact and re-quantizing it will not "
            f"help. If a supported transformers still reports this, the sidecar tensors "
            f"really are missing from the files and the checkpoint does need rebuilding."
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
            # A layer that is genuinely unquantized holds an ordinary [out, in] weight,
            # and the fallback below is correct for it. A PACKED 4-bit buffer is [N, 1]
            # uint8, and handing that to F.linear only ever produces
            #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x5120 and 1x15728640)
            # which reads like a corrupt checkpoint and sent the reporters of unsloth
            # #9867, #10010, #10017 and #10276 off regenerating good ones. It is not the
            # checkpoint: transformers 5.4.0 and 5.5.x discard the quant_state sidecars of
            # pre-quantized composite (multimodal) checkpoints while loading them. Name
            # that instead of letting the shape error stand. Note the recovery attempt
            # above cannot help here: fix_4bit_weight_quant_state_from_module only copies
            # module.quant_state onto the weight, and module.quant_state is itself None
            # because Params4bit.from_prequantized never ran.
            # A packed blob is (out_features * in_features / 2, 1). A legitimate
            # unquantized Linear4bit with in_features == 1 is (out_features, 1) and
            # matches the shape test alone, so compare against out_features to tell
            # them apart. in_features == 2 makes the two shapes equal; that collapses
            # to not raising, which is the old behaviour, never a false accusation.
            if (
                weight.dim() == 2
                and weight.shape[-1] == 1
                and weight.shape[0] != getattr(self, "out_features", -1)
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
