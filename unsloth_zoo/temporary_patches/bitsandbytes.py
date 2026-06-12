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
        # In transformers 5.0+, weights may not be in packed format yet during init.
        # Recover the missing `quant_state` for both packed layouts that appear
        # in transformers 4.x and 5.x with bitsandbytes >= 0.43:
        #   * `[N, 1]` -- legacy column-packed (handled by bnb upstream's
        #     `fix_4bit_weight_quant_state_from_module`, which asserts
        #     `shape[1] == 1`).
        #   * `[1, N]` -- flat-row-packed used by some Gemma3n /
        #     conditional-generation layers (e.g. `per_layer_model_projection`).
        #     bnb's recovery function asserts column-packed, so briefly reshape
        #     the weight metadata to `[N, 1]` to satisfy the assertion, call
        #     the recovery, and restore the original shape. Reshape is
        #     metadata-only on the contiguous nibble buffer; storage bytes are
        #     untouched and `quant_state.shape` carries the original
        #     `[out_features, in_features]` for the matmul.
        w = self.weight
        if (
            getattr(w, "quant_state", None) is None
            and w.dim() == 2
            and (w.shape[-1] == 1 or w.shape[0] == 1)
        ):
            original_shape = tuple(w.shape)
            try:
                if w.shape[-1] != 1:
                    # `[1, N]` -> transient `[N, 1]` so bnb's assert passes.
                    w.data = w.data.reshape(-1, 1)
                fix_4bit_weight_quant_state_from_module(self)
            except AssertionError:
                # Module also lacks a stashed `module.quant_state`; fall
                # through to the eager error below with a clearer message.
                pass
            finally:
                if tuple(w.shape) != original_shape:
                    w.data = w.data.reshape(*original_shape)

        # Some layers may not be quantized (no quant_state) - fall back to regular matmul
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is None:
            bias = None if self.bias is None else self.bias
            weight = self.weight
            # Hard-detect "still packed but unrecoverable": shape doesn't
            # match (out_features, in_features). Emit a specific error
            # instead of a confusing `mat1 x mat2` mismatch.
            of = getattr(self, "out_features", None)
            if_ = getattr(self, "in_features", None)
            if of is not None and if_ is not None and tuple(weight.shape) != (of, if_):
                raise RuntimeError(
                    f"Unsloth: Linear4bit weight is in packed layout {tuple(weight.shape)} "
                    f"but has no `quant_state` to dequantize. Expected dense "
                    f"shape ({of}, {if_}). This usually means the model loader "
                    f"did not register a `quant_state` for this layer. "
                    f"Workarounds: (1) load with `load_in_4bit=False`, or "
                    f"(2) add the layer prefix to `llm_int8_skip_modules` so "
                    f"it is materialised as dense fp16/bf16."
                )
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
