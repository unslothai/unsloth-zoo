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
    except Exception as e:
        return raise_error("bitsandbytes.Linear4bit", e)

    # Fix Params4bit.__torch_function__ infinite recursion under torch.compile.
    # bitsandbytes >= 0.46 added a __torch_function__ to Params4bit that calls
    # super().__torch_function__() for all ops except chunk/split. The super()
    # call goes to Parameter.__torch_function__ which re-dispatches back to
    # Params4bit.__torch_function__ since Params4bit is still in the types tuple.
    # In eager mode, torch's _disabled_torch_function_impl prevents this, but
    # under torch.compile's AOT autograd runtime, the re-dispatch is not blocked,
    # causing infinite recursion (observed on T4 with torch 2.8.0 + bnb 0.49.2).
    #
    # Fix: remove Params4bit's __torch_function__ entirely so it inherits from
    # Parameter/Tensor which use C-level dispatch that cannot recurse. The bnb
    # implementation only adds chunk/split handling (rarely used for 4-bit weights)
    # and delegates everything else to super() anyway.
    if hasattr(Params4bit, "__torch_function__") and \
       "__torch_function__" in Params4bit.__dict__:
        delattr(Params4bit, "__torch_function__")
    pass

    def forward(self, x: torch.Tensor):
        # In transformers 5.0+, weights may not be in packed format yet during init.
        # Detect packed weights needing quant_state recovery (FSDP re-wrap case)
        # without accessing .shape or .data on Params4bit -- both trigger
        # __torch_function__ recursion under torch.compile.
        if getattr(self.weight, "quant_state", None) is None and \
           getattr(self, "quant_state", None) is not None:
            if not isinstance(self.weight, Params4bit):
                self.weight = Params4bit(
                    self.weight, quant_storage=self.quant_storage, bnb_quantized=True,
                )
            self.weight.quant_state = self.quant_state

        # Some layers may not be quantized (no quant_state) - fall back to regular matmul
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is None:
            bias = None if self.bias is None else self.bias
            weight = self.weight
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
