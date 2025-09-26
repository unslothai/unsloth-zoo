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
        fix_4bit_weight_quant_state_from_module = bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module
    except Exception as e:
        return raise_error("bitsandbytes.Linear4bit", e)

    def forward(self, x: torch.Tensor):
        fix_4bit_weight_quant_state_from_module(self)

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

        # Cannot do .t() on Params4bit, instead do it on torch.Tensor
        weight = self.weight.data.t()

        return bitsandbytes.matmul_4bit(x, weight, bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)

    patch_function(bitsandbytes.nn.modules.Linear4bit, "forward", forward)
    try:
        patch_function(bitsandbytes.nn.Linear4bit, "forward", forward)
    except:
        pass
pass
TEMPORARY_PATCHES.append(patch_bitsandbytes_linear4bit_forward)
