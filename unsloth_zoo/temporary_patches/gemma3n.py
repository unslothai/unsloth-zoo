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
import torch
import torch.nn as nn
from .common import TEMPORARY_PATCHES, torch_compile_options
from .utils import (
    patch_function,
    KWARGS_TYPE,
    raise_error,
)

def patch_Gemma3nConvNormAct_forward():
    try:
        import timm.layers.conv_bn_act
        timm.layers.conv_bn_act.ConvNormAct
    except Exception as e:
        return raise_error("timm.layers.conv_bn_act.ConvNormAct", e)

    # Counteract high weights in Conv layers for Gemma 3N by forcing to float32
    def forward(self, x):
        old_dtype = x.dtype
        x = x.to(torch.float32)
        with torch.autocast(device_type = "cuda", dtype = torch.float32, enabled = True):
            x = self.conv(x)
        x = self.bn(x)
        aa = getattr(self, 'aa', None)
        if aa is not None:
            x = self.aa(x)
        return x.to(old_dtype)
    pass
    patch_function(timm.layers.conv_bn_act.ConvNormAct, "forward", forward, fullgraph = True)
pass
# We only execute this for float16 so it's not always executed
# TEMPORARY_PATCHES.append(patch_Gemma3nConvNormAct_forward)


def patch_Gemma3nTextAltUp_functions():
    try:
        import transformers.models.gemma3n.modeling_gemma3n
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp
    except Exception as e:
        return raise_error("Gemma3nTextAltUp", e)

    if hasattr(Gemma3nTextAltUp, "predict"):
        patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "predict", Gemma3nTextAltUp.predict, fullgraph = True)
    if hasattr(Gemma3nTextAltUp, "correct"):
        patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "correct", Gemma3nTextAltUp.correct, fullgraph = True)
    if hasattr(Gemma3nTextAltUp, "scale_corrected_output"):
        patch_function(transformers.models.gemma3n.modeling_gemma3n.Gemma3nTextAltUp, "scale_corrected_output", Gemma3nTextAltUp.scale_corrected_output, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_Gemma3nTextAltUp_functions)
