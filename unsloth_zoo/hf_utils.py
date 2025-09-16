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
from transformers import PretrainedConfig
HAS_TORCH_DTYPE = "torch_dtype" in PretrainedConfig.__doc__

__all__ = [
    "HAS_TORCH_DTYPE",
    "dtype_from_config",
    "add_dtype_kwargs",
    "set_dtype_in_config",
]

def dtype_from_config(config):
    check_order = ['dtype', 'torch_dtype']
    if HAS_TORCH_DTYPE:
        check_order = ['torch_dtype', 'dtype']
    dtype = None
    for dtype_name in check_order:
        if dtype is None:
            dtype = getattr(config, dtype_name, None)
    return dtype

def set_dtype_in_config(config, dtype):
    try:
        # if dtype is not a string, convert it to a string
        string_dtype = str(dtype).split(".")[-1] if isinstance(dtype, torch.dtype) else dtype
        if HAS_TORCH_DTYPE:
            setattr(config, "torch_dtype", string_dtype)
        else:
            setattr(config, "dtype", string_dtype)
    except:
        set_dtype_in_config_fallback(config, string_dtype)

def set_dtype_in_config_fallback(config, dtype):
    try:
        string_dtype = str(dtype).split(".")[-1] if isinstance(dtype, torch.dtype) else dtype
        if HAS_TORCH_DTYPE:
            config.__dict__["torch_dtype"] = string_dtype
        else:
            config.__dict__["dtype"] = string_dtype
    except:
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            print("Unsloth: Failed to set dtype in config, fallback failed too")

def add_dtype_kwargs(dtype, kwargs_dict=None):
    if kwargs_dict is None:
        kwargs_dict = {}
    if HAS_TORCH_DTYPE:
        kwargs_dict["torch_dtype"] = dtype
    else:
        kwargs_dict["dtype"] = dtype
    return kwargs_dict

def _dtype_stringify(x):
    # Convert *values* (not the config) into JSON-safe strings when they are dtypes
    try:
        if isinstance(x, torch.dtype):
            # str(torch.float16) -> "torch.float16" -> "float16"
            return str(x).split(".", 1)[-1]
        if isinstance(x, str) and x.startswith("torch."):
            tail = x.split(".", 1)[-1]
            # Only strip "torch." if the tail is actually a dtype on torch
            if hasattr(torch, tail) and isinstance(getattr(torch, tail), torch.dtype):
                return tail
    except Exception:
        pass

    return x

def _normalize_dict_dtypes(obj):
    if isinstance(obj, dict):
        return {k: _normalize_dict_dtypes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_dict_dtypes(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_dict_dtypes(v) for v in obj)
    return _dtype_stringify(obj)
    