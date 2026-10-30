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

"""Activation memory budget helpers shared by sequence tiling and the fused losses."""

import functools
import math

import torch

from unsloth_zoo.device_type import DEVICE_TYPE

__all__ = [
    "get_max_flat_qlen",
    "_default_target_gb",
    "_host_target_gb",
]

def _max_flat_qlen(
    hd = 4096,
    mlp_size = 14336,
    nbytes = 2,
    target_gb = 0.5,
    padded_length = 64,
):
    # Saved tensors are 10*hd per token (2 norms, 2 residual, 4 QKVO, 2 attention); MLP terms added.
    numerator = target_gb * 1024 * 1024 * 1024 / nbytes - (3*hd*mlp_size)
    denominator = (10*hd + 3*mlp_size + hd)
    max_flat_qlen = math.ceil(numerator / denominator)
    max_flat_qlen = max(padded_length, (max_flat_qlen // padded_length) * padded_length)
    return max_flat_qlen
pass

# Cached for eager callers; compiled callers use _max_flat_qlen (plain arithmetic, nothing for Dynamo to trace around).
get_max_flat_qlen = functools.cache(_max_flat_qlen)


def _default_target_gb(device = None):
    # Budget from free memory on the current device, keeping half; None means the current device, never GPU 0, because
    # under torchrun GPU 0 is the wrong card and opens a CUDA context
    index = None
    if device is not None:
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type in ("cpu", "meta", "mps"):
            return _host_target_gb()
        index = device.index
    if DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
        # torch.cuda.mem_get_info also covers ROCm (PyTorch aliases the cuda API).
        free = torch.cuda.mem_get_info(index)[0]
        return free / 1024 / 1024 / 1024 * 0.5
    if DEVICE_TYPE == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        # torch.xpu.mem_get_info is present from torch 2.6 but raises on some Intel GPUs (e.g. Arc B580, Lunar Lake);
        # then use a small bounded default, since host RAM would overshoot VRAM
        try:
            free = torch.xpu.mem_get_info(index)[0]
            return free / 1024 / 1024 / 1024 * 0.5
        except Exception:
            return 4.0
    if DEVICE_TYPE == "npu" and hasattr(torch, "npu"):
        try:
            free = torch.npu.mem_get_info(index)[0]
            return free / 1024 / 1024 / 1024 * 0.5
        except Exception:
            return 4.0
    return _host_target_gb()
pass

def _host_target_gb():
    # CPU / MPS / unified-memory backends: activations live in host RAM, budget from it.
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024 / 1024 * 0.5
    except Exception:
        return 4.0
pass
