# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Activation memory budget helpers shared by sequence tiling and the fused losses."""

import functools
import math
import os

import torch

from unsloth_zoo.device_type import DEVICE_TYPE

__all__ = [
    "get_max_flat_qlen",
    "_default_target_gb",
    "_host_target_gb",
    "host_memory_bytes",
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

get_max_flat_qlen = functools.cache(_max_flat_qlen)


def _default_target_gb(device = None):
    # Current device, never GPU 0: under torchrun GPU 0 is the wrong card and opens a CUDA context.
    index = None
    if device is not None:
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type in ("cpu", "meta", "mps"):
            return _host_target_gb()
        index = device.index
    if DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
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
    try:
        return host_memory_bytes()[1] / 1024 / 1024 / 1024 * 0.5
    except Exception:
        return 4.0
pass


# psutil reports the machine inside a container: a 16GB cgroup on a 64GB host would size host buffers for 48GB.
# 0 restores the psutil-only reading.
HOST_BUDGET_CGROUP_ENV = "UNSLOTH_HOST_BUDGET_CGROUP"


def _cgroup_memory_bounds():
    """(smallest limit, smallest headroom) of the enforcing cgroup memory limits, or (None, None) outside one. Each
    limit pairs with its own directory's usage, minus that directory's file LRU (reclaimable page cache, which
    MemAvailable also counts as free), so a load's mmap cache does not read as used."""
    if os.environ.get(HOST_BUDGET_CGROUP_ENV, "").strip().lower() in ("0", "off", "false", "no"):
        return None, None
    try:
        from unsloth_zoo.hf_xet_tuning import (
            _cgroup_v1_dirs,
            _cgroup_v2_dirs,
            _parse_limit,
            _read_first_line,
        )
    except Exception:
        return None, None

    def _int(raw):
        try: return int(raw.strip()) if raw else None
        except ValueError: return None

    def _file_lru(directory, names):
        try:
            with open(directory / "memory.stat", "r") as f:
                stat = dict(line.split()[:2] for line in f if len(line.split()) >= 2)
        except OSError:
            return 0
        return sum(_int(stat.get(name)) or 0 for name in names)

    limits, headroom = [], []
    dirs = [(d, ("memory.max", "memory.high"), "memory.current", ("active_file", "inactive_file"))
            for d in _cgroup_v2_dirs()]
    dirs += [(d, ("memory.limit_in_bytes",), "memory.usage_in_bytes", ("total_active_file", "total_inactive_file"))
             for d in _cgroup_v1_dirs("memory")]
    for directory, limit_names, usage_name, lru_names in dirs:
        found = [_parse_limit(_read_first_line(directory / name)) for name in limit_names]
        found = [value for value in found if value is not None]
        if not found:
            continue
        limit = min(found)
        used = _int(_read_first_line(directory / usage_name))
        limits.append(limit)
        headroom.append(limit if used is None else limit - max(0, used - _file_lru(directory, lru_names)))
    if not limits:
        return None, None
    return min(limits), max(0, min(headroom))
pass


def host_memory_bytes():
    """(total, available) host RAM this process may use: psutil's, capped by an enforcing cgroup limit."""
    import psutil
    vm = psutil.virtual_memory()
    total, available = getattr(vm, "total", None), vm.available
    limit, headroom = _cgroup_memory_bounds()
    if limit is not None and total is not None:
        total = min(total, limit)
    if headroom is not None:
        available = min(available, headroom)
    return total, available
pass
