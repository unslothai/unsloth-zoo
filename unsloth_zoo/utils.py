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

__all__ = [
    "Version",
    "_get_dtype",
    "is_main_process",
    "is_distributed",
    "distributed_function",
    "torch_distributed_get_rank",
]

from packaging.version import Version as TrueVersion
import torch
import torch.distributed as dist
import os
import time
import contextlib
import re
import pathlib
from typing import Optional
from filelock import FileLock

def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        new_version = str(version)
        new_version = re.match(r"[0-9\.]{1,}", new_version)
        if new_version is None:
            raise ValueError(f"Invalid version format: {version}")
        new_version = new_version.group(0).rstrip(".")
        if new_version != version:
            new_version += ".1" # Add .1 for dev / alpha / beta / rc
        return TrueVersion(new_version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n"\
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass
pass


__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}
def _get_dtype(dtype):
    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            dtype = dtype.lower()
            return getattr(torch, dtype, None)
        elif isinstance(dtype, torch.dtype):
            return dtype
    return None
pass


import functools
torch_distributed_is_initialized = torch.distributed.is_initialized
torch_distributed_is_torchelastic_launched = torch.distributed.is_torchelastic_launched
torch_distributed_get_rank = torch.distributed.get_rank

def is_main_process():
    if torch_distributed_is_initialized():
        # torch.distributed.init_process_group was run, so get_rank works
        return torch_distributed_get_rank() == 0
    elif torch_distributed_is_torchelastic_launched():
        # accelerate launch for example calls init_process_group later
        return os.environ.get("RANK", "0") == "0"
    return True
pass

def is_distributed():
    return torch_distributed_is_initialized() or torch_distributed_is_torchelastic_launched()
pass

def distributed_function(n = 1, function = None, *args, **kwargs):
    assert function is not None

    # Run independently if process group isn't initialized yet.
    # This covers both: (1) not distributed at all, and (2) torchrun launched
    # but init_process_group() wasn't called yet (e.g. during module imports).
    # Ref: https://github.com/unslothai/unsloth/issues/3703
    if not torch_distributed_is_initialized():
        out = function(*args, **kwargs)
        return out if n == 1 else out

    # Multi-process: only main executes the function
    if is_main_process():
        out = function(*args, **kwargs)
        obj_list = [out] if n == 1 else list(out)
    else:
        obj_list = [None for _ in range(n)]

    # If the process group is initialized, we can synchronize / share the result
    if torch_distributed_is_initialized():
        # Broadcast result to all ranks
        dist.broadcast_object_list(obj_list, src = 0)
        # Barrier to make sure everyone waits until main is done
        dist.barrier()

    return obj_list[0] if n == 1 else obj_list
pass

def _lock_path_for(target: str) -> str:
    """ str needs to be a valid file path """
    locks_dir = pathlib.Path(target).parent / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return str(locks_dir / f".lock.{pathlib.Path(target).name}")

def get_lock(target: str, timeout: Optional[int] = None) -> FileLock:
    """
    Get a lock for a target file.
    target: str, the path to the file to lock
    timeout: int, the timeout in seconds for the lock
    If timeout is not provided, it will use the value of
    the environment variable UNSLOTH_LOCK_TIMEOUT, otherwise 10 seconds.

    Returns:
        FileLock, the lock for the target file
    """
    lock_path = _lock_path_for(target)
    if timeout is None:
        timeout = int(os.environ.get("UNSLOTH_LOCK_TIMEOUT", "10"))
    return FileLock(lock_path, timeout=timeout)


def get_quant_type(config):
    quant_config = getattr(config, 'quantization_config', None)
    if quant_config:
        from transformers.quantizers import AutoQuantizationConfig
        if isinstance(quant_config, dict):
            return quant_config.get('quant_method', None)
        elif isinstance(quant_config, AutoQuantizationConfig):
            return getattr(quant_config, 'quant_method', None)
    return None

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
