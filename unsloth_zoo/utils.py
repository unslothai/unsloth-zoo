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
import os
import time
import contextlib

def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        return TrueVersion(version)
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
            if hasattr(torch, dtype.lower()):
                return getattr(torch, dtype.lower())
        if isinstance(dtype, torch.dtype): return dtype
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
    if is_distributed():
        if is_main_process():
            object_list = function(*args, **kwargs)
            if n == 1: object_list = [object_list]
        else:
            object_list = [None for _ in range(n)]
        # broadcast_object_list auto blocks so no need for barrier
        if not torch_distributed_is_initialized():
            # But check if the function even works!
            # This happens when torch_distributed_is_torchelastic_launched()==True but
            # torch_distributed_is_initialized()==False
            # Trick is to just add a 0.01+0.01*RANK second sleep and print with flush
            time.sleep(0.01 + 0.01*int(os.environ.get("RANK", "0")))
            with contextlib.redirect_stdout(None):
                print("", flush = True)
            object_list = function(*args, **kwargs)
            if n == 1: object_list = [object_list]
        else:
            torch.distributed.broadcast_object_list(object_list, src = 0)
        if n == 1:
            result = object_list[0]
        else:
            result = object_list
    else:
        result = function(*args, **kwargs)
    return result
pass

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
