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
    "torch_distributed_get_rank",
]

from packaging.version import Version as TrueVersion
import torch
import os
import re


def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        version = str(version)
        try:
            return TrueVersion(version)
        except Exception as e:
            version = re.match(r"[0-9\.]{1,}", version)
            if version is None:
                raise Exception(str(e))
            version = version.group(0).rstrip(".")
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
            dtype = dtype.lower()
            return getattr(torch, dtype, None)
        elif isinstance(dtype, torch.dtype):
            return dtype
    return None
pass


torch_distributed_is_initialized = torch.distributed.is_initialized
torch_distributed_get_rank = torch.distributed.get_rank


def is_main_process():
    if torch_distributed_is_initialized():
        # torch.distributed.init_process_group was run, so get_rank works
        return torch_distributed_get_rank() == 0
    return os.environ.get("RANK", "0") == "0"
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
