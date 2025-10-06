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
    "UNSLOTH_COMPILE_LOCATION",
    "UNSLOTH_COMPILE_USE_TEMP",
    "COMBINED_UNSLOTH_NAME",
    "get_compile_folder",
]

from packaging.version import Version as TrueVersion
import torch
import os
import time
import contextlib
import re
import pathlib
import zlib
from typing import Optional
from filelock import FileLock
import tempfile
from unsloth_zoo.log import logger

# Compiled cache location
global COMBINED_UNSLOTH_NAME
COMBINED_UNSLOTH_NAME = "unsloth_compiled_module"

global UNSLOTH_COMPILE_LOCATION
if 'UNSLOTH_COMPILE_LOCATION' not in globals():
    _loc = os.getenv("UNSLOTH_COMPILE_LOCATION", None)
    if _loc:
        UNSLOTH_COMPILE_LOCATION = _loc
    else:
        UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache"

global UNSLOTH_COMPILE_USE_TEMP
UNSLOTH_COMPILE_USE_TEMP = False

def _get_compile_folder(use_tempfile = False):
    global UNSLOTH_COMPILE_LOCATION
    global UNSLOTH_COMPILE_USE_TEMP
    if UNSLOTH_COMPILE_USE_TEMP or use_tempfile:
        UNSLOTH_COMPILE_USE_TEMP = True
        leaf = os.path.basename(UNSLOTH_COMPILE_LOCATION)
        location = os.path.join(tempfile.gettempdir(), leaf)
        logger.info(
            f"Unsloth: We'll be using `{location}` for temporary Unsloth patches."
        )
        os.makedirs(location, exist_ok = True)
    else:
        location = UNSLOTH_COMPILE_LOCATION
        try:
            # Try creating the directory
            os.makedirs(location, exist_ok = True)
            return location, UNSLOTH_COMPILE_USE_TEMP
        except Exception as e:
            logger.error(f"Unsloth: Failed to create directory `{UNSLOTH_COMPILE_LOCATION}` because {str(e)}")

            # Instead use a temporary location!
            location, UNSLOTH_COMPILE_USE_TEMP = _get_compile_folder(use_tempfile = True)
    return location, UNSLOTH_COMPILE_USE_TEMP
pass

def get_compile_folder(use_tempfile = False, distributed = True):
    if distributed:
        location, UNSLOTH_COMPILE_USE_TEMP = distributed_function(2, _get_compile_folder, use_tempfile)
    else:
        location, UNSLOTH_COMPILE_USE_TEMP = _get_compile_folder(use_tempfile)
    return location, UNSLOTH_COMPILE_USE_TEMP
pass

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

def _canon_key(p: str) -> str:
    s = os.path.abspath(p)
    if os.name == "nt":
        s = os.path.normcase(s)
    return os.path.normpath(s)

def _slug(name: str, maxlen: int = 100) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return (name or "_")[:maxlen]

def _lock_name_for(target: str) -> str:
    canon = _canon_key(target)
    base  = _slug(pathlib.Path(canon).name)
    h8    = f"{zlib.crc32(canon.encode('utf-8')) & 0xffffffff:08x}"
    return f"{base}.{h8}.lock"

def _lock_path_for(target: str) -> str:
    """ str needs to be a valid file path """
    base_dir = _get_compile_folder()[0]
    locks_dir = pathlib.Path(base_dir) / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_name = _lock_name_for(target)
    return str(locks_dir / lock_name)

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
  
def write_file(function_location, write_new_source):
    lock = get_lock(function_location)
    new_write_bytes = write_new_source.encode("utf-8")
    try:
        with lock:
            # existence check
            try:
                st = os.stat(function_location)
            except Exception as e:
                st = None

            need_write = False
            if st is None or st.st_size != len(new_write_bytes):
                need_write = True
            else:
                with open(function_location, "rb") as f:
                    need_write = f.read() != new_write_bytes

            if need_write:
                with open(function_location, "wb", buffering = 0) as file:
                    file.write(new_write_bytes)
                    file.flush()
                    os.fsync(file.fileno())
        return None
    except Exception as e:
        # consider adding logging to main_process only
        # counterpoint: we may want to see errors on all processes
        if os.environ.get("UNSLOTH_LOGGING_ENABLED", "0") == "1":
            logger.error(f"Unsloth: Failed to write file {function_location} because {str(e)}")
        return None
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
