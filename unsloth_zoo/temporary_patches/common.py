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
    "TEMPORARY_PATCHES", 
    "torch_compile_options",
    "UNSLOTH_ENABLE_LOGGING",
    "UNSLOTH_COMPILE_DISABLE",
    "get_torch_compile_options",
    "logger",
    "torch_compile",
    "_torch_compile",
]

import os
import sys
import logging
from ..log import logger
import functools
UNSLOTH_ENABLE_LOGGING  = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1"

# Get only allowed options
import inspect
import torch
inductor_config_source = inspect.getsource(torch._inductor.config)

@functools.lru_cache(1)
def determine_compile_threads():
    # See https://github.com/pytorch/pytorch/blob/ab2294d8289a7757a2fc321cdefac88e2b378edf/torch/_inductor/config.py#L771
    # Windows thread count = 1. See https://github.com/unslothai/unsloth-zoo/pull/187
    if sys.platform == "win32": return 1
    cpu_count = os.cpu_count()
    return min(32, max(4, cpu_count))
pass

def get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
    coordinate_descent_tuning = False,
    logging = False,
    combo_kernels = False,
    group_fusion = True,
    memory_planning = True,
    multi_kernel = False,
    use_block_ptr = False,
):
    UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
    UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
    UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
    if UNSLOTH_ENABLE_LOGGING: logging = True

    # https://github.com/pytorch/pytorch/blob/c665594c1edca9a507b0ec8b18ab74a0ecb65bc3/torch/_inductor/config.py#L1283
    # Needs integer
    multi_kernel = 1 if multi_kernel else 0

    # Instead of Inductor Compilation:
    try:
        import torch._inductor.async_compile
        from torch.hub import tqdm
        def replaced_tqdm(*args, **kwargs):
            kwargs["desc"] = "Unsloth: Compiling kernels"
            return tqdm(*args, **kwargs)
        torch._inductor.async_compile.tqdm = replaced_tqdm
    except:
        print("Unsloth: Failed editing tqdm to replace Inductor Compilation:")
    pass

    torch_compile_options = {
        "epilogue_fusion"           : epilogue_fusion,
        "max_autotune"              : max_autotune,
        "shape_padding"             : shape_padding,
        "trace.enabled"             : UNSLOTH_COMPILE_DEBUG or debug,
        "triton.cudagraphs"         : cudagraphs,
        "debug"                     : UNSLOTH_COMPILE_DEBUG or debug,
        "dce"                       : True,
        "memory_planning"           : memory_planning,
        "coordinate_descent_tuning" : coordinate_descent_tuning or UNSLOTH_COMPILE_MAXIMUM,
        "trace.graph_diagram"       : UNSLOTH_COMPILE_DEBUG or debug,
        "compile_threads"           : determine_compile_threads(), # Auto detects via https://github.com/unslothai/unsloth-zoo/pull/187
        "group_fusion"              : group_fusion, # [DEPRECATED]
        "disable_progress"          : not logging,
        "verbose_progress"          : logging,

        "triton.multi_kernel"       : multi_kernel, # RuntimeError: name 'multi_kernel_0' is not defined
        "triton.use_block_ptr"      : use_block_ptr,
        "triton.enable_persistent_tma_matmul" : True,
        "triton.autotune_at_compile_time"     : False,
        "triton.cooperative_reductions"       : False,
        # "reorder_for_compute_comm_overlap"  : True, # Fails for single GPU
        "cuda.compile_opt_level"              : "-O2",
        "cuda.enable_cuda_lto"                : True,
        # "cuda.use_fast_math"                : True, # Disable fast math
        # Causes incompatible gradient sizes on 2.6
        # And TypeError: bad operand type for unary -: 'SymbolicCallArg'
        "combo_kernels"                       : combo_kernels,
        "benchmark_combo_kernel"              : True,
        "combo_kernel_foreach_dynamic_shapes" : True,
    }
    final_torch_compile_options = {}
    for key, value in torch_compile_options.items():
        splits = key.split(".")
        if all(k in inductor_config_source for k in splits):
            final_torch_compile_options[key] = value
    return final_torch_compile_options
pass
torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
    coordinate_descent_tuning = False,
    logging = UNSLOTH_ENABLE_LOGGING,
    combo_kernels = False,
    memory_planning = False,
    multi_kernel = False,
    use_block_ptr = False,
)

from typing import Any, Callable, TypeVar
F = TypeVar("F", bound=Callable[..., Any])
def noop(*args: Any, **kwargs: Any):
    """
    A do-nothing decorator/adapter.

    Works as:
      - @noop
      - @noop(...)
      - noop(func, ...)
    
    Returns the original function unchanged in every case.
    """
    # If used like noop(func, **kwargs) or as @noop on a function,
    # the first positional arg will be the function. Return it directly.
    if args and callable(args[0]):
        return torch.compiler.disable(args[0]) # type: ignore[return-value]

    # Otherwise, used as @noop(...): return a decorator that returns the function unchanged.
    def _decorator(func: F) -> F:
        return torch.compiler.disable(func)
    return _decorator
pass

if UNSLOTH_COMPILE_DISABLE:
    torch_compile = noop
else:
    torch_compile = functools.partial(
        torch.compile,
        options = torch_compile_options,
    )

if UNSLOTH_COMPILE_DISABLE:
    _torch_compile = noop
else:
    _torch_compile = functools.partial(
        torch.compile,
    )

global TEMPORARY_PATCHES
TEMPORARY_PATCHES = []
