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
]

import os
import logging
UNSLOTH_ENABLE_LOGGING  = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") == "1"
logger = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger.setLevel(logging.DEBUG)

def get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
):
    UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
    UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
    UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
    torch_compile_options = {
        "epilogue_fusion"           : epilogue_fusion,
        "max_autotune"              : max_autotune,
        "shape_padding"             : shape_padding,
        "trace.enabled"             : UNSLOTH_COMPILE_DEBUG or debug,
        "triton.cudagraphs"         : cudagraphs,
        "debug"                     : UNSLOTH_COMPILE_DEBUG or debug,
        "dce"                       : False,
        "memory_planning"           : False,
        "coordinate_descent_tuning" : UNSLOTH_COMPILE_MAXIMUM,
        "trace.graph_diagram"       : UNSLOTH_COMPILE_DEBUG or debug,
        # "compile_threads"           : 24, # Auto detects via https://github.com/unslothai/unsloth-zoo/pull/187
        "combo_kernels"             : False, # Causes incompatible gradient sizes on 2.6
        "group_fusion"              : False,
        "disable_progress"          : not UNSLOTH_ENABLE_LOGGING,
        "verbose_progress"          : UNSLOTH_ENABLE_LOGGING,
        "triton.multi_kernel"       : False, # Sometimes fails
        "triton.use_block_ptr"      : False,
        "triton.enable_persistent_tma_matmul" : False,
        "triton.autotune_at_compile_time"     : False,
    }
    return torch_compile_options
pass
torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
)

global TEMPORARY_PATCHES
TEMPORARY_PATCHES = []
