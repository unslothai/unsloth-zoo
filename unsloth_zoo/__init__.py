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

__version__ = "2026.1.1"

import os
import warnings
import re
# Stop TOKENIZERS_PARALLELISM warning
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# More stable downloads
if os.environ.get("UNSLOTH_STABLE_DOWNLOADS", "0") == "1":
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DISABLE_XET"] = "1" # Disable XET

# Check offline mode as well
if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"

# Disable XET Cache for now
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "0"
os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "64"
# More verbose HF Hub info
if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
    os.environ["HF_HUB_VERBOSITY"] = "info"

# More logging for Triton
os.environ["TRITON_DISABLE_LINE_INFO"] = "1" # Reduces Triton binary size
os.environ["TRITON_FRONT_END_DEBUGGING"] = "0" # Disables debugging

if (os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1") or \
    (os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1"):
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1" # Prints out Triton best configs
    os.environ["TRITON_DISABLE_LINE_INFO"] = "0" # Enables Triton line info
    os.environ["TRITON_FRONT_END_DEBUGGING"] = "0" # Debugging
    os.environ["TRITON_ALWAYS_COMPILE"] = "1" # Always compile kernels
    os.environ["NCCL_DEBUG"] = "WARN" # Warn on NCCL issues

# Triton compile debugging
if (os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1"):
    # Lots of debugging info
    # BUT weirdly blocks torch.compile, so we disable
    os.environ["TRITON_ENABLE_LLVM_DEBUG"] = "0"
    # Can add print statements, but slower so disable
    # Also fails on get_int1_ty for example (bool)
    os.environ["TRITON_INTERPRET"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Blocking calls for debugging


from importlib.util import find_spec
if find_spec("unsloth") is None:
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
if find_spec("torch") is None:
    raise ImportError(
        "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"\
        "We also have some installation instructions on our Github page."
    )

# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "0":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
            "expandable_segments:True,"\
            "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
    if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
        # [TODO] Check if AMD works with roundup_power2_divisions
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
elif os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "1":
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF",):
        if "expandable_segments:True" in os.environ.get(key, ""):
            warnings.warn(
                "Unsloth: `UNSLOTH_VLLM_STANDBY` is on, but requires `expandable_segments` to be off. "\
                "We will remove `expandable_segments`.",
                stacklevel = 2,
            )
            os.environ[key] = re.sub(r"expandable\_segments\:True\,?", "", os.environ[key])

# We support Pytorch 2
# Fixes https://github.com/unslothai/unsloth/issues/38
from importlib.metadata import version as importlib_version
torch_version = str(re.match(r"[0-9\.]{3,}", str(importlib_version("torch"))).group(0)).split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
def delete_key(key):
    if key in os.environ: del os.environ[key]
if (major_torch < 2):
    raise ImportError("Unsloth only supports Pytorch 2 for now. Please update your Pytorch to 2.1.\n"\
                      "We have some installation instructions on our Github page.")
elif (major_torch == 2) and (minor_torch < 2):
    # Disable expandable_segments
    delete_key("PYTORCH_CUDA_ALLOC_CONF")
    delete_key("PYTORCH_HIP_ALLOC_CONF")
    delete_key("PYTORCH_ALLOC_CONF")
elif bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")):
    # Expandable segments does NOT work on WSL
    delete_key("PYTORCH_CUDA_ALLOC_CONF")
    delete_key("PYTORCH_HIP_ALLOC_CONF")
    delete_key("PYTORCH_ALLOC_CONF")
elif os.name == 'nt':
    # Expandable segments does NOT work on Windows
    delete_key("PYTORCH_CUDA_ALLOC_CONF")
    delete_key("PYTORCH_HIP_ALLOC_CONF")
    delete_key("PYTORCH_ALLOC_CONF")

# Suppress WARNING:torchao:Skipping import of cpp extensions due to incompatible torch version 2.7.0+cu126 for torchao version 0.14.1
# Please see https://github.com/pytorch/ao/issues/2919 for more info
import logging
torchao_logger = logging.getLogger("torchao")
# Ignore logging messages
class HideLoggingMessage(logging.Filter):
    __slots__ = "text",
    def __init__(self, text): self.text = text
    def filter(self, x): return not (self.text in x.getMessage())

torchao_logger.addFilter(HideLoggingMessage("Skipping import"))
del logging, torchao_logger, HideLoggingMessage

# Get device types and other variables
from .device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)

# Torch 2.9 removed PYTORCH_HIP_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF
if major_torch == 2 and minor_torch >= 9:
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF",):
        if key in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_ALLOC_CONF"] = os.environ[key]
            delete_key(key)
else:
    # Specify PYTORCH_CUDA_ALLOC_CONF or PYTORCH_HIP_ALLOC_CONF
    if DEVICE_TYPE == "hip":
        if "PYTORCH_HIP_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            delete_key("PYTORCH_CUDA_ALLOC_CONF")
        if "PYTORCH_HIP_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" in os.environ:
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
            delete_key("PYTORCH_ALLOC_CONF")
    elif DEVICE_TYPE == "cuda":
        delete_key("PYTORCH_HIP_ALLOC_CONF")
        delete_key("PYTORCH_ALLOC_CONF")

# CCE fails on Torch 2.8 and above
# OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536. Reducing block sizes or `num_stages`
if (major_torch >= 2 and minor_torch >= 8) or (major_torch > 2):
    os.environ["UNSLOTH_ENABLE_CCE"] = "0"
elif DEVICE_TYPE == "hip":
    # CCE also fails in HIP / AMD
    os.environ["UNSLOTH_ENABLE_CCE"] = "0"
del delete_key, major_torch, minor_torch, torch_version, importlib_version, find_spec

if not ("UNSLOTH_IS_PRESENT" in os.environ):
    raise ImportError("Please install Unsloth via `pip install unsloth`!")

try:
    print("🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
except:
    print("Unsloth: Will patch your computer to enable 2x faster free finetuning.")

# Log Unsloth-Zoo Utilities
os.environ["UNSLOTH_ZOO_IS_PRESENT"] = "1"

from .temporary_patches import (
    encode_conversations_with_harmony,
)
from .rl_environments import (
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
    Benchmarker,
    is_port_open,
    launch_openenv,
)

# Top some pydantic warnings
try:
    # pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True
    # was provided to the `Field()` function, which has no effect in the context it was used.
    # 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment.
    # This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings(action = "ignore", category = UnsupportedFieldAttributeWarning)
    del UnsupportedFieldAttributeWarning
except:
    pass

del os, warnings, re
