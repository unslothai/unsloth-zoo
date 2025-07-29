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

__version__ = "2025.7.11"

import os
# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
pass

# More stable downloads
if os.environ.get("UNSLOTH_STABLE_DOWNLOADS", "0") == "1":
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DISABLE_XET"] = "1" # Disable XET
pass

# Disable XET Cache for now
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "0"
os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "64"
# More verbose HF Hub info
if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
    os.environ["HF_HUB_VERBOSITY"] = "info"
pass

# More logging for Triton
os.environ["TRITON_DISABLE_LINE_INFO"] = "1" # Reduces Triton binary size
os.environ["TRITON_FRONT_END_DEBUGGING"] = "0" # Disables debugging

if (os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1") or \
    (os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1"):
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1" # Prints out Triton best configs
    os.environ["TRITON_DISABLE_LINE_INFO"] = "0" # Enables Triton line info
    os.environ["TRITON_FRONT_END_DEBUGGING"] = "0" # Debugging
    os.environ["TRITON_ALWAYS_COMPILE"] = "1" # Always compile kernels
pass

# Triton compile debugging
if (os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1"):
    os.environ["TRITON_ENABLE_LLVM_DEBUG"] = "1" # Lots of debugging info
    os.environ["TRITON_INTERPRET"] = "1" # Can add print statements!
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Blocking calls for debugging
pass


from importlib.util import find_spec
if find_spec("unsloth") is None:
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
pass
del find_spec

def get_device_type():
    import torch
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise NotImplementedError("Unsloth currently only works on NVIDIA GPUs and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()

if not ("UNSLOTH_IS_PRESENT" in os.environ):
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
pass

try:
    print("🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
except:
    print("Unsloth: Will patch your computer to enable 2x faster free finetuning.")
pass
# Log Unsloth-Zoo Utilities
os.environ["UNSLOTH_ZOO_IS_PRESENT"] = "1"
del os
