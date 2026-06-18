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

__version__ = "2026.6.6"

import os
import warnings
import re
# Stop TOKENIZERS_PARALLELISM warning
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Detect offline mode first. hf_transfer is a Rust downloader that bypasses
# huggingface_hub's offline guard, so leaving it on defeats HF_HUB_OFFLINE
# and TRANSFORMERS_OFFLINE entirely.
_OFFLINE_TRUE = {"1", "true", "yes", "on"}
_offline_env = (
    os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_TRUE
    or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _OFFLINE_TRUE
    or os.environ.get("HF_DATASETS_OFFLINE", "").strip().lower() in _OFFLINE_TRUE
)

# Hugging Face Hub faster downloads (skipped when offline mode is requested).
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ and not _offline_env:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# More stable downloads
if os.environ.get("UNSLOTH_STABLE_DOWNLOADS", "0") == "1":
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DISABLE_XET"] = "1" # Disable XET
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0" # This causes "429 Too Many Requests"

# Cross-sync the three offline flags: setting any one implies all three.
# Without HF_DATASETS_OFFLINE, load_dataset() still hits the network for
# dataset metadata even when the rest of the HF stack is offline.
if _offline_env:
    for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[_v] = "1"
del _OFFLINE_TRUE, _offline_env

# Check "429 Too Many Requests" and set HF_XET_HIGH_PERFORMANCE
from pathlib import Path
def has_429_exact_full_read(log_dir: str | Path) -> str:
    log_dir = Path(log_dir).expanduser()
    if not log_dir.is_dir():
        return "1"
    for log_file in log_dir.glob("*.log"):
        try:
            if b"429 Too Many Requests" in log_file.read_bytes():
                return "0"
        except OSError:
            continue
    return "1"

# Redirect the HF cache off a read-only default (locked-down machines) so
# snapshot_download() can write. Runs before any huggingface_hub import.
from .hf_cache import redirect_hf_cache_if_readonly, _active_caches
redirect_hf_cache_if_readonly()

# _active_caches mirrors Hub's env layering (XDG_CACHE_HOME included) and
# returns None entries instead of raising when home is unresolvable; "1"
# matches the probe's no-logs-found default.
_, _, xet_cache = _active_caches()
os.environ.setdefault(
    "HF_XET_HIGH_PERFORMANCE",
    has_429_exact_full_read(xet_cache / "logs") if xet_cache is not None else "1",
)
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
os.environ.setdefault("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "0")
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "64")
del has_429_exact_full_read, xet_cache, redirect_hf_cache_if_readonly, _active_caches

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
from .mlx.runtime import is_mlx_available
from .model_lists import FORCE_FLOAT32

# Import-time fixes live in ``unsloth/import_fixes.py`` and run at ``import
# unsloth`` time. Zoo cannot be imported standalone (the GPU init below
# requires ``find_spec("unsloth")``), so they are always already in place.

# Detect Apple Silicon MLX mode: torch absent (pure MLX) or unsloth detected MLX
_is_mlx_only = is_mlx_available()

if _is_mlx_only:
    # MLX mode: skip all CUDA/torch-specific initialization.
    os.environ["UNSLOTH_ZOO_IS_PRESENT"] = "1"
    UNSLOTH_ZOO_IS_PRESENT = True
    DEVICE_TYPE = "mlx"
    DEVICE_TYPE_TORCH = "mps"
    DEVICE_COUNT = 1
    ALLOW_PREQUANTIZED_MODELS = True
    del _is_mlx_only, is_mlx_available, find_spec
    # Everything below this point is GPU-only. Use a flag to gate it.
    _SKIP_GPU_INIT = True
else:
    _SKIP_GPU_INIT = False
    del _is_mlx_only, is_mlx_available

# Inject triton & bitsandbytes stubs on Apple Silicon with MLX so unsloth's
# CUDA-only imports don't error at startup. _SKIP_GPU_INIT is True only on
# Darwin/arm64 with mlx installed (the exact case stubs are needed).
if _SKIP_GPU_INIT:
    from .stubs.triton_stub import inject_into_sys_modules as _inject_triton
    _inject_triton()
    from .stubs.bitsandbytes_stub import inject_into_sys_modules as _inject_bnb
    _inject_bnb()
    del _inject_triton, _inject_bnb

# Lazy bridge for downstream code that still imports the old flat MLX module
# names. Installed on every host so external scripts don't hit a hard
# ModuleNotFoundError at import time; the real import (which pulls in mlx)
# is deferred to first attribute access. On non-MLX hosts that access
# surfaces the same ModuleNotFoundError("mlx") users got pre-refactor.
import importlib as _importlib
import sys as _sys
import types as _types

class _LazyMLXAlias(_types.ModuleType):
    __slots__ = ()
    _LEGACY_TO_NEW = {
        "unsloth_zoo.mlx_loader": "unsloth_zoo.mlx.loader",
        "unsloth_zoo.mlx_trainer": "unsloth_zoo.mlx.trainer",
        "unsloth_zoo.mlx_utils": "unsloth_zoo.mlx.utils",
        "unsloth_zoo.mlx_compile": "unsloth_zoo.mlx.compile",
        "unsloth_zoo.mlx_cce": "unsloth_zoo.mlx.cce",
        "unsloth_zoo.mlx_cce.runtime_cce": "unsloth_zoo.mlx.cce.runtime_cce",
    }

    def _resolve(self):
        import importlib, sys
        target = self._LEGACY_TO_NEW[self.__name__]
        real = importlib.import_module(target)
        sys.modules[self.__name__] = real
        return real

    def __getattr__(self, name):
        # Skip dunder probes (inspect.getmodule, hasattr(..., '__file__'), etc.)
        # so we don't trigger an mlx import while torch walks sys.modules during
        # its own init. Real attribute access (e.g. FastMLXModel) still resolves.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            real = self._resolve()
        except ModuleNotFoundError:
            # mlx is Apple-only; on non-mlx hosts the submodule import fails.
            # Surface as AttributeError so sys.modules walkers (notably
            # pickle.whichmodule, used by torch._inductor's FX graph hash
            # pickler) skip this stub cleanly instead of crashing the compile.
            raise AttributeError(name)
        return getattr(real, name)

for _old_name in _LazyMLXAlias._LEGACY_TO_NEW:
    if _old_name in _sys.modules:
        continue
    _sys.modules[_old_name] = _LazyMLXAlias(_old_name)

del _old_name, _importlib, _sys, _types

if not _SKIP_GPU_INIT:
    if find_spec("unsloth") is None:
        raise ImportError("Please install Unsloth via `pip install unsloth`!")
    if find_spec("torch") is None:
        raise ImportError(
            "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"\
            "We also have some installation instructions on our Github page."
        )

if not _SKIP_GPU_INIT:
    # Keep original allocator settings to preserve explicit user config precedence.
    _ORIGINAL_PYTORCH_CUDA_ALLOC_CONF = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    _ORIGINAL_PYTORCH_HIP_ALLOC_CONF = os.environ.get("PYTORCH_HIP_ALLOC_CONF")
    _HAS_ORIGINAL_PYTORCH_ALLOC_CONF = "PYTORCH_ALLOC_CONF" in os.environ

    # We support Pytorch 2
    # Fixes https://github.com/unslothai/unsloth/issues/38
    from importlib.metadata import version as importlib_version
    torch_version_raw = str(importlib_version("torch"))
    torch_version = str(re.match(r"[0-9\.]{3,}", torch_version_raw).group(0)).split(".")
    major_torch, minor_torch = torch_version[0], torch_version[1]
    major_torch, minor_torch = int(major_torch), int(minor_torch)
    IS_TORCH_2_9_OR_NEWER = (major_torch > 2) or (major_torch == 2 and minor_torch >= 9)
    IS_TORCH_ROCM_BUILD = "+rocm" in torch_version_raw.lower()

    # Reduce VRAM fragmentation and optimize memory pinning
    if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "0":
        if IS_TORCH_2_9_OR_NEWER:
            if "PYTORCH_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        else:
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

    def delete_key(key):
        if key in os.environ: del os.environ[key]


    def remove_expandable_segments(key):
        value = os.environ.get(key, "")
        if "expandable_segments" not in value:
            return
        parts = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part.startswith("expandable_segments:"):
                continue
            parts.append(part)
        if parts:
            os.environ[key] = ",".join(parts)
        else:
            delete_key(key)


    def clean_expandable_segments_value(value):
        if value is None or "expandable_segments" not in value:
            return value
        parts = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part.startswith("expandable_segments:"):
                continue
            parts.append(part)
        return ",".join(parts) if len(parts) else None


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

    # IMPORTANT: run ROCm cleanup before importing device_type (which imports torch).
    # HIP allocator settings can be read during torch initialization.
    if IS_TORCH_ROCM_BUILD:
        remove_expandable_segments("PYTORCH_CUDA_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_HIP_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_ALLOC_CONF")
        delete_key("PYTORCH_CUDA_ALLOC_CONF")
        delete_key("PYTORCH_HIP_ALLOC_CONF")

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
    IS_HIP_RUNTIME = (DEVICE_TYPE == "hip") or bool(is_hip())

    # Torch >= 2.9 uses PYTORCH_ALLOC_CONF and treats legacy per-backend vars as deprecated.
    if IS_TORCH_2_9_OR_NEWER:
        # Preserve explicit legacy allocator settings when user did not directly set PYTORCH_ALLOC_CONF.
        if not _HAS_ORIGINAL_PYTORCH_ALLOC_CONF:
            promoted = _ORIGINAL_PYTORCH_CUDA_ALLOC_CONF
            if promoted is None:
                promoted = _ORIGINAL_PYTORCH_HIP_ALLOC_CONF
            # Keep standby + ROCm protections when promoting legacy values.
            if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "1" or IS_TORCH_ROCM_BUILD:
                promoted = clean_expandable_segments_value(promoted)
            if promoted is not None:
                os.environ["PYTORCH_ALLOC_CONF"] = promoted
        delete_key("PYTORCH_CUDA_ALLOC_CONF")
        delete_key("PYTORCH_HIP_ALLOC_CONF")

    # Specify PYTORCH_CUDA_ALLOC_CONF or PYTORCH_HIP_ALLOC_CONF
    if IS_HIP_RUNTIME:
        if IS_TORCH_2_9_OR_NEWER:
            # PyTorch >= 2.9 uses PYTORCH_ALLOC_CONF. expandable_segments is unsupported on HIP.
            remove_expandable_segments("PYTORCH_ALLOC_CONF")
            delete_key("PYTORCH_CUDA_ALLOC_CONF")
            delete_key("PYTORCH_HIP_ALLOC_CONF")
        else:
            if "PYTORCH_HIP_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
                os.environ["PYTORCH_HIP_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
                delete_key("PYTORCH_CUDA_ALLOC_CONF")
            if "PYTORCH_HIP_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" in os.environ:
                os.environ["PYTORCH_HIP_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
                delete_key("PYTORCH_ALLOC_CONF")
            # expandable_segments is not supported on ROCm/HIP
            remove_expandable_segments("PYTORCH_HIP_ALLOC_CONF")
            remove_expandable_segments("PYTORCH_ALLOC_CONF")
            delete_key("PYTORCH_CUDA_ALLOC_CONF")
    elif DEVICE_TYPE == "cuda" and not IS_HIP_RUNTIME and not IS_TORCH_2_9_OR_NEWER:
        delete_key("PYTORCH_HIP_ALLOC_CONF")
        delete_key("PYTORCH_ALLOC_CONF")

    # CCE fails on Torch 2.8 and above
    # OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536. Reducing block sizes or `num_stages`
    if (major_torch >= 2 and minor_torch >= 8) or (major_torch > 2):
        os.environ["UNSLOTH_ENABLE_CCE"] = "0"
    elif DEVICE_TYPE == "hip":
        # CCE also fails in HIP / AMD
        os.environ["UNSLOTH_ENABLE_CCE"] = "0"
    del remove_expandable_segments, delete_key, IS_HIP_RUNTIME, IS_TORCH_2_9_OR_NEWER, IS_TORCH_ROCM_BUILD, major_torch, minor_torch, torch_version, torch_version_raw, importlib_version, find_spec
    del clean_expandable_segments_value
    del _ORIGINAL_PYTORCH_CUDA_ALLOC_CONF, _ORIGINAL_PYTORCH_HIP_ALLOC_CONF, _HAS_ORIGINAL_PYTORCH_ALLOC_CONF

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

    # Fused lm_head + cross_entropy auto-installer. On by default; set
    # UNSLOTH_FUSED_FORWARD=0 to disable.
    try:
        from .fused_losses.forward_install import install_modeling_import_hook as _install_fused_forward
        _install_fused_forward()
        del _install_fused_forward
    except Exception:
        pass
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
