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

# Keeps PEP 604 annotations (`str | Path`) from being evaluated at def time, which
# is a TypeError on the 3.9 floor pyproject declares.
from __future__ import annotations

__version__ = "2026.10.10"

import os
import platform
import sys
import warnings
import re
# Stop TOKENIZERS_PARALLELISM warning
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Detect offline mode first: hf_transfer bypasses huggingface_hub's offline guard, so leaving it on defeats HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE.
_OFFLINE_TRUE = {"1", "true", "yes", "on"}
_offline_env = (
    os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_TRUE
    or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _OFFLINE_TRUE
    or os.environ.get("HF_DATASETS_OFFLINE", "").strip().lower() in _OFFLINE_TRUE
)

# hf_transfer cannot complete downloads on Windows on ARM (every fetch fails), so it is not enabled there; the same fetch succeeds once it is off.
def _detect_windows_on_arm() -> bool:
    if sys.platform != "win32":
        return False
    if platform.machine().lower() in ("arm64", "aarch64"):
        return True
    # Emulated x64 on ARM64 reports AMD64 on Python below 3.12, which reads only PROCESSOR_ARCHITECTURE/ARCHITEW6432;
    # IsWow64Process2 pNativeMachine names the real host.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        process, native = ctypes.c_ushort(), ctypes.c_ushort()
        if kernel32.IsWow64Process2(
            kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)
        ):
            return native.value == 0xAA64  # IMAGE_FILE_MACHINE_ARM64
    except Exception:
        pass  # pre-1709 Windows has no IsWow64Process2; fall back to "not ARM".
    return False


_windows_on_arm = _detect_windows_on_arm()

# Set hf_transfer only if importable: huggingface_hub below 1.0 refuses every download when the flag is set without the package, e.g. after a --no-deps install.
import importlib.util as _importlib_util


def _hf_transfer_importable() -> bool:
    try:
        return _importlib_util.find_spec("hf_transfer") is not None
    except (ImportError, ValueError):
        # A sys.modules stub without __spec__ makes find_spec raise; the probe must not fail on it, and a stub is not the real downloader.
        return False


_hf_transfer_installed = _hf_transfer_importable()

# Hugging Face Hub faster downloads (skipped when offline mode is requested).
if (
    "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
    and not _offline_env
    and not _windows_on_arm
    and _hf_transfer_installed
):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# More stable downloads
if os.environ.get("UNSLOTH_STABLE_DOWNLOADS", "0") == "1":
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30" # Default is 10 seconds
    os.environ["HF_HUB_DISABLE_XET"] = "1" # Disable XET
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0" # This causes "429 Too Many Requests"

# Setting any offline flag implies all three; HF_DATASETS_OFFLINE is needed because load_dataset() otherwise still hits the network for metadata.
if _offline_env:
    for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[_v] = "1"
del _OFFLINE_TRUE, _offline_env

# A 429 in hf_xet logs means the account, not the machine, is the bottleneck, so it lowers the stream ceiling rather than the memory caps.
from pathlib import Path
def has_429_exact_full_read(log_dir: str | Path) -> bool:
    log_dir = Path(log_dir).expanduser()
    if not log_dir.is_dir():
        return False
    for log_file in log_dir.glob("*.log"):
        try:
            if b"429 Too Many Requests" in log_file.read_bytes():
                return True
        except OSError:
            continue
    return False

# Redirect the HF cache off a read-only default so snapshot_download() can write; must run before any huggingface_hub import.
from .hf_cache import redirect_hf_cache_if_readonly, _active_caches
redirect_hf_cache_if_readonly()

# Size hf_xet's download buffers from THIS machine's RAM and cores. HF_XET_HIGH_PERFORMANCE=1 (the
# old default here) is an xet-core preset applied AFTER the environment is read: it raises the
# reconstruction buffer cap to 64GB and the stream count to 124 and overwrites any explicit
# HF_XET_RECONSTRUCTION_* cap, which is the source of the multi-GB RSS spikes. apply_xet_env()
# turns it off and writes RAM-derived caps instead, leaving user-set variables untouched.
# _active_caches mirrors Hub's env layering (XDG_CACHE_HOME included) and returns None entries
# instead of raising when home is unresolvable.
from .hf_xet_tuning import apply_xet_env
_, _, xet_cache = _active_caches()
apply_xet_env(throttled = has_429_exact_full_read(xet_cache / "logs") if xet_cache is not None else False)
del has_429_exact_full_read, xet_cache, redirect_hf_cache_if_readonly, _active_caches, apply_xet_env

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
    # Lots of debugging info, but it weirdly blocks torch.compile, so TRITON_ENABLE_LLVM_DEBUG is forced to 0.
    os.environ["TRITON_ENABLE_LLVM_DEBUG"] = "0"
    # TRITON_INTERPRET allows print statements but is slower and fails on bool types (get_int1_ty), so it is forced to 0.
    os.environ["TRITON_INTERPRET"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Blocking calls for debugging


from importlib.util import find_spec
from unsloth_zoo.mlx.runtime import is_mlx_available
from .model_lists import FORCE_FLOAT32

# Import-time fixes live in unsloth/import_fixes.py and run at import unsloth; zoo's GPU init needs find_spec for unsloth, so they are always in place.

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
    # Opt-in: a download-only helper child (e.g. hf_xet_fallback) sets
    # UNSLOTH_ZOO_DISABLE_GPU_INIT=1 to skip the heavy torch/transformers/device
    # init it never uses. Off by default, so normal CUDA/CPU runs are unchanged.
    # The HF cache redirect above still runs, so the child shares the parent's cache.
    _SKIP_GPU_INIT = os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT", "0") == "1"
    del _is_mlx_only, is_mlx_available
    if _SKIP_GPU_INIT:
        # `compiler.py` imports DEVICE_TYPE at module scope, so these constants must still be defined when GPU init is skipped.
        DEVICE_TYPE = "cpu"
        DEVICE_TYPE_TORCH = "cpu"
        DEVICE_COUNT = 0
        ALLOW_PREQUANTIZED_MODELS = False

_IS_CPU_BACKEND = False

# Stub CUDA-only imports when GPU init is skipped (MLX host or download child) so they resolve to a loud no-op, not a
# hard ImportError; CUDA/CPU keeps real modules.
if _SKIP_GPU_INIT:
    from unsloth_zoo.stubs.triton_stub import inject_into_sys_modules as _inject_triton
    _inject_triton()
    # bitsandbytes, unlike triton, ships a working arm64 macOS wheel, and shadowing a
    # real install makes bnb-quantized checkpoints unloadable. Locating it imports
    # nothing, so the download-only child can take this path too.
    from unsloth_zoo.stubs.bitsandbytes_stub import (
        inject_into_sys_modules as _inject_bnb,
        real_bitsandbytes_available as _real_bnb,
    )
    if not _real_bnb():
        _inject_bnb()
    del _inject_triton, _inject_bnb, _real_bnb

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
        # Dunder lookups raise AttributeError so torch walking sys.modules during init does not trigger an mlx import;
        # real attributes such as FastMLXModel still resolve.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            real = self._resolve()
        except ModuleNotFoundError:
            # mlx is Apple-only; raising AttributeError instead of ModuleNotFoundError lets sys.modules walkers such as
            # the torch._inductor FX hash pickler skip this stub.
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

    # Supports PyTorch 2; fixes unslothai/unsloth issue 38.
    from importlib.metadata import version as importlib_version
    torch_version_raw = str(importlib_version("torch"))
    torch_version = str(re.match(r"[0-9\.]{3,}", torch_version_raw).group(0)).split(".")
    major_torch, minor_torch = torch_version[0], torch_version[1]
    major_torch, minor_torch = int(major_torch), int(minor_torch)
    # Unified PYTORCH_ALLOC_CONF is only read from torch 2.10; <= 2.9.x reads the legacy vars.
    IS_TORCH_2_10_OR_NEWER = (major_torch > 2) or (major_torch == 2 and minor_torch >= 10)
    IS_TORCH_ROCM_BUILD = "+rocm" in torch_version_raw.lower()
    # expandable_segments is unsupported on Windows/WSL.
    def _is_wsl():
        if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
            return True
        # Docker on WSL2 lacks WSL_* env vars but shares the *-microsoft-standard-WSL2 kernel, where expandable_segments
        # dies with a CUDA driver error (unslothai/unsloth issue 3511).
        try:
            return "microsoft" in os.uname().release.lower()
        except Exception:
            return False
    IS_WSL_OR_WINDOWS = os.name == "nt" or _is_wsl()
    del _is_wsl

    # Also unsupported on NVIDIA Tegra boards, where the CUDA VMM calls fail even with memory free (unslothai/unsloth
    # issue 2401); detection is filesystem-only since torch is not yet imported; see unsloth_zoo/integrated_device.py.
    from .integrated_device import expandable_segments_unsupported
    EXPANDABLE_SEGMENTS_UNSUPPORTED = expandable_segments_unsupported()

    # Reduce VRAM fragmentation and optimize memory pinning
    if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "0" and not EXPANDABLE_SEGMENTS_UNSUPPORTED:
        if IS_TORCH_2_10_OR_NEWER:
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
    elif IS_WSL_OR_WINDOWS:
        # Strip only expandable_segments and keep other user config; a roundup fallback is applied below (unslothai/unsloth issue 7203).
        remove_expandable_segments("PYTORCH_CUDA_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_HIP_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_ALLOC_CONF")

    # Unified-memory boards get no roundup fallback, unlike WSL: there is no separate VRAM pool to defragment and
    # rounding up wastes shared RAM. Same ordering rule as ROCm: runs before torch import.
    if EXPANDABLE_SEGMENTS_UNSUPPORTED:
        remove_expandable_segments("PYTORCH_CUDA_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_HIP_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_ALLOC_CONF")

    # Run the ROCm cleanup before importing device_type (which imports torch), since HIP allocator settings can be read during torch init.
    if IS_TORCH_ROCM_BUILD:
        remove_expandable_segments("PYTORCH_CUDA_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_HIP_ALLOC_CONF")
        remove_expandable_segments("PYTORCH_ALLOC_CONF")
        delete_key("PYTORCH_CUDA_ALLOC_CONF")
        delete_key("PYTORCH_HIP_ALLOC_CONF")

    # Hides torchao's 'skipping cpp extensions' warning for incompatible torch/torchao versions (e.g. torch 2.7.0 with
    # torchao 0.14.1); see pytorch/ao issue 2919.
    import logging
    torchao_logger = logging.getLogger("torchao")
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
    # No accelerator: unsloth_zoo.cpu trains with stock torch, so none of the GPU patches below load.
    _IS_CPU_BACKEND = DEVICE_TYPE == "cpu"

    # Torch >= 2.10 reads PYTORCH_ALLOC_CONF and treats legacy per-backend vars as deprecated.
    if IS_TORCH_2_10_OR_NEWER:
        # Preserve explicit legacy allocator settings when user did not directly set PYTORCH_ALLOC_CONF.
        if not _HAS_ORIGINAL_PYTORCH_ALLOC_CONF:
            promoted = _ORIGINAL_PYTORCH_CUDA_ALLOC_CONF
            if promoted is None:
                promoted = _ORIGINAL_PYTORCH_HIP_ALLOC_CONF
            # Keep standby + ROCm + Tegra protections when promoting legacy values.
            if (
                os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "1"
                or IS_TORCH_ROCM_BUILD
                or EXPANDABLE_SEGMENTS_UNSUPPORTED
            ):
                promoted = clean_expandable_segments_value(promoted)
            if promoted is not None:
                os.environ["PYTORCH_ALLOC_CONF"] = promoted
        delete_key("PYTORCH_CUDA_ALLOC_CONF")
        delete_key("PYTORCH_HIP_ALLOC_CONF")

    # Specify PYTORCH_CUDA_ALLOC_CONF or PYTORCH_HIP_ALLOC_CONF
    if IS_HIP_RUNTIME:
        if IS_TORCH_2_10_OR_NEWER:
            # PyTorch >= 2.10 uses PYTORCH_ALLOC_CONF. expandable_segments is unsupported on HIP.
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
    elif DEVICE_TYPE == "cuda" and not IS_HIP_RUNTIME and not IS_TORCH_2_10_OR_NEWER:
        delete_key("PYTORCH_HIP_ALLOC_CONF")
        delete_key("PYTORCH_ALLOC_CONF")

    # Windows/WSL lack expandable_segments, so the branches above leave long context
    # training with no fragmentation mitigation (cudaMalloc retry storms / OOM). Give
    # NVIDIA CUDA a roundup_power2_divisions fallback instead (unslothai/unsloth#7203).
    if (
        IS_WSL_OR_WINDOWS
        and DEVICE_TYPE == "cuda" and not IS_HIP_RUNTIME
        and os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "0"
        and os.environ.get("UNSLOTH_DISABLE_ALLOC_FALLBACK", "0") == "0"
        and not (major_torch == 2 and minor_torch < 2)
    ):
        # torch <= 2.9.x reads the legacy var; >= 2.10 reads the unified one.
        _alloc_key = "PYTORCH_ALLOC_CONF" if IS_TORCH_2_10_OR_NEWER else "PYTORCH_CUDA_ALLOC_CONF"
        # Promotion above can re-add expandable_segments (only cleaned for standby/ROCm); strip it.
        remove_expandable_segments(_alloc_key)
        # Only set when absent; an explicit empty value opts out, as gradient_checkpointing.py advises with PYTORCH_CUDA_ALLOC_CONF set to empty.
        if _alloc_key not in os.environ:
            os.environ[_alloc_key] = "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
        del _alloc_key

    # CCE fails on torch 2.8 and above (OutOfResources: shared memory 98304 over the 65536 hardware limit), so UNSLOTH_ENABLE_CCE is disabled.
    if (major_torch >= 2 and minor_torch >= 8) or (major_torch > 2):
        os.environ["UNSLOTH_ENABLE_CCE"] = "0"
    elif DEVICE_TYPE == "hip":
        # CCE also fails in HIP / AMD
        os.environ["UNSLOTH_ENABLE_CCE"] = "0"

    # ROCm RDNA2/3/3.5/4: prefer rocBLAS (hipBLASLt can lack fallback Tensile kernels, so odd GEMMs JIT on first step);
    # DISABLE_ADDMM_HIP_LT set early since addmm reads it at dispatch and runtime setter is a no-op on Windows; NVIDIA,
    # Intel, Mac, CDNA untouched; UNSLOTH_ROCM_PREFER_ROCBLAS=0 disables.
    if IS_HIP_RUNTIME and os.environ.get(
        "UNSLOTH_ROCM_PREFER_ROCBLAS", "1"
    ).strip().lower() not in ("0", "off", "false", "no"):
        import sys as _sys  # the MLX-alias _sys was del'd above; re-import locally
        # Did the user pin a BLAS backend? If so, do not override it at runtime.
        _user_blas = "TORCH_BLAS_PREFER_HIPBLASLT" in os.environ or "TORCH_BLAS_PREFER_CUBLASLT" in os.environ
        # Did they pin it to the LT backend specifically? Then leave addmm on it too.
        _user_wants_lt = any(
            os.environ.get(_k, "").strip().lower() in ("1", "on", "true", "yes")
            for _k in ("TORCH_BLAS_PREFER_HIPBLASLT", "TORCH_BLAS_PREFER_CUBLASLT")
        )
        try:
            import torch as _torch
            _rocm_arch = str(getattr(
                _torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
            ).split(":")[0].strip()
        except Exception:
            _torch, _rocm_arch = None, ""
        # is_rdna set (RDNA2/3/3.5/4); CDNA (MI, gfx9xx) excluded on purpose.
        if _rocm_arch in (
            "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103",
            "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
        ):
            os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
            os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
            if not _user_wants_lt:
                os.environ.setdefault("DISABLE_ADDMM_HIP_LT", "1")
            # Non-Windows also flips the backend at runtime (a no-op on Windows); skipped when the user pinned a backend
            # so an explicit hipBLASLt choice is honored.
            if not _user_blas and _sys.platform != "win32" and _torch is not None:
                _pref = getattr(getattr(_torch.backends, "cuda", None), "preferred_blas_library", None)
                if callable(_pref):
                    try: _pref("cublas")  # prefer rocBLAS on ROCm
                    except Exception: pass  # best-effort; some builds lack the setter
                del _pref
        del _torch, _rocm_arch, _sys, _user_blas, _user_wants_lt
    del remove_expandable_segments, delete_key, IS_HIP_RUNTIME, IS_TORCH_2_10_OR_NEWER, IS_WSL_OR_WINDOWS, IS_TORCH_ROCM_BUILD, major_torch, minor_torch, torch_version, torch_version_raw, importlib_version, find_spec
    del clean_expandable_segments_value
    del expandable_segments_unsupported, EXPANDABLE_SEGMENTS_UNSUPPORTED
    del _ORIGINAL_PYTORCH_CUDA_ALLOC_CONF, _ORIGINAL_PYTORCH_HIP_ALLOC_CONF, _HAS_ORIGINAL_PYTORCH_ALLOC_CONF

    if not ("UNSLOTH_IS_PRESENT" in os.environ):
        raise ImportError("Please install Unsloth via `pip install unsloth`!")

    if not _IS_CPU_BACKEND:
        try:
            print("🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
        except:
            print("Unsloth: Will patch your computer to enable 2x faster free finetuning.")

    # Log Unsloth-Zoo Utilities
    os.environ["UNSLOTH_ZOO_IS_PRESENT"] = "1"

if not _SKIP_GPU_INIT and not _IS_CPU_BACKEND:
    from unsloth_zoo.temporary_patches import (
        encode_conversations_with_harmony,
    )

    # Fused lm_head + cross_entropy auto-installer. On by default; set
    # UNSLOTH_FUSED_FORWARD=0 to disable.
    try:
        from unsloth_zoo.fused_losses.forward_install import install_modeling_import_hook as _install_fused_forward
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
        # Silences pydantic UnsupportedFieldAttributeWarning: frozen=True on Field() has no effect inside type-statement aliases or union members.
        from pydantic.warnings import UnsupportedFieldAttributeWarning
        warnings.filterwarnings(action = "ignore", category = UnsupportedFieldAttributeWarning)
        del UnsupportedFieldAttributeWarning
    except:
        pass
elif not _SKIP_GPU_INIT and _IS_CPU_BACKEND:
    # Device-independent (stdlib + numpy): keep the root exports on the CPU backend too.
    from .rl_environments import (
        check_python_modules,
        create_locked_down_function,
        execute_with_time_limit,
        Benchmarker,
        is_port_open,
        launch_openenv,
    )

if not _SKIP_GPU_INIT:
    del os, warnings, re


# Device constants under UNSLOTH_ZOO_DISABLE_GPU_INIT. The MLX branch sets these
# four eagerly and the normal path imports them from `.device_type`; the skip
# branch did neither, so `from . import DEVICE_TYPE` (compiler.py) raised.
#
# Lazy, not a top-level import: `.device_type` costs ~1.4s and pulls in torch,
# and the download-only child the flag exists for never reads a constant.
#
# PEP 562: __getattr__ runs only when normal lookup fails, so the MLX and normal
# paths, where all four are real globals, are unaffected.
#
# No "cpu" fallback: compiler.py has cuda/hip/xpu arms only, so "cpu" would fall
# through all three. Driverless hosts opt in with UNSLOTH_ALLOW_CPU=1, which
# `get_device_type` honours by returning the "cuda" sentinel.
_LAZY_DEVICE_CONSTANTS = frozenset((
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "ALLOW_PREQUANTIZED_MODELS",
))


def __getattr__(name):
    if name not in _LAZY_DEVICE_CONSTANTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import device_type as _device_type
    value = getattr(_device_type, name)
    globals()[name] = value # resolve once; later lookups skip __getattr__
    return value
