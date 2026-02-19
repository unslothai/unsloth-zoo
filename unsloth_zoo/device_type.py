# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "is_hip",
    "get_device_type",
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "ALLOW_PREQUANTIZED_MODELS",
    "ALLOW_BITSANDBYTES",
    "device_synchronize",
]

import torch
import os
import functools
from .utils import Version
import inspect
import os
import re
import shutil
import subprocess
import urllib.request

_PYTORCH_WHL_BASE_URL = "https://download.pytorch.org/whl"

def _safe_run_command(command, timeout = 2.0):
    try:
        result = subprocess.run(
            command,
            capture_output = True,
            text = True,
            check = False,
            timeout = timeout,
        )
        return result.stdout or ""
    except Exception:
        return ""
pass

def _extract_major_minor(version_text):
    if not version_text:
        return None
    match = re.search(r"([0-9]+)\.([0-9]+)", str(version_text))
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"
pass

def _version_sort_key(version_text):
    parts = [int(x) for x in re.findall(r"[0-9]+", str(version_text))]
    if len(parts) < 2: parts = parts + [0]
    return tuple(parts)
pass

@functools.cache
def _pytorch_rocm_index_exists(rocm_index):
    index_url = f"{_PYTORCH_WHL_BASE_URL}/{rocm_index}/"
    # Some endpoints reject HEAD, so fallback to GET if needed.
    methods = ("HEAD", "GET")
    for method in methods:
        try:
            request = urllib.request.Request(
                index_url,
                headers = {"User-Agent" : "unsloth-zoo"},
                method = method,
            )
            with urllib.request.urlopen(request, timeout = 2.5) as response:
                if 200 <= getattr(response, "status", 200) < 400:
                    return True
        except Exception:
            pass
    return False
pass

@functools.cache
def _available_pytorch_rocm_indices():
    # Parse official wheel listing so we can suggest only valid ROCm endpoints.
    known_defaults = ["rocm7.1", "rocm7.0", "rocm6.4", "rocm6.3", "rocm6.2", "rocm6.1"]
    try:
        request = urllib.request.Request(
            f"{_PYTORCH_WHL_BASE_URL}/",
            headers = {"User-Agent" : "unsloth-zoo"},
        )
        with urllib.request.urlopen(request, timeout = 2.5) as response:
            html = response.read().decode("utf-8", errors = "ignore")
        matches = set(re.findall(r"rocm[0-9]+\.[0-9]+(?:\.[0-9]+)?", html))
        if matches:
            return sorted(matches, key = _version_sort_key, reverse = True)
    except Exception:
        pass
    return known_defaults
pass

def _nearest_rocm_index(detected_major_minor, available_indices):
    if not detected_major_minor:
        return None
    exact = f"rocm{detected_major_minor}"
    if exact in available_indices:
        return exact
    detected_major = detected_major_minor.split(".")[0]
    same_major = [x for x in available_indices if x.startswith(f"rocm{detected_major}.")]
    if same_major:
        return same_major[0]
    return None
pass

@functools.cache
def _detect_rocm_major_minor():
    # Preferred sources ordered from most direct to fallback.
    sources = []
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    if hip_version:
        sources.append(str(hip_version))
    for key in ("ROCM_VERSION", "ROCM_VERSION_FULL", "ROCM_VER"):
        value = os.environ.get(key, "")
        if value:
            sources.append(value)
    for filename in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            with open(filename, "r", encoding = "utf-8") as file:
                sources.append(file.read().strip())
        except Exception:
            pass
    if shutil.which("hipcc") is not None:
        sources.append(_safe_run_command(["hipcc", "--version"]))
    if shutil.which("rocm-smi") is not None:
        sources.append(_safe_run_command(["rocm-smi", "--showdriverversion"]))
    for source in sources:
        major_minor = _extract_major_minor(source)
        if major_minor is not None:
            return major_minor
    return None
pass

@functools.cache
def _detect_amd_rocm_runtime():
    # Fast path for Linux ROCm installs.
    if os.path.exists("/dev/kfd"):
        return True
    for env_key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if env_key in os.environ:
            return True
    if shutil.which("rocminfo") is not None:
        info = _safe_run_command(["rocminfo"])
        if ("gfx" in info.lower()) or ("amd" in info.lower()):
            return True
    if shutil.which("rocm-smi") is not None:
        info = _safe_run_command(["rocm-smi", "-i"])
        if ("gpu" in info.lower()) or ("amd" in info.lower()):
            return True
    return False
pass

@functools.cache
def _amd_installation_hint():
    if not _detect_amd_rocm_runtime():
        return None
    available_indices = _available_pytorch_rocm_indices()
    detected_major_minor = _detect_rocm_major_minor()
    chosen_index = _nearest_rocm_index(detected_major_minor, available_indices)
    if chosen_index is None:
        chosen_index = available_indices[0] if len(available_indices) else "rocm7.0"
    index_url = f"{_PYTORCH_WHL_BASE_URL}/{chosen_index}/"
    index_is_valid = _pytorch_rocm_index_exists(chosen_index)

    lines = [
        "Unsloth detected signs of an AMD ROCm GPU, but your current PyTorch build has no usable HIP accelerator.",
        "This usually means torch/torchvision/torchaudio were installed from default PyPI wheels instead of ROCm wheels.",
    ]
    if detected_major_minor is not None:
        lines.append(f"Detected ROCm version hint: {detected_major_minor}")
    else:
        lines.append("Could not determine ROCm version exactly; choosing the latest known ROCm wheel index.")
    lines.append("Try reinstalling PyTorch wheels with:")
    lines.append(
        f"uv pip install torch torchvision torchaudio --index-url {index_url} --upgrade --force-reinstall"
    )
    if index_is_valid:
        lines.append(f"Verified index URL is reachable: {index_url}")
    else:
        lines.append(
            "Could not verify index URL reachability from this environment; if needed, choose a ROCm index from https://pytorch.org/get-started/locally/"
        )
    return "\n".join(lines)
pass

@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))
pass

@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Check torch.accelerator
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            amd_hint = _amd_installation_hint()
            if amd_hint is not None:
                raise NotImplementedError(amd_hint)
            raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
        accelerator = str(torch.accelerator.current_accelerator())
        if accelerator in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                f"Unsloth: Weirdly `torch.cuda.is_available()`, `torch.xpu.is_available()` and `is_hip` all failed.\n"\
                f"But `torch.accelerator.current_accelerator()` works with it being = `{accelerator}`\n"\
                f"Please reinstall torch - it's most likely broken :("
            )
    amd_hint = _amd_installation_hint()
    if amd_hint is not None:
        raise NotImplementedError(amd_hint)
    raise NotImplementedError("Unsloth currently only works on NVIDIA, AMD and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()
# HIP fails for autocast and other torch functions. Use CUDA instead
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip": DEVICE_TYPE_TORCH = "cuda"

@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    else:
        return 1
pass

DEVICE_COUNT : int = get_device_count()

# Check blocksize for 4bit -> 64 for CUDA, 128 for AMD
# If AMD, we cannot load pre-quantized models for now :(
ALLOW_PREQUANTIZED_MODELS : bool = True
# HSA_STATUS_ERROR_EXCEPTION checks - sometimes AMD fails for BnB
ALLOW_BITSANDBYTES : bool = True
if DEVICE_TYPE == "hip":
    # Disable AITER by default on ROCm to avoid JIT build locks and runtime faults.
    # Users can override by explicitly setting env vars.
    os.environ.setdefault("AITER_DISABLE", "1")
    os.environ.setdefault("USE_ROCM_AITER_ROPE_BACKEND", "0")
    try:
        from bitsandbytes.nn.modules import Params4bit
        if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(Params4bit):
            ALLOW_PREQUANTIZED_MODELS = False
        import bitsandbytes
        ALLOW_BITSANDBYTES = Version(bitsandbytes.__version__) > Version("0.48.2.dev0")
    except:
        pass
pass

def device_synchronize():
    """
    Synchronize the current device (CUDA, XPU, or HIP).
    This is a cross-platform replacement for torch.cuda.synchronize().
    """
    if DEVICE_TYPE in ("cuda", "hip"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif DEVICE_TYPE == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
pass
