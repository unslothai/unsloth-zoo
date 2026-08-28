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
    "get_amd_attention_implementation",
    "get_amd_flash_attn_func",
    "get_device_type",
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "ALLOW_PREQUANTIZED_MODELS",
    "ALLOW_BITSANDBYTES",
    "device_synchronize",
    "device_empty_cache",
    "device_is_bf16_supported",
    "is_mlx_available",
    "get_recommended_attn_implementation",
]

import functools
from .utils import Version
from unsloth_zoo.mlx.runtime import is_mlx_available
import inspect
import os
import re
import shutil
import subprocess
import urllib.request

_IS_MLX = is_mlx_available()

if not _IS_MLX:
    import torch

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
    if _IS_MLX:
        return None
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
    if _IS_MLX:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))
pass


@functools.cache
def _detect_gfx_arch():
    """Return the GPU architecture string (e.g. 'gfx942') or None if undetectable.

    Uses the PyTorch-active device first so that HIP_VISIBLE_DEVICES and multi-arch
    hosts are handled correctly. Subprocess probes (rocminfo, hipconfig) scan
    system-wide output and may return a different arch than the active device.
    """
    import re as _re
    # Active device first: right arch for the GPU torch is using, and it
    # respects HIP_VISIBLE_DEVICES.
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            arch = torch.cuda.get_device_properties(dev).gcnArchName
            # gcnArchName may include suffixes e.g. "gfx942:sramecc+:xnack-"
            m = _re.match(r"gfx[0-9a-f]+", arch)
            if m:
                return m.group(0)
    except Exception:
        pass
    # Subprocess fallbacks for unusual builds where PyTorch cannot query the device.
    import subprocess
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        m = _re.search(r"gfx[0-9a-f]+", r.stdout)
        if m:
            return m.group(0)
    except Exception:
        pass
    try:
        r = subprocess.run(["hipconfig", "--full"], capture_output=True, text=True, timeout=10)
        m = _re.search(r"gfx[0-9a-f]+", r.stdout)
        if m:
            return m.group(0)
    except Exception:
        pass
    return None
pass


@functools.cache
def get_amd_attention_implementation():
    """
    Return the best available attention implementation for AMD ROCm.

    Priority:
    1. "amd_aiter": AMD aiter (pip install amd-aiter, ROCm >= 7.0 required)
    2. "sdpa": PyTorch SDPA via MIOpen (always available on ROCm)

    Returns: "amd_aiter" | "sdpa"
    """
    if not is_hip():
        return "sdpa"

    # Gate on ROCm >= 7.0 (amd-aiter has hard ABI dep on libamdhip64.so.7).
    # Prefer torch.version.hip: always present in ROCm wheels. rocm-smi
    # reports the kernel driver version, which can differ from the runtime.
    try:
        _hip_ver = getattr(torch.version, "hip", None)
        if _hip_ver is not None:
            _hip_major = int(str(_hip_ver).split(".")[0])
            if _hip_major < 7:
                return "sdpa"
        else:
            # Fallback for unusual builds without torch.version.hip
            rocm_version = _detect_rocm_major_minor()
            if rocm_version is not None:
                if int(rocm_version.split(".")[0]) < 7:
                    return "sdpa"
    except Exception:
        return "sdpa"

    # aiter's CK/ASM kernels are CDNA-only: gfx942 (MI300X/MI325X) and gfx950
    # (MI355X). flash_attn_func imports on all archs but only runs on these two.
    try:
        _gfx = _detect_gfx_arch()
        _AITER_SUPPORTED_ARCHS = ("gfx942", "gfx950")  # CDNA3, CDNA4
        if _gfx not in _AITER_SUPPORTED_ARCHS:
            return "sdpa"
    except Exception:
        return "sdpa"

    # Check for amd-aiter (AMD AI Tensor Engine for ROCm)
    try:
        import importlib.util  # must import submodule explicitly (not bare importlib)
        if importlib.util.find_spec("aiter") is not None:
            import aiter as _aiter
            # Validate: AMD AI tensor engine exposes the functional flash_attn_func API.
            # FlashAttnFunc (class API) requires 13+ positional args and cannot be
            # wrapped safely; only accept the simpler flash_attn_func functional API.
            if hasattr(_aiter, "flash_attn_func"):
                return "amd_aiter"
    except Exception:
        pass

    return "sdpa"


@functools.cache
def get_amd_flash_attn_func():
    """
    Return the amd-aiter flash attention function, or None if unavailable.

    Only the functional API `flash_attn_func` (aiter >= 0.7) is accepted. The
    class API `FlashAttnFunc` is deliberately not wrapped, so environments with
    only that one get None and fall back to SDPA.

    Call signature: func(q, k, v, causal=True)
    Shapes: q/k/v = (batch, seqlen, nheads, headdim), float16 or bfloat16
    Returns: output tensor, same shape as q
    """
    if get_amd_attention_implementation() != "amd_aiter":
        return None
    try:
        import aiter as _aiter
        if hasattr(_aiter, "flash_attn_func"):
            return _aiter.flash_attn_func
        # FlashAttnFunc is a torch.autograd.Function whose .apply() requires 13+
        # positional arguments (dropout_p, softmax_scale, causal, window_size,
        # bias, alibi_slopes, deterministic, return_lse, return_softmax,
        # is_grad_enabled, ...), see ROCm/aiter:aiter/ops/mha.py.
        # We cannot safely wrap it without knowing the required defaults for the
        # installed aiter version.  Return None so callers fall back to SDPA.
        # (FlashAttnFunc environments should expose flash_attn_func in aiter >= 0.7)
    except Exception:
        pass
    return None


def _cpu_fallback():
    """The device name for a host with no accelerator, or None.

    `UNSLOTH_ALLOW_CPU=1` answers "cuda" so callers keep their CUDA branches against a
    CPU torch. `UNSLOTH_ZOO_DISABLE_GPU_INIT=1` answers "cpu" to match `__init__.py`.
    """
    if os.environ.get("UNSLOTH_ALLOW_CPU", "0") == "1":
        return "cuda"
    if os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT", "0") == "1":
        return "cpu"
    return None


_GPU_INIT_SKIPPED = os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT", "0") == "1"


@functools.cache
def get_device_type():
    if _IS_MLX:
        return "mlx"
    if _GPU_INIT_SKIPPED:
        # BEFORE the hardware probes: `__init__.py` publishes "cpu" unconditionally, so
        # checking only in the no-accelerator fallback left a CUDA host with both.
        return "cpu"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            # Test-only CPU fallback; get_device_type is @functools.cache'd.
            fallback = _cpu_fallback()
            if fallback is not None:
                return fallback
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
    fallback = _cpu_fallback()
    if fallback is not None:
        return fallback
    amd_hint = _amd_installation_hint()
    if amd_hint is not None:
        raise NotImplementedError(amd_hint)
    raise NotImplementedError("Unsloth currently only works on NVIDIA, AMD and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()
# HIP fails for autocast and other torch functions. Use CUDA instead
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip": DEVICE_TYPE_TORCH = "cuda"
elif DEVICE_TYPE_TORCH == "mlx": DEVICE_TYPE_TORCH = "mps"

@functools.cache
def get_device_count():
    if _GPU_INIT_SKIPPED and DEVICE_TYPE == "cpu":
        # The getter too, not only the constant below: they were handed 1 and 0.
        return 0
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
if _GPU_INIT_SKIPPED and DEVICE_TYPE == "cpu":
    # As with DEVICE_COUNT: the two import paths must not disagree.
    ALLOW_PREQUANTIZED_MODELS = False
# HSA_STATUS_ERROR_EXCEPTION checks - sometimes AMD fails for BnB
ALLOW_BITSANDBYTES : bool = True
if DEVICE_TYPE == "hip":
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
    """Cross-platform torch.cuda.synchronize() (CUDA, XPU, or HIP)."""
    if DEVICE_TYPE in ("cuda", "hip"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif DEVICE_TYPE == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
pass

def device_empty_cache():
    """Cross-platform torch.cuda.empty_cache() (CUDA, XPU, or HIP)."""
    if DEVICE_TYPE in ("cuda", "hip"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif DEVICE_TYPE == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
pass

def device_is_bf16_supported():
    """Cross-platform torch.cuda.is_bf16_supported() (CUDA, XPU, or HIP)."""
    if DEVICE_TYPE in ("cuda", "hip"):
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
    elif DEVICE_TYPE == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if hasattr(torch.xpu, "is_bf16_supported"):
                return torch.xpu.is_bf16_supported()
    return False
pass


def get_recommended_attn_implementation():
    """
    Return "sdpa" on AMD ROCm, None elsewhere (no override, keep your default).

    Callers MUST check the resolved model class first. 43 causal LM
    architectures on transformers 4.57 (GptOss, Mamba, Bloom, GPT-J, MPT, ...)
    set `_supports_sdpa = False` and `from_config` raises ValueError for them:
    `AutoModelForCausalLM._model_mapping[type(config)]._supports_sdpa`.
    """
    if is_hip():
        return "sdpa"
    return None
pass
