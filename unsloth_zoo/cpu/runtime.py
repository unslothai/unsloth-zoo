# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "cpu_flags",
    "cpu_bf16_supported",
    "cpu_bnb_4bit_supported",
    "cpu_capabilities",
    "apply_cpu_security_patches",
]

import functools
import os
import platform
import warnings


@functools.cache
def cpu_flags():
    """Lower-cased ISA feature flags of the host CPU (empty when unreadable)."""
    flags = set()
    try:
        with open("/proc/cpuinfo", "r", encoding = "utf-8", errors = "ignore") as f:
            for line in f:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("flags", "features"):
                    flags.update(line.split(":", 1)[1].lower().split())
                    break
    except OSError:
        pass
    if not flags and platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output = True, text = True, timeout = 5,
            ).stdout
            flags.update(out.lower().split())
        except Exception:
            pass
    return frozenset(flags)


@functools.cache
def cpu_bf16_supported():
    """bfloat16 compute in hardware: AMX / AVX512-BF16 on x86, the BF16 extension on Arm.

    Without it torch emulates bf16 matmuls and bf16 training is several times slower than fp32,
    so this is the answer to "should CPU training default to bf16". UNSLOTH_CPU_BF16=0/1 overrides.
    """
    override = os.environ.get("UNSLOTH_CPU_BF16", "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override in ("1", "true", "yes", "on"):
        return True
    flags = cpu_flags()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "i686", "x86"):
        return bool(flags & {"amx_bf16", "avx512_bf16"})
    if machine in ("aarch64", "arm64"):
        return "bf16" in flags
    return False


@functools.cache
def cpu_bnb_4bit_supported():
    """A real NF4 Linear4bit forward + backward on CPU, not an import check: bitsandbytes resolves
    every native symbol to a raising stub when its library is dead, so only a call proves it."""
    if os.environ.get("UNSLOTH_CPU_LOAD_IN_4BIT", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import bitsandbytes as bnb
            dtype = torch.bfloat16 if cpu_bf16_supported() else torch.float32
            layer = bnb.nn.Linear4bit(128, 64, bias = False, compute_dtype = dtype, quant_type = "nf4")
            weight = layer.weight
            if not getattr(weight, "bnb_quantized", False):
                layer.weight = weight._quantize("cpu")
            x = torch.randn(4, 128, dtype = dtype, requires_grad = True)
            y = layer(x)
            y.float().sum().backward()
            return bool(torch.isfinite(y).all()) and x.grad is not None and bool(torch.isfinite(x.grad).all())
    except Exception:
        return False


def cpu_capabilities():
    import torch
    return {
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "bf16": cpu_bf16_supported(),
        "amx": "amx_bf16" in cpu_flags(),
        "bnb_4bit": cpu_bnb_4bit_supported(),
    }


def apply_cpu_security_patches():
    """Install the device-independent security guards from temporary_patches.

    That package also pulls the GPU kernels, so each guard file is loaded on its own with a private
    TEMPORARY_PATCHES list standing in for `.common`. Idempotent: the guards check their own flags.
    """
    import importlib.util
    import sys
    import types

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temporary_patches")
    applied = []
    for name in ("datasets_folder_metadata",):
        module_name = f"{__package__}._guard_{name}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(root, f"{name}.py"))
            if spec is None:
                continue
            common_name = f"{__package__}.common"
            common = types.ModuleType(common_name)
            common.TEMPORARY_PATCHES = []
            previous = sys.modules.get(common_name)
            sys.modules[common_name] = common
            try:
                module = importlib.util.module_from_spec(spec)
                module.__package__ = __package__
                spec.loader.exec_module(module)
            except Exception:
                continue
            finally:
                if previous is None:
                    sys.modules.pop(common_name, None)
                else:
                    sys.modules[common_name] = previous
            sys.modules[module_name] = module
            module._unsloth_cpu_patches = list(common.TEMPORARY_PATCHES)
        for patch in getattr(module, "_unsloth_cpu_patches", []):
            try:
                patch()
                applied.append(patch.__name__)
            except Exception:
                pass
    return applied
