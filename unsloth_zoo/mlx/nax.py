# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Apple GPU neural accelerators (NAX) and the MLX dispatch gaps unsloth routes around."""

import functools
import json
import logging
import mmap
import os
import platform
import re
import subprocess
import sys
from threading import Lock
from typing import NamedTuple

import mlx.core as mx


logger = logging.getLogger(__name__)


# MLX's `is_nax_available`, which Python cannot call: macOS 26.2 and a generation-17 GPU,
# 18 for the `p` class.
_MIN_MACOS = (26, 2)
_GPU_ARCHITECTURE_PATTERN = re.compile(r"applegpu_g(\d+)([a-z])")


def _gpu_architecture():
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    except RuntimeError:
        return ""
    return str(info.get("architecture", ""))


def _gpu_generation():
    match = _GPU_ARCHITECTURE_PATTERN.fullmatch(_gpu_architecture())
    return int(match.group(1)) if match else None


@functools.cache
def _gpu_core_count():
    """The IORegistry `gpu-core-count`, which `mx.device_info()` does not report; None if unreadable."""
    try:
        result = subprocess.run(["/usr/sbin/ioreg", "-rc", "AGXAccelerator", "-d", "1", "-k", "gpu-core-count"],
                                capture_output = True, text = True, timeout = 5)
    except Exception:
        return None
    match = re.search(r'"gpu-core-count" = (\d+)', result.stdout) if result.returncode == 0 else None
    return int(match.group(1)) if match else None


@functools.cache
def _nax_gpu():
    if platform.system() != "Darwin" or not mx.metal.is_available():
        return False
    try:
        release = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
    except ValueError:
        return False
    match = _GPU_ARCHITECTURE_PATTERN.fullmatch(_gpu_architecture())
    if match is None or release < _MIN_MACOS:
        return False
    return int(match.group(1)) >= (18 if match.group(2) == "p" else 17)


@functools.cache
def _stock_nax_kernels():
    # A wheel built for macOS below 26.2 compiles MLX_METAL_NO_NAX and ships none.
    path = os.path.join(os.path.dirname(mx.__file__), "lib", "mlx.metallib")
    try:
        with open(path, "rb") as file, mmap.mmap(file.fileno(), 0, access = mmap.ACCESS_READ) as data:
            return data.find(b"_nax_") >= 0
    except (OSError, ValueError):
        return False


def nax_available():
    """Whether stock MLX dispatches its NAX kernels here. `UNSLOTH_MLX_NAX=0` turns every route off."""
    return os.environ.get("UNSLOTH_MLX_NAX", "1") != "0" and _nax_gpu() and _stock_nax_kernels()


class Gap(NamedTuple):
    open_in: tuple  # MLX releases measured to leave the gap open
    closed_on_main: bool


# A release missing from `open_in` keeps a gap only while MLX main has not closed it, so both
# the first release with the upstream fix and older unmeasured releases turn the route off.
_GAPS = {}


def gap_open(name):
    gap = _GAPS[name]
    return mx.__version__ in gap.open_in or not gap.closed_on_main


_PROBE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "unsloth", "mlx_nax_probes.json")
_PROBE_TIMEOUT = 60
_PROBES = {}
_PROBE_LOCK = Lock()


@functools.cache
def _os_build():
    try:
        return subprocess.run(["sysctl", "-n", "kern.osversion"], capture_output = True,
                              text = True, timeout = 5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _stored_probes():
    try:
        with open(_PROBE_PATH) as file:
            stored = json.load(file)
        return stored if isinstance(stored, dict) else {}
    except (OSError, ValueError):
        return {}


def _store_probe(entry, passed):
    stored = _stored_probes()
    stored[entry] = passed
    try:
        os.makedirs(os.path.dirname(_PROBE_PATH), exist_ok = True)
        temporary = f"{_PROBE_PATH}.{os.getpid()}"
        with open(temporary, "w") as file:
            json.dump(stored, file)
        os.replace(temporary, _PROBE_PATH)
    except OSError:
        pass


def _run_probe(module, function):
    code = f"import importlib; getattr(importlib.import_module({module!r}), {function!r})()"
    env = dict(os.environ, PYTHONPATH = os.pathsep.join(path for path in sys.path if path))
    try:
        result = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True,
                                text = True, timeout = _PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    except OSError as error:
        return f"{type(error).__name__}: {error}"
    if result.returncode == 0:
        return ""
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return lines[-1] if lines else f"exit code {result.returncode}"


def kernel_probe_passed(key: str, module: str, function: str) -> bool:
    """Run `module.function()` once in a subprocess; cache pass/fail per (macOS build, MLX version, key).

    A Metal build failure inside `mx.fast.metal_kernel` can abort the process, so each NAX kernel is
    first built and checked here, where a failure only disables it.
    """
    try:
        if not nax_available():
            return False
        entry = f"{platform.mac_ver()[0]}|{_os_build()}|{mx.__version__}|{key}"
        with _PROBE_LOCK:
            if entry not in _PROBES:
                passed = _stored_probes().get(entry)
                if not isinstance(passed, bool):
                    failure = _run_probe(module, function)
                    passed = failure == ""
                    if failure is not None:  # a timeout is retried by the next process
                        _store_probe(entry, passed)
                    if not passed:
                        logger.warning("NAX kernel %s failed its probe (%s); the native path stays in use",
                                       key, failure or "timed out")
                _PROBES[entry] = passed
            return _PROBES[entry]
    except Exception as error:
        logger.warning("NAX kernel %s could not be probed (%s); the native path stays in use", key, error)
        return False
