# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""`UNSLOTH_ALLOW_CPU=1` has to survive the import on a driverless host.

A CUDA-built torch with no usable device -- a driverless container, a CI runner,
a laptop with the runtime and no card -- is what that variable exists for.
`get_device_type()` deliberately keeps `DEVICE_TYPE` at `"cuda"` there, so a
module-scope `torch.cuda.get_device_capability()` runs with nothing to query and
raises `RuntimeError: No CUDA GPUs are available` out of `_lazy_init()`.

`compiler` and `loss_utils` are both on the path `import unsloth` walks, so a
regression in either takes the whole import down with it.

Module import is process-global and one-shot, so each case runs in a fresh
interpreter with `CUDA_VISIBLE_DEVICES=""`.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_NO_DEVICE = "No CUDA GPUs are available"

# Every unsloth_zoo module reached at import time that reads a device capability
# at module scope. `vllm_utils` is deliberately absent: its probes sit behind
# `fast_inference = True` and are not import-time.
_MODULES = ("unsloth_zoo", "unsloth_zoo.compiler", "unsloth_zoo.loss_utils", "unsloth_zoo.patching_utils")


def _needs_cuda_build():
    """Only a CUDA-built torch reaches the branch under test. On ROCm or a
    CPU-only build `DEVICE_TYPE` is not `"cuda"` and there is nothing to prove."""
    if getattr(torch.version, "hip", None):
        pytest.skip("ROCm build: DEVICE_TYPE is not the cuda branch this covers")
    if not getattr(torch.version, "cuda", None):
        pytest.skip("torch is not built against CUDA, so no device can go missing")


def _run(code, **env):
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    # `conftest.py` sets UNSLOTH_ALLOW_CPU=1 for the whole session, so each case
    # has to say for itself whether the child gets it.
    clean = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ALLOW_CPU"}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = dict(
            clean,
            PYTHONPATH = os.pathsep.join(path),
            CUDA_VISIBLE_DEVICES = "",
            **env,
        ),
        timeout = 900,
    )


def test_the_devices_really_are_hidden():
    """If a device is visible here every other case in this file is vacuous."""
    _needs_cuda_build()
    out = _run("import torch; print('AVAILABLE', torch.cuda.is_available())")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "AVAILABLE False" in out.stdout, out.stdout


@pytest.mark.parametrize("module", _MODULES)
def test_the_module_imports_on_a_driverless_host(module):
    _needs_cuda_build()
    out = _run(f"import {module}; print('IMPORT_OK')", UNSLOTH_ALLOW_CPU = "1")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "IMPORT_OK" in out.stdout, out.stdout
    assert _NO_DEVICE not in out.stderr, out.stderr[-2000:]


def test_a_host_with_no_device_claims_no_capability():
    """Where a capability cannot be read, take the conservative answer. Cut cross
    entropy is a Triton GPU kernel, and the old-arch compile workarounds only ever
    run on a real GPU, so False is the answer that also changes nothing."""
    _needs_cuda_build()
    out = _run(
        "from unsloth_zoo.device_type import DEVICE_TYPE\n"
        "from unsloth_zoo.compiler import OLD_CUDA_ARCH_VERSION\n"
        "from unsloth_zoo.loss_utils import HAS_CUT_CROSS_ENTROPY\n"
        "print('DEVICE_TYPE', DEVICE_TYPE)\n"
        "print('OLD_CUDA_ARCH_VERSION', OLD_CUDA_ARCH_VERSION)\n"
        "print('HAS_CUT_CROSS_ENTROPY', HAS_CUT_CROSS_ENTROPY)\n",
        UNSLOTH_ALLOW_CPU = "1",
    )
    assert out.returncode == 0, out.stderr[-3000:]
    # The variable is what keeps DEVICE_TYPE on the cuda branch; without that
    # there is no branch to guard and this file is testing nothing.
    assert "DEVICE_TYPE cuda" in out.stdout, out.stdout
    assert "OLD_CUDA_ARCH_VERSION False" in out.stdout, out.stdout
    assert "HAS_CUT_CROSS_ENTROPY False" in out.stdout, out.stdout


def test_without_the_variable_the_import_still_refuses():
    """The documented refusal is not what this fix relaxes."""
    _needs_cuda_build()
    out = _run("import unsloth_zoo.compiler")
    assert out.returncode != 0, out.stdout
    assert "You need a GPU" in out.stderr, out.stderr[-2000:]
