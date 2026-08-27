# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_ZOO_DISABLE_GPU_INIT=1` must leave the package fully defined.

This lane sets that flag (see `.github/workflows/security-audit.yml`) so
`import unsloth_zoo` skips the device init that would otherwise demand
`unsloth` be installed in a supply-chain-hygiene job. The skip branch used to
define none of the four device constants that the MLX branch defines eagerly
and the normal path imports from `.device_type`, so `from unsloth_zoo import
DEVICE_TYPE` raised ImportError -- which `unsloth_zoo/compiler.py` does at
import time, taking `test_compile_model_type_sink_guard.py` out at COLLECTION.
A whole security test module stopped running and the lane reported it as one
red rather than as missing coverage.

Two properties, and the second is the one that rots quietly. The constants have
to be reachable, and reaching them has to stay lazy: resolving `.device_type`
costs ~1.4s and pulls torch into the process, and the download-only child that
this flag exists for never reads one. Someone "simplifying" the resolver into a
top-level import would satisfy the first test and silently undo the reason the
flag is there at all, so the torch check below is not decoration.

Everything runs in a subprocess: both properties are about what happens during
`import unsloth_zoo`, and this process has already imported it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

_NAMES = ("DEVICE_TYPE", "DEVICE_TYPE_TORCH", "DEVICE_COUNT", "ALLOW_PREQUANTIZED_MODELS")


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    # A CI runner has no accelerator; this is the sentinel `get_device_type`
    # already honours, and is what `tests/conftest.py` sets for the same reason.
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd = str(REPO_ROOT),
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )


@pytest.mark.parametrize("name", _NAMES)
def test_device_constant_is_importable_under_the_skip(name):
    """`from unsloth_zoo import <constant>` resolves rather than raising."""
    result = _run(f"from unsloth_zoo import {name}\nprint(repr({name}))\n")
    assert result.returncode == 0, (
        f"`from unsloth_zoo import {name}` failed under "
        f"UNSLOTH_ZOO_DISABLE_GPU_INIT=1:\n{result.stderr}"
    )


def test_the_module_that_actually_broke_imports():
    """The concrete regression: compiler.py does `from . import DEVICE_TYPE`."""
    result = _run("import unsloth_zoo.compiler\nprint('ok')\n")
    assert result.returncode == 0, (
        "unsloth_zoo.compiler failed to import under "
        f"UNSLOTH_ZOO_DISABLE_GPU_INIT=1:\n{result.stderr}"
    )


def test_importing_the_package_does_not_pull_in_torch():
    """The skip stays a skip: torch is not imported until a constant is read.

    Fails if the lazy resolver is ever replaced by a top-level
    `from .device_type import ...`, which would pass every test above.
    """
    result = _run(
        "import sys\n"
        "import unsloth_zoo\n"
        "print('torch' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False"), (
        "importing unsloth_zoo under UNSLOTH_ZOO_DISABLE_GPU_INIT=1 pulled torch "
        "into the process; the device constants are no longer resolved lazily and "
        "the flag no longer skips the work it exists to skip"
    )


def test_reading_a_constant_is_what_pulls_torch_in():
    """The other half of the pair, so the check above cannot pass vacuously."""
    result = _run(
        "import sys\n"
        "import unsloth_zoo\n"
        "unsloth_zoo.DEVICE_TYPE\n"
        "print('torch' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("True"), (
        "reading unsloth_zoo.DEVICE_TYPE did not import torch, so the previous "
        f"test proves nothing about laziness:\n{result.stdout}"
    )


def test_an_unknown_attribute_still_raises_attribute_error():
    """The resolver is a narrow shim, not a catch-all that hides typos."""
    result = _run(
        "import unsloth_zoo\n"
        "try:\n"
        "    unsloth_zoo.NOT_A_REAL_NAME\n"
        "except AttributeError:\n"
        "    print('raised')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("raised"), result.stdout
