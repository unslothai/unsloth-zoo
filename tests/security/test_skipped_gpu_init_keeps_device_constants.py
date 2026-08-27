# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_ZOO_DISABLE_GPU_INIT=1` must not remove the device constants.

The flag skips the heavy torch/device init. It is what CI sets to run this very
suite, because the runner has no GPU and the init ends in
`find_spec("unsloth") is None -> ImportError`.

But modules that are otherwise import-safe read those constants at module scope -
`compiler.py` does `from . import DEVICE_TYPE` - so skipping the init used to turn
any such import into `ImportError: cannot import name 'DEVICE_TYPE'`. The whole
security job then failed during COLLECTION, which means it was not running at all
while still reporting a result.

CPU-only and network-free: this reads source and runs a subprocess with the flag set.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_INIT = _ROOT / "unsloth_zoo" / "__init__.py"

# Bound by the MLX branch and by the skip branch alike, and by the real init below
# both. A module reading any of them must not care which path ran.
_CONSTANTS = ("DEVICE_TYPE", "DEVICE_TYPE_TORCH", "DEVICE_COUNT", "ALLOW_PREQUANTIZED_MODELS")


def _child(code: str):
    environment = dict(os.environ)
    environment["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    # `tests/conftest.py` sets `UNSLOTH_ALLOW_CPU=1` for the whole run, and inheriting
    # it here would have let that unrelated escape hatch stand in for the skip path
    # this test is about: the promise is that `UNSLOTH_ZOO_DISABLE_GPU_INIT=1` alone
    # keeps the module importable, so the child is asked exactly that question.
    environment.pop("UNSLOTH_ALLOW_CPU", None)
    # A subprocess, not an import: this package is already imported in-process with
    # the flag unset, so an in-process check would prove nothing about the flag.
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd = _ROOT, env = environment, capture_output = True, text = True, timeout = 300,
    )


@pytest.mark.parametrize("name", _CONSTANTS)
def test_the_constant_survives_a_skipped_init(name):
    result = _child(f"import unsloth_zoo; print(unsloth_zoo.{name})")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip(), f"{name} is bound but empty"


def test_a_module_that_reads_them_still_imports():
    """`compiler.py` is the one CI actually tripped over."""
    result = _child("import unsloth_zoo.compiler")
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_skip_branch_binds_every_constant_the_mlx_branch_does():
    """Read off the source, so a constant added to one branch cannot be forgotten.

    The two branches answer the same question - "the heavy init did not run" - and a
    reader of these names cannot tell which one it got.
    """
    tree = ast.parse(_INIT.read_text(encoding = "utf-8"))

    bound = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            target.id
            for statement in node.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        if any(constant in names for constant in _CONSTANTS):
            bound.append(names)

    assert len(bound) >= 2, (
        "expected both the MLX branch and the skipped-init branch to bind the device "
        f"constants, found {len(bound)} branch(es) that bind any of them"
    )
    for names in bound:
        missing = sorted(set(_CONSTANTS) - names)
        assert not missing, f"a branch binds only part of the device constants: {missing}"
