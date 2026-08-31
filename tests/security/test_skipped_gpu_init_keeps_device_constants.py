# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_ZOO_DISABLE_GPU_INIT=1` must not remove the device constants.

CI sets the flag to run this suite, and `compiler.py` imports `DEVICE_TYPE` at module
scope, so the job failed during COLLECTION while still reporting a result.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[2]

# MLX answers before the skip flag is read, so the CPU expectations do not hold there.
try:
    from unsloth_zoo.mlx.runtime import is_mlx_available as _is_mlx_available
    _IS_MLX_HOST = bool(_is_mlx_available())
except Exception:
    _IS_MLX_HOST = False

# The security lane installs `.[core]` and not `unsloth`, so the full init raises.
_HAS_UNSLOTH = importlib.util.find_spec("unsloth") is not None

# Clearing `CUDA_VISIBLE_DEVICES` hides CUDA but not XPU, answered before the hatch.
def _has_non_cuda_accelerator():
    try:
        import torch
    except Exception:
        return False
    return bool(getattr(getattr(torch, "xpu", None), "is_available", bool)())


_HAS_NON_CUDA_ACCELERATOR = _has_non_cuda_accelerator()
_INIT = _ROOT / "unsloth_zoo" / "__init__.py"

_CONSTANTS = ("DEVICE_TYPE", "DEVICE_TYPE_TORCH", "DEVICE_COUNT", "ALLOW_PREQUANTIZED_MODELS")


def _child(code: str):
    environment = dict(os.environ)
    environment["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    # conftest sets this run-wide; inheriting it would stand in for the skip path.
    environment.pop("UNSLOTH_ALLOW_CPU", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd = _ROOT, env = environment, capture_output = True, text = True, timeout = 300,
    )


@pytest.mark.parametrize("name", _CONSTANTS)
def test_the_constant_survives_a_skipped_init(name):
    result = _child(f"import unsloth_zoo; print(unsloth_zoo.{name})")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip(), f"{name} is bound but empty"


@pytest.mark.skipif(_IS_MLX_HOST, reason = "the MLX install ships no torch to import")
def test_a_module_that_reads_them_still_imports():
    """`compiler.py` is the one CI actually tripped over."""
    result = _child("import unsloth_zoo.compiler")
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.skipif(_IS_MLX_HOST, reason = "MLX takes precedence over the skip flag")
def test_the_two_device_type_spellings_agree_on_the_skip_path():
    """One process must not hold two answers: two modules would branch differently."""
    result = _child(
        "import unsloth_zoo, unsloth_zoo.device_type as d;"
        "print(unsloth_zoo.DEVICE_TYPE, d.DEVICE_TYPE, d.DEVICE_TYPE_TORCH)"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    package, direct, torch_name = result.stdout.strip().splitlines()[-1].split()
    assert package == direct == torch_name == "cpu", (package, direct, torch_name)


@pytest.mark.skipif(_IS_MLX_HOST, reason = "MLX takes precedence over the skip flag")
@pytest.mark.parametrize("name", _CONSTANTS)
def test_both_import_paths_publish_the_same_constant(name):
    """`DEVICE_COUNT` and `ALLOW_PREQUANTIZED_MODELS` disagreed too: 0/False vs 1/True."""
    result = _child(
        "import unsloth_zoo, unsloth_zoo.device_type as d;"
        f"print(unsloth_zoo.{name}, d.{name})"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    package, direct = result.stdout.strip().splitlines()[-1].split()
    assert package == direct, (name, package, direct)


@pytest.mark.skipif(_IS_MLX_HOST, reason = "MLX takes precedence over the skip flag")
@pytest.mark.skipif(_IS_MLX_HOST, reason = "the MLX install ships no torch to import")
def test_the_compiler_binds_its_old_arch_flag_on_the_skip_path():
    """`fuse_lm_head` reads `OLD_CUDA_ARCH_VERSION`, which "cpu" left unbound."""
    result = _child(
        "import unsloth_zoo.compiler as c; print(c.OLD_CUDA_ARCH_VERSION)"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.skipif(
    not _HAS_UNSLOTH,
    reason = "this one turns the skip flag off, so the init needs `unsloth` installed",
)
@pytest.mark.skipif(_IS_MLX_HOST, reason = "MLX answers before the CPU hatch is read")
@pytest.mark.skipif(
    _HAS_NON_CUDA_ACCELERATOR,
    reason = "clearing CUDA_VISIBLE_DEVICES leaves another accelerator visible",
)
def test_the_legacy_cpu_hatch_still_answers_cuda(monkeypatch):
    """`UNSLOTH_ALLOW_CPU=1` keeps its own meaning, unchanged by the skip flag."""
    environment = dict(os.environ)
    environment.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
    environment["UNSLOTH_ALLOW_CPU"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "import unsloth_zoo.device_type as d; print(d.DEVICE_TYPE)"],
        cwd = _ROOT, env = environment, capture_output = True, text = True, timeout = 300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "cuda"


def test_the_skip_branch_binds_every_constant_the_mlx_branch_does():
    """Read off the source, so a constant added to one branch cannot be forgotten."""
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
