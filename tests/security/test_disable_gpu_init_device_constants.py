# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_ZOO_DISABLE_GPU_INIT=1` must leave the package fully defined.

The security-audit lane sets that flag, and the skip branch used to define none
of the four device constants, so `from . import DEVICE_TYPE` in compiler.py
raised at import time and took a whole security module out at COLLECTION.

Two properties: the constants must be reachable, and reaching them must stay
lazy (`.device_type` costs ~1.4s and pulls in torch, and the download-only child
this flag exists for never reads one). Hence the torch checks below.

Everything runs in a subprocess, since both properties are about what happens
during `import unsloth_zoo` and this process already imported it.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INIT_PY = REPO_ROOT / "unsloth_zoo" / "__init__.py"

_NAMES = ("DEVICE_TYPE", "DEVICE_TYPE_TORCH", "DEVICE_COUNT", "ALLOW_PREQUANTIZED_MODELS")


def _mlx_branch_is_live() -> bool:
    """Mirror `is_mlx_available()`: importing the real predicate would drag in the package under test."""
    return (
        os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") != "1"
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    )


def _load_real_predicate():
    """`is_mlx_available` loaded by file path, so the package (and torch) stays out of it.

    `unsloth_zoo/mlx/runtime.py` imports stdlib only, so this is safe.
    """
    path = REPO_ROOT / "unsloth_zoo" / "mlx" / "runtime.py"
    spec = importlib.util.spec_from_file_location("_zoo_mlx_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_mlx_available


# MLX binds the constants eagerly, so PEP 562 __getattr__ never runs and "does not
# pull torch" would pass for the wrong reason. Skip the pair together, not vacuously green.
_needs_lazy_path = pytest.mark.skipif(
    _mlx_branch_is_live(),
    reason = "MLX branch defines the device constants eagerly; the lazy path is not taken",
)

# `unsloth_zoo.compiler` imports torch at module scope; a torch-less Apple Silicon install is supported.
_needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason = "torch is not installed; unsloth_zoo.compiler cannot import by design",
)


def _public_upper_names(body) -> set:
    """Module-level UPPER_CASE names bound by a branch of `__init__.py`.

    Descends into nested `if`s (still module-level bindings) but not `def`/`class`.

    Annotated assignment and unpacked targets count: `device_type.py` uses
    `DEVICE_TYPE : str = ...`, and a binding form we miss shrinks the
    intersection, letting the contract test pass while the skip path is short a
    name. A bare `X: int` with no value binds nothing.
    """
    found = set()
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        targets = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = (node.target,)
        for target in targets:
            for leaf in ast.walk(target):
                if isinstance(leaf, ast.Name) and leaf.id.isupper():
                    found.add(leaf.id)
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name.isupper():
                    found.add(name)
        stack.extend(ast.iter_child_nodes(node))
    return {name for name in found if not name.startswith("_")}


def _cross_path_contract() -> set:
    """Names that BOTH the MLX branch and the normal path bind at module level.

    A name bound on only one path (`UNSLOTH_ZOO_IS_PRESENT` is MLX-only,
    `IS_HIP_RUNTIME` normal-only) is that path's private business and is not read
    off the package root. The intersection is what every path promises, so it is
    what the skip path has to keep reachable.
    """
    tree = ast.parse(INIT_PY.read_text(encoding = "utf-8"))
    mlx, normal = set(), set()
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "_is_mlx_only" in test:
            mlx |= _public_upper_names(node.body)
        elif test == "not _SKIP_GPU_INIT":
            normal |= _public_upper_names(node.body)
    assert mlx, "could not find the MLX branch in __init__.py; this test needs rewriting"
    assert normal, "could not find the normal branch in __init__.py; this test needs rewriting"
    return mlx & normal


def _run(code: str, skip_gpu_init: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if skip_gpu_init:
        env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    else:
        env.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
    # CI runners have no accelerator; `get_device_type` honours this sentinel,
    # and `tests/conftest.py` sets it for the same reason.
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


@_needs_torch
def test_the_module_that_actually_broke_imports():
    """The concrete regression: compiler.py does `from . import DEVICE_TYPE`."""
    result = _run("import unsloth_zoo.compiler\nprint('ok')\n")
    assert result.returncode == 0, (
        "unsloth_zoo.compiler failed to import under "
        f"UNSLOTH_ZOO_DISABLE_GPU_INIT=1:\n{result.stderr}"
    )


@_needs_lazy_path
@_needs_torch
def test_importing_the_package_does_not_pull_in_torch():
    """The skip stays a skip: torch is not imported until a constant is read.

    Fails if the lazy resolver becomes a top-level `from .device_type import
    ...`, which would pass every test above.
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


# Without the flag the init raises when unsloth is absent, as in the security lane.
_needs_unsloth = pytest.mark.skipif(
    importlib.util.find_spec("unsloth") is None,
    reason = "unsloth is not installed; importing unsloth_zoo without the skip flag raises",
)


@_needs_lazy_path
@_needs_torch
@_needs_unsloth
def test_reading_a_constant_is_what_pulls_torch_in():
    """The non-vacuous half: without the flag, reading a constant does pull torch in."""
    result = _run(
        "import sys\n"
        "import unsloth_zoo\n"
        "unsloth_zoo.DEVICE_TYPE\n"
        "print('torch' in sys.modules)\n",
        skip_gpu_init = False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("True"), (
        "reading unsloth_zoo.DEVICE_TYPE did not import torch, so the previous "
        f"test proves nothing about laziness:\n{result.stdout}"
    )


@_needs_lazy_path
@_needs_torch
def test_reading_a_constant_under_the_skip_asks_torch_nothing():
    """And under the flag it does not: the skip answers `cpu` on its own."""
    result = _run(
        "import sys\n"
        "import unsloth_zoo\n"
        "print(unsloth_zoo.DEVICE_TYPE, unsloth_zoo.DEVICE_COUNT)\n"
        "print('torch' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-2].split() == ["cpu", "0"], result.stdout
    assert lines[-1] == "False", (
        "reading a device constant under UNSLOTH_ZOO_DISABLE_GPU_INIT=1 pulled torch "
        f"in, so the flag no longer skips the work it exists to skip:\n{result.stdout}"
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


def test_the_skip_path_covers_every_name_the_other_two_paths_promise():
    """The contract is DERIVED from `__init__.py`, not restated here.

    A hardcoded list of names is the trap that caused the original bug: add a
    fifth constant to the MLX branch and the normal path, and the skip path
    silently lacks it. Reading the branches from source fails loudly instead.
    """
    contract = _cross_path_contract()
    assert contract == set(_NAMES), (
        f"the set of names bound by BOTH the MLX branch and the normal path is "
        f"{sorted(contract)}, but this suite checks {sorted(_NAMES)}. Extend the "
        f"lazy resolver in unsloth_zoo/__init__.py and _NAMES here to match, or "
        f"the skip path is short a name that every other path defines."
    )


@pytest.mark.parametrize(
    "snippet",
    (
        "NEW_CONST = 1",                      # plain assignment
        "NEW_CONST : bool = True",            # the device_type.py idiom
        "NEW_CONST, OTHER = 1, 2",            # unpacked
        "if True:\n    NEW_CONST = 1",        # nested branch
        "from .device_type import NEW_CONST", # the normal path's shape
    ),
)
def test_the_contract_parser_sees_every_shape_a_constant_can_arrive_in(snippet):
    """The derivation is only as good as the binding forms it recognises.

    A missed form shrinks the intersection rather than growing it, so the
    contract test passes while the skip path is missing a name.
    """
    assert "NEW_CONST" in _public_upper_names(ast.parse(snippet).body), snippet


@pytest.mark.parametrize(
    "system, machine, has_mlx, force_gpu, expected",
    (
        ("Darwin", "arm64", True,  None,  True),   # the guarded branch
        ("Darwin", "arm64", False, None,  False),
        ("Darwin", "x86_64", True, None,  False),
        ("Linux",  "x86_64", True, None,  False),
        ("Windows", "AMD64", True, None,  False),
        ("Darwin", "arm64", True,  "1",   False),  # escape hatch wins
        ("Darwin", "arm64", True,  "0",   True),   # only when exactly "1"
    ),
)
def test_the_mlx_guard_agrees_with_is_mlx_available(
    monkeypatch, system, machine, has_mlx, force_gpu, expected,
):
    """The guard copies `is_mlx_available`, so check it against the real one, not just a table.

    A table alone would stay green if the production predicate gained a condition,
    while `_needs_lazy_path` silently skipped the wrong environments. Asserting both
    against `expected` catches drift on either side.
    """
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            object() if has_mlx else None
        ) if name == "mlx" else real_find_spec(name, *a, **k),
    )
    if force_gpu is None:
        monkeypatch.delenv("UNSLOTH_FORCE_GPU_PATH", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_FORCE_GPU_PATH", force_gpu)
    real = _load_real_predicate()
    real.cache_clear() # functools.cache: one stale answer would fix every arm below
    assert _mlx_branch_is_live() is expected
    assert real() is expected, (
        f"unsloth_zoo/mlx/runtime.py::is_mlx_available returned {real()!r} where the "
        f"copy in this file returned {expected!r}. The two have diverged, so "
        f"_needs_lazy_path now skips the wrong environments; update the copy."
    )


def test_every_contract_name_actually_resolves_under_the_skip():
    """Derivation is worth nothing if nothing checks it against a live import."""
    names = sorted(_cross_path_contract())
    code = (
        "import unsloth_zoo\n"
        f"for name in {names!r}:\n"
        "    getattr(unsloth_zoo, name)\n"
        "print('all resolved')\n"
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"one of {names} did not resolve under UNSLOTH_ZOO_DISABLE_GPU_INIT=1:\n"
        f"{result.stderr}"
    )
    assert result.stdout.strip().endswith("all resolved"), result.stdout
