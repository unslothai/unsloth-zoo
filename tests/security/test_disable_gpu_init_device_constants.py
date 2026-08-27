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

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INIT_PY = REPO_ROOT / "unsloth_zoo" / "__init__.py"

_NAMES = ("DEVICE_TYPE", "DEVICE_TYPE_TORCH", "DEVICE_COUNT", "ALLOW_PREQUANTIZED_MODELS")


def _public_upper_names(body) -> set:
    """Module-level UPPER_CASE names bound by a branch of `__init__.py`.

    Descends into nested `if`s, because those still bind at module level when
    the branch is taken, but not into `def`/`class`, whose locals do not.
    """
    found = set()
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    found.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name.isupper():
                    found.add(name)
        stack.extend(ast.iter_child_nodes(node))
    return {name for name in found if not name.startswith("_")}


def _cross_path_contract() -> set:
    """Names that BOTH the MLX branch and the normal path bind at module level.

    A name bound on only one path is that path's private business:
    `UNSLOTH_ZOO_IS_PRESENT` is MLX-only (the normal path sets just the
    environment variable), `IS_HIP_RUNTIME` and the torch-version flags are
    normal-only. Neither kind is read off the package root anywhere. The
    intersection is the set every path genuinely promises, and it is therefore
    the set the skip path has to keep reachable.
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


def test_the_skip_path_covers_every_name_the_other_two_paths_promise():
    """The contract is DERIVED from `__init__.py`, not restated here.

    The bug this file exists for was a hardcoded assumption about which names
    the skip path needs, made in a workflow comment in #1089 and falsified
    three days later by an ordinary new test in #1108. Hardcoding the same
    assumption in the resolver would leave the identical trap: add a fifth
    constant to the MLX branch and the normal path, and the skip path silently
    lacks it again with nothing to notice.

    So this reads the three branches out of the source. If the cross-path
    contract grows, this fails until the resolver is extended to match.
    """
    contract = _cross_path_contract()
    assert contract == set(_NAMES), (
        f"the set of names bound by BOTH the MLX branch and the normal path is "
        f"{sorted(contract)}, but this suite checks {sorted(_NAMES)}. Extend the "
        f"lazy resolver in unsloth_zoo/__init__.py and _NAMES here to match, or "
        f"the skip path is short a name that every other path defines."
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
