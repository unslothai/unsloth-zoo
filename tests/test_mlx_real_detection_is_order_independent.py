# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Asking "is mlx importable" is not asking "is this real mlx".

``simulate_mlx_on_torch`` installs process-wide: it puts ``mlx`` in sys.modules and
a finder in sys.meta_path. One test module calls it while being IMPORTED, and
collection imports every module in the session, so from that point on
``importlib.util.find_spec("mlx")`` returns a spec for every module collected
afterwards -- whether or not it asked for the shim.

A module that gates real-MLX tests on that spec therefore decides what to run based
on COLLECTION ORDER. That is what happened: test_mlx_gated_delta_vjp.py read
``find_spec("mlx") is not None`` as "on real MLX", and when a new sibling sorting
before it (test_mlx_arrays_cache_advance.py, #1127) started installing the shim at
import, five `requires_real_mlx` tests stopped skipping and ran against a shim where
``mx.argpartition`` is a ``_Noop``. Alone the file passed; in the suite it failed.
Nothing was wrong with the new test, and nothing was wrong with the code under test.

``mlx_is_simulated()`` is the question that has an order-independent answer, and
these pin both halves: that the trap is real, and that no test module falls into it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent


def _find_spec_mlx_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``find_spec("mlx")`` call in the module, however it is spelled."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "find_spec" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "mlx":
            calls.append(node)
    return calls


def _is_install_guard(tree: ast.AST, call: ast.Call) -> bool:
    """True for ``if find_spec("mlx") is None: simulate_mlx_on_torch()``.

    That spelling is fine and is not what breaks: it only decides whether to INSTALL
    the shim, and installing when one is already installed is a no-op. The dangerous
    spelling is the negation, ``is not None``, which is read as "real mlx is here".
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare)
                and test.left is call
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Is)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None):
            return True
    return False


def _test_modules() -> list[Path]:
    return sorted(p for p in _TESTS.glob("test_*.py"))


@pytest.mark.parametrize("path", _test_modules(), ids=lambda p: p.name)
def test_real_mlx_is_not_decided_by_find_spec_alone(path: Path):
    source = path.read_text(encoding="utf-8")
    if "find_spec" not in source:
        return
    tree = ast.parse(source)
    calls = _find_spec_mlx_calls(tree)
    if not calls:
        return
    gating = [c for c in calls if not _is_install_guard(tree, c)]
    if not gating:
        return
    assert "mlx_is_simulated" in source, (
        f"{path.name} consults find_spec('mlx') for something other than the "
        f"`if ... is None:` install guard (line(s) "
        f"{', '.join(str(c.lineno) for c in gating)}), but never calls "
        f"mlx_is_simulated(). A spec exists as soon as ANY sibling module has "
        f"installed the torch shim, so this file's real-MLX decision depends on "
        f"collection order. Use:\n"
        f"    _HAS_REAL_MLX = (importlib.util.find_spec('mlx') is not None\n"
        f"                     and not mlx_is_simulated())"
    )


def test_the_shim_makes_find_spec_lie():
    """The trap itself, so the rule above keeps its reason when someone reads it.

    If this ever fails because find_spec stops seeing the shim, the guard is
    obsolete rather than merely unsatisfied, and should be reconsidered, not
    silenced.
    """
    import importlib.util

    from mlx_simulation import mlx_is_simulated, simulate_mlx_on_torch

    simulate_mlx_on_torch()
    assert importlib.util.find_spec("mlx") is not None, (
        "the shim no longer registers mlx, so find_spec is no longer misleading"
    )
    assert mlx_is_simulated() is True, (
        "mlx_is_simulated() must see through the shim; it is the only "
        "order-independent answer available to a test module"
    )


def test_the_shim_does_not_implement_the_ops_the_gate_protects():
    """Why an order-dependent gate is a real failure and not a cosmetic one.

    These are exactly the symbols the five gated-delta VJP tests reached when the
    gate wrongly said "real MLX". A `_Noop` raises on CALL, not on attribute access,
    so nothing catches it until the test is already running.
    """
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()
    import mlx.core as mx

    for symbol in ("argpartition", "put_along_axis"):
        with pytest.raises(NotImplementedError, match="mlx-shim"):
            getattr(mx, symbol)()
