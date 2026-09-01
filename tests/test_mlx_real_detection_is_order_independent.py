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


def _in_a_shimmed_process(body: str):
    """Run ``body`` in a fresh interpreter that has the shim installed.

    Installing it HERE would be wrong on a machine that has real MLX. The shim is
    process-wide and permanent -- it displaces `mlx` in sys.modules for every test
    that follows, and mlx-vlm registers an atexit handler calling mx.clear_streams,
    which then lands on a `_Noop` and prints an ignored exception at interpreter
    shutdown. On Linux CI nothing notices, because there the shim is all there is.
    A subprocess is what keeps these checks honest on Apple Silicon.
    """
    import subprocess
    import sys as _sys

    script = (
        f"import sys\n"
        f"sys.path.insert(0, {str(_TESTS)!r})\n"
        f"from mlx_simulation import mlx_is_simulated, simulate_mlx_on_torch\n"
        f"simulate_mlx_on_torch()\n"
    ) + body
    result = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(_TESTS.parent), timeout=300,
    )
    assert result.returncode == 0, f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    return result.stdout


def test_the_shim_makes_find_spec_lie():
    """The trap itself, so the rule above keeps its reason when someone reads it.

    If this ever fails because find_spec stops seeing the shim, the guard is
    obsolete rather than merely unsatisfied, and should be reconsidered, not
    silenced.
    """
    _in_a_shimmed_process(
        "import importlib.util\n"
        "assert importlib.util.find_spec('mlx') is not None, (\n"
        "    'the shim no longer registers mlx, so find_spec is no longer misleading')\n"
        "assert mlx_is_simulated() is True, (\n"
        "    'mlx_is_simulated() must see through the shim; it is the only '\n"
        "    'order-independent answer available to a test module')\n"
    )


def test_the_shim_does_not_implement_the_ops_the_gate_protects():
    """Why an order-dependent gate is a real failure and not a cosmetic one.

    These are exactly the symbols the five gated-delta VJP tests reached when the
    gate wrongly said "real MLX". A `_Noop` raises on CALL, not on attribute access,
    so nothing catches it until the test is already running.
    """
    _in_a_shimmed_process(
        "import mlx.core as mx\n"
        "for symbol in ('argpartition', 'put_along_axis'):\n"
        "    try:\n"
        "        getattr(mx, symbol)()\n"
        "    except NotImplementedError as error:\n"
        "        assert 'mlx-shim' in str(error), error\n"
        "    else:\n"
        "        raise AssertionError(\n"
        "            f'mx.{symbol} no longer raises under the shim, so the gate it '\n"
        "            f'protects would fail later and less legibly')\n"
    )


# --------------------------------------------------------------------------- #
# The same rule, asked of the VALUE rather than of the source.
#
# Everything above is syntactic: it proves a module consults mlx_is_simulated(),
# not that it consults it correctly. `find_spec("mlx") is not None OR not
# mlx_is_simulated()` satisfies the rule and is still True under the shim, and a
# gate computed before the module installs its own shim would pass too. So each
# gate is also EVALUATED, in a subprocess with the shim already up, which is the
# state a sibling module leaves behind during collection.
# --------------------------------------------------------------------------- #

def _gate_assignments(tree: ast.AST) -> set:
    """Module-level names whose value is derived from a real-mlx probe.

    Three spellings are in use: find_spec("mlx"), mlx_is_simulated(), and reading
    "mlx_simulation" out of mx.__file__. Anything built from one of those is a gate,
    whatever it is named -- the suite currently uses _HAS_REAL_MLX, _HAS_MLX and _MLX.
    """
    def _is_probe(value) -> bool:
        for node in ast.walk(value):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "mlx_is_simulated":
                    return True
                if name == "find_spec" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value == "mlx":
                        return True
            if isinstance(node, ast.Constant) and node.value == "mlx_simulation":
                return True
        return False

    names, queue = set(), list(tree.body)
    while queue:
        node = queue.pop()
        if isinstance(node, ast.Assign) and _is_probe(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Try, ast.If)):
            queue.extend(node.body + node.orelse + getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                queue.extend(handler.body)
    return names


_EVAL_PROBE = """
import sys
sys.path.insert(0, {tests_dir!r})
from mlx_simulation import simulate_mlx_on_torch, mlx_is_simulated
simulate_mlx_on_torch()
assert mlx_is_simulated(), "shim did not install; this probe would prove nothing"
import importlib
module = importlib.import_module({module!r})
for name in {names!r}:
    print("GATE", name, bool(getattr(module, name)))
"""


def _modules_with_gates():
    for path in _test_modules():
        if path.name == Path(__file__).name:
            continue
        names = _gate_assignments(ast.parse(path.read_text(encoding="utf-8")))
        if names:
            yield path, sorted(names)


@pytest.mark.parametrize("path, names", list(_modules_with_gates()), ids=lambda a: getattr(a, "name", ""))
def test_a_gate_evaluates_to_false_under_the_shim(path: Path, names: list):
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [_sys.executable, "-c",
         _EVAL_PROBE.format(tests_dir=str(_TESTS), module=path.stem, names=names)],
        capture_output=True, text=True, cwd=str(_TESTS.parent), timeout=300,
    )
    assert result.returncode == 0, (
        f"{path.name} does not import with the MLX shim installed, which is the state "
        f"any sibling leaves behind during collection:\n{result.stderr[-2000:]}"
    )
    verdicts = dict(
        (line.split()[1], line.split()[2] == "True")
        for line in result.stdout.splitlines() if line.startswith("GATE ")
    )
    assert verdicts, f"probe printed nothing for {path.name}:\n{result.stdout}\n{result.stderr}"
    true_gates = [name for name, value in verdicts.items() if value]
    assert not true_gates, (
        f"{path.name} has {', '.join(true_gates)} True while `mlx` in sys.modules is the "
        f"torch shim. The source mentions the right helper, so the syntactic rule above is "
        f"satisfied, but the expression still reads the shim as real -- an `or` where an "
        f"`and` belongs, or a probe evaluated before this module installs its own shim. "
        f"Tests behind that gate will run against a stub whose ops raise on CALL."
    )
