# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real-MLX gates must ask mlx_is_simulated(), not find_spec('mlx'): the process-wide shim makes find_spec see mlx for
every module collected after one installs it; these tests pin the trap and check that no module falls into it."""

from __future__ import annotations

import ast
import re
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
    """Install guard 'if find_spec('mlx') is None' is safe since re-installing is a no-op; the negated 'is not None' spelling is the dangerous one, read as real mlx."""
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
    """Runs body in a subprocess, since the shim is process-wide and permanent and would displace real mlx on Apple
    Silicon; mlx-vlm's atexit mx.clear_streams then hits a _Noop."""
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
    """Pins the find_spec trap itself; if find_spec stops seeing the shim the guard is obsolete and should be reconsidered, not silenced."""
    _in_a_shimmed_process(
        "import importlib.util\n"
        "assert importlib.util.find_spec('mlx') is not None, (\n"
        "    'the shim no longer registers mlx, so find_spec is no longer misleading')\n"
        "assert mlx_is_simulated() is True, (\n"
        "    'mlx_is_simulated() must see through the shim; it is the only '\n"
        "    'order-independent answer available to a test module')\n"
    )


def test_the_shim_does_not_implement_the_ops_the_gate_protects():
    """An order-dependent gate is a real failure: these are the symbols the five gated-delta VJP tests reached, and _Noop raises on call, not on attribute access."""
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


# Source checks are syntactic, so each gate is also evaluated with the shim already up; 'find_spec('mlx') is not None or
# not mlx_is_simulated()' passes the source rule yet is still True under the shim.

def _gate_assignments(tree: ast.AST) -> set:
    """Module-level names built from a real-mlx probe (find_spec('mlx'), mlx_is_simulated(), or 'mlx_simulation' in
    mx.__file__) count as gates whatever they are named."""
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


# importorskip('mlx.core') passes on the shim too, so flag only modules calling an op the shim refuses (12 of the 14
# such modules are fine); the refused set is read off the shim.


# Matches importorskip of mlx or its submodules only; mlx_vlm is a different package, and a bare prefix match once read it as an mlx gate (issue 1329).
_MLX_IMPORTORSKIP = re.compile(r"""importorskip\(\s*["']mlx(?:["']|\.)""")


def _shim_refused_ops() -> set:
    """Public ``mx.*`` names whose shim body raises NotImplementedError."""
    stub = _TESTS / "mlx_simulation" / "mlx_stub.py"
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    refused = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Raise):
                continue
            exc = sub.exc
            name = (exc.func.id if isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name) else getattr(exc, "id", ""))
            if name == "NotImplementedError":
                refused.add(node.name)
    return refused


def _consults_the_simulation(source: str) -> bool:
    """Either mlx_is_simulated() or an open-coded mx.__file__ check answers the order-independent question; test_mlx_quantized_optimizer_state.py uses the latter."""
    return "mlx_is_simulated" in source or "mlx_simulation" in source


@pytest.mark.parametrize("path", _test_modules(), ids=lambda p: p.name)
def test_a_module_calling_an_op_the_shim_refuses_asks_whether_it_is_real(path: Path):
    source = path.read_text(encoding="utf-8")
    if not _MLX_IMPORTORSKIP.search(source):
        return
    called = sorted(op for op in _shim_refused_ops() if f"mx.{op}" in source)
    if not called or _consults_the_simulation(source):
        return
    pytest.fail(
        f"{path.name} gates on importorskip('mlx...') alone but calls "
        f"mx.{', mx.'.join(called)}, which the shim raises NotImplementedError "
        f"for. A sibling module installs the shim while being imported, so "
        f"collection order alone decides whether this file runs against real "
        f"MLX or against a stub that refuses the operation it is built on. Add:\n"
        f"    from mlx_simulation import mlx_is_simulated\n"
        f"    if mlx_is_simulated():\n"
        f"        pytest.skip(..., allow_module_level = True)"
    )


@pytest.mark.parametrize(
    "call, gates",
    [
        ('pytest.importorskip("mlx")', True),
        ('pytest.importorskip("mlx.core")', True),
        ("pytest.importorskip('mlx.nn')", True),
        ('pytest.importorskip( "mlx.core" )', True),
        ('pytest.importorskip("mlx_vlm.models.diffusion_gemma.config")', False),
        ('pytest.importorskip("mlx_lm")', False),
    ],
)
def test_only_an_importorskip_of_mlx_itself_counts_as_an_mlx_gate(call, gates):
    assert bool(_MLX_IMPORTORSKIP.search(call)) is gates
