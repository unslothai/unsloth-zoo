# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Every "needs real MLX" gate in the suite must reject the torch shim.

Several test modules call ``simulate_mlx_on_torch()`` while being IMPORTED, and
pytest's collection imports every module in the session. So by the time a module
that wants REAL mlx is imported, ``import mlx`` may already resolve to the shim,
and a gate that only asks whether the import worked reads the shim as real. The
tests behind it then run against a stub that implements neither ``mx.checkpoint``
nor a vjp over integer arrays, and fail.

Which module is imported first is alphabetical serially -- where no installer
happens to come first -- and worker assignment under ``-n N --dist loadfile``,
which is how unsloth's Core job runs this suite. That is why the same tests pass
locally and fail in CI: not flakiness, an ordering the local run never produces.

The fix in each module is to gate on ``mlx_is_simulated()`` rather than on the
import. This file is the standing check that no module loses that gate again, and
that a new one does not arrive without it. Each case runs in a SUBPROCESS with the
shim installed first, because the point is what a module decides at import time and
this session has already made those decisions.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent

# module -> the module-level flag holding "a REAL mlx is importable". Every module
# whose tests need real MLX semantics is here; the flag names differ because the
# modules were written apart.
REAL_MLX_GATES = {
    "test_mlx_gated_delta_vjp": "_HAS_REAL_MLX",
    "test_mlx_gated_delta_batch_grad": "_HAS_MLX",
    "test_mlx_neftune_quant_map": "_MLX",
}

_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {tests_dir!r})

    # Exactly what a sibling module does to this process during collection.
    from mlx_simulation import simulate_mlx_on_torch, mlx_is_simulated
    simulate_mlx_on_torch()
    assert mlx_is_simulated(), "the shim did not install; the probe proves nothing"

    import importlib
    module = importlib.import_module({module!r})
    print("GATE:", bool(getattr(module, {flag!r})))
    """
)


@pytest.mark.parametrize("module, flag", sorted(REAL_MLX_GATES.items()))
def test_a_real_mlx_gate_is_false_under_the_shim(module, flag):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(tests_dir=str(TESTS_DIR), module=module, flag=flag)],
        capture_output=True,
        text=True,
        cwd=str(TESTS_DIR.parent),
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{module} did not import with the MLX shim installed:\n{result.stderr}"
    )
    gate = [ln for ln in result.stdout.splitlines() if ln.startswith("GATE:")]
    assert gate, f"probe printed no verdict for {module}:\n{result.stdout}\n{result.stderr}"
    assert gate[-1] == "GATE: False", (
        f"{module}.{flag} is True while `mlx` in sys.modules is the torch shim, so the "
        f"cases it guards will run against the shim whenever another module installed it "
        f"first. Gate on mlx_simulation.mlx_is_simulated(), not on the import succeeding."
    )


def _module_level_names(source: str) -> set:
    """Names assigned at module level, including inside a module-level try/if.

    Both shapes are in use: one module assigns its gate at top level, another inside
    `try: import mlx.core`. Walking the whole tree would also count names bound inside
    functions, which is not what "module-level flag" means.
    """
    tree = ast.parse(source)
    names, queue = set(), list(tree.body)
    while queue:
        node = queue.pop()
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign,)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Try, ast.If)):
            queue.extend(node.body + node.orelse + getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                queue.extend(handler.body)
    return names


def test_the_gate_names_still_exist():
    """Non-vacuity: a renamed flag would make getattr raise, but a DELETED module or a
    gate that stopped being module-level would quietly leave this file testing nothing."""
    for module, flag in REAL_MLX_GATES.items():
        path = TESTS_DIR / f"{module}.py"
        assert path.exists(), f"{module} is gone; update REAL_MLX_GATES."
        assert flag in _module_level_names(path.read_text()), (
            f"{module} no longer defines a module-level {flag}; update REAL_MLX_GATES so "
            f"this file keeps guarding a gate that exists."
        )
