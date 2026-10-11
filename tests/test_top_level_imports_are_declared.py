# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Every third-party package a zoo module imports at module level is a declared dependency.

`import unsloth_zoo` imports these modules eagerly, so an undeclared one fails at import time on any
install that happens not to pull it in transitively. `requests` was exactly that: llama_cpp,
rl_environments and vision_utils import it at the top, and it only ever arrived through datasets,
which dropped it in 5.x. The version-compat `latest` lane hit `No module named 'requests'`.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]

# Import name -> distribution name, where they differ.
DISTRIBUTION = {"PIL": "pillow"}

# Imported at module level but deliberately not declared, with the declared dependency that
# guarantees it. Each one is a hard requirement of that dependency, not an accident of a resolve.
GUARANTEED_BY = {"safetensors": "transformers"}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding = "utf-8"))["project"]
    return {_normalise(re.split(r"[<>=!~ ;\[]", spec, maxsplit = 1)[0]) for spec in project["dependencies"]}


def _module_level_imports() -> dict[str, list[str]]:
    # sys.stdlib_module_names is 3.10+; on the 3.9 floor this check is left to the newer lanes.
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        pytest.skip("needs sys.stdlib_module_names (Python 3.10+)")
    found: dict[str, list[str]] = {}
    for path in sorted((ROOT / "unsloth_zoo").glob("*.py")):
        for node in ast.parse(path.read_text(encoding = "utf-8")).body:
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in stdlib or name == "unsloth_zoo":
                    continue
                found.setdefault(name, []).append(path.name)
    return found


def test_module_level_imports_are_declared_dependencies():
    declared = _declared()
    imports = _module_level_imports()
    assert {"torch", "transformers", "requests"} <= set(imports), (
        "the scan found fewer module-level imports than the zoo has; it would pass on nothing"
    )
    missing = {
        name: sorted(set(files))
        for name, files in imports.items()
        if _normalise(DISTRIBUTION.get(name, name)) not in declared
        and _normalise(GUARANTEED_BY.get(name, "")) not in declared
    }
    assert not missing, (
        f"imported at module level but not in [project].dependencies: {missing}. Declare them, "
        "or move the import inside the function that needs it."
    )


def test_every_guarantor_is_itself_declared():
    declared = _declared()
    assert all(_normalise(dep) in declared for dep in GUARANTEED_BY.values())
