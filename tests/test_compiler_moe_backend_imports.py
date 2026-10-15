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

import ast
import builtins
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_names(source, name):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - bound - set(dir(builtins))


def test_compiled_module_imports_every_name_forward_moe_backend_reads():
    # The compiler copies forward_moe_backend into unsloth_compiled_module_*.py, where only
    # the names in its injected `from moe_utils import (...)` block resolve.
    compiler = (ROOT / "unsloth_zoo" / "compiler.py").read_text(encoding = "utf-8")
    block = re.search(r'"    from moe_utils import \(\\n"(.*?)"    \)\\n"', compiler, re.S)
    assert block is not None
    imported = set(re.findall(r'"\s+(\w+),\\n"', block.group(1)))
    moe_utils = (ROOT / "unsloth_zoo" / "temporary_patches" / "moe_utils.py").read_text(encoding = "utf-8")
    missing = _free_names(moe_utils, "forward_moe_backend") - imported - {"torch"}
    assert not missing, f"compiled modules would raise NameError on {sorted(missing)}"
