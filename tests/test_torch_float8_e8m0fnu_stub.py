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

"""torch.float8_e8m0fnu must be stubbed before transformers imports finegrained_fp8.

Studio users on torch < 2.7 hit:
    module 'torch' has no attribute 'float8_e8m0fnu'
during `import unsloth` even when they are not training UE8M0 FP8 models.
"""

import ast
import types
from pathlib import Path

import pytest

UTILS = (Path(__file__).resolve().parents[1] / "unsloth_zoo"
         / "temporary_patches" / "utils.py")


def _run_ensure(torch_mod):
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_torch_float8_e8m0fnu":
            ns = {"torch": torch_mod}
            exec(compile(ast.Module([node], []), "<utils>", "exec"), ns)
            ns["_ensure_torch_float8_e8m0fnu"]()
            return
    raise AssertionError("_ensure_torch_float8_e8m0fnu not found in utils.py")


def test_stub_aliases_e4m3_when_e8m0_missing():
    torch_mod = types.SimpleNamespace(float8_e4m3fn = object())
    _run_ensure(torch_mod)
    assert torch_mod.float8_e8m0fnu is torch_mod.float8_e4m3fn


def test_stub_is_a_noop_when_e8m0_already_exists():
    sentinel = object()
    torch_mod = types.SimpleNamespace(float8_e8m0fnu = sentinel, float8_e4m3fn = object())
    _run_ensure(torch_mod)
    assert torch_mod.float8_e8m0fnu is sentinel


def test_stub_leaves_torch_alone_when_no_fp8_dtypes_exist():
    torch_mod = types.SimpleNamespace()
    _run_ensure(torch_mod)
    assert not hasattr(torch_mod, "float8_e8m0fnu")


def test_the_helper_runs_before_the_unpack_import():
    src = UTILS.read_text(encoding="utf-8")
    stub = src.index("_ensure_torch_float8_e8m0fnu()")
    unpack = src.index("from transformers.processing_utils import Unpack")
    assert stub < unpack, "the stub must run before transformers is imported"


def test_attribute_error_branch_names_e8m0():
    src = UTILS.read_text(encoding="utf-8")
    assert '"float8_e8m0fnu" in e_str and "has no attribute" in e_str' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
