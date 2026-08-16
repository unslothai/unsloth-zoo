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

_HELPER_NAMES = (
    "torch_supports_float8_e8m0fnu",
    "_temporary_float8_e8m0fnu_import_stub",
    "require_native_float8_e8m0fnu",
)


def _helper_namespace(torch_mod, *, include_require_native: bool = False):
    """Extract e8m0 helpers with the same module globals they expect at runtime."""
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    wanted = set(_HELPER_NAMES)
    if not include_require_native:
        wanted.discard("require_native_float8_e8m0fnu")
    ns: dict = {
        "torch": torch_mod,
        "contextlib": __import__("contextlib"),
        "_E8M0_IMPORT_STUB_ACTIVE": False,
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), "<utils>", "exec"), ns)
    missing = wanted - set(ns)
    assert not missing, f"helpers not found in utils.py: {missing}"
    return ns


def test_temporary_stub_aliases_e4m3_only_during_context():
    torch_mod = types.SimpleNamespace(float8_e4m3fn = object())
    stub = _helper_namespace(torch_mod)["_temporary_float8_e8m0fnu_import_stub"]
    with stub() as used:
        assert used is True
        assert torch_mod.float8_e8m0fnu is torch_mod.float8_e4m3fn
    assert not hasattr(torch_mod, "float8_e8m0fnu")


def test_temporary_stub_is_noop_when_e8m0_already_exists():
    sentinel = object()
    torch_mod = types.SimpleNamespace(float8_e8m0fnu = sentinel)
    stub = _helper_namespace(torch_mod)["_temporary_float8_e8m0fnu_import_stub"]
    with stub() as used:
        assert used is False
        assert torch_mod.float8_e8m0fnu is sentinel


def test_the_import_context_runs_before_unpack_import():
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    with_line = None
    unpack_line = None
    for node in tree.body:
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name):
                    if ctx.func.id == "_temporary_float8_e8m0fnu_import_stub":
                        with_line = node.lineno
        if isinstance(node, ast.With):
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module == "transformers.processing_utils":
                    for alias in child.names:
                        if alias.name == "Unpack":
                            unpack_line = child.lineno
    assert with_line is not None, "import context manager not found"
    assert unpack_line is not None, "Unpack import not found"
    assert with_line < unpack_line


def test_attribute_error_branch_names_e8m0():
    src = UTILS.read_text(encoding="utf-8")
    assert '"float8_e8m0fnu" in e_str and "has no attribute" in e_str' in src


def test_require_native_raises_when_e8m0_missing():
    pytest.importorskip("torch")
    import torch

    if hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("need torch without native e8m0 for this check")

    require_native = _helper_namespace(
        torch, include_require_native = True,
    )["require_native_float8_e8m0fnu"]
    with pytest.raises(RuntimeError, match="PyTorch >= 2.7"):
        require_native()


@pytest.mark.integration
def test_old_transformers_bind_inside_temporary_stub():
    """Mimic pinned transformers that bind _UE8M0_SF_DTYPE at import time."""
    pytest.importorskip("torch")
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks float8_e4m3fn")
    if hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("need torch without native e8m0 for this check")

    helpers = _helper_namespace(torch, include_require_native = True)
    stub = helpers["_temporary_float8_e8m0fnu_import_stub"]
    require_native = helpers["require_native_float8_e8m0fnu"]
    with stub():
        _UE8M0_SF_DTYPE = torch.float8_e8m0fnu  # noqa: N806
        assert _UE8M0_SF_DTYPE is torch.float8_e4m3fn
    assert not hasattr(torch, "float8_e8m0fnu")
    with pytest.raises(RuntimeError, match="PyTorch >= 2.7"):
        require_native()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
