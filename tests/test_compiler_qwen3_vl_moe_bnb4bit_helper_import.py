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

"""Emitted Qwen3-VL-MoE modules must import _bnb4bit_experts_forward_or_none; without it every expert call raised NameError, 16-bit experts included."""

import ast
import pathlib

import pytest
import torch

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches import qwen3_vl_moe

PROBE_NAME = "UnslothQwen3VLMoeHelperImportProbe"
HELPER = "_bnb4bit_experts_forward_or_none"
HELPER_IMPORT = f"from unsloth_zoo.temporary_patches.qwen3_vl_moe import {HELPER}\n"
QWEN3_VL_MOE_LOCATION = "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"


def _patched_expert_forward_source(dispatcher):
    """Source of the forward in patch_qwen3_vl_moe whose last statement returns dispatcher(...)."""
    tree = ast.parse(pathlib.Path(qwen3_vl_moe.__file__).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "forward"
            and isinstance(node.body[-1], ast.Return)
            and isinstance(node.body[-1].value, ast.Call)
            and getattr(node.body[-1].value.func, "id", None) == dispatcher
        ):
            node.decorator_list = []
            return ast.unparse(node)
    raise AssertionError(f"no patched forward returning {dispatcher}(...)")


@pytest.fixture
def compile_location(tmp_path, monkeypatch):
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    return tmp_path


@pytest.mark.parametrize(
    ("source", "expect_import"),
    [
        pytest.param(
            f"def _probe(self, x):\n    return {HELPER}(self, x, None, None)\n",
            True,
            id = "references-helper",
        ),
        pytest.param("def _probe(self, x):\n    return x\n", False, id = "no-helper-ref"),
    ],
)
def test_emitted_module_imports_helper_iff_referenced(compile_location, source, expect_import):
    compiler.create_new_function(PROBE_NAME, source, QWEN3_VL_MOE_LOCATION, [], overwrite = True)
    emitted = (compile_location / f"{PROBE_NAME}.py").read_text()
    assert (HELPER_IMPORT in emitted.split("def _probe")[0]) is expect_import


@pytest.mark.parametrize("dispatcher", ["forward_triton_grouped_gemm", "forward_native_moe_loop"])
def test_generated_qwen3_vl_moe_expert_forward_resolves_helper(compile_location, dispatcher):
    pytest.importorskip(QWEN3_VL_MOE_LOCATION)
    source = _patched_expert_forward_source(dispatcher)
    assert HELPER in source
    module = compiler.create_new_function(
        PROBE_NAME + "_" + dispatcher, source, QWEN3_VL_MOE_LOCATION, [], overwrite = True,
    )
    assert getattr(module, HELPER) is qwen3_vl_moe._bnb4bit_experts_forward_or_none
    # Stub the GEMM: only the generated module's name resolution is under test.
    setattr(module, dispatcher, lambda self, hidden_states, *args: hidden_states + 1)
    hidden_states = torch.zeros(2, 8)
    out = module.forward(
        torch.nn.Module(), hidden_states, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1),
    )
    # A module without packed bnb 4-bit experts gets None from the helper, so the dispatcher runs.
    torch.testing.assert_close(out, hidden_states + 1)
