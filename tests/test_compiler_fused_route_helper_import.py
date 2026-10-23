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

"""The zoo MoE routers call _fused_route_or_none on every forward (None unless
UNSLOTH_MOE_FUSED_ROUTER=1). The compiler copies those forwards into a generated module
that does not carry the patch module's globals, so the emitted module must import the
helper itself. Without it gpt-oss training raised NameError on the first step."""

import ast
import inspect
import pathlib
import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo import compiler
from unsloth_zoo import temporary_patches

HELPER = "_fused_route_or_none"
GPT_OSS_LOCATION = "transformers.models.gpt_oss.modeling_gpt_oss"
PATCH_DIR = pathlib.Path(temporary_patches.__file__).parent


@pytest.fixture
def compile_location(tmp_path, monkeypatch):
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    return tmp_path


def _functions_calling_helper():
    """(file, qualname, source) of every function in temporary_patches that calls the helper."""
    found = []
    for path in sorted(PATCH_DIR.glob("*.py")):
        text = path.read_text()
        if HELPER not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name == HELPER:
                continue
            calls = any(
                isinstance(n, ast.Call) and getattr(n.func, "id", None) == HELPER
                for n in ast.walk(node)
            )
            if calls:
                node.decorator_list = []
                found.append((path.stem, node.name, ast.unparse(node)))
    return found


CALLERS = _functions_calling_helper()


def test_helper_callers_exist():
    # gpt_oss, qwen3_moe, qwen3_vl_moe, moe_grouped_modulelist at the time of writing.
    assert {stem for stem, _, _ in CALLERS} >= {"gpt_oss", "qwen3_moe", "moe_grouped_modulelist"}


@pytest.mark.parametrize(
    ("stem", "fn_name", "source"),
    CALLERS,
    ids = [f"{stem}.{name}.{i}" for i, (stem, name, _) in enumerate(CALLERS)],
)
def test_emitted_copy_resolves_helper(compile_location, stem, fn_name, source):
    name = f"UnslothFusedRouteProbe_{stem}_{fn_name}_{abs(hash(source)) % 10**8}"
    module = compiler.create_new_function(name, source, GPT_OSS_LOCATION, [], overwrite = True)
    helper = getattr(module, HELPER, None)
    assert callable(helper), f"{stem}.{fn_name} copied without {HELPER}"
    # Unset flag: the helper declines, so every router keeps its eager path.
    assert helper(torch.randn(4, 8), 2, "softmax_topk", True) is None


def test_compiled_gpt_oss_router_forward_runs(compile_location, monkeypatch):
    """The real case: GptOssTopKRouter.forward copied into the compiled gpt_oss module."""
    modeling = pytest.importorskip(GPT_OSS_LOCATION)
    from unsloth_zoo.temporary_patches import gpt_oss

    # What patch_gpt_oss_compiler_exports puts on the modeling module for the copied source.
    monkeypatch.setattr(modeling, "transformers_version", gpt_oss.transformers_version, raising = False)
    monkeypatch.setattr(modeling, "Version", gpt_oss.Version, raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_FUSED_ROUTER", raising = False)

    source = textwrap.dedent(inspect.getsource(gpt_oss.GptOssTopKRouter.forward))
    assert HELPER in source
    module = compiler.create_new_function(
        "UnslothFusedRouteProbe_gpt_oss_router", source, GPT_OSS_LOCATION, dir(modeling), overwrite = True,
    )
    config = SimpleNamespace(
        num_experts_per_tok = 2, num_local_experts = 4, hidden_size = 8, torch_dtype = torch.float32, dtype = torch.float32,
    )
    router = gpt_oss.GptOssTopKRouter(config)
    torch.nn.init.normal_(router.linear.weight)
    hidden = torch.randn(3, 8)
    out = module.forward(router, hidden)
    expected = router.forward(hidden)
    for a, b in zip(out, expected):
        torch.testing.assert_close(a, b)
