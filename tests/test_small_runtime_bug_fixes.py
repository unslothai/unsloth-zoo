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

"""Three paths that raise instead of doing their job, checked from source so no GPU or vLLM is needed."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _module(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding = "utf-8"))


def _pushing_code() -> str:
    for node in ast.walk(_module("saving_utils.py")):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_PUSHING_CODE" for t in node.targets
        ):
            return node.value.value
    raise AssertionError("_PUSHING_CODE not found")


def test_the_low_disk_push_template_quotes_its_values():
    # The template is executed, so `revision = main` raised NameError for any named branch.
    code = _pushing_code().format(repo_id = "me/model", revision = "main", use_temp_file = False)
    call = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call))
    kwargs = {k.arg: k.value for k in call.keywords}
    assert isinstance(kwargs["revision"], ast.Constant) and kwargs["revision"].value == "main"
    assert isinstance(kwargs["repo_id"], ast.Constant) and kwargs["repo_id"].value == "me/model"
    # No revision still means the default branch.
    code = _pushing_code().format(repo_id = "me/model", revision = None, use_temp_file = False)
    call = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call))
    assert {k.arg: k.value for k in call.keywords}["revision"].value is None


def test_the_sink_block_mask_call_matches_the_helper_signature():
    # compiled_create_block_mask(mask_mod, bsz, head, M, N): four positionals raised TypeError.
    helper = next(
        n
        for n in ast.walk(_module("flex_attention/utils.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "compiled_create_block_mask"
    )
    required = [a.arg for a in helper.args.args][: len(helper.args.args) - len(helper.args.defaults)]
    calls = [
        n
        for n in ast.walk(_module("flex_attention/attention_sink.py"))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "compiled_create_block_mask"
    ]
    assert calls
    for call in calls:
        assert len(call.args) == len(required), ast.unparse(call)


def test_vllm_supports_flashinfer_reports_a_failed_import_without_a_name_error():
    fn = next(
        n
        for n in ast.walk(_module("vllm_utils.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "vllm_supports_flashinfer"
    )
    handler = next(n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler))
    # Every name the handler reads must exist there: builtins, the caught error, or a parameter.
    bound = {"e"} | {a.arg for a in fn.args.args} | set(dir(builtins))
    used = {n.id for n in ast.walk(handler) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert used <= bound, used - bound
