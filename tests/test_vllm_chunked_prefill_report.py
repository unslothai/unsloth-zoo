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

"""Regression guard for the chunked-prefill number load_vllm prints.

vLLM's chunked-prefill budget is the `max_num_batched_tokens` load_vllm passes to
EngineArgs. A separate `chunked_prefill_tokens` local used to be sized by its own
memory ladder, overwritten by `max_seq_length` on the next line, and then printed,
so the reported number described nothing that reached vLLM. The surrounding
function needs a live GPU, so this checks the source-level invariant the way
tests/test_vllm_utils_xpu_sm_cap.py does.

Matched on the AST rather than on text: the invariant is "the name rendered into
the log line is the name handed to EngineArgs", which a regex can only
approximate. Reformatting the print or the engine_args call must not fail this,
and a plain `max_num_batched_tokens = max_num_batched_tokens` sitting anywhere
else in the module must not satisfy it.
"""

import ast
import pathlib

LOG_PREFIX = "Chunked prefill tokens = "


def _module_source_text() -> str:
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
    # utf-8-sig, not utf-8: CPython's tokenizer strips a BOM, so a file saved by
    # a Windows editor imports fine but would hand ast.parse a leading U+FEFF
    # and fail this guard for a reason that has nothing to do with the invariant.
    return path.read_text(encoding = "utf-8-sig")


def _load_vllm_node() -> ast.FunctionDef:
    tree = ast.parse(_module_source_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_vllm":
            return node
    raise AssertionError("load_vllm is gone from unsloth_zoo/vllm_utils.py")


def _printed_name(load_vllm : ast.FunctionDef) -> str:
    """The variable interpolated straight after the log line's prefix."""
    for node in ast.walk(load_vllm):
        if not isinstance(node, ast.JoinedStr): continue
        for prefix, value in zip(node.values, node.values[1:]):
            if not isinstance(prefix, ast.Constant): continue
            if not str(prefix.value).endswith(LOG_PREFIX): continue
            assert isinstance(value, ast.FormattedValue) and \
                isinstance(value.value, ast.Name), \
                f"{LOG_PREFIX!r} no longer interpolates a plain variable"
            return value.value.id
    raise AssertionError(f"{LOG_PREFIX!r} vanished from load_vllm")


def _engine_args_value(load_vllm : ast.FunctionDef, keyword : str) -> str:
    """The variable passed as `keyword` in `engine_args = dict(...)`."""
    for node in ast.walk(load_vllm):
        if not isinstance(node, ast.Assign): continue
        if not any(isinstance(t, ast.Name) and t.id == "engine_args"
                   for t in node.targets): continue
        assert isinstance(node.value, ast.Call), "engine_args is no longer a dict(...) call"
        for kw in node.value.keywords:
            if kw.arg != keyword: continue
            assert isinstance(kw.value, ast.Name), \
                f"engine_args[{keyword!r}] is no longer a plain variable"
            return kw.value.id
        raise AssertionError(f"engine_args no longer passes {keyword!r} to vLLM")
    raise AssertionError("engine_args is not built in load_vllm any more")


def test_chunked_prefill_line_reports_the_value_vllm_receives():
    load_vllm = _load_vllm_node()

    printed = _printed_name(load_vllm)
    given = _engine_args_value(load_vllm, "max_num_batched_tokens")

    assert printed == given, (
        f"the startup line prints {printed!r} but vLLM is given {given!r}; "
        f"the number users read does not describe the run"
    )


def test_no_second_chunked_prefill_variable_is_computed_and_dropped():
    load_vllm = _load_vllm_node()

    # Every assignment to the old local was overwritten before anything read it, so the
    # eight-branch ladder that produced them could not change the run. Checked as
    # assignments, not as a substring: prose about the old bug is allowed to survive
    # in comments and docstrings, a revived variable is not.
    assigned = {
        target.id
        for node in ast.walk(load_vllm) if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert "chunked_prefill_tokens" not in assigned, (
        "chunked_prefill_tokens is back; if it is ever used again it must reach "
        "vLLM, not just the log line"
    )
