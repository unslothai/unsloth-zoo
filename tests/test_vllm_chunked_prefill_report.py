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

"""The chunked prefill number load_vllm prints must be the one vLLM is given.

`chunked_prefill_tokens` was sized by its own ladder, overwritten by
`max_seq_length`, then printed, so the reported budget described nothing that
reached EngineArgs. Matched on the AST, not on text, so reformatting the print
or the engine_args call is free and only a real regression fails. Source-level
because load_vllm needs a live GPU, as in tests/test_vllm_utils_xpu_sm_cap.py.
"""

import ast
import pathlib

LOG_PREFIX = "Chunked prefill tokens = "


def _module_source_text() -> str:
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
    # utf-8-sig: CPython strips a BOM, so a file saved by a Windows editor imports
    # fine but would hand ast.parse a leading U+FEFF.
    return path.read_text(encoding = "utf-8-sig")


def _load_vllm_node() -> ast.FunctionDef:
    for node in ast.walk(ast.parse(_module_source_text())):
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
        if not any(isinstance(t, ast.Name) and t.id == "engine_args" for t in node.targets):
            continue
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

    # Assignments only, not a substring: prose about the old bug may stay in
    # comments, a revived variable may not.
    assigned = {
        target.id
        for node in ast.walk(load_vllm)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert "chunked_prefill_tokens" not in assigned, (
        "chunked_prefill_tokens is back; if it is used again it must reach vLLM, "
        "not just the log line"
    )
