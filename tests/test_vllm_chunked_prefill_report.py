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

"""load_vllm's startup banner must report what vLLM was actually given.

Two numbers in it described nothing that reached the engine: the chunked prefill
budget came from a dead local, and swap space is dropped by the good_keys filter
on vLLM 0.18.0 and newer. Both now read back from engine_args after that filter.
On the AST, not on text, so reformatting is free. Source-level because load_vllm
needs a live GPU, as in test_vllm_utils_xpu_sm_cap.py.
"""

import ast
import pathlib

LOG_PREFIX = "Chunked prefill tokens = "


def _module_source_text() -> str:
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
    # utf-8-sig: a BOM CPython itself strips would otherwise break ast.parse.
    return path.read_text(encoding = "utf-8-sig")


def _load_vllm_node() -> ast.FunctionDef:
    tree = ast.parse(_module_source_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_vllm":
            return node
    raise AssertionError("load_vllm is gone from unsloth_zoo/vllm_utils.py")


def _banner_value(load_vllm : ast.FunctionDef) -> ast.expr:
    """The expression interpolated straight after the log line's prefix."""
    for node in ast.walk(load_vllm):
        if not isinstance(node, ast.JoinedStr): continue
        for prefix, value in zip(node.values, node.values[1:]):
            if not isinstance(prefix, ast.Constant): continue
            if not str(prefix.value).endswith(LOG_PREFIX): continue
            assert isinstance(value, ast.FormattedValue), \
                f"{LOG_PREFIX!r} no longer interpolates an expression"
            return value.value
    raise AssertionError(f"{LOG_PREFIX!r} vanished from load_vllm")


def _engine_args_lookup(node : ast.expr) -> str:
    """The key `node` reads out of engine_args, or "" if it reads something else."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
            and node.value.id == "engine_args" and isinstance(node.slice, ast.Constant):
        return node.slice.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "get" and isinstance(node.func.value, ast.Name) \
            and node.func.value.id == "engine_args" and node.args \
            and isinstance(node.args[0], ast.Constant):
        return node.args[0].value
    return ""


def test_banner_reports_the_budget_vllm_receives():
    banner = _banner_value(_load_vllm_node())
    assert _engine_args_lookup(banner) == "max_num_batched_tokens", (
        "the startup line no longer reads the budget out of engine_args, so it can "
        "print a number vLLM was never given"
    )


def test_banner_is_printed_after_the_unsupported_key_filter():
    load_vllm = _load_vllm_node()
    banner = _banner_value(load_vllm)
    # engine_args is filtered against EngineArgs' signature; printing before that
    # is how swap space kept being reported after vLLM 0.18.0 stopped taking it.
    deletes = [n.lineno for n in ast.walk(load_vllm)
               if isinstance(n, ast.Delete)
               for t in n.targets
               if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
               and t.value.id == "engine_args"]
    assert deletes, "engine_args is no longer filtered against the vLLM signature"
    assert banner.lineno > max(deletes), "the banner is printed before the filter runs"


def test_budget_is_clamped_when_chunked_prefill_is_off():
    # vLLM raises when max_num_batched_tokens < max_model_len and it cannot chunk.
    # Gated, not blanket: a blanket floor would disable chunking for text models.
    for node in ast.walk(_load_vllm_node()):
        if not isinstance(node, ast.If): continue
        if not (isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == "enable_chunked_prefill"): continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign): continue
            if not any(isinstance(t, ast.Name) and t.id == "max_num_batched_tokens"
                       for t in stmt.targets): continue
            call = stmt.value
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                and call.func.id == "max", "the clamp no longer takes a maximum"
            names = {a.id for a in call.args if isinstance(a, ast.Name)}
            assert names == {"max_num_batched_tokens", "max_seq_length"}, \
                f"the clamp compares {names}, not the budget against max_seq_length"
            return
    raise AssertionError(
        "max_num_batched_tokens is no longer clamped to max_seq_length when "
        "enable_chunked_prefill is False; vLLM rejects that combination"
    )


def test_no_second_chunked_prefill_variable_is_computed_and_dropped():
    load_vllm = _load_vllm_node()

    # Assignments only: prose about the old bug may stay, a revived variable may not.
    assigned = {
        target.id
        for node in ast.walk(load_vllm)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert "chunked_prefill_tokens" not in assigned, (
        "chunked_prefill_tokens is back; if it is ever used again it must reach "
        "vLLM, not just the log line"
    )
