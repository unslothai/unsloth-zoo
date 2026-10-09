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

"""A GPU whose KV cache estimate cannot hold max_seq_length gets vLLM's
max_model_len lowered. The notice must name the requested and the reduced
length (#2666 printed 256 twice), and the clamp itself must not move.
Run off the source: load_vllm needs a GPU."""

import ast
import pathlib

import pytest

HELPER = "_fit_max_seq_length_to_kv_cache"


def _tree() -> ast.Module:
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
    return ast.parse(path.read_text(encoding = "utf-8-sig"))


def _helper():
    tree = _tree()
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == HELPER)
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), "<helper>", "exec"), namespace)
    return namespace[HELPER]


@pytest.mark.parametrize("requested, kv_tokens, expected", [
    (5120, 0,     256),   # no KV room at all: 256 floor
    (5120, -512,  256),
    (5120, 3072,  3072),  # partial room
    (2048, 2048,  2048),  # exactly fits
    (2048, 8192,  2048),  # fits
    (128,  0,     256),   # unchanged: the floor still applies below 256
])
def test_clamp_values_unchanged(requested, kv_tokens, expected, capsys):
    assert _helper()(requested, kv_tokens) == expected


@pytest.mark.parametrize("requested, kv_tokens, reduced", [(5120, 0, 256), (20480, 3072, 3072)])
def test_notice_names_requested_and_reduced_length(requested, kv_tokens, reduced, capsys):
    _helper()(requested, kv_tokens)
    out = capsys.readouterr().out
    assert f"cannot handle sequence lengths of {requested} " in out
    assert f"maximum sequence length of {reduced}." in out
    assert "gpu_memory_utilization" in out


@pytest.mark.parametrize("requested, kv_tokens", [(2048, 2048), (2048, 8192), (128, 0)])
def test_no_notice_when_nothing_is_reduced(requested, kv_tokens, capsys):
    _helper()(requested, kv_tokens)
    assert capsys.readouterr().out == ""


def test_load_vllm_uses_the_helper():
    load_vllm = next(
        n for n in ast.walk(_tree())
        if isinstance(n, ast.FunctionDef) and n.name == "load_vllm"
    )
    calls = [
        n for n in ast.walk(load_vllm)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == HELPER
    ]
    assert calls, f"load_vllm no longer clamps max_seq_length through {HELPER}"
