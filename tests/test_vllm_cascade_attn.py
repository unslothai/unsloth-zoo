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

"""vLLM 0.11.0 pins flash-attention ee4d25bd, one commit before vllm-project/flash-attention#87."""

import ast
import pathlib

import pytest
from packaging.version import Version

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"


def _helper():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vllm_cascade_attn_needs_disable"
    )
    namespace = {"Version": Version}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace["_vllm_cascade_attn_needs_disable"]


@pytest.mark.parametrize(
    "vllm_version, major, expected",
    [
        ("0.10.2", 8, True),
        ("0.11.0", 8, True),
        ("0.11.1", 8, False),
        ("0.11.2", 8, False),
        ("0.18.0", 8, False),  # vLLM's own default (disabled) must stand
        ("0.10.2", 9, False),
        ("0.11.0", 10, False),
        ("0.11.0", 12, False),
    ],
)
def test_cascade_attention_is_only_ever_disabled(vllm_version, major, expected):
    assert _helper()(vllm_version, major) is expected


def test_load_vllm_never_forces_cascade_attention_on():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
            for target in targets:
                name = None
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    name = target.slice.value
                elif isinstance(target, ast.Name):
                    name = target.id
                if name == "disable_cascade_attn":
                    assert not (isinstance(value, ast.Constant) and value.value is False), (
                        f"line {node.lineno} forces cascade attention on"
                    )
