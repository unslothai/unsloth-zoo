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

"""load_vllm must not set VLLM_USE_FLASHINFER_SAMPLER=1 on sm_75 from vLLM 0.24: it raises at engine init when the
variable is explicitly set and FlashInfer rejects the GPU (T4 is sm_75); AST-based, no GPU."""

import ast
import importlib
import inspect
import textwrap
import types

import pytest
from packaging.version import Version

vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")


def _sets_sampler(stmt, value):
    return (
        isinstance(stmt, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and getattr(target.slice, "value", "") == "VLLM_USE_FLASHINFER_SAMPLER"
            for target in stmt.targets
        )
        and getattr(stmt.value, "value", "") == value
    )


def _sampler_env(major_version, vllm_version):
    """Runs load_vllm's own sampler if-chain with the given GPU major and vLLM version and returns what it sets."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    chain = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(_sets_sampler(stmt, "1") for stmt in node.body)
    )
    env = {}
    exec(
        compile(ast.Module(body = [chain], type_ignores = []), "<sampler chain>", "exec"),
        {
            "os": types.SimpleNamespace(environ = env),
            "Version": Version,
            "major_version": major_version,
            "vllm_version": vllm_version,
        },
    )
    return env["VLLM_USE_FLASHINFER_SAMPLER"]


@pytest.mark.parametrize("vllm_version", ["0.10.0", "0.24.0", "0.26.0", "0.33.0"])
def test_sampler_stays_off_on_sm75_where_flashinfer_rejects_it(vllm_version):
    assert _sampler_env(7, vllm_version) == "0"


@pytest.mark.parametrize("vllm_version", ["0.11.0", "0.20.0", "0.23.2"])
def test_sampler_is_still_on_for_sm75_before_vllm_0_24(vllm_version):
    assert _sampler_env(7, vllm_version) == "1"


@pytest.mark.parametrize("major_version", [8, 9, 10, 12])
@pytest.mark.parametrize("vllm_version", ["0.10.0", "0.26.0"])
def test_sampler_is_on_from_sm80(major_version, vllm_version):
    assert _sampler_env(major_version, vllm_version) == "1"
