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

"""load_vllm must not force VLLM_USE_FLASHINFER_SAMPLER=1 where vLLM rejects the GPU: engine init then raises."""

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


def _sampler_env(major_version, vllm_version, inherited = None):
    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    chain = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(_sets_sampler(stmt, "1") for stmt in node.body)
    )
    env = {} if inherited is None else {"VLLM_USE_FLASHINFER_SAMPLER": inherited}
    exec(
        compile(ast.Module(body = [chain], type_ignores = []), "<sampler chain>", "exec"),
        {
            "os": types.SimpleNamespace(environ = env),
            "Version": Version,
            "major_version": major_version,
            "vllm_version": vllm_version,
        },
    )
    return env.get("VLLM_USE_FLASHINFER_SAMPLER")


def _vllm_sampler_init(value, major_version, vllm_version):
    """vLLM >= 0.21 flashinfer_sampler_supported: default on, sm_75 floor (sm_80 from 0.24), raises only if forced."""
    if value is not None and not int(value):
        return "native"
    floor = 8 if Version(vllm_version) >= Version("0.24.0") else 7
    if major_version >= floor:
        return "flashinfer"
    if value is not None:
        raise RuntimeError("FlashInfer top-p/top-k sampling unavailable: unsupported compute capability")
    return "native"


@pytest.mark.parametrize("inherited", [None, "1", "0"])
@pytest.mark.parametrize("vllm_version", ["0.21.0", "0.23.0", "0.24.0", "0.26.0", "0.33.0", "0.26.0+cu130"])
@pytest.mark.parametrize("major_version", [7, 8, 9, 10, 12])
def test_engine_init_never_raises_from_vllm_0_21(major_version, vllm_version, inherited):
    value = _sampler_env(major_version, vllm_version, inherited)
    used = _vllm_sampler_init(value, major_version, vllm_version)
    floor = 8 if Version(vllm_version) >= Version("0.24.0") else 7
    assert used == ("flashinfer" if major_version >= floor else "native")


@pytest.mark.parametrize("vllm_version", ["0.24.0", "0.26.0", "0.33.0"])
def test_sm75_leaves_the_choice_to_vllm(vllm_version):
    assert _sampler_env(7, vllm_version, inherited = "1") is None


@pytest.mark.parametrize("vllm_version", ["0.11.0", "0.15.0", "0.20.0"])
def test_sm75_still_opts_in_before_vllm_default_on(vllm_version):
    assert _sampler_env(7, vllm_version) == "1"


def test_sm75_before_0_11_stays_off():
    assert _sampler_env(7, "0.10.2") == "0"


@pytest.mark.parametrize("major_version", [8, 9, 10, 12])
@pytest.mark.parametrize("vllm_version", ["0.10.0", "0.20.0", "0.26.0"])
def test_sm80_plus_unchanged(major_version, vllm_version):
    assert _sampler_env(major_version, vllm_version) == "1"
