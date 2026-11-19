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

"""load_vllm turns off vLLM >= 0.34's extensible KV cache and keeps dense prefix-cache retention
(vLLM >= 0.29) for the engines it builds in-process. AST-level: load_vllm needs a GPU."""

import ast
import inspect
import os
import pathlib

import pytest

VLLM_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
TREE = ast.parse(VLLM_UTILS.read_text(encoding = "utf-8-sig"))
HELPER = "_vllm_colocated_cache_args"
ENV = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"

# EngineArgs fields per vLLM line: 0.34 has both, 0.29 to 0.33 only the retention interval.
KEYS_0_34 = {"model", "enable_prefix_caching", "prefix_cache_retention_interval", "enable_extensible_kv_cache"}
KEYS_0_29 = {"model", "enable_prefix_caching", "prefix_cache_retention_interval"}
KEYS_0_11 = {"model", "enable_prefix_caching"}


def _helper():
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == HELPER]
    assert len(nodes) == 1, f"{HELPER} missing from vllm_utils.py"
    ns = {"os": os}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(VLLM_UTILS), "exec"), ns)
    return ns[HELPER]


@pytest.fixture(autouse = True)
def _no_env(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)


def test_vllm_0_34_gets_both():
    assert _helper()(KEYS_0_34, True) == {
        "enable_extensible_kv_cache": False, "prefix_cache_retention_interval": None,
    }


def test_vllm_0_29_gets_retention_only():
    assert _helper()(KEYS_0_29, True) == {"prefix_cache_retention_interval": None}


def test_older_vllm_untouched():
    assert _helper()(KEYS_0_11, True) == {}


def test_prefix_caching_off_keeps_retention_default():
    assert _helper()(KEYS_0_34, False) == {"enable_extensible_kv_cache": False}


def test_user_env_retention_wins(monkeypatch):
    monkeypatch.setenv(ENV, "1024")
    assert _helper()(KEYS_0_34, True) == {"enable_extensible_kv_cache": False}


def _load_vllm_body():
    fn = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm"]
    assert len(fn) == 1
    return fn[0]


def test_load_vllm_applies_helper_after_key_filter_and_not_for_serve():
    fn = _load_vllm_body()
    src = ast.get_source_segment(VLLM_UTILS.read_text(encoding = "utf-8-sig"), fn)
    filter_at = src.index("good_keys = inspect.signature(")
    call_at = src.index(f"{HELPER}(good_keys")
    build_at = src.index("LLM(**engine_args)")
    serve_exit = src.index("if return_args:\n            _warn_padded_vocab_for_serve")
    assert filter_at < call_at < serve_exit < build_at
    guard = src.rfind("if not return_args:", 0, call_at)
    assert guard != -1 and call_at - guard < 200


def test_real_vllm_engine_args_accept_values():
    vllm_arg_utils = pytest.importorskip("vllm.engine.arg_utils")
    keys = set(inspect.signature(vllm_arg_utils.EngineArgs).parameters)
    args = _helper()(keys, True)
    from packaging.version import Version
    import vllm
    v = Version(vllm.__version__).release
    assert ("enable_extensible_kv_cache" in args) == (v >= (0, 34))
    assert ("prefix_cache_retention_interval" in args) == ("prefix_cache_retention_interval" in keys)
    vllm_arg_utils.EngineArgs(model = "x", **args)
