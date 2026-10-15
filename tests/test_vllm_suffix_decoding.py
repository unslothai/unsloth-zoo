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

"""UNSLOTH_VLLM_SUFFIX_DECODING turns on vLLM suffix decoding for fast_inference engines.

Off by default. On, load_vllm passes speculative_config={"method": "suffix", ...} only
when the installed vLLM lists the "suffix" method (vLLM >= 0.11.1) and Arctic Inference
is importable, and refuses with a clear message otherwise. load_vllm needs a GPU, so the
helper is exercised on its own and the wiring is checked on the AST.
"""

import ast
import json
import os
import pathlib
import sys
import types
import typing

import pytest

import unsloth_zoo

ROOT = pathlib.Path(unsloth_zoo.__file__).resolve().parents[1]


def _tree():
    return ast.parse((ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8-sig"))


def _helper(monkeypatch, methods, has_arctic, version = "0.29.0"):
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "_vllm_suffix_decoding_config"]
    assert len(nodes) == 1, "_vllm_suffix_decoding_config is missing from vllm_utils.py"
    fake_util = types.SimpleNamespace(
        find_spec = lambda name: object() if (name == "arctic_inference" and has_arctic) else None,
    )
    ns = {"os": os, "json": json, "importlib": types.SimpleNamespace(util = fake_util), "vllm_version": version}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_helper", "exec"), ns)
    names = ["vllm", "vllm.config", "vllm.config.speculative"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    if methods is not None:
        sys.modules["vllm.config.speculative"].SpeculativeMethod = typing.Literal[methods]
    return ns["_vllm_suffix_decoding_config"]


@pytest.mark.parametrize("value", [None, "", "0", "false"])
def test_off_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("UNSLOTH_VLLM_SUFFIX_DECODING", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", value)
    # Off must not even probe vLLM: an old vLLM without arctic-inference stays untouched.
    assert _helper(monkeypatch, None, has_arctic = False)() is None


def test_on_supported(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", "1")
    fn = _helper(monkeypatch, ("ngram", "suffix"), has_arctic = True)
    assert fn() == {"method": "suffix"}


def test_json_extra_keys(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", '{"num_speculative_tokens": 16, "method": "ngram"}')
    fn = _helper(monkeypatch, ("ngram", "suffix"), has_arctic = True)
    assert fn() == {"num_speculative_tokens": 16, "method": "suffix"}


@pytest.mark.parametrize("value", ["2", "suffix", "[1]", "{bad json"])
def test_bad_values_raise(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", value)
    with pytest.raises(ValueError, match = "UNSLOTH_VLLM_SUFFIX_DECODING"):
        _helper(monkeypatch, ("ngram", "suffix"), has_arctic = True)()


def test_old_vllm_refused(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", "1")
    with pytest.raises(RuntimeError, match = "vLLM >= 0.11.1"):
        _helper(monkeypatch, ("ngram", "eagle"), has_arctic = True, version = "0.10.2")()


def test_missing_literal_refused(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", "1")
    with pytest.raises(RuntimeError, match = "vLLM >= 0.11.1"):
        _helper(monkeypatch, None, has_arctic = True, version = "0.9.0")()


def test_missing_arctic_refused(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SUFFIX_DECODING", "1")
    with pytest.raises(RuntimeError, match = "arctic-inference"):
        _helper(monkeypatch, ("ngram", "suffix"), has_arctic = False)()


def test_load_vllm_passes_speculative_config_before_arg_filter():
    load_vllm = next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    src = ast.unparse(load_vllm)
    call = src.find("_vllm_suffix_decoding_config()")
    assign = src.find("engine_args['speculative_config'] = suffix_config")
    arg_filter = src.find("good_keys = inspect.signature(AsyncEngineArgs", max(assign, 0))
    assert -1 < call < assign < arg_filter
