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


_FIT_HELPERS = ("_speculation_lookahead", "_fit_max_num_seqs_to_speculation", "_fit_max_num_batched_tokens_to_speculation")


def _fit_ns():
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in _FIT_HELPERS]
    assert len(nodes) == len(_FIT_HELPERS), "speculation fit helpers are missing from vllm_utils.py"
    ns = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_fit", "exec"), ns)
    return ns


def _fit():
    return _fit_ns()["_fit_max_num_seqs_to_speculation"]


@pytest.mark.parametrize("seqs, tokens, config, expected", [
    # >80 GiB tier: 256 seqs, 8192 tokens. vLLM needs tokens >= seqs * (k + 1) with LoRA.
    (256, 8192, {"method": "suffix", "num_speculative_tokens": 32}, 248),
    (256, 8192, {"method": "suffix"}, 256),  # default k = 24 already fits (256 * 25 = 6400)
    (256, 4096, {"method": "suffix"}, 163),
    (256, 8192, {"method": "suffix", "suffix_decoding_max_tree_depth": 63}, 128),
    (8, 2048, {"method": "suffix", "num_speculative_tokens": 32}, 8),
    (4, 16, {"method": "suffix", "num_speculative_tokens": 32}, 1),
])
def test_max_num_seqs_fits_lora_speculation(seqs, tokens, config, expected):
    fitted = _fit()(seqs, tokens, config)
    assert fitted == expected
    k = config.get("num_speculative_tokens", config.get("suffix_decoding_max_tree_depth", 24))
    assert fitted == 1 or tokens >= fitted * (k + 1)


def test_load_vllm_fits_max_num_seqs_with_lora():
    load_vllm = next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    src = ast.unparse(load_vllm)
    assign = src.find("engine_args['speculative_config'] = suffix_config")
    fit = src.find("_fit_max_num_seqs_to_speculation(", max(assign, 0))
    lowered = src.find("engine_args['max_num_seqs'] = fitted_max_num_seqs", max(fit, 0))
    arg_filter = src.find("good_keys = inspect.signature(AsyncEngineArgs", max(lowered, 0))
    assert -1 < assign < fit < lowered < arg_filter


def _shrink():
    names = _FIT_HELPERS + ("_shrink_for_memory_retry",)
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), "memory retry helpers are missing from vllm_utils.py"
    ns = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils_shrink", "exec"), ns)
    return ns["_shrink_for_memory_retry"]


@pytest.mark.parametrize("seqs, tokens, max_seq_length, config, expected", [
    # 15 seqs * 33 <= 512 fits; the retry's 11 seqs and 176 tokens would not (11 * 33 = 363).
    (15, 512, 16, {"method": "suffix", "num_speculative_tokens": 32}, (5, 176)),
    (15, 512, 16, None, (11, 176)),
    (256, 8192, 2048, {"method": "suffix"}, (192, 8192)),
])
def test_memory_retry_keeps_the_lora_speculation_bound(seqs, tokens, max_seq_length, config, expected):
    got = _shrink()(seqs, tokens, max_seq_length, config)
    assert got == expected
    if config is not None:
        k = config.get("num_speculative_tokens", 24)
        assert got[1] >= got[0] * (k + 1)


def test_load_vllm_memory_retry_uses_the_refitting_shrink():
    load_vllm = next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    src = ast.unparse(load_vllm)
    assert "_shrink_for_memory_retry(" in src
    assert "approx_max_num_seqs = max(int(approx_max_num_seqs * 0.75), 1)" not in src


def test_return_args_callers_get_no_speculative_config():
    # SyntheticDataKit stringifies load_vllm(return_args = True) into `vllm serve` flags;
    # str(dict) is not JSON, so --speculative-config would make vLLM's parser exit.
    load_vllm = next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    assigns = [
        n for n in ast.walk(load_vllm)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "suffix_config" for t in n.targets)
    ]
    assert len(assigns) == 1
    value = ast.unparse(assigns[0].value)
    assert "return_args" in value and "_vllm_suffix_decoding_config()" in value


@pytest.mark.parametrize("seqs, tokens, config, expected", [
    # A budget below k + 1 cannot be met by max_num_seqs = 1: vLLM needs tokens >= seqs * (k + 1).
    (4, 16, {"method": "suffix"}, (1, 25)),
    (4, 16, {"method": "suffix", "num_speculative_tokens": 32}, (1, 33)),
    (256, 8192, {"method": "suffix"}, (256, 8192)),
])
def test_lora_speculation_budget_below_lookahead_is_raised(seqs, tokens, config, expected):
    ns = _fit_ns()
    fitted = ns["_fit_max_num_seqs_to_speculation"](seqs, tokens, config)
    got = (fitted, ns["_fit_max_num_batched_tokens_to_speculation"](fitted, tokens, config))
    assert got == expected
    k = config.get("num_speculative_tokens", 24)
    assert got[1] >= got[0] * (k + 1)


def test_memory_retry_raises_a_budget_below_lookahead():
    got = _shrink()(2, 64, 8, {"method": "suffix"})
    assert got == (1, 25)


@pytest.mark.parametrize("config", [
    {"method": "suffix", "suffix_decoding_max_tree_depth": None},
    {"method": "suffix", "num_speculative_tokens": -1},
    {"method": "suffix", "num_speculative_tokens": "many"},
    {"method": "suffix", "num_speculative_tokens": True},
])
def test_malformed_lookahead_is_left_to_vllm(config):
    # Was TypeError / ZeroDivisionError from the fitter; now vLLM's own validation reports the value.
    ns = _fit_ns()
    assert ns["_fit_max_num_seqs_to_speculation"](8, 512, config) == 8
    assert ns["_fit_max_num_batched_tokens_to_speculation"](8, 512, config) == 512


def test_load_vllm_raises_the_lora_speculation_budget():
    load_vllm = next(n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    src = ast.unparse(load_vllm)
    lowered = src.find("engine_args['max_num_seqs'] = fitted_max_num_seqs")
    raised = src.find("engine_args['max_num_batched_tokens'] = fitted_max_num_batched_tokens", max(lowered, 0))
    arg_filter = src.find("good_keys = inspect.signature(AsyncEngineArgs", max(raised, 0))
    assert -1 < lowered < raised < arg_filter
