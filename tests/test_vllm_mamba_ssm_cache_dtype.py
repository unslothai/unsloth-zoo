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

"""UNSLOTH_VLLM_MAMBA_SSM_CACHE_DTYPE -> vLLM mamba_ssm_cache_dtype. AST-level: load_vllm needs a GPU."""

import ast
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

VLLM_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "vllm_utils.py"
TREE = ast.parse(VLLM_UTILS.read_text(encoding = "utf-8-sig"))
HELPERS = ("_config_has_gdn_layers", "_vllm_mamba_dtype_choices", "_vllm_gdn_reads_ssm_dtype",
           "_vllm_mamba_ssm_cache_dtype_args")
CONSTANTS = ("_VLLM_MAMBA_SSM_DTYPE_ALIASES",)
ENV = "UNSLOTH_VLLM_MAMBA_SSM_CACHE_DTYPE"

# MambaDType exactly as vLLM declares it (vllm/config/cache.py).
VLLM_0_11_2 = 'MambaDType = Literal["auto", "float32"]'
VLLM_0_20 = 'MambaDType = Literal["auto", "float32", "float16"]'
VLLM_LATEST = 'MambaDType = Literal["auto", "float32", "float16", "bfloat16"]'


def _helpers():
    nodes = [
        n for n in TREE.body
        if (isinstance(n, ast.FunctionDef) and n.name in HELPERS)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in CONSTANTS for t in n.targets))
    ]
    assert len(nodes) == len(HELPERS) + len(CONSTANTS)
    ns = {"os": os}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(VLLM_UTILS), "exec"), ns)
    return ns


NS = _helpers()
KEYS = {"model", "mamba_ssm_cache_dtype"}
QWEN3_NEXT = SimpleNamespace(model_type = "qwen3_next", linear_key_head_dim = 128)
QWEN3_5 = SimpleNamespace(
    model_type = "qwen3_5",
    text_config = SimpleNamespace(model_type = "qwen3_5_text", layer_types = ["linear_attention", "full_attention"],
                                  linear_key_head_dim = 128, mamba_ssm_dtype = "float32"),
)
MINIMAX = SimpleNamespace(model_type = "minimax", layer_types = ["linear_attention", "full_attention"])
LLAMA = SimpleNamespace(model_type = "llama")


# MambaStateDtypeCalculator.gated_delta_net_state_dtype (vllm/model_executor/layers/mamba/mamba_utils.py).
GDN_OLD = "def gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype): pass"  # <= 0.16
GDN_NEW = 'def gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype = "auto"): pass'


@pytest.fixture
def fake_vllm(monkeypatch):
    def install(source, gdn = GDN_NEW):
        module = types.ModuleType("vllm.config.cache")
        exec("from typing import Literal\n" + source, module.__dict__)
        mamba_utils = types.ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
        exec("class MambaStateDtypeCalculator:\n    @classmethod\n    " + gdn, mamba_utils.__dict__)
        for name in ("vllm", "vllm.config", "vllm.model_executor", "vllm.model_executor.layers",
                     "vllm.model_executor.layers.mamba"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "vllm.config.cache", module)
        monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.mamba.mamba_utils", mamba_utils)
    return install


@pytest.mark.parametrize("config", [QWEN3_NEXT, QWEN3_5, LLAMA])
def test_unset_passes_nothing(monkeypatch, config):
    monkeypatch.delenv(ENV, raising = False)
    for training in (False, True):
        args, _ = NS["_vllm_mamba_ssm_cache_dtype_args"](config, KEYS, training)
        assert args == {}


def test_unset_hints_only_for_training_gdn_without_a_config_dtype(monkeypatch, fake_vllm):
    fake_vllm(VLLM_LATEST)
    monkeypatch.delenv(ENV, raising = False)
    fn = NS["_vllm_mamba_ssm_cache_dtype_args"]
    assert ENV in fn(QWEN3_NEXT, KEYS, True)[1]
    assert fn(QWEN3_NEXT, KEYS, False)[1] is None
    assert fn(QWEN3_NEXT, {"model"}, True)[1] is None
    assert fn(QWEN3_5, KEYS, True)[1] is None  # vLLM already reads mamba_ssm_dtype = float32
    assert fn(LLAMA, KEYS, True)[1] is None


@pytest.mark.parametrize("value, expected", [
    ("float32", "float32"), ("FP32", "float32"), (" fp32 ", "float32"),
    ("bfloat16", "bfloat16"), ("bf16", "bfloat16"), ("float16", "float16"), ("fp16", "float16"),
    ("auto", "auto"),
])
def test_latest_vllm_accepts(monkeypatch, fake_vllm, value, expected):
    fake_vllm(VLLM_LATEST)
    monkeypatch.setenv(ENV, value)
    assert NS["_vllm_mamba_ssm_cache_dtype_args"](QWEN3_NEXT, KEYS, True) == ({"mamba_ssm_cache_dtype": expected}, None)


@pytest.mark.parametrize("value, passed", [("float32", True), ("auto", True), ("bfloat16", False), ("fp16", False)])
def test_vllm_0_11_2_only_takes_auto_and_float32(monkeypatch, fake_vllm, value, passed):
    # Mamba hybrids read mamba_ssm_cache_dtype on 0.11.2; Gated DeltaNet does not (next test).
    fake_vllm(VLLM_0_11_2, gdn = GDN_OLD)
    monkeypatch.setenv(ENV, value)
    args, message = NS["_vllm_mamba_ssm_cache_dtype_args"](LLAMA, KEYS, True)
    if passed:
        assert args == {"mamba_ssm_cache_dtype": value} and message is None
    else:
        assert args == {} and "auto, float32" in message


def test_vllm_0_20_has_no_bfloat16_yet(monkeypatch, fake_vllm):
    fake_vllm(VLLM_0_20)
    monkeypatch.setenv(ENV, "bf16")
    assert NS["_vllm_mamba_ssm_cache_dtype_args"](QWEN3_NEXT, KEYS, True)[0] == {}
    monkeypatch.setenv(ENV, "fp16")
    assert NS["_vllm_mamba_ssm_cache_dtype_args"](QWEN3_NEXT, KEYS, True)[0] == {"mamba_ssm_cache_dtype": "float16"}


def test_unreadable_literal_falls_back_to_the_0_11_2_set(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.config.cache", None)
    assert NS["_vllm_mamba_dtype_choices"]() == ("auto", "float32")


def test_engine_without_the_arg_is_left_alone(monkeypatch):
    monkeypatch.setenv(ENV, "float32")
    args, message = NS["_vllm_mamba_ssm_cache_dtype_args"](QWEN3_NEXT, {"model"}, True, choices = ("auto", "float32"))
    assert args == {} and "no mamba_ssm_cache_dtype" in message


def test_typo_is_ignored_not_raised(monkeypatch):
    monkeypatch.setenv(ENV, "float33")
    args, message = NS["_vllm_mamba_ssm_cache_dtype_args"](QWEN3_NEXT, KEYS, True, choices = ("auto", "float32"))
    assert args == {} and "float33" in message


def test_load_vllm_applies_it_after_the_engine_arg_filter():
    load_vllm = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    source = ast.get_source_segment(VLLM_UTILS.read_text(encoding = "utf-8-sig"), load_vllm)
    call = source.index("_vllm_mamba_ssm_cache_dtype_args(config, good_keys, training)")
    assert source.index("good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs)") < call
    assert call < source.index("engine_args.update(ssm_dtype_args)") < source.index("if return_args:")
    # Engines are built only after the update, so every LLM / LLMEngine / AsyncLLMEngine gets it.
    assert source.index("engine_args.update(ssm_dtype_args)") < source.index("llm = LLM(**engine_args)")


@pytest.mark.parametrize("value", ["float32", "auto"])
def test_gdn_override_is_refused_where_vllm_does_not_read_it(monkeypatch, fake_vllm, value):
    fake_vllm(VLLM_0_11_2, gdn = GDN_OLD)
    monkeypatch.setenv(ENV, value)
    fn = NS["_vllm_mamba_ssm_cache_dtype_args"]
    args, message = fn(QWEN3_NEXT, KEYS, True)
    assert args == {} and "vLLM < 0.17" in message
    monkeypatch.delenv(ENV)
    assert fn(QWEN3_NEXT, KEYS, True) == ({}, None)  # no hint either: the override cannot work here
    fake_vllm(VLLM_0_11_2, gdn = GDN_NEW)
    monkeypatch.setenv(ENV, value)
    assert fn(QWEN3_NEXT, KEYS, True) == ({"mamba_ssm_cache_dtype": value}, None)


def test_gdn_support_unreadable_refuses(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.mamba.mamba_utils", None)
    assert NS["_vllm_gdn_reads_ssm_dtype"]() is False


def test_non_gdn_linear_attention_is_refused_and_not_hinted(monkeypatch, fake_vllm):
    fake_vllm(VLLM_LATEST)
    fn = NS["_vllm_mamba_ssm_cache_dtype_args"]
    monkeypatch.setenv(ENV, "float32")
    args, message = fn(MINIMAX, KEYS, True)
    assert args == {} and "linear attention" in message
    monkeypatch.delenv(ENV)
    assert fn(MINIMAX, KEYS, True) == ({}, None)
