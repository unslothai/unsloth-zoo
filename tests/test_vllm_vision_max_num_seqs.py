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

"""load_vllm pinned every vision model to max_num_seqs = 1, so GRPO rollouts for
Qwen3.5 / Gemma-4 ran one sequence at a time. vLLM V1 bounds the profiled images
by the encoder budget, not by max_num_seqs, so vision models now keep the text
default unless vLLM is older than 0.11 or UNSLOTH_VLLM_VISION_MAX_NUM_SEQS caps it."""

import ast
import os
import pathlib
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from types import SimpleNamespace

from unsloth_zoo import vllm_utils


def _config(hidden_size = 4096):
    return SimpleNamespace(text_config = SimpleNamespace(hidden_size = hidden_size))


def _seqs(approx = 128, max_num_seqs = 256, kv_gb = 40.0, version = "0.29.0", hidden_size = 4096):
    return vllm_utils.vision_max_num_seqs(
        approx, max_num_seqs, _config(hidden_size), 8192, kv_gb, version,
    )


@pytest.fixture(autouse = True)
def _no_cap(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_VISION_MAX_NUM_SEQS", raising = False)


@pytest.mark.parametrize("version", ["0.13.0", "0.18.1", "0.29.0"])
def test_v1_keeps_the_kv_based_default(version):
    assert _seqs(version = version) == 128


@pytest.mark.parametrize("max_num_seqs", [1, 8, 64, 512])
def test_explicit_max_num_seqs_is_honored(max_num_seqs):
    for version in ("0.10.2", "0.11.2", "0.29.0"):
        assert _seqs(max_num_seqs = max_num_seqs, version = version) == max_num_seqs


@pytest.mark.parametrize("max_num_seqs", [256, None])
def test_v0_capable_vllm_keeps_one_sequence(max_num_seqs):
    assert _seqs(max_num_seqs = max_num_seqs, version = "0.10.2") == 1


def test_padded_profiling_window_caps_by_kv_memory():
    # 0.11 / 0.12 pad each dummy image to (8192, hidden): 64 MiB at hidden 4096.
    assert _seqs(version = "0.11.2", kv_gb = 40.0) == 64
    assert _seqs(version = "0.12.0", kv_gb = 2.0) == 3
    assert _seqs(version = "0.11.2", kv_gb = 0.1) == 1
    assert _seqs(version = "0.11.2", kv_gb = 400.0) == 128


def test_kill_switch_restores_the_old_cap(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_VISION_MAX_NUM_SEQS", "1")
    assert _seqs() == 1
    monkeypatch.setenv("UNSLOTH_VLLM_VISION_MAX_NUM_SEQS", "32")
    assert _seqs() == 32
    assert _seqs(approx = 16) == 16
    assert _seqs(max_num_seqs = 200) == 200


def _load_vllm_node():
    source = pathlib.Path(vllm_utils.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "load_vllm":
            return node
    raise AssertionError("load_vllm not found")


def test_load_vllm_no_longer_hardcodes_one_sequence():
    for node in ast.walk(_load_vllm_node()):
        if not isinstance(node, ast.Assign): continue
        if not any(isinstance(t, ast.Name) and t.id == "approx_max_num_seqs" for t in node.targets): continue
        assert not (isinstance(node.value, ast.Constant) and node.value.value == 1), \
            f"load_vllm still sets approx_max_num_seqs = 1 at line {node.lineno}"


def test_load_vllm_routes_vision_models_through_the_rule():
    calls = [
        node for node in ast.walk(_load_vllm_node())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "vision_max_num_seqs"
    ]
    assert len(calls) == 1, "load_vllm does not call vision_max_num_seqs"


def _engine_max_num_seqs(monkeypatch, float8_kv_cache):
    import torch
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen2_5_VLConfig"):
        pytest.skip("Qwen2_5_VLConfig needs a newer transformers")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (179 * 1024**3, 180 * 1024**3))
    config = transformers.Qwen2_5_VLConfig(
        text_config = dict(hidden_size = 1024, intermediate_size = 2048, num_hidden_layers = 4,
                           num_attention_heads = 8, num_key_value_heads = 2),
        vision_config = dict(depth = 2, hidden_size = 256, out_hidden_size = 1024),
    )
    args = vllm_utils.load_vllm(
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct", config = config, max_seq_length = 2048,
        gpu_memory_utilization = 0.9, use_bitsandbytes = False, is_vision_model = True,
        return_args = True, dtype = torch.bfloat16, float8_kv_cache = float8_kv_cache,
    )
    return args["max_num_seqs"]


@pytest.mark.parametrize("float8_kv_cache", [False, True])
def test_the_vision_cap_holds_after_the_float8_bump(monkeypatch, float8_kv_cache):
    monkeypatch.setenv("UNSLOTH_VLLM_VISION_MAX_NUM_SEQS", "32")
    assert _engine_max_num_seqs(monkeypatch, float8_kv_cache) == 32
