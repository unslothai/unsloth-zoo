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

"""load_vllm forwards the caller's rope_scaling to vLLM (unslothai/unsloth#3083)."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inspect

import pytest

from unsloth_zoo import vllm_utils

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


def test_load_vllm_accepts_rope_scaling():
    # unsloth passes load_vllm only the kwargs named in its signature.
    assert "rope_scaling" in inspect.getfullargspec(vllm_utils.load_vllm).args


def _engine_args(monkeypatch, max_seq_length = 131072, **kwargs):
    import torch
    transformers = pytest.importorskip("transformers")
    if "vllm_version" not in vars(vllm_utils):
        pytest.skip("load_vllm needs vLLM installed")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (179 * 1024**3, 180 * 1024**3))
    config = transformers.Qwen2Config(
        hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 32768,
    )
    return vllm_utils.load_vllm(
        model_name = "unsloth/Qwen2.5-0.5B-Instruct", config = config, max_seq_length = max_seq_length,
        gpu_memory_utilization = 0.5, use_bitsandbytes = False, return_args = True,
        dtype = torch.bfloat16, **kwargs,
    )


def test_rope_scaling_reaches_hf_overrides(monkeypatch):
    args = _engine_args(monkeypatch, rope_scaling = dict(YARN))
    assert args["hf_overrides"]["rope_scaling"] == YARN
    # Newer vLLM does not scale a YaRN max_position_embeddings itself.
    assert args["hf_overrides"]["max_position_embeddings"] == 131072


def test_window_is_not_raised_past_the_scaled_rope(monkeypatch):
    args = _engine_args(monkeypatch, max_seq_length = 262144, rope_scaling = dict(YARN))
    assert "max_position_embeddings" not in args["hf_overrides"]


def test_no_rope_scaling_adds_no_override(monkeypatch):
    args = _engine_args(monkeypatch)
    assert "rope_scaling" not in args.get("hf_overrides", {})
    assert "max_position_embeddings" not in args.get("hf_overrides", {})
