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
"""load_lora(load_tensors = True) on a PEFT model whose adapter is not named "default": the live config and the
tensors of the active adapter reach vLLM. CPU, tiny random Llama, vLLM's LoRARequest stubbed."""
import json
import os
import sys
import types

import pytest
import torch

import unsloth_zoo.vllm_utils as vu


def _peft_model(*names):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 2,
    )
    torch.manual_seed(0)
    model = get_peft_model(
        LlamaForCausalLM(config), LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"]), adapter_name = names[0]
    )
    for i, name in enumerate(names[1:]):
        model.add_adapter(name, LoraConfig(r = 2 + 2 * (i + 2), lora_alpha = 8, target_modules = ["q_proj"]))
    return model


@pytest.fixture
def lora_request(monkeypatch):
    class LoRARequest:
        def __init__(self, name, lora_int_id, lora_tensors = None, lora_config = None, lora_path = None):
            self.lora_tensors, self.lora_config = lora_tensors, lora_config

    for mod, attrs in (("vllm", {}), ("vllm.lora", {}), ("vllm.lora.request", {"LoRARequest": LoRARequest})):
        monkeypatch.setitem(sys.modules, mod, types.SimpleNamespace(__name__ = mod, **attrs))
    monkeypatch.setattr(vu, "LORA_REQUEST_ID", 1, raising = False)
    seen = {}
    monkeypatch.setattr(vu, "_check_lora_is_servable", lambda m, keys, src, cfg: seen.setdefault("keys", list(keys)))
    vu._LIVE_LORA_CONFIGS.clear()
    yield seen
    vu._LIVE_LORA_CONFIGS.clear()


@pytest.mark.parametrize("names", [("default",), ("a1",), ("a1", "other")], ids = ["default", "renamed", "two_adapters"])
def test_load_tensors_serves_the_active_adapter(lora_request, tmp_path, names):
    model = _peft_model(*names)
    request = vu._build_lora_request(model, str(tmp_path), True, 1)
    with open(tmp_path / "adapter_config.json") as f:
        assert json.load(f)["r"] == 4
    assert request.lora_config["r"] == 4
    keys = sorted(request.lora_tensors)
    assert keys == [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    ], keys
    q = model.base_model.model.model.layers[0].self_attn.q_proj
    assert torch.equal(request.lora_tensors[keys[0]], q.lora_A[names[0]].weight)
    assert torch.equal(request.lora_tensors[keys[1]], q.lora_B[names[0]].weight)


def test_default_adapter_drops_other_adapters(lora_request, tmp_path):
    # DPO adds a frozen "ref" adapter next to the trained "default" one: only "default" may reach vLLM.
    model = _peft_model("default", "ref")
    model.set_adapter("default")
    request = vu._build_lora_request(model, str(tmp_path), True, 1)
    keys = sorted(request.lora_tensors)
    assert keys == [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    ], keys
    q = model.base_model.model.model.layers[0].self_attn.q_proj
    assert torch.equal(request.lora_tensors[keys[0]], q.lora_A["default"].weight)
    assert torch.equal(request.lora_tensors[keys[1]], q.lora_B["default"].weight)


@pytest.mark.parametrize("name", ["default", "a1"])
def test_strip_keeps_module_names_that_contain_the_adapter_name(name):
    sd = {
        f"m.layers.0.{name}_mlp.up.lora_A.{name}.weight": 1,
        f"m.layers.0.{name}_mlp.up.lora_B.{name}.weight": 2,
        f"m.layers.0.q_proj.lora_A.{name}x.weight": 3,
        "m.layers.0.q_proj.lora_A.ref.weight": 4,
    }
    assert vu._strip_lora_adapter_name(sd, name) == {
        f"m.layers.0.{name}_mlp.up.lora_A.weight": 1,
        f"m.layers.0.{name}_mlp.up.lora_B.weight": 2,
    }
