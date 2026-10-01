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

"""`merged_16bit` must export biases trained via `bias="lora_only"|"all"` or `modules_to_save`.

The shard writers rewrite weights in place from the base checkpoint, so a trained bias was
written back as the base checkpoint's value. Untrained biases must stay byte-identical.
"""

from __future__ import annotations

import pytest
import torch

import _merge_e2e_helpers as H

TARGETS = ["q_proj", "v_proj", "down_proj"]


def _build(tmp_path, *, bias = "none", modules_to_save = None, runtime_dtype = torch.float32,
           shard_size = None):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    base = str(tmp_path / "base")
    spec = H.make_spec("llama")
    spec.config.attention_bias = True
    spec.config.mlp_bias = True
    torch.manual_seed(H.SEED)
    model = AutoModelForCausalLM.from_config(spec.config).to(torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.copy_(torch.randn_like(p))
    kwargs = {"max_shard_size": shard_size} if shard_size else {}
    model.save_pretrained(base, safe_serialization = True, **kwargs)

    model = AutoModelForCausalLM.from_pretrained(base, dtype = runtime_dtype)
    model.config._name_or_path = base
    peft_model = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, bias = bias,
        target_modules = TARGETS, modules_to_save = modules_to_save))
    H.seed_lora(peft_model)
    return base, peft_model


def _train_biases(peft_model):
    """Shift every trainable bias, standing in for an optimizer step; returns the
    expected {checkpoint key: value}."""
    expected = {}
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if not name.endswith(".bias") or not p.requires_grad:
                continue
            p.add_(0.5)
            key = H._strip_peft_prefix(name)
            key = key.replace(".base_layer", "").replace(".modules_to_save.default", "")
            expected[key] = p.detach().float().cpu().clone()
    assert expected, "no trainable bias: the case tests nothing"
    return expected


def _merge(peft_model, base, out):
    H.run_merge(peft_model, base, str(out), save_dtype = torch.float32)
    return H.read_safetensors_dir(str(out))


@pytest.mark.parametrize("bias,modules_to_save", [
    ("lora_only", None),
    ("all", None),
    ("none", ["gate_proj"]),
])
@pytest.mark.parametrize("shard_size", [None, "4KB"])
def test_trained_bias_is_exported(tmp_path, bias, modules_to_save, shard_size):
    base, peft_model = _build(tmp_path, bias = bias, modules_to_save = modules_to_save,
                              shard_size = shard_size)
    expected = _train_biases(peft_model)
    base_tensors = H.read_safetensors_dir(base)
    saved = _merge(peft_model, base, tmp_path / "out")

    for key, value in expected.items():
        assert torch.equal(saved[key], value), key
    for key, value in base_tensors.items():
        if key.endswith(".bias") and key not in expected:
            assert torch.equal(saved[key], value), f"untrained bias moved: {key}"


def test_frozen_reloaded_adapter_still_exports_bias(tmp_path):
    # An adapter reloaded for inference has requires_grad=False, so selection must
    # come from the adapter config.
    base, peft_model = _build(tmp_path, bias = "lora_only")
    expected = _train_biases(peft_model)
    for p in peft_model.parameters():
        p.requires_grad_(False)
    saved = _merge(peft_model, base, tmp_path / "out")
    for key, value in expected.items():
        assert torch.equal(saved[key], value), key


def test_untrained_fp32_bias_survives_bf16_runtime(tmp_path):
    base, peft_model = _build(tmp_path, bias = "none", runtime_dtype = torch.bfloat16)
    base_tensors = H.read_safetensors_dir(base)
    saved = _merge(peft_model, base, tmp_path / "out")
    for key, value in base_tensors.items():
        if key.endswith(".bias"):
            assert saved[key].dtype == torch.float32
            assert torch.equal(saved[key], value), key
