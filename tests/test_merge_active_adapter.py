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

"""Merged export follows the active adapter (A/B, scaling, modules_to_save) like PEFT's own merge."""

import copy
import os

import pytest
import torch

from _merge_e2e_helpers import read_safetensors_dir, run_merge, set_offline_cpu_env


def _base(base_dir):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).float()
    model.save_pretrained(base_dir, safe_serialization=True)
    model.config._name_or_path = base_dir
    return model


def _seed(peft_model, adapter, seed):
    g = torch.Generator().manual_seed(seed)
    for n, p in peft_model.named_parameters():
        if f".{adapter}." in n and (".lora_" in n or ".modules_to_save." in n):
            p.data.copy_(torch.randn(p.shape, generator = g) * 0.1)


def _peft(model, adapters, active):
    from peft import LoraConfig, PeftModel, get_peft_model
    peft_model = None
    for i, (name, targets, r, alpha, saved) in enumerate(adapters):
        cfg = LoraConfig(
            r = r, lora_alpha = alpha, lora_dropout = 0.0, target_modules = targets,
            modules_to_save = saved,
        )
        if peft_model is None:
            peft_model = get_peft_model(model, cfg, adapter_name = name)
        else:
            peft_model.add_adapter(name, cfg)
        _seed(peft_model, name, 100 + i)
    peft_model.set_adapter(active)
    return peft_model


def _merge_and_compare(tmp_path, adapters, active):
    set_offline_cpu_env()
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "merged")
    peft_model = _peft(_base(base_dir), adapters, active)
    x = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        live = peft_model(x).logits
    try:
        ref = copy.deepcopy(peft_model).merge_and_unload().state_dict()
    except TypeError:
        # PEFT <= 0.20 cannot merge a saved module whose active adapter list is empty.
        ref = None
    run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    saved = read_safetensors_dir(out_dir)
    worst = 0.0 if ref is None else max((saved[k] - ref[k]).abs().max().item() for k in saved)
    from transformers import LlamaForCausalLM
    with torch.no_grad():
        reloaded = LlamaForCausalLM.from_pretrained(out_dir, dtype = torch.float32)(x).logits
    return worst, (reloaded - live).abs().max().item()


def test_switched_adapter_matches_peft(tmp_path):
    adapters = [
        ("default", ["q_proj", "v_proj"], 2, 4, ["lm_head"]),
        ("task", ["q_proj", "v_proj"], 3, 9, ["lm_head"]),
    ]
    weight_err, logit_err = _merge_and_compare(tmp_path, adapters, "task")
    assert weight_err < 1e-5 and logit_err < 1e-4, (weight_err, logit_err)


def test_named_only_adapter_matches_peft(tmp_path):
    adapters = [("task", ["q_proj", "o_proj", "embed_tokens"], 4, 8, None)]
    weight_err, logit_err = _merge_and_compare(tmp_path, adapters, "task")
    assert weight_err < 1e-5 and logit_err < 1e-4, (weight_err, logit_err)


def test_partial_targets_and_missing_saved_module(tmp_path, capsys):
    # `task` targets only q_proj and saves no lm_head: k_proj and lm_head must stay base.
    adapters = [
        ("default", ["q_proj", "k_proj"], 2, 4, ["lm_head"]),
        ("task", ["q_proj"], 3, 9, None),
    ]
    weight_err, logit_err = _merge_and_compare(tmp_path, adapters, "task")
    assert weight_err < 1e-5 and logit_err < 1e-4, (weight_err, logit_err)
    assert "LoRA count mismatch" not in capsys.readouterr().out


def test_default_adapter_unchanged(tmp_path):
    adapters = [
        ("default", ["q_proj", "v_proj"], 2, 4, ["lm_head"]),
        ("task", ["q_proj", "v_proj"], 3, 9, ["lm_head"]),
    ]
    weight_err, logit_err = _merge_and_compare(tmp_path, adapters, "default")
    assert weight_err < 1e-5 and logit_err < 1e-4, (weight_err, logit_err)


def test_multiple_active_adapters_refused(tmp_path):
    from peft.tuners.lora import LoraLayer
    from unsloth_zoo.saving_utils import create_lora_statistics
    adapters = [
        ("default", ["q_proj"], 2, 4, None),
        ("task", ["q_proj"], 3, 9, None),
    ]
    peft_model = _peft(_base(str(tmp_path / "base")), adapters, "default")
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            module.set_adapter(["default", "task"])
    with pytest.raises(ValueError, match = "single active adapter"):
        create_lora_statistics(peft_model)


def test_untargeted_layer_keeps_downloaded_weight(tmp_path):
    # The live base_layer weight can differ from the checkpoint (4-bit packed for QLoRA); a layer the
    # active adapter does not target must keep the downloaded tensor, not the live one.
    set_offline_cpu_env()
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "merged")
    adapters = [
        ("default", ["q_proj", "k_proj"], 2, 4, None),
        ("task", ["q_proj"], 3, 9, None),
    ]
    peft_model = _peft(_base(base_dir), adapters, "task")
    disk = read_safetensors_dir(base_dir)
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if name.endswith("k_proj.base_layer"):
                module.weight.mul_(2)
    run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    saved = read_safetensors_dir(out_dir)
    keys = [k for k in saved if k.endswith("k_proj.weight")]
    assert keys and all(torch.equal(saved[k], disk[k]) for k in keys)
