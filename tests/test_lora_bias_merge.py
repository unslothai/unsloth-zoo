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

"""PEFT lora_bias=True: the merged export folds scaling * lora_B.bias into the base bias, like merge_and_unload."""

import copy

import pytest
import torch

from _merge_e2e_helpers import read_safetensors_dir, run_merge, set_offline_cpu_env


def _peft(base_dir, targets):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False,
        attention_bias=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).float()
    model.save_pretrained(base_dir, safe_serialization=True)
    model.config._name_or_path = base_dir
    lora = LoraConfig(r = 4, lora_alpha = 8, lora_dropout = 0.0, target_modules = targets, lora_bias = True)
    peft_model = get_peft_model(model, lora)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if ".lora_" in n:
                p.copy_(torch.randn(p.shape, generator = g) * 0.1)
    return peft_model


def test_lora_bias_merges_into_base_bias(tmp_path):
    set_offline_cpu_env()
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "merged")
    peft_model = _peft(base_dir, ["q_proj", "v_proj"])
    x = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        live = peft_model(x).logits
    ref = copy.deepcopy(peft_model).merge_and_unload().state_dict()
    run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    saved = read_safetensors_dir(out_dir)
    for k in saved:
        torch.testing.assert_close(saved[k], ref[k], rtol = 0, atol = 1e-5, msg = k)
    from transformers import LlamaForCausalLM
    with torch.no_grad():
        reloaded = LlamaForCausalLM.from_pretrained(out_dir, dtype = torch.float32)(x).logits
    torch.testing.assert_close(reloaded, live, rtol = 0, atol = 1e-4)


def test_lora_bias_without_base_bias_refused(tmp_path):
    from unsloth_zoo.saving_utils import create_lora_statistics
    # Llama's MLP has no bias, so there is nothing to fold lora_B's bias into.
    peft_model = _peft(str(tmp_path / "base"), ["gate_proj"])
    with pytest.raises(RuntimeError, match = "lora_bias=True"):
        create_lora_statistics(peft_model)
