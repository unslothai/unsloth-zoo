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

"""CUDA `modules_to_save` copies must land in their own merged keys, not a stale pinned buffer."""

from __future__ import annotations

import pytest
import torch

import _merge_e2e_helpers as H

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

NORMS = ("input_layernorm", "post_attention_layernorm")
LAYERS = 12


def _fill_value(norm_index, layer):
    return float(100 * norm_index + layer + 2)


def test_cuda_modules_to_save_copies_merge_to_their_own_keys(tmp_path):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    spec = H.make_spec("llama")
    spec.config.num_hidden_layers = LAYERS
    base = str(tmp_path / "base")
    torch.manual_seed(H.SEED)
    AutoModelForCausalLM.from_config(spec.config).to(torch.bfloat16).save_pretrained(
        base, safe_serialization = True)

    model = AutoModelForCausalLM.from_pretrained(base, dtype = torch.bfloat16).cuda()
    model.config._name_or_path = base
    peft_model = get_peft_model(model, LoraConfig(
        r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
        target_modules = ["q_proj", "v_proj"], modules_to_save = list(NORMS)))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if ".modules_to_save." in name:
                layer = int(name.split(".layers.")[1].split(".")[0])
                k = next(i for i, n in enumerate(NORMS) if f".{n}." in name)
                param.data = torch.full_like(param, _fill_value(k, layer), dtype = torch.float32)

    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = torch.bfloat16)

    written = H.read_safetensors_dir(out)
    wrong = {}
    for k, norm in enumerate(NORMS):
        for layer in range(LAYERS):
            got = written[f"model.layers.{layer}.{norm}.weight"].float().unique().tolist()
            if got != [_fill_value(k, layer)]:
                wrong[f"{norm}[{layer}]"] = got
    assert wrong == {}
