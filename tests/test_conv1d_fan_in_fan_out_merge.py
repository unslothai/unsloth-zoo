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

"""A LoRA adapter on a Conv1D (GPT-2) sets fan_in_fan_out and the base weight is stored (in, out), so the delta has to be
added transposed, as PEFT's own merge does. Rectangular layers used to crash with an in-place shape error and square ones
got a silently wrong delta."""
import copy

import pytest
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from unsloth_zoo.saving_utils import create_lora_statistics, _merge_lora


class _Tiny(nn.Module):
    def __init__(self, d=16, d_ff=24):
        super().__init__()
        self.c_attn = Conv1D(3 * d, d)   # weight (d, 3d)
        self.c_proj = Conv1D(d, d)       # square, the case that used to merge silently wrong
        self.c_fc = Conv1D(d_ff, d)      # weight (d, d_ff)

    def forward(self, x):
        return self.c_proj(self.c_fc(x)[..., :16]) + self.c_attn(x)[..., :16]


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("targets", [["c_proj"], ["c_attn", "c_proj", "c_fc"]])
def test_conv1d_merge_matches_peft(use_dora, targets):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = _Tiny()
    base_weights = {k: v.detach().clone() for k, v in base.state_dict().items()}
    cfg = LoraConfig(
        r=4, lora_alpha=8, use_dora=use_dora, fan_in_fan_out=True,
        target_modules=targets,
    )
    pm = get_peft_model(copy.deepcopy(base), cfg)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.1)
            if "lora_magnitude_vector" in n:
                p.add_(torch.randn_like(p) * 0.1)

    merged = copy.deepcopy(pm).merge_and_unload().state_dict()
    lora_weights, _ = create_lora_statistics(pm, merge_into_original=True, return_state_dict=True)
    checked = 0
    for key, stats in lora_weights.items():
        if stats.lora_A is None or stats.lora_B is None:
            continue
        W0 = base_weights[key + ".weight"]
        W = _merge_lora(W0.clone(), stats, key).cpu().float()
        assert W.shape == W0.shape, f"{key}: merged {tuple(W.shape)} vs stored {tuple(W0.shape)}"
        assert torch.allclose(W, merged[key + ".weight"], atol=1e-5, rtol=1e-5), (
            f"{key}: max abs diff {(W - merged[key + '.weight']).abs().max().item()}"
        )
        checked += 1
    assert checked == len(targets)


def test_conv1d_merge_end_to_end(tmp_path):
    """The public export path on a real GPT-2 matches PEFT's merge_and_unload."""
    import _merge_e2e_helpers as H
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    H.set_offline_cpu_env()
    torch.manual_seed(0)
    config = GPT2Config(
        n_embd=32, n_layer=1, n_head=2, vocab_size=64, n_positions=32, bos_token_id=1, eos_token_id=2,
    )
    base = GPT2LMHeadModel(config).eval()
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "out")
    base.save_pretrained(base_dir)
    pm = get_peft_model(copy.deepcopy(base), LoraConfig(
        r=4, lora_alpha=8, fan_in_fan_out=True, target_modules=["c_attn", "c_proj", "c_fc"],
    ))
    H.seed_lora(pm)
    expected = copy.deepcopy(pm).merge_and_unload().state_dict()

    H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)
    got = H.read_safetensors_dir(out_dir)
    n_adapted = 0
    for key, tensor in got.items():
        if key not in expected:
            continue
        assert tensor.shape == expected[key].shape, key
        assert torch.allclose(tensor, expected[key], atol=1e-5, rtol=1e-5), (
            f"{key}: max abs diff {(tensor - expected[key]).abs().max().item()}"
        )
        n_adapted += key.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight"))
    assert n_adapted == 4
