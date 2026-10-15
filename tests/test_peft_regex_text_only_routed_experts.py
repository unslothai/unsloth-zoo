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

"""The text-only decoder branch reaches shared experts but not routed ones.

Routed experts (mlp.experts.<N>.<leaf>) were never a default target on model.layers.N:
unsloth adds them itself for remote expert blocks and keeps native ones (Qwen3-MoE on
transformers 4.x) off LoRA on every routed expert, and it only widens when the generated
regex misses them. Reaching nested leaves for shared experts must not pull them in.
"""
import re

import pytest
from torch import nn


def _expert():
    e = nn.Module()
    for n in ("gate_proj", "up_proj", "down_proj"):
        setattr(e, n, nn.Linear(8, 8))
    return e


def _moe_decoder(mixer_name):
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList()
    for _ in range(2):
        block = nn.Module()
        block.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, n, nn.Linear(8, 8))
        moe = nn.Module()
        moe.gate = nn.Linear(8, 4)
        moe.experts = nn.ModuleList([_expert() for _ in range(4)])
        moe.shared_experts = _expert()
        setattr(block, mixer_name, moe)
        m.model.layers.append(block)
    m.lm_head = nn.Linear(8, 16)
    return m


def _selected(model, **kw):
    from unsloth_zoo.peft_utils import get_peft_regex
    regex = get_peft_regex(model, finetune_vision_layers = False, **kw)
    return {n for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and re.fullmatch(regex, n)}


@pytest.mark.parametrize("mixer_name", ["mlp", "mixer"])
@pytest.mark.parametrize("targets", [None, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]])
def test_routed_experts_stay_out_of_the_text_only_branch(mixer_name, targets):
    model = _moe_decoder(mixer_name)
    selected = _selected(model, target_modules = targets)
    assert not any(".experts." in n for n in selected), sorted(selected)
    assert "model.layers.0.self_attn.q_proj" in selected
    if mixer_name == "mlp":
        assert f"model.layers.1.{mixer_name}.shared_experts.down_proj" in selected
