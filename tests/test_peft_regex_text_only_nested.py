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

"""Text-only decoders (model.layers.N) get the same LoRA leaves as model.language_model.layers.N."""
import re

import pytest
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, linear_attention):
        super().__init__()
        if linear_attention:
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = nn.Linear(8, 8)
            self.linear_attn.in_proj_z = nn.Linear(8, 8)
            self.linear_attn.out_proj = nn.Linear(8, 8)
        else:
            self.self_attn = nn.Module()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self.self_attn, n, nn.Linear(8, 8))
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(8, 4)
        self.mlp.shared_expert = nn.Module()
        for n in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp.shared_expert, n, nn.Linear(8, 8))
        self.ple = nn.Module()
        self.ple.key_proj = nn.Linear(8, 8)


def decoder():
    m = nn.Module()
    m.layers = nn.ModuleList([Block(True), Block(False)])
    return m


class TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = decoder()
        self.lm_head = nn.Linear(8, 16)


class Composite(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = decoder()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([nn.Module()])
        self.model.visual.blocks[0].attn = nn.Module()
        self.model.visual.blocks[0].attn.qkv = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 16)


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def selected(model, **kw):
    from unsloth_zoo.peft_utils import get_peft_regex
    regex = get_peft_regex(model, **kw)
    return {n for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and re.fullmatch(regex, n)}


def strip(names):
    return {n.replace("model.language_model.", "model.") for n in names}


@pytest.mark.parametrize("targets", [TARGETS, None])
def test_text_only_matches_composite(targets):
    kw = dict(finetune_vision_layers=False, finetune_language_layers=True,
              finetune_attention_modules=True, finetune_mlp_modules=True, target_modules=targets)
    text = selected(TextOnly(), **kw)
    comp = strip(selected(Composite(), **kw))
    assert text == comp
    assert "model.layers.1.mlp.shared_expert.down_proj" in text
    if targets is None:
        return
    assert "model.layers.0.linear_attn.in_proj_qkv" in text
    assert "model.layers.0.linear_attn.out_proj" in text
    assert "model.layers.1.mlp.shared_expert.down_proj" in text
    assert "model.layers.1.self_attn.q_proj" in text
    assert not any(n.endswith("mlp.gate") or "ple." in n or n == "lm_head" for n in text), text


def test_attention_only_scoping_still_scopes():
    kw = dict(finetune_vision_layers=False, finetune_language_layers=True,
              finetune_attention_modules=True, finetune_mlp_modules=False, target_modules=TARGETS)
    text = selected(TextOnly(), **kw)
    assert "model.layers.0.linear_attn.in_proj_qkv" in text
    assert not any(".mlp." in n for n in text), text
