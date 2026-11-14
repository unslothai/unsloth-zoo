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

"""Qwen4Exp (Qwen3.8-Flash-Next) auto LoRA targets: QSA indexer out, PLE and the hyper-connection mixer in.

Module trees mirror transformers' Qwen4ExpForConditionalGeneration / Qwen4ExpForCausalLM names, so the
test runs without a transformers that ships qwen4_exp.
"""
import re
from types import SimpleNamespace

import pytest
from torch import nn


def hc():
    m = nn.Module()
    m.input_mix_weight_down = nn.Linear(8, 4)
    m.input_mix_weight_up = nn.Linear(4, 8)
    m.block_inject_weight = nn.Linear(8, 2)
    return m


class Block(nn.Module):
    def __init__(self, linear_attention, ple):
        super().__init__()
        if linear_attention:
            self.linear_attn = nn.Module()
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                setattr(self.linear_attn, n, nn.Linear(8, 8))
        else:
            self.self_attn = nn.Module()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self.self_attn, n, nn.Linear(8, 8))
            self.self_attn.indexer = nn.Module()
            self.self_attn.indexer.index_qk_proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.shared_expert = nn.Module()
        for n in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp.shared_expert, n, nn.Linear(8, 8))
        self.mlp.shared_expert_gate = nn.Linear(8, 1)
        self.attn_hyper_connection = hc()
        self.mlp_hyper_connection = hc()
        if ple:
            self.ple = nn.Module()
            self.ple.key_proj = nn.Linear(8, 16)
            self.ple.value_proj = nn.Linear(8, 8)


def decoder():
    m = nn.Module()
    m.layers = nn.ModuleList([Block(True, False), Block(True, True), Block(True, False), Block(False, False)])
    m.hyper_connection_mixer = hc()
    del m.hyper_connection_mixer.block_inject_weight
    return m


def text_config(model_type="qwen4_exp_text"):
    return SimpleNamespace(model_type=model_type)


class TextOnly(nn.Module):
    def __init__(self, model_type="qwen4_exp_text"):
        super().__init__()
        self.config = text_config(model_type)
        self.model = decoder()
        self.lm_head = nn.Linear(8, 16)


class Composite(nn.Module):
    def __init__(self, model_type="qwen4_exp"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, text_config=text_config(model_type + "_text"))
        self.model = nn.Module()
        self.model.language_model = decoder()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([nn.Module()])
        self.model.visual.blocks[0].attn = nn.Module()
        self.model.visual.blocks[0].attn.qkv = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 16)


def selected(model, **kw):
    from unsloth_zoo.peft_utils import get_peft_regex
    regex = get_peft_regex(model, **kw)
    return {n for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and re.fullmatch(regex, n)}


def linears(model):
    return {n for n, mod in model.named_modules() if isinstance(mod, nn.Linear)}


@pytest.mark.parametrize("cls", [TextOnly, Composite])
def test_auto_targets_cover_every_linear_but_indexer_and_head(cls, capsys):
    model = cls()
    got = selected(model)
    expected = {n for n in linears(model) if ".indexer." not in n and n != "lm_head"}
    assert got == expected, sorted(expected ^ got)
    assert "1 QSA indexer projection" in capsys.readouterr().out


def test_explicit_indexer_target_still_opts_in():
    got = selected(TextOnly(), target_modules=["q_proj", "index_qk_proj"])
    assert got == {"model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.indexer.index_qk_proj"}


def test_explicit_list_scopes_ple_and_mixer():
    got = selected(TextOnly(), target_modules=["q_proj", "key_proj"])
    assert got == {"model.layers.3.self_attn.q_proj", "model.layers.1.ple.key_proj"}


def test_mlp_off_keeps_ple_out():
    got = selected(TextOnly(), finetune_mlp_modules=False)
    assert not any(".ple." in n or ".mlp." in n for n in got), got
    assert "model.hyper_connection_mixer.input_mix_weight_down" in got


@pytest.mark.parametrize("cls,model_type", [(TextOnly, "qwen3_5_moe_text"), (Composite, "qwen3_5_moe")])
def test_other_model_types_unchanged(cls, model_type):
    # Same tree under another model_type: no indexer exclusion and no PLE / mixer branches.
    got = selected(cls(model_type))
    assert any(".indexer." in n for n in got)
    assert not any(".ple." in n for n in got)
