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

"""Automatic LoRA targets of a Mamba-family model leave out out_proj and conv1d, which PEFT refuses."""
import re
from types import SimpleNamespace

import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex, _mamba_only_leaves


class _Mixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.conv1d = nn.Conv1d(dim, dim, 3)


class _Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)


class _Layer(nn.Module):
    def __init__(self, dim, mixer):
        super().__init__()
        self.mixer = _Mixer(dim) if mixer else _Mlp(dim)


class _Model(nn.Module):
    def __init__(self, model_type, dim = 8, layers = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer(dim, i % 2 == 0) for i in range(layers))
        self.config = SimpleNamespace(model_type = model_type, _name_or_path = "test/mamba-leaves")


def _matched(model, regex):
    return {n for n, _ in model.named_modules() if n and re.fullmatch(regex, n, flags = re.DOTALL)}


def test_nemotron_h_leaves_out_proj_and_conv1d_out():
    model = _Model("nemotron_h")
    matched = _matched(model, get_peft_regex(model))
    assert matched
    assert not any(n.endswith(".out_proj") or n.endswith(".conv1d") for n in matched), sorted(matched)
    assert "model.layers.0.mixer.in_proj" in matched
    assert "model.layers.1.mixer.up_proj" in matched and "model.layers.1.mixer.down_proj" in matched


def test_every_mamba_family_answers_the_same():
    for model_type in ("mamba", "mamba2", "falcon_mamba", "falcon_h1", "nemotron_h"):
        assert _mamba_only_leaves(_Model(model_type)) == frozenset(("out_proj", "conv1d")), model_type


def test_other_models_keep_out_proj():
    model = _Model("llama")
    assert _mamba_only_leaves(model) == frozenset()
    matched = _matched(model, get_peft_regex(model))
    assert "model.layers.0.mixer.out_proj" in matched


def test_wrapped_mamba_language_model_is_recognised():
    model = _Model("omni-wrapper")
    model.config.llm_config = SimpleNamespace(model_type = "nemotron_h")
    assert _mamba_only_leaves(model) == frozenset(("out_proj", "conv1d"))


def test_explicit_target_modules_are_the_caller_s_decision():
    model = _Model("nemotron_h")
    regex = get_peft_regex(model, target_modules = ["in_proj", "out_proj"])
    assert "model.layers.0.mixer.out_proj" in _matched(model, regex)


def test_model_without_config_is_untouched():
    assert _mamba_only_leaves(nn.Linear(2, 2)) == frozenset()


class _Tower(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)


class _Wrapper(nn.Module):
    def __init__(self, dim = 8):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList(nn.Module() for _ in range(2))
        for layer in self.vision_model.encoder.layers:
            layer.self_attn = _Tower(dim)
        self.language_model = _Model("nemotron_h", dim = dim).model
        self.config = SimpleNamespace(
            model_type = "omni_wrapper",
            llm_config = SimpleNamespace(model_type = "nemotron_h"),
            _name_or_path = "test/wrapped-mamba",
        )


def test_nested_mamba_excludes_only_the_mixer_out_proj():
    model = _Wrapper()
    regex = get_peft_regex(model, finetune_vision_layers = True, finetune_language_layers = True)
    matched = _matched(model, regex)
    assert "vision_model.encoder.layers.0.self_attn.out_proj" in matched
    assert "vision_model.encoder.layers.1.self_attn.out_proj" in matched
    assert not any(".mixer.out_proj" in n or ".mixer.conv1d" in n for n in matched), sorted(matched)
    assert "language_model.layers.0.mixer.in_proj" in matched
