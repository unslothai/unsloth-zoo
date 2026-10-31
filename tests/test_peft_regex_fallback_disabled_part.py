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

"""finetune_vision_layers = False keeps a tagged vision tower out even when the decoder has no language tag (GIT)."""
import re

import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex


class _VisionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, nn.Linear(dim, dim))
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim)
        self.mlp.fc2 = nn.Linear(dim, dim)


class _BertSelf(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class _BertLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = _BertSelf(dim)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(dim, dim)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(dim, dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)


class _GitLike(nn.Module):
    """GitForCausalLM's layout: a vision tower tagged vision_model, a BERT-style decoder at git.encoder.layer.N."""
    def __init__(self, dim = 8, n = 2):
        super().__init__()
        self.git = nn.Module()
        self.git.image_encoder = nn.Module()
        self.git.image_encoder.vision_model = nn.Module()
        self.git.image_encoder.vision_model.encoder = nn.Module()
        self.git.image_encoder.vision_model.encoder.layers = nn.ModuleList(_VisionLayer(dim) for _ in range(n))
        self.git.encoder = nn.Module()
        self.git.encoder.layer = nn.ModuleList(_BertLayer(dim) for _ in range(n))
        self.output = nn.Linear(dim, 4)

    class config:
        _name_or_path = "test/git-like"
        model_type = "git"


def _targets(model, **kwargs):
    regex = get_peft_regex(model, **kwargs)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    # PEFT matches a string target_modules with re.fullmatch.
    return {n for n in linears if re.fullmatch(regex, n)}


def test_vision_off_keeps_vision_tower_out():
    targets = _targets(_GitLike(), finetune_vision_layers = False)
    assert not [n for n in targets if "vision_model" in n]
    assert "git.encoder.layer.0.attention.self.query" in targets


def test_vision_only_still_targets_vision_tower():
    model = _GitLike()
    targets = _targets(model, finetune_language_layers = False)
    tower = {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and "vision_model" in n}
    assert targets == tower


def test_defaults_still_target_vision_tower():
    targets = _targets(_GitLike())
    assert "git.image_encoder.vision_model.encoder.layers.0.self_attn.q_proj" in targets
