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

"""Vision towers looping `for i, blk in enumerate(self.blocks):` (Qwen3-VL, Qwen3-Omni) need the
requires-grad hook: fed by a frozen patch embedding, reentrant-checkpointed blocks get no gradient."""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return hidden_states + self.proj(hidden_states)


class _EnumerateTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(8, 8, bias = False)
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = checkpoint(blk, hidden_states, use_reentrant = True)
        return hidden_states


class _EnumerateSlicedTower(_EnumerateTower):
    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        for i, blk in enumerate(self.blocks[: len(self.blocks)]):
            hidden_states = checkpoint(blk, hidden_states, use_reentrant = True)
        return hidden_states


class _PlainTower(_EnumerateTower):
    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        for blk in self.blocks:
            hidden_states = checkpoint(blk, hidden_states, use_reentrant = True)
        return hidden_states


class _Composite(nn.Module):
    # Tower reached only via a helper, never as `self.tower(` in forward.
    def __init__(self, tower_cls):
        super().__init__()
        self.tower = tower_cls()
        self.head = nn.Linear(8, 8, bias = False)

    def get_image_features(self, pixel_values):
        return self.tower(pixel_values)

    def forward(self, pixel_values):
        return self.head(self.get_image_features(pixel_values))


@pytest.mark.parametrize("tower_cls", [_EnumerateTower, _EnumerateSlicedTower, _PlainTower])
def test_every_tower_block_gets_a_gradient(tower_cls):
    torch.manual_seed(0)
    model = _Composite(tower_cls)
    model.tower.patch_embed.weight.requires_grad_(False)
    requires_grad_for_gradient_checkpointing(model)
    model(torch.randn(2, 8)).sum().backward()
    for blk in model.tower.blocks:
        assert blk.proj.weight.grad is not None
        assert blk.proj.weight.grad.abs().sum() > 0


def test_unhooked_enumerate_tower_really_loses_the_gradient():
    torch.manual_seed(0)
    model = _Composite(_EnumerateTower)
    model.tower.patch_embed.weight.requires_grad_(False)
    with pytest.warns(UserWarning):
        model(torch.randn(2, 8)).sum().backward()
    assert all(blk.proj.weight.grad is None for blk in model.tower.blocks)
