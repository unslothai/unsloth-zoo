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

"""Vision / audio towers have no token embedding: transformers' fallback `get_input_embeddings`
raises NotImplementedError there (Qwen2.5-VL `visual`, Gemma 4 `audio_tower`). The requires-grad
hook must take its pre-forward path quietly instead of warning at every LoRA setup."""

import logging

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


class _Tower(nn.Module):
    def __init__(self, error):
        super().__init__()
        self.error = error
        self.patch_embed = nn.Linear(8, 8, bias = False)
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])

    def get_input_embeddings(self):
        raise self.error

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        for blk in self.blocks:
            hidden_states = checkpoint(blk, hidden_states, use_reentrant = True)
        return hidden_states


class _Composite(nn.Module):
    def __init__(self, error):
        super().__init__()
        self.tower = _Tower(error)
        self.head = nn.Linear(8, 8, bias = False)

    def forward(self, pixel_values):
        return self.head(self.tower(pixel_values))


def _run(error, caplog):
    torch.manual_seed(0)
    model = _Composite(error)
    model.tower.patch_embed.weight.requires_grad_(False)
    with caplog.at_level(logging.INFO, logger = "unsloth_zoo.log"):
        requires_grad_for_gradient_checkpointing(model)
    model(torch.randn(2, 8)).sum().backward()
    for blk in model.tower.blocks:
        assert blk.proj.weight.grad is not None
        assert blk.proj.weight.grad.abs().sum() > 0
    return [r for r in caplog.records if "input-embedding hook" in r.getMessage() or "no input embeddings" in r.getMessage()]


def test_tower_without_embeddings_is_hooked_quietly(caplog):
    records = _run(NotImplementedError("`get_input_embeddings` not auto-handled"), caplog)
    assert records, "expected the info line naming the pre-forward hook"
    assert all(r.levelno < logging.WARNING for r in records)


def test_unexpected_failure_still_warns(caplog):
    records = _run(RuntimeError("boom"), caplog)
    assert any(r.levelno == logging.WARNING and "boom" in r.getMessage() for r in records)
