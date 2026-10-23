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

"""A tower fed only token ids with no get_input_embeddings (Dia's text encoder) must still get
gradients under reentrant checkpointing: its own Embedding output is made to require grad."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return hidden_states + self.q_proj(hidden_states)


class _Encoder(nn.Module):
    """Like DiaEncoder: token ids in, `self.embedding`, no get_input_embeddings."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = checkpoint(layer, hidden_states, use_reentrant = True)
        return hidden_states


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()

    def forward(self, input_ids):
        return self.encoder(input_ids)


def test_token_id_tower_gets_gradients_under_reentrant_checkpointing():
    torch.manual_seed(0)
    model = _Model()
    for name, param in model.named_parameters():
        param.requires_grad_("q_proj" in name)
    requires_grad_for_gradient_checkpointing(model)
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    grads = [p.grad for n, p in model.named_parameters() if "q_proj" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
    requires_grad_for_gradient_checkpointing(model)  # idempotent: hooks are replaced, not stacked
    assert len(model.encoder.embedding._forward_hooks) == 1
