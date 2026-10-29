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

"""When the compiler swaps in the class that calls the input embedding (AST's ASTEmbeddings), the
requires_grad_ hook must sit on that class's output, outside torch.compile, not inside it."""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(4, 8, 1)

    def forward(self, x):
        return self.projection(x).transpose(1, 2)


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embeddings = _Patch()
        self.position = nn.Parameter(torch.zeros(1, 3, 8))

    def forward(self, x):
        return self.patch_embeddings(x) + self.position


_Embeddings.__module__ = "unsloth_compiled_module_fake"  # what the compiler's swapped classes look like
_Embeddings._compiled_forward = torch.compile(_Embeddings.forward, backend = "aot_eager", fullgraph = True)
_Embeddings.forward = lambda self, x: _Embeddings._compiled_forward(self, x)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(8, 8, bias = False)

    def forward(self, hidden_states):
        return hidden_states + self.query(hidden_states)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layers = nn.ModuleList([_Layer()])

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(self, x):
        hidden_states = self.embeddings(x)
        for layer in self.layers:
            hidden_states = checkpoint(layer, hidden_states, use_reentrant = True)
        return hidden_states


class _Top(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Model()

    def forward(self, x):
        return self.model(x)


def test_hook_moves_outside_the_compiled_embeddings():
    torch._dynamo.reset()
    top = _Top()
    for name, param in top.named_parameters():
        param.requires_grad_("query" in name)
    requires_grad_for_gradient_checkpointing(top)
    assert len(top.model.embeddings._forward_hooks) == 1
    assert len(top.model.embeddings.patch_embeddings._forward_hooks) == 0
    top(torch.randn(2, 4, 3)).sum().backward()
    grad = top.model.layers[0].query.weight.grad
    assert grad is not None and grad.abs().sum() > 0
