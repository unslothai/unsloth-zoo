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
"""Deepcopied tiled MLP decides RNG preservation from its own dropout state; deepcopy keeps the closure over the
original module, so an eval original with a training copy broke recompute masks; CPU only."""
import copy

import pytest
import torch

from unsloth_zoo.tiled_mlp import patch_mlp

HIDDEN = 16


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(HIDDEN, 4 * HIDDEN, bias = False)
        self.dropout = torch.nn.Dropout(0.25)
        self.down_proj = torch.nn.Linear(4 * HIDDEN, HIDDEN, bias = False)

    def forward(self, x):
        return self.down_proj(self.dropout(torch.nn.functional.gelu(self.up_proj(x))))
pass


@pytest.mark.parametrize("target_arctic", [True, False])
def test_training_deepcopy_of_eval_original_replays_dropout(target_arctic):
    torch.manual_seed(0)
    original = _MLP()
    patch_mlp(original, target_arctic = target_arctic, target_gb = 1e-6, padded_length = HIDDEN)
    original.eval()
    copied = copy.deepcopy(original).train()
    reference = _MLP()
    reference.load_state_dict(copied.state_dict())
    reference.train()

    torch.manual_seed(2)
    x = torch.randn(2, 32, HIDDEN)
    grad_out = torch.randn(2, 32, HIDDEN)

    # Reference: same tiles, same RNG order, ordinary autograd (no recompute).
    torch.manual_seed(3)
    x_ref = x.clone().requires_grad_(True)
    tiles = [reference(t.unsqueeze(0)) for t in torch.split(x_ref.view(-1, HIDDEN), HIDDEN, dim = 0)]
    out_ref = torch.cat(tiles, dim = 1).view_as(x)
    out_ref.backward(grad_out)

    torch.manual_seed(3)
    x_copy = x.clone().requires_grad_(True)
    out_copy = copied(x_copy)
    out_copy.backward(grad_out)

    torch.testing.assert_close(out_copy, out_ref)
    torch.testing.assert_close(x_copy.grad, x_ref.grad)
    for (name, p_ref), (_, p_copy) in zip(reference.named_parameters(), copied.named_parameters()):
        torch.testing.assert_close(p_copy.grad, p_ref.grad, msg = name)
pass
