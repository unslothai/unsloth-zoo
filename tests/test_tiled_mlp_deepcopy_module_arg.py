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
"""A deepcopied tiled MLP must pass itself, not the closure-captured original, to TiledMLP.apply, since the DDP shim
takes grads for mlp_module.parameters(); the original would leave the copy without grads."""
import copy

import pytest
import torch

from unsloth_zoo import tiled_mlp
from unsloth_zoo.tiled_mlp import patch_mlp

HIDDEN = 16


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(HIDDEN, 4 * HIDDEN, bias = False)
        self.down_proj = torch.nn.Linear(4 * HIDDEN, HIDDEN, bias = False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.up_proj(x)))
pass


class _ParamsAsInputs(torch.autograd.Function):
    # Same contract as Unsloth's DDP shim: gradients only for mlp_module's parameters.
    @staticmethod
    def forward(ctx, mlp_forward, x, *params):
        ctx.mlp_forward, ctx.params = mlp_forward, params
        ctx.save_for_backward(x)
        with torch.no_grad():
            return mlp_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0].detach().requires_grad_(True)
        with torch.enable_grad():
            out = ctx.mlp_forward(x)
        grads = torch.autograd.grad(out, (x, *ctx.params), grad_output, allow_unused = True)
        return (None, *grads)
pass


@pytest.fixture
def params_as_inputs_apply(monkeypatch):
    seen = []

    def apply(mlp_forward, mlp_module, x, preserve_rng_state, num_shards, max_flat_qlen):
        seen.append(mlp_module)
        params = [p for p in mlp_module.parameters() if p.requires_grad]
        return _ParamsAsInputs.apply(mlp_forward, x, *params)

    # The class patch_mlp resolves now, not the one imported at collection (a reload swaps it).
    monkeypatch.setattr(tiled_mlp.TiledMLP, "apply", staticmethod(apply))
    return seen
pass


@pytest.mark.parametrize("target_arctic", [True, False])
def test_deepcopy_passes_running_module_to_apply(target_arctic, params_as_inputs_apply):
    torch.manual_seed(0)
    original = _MLP()
    patch_mlp(original, target_arctic = target_arctic, target_gb = 1e-6, padded_length = HIDDEN)
    copied = copy.deepcopy(original)
    reference = _MLP()
    reference.load_state_dict(copied.state_dict())

    x = torch.randn(2, 32, HIDDEN)
    copied(x).pow(2).sum().backward()
    reference(x).pow(2).sum().backward()

    assert params_as_inputs_apply[-1] is copied
    for (name, got), want in zip(copied.named_parameters(), reference.parameters()):
        assert got.grad is not None, name
        torch.testing.assert_close(got.grad, want.grad)
    assert all(p.grad is None for p in original.parameters())
pass


@pytest.mark.parametrize("target_arctic", [True, False])
def test_original_still_passes_itself(target_arctic, params_as_inputs_apply):
    torch.manual_seed(0)
    original = _MLP()
    patch_mlp(original, target_arctic = target_arctic, target_gb = 1e-6, padded_length = HIDDEN)
    original(torch.randn(2, 32, HIDDEN)).sum().backward()
    assert params_as_inputs_apply[-1] is original
    assert all(p.grad is not None for p in original.parameters())
pass
