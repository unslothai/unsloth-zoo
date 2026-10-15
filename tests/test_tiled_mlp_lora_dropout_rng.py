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
"""TiledMLP must replay dropout masks when PEFT adds lora_dropout after patching (CPU only).

Unsloth patches MLPs inside from_pretrained, before get_peft_model injects LoRA and its
lora_dropout. A patch time scan found no dropout, so the backward recompute drew fresh masks
and the gradients belonged to a different function than the one the forward computed.
"""
import copy

import pytest
import torch

from unsloth_zoo.tiled_mlp import patch_mlp

peft = pytest.importorskip("peft")

HIDDEN = 16


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(HIDDEN, 4 * HIDDEN, bias = False)
        self.down_proj = torch.nn.Linear(4 * HIDDEN, HIDDEN, bias = False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.up_proj(x)))
pass


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x):
        return self.mlp(x)
pass


def _lora(model, dropout):
    config = peft.LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = dropout,
        target_modules = ["up_proj", "down_proj"], init_lora_weights = False,
    )
    return peft.inject_adapter_in_model(config, model)


def _grads(model):
    return {n : p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("target_arctic", [True, False])
def test_lora_dropout_added_after_patch_gets_matching_grads(target_arctic):
    torch.manual_seed(0)
    base = _Block()
    tiled = copy.deepcopy(base)
    # Patch first, as from_pretrained does, then add LoRA with dropout.
    patch_mlp(tiled.mlp, target_arctic = target_arctic, target_gb = 1e-6, padded_length = HIDDEN)
    torch.manual_seed(1)
    tiled = _lora(tiled, 0.1)
    torch.manual_seed(1)
    reference = _lora(base, 0.1)
    reference.load_state_dict(tiled.state_dict())
    tiled.train(); reference.train()

    torch.manual_seed(2)
    x = torch.randn(2, 32, HIDDEN)
    grad_out = torch.randn(2, 32, HIDDEN)

    # Reference: same tiles, same RNG order, ordinary autograd (no recompute).
    torch.manual_seed(3)
    x_ref = x.clone().requires_grad_(True)
    tiles = [reference.mlp(t.unsqueeze(0)) for t in torch.split(x_ref.view(-1, HIDDEN), HIDDEN, dim = 0)]
    out_ref = torch.cat(tiles, dim = 1).view_as(x)
    out_ref.backward(grad_out)

    torch.manual_seed(3)
    x_tiled = x.clone().requires_grad_(True)
    out_tiled = tiled.mlp(x_tiled)
    out_tiled.backward(grad_out)

    torch.testing.assert_close(out_tiled, out_ref)
    torch.testing.assert_close(x_tiled.grad, x_ref.grad)
    ref_grads, tiled_grads = _grads(reference), _grads(tiled)
    assert ref_grads.keys() == tiled_grads.keys() and any("lora" in n for n in ref_grads)
    for name in ref_grads:
        torch.testing.assert_close(tiled_grads[name], ref_grads[name], msg = name)
pass


def test_no_dropout_skips_rng_state():
    from unsloth_zoo.tiled_mlp import _mlp_needs_rng_state
    mlp = _MLP()
    patch_mlp(mlp, target_arctic = True)
    assert not _mlp_needs_rng_state(mlp)
    holder = _lora(_Block(), 0.0)
    assert not _mlp_needs_rng_state(holder.mlp)
    holder = _lora(_Block(), 0.1)
    assert _mlp_needs_rng_state(holder.mlp)
    holder.eval()
    assert not _mlp_needs_rng_state(holder.mlp)
pass
