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

"""Released Unsloth imports the old TiledMLP names at module level; an upgraded Unsloth Zoo must keep them working."""

import os
import runpy

import torch

from unsloth_zoo.tiled_mlp import (
    TiledMLP,
    _mxfp4_instance_forward,
    is_custom_module,
    torch_amp_custom_bwd,
    torch_amp_custom_fwd,
)

RELEASED_DDP = os.path.join(os.path.dirname(__file__), "released_unsloth", "_tiled_mlp_ddp.py")


class _MLP(torch.nn.Module):
    def __init__(self, h = 8):
        super().__init__()
        self.up = torch.nn.Linear(h, 4 * h)
        self.down = torch.nn.Linear(4 * h, h)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


def _grads(m, x, apply):
    m.zero_grad()
    xr = x.detach().clone().requires_grad_()
    y = apply(m.forward, m, xr, False, 3, 4) if apply else m(xr)
    y.pow(2).sum().backward()
    return xr.grad, [p.grad.clone() for p in m.parameters()]


def test_legacy_names_import():
    assert callable(torch_amp_custom_fwd) and callable(torch_amp_custom_bwd)
    assert is_custom_module("model.layers.0.mlp", ["mlp"])
    assert _mxfp4_instance_forward(_MLP()) is None


def test_released_ddp_patch_runs_against_the_shim():
    torch.manual_seed(0)
    m = _MLP()
    x = torch.randn(1, 12, 8)
    ref = _grads(m, x, None)
    original_apply = TiledMLP.__dict__["apply"] if "apply" in TiledMLP.__dict__ else None
    try:
        released = runpy.run_path(RELEASED_DDP)
        tiled = _grads(m, x, TiledMLP.apply)
        torch.testing.assert_close(tiled[0], ref[0])
        for a, b in zip(tiled[1], ref[1]):
            torch.testing.assert_close(a, b)
        released["patch_tiled_mlp_for_ddp"]()
        patched = _grads(m, x, TiledMLP.apply)
        torch.testing.assert_close(patched[0], ref[0])
        for a, b in zip(patched[1], ref[1]):
            torch.testing.assert_close(a, b)
    finally:
        if original_apply is None:
            if "apply" in TiledMLP.__dict__:
                delattr(TiledMLP, "apply")
        else:
            TiledMLP.apply = original_apply
