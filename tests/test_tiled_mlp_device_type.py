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
"""TiledMLP's autocast decorators need a device string torch knows.

`DEVICE_TYPE` is "hip" on ROCm and "mlx" on Apple silicon with mlx installed; torch's amp
rejects both on the first decorated call, so every tiled forward died. The test re-imports
`tiled_mlp` under each label and runs a real tiled forward + backward on CPU tensors.
"""

import importlib

import pytest
import torch

import unsloth_zoo.device_type as device_type
from unsloth_zoo import tiled_mlp


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(16, 64)
        self.down = torch.nn.Linear(64, 16)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


@pytest.fixture
def reload_tiled_mlp():
    saved = (device_type.DEVICE_TYPE, device_type.DEVICE_TYPE_TORCH)

    def _reload(label, torch_label):
        device_type.DEVICE_TYPE = label
        device_type.DEVICE_TYPE_TORCH = torch_label
        return importlib.reload(tiled_mlp)

    yield _reload
    device_type.DEVICE_TYPE, device_type.DEVICE_TYPE_TORCH = saved
    importlib.reload(tiled_mlp)


# (DEVICE_TYPE, DEVICE_TYPE_TORCH) as unsloth_zoo.device_type translates them.
@pytest.mark.parametrize("label, torch_label", [("hip", "cuda"), ("mlx", "mps"), ("cuda", "cuda")])
def test_tiled_forward_and_backward_on_every_device_label(reload_tiled_mlp, label, torch_label):
    module = reload_tiled_mlp(label, torch_label)
    torch.manual_seed(0)
    reference = _MLP()
    tiled = _MLP()
    tiled.load_state_dict(reference.state_dict())
    module.patch_mlp(tiled, target_arctic = True)

    x = torch.randn(2, 40, 16, requires_grad = True)
    x_ref = x.detach().clone().requires_grad_()
    out = tiled(x)
    out_ref = reference(x_ref)
    out.pow(2).sum().backward()
    out_ref.pow(2).sum().backward()

    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    for p, q in zip(tiled.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
pass
