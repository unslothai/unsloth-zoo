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

"""Tiled MLP must tile the per-instance mlp_forward transformers binds on MXFP4 GptOssMLP (unsloth-zoo#385)."""

import sys
import types
from types import MethodType

import torch

from unsloth_zoo import tiled_mlp


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x) + 1000.0


def _mlp_forward(self, x):
    return self.proj(x)


def _fake_mxfp4(monkeypatch):
    mod = types.ModuleType("transformers.integrations.mxfp4")
    mod.mlp_forward = _mlp_forward
    monkeypatch.setitem(sys.modules, "transformers.integrations.mxfp4", mod)


def test_tiles_the_instance_bound_mxfp4_forward(monkeypatch):
    _fake_mxfp4(monkeypatch)
    m = _MLP()
    m.forward = MethodType(_mlp_forward, m)
    x = torch.randn(2, 16, 8)
    want = _mlp_forward(m, x)
    tiled_mlp.patch_mlp(m, target_arctic=True)
    torch.testing.assert_close(m(x), want)
    tiled_mlp.patch_mlp(m, target_arctic=True)  # re-patch keeps it
    torch.testing.assert_close(m(x), want)


def test_other_modules_keep_the_class_forward(monkeypatch):
    _fake_mxfp4(monkeypatch)
    m = _MLP()
    x = torch.randn(2, 16, 8)
    want = _MLP.forward(m, x)
    tiled_mlp.patch_mlp(m, target_arctic=True)
    torch.testing.assert_close(m(x), want)
    assert m._unsloth_forward is _MLP.forward
