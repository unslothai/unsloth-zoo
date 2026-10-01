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

"""Dense torch.Tensor expert weights must not resolve to "torch.matmul_ogs" (broke native MXFP4 training)."""
import sys
import types

import torch

from unsloth_zoo.temporary_patches import gpt_oss


def _fake_copy(monkeypatch, root):
    pkg = types.ModuleType(root)
    pkg.__path__ = []
    routing = types.ModuleType(root + ".routing")
    tensor = types.ModuleType(root + ".tensor")
    mo = types.ModuleType(root + ".matmul_ogs")

    class RoutingData:
        pass

    class Tensor:
        pass

    RoutingData.__module__ = routing.__name__
    Tensor.__module__ = tensor.__name__
    routing.RoutingData, tensor.Tensor = RoutingData, Tensor
    mo.matmul_ogs = lambda *a, **k: root
    for m in (pkg, routing, tensor, mo):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    return RoutingData, Tensor, mo


def test_dense_weight_uses_routing_data_copy(monkeypatch):
    RoutingData, _, mo = _fake_copy(monkeypatch, "fake_vendored.triton_kernels")
    dense = torch.zeros(2, 4, 6, dtype = torch.bfloat16)
    assert gpt_oss._matmul_ogs_for(dense, RoutingData()) is mo.matmul_ogs


def test_dense_parameter_uses_routing_data_copy(monkeypatch):
    RoutingData, _, mo = _fake_copy(monkeypatch, "fake_tk_param")
    dense = torch.nn.Parameter(torch.zeros(2, 4, 6), requires_grad = False)
    assert gpt_oss._matmul_ogs_for(dense, RoutingData()) is mo.matmul_ogs


def test_dense_weight_without_routing_falls_back_to_resolved(monkeypatch):
    _, _, mo = _fake_copy(monkeypatch, "fake_tk_resolved")
    import unsloth_zoo.triton_kernels_compat as compat
    monkeypatch.setattr(compat, "get_triton_kernels", lambda: sys.modules["fake_tk_resolved"])
    assert gpt_oss._matmul_ogs_for(torch.zeros(1, 2, 2)) is mo.matmul_ogs


def test_triton_tensor_weight_keeps_its_own_copy(monkeypatch):
    _, Tensor, mo = _fake_copy(monkeypatch, "fake_tk_weight")
    OtherRouting, _, _ = _fake_copy(monkeypatch, "fake_tk_other")
    assert gpt_oss._matmul_ogs_for(Tensor(), OtherRouting()) is mo.matmul_ogs
