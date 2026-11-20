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

"""Checks moe_triton_kernels_available() is probed at import; compiled frames then never guard on an unprobed _TRITON or recompile after an eager call."""

import importlib.util
import sys

import pytest
import torch
from torch._dynamo.utils import counters

M = pytest.importorskip("unsloth_zoo.temporary_patches.moe_triton_kernels")


@pytest.fixture
def fresh_module(monkeypatch, request):
    name = f"_moe_triton_kernels_fresh_{request.node.name}"
    spec = importlib.util.spec_from_file_location(name, M.__file__)
    module = importlib.util.module_from_spec(spec)
    # Dynamo looks a traced global's module up in sys.modules.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_probe_runs_at_import(fresh_module):
    module = fresh_module
    assert module._TRITON is module._probe_triton()


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_compiled_frame_does_not_recompile_after_an_eager_call(fresh_module, device):
    module = fresh_module
    device = None if device is None else torch.device(device)
    torch._dynamo.reset()
    counters.clear()

    @torch.compile(backend = "eager", fullgraph = True)
    def f(x):
        return x + 1 if module.moe_triton_kernels_available(device) else x - 1

    x = torch.ones(4)
    first = f(x)
    expected = module.moe_triton_kernels_available(device)
    second = f(x)
    assert torch.equal(first, second)
    assert torch.equal(second, x + 1 if expected else x - 1)
    assert counters["stats"]["unique_graphs"] == 1
    assert not counters["recompiles"]
