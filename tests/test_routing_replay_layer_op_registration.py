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

"""Layer input address op registers on torch >= 2.4, survives compilation; one op per module copy."""

import importlib.util
import sys

import pytest
import torch

import unsloth_zoo.routing_replay as rr


def test_layer_inputs_op_registers_and_returns_addresses():
    op = rr._layer_inputs_op()  # torch < 2.7: ValueError from infer_schema on list[torch.Tensor]
    x, y = torch.randn(3), torch.randn(2)
    # Distinct non-empty addresses, sorted, zero padded to one slot per tensor.
    out = op([x, torch.empty(0), x, y])
    assert out.dtype == torch.int64 and out.shape == (4,)
    assert out.tolist() == sorted([x.data_ptr(), y.data_ptr()]) + [0, 0]


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_layer_inputs_op_survives_a_compiled_graph(backend):
    # torch 2.4 Inductor dropped the earlier output-less op even when marked side effecting.
    op = rr._layer_inputs_op()
    torch._dynamo.reset()
    fn = torch.compile(lambda x: (x.sin(), op([x])), backend = backend, fullgraph = True)
    for n in (3, 3, 5):
        x = torch.randn(n, requires_grad = True)
        out, addresses = fn(x)
        out.sum().backward()
        assert addresses.tolist() == [x.data_ptr()]
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA graphs need a CUDA device")
def test_reduce_overhead_records_each_calls_addresses():
    # Under mode = "reduce-overhead" an untagged op either broke CUDA graph recording (CPU output)
    # or, replayed from the graph, kept the first call's addresses. Each call must see its own.
    op = rr._layer_inputs_op()
    torch._dynamo.reset()
    fn = torch.compile(lambda x: (x.sin(), op([x])), mode = "reduce-overhead", fullgraph = True)
    for _ in range(5):
        x = torch.randn(4, device = "cuda")
        out, addresses = fn(x)
        assert addresses.tolist() == [x.data_ptr()]
        out.clone()
    torch._dynamo.reset()


def test_second_module_copy_gets_its_own_op():
    first = rr._layer_inputs_op()
    spec = importlib.util.spec_from_file_location("_rr_second_copy", rr.__file__)
    copy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = copy
    try:
        spec.loader.exec_module(copy)
        second = copy._layer_inputs_op()
        assert str(second) != str(first)
        x = torch.randn(2)
        assert first([x]).tolist() == second([x]).tolist() == [x.data_ptr()]
    finally:
        sys.modules.pop(spec.name, None)
