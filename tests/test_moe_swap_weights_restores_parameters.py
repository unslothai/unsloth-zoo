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

"""swap_moe_weights_for_call (the dequantize-then-dispatch path of the bnb 4-bit and FP8 MoE
backends) must leave the experts module's parameters registered exactly as it found them.

It installs the dense weights with object.__setattr__, which writes the instance __dict__. Writing
the original Parameter back the same way left it in __dict__, shadowing `_parameters`: FSDP2 swaps
`_parameters[name]` between the sharded and the all-gathered parameter around every forward, so
after one call the module kept returning a stale all-gather buffer (`'Params4bit' object has no
attribute '_local_tensor'` in FSDP2's lazy init, or freed storage in the next forward).
"""

import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches.moe_utils import swap_moe_weights_for_call


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4), requires_grad = False)
        self.down_proj = nn.Parameter(torch.zeros(2, 4, 4), requires_grad = False)


def test_swap_does_not_shadow_registered_parameters():
    experts = _Experts()
    dense_gu, dense_dn = torch.ones(2, 8, 4), torch.ones(2, 4, 4)
    seen = {}

    def forward_fn(module, x):
        seen["gu"] = module.gate_up_proj
        seen["dn"] = module.down_proj
        return x + 1

    out = swap_moe_weights_for_call(experts, dense_gu, dense_dn, forward_fn, torch.zeros(1))
    assert out.item() == 1
    assert seen["gu"] is dense_gu and seen["dn"] is dense_dn
    assert "gate_up_proj" not in experts.__dict__
    assert "down_proj" not in experts.__dict__
    assert set(dict(experts.named_parameters())) == {"gate_up_proj", "down_proj"}
    # What FSDP2's unsafe_setattr_param does around each forward: replace the registered Parameter.
    replacement = nn.Parameter(torch.full((2, 8, 4), 3.0), requires_grad = False)
    experts._parameters["gate_up_proj"] = replacement
    assert experts.gate_up_proj is replacement


def test_swap_restores_plain_instance_attributes_and_raises_through():
    experts = nn.Module()
    plain_gu, plain_dn = torch.zeros(1), torch.zeros(2)
    experts.gate_up_proj = plain_gu  # a plain tensor attribute lives in __dict__
    experts.down_proj = plain_dn

    def boom(module, x):
        assert module.gate_up_proj is not plain_gu
        raise RuntimeError("inner")

    try:
        swap_moe_weights_for_call(experts, torch.ones(1), torch.ones(2), boom, None)
    except RuntimeError as error:
        assert str(error) == "inner"
    else:
        raise AssertionError("expected the forward's error")
    assert experts.__dict__["gate_up_proj"] is plain_gu
    assert experts.__dict__["down_proj"] is plain_dn


import pytest


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_swap_traces_fullgraph_and_restores_under_compile(backend, dynamic):
    # fullgraph=True on every supported torch (2.10 rejected the old `__dict__.pop` restore).
    from torch._dynamo.utils import counters

    experts = _Experts()
    original_gu = experts._parameters["gate_up_proj"]
    original_dn = experts._parameters["down_proj"]

    def forward_fn(module, x):
        return (x @ module.gate_up_proj[0].T).sum() + module.down_proj.sum()

    @torch.compile(backend = backend, fullgraph = True, dynamic = dynamic)
    def call(x, gu, dn):
        return swap_moe_weights_for_call(experts, gu, dn, forward_fn, x) * 2

    torch._dynamo.reset()
    counters.clear()
    try:
        for rows in (3, 5, 9):
            gu = torch.randn(2, 8, 4, requires_grad = True)
            dn = torch.randn(2, 4, 4, requires_grad = True)
            x = torch.randn(rows, 4)
            out = call(x, gu, dn)
            torch.testing.assert_close(out, 2 * ((x @ gu[0].T).sum() + dn.sum()))
            out.backward()
            assert gu.grad is not None and dn.grad is not None
            assert "gate_up_proj" not in experts.__dict__
            assert "down_proj" not in experts.__dict__
            assert experts._parameters["gate_up_proj"] is original_gu
            assert experts._parameters["down_proj"] is original_dn
            assert experts.gate_up_proj is original_gu and experts.down_proj is original_dn
        breaks = dict(counters["graph_break"])
    finally:
        torch._dynamo.reset()
    assert sum(breaks.values()) == 0, breaks


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("fullgraph", [False, True])
def test_compiled_swap_keeps_parameters_registered(backend, fullgraph):
    # Dynamo on torch 2.12 replays a traced `__dict__.pop(name)` as DELETE_ATTR, and
    # nn.Module.__delattr__ then unregisters the Parameter: the first compiled call returned the
    # right value but left `_parameters` empty, and the module lost both expert weights.
    experts = _Experts()
    originals = dict(experts._parameters)

    def forward_fn(module, x):
        return x * module.gate_up_proj.sum() + module.down_proj.sum()

    call = torch.compile(
        lambda x: swap_moe_weights_for_call(experts, torch.ones(2, 8, 4), torch.ones(2, 4, 4), forward_fn, x),
        backend = backend,
        fullgraph = fullgraph,
    )
    torch._dynamo.reset()
    try:
        for _ in range(3):
            out = call(torch.ones(1))
            assert out.item() == 64 + 32
            assert list(experts._parameters) == ["gate_up_proj", "down_proj"]
            assert all(experts._parameters[n] is p for n, p in originals.items())
            assert set(dict(experts.named_parameters())) == set(originals)
            assert "gate_up_proj" not in experts.__dict__ and "down_proj" not in experts.__dict__
    finally:
        torch._dynamo.reset()
