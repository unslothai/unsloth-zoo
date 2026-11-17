# SPDX-License-Identifier: AGPL-3.0-only
"""A singleton tail must train under fullgraph/dynamic Inductor compilation."""
import copy

import pytest
import torch
from unsloth_zoo import tiling_registry as R


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))


@pytest.mark.parametrize("rows", [(10,), (9, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_registry_fullgraph_inductor_short_tail(monkeypatch, rows, dtype):
    # Exercise actual row tiling even when auto mode uses a whole-module checkpoint.
    monkeypatch.setenv("UNSLOTH_TILED_COMPILE", "unroll")
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = TinyMLP().to(dtype)
    ref = copy.deepcopy(model)
    R.install_module(model, "rows:4")
    compiled = torch.compile(model, backend = "inductor", fullgraph = True, dynamic = True)
    for n in rows:
        model.zero_grad(set_to_none = True)
        ref.zero_grad(set_to_none = True)
        x = torch.randn(1, n, 8, dtype = dtype, requires_grad = True)
        xr = x.detach().clone().requires_grad_()
        actual, expected = compiled(x), ref(xr)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol = 1e-4, atol = 1e-6)
        torch.testing.assert_close(x.grad, xr.grad, rtol = 1e-4, atol = 1e-6)
        for p, q in zip(model.parameters(), ref.parameters()):
            assert p.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol = 1e-4, atol = 1e-6)
