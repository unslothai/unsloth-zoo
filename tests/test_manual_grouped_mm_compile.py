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

"""The per-group grouped GEMM fallback (CPU, devices or torch builds without torch._grouped_mm) inside a compiled
frame: no graph break, fullgraph traces, and outputs and gradients bit-identical to eager."""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M

needs_op = pytest.mark.skipif(
    not (hasattr(torch, "library") and hasattr(torch.library, "custom_op")),
    reason = "torch without torch.library.custom_op keeps the compile-disabled loop",
)


def _case(dtype, transposed):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(13, 16, generator = g).to(dtype).requires_grad_()
    if transposed:  # HF stores the stacks [E, out, in]; Zoo passes the [E, in, out] view
        w = torch.randn(4, 8, 16, generator = g).to(dtype).transpose(-2, -1).detach().requires_grad_()
    else:
        w = torch.randn(4, 16, 8, generator = g).to(dtype).requires_grad_()
    offsets = torch.tensor([3, 3, 9, 13], dtype = torch.int32)  # one empty group
    return x, w, offsets


def _run(fn, x, w, offsets):
    x = x.detach().clone().requires_grad_()
    w = w.detach().clone().requires_grad_()
    out = fn(x, w, offsets)
    (out.float() * torch.linspace(-1, 1, out.shape[-1])).sum().backward()
    return out.detach(), x.grad, w.grad


@needs_op
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_manual_grouped_mm_compiles_without_graph_break(dtype, transposed, backend):
    x, w, offsets = _case(dtype, transposed)
    torch._dynamo.reset()
    explained = torch._dynamo.explain(M._manual_grouped_mm)(x, w, offsets)
    assert explained.graph_break_count == 0, [str(b.reason) for b in explained.break_reasons]
    assert explained.graph_count == 1

    eager = _run(M._manual_grouped_mm, x, w, offsets)
    torch._dynamo.reset()
    compiled = _run(torch.compile(M._manual_grouped_mm, backend = backend, fullgraph = True), x, w, offsets)
    for got, want in zip(compiled, eager):
        assert got.dtype == want.dtype and got.shape == want.shape and got.stride() == want.stride()
        assert torch.equal(got, want)


@needs_op
def test_cpu_grouped_mm_fallback_compiles_without_graph_break():
    # The path MoE LoRA takes on a CPU: Zoo's shared grouped GEMM entry point, base and LoRA alike.
    x, w, offsets = _case(torch.bfloat16, True)
    assert not M._grouped_mm_supported_on(x.device)
    torch._dynamo.reset()
    explained = torch._dynamo.explain(M._grouped_mm_with_backward_fix)(x, w, offsets)
    assert explained.graph_break_count == 0, [str(b.reason) for b in explained.break_reasons]
    eager = _run(M._grouped_mm_with_backward_fix, x, w, offsets)
    torch._dynamo.reset()
    fn = torch.compile(M._grouped_mm_with_backward_fix, backend = "aot_eager", fullgraph = True)
    for got, want in zip(_run(fn, x, w, offsets), eager):
        assert torch.equal(got, want)


@needs_op
def test_compiled_backward_keeps_the_routing_guard():
    # The forward's routing signature rides as a saved tensor; a backward seeing other rows must refuse, as eager does.
    x, w, offsets = _case(torch.float32, False)
    out, routing = M._MANUAL_GROUPED_MM_OP(x.detach(), w.detach(), offsets)
    grad = torch.ones_like(out)
    op = torch.ops.unsloth_zoo.manual_grouped_mm_backward
    gx, gw = op(grad, x.detach(), w.detach(), offsets, routing, True, True)
    assert torch.isfinite(gx).all() and torch.isfinite(gw).all()
    swapped = x.detach().clone()
    swapped[[0, 12]] = swapped[[12, 0]]
    with pytest.raises(RuntimeError, match = "assigned tokens differently"):
        op(grad, swapped, w.detach(), offsets, routing, True, True)


def test_eager_call_is_unchanged():
    # Outside a compiled frame the compile-disabled autograd Function runs, as before.
    x, w, offsets = _case(torch.float32, False)
    out = M._manual_grouped_mm(x, w, offsets)
    assert type(out.grad_fn).__name__.startswith("_ManualGroupedMM")
    with torch.no_grad():
        assert torch.equal(M._manual_grouped_mm(x, w, offsets), M._grouped_matmul_loop(x, w, offsets))
