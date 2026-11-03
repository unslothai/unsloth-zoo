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
frame. With no gradient through it: one opaque op, no graph break, outputs and autocast dtypes as eager. With a gradient:
the guarded eager path, so a checkpoint replay that routes differently still raises."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.temporary_patches import moe_utils as M

needs_op = pytest.mark.skipif(
    not (hasattr(torch, "library") and hasattr(torch.library, "custom_op")),
    reason = "torch without torch.library.custom_op keeps the compile-disabled loop",
)


def _case(dtype, transposed, w_dtype = None):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(13, 16, generator = g).to(dtype)
    w_dtype = w_dtype or dtype
    if transposed:  # HF stores the stacks [E, out, in]; Zoo passes the [E, in, out] view
        w = torch.randn(4, 8, 16, generator = g).to(w_dtype).transpose(-2, -1)
    else:
        w = torch.randn(4, 16, 8, generator = g).to(w_dtype)
    offsets = torch.tensor([3, 3, 9, 13], dtype = torch.int32)  # one empty group
    return x, w, offsets


def _breaks(fn, *args):
    torch._dynamo.reset()
    explained = torch._dynamo.explain(fn)(*args)
    return explained.graph_break_count, explained.graph_count


@needs_op
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_no_grad_call_compiles_without_graph_break(dtype, transposed, backend):
    x, w, offsets = _case(dtype, transposed)
    with torch.no_grad():
        assert _breaks(M._manual_grouped_mm, x, w, offsets) == (0, 1)
        want = M._manual_grouped_mm(x, w, offsets)
        torch._dynamo.reset()
        got = torch.compile(M._manual_grouped_mm, backend = backend, fullgraph = True)(x, w, offsets)
    assert got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want)


@needs_op
def test_frozen_experts_compile_without_graph_break():
    # Grad mode on, but nothing to backpropagate into (frozen experts, no grad on the hidden states).
    x, w, offsets = _case(torch.bfloat16, True)
    assert _breaks(M._grouped_mm_with_backward_fix, x, w, offsets) == (0, 1)


@needs_op
@pytest.mark.parametrize(
    "x_dtype,w_dtype",
    [(torch.float32, torch.float32), (torch.float32, torch.bfloat16), (torch.float64, torch.float64)],
)
def test_autocast_dtype_matches_eager(x_dtype, w_dtype):
    # Eager matmuls inside the loop autocast (float64 operands excluded); the opaque op must return what they would.
    x, w, offsets = _case(x_dtype, False, w_dtype = w_dtype)
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
        want = M._manual_grouped_mm(x, w, offsets)
        torch._dynamo.reset()
        got = torch.compile(M._manual_grouped_mm, backend = "aot_eager", fullgraph = True)(x, w, offsets)
    expected = torch.float64 if x_dtype == torch.float64 else torch.bfloat16
    assert want.dtype == expected and got.dtype == want.dtype and torch.equal(got, want)


@needs_op
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_training_keeps_the_routing_guard_under_checkpoint_replay(backend):
    # A gradient through the call keeps the guarded eager path: its forward routing is a Python list on ctx, which
    # non-reentrant checkpoint replay cannot swap (a saved tensor would be swapped, hiding the reroute).
    seen = {"n": 0}

    def region(x, w):
        seen["n"] += 1
        ends = [4, 8, 12] if seen["n"] == 1 else [4, 9, 12]
        return M._grouped_mm_with_backward_fix(x, w, torch.tensor(ends, dtype = torch.int32))

    torch._dynamo.reset()
    compiled = torch.compile(region, backend = backend)
    x = torch.randn(12, 8, requires_grad = True)
    w = torch.randn(3, 8, 6, requires_grad = True)
    out = checkpoint(compiled, x, w, use_reentrant = False)
    with pytest.raises(RuntimeError, match = "assigned tokens differently"):
        out.sum().backward()


@needs_op
def test_compiled_training_matches_eager():
    x, w, offsets = _case(torch.float32, False)

    def run(fn):
        xi, wi = x.clone().requires_grad_(), w.clone().requires_grad_()
        out = fn(xi, wi, offsets)
        (out * torch.linspace(-1, 1, out.shape[-1])).sum().backward()
        return out.detach(), xi.grad, wi.grad

    eager = run(M._manual_grouped_mm)
    torch._dynamo.reset()
    for got, want in zip(run(torch.compile(M._manual_grouped_mm, backend = "aot_eager")), eager):
        assert torch.equal(got, want)


def test_eager_call_is_unchanged():
    # Outside a compiled frame the compile-disabled autograd Function runs, as before.
    x, w, offsets = _case(torch.float32, False)
    x.requires_grad_()
    out = M._manual_grouped_mm(x, w, offsets)
    assert type(out.grad_fn).__name__.startswith("_ManualGroupedMM")
    with torch.no_grad():
        assert torch.equal(M._manual_grouped_mm(x, w, offsets), M._grouped_matmul_loop(x, w, offsets))


@needs_op
def test_no_grad_op_skips_the_routing_signature(monkeypatch):
    # Like the eager no-grad call: nothing reads the checksum, so the op must not pay for it.
    x, w, offsets = _case(torch.float32, False)
    calls = []
    real = M._routing_signature
    monkeypatch.setattr(M, "_routing_signature", lambda *a: calls.append(1) or real(*a))
    with torch.no_grad():
        torch._dynamo.reset()
        got = torch.compile(M._manual_grouped_mm, backend = "aot_eager", fullgraph = True)(x, w, offsets)
        assert torch.equal(got, M._grouped_matmul_loop(x, w, offsets))
    assert not calls


@needs_op
def test_autocast_casts_only_the_active_experts():
    # Eager autocasts one matmul per nonempty group; the compiled graph must not convert the whole stack per call.
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm.graph)
        return gm.forward

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, generator = g)
    w = torch.randn(8, 16, 8, generator = g)
    offsets = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype = torch.int32)  # one token, one active expert
    torch._dynamo.reset()
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
        got = torch.compile(M._manual_grouped_mm, backend = backend, fullgraph = True)(x, w, offsets)
        want = M._manual_grouped_mm(x, w, offsets)
    assert torch.equal(got, want) and got.dtype == torch.bfloat16
    casts = [n for graph in graphs for n in graph.nodes if n.op == "call_method" and n.target in ("to", "type")]
    assert not casts, casts
