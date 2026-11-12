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


from __future__ import annotations

import pytest
import torch

from unsloth_zoo.temporary_patches import utils as U

SKIPPED = (
    "torch.compile with fullgraph=True found no compiled frames. Skipped frames:\n"
    "  - Tesla T4 does not support bfloat16 compilation natively, skipping"
)
NOT_ATTEMPTED = (
    "torch.compile with fullgraph=True found no compiled frames. Compilation was not attempted "
    "and no cached compiled code was found."
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(U, "_EAGER_FALLBACK_WRAPPERS", [])
    monkeypatch.setattr(U, "_LATCHED_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_PENDING_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_RECENT_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_COMPILED_OK_LABELS", set())
    monkeypatch.setattr(U, "_PACKED_COMPILED_IN_CHECKPOINT", False)
    monkeypatch.setattr(U, "_RAISED_INSIDE_CHECKPOINT", False)


def _pair(message):
    calls = {"compiled": 0, "eager": 0}

    def compiled(x):
        calls["compiled"] += 1
        raise RuntimeError(message)

    def eager(x):
        calls["eager"] += 1
        return x * 2

    return compiled, eager, calls


@pytest.mark.parametrize("message", [SKIPPED, NOT_ATTEMPTED])
def test_no_compiled_frames_falls_back_and_latches(message):
    compiled, eager, calls = _pair(message)
    wrapped = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "test-skipped")
    assert wrapped(21) == 42
    assert wrapped(5) == 10
    assert calls == {"compiled": 1, "eager": 2}
    assert "test-skipped" in U._LATCHED_EAGER_LABELS
    assert U.eager_fallback_state()["test-skipped"] is True


def test_an_unrelated_runtime_error_still_raises():
    compiled, eager, calls = _pair("CUDA error: an illegal memory access was encountered")
    wrapped = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "test-unrelated")
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        wrapped(1)
    assert calls == {"compiled": 1, "eager": 0}
    assert "test-unrelated" not in U._LATCHED_EAGER_LABELS


def test_compile_with_eager_fallback_end_to_end(monkeypatch):
    from unsloth_zoo.temporary_patches import common

    state = {"n": 0}

    def fake_raw_compile(**kwargs):
        def deco(fn):
            def compiled(*a, **k):
                state["n"] += 1
                raise RuntimeError(SKIPPED if state["n"] == 1 else NOT_ATTEMPTED)
            return compiled
        return deco

    monkeypatch.setattr(common, "_raw_torch_compile", fake_raw_compile)

    def kernel(x):
        return x + 1

    wrapped = U.compile_with_eager_fallback(kernel, "test-e2e")
    x = torch.arange(3.0)
    assert torch.equal(wrapped(x), x + 1)
    assert torch.equal(wrapped(x.to(torch.bfloat16)), (x + 1).to(torch.bfloat16))
    assert state["n"] == 1
    assert "test-e2e" in U._LATCHED_EAGER_LABELS


def test_real_torch_skipped_frame_latches_to_eager():
    import torch._dynamo.exc as dynamo_exc

    if not hasattr(dynamo_exc, "SkipFrame"):
        pytest.skip("torch has no SkipFrame")

    def skipping_backend(gm, example_inputs):
        raise dynamo_exc.SkipFrame("test skip")

    def probe(x):
        return x - 1

    try:
        torch.compile(probe, backend = skipping_backend, fullgraph = True)(torch.ones(2))
    except RuntimeError as e:
        if not U._is_no_compiled_frames_error(e):
            raise
    else:
        pytest.skip("this torch runs a skipped fullgraph frame eagerly without raising")

    def kernel(x):
        return x * 3

    compiled = torch.compile(kernel, backend = skipping_backend, fullgraph = True)
    wrapped = U._fall_back_to_eager_on_recompile_limit(compiled, kernel, "test-real-skip")
    x = torch.ones(2)
    assert torch.equal(wrapped(x), x * 3)
    assert torch.equal(wrapped(x), x * 3)
    assert "test-real-skip" in U._LATCHED_EAGER_LABELS
    with pytest.raises(RuntimeError, match = "Compilation was not attempted"):
        compiled(x)


def _skipping_compile(kernel):
    import torch._dynamo.exc as dynamo_exc

    if not hasattr(dynamo_exc, "SkipFrame"):
        pytest.skip("torch has no SkipFrame")

    def skipping_backend(gm, example_inputs):
        raise dynamo_exc.SkipFrame("test skip")

    probe = torch.compile(lambda x: x - 1, backend = skipping_backend, fullgraph = True)
    try:
        probe(torch.ones(2))
    except RuntimeError as e:
        if not U._is_no_compiled_frames_error(e):
            raise
    else:
        pytest.skip("this torch runs a skipped fullgraph frame eagerly without raising")
    return torch.compile(kernel, backend = skipping_backend, fullgraph = True)


def test_a_skipped_frame_runs_its_body_once():
    calls = []

    def kernel(x):
        calls.append(1)
        return x * 3

    wrapped = U._fall_back_to_eager_on_recompile_limit(_skipping_compile(kernel), kernel, "test-once")
    x = torch.ones(2)
    assert torch.equal(wrapped(x), x * 3) and len(calls) == 1
    assert torch.equal(wrapped(x), x * 3) and len(calls) == 2
    assert "test-once" in U._LATCHED_EAGER_LABELS

    bn = torch.nn.BatchNorm1d(4).train()
    stateful = U._fall_back_to_eager_on_recompile_limit(_skipping_compile(bn.forward), bn.forward, "test-bn")
    stateful(torch.randn(8, 4))
    assert int(bn.num_batches_tracked) == 1


def test_skipped_frame_inside_a_checkpoint_backpropagates():
    from torch.utils.checkpoint import checkpoint

    if U._in_non_reentrant_checkpoint() is None:
        pytest.skip("this torch cannot report checkpoint regions")

    def kernel(x):
        return x.sin() * x

    wrapped = U._fall_back_to_eager_on_recompile_limit(_skipping_compile(kernel), kernel, "test-ckpt-skip")
    x = torch.linspace(-1, 1, 5, requires_grad = True)
    checkpoint(wrapped, x, use_reentrant = False).sum().backward()
    assert "test-ckpt-skip" in U._LATCHED_EAGER_LABELS
    assert not U._RAISED_INSIDE_CHECKPOINT
    ref = torch.linspace(-1, 1, 5, requires_grad = True)
    kernel(ref).sum().backward()
    assert torch.allclose(x.grad, ref.grad)


def test_skip_after_a_compiled_pack_in_the_step_ends_it(monkeypatch):
    from torch.utils.checkpoint import checkpoint

    if U._in_non_reentrant_checkpoint() is None:
        pytest.skip("this torch cannot report checkpoint regions")

    def kernel(x):
        return x.sin() * x

    wrapped = U._fall_back_to_eager_on_recompile_limit(_skipping_compile(kernel), kernel, "test-ckpt-packed")
    U._COMPILED_OK_LABELS.add("test-ckpt-packed")
    x = torch.linspace(-1, 1, 5, requires_grad = True)
    with pytest.raises(RuntimeError, match = "ending the step"):
        checkpoint(wrapped, x, use_reentrant = False)
    assert U._RAISED_INSIDE_CHECKPOINT and "test-ckpt-packed" in U._LATCHED_EAGER_LABELS


def test_a_skipped_outer_frame_with_a_compiled_child_is_not_eager():
    # A skipped frame whose nested frame compiled is not "no compiled frames": latching it eager inside a
    # non-reentrant checkpoint recomputes the region eagerly against the compiled child's saved tensors.
    if not U._resolve_skip_hooks():
        pytest.skip("torch without the fullgraph compiled-frame counter")
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.exc import SkipFrame
    from torch.utils.checkpoint import checkpoint

    torch._dynamo.reset()
    calls = []

    def backend(gm, args):
        outer = any(n.op == "call_method" and n.target == "exp" for n in gm.graph.nodes)
        calls.append("skip" if outer else "compile")
        if outer:
            raise SkipFrame("outer frame refused, nested frame compiles")
        return lookup_backend("aot_eager")(gm, args)

    def inner(t):
        return t.sigmoid() * t

    def kernel(x):
        return inner(x.exp())

    wrapped = U._fall_back_to_eager_on_recompile_limit(
        torch.compile(kernel, backend=backend, fullgraph=True), kernel, "mixed_frames"
    )
    x = torch.linspace(-1, 1, 5, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    kernel(ref).sum().backward()
    checkpoint(wrapped, x, use_reentrant=False).sum().backward()
    assert calls == ["skip", "compile"], calls
    assert "mixed_frames" not in U._LATCHED_EAGER_LABELS
    torch.testing.assert_close(x.grad, ref.grad)
