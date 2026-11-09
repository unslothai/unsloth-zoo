"""torch >= 2.14 raises a plain RuntimeError when a fullgraph region ran with every frame skipped (for example
Inductor declining bf16 on a T4): "found no compiled frames. Skipped frames: ..." on the first call, then
"... Compilation was not attempted and no cached compiled code was found." on every later call. The eager
fallback must latch on both, and any other RuntimeError must still propagate."""

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
    # Latched: the compiled callable is not retried after the first refusal.
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
    """Through the public entry point, with the raw compiled callable replaced by one that raises like torch 2.14."""
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
    """Real Dynamo, CPU: a backend that skips the frame reproduces the torch >= 2.14 error without a GPU."""
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
    # First call: "Skipped frames: ..."; a direct second call would be "Compilation was not attempted ...".
    assert torch.equal(wrapped(x), x * 3)
    assert torch.equal(wrapped(x), x * 3)
    assert "test-real-skip" in U._LATCHED_EAGER_LABELS
    with pytest.raises(RuntimeError, match = "Compilation was not attempted"):
        compiled(x)
