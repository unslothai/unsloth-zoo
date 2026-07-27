"""Adaptive pin-vs-recompute policy for the grouped MoE base GEMM.

The dequantized bf16 expert stack is either pinned (kept from forward to backward)
or recomputed in backward. Pinning is fast but memory-heavy; recompute is the
reverse. The choice is adaptive: pin only inside a gradient-checkpoint recompute
pass (where the stack is rebuilt right before the layer's own backward, so the pin
is momentary), otherwise recompute so a non-checkpointed forward does not hold every
layer's stack across the whole backward. UNSLOTH_MOE_RECOMPUTE overrides it.
"""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.gradient_checkpointing import (
    Unsloth_Gradient_Checkpointer,
    in_gradient_checkpoint_recompute,
    _gradient_checkpoint_recompute_marker,
)
import unsloth_zoo.temporary_patches.moe_utils as mu


# ---------------------------------------------------------------------------
# The recompute marker
# ---------------------------------------------------------------------------
def test_marker_idempotent_nesting_and_exception_safe():
    assert in_gradient_checkpoint_recompute() is False
    with _gradient_checkpoint_recompute_marker():
        assert in_gradient_checkpoint_recompute() is True
        with _gradient_checkpoint_recompute_marker():        # nested
            assert in_gradient_checkpoint_recompute() is True
        assert in_gradient_checkpoint_recompute() is True    # restored to outer
    assert in_gradient_checkpoint_recompute() is False       # restored to base

    with pytest.raises(ValueError):
        with _gradient_checkpoint_recompute_marker():
            assert in_gradient_checkpoint_recompute() is True
            raise ValueError("boom")
    # The finally clause must clear the flag even when the body raised.
    assert in_gradient_checkpoint_recompute() is False


def test_marker_true_only_during_gc_recompute():
    # The checkpointer runs the wrapped forward twice: once under no_grad (original
    # forward, marker False) and once under enable_grad in backward (recompute pass,
    # marker True). This proves the flag fires exactly during the recompute pass and
    # is cleared afterwards -- i.e. no other gradient-checkpoint run is disturbed.
    seen = []

    def fn(x, *args):
        seen.append(in_gradient_checkpoint_recompute())
        return (x * 2,)

    x = torch.randn(4, requires_grad=True)
    (out,) = Unsloth_Gradient_Checkpointer.apply(fn, x)
    assert seen == [False], seen                    # original forward
    out.sum().backward()
    assert seen == [False, True], seen              # + recompute pass
    assert in_gradient_checkpoint_recompute() is False
    # Backward still produced the correct gradient (d(2x)/dx = 2).
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))


# ---------------------------------------------------------------------------
# The adaptive decision
# ---------------------------------------------------------------------------
def test_recompute_policy_adaptive_and_overrides(monkeypatch):
    # Isolate the policy layer from the base-recomputable guards (which need CUDA +
    # torch._grouped_mm support).
    monkeypatch.setattr(mu, "_base_is_recomputable", lambda src: True)
    src = object()

    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    assert mu._moe_recompute_enabled(src) is True                 # non-GC -> recompute
    with _gradient_checkpoint_recompute_marker():
        assert mu._moe_recompute_enabled(src) is False            # GC recompute -> pin

    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "1")
    with _gradient_checkpoint_recompute_marker():
        assert mu._moe_recompute_enabled(src) is True             # force recompute

    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "0")
    assert mu._moe_recompute_enabled(src) is False                # force pin

    # A trainable / unsupported base can never recompute -> pinned eager path.
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    monkeypatch.setattr(mu, "_base_is_recomputable", lambda src: False)
    assert mu._moe_recompute_enabled(src) is False
