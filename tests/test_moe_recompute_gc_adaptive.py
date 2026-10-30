"""Adaptive pin-vs-recompute policy for the grouped MoE base GEMM: pin the bf16 expert stack only during
gradient-checkpoint recompute, else recompute; UNSLOTH_MOE_RECOMPUTE overrides."""

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
    # Forward runs under no_grad (marker False) and again in backward (marker True); the flag must fire only in that
    # recompute and clear afterwards; other checkpoint runs stay undisturbed.
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


def test_recompute_policy_adaptive_and_overrides(monkeypatch):
    # Stub _base_is_recomputable to isolate the policy layer, since the real guard needs CUDA and torch._grouped_mm support.
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
