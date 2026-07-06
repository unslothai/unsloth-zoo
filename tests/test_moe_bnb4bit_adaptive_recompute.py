"""The bnb4bit grouped-mm dispatcher follows the adaptive pin-vs-recompute policy.

forward_moe_backend_bnb4bit must not pre-dequantize the packed Params4bit into a dense
bf16 stack and then re-hold that stack for a backward recompute. It keeps the 4-bit
weights and defers dequant to forward_native_grouped_mm's providers exactly when the
adaptive policy says recompute (non gradient-checkpoint, or UNSLOTH_MOE_RECOMPUTE=1),
and pre-dequantizes-then-pins only when the policy says pin (inside a gradient-checkpoint
recompute pass, or UNSLOTH_MOE_RECOMPUTE=0). In the pin case recompute is disabled, so
the dense stack is never re-held for a backward recompute with no memory benefit.
"""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.nn import Params4bit

if not torch.cuda.is_available():
    pytest.skip("bnb 4-bit dequant needs CUDA", allow_module_level=True)

import unsloth_zoo.temporary_patches.moe_utils as mu
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit
from unsloth_zoo.gradient_checkpointing import _gradient_checkpoint_recompute_marker


def _quantized_expert_param(shape=(4, 32, 64)):
    w = torch.randn(*shape, dtype=torch.bfloat16)
    p = Params4bit(w, requires_grad=False, quant_type="nf4", compress_statistics=True).to("cuda")
    p._original_shape = torch.Size(shape)
    return p


class _FakeExperts:
    def __init__(self):
        self.gate_up_proj = _quantized_expert_param()
        self.down_proj = _quantized_expert_param()


def _run_and_record(monkeypatch):
    """Dispatch once with the real backend picked as grouped_mm; return the path taken."""
    calls = []
    monkeypatch.setattr(mu, "select_moe_backend", lambda: "grouped_mm")
    monkeypatch.setattr(
        mu, "forward_native_grouped_mm",
        lambda self, hs, ti, tw: (calls.append("provider"), torch.zeros(1, device="cuda"))[1],
    )
    monkeypatch.setattr(
        mu, "swap_moe_weights_for_call",
        lambda self, gu, dn, fn, *a: (calls.append("swap"), torch.zeros(1, device="cuda"))[1],
    )
    self = _FakeExperts()
    hs = torch.randn(3, 32, dtype=torch.bfloat16, device="cuda")
    forward_moe_backend_bnb4bit(self, hs, torch.zeros(3, 1, dtype=torch.long, device="cuda"), None)
    return calls


def test_default_non_gc_uses_recompute_provider(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    assert _run_and_record(monkeypatch) == ["provider"]  # keep 4-bit, recompute


def test_gc_recompute_pass_pins_via_swap(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    with _gradient_checkpoint_recompute_marker():
        assert _run_and_record(monkeypatch) == ["swap"]  # pin (pre-dequant)


def test_env_override_forces_recompute(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "1")
    with _gradient_checkpoint_recompute_marker():  # override beats the GC pin
        assert _run_and_record(monkeypatch) == ["provider"]


def test_env_override_forces_pin(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "0")
    assert _run_and_record(monkeypatch) == ["swap"]  # forced pin


def test_pre_dequantized_dense_is_never_recomputed(monkeypatch):
    # The invariant behind the swap path: when the policy says pin, a pre-dequantized
    # dense stack must not be scheduled for a backward recompute (which would re-hold it
    # for no memory benefit).
    monkeypatch.setattr(mu, "_base_is_recomputable", lambda src: True)
    dense = torch.randn(4, 32, 64, dtype=torch.bfloat16, device="cuda")  # requires_grad False
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "0")
    assert mu._moe_recompute_enabled(dense) is False
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    with _gradient_checkpoint_recompute_marker():
        assert mu._moe_recompute_enabled(dense) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
