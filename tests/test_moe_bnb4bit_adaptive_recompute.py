"""The bnb4bit grouped-mm dispatcher follows the pin-vs-recompute policy, biased to
recompute for the 4-bit base.

forward_moe_backend_bnb4bit must not pre-dequantize the packed Params4bit into a dense
bf16 stack and then re-hold that stack for a backward recompute. It keeps the 4-bit
weights and defers dequant to forward_native_grouped_mm's providers by default: the
4-bit base prefers recompute even inside a gradient-checkpoint recompute pass, because
pinning it would materialize the full bf16 expert stack the 4-bit storage exists to
avoid (several GiB per layer on large MoEs, exactly when 4-bit + gradient checkpointing
means the run is already memory-constrained). Only UNSLOTH_MOE_RECOMPUTE=0 forces the
pre-dequantize-then-pin path (for memory-rich runs); UNSLOTH_MOE_RECOMPUTE=1 also
recomputes. In the pin case recompute is disabled, so the dense stack is never re-held
for a backward recompute with no memory benefit.
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


def test_gc_recompute_pass_recomputes_for_4bit(monkeypatch):
    # A 4-bit base prefers recompute even inside a GC recompute pass, so the packed
    # Params4bit is kept and dequant deferred rather than pinning the full bf16 stack.
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    with _gradient_checkpoint_recompute_marker():
        assert _run_and_record(monkeypatch) == ["provider"]  # recompute, no bf16 pin


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


def test_source_pins_large_dequant_classifies_4bit_only(monkeypatch):
    # Only a frozen bnb 4-bit expert reports a "large dequant" pinned form; a plain
    # dense base pins its own storage, so it is not flagged.
    q = _quantized_expert_param()
    dense = torch.randn(4, 32, 64, dtype=torch.bfloat16, device="cuda")
    assert mu._source_pins_large_dequant(q) is True
    assert mu._source_pins_large_dequant(dense) is False


def test_4bit_prefers_recompute_but_dense_still_pins_under_gc(monkeypatch):
    # The scope guard: the recompute-under-GC default is 4-bit-specific. A dense
    # (already bf16) base keeps the speed-oriented adaptive policy and pins under a
    # GC recompute pass; the 4-bit base recomputes to avoid the bf16 dequant pin.
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    q = _quantized_expert_param()
    dense = torch.randn(4, 32, 64, dtype=torch.bfloat16, device="cuda")  # frozen
    with _gradient_checkpoint_recompute_marker():
        assert mu._moe_recompute_enabled(q) is True       # 4-bit -> recompute
        assert mu._moe_recompute_enabled(dense) is False  # dense -> pin (unchanged)
    # UNSLOTH_MOE_RECOMPUTE=0 still forces the pin even for a 4-bit base.
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "0")
    assert mu._moe_recompute_enabled(q) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
