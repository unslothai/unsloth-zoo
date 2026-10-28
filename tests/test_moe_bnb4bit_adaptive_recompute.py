"""The bnb4bit grouped-mm dispatcher follows the pin-vs-recompute policy, biased to
recompute for the 4-bit base; it never pre-dequantizes and then re-holds the stack."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.nn import Params4bit

# conftest sets UNSLOTH_ALLOW_CPU=1, so DEVICE_TYPE_TORCH says "cuda" even with no GPU: probe torch.
gpu_available = (
    (hasattr(torch, "cuda") and torch.cuda.is_available())
    or (hasattr(torch, "xpu") and torch.xpu.is_available())
)

# Skip before importing unsloth_zoo so a CPU-only host stays a clean module skip.
if not gpu_available:
    pytest.skip("bnb 4-bit dequant needs CUDA or XPU", allow_module_level=True)

from unsloth_zoo.device_type import DEVICE_TYPE_TORCH
import unsloth_zoo.temporary_patches.moe_utils as mu
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit
from unsloth_zoo.gradient_checkpointing import _gradient_checkpoint_recompute_marker


def _quantized_expert_param(shape=(4, 32, 64)):
    w = torch.randn(*shape, dtype=torch.bfloat16)
    p = Params4bit(w, requires_grad=False, quant_type="nf4", compress_statistics=True).to(DEVICE_TYPE_TORCH)
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
        lambda self, hs, ti, tw: (calls.append("provider"), torch.zeros(1, device=DEVICE_TYPE_TORCH))[1],
    )
    monkeypatch.setattr(
        mu, "swap_moe_weights_for_call",
        lambda self, gu, dn, fn, *a: (calls.append("swap"), torch.zeros(1, device=DEVICE_TYPE_TORCH))[1],
    )
    self = _FakeExperts()
    hs = torch.randn(3, 32, dtype=torch.bfloat16, device=DEVICE_TYPE_TORCH)
    forward_moe_backend_bnb4bit(self, hs, torch.zeros(3, 1, dtype=torch.long, device=DEVICE_TYPE_TORCH), None)
    return calls


def test_default_non_gc_uses_recompute_provider(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    assert _run_and_record(monkeypatch) == ["provider"]  # keep 4-bit, recompute


def test_gc_recompute_pass_recomputes_for_4bit_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    monkeypatch.delenv("UNSLOTH_MOE_GC_REPLAY_PIN", raising=False)
    monkeypatch.setattr(mu, "_momentary_pin_fits", lambda src, dtype = None: True)
    with _gradient_checkpoint_recompute_marker():
        assert _run_and_record(monkeypatch) == ["provider"]


def test_gc_recompute_pass_pins_for_4bit_when_asked_and_the_stack_fits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    monkeypatch.setenv("UNSLOTH_MOE_GC_REPLAY_PIN", "1")
    monkeypatch.setattr(mu, "_momentary_pin_fits", lambda src, dtype = None: True)
    with _gradient_checkpoint_recompute_marker():
        assert _run_and_record(monkeypatch) == ["swap"]  # momentary pin


def test_gc_recompute_pass_recomputes_for_4bit_when_the_stack_does_not_fit(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    monkeypatch.setenv("UNSLOTH_MOE_GC_REPLAY_PIN", "1")
    monkeypatch.setattr(mu, "_momentary_pin_fits", lambda src, dtype = None: False)
    with _gradient_checkpoint_recompute_marker():
        assert _run_and_record(monkeypatch) == ["provider"]  # recompute, no bf16 pin


def test_momentary_pin_fits_reads_the_real_stack_size(monkeypatch):
    q = _quantized_expert_param()
    if q.device.type != "cuda":
        pytest.skip("measures CUDA free memory")
    assert mu._momentary_pin_fits(q) is True              # 4 x 32 x 64 bf16 always fits
    monkeypatch.setattr(mu, "_MOMENTARY_PIN_HEADROOM", 1e12)
    assert mu._momentary_pin_fits(q) is False             # and the headroom is honoured


def test_momentary_pin_fits_measures_xpu_and_refuses_what_it_cannot_measure(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(mu, "_logical_expert_shape", lambda param: (4, 32, 64))
    need = 4 * 32 * 64 * 2  # bf16
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        free = {"bytes": 0}
        monkeypatch.setattr(xpu, "mem_get_info", lambda device = None: (free["bytes"], 1 << 40), raising = False)
        monkeypatch.setattr(xpu, "memory_reserved", lambda device = None: 0, raising = False)
        monkeypatch.setattr(xpu, "memory_allocated", lambda device = None: 0, raising = False)
        param = SimpleNamespace(device = torch.device("xpu"), quant_state = None)
        free["bytes"] = int(mu._MOMENTARY_PIN_HEADROOM * need)
        assert mu._momentary_pin_fits(param) is True
        free["bytes"] = need
        mu._MOMENTARY_PIN_FITS_CACHE.clear()
        assert mu._momentary_pin_fits(param) is False
    assert mu._momentary_pin_fits(SimpleNamespace(device = torch.device("meta"), quant_state = None)) is False


def test_momentary_pin_fits_budgets_the_dtype_the_stack_is_dequantized_to(monkeypatch):
    from types import SimpleNamespace

    backend = getattr(torch, "xpu", None) if not torch.cuda.is_available() else torch.cuda
    if backend is None:
        pytest.skip("needs torch.cuda or torch.xpu to patch")
    monkeypatch.setattr(mu, "_logical_expert_shape", lambda param: (4, 32, 64))
    need_bf16 = 4 * 32 * 64 * 2
    free = int(mu._MOMENTARY_PIN_HEADROOM * need_bf16)
    monkeypatch.setattr(backend, "mem_get_info", lambda device = None: (free, 1 << 40), raising = False)
    monkeypatch.setattr(backend, "memory_reserved", lambda device = None: 0, raising = False)
    monkeypatch.setattr(backend, "memory_allocated", lambda device = None: 0, raising = False)
    mu._MOMENTARY_PIN_FITS_CACHE.clear()
    kind = "cuda" if backend is torch.cuda else "xpu"
    param = SimpleNamespace(device = torch.device(kind, 0), quant_state = SimpleNamespace(dtype = torch.bfloat16))
    assert mu._momentary_pin_fits(param) is True
    assert mu._momentary_pin_fits(param, dtype = torch.bfloat16) is True
    assert mu._momentary_pin_fits(param, dtype = torch.float32) is False


def test_momentary_pin_fits_does_not_query_the_driver_per_call(monkeypatch):
    from types import SimpleNamespace

    backend = getattr(torch, "xpu", None) if not torch.cuda.is_available() else torch.cuda
    if backend is None:
        pytest.skip("needs torch.cuda or torch.xpu to patch")
    monkeypatch.setattr(mu, "_logical_expert_shape", lambda param: (4, 32, 64))
    calls = []
    monkeypatch.setattr(backend, "mem_get_info", lambda device = None: calls.append(1) or (1 << 40, 1 << 40), raising = False)
    monkeypatch.setattr(backend, "memory_reserved", lambda device = None: 0, raising = False)
    monkeypatch.setattr(backend, "memory_allocated", lambda device = None: 0, raising = False)
    mu._MOMENTARY_PIN_FITS_CACHE.clear()
    kind = "cuda" if backend is torch.cuda else "xpu"
    param = SimpleNamespace(device = torch.device(kind, 0), quant_state = None)
    for _ in range(100):
        assert mu._momentary_pin_fits(param) is True
    assert len(calls) == 1


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
    monkeypatch.setattr(mu, "_base_is_recomputable", lambda src, dtype = None: True)
    dense = torch.randn(4, 32, 64, dtype=torch.bfloat16, device=DEVICE_TYPE_TORCH)  # requires_grad False
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "0")
    assert mu._moe_recompute_enabled(dense) is False
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    with _gradient_checkpoint_recompute_marker():
        assert mu._moe_recompute_enabled(dense) is False


def test_source_pins_large_dequant_classifies_4bit_only(monkeypatch):
    # Only a frozen bnb 4-bit expert reports a "large dequant" pinned form; a plain
    # dense base pins its own storage, so it is not flagged.
    q = _quantized_expert_param()
    dense = torch.randn(4, 32, 64, dtype=torch.bfloat16, device=DEVICE_TYPE_TORCH)
    assert mu._source_pins_large_dequant(q) is True
    assert mu._source_pins_large_dequant(dense) is False


def test_4bit_recomputes_outside_gc_and_pins_inside_when_it_fits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising=False)
    monkeypatch.setenv("UNSLOTH_MOE_GC_REPLAY_PIN", "1")
    q = _quantized_expert_param()
    dense = torch.randn(4, 32, 64, dtype=torch.bfloat16, device=DEVICE_TYPE_TORCH)  # frozen
    assert mu._moe_recompute_enabled(q) is True           # outside GC -> recompute
    with _gradient_checkpoint_recompute_marker():
        monkeypatch.setattr(mu, "_momentary_pin_fits", lambda src, dtype = None: True)
        assert mu._moe_recompute_enabled(q) is False      # fits -> momentary pin
        monkeypatch.setattr(mu, "_momentary_pin_fits", lambda src, dtype = None: False)
        assert mu._moe_recompute_enabled(q) is True       # does not fit -> recompute
        assert mu._moe_recompute_enabled(dense) is False  # dense -> pin (unchanged)
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "0")
    assert mu._moe_recompute_enabled(q) is False
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "1")
    with _gradient_checkpoint_recompute_marker():
        assert mu._moe_recompute_enabled(q) is True       # override beats the pin


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
