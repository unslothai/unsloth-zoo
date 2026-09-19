"""Regression tests for the copy-elimination in _grouped_mm_with_backward_fix.

The fix passes the frozen base stack to torch._grouped_mm as a transposed view instead of
copying it (~805 MB / ~57% of MoE GPU time on Qwen3-30B) every step, gated on the #186365
safety probe. These pin: the view is kept on a probe-safe stack (else a copy is forced), the
probe gates that choice, the probe leaves global RNG untouched, and view == contiguous
bit-exactly in forward and backward where torch._grouped_mm runs for real.
"""
import pytest
import torch


def _grouped_mm_ok():
    try:
        from unsloth_zoo.device_type import DEVICE_TYPE_TORCH
    except Exception:
        return False
    gpu = DEVICE_TYPE_TORCH if DEVICE_TYPE_TORCH in ("cuda", "xpu") else None
    if gpu is None:
        return False
    try:
        x = torch.randn(2, 8, device=gpu, dtype=torch.bfloat16)
        w = torch.randn(1, 8, 8, device=gpu, dtype=torch.bfloat16)
        torch._grouped_mm(x, w, offs=torch.tensor([2], dtype=torch.int32, device=gpu))
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _pretend_the_kernel_is_supported(monkeypatch):
    """These pin what happens AT the kernel, so the capability gate in front of
    it must not answer for a CPU runner and route the call away first."""
    from unsloth_zoo.temporary_patches import moe_utils
    monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: True)


def test_no_forced_copy_on_happy_path(monkeypatch):
    """Probe-safe stack: weight kept as a non-contiguous view in one attempt; unproven: copied."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        _grouped_mm_with_backward_fix, _transposed_view_grouped_mm_is_safe,
    )
    safe = _transposed_view_grouped_mm_is_safe()  # warm the cached probe with the REAL op first

    inputs = torch.randn(5, 4)
    weight_view = torch.randn(3, 2, 4).transpose(1, 2)  # (E, 4, 2) non-contiguous, like the base stack
    assert not weight_view.is_contiguous()
    offsets = torch.tensor([2, 2, 5], dtype=torch.int32)

    seen = []

    def spy(inp, w, offs=None):
        seen.append(w.is_contiguous())
        return torch.zeros(inp.shape[0], w.shape[-1], dtype=inp.dtype)

    monkeypatch.setattr(torch, "_grouped_mm", spy, raising=False)
    _grouped_mm_with_backward_fix(inputs, weight_view, offsets)

    expected = [False] if safe else [True]  # safe -> view kept; unproven -> contiguous copy
    assert seen == expected, f"probe safe={safe}: expected {expected}, got {seen}"


def test_probe_gates_the_forced_copy(monkeypatch):
    """The #186365 gate: probe unsafe -> weight made contiguous; safe -> view passed as-is."""
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    inputs = torch.randn(5, 4)
    weight_view = torch.randn(3, 2, 4).transpose(1, 2)
    offsets = torch.tensor([2, 2, 5], dtype=torch.int32)

    for probe_safe in (True, False):
        seen = []

        def spy(inp, w, offs=None):
            seen.append(w.is_contiguous())
            return torch.zeros(inp.shape[0], w.shape[-1], dtype=inp.dtype)

        monkeypatch.setattr(
            "unsloth_zoo.temporary_patches.moe_utils._TRANSPOSED_VIEW_GROUPED_MM_SAFE",
            probe_safe, raising=False)
        monkeypatch.setattr(torch, "_grouped_mm", spy, raising=False)
        _grouped_mm_with_backward_fix(inputs, weight_view, offsets)
        assert seen == [not probe_safe], f"probe_safe={probe_safe}: got {seen}"


def test_probe_does_not_perturb_global_rng(monkeypatch):
    """The probe uses a local generator, so it leaves the process-wide RNG untouched."""
    from unsloth_zoo.temporary_patches.moe_utils import _transposed_view_grouped_mm_is_safe

    monkeypatch.setattr(
        "unsloth_zoo.temporary_patches.moe_utils._TRANSPOSED_VIEW_GROUPED_MM_SAFE",
        None, raising=False)
    torch.manual_seed(1234)
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    _transposed_view_grouped_mm_is_safe()

    assert torch.equal(torch.get_rng_state(), cpu_before), "probe changed the CPU RNG state"
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_before), "probe changed the CUDA RNG state"


@pytest.mark.skipif(not _grouped_mm_ok(), reason="torch._grouped_mm unsupported on this device")
def test_view_matches_copy_forward():
    from unsloth_zoo.device_type import DEVICE_TYPE_TORCH
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    torch.manual_seed(0)
    E, K, N, T = 3, 64, 128, 40
    inputs = torch.randn(T, K, device=DEVICE_TYPE_TORCH, dtype=torch.bfloat16)
    base = torch.randn(E, N, K, device=DEVICE_TYPE_TORCH, dtype=torch.bfloat16)  # (E, out, in) as stored
    weight_view = base.transpose(1, 2)          # (E, K, N) non-contiguous view (what the fix keeps)
    weight_copy = weight_view.contiguous()      # what the old path forced
    offsets = torch.tensor([16, 28, 40], dtype=torch.int32, device=DEVICE_TYPE_TORCH)

    out_view = _grouped_mm_with_backward_fix(inputs, weight_view, offsets)
    out_copy = _grouped_mm_with_backward_fix(inputs, weight_copy, offsets)
    assert (out_view - out_copy).abs().max().item() == 0.0, "view vs contiguous-copy result differs"


@pytest.mark.skipif(not _grouped_mm_ok(), reason="torch._grouped_mm unsupported on this device")
def test_view_matches_copy_backward():
    from unsloth_zoo.device_type import DEVICE_TYPE_TORCH
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    torch.manual_seed(0)
    E, K, N, T = 3, 64, 128, 40
    offsets = torch.tensor([16, 28, 40], dtype=torch.int32, device=DEVICE_TYPE_TORCH)

    def run(force_copy):
        x = torch.randn(T, K, device=DEVICE_TYPE_TORCH, dtype=torch.bfloat16, requires_grad=True)
        base = torch.randn(E, N, K, device=DEVICE_TYPE_TORCH, dtype=torch.bfloat16, requires_grad=True)
        torch.manual_seed(1)  # identical draws across both runs
        x.data.normal_(); base.data.normal_()
        w = base.transpose(1, 2)
        w = w.contiguous() if force_copy else w
        _grouped_mm_with_backward_fix(x, w, offsets).float().pow(2).sum().backward()
        return x.grad.clone(), base.grad.clone()

    gx_view, gb_view = run(force_copy=False)
    gx_copy, gb_copy = run(force_copy=True)
    assert (gx_view - gx_copy).abs().max().item() == 0.0, "input grad differs"
    assert (gb_view - gb_copy).abs().max().item() == 0.0, "weight grad differs"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))


def test_the_routing_signature_sees_a_same_sum_swap():
    """A sum ignores WHERE a row's values sit, so `[1, 0]` and `[0, 1]` reduced alike
    and swapping them across an expert boundary left the signature unchanged: the guard
    would accept a replay whose gradients belong to a different routing."""
    from unsloth_zoo.temporary_patches.moe_utils import _routing_signature

    offsets = torch.tensor([2], dtype = torch.int32)
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert _routing_signature(rows, offsets) != _routing_signature(rows.flip(0), offsets)


def test_the_routing_signature_is_stable_for_the_same_rows():
    """A false positive here turns a healthy step into a hard error."""
    from unsloth_zoo.temporary_patches.moe_utils import _routing_signature

    offsets = torch.tensor([8], dtype = torch.int32)
    rows = torch.randn(8, 32)
    assert _routing_signature(rows, offsets) == _routing_signature(rows.clone(), offsets)


def test_the_routing_signature_sees_an_arbitrary_permutation():
    from unsloth_zoo.temporary_patches.moe_utils import _routing_signature

    offsets = torch.tensor([64], dtype = torch.int32)
    rows = torch.randn(64, 128)
    shuffled = rows[torch.randperm(64)]
    if torch.equal(rows, shuffled):
        pytest.skip("permutation happened to be the identity")
    assert _routing_signature(rows, offsets) != _routing_signature(shuffled, offsets)


def test_the_routing_signature_does_not_upcast_the_whole_input():
    """Upcasting on the way in materialised a whole `[routed_tokens, hidden]` transient
    for one number per row: 512MB at 32K by 4096, on exactly the memory-constrained
    runs this fallback exists for."""
    import inspect

    from unsloth_zoo.temporary_patches.moe_utils import _routing_signature

    body = inspect.getsource(_routing_signature)
    assert ".float().sum(" not in body
    assert ".detach().float()" not in body, "the full-width FP32 copy is back"


def test_the_routing_signature_runs_in_half_precision():
    from unsloth_zoo.temporary_patches.moe_utils import _routing_signature

    offsets = torch.tensor([4], dtype = torch.int32)
    rows = torch.randn(4, 16).half()
    assert isinstance(_routing_signature(rows, offsets), list)
