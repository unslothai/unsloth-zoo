"""Regression tests for the copy-elimination in _grouped_mm_with_backward_fix.

The frozen base expert stacks are stored (E, 2I, H) / (E, H, I) and preprocess_weight
transposes them into grouped layout, producing a non-contiguous view. Forcing
weight.contiguous() copied that whole frozen stack every step (~805 MB for gate_up on
Qwen3-30B; ~57% of MoE GPU time). The fix passes the weight as-is and only falls back to a
contiguous copy on the narrow 16-byte-stride error.

test_no_forced_copy pins the control flow: on a stack the #186365 probe deems safe the weight
reaches torch._grouped_mm as a non-contiguous view in one attempt, otherwise a contiguous copy
is forced. test_probe_gates_the_forced_copy pins that gate directly. test_view_matches_copy_*
pin bit-exactness against the always-contiguous path where torch._grouped_mm runs for real
(H100+/B200), forward and backward.
"""
import pytest
import torch


def _grouped_mm_ok():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1, 8, 8, device="cuda", dtype=torch.bfloat16)
        torch._grouped_mm(x, w, offs=torch.tensor([2], dtype=torch.int32, device="cuda"))
        return True
    except Exception:
        return False


def test_no_forced_copy_on_happy_path(monkeypatch):
    """On a probe-safe stack the weight reaches torch._grouped_mm as a non-contiguous view in
    one attempt (no forced copy of the frozen stack); on an unproven stack it is made
    contiguous. Fails against the old code, which always copied before the single try."""
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
    """The #186365 gate: when the probe reports the transposed-view path unsafe, the weight is
    made contiguous before torch._grouped_mm; when safe, the view is passed as-is."""
    import unsloth_zoo.temporary_patches.moe_utils as mu

    inputs = torch.randn(5, 4)
    weight_view = torch.randn(3, 2, 4).transpose(1, 2)
    offsets = torch.tensor([2, 2, 5], dtype=torch.int32)

    for probe_safe in (True, False):
        seen = []

        def spy(inp, w, offs=None):
            seen.append(w.is_contiguous())
            return torch.zeros(inp.shape[0], w.shape[-1], dtype=inp.dtype)

        monkeypatch.setattr(mu, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", probe_safe, raising=False)
        monkeypatch.setattr(torch, "_grouped_mm", spy, raising=False)
        mu._grouped_mm_with_backward_fix(inputs, weight_view, offsets)
        assert seen == [not probe_safe], f"probe_safe={probe_safe}: got {seen}"


@pytest.mark.skipif(not _grouped_mm_ok(), reason="torch._grouped_mm unsupported on this device")
def test_view_matches_copy_forward():
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    torch.manual_seed(0)
    E, K, N, T = 3, 64, 128, 40
    inputs = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)  # (E, out, in) as stored
    weight_view = base.transpose(1, 2)          # (E, K, N) non-contiguous view (what the fix keeps)
    weight_copy = weight_view.contiguous()      # what the old path forced
    offsets = torch.tensor([16, 28, 40], dtype=torch.int32, device="cuda")

    out_view = _grouped_mm_with_backward_fix(inputs, weight_view, offsets)
    out_copy = _grouped_mm_with_backward_fix(inputs, weight_copy, offsets)
    assert (out_view - out_copy).abs().max().item() == 0.0, "view vs contiguous-copy result differs"


@pytest.mark.skipif(not _grouped_mm_ok(), reason="torch._grouped_mm unsupported on this device")
def test_view_matches_copy_backward():
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    torch.manual_seed(0)
    E, K, N, T = 3, 64, 128, 40
    offsets = torch.tensor([16, 28, 40], dtype=torch.int32, device="cuda")

    def run(force_copy):
        x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        base = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
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
