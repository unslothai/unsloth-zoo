import pytest
import torch


def test_grouped_mm_alignment_fallback_is_differentiable(monkeypatch):
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    inputs = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(3, 4, 2, requires_grad=True)
    offsets = torch.tensor([2, 2, 5], dtype=torch.int32)

    def raise_alignment_error(*args, **kwargs):
        raise RuntimeError("strides should be multiple of 16 bytes")

    monkeypatch.setattr(torch, "_grouped_mm", raise_alignment_error, raising=False)

    actual = _grouped_mm_with_backward_fix(inputs, weight, offsets)
    expected = torch.cat(
        [
            inputs[:2] @ weight[0],
            inputs[2:5] @ weight[2],
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert inputs.grad is not None
    assert weight.grad is not None


def test_grouped_mm_alignment_fallback_reraises_other_errors(monkeypatch):
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix

    inputs = torch.randn(1, 4)
    weight = torch.randn(1, 4, 2)
    offsets = torch.tensor([1], dtype=torch.int32)

    def raise_shape_error(*args, **kwargs):
        raise RuntimeError("matmul shape mismatch")

    monkeypatch.setattr(torch, "_grouped_mm", raise_shape_error, raising=False)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        _grouped_mm_with_backward_fix(inputs, weight, offsets)

