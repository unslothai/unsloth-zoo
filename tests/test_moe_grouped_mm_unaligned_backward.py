import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


def _native_cpu_grouped_mm_works():
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:
        x = torch.randn(8, 16, dtype = torch.bfloat16)
        w = torch.randn(1, 16, 16, dtype = torch.bfloat16)
        torch._grouped_mm(x, w, offs = torch.tensor([8], dtype = torch.int32))
        return True
    except (RuntimeError, NotImplementedError):
        return False


pytestmark = pytest.mark.skipif(not _native_cpu_grouped_mm_works(), reason = "torch._grouped_mm has no CPU kernel")


@pytest.fixture
def native(monkeypatch):
    """Route to the real torch._grouped_mm on CPU (the probes say no on CPU) and count its calls."""
    monkeypatch.setattr(M, "_grouped_mm_supported_on", lambda device: True)
    monkeypatch.setattr(M, "_transposed_view_safe_on", lambda device: True)
    real = torch._grouped_mm
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "_grouped_mm", counted)
    return calls


def _case(dtype, N, requires_grad = True):
    torch.manual_seed(0)
    E, K, M_rows = 3, 64, 24
    inputs = torch.randn(M_rows, K, dtype = dtype, requires_grad = requires_grad)
    # HF stores expert stacks as [E, out, in]; the MoE forward passes the [E, in, out] view.
    leaf = (torch.randn(E, N, K, dtype = dtype) * 0.1).requires_grad_(requires_grad)
    offsets = torch.tensor([8, 8, 24], dtype = torch.int32)
    return inputs, leaf, offsets


def _reference(inputs, leaf, offsets, grad):
    x = inputs.detach().double().requires_grad_()
    w = leaf.detach().double().requires_grad_()
    out, start = [], 0
    for g, end in enumerate(offsets.tolist()):
        out.append(x[start:end] @ w[g].transpose(0, 1))
        start = end
    y = torch.cat(out)
    y.backward(grad.double())
    return y, x.grad, w.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trainable_stack_with_unaligned_rows_trains(native, dtype):
    # 2 * I = 20 columns: 40 bytes per row in 16-bit, which torch._grouped_mm's own backward rejects.
    inputs, leaf, offsets = _case(dtype, N = 20)
    y = M._grouped_mm_with_backward_fix(inputs, leaf.transpose(-2, -1), offsets)
    grad = torch.randn_like(y)
    y.backward(grad)
    ref_y, ref_dx, ref_dw = _reference(inputs, leaf, offsets, grad)
    tol = dict(atol = 3e-2, rtol = 3e-2)
    torch.testing.assert_close(y.double(), ref_y, **tol)
    torch.testing.assert_close(inputs.grad.double(), ref_dx, **tol)
    torch.testing.assert_close(leaf.grad.double(), ref_dw, **tol)
    assert leaf.grad[1].abs().max() == 0  # the empty expert
    assert not native  # the per-expert fallback, not the op whose backward raises


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_backward_really_rejects_unaligned_rows(dtype):
    # Negative control: the raw op is what failed; without the guard this is the user's error.
    inputs, leaf, offsets = _case(dtype, N = 20)
    y = torch._grouped_mm(inputs, leaf.transpose(-2, -1), offs = offsets)
    with pytest.raises(RuntimeError, match = "strides should be multiple of 16 bytes"):
        y.backward(torch.randn_like(y))


def test_aligned_rows_keep_the_native_kernel(native):
    inputs, leaf, offsets = _case(torch.bfloat16, N = 24)
    y = M._grouped_mm_with_backward_fix(inputs, leaf.transpose(-2, -1), offsets)
    y.backward(torch.randn_like(y))
    assert len(native) == 1 and leaf.grad is not None


def test_frozen_or_no_grad_unaligned_keeps_the_native_kernel(native):
    # A frozen base (LoRA, recompute) never runs torch's backward, so the forward stays on the kernel.
    inputs, leaf, offsets = _case(torch.bfloat16, N = 20, requires_grad = False)
    M._grouped_mm_with_backward_fix(inputs, leaf.transpose(-2, -1), offsets)
    inputs, leaf, offsets = _case(torch.bfloat16, N = 20)
    with torch.no_grad():
        M._grouped_mm_with_backward_fix(inputs, leaf.transpose(-2, -1), offsets)
    assert len(native) == 2


def test_fp32_unaligned_keeps_the_native_kernel(native):
    # 20 fp32 columns are 80 bytes: aligned, so nothing changes in fp32.
    inputs, leaf, offsets = _case(torch.float32, N = 20)
    y = M._grouped_mm_with_backward_fix(inputs, leaf.transpose(-2, -1), offsets)
    y.backward(torch.randn_like(y))
    assert len(native) == 1
