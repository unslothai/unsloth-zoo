# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
PR-A end-to-end component exercises through the shim.

Goes one level deeper than test_pr_a_imports.py: actually constructs
inputs and runs the function bodies.  Each test focuses on one
critical PR-A code path.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ---------------------------------------------------------------------------
# 1. CCE end-to-end: forward + backward (autograd) match a numpy reference.
# ---------------------------------------------------------------------------

def _ce_reference(hidden, weight, targets, ignore_index=-100, softcap=0.0):
    """Numpy/torch reference for cross_entropy with optional logit softcap."""
    logits = hidden @ weight.T  # (n, vocab)
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    valid = targets != ignore_index
    log_probs = torch.log_softmax(logits, dim=-1)
    targ_safe = torch.where(valid, targets.long(), torch.zeros_like(targets, dtype=torch.long))
    nll = -log_probs.gather(1, targ_safe.unsqueeze(1)).squeeze(1)
    return torch.where(valid, nll, torch.zeros_like(nll))


def test_cce_forward_pure_python_matches_reference():
    """The kernel-less branch of _forward_chunked_fused_finalize."""
    from unsloth_zoo.mlx.cce.runtime_cce import _forward_chunked_fused_finalize

    torch.manual_seed(0)
    n, hidden_dim, vocab = 4, 8, 32
    hidden = torch.randn(n, hidden_dim, dtype=torch.float32)
    weight = torch.randn(vocab, hidden_dim, dtype=torch.float32) * 0.1
    targets = torch.tensor([3, 17, 5, 29], dtype=torch.int32)

    loss, lse = _forward_chunked_fused_finalize(
        hidden, weight, targets,
        scales=None, biases=None, group_size=None, bits=None, mode="affine",
        ignore_index=-100, logit_softcap=0.0, chunk_size=16,
        forward_update_kernel=None, forward_update_finalize_kernel=None,
    )
    expected = _ce_reference(hidden, weight, targets)
    torch.testing.assert_close(loss.float(), expected.float(), atol=1e-4, rtol=1e-4)


def test_cce_with_softcap():
    """Softcap path matches reference too."""
    from unsloth_zoo.mlx.cce.runtime_cce import _forward_chunked_fused_finalize

    torch.manual_seed(0)
    n, hidden_dim, vocab = 4, 8, 32
    hidden = torch.randn(n, hidden_dim, dtype=torch.float32)
    weight = torch.randn(vocab, hidden_dim, dtype=torch.float32) * 0.1
    targets = torch.tensor([3, 17, 5, 29], dtype=torch.int32)
    softcap = 30.0

    loss, _ = _forward_chunked_fused_finalize(
        hidden, weight, targets,
        scales=None, biases=None, group_size=None, bits=None, mode="affine",
        ignore_index=-100, logit_softcap=softcap, chunk_size=16,
        forward_update_kernel=None, forward_update_finalize_kernel=None,
    )
    expected = _ce_reference(hidden, weight, targets, softcap=softcap)
    torch.testing.assert_close(loss.float(), expected.float(), atol=1e-4, rtol=1e-4)


def test_cce_with_ignore_index():
    """Ignore-index zeros out loss for those positions."""
    from unsloth_zoo.mlx.cce.runtime_cce import _forward_chunked_fused_finalize

    torch.manual_seed(0)
    n, hidden_dim, vocab = 4, 8, 32
    hidden = torch.randn(n, hidden_dim, dtype=torch.float32)
    weight = torch.randn(vocab, hidden_dim, dtype=torch.float32) * 0.1
    targets = torch.tensor([3, -100, 5, -100], dtype=torch.int32)
    loss, _ = _forward_chunked_fused_finalize(
        hidden, weight, targets,
        scales=None, biases=None, group_size=None, bits=None, mode="affine",
        ignore_index=-100, logit_softcap=0.0, chunk_size=16,
        forward_update_kernel=None, forward_update_finalize_kernel=None,
    )
    assert loss[1].item() == 0.0
    assert loss[3].item() == 0.0
    assert loss[0].item() > 0.0
    assert loss[2].item() > 0.0


def test_cce_chunked_matches_unchunked():
    """Chunked online LSE must equal a single-shot computation."""
    from unsloth_zoo.mlx.cce.runtime_cce import _forward_chunked_fused_finalize

    torch.manual_seed(0)
    n, hidden_dim, vocab = 4, 8, 64
    hidden = torch.randn(n, hidden_dim, dtype=torch.float32)
    weight = torch.randn(vocab, hidden_dim, dtype=torch.float32) * 0.1
    targets = torch.tensor([3, 17, 5, 29], dtype=torch.int32)
    expected_loss = _ce_reference(hidden, weight, targets)

    for chunk_size in [8, 16, 32, 64]:
        loss, _ = _forward_chunked_fused_finalize(
            hidden, weight, targets,
            scales=None, biases=None, group_size=None, bits=None, mode="affine",
            ignore_index=-100, logit_softcap=0.0, chunk_size=chunk_size,
            forward_update_kernel=None, forward_update_finalize_kernel=None,
        )
        torch.testing.assert_close(loss.float(), expected_loss.float(), atol=1e-4, rtol=1e-4,
                                   msg=f"mismatch at chunk_size={chunk_size}")


# ---------------------------------------------------------------------------
# 2. Dequantize: the affine helper round-trips correctly.
# ---------------------------------------------------------------------------

def test_dequantize_affine_roundtrip():
    """Construct a known-good packed weight and verify dequant."""
    from mlx_simulation.mlx_helpers.quant import dequantize_affine

    # 4-bit affine: 8 elements packed into 1 uint32 word (4 bits each, 8 elements per word).
    # Group size = 8, bits = 4. Use a simple test vector.
    bits = 4
    group_size = 8
    elements_per_word = 32 // bits  # = 8

    # Pack values 0..7 (4-bit each) into a single uint32.
    # Element 0 in lowest bits, element 7 in highest.
    packed_value = 0
    for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        packed_value |= v << (i * bits)
    packed = torch.tensor([[packed_value]], dtype=torch.int32)  # shape (1, 1)
    scales = torch.tensor([[2.0]])  # shape (1, 1) - per-group scale
    biases = torch.tensor([[1.0]])  # shape (1, 1) - per-group bias

    out = dequantize_affine(packed, scales, biases, group_size=group_size, bits=bits)
    expected = torch.tensor([[0*2+1, 1*2+1, 2*2+1, 3*2+1, 4*2+1, 5*2+1, 6*2+1, 7*2+1]],
                            dtype=scales.dtype)
    torch.testing.assert_close(out, expected)


def test_dequantize_unsupported_mode_raises():
    """Non-affine modes raise NotImplementedError with a clear message."""
    from mlx_simulation.mlx_helpers.quant import dequantize_affine
    import mlx.core as mx
    with pytest.raises(NotImplementedError, match="mxfp4"):
        mx.dequantize(
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            group_size=32, bits=4, mode="mxfp4",
        )


# ---------------------------------------------------------------------------
# 3. mx.custom_function VJP cycle: forward + backward via torch.autograd.
# ---------------------------------------------------------------------------

def test_custom_function_forward_and_backward():
    """Define a simple square op with custom VJP; verify autograd traverses it."""
    import mlx.core as mx

    @mx.custom_function
    def square(x):
        return x * x

    @square.vjp
    def square_vjp(primals, cotangents, outputs):
        (x,) = primals
        (g,) = cotangents
        return (2 * x * g,)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = square(x)
    torch.testing.assert_close(y, x * x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, 2 * torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 4. mx.array isinstance contract that PR-A relies on.
# ---------------------------------------------------------------------------

def test_torch_tensor_is_mx_array():
    import mlx.core as mx
    t = torch.tensor([1.0, 2.0])
    assert isinstance(t, mx.array)
    # Reverse contract: mx.array(...) returns torch.Tensor
    a = mx.array([1, 2, 3], dtype=mx.int32)
    assert isinstance(a, torch.Tensor)
    assert a.dtype == torch.int32


# ---------------------------------------------------------------------------
# 5. Tensor monkey-patches: .astype, .expand_dims, .at[].add
# ---------------------------------------------------------------------------

def test_tensor_astype():
    t = torch.tensor([1.0, 2.0])
    out = t.astype(torch.bfloat16)
    assert out.dtype == torch.bfloat16


def test_tensor_expand_dims():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = t.expand_dims(0)
    assert out.shape == (1, 3)


def test_tensor_at_add_functional_update():
    """t.at[idx].add(v) returns a new tensor with t[idx] += v, original unchanged."""
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = t.at[1].add(10.0)
    torch.testing.assert_close(out, torch.tensor([1.0, 12.0, 3.0, 4.0]))
    # Original unchanged
    torch.testing.assert_close(t, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_tensor_at_set():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = t.at[0].set(99.0)
    torch.testing.assert_close(out, torch.tensor([99.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 6. multi-arg .transpose() = MLX permute semantics.
# ---------------------------------------------------------------------------

def test_transpose_multi_axis_is_permute():
    t = torch.zeros(2, 3, 4, 5)
    # 4-axis transpose = full permute, shape becomes (5, 4, 3, 2)
    out = t.transpose(3, 2, 1, 0)
    assert out.shape == (5, 4, 3, 2)


def test_transpose_two_args_unchanged():
    """torch's standard 2-axis swap behavior is preserved."""
    t = torch.zeros(2, 3, 4)
    out = t.transpose(0, 1)
    assert out.shape == (3, 2, 4)
