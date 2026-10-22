# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Trainable fp16 head, vocab sliced backward: no fp16 overflow, no long context memory rise."""

import pytest
import torch

import unsloth_zoo.fused_losses.preference_loss as module
from unsloth_zoo.fused_losses.preference_loss import fused_linear_token_logps

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _run(hidden, weight, upstream):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    labels = torch.zeros(h.shape[:-1], dtype = torch.long, device = h.device)
    mask = torch.ones_like(labels, dtype = torch.bool)
    out = fused_linear_token_logps(h, w, None, labels, mask)["per_token_logps"]
    (out * upstream).sum().backward()
    return out.detach(), h.grad, w.grad


@pytest.mark.parametrize("device", DEVICES)
def test_sliced_head_grad_does_not_overflow_fp16(monkeypatch, device):
    # Two vocab slices double the rows per GEMM; opposite signed upstream grads (chosen vs
    # rejected) cancel in the sum, but an fp16 GEMM output overflowed first and gave NaNs.
    monkeypatch.setattr(module, "_target_chunk_bytes", lambda: 1024 * 1024)
    hidden = torch.full((1, 4, 8), 1.5, device = device, dtype = torch.float16)
    weight = torch.zeros((65536, 8), device = device, dtype = torch.float16)
    upstream = torch.tensor([[32768.0, 32768.0, -32768.0, -32768.0]], device = device)
    out, grad_hidden, grad_weight = _run(hidden, weight, upstream)
    assert out.isfinite().all()
    assert grad_hidden.isfinite().all()
    assert grad_weight.isfinite().all()
    assert grad_weight[0, 0].item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_sliced_hidden_grad_does_not_overflow_fp16(monkeypatch, device):
    # Each vocab slice's hidden partial overflows fp16 before the slices cancel.
    monkeypatch.setattr(module, "_target_chunk_bytes", lambda: 1024 * 1024)
    hidden = torch.zeros((1, 1, 128), device = device, dtype = torch.float16)
    weight = torch.full((4096, 128), 4.0, device = device, dtype = torch.float16)
    upstream = torch.full((1, 1), 32768.0, device = device)
    _, grad_hidden, grad_weight = _run(hidden, weight, upstream)
    assert grad_hidden.isfinite().all()
    assert grad_weight.isfinite().all()


@pytest.mark.parametrize("n_rows, hidden_grad, want_sliced", [
    (20, True, True),
    (64, True, False),
    (64, False, True),
])
def test_long_context_keeps_per_chunk_backward(monkeypatch, n_rows, hidden_grad, want_sliced):
    # Sliced backward's float32 rows x hidden gradient outgrows the per chunk loop's head sized
    # copies once rows exceed about twice the vocab, so long contexts keep the per chunk loop.
    calls = []
    real = module._sliced_backward
    monkeypatch.setattr(module, "_sliced_backward", lambda *a: calls.append(1) or real(*a))
    vocab, hidden = 16, 8
    states = torch.randn(1, n_rows, hidden, dtype = torch.bfloat16).requires_grad_(hidden_grad)
    weight = torch.randn(vocab, hidden, dtype = torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, vocab, (1, n_rows))
    mask = torch.ones(1, n_rows, dtype = torch.bool)
    out = fused_linear_token_logps(states, weight, None, labels, mask)["per_token_logps"]
    out.sum().backward()
    assert bool(calls) == want_sliced
    assert weight.grad is not None and weight.grad.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
def test_per_chunk_head_grad_does_not_overflow_fp16(monkeypatch, device):
    # Rows past about twice the vocab take the per chunk loop; its head GEMM must not round each
    # chunk to fp16 either, or +inf and -inf chunks sum to NaN.
    monkeypatch.setattr(module, "_target_chunk_bytes", lambda: 1024)  # 4 rows per chunk
    calls = []
    real = module._sliced_backward
    monkeypatch.setattr(module, "_sliced_backward", lambda *a: calls.append(1) or real(*a))
    hidden = torch.full((1, 40, 8), 1.5, device = device, dtype = torch.float16)
    weight = torch.zeros((16, 8), device = device, dtype = torch.float16)
    sign = torch.tensor([1.0] * 4 + [-1.0] * 4, device = device).repeat(5)
    upstream = (32768.0 * sign).unsqueeze(0)
    _, grad_hidden, grad_weight = _run(hidden, weight, upstream)
    assert not calls
    assert grad_weight.isfinite().all()
    assert grad_hidden.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
def test_per_chunk_bf16_hidden_grad_with_fp16_head_does_not_overflow(monkeypatch, device):
    # Enough rows for the per chunk backward. Row 0's hidden gradient, about -72000, fits bf16 but
    # not an fp16 GEMM output, which gave -inf before the cast to the hidden dtype.
    monkeypatch.setattr(module, "_target_chunk_bytes", lambda: 1024 * 1024)
    vocab, hidden_size, n_rows = 129, 2048, 259
    weight = torch.full((vocab, hidden_size), 600.0, device = device)
    weight[0] = 0.0
    weight = weight.half()
    gen = torch.Generator().manual_seed(0)
    hidden = (torch.randn(1, n_rows, hidden_size, generator = gen) * 1e-3).to(device = device, dtype = torch.bfloat16)
    sliced = []
    original = module._sliced_backward
    monkeypatch.setattr(module, "_sliced_backward", lambda *args: sliced.append(1) or original(*args))
    _, grad_hidden, grad_weight = _run(hidden, weight, torch.full((1, n_rows), 120.0, device = device))
    assert not sliced
    h = hidden.detach().double().requires_grad_(True)
    w = weight.detach().double()
    logps = torch.log_softmax(h @ w.t(), -1)[..., 0]
    (want,) = torch.autograd.grad((logps * 120.0).sum(), [h])
    assert grad_hidden.isfinite().all()
    assert grad_weight.isfinite().all()
    # Within bf16 / fp16 rounding of the float64 oracle.
    assert float((grad_hidden.double() - want).abs().max()) <= 2 ** -7 * float(want.abs().max())
