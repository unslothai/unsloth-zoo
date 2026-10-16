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

"""Chunked linear preference loss against a float64 full-logits oracle."""

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.fused_losses.preference_loss import (
    fused_linear_token_logps,
    linear_mean_logits,
)

DT = torch.float64


def _oracle_logits(hidden, weight, bias, multiply, divide, softcap):
    logits = F.linear(hidden, weight, bias)
    if multiply:
        logits = logits * multiply
    if divide:
        logits = logits / divide
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def _tensors(seed, rows = 4, seq = 9, hidden = 12, vocab = 53, bias = False):
    gen = torch.Generator().manual_seed(seed)
    weight = (3 * torch.randn(vocab, hidden, generator = gen, dtype = DT)).requires_grad_(True)
    b = torch.randn(vocab, generator = gen, dtype = DT).requires_grad_(True) if bias else None
    states = torch.randn(rows, seq, hidden, generator = gen, dtype = DT).requires_grad_(True)
    labels = torch.randint(0, vocab, (rows, seq), generator = gen)
    mask = torch.rand(rows, seq, generator = gen) > 0.35
    return states, weight, b, labels, mask


TRANSFORMS = [(0.0, 0.0, 0.0), (0.0, 0.0, 30.0), (0.0, 0.0, 2.0), (0.25, 0.0, 0.0), (0.0, 4.0, 0.0), (2.0, 0.0, 5.0)]


@pytest.mark.parametrize("multiply, divide, softcap", TRANSFORMS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, None])
def test_token_logps_match_full_logits(multiply, divide, softcap, bias, chunk_size):
    states, weight, b, labels, mask = _tensors(0, bias = bias)
    out = fused_linear_token_logps(
        states, weight, b, labels, mask, multiply, divide, softcap,
        chunk_size = chunk_size, return_mean_logits = True, return_log_sum_sq_probs = True,
    )
    logits = _oracle_logits(states, weight, b, multiply, divide, softcap)
    logprobs = logits.log_softmax(-1)
    expected = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
    torch.testing.assert_close(out["per_token_logps"], expected, rtol = 0, atol = 1e-12)
    torch.testing.assert_close(out["mean_logits"], logits.mean(-1) * mask, rtol = 0, atol = 1e-12)
    torch.testing.assert_close(
        out["log_sum_sq_probs"], torch.logsumexp(2 * logprobs, -1) * mask, rtol = 0, atol = 1e-12,
    )
    upstream = torch.randn(mask.shape, dtype = DT, generator = torch.Generator().manual_seed(1))
    params = [states, weight] + ([b] if bias else [])
    got = torch.autograd.grad((out["per_token_logps"] * upstream).sum(), params)
    want = torch.autograd.grad((expected * upstream).sum(), params)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol = 0, atol = 1e-11)
    # Off-mask positions get exactly zero gradient.
    assert got[0][~mask].abs().max() == 0


def test_token_logps_gradcheck():
    states, weight, b, labels, mask = _tensors(2, rows = 2, seq = 4, hidden = 5, vocab = 7, bias = True)
    fn = lambda s, w, bb: fused_linear_token_logps(s, w, bb, labels, mask, 0.5, 0.0, 3.0, chunk_size = 2)["per_token_logps"]
    assert torch.autograd.gradcheck(fn, (states, weight, b))


def test_token_logps_empty_and_frozen_head():
    states, weight, _, labels, mask = _tensors(3)
    out = fused_linear_token_logps(states, weight.detach(), None, labels, torch.zeros_like(mask))
    assert out["per_token_logps"].abs().sum() == 0
    out["per_token_logps"].sum().backward()
    assert states.grad is not None and states.grad.abs().sum() == 0
    one = torch.zeros_like(mask)
    one[1, 2] = True
    got = fused_linear_token_logps(states, weight.detach(), None, labels, one)["per_token_logps"]
    want = F.linear(states, weight).log_softmax(-1)[1, 2, labels[1, 2]]
    assert got[1, 2].item() == pytest.approx(want.item(), abs = 1e-12)


def test_token_logps_large_logits_stable():
    states, weight, _, labels, mask = _tensors(4)
    big = weight.detach() * 1e4
    out = fused_linear_token_logps(states.detach(), big, None, labels, mask)["per_token_logps"]
    want = F.linear(states.detach(), big).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, want, rtol = 1e-12, atol = 1e-9)


@pytest.mark.parametrize("multiply, divide, softcap", TRANSFORMS)
def test_linear_mean_logits(multiply, divide, softcap):
    states, weight, b, _, _ = _tensors(5, bias = True)
    got = linear_mean_logits(states, weight, b, multiply, divide, softcap, chunk_size = 4)
    want = _oracle_logits(states, weight, b, multiply, divide, softcap).mean(-1)
    torch.testing.assert_close(got, want, rtol = 0, atol = 1e-12)


def test_float32_and_half_heads_promote_to_float32():
    states, weight, _, labels, mask = _tensors(6)
    for dtype in (torch.float32, torch.bfloat16):
        out = fused_linear_token_logps(states.detach().to(dtype), weight.detach().to(dtype), None, labels, mask)
        assert out["per_token_logps"].dtype == torch.float32
        oracle = F.linear(states.detach(), weight.detach()).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
        tol = 1e-4 if dtype == torch.float32 else 0.6
        assert (out["per_token_logps"].double() - oracle).abs().max() < tol


def test_out_of_memory_does_not_disable_float32_gemm(monkeypatch):
    # A transient OOM in the float32-output GEMM must surface, not silently switch every later
    # call in the process to half-precision logits.
    import unsloth_zoo.fused_losses.preference_loss as module

    monkeypatch.setattr(module, "_MM_OUT_DTYPE_OK", {})

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(module.torch, "mm", oom)
    hidden = torch.randn(3, 4, dtype = torch.bfloat16)
    weight = torch.randn(5, 4, dtype = torch.bfloat16)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        module._linear_float32(hidden, weight, None)
    assert module._MM_OUT_DTYPE_OK.get(("cpu", torch.bfloat16), True) is True


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_trainable_head_grad_has_no_head_sized_float32_temporaries(dtype):
    # A trainable head accumulates its vocab x hidden gradient in place in the head dtype; a float32
    # accumulator plus per chunk head sized GEMM / cast copies used to add GiBs of peak memory.
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.manual_seed(0)
    vocab, hidden = 64, 16
    states = torch.randn(2, 9, hidden, dtype = dtype, requires_grad = True)
    weight = (torch.randn(vocab, hidden) * 0.3).to(dtype).requires_grad_(True)
    labels = torch.randint(0, vocab, (2, 9))
    mask = torch.ones(2, 9, dtype = torch.bool)
    mask[1, 5:] = False
    head_numel = vocab * hidden

    class Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.big = []

        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(t, torch.Tensor) and t.numel() >= head_numel and t.dtype != dtype:
                    self.big.append((str(func), t.dtype))
            return out

    out = fused_linear_token_logps(states, weight, None, labels, mask, chunk_size = 3)["per_token_logps"]
    record = Record()
    with record:
        out.sum().backward()
    assert record.big == []
    assert weight.grad.dtype == dtype

    w64 = weight.detach().double().requires_grad_(True)
    s64 = states.detach().double()
    want = F.linear(s64, w64).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
    want.sum().backward()
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    rel = (weight.grad.double() - w64.grad).norm() / w64.grad.norm()
    assert rel < tol
