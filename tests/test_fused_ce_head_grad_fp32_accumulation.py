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

"""A trainable bf16 lm_head bias gradient summed over many chunks must not drift from fp64."""
import torch

import unsloth_zoo.fused_losses.cross_entropy_loss as ce


def _inputs(bsz = 2, qlen = 512, hd = 32, vocab = 256):
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(bsz, qlen, hd, generator = g).bfloat16()
    weight = (torch.randn(vocab, hd, generator = g) * hd ** -0.5).bfloat16()
    bias = (torch.randn(vocab, generator = g) * 0.1).bfloat16()
    labels = torch.randint(0, vocab, (bsz, qlen), generator = g)
    return hidden, weight, bias, labels


def _fused_grads(hidden, weight, bias, labels, n_chunks):
    h = hidden.clone().requires_grad_()
    w = weight.clone().requires_grad_()
    b = bias.clone().requires_grad_()
    loss = ce.unsloth_fused_ce_loss(
        None, h, w, b, labels, torch_compile = False, n_chunks = n_chunks,
    )
    loss.backward()
    assert w.grad.dtype == torch.bfloat16 and b.grad.dtype == torch.bfloat16
    return w.grad, b.grad


def _oracle(hidden, weight, bias, labels):
    w = weight.double().requires_grad_()
    b = bias.double().requires_grad_()
    logits = torch.nn.functional.linear(hidden.double(), w, b)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
    )
    loss.backward()
    return w.grad, b.grad


def _rel(a, ref):
    return float((a.double() - ref).norm() / ref.norm())


def test_bias_grad_chunked_matches_single_chunk_precision():
    hidden, weight, bias, labels = _inputs()
    _, ref_b = _oracle(hidden, weight, bias, labels)
    _, b1 = _fused_grads(hidden, weight, bias, labels, n_chunks = 1)
    _, b64 = _fused_grads(hidden, weight, bias, labels, n_chunks = 64)
    err1, err64 = _rel(b1, ref_b), _rel(b64, ref_b)
    # bf16 running sum over 64 chunks was ~4x the single chunk error here.
    assert err64 < 2 * err1 + 1e-3, (err1, err64)


def test_fp32_head_grad_opt_in(monkeypatch):
    hidden, weight, bias, labels = _inputs()
    ref_w, _ = _oracle(hidden, weight, bias, labels)
    w_default, _ = _fused_grads(hidden, weight, bias, labels, n_chunks = 64)
    monkeypatch.setattr(ce, "FP32_HEAD_GRAD", True)
    w_fp32, _ = _fused_grads(hidden, weight, bias, labels, n_chunks = 64)
    assert _rel(w_fp32, ref_w) < _rel(w_default, ref_w)
    w1, _ = _fused_grads(hidden, weight, bias, labels, n_chunks = 1)
    monkeypatch.setattr(ce, "FP32_HEAD_GRAD", False)
    w1_default = _fused_grads(hidden, weight, bias, labels, n_chunks = 1)[0]
    # One chunk: the float32 buffer rounds once, exactly like the bf16 one.
    torch.testing.assert_close(w1, w1_default, rtol = 0, atol = 0)
