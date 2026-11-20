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

"""Softmax collapse (arXiv 2610.31642): targets ~16.6 nats ahead keep nonzero, accurate logp and grad."""

import pytest
import torch

import unsloth_zoo.fused_losses.preference_loss as module
from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy, cpu_chunked_log_probs

MARGINS = [5.0, 10.0, 16.0, 17.0, 20.0, 30.0, 50.0, 80.0]
N_TARGETS = len(MARGINS)


def _exact_neg_softplus(x):
    return -(x.clamp_min(0) + torch.log1p(torch.exp(-x.abs())))


def _oracle(logits, labels):
    z = logits.double()
    z_t = z.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    others = z.scatter(-1, labels.unsqueeze(-1), float("-inf")).logsumexp(-1)
    logp = _exact_neg_softplus(others - z_t)
    return logp, -torch.expm1(logp)


def _problem(vocab, dtype, seed = 0):
    """Row r targets class r at margin MARGINS[r], others in [-1, 0]; hidden column r feeds only that row."""
    gen = torch.Generator().manual_seed(seed)
    hidden_size = N_TARGETS + 1
    weight = torch.zeros(vocab, hidden_size, dtype = torch.float64)
    weight[:N_TARGETS, :N_TARGETS] = torch.eye(N_TARGETS, dtype = torch.float64)
    weight[N_TARGETS:, N_TARGETS] = -torch.rand(vocab - N_TARGETS, generator = gen, dtype = torch.float64)
    hidden = torch.zeros(N_TARGETS, hidden_size, dtype = torch.float64)
    hidden[range(N_TARGETS), range(N_TARGETS)] = torch.tensor(MARGINS, dtype = torch.float64)
    hidden[:, N_TARGETS] = 1.0
    weight, hidden = weight.to(dtype), hidden.to(dtype)
    labels = torch.arange(N_TARGETS)
    logits = hidden.double() @ weight.double().t()  # exact: one nonzero product per logit
    return hidden, weight, labels, logits


def _rel(got, want):
    return ((got.double() - want).abs() / want.abs()).max().item()


def _paths(monkeypatch, compact):
    monkeypatch.setattr(module, "_FUSED_LINEAR", False)
    monkeypatch.setattr(module, "_HAS_CUSTOM_OP", compact and module._HAS_CUSTOM_OP)


# (head dtype, trainable head, expected backward, logp tolerance, gradient tolerance)
CASES = [
    (torch.float32, False, "chunked", 1e-5, 1e-4),
    (torch.float32, True, "chunked", 1e-5, 1e-4),
    (torch.bfloat16, True, "sliced", 1e-5, 1e-2),  # the gradient GEMM rounds to bf16
    (torch.float16, True, "sliced", 1e-5, 1e-2),
]


@pytest.mark.parametrize("vocab", [1000, 32000])
@pytest.mark.parametrize("dtype, trainable, path, logp_tol, grad_tol", CASES)
@pytest.mark.parametrize("compact", [True, False])
def test_fused_linear_token_logps_margin_sweep(monkeypatch, vocab, dtype, trainable, path, logp_tol, grad_tol,
                                               compact):
    _paths(monkeypatch, compact)
    calls = []
    real = module._sliced_backward
    monkeypatch.setattr(module, "_sliced_backward", lambda *a: calls.append(1) or real(*a))
    hidden, weight, labels, logits = _problem(vocab, dtype)
    hidden.requires_grad_(True)
    weight.requires_grad_(trainable)
    # One masked-off row in front: the compact path must skip it.
    states = torch.cat([torch.zeros_like(hidden[:1]), hidden]).unsqueeze(0)
    mask = torch.tensor([[False] + [True] * N_TARGETS])
    full_labels = torch.cat([torch.zeros(1, dtype = torch.long), labels]).unsqueeze(0)
    out = module.fused_linear_token_logps(states, weight, None, full_labels, mask)["per_token_logps"][0, 1:]
    # fp16 needs a loss scale; rows whose scaled target grad falls below fp16's normal range are skipped.
    scale = 2.0 ** 16 if dtype == torch.float16 else 1.0
    (grad_hidden,) = torch.autograd.grad(out.sum() * scale, [hidden])
    assert bool(calls) == (path == "sliced")
    logp, grad = _oracle(logits, labels)
    assert (out < 0).all(), out
    assert _rel(out, logp) < logp_tol
    rows = (grad * scale >= torch.finfo(dtype).tiny).nonzero().squeeze(-1)
    assert len(rows) >= 5  # margins 5 .. 20 at least
    got = grad_hidden[rows, rows].double() / scale
    assert _rel(got, grad[rows]) < grad_tol


@pytest.mark.parametrize("dtype, trainable, path", [(torch.float32, False, "chunked"), (torch.float32, True, "chunked"),
                                                    (torch.bfloat16, True, "sliced")])
def test_identity_head_logit_gradients(monkeypatch, dtype, trainable, path):
    # W = I: the hidden gradient is d logp / d logits, target and non-target entries.
    _paths(monkeypatch, True)
    vocab = 1000
    gen = torch.Generator().manual_seed(3)
    z = -torch.rand(N_TARGETS, vocab, generator = gen, dtype = torch.float64)
    labels = torch.randint(0, vocab, (N_TARGETS,), generator = gen)
    z[range(N_TARGETS), labels] = torch.tensor(MARGINS, dtype = torch.float64)
    hidden = z.to(dtype).requires_grad_(True)
    weight = torch.eye(vocab, dtype = dtype).requires_grad_(trainable)
    out = module.fused_linear_token_logps(hidden.unsqueeze(0), weight, None, labels.unsqueeze(0),
                                          torch.ones(1, N_TARGETS, dtype = torch.bool))["per_token_logps"][0]
    (g,) = torch.autograd.grad(out.sum(), [hidden])
    zd = hidden.detach().double()
    logp, target_grad = _oracle(zd, labels)
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    assert _rel(out, logp) < 1e-5
    assert _rel(g[range(N_TARGETS), labels], target_grad) < tol
    want = -torch.softmax(zd, -1)
    want[range(N_TARGETS), labels] = target_grad
    off = torch.ones_like(want, dtype = torch.bool)
    off[range(N_TARGETS), labels] = False
    assert ((g.double() - want).abs()[off] <= tol * want.abs()[off] + 1e-30).all()


@pytest.mark.parametrize("vocab", [1000, 32000])
def test_cpu_chunked_cross_entropy_margin_sweep(vocab):
    hidden, weight, labels, logits = _problem(vocab, torch.float32)
    logp, grad = _oracle(logits, labels)
    for r in range(N_TARGETS):
        h = hidden[r:r + 1].clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        loss = chunked_linear_cross_entropy(h, w, None, shift_labels = labels[r:r + 1], chunk_rows = 1)
        gh, gw = torch.autograd.grad(loss, [h, w])
        assert abs(-loss.item() - logp[r].item()) / abs(logp[r].item()) < 1e-5
        assert abs(-gh[0, r].item() - grad[r].item()) / grad[r].item() < 1e-4
        assert abs(-gw[r, r].item() - grad[r].item() * hidden[r, r].item()) / (grad[r].item() * hidden[r, r].item()) < 1e-4


def test_cpu_chunked_cross_entropy_all_ignored():
    hidden, weight, labels, _ = _problem(1000, torch.float32)
    h = hidden.clone().requires_grad_(True)
    loss = chunked_linear_cross_entropy(h, weight, None, shift_labels = torch.full_like(labels, -100))
    assert torch.isnan(loss)
    (gh,) = torch.autograd.grad(loss, [h])
    assert (gh == 0).all()


@pytest.mark.parametrize("vocab", [1000, 32000])
@pytest.mark.parametrize("trainable", [False, True])
def test_cpu_chunked_log_probs_margin_sweep(vocab, trainable):
    hidden, weight, labels, logits = _problem(vocab, torch.float32)
    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(trainable)
    log_probs = cpu_chunked_log_probs(h, w, None, labels, 1.0, None, outputs = ())[0]
    (gh,) = torch.autograd.grad(log_probs.sum(), [h])
    logp, grad = _oracle(logits, labels)
    assert (log_probs < 0).all()
    assert _rel(log_probs, logp) < 1e-5
    assert _rel(gh[range(N_TARGETS), range(N_TARGETS)], grad) < 1e-4
