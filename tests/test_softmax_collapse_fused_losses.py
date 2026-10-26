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

"""Softmax collapse (arXiv 2610.31642): a target leading the logsumexp of the others by more than ~16.6
nats used to get loss 0 and target gradient 0 from every fused path. Each route is checked against a
target-excluded fp64 oracle with relative error (fp64 F.cross_entropy itself collapses at ~36.7 nats)."""

import math

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.fused_losses import chunked_logprob
from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_ce_loss, unsloth_count_aware_cross_entropy
from unsloth_zoo import rl_replacements as rl

V = 257
MARGINS = [5.0, 17.0, 25.0, 40.0, 80.0]  # past ~87 nats float32 results are subnormal


def _logits(margin, rows = 3):
    # Row r: target (7 * r + 3) % V leads V - 1 zero logits by `margin` over their logsumexp.
    target = torch.tensor([(7 * r + 3) % V for r in range(rows)])
    z = torch.zeros(rows, V, dtype = torch.float64)
    z[torch.arange(rows), target] = margin + math.log(V - 1)
    return z, target


def _oracle(z, target):
    """fp64 log p_t and d log p_t / d z_t from the target-excluded form."""
    z = z.double().clone().requires_grad_()
    z_t = z.gather(-1, target[:, None])
    others = (z - z_t).masked_fill(F.one_hot(target, z.shape[-1]).bool(), -float("inf"))
    logp = -F.softplus(torch.logsumexp(others, -1))
    logp.sum().backward()
    return logp.detach(), z.grad[torch.arange(len(target)), target]


def _rel(a, b):
    return ((a.double() - b.double()).abs() / b.double().abs()).max().item()


def _check(logp, grad_t, z, target, sign = 1.0, tol = 1e-5):
    ref_logp, ref_grad = _oracle(z, target)
    assert (logp != 0).all() and (grad_t != 0).all(), "collapsed to 0"
    assert _rel(logp, ref_logp) < tol
    assert _rel(sign * grad_t, ref_grad) < 10 * tol


@pytest.mark.parametrize("margin", MARGINS)
def test_target_logprob_helper(margin):
    from unsloth_zoo.fused_losses.stable_softmax import target_logprob

    z, t = _logits(margin)
    x = z.float().requires_grad_()
    logp, _ = target_logprob(x, t)
    logp.sum().backward()
    _check(logp, x.grad[torch.arange(3), t], z, t)


@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_sft_fused_ce_chunk(margin, smoothing):
    z, t = _logits(margin)
    h = z.float().clone().requires_grad_()  # hidden @ I.T == logits
    loss, _ = compute_fused_ce_loss(
        h[None], torch.eye(V), None, t[None], shift_labels = False, n_items = torch.tensor(3),
        label_smoothing = smoothing,
    )
    loss.backward()
    if smoothing == 0.0:
        ref_logp, ref_grad = _oracle(z, t)
        assert abs(loss.item() + ref_logp.mean().item()) / abs(ref_logp.mean().item()) < 1e-5
        assert _rel(-3 * h.grad[torch.arange(3), t], ref_grad) < 1e-4
    else:
        ref = F.cross_entropy(z, t, label_smoothing = smoothing, reduction = "sum") / 3
        assert abs(loss.item() - ref.item()) / ref.item() < 1e-5


@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_count_aware_ce(margin, dtype):
    z, t = _logits(margin)
    x = z.to(dtype).clone().requires_grad_()
    loss = unsloth_count_aware_cross_entropy(x[None], t[None], torch.tensor(3), shift = False)
    loss.backward()
    zr = x.detach().double()  # oracle on the logits the loss actually saw
    ref_logp, ref_grad = _oracle(zr, t)
    assert abs(loss.item() + ref_logp.mean().item()) / abs(ref_logp.mean().item()) < 1e-5
    g = x.grad.float()[torch.arange(3), t]
    if dtype == torch.float32:
        assert _rel(-3 * g, ref_grad) < 1e-4
    else:
        assert (g != 0).all()


@pytest.mark.parametrize("margin", MARGINS)
def test_chunked_logprobs_and_ce(monkeypatch, margin):
    monkeypatch.setattr(chunked_logprob, "_FUSED_LINEAR", False)
    z, t = _logits(margin)
    h = z.float().clone().requires_grad_()
    logp = chunked_logprob.unsloth_chunked_logprobs(h[None], torch.eye(V), t[None], rows_per_chunk = 2)[0]
    logp.sum().backward()
    _check(logp, h.grad[torch.arange(3), t], z, t)
    h2 = z.float().clone().requires_grad_()
    loss = chunked_logprob.unsloth_chunked_ce_loss(
        h2[None], torch.eye(V), None, t[None], shift_labels = False, n_items = torch.tensor(3), rows_per_chunk = 2,
    )
    loss.backward()
    ref_logp, ref_grad = _oracle(z, t)
    assert abs(loss.item() + ref_logp.mean().item()) / abs(ref_logp.mean().item()) < 1e-5
    assert _rel(-3 * h2.grad[torch.arange(3), t], ref_grad) < 1e-4


@pytest.mark.parametrize("margin", MARGINS)
def test_grpo_selective_log_softmax(margin):
    z, t = _logits(margin)
    for fn in (rl.selective_log_softmax, lambda x, i: rl.chunked_selective_log_softmax(x[None], i[None], chunks = 2)[0]):
        x = z.float().clone().requires_grad_()
        logp = fn(x, t)
        logp.sum().backward()
        _check(logp, x.grad[torch.arange(3), t], z, t)


@pytest.mark.parametrize("margin", MARGINS)
def test_grpo_truncated_support(margin):
    z, t = _logits(margin)
    rows = torch.arange(3)
    z[rows, (t + 1) % V] = 1.0  # one runner-up per row that survives top_k = 2 (gap grows with margin)
    x = z.float().clone().requires_grad_()
    logp = rl.chunked_selective_log_softmax(x[None], t[None], chunks = 1, support_top_k = 2)[0]
    logp.sum().backward()
    kept = torch.full_like(z, -float("inf"))
    kept[rows, t] = z[rows, t]
    kept[rows, (t + 1) % V] = z[rows, (t + 1) % V]
    _check(logp, x.grad[rows, t], kept, t)


def test_grpo_truncated_support_lone_target_has_finite_grads():
    z, t = _logits(5.0)
    x = z.float().clone().requires_grad_()
    logp = rl.chunked_selective_log_softmax(x[None], t[None], chunks = 1, support_top_k = 1)[0]
    logp.sum().backward()
    assert torch.equal(logp, torch.zeros_like(logp)) and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("fused_linear", [False, True])
def test_grpo_hidden_states_scorers(monkeypatch, margin, recompute, fused_linear):
    monkeypatch.setattr(chunked_logprob, "_FUSED_LINEAR", fused_linear)
    z, t = _logits(margin)
    h = z.float().clone().requires_grad_()
    fn = rl.chunked_hidden_states_selective_log_softmax_recompute if recompute else rl.chunked_hidden_states_selective_log_softmax
    logp = fn(h[None], torch.eye(V), t[None], chunks = 2)
    logp = logp[0] if isinstance(logp, tuple) else logp
    logp = logp.reshape(-1)
    logp.sum().backward()
    _check(logp, h.grad[torch.arange(3), t], z, t)


def test_ordinary_logits_match_stock():
    from unsloth_zoo.fused_losses.stable_softmax import target_logprob

    torch.manual_seed(0)
    z = torch.randn(32, V) * 3
    t = torch.randint(0, V, (32,))
    stock = torch.log_softmax(z, -1).gather(-1, t[:, None]).squeeze(-1)
    assert torch.allclose(rl.selective_log_softmax(z, t), stock, rtol = 1e-5, atol = 1e-6)
    assert torch.allclose(target_logprob(z, t)[0], stock, rtol = 1e-5, atol = 1e-6)
    loss, _ = compute_fused_ce_loss(z[None], torch.eye(V), None, t[None], shift_labels = False, label_smoothing = 0.1)
    ref = F.cross_entropy(z, t, label_smoothing = 0.1)
    assert abs(loss.item() - ref.item()) < 1e-5 * ref.item()


def test_tie_gradient_and_empty_rows():
    from unsloth_zoo.fused_losses.stable_softmax import cross_entropy_rows, smoothed_nll

    # Target logit equal to the logsumexp of the others: d nll / dz = p - onehot = [-0.5, 0.5].
    for fn in (lambda z, t: smoothed_nll(z, t), lambda z, t: cross_entropy_rows(z, t)):
        z = torch.zeros(1, 2, requires_grad = True)
        fn(z, torch.tensor([0])).sum().backward()
        assert torch.allclose(z.grad, torch.tensor([[-0.5, 0.5]]))
    x = torch.tensor([-100.0, -30.0, 0.0, 1e-8, 20.0, 30.0, 100.0], dtype = torch.float64)
    ref = torch.log1p(torch.exp(x))
    assert torch.allclose(smoothed_nll(torch.stack([x, torch.zeros_like(x)], -1), torch.ones(7, dtype = torch.long)),
                          ref, rtol = 1e-12, atol = 0)
    compiled = torch.compile(cross_entropy_rows, backend = "eager", fullgraph = True)
    assert compiled(torch.zeros(0, 5), torch.zeros(0, dtype = torch.long)).shape == (0,)


def test_rows_ce_offset_logits_and_float64():
    from unsloth_zoo.fused_losses.stable_softmax import cross_entropy_rows

    # A large shared offset: the backward must normalise like the forward ([-0.5, 0.5], as stock).
    z = torch.tensor([[1e8, 1e8]], requires_grad = True)
    cross_entropy_rows(z, torch.tensor([0])).sum().backward()
    assert torch.equal(z.grad, torch.tensor([[-0.5, 0.5]]))
    # float64 logits keep float64 math: 1e8 and 1e8 + 1 are equal in float32.
    z = torch.tensor([[1e8, 1e8 + 1]], dtype = torch.float64, requires_grad = True)
    nll = cross_entropy_rows(z, torch.tensor([0]))
    ref = F.cross_entropy(z.detach(), torch.tensor([0]), reduction = "none")
    assert nll.dtype == torch.float64 and torch.allclose(nll, ref, rtol = 1e-12)
    nll.sum().backward()
    assert torch.allclose(z.grad, torch.softmax(z.detach(), -1) - torch.tensor([[1.0, 0.0]], dtype = torch.float64))


def test_smoothed_nll_offset_and_negative_tail():
    from unsloth_zoo.fused_losses.stable_softmax import smoothed_nll

    z = torch.full((1, 3), 1e8, requires_grad = True)
    nll = smoothed_nll(z, torch.tensor([0]))
    nll.sum().backward()
    assert torch.allclose(nll, torch.tensor([math.log(3.0)]))
    assert torch.allclose(z.grad, torch.tensor([[-2 / 3, 1 / 3, 1 / 3]]))
    # Margin 90: the non-target gradient e^-90 is a float32 subnormal, still nonzero.
    z = torch.tensor([[90.0, 0.0]], requires_grad = True)
    smoothed_nll(z, torch.tensor([0])).sum().backward()
    assert z.grad[0, 1] > 0 and z.grad[0, 0] < 0


def test_trailing_target_and_smoothing_offset():
    from unsloth_zoo.fused_losses.stable_softmax import cross_entropy_rows, smoothed_nll

    # Target 1e8 below 256 equal others: logsumexp = log 256, gradients sum to 0 (z_t - logp cancels).
    z = torch.zeros(1, 257)
    z[0, 0] = -1e8
    x = z.clone().requires_grad_()
    nll = cross_entropy_rows(x, torch.tensor([0]))
    nll.sum().backward()
    assert torch.allclose(x.grad[0, 1:], torch.full((256,), 1 / 256)) and abs(x.grad.sum().item()) < 1e-6
    x = z.clone().requires_grad_()
    smoothed_nll(x, torch.tensor([0])).sum().backward()
    assert torch.allclose(x.grad[0, 1:], torch.full((256,), 1 / 256))
    # Label smoothing is offset invariant: three equal 1e8 logits give log 3.
    s = smoothed_nll(torch.full((1, 3), 1e8), torch.tensor([0]), label_smoothing = 0.1)
    assert torch.allclose(s, torch.tensor([math.log(3.0)]))


def test_recompute_trailing_target(monkeypatch):
    monkeypatch.setattr(chunked_logprob, "_FUSED_LINEAR", False)
    z = torch.tensor([[0.0, -1e8, 0.0]])
    h = z.clone().requires_grad_()
    logp, ent = rl.chunked_hidden_states_selective_log_softmax_recompute(
        h[None], torch.eye(3), torch.tensor([[1]]), chunks = 1, return_entropy = True, fused_linear = False,
    )
    logp.sum().backward()
    assert torch.allclose(ent.reshape(-1), torch.tensor([math.log(2.0)]))
    assert torch.allclose(h.grad, torch.tensor([[-0.5, 1.0, -0.5]]))
