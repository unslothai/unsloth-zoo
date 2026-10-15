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

"""Chunked linear preference loss against a float64 full-logits oracle and TRL's own formulas."""

import math
import types

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.fused_losses.preference_loss import (
    CPO_LOSS_TYPES,
    DPO_LOSS_TYPES,
    cpo_preference_loss,
    dpo_preference_loss,
    fused_linear_preference_loss,
    fused_linear_token_logps,
    linear_mean_logits,
    orpo_preference_loss,
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


# ---- loss formulas against TRL's own methods -------------------------------------------------


def _logps(seed, n = 6):
    gen = torch.Generator().manual_seed(seed)
    draw = lambda: -torch.rand(n, generator = gen, dtype = DT) * 20 - 0.01
    return draw(), draw(), draw(), draw()


def _trl_dpo():
    trl = pytest.importorskip("trl")
    from trl import DPOTrainer

    if not hasattr(DPOTrainer, "dpo_loss"):
        pytest.skip("TRL >= 0.29 inlines the DPO loss")
    return DPOTrainer


@pytest.mark.parametrize("loss_type", DPO_LOSS_TYPES)
@pytest.mark.parametrize("beta, label_smoothing", [(0.1, 0.0), (0.5, 0.2)])
@pytest.mark.parametrize("reference_free", [False, True])
def test_dpo_loss_matches_trl(loss_type, beta, label_smoothing, reference_free):
    DPOTrainer = _trl_dpo()
    import inspect

    trl_source = inspect.getsource(DPOTrainer.dpo_loss)
    if f'"{loss_type}"' not in trl_source:
        pytest.skip(f"{loss_type} not in this TRL")
    if loss_type == "robust" and label_smoothing == 0.0:
        label_smoothing = 0.1
    stub = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(device = torch.device("cpu")),
        reference_free = reference_free, f_divergence_type = "reverse_kl", f_divergence_params = {},
        beta = beta, label_smoothing = label_smoothing, loss_type = [loss_type],
        args = types.SimpleNamespace(discopop_tau = 0.05),
    )
    chosen, rejected, ref_chosen, ref_rejected = _logps(7)
    want = DPOTrainer.dpo_loss(stub, chosen, rejected, ref_chosen, ref_rejected, loss_type)
    got = dpo_preference_loss(
        chosen, rejected, ref_chosen, ref_rejected, loss_type = loss_type, beta = beta,
        label_smoothing = label_smoothing, reference_free = reference_free,
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol = 1e-13, atol = 1e-13)


def _trl_class(name):
    trl = pytest.importorskip("trl")
    cls = getattr(trl, name, None)
    if cls is None:
        try:
            module = __import__(f"trl.experimental.{name[:-7].lower()}", fromlist = [name])
            cls = getattr(module, name)
        except Exception:
            pytest.skip(f"{name} not available")
    return cls


@pytest.mark.parametrize("beta", [0.1, 1.0])
def test_orpo_loss_matches_trl(beta):
    ORPOTrainer = _trl_class("ORPOTrainer")
    stub = types.SimpleNamespace(accelerator = types.SimpleNamespace(device = torch.device("cpu")), beta = beta)
    chosen, rejected, _, _ = _logps(8)
    chosen, rejected = chosen / 20, rejected / 20  # length-averaged logps
    want = ORPOTrainer.odds_ratio_loss(stub, chosen, rejected)
    got = orpo_preference_loss(chosen, rejected, beta = beta)
    # TRL >= 0.29 casts to float32 first; older TRL keeps float64.
    tol = 1e-13 if want[0].dtype == DT else 1e-5
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w.to(g.dtype), rtol = tol, atol = tol)


@pytest.mark.parametrize("loss_type", CPO_LOSS_TYPES)
@pytest.mark.parametrize("alpha", [0.0, 0.3])
def test_cpo_loss_matches_trl(loss_type, alpha):
    CPOTrainer = _trl_class("CPOTrainer")
    stub = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(device = torch.device("cpu")), beta = 0.2, loss_type = loss_type,
        label_smoothing = 0.1, simpo_gamma = 0.7, alpha = alpha,
    )
    chosen, rejected, _, _ = _logps(9)
    chosen, rejected = chosen / 20, rejected / 20
    want = CPOTrainer.cpo_loss(stub, chosen, rejected)
    got = cpo_preference_loss(
        chosen, rejected, loss_type = loss_type, beta = 0.2, label_smoothing = 0.1, simpo_gamma = 0.7, alpha = alpha,
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol = 1e-13, atol = 1e-13)


# ---- end to end from hidden states -------------------------------------------------------------


def _full_logits_preference(method, loss_type, states, weight, labels, mask, ref_states = None, **kw):
    logprobs = F.linear(states, weight).log_softmax(-1)
    per_token = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
    n = states.shape[0] // 2
    average = method == "orpo" or (method == "dpo" and loss_type == "ipo") or (method == "cpo" and loss_type in ("ipo", "simpo"))
    logps = per_token.sum(-1) / (mask.sum(-1) if average else 1)
    if method == "dpo":
        with torch.no_grad():
            ref = F.linear(ref_states, weight).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
            ref = ref.sum(-1) / (mask.sum(-1) if average else 1)
        losses, cr, rr = dpo_preference_loss(logps[:n], logps[n:], ref[:n], ref[n:], loss_type = loss_type, **kw)
        return losses.mean(), cr, rr
    nll = -(per_token[:n]).sum() / mask[:n].sum()
    if method == "orpo":
        losses, cr, rr, _, _ = orpo_preference_loss(logps[:n], logps[n:], **kw)
        return nll - losses.mean(), cr, rr
    losses, cr, rr = cpo_preference_loss(logps[:n], logps[n:], loss_type = loss_type, **kw)
    return losses.mean() + nll, cr, rr


@pytest.mark.parametrize(
    "method, loss_type",
    [("dpo", "sigmoid"), ("dpo", "ipo"), ("dpo", "hinge"), ("orpo", "sigmoid"), ("cpo", "sigmoid"), ("cpo", "simpo")],
)
def test_fused_preference_loss_end_to_end(method, loss_type):
    states, weight, _, labels, mask = _tensors(10, rows = 4)
    mask[:, 0] = True  # no empty completion
    ref_states = (states + 0.1 * torch.randn_like(states)).detach()
    kwargs = dict(beta = 0.1)
    loss, metrics = fused_linear_preference_loss(
        states, weight, None, labels, mask, method = method, loss_type = loss_type,
        ref_hidden_states = ref_states if method == "dpo" else None, chunk_size = 5, cpo_alpha = 1.0, **kwargs,
    )
    states_b = states.detach().clone().requires_grad_(True)
    weight_b = weight.detach().clone().requires_grad_(True)
    want, cr, rr = _full_logits_preference(method, loss_type, states_b, weight_b, labels, mask, ref_states, **kwargs)
    torch.testing.assert_close(loss, want, rtol = 0, atol = 1e-12)
    assert metrics["rewards/chosen"].item() == pytest.approx(cr.mean().item(), abs = 1e-12)
    assert metrics["rewards/margins"].item() == pytest.approx((cr - rr).mean().item(), abs = 1e-12)
    assert metrics["rewards/accuracies"].item() == pytest.approx((cr > rr).double().mean().item(), abs = 1e-12)
    loss.backward()
    want.backward()
    torch.testing.assert_close(states.grad, states_b.grad, rtol = 0, atol = 1e-11)
    torch.testing.assert_close(weight.grad, weight_b.grad, rtol = 0, atol = 1e-11)


@pytest.mark.parametrize(
    "method, loss_type, extra",
    [("dpo", "discopop", dict(discopop_tau = 0.3)), ("cpo", "simpo", dict(alpha = 0.4)), ("cpo", "sigmoid", dict(alpha = -0.2))],
)
def test_fused_preference_loss_forwards_discopop_tau_and_alphapo_alpha(method, loss_type, extra):
    states, weight, _, labels, mask = _tensors(12, rows = 4)
    mask[:, 0] = True
    ref_states = (states + 0.1 * torch.randn_like(states)).detach()
    loss, _ = fused_linear_preference_loss(
        states, weight, None, labels, mask, method = method, loss_type = loss_type, beta = 0.1,
        ref_hidden_states = ref_states if method == "dpo" else None, chunk_size = 5, **extra,
    )
    want, _, _ = _full_logits_preference(method, loss_type, states.detach(), weight.detach(), labels, mask, ref_states, beta = 0.1, **extra)
    default, _, _ = _full_logits_preference(method, loss_type, states.detach(), weight.detach(), labels, mask, ref_states, beta = 0.1)
    torch.testing.assert_close(loss.detach(), want, rtol = 0, atol = 1e-12)
    assert abs(float(want - default)) > 1e-6


def test_dpo_policy_equals_reference_is_log2_and_precomputed_matches():
    states, weight, _, labels, mask = _tensors(11, rows = 4)
    loss, metrics = fused_linear_preference_loss(
        states, weight, None, labels, mask, ref_hidden_states = states.detach(),
    )
    assert loss.item() == pytest.approx(math.log(2), abs = 1e-12)
    assert metrics["rewards/margins"].item() == pytest.approx(0.0, abs = 1e-12)
    per_token = fused_linear_token_logps(states.detach(), weight.detach(), None, labels, mask)["per_token_logps"]
    ref = per_token.sum(-1)
    loss2, _ = fused_linear_preference_loss(
        states, weight, None, labels, mask, ref_chosen_logps = ref[:2], ref_rejected_logps = ref[2:],
    )
    torch.testing.assert_close(loss2, loss, rtol = 0, atol = 1e-13)


def test_alphapo_and_discopop_stay_finite_on_long_completions():
    # float32: exp(-200) underflows (AlphaPO) and exp(150) overflows against a zero modulation (DiscoPOP).
    chosen = torch.tensor([-200.0, -2.0], requires_grad = True)
    rejected = torch.tensor([-210.0, -3.0], requires_grad = True)
    for alpha in (0.1, -0.2):
        losses = cpo_preference_loss(chosen, rejected, loss_type = "sigmoid", beta = 0.1, alpha = alpha)[0]
        grads = torch.autograd.grad(losses.sum(), (chosen, rejected))
        assert torch.isfinite(losses).all() and all(torch.isfinite(g).all() for g in grads)
        exact = cpo_preference_loss(chosen.double(), rejected.double(), loss_type = "sigmoid", beta = 0.1, alpha = alpha)[0]
        torch.testing.assert_close(losses, exact.float())
    zeros = torch.zeros(2)
    chosen = torch.tensor([-1600.0, -2.0], requires_grad = True)
    rejected = torch.tensor([-100.0, -3.0], requires_grad = True)
    losses = dpo_preference_loss(chosen, rejected, zeros, zeros, loss_type = "discopop", beta = 0.1)[0]
    grads = torch.autograd.grad(losses.sum(), (chosen, rejected))
    assert torch.isfinite(losses).all() and all(torch.isfinite(g).all() for g in grads)
    exact = dpo_preference_loss(chosen.double(), rejected.double(), zeros.double(), zeros.double(), loss_type = "discopop", beta = 0.1)[0]
    torch.testing.assert_close(losses, exact.float())
