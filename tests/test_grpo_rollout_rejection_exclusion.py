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

"""Rollout rejection (UNSLOTH_GRPO_ROLLOUT_RS) removes a rejected token from every loss term.

verl compute_rollout_rejection_mask returns response_mask * keep and the trainer replaces the
batch response_mask with it, so the policy term and the KL loss both drop rejected tokens. Here a
rejected token must contribute 0 to the loss and exactly 0 gradient, policy and KL alike, and a
rejected token whose vLLM IS ratio overflowed to inf must not turn into nan (inf * 0). The loss
normaliser keeps its documented TRL *_mask convention (rejected tokens stay counted).
"""
import math

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

EPS_LOW, EPS_HIGH = 0.2, 0.28


def _run(ref, new, old, sampling, mask, adv, beta, rollout_rs, loss_type = "dapo", mode = "token_mask",
         clip = (None, None), level = "token"):
    kwargs = dict(
        loss_type = loss_type, epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, max_completion_length = new.shape[1],
        num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = 1, steps_per_generation = 1,
        num_processes = 1, importance_sampling_level = level, use_vllm = True,
        vllm_importance_sampling_correction = True, vllm_importance_sampling_mode = mode,
        vllm_importance_sampling_clip_min = clip[0], vllm_importance_sampling_clip_max = clip[1],
        use_bias_correction_kl = False,
    )
    if rollout_rs is not None:
        kwargs["rollout_rs"] = rollout_rs
    out = rr.grpo_compute_loss(ref, new, old, sampling, torch.zeros_like(mask), mask, beta, adv, **kwargs)
    return out[0], out[4]


def _oracle_keep(old, sampling, mask, parsed):
    # float64 restatement of verl compute_rollout_rejection_mask (log_ratio = old - sampling, clamp 20).
    m = mask.double()
    lr = torch.nan_to_num((old.double() - sampling.double()) * m, nan = 0.0).clamp(-20, 20)
    keep = torch.ones_like(lr)
    for opt, lo, hi in parsed:
        if opt.endswith("k1"):
            s = -lr
        elif opt.endswith("k2"):
            s = 0.5 * lr * lr
        else:
            s = lr.exp() - 1 - lr
        s = s * m
        if opt.startswith("seq_sum"):
            s = s.sum(-1, keepdim = True)
        elif opt.startswith("seq_mean"):
            s = s.sum(-1, keepdim = True) / m.sum(-1, keepdim = True).clamp(min = 1)
        elif opt.startswith("seq_max"):
            s = s.amax(-1, keepdim = True)
        ok = (s <= hi) if lo is None else ((s >= lo) & (s <= hi))
        keep = keep * ok.double()
    return keep.expand_as(lr)


def _oracle_loss(ref, new, old, sampling, mask, adv, beta, parsed, loss_type, clip = (None, None)):
    """float64 objective: rejected tokens excluded from policy and KL; normaliser unchanged."""
    m = mask.double()
    new, old, sampling = new.double(), old.double(), sampling.double()
    lr = torch.nan_to_num((old - sampling) * m, nan = 0.0)
    keep = _oracle_keep(old, sampling, mask, parsed) * m
    # Selected before exp so a rejected token never produces inf.
    ratio = torch.where(keep > 0, lr, torch.zeros_like(lr)).exp()
    lo = -math.inf if clip[0] is None else clip[0]
    hi = math.inf if clip[1] is None else clip[1]
    ratio = torch.where((ratio < lo) | (ratio > hi), torch.zeros_like(ratio), ratio)
    coef = torch.where(m > 0, new - old, torch.zeros_like(new)).exp()
    a = adv.double().unsqueeze(1)
    pol = -torch.min(coef * a, coef.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * a) * ratio
    per_token = pol
    if beta != 0.0:
        d = torch.where(m > 0, ref.double() - new, torch.zeros_like(new)).clamp(-20, 20)
        per_token = per_token + beta * (d.exp() - d - 1).clamp(-10, 10)
    per_token = torch.where(keep > 0, per_token, torch.zeros_like(per_token))
    if loss_type == "dapo":
        return per_token.sum() / m.sum().clamp(min = 1)
    if loss_type == "grpo":
        return (per_token.sum(-1) / m.sum(-1).clamp(min = 1)).mean()
    if loss_type == "dr_grpo":
        return per_token.sum() / (per_token.shape[0] * per_token.shape[1])
    raise ValueError(loss_type)


def _data(seed, rows = 6, cols = 9, scale = 0.2, dtype = torch.float64):
    g = torch.Generator().manual_seed(seed)
    new = -torch.rand(rows, cols, generator = g, dtype = torch.float64) * 3
    old = new + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.05
    ref = new + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.3
    sampling = old + torch.randn(rows, cols, generator = g, dtype = torch.float64) * scale
    mask = (torch.rand(rows, cols, generator = g) > 0.2).long()
    mask[1, 1:] = 0
    adv = torch.randn(rows, generator = g, dtype = torch.float64)
    cast = lambda t: t.to(dtype)
    return cast(ref), cast(new).requires_grad_(True), cast(old), cast(sampling), mask, cast(adv)


def test_overflowed_is_ratio_on_rejected_token_is_not_nan():
    # Review repro: token_mask with no clip, log ratios [0, 100] in float32. exp(100) = inf and the
    # rejected token used to become inf * 0 = nan in the loss and in the gradient.
    new = torch.zeros(1, 2, dtype = torch.float32, requires_grad = True)
    old = torch.zeros(1, 2, dtype = torch.float32)
    sampling = torch.tensor([[0.0, -100.0]])
    mask = torch.ones(1, 2, dtype = torch.long)
    adv = torch.ones(1)
    parsed = rr.grpo_parse_rollout_rs("token_k3", "0.1")
    loss, flat = _run(None, new, old, sampling, mask, adv, 0.0, parsed)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all() and torch.isfinite(flat).all()
    # Survivor: ratio 1, coef 1, advantage 1 -> -1; dapo divides by the 2 valid tokens.
    torch.testing.assert_close(loss, torch.tensor(-0.5), rtol = 0, atol = 1e-7)
    assert new.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rejected_row_carries_no_kl_gradient(dtype):
    # Review repro: two two-token rows, the first rejected by seq_mean_k3 0.01, zero advantages,
    # beta 0.1, ref - new = 1. The rejected row used to keep a -0.043 KL gradient per token.
    new = torch.zeros(2, 2, dtype = dtype, requires_grad = True)
    ref = torch.ones(2, 2, dtype = dtype)
    old = torch.zeros(2, 2, dtype = dtype)
    sampling = torch.tensor([[-1.0, -1.0], [0.0, 0.0]], dtype = dtype)
    mask = torch.ones(2, 2, dtype = torch.long)
    adv = torch.zeros(2, dtype = dtype)
    parsed = rr.grpo_parse_rollout_rs("seq_mean_k3", "0.01")
    loss, _ = _run(ref, new, old, sampling, mask, adv, 0.1, parsed)
    loss.backward()
    assert torch.equal(new.grad[0], torch.zeros(2, dtype = dtype))
    # Kept row: k3(1) = e - 2 per token, its gradient d/dnew beta * (e^(ref-new) - (ref-new) - 1) / 4.
    kept = 0.1 * (1.0 - math.e) / 4
    torch.testing.assert_close(new.grad[1], torch.full((2,), kept, dtype = dtype), rtol = 1e-6, atol = 0)
    torch.testing.assert_close(loss, torch.tensor(0.1 * 2 * (math.e - 2) / 4, dtype = dtype), rtol = 1e-6, atol = 0)


RS = [("token_k3", "0.02"), ("seq_mean_k3", "0.01"), ("seq_max_k2", "0.05"), ("token_k1", "0.8_1.25"),
      ("token_k3,seq_mean_k3", "0.05,0.01")]


@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("loss_type", ["dapo", "grpo", "dr_grpo"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("seed", [0, 3])
def test_rejection_with_kl_matches_fp64_oracle(options, thresholds, loss_type, beta, seed):
    ref, new, old, sampling, mask, adv = _data(seed)
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    keep = _oracle_keep(old, sampling, mask, parsed) * mask
    assert 0 < int(((keep == 0) & (mask > 0)).sum()), "the threshold should reject something"
    loss, _ = _run(ref, new, old, sampling, mask, adv, beta, parsed, loss_type = loss_type)
    want = _oracle_loss(ref, new, old, sampling, mask, adv, beta, parsed, loss_type)
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-15)
    loss.backward()
    rejected = (keep == 0) & (mask > 0)
    assert torch.equal(new.grad[rejected], torch.zeros_like(new.grad[rejected]))
    oracle_new = new.detach().clone().requires_grad_(True)
    _oracle_loss(ref, oracle_new, old, sampling, mask, adv, beta, parsed, loss_type).backward()
    torch.testing.assert_close(new.grad, oracle_new.grad, rtol = 1e-10, atol = 1e-14)


@pytest.mark.parametrize("options,thresholds", RS[:2])
def test_rejection_gradcheck_fp64(options, thresholds):
    ref, new, old, sampling, mask, adv = _data(1)
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    fn = lambda n: _run(ref, n, old, sampling, mask, adv, 0.04, parsed)[0]
    assert torch.autograd.gradcheck(fn, (new,), eps = 1e-6, atol = 1e-8)


@pytest.mark.parametrize("gap", [20.0, 88.0, 100.0, 1e4])
@pytest.mark.parametrize("mode,clip", [("token_mask", (None, None)), ("sequence_mask", (None, None)),
                                       ("token_truncate", (None, math.inf)), ("geometric_mask", (None, None))])
def test_large_gap_on_rejected_tokens_stays_finite_fp32(gap, mode, clip):
    ref, new, old, sampling, mask, adv = _data(2, dtype = torch.float32)
    sampling = sampling.clone()
    sampling[0, :] = old[0, :] - gap  # row 0 far off-policy, rejected by seq_max_k3 and token_k3
    parsed = rr.grpo_parse_rollout_rs("token_k3,seq_max_k3", "0.05,0.05")
    loss, flat = _run(ref, new, old, sampling, mask, adv, 0.04, parsed, mode = mode, clip = clip)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(flat).all() and torch.isfinite(new.grad).all()
    assert torch.equal(new.grad[0], torch.zeros_like(new.grad[0]))


def test_default_path_without_rejection_keeps_kl_on_is_masked_tokens():
    # The TRL *_mask modes zero only the policy term: TRL adds beta * KL after the IS product.
    ref, new, old, sampling, mask, adv = _data(4, scale = 1.0)
    loss, flat = _run(ref, new, old, sampling, mask, adv, 0.04, None, mode = "token_mask", clip = (0.5, 2.0))
    assert ((flat == 0) & (mask > 0)).any()
    loss.backward()
    masked = (flat == 0) & (mask > 0)
    assert (new.grad[masked] != 0).all()


# Rollout logprobs as pi_old (UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD, verl bypass mode): sampling is
# dropped, old = rollout. verl compute_policy_loss_bypass_mode still runs
# compute_rollout_correction_and_rejection_mask with old_log_prob = the current log_prob, so the
# rejection statistic is on log(pi_theta / pi_rollout) and the rejected tokens leave the loss.


def _run_bypass(ref, new, rollout, mask, adv, beta, rollout_rs, loss_type = "dapo"):
    old, sampling = rr.grpo_rollout_logps_as_old(rollout)
    kwargs = dict(
        loss_type = loss_type, epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, max_completion_length = new.shape[1],
        num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = 1, steps_per_generation = 1,
        num_processes = 1, importance_sampling_level = "token", use_vllm = True,
        vllm_importance_sampling_correction = True, rollout_logps_as_old = True, use_bias_correction_kl = False,
    )
    if rollout_rs is not None:
        kwargs["rollout_rs"] = rollout_rs
    return rr.grpo_compute_loss(ref, new, old, sampling, torch.zeros_like(mask), mask, beta, adv, **kwargs)[0]


def _oracle_bypass(ref, new, rollout, mask, adv, beta, parsed, loss_type):
    m = mask.double()
    new, rollout = new.double(), rollout.double()
    keep = _oracle_keep(new.detach(), rollout, mask, parsed) * m
    lr = torch.where(m > 0, new - rollout, torch.zeros_like(new)).clamp(-20, 20)
    coef = lr.exp()
    a = adv.double().unsqueeze(1)
    per_token = -torch.min(coef * a, coef.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * a)
    if beta != 0.0:
        d = torch.where(m > 0, ref.double() - new, torch.zeros_like(new)).clamp(-20, 20)
        per_token = per_token + beta * (d.exp() - d - 1).clamp(-10, 10)
    per_token = torch.where(keep > 0, per_token, torch.zeros_like(per_token))
    if loss_type == "dapo":
        return per_token.sum() / m.sum().clamp(min = 1)
    return (per_token.sum(-1) / m.sum(-1).clamp(min = 1)).mean()


def test_bypass_mode_applies_rejection():
    # Review repro (zoo#1927): new = -2, rollout = -1, advantage 1, token_k1 bounds [0.5, 2]. The
    # statistic -log(pi_theta / pi_rollout) = 1 > log 2 rejects the token; main ignored the filter
    # and returned -exp(-1).
    new = torch.full((1, 1), -2.0, dtype = torch.float64, requires_grad = True)
    rollout = torch.full((1, 1), -1.0, dtype = torch.float64)
    mask = torch.ones(1, 1, dtype = torch.long)
    loss = _run_bypass(None, new, rollout, mask, torch.ones(1, dtype = torch.float64), 0.0,
                       rr.grpo_parse_rollout_rs("token_k1", "0.5_2"))
    loss.backward()
    assert loss.item() == 0.0 and new.grad.item() == 0.0


@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("loss_type", ["dapo", "grpo"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_bypass_rejection_matches_fp64_oracle(options, thresholds, loss_type, beta):
    ref, new, old, sampling, mask, adv = _data(7)
    rollout = sampling.float().double()  # grpo_rollout_logps_as_old hands the loss float32 rollout logps
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    keep = _oracle_keep(new.detach(), rollout, mask, parsed) * mask
    assert 0 < int(((keep == 0) & (mask > 0)).sum()), "the threshold should reject something"
    loss = _run_bypass(ref, new, rollout, mask, adv, beta, parsed, loss_type)
    torch.testing.assert_close(loss, _oracle_bypass(ref, new, rollout, mask, adv, beta, parsed, loss_type),
                               rtol = 1e-12, atol = 1e-15)
    loss.backward()
    oracle_new = new.detach().clone().requires_grad_(True)
    _oracle_bypass(ref, oracle_new, rollout, mask, adv, beta, parsed, loss_type).backward()
    torch.testing.assert_close(new.grad, oracle_new.grad, rtol = 1e-10, atol = 1e-14)


def test_bypass_without_rejection_is_unchanged():
    ref, new, old, sampling, mask, adv = _data(8)
    plain = _run_bypass(ref, new, sampling, mask, adv, 0.04, None)
    wide = _run_bypass(ref, new, sampling, mask, adv, 0.04, rr.grpo_parse_rollout_rs("seq_mean_k3", "1e9"))
    assert torch.equal(plain, wide)


def test_bypass_metrics_report_gaps_past_the_loss_clamp():
    # Review repro: gaps 30 and 20 nats. coef_1 is clamped to e^20 in the loss; |log r| must stay raw.
    mask = torch.ones(1, 2, dtype = torch.long)
    raw = torch.tensor([[30.0, 20.0]], dtype = torch.float64)
    coef_1 = raw.clamp(-20, 20).exp().float()
    delta, ratio = rr.grpo_rollout_as_old_ratio_metrics(coef_1, mask, raw)
    torch.testing.assert_close(delta, raw.float())
    torch.testing.assert_close(ratio, coef_1)


@pytest.mark.parametrize("as_old", ["0", "1"])
def test_rollout_rs_warning_follows_bypass_support(monkeypatch, as_old):
    # Without vLLM IS correction rejection only runs with rollout logps as pi_old, so only the
    # other case is reported as ignored.
    import types
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_RS", "token_k3")
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", as_old)
    seen = []
    monkeypatch.setattr(rr.logger, "warning", lambda msg, *a, **k: seen.append(msg))
    trainer = types.SimpleNamespace(args = types.SimpleNamespace(top_entropy_quantile = 1.0), use_vllm = True,
                                    vllm_importance_sampling_correction = False)
    rr._warn_unsupported_grpo_options(trainer)
    assert any("UNSLOTH_GRPO_ROLLOUT_RS" in m for m in seen) == (as_old == "0")


@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("bypass", [False, True])
def test_reported_kl_leaves_out_rejected_tokens(options, thresholds, bypass):
    # mean_kl: per-row sum of KL over kept tokens / all completion tokens of that row (the loss
    # normaliser's count), averaged over the rows that have tokens.
    ref, new, old, sampling, mask, adv = _data(5)
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    if bypass:
        rollout = sampling.float().double()
        o, s = rr.grpo_rollout_logps_as_old(rollout)
        keep = _oracle_keep(new.detach(), rollout, mask, parsed) * mask
        extra = dict(rollout_logps_as_old = True)
    else:
        o, s = old, sampling
        keep = _oracle_keep(old, sampling, mask, parsed) * mask
        extra = dict(vllm_importance_sampling_mode = "token_mask")
    assert 0 < int(((keep == 0) & (mask > 0)).sum())
    out = rr.grpo_compute_loss(
        ref, new, o, s, torch.zeros_like(mask), mask, 0.04, adv, loss_type = "dapo", num_items_in_batch = mask.sum(),
        steps_per_generation = 1, use_vllm = True, vllm_importance_sampling_correction = True, rollout_rs = parsed,
        max_completion_length = new.shape[1], **extra,
    )
    m = mask.double()
    d = torch.where(m > 0, ref - new.detach(), torch.zeros_like(ref)).clamp(-20, 20)
    k3 = (d.exp() - d - 1).clamp(-10, 10) * keep
    n = m.sum(-1)
    rows = n > 0
    want = (k3.sum(-1)[rows] / n[rows]).mean()
    torch.testing.assert_close(out[2].double(), want, rtol = 1e-6, atol = 1e-9)


def test_reported_kl_without_rejection_is_unchanged():
    ref, new, old, sampling, mask, adv = _data(6)
    kw = dict(loss_type = "dapo", num_items_in_batch = mask.sum(), steps_per_generation = 1, use_vllm = True,
              vllm_importance_sampling_correction = True, max_completion_length = new.shape[1])
    a = rr.grpo_compute_loss(ref, new, old, sampling, torch.zeros_like(mask), mask, 0.04, adv, **kw)[2]
    b = rr.grpo_compute_loss(ref, new, old, sampling, torch.zeros_like(mask), mask, 0.04, adv,
                             rollout_rs = rr.grpo_parse_rollout_rs("seq_mean_k3", "1e9"), **kw)[2]
    assert torch.equal(a, b)


@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("bypass", [False, True])
def test_keep_helper_matches_the_loss(options, thresholds, bypass):
    # grpo_rollout_rs_keep (caller side) must reject exactly the tokens grpo_compute_loss rejects.
    ref, new, old, sampling, mask, adv = _data(9)
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    if bypass:
        o, s = rr.grpo_rollout_logps_as_old(sampling)
        keep = rr.grpo_rollout_rs_keep(new.detach() - o, mask, parsed)
        loss = _run_bypass(ref, new, sampling, mask, adv, 0.04, parsed)
    else:
        keep = rr.grpo_rollout_rs_keep(old - sampling, mask, parsed)
        loss, _ = _run(ref, new, old, sampling, mask, adv, 0.04, parsed)
    loss.backward()
    rejected = mask.bool() & ~keep
    assert int(rejected.sum()) > 0
    assert torch.all(new.grad[rejected] == 0) and torch.all(new.grad[keep & mask.bool()] != 0)
