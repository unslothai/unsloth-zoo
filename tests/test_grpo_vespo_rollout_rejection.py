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

"""VESPO gamma weights: TRL parity without rollout_rs, and a rollout_rs rejected token leaves the
sequence weight (verl response_mask *= keep) instead of moving the kept tokens' gradients."""
import math

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

trl_grpo = pytest.importorskip("trl.trainer.grpo_trainer")
if not hasattr(trl_grpo.GRPOTrainer, "get_gamma_weights"):
    pytest.skip("TRL without VESPO (added in TRL 1.0.0)", allow_module_level = True)
gamma = trl_grpo.GRPOTrainer.get_gamma_weights

VESPO = dict(k_pos = 2.0, lambda_pos = 3.0, k_neg = 3.0, lambda_neg = 2.0)


def _zoo(new, old, sampling, mask, adv, beta = 0.0, ref = None, mode = "token_mask", clip = (None, 3.0),
         level = "token", rollout_rs = None, bypass = False):
    kw = dict(
        loss_type = "vespo", max_completion_length = mask.shape[1], num_items_in_batch = mask.sum(),
        current_gradient_accumulation_steps = 1, steps_per_generation = 1, num_processes = 1,
        importance_sampling_level = level, use_vllm = True, vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = mode, vllm_importance_sampling_clip_min = clip[0],
        vllm_importance_sampling_clip_max = clip[1], get_gamma_weights = gamma,
        vespo_k_pos = 2.0, vespo_lambda_pos = 3.0, vespo_k_neg = 3.0, vespo_lambda_neg = 2.0,
    )
    if rollout_rs is not None:
        kw["rollout_rs"] = rollout_rs
    if bypass:
        kw["rollout_logps_as_old"] = True
        old, sampling = rr.grpo_rollout_logps_as_old(sampling)
    return rr.grpo_compute_loss(ref, new, old, sampling, torch.zeros_like(mask), mask, beta, adv, **kw)[0]


def _trl_is_ratio(old, sampling, mask, mode, clip):
    """TRL _generate_and_score_completions vLLM IS ratio, token modes."""
    diff = torch.nan_to_num((old - sampling) * mask, nan = 0.0)
    ratio = torch.exp(diff)
    if mode == "token_truncate":
        return torch.clamp(ratio, min = clip[0], max = clip[1])
    lo = clip[0] if clip[0] is not None else -math.inf
    hi = clip[1] if clip[1] is not None else math.inf
    return ratio.masked_fill((ratio < lo) | (ratio > hi), 0.0)


def _trl_vespo(new, old, mask, adv, is_ratio, beta = 0.0, ref = None, normalizer = None):
    """TRL GRPOTrainer._compute_loss, loss_type="vespo", in float64."""
    m = mask.double()
    a = adv.double().unsqueeze(1)
    phi = gamma(advantages = a, log_ratio_per_token = new - old, mask = m, importance_sampling_ratio = is_ratio, **VESPO)
    per_token = -phi * a * new
    if beta != 0.0:
        d = ref - new
        per_token = per_token + beta * (torch.exp(d) - d - 1)
    if normalizer is None:
        normalizer = m.sum().clamp(min = 1.0)
    return (per_token * m).sum() / normalizer


def _data(seed, rows = 6, cols = 9, spread = 0.4):
    g = torch.Generator().manual_seed(seed)
    new = -torch.rand(rows, cols, generator = g, dtype = torch.float64) * 3
    old = new + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.05
    ref = new + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.3
    sampling = old + torch.randn(rows, cols, generator = g, dtype = torch.float64) * spread
    mask = (torch.rand(rows, cols, generator = g) > 0.2).long()
    mask[1, 1:] = 0
    mask[2, 0] = 0
    mask[3] = 0
    adv = torch.randn(rows, generator = g, dtype = torch.float64)
    adv[0] = 0.0
    # Ratios outside every clip window used below, so TRL's token_mask / truncate really fire.
    sampling[4, 2] = old[4, 2] - 2.0
    sampling[5, 3] = old[5, 3] + 2.0
    mask[4, 2] = mask[5, 3] = 1
    return ref, new, old, sampling, mask, adv


def _grad(fn, new):
    new = new.detach().clone().requires_grad_(True)
    loss = fn(new)
    loss.backward()
    return loss.detach(), new.grad


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode,clip", [("token_mask", (None, 3.0)), ("token_mask", (0.8, 1.25)),
                                       ("token_truncate", (None, 3.0)), ("token_truncate", (0.5, 1.5))])
@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_vespo_without_rollout_rs_matches_trl(seed, mode, clip, level, beta):
    # TRL's own token_mask keeps every completion token in the gamma sum (IS factor 0 -> log 1e-8).
    ref, new, old, sampling, mask, adv = _data(seed)
    is_ratio = _trl_is_ratio(old, sampling, mask, mode, clip)
    if mode == "token_mask":
        assert (is_ratio[mask > 0] == 0).any(), "the case must exercise TRL's token mask"
    lz, gz = _grad(lambda n: _zoo(n, old, sampling, mask, adv, beta, ref, mode, clip, level), new)
    lt, gt = _grad(lambda n: _trl_vespo(n, old, mask, adv, is_ratio, beta, ref), new)
    torch.testing.assert_close(lz, lt, rtol = 1e-12, atol = 1e-12)
    torch.testing.assert_close(gz * mask, gt * mask, rtol = 1e-12, atol = 1e-12)


def _compact(x, keep, fill):
    """Move each row's kept tokens to the front, as if rejected tokens were never generated."""
    out = torch.full_like(x, fill)
    for r in range(x.shape[0]):
        idx = keep[r].nonzero().flatten()
        out[r, : idx.numel()] = x[r, idx]
    return out


def _dropped_oracle(new, old, sampling, mask, adv, keep, mode, clip, beta, ref, bypass):
    """TRL VESPO run on the sequences with the rejected tokens physically removed; the normaliser still
    counts every completion token (as on the zoo rollout_rs path)."""
    k = (keep & (mask > 0))
    m = _compact(mask.double(), k, 0.0)
    n = _compact(new, k, 0.0)
    if bypass:
        o = _compact(sampling.float().double(), k, 0.0)
        is_ratio = None
        lr_clamped = True
    else:
        o = _compact(old, k, 0.0)
        is_ratio = _trl_is_ratio(o, _compact(sampling, k, 0.0), m, mode, clip)
        lr_clamped = False
    a = adv.double().unsqueeze(1)
    lr = n - o
    if lr_clamped:
        lr = lr.clamp(-20, 20)
    phi = gamma(advantages = a, log_ratio_per_token = lr, mask = m, importance_sampling_ratio = is_ratio, **VESPO)
    per_token = -phi * a * n
    if beta != 0.0:
        d = _compact(ref, k, 0.0) - n
        per_token = per_token + beta * (torch.exp(d) - d - 1)
    return (per_token * m).sum() / mask.double().sum().clamp(min = 1.0)


RS = [("token_k3", "0.05"), ("token_k1", "0.8_1.25"), ("token_k2", "0.02"), ("token_k3,token_k1", "0.05,0.7_1.4")]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("mode,clip", [("token_mask", (None, None)), ("token_truncate", (None, 3.0))])
@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("bypass", [False, True])
def test_vespo_rollout_rs_matches_dropped_token_oracle(seed, options, thresholds, mode, clip, level, beta, bypass):
    ref, new, old, sampling, mask, adv = _data(seed)
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    gap = (new - sampling.float().double()) if bypass else (old - sampling)
    keep = rr.grpo_rollout_rs_keep(gap, mask, parsed)
    kept = keep & (mask > 0)
    assert (~keep & (mask > 0)).any() and kept.any(), "the case must reject some and keep some tokens"
    lz, gz = _grad(lambda n: _zoo(n, old, sampling, mask, adv, beta, ref, mode, clip, level, parsed, bypass), new)
    lo, go = _grad(lambda n: _dropped_oracle(n, old, sampling, mask, adv, keep, mode, clip, beta, ref, bypass), new)
    torch.testing.assert_close(lz, lo, rtol = 1e-10, atol = 1e-12)
    # The oracle compacts inside autograd, so its gradient is already at the original positions.
    torch.testing.assert_close(gz * mask, go, rtol = 1e-10, atol = 1e-12)


def test_rejected_token_does_not_move_kept_token_grad():
    # Token 2 is rollout_rs rejected (token_k3 = e - 2 > 0.1); its new logprob must not matter.
    old = torch.tensor([[-1.0, -20.0]], dtype = torch.float64)
    sampling = torch.tensor([[-1.0, -21.0]], dtype = torch.float64)
    mask = torch.ones(1, 2, dtype = torch.long)
    adv = torch.ones(1, dtype = torch.float64)
    parsed = rr.grpo_parse_rollout_rs("token_k3", "0.1")
    grads = []
    for second in (-20.0, -1.0, -50.0):
        new = torch.tensor([[-1.0, second]], dtype = torch.float64)
        _, g = _grad(lambda n: _zoo(n, old, sampling, mask, adv, mode = "token_mask", rollout_rs = parsed), new)
        grads.append(g)
        assert g[0, 1].item() == 0.0
    # Kept token alone: log w = 0, phi = 1, grad = -A / N = -0.5.
    for g in grads:
        assert g[0, 0].item() == pytest.approx(-0.5, rel = 1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("bypass", [False, True])
def test_rejected_tokens_perturbation_invariance(seed, bypass):
    ref, new, old, sampling, mask, adv = _data(seed)
    parsed = rr.grpo_parse_rollout_rs("token_k3", "0.05")
    gap = (new - sampling.float().double()) if bypass else (old - sampling)
    keep = rr.grpo_rollout_rs_keep(gap, mask, parsed)
    rejected = ~keep & (mask > 0)
    assert rejected.any()
    g = torch.Generator().manual_seed(100 + seed)
    shift = torch.randn(new.shape, generator = g, dtype = torch.float64) * 5.0
    new2 = torch.where(rejected, new + shift, new)
    if bypass:
        # Rejection reads new - rollout: push that gap further out on its own side.
        new2 = torch.where(rejected, new + torch.sign(gap) * shift.abs(), new)
        sampling2 = sampling
        gap2 = new2 - sampling2.float().double()
        old2 = old
    else:
        # Same rollout gap (same rejection) but a policy ratio of about 1 / 1e-8 on each rejected token,
        # cancelling its zeroed IS factor's log(1e-8): before the fix this un-crushes the sequence weight.
        lift = math.log(1e-8)
        new2 = torch.where(rejected, new + 0.1 * shift, new)
        old2 = torch.where(rejected, old + lift, old)
        sampling2 = torch.where(rejected, sampling + lift, sampling)
        gap2 = old2 - sampling2
    keep2 = rr.grpo_rollout_rs_keep(gap2, mask, parsed)
    assert torch.equal(keep, keep2)
    _, g1 = _grad(lambda n: _zoo(n, old, sampling, mask, adv, mode = "token_mask", clip = (None, 3.0),
                                 rollout_rs = parsed, bypass = bypass), new)
    _, g2 = _grad(lambda n: _zoo(n, old2, sampling2, mask, adv, mode = "token_mask", clip = (None, 3.0),
                                 rollout_rs = parsed, bypass = bypass), new2)
    kept = keep & (mask > 0)
    torch.testing.assert_close(g1[kept], g2[kept], rtol = 1e-12, atol = 1e-14)
    assert (g1[rejected] == 0).all() and (g2[rejected] == 0).all()
