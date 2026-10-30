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

"""A rollout-rejected token must not reach the policy ratio (verl: response_mask *= keep)."""
import math

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

EPS_LOW, EPS_HIGH = 0.2, 0.28


def _kw(level, bias_kl, bypass, loss_type, mask):
    kw = dict(
        loss_type = loss_type, epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, max_completion_length = mask.shape[1],
        num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = 1, steps_per_generation = 1,
        num_processes = 1, importance_sampling_level = level, use_vllm = True,
        vllm_importance_sampling_correction = True, vllm_importance_sampling_mode = "token_mask",
        vllm_importance_sampling_clip_min = None, vllm_importance_sampling_clip_max = None,
        use_bias_correction_kl = bias_kl,
    )
    if bypass:
        kw["rollout_logps_as_old"] = True
    return kw


def _loss(ref, new, old, sampling, mask, adv, beta, parsed, level = "sequence", bias_kl = False,
          bypass = False, loss_type = "dapo"):
    kw = _kw(level, bias_kl, bypass, loss_type, mask)
    if parsed is not None:
        kw["rollout_rs"] = parsed
    if bypass:
        old, sampling = rr.grpo_rollout_logps_as_old(sampling)
    return rr.grpo_compute_loss(ref, new, old, sampling, torch.zeros_like(mask), mask, beta, adv, **kw)[0]


def _oracle(ref, new, old, sampling, mask, adv, beta, keep, level, bias_kl, bypass, loss_type):
    """float64 verl oracle; the loss normaliser still counts every completion token."""
    m = mask.double()
    k = keep.double() * m
    new = new.double()
    if bypass:
        old = sampling.double()
        is_ratio = torch.ones_like(new)
    else:
        old = old.double()
        lr_is = torch.where(m > 0, old - sampling.double(), torch.zeros_like(new))
        is_ratio = torch.where(k > 0, lr_is, torch.zeros_like(new)).exp()
    lr = torch.where(k > 0, new - old, torch.zeros_like(new))
    if bypass:
        lr = lr.clamp(-20, 20)
    if level == "sequence":
        lr = ((lr * k).sum(-1) / k.sum(-1).clamp(min = 1)).unsqueeze(-1)
    coef = lr.exp()
    a = adv.double().unsqueeze(1)
    per_token = -torch.min(coef * a, coef.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * a) * is_ratio
    if beta != 0.0:
        d = torch.where(m > 0, ref.double() - new, torch.zeros_like(new)).clamp(-20, 20)
        kl = d.exp() - d - 1
        if bias_kl:
            kl = kl * coef
        per_token = per_token + beta * kl.clamp(-10, 10)
    per_token = torch.where(k > 0, per_token, torch.zeros_like(per_token))
    if loss_type == "dapo":
        return per_token.sum() / m.sum().clamp(min = 1)
    return (per_token.sum(-1) / m.sum(-1).clamp(min = 1)).mean()


def _keep(new, old, sampling, mask, parsed, bypass):
    gap = (new.detach() - sampling) if bypass else (old - sampling)
    return rr.grpo_rollout_rs_keep(gap, mask, parsed)


def test_sequence_ratio_leaves_out_rejected_token():
    # Token 2 rejected; main leaked it into the shared GSPO ratio.
    new = torch.tensor([[-1.0, -1.0]], requires_grad = True)
    old = torch.tensor([[-1.0, -1.0]])
    sampling = torch.tensor([[-1.0, -2.0]])
    mask = torch.ones(1, 2, dtype = torch.long)
    loss = _loss(None, new, old, sampling, mask, torch.ones(1), 0.0, rr.grpo_parse_rollout_rs("token_k3", "0.1"))
    loss.backward()
    assert loss.item() == -0.5
    assert new.grad.tolist() == [[-0.5, 0.0]]


@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("gap", [100.0, 1e4])
def test_trainable_gap_on_rejected_token_has_finite_grad(level, gap):
    # exp(new - old) on the rejected token overflows float32; main gave grad nan.
    new = torch.tensor([[-1.0, -1.0]], requires_grad = True)
    old = torch.tensor([[-1.0, -1.0 - gap]])
    sampling = torch.tensor([[-1.0, -2.0 - gap]])
    mask = torch.ones(1, 2, dtype = torch.long)
    loss = _loss(None, new, old, sampling, mask, torch.ones(1), 0.0, rr.grpo_parse_rollout_rs("token_k3", "0.1"),
                 level = level)
    loss.backward()
    assert loss.item() == -0.5
    assert new.grad.tolist() == [[-0.5, 0.0]]


def _data(seed, rows = 6, cols = 9):
    g = torch.Generator().manual_seed(seed)
    new = -torch.rand(rows, cols, generator = g, dtype = torch.float64) * 3
    old = new + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.05
    ref = new + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.3
    sampling = old + torch.randn(rows, cols, generator = g, dtype = torch.float64) * 0.2
    mask = (torch.rand(rows, cols, generator = g) > 0.2).long()
    mask[1, 1:] = 0
    mask[2, 0] = 0
    adv = torch.randn(rows, generator = g, dtype = torch.float64)
    return ref, new.requires_grad_(True), old, sampling, mask, adv


RS = [("token_k3", "0.02"), ("token_k1", "0.8_1.25"), ("seq_mean_k3", "0.01"), ("token_k3,seq_mean_k3", "0.05,0.01")]


@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("beta,bias_kl", [(0.0, False), (0.04, False), (0.04, True)])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("loss_type", ["dapo", "grpo"])
def test_rejection_matches_verl_mask_fp64_oracle(options, thresholds, level, beta, bias_kl, bypass, loss_type):
    ref, new, old, sampling, mask, adv = _data(11)
    if bypass:
        sampling = sampling.float().double()  # grpo_rollout_logps_as_old hands the loss float32 rollout logps
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    keep = _keep(new, old, sampling, mask, parsed, bypass)
    rejected = mask.bool() & ~keep
    assert int(rejected.sum()) > 0, "the threshold should reject something"
    loss = _loss(ref, new, old, sampling, mask, adv, beta, parsed, level, bias_kl, bypass, loss_type)
    oracle_new = new.detach().clone().requires_grad_(True)
    want = _oracle(ref, oracle_new, old, sampling, mask, adv, beta, keep, level, bias_kl, bypass, loss_type)
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-15)
    loss.backward()
    want.backward()
    assert torch.equal(new.grad[rejected], torch.zeros_like(new.grad[rejected]))
    torch.testing.assert_close(new.grad, oracle_new.grad, rtol = 1e-10, atol = 1e-14)


@pytest.mark.parametrize("level", ["token", "sequence"])
def test_rejection_gradcheck_fp64(level):
    ref, new, old, sampling, mask, adv = _data(12)
    parsed = rr.grpo_parse_rollout_rs("token_k3", "0.02")
    fn = lambda n: _loss(ref, n, old, sampling, mask, adv, 0.04, parsed, level, True)
    assert torch.autograd.gradcheck(fn, (new,), eps = 1e-6, atol = 1e-8)


@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_without_rejection_is_bitwise_unchanged(level, dtype):
    ref, new, old, sampling, mask, adv = _data(13)
    ref, old, sampling, adv = (t.to(dtype) for t in (ref, old, sampling, adv))
    a_new = new.detach().to(dtype).requires_grad_(True)
    b_new = new.detach().to(dtype).requires_grad_(True)
    a = _loss(ref, a_new, old, sampling, mask, adv, 0.04, None, level, True)
    b = _loss(ref, b_new, old, sampling, mask, adv, 0.04, rr.grpo_parse_rollout_rs("seq_mean_k3", "1e9"), level, True)
    a.backward()
    b.backward()
    assert torch.equal(a, b) and torch.equal(a_new.grad, b_new.grad)
    assert math.isfinite(a.item())
