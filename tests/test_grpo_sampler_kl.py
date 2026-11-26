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

"""grpo_compute_loss against float64 oracles for two opt-in pieces:
- the StableReinforce sampler KL (Tinker stable_reinforce: coef * sum_t k(q, p_theta), k1 = log q - log p,
  k3 = r - 1 - log r with r = p / q), added next to the reference KL and not IS-weighted;
- per-token (B, T) advantages, as on-policy distillation adds A_t = -coef * (log q - log p_teacher)."""

import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo")
DELTAS = (0.0, 1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0)


def _kwargs(mask, loss_type, **extra):
    kw = dict(
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28,
        max_completion_length = mask.shape[1], num_items_in_batch = mask.sum(),
        current_gradient_accumulation_steps = 1, steps_per_generation = 1, num_processes = 1,
        importance_sampling_level = "token",
    )
    kw.update(extra)
    return kw


def _agg(per_token, mask, loss_type):
    mask = mask.to(per_token.dtype)
    if loss_type == "grpo":
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min = 1)).mean()
    if loss_type == "dr_grpo":
        return (per_token * mask).sum() / (mask.shape[0] * mask.shape[1])
    return (per_token * mask).sum() / mask.sum().clamp(min = 1)


def _oracle(new, sampler, adv, mask, loss_type, coef, estimator, is_ratio = None):
    """On-policy (old = None): the policy term is -A_t * exp(new - new.detach()) [* IS], clipping inactive."""
    adv = adv.unsqueeze(1) if adv.dim() == 1 else adv
    ratio = torch.exp(new - new.detach())
    policy = -adv * ratio
    if is_ratio is not None:
        policy = policy * is_ratio
    log_r = new - sampler
    kl = -log_r if estimator == "k1" else torch.exp(log_r) - 1 - log_r
    return _agg(policy + coef * kl, mask, loss_type)


def _data(d, rows = 4, cols = 6, seed = 0, dtype = torch.float64):
    gen = torch.Generator().manual_seed(seed)
    sampler = -torch.rand(rows, cols, generator = gen, dtype = torch.float64) * 4
    signs = torch.where(torch.rand(rows, cols, generator = gen) > 0.5, 1.0, -1.0).double()
    new = (sampler + d * signs).to(dtype).requires_grad_(True)
    adv = torch.randn(rows, generator = gen, dtype = torch.float64).to(dtype)
    mask = torch.ones(rows, cols, dtype = torch.int64)
    mask[0, 4:] = 0  # right padding
    mask[1, :2] = 0  # left padding
    mask[2, 1:] = 0  # one token
    return new, sampler.to(dtype), adv, mask


def _zoo(new, sampler, adv, mask, loss_type, coef, estimator, beta = 0.0, ref = None, **extra):
    kw = _kwargs(mask, loss_type, sampler_kl_coef = coef, sampler_kl_estimator = estimator, use_vllm = True, **extra)
    return rr.grpo_compute_loss(ref, new, None, sampler, torch.zeros_like(mask), mask, beta, adv, **kw)[0]


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("estimator", ("k1", "k3"))
@pytest.mark.parametrize("d", DELTAS)
def test_sampler_kl_matches_fp64_oracle(loss_type, estimator, d):
    coef = 0.01
    new, sampler, adv, mask = _data(d)
    loss = _zoo(new, sampler, adv, mask, loss_type, coef, estimator)
    (grad,) = torch.autograd.grad(loss, new)
    new64 = new.detach().clone().requires_grad_(True)
    want = _oracle(new64, sampler, adv, mask, loss_type, coef, estimator)
    (want_grad,) = torch.autograd.grad(want, new64)
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-12)
    torch.testing.assert_close(grad, want_grad, rtol = 1e-12, atol = 1e-12)
    # Masked tokens get no gradient.
    assert torch.all(grad[mask == 0] == 0)


@pytest.mark.parametrize("estimator", ("k1", "k3"))
def test_sampler_kl_float32_close_to_fp64(estimator):
    new, sampler, adv, mask = _data(1.0, dtype = torch.float32)
    loss = _zoo(new, sampler, adv, mask, "dapo", 0.5, estimator)
    (grad,) = torch.autograd.grad(loss, new)
    new64 = new.detach().double().requires_grad_(True)
    want = _oracle(new64, sampler.double(), adv.double(), mask, "dapo", 0.5, estimator)
    (want_grad,) = torch.autograd.grad(want, new64)
    torch.testing.assert_close(loss.double(), want, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(grad.double(), want_grad, rtol = 1e-5, atol = 1e-7)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_coef_zero_is_bitwise_unchanged(loss_type):
    new, sampler, adv, mask = _data(1.0, dtype = torch.float32)
    base_kw = _kwargs(mask, loss_type, use_vllm = True)
    base = rr.grpo_compute_loss(None, new, None, sampler, torch.zeros_like(mask), mask, 0.0, adv, **base_kw)[0]
    off = _zoo(new, sampler, adv, mask, loss_type, 0.0, "k3")
    assert torch.equal(base, off)


@pytest.mark.parametrize("estimator", ("k1", "k3"))
def test_saturated_log_ratio_stays_finite(estimator):
    # |log p - log q| past 20 is clamped (expm1 would overflow float32 near 88): finite loss, no gradient there.
    for d in (88.0, -88.0):
        new, sampler, adv, mask = _data(d, dtype = torch.float32)
        loss = _zoo(new, sampler, torch.zeros_like(adv), mask, "dapo", 1.0, estimator)
        (grad,) = torch.autograd.grad(loss, new)
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
        assert torch.all(grad == 0)


def test_sampler_falls_back_to_old_then_policy():
    new, sampler, adv, mask = _data(1.0)
    zero_adv = torch.zeros_like(adv)
    kw = _kwargs(mask, "dapo", sampler_kl_coef = 1.0, sampler_kl_estimator = "k3")
    # No vLLM logprobs: pi_old is the sampler.
    with_old = rr.grpo_compute_loss(None, new, sampler, None, torch.zeros_like(mask), mask, 0.0, zero_adv, **kw)[0]
    want = _oracle(new.detach().clone().requires_grad_(True), sampler, zero_adv, mask, "dapo", 1.0, "k3")
    torch.testing.assert_close(with_old, want, rtol = 1e-12, atol = 1e-12)
    # Neither: the policy sampled its own tokens, so k3 is 0 with no gradient and k1 is 0 with gradient -coef.
    k3 = rr.grpo_compute_loss(None, new, None, None, torch.zeros_like(mask), mask, 0.0, zero_adv, **kw)[0]
    (g3,) = torch.autograd.grad(k3, new)
    assert k3.item() == 0.0 and torch.all(g3 == 0)
    kw["sampler_kl_estimator"] = "k1"
    k1 = rr.grpo_compute_loss(None, new, None, None, torch.zeros_like(mask), mask, 0.0, zero_adv, **kw)[0]
    (g1,) = torch.autograd.grad(k1, new)
    assert k1.item() == 0.0
    torch.testing.assert_close(g1, -mask.double() / mask.sum())


def test_unscored_sampler_tokens_add_nothing():
    new, sampler, adv, mask = _data(1.0)
    holes = sampler.clone()
    holes[3, 2] = float("nan")
    loss = _zoo(new, holes, torch.zeros_like(adv), mask, "dapo", 1.0, "k3")
    (grad,) = torch.autograd.grad(loss, new)
    filled = sampler.clone()
    filled[3, 2] = new.detach()[3, 2]
    want = _oracle(new.detach().clone().requires_grad_(True), filled, torch.zeros_like(adv), mask, "dapo", 1.0, "k3")
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-12)
    assert grad[3, 2] == 0 and torch.isfinite(grad).all()


def test_fully_masked_batch_is_zero():
    new, sampler, adv, mask = _data(1.0)
    mask = torch.zeros_like(mask)
    for estimator in ("k1", "k3"):
        loss = _zoo(new, sampler, adv, mask, "dapo", 1.0, estimator)
        (grad,) = torch.autograd.grad(loss, new)
        assert loss.item() == 0.0 and torch.all(grad == 0)


def test_sampler_kl_with_reference_kl_adds():
    new, sampler, adv, mask = _data(0.5)
    ref = sampler - 0.3
    both = _zoo(new, sampler, adv, mask, "dapo", 0.1, "k1", beta = 0.04, ref = ref)
    ref_only = _zoo(new, sampler, adv, mask, "dapo", 0.0, "k1", beta = 0.04, ref = ref)
    kl_only = _zoo(new, sampler, torch.zeros_like(adv), mask, "dapo", 0.1, "k1")
    # Every term shares the dapo normaliser, so the sampler KL adds on top of the reference KL unchanged.
    torch.testing.assert_close(both - ref_only, kl_only, rtol = 1e-10, atol = 1e-12)


def test_sampler_kl_not_is_weighted():
    # The vLLM IS ratio scales the policy term only: the sampler KL is an expectation under the sampler already.
    _, sampler, adv, mask = _data(0.0)
    old = sampler + 0.05
    new = (old + 0.01).requires_grad_(True)  # inside the PPO clip range
    kw = _kwargs(
        mask, "dapo", sampler_kl_coef = 0.2, sampler_kl_estimator = "k3", use_vllm = True,
        vllm_importance_sampling_correction = True, vllm_importance_sampling_mode = "token_truncate",
        vllm_importance_sampling_clip_min = None, vllm_importance_sampling_clip_max = 3.0,
    )
    loss = rr.grpo_compute_loss(None, new, old, sampler, torch.zeros_like(mask), mask, 0.0, adv, **kw)[0]
    is_ratio = torch.exp(old - sampler).clamp(max = 3.0) * mask
    new64 = new.detach().clone().requires_grad_(True)
    policy = -adv.unsqueeze(1) * torch.exp(new64 - old) * is_ratio
    log_r = new64 - sampler
    want = _agg(policy + 0.2 * (torch.exp(log_r) - 1 - log_r), mask, "dapo")
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-12)


@pytest.mark.parametrize("estimator", ("k1", "k3"))
def test_sampler_kl_gradcheck(estimator):
    _, sampler, adv, mask = _data(0.3)
    gen = torch.Generator().manual_seed(1)
    new = (sampler + 0.3 * torch.randn(sampler.shape, generator = gen, dtype = torch.float64)).requires_grad_(True)
    # With pi_old given (no detach trick) the whole loss is a smooth function of new inside the clip range.
    old = new.detach() + 0.01
    kw = _kwargs(mask, "dapo", sampler_kl_coef = 0.3, sampler_kl_estimator = estimator, use_vllm = True)
    assert torch.autograd.gradcheck(
        lambda x: rr.grpo_compute_loss(None, x, old, sampler, torch.zeros_like(mask), mask, 0.0, adv, **kw)[0],
        (new,), eps = 1e-6, atol = 1e-8,
    )


# Per-token advantages (on-policy distillation).

def _opd_advantages(sampler, teacher, mask, coef = 1.0):
    return -coef * (sampler - teacher) * mask


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_token_advantages_match_fp64_oracle(loss_type):
    new, sampler, adv, mask = _data(0.0)
    gen = torch.Generator().manual_seed(2)
    teacher = sampler + 0.5 * torch.randn(sampler.shape, generator = gen, dtype = torch.float64)
    token_adv = adv.unsqueeze(1) + _opd_advantages(sampler, teacher, mask)
    loss = rr.grpo_compute_loss(
        None, new, None, None, torch.zeros_like(mask), mask, 0.0, token_adv, **_kwargs(mask, loss_type),
    )[0]
    (grad,) = torch.autograd.grad(loss, new)
    new64 = new.detach().clone().requires_grad_(True)
    want = _oracle(new64, sampler, token_adv, mask, loss_type, 0.0, "k1")
    (want_grad,) = torch.autograd.grad(want, new64)
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-12)
    torch.testing.assert_close(grad, want_grad, rtol = 1e-12, atol = 1e-12)
    # On policy, the gradient is the per-token reverse-KL gradient: -A_t * grad log p / normaliser.
    norm = {"grpo": None, "bnpo": mask.sum(), "dapo": mask.sum(), "dr_grpo": mask.numel()}[loss_type]
    if norm is not None:
        torch.testing.assert_close(grad, -token_adv * mask / norm, rtol = 1e-12, atol = 1e-12)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_broadcast_token_advantages_equal_row_advantages(loss_type):
    new, sampler, adv, mask = _data(0.2, dtype = torch.float32)
    old = sampler + 0.01
    kw = _kwargs(mask, loss_type)
    row = rr.grpo_compute_loss(None, new, old, None, torch.zeros_like(mask), mask, 0.0, adv, **kw)[0]
    token = rr.grpo_compute_loss(
        None, new, old, None, torch.zeros_like(mask), mask, 0.0, adv.unsqueeze(1).expand_as(new).contiguous(), **kw,
    )[0]
    assert torch.equal(row, token)


def test_token_advantages_with_vllm_is_ratio():
    new, sampler, _, mask = _data(0.0)
    gen = torch.Generator().manual_seed(3)
    teacher = sampler + 0.5 * torch.randn(sampler.shape, generator = gen, dtype = torch.float64)
    old = sampler + 0.05 * torch.randn(sampler.shape, generator = gen, dtype = torch.float64)
    token_adv = _opd_advantages(sampler, teacher, mask)
    kw = _kwargs(
        mask, "dapo", use_vllm = True, vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = "token_truncate", vllm_importance_sampling_clip_min = None,
        vllm_importance_sampling_clip_max = 3.0,
    )
    new = old.clone().requires_grad_(True)
    loss = rr.grpo_compute_loss(None, new, old, sampler, torch.zeros_like(mask), mask, 0.0, token_adv, **kw)[0]
    (grad,) = torch.autograd.grad(loss, new)
    is_ratio = torch.exp(old - sampler).clamp(max = 3.0) * mask
    torch.testing.assert_close(grad, -token_adv * is_ratio / mask.sum(), rtol = 1e-12, atol = 1e-12)


def test_zero_advantage_keep_keeps_rows_with_token_signal():
    adv = torch.zeros(4, 5)
    adv[1, 3] = 0.25
    # A row whose environment advantage is 0 but whose distillation advantage is not must still be scored.
    assert rr.grpo_zero_advantage_keep(adv, 4).tolist() == [1]


def test_fused_route_off_with_sampler_kl():
    base = dict(loss_type = "dapo", importance_sampling_level = "token")
    on = rr.grpo_fused_route_ok(dict(base, sampler_kl_coef = 0.01))
    assert on is False
    assert rr.grpo_fused_route_ok(dict(base, sampler_kl_coef = 0.0)) == rr.grpo_fused_route_ok(base)


@pytest.mark.parametrize("bad", (dict(unsloth_sampler_kl_estimator = "k2"), dict(unsloth_sampler_kl_coef = -0.1)))
def test_accumulated_loss_rejects_bad_config(bad):
    trainer = types.SimpleNamespace(args = types.SimpleNamespace(**bad), use_vllm = False)
    with pytest.raises(ValueError, match = "sampler_kl"):
        rr.grpo_accumulated_loss(
            trainer, torch.zeros(1, 2, dtype = torch.long), torch.ones(1, 2), 1, torch.ones(1, 1),
            torch.zeros(1), None, None,
        )


def test_token_advantages_follow_the_left_packed_layout():
    # grpo_accumulated_loss left-packs every row and scores the last C + max_left_pad columns; (B, C) per-token
    # advantages must land on the same tokens there, exactly as vLLM's sampling logprobs do.
    pad = 0
    prompts = torch.tensor([[pad, pad, 11, 12], [pad, 21, 22, 23], [31, 32, 33, 34]])
    completions = torch.tensor([[101, 102, 103, pad], [201, 202, pad, pad], [301, 302, 303, 304]])
    input_ids = torch.cat([prompts, completions], dim = 1)
    C = completions.shape[1]
    left_pad = rr.calculate_pad_tokens_in_prompt(input_ids, C, pad)
    max_left_pad = int(left_pad.max())
    packed = rr.left_pack_padding(input_ids, pad)[:, -(C + max_left_pad):]
    mask = rr.create_completion_attention_mask(packed, left_pad, max_left_pad, pad).to(torch.long)
    tags = completions.to(torch.float32)  # one advantage per completion token, equal to its token id
    aligned = rr.align_logprobs_with_mask(tags, mask)
    keep = mask.bool()
    assert aligned.shape == mask.shape
    assert torch.equal(aligned[keep], packed[keep].to(torch.float32))
    assert torch.all(aligned[~keep] == 0)
