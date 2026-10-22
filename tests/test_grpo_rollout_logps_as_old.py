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

"""Rollout logprobs as pi_old (unsloth UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1) against float64 oracles.

verl rollout_correction bypass_mode with loss_type="ppo_clip" (verl/trainer/ppo/rollout_corr_helper.py
apply_bypass_mode, core_algos.py compute_policy_loss_bypass_mode): old_log_prob = rollout_log_prob, then
compute_policy_loss_vanilla with no IS weight (would be double-counting):
    r = exp(log_prob - rollout_log_prob)
    l = max(-A r, -A clip(r, 1 - eps_low, 1 + eps_high)), and for A < 0 the dual clip min(l, -A c), c = 3.0
vime (--use-rollout-logprobs, ppo_utils.compute_policy_loss with eps_clip_c) is the same per-token loss.
TRL's delta (clamp(r, max = delta) on the unclipped term) equals that dual clip for delta > 1 + eps_high,
which unsloth sets to verl's c = 3.0 when delta is unset. Bypass mismatch metrics are verl's
compute_offpolicy_metrics with the current log_prob in place of old_log_prob: k1 = rollout - current,
k3 = r - 1 - log r, ESS = (sum r)^2 / (N sum r^2).
"""

import collections
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

EPS_LOW, EPS_HIGH = 0.2, 0.28


def _verl_bypass_ppo_clip(log_prob, rollout_log_prob, advantages, mask, clip_ratio_c):
    """verl compute_policy_loss_vanilla per token (core_algos.py), old_log_prob = rollout_log_prob."""
    a = advantages.unsqueeze(1)
    negative_approx_kl = torch.clamp(log_prob - rollout_log_prob, min = -20.0, max = 20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -a * ratio
    pg_losses2 = -a * torch.clamp(ratio, 1 - EPS_LOW, 1 + EPS_HIGH)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    if clip_ratio_c is None:
        return clip_pg_losses1
    pg_losses3 = -a * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    return torch.where(a < 0, clip_pg_losses2, clip_pg_losses1)


def _agg(per_token, mask, loss_type):
    mask = mask.to(per_token.dtype)
    if loss_type == "grpo":  # verl seq-mean-token-mean
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min = 1)).mean()
    return (per_token * mask).sum() / mask.sum().clamp(min = 1)  # verl token-mean


def _zoo(new, rollout, advantages, mask, loss_type, delta, sampling = "dropped"):
    old, sampling_logps = rr.grpo_rollout_logps_as_old(rollout)
    kwargs = dict(
        loss_type = loss_type, epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, delta = delta,
        max_completion_length = new.shape[1], num_items_in_batch = mask.sum(),
        current_gradient_accumulation_steps = 1, steps_per_generation = 1, num_processes = 1,
        importance_sampling_level = "token", use_vllm = True, vllm_importance_sampling_correction = True,
        vllm_importance_sampling_mode = "token_truncate", vllm_importance_sampling_clip_max = 3.0,
        rollout_logps_as_old = True,
    )
    return rr.grpo_compute_loss(
        None, new, old.to(new.dtype), sampling_logps, torch.zeros_like(mask), mask, 0.0, advantages, **kwargs,
    )


def _data(d, rows = 5, cols = 7, seed = 0, dtype = torch.float64):
    gen = torch.Generator().manual_seed(seed)
    rollout = -torch.rand(rows, cols, generator = gen, dtype = torch.float64) * 4
    signs = torch.where(torch.rand(rows, cols, generator = gen) > 0.5, 1.0, -1.0).double()
    new = (rollout + d * signs).to(dtype).requires_grad_(True)
    mask = (torch.rand(rows, cols, generator = gen) > 0.2).long()
    mask[0] = 0  # fully masked row
    mask[1, 1:] = 0  # one token
    mask[2, :3] = 0  # left padding
    mask[3, -3:] = 0  # right padding
    advantages = torch.randn(rows, generator = gen, dtype = torch.float64).to(dtype)
    return new, rollout.to(torch.float32), advantages, mask


DS = [0.0, 1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0]


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "bnpo"])
@pytest.mark.parametrize("delta", [None, 3.0])
def test_loss_matches_verl_bypass_ppo_clip(d, loss_type, delta):
    new, rollout, advantages, mask = _data(d)
    got = _zoo(new, rollout, advantages, mask, loss_type, delta)[0]
    # Both clamp the log ratio to [-20, 20], so they agree at every gap, 88 nats included.
    want = _agg(_verl_bypass_ppo_clip(new.detach(), rollout.double(), advantages, mask, delta), mask, loss_type)
    torch.testing.assert_close(got.detach(), want, rtol = 1e-9, atol = 1e-12)
    assert torch.isfinite(got)


@pytest.mark.parametrize("d", [1e-4, 1.0, 20.0])
def test_float32_error_stays_at_float32_level(d):
    new64, rollout, advantages64, mask = _data(d)
    want = _agg(_verl_bypass_ppo_clip(new64.detach(), rollout.double(), advantages64, mask, 3.0), mask, "dapo")
    new32 = new64.detach().float().requires_grad_(True)
    got = _zoo(new32, rollout, advantages64.float(), mask, "dapo", 3.0)[0]
    assert (got.double() - want).abs() <= 1e-5 * max(1.0, want.abs().item())


def test_rollout_as_old_drops_the_is_weight():
    new, rollout, advantages, mask = _data(0.3)
    dropped = _zoo(new, rollout, advantages, mask, "dapo", 3.0)
    assert dropped[3] is not None and dropped[3].numel() == 0  # no vLLM IS block ran
    old, sampling = rr.grpo_rollout_logps_as_old(rollout)
    assert sampling is None and old.dtype == torch.float32
    torch.testing.assert_close(old, rollout)


def test_gradcheck_float64():
    new, rollout, advantages, mask = _data(0.5, rows = 5, cols = 4)
    rollout = rollout.double()

    def f(x):
        old = rollout
        kwargs = dict(
            loss_type = "dapo", epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, delta = 3.0,
            max_completion_length = 4, num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = 1,
            steps_per_generation = 1, num_processes = 1, importance_sampling_level = "token",
            rollout_logps_as_old = True,
        )
        return rr.grpo_compute_loss(None, x, old, None, torch.zeros_like(mask), mask, 0.0, advantages, **kwargs)[0]

    assert torch.autograd.gradcheck(f, (new,), eps = 1e-6, atol = 1e-8)
    x = (rollout + 2.0).requires_grad_(True)
    f(x).backward()
    assert torch.all(x.grad[(advantages < 0).unsqueeze(1).expand_as(x) & mask.bool()] == 0)


@pytest.mark.parametrize("d", [30.0, 89.0, 100.0, 300.0])
@pytest.mark.parametrize("delta", [None, 3.0])
def test_large_rollout_gap_keeps_float32_grads_finite(d, delta):
    # exp(new - rollout) overflows float32 past ~88.7 nats; the clipped branch's backward is then 0 * inf.
    new64, rollout, advantages, mask = _data(d)
    new = new64.detach().float().requires_grad_(True)
    loss = _zoo(new, rollout, advantages.float(), mask, "dapo", delta)[0]
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all()


def test_clamped_path_compiles_fullgraph():
    torch._dynamo.reset()
    new64, rollout, advantages, mask = _data(100.0)
    new = new64.detach().float()
    old, sampling = rr.grpo_rollout_logps_as_old(rollout)
    kwargs = dict(
        loss_type = "dapo", epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, delta = 3.0, num_items_in_batch = mask.sum(),
        steps_per_generation = 1, num_processes = 1, rollout_logps_as_old = True,
    )
    args = (None, new, old, sampling, torch.zeros_like(mask), mask, 0.0, advantages.float())
    eager = rr.grpo_compute_loss(*args, **kwargs)
    compiled = torch.compile(rr.grpo_compute_loss, fullgraph = True, dynamic = True)(*args, **kwargs)
    torch.testing.assert_close(compiled[0], eager[0], rtol = 1e-5, atol = 1e-6)
    assert torch.isfinite(compiled[0])


def test_default_path_is_not_clamped():
    new, rollout, advantages, mask = _data(30.0)
    old = rollout.double()
    kwargs = dict(
        loss_type = "dapo", epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH, delta = 3.0,
        max_completion_length = new.shape[1], num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = 1,
        steps_per_generation = 1, num_processes = 1, importance_sampling_level = "token",
    )
    got = rr.grpo_compute_loss(None, new, old, None, torch.zeros_like(mask), mask, 0.0, advantages, **kwargs)
    coef = (new.detach() - old).exp()  # unclamped
    a = advantages.unsqueeze(1)
    per_token = -torch.min(coef.clamp(max = 3.0) * a, coef.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * a)
    want = (per_token * mask).sum() / mask.sum()
    torch.testing.assert_close(got[0].detach(), want, rtol = 1e-12, atol = 0)
    assert float(got[5].detach().max()) > float(torch.tensor(20.0, dtype = torch.float64).exp())


def test_helper_refuses_bad_inputs():
    with pytest.raises(ValueError):
        rr.grpo_rollout_logps_as_old(None)
    with pytest.raises(ValueError):
        rr.grpo_rollout_logps_as_old(torch.zeros(2, 3), torch.zeros(2, 3))


@pytest.mark.parametrize("gap", [0.0, 1e-6, 0.3, 5.0, 100.0])
def test_bypass_metrics_match_verl_offpolicy_metrics(gap):
    gen = torch.Generator().manual_seed(1)
    current = -torch.rand(4, 6, generator = gen, dtype = torch.float64) * 3
    rollout = current + gap * torch.randn(4, 6, generator = gen, dtype = torch.float64)
    mask = (torch.rand(4, 6, generator = gen) > 0.3).long()
    mask[0] = 0
    coef_1 = torch.exp((current - rollout).float())
    delta, ratio = rr.grpo_rollout_as_old_ratio_metrics(coef_1, mask)
    keep = mask.bool()
    log_r = (current - rollout).float()
    finite = torch.isfinite(coef_1) & (coef_1 > 0)
    sel = keep & finite
    torch.testing.assert_close(delta[sel], log_r.abs()[sel], rtol = 1e-5, atol = 1e-6)
    assert torch.all(delta[~keep] == 0) and torch.all(ratio[~keep] == 0)
    if gap <= 5.0:
        log_ratio = torch.log(coef_1.double())
        stats = rr.grpo_rollout_mismatch_stats(log_ratio, torch.zeros_like(log_ratio), mask, ratio)
        metrics = rr.grpo_rollout_mismatch_metrics(stats)
        lr = (current - rollout)[keep]
        r = lr.exp()
        assert metrics["sampling/rollout_kl_k1/mean"] == pytest.approx((-lr).mean().item(), rel = 1e-5, abs = 1e-6)
        assert metrics["sampling/rollout_kl_k3/mean"] == pytest.approx((r - 1 - lr).mean().item(), rel = 1e-4, abs = 1e-7)
        ess = (r.sum() ** 2 / (r.numel() * (r * r).sum())).item()
        assert metrics["sampling/importance_sampling_ess"] == pytest.approx(ess, rel = 1e-5)


def test_sequence_level_ratio_reports_nothing():
    coef_1 = torch.ones(3, 1)
    assert rr.grpo_rollout_as_old_ratio_metrics(coef_1, torch.ones(3, 4)) == (None, None)


# Rollout logprobs arrive right padded (TRL) and must land in the left-packed completion columns.
_IDS = torch.tensor([[0, 0, 1, 2, 3, 4, 5], [1, 2, 6, 7, 3, 4, 6], [0, 1, 2, 3, 5, 0, 0]])
_MASK = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 0]])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _drive(monkeypatch, as_old, old_logps = None, use_vllm = True, correction = True, coef_1 = None, metrics_on = True,
           unscored = None):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_METRICS", "1" if metrics_on else "0")
    seen = {}

    class _Loss:
        @staticmethod
        def apply(new, old, ref, sampling, lm_head, ids, mask, adv, beta, scaler, n_chunks, kwargs, upstream):
            seen.update(new = new.detach().clone(), old = old, sampling = sampling, mask = mask.clone(), ids = ids.clone(),
                        kwargs = dict(kwargs))
            zero = torch.zeros(())
            c = torch.ones(mask.shape) if coef_1 is None else coef_1(mask)
            if sampling is not None and correction:
                return zero, zero, zero, torch.zeros(mask.shape), torch.ones(mask.shape), c
            return zero, zero, zero, torch.tensor([]), torch.tensor([]), c

    monkeypatch.setattr(rr, "UnslothEfficientGRPO", _Loss)
    model = _Model()
    model.train(True)
    metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)}
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = None,
            vllm_importance_sampling_mode = "token_truncate",
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = False),
        _metrics = metrics,
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = correction,
        _autocast_dtype = None,
        beta = 0.0,
        loss_type = "dapo",
    )
    # Distinct values per row and column, so a misaligned copy cannot pass.
    sampling = -(torch.arange(9, dtype = torch.float32).view(3, 3) + 1) / 10
    if unscored is not None:
        sampling[unscored] = float("nan")
    kwargs = dict(loss_type = "dapo", sampling_per_token_logps = sampling)
    if as_old:
        kwargs["rollout_logps_as_old"] = True
    out = rr.grpo_accumulated_loss(
        trainer, _IDS, (_IDS != 0).long(), 3, _MASK, torch.zeros(3), old_logps, None, **kwargs,
    )
    return seen, sampling, out, metrics


@pytest.mark.parametrize("use_vllm, correction", [(True, True), (True, False), (False, False)])
def test_accumulated_loss_scores_against_the_aligned_rollout(monkeypatch, use_vllm, correction):
    seen, sampling, _, _ = _drive(monkeypatch, True, use_vllm = use_vllm, correction = correction)
    assert seen["sampling"] is None
    assert seen["old"] is not None and seen["old"].shape == seen["mask"].shape
    want = rr.align_logprobs_with_mask(sampling, seen["mask"])
    torch.testing.assert_close(seen["old"], want)
    for row in range(3):
        n = int(_MASK[row].sum())
        torch.testing.assert_close(seen["old"][row][seen["mask"][row].bool()], sampling[row, :n])


def test_accumulated_loss_without_the_marker_is_unchanged(monkeypatch):
    seen, sampling, _, _ = _drive(monkeypatch, False)
    assert seen["old"] is None
    torch.testing.assert_close(seen["sampling"], rr.align_logprobs_with_mask(sampling, seen["mask"]))
    seen, _, _, _ = _drive(monkeypatch, False, use_vllm = False, correction = False)
    assert seen["old"] is None and seen["sampling"] is None


def test_accumulated_loss_reports_the_clipped_ratio(monkeypatch):
    # r (and the ESS built on it) is the loss ratio coef_1, like verl's SAFETY_BOUND clamped IS
    # weights; |log r| and the k1 / k3 metrics use the raw new - rollout, like verl
    # compute_offpolicy_metrics, so a gap past the [-20, 20] loss clamp is not under-reported.
    values = torch.tensor([1.5, 0.5, 1.0, 2.0, 1.0, 0.25, 1.1, 7.0, 9.0, 3.0, 0.1, 4.0, 2.5, 1.25, 0.75])

    def coef_of(mask):
        return values[: mask.numel()].view(mask.shape)

    seen, _, out, metrics = _drive(monkeypatch, True, coef_1 = coef_of)
    delta, ratio, mask = out[3], out[4], out[6]
    coef_1 = coef_of(mask)
    keep = mask.bool()
    raw = seen["new"].double() - seen["old"].double()
    torch.testing.assert_close(ratio[keep], coef_1[keep])
    torch.testing.assert_close(delta[keep], raw.abs().float()[keep])
    lr = raw[keep]
    r = coef_1.double()[keep]
    m = metrics["train"]
    assert m["sampling/rollout_kl_k3/mean"] == [pytest.approx((lr.exp() - 1 - lr).mean().item(), rel = 1e-6)]
    assert m["sampling/importance_sampling_ess"] == [
        pytest.approx((r.sum() ** 2 / (r.numel() * (r * r).sum())).item(), rel = 1e-6)
    ]


def test_accumulated_loss_metrics_stay_finite_on_an_overflowed_ratio(monkeypatch):
    def coef_of(mask):
        c = torch.full(mask.shape, 2.0)
        c[~mask.bool()] = float("inf")  # overflowed on padding: must not reach the sums
        c[1, -1] = float("inf") if mask[1, -1] else c[1, -1]
        return c

    _, _, out, metrics = _drive(monkeypatch, True, coef_1 = coef_of)
    for name in ("sampling/rollout_kl_k1/mean", "sampling/rollout_kl_k3/mean", "sampling/importance_sampling_ess"):
        values = metrics["train"][name]
        assert values and all(v == v and abs(v) != float("inf") for v in values), name
    assert torch.isfinite(out[3]).all()


def test_unscored_rollout_tokens_get_ratio_one(monkeypatch):
    probe = -(torch.arange(9, dtype = torch.float32).view(3, 3) + 1) / 10
    probe[1, 1] = float("nan")

    def coef_of(mask):
        return torch.where(torch.isnan(rr.align_logprobs_with_mask(probe, mask)), 5.0, 1.0)

    seen, sampling, out, metrics = _drive(monkeypatch, True, unscored = (1, 1), coef_1 = coef_of)
    old, new, mask = seen["old"], seen["new"], seen["mask"]
    assert torch.isfinite(old).all()
    hole = torch.isnan(rr.align_logprobs_with_mask(sampling, mask))
    assert hole.sum() == 1 and mask[hole].all()
    torch.testing.assert_close(old[hole], new[hole].to(old.dtype))
    keep = mask.bool() & ~hole
    torch.testing.assert_close(old[keep], rr.align_logprobs_with_mask(sampling, mask)[keep])
    assert seen["kwargs"]["rollout_logps_as_old"] is True
    # The hole is left out of the sums; the scored tokens report their raw new - rollout gap.
    lr = (new.double() - old.double())[keep]
    assert metrics["train"]["sampling/rollout_kl_k3/mean"] == [pytest.approx((lr.exp() - 1 - lr).mean().item(), rel = 1e-6)]
    assert metrics["train"]["sampling/rollout_kl_k1/mean"] == [pytest.approx((-lr).mean().item(), rel = 1e-6)]


def test_default_batch_does_not_ask_for_the_clamp(monkeypatch):
    seen, _, _, _ = _drive(monkeypatch, False)
    assert "rollout_logps_as_old" not in seen["kwargs"]


def test_marker_with_trainer_old_logps_is_refused(monkeypatch):
    with pytest.raises(ValueError):
        _drive(monkeypatch, True, old_logps = torch.zeros(3, 3))
