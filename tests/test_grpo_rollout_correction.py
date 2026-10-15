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

"""vLLM rollout correction in grpo_compute_loss against float64 oracles.

TRL modes (trl/trainer/grpo_trainer.py, _generate_and_score_completions, TRL 1.15): ratio =
exp((old - sampling) * mask), summed per row for sequence_*, then clamped (*_truncate) or zeroed
outside [C_min, C_max] (*_mask). geometric_*: exp of the masked mean log ratio (verl
rollout_corr seq_mean_k1 statistic). Rejection options follow verl
compute_rollout_rejection_mask with log_ratio = old - sampling.
"""
import math

import pytest
import torch

import unsloth_zoo.rl_replacements as rr


def _data(rows = 6, cols = 11, seed = 0, scale = 0.3):
    gen = torch.Generator().manual_seed(seed)
    new = (-torch.rand(rows, cols, generator = gen, dtype = torch.float64) * 3).requires_grad_(True)
    old = new.detach() + torch.randn(rows, cols, generator = gen, dtype = torch.float64) * 0.05
    sampling = old + torch.randn(rows, cols, generator = gen, dtype = torch.float64) * scale
    mask = (torch.rand(rows, cols, generator = gen) > 0.2).long()
    mask[0] = 0
    mask[1, 1:] = 0
    advantages = torch.randn(rows, generator = gen, dtype = torch.float64)
    return new, old, sampling, mask, advantages


def _loss(new, old, sampling, mask, advantages, mode, clip_min, clip_max, rollout_rs = None, loss_type = "dapo"):
    kwargs = dict(
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28, max_completion_length = new.shape[1],
        num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = 1, steps_per_generation = 1,
        num_processes = 1, importance_sampling_level = "token", use_vllm = True,
        vllm_importance_sampling_correction = True, vllm_importance_sampling_mode = mode,
        vllm_importance_sampling_clip_min = clip_min, vllm_importance_sampling_clip_max = clip_max,
    )
    if rollout_rs is not None:
        kwargs["rollout_rs"] = rollout_rs
    out = rr.grpo_compute_loss(None, new, old, sampling, torch.zeros_like(mask), mask, 0.0, advantages, **kwargs)
    return out[0], out[4]


def _oracle_ratio(old, sampling, mask, mode, clip_min, clip_max):
    lr = torch.nan_to_num((old - sampling) * mask, nan = 0.0)
    n = mask.sum(-1, keepdim = True).double()
    if mode.startswith("sequence"):
        lr = lr.sum(-1, keepdim = True)
    elif mode.startswith("geometric"):
        lr = lr.sum(-1, keepdim = True) / n.clamp(min = 1)
    r = lr.exp()
    if mode.endswith("truncate"):
        return r.clamp(min = clip_min, max = clip_max)
    lo = -math.inf if clip_min is None else clip_min
    hi = math.inf if clip_max is None else clip_max
    return torch.where((r < lo) | (r > hi), torch.zeros_like(r), r)


def _oracle_loss(new, old, mask, advantages, ratio):
    coef = (new - old).exp()
    a = advantages.unsqueeze(1)
    per_token = -torch.min(coef * a, coef.clamp(0.8, 1.28) * a) * ratio
    return (per_token * mask).sum() / mask.sum().clamp(min = 1)


MODES = [
    ("token_truncate", None, 3.0), ("token_mask", 0.5, 2.0), ("sequence_truncate", 0.1, 3.0),
    ("sequence_mask", 0.25, 4.0), ("geometric_truncate", 0.9, 1.1), ("geometric_mask", 0.95, 1.05),
    ("geometric_mask", None, 1.02), ("geometric_truncate", 0.98, None),
]


@pytest.mark.parametrize("mode,clip_min,clip_max", MODES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_is_modes_match_oracle(mode, clip_min, clip_max, seed):
    new, old, sampling, mask, adv = _data(seed = seed)
    loss, flat = _loss(new, old, sampling, mask, adv, mode, clip_min, clip_max)
    ratio = _oracle_ratio(old, sampling, mask, mode, clip_min, clip_max)
    torch.testing.assert_close(flat, (ratio * mask).expand_as(flat), rtol = 1e-12, atol = 0)
    want = _oracle_loss(new, old, mask, adv, ratio)
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-15)
    got_grad, = torch.autograd.grad(loss, new)
    want_grad, = torch.autograd.grad(_oracle_loss(new, old, mask, adv, ratio), new)
    torch.testing.assert_close(got_grad, want_grad, rtol = 1e-12, atol = 1e-15)


def test_geometric_engages_where_the_product_saturates():
    # 400 tokens with a mean log ratio of 0.01: the product is e^4 ~ 55 (masked at 3), the
    # geometric mean e^0.01 stays inside [0.9, 1.1].
    old = torch.zeros(1, 400, dtype = torch.float64)
    sampling = old - 0.01
    mask = torch.ones(1, 400, dtype = torch.long)
    seq = _oracle_ratio(old, sampling, mask, "sequence_mask", None, 3.0)
    geo = _oracle_ratio(old, sampling, mask, "geometric_mask", 0.9, 1.1)
    assert float(seq) == 0.0
    assert abs(float(geo) - math.exp(0.01)) < 1e-12
    new = old.clone().requires_grad_(True)
    _, flat = _loss(new, old, sampling, mask, torch.ones(1, dtype = torch.float64), "geometric_mask", 0.9, 1.1)
    torch.testing.assert_close(flat[0, 0], torch.tensor(math.exp(0.01), dtype = torch.float64))


def _oracle_keep(old, sampling, mask, options):
    lr = ((old - sampling) * mask).clamp(-20, 20)
    m = mask.double()
    keep = torch.ones_like(lr)
    for name, lower, upper in options:
        if name.endswith("k1"):
            stat = -lr
        elif name.endswith("k2"):
            stat = 0.5 * lr ** 2
        else:
            stat = lr.exp() - 1 - lr
        if name.startswith("seq_sum"):
            stat = (stat * m).sum(-1, keepdim = True)
        elif name.startswith("seq_mean"):
            stat = (stat * m).sum(-1, keepdim = True) / m.sum(-1, keepdim = True).clamp(min = 1)
        elif name.startswith("seq_max"):
            stat = torch.where(m > 0, stat, torch.full_like(stat, -math.inf)).amax(-1, keepdim = True)
            stat = torch.where(stat == -math.inf, torch.zeros_like(stat), stat)
        ok = (stat <= upper) if lower is None else (stat >= lower) & (stat <= upper)
        keep = keep * ok.double()
    return keep


RS = [
    ("token_k3", "0.02"), ("token_k2", "0.02"), ("token_k1", "0.8_1.25"), ("seq_sum_k3", "0.1"),
    ("seq_mean_k3", "0.01"), ("seq_mean_k1", "0.98_1.02"), ("seq_max_k3", "0.05"), ("seq_max_k2", "0.05"),
    ("seq_sum_k1", "0.5_2"), ("seq_mean_k2", "0.01"), ("seq_sum_k2", "0.1"),
    ("token_k3,seq_mean_k3", "0.05,0.01"), ("seq_mean_k1,seq_max_k3", "1.03,0.08"),
]


@pytest.mark.parametrize("options,thresholds", RS)
@pytest.mark.parametrize("mode", ["token_mask", "sequence_truncate"])
@pytest.mark.parametrize("seed", [0, 5])
def test_rejection_matches_oracle(options, thresholds, mode, seed):
    new, old, sampling, mask, adv = _data(seed = seed, scale = 0.15)
    parsed = rr.grpo_parse_rollout_rs(options, thresholds)
    clip = (None, None) if mode == "token_mask" else (0.0, 50.0)
    loss, flat = _loss(new, old, sampling, mask, adv, mode, *clip, rollout_rs = parsed)
    ratio = _oracle_ratio(old, sampling, mask, mode, *clip) * _oracle_keep(old, sampling, mask, parsed)
    torch.testing.assert_close(flat, ratio * mask, rtol = 1e-12, atol = 0)
    torch.testing.assert_close(loss, _oracle_loss(new, old, mask, adv, ratio), rtol = 1e-12, atol = 1e-15)
    keep = (flat != 0) | (mask == 0)
    assert 0 < int((~keep).sum()) or options in ("seq_sum_k1",), "the threshold should reject something"


def test_parse_matches_verl_threshold_rules():
    parsed = rr.grpo_parse_rollout_rs("seq_mean_k1, token_k3,seq_mean_k1", "1.25")
    assert parsed == (("seq_mean_k1", math.log(0.8), math.log(1.25)), ("token_k3", None, 1.25))
    assert rr.grpo_parse_rollout_rs("token_k1", "0.5_2")[0][1:] == (math.log(0.5), math.log(2.0))
    assert rr.grpo_parse_rollout_rs("", "") is None
    assert rr.grpo_parse_rollout_rs(None, None) is None
    for bad in [("token_k4", "1"), ("token_k3", ""), ("token_k3", "0.1_0.2"), ("token_k1", "2_1"),
                ("token_k3,seq_mean_k3", "1,2,3"), ("token_k3", "-1")]:
        with pytest.raises(ValueError):
            rr.grpo_parse_rollout_rs(*bad)


def test_no_rejection_is_identity():
    new, old, sampling, mask, adv = _data(seed = 3)
    base = _loss(new, old, sampling, mask, adv, "sequence_mask", None, 3.0)
    wide = _loss(new, old, sampling, mask, adv, "sequence_mask", None, 3.0,
                 rollout_rs = rr.grpo_parse_rollout_rs("seq_mean_k3", "1e9"))
    assert torch.equal(base[0], wide[0]) and torch.equal(base[1], wide[1])


def test_mismatch_metrics_match_oracle():
    new, old, sampling, mask, adv = _data(seed = 7)
    _, flat = _loss(new, old, sampling, mask, adv, "token_mask", 0.8, 1.25)
    stats = rr.grpo_rollout_mismatch_stats(old, sampling, mask, flat)
    got = rr.grpo_rollout_mismatch_metrics(stats)
    m = mask.bool()
    lr = (old - sampling)[m]
    w = flat[m]
    rows = m.any(-1)
    want = {
        "sampling/rollout_kl_k1/mean": float((-lr).mean()),
        "sampling/rollout_kl_k3/mean": float((lr.exp() - 1 - lr).mean()),
        "sampling/rollout_kl_k3/max": float((lr.exp() - 1 - lr).max()),
        "sampling/importance_sampling_ess": float(1.0 / ((w / w.mean()) ** 2).mean()),
        "sampling/importance_sampling_rejected_frac": float((w == 0).double().mean()),
        "sampling/importance_sampling_rejected_seq_frac": float(((flat == 0) & m).any(-1)[rows].double().mean()),
    }
    assert got.keys() == want.keys()
    for k in want:
        assert abs(got[k] - want[k]) <= 1e-12 * max(1.0, abs(want[k])), (k, got[k], want[k])
    assert 0 < want["sampling/importance_sampling_rejected_frac"] < 1


def test_mismatch_metrics_pool_across_shards():
    new, old, sampling, mask, adv = _data(rows = 8, seed = 9)
    _, flat = _loss(new, old, sampling, mask, adv, "sequence_truncate", 0.5, 2.0)
    whole = rr.grpo_rollout_mismatch_stats(old, sampling, mask, flat)
    parts = [rr.grpo_rollout_mismatch_stats(old[s], sampling[s], mask[s], flat[s]) for s in (slice(0, 3), slice(3, 8))]
    pooled = torch.stack(parts)
    pooled = torch.cat([pooled[:, :3].sum(0), pooled[:, 3:4].amax(0), pooled[:, 4:].sum(0)])
    torch.testing.assert_close(pooled, whole, rtol = 1e-12, atol = 1e-14)


def test_mismatch_metrics_empty_and_on_policy():
    mask = torch.zeros(2, 4, dtype = torch.long)
    x = torch.zeros(2, 4, dtype = torch.float64)
    got = rr.grpo_rollout_mismatch_metrics(rr.grpo_rollout_mismatch_stats(x, x, mask, x))
    assert all(v == 0.0 for v in got.values())
    mask[0, :3] = 1
    got = rr.grpo_rollout_mismatch_metrics(rr.grpo_rollout_mismatch_stats(x, x, mask, torch.ones(2, 1, dtype = torch.float64)))
    assert got["sampling/rollout_kl_k3/mean"] == 0.0 and got["sampling/importance_sampling_ess"] == 1.0
    assert got["sampling/importance_sampling_rejected_frac"] == 0.0


@pytest.mark.parametrize("options", [None, "token_k3,seq_mean_k1"])
@pytest.mark.parametrize("mode", ["geometric_mask", "sequence_mask"])
def test_compiles_fullgraph(options, mode):
    torch._dynamo.reset()
    new, old, sampling, mask, adv = _data(seed = 4)
    parsed = rr.grpo_parse_rollout_rs(options, "0.05,1.02") if options else None
    kwargs = dict(
        loss_type = "dapo", num_items_in_batch = mask.sum(), steps_per_generation = 1, num_processes = 1,
        use_vllm = True, vllm_importance_sampling_correction = True, vllm_importance_sampling_mode = mode,
        vllm_importance_sampling_clip_min = 0.9, vllm_importance_sampling_clip_max = 1.1,
    )
    if parsed is not None:
        kwargs["rollout_rs"] = parsed
    args = (None, new.float(), old.float(), sampling.float(), torch.zeros_like(mask), mask, 0.0, adv.float())
    eager = rr.grpo_compute_loss(*args, **kwargs)
    compiled = torch.compile(rr.grpo_compute_loss, fullgraph = True, dynamic = True)(*args, **kwargs)
    torch.testing.assert_close(compiled[0], eager[0], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(compiled[4], eager[4], rtol = 1e-5, atol = 1e-6)


def _micro_losses(new, old, mask, adv, weight, ga, spg, num_processes = 1, loss_type = "dapo"):
    kwargs = dict(
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28, max_completion_length = new.shape[1],
        num_items_in_batch = mask.sum(), current_gradient_accumulation_steps = ga, steps_per_generation = spg,
        num_processes = num_processes, importance_sampling_level = "token",
    )
    rows = new.shape[0] // spg
    total = 0
    for i in range(spg):
        s = slice(i * rows, (i + 1) * rows)
        if weight is not None:
            kwargs["sample_weight"] = weight[s]
        total = total + rr.grpo_compute_loss(None, new[s], old[s], None, torch.zeros_like(mask[s]), mask[s], 0.0, adv[s], **kwargs)[0]
    return total


@pytest.mark.parametrize("ga,spg", [(1, 1), (2, 2), (4, 4), (2, 4)])
def test_sample_weight_of_one_over_tokens_is_dapo(ga, spg):
    new, old, _, mask, adv = _data(rows = 8, seed = 2)
    weight = torch.full((8,), 1.0 / float(mask.sum()), dtype = torch.float64)
    got = _micro_losses(new, old, mask, adv, weight, ga, spg)
    want = _micro_losses(new, old, mask, adv, None, ga, spg)
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-15)
    g1, = torch.autograd.grad(got, new)
    g2, = torch.autograd.grad(want, new)
    torch.testing.assert_close(g1, g2, rtol = 1e-12, atol = 1e-15)


@pytest.mark.parametrize("loss_type", ["dapo", "grpo", "cispo", "dr_grpo"])
def test_sample_weight_prompt_level_matches_scalerl_formula(loss_type):
    # ScaleRL prompt-level aggregation over 3 groups of 4, group 1 excluded (zero variance):
    # (1 / kept groups) sum_g (1 / tokens in g) sum_{i in g, t} l_it, summed over the micro-batches.
    new, old, _, mask, adv = _data(rows = 12, cols = 9, seed = 6)
    keep = torch.tensor([1.0] * 4 + [0.0] * 4 + [1.0] * 4, dtype = torch.float64)
    tokens = mask.sum(-1).double().view(3, 4).sum(-1).repeat_interleave(4)
    weight = keep / (tokens.clamp(min = 1) * 2)
    got = _micro_losses(new, old, mask, adv, weight, ga = 2, spg = 2, loss_type = loss_type)
    if loss_type == "cispo":
        per_token = -(new - old).exp().clamp(max = 0.28).detach() * adv.unsqueeze(1) * new
    else:
        coef = (new - old).exp()
        a = adv.unsqueeze(1)
        per_token = -torch.min(coef * a, coef.clamp(0.8, 1.28) * a)
    per_row = (per_token * mask).sum(-1)
    want = sum(per_row.view(3, 4)[g].sum() / tokens[4 * g] for g in (0, 2)) / 2
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-15)


_DRIVER_IDS = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 3]])
_DRIVER_MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])


class _DriverModel(torch.nn.Module):
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
        return __import__("types").SimpleNamespace(logits = hidden)


def _drive(monkeypatch, correction, training = True, should_evaluate = False, loss_type = "dapo", is_mode = None):
    import collections
    import types

    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_METRICS", "1")
    if is_mode is not None:
        monkeypatch.setenv("UNSLOTH_GRPO_VLLM_IS_MODE", is_mode)

    class _Loss:
        @staticmethod
        def apply(*args):
            zero = torch.zeros(())
            # What the real forward returns: empty when the IS correction is off, else ratio * mask.
            ratio = _DRIVER_MASK.double() if correction else torch.tensor([])
            return zero, zero, zero, zero, ratio, zero

    monkeypatch.setattr(rr, "UnslothEfficientGRPO", _Loss)
    model = _DriverModel()
    model.train(training)
    metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)}
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = 0.5,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = should_evaluate),
        _metrics = metrics,
        use_vllm = True,
        vllm_importance_sampling_correction = correction,
        _autocast_dtype = None,
        beta = 0.0,
        loss_type = loss_type,
    )
    logps = torch.zeros(_DRIVER_MASK.shape, dtype = torch.float64)
    rr.grpo_accumulated_loss(
        trainer, _DRIVER_IDS, (_DRIVER_IDS != 0).long(), 3, _DRIVER_MASK,
        torch.zeros(_DRIVER_IDS.shape[0]), logps, None,
        loss_type = loss_type, sampling_per_token_logps = logps - 0.1,
    )
    return metrics


def test_rollout_metrics_skip_when_is_correction_is_off(monkeypatch):
    metrics = _drive(monkeypatch, correction = False)
    assert not metrics["train"] and not metrics["eval"]
    metrics = _drive(monkeypatch, correction = True)
    assert "sampling/rollout_kl_k1/mean" in metrics["train"]


def test_rollout_metrics_bucket_follows_model_mode(monkeypatch):
    metrics = _drive(monkeypatch, correction = True, training = False, should_evaluate = False)
    assert "sampling/rollout_kl_k1/mean" in metrics["eval"] and not metrics["train"]


@pytest.mark.parametrize("mode", ["geometric_mask", "geometric_truncate", "sequence_mask"])
def test_vespo_rejects_non_token_is_mode_override(monkeypatch, mode):
    with pytest.raises(ValueError, match = "VESPO"):
        _drive(monkeypatch, correction = True, loss_type = "vespo", is_mode = mode)
    _drive(monkeypatch, correction = True, loss_type = "vespo", is_mode = "token_mask")


def test_is_clip_env_parse_and_validation():
    assert rr.grpo_parse_is_clip("0.999_1.001", "geometric_mask") == (0.999, 1.001)
    assert rr.grpo_parse_is_clip("_2", "token_truncate") == (None, 2.0)
    assert rr.grpo_parse_is_clip("0_2", "token_truncate") == (0.0, 2.0)
    assert rr.grpo_parse_is_clip("0.5_", "sequence_mask") == (0.5, None)
    assert rr.grpo_parse_is_clip("_", "token_mask") == (None, None)
    for bad, mode in [("2_1", "token_truncate"), ("2_1", "sequence_mask"), ("-1_2", "token_mask"),
                      ("_0", "token_truncate"), ("_nan", "token_mask"), ("_inf", "token_truncate"),
                      ("_", "token_truncate"), ("_", "geometric_truncate")]:
        with pytest.raises(ValueError, match = "UNSLOTH_GRPO_VLLM_IS_CLIP"):
            rr.grpo_parse_is_clip(bad, mode)


@pytest.mark.parametrize("spec", ["2_1", "_"])
def test_reversed_or_empty_is_clip_override_is_rejected(monkeypatch, spec):
    monkeypatch.setenv("UNSLOTH_GRPO_VLLM_IS_CLIP", spec)
    with pytest.raises(ValueError, match = "UNSLOTH_GRPO_VLLM_IS_CLIP"):
        _drive(monkeypatch, correction = True, is_mode = "token_truncate")
    monkeypatch.setenv("UNSLOTH_GRPO_VLLM_IS_CLIP", "0.5_2")
    _drive(monkeypatch, correction = True, is_mode = "token_truncate")
