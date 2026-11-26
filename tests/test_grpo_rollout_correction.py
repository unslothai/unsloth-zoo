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

"""Rollout correction in grpo_compute_loss vs float64 oracles per TRL 1.15 (trl/trainer/grpo_trainer.py): sequence_*
sums per row, *_truncate clamps, *_mask zeroes outside [C_min, C_max]; geometric_* is exp of the masked mean;
rejection follows verl."""
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
    # 400 tokens at mean log ratio 0.01: the product is e^4 ~ 55 (masked at 3) while the geometric mean e^0.01 stays inside [0.9, 1.1].
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
    assert want.keys() <= got.keys()
    for k in want:
        assert abs(got[k] - want[k]) <= 1e-12 * max(1.0, abs(want[k])), (k, got[k], want[k])
    assert 0 < want["sampling/importance_sampling_rejected_frac"] < 1


def test_mismatch_metrics_pool_across_shards():
    new, old, sampling, mask, adv = _data(rows = 8, seed = 9)
    _, flat = _loss(new, old, sampling, mask, adv, "sequence_truncate", 0.5, 2.0)
    whole = rr.grpo_rollout_mismatch_stats(old, sampling, mask, flat)
    parts = [rr.grpo_rollout_mismatch_stats(old[s], sampling[s], mask[s], flat[s]) for s in (slice(0, 3), slice(3, 8))]
    pooled = rr.grpo_rollout_stats_reduce(torch.stack(parts))
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


def test_mismatch_stats_leave_out_unscored_tokens():
    new, old, sampling, mask, adv = _data(rows = 4, seed = 3)
    _, flat = _loss(new, old, sampling, mask, adv, "token_truncate", 0.5, 2.0)
    holed = sampling.clone()
    hole = mask.to(torch.bool) & (torch.rand(mask.shape, generator = torch.Generator().manual_seed(1)) < 0.3)
    assert hole.any()
    holed[hole] = float("nan")
    got = rr.grpo_rollout_mismatch_stats(old, holed, mask, flat)
    want = rr.grpo_rollout_mismatch_stats(old, sampling, mask * (~hole), flat)
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-14)


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
    # ScaleRL prompt-level weights, 3 groups of 4 with group 1 excluded for zero variance: (1/kept groups) sum_g
    # (1/tokens in g) sum_{i in g, t} l_it, summed over micro-batches.
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


def _drive(monkeypatch, correction, training = True, should_evaluate = False, loss_type = "dapo", is_mode = None, args_mode = None, use_vllm = True,
           ratio = None, sampling_shift = 0.1, state = None):
    import collections
    import types

    mask = _DRIVER_MASK

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
            out = (mask.double() if ratio is None else ratio) if correction else torch.tensor([])
            return zero, zero, zero, zero, out, zero

    monkeypatch.setattr(rr, "UnslothEfficientGRPO", _Loss)
    model = _DriverModel()
    model.train(training)
    metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)}
    if state is not None and "trainer" in state:
        trainer = state["trainer"]
        metrics = trainer._metrics
    else:
        trainer = None
    trainer = trainer or types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = 0.5,
            vllm_importance_sampling_mode = args_mode,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = should_evaluate),
        _metrics = metrics,
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = correction,
        _autocast_dtype = None,
        beta = 0.0,
        loss_type = loss_type,
    )
    if state is not None:
        state["trainer"] = trainer
    logps = torch.zeros(mask.shape, dtype = torch.float64)
    rr.grpo_accumulated_loss(
        trainer, _DRIVER_IDS, (_DRIVER_IDS != 0).long(), 3, mask,
        torch.zeros(_DRIVER_IDS.shape[0]), logps, None,
        loss_type = loss_type, sampling_per_token_logps = logps - sampling_shift,
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


@pytest.mark.parametrize("mode", ["token_truncate", "token_mask", "geometric_mask"])
@pytest.mark.parametrize("spec", ["2.0", "0.5", " 3 "])
def test_is_clip_without_separator_is_rejected(spec, mode):
    # A bare number used to parse as a lower bound, flooring (truncate) or rejecting (mask) every ratio below it.
    with pytest.raises(ValueError, match = "lower_upper"):
        rr.grpo_parse_is_clip(spec, mode)


@pytest.mark.parametrize("spec", ["2_1", "_"])
def test_reversed_or_empty_is_clip_override_is_rejected(monkeypatch, spec):
    monkeypatch.setenv("UNSLOTH_GRPO_VLLM_IS_CLIP", spec)
    with pytest.raises(ValueError, match = "UNSLOTH_GRPO_VLLM_IS_CLIP"):
        _drive(monkeypatch, correction = True, is_mode = "token_truncate")
    monkeypatch.setenv("UNSLOTH_GRPO_VLLM_IS_CLIP", "0.5_2")
    _drive(monkeypatch, correction = True, is_mode = "token_truncate")


@pytest.mark.parametrize("options, thresholds", [
    ("token_k3", "nan"), ("token_k3", "inf"), ("token_k1", "nan"),
    ("token_k1", "nan_2"), ("token_k1", "0.5_nan"), ("seq_mean_k1", "0.5_inf"),
])
def test_rollout_rs_rejects_non_finite_thresholds(options, thresholds):
    with pytest.raises(ValueError, match = "finite"):
        rr.grpo_parse_rollout_rs(options, thresholds)


@pytest.mark.parametrize("use_vllm, correction", [(False, False), (True, False), (False, True)])
def test_unknown_trl_is_mode_is_ignored_without_is_correction(monkeypatch, use_vllm, correction):
    # TRL reads vllm_importance_sampling_mode only inside its vLLM IS correction block, so an unknown mode must stay a no-op when that correction is off.
    _drive(monkeypatch, correction = correction, use_vllm = use_vllm, args_mode = "some_future_mode")
    with pytest.raises(ValueError, match = "importance sampling mode"):
        _drive(monkeypatch, correction = True, args_mode = "some_future_mode")


@pytest.mark.parametrize("spec", ["1_1", "0.5_0.5", "2_2"])
def test_is_clip_equal_bounds_are_rejected(spec):
    # TRL GRPOConfig: vllm_importance_sampling_clip_min must be less than clip_max.
    with pytest.raises(ValueError, match = "UNSLOTH_GRPO_VLLM_IS_CLIP"):
        rr.grpo_parse_is_clip(spec, "token_truncate")
    assert rr.grpo_parse_is_clip("0.99_1", "token_mask") == (0.99, 1.0)


@pytest.mark.parametrize("gap", [25.0, 49.0, -30.0])
def test_mismatch_k1_k3_are_not_clamped(gap):
    # verl compute_offpolicy_metrics: kl = mean(rollout - old), k3_kl = mean(exp(lr) - lr - 1), no clamp.
    old = torch.zeros(1, 4, dtype = torch.float64)
    sampling = old - gap
    mask = torch.tensor([[1, 1, 1, 0]])
    got = rr.grpo_rollout_mismatch_metrics(rr.grpo_rollout_mismatch_stats(old, sampling, mask, torch.ones(1, 4, dtype = torch.float64)))
    k3 = math.expm1(gap) - gap
    assert got["sampling/rollout_kl_k1/mean"] == pytest.approx(-gap, rel = 1e-12)
    assert got["sampling/rollout_kl_k3/mean"] == pytest.approx(k3, rel = 1e-12)
    assert got["sampling/rollout_kl_k3/max"] == pytest.approx(k3, rel = 1e-12)


def test_mismatch_stats_stay_finite_on_infinite_logps():
    old = torch.zeros(1, 3, dtype = torch.float64)
    sampling = torch.tensor([[0.0, -float("inf"), -1.0]], dtype = torch.float64)
    stats = rr.grpo_rollout_mismatch_stats(old, sampling, torch.ones(1, 3, dtype = torch.long), torch.ones(1, 3, dtype = torch.float64))
    assert torch.isfinite(stats).all()


def test_mismatch_stats_stay_finite_on_many_saturated_tokens():
    # Every token at +inf log ratio: the per token cap must leave the float64 K3 sum finite too.
    n = 20000
    old = torch.zeros(1, n, dtype = torch.float64)
    sampling = torch.full((1, n), -float("inf"), dtype = torch.float64)
    stats = rr.grpo_rollout_mismatch_stats(old, sampling, torch.ones(1, n, dtype = torch.long), torch.ones(1, n, dtype = torch.float64))
    assert torch.isfinite(stats).all()


def test_rollout_metrics_pool_micro_batches_until_log(monkeypatch):
    # Docstring promise: sums pooled over micro-batches before dividing, not a mean of per-micro-batch ratios.
    state = {}
    skewed = torch.tensor([[2.0, 0, 0], [0, 0, 0]], dtype = torch.float64)
    flat = torch.ones(2, 3, dtype = torch.float64)
    first = _drive(monkeypatch, correction = True, ratio = skewed, state = state)["train"]
    n = len(first) and first["sampling/importance_sampling_ess"][0]
    metrics = _drive(monkeypatch, correction = True, ratio = flat, state = state)["train"]
    tokens = 6  # tokens each micro-batch scores
    want = (2 + tokens) ** 2 / (2 * tokens * (4 + tokens))
    assert n == pytest.approx(4 / (tokens * 4), rel = 1e-12)
    assert metrics["sampling/importance_sampling_ess"] == [pytest.approx(want, rel = 1e-12)]
    # TRL log() clears the bucket: the next window starts from the next micro-batch alone.
    metrics.clear()
    metrics = _drive(monkeypatch, correction = True, ratio = flat, state = state)["train"]
    assert metrics["sampling/importance_sampling_ess"] == [pytest.approx(1.0, rel = 1e-12)]


def _bucket_oracle(train, sampling, mask):
    sums = [[0, 0.0], [0, 0.0], [0, 0.0]]
    neg_train = neg_sampling = 0.0
    for t, s, m in zip(train.flatten().tolist(), sampling.flatten().tolist(), mask.flatten().tolist()):
        if not m or not (math.isfinite(t) and math.isfinite(s)):
            continue
        p = math.exp(s)
        b = 0 if p > 0.9 else (2 if p < 0.1 else 1)
        sums[b][0] += 1
        sums[b][1] += abs(t - s)
        neg_train -= t
        neg_sampling -= s
    return [x for pair in sums for x in pair] + [neg_train, neg_sampling]


def test_bucket_stats_match_oracle_and_split_the_mean():
    gen = torch.Generator().manual_seed(0)
    sampling = -torch.rand(4, 9, generator = gen, dtype = torch.float64) * 4.0
    train = sampling + 0.05 * torch.randn(4, 9, generator = gen, dtype = torch.float64)
    mask = torch.rand(4, 9, generator = gen) > 0.2
    sampling[0, 0] = float("nan")  # unscored by vLLM: left out
    train[1, 1] = float("-inf")
    stats = rr.grpo_rollout_bucket_stats(train.float(), sampling.float(), mask)
    assert stats.device.type == "cpu" and stats.dtype == torch.float64 and stats.numel() == 10
    want = _bucket_oracle(train.float().double(), sampling.float().double(), mask)
    assert stats.tolist() == pytest.approx(want + [0.0, 0.0], rel = 1e-9, abs = 1e-12)
    metrics = rr.grpo_rollout_bucket_metrics(stats)
    n = want[0] + want[2] + want[4]
    names = ("p_gt_0.9", "p_0.1_0.9", "p_lt_0.1")
    assert sum(metrics[f"sampling/token_frac/{b}"] for b in names) == pytest.approx(1.0)
    # share x mean over the buckets is the overall mean |dlogp|.
    overall = sum(metrics[f"sampling/token_frac/{b}"] * metrics[f"sampling/logp_abs_diff/{b}"] for b in names)
    assert overall == pytest.approx((want[1] + want[3] + want[5]) / n)
    assert metrics["sampling/policy_nll/mean"] == pytest.approx(want[6] / n)
    assert metrics["sampling/rollout_nll/mean"] == pytest.approx(want[7] / n)
    # Without entropies from the loss there is no entropy metric: the surprisals are not entropies.
    assert "sampling/policy_entropy/mean" not in metrics


def test_bucket_stats_report_the_policy_entropy_the_loss_computed():
    train = torch.tensor([[-0.1, -2.0, -0.5]])
    sampling = torch.tensor([[-0.2, -2.5, -0.4]])
    mask = torch.tensor([[1, 1, 0]])
    entropies = torch.tensor([[0.7, 1.3, 9.0]])
    stats = rr.grpo_rollout_bucket_stats(train, sampling, mask, entropies)
    assert stats[8:].tolist() == pytest.approx([2.0, 2.0])
    metrics = rr.grpo_rollout_bucket_metrics(stats)
    assert metrics["sampling/policy_entropy/mean"] == pytest.approx(1.0)
    # The sampled-token surprisal is a cross-entropy under a mismatch, kept apart from the entropy.
    assert metrics["sampling/policy_nll/mean"] == pytest.approx(1.05)


def test_bucket_metrics_pool_like_sums_and_handle_empty():
    a = rr.grpo_rollout_bucket_stats(torch.tensor([[-0.01, -1.0]]), torch.tensor([[-0.02, -1.5]]), torch.ones(1, 2))
    b = rr.grpo_rollout_bucket_stats(torch.tensor([[-4.0]]), torch.tensor([[-3.0]]), torch.ones(1, 1))
    metrics = rr.grpo_rollout_bucket_metrics(a + b)
    assert metrics["sampling/token_frac/p_lt_0.1"] == pytest.approx(1 / 3)
    assert metrics["sampling/logp_abs_diff/p_lt_0.1"] == pytest.approx(1.0)
    assert metrics["sampling/logp_abs_diff/p_gt_0.9"] == pytest.approx(0.01)
    assert metrics["sampling/logp_abs_diff/p_0.1_0.9"] == pytest.approx(0.5)
    empty = rr.grpo_rollout_bucket_metrics(torch.zeros(10, dtype = torch.float64))
    assert all(value == 0.0 for value in empty.values())


@pytest.mark.parametrize("shift, bucket", [(0.05, "p_gt_0.9"), (1.0, "p_0.1_0.9"), (3.0, "p_lt_0.1")])
def test_rollout_metrics_log_the_bucket_split_and_entropy(monkeypatch, shift, bucket):
    metrics = _drive(monkeypatch, correction = True, sampling_shift = shift)["train"]
    assert metrics[f"sampling/token_frac/{bucket}"] == [pytest.approx(1.0)]
    assert metrics[f"sampling/logp_abs_diff/{bucket}"] == [pytest.approx(shift)]
    assert metrics["sampling/rollout_nll/mean"] == [pytest.approx(shift)]
    assert metrics["sampling/policy_nll/mean"] == [pytest.approx(0.0, abs = 1e-12)]
    assert "sampling/policy_entropy/mean" not in metrics
    # The existing metrics are unchanged by the appended sums.
    assert metrics["sampling/rollout_kl_k1/mean"] == [pytest.approx(-shift)]


def test_rollout_bucket_metrics_pool_micro_batches_until_log(monkeypatch):
    state = {}
    _drive(monkeypatch, correction = True, sampling_shift = 0.05, state = state)
    metrics = _drive(monkeypatch, correction = True, sampling_shift = 3.0, state = state)["train"]
    assert metrics["sampling/token_frac/p_gt_0.9"] == [pytest.approx(0.5)]
    assert metrics["sampling/token_frac/p_lt_0.1"] == [pytest.approx(0.5)]
    assert metrics["sampling/rollout_nll/mean"] == [pytest.approx((0.05 + 3.0) / 2)]


def _penalty_trainer(correction = True, use_vllm = True, **args):
    import types

    return types.SimpleNamespace(
        args = types.SimpleNamespace(**args), use_vllm = use_vllm, vllm_importance_sampling_correction = correction,
    )


@pytest.mark.parametrize("args, found", [
    ({"repetition_penalty": 1.1}, "repetition_penalty=1.1"),
    ({"generation_kwargs": {"presence_penalty": 0.5}}, "presence_penalty=0.5"),
    ({"generation_kwargs": {"frequency_penalty": -0.2}}, "frequency_penalty=-0.2"),
    ({"generation_kwargs": {"logit_bias": {7: -5.0}}}, "logit_bias"),
])
def test_penalties_with_is_on_warn_once(monkeypatch, args, found):
    seen = []
    monkeypatch.setattr(rr.logger, "warning", lambda message, *a, **k: seen.append(message))
    trainer = _penalty_trainer(**args)
    rr.grpo_warn_unreplayed_penalties(trainer)
    rr.grpo_warn_unreplayed_penalties(trainer)
    assert len(seen) == 1 and found in seen[0]


@pytest.mark.parametrize("trainer_kwargs", [
    {"repetition_penalty": 1.0, "generation_kwargs": {"presence_penalty": 0.0, "logit_bias": {}}},
    {"repetition_penalty": 1.3, "correction": False},
    {"repetition_penalty": 1.3, "use_vllm": False},
])
def test_default_penalties_or_is_off_stay_silent(monkeypatch, trainer_kwargs):
    seen = []
    monkeypatch.setattr(rr.logger, "warning", lambda message, *a, **k: seen.append(message))
    rr.grpo_warn_unreplayed_penalties(_penalty_trainer(**trainer_kwargs))
    assert seen == []


def test_the_loss_warns_about_penalties(monkeypatch):
    seen = []
    monkeypatch.setattr(rr.logger, "warning", lambda message, *a, **k: seen.append(message))
    state = {}
    _drive(monkeypatch, correction = True, state = state)
    state["trainer"].args.repetition_penalty = 1.2
    state["trainer"]._unsloth_grpo_penalty_warned = False
    _drive(monkeypatch, correction = True, state = state)
    _drive(monkeypatch, correction = True, state = state)
    assert sum("repetition_penalty=1.2" in m for m in seen) == 1


def test_rollout_metrics_log_the_policy_entropy_when_the_loss_has_it(monkeypatch):
    state = {}
    first = _drive(monkeypatch, correction = True, state = state)["train"]
    assert "sampling/policy_entropy/mean" not in first
    state["trainer"].args.entropy_coef = 0.01
    # The bonus weights need TRL's num_items_in_batch, which the driver does not pass.
    monkeypatch.setattr(rr, "grpo_entropy_bonus_weights", lambda mask, *a, **k: torch.zeros(mask.shape))
    first.clear()
    metrics = _drive(monkeypatch, correction = True, state = state)["train"]
    entropy = metrics["sampling/policy_entropy/mean"][0]
    assert 0.0 < entropy <= math.log(8) + 1e-6  # the driver model's vocabulary is 8


# verl 95ede34 (Apache-2.0), verl/trainer/ppo/rollout_corr_helper.py:940-1057 compute_offpolicy_metrics and the
# token-level branch of compute_is_metrics, with verl.utils.torch_functional's masked helpers: the parity oracle.
_VERL_SAFETY_BOUND = 20.0


def _verl_masked_sum(values, mask, axis = None):
    return (torch.where(mask.bool(), values, 0.0) * mask).sum(axis = axis)


def _verl_masked_mean(values, mask, axis = None):
    return _verl_masked_sum(values, mask, axis) / (mask.sum(axis = axis) + 1e-8)


def _verl_compute_offpolicy_metrics(old_log_prob, rollout_log_prob, response_mask):
    metrics = {}
    seq_valid_mask = response_mask.sum(dim = -1) > 0
    mean_log_prob_training = _verl_masked_mean(old_log_prob, response_mask, axis = -1)
    metrics["training_ppl"] = _verl_masked_mean(torch.exp(-mean_log_prob_training), seq_valid_mask).item()
    metrics["training_log_ppl"] = _verl_masked_mean(-mean_log_prob_training, seq_valid_mask).item()
    metrics["kl"] = _verl_masked_mean(rollout_log_prob - old_log_prob, response_mask).item()
    log_ratio = old_log_prob - rollout_log_prob
    metrics["k3_kl"] = _verl_masked_mean(torch.exp(log_ratio) - log_ratio - 1, response_mask).item()
    mean_log_prob_rollout = _verl_masked_mean(rollout_log_prob, response_mask, axis = -1)
    metrics["rollout_ppl"] = _verl_masked_mean(torch.exp(-mean_log_prob_rollout), seq_valid_mask).item()
    metrics["rollout_log_ppl"] = _verl_masked_mean(-mean_log_prob_rollout, seq_valid_mask).item()
    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics["log_ppl_diff"] = _verl_masked_mean(log_ppl_diff, seq_valid_mask).item()
    metrics["log_ppl_abs_diff"] = _verl_masked_mean(log_ppl_diff.abs(), seq_valid_mask).item()
    metrics["log_ppl_diff_max"] = log_ppl_diff.masked_fill(~seq_valid_mask, float("-inf")).max().item()
    metrics["log_ppl_diff_min"] = log_ppl_diff.masked_fill(~seq_valid_mask, float("inf")).min().item()
    metrics["ppl_ratio"] = _verl_masked_mean(torch.exp(log_ppl_diff), seq_valid_mask).item()
    rho_token = torch.exp(torch.clamp(log_ratio, min = -_VERL_SAFETY_BOUND, max = _VERL_SAFETY_BOUND))
    metrics["chi2_token"] = (_verl_masked_mean(rho_token.square(), response_mask) - 1.0).item()
    log_ratio_sum = _verl_masked_sum(log_ratio, response_mask, axis = -1)
    rho_squared_seq = torch.exp(2.0 * torch.clamp(log_ratio_sum, min = -_VERL_SAFETY_BOUND, max = _VERL_SAFETY_BOUND))
    metrics["chi2_seq"] = (_verl_masked_mean(rho_squared_seq, seq_valid_mask) - 1.0).item()
    return metrics


def _verl_token_is_metrics(weights, raw_weights, response_mask, threshold, threshold_lower):
    metrics = {"rollout_is_mean": _verl_masked_mean(weights, response_mask).item()}
    metrics["rollout_is_ratio_fraction_high"] = _verl_masked_mean((raw_weights > threshold).double(), response_mask).item()
    metrics["rollout_is_ratio_fraction_low"] = _verl_masked_mean((raw_weights < threshold_lower).double(), response_mask).item()
    m = response_mask.bool()
    metrics["rollout_is_max"] = weights.masked_fill(~m, float("-inf")).max().item()
    metrics["rollout_is_min"] = weights.masked_fill(~m, float("inf")).min().item()
    clipped = weights.clamp(min = 0.0, max = threshold)
    mean = _verl_masked_mean(clipped, response_mask)
    metrics["rollout_is_std"] = torch.sqrt(torch.clamp(_verl_masked_mean((clipped - mean) ** 2, response_mask), min = 0.0)).item()
    normalized = clipped / (_verl_masked_mean(clipped, response_mask) + 1e-8)
    metrics["rollout_is_eff_sample_size"] = 1.0 / _verl_masked_mean(normalized.square(), response_mask).item()
    return metrics


_VERL_TO_UNSLOTH = {
    "kl": "sampling/rollout_kl_k1/mean",
    "k3_kl": "sampling/rollout_kl_k3/mean",
    "training_ppl": "sampling/training_ppl",
    "training_log_ppl": "sampling/training_log_ppl",
    "rollout_ppl": "sampling/rollout_ppl",
    "rollout_log_ppl": "sampling/rollout_log_ppl",
    "log_ppl_diff": "sampling/log_ppl_diff/mean",
    "log_ppl_abs_diff": "sampling/log_ppl_diff/abs_mean",
    "log_ppl_diff_max": "sampling/log_ppl_diff/max",
    "log_ppl_diff_min": "sampling/log_ppl_diff/min",
    "ppl_ratio": "sampling/ppl_ratio",
    "chi2_token": "sampling/rollout_chi2_token",
    "chi2_seq": "sampling/rollout_chi2_seq",
    "rollout_is_mean": "sampling/importance_sampling_weight/mean",
    "rollout_is_std": "sampling/importance_sampling_weight/std",
    "rollout_is_max": "sampling/importance_sampling_weight/max",
    "rollout_is_min": "sampling/importance_sampling_weight/min",
    "rollout_is_ratio_fraction_high": "sampling/importance_sampling_ratio_fraction_high",
    "rollout_is_ratio_fraction_low": "sampling/importance_sampling_ratio_fraction_low",
    "rollout_is_eff_sample_size": "sampling/importance_sampling_ess",
}


@pytest.mark.parametrize("seed, scale", [(11, 0.05), (12, 0.3), (13, 1.0)])
def test_mismatch_metrics_match_verl(seed, scale):
    new, old, sampling, mask, adv = _data(rows = 7, seed = seed, scale = scale)
    clip_min, clip_max = 0.5, 2.0
    _, flat = _loss(new, old, sampling, mask, adv, "token_truncate", clip_min, clip_max)
    stats = rr.grpo_rollout_mismatch_stats(old, sampling, mask, flat, clip_max = clip_max, clip_min = clip_min)
    # Pooled over two shards first: the sums must give the whole batch's numbers.
    parts = [rr.grpo_rollout_mismatch_stats(old[s], sampling[s], mask[s], flat[s], clip_max = clip_max, clip_min = clip_min)
             for s in (slice(0, 3), slice(3, 7))]
    torch.testing.assert_close(rr.grpo_rollout_stats_reduce(torch.stack(parts)), stats, rtol = 1e-12, atol = 1e-12)
    got = rr.grpo_rollout_mismatch_metrics(stats)
    want = _verl_compute_offpolicy_metrics(old, sampling, mask)
    raw = torch.exp(old - sampling)
    want.update(_verl_token_is_metrics(torch.broadcast_to(flat, mask.shape), raw, mask, clip_max, clip_min))
    for key, name in _VERL_TO_UNSLOTH.items():
        assert abs(got[name] - want[key]) <= 1e-6 * max(1.0, abs(want[key])), (key, got[name], want[key])
    # ms-swift's geometric chi2_seq: E_seq[exp(2 clamp(mean log rho))] - 1 over rows with tokens.
    rows = mask.sum(-1) > 0
    geo = torch.exp(2 * torch.clamp(((old - sampling) * mask).sum(-1) / mask.sum(-1).clamp(min = 1), -20, 20))
    assert got["sampling/rollout_chi2_seq_geometric"] == pytest.approx(float(geo[rows].mean() - 1), rel = 1e-9)
    assert got["sampling/rollout_logprobs_mismatch_count"] == float(((old != sampling) & mask.bool()).sum())


def test_mismatch_count_is_bitwise():
    x = torch.tensor([[-1.0, -2.0, -3.0]])
    y = x.clone()
    y[0, 1] = torch.nextafter(y[0, 1], torch.tensor(0.0))
    stats = rr.grpo_rollout_mismatch_stats(x, y, torch.ones(1, 3), torch.ones(1, 3))
    assert rr.grpo_rollout_mismatch_metrics(stats)["sampling/rollout_logprobs_mismatch_count"] == 1.0


@pytest.mark.parametrize("metrics, warns", [
    ({"sampling/importance_sampling_ess": 0.2, "sampling/rollout_chi2_token": 0.1}, True),
    ({"sampling/importance_sampling_ess": 0.9, "sampling/rollout_chi2_token": 1.5}, True),
    ({"sampling/importance_sampling_ess": 0.9, "sampling/rollout_chi2_token": 0.1}, False),
])
def test_large_mismatch_warns_once(monkeypatch, metrics, warns):
    import types

    seen = []
    monkeypatch.setattr(rr.logger, "warning", lambda message, *a, **k: seen.append(message))
    trainer = types.SimpleNamespace()
    rr.grpo_warn_rollout_mismatch(trainer, metrics)
    rr.grpo_warn_rollout_mismatch(trainer, metrics)
    assert len(seen) == (1 if warns else 0)


@pytest.mark.parametrize("mode", ["sequence_mask", "sequence_truncate", "token_mask"])
def test_is_ratio_stays_finite_when_a_row_sums_past_float32_exp(mode):
    # A row whose log rho sums to 100 (exp overflows float32 past ~88.7): an unbounded *_mask mode kept an
    # infinite weight, and the sequence metrics went inf / nan.
    new, old, sampling, mask, adv = _data(rows = 3, cols = 10, seed = 5)
    mask = torch.ones_like(mask)
    new, old, sampling = new.detach().float().requires_grad_(True), old.float(), sampling.float()
    sampling = sampling.clone()
    sampling[2] = old[2] - (100.0 if mode != "token_mask" else torch.tensor([100.0] + [0.0] * 9))
    clip_max = None if mode.endswith("_mask") else 2.0
    loss, flat = _loss(new, old, sampling, mask, adv, mode, None, clip_max)
    assert torch.isfinite(flat).all() and torch.isfinite(loss)
    stats = rr.grpo_rollout_mismatch_stats(old, sampling, mask, flat, clip_max = clip_max)
    got = rr.grpo_rollout_mismatch_metrics(stats)
    assert all(math.isfinite(v) for v in got.values()), got
    assert float(flat.max()) <= math.exp(20.0) * (1 + 1e-6)


def test_fp8_kv_cache_without_is_warns_once(monkeypatch):
    import types

    seen = []
    monkeypatch.setattr(rr.logger, "warning", lambda message, *a, **k: seen.append(message))

    def trainer(correction, fp8, by_config = False):
        config = types.SimpleNamespace(cache_config = types.SimpleNamespace(cache_dtype = "fp8" if by_config else "auto"))
        llm = types.SimpleNamespace(llm_engine = types.SimpleNamespace(vllm_config = config))
        if fp8 and not by_config:
            llm._unsloth_float8_kv_cache = True
        return types.SimpleNamespace(use_vllm = True, vllm_importance_sampling_correction = correction, llm = llm)

    off = trainer(False, True)
    rr.grpo_warn_fp8_kv_cache_without_is(off)
    rr.grpo_warn_fp8_kv_cache_without_is(off)
    assert len(seen) == 1 and "float8_kv_cache" in seen[0]
    rr.grpo_warn_fp8_kv_cache_without_is(trainer(False, True, by_config = True))
    assert len(seen) == 2
    rr.grpo_warn_fp8_kv_cache_without_is(trainer(True, True))
    rr.grpo_warn_fp8_kv_cache_without_is(trainer(False, False))
    assert len(seen) == 2


def test_load_vllm_tags_the_fp8_kv_cache():
    import inspect
    import unsloth_zoo.vllm_utils as vu

    assert "llm._unsloth_float8_kv_cache = bool(float8_kv_cache)" in inspect.getsource(vu.load_vllm)
