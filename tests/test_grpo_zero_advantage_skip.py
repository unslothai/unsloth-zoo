# SPDX-License-Identifier: AGPL-3.0-or-later
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""GRPO zero-advantage row skip and old_logps_from_new vs float64 oracles of TRL 1.18 _compute_loss."""

import collections
import math
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

EPS_LOW, EPS_HIGH = 0.2, 0.28
LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo")
IS_MODES = (None, "token_truncate", "token_mask", "sequence_truncate", "sequence_mask")


@pytest.fixture(autouse = True)
def _log_prob_route(monkeypatch):
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)
    monkeypatch.delenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", raising = False)


_IS_LOG_RATIO_BOUND = 20.0


def _trl_loss(new, old, sampling, adv, mask, loss_type, level, is_mode, ga = 1, max_len = 16, num_items = None,
              clip_min = None, clip_max = 3.0, beta = 0.0, ref = None, spg = 4):
    """TRL 1.18 _compute_loss (+ the IS ratio _generate_and_score_completions builds), float64, written out."""
    old = new.detach() if old is None else old
    a = adv.unsqueeze(1)
    log_ratio = new - old
    if level == "token":
        log_w = log_ratio
    else:
        log_w = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min = 1.0)).unsqueeze(-1)
    coef_1 = torch.exp(log_w)
    coef_2 = torch.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    per_token = -torch.min(coef_1 * a, coef_2 * a)
    if is_mode is not None:
        diff = torch.nan_to_num((old - sampling) * mask, nan = 0.0)
        if is_mode.startswith("sequence"):
            diff = diff.sum(-1, keepdim = True)
        # grpo_compute_loss bounds the IS log ratio to +-20 like verl's SAFETY_BOUND (#2818); TRL does not. Only the lower
        # bound changes a result here (d = +20 / +88): above +20 the clip_max = 3 truncation or mask already decides.
        ratio = torch.exp(torch.clamp(diff, min = -_IS_LOG_RATIO_BOUND, max = _IS_LOG_RATIO_BOUND))
        if is_mode.endswith("truncate"):
            ratio = torch.clamp(ratio, min = clip_min, max = clip_max)
        else:
            lo = clip_min if clip_min is not None else -math.inf
            hi = clip_max if clip_max is not None else math.inf
            ratio = ratio.masked_fill((ratio < lo) | (ratio > hi), 0.0)
        per_token = per_token * ratio
    if beta != 0.0:
        kl = torch.exp(ref - new) - (ref - new) - 1
        per_token = per_token + beta * kl
    if loss_type == "grpo":
        loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min = 1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token * mask).sum() / mask.sum().clamp(min = 1.0)
    elif loss_type == "dr_grpo":
        loss = (per_token * mask).sum() / (per_token.size(0) * max_len)
    else:
        # num_items_in_batch spans the generation batch: rescaled to one accumulation window.
        return (per_token * mask).sum() / (max(float(num_items), 1.0) * ga / spg)
    return loss / ga


def _zoo_kwargs(loss_type, level, is_mode, ga = 1, max_len = 16, num_items = None, **extra):
    kw = dict(
        loss_type = loss_type, importance_sampling_level = level, epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH,
        max_completion_length = max_len, current_gradient_accumulation_steps = ga, num_processes = 1,
        steps_per_generation = 4,
        num_items_in_batch = num_items, use_vllm = is_mode is not None,
        vllm_importance_sampling_correction = is_mode is not None,
        vllm_importance_sampling_mode = is_mode or "token_truncate",
        vllm_importance_sampling_clip_min = None, vllm_importance_sampling_clip_max = 3.0,
    )
    kw.update(extra)
    return kw


def _batch(rows = 8, cols = 9, seed = 0, d = 0.3, zero_rows = (1, 2, 5), dtype = torch.float64):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, cols + 1, (rows,), generator = g)
    lengths[0] = cols   # full row
    lengths[3] = 0      # fully masked row
    lengths[4] = 1      # one-token row
    mask = (torch.arange(cols)[None] < lengths[:, None]).to(dtype)
    new = -torch.rand(rows, cols, generator = g, dtype = dtype) * 3
    old = new + d * torch.randn(rows, cols, generator = g, dtype = dtype)
    sampling = new + 0.2 * torch.randn(rows, cols, generator = g, dtype = dtype)
    adv = torch.randn(rows, generator = g, dtype = dtype)
    adv[list(zero_rows)] = 0.0
    ids = torch.randint(0, 50, (rows, cols), generator = g)
    return new, old, sampling, adv, mask, ids


def _zoo_loss(new, old, sampling, adv, mask, ids, kw):
    return rr.grpo_compute_loss(None, new, old, sampling, ids, mask, 0.0, adv, **kw)[0]


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("is_mode", IS_MODES)
@pytest.mark.parametrize("d", [0.0, 1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0])
def test_scored_rows_match_the_whole_batch(loss_type, level, is_mode, d):
    """Loss and gradient of the kept rows with the full-batch normaliser == TRL on every row (float64)."""
    new, old, sampling, adv, mask, ids = _batch(d = 0.0)
    old = new.detach() - d if d != 0.0 else old.detach()   # log ratio new - old = d (plus noise at d = 0)
    num_items = mask.sum()
    ga = 3
    leaf = new.clone().requires_grad_(True)
    want = _trl_loss(leaf, old, sampling, adv, mask, loss_type, level, is_mode, ga = ga, num_items = num_items)
    (want_grad,) = torch.autograd.grad(want, leaf)

    keep = rr.grpo_zero_advantage_keep(adv, adv.shape[0])
    assert keep is not None and keep.tolist() == [0, 3, 4, 6, 7]
    leaf_k = new[keep].clone().requires_grad_(True)
    kw = _zoo_kwargs(
        loss_type, level, is_mode, ga = ga, num_items = num_items,
        full_batch_rows = torch.tensor(float(adv.shape[0]), dtype = torch.float64),
        full_batch_tokens = mask.sum(),
    )
    got = _zoo_loss(leaf_k, old[keep], sampling[keep], adv[keep], mask[keep], ids[keep], kw)
    (got_grad,) = torch.autograd.grad(got, leaf_k)

    if not torch.isfinite(want):
        # exp(+88) overflows only in float32; float64 stays finite. Anything else must match.
        pytest.fail(f"oracle not finite: {want}")
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-15)
    full_grad = torch.zeros_like(want_grad)
    full_grad[keep] = got_grad
    torch.testing.assert_close(full_grad, want_grad, rtol = 1e-12, atol = 1e-15)
    dropped = torch.ones(adv.shape[0], dtype = torch.bool)
    dropped[keep] = False
    assert torch.count_nonzero(want_grad[dropped]) == 0


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_default_normaliser_is_unchanged_without_the_counts(loss_type):
    """No full_batch_* kwargs (every existing caller): bitwise the same as before, i.e. TRL on the rows given."""
    new, old, sampling, adv, mask, ids = _batch()
    kw = _zoo_kwargs(loss_type, "token", None, num_items = mask.sum())
    got = _zoo_loss(new, old, sampling, adv, mask, ids, kw)
    want = _trl_loss(new, old, sampling, adv, mask, loss_type, "token", None, num_items = mask.sum())
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 0.0)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_float32_error_does_not_grow(loss_type):
    """Real dtype: the kept-rows loss sits as close to the float64 oracle as the whole-batch float32 loss."""
    new, old, sampling, adv, mask, ids = _batch(rows = 16, cols = 33, zero_rows = range(0, 16, 2))
    want = _trl_loss(new, old, sampling, adv, mask, loss_type, "token", "token_truncate", num_items = mask.sum())
    f = lambda x: x.to(torch.float32)
    full = _zoo_loss(f(new), f(old), f(sampling), f(adv), f(mask), ids,
                     _zoo_kwargs(loss_type, "token", "token_truncate", num_items = f(mask).sum()))
    keep = rr.grpo_zero_advantage_keep(adv, adv.shape[0])
    kept = _zoo_loss(f(new)[keep], f(old)[keep], f(sampling)[keep], f(adv)[keep], f(mask)[keep], ids[keep],
                     _zoo_kwargs(loss_type, "token", "token_truncate", num_items = f(mask).sum(),
                                 full_batch_rows = torch.tensor(16.0), full_batch_tokens = f(mask).sum()))
    err_full = (full.double() - want).abs().item()
    err_kept = (kept.double() - want).abs().item()
    assert err_kept <= max(2 * err_full, 4 * torch.finfo(torch.float32).eps * want.abs().item() + 1e-12)


def test_gradcheck_float64_with_full_counts():
    new, old, sampling, adv, mask, ids = _batch(rows = 6, cols = 5, zero_rows = (1, 2))
    keep = rr.grpo_zero_advantage_keep(adv, 6)
    for loss_type in LOSS_TYPES:
        kw = _zoo_kwargs(loss_type, "token", "sequence_mask", num_items = mask.sum(),
                         full_batch_rows = torch.tensor(6.0, dtype = torch.float64), full_batch_tokens = mask.sum())

        def f(x):
            return _zoo_loss(x, old[keep], sampling[keep], adv[keep], mask[keep], ids[keep], kw)
        assert torch.autograd.gradcheck(f, (new[keep].clone().requires_grad_(True),))


def test_keep_rows_helper():
    adv = torch.tensor([0.0, 1.0, 0.0, -2.0, 0.0, 0.0])
    assert rr.grpo_zero_advantage_keep(adv, 6).tolist() == [1, 3]
    # Whole groups only (PrefixGrouper layout): group [0, 1] has a nonzero row, [4, 5] is all zero.
    assert rr.grpo_zero_advantage_keep(adv, 6, group_size = 2).tolist() == [0, 1, 2, 3]
    # Every row zero: one row is still scored, so the forward / backward run.
    assert rr.grpo_zero_advantage_keep(torch.zeros(4), 4).tolist() == [0]
    # Nothing to drop, NaN is not zero, a near zero is not zero, token-level advantages need all zeros.
    assert rr.grpo_zero_advantage_keep(torch.ones(3), 3) is None
    assert rr.grpo_zero_advantage_keep(torch.tensor([float("nan"), 0.0]), 2).tolist() == [0]
    assert rr.grpo_zero_advantage_keep(torch.tensor([1e-30, 0.0]), 2).tolist() == [0]
    assert rr.grpo_zero_advantage_keep(torch.tensor([[0.0, 0.0], [0.0, 1.0]]), 2).tolist() == [1]
    assert rr.grpo_zero_advantage_keep(torch.zeros(3), 4) is None
    assert rr.grpo_zero_advantage_keep(None, 4) is None


@pytest.mark.parametrize("is_mode", IS_MODES[1:])
@pytest.mark.parametrize("level", ["token", "sequence"])
def test_old_from_new_matches_trl_aligned_pi_old(is_mode, level):
    """pi_old = pi_theta (aligned generation): the IS weight is exp(pi_theta - pi_vllm) of the detached
    forward and the PPO ratio is 1, exactly TRL's loss and gradient with old_per_token_logps = pi_theta."""
    new, _, sampling, adv, mask, ids = _batch()
    leaf = new.clone().requires_grad_(True)
    want = _trl_loss(leaf, new.detach(), sampling, adv, mask, "dapo", level, is_mode, num_items = mask.sum())
    (want_grad,) = torch.autograd.grad(want, leaf)
    leaf2 = new.clone().requires_grad_(True)
    old = rr.grpo_old_logps_from_new(leaf2).to(torch.float64)
    assert not old.requires_grad
    got = _zoo_loss(leaf2, old, sampling, adv, mask, ids, _zoo_kwargs("dapo", level, is_mode, num_items = mask.sum()))
    (got_grad,) = torch.autograd.grad(got, leaf2)
    torch.testing.assert_close(got, want, rtol = 1e-12, atol = 1e-15)
    torch.testing.assert_close(got_grad, want_grad, rtol = 1e-12, atol = 1e-15)


def test_old_from_new_is_closer_to_the_oracle_than_a_second_forward():
    """A bf16-noisy second pi_old forward drifts from the float64 oracle; the detached one does not."""
    new, _, sampling, adv, mask, ids = _batch(rows = 16, cols = 33, zero_rows = ())
    want = _trl_loss(new, new, sampling, adv, mask, "dapo", "token", "token_truncate", num_items = mask.sum())
    f = lambda x: x.to(torch.float32)
    kw = _zoo_kwargs("dapo", "token", "token_truncate", num_items = f(mask).sum())
    second_forward = f(new).to(torch.bfloat16).to(torch.float32)
    base = _zoo_loss(f(new), second_forward, f(sampling), f(adv), f(mask), ids, kw)
    head = _zoo_loss(f(new), rr.grpo_old_logps_from_new(f(new)), f(sampling), f(adv), f(mask), ids, kw)
    assert abs(head.double() - want).item() <= abs(base.double() - want).item()
    assert abs(head.double() - want).item() < 1e-5 * max(1.0, abs(want.item()))


# ---- grpo_accumulated_loss end to end on CPU (real UnslothEfficientGRPO) ----

V, H = 16, 6
_APPLY = rr.UnslothEfficientGRPO.apply
# Left-padded prompts (pad 0), right-padded completions; 6 rows = 3 groups of 2.
_IDS = torch.tensor([
    [0, 0, 1, 2, 3, 4, 5, 6],
    [0, 0, 1, 2, 7, 8, 9, 0],
    [1, 2, 6, 7, 3, 4, 6, 9],
    [1, 2, 6, 7, 5, 0, 0, 0],
    [0, 3, 2, 3, 5, 9, 8, 7],
    [0, 3, 2, 3, 11, 12, 0, 0],
])
_COMPLETION = 4


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.mix = torch.nn.Linear(H, H)
        self.head = torch.nn.Linear(H, V, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = torch.tanh(self.mix(self.emb(input_ids)))
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _trainer(model, beta = 0.0, entropy_coef = 0.0, use_vllm = True, correction = True, is_mode = "token_truncate"):
    return types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = 2, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = None,
            vllm_importance_sampling_mode = is_mode, vllm_importance_sampling_cap = 3.0,
            vllm_importance_sampling_clip_min = None, vllm_importance_sampling_clip_max = 3.0,
            sapo_temperature_pos = 1.0, sapo_temperature_neg = 1.05,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = False),
        _metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)},
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = correction,
        _autocast_dtype = None,
        beta = beta,
        num_generations = 2,
        entropy_coef = entropy_coef,
    )


def _sampling():
    g = torch.Generator().manual_seed(1)
    return -torch.rand(_IDS.shape[0], _COMPLETION, generator = g) * 2


def _score_old(model):
    """The trainer's own no-grad pi_old forward (what TRL runs when the old logps are not skipped)."""
    with torch.no_grad():
        trainer = _trainer(model, use_vllm = False, correction = False)
        seen = {}

        def spy(new, *args):
            seen["new"] = new.detach().clone()
            return _APPLY(new, *args)
        rr.UnslothEfficientGRPO.apply = spy
        try:
            rr.grpo_accumulated_loss(trainer, _IDS, (_IDS != 0).long(), _COMPLETION, (_IDS[:, -_COMPLETION:] != 0).long(),
                                     torch.ones(_IDS.shape[0]), None, None, loss_type = "dapo",
                                     num_items_in_batch = torch.tensor(1.0))
        finally:
            rr.UnslothEfficientGRPO.apply = _APPLY
    return seen["new"]


def _run(monkeypatch, adv, loss_type = "dapo", skip = True, old = None, old_from_new = False, beta = 0.0, ref = None,
         entropy_coef = 0.0, use_vllm = True, correction = True, is_mode = "token_truncate", prefix_grouper = "0"):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", prefix_grouper)
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "1" if skip else "0")
    model = _Model()
    model.train(True)
    trainer = _trainer(model, beta = beta, entropy_coef = entropy_coef, use_vllm = use_vllm, correction = correction,
                       is_mode = is_mode)
    calls = []

    def spy(new, *args):
        calls.append(new.shape[0])
        return _APPLY(new, *args)
    monkeypatch.setattr(rr.UnslothEfficientGRPO, "apply", spy)
    mask = (_IDS[:, -_COMPLETION:] != 0).long()
    kwargs = dict(loss_type = loss_type, sampling_per_token_logps = _sampling(), max_completion_length = 8,
                  num_items_in_batch = mask.sum().float(), current_gradient_accumulation_steps = 2,
                  epsilon_low = EPS_LOW, epsilon_high = EPS_HIGH)
    if old_from_new:
        kwargs["old_logps_from_new"] = True
    out = rr.grpo_accumulated_loss(trainer, _IDS, (_IDS != 0).long(), _COMPLETION, mask, adv, old, ref, **kwargs)
    out[0].backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return out, grads, calls, trainer


_ADV = torch.tensor([0.0, 0.0, 1.5, -1.5, 0.0, 0.0])   # groups 0 and 2 have zero variance


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("is_mode", ["token_truncate", "sequence_mask"])
def test_accumulated_loss_skips_rows_with_identical_loss_and_grads(monkeypatch, loss_type, is_mode):
    old = _score_old(_Model())
    base, base_grads, base_calls, _ = _run(monkeypatch, _ADV, loss_type, skip = False, old = old, is_mode = is_mode)
    head, head_grads, head_calls, trainer = _run(monkeypatch, _ADV, loss_type, skip = True, old = old, is_mode = is_mode)
    assert base_calls == [6] and head_calls == [2]
    assert trainer._unsloth_zero_advantage_rows_skipped == 4
    assert trainer._metrics["train"]["unsloth/zero_advantage_rows_skipped"] == [pytest.approx(4 / 6)]
    torch.testing.assert_close(head[0], base[0], rtol = 1e-6, atol = 1e-9)
    for name in base_grads:
        torch.testing.assert_close(head_grads[name], base_grads[name], rtol = 1e-5, atol = 1e-9)
    assert any(g.abs().sum() > 0 for g in head_grads.values())
    # Metrics keep every row: same completion length, mask, ratio and vLLM IS values (pi_old is given).
    torch.testing.assert_close(head[1], base[1])
    torch.testing.assert_close(head[6], base[6])
    torch.testing.assert_close(head[5], base[5], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(head[3], base[3], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(head[4], base[4], rtol = 1e-5, atol = 1e-6)


def test_accumulated_loss_keeps_whole_groups_for_the_prefix_grouper(monkeypatch):
    adv = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # row 0 is zero inside a non-degenerate group
    _, _, calls, _ = _run(monkeypatch, adv, use_vllm = False, correction = False, prefix_grouper = "1")
    assert calls == [2]
    _, _, calls, _ = _run(monkeypatch, adv, use_vllm = False, correction = False, prefix_grouper = "0")
    assert calls == [1]


def test_all_rows_zero_still_runs_one_row(monkeypatch):
    out, grads, calls, _ = _run(monkeypatch, torch.zeros(6), old = _score_old(_Model()))
    assert calls == [1]
    assert float(out[0]) == 0.0 and all(float(g.abs().sum()) == 0.0 for g in grads.values())


@pytest.mark.parametrize("case", ["beta", "entropy", "kill_switch", "sapo", "cispo"])
def test_no_skip_outside_the_exact_cases(monkeypatch, case):
    ref = None
    kw = {}
    if case == "beta":
        ref = _score_old(_Model()) - 0.1
        kw = dict(beta = 0.04, ref = ref)
    elif case == "entropy":
        kw = dict(entropy_coef = 0.01)
    elif case == "kill_switch":
        kw = dict(skip = False)
    elif case in ("sapo", "cispo"):
        kw = dict(loss_type = case)
    _, _, calls, trainer = _run(monkeypatch, _ADV, old = _score_old(_Model()), **kw)
    assert calls == [6]
    assert not hasattr(trainer, "_unsloth_zero_advantage_rows_skipped")


@pytest.mark.parametrize("is_mode", ["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"])
def test_accumulated_loss_old_from_new_equals_the_scored_pi_old(monkeypatch, is_mode):
    """old_logps_from_new matches a separate pi_old forward of the same weights."""
    adv = torch.tensor([0.5, -0.5, 1.5, -1.5, 0.2, -0.2])
    old = _score_old(_Model())
    base, base_grads, _, _ = _run(monkeypatch, adv, old = old, is_mode = is_mode, skip = False)
    head, head_grads, _, _ = _run(monkeypatch, adv, old = None, old_from_new = True, is_mode = is_mode, skip = False)
    torch.testing.assert_close(head[0], base[0], rtol = 1e-6, atol = 1e-9)
    for name in base_grads:
        torch.testing.assert_close(head_grads[name], base_grads[name], rtol = 1e-5, atol = 1e-9)
    torch.testing.assert_close(head[3], base[3], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(head[4], base[4], rtol = 1e-5, atol = 1e-6)


def test_old_from_new_with_skipped_rows_marks_their_is_metrics_unscored(monkeypatch):
    old = _score_old(_Model())
    base, base_grads, _, _ = _run(monkeypatch, _ADV, old = old, skip = False)
    head, head_grads, calls, _ = _run(monkeypatch, _ADV, old = None, old_from_new = True, skip = True)
    assert calls == [2]
    torch.testing.assert_close(head[0], base[0], rtol = 1e-6, atol = 1e-9)
    for name in base_grads:
        torch.testing.assert_close(head_grads[name], base_grads[name], rtol = 1e-5, atol = 1e-9)
    kept = torch.tensor([2, 3])
    dropped = torch.tensor([0, 1, 4, 5])
    assert torch.isnan(head[3][dropped]).all() and torch.isnan(head[4][dropped]).all()
    torch.testing.assert_close(head[3][kept], base[3][kept], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(head[4][kept], base[4][kept], rtol = 1e-5, atol = 1e-6)


def test_old_from_new_without_the_flag_keeps_the_none_path(monkeypatch):
    """old_logps None without the flag (IS off, aligned): unchanged, ratio 1 from new - new.detach()."""
    adv = torch.tensor([0.5, -0.5, 1.5, -1.5, 0.2, -0.2])
    a, ga, _, _ = _run(monkeypatch, adv, old = None, use_vllm = False, correction = False, skip = False)
    b, gb, _, _ = _run(monkeypatch, adv, old = None, old_from_new = True, use_vllm = False, correction = False, skip = False)
    torch.testing.assert_close(a[0], b[0])
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name])


# ---- what the two savings refuse: random forwards, mixture of experts, sharded forwards ----


def test_stochastic_policy_detector():
    plain = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    assert not rr.grpo_policy_is_stochastic(plain)
    assert rr.grpo_policy_is_stochastic(torch.nn.Sequential(torch.nn.Dropout(0.1)))
    assert rr.grpo_policy_is_stochastic(torch.nn.Sequential(torch.nn.Dropout1d(0.1)))
    loha = torch.nn.Linear(2, 2)
    loha.rank_dropout = {"default": 0.0}
    loha.module_dropout = {"default": 0.0}
    assert not rr.grpo_policy_is_stochastic(torch.nn.Sequential(loha))
    loha.module_dropout = {"default": 0.05}
    assert rr.grpo_policy_is_stochastic(torch.nn.Sequential(loha))
    moe = torch.nn.Linear(2, 2)
    moe.jitter_noise = 0.01
    assert rr.grpo_policy_is_stochastic(torch.nn.Sequential(moe))
    for config, expected in [
        (types.SimpleNamespace(attention_dropout = 0.0, classifier_dropout = 0.1), False),
        (types.SimpleNamespace(attention_dropout = 0.1), True),
        (types.SimpleNamespace(resid_pdrop = 0.1), True),
        (types.SimpleNamespace(router_jitter_noise = 0.1), True),
        (types.SimpleNamespace(router_jitter_noise = 0.0, hidden_dropout = 0), False),
        (types.SimpleNamespace(text_config = types.SimpleNamespace(attention_dropout = 0.2)), True),
        (types.SimpleNamespace(use_dropout = True), False),   # a flag, not a rate
    ]:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        model.config = config
        assert rr.grpo_policy_is_stochastic(model) is expected, config


def test_expert_detector():
    dense = torch.nn.Sequential(torch.nn.Linear(2, 2))
    dense.config = types.SimpleNamespace(hidden_size = 2)
    assert not rr.grpo_policy_has_experts(dense)
    for name in ("num_experts", "num_local_experts", "n_routed_experts"):
        moe = torch.nn.Sequential(torch.nn.Linear(2, 2))
        moe.config = types.SimpleNamespace(**{name: 8})
        assert rr.grpo_policy_has_experts(moe)
    moe = torch.nn.Sequential(torch.nn.Linear(2, 2))
    moe.config = types.SimpleNamespace(text_config = types.SimpleNamespace(num_experts = 4))
    assert rr.grpo_policy_has_experts(moe)


@pytest.mark.parametrize("case", ["dropout", "experts", "deepspeed", "fsdp"])
def test_no_skip_for_random_expert_or_sharded_policies(monkeypatch, case):
    old = _score_old(_Model())
    orig_trainer = _trainer

    def patched_trainer(model, **kw):
        trainer = orig_trainer(model, **kw)
        if case == "dropout":
            model.drop = torch.nn.Dropout(0.1)
        elif case == "experts":
            model.config = types.SimpleNamespace(num_local_experts = 8)
        elif case == "deepspeed":
            trainer.is_deepspeed_enabled = True
        elif case == "fsdp":
            trainer.is_fsdp_enabled = True
        return trainer
    monkeypatch.setitem(globals(), "_trainer", patched_trainer)
    _, _, calls, trainer = _run(monkeypatch, _ADV, old = old)
    assert calls == [6]
    assert not hasattr(trainer, "_unsloth_zero_advantage_rows_skipped")
