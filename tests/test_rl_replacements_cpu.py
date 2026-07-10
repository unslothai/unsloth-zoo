# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CPU-pure unit tests for `unsloth_zoo.rl_replacements`.

The GRPO replacement helpers are normally exercised inside a torch.compile'd
GRPO step on GPU. Several are pure-torch shape ops with well-defined IO
contracts; this pins them with tiny CPU fixtures so refactors can't silently
break the contract.

Covers:
  - `calculate_pad_tokens_in_prompt` (left-pad counter)
  - `create_completion_attention_mask` (0/1 mask after slicing prompt off)
  - `left_pack_padding` (stable sort that moves pad tokens to the right)
  - `align_logprobs_with_mask` (insert per-batch left padding into logprobs)
  - `sanitize_logprob` (filter NaN logprob values from vLLM outputs)
  - `_warn_unsupported_grpo_options` (warn-once for ignored TRL GRPO options:
    top_entropy_quantile; use_bias_correction_kl is supported so it must not warn)
  - `grpo_compute_loss` with `use_bias_correction_kl=True` (KL x importance-sampling
    ratio, TRL GRPOConfig.use_bias_correction_kl) matches an inline TRL-mirror
    reference in loss, gradient and mean_kl for token and sequence IS levels
  - `_warn_deprecated_n_chunks` (warn-once that unsloth_num_chunks has no effect)
  - `UnslothEfficientGRPO` on the single-chunk path (the only path
    grpo_accumulated_loss uses) matches a naive grpo_compute_loss pass in loss
    and gradient for every loss type
  - `RL_REPLACEMENTS` dict integrity (every value is callable; the
    well-known public-API keys are populated).
"""

from __future__ import annotations

import inspect
import logging
import math
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo import rl_replacements as rr


# ---------------------------------------------------------------------------
# calculate_pad_tokens_in_prompt
# ---------------------------------------------------------------------------


def test_calculate_pad_tokens_in_prompt_counts_left_pads():
    PAD = 0
    # batch=2, seq_len=6, logits_to_keep=3 -> prompt_section is the
    # first 3 cols. Row 0 has 3 pads, row 1 has 1 pad.
    input_ids = torch.tensor(
        [
            [PAD, PAD, PAD, 7, 8, 9],
            [PAD,   1,   2, 7, 8, 9],
        ]
    )
    counts = rr.calculate_pad_tokens_in_prompt(input_ids, logits_to_keep = 3, pad_token_id = PAD)
    assert counts.tolist() == [3, 1]


def test_calculate_pad_tokens_in_prompt_rejects_invalid_keep():
    PAD = 0
    input_ids = torch.zeros((1, 4), dtype = torch.long)
    with pytest.raises(ValueError):
        rr.calculate_pad_tokens_in_prompt(input_ids, logits_to_keep = 4, pad_token_id = PAD)
    with pytest.raises(ValueError):
        rr.calculate_pad_tokens_in_prompt(input_ids, logits_to_keep = 5, pad_token_id = PAD)


# ---------------------------------------------------------------------------
# create_completion_attention_mask
# ---------------------------------------------------------------------------


def test_create_completion_attention_mask_zeros_left_prompt_and_right_pads():
    PAD = 0
    # batch=2, completion_len=6. left_pad_tokens_per_prompt says
    # row 0 had 0 left pads, row 1 had 2 left pads. max_left_pad=3
    # means we need to also zero out an extra (max - row_pad) leading
    # cols on each row.
    completion_input_ids = torch.tensor(
        [
            [10, 11, 12, 13,  PAD, PAD],
            [10, 11, 12,  PAD, PAD, PAD],
        ]
    )
    left_pad = torch.tensor([0, 2])
    mask = rr.create_completion_attention_mask(
        completion_input_ids   = completion_input_ids,
        left_pad_tokens_per_prompt = left_pad,
        max_left_pad           = 3,
        pad_token_id           = PAD,
    )
    assert mask.dtype == torch.bool
    # row 0: zero the first 3 cols (max-0), keep non-pad. shape mask = [0,0,0,1,0,0]
    assert mask[0].tolist() == [False, False, False, True, False, False]
    # row 1: zero the first 1 col (max-2), keep non-pad. shape mask = [0,1,1,0,0,0]
    assert mask[1].tolist() == [False, True, True, False, False, False]


# ---------------------------------------------------------------------------
# left_pack_padding
# ---------------------------------------------------------------------------


def test_left_pack_padding_moves_pads_to_right_stable():
    PAD = 0
    t = torch.tensor(
        [
            [PAD,   1,   2, PAD,   3],
            [  4, PAD, PAD,   5,   6],
        ]
    )
    packed = rr.left_pack_padding(t, pad_id = PAD)
    # Non-pad tokens preserve their relative order (stable sort).
    assert packed[0].tolist() == [1, 2, 3, PAD, PAD]
    assert packed[1].tolist() == [4, 5, 6, PAD, PAD]


def test_left_pack_padding_idempotent_on_already_packed():
    PAD = -1
    t = torch.tensor([[1, 2, 3, PAD, PAD]])
    out = rr.left_pack_padding(t, pad_id = PAD)
    assert out.tolist() == t.tolist()


# ---------------------------------------------------------------------------
# align_logprobs_with_mask
# ---------------------------------------------------------------------------


def test_align_logprobs_with_mask_inserts_per_row_left_padding():
    # Each row's left-pad count in attention_mask determines where
    # the row's logprob block starts in the output tensor.
    # row 0: attention_mask has 1 leading 0 then 3 ones; logprob_seq_len=2.
    # row 1: attention_mask has 0 leading 0s then 4 ones; logprob_seq_len=2.
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype = torch.long,
    )
    logprobs = torch.tensor(
        [
            [0.5, 0.7],
            [0.1, 0.2],
        ]
    )
    aligned = rr.align_logprobs_with_mask(
        logprob_tensor = logprobs,
        attention_mask = attention_mask,
        pad_value      = 0.0,
    )
    # Output shape matches attention_mask's seq_len = 4.
    assert aligned.shape == (2, 4)
    # row 0: shift by 1 left pad -> logprobs land at cols 1,2; cols 0,3 stay pad.
    # row 1: shift by 0 left pads -> logprobs land at cols 0,1; cols 2,3 stay pad.
    # `pytest.approx` for float32 round-trip tolerance.
    assert aligned[0].tolist() == pytest.approx([0.0, 0.5, 0.7, 0.0])
    assert aligned[1].tolist() == pytest.approx([0.1, 0.2, 0.0, 0.0])


# ---------------------------------------------------------------------------
# sanitize_logprob
# ---------------------------------------------------------------------------


def test_sanitize_logprob_returns_value_for_finite():
    p = SimpleNamespace(logprob = -1.234)
    assert rr.sanitize_logprob(p) == pytest.approx(-1.234)


def test_sanitize_logprob_returns_none_for_nan():
    p = SimpleNamespace(logprob = float("nan"))
    assert rr.sanitize_logprob(p) is None


# ---------------------------------------------------------------------------
# RL_REPLACEMENTS dict integrity
# ---------------------------------------------------------------------------


def test_RL_REPLACEMENTS_values_are_callables_or_source_strings():
    """`RL_REPLACEMENTS` mixes two kinds of values:

      - callables (regular Python functions) used by direct callers,
      - source strings (raw `def ...` text) that the compiler
        injects verbatim into a generated module at compile time.

    Both are valid; what's NOT valid is `None`, an int, a torch
    tensor, etc. -- any other type would mean a registration bug.
    """
    table = rr.RL_REPLACEMENTS
    assert isinstance(table, dict)
    assert len(table) >= 5, (
        f"RL_REPLACEMENTS unexpectedly small ({len(table)} entries) -- a "
        f"refactor likely dropped registrations. keys: {sorted(table)}"
    )
    for name, value in table.items():
        assert callable(value) or isinstance(value, str), (
            f"RL_REPLACEMENTS[{name!r}] has unexpected type "
            f"{type(value).__name__}: {value!r}"
        )


def test_RL_REPLACEMENTS_contains_public_api_keys():
    # The known-good keys that downstream unsloth + Studio code calls
    # by name. If any of these go missing the consumer side breaks.
    expected = {
        "calculate_pad_tokens_in_prompt",
        "create_completion_attention_mask",
        "left_pack_padding",
        "sanitize_logprob",
    }
    missing = expected - set(rr.RL_REPLACEMENTS.keys())
    assert not missing, f"RL_REPLACEMENTS missing public-API keys: {sorted(missing)}"


# ---------------------------------------------------------------------------
# _warn_unsupported_grpo_options  (Issue 1: silently-ignored GRPO config options)
# ---------------------------------------------------------------------------
# The fast GRPO path ignores top_entropy_quantile < 1.0; the helper warns once per
# trainer on non-defaults only (the TRL 1.7.1 default is 1.0, so the default must NOT
# warn). use_bias_correction_kl is implemented (grpo_compute_loss applies kl_i * coef_1)
# so setting it must never warn.


def _make_grpo_trainer(**args):
    return SimpleNamespace(args=SimpleNamespace(**args))


def test_warn_unsupported_grpo_options_silent_on_defaults(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=1.0, use_bias_correction_kl=False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []
    # Flag set even on defaults so the check runs at most once.
    assert trainer._unsloth_grpo_unsupported_warned is True


def test_warn_unsupported_grpo_options_silent_when_attrs_missing(caplog):
    # Missing attrs -> defaults assumed -> no warning.
    trainer = _make_grpo_trainer()
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []


def test_warn_unsupported_grpo_options_fires_for_top_entropy_quantile(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=0.2, use_bias_correction_kl=False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "top_entropy_quantile=0.2" in msgs[0]
    assert "use_bias_correction_kl" not in msgs[0]


def test_warn_unsupported_grpo_options_silent_for_use_bias_correction_kl(caplog):
    # use_bias_correction_kl is supported (grpo_compute_loss applies kl_i * coef_1),
    # so enabling it must NOT warn.
    trainer = _make_grpo_trainer(top_entropy_quantile=1.0, use_bias_correction_kl=True)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []


def test_warn_unsupported_grpo_options_never_mentions_use_bias_correction_kl(caplog):
    # Even when another option warns, the supported use_bias_correction_kl must not
    # appear in the message.
    trainer = _make_grpo_trainer(top_entropy_quantile=0.5, use_bias_correction_kl=True)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "top_entropy_quantile=0.5" in msgs[0]
    assert "use_bias_correction_kl" not in msgs[0]


def test_warn_unsupported_grpo_options_fires_once(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=0.2)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
        rr._warn_unsupported_grpo_options(trainer)
        rr._warn_unsupported_grpo_options(trainer)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1


def test_warn_unsupported_grpo_options_registered():
    assert rr.RL_REPLACEMENTS.get("_warn_unsupported_grpo_options") is rr._warn_unsupported_grpo_options


# ---------------------------------------------------------------------------
# _warn_deprecated_n_chunks  (unsloth_num_chunks is accepted but has no effect)
# ---------------------------------------------------------------------------
# grpo_accumulated_loss keeps n_chunks in its signature because unsloth's generated
# GRPO trainer passes unsloth_num_chunks, but loss chunking was removed: the
# UnslothEfficientGRPO call always runs the whole batch as a single chunk. Any
# non-default request (not None / -1 / 1) warns once per process.


def test_warn_deprecated_n_chunks_silent_on_defaults(caplog, monkeypatch):
    monkeypatch.setattr(rr, "_n_chunks_deprecation_warned", False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_deprecated_n_chunks(None)
        rr._warn_deprecated_n_chunks(-1)
        rr._warn_deprecated_n_chunks(1)
    assert caplog.records == []
    # Defaults must not consume the warn-once flag.
    assert rr._n_chunks_deprecation_warned is False


@pytest.mark.parametrize("n_chunks", [-2, 0, 2, 4, 1000])
def test_warn_deprecated_n_chunks_fires_for_non_default(n_chunks, caplog, monkeypatch):
    monkeypatch.setattr(rr, "_n_chunks_deprecation_warned", False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_deprecated_n_chunks(n_chunks)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "unsloth_num_chunks is deprecated" in msgs[0]


def test_warn_deprecated_n_chunks_fires_once(caplog, monkeypatch):
    monkeypatch.setattr(rr, "_n_chunks_deprecation_warned", False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_deprecated_n_chunks(4)
        rr._warn_deprecated_n_chunks(2)
        rr._warn_deprecated_n_chunks(4)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1


def test_warn_deprecated_n_chunks_registered():
    assert rr.RL_REPLACEMENTS.get("_warn_deprecated_n_chunks") is rr._warn_deprecated_n_chunks


def test_grpo_accumulated_loss_does_not_forward_n_chunks():
    # The n_chunks request must not reach UnslothEfficientGRPO.apply: the whole batch
    # runs as a single chunk regardless of the requested value, so a non-default
    # request computes the same loss as the default (pinned against the naive pass in
    # test_efficient_grpo_single_chunk_matches_naive below).
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "UnslothEfficientGRPO.apply" in src
    apply_args = src.split("UnslothEfficientGRPO.apply(", 1)[1].split(")", 1)[0]
    assert "n_chunks" not in apply_args


# ---------------------------------------------------------------------------
# UnslothEfficientGRPO single-chunk path
# ---------------------------------------------------------------------------
# grpo_accumulated_loss always invokes UnslothEfficientGRPO with a single chunk
# (the whole batch). Pin that this path is numerically identical (loss and
# gradient) to a naive grpo_compute_loss pass for every loss type.


@pytest.fixture
def disable_dynamo():
    # UnslothEfficientGRPO.forward wraps its inner step in torch.compile; disable
    # dynamo so it runs eagerly on CPU. The math is identical to the compiled path.
    import torch._dynamo
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = prev


def _vespo_gamma_weights(advantages, log_ratio_per_token, mask, importance_sampling_ratio,
                         k_pos=2.0, lambda_pos=3.0, k_neg=3.0, lambda_neg=2.0):
    # Faithful per-sequence VESPO gamma weights (mirrors TRL GRPOTrainer.get_gamma_weights):
    # phi(w) = e^lambda * w^k * e^{-lambda w} with w the sequence-level ratio. Each row depends
    # only on its own tokens (seq_log_ratio sums over that row).
    lower = math.log(1e-8)
    seq_log_ratio = (torch.clamp(log_ratio_per_token, -20.0, 20.0) * mask).sum(-1, keepdim=True)
    if importance_sampling_ratio is not None:
        seq_log_ratio = seq_log_ratio + torch.clamp(
            torch.log(importance_sampling_ratio), lower, 20.0
        ).sum(-1, keepdim=True)
    log_w = torch.clamp(seq_log_ratio, lower, 20.0)
    w = torch.exp(log_w)
    k = torch.where(advantages >= 0, k_pos, k_neg)
    lam = torch.where(advantages >= 0, lambda_pos, lambda_neg)
    return torch.exp(lam + k * log_w - lam * w).detach()


def _grpo_loss_fixture(loss_type, B=6, T=5, V=17):
    torch.manual_seed(123)
    new = torch.randn(B, T, dtype=torch.float64)
    old = new + 0.05 * torch.randn(B, T, dtype=torch.float64)
    ref = new + 0.05 * torch.randn(B, T, dtype=torch.float64)
    input_ids = torch.randint(0, V, (B, T))
    mask = (torch.rand(B, T) > 0.3).to(torch.float64)
    mask[:, 0] = 1.0  # guarantee at least one active token per row
    advantages = torch.randn(B, dtype=torch.float64)
    kwargs = dict(
        loss_type=loss_type,
        num_items_in_batch=float(mask.sum().item()),
        num_processes=1,
        current_gradient_accumulation_steps=1,
        max_completion_length=T,
    )
    if loss_type == "vespo":
        # vespo raises without a get_gamma_weights callable (TRL 0.26+ supplies it).
        kwargs["get_gamma_weights"] = _vespo_gamma_weights
    return new, old, ref, input_ids, mask, advantages, kwargs


@pytest.mark.parametrize(
    "loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"]
)
def test_efficient_grpo_single_chunk_matches_naive(loss_type, disable_dynamo):
    beta = 0.04
    lm_head = torch.randn(17, 8, dtype=torch.float64)  # unused on the logps-in path
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture(loss_type)

    new_ref = new.clone().requires_grad_(True)
    loss_ref = rr.grpo_compute_loss(
        ref, new_ref, old, None, input_ids, mask, beta, advantages, **kwargs
    )[0]
    loss_ref.backward()

    new_eff = new.clone().requires_grad_(True)
    out = rr.UnslothEfficientGRPO.apply(
        new_eff, old, ref, None, lm_head, input_ids, mask, advantages,
        beta, None, 1, kwargs,
    )
    out[0].backward()

    assert torch.allclose(
        out[0].detach().double(), loss_ref.detach().double(), atol=1e-8, rtol=1e-6
    ), f"{loss_type}: loss mismatch"
    assert torch.allclose(
        new_eff.grad, new_ref.grad, atol=1e-8, rtol=1e-6
    ), f"{loss_type}: gradient mismatch"


# ---------------------------------------------------------------------------
# use_bias_correction_kl  (TRL GRPOConfig.use_bias_correction_kl, DeepSeek-V3.2)
# ---------------------------------------------------------------------------
# TRL grpo_trainer._compute_loss (main @ f782735, identical in 0.27.0):
#
#     coef_1 = torch.exp(log_importance_weights)
#     if self.beta != 0.0:
#         per_token_kl = (
#             torch.exp(ref_per_token_logps - per_token_logps)
#             - (ref_per_token_logps - per_token_logps) - 1
#         )
#         # Importance sampling correction for the KL divergence
#         if self.args.use_bias_correction_kl:
#             per_token_kl = per_token_kl * coef_1
#
# The correction runs before the loss_type dispatch (so all loss types), uses the
# pre-delta-clamp NON-detached coef_1 ((B,T) token level, (B,1) sequence level
# broadcast), and the corrected KL feeds both `beta * per_token_kl` in the loss and
# the kl metric. grpo_compute_loss must reproduce that.


def _trl_mirror_grpo_loss(
    ref, new, old, mask, beta, advantages,
    importance_sampling_level="token", use_bias_correction_kl=False,
    epsilon_low=0.2, epsilon_high=0.2,
):
    # Line-by-line mirror of TRL _compute_loss for loss_type="grpo" (quoted above),
    # kept independent of unsloth_zoo internals. mean_kl uses unsloth's per-row
    # masked-mean convention so it is comparable with grpo_compute_loss's metric.
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    log_ratio = new - old
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    else:
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    coef_1 = torch.exp(log_importance_weights)
    per_token_kl = torch.exp(ref - new) - (ref - new) - 1
    if use_bias_correction_kl:
        per_token_kl = per_token_kl * coef_1
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
    per_token_loss = per_token_loss + beta * per_token_kl
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    mean_kl = ((per_token_kl * mask).sum(-1) / mask.sum(-1)).mean()
    return loss, mean_kl


@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
@pytest.mark.parametrize("use_bias_correction_kl", [False, True])
def test_grpo_compute_loss_bias_correction_kl_matches_trl_mirror(
    importance_sampling_level, use_bias_correction_kl
):
    beta = 0.04
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture("grpo")
    kwargs["importance_sampling_level"] = importance_sampling_level
    kwargs["use_bias_correction_kl"] = use_bias_correction_kl

    new_ours = new.clone().requires_grad_(True)
    loss, _completion_length, mean_kl, *_ = rr.grpo_compute_loss(
        ref, new_ours, old, None, input_ids, mask, beta, advantages, **kwargs
    )
    loss.backward()

    new_trl = new.clone().requires_grad_(True)
    loss_trl, mean_kl_trl = _trl_mirror_grpo_loss(
        ref, new_trl, old, mask, beta, advantages,
        importance_sampling_level=importance_sampling_level,
        use_bias_correction_kl=use_bias_correction_kl,
    )
    loss_trl.backward()

    assert torch.allclose(loss.detach(), loss_trl.detach(), atol=1e-10, rtol=1e-8)
    assert torch.allclose(mean_kl.detach(), mean_kl_trl.detach(), atol=1e-10, rtol=1e-8)
    # Gradient must flow through the NON-detached coef_1 in the corrected KL term.
    assert torch.allclose(new_ours.grad, new_trl.grad, atol=1e-10, rtol=1e-8)


def test_grpo_compute_loss_bias_correction_kl_changes_loss_and_mean_kl():
    # Guard against the kwarg being silently dropped: with beta != 0 and old != new,
    # enabling the correction must change both the loss and the kl metric. Shift old
    # by a constant so coef_1 ~= e^0.5 everywhere: with the fixture's symmetric noise
    # alone, E[kl_i * (coef_1 - 1)] ~= 0 and the loss shift can fall inside allclose's
    # default tolerance.
    beta = 0.04
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture("grpo")
    old = old - 0.5
    loss_off, _, mean_kl_off, *_ = rr.grpo_compute_loss(
        ref, new, old, None, input_ids, mask, beta, advantages, **kwargs
    )
    loss_on, _, mean_kl_on, *_ = rr.grpo_compute_loss(
        ref, new, old, None, input_ids, mask, beta, advantages,
        use_bias_correction_kl=True, **kwargs
    )
    assert not torch.allclose(loss_on, loss_off)
    assert not torch.allclose(mean_kl_on, mean_kl_off)


def test_grpo_compute_loss_bias_correction_kl_defaults_off():
    # Omitting the kwarg must behave exactly like use_bias_correction_kl=False
    # (the TRL GRPOConfig default).
    beta = 0.04
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture("grpo")
    loss_default, _, kl_default, *_ = rr.grpo_compute_loss(
        ref, new, old, None, input_ids, mask, beta, advantages, **kwargs
    )
    loss_off, _, kl_off, *_ = rr.grpo_compute_loss(
        ref, new, old, None, input_ids, mask, beta, advantages,
        use_bias_correction_kl=False, **kwargs
    )
    assert torch.equal(loss_default, loss_off)
    assert torch.equal(kl_default, kl_off)


def test_grpo_compute_loss_bias_correction_kl_noop_when_beta_zero():
    # TRL only computes (and corrects) the KL term when beta != 0; with beta == 0 the
    # flag must have no effect on the loss.
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture("grpo")
    loss_off, *_ = rr.grpo_compute_loss(
        ref, new, old, None, input_ids, mask, 0.0, advantages, **kwargs
    )
    loss_on, *_ = rr.grpo_compute_loss(
        ref, new, old, None, input_ids, mask, 0.0, advantages,
        use_bias_correction_kl=True, **kwargs
    )
    assert torch.equal(loss_on, loss_off)


@pytest.mark.parametrize(
    "loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"]
)
def test_efficient_grpo_forwards_use_bias_correction_kl(loss_type, disable_dynamo):
    # UnslothEfficientGRPO passes extra_kwargs through to grpo_compute_loss, so the
    # single-chunk path with the flag on must match the naive corrected pass (loss and
    # gradient) for every loss type -- TRL applies the correction before the loss_type
    # dispatch, so it is loss_type independent.
    beta = 0.04
    lm_head = torch.randn(17, 8, dtype=torch.float64)  # unused on the logps-in path
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture(loss_type)
    # Constant shift -> coef_1 ~= e^0.5 everywhere, so the corrected KL term moves the
    # loss well past allclose tolerance (the fixture's symmetric noise alone averages
    # the correction to ~0).
    old = old - 0.5
    kwargs["use_bias_correction_kl"] = True

    new_ref = new.clone().requires_grad_(True)
    loss_ref = rr.grpo_compute_loss(
        ref, new_ref, old, None, input_ids, mask, beta, advantages, **kwargs
    )[0]
    loss_ref.backward()

    new_eff = new.clone().requires_grad_(True)
    out = rr.UnslothEfficientGRPO.apply(
        new_eff, old, ref, None, lm_head, input_ids, mask, advantages,
        beta, None, 1, kwargs,
    )
    out[0].backward()

    assert torch.allclose(
        out[0].detach().double(), loss_ref.detach().double(), atol=1e-8, rtol=1e-6
    ), f"{loss_type}: loss mismatch"
    assert torch.allclose(
        new_eff.grad, new_ref.grad, atol=1e-8, rtol=1e-6
    ), f"{loss_type}: gradient mismatch"

    # And the flag must actually bite: with beta != 0 and old != new, the same
    # efficient path with the flag off must produce a different (uncorrected) loss.
    kwargs_off = dict(kwargs)
    kwargs_off["use_bias_correction_kl"] = False
    out_off = rr.UnslothEfficientGRPO.apply(
        new.clone().requires_grad_(True), old, ref, None, lm_head, input_ids, mask,
        advantages, beta, None, 1, kwargs_off,
    )
    assert not torch.allclose(out[0].detach(), out_off[0].detach()), (
        f"{loss_type}: use_bias_correction_kl had no effect"
    )


def test_grpo_accumulated_loss_forwards_use_bias_correction_kl():
    # grpo_accumulated_loss must read the flag off trainer.args (same pattern as the
    # other TRL config passthroughs) so it reaches grpo_compute_loss via extra_kwargs;
    # getattr default False keeps older TRL (no such field) on the uncorrected path.
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert 'kwargs["use_bias_correction_kl"]' in src
    assert 'getattr(trainer.args, "use_bias_correction_kl", False)' in src
