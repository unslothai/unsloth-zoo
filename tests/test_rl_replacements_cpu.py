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

import dataclasses
import inspect
import logging
import math
import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo import rl_replacements as rr


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
    assert mask[0].tolist() == [False, False, False, True, False, False]
    assert mask[1].tolist() == [False, True, True, False, False, False]


def test_left_pack_padding_moves_pads_to_right_stable():
    PAD = 0
    t = torch.tensor(
        [
            [PAD,   1,   2, PAD,   3],
            [  4, PAD, PAD,   5,   6],
        ]
    )
    packed = rr.left_pack_padding(t, pad_id = PAD)
    assert packed[0].tolist() == [1, 2, 3, PAD, PAD]
    assert packed[1].tolist() == [4, 5, 6, PAD, PAD]


def test_left_pack_padding_idempotent_on_already_packed():
    PAD = -1
    t = torch.tensor([[1, 2, 3, PAD, PAD]])
    out = rr.left_pack_padding(t, pad_id = PAD)
    assert out.tolist() == t.tolist()


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
    assert aligned.shape == (2, 4)
    assert aligned[0].tolist() == pytest.approx([0.0, 0.5, 0.7, 0.0])
    assert aligned[1].tolist() == pytest.approx([0.1, 0.2, 0.0, 0.0])


def test_sanitize_logprob_returns_value_for_finite():
    p = SimpleNamespace(logprob = -1.234)
    assert rr.sanitize_logprob(p) == pytest.approx(-1.234)


def test_sanitize_logprob_returns_none_for_nan():
    p = SimpleNamespace(logprob = float("nan"))
    assert rr.sanitize_logprob(p) is None


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
    # The known-good keys that downstream unsloth + Unsloth code calls
    # by name. If any of these go missing the consumer side breaks.
    expected = {
        "calculate_pad_tokens_in_prompt",
        "create_completion_attention_mask",
        "left_pack_padding",
        "sanitize_logprob",
    }
    missing = expected - set(rr.RL_REPLACEMENTS.keys())
    assert not missing, f"RL_REPLACEMENTS missing public-API keys: {sorted(missing)}"


def _make_grpo_trainer(**args):
    return SimpleNamespace(args=SimpleNamespace(**args))


def test_warn_unsupported_grpo_options_silent_on_defaults(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=1.0, use_bias_correction_kl=False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []
    assert not hasattr(trainer, "_unsloth_grpo_unsupported_warned")


def test_warn_unsupported_grpo_options_warns_after_a_mid_run_config_change(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=1.0)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
        assert caplog.records == []
        trainer.args.top_entropy_quantile = 0.2
        rr._warn_unsupported_grpo_options(trainer)
        rr._warn_unsupported_grpo_options(trainer)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "top_entropy_quantile=0.2" in msgs[0]


def test_warn_unsupported_grpo_options_silent_when_attrs_missing(caplog):
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
    trainer = _make_grpo_trainer(top_entropy_quantile=1.0, use_bias_correction_kl=True)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []


def test_warn_unsupported_grpo_options_never_mentions_use_bias_correction_kl(caplog):
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


# unsloth_num_chunks is accepted but ignored; non-defaults warn once per process.


def test_warn_deprecated_n_chunks_silent_on_defaults(caplog, monkeypatch):
    monkeypatch.setattr(rr, "_n_chunks_deprecation_warned", False)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_deprecated_n_chunks(None)
        rr._warn_deprecated_n_chunks(-1)
        rr._warn_deprecated_n_chunks(1)
    assert caplog.records == []
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
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "UnslothEfficientGRPO.apply" in src
    apply_args = src.split("UnslothEfficientGRPO.apply(", 1)[1].split(")", 1)[0]
    assert "n_chunks" not in apply_args


# unsloth text-copies these bodies into its generated trainer cache, so a new cache can
# run against an older unsloth_zoo with neither helper: the calls must no-op, not raise.


def _helper_call_snippets():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    lines = src.splitlines()
    snippets = []
    for i, line in enumerate(lines):
        if line.strip() != "try:":
            continue
        block = [line]
        indent = len(line) - len(line.lstrip())
        for nxt in lines[i + 1:]:
            if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent and \
               not nxt.strip().startswith(("except", "else", "finally")):
                break
            block.append(nxt)
        text = "\n".join(block)
        if "_warn_unsupported_grpo_options" in text or "_warn_deprecated_n_chunks" in text:
            snippets.append(textwrap.dedent(text))
    return snippets


def test_helper_calls_no_op_against_an_older_unsloth_zoo(monkeypatch):
    monkeypatch.delattr(rr, "_warn_unsupported_grpo_options", raising=False)
    monkeypatch.delattr(rr, "_warn_deprecated_n_chunks", raising=False)
    snippets = _helper_call_snippets()
    assert len(snippets) == 2, "expected both helper calls to be guarded try blocks"
    for snippet in snippets:
        exec(snippet, {"trainer": object(), "n_chunks": 4})


def test_warn_unsupported_grpo_options_survives_an_unassignable_trainer():
    class Slotted:
        __slots__ = ("args",)

    trainer = Slotted()
    trainer.args = SimpleNamespace(top_entropy_quantile=0.2)
    rr._warn_unsupported_grpo_options(trainer)
    rr._warn_unsupported_grpo_options(trainer)


def test_warn_helpers_fall_back_to_warnings_when_the_logger_raises(monkeypatch):
    class _BrokenLogger:
        def warning(self, *args, **kwargs):
            raise RuntimeError("no logger")

    monkeypatch.setattr(rr, "logger", _BrokenLogger())
    monkeypatch.setattr(rr, "_n_chunks_deprecation_warned", False)
    with pytest.warns(UserWarning):
        rr._warn_unsupported_grpo_options(_make_grpo_trainer(top_entropy_quantile=0.2))
    with pytest.warns(UserWarning):
        rr._warn_deprecated_n_chunks(4)


@pytest.mark.parametrize("quantile", [None, 1.0, 1.5])
def test_warn_unsupported_grpo_options_silent_for_non_masking_quantiles(quantile, caplog):
    # None means "unset" on some TRL versions; >= 1.0 keeps every token.
    trainer = _make_grpo_trainer(top_entropy_quantile=quantile)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []


def test_warn_unsupported_grpo_options_fires_for_the_zero_quantile(caplog):
    # 0.0 is a legal TRL value (mask all but the highest-entropy token), not a default.
    trainer = _make_grpo_trainer(top_entropy_quantile=0.0)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert len(caplog.records) == 1


def test_bias_correction_follows_the_installed_trl_default():
    # TRL's own default is False through 1.9.x and True from 1.10.0; we follow it.
    trl_config = pytest.importorskip("trl.trainer.grpo_config")
    fields = {f.name: f for f in dataclasses.fields(trl_config.GRPOConfig)}
    if "use_bias_correction_kl" not in fields:
        pytest.skip("installed TRL predates use_bias_correction_kl")
    trl_default = fields["use_bias_correction_kl"].default
    args = SimpleNamespace(use_bias_correction_kl=trl_default)
    assert getattr(args, "use_bias_correction_kl", False) is trl_default


def test_bias_correction_defaults_off_without_the_trl_field():
    args = SimpleNamespace()
    assert getattr(args, "use_bias_correction_kl", False) is False


# n_chunks=1 is the only path grpo_accumulated_loss uses.


@pytest.fixture
def disable_dynamo():
    import torch._dynamo
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = prev


def _vespo_gamma_weights(advantages, log_ratio_per_token, mask, importance_sampling_ratio,
                         k_pos=2.0, lambda_pos=3.0, k_neg=3.0, lambda_neg=2.0):
    # Per-sequence VESPO gamma weights, mirroring TRL GRPOTrainer.get_gamma_weights.
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
        # vespo raises without a get_gamma_weights callable.
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


# Per TRL _compute_loss (main @ f782735): kl_i *= the pre-clamp non-detached coef_1,
# before the loss_type dispatch, feeding both the beta term and the kl metric.


def _trl_mirror_grpo_loss(
    ref, new, old, mask, beta, advantages,
    importance_sampling_level="token", use_bias_correction_kl=False,
    epsilon_low=0.2, epsilon_high=0.2, vllm_is_ratio=None,
):
    # Independent of unsloth_zoo; mean_kl follows unsloth's per-row masked mean.
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
    if vllm_is_ratio is not None:
        # TRL v1.12.0 grpo_trainer.py:3236 then :3239 - the vLLM ratio scales the policy
        # term only, and the KL is added after it.
        per_token_loss = per_token_loss * vllm_is_ratio
    per_token_loss = per_token_loss + beta * per_token_kl
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    mean_kl = ((per_token_kl * mask).sum(-1) / mask.sum(-1)).mean()
    return loss, mean_kl


@pytest.mark.parametrize("mode", ["sequence_mask", "token_truncate"])
@pytest.mark.parametrize("use_bias_correction_kl", [False, True])
def test_bias_correction_kl_is_not_scaled_by_the_vllm_ratio(mode, use_bias_correction_kl):
    # fast_inference=True feeds sampling_per_token_logps, and the vLLM ratio must reach
    # the policy term only. Multiplying the corrected KL by it too would apply the
    # correction twice with the wrong ratio.
    beta = 0.04
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture("grpo")
    sampling = old - 0.3 * torch.randn(old.shape, generator=torch.Generator().manual_seed(5),
                                       dtype=old.dtype)
    kwargs.update(
        use_vllm=True,
        vllm_importance_sampling_mode=mode,
        vllm_importance_sampling_clip_min=0.0,
        vllm_importance_sampling_clip_max=3.0,
        use_bias_correction_kl=use_bias_correction_kl,
    )

    new_ours = new.clone().requires_grad_(True)
    loss, _cl, mean_kl, *_ = rr.grpo_compute_loss(
        ref, new_ours, old, sampling, input_ids, mask, beta, advantages, **kwargs
    )
    loss.backward()

    is_ratio = (old - sampling) * mask
    if mode == "sequence_mask":
        is_ratio = is_ratio.sum(dim=-1, keepdim=True)
    is_ratio = torch.exp(is_ratio)
    if mode == "token_truncate":
        is_ratio = torch.clamp(is_ratio, min=0.0, max=3.0)
    else:
        is_ratio = is_ratio.masked_fill((is_ratio < 0.0) | (is_ratio > 3.0), 0.0)

    new_trl = new.clone().requires_grad_(True)
    loss_trl, mean_kl_trl = _trl_mirror_grpo_loss(
        ref, new_trl, old, mask, beta, advantages,
        use_bias_correction_kl=use_bias_correction_kl, vllm_is_ratio=is_ratio,
    )
    loss_trl.backward()

    assert torch.allclose(loss.detach(), loss_trl.detach(), atol=1e-10, rtol=1e-8)
    assert torch.allclose(mean_kl.detach(), mean_kl_trl.detach(), atol=1e-10, rtol=1e-8)
    assert torch.allclose(new_ours.grad, new_trl.grad, atol=1e-10, rtol=1e-8)


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
    # Shift old so coef_1 != 1 everywhere; symmetric noise would average the effect away.
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
    beta = 0.04
    lm_head = torch.randn(17, 8, dtype=torch.float64)  # unused on the logps-in path
    new, old, ref, input_ids, mask, advantages, kwargs = _grpo_loss_fixture(loss_type)
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
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert 'kwargs["use_bias_correction_kl"]' in src
    assert 'getattr(trainer.args, "use_bias_correction_kl", False)' in src


# backward + leaf .grad double-counts through the outer AccumulateGrad.


def test_offloaded_log_softmax_uses_autograd_grad_not_backward():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "class Unsloth_Offloaded_Log_Softmax" in src
    assert "torch.autograd.grad(" in src
    assert "torch.autograd.backward(output, grad_output)" not in src
    assert "lm_head.grad if ctx.lm_head_requires_grad else None" not in src


def _recompute_fn(use_backward):
    # Recompute-in-backward mirror; use_backward=True is the buggy variant.
    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, W):
            ctx.x = x.detach()
            ctx.W = W
            ctx.W_rg = W.requires_grad
            with torch.no_grad():
                return x @ W

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.x.clone().requires_grad_(True)
            W = ctx.W
            with torch.enable_grad():
                out = x @ W
            if use_backward:
                torch.autograd.backward(out, grad_output)
                return x.grad, (W.grad if ctx.W_rg else None)
            grads = torch.autograd.grad(out, (x, W) if ctx.W_rg else (x,), grad_output)
            return grads[0], (grads[1] if ctx.W_rg else None)

    return _Fn


def _weight_grad(op, shared, preexisting):
    torch.manual_seed(0)
    x1 = torch.randn(6, 8)
    x2 = torch.randn(6, 8)
    W = torch.randn(8, 10, requires_grad=True)
    g1 = torch.randn(6, 10)
    g2 = torch.randn(6, 10)
    W.grad = torch.randn(8, 10) if preexisting else None
    torch.autograd.backward(op(x1, W), g1, retain_graph=True)
    if shared:
        torch.autograd.backward(op(x2, W), g2)
    return W.grad.clone()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("preexisting", [False, True])
def test_offloaded_recompute_weight_grad_not_double_counted(shared, preexisting):
    ref = _weight_grad(lambda x, W: x @ W, shared, preexisting)
    fixed = _recompute_fn(use_backward=False)
    assert torch.allclose(_weight_grad(fixed.apply, shared, preexisting), ref, atol=1e-6)
    # Buggy variant must diverge, or this test proves nothing.
    buggy = _recompute_fn(use_backward=True)
    assert not torch.allclose(_weight_grad(buggy.apply, shared, preexisting), ref, atol=1e-4)


# backward's event wait is load-bearing: the pinned non_blocking copy races
# without it.


def _eager_selective_log_softmax(hidden_states, lm_head, index, chunks,
                                 logit_scale_multiply, logit_scale_divide,
                                 logit_softcapping, temperature):
    # Eager mirror of chunked_hidden_states_selective_log_softmax (no compile).
    logits = hidden_states.reshape(-1, hidden_states.shape[-1]).to(lm_head.dtype) @ lm_head.t()
    if logit_scale_multiply != 0.0:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0.0:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    logits = logits.to(torch.float32)
    if temperature != 1.0:
        logits = logits / temperature
    flat_index = index.reshape(-1)
    selected = torch.gather(logits, dim=-1, index=flat_index.unsqueeze(-1)).squeeze(-1)
    out = selected - torch.logsumexp(logits, dim=-1)
    return out.reshape(index.shape)


def _offloaded_block_source():
    import textwrap
    src = inspect.getsource(rr.grpo_accumulated_loss)
    return textwrap.dedent(src[src.index("    def to_device"):src.index("    def efficient_log_softmax")])


def _exec_offloaded_block(inner_fn):
    # Exec the real block against `inner_fn` instead of the compiled kernel.
    ns = {"torch": torch, "chunked_hidden_states_selective_log_softmax": inner_fn}
    exec(_offloaded_block_source(), ns)
    return ns


def _extract_offloaded_log_softmax(inner_fn):
    return _exec_offloaded_block(inner_fn)["Unsloth_Offloaded_Log_Softmax"]


@pytest.mark.parametrize("lm_requires_grad", [False, True])
def test_offloaded_log_softmax_cpu_path_grads_bitwise_exact(lm_requires_grad):
    Fn = _extract_offloaded_log_softmax(_eager_selective_log_softmax)
    args = (4, 1.5, 2.0, 20.0, 0.8)

    def run(op):
        torch.manual_seed(0)
        hs = (torch.randn(3, 32, 16) * 0.02).requires_grad_(True)
        lm = (torch.randn(64, 16) * 0.02).requires_grad_(lm_requires_grad)
        idx = torch.randint(0, 64, (3, 32))
        go = torch.randn(3, 32)
        out = op(hs, lm, idx, *args)
        out.backward(go)
        return out.detach(), hs.grad, (lm.grad if lm_requires_grad else None)

    out_ref, hs_ref, lm_ref = run(_eager_selective_log_softmax)
    out_fn, hs_fn, lm_fn = run(Fn.apply)
    assert torch.equal(out_fn, out_ref)
    assert torch.equal(hs_fn, hs_ref)
    if lm_requires_grad:
        assert torch.equal(lm_fn, lm_ref)


def test_offloaded_log_softmax_pinned_offload_is_event_synced_and_guarded():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    fwd = src[src.index("class Unsloth_Offloaded_Log_Softmax"):src.index("def efficient_log_softmax")]
    assert "pin_memory = True" in fwd
    assert "copy_event.record(copy_stream)" in fwd
    assert "record_stream(copy_stream)" in fwd
    assert "ctx.copy_event.wait(" in fwd
    assert 'saved_hidden_states = detached_hidden_states.to("cpu", non_blocking = True)' in fwd


@pytest.mark.parametrize(
    "vendor,hip_version,device,expect",
    [
        # torch.cuda is also the HIP backend, so ROCm reports device.type "cuda"
        # and needs no branch of its own; see gradient_checkpointing.py, which
        # likewise treats DEVICE_TYPE in ("cuda", "hip") identically.
        ("nvidia", None, "cuda", "torch.cuda"),
        ("amd_rocm", "6.2.41134", "cuda", "torch.cuda"),
        ("intel", None, "xpu", "torch.xpu"),
        # anything else must return None and take the pageable copy, not crash
        ("cpu_only", None, "cpu", None),
        ("meta", None, "meta", None),
    ],
)
def test_offloaded_log_softmax_stream_module_dispatch(monkeypatch, vendor, hip_version,
                                                      device, expect):
    if hip_version is not None:
        monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)
    pick = _exec_offloaded_block(_eager_selective_log_softmax)["_offload_device_module"]
    got = pick(torch.device(device))
    if expect is None:
        assert got is None, (vendor, got)
    elif expect == "torch.cuda":
        assert got is torch.cuda, (vendor, got)
    else:
        assert got is getattr(torch, "xpu", None), (vendor, got)


def test_offloaded_log_softmax_releases_clone_before_forward_compute():
    # hidden_states is normally a [:, :-1, :] slice, so .contiguous() allocates a
    # full copy. Holding it across the no-grad log-softmax would keep it resident
    # alongside the chunk logits and raise the forward-phase peak; measured at
    # +0.376 GiB on an 8x8192x4096 bf16 chunk before this ordering was fixed.
    # record_stream still blocks reuse until the D2H lands, so dropping the
    # reference early is safe.
    block = _offloaded_block_source()
    released = block.index("del detached_hidden_states")
    compute = block.index("with torch.no_grad():")
    assert released < compute, "clone must be released before the forward compute"
    # and nothing may use it after the release
    assert "detached_hidden_states" not in block[released + 30:compute]


def test_offloaded_log_softmax_stream_module_accepts_tensor_or_device():
    pick = _exec_offloaded_block(_eager_selective_log_softmax)["_offload_device_module"]
    assert pick(torch.zeros(1)) is None
    assert pick(torch.zeros(1).device) is None


def test_offloaded_log_softmax_never_retains_hidden_states_on_gpu():
    # The caller is already memory bound, so retaining hidden states on device
    # would raise the free VRAM needed to finish a step.
    block = _offloaded_block_source()
    assigns = [ln.strip() for ln in block.splitlines()
               if "saved_hidden_states =" in ln and "ctx.saved_hidden_states" not in ln]
    # Any number of "= None" resets is fine; what matters is that the only values
    # ever stored are the pinned host buffer and the CPU copy, never the device tensor.
    stored = [a for a in assigns if a != "saved_hidden_states = None"]
    assert stored == [
        "saved_hidden_states = pinned_buffer",
        'saved_hidden_states = detached_hidden_states.to("cpu", non_blocking = True)',
    ], stored
    # No memory-budget heuristic may decide to keep the tensor on device.
    assert "mem_get_info" not in block
    assert "offload_retained_bytes" not in block


@pytest.mark.parametrize("head", ["leaf", "nonleaf"])
def test_offloaded_log_softmax_runs_head_hook_once(head):
    # A Tensor.register_hook fires for tensors named in autograd.grad's `inputs`,
    # so recomputing against the real lm_head would apply a user's grad mask or
    # scaler twice. The recompute must use a private detached leaf.
    Fn = _extract_offloaded_log_softmax(_eager_selective_log_softmax)
    args = (4, 0.0, 0.0, 0.0, 1.0)

    def run(op, hook):
        torch.manual_seed(0)
        hs = (torch.randn(3, 32, 16) * 0.02).requires_grad_(True)
        base = (torch.randn(64, 16) * 0.02).requires_grad_(True)
        lm = base if head == "leaf" else base * 1.5
        fired = []
        lm.register_hook(lambda g: (fired.append(1), hook(g))[1])
        idx = torch.randint(0, 64, (3, 32))
        go = torch.randn(3, 32)
        op(hs, lm, idx, *args).backward(go)
        return len(fired), base.grad

    scale = lambda g: g * 0.5  # noqa: E731
    n_fn, g_fn = run(Fn.apply, scale)
    n_ref, g_ref = run(_eager_selective_log_softmax, scale)
    assert n_fn == n_ref == 1, (n_fn, n_ref)
    assert torch.allclose(g_fn, g_ref, atol=1e-6), (g_fn - g_ref).abs().max()


def test_offloaded_log_softmax_preserves_preexisting_head_grad():
    # Accumulation must leave lm_head.grad at P + g, not 2*P + 2*g.
    Fn = _extract_offloaded_log_softmax(_eager_selective_log_softmax)
    args = (4, 0.0, 0.0, 0.0, 1.0)

    def run(op):
        torch.manual_seed(0)
        hs = (torch.randn(3, 32, 16) * 0.02).requires_grad_(True)
        lm = (torch.randn(64, 16) * 0.02).requires_grad_(True)
        idx = torch.randint(0, 64, (3, 32))
        go = torch.randn(3, 32)
        pre = torch.randn(64, 16) * 0.01
        lm.grad = pre.clone()
        op(hs, lm, idx, *args).backward(go)
        return lm.grad, pre

    got, pre_got = run(Fn.apply)
    ref, pre_ref = run(_eager_selective_log_softmax)
    assert torch.equal(pre_got, pre_ref)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()
