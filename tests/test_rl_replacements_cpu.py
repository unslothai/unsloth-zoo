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
    top_entropy_quantile, use_bias_correction_kl)
  - `UnslothEfficientGRPO` at n_chunks=1 matches a naive grpo_compute_loss pass,
    and n_chunks>1 (equal or ragged chunks) is loss- and gradient-identical to
    n_chunks=1 for every loss type (global normalizers threaded per chunk)
  - `RL_REPLACEMENTS` dict integrity (every value is callable; the
    well-known public-API keys are populated).
"""

from __future__ import annotations

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
# The fast GRPO path ignores top_entropy_quantile < 1.0 and use_bias_correction_kl
# = True; the helper warns once per trainer on non-defaults only (TRL 1.7.1 defaults
# are 1.0 and False, so defaults must NOT warn).


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


def test_warn_unsupported_grpo_options_fires_for_use_bias_correction_kl(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=1.0, use_bias_correction_kl=True)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "use_bias_correction_kl=True" in msgs[0]
    assert "top_entropy_quantile" not in msgs[0]


def test_warn_unsupported_grpo_options_lists_both(caplog):
    trainer = _make_grpo_trainer(top_entropy_quantile=0.5, use_bias_correction_kl=True)
    with caplog.at_level(logging.WARNING, logger="unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "top_entropy_quantile=0.5" in msgs[0]
    assert "use_bias_correction_kl=True" in msgs[0]


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
# UnslothEfficientGRPO batch chunking
# ---------------------------------------------------------------------------
# UnslothEfficientGRPO threads full-batch (global) normalizers into every chunk's
# grpo_compute_loss call, so chunk losses sum to the exact full-batch loss and each
# chunk's gradient slice is already globally scaled (no division by n_chunks). These
# tests pin (1) n_chunks=1 is numerically identical (loss and gradient) to a naive
# grpo_compute_loss pass (the global kwargs default to None, keeping the single-chunk
# path bit-identical), and (2) n_chunks>1, including chunk counts that do not divide
# the batch evenly, matches n_chunks=1 for every loss type in loss AND gradient.


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
    return new, old, ref, input_ids, mask, advantages, kwargs


@pytest.mark.parametrize(
    "loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo"]
)
def test_efficient_grpo_nchunks1_matches_naive(loss_type, disable_dynamo):
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


_ALL_LOSS_TYPES = ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo"]


def _run_efficient_grpo(new, old, ref, input_ids, mask, advantages, kwargs,
                        n_chunks, beta=0.04):
    lm_head = torch.randn(17, 8, dtype=torch.float64)  # unused on the logps-in path
    ne = new.clone().requires_grad_(True)
    out = rr.UnslothEfficientGRPO.apply(
        ne, old, ref, None, lm_head, input_ids, mask, advantages,
        beta, None, n_chunks, kwargs,
    )
    out[0].backward()
    # (loss, completion_length, mean_kl, grad)
    return out[0].detach().double(), out[1].detach().double(), \
        out[2].detach().double(), ne.grad.clone()


@pytest.mark.parametrize("n_chunks", [2, 3, 4])
@pytest.mark.parametrize("loss_type", _ALL_LOSS_TYPES)
def test_efficient_grpo_nchunks_invariant(loss_type, n_chunks, disable_dynamo):
    # Core correctness proof for batch chunking: n_chunks in {2,3,4} must give the
    # same loss AND gradient (and folded metrics) as n_chunks=1 for every loss type.
    # B=6 makes n_chunks=2,3 equal-sized splits; n_chunks=4 exercises a chunk count
    # that does not divide the batch (torch.chunk(6, 4) -> 3 chunks of 2 rows).
    fixture = _grpo_loss_fixture(loss_type, B=6)
    new, old, ref, input_ids, mask, advantages, kwargs = fixture

    loss_1, clen_1, kl_1, grad_1 = _run_efficient_grpo(*fixture, n_chunks=1)
    loss_k, clen_k, kl_k, grad_k = _run_efficient_grpo(*fixture, n_chunks=n_chunks)

    # The loss accumulator buffer in UnslothEfficientGRPO is float32, so summed
    # chunk losses agree with the single-chunk loss at fp32 resolution; the fp64
    # gradient below is the strict 1e-8 invariance proof.
    assert torch.allclose(loss_k, loss_1, atol=1e-8, rtol=1e-6), \
        f"{loss_type}: loss mismatch at n_chunks={n_chunks}"
    assert torch.allclose(grad_k, grad_1, atol=1e-8, rtol=1e-8), \
        f"{loss_type}: gradient mismatch at n_chunks={n_chunks}"
    # Metrics accumulate in float32 buffers, so compare at fp32 resolution.
    assert torch.allclose(clen_k, clen_1, atol=1e-6, rtol=1e-6), \
        f"{loss_type}: completion_length mismatch at n_chunks={n_chunks}"
    assert torch.allclose(kl_k, kl_1, atol=1e-6, rtol=1e-6), \
        f"{loss_type}: mean_kl mismatch at n_chunks={n_chunks}"


@pytest.mark.parametrize("loss_type", _ALL_LOSS_TYPES)
def test_efficient_grpo_ragged_chunks_match(loss_type, disable_dynamo):
    # Unequal chunk sizes must still be exact: B=7 with n_chunks=3 splits into
    # (3, 3, 1)-row chunks. Global normalization makes ragged chunks correct too
    # (the caller only snaps to divisors to avoid torch.compile recompiles).
    fixture = _grpo_loss_fixture(loss_type, B=7)

    loss_1, clen_1, kl_1, grad_1 = _run_efficient_grpo(*fixture, n_chunks=1)
    loss_3, clen_3, kl_3, grad_3 = _run_efficient_grpo(*fixture, n_chunks=3)

    # fp32 loss accumulator: see test_efficient_grpo_nchunks_invariant.
    assert torch.allclose(loss_3, loss_1, atol=1e-8, rtol=1e-6), \
        f"{loss_type}: loss mismatch on ragged chunks"
    assert torch.allclose(grad_3, grad_1, atol=1e-8, rtol=1e-8), \
        f"{loss_type}: gradient mismatch on ragged chunks"
    # Metrics accumulate in float32 buffers, so compare at fp32 resolution.
    assert torch.allclose(clen_3, clen_1, atol=1e-6, rtol=1e-6), \
        f"{loss_type}: completion_length mismatch on ragged chunks"
    assert torch.allclose(kl_3, kl_1, atol=1e-6, rtol=1e-6), \
        f"{loss_type}: mean_kl mismatch on ragged chunks"
