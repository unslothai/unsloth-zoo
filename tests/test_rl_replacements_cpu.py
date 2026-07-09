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
  - `RL_REPLACEMENTS` dict integrity (every value is callable; the
    well-known public-API keys are populated).
"""

from __future__ import annotations

import inspect
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


# get_off_policy_mask adapter: TRL renamed its 3rd parameter old_per_token_logps ->
# sampling_per_token_logps in 0.27.1 (huggingface/trl#4857), so the old hardcoded keyword
# crashed on TRL >= 0.27.1. grpo_compute_loss now passes a version-stable keyword that
# grpo_accumulated_loss adapts to the installed parameter name.


def test_grpo_compute_loss_uses_version_stable_off_policy_keyword():
    src = inspect.getsource(rr.grpo_compute_loss)
    flat = src.replace(" ", "")
    # The off-policy branch calls get_off_policy_mask with the mismatch logprobs.
    assert "get_off_policy_mask(" in src
    assert "sampling_per_token_logps=sampling_per_token_logps" in flat, (
        "grpo_compute_loss must pass the version-stable keyword; the adapter in "
        "grpo_accumulated_loss maps it to the installed TRL parameter name."
    )
    # The removed TRL 0.27.0 keyword must NOT be passed here -- it is the crash source
    # on TRL >= 0.27.1 and branching on it would live inside the compiled function.
    assert "old_per_token_logps=" not in src, (
        "grpo_compute_loss must not hardcode the old_per_token_logps= keyword "
        "(TypeError on TRL >= 0.27.1)."
    )


def test_grpo_accumulated_loss_adapts_both_trl_off_policy_signatures():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    flat = src.replace(" ", "")
    # Version detection happens here (eager, outside torch.compile), not in the loss.
    assert "signature(" in src and "get_off_policy_mask" in src
    # The adapter forwards the stable arg to BOTH the TRL 0.27.0 parameter name and
    # the TRL >= 0.27.1 name, so it works across the huggingface/trl#4857 rename.
    assert "old_per_token_logps=sampling_per_token_logps" in flat
    assert "sampling_per_token_logps=sampling_per_token_logps" in flat


def test_installed_trl_get_off_policy_mask_needs_signature_adapter():
    """Behavioral pin against the actually-installed TRL: calling
    get_off_policy_mask with the wrong (renamed-away) keyword raises TypeError,
    while the installed name returns the (B, 1) Keep/Drop mask. This is the exact
    failure the adapter routes around."""
    pytest.importorskip("trl")
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
    except Exception:  # pragma: no cover - environment dependent
        pytest.skip("trl GRPOTrainer not importable")
    fn = getattr(GRPOTrainer, "get_off_policy_mask", None)
    if fn is None:
        pytest.skip("TRL < 0.27.0 has no get_off_policy_mask")

    params = list(inspect.signature(fn).parameters)
    # (advantages, per_token_logps, <mismatch logprobs>, mask, off_policy_threshold)
    mismatch_kw = params[2]
    assert mismatch_kw in ("old_per_token_logps", "sampling_per_token_logps")
    wrong_kw = (
        "sampling_per_token_logps"
        if mismatch_kw == "old_per_token_logps"
        else "old_per_token_logps"
    )

    B, T = 4, 5
    common = dict(
        advantages=torch.randn(B, 1),
        per_token_logps=torch.randn(B, T),
        mask=torch.ones(B, T),
        off_policy_threshold=0.5,
    )
    mismatch = torch.randn(B, T)

    # Hardcoding the wrong keyword is the bug.
    with pytest.raises(TypeError):
        fn(**common, **{wrong_kw: mismatch})

    # Routing to the installed name (what the adapter does) works.
    out = fn(**common, **{mismatch_kw: mismatch})
    assert out.shape == (B, 1)
    assert set(out.unique().tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# off-policy mask None fallback (num_iterations == 1, no vLLM)
# ---------------------------------------------------------------------------
#
# TRL defaults old_per_token_logps to per_token_logps.detach() when it is absent
# (num_iterations == 1), then feeds that as the mismatch logprobs, so
# get_off_policy_mask never receives None. grpo_compute_loss must do the same:
# when both sampling_per_token_logps and old are None, fall back to new.detach().
# get_off_policy_mask computes `mismatch - per_token_logps.detach()`, so None would
# crash; new.detach() gives a zero-KL keep-all mask.


def _trl_faithful_off_policy_mask(
    advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold
):
    # Byte-for-byte TRL GRPOTrainer.get_off_policy_mask: crashes on a None mismatch.
    kl_div = sampling_per_token_logps - per_token_logps.detach()
    seq_kl_sum = (kl_div * mask).sum(dim=1, keepdim=True)
    avg_seq_kl = seq_kl_sum / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    is_pos_adv = advantages >= 0
    is_low_kl = avg_seq_kl <= off_policy_threshold
    return (is_pos_adv | is_low_kl).to(dtype=mask.dtype)


def test_grpo_compute_loss_off_policy_mask_falls_back_to_new_when_old_absent():
    B, T, V = 4, 5, 11
    torch.manual_seed(0)
    new = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.float64)
    advantages = torch.randn(B, dtype=torch.float64)

    # num_iterations == 1 with no vLLM: old and sampling are both None.
    loss, *_ = rr.grpo_compute_loss(
        None, new, None, None, input_ids, mask, 0.0, advantages,
        loss_type="grpo",
        num_items_in_batch=float(mask.sum().item()),
        num_processes=1,
        current_gradient_accumulation_steps=1,
        max_completion_length=T,
        get_off_policy_mask=_trl_faithful_off_policy_mask,
        off_policy_mask_threshold=0.5,
    )
    # No TypeError from a None mismatch, and a real finite loss came out.
    assert torch.isfinite(loss)


def test_grpo_compute_loss_source_has_new_detach_off_policy_fallback():
    flat = inspect.getsource(rr.grpo_compute_loss).replace(" ", "")
    # The mismatch logprobs must fall back to new.detach() when old is also None,
    # matching TRL's old_per_token_logps = per_token_logps.detach() default.
    assert "new.detach()" in flat
    assert "oldifoldisnotNoneelsenew.detach()" in flat


# ---------------------------------------------------------------------------
# off-policy mask uses vLLM sampling logps independently of IS correction
# ---------------------------------------------------------------------------
#
# TRL feeds the vLLM sampling logprobs to get_off_policy_mask whenever the batch
# has them, regardless of vllm_importance_sampling_correction (which only gates the
# IS ratio applied to the loss). grpo_compute_loss must do the same: the off-policy
# mask sees sampling_per_token_logps even with correction off, while the IS ratio
# stays gated on the flag.


def _grpo_vllm_kwargs(mask, **extra):
    base = dict(
        loss_type="grpo",
        use_vllm=True,
        num_items_in_batch=float(mask.sum().item()),
        num_processes=1,
        current_gradient_accumulation_steps=1,
        max_completion_length=mask.shape[1],
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_clip_min=0.0,
        vllm_importance_sampling_clip_max=3.0,
    )
    base.update(extra)
    return base


def test_opsm_uses_vllm_sampling_when_is_correction_off():
    B, T, V = 4, 5, 11
    torch.manual_seed(0)
    new = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    old = new.detach() + 0.1
    sampling = new.detach() + 0.3  # distinct from old and new.detach()
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.float64)
    advantages = torch.randn(B, dtype=torch.float64)

    captured = {}

    def recording_mask(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
        captured["sampling"] = sampling_per_token_logps
        return torch.ones(per_token_logps.shape[0], 1, dtype=mask.dtype)

    rr.grpo_compute_loss(
        None, new, old, sampling, input_ids, mask, 0.0, advantages,
        get_off_policy_mask=recording_mask, off_policy_mask_threshold=0.5,
        **_grpo_vllm_kwargs(mask, vllm_importance_sampling_correction=False),
    )
    # With correction off, the mask must still receive the real vLLM sampling logps.
    assert torch.equal(captured["sampling"], sampling)


def test_grpo_compute_loss_is_ratio_gated_on_correction_flag():
    B, T, V = 4, 5, 11
    torch.manual_seed(1)
    new = torch.randn(B, T, dtype=torch.float64)
    old = new + 0.1
    sampling = new + 0.3
    input_ids = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.float64)
    advantages = torch.randn(B, dtype=torch.float64)

    def loss_for(sampling_arg, correction):
        return rr.grpo_compute_loss(
            None, new.clone(), old, sampling_arg, input_ids, mask, 0.0, advantages,
            **_grpo_vllm_kwargs(mask, vllm_importance_sampling_correction=correction),
        )[0]

    # No OPSM here (off_policy_mask_threshold defaults to None), so only the IS ratio matters.
    loss_off = loss_for(sampling, False)
    loss_none = loss_for(None, False)
    loss_on = loss_for(sampling, True)
    # Correction off: IS ratio not applied, so sampling present matches sampling absent.
    assert torch.allclose(loss_off, loss_none)
    # Correction on: IS ratio applied, so the loss changes.
    assert not torch.allclose(loss_on, loss_none)


def test_grpo_compute_loss_is_ratio_blocks_gated_on_correction_flag_source():
    flat = inspect.getsource(rr.grpo_compute_loss).replace(" ", "")
    # Both IS-ratio blocks must require the correction flag; the off-policy mask call must not.
    assert flat.count("sampling_per_token_logpsisnotNoneandvllm_importance_sampling_correction") >= 2


def test_off_policy_adapter_cache_compares_bound_method_by_value():
    # trainer.get_off_policy_mask returns a fresh bound-method object every access, so the adapter
    # cache must compare `_unsloth_wrapped` by value (`!=`), not identity (`is not`) - otherwise the
    # cache never hits and a new closure is built every step, re-triggering torch.compile.
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert 'getattr(_adapter, "_unsloth_wrapped", None) != _off_policy_mask_fn' in src
    assert 'getattr(_adapter, "_unsloth_wrapped", None) is not _off_policy_mask_fn' not in src

    # Behavioral proof of the underlying semantics the fix relies on.
    class _T:
        def get_off_policy_mask(self):  # pragma: no cover - only identity/eq is exercised
            pass

    t = _T()
    assert t.get_off_policy_mask is not t.get_off_policy_mask  # fresh object each access
    assert t.get_off_policy_mask == t.get_off_policy_mask      # but equal by value
    assert (None != t.get_off_policy_mask) is True             # first-call guard still rebuilds
