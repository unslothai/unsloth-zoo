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


# ---------------------------------------------------------------------------
# Unsloth_Offloaded_Log_Softmax backward returns LOCAL input gradients
# ---------------------------------------------------------------------------
# backward must return LOCAL grads via autograd.grad; backward + leaf .grad
# double-counts through the outer AccumulateGrad when lm_head is trainable.


def test_offloaded_log_softmax_uses_autograd_grad_not_backward():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "class Unsloth_Offloaded_Log_Softmax" in src
    assert "torch.autograd.grad(" in src
    assert "torch.autograd.backward(output, grad_output)" not in src
    assert "lm_head.grad if ctx.lm_head_requires_grad else None" not in src


def _recompute_fn(use_backward):
    # Recompute-in-backward mirror; use_backward=True is the buggy variant, False the fix.
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


# ---------------------------------------------------------------------------
# Unsloth_Offloaded_Log_Softmax offload paths (pinned / keep-on-GPU / CPU)
# ---------------------------------------------------------------------------
# Pins CPU-fallback numerics and CUDA-path source invariants; backward's event
# wait is load-bearing (pinned non_blocking copy races without it).


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


def _extract_offloaded_log_softmax(inner_fn):
    # Exec the real Unsloth_Offloaded_Log_Softmax block against `inner_fn`.
    import textwrap
    src = inspect.getsource(rr.grpo_accumulated_loss)
    block = textwrap.dedent(src[src.index("    def to_device"):src.index("    def efficient_log_softmax")])
    ns = {"torch": torch, "chunked_hidden_states_selective_log_softmax": inner_fn}
    exec(block, ns)
    return ns["Unsloth_Offloaded_Log_Softmax"]


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
    # Pinned D2H must record an event, keep the source alive, and backward must wait on it.
    assert "pin_memory = True" in fwd
    assert "copy_event.record(copy_stream)" in fwd
    assert "record_stream(copy_stream)" in fwd
    assert "ctx.copy_event.wait(" in fwd
    # CUDA machinery stays behind is_cuda; CPU-only platforms take the pageable path.
    assert "is_cuda" in fwd
    assert 'saved_hidden_states = detached_hidden_states.to("cpu", non_blocking = True)' in fwd


def test_offloaded_log_softmax_keep_on_gpu_budget_is_cumulative():
    # The padded GRPO loop runs N forwards before any backward; a per-chunk
    # 4x-free check alone compounds retained chunks toward all free memory.
    src = inspect.getsource(rr.grpo_accumulated_loss)
    block = src[src.index("    def to_device"):src.index("    def efficient_log_softmax")]
    assert "offload_retained_bytes = [0]" in block
    assert "4 * (tensor_bytes + offload_retained_bytes[0]) <= free_bytes" in block
    assert "offload_retained_bytes[0] += tensor_bytes" in block
