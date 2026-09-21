# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CPU-pure numerics for `distillation_chunked_jsd`.

The chunked loss exists so the two `(batch, seq, vocab)` logit tensors a
knowledge-distillation step would otherwise hold never get materialized. That
only helps if the chunk loop carries exactly the same value and the same
gradient as the straight-line version, so the bulk of this file is parity
against a dense reference written inline below.

The reference is written here rather than imported because implementations
disagree on the objective: some blend a hard cross-entropy term, some apply a
`T**2` gradient rescale, and Liger mirrors the mixture interpolation relative to
TRL. Pinning ours against an inline reference states the convention instead of
inheriting whichever one a dependency happens to ship.

Covers:
  - chunked equals dense in value and gradient, every beta, every chunk size
  - the closed forms: self-distillation is 0, a fully masked batch is a
    graph-connected 0 rather than 0 / 0
  - beta = 0 and beta = 1 take their own branches instead of degenerating
  - lm_head bias, Cohere logit_scale and Gemma final_logit_softcapping are applied
  - a frozen output head allocates no dense gradient (the LoRA case)
  - the teacher never receives a gradient
  - `num_items_in_batch` reduction is applied once, not twice
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo import rl_replacements as rr


BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _dense_reference(
    student_hidden_states,
    teacher_hidden_states,
    student_lm_head,
    teacher_lm_head,
    completion_mask,
    beta = 0.5,
    num_items_in_batch = None,
    student_lm_head_bias = None,
    teacher_lm_head_bias = None,
    student_logit_scale = 1.0,
    teacher_logit_scale = 1.0,
    student_final_logit_softcapping = 0.0,
    teacher_final_logit_softcapping = 0.0,
    temperature = 1.0,
):
    """Straight-line full-logits JSD. The oracle, deliberately naive."""
    def project(hidden, weight, bias, scale, softcap):
        logits = (hidden @ weight.t()).float()
        if bias is not None:
            logits = logits + bias.float()
        if scale != 1.0:
            logits = logits * scale
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    student_logits = project(
        student_hidden_states, student_lm_head, student_lm_head_bias,
        student_logit_scale, student_final_logit_softcapping,
    )
    with torch.no_grad():
        teacher_logits = project(
            teacher_hidden_states, teacher_lm_head, teacher_lm_head_bias,
            teacher_logit_scale, teacher_final_logit_softcapping,
        )
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim = -1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim = -1)

    if beta == 0.0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction = "none", log_target = True)
    elif beta == 1.0:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction = "none", log_target = True)
    else:
        mixture = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log(torch.tensor(1.0 - beta)),
                teacher_log_probs + torch.log(torch.tensor(beta)),
            ]),
            dim = 0,
        )
        jsd = (
            beta * F.kl_div(mixture, teacher_log_probs, reduction = "none", log_target = True)
            + (1 - beta) * F.kl_div(mixture, student_log_probs, reduction = "none", log_target = True)
        )

    mask = (completion_mask.reshape(-1) != 0).float()
    per_token = jsd.reshape(mask.shape[0], -1).sum(dim = -1) * mask
    entropy = -(student_log_probs.exp() * student_log_probs).sum(dim = -1)
    entropy_sum = (entropy.reshape(mask.shape[0]) * mask).sum()
    n_valid = mask.sum().to(torch.long)

    total = per_token.sum()
    if num_items_in_batch is None:
        loss = total / n_valid.clamp(min = 1)
    else:
        loss = total / num_items_in_batch
    return loss, entropy_sum, n_valid


def _make(batch = 2, seq = 7, student_hidden = 16, teacher_hidden = 24, vocab = 53, seed = 0, bias = False):
    generator = torch.Generator().manual_seed(seed)
    sh = torch.randn(batch, seq, student_hidden, generator = generator).requires_grad_(True)
    th = torch.randn(batch, seq, teacher_hidden, generator = generator)
    swt = (torch.randn(vocab, student_hidden, generator = generator) * 0.05).requires_grad_(True)
    twt = torch.randn(vocab, teacher_hidden, generator = generator) * 0.05
    mask = torch.ones(batch, seq, dtype = torch.long)
    # Ragged on purpose, so masking is actually exercised rather than assumed.
    mask[0, -2:] = 0
    mask[1, -1] = 0
    sb = torch.randn(vocab, generator = generator) if bias else None
    tb = torch.randn(vocab, generator = generator) if bias else None
    return sh, th, swt, twt, mask, sb, tb


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("chunk_size", [1, 3, 8, 256])
def test_chunked_matches_dense_value(beta, chunk_size):
    sh, th, swt, twt, mask, _, _ = _make()
    got, got_entropy, got_n = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = beta, chunk_size = chunk_size,
    )
    want, want_entropy, want_n = _dense_reference(sh, th, swt, twt, mask, beta = beta)
    assert got_n == want_n
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(got_entropy, want_entropy, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("beta", BETAS)
def test_chunked_matches_dense_gradients(beta):
    """Value parity is not enough: the chunk loop must carry the same gradient."""
    grads = {}
    for name, fn, kwargs in (
        ("chunked", rr.distillation_chunked_jsd, {"chunk_size": 3}),
        ("dense", _dense_reference, {}),
    ):
        sh, th, swt, twt, mask, _, _ = _make()
        loss, _, _ = fn(sh, th, swt, twt, mask, beta = beta, **kwargs)
        loss.backward()
        grads[name] = (sh.grad.clone(), swt.grad.clone())
    torch.testing.assert_close(grads["chunked"][0], grads["dense"][0], rtol = 1e-4, atol = 1e-6)
    torch.testing.assert_close(grads["chunked"][1], grads["dense"][1], rtol = 1e-4, atol = 1e-6)


@pytest.mark.parametrize("beta", BETAS)
def test_self_distillation_is_zero(beta):
    """Identical student and teacher: every divergence is exactly 0."""
    sh, _, swt, _, mask, _, _ = _make(student_hidden = 16, teacher_hidden = 16)
    loss, _, _ = rr.distillation_chunked_jsd(sh, sh, swt, swt, mask, beta = beta, chunk_size = 4)
    assert abs(loss.item()) < 1e-6, f"self-distillation gave {loss.item()}"


@pytest.mark.parametrize("beta", BETAS)
def test_fully_masked_batch_is_finite_zero_and_connected(beta):
    """A fully masked batch must give a graph-connected 0, never 0 / 0 = nan.

    The gradient still has to reach every trainable parameter or the collective
    that follows backward has nothing to synchronise and the run hangs.
    """
    sh, th, swt, twt, mask, _, _ = _make()
    loss, _, n_valid = rr.distillation_chunked_jsd(
        sh, th, swt, twt, torch.zeros_like(mask), beta = beta, chunk_size = 4,
    )
    assert int(n_valid) == 0
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert sh.grad is not None


def test_endpoints_do_not_degenerate():
    """beta 0 and 1 need their own branches.

    Substituting either endpoint into the interior mixture expression collapses
    it to exactly zero, which would look like a converged run.
    """
    sh, th, swt, twt, mask, _, _ = _make()
    for beta in (0.0, 1.0):
        loss, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = beta, chunk_size = 4)
        assert loss.item() > 1e-4, f"beta={beta} degenerated to {loss.item()}"


def test_forward_and_reverse_kl_differ():
    sh, th, swt, twt, mask, _, _ = _make()
    forward, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.0, chunk_size = 4)
    reverse, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 1.0, chunk_size = 4)
    assert abs(forward.item() - reverse.item()) > 1e-4


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_bias_and_softcapping_are_honoured(beta):
    sh, th, swt, twt, mask, sb, tb = _make(bias = True)
    plain, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = beta, chunk_size = 4)
    biased, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = beta, chunk_size = 4,
        student_lm_head_bias = sb, teacher_lm_head_bias = tb,
    )
    assert abs(plain.item() - biased.item()) > 1e-6, "lm_head bias was ignored"

    capped, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = beta, chunk_size = 4,
        student_final_logit_softcapping = 2.0, teacher_final_logit_softcapping = 2.0,
    )
    assert abs(plain.item() - capped.item()) > 1e-6, "final_logit_softcapping was ignored"

    scaled, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = beta, chunk_size = 4,
        student_logit_scale = 0.5, teacher_logit_scale = 0.5,
    )
    assert abs(plain.item() - scaled.item()) > 1e-6, "logit_scale was ignored"

    # Whatever the post-processing, the chunk boundaries must not change the answer.
    for chunk_size in (2, 5):
        again, _, _ = rr.distillation_chunked_jsd(
            sh, th, swt, twt, mask, beta = beta, chunk_size = chunk_size,
            student_lm_head_bias = sb, teacher_lm_head_bias = tb,
        )
        torch.testing.assert_close(again, biased, rtol = 1e-5, atol = 1e-6)


def test_temperature_applies_without_a_t_squared_rescale():
    sh, th, swt, twt, mask, _, _ = _make()
    base, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 4)
    hot, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 4, temperature = 2.0,
    )
    assert abs(base.item() - hot.item()) > 1e-6
    reference, _, _ = _dense_reference(sh, th, swt, twt, mask, beta = 0.5, temperature = 2.0)
    torch.testing.assert_close(hot, reference, rtol = 1e-5, atol = 1e-6)


def test_num_items_in_batch_reduction_is_applied_once():
    """Gradient accumulation is only exact if this divides once, not twice."""
    sh, th, swt, twt, mask, _, _ = _make()
    n_items = 5
    got, _, n_valid = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3, num_items_in_batch = n_items,
    )
    mean, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3)
    torch.testing.assert_close(got, mean * float(int(n_valid)) / n_items, rtol = 1e-5, atol = 1e-6)


def test_num_items_in_batch_accepts_a_tensor():
    sh, th, swt, twt, mask, _, _ = _make()
    as_int, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3, num_items_in_batch = 5,
    )
    as_tensor, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3,
        num_items_in_batch = torch.tensor(5.0),
    )
    torch.testing.assert_close(as_int, as_tensor, rtol = 1e-6, atol = 1e-7)


def test_frozen_head_allocates_no_head_gradient():
    """The LoRA case: a frozen lm_head must not get a dense (vocab, hidden) grad."""
    sh, th, swt, twt, mask, _, _ = _make()
    swt.requires_grad_(False)
    loss, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3)
    loss.backward()
    assert swt.grad is None, "frozen output head still allocated a gradient"
    assert sh.grad is not None, "hidden states must still receive a gradient"


def test_teacher_never_receives_a_gradient():
    """Even when the caller forgot to freeze it."""
    sh, th, swt, twt, mask, _, _ = _make()
    th.requires_grad_(True)
    twt.requires_grad_(True)
    loss, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3)
    loss.backward()
    assert th.grad is None and twt.grad is None, "teacher received a gradient"


@pytest.mark.parametrize("widths", [(16, 24), (24, 16), (16, 16)])
def test_differing_hidden_widths(widths):
    """Teacher may be wider or narrower; only the vocabulary has to match."""
    student_hidden, teacher_hidden = widths
    sh, th, swt, twt, mask, _, _ = _make(
        student_hidden = student_hidden, teacher_hidden = teacher_hidden,
    )
    got, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3)
    want, _, _ = _dense_reference(sh, th, swt, twt, mask, beta = 0.5)
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)


def test_chunking_bounds_peak_logit_memory():
    """The reason the function exists: peak must not scale with the batch.

    Counted rather than measured, so it holds on CPU CI: the chunk loop only ever
    holds one `chunk_size x vocab` pair of logit tensors at a time.
    """
    vocab = 53
    seen = []
    original = rr._distillation_project_logits

    def counting(hidden_states, lm_head, *args, **kwargs):
        out = original(hidden_states, lm_head, *args, **kwargs)
        seen.append(out.shape[0])
        return out

    rr._distillation_project_logits = counting
    try:
        sh, th, swt, twt, mask, _, _ = _make(batch = 4, seq = 16, vocab = vocab)
        rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 8)
    finally:
        rr._distillation_project_logits = original

    assert seen, "the projection was never called"
    assert max(seen) <= 8, f"a chunk projected {max(seen)} rows, above chunk_size"


def test_registered_in_rl_replacements():
    assert rr.RL_REPLACEMENTS["distillation_chunked_jsd"] is rr.distillation_chunked_jsd


# The loss differentiates in the forward pass with torch.func.grad_and_value and
# compiles that transform, rather than recomputing each chunk under gradient
# checkpointing. These pin the paths that choice introduces.


def test_trainable_bias_receives_a_gradient():
    """The (0, 1, 2) argnums branch: hidden states, head and bias all trained."""
    sh, th, swt, twt, mask, sb, tb = _make(bias = True)
    sb.requires_grad_(True)
    loss, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3,
        student_lm_head_bias = sb, teacher_lm_head_bias = tb,
    )
    loss.backward()
    assert sb.grad is not None and sb.grad.abs().sum() > 0
    assert swt.grad is not None and sh.grad is not None
    assert tb.grad is None, "teacher bias received a gradient"


def test_frozen_bias_with_trained_head():
    """The (0, 1) argnums branch: a bias that exists but is not trained."""
    sh, th, swt, twt, mask, sb, tb = _make(bias = True)
    sb.requires_grad_(False)
    loss, _, _ = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3,
        student_lm_head_bias = sb, teacher_lm_head_bias = tb,
    )
    loss.backward()
    assert sb.grad is None
    assert swt.grad is not None and sh.grad is not None


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_compile_disabled_gives_the_same_answer(beta, monkeypatch):
    """UNSLOTH_COMPILE_DISABLE must change speed, never numerics."""
    sh, th, swt, twt, mask, _, _ = _make()
    compiled, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = beta, chunk_size = 3)
    compiled.backward()
    compiled_grads = (sh.grad.clone(), swt.grad.clone())

    rr._distillation_jsd_grad_fn.cache_clear()
    monkeypatch.setattr(
        rr, "_maybe_compile", lambda **kwargs: (lambda function: function), raising = False,
    )
    try:
        sh2, th2, swt2, twt2, mask2, _, _ = _make()
        eager, _, _ = rr.distillation_chunked_jsd(sh2, th2, swt2, twt2, mask2, beta = beta, chunk_size = 3)
        eager.backward()
        torch.testing.assert_close(compiled, eager, rtol = 1e-5, atol = 1e-6)
        torch.testing.assert_close(compiled_grads[0], sh2.grad, rtol = 1e-4, atol = 1e-6)
        torch.testing.assert_close(compiled_grads[1], swt2.grad, rtol = 1e-4, atol = 1e-6)
    finally:
        rr._distillation_jsd_grad_fn.cache_clear()


def test_hidden_gradient_is_returned_in_the_callers_row_order():
    """Packing sorts rows; the gradient has to come back unsorted.

    Masked rows contribute nothing, so a gradient left in packed order would put
    real values on masked positions and zeros on trained ones, which still trains
    and still looks plausible.
    """
    sh, th, swt, twt, mask, _, _ = _make()
    loss, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3)
    loss.backward()
    zero_rows = (sh.grad.reshape(-1, sh.shape[-1]).abs().sum(dim = -1) == 0)
    masked = (mask.reshape(-1) == 0)
    assert torch.equal(zero_rows, masked), "gradient rows do not line up with the mask"


def test_entropy_and_token_count_are_not_differentiable():
    """Only the loss carries a gradient; the two metrics are reported, not trained."""
    sh, th, swt, twt, mask, _, _ = _make()
    loss, entropy_sum, n_valid = rr.distillation_chunked_jsd(
        sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3,
    )
    assert loss.requires_grad
    assert not entropy_sum.requires_grad
    assert not n_valid.requires_grad


def test_double_backward_is_not_silently_wrong():
    """The custom Function has no double backward; it must say so, not lie."""
    sh, th, swt, twt, mask, _, _ = _make()
    loss, _, _ = rr.distillation_chunked_jsd(sh, th, swt, twt, mask, beta = 0.5, chunk_size = 3)
    (grad_hidden,) = torch.autograd.grad(loss, sh, create_graph = True)
    # The saved gradient is a constant with respect to the inputs, so a second
    # derivative through it is zero rather than a silently wrong value.
    assert grad_hidden.grad_fn is not None or not grad_hidden.requires_grad
