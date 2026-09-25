# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Post-lm_head scales (Falcon-H1, HyperCLOVAX) must reach the fused kernel; dropping
one trained on 128x-off logits and NaN'd on step 1. CPU-only."""

from __future__ import annotations

import ast
import textwrap

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source


PLAIN_FORWARD = '''
def forward(self, hidden_states, labels=None, **kwargs):
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return loss, logits
'''

SCALED_FORWARD = '''
def forward(self, hidden_states, labels=None, **kwargs):
    logits = self.lm_head(hidden_states) * self.model.lm_head_multiplier
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return loss, logits
'''

DIVIDED_FORWARD = SCALED_FORWARD.replace(
    ") * self.model.lm_head_multiplier", ") / self.model.lm_head_divisor"
)

REVERSED_FORWARD = SCALED_FORWARD.replace(
    "self.lm_head(hidden_states) * self.model.lm_head_multiplier",
    "self.model.lm_head_multiplier * self.lm_head(hidden_states)",
)

BIASED_FORWARD = SCALED_FORWARD.replace(
    ") * self.model.lm_head_multiplier", ") + self.final_logits_bias"
)

FLOAT_FORWARD = PLAIN_FORWARD.replace(
    "self.lm_head(hidden_states)", "self.lm_head(hidden_states).float()"
)


def _fused_call_src(new_src: str) -> str:
    tree = ast.parse(new_src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "unsloth_fused_lm_head_loss"):
            return ast.unparse(node)
    raise AssertionError("no fused call emitted:\n" + new_src)


def _loss_function_call_src(new_src: str) -> str:
    tree = ast.parse(new_src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "loss_function"):
            return ast.unparse(node)
    raise AssertionError("no loss_function call kept:\n" + new_src)


def test_multiplier_reaches_the_fused_call():
    new_src, cap = rewrite_forward_source(SCALED_FORWARD)
    assert new_src is not None, "the scaled triplet must still be rewritten"
    assert [(n, ast.unparse(v)) for n, v in cap.scale_kws] == [
        ("logit_scale_multiply", "self.model.lm_head_multiplier")
    ]
    assert "logit_scale_multiply=self.model.lm_head_multiplier" in _fused_call_src(new_src)


def test_divisor_reaches_the_fused_call():
    new_src, cap = rewrite_forward_source(DIVIDED_FORWARD)
    assert new_src is not None
    assert "logit_scale_divide=self.model.lm_head_divisor" in _fused_call_src(new_src)


def test_scale_on_the_left_is_still_a_multiply():
    new_src, cap = rewrite_forward_source(REVERSED_FORWARD)
    assert new_src is not None
    assert "logit_scale_multiply=self.model.lm_head_multiplier" in _fused_call_src(new_src)


def test_scale_is_not_applied_twice_in_the_return_logits_branch():
    new_src, _ = rewrite_forward_source(SCALED_FORWARD)
    assert "logit_scale_multiply" not in _loss_function_call_src(new_src)


def test_unscaled_forward_is_unchanged():
    new_src, cap = rewrite_forward_source(PLAIN_FORWARD)
    assert new_src is not None
    assert cap.scale_kws == []
    assert "logit_scale" not in new_src


def test_float_wrapper_still_rewrites():
    new_src, cap = rewrite_forward_source(FLOAT_FORWARD)
    assert new_src is not None, "a .float() wrapper must not lose the fused path"
    assert cap.scale_kws == []


def test_repeated_scale_bails_out():
    chained = SCALED_FORWARD.replace(
        ") * self.model.lm_head_multiplier",
        ") * self.model.lm_head_multiplier * self.config.extra_scale",
    )
    new_src, cap = rewrite_forward_source(chained)
    assert new_src is None and cap is None


def test_additive_bias_bails_out():
    new_src, cap = rewrite_forward_source(BIASED_FORWARD)
    assert new_src is None and cap is None


class _Config:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class _Inner(torch.nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.lm_head_multiplier = multiplier


class _Tiny(torch.nn.Module):
    def __init__(self, hidden=16, vocab=32, multiplier=0.0078125):
        super().__init__()
        torch.manual_seed(0)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.model = _Inner(multiplier)
        self.config = _Config(vocab)

    def loss_function(self, logits=None, labels=None, vocab_size=None, **kwargs):
        logits = logits.float()
        shifted = torch.empty_like(labels)
        shifted[..., :-1] = labels[..., 1:]
        shifted[..., -1] = -100
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), shifted.view(-1), ignore_index=-100
        )

    def forward(self, hidden_states, labels=None, **kwargs):
        logits = self.lm_head(hidden_states) * self.model.lm_head_multiplier
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )
        return loss, logits


def _install_rewritten(cls):
    import os

    from unsloth_zoo.fused_losses.forward_adapter import (
        EMPTY_LOGITS,
        unsloth_fused_lm_head_loss,
    )

    src = textwrap.dedent(inspect_source(cls.forward))
    new_src, _cap = rewrite_forward_source(src)
    assert new_src is not None
    ns = {
        "os": os,
        "torch": torch,
        "unsloth_fused_lm_head_loss": unsloth_fused_lm_head_loss,
        "EMPTY_LOGITS": EMPTY_LOGITS,
    }
    exec(compile(new_src, "<rewritten>", "exec"), ns)
    return ns["forward"]


def inspect_source(fn):
    import inspect

    return inspect.getsource(fn)


def test_rewritten_forward_matches_the_reference_loss():
    model = _Tiny()
    torch.manual_seed(1)
    hidden = torch.randn(1, 8, 16, dtype=torch.float32)
    labels = torch.randint(0, 32, (1, 8))

    reference, _ = model(hidden, labels=labels)

    rewritten = _install_rewritten(_Tiny)
    fused, logits = rewritten(model, hidden, labels=labels)

    assert logits.numel() == 0, "fused path must not materialise logits"
    assert torch.isfinite(fused), "fused loss must be finite"
    torch.testing.assert_close(
        fused.float(), reference.float(), rtol=1e-4, atol=1e-4,
        msg=lambda m: "fused loss ignored the post-lm_head multiplier: " + m,
    )


def test_unscaled_model_is_unaffected():
    """Multiplier 1.0 control: proves the check above is not vacuous."""
    model = _Tiny(multiplier=1.0)
    torch.manual_seed(1)
    hidden = torch.randn(1, 8, 16, dtype=torch.float32)
    labels = torch.randint(0, 32, (1, 8))

    reference, _ = model(hidden, labels=labels)
    rewritten = _install_rewritten(_Tiny)
    fused, _ = rewritten(model, hidden, labels=labels)
    torch.testing.assert_close(fused.float(), reference.float(), rtol=1e-4, atol=1e-4)
