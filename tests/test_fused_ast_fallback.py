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

"""Compiler AST fallback for forwards the regex patterns miss; fixtures are verbatim upstream loss regions."""

import os
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch.nn import CrossEntropyLoss

from unsloth_zoo import compiler
from unsloth_zoo.fused_losses.ast_rewriter import (
    rewrite_forward_source,
    rewrite_forward_source_spliced,
)
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_count_aware_cross_entropy
from unsloth_zoo.fused_losses.forward_adapter import EMPTY_LOGITS, unsloth_fused_lm_head_loss

# transformers 5.16.0+ CohereCompassForConditionalGeneration: guarded logit scale.
COHERE_COMPASS = """
    @can_return_tuple
    def forward(self, hidden_states=None, labels=None, logits_to_keep=0, **kwargs):
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.logit_scale is not None:
            logits = logits * self.logit_scale

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return loss, logits
"""

# transformers 5.15.0+ PPFormulaNetForConditionalGeneration: aligned targets via shift_labels.
PP_FORMULANET = """
    def forward(self, hidden_states=None, labels=None, logits_to_keep=0, **kwargs):
        shift_labels = kwargs.pop("shift_labels", None)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Encoder-decoder logits are position-aligned with the targets, so pass them as `shift_labels`
            # (with `labels=None`) to stop `ForCausalLMLoss` shifting them a second time.
            loss = self.loss_function(
                logits=logits,
                labels=None,
                vocab_size=self.config.text_config.vocab_size,
                shift_labels=shift_labels if shift_labels is not None else labels,
                **kwargs,
            )

        return loss, logits
"""

# transformers 5.13.0+ Florence2ForConditionalGeneration.
FLORENCE2 = """
    def forward(self, hidden_states=None, labels=None, logits_to_keep=0, **kwargs):
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                shift_labels=labels,
                **kwargs,
            )

        return loss, logits
"""

# transformers <= 5.16.1 MambaForCausalLM / FalconMambaForCausalLM: legacy shifted CE.
MAMBA = """
    def forward(self, hidden_states=None, labels=None, logits_to_keep=0, **kwargs):
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :].to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss, logits
"""

# ClvpForCausalLM (every release): norm, then the head, then legacy shifted CE.
CLVP = """
    def forward(self, hidden_states=None, labels=None, **kwargs):
        lm_logits = self.final_norm(hidden_states)
        lm_logits = self.lm_head(lm_logits)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss, lm_logits
"""

# transformers 4.55.x OPTForCausalLM (the import hook is gated to >= 4.56).
OPT_455 = """
    def forward(self, hidden_states=None, labels=None, **kwargs):
        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return loss, logits
"""

FIXTURES = {
    "cohere_compass": COHERE_COMPASS,
    "pp_formulanet": PP_FORMULANET,
    "florence2": FLORENCE2,
    "mamba": MAMBA,
    "clvp": CLVP,
    "opt_455": OPT_455,
}
EXTENDED_ONLY = ("cohere_compass", "pp_formulanet", "mamba", "clvp")
V, H = 37, 16


def _ns():
    from transformers.loss.loss_utils import ForCausalLMLoss

    return dict(
        torch = torch,
        os = os,
        CrossEntropyLoss = CrossEntropyLoss,
        can_return_tuple = lambda f: f,
        unsloth_fused_lm_head_loss = unsloth_fused_lm_head_loss,
        unsloth_count_aware_cross_entropy = unsloth_count_aware_cross_entropy,
        EMPTY_LOGITS = EMPTY_LOGITS,
        ForCausalLMLoss = ForCausalLMLoss,
    )


def _compile(src):
    ns = _ns()
    exec(textwrap.dedent(src), ns)
    return ns["forward"]


def _model(logit_scale):
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.manual_seed(0)
    cfg = SimpleNamespace(vocab_size = V, text_config = SimpleNamespace(vocab_size = V))
    return SimpleNamespace(
        lm_head = torch.nn.Linear(H, V, bias = False),
        final_norm = torch.nn.LayerNorm(H),
        logit_scale = logit_scale,
        config = cfg,
        loss_function = ForCausalLMLoss,
    )


def _run(fn, model, labels, hidden, **kwargs):
    hidden = hidden.detach().clone().requires_grad_(True)
    model.lm_head.weight.grad = None
    loss, logits = fn(model, hidden_states = hidden, labels = labels, **kwargs)
    if loss is None:
        return None, logits, None, None
    loss.backward()
    return loss.detach(), logits, hidden.grad.clone(), model.lm_head.weight.grad.clone()


@pytest.mark.parametrize("name", list(FIXTURES))
def test_spliced_rewrite_keeps_signature_and_parses(name):
    src = FIXTURES[name]
    new, cap = rewrite_forward_source_spliced(src)
    assert new is not None and cap.head_attr == "lm_head"
    old_lines = src.splitlines()
    head = [x for x in old_lines if x.strip().startswith(("@", "def "))]
    assert head == [x for x in new.splitlines() if x.strip().startswith(("@", "def "))]
    assert "unsloth_fused_lm_head_loss(" in new and "EMPTY_LOGITS" in new


@pytest.mark.parametrize("name", EXTENDED_ONLY)
def test_import_hook_shapes_unchanged(name):
    # The hook keeps its narrower match set, so no class changes route at import time.
    assert rewrite_forward_source(textwrap.dedent(FIXTURES[name])) == (None, None)


@pytest.mark.parametrize("name", list(FIXTURES))
@pytest.mark.parametrize("logit_scale", [None, 0.25])
def test_fused_matches_original(name, logit_scale, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    src = FIXTURES[name]
    original = _compile(src)
    fused = _compile(rewrite_forward_source_spliced(src)[0])
    model = _model(logit_scale)
    torch.manual_seed(1)
    hidden = torch.randn(2, 9, H)
    labels = torch.randint(0, V, (2, 9))
    labels[0, :3] = -100
    ref = _run(original, model, labels, hidden)
    out = _run(fused, model, labels, hidden)
    assert out[1] is EMPTY_LOGITS
    torch.testing.assert_close(out[0], ref[0], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(out[2], ref[2], rtol = 1e-4, atol = 1e-6)
    torch.testing.assert_close(out[3], ref[3], rtol = 1e-4, atol = 1e-6)

    _, logits_ref, _, _ = _run(original, model, None, hidden)
    _, logits_new, _, _ = _run(fused, model, None, hidden)
    torch.testing.assert_close(logits_new, logits_ref)

    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    out = _run(fused, model, labels, hidden)
    torch.testing.assert_close(out[1], ref[1])
    torch.testing.assert_close(out[0], ref[0])


def test_zero_guarded_scale_keeps_original_loss(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    original = _compile(COHERE_COMPASS)
    fused = _compile(rewrite_forward_source_spliced(COHERE_COMPASS)[0])
    model = _model(0.0)
    torch.manual_seed(1)
    hidden = torch.randn(2, 9, H)
    labels = torch.randint(0, V, (2, 9))
    ref = _run(original, model, labels, hidden)
    out = _run(fused, model, labels, hidden)
    torch.testing.assert_close(out[0], ref[0])
    torch.testing.assert_close(out[1], ref[1])


def test_explicit_shift_labels_are_not_shifted_again(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    original = _compile(PP_FORMULANET)
    fused = _compile(rewrite_forward_source_spliced(PP_FORMULANET)[0])
    model = _model(None)
    hidden = torch.randn(2, 9, H)
    labels = torch.randint(0, V, (2, 9))
    targets = torch.randint(0, V, (2, 9))
    targets[1, 4:] = -100
    ref = _run(original, model, labels, hidden, shift_labels = targets)
    out = _run(fused, model, labels, hidden, shift_labels = targets)
    torch.testing.assert_close(out[0], ref[0], rtol = 1e-5, atol = 1e-6)


def test_compiler_fallback_only_for_unfused_lm_head_sources():
    assert compiler._ast_fused_lm_head_fallback(OPT_455, "OPTForCausalLM") is not None
    hooked, _ = rewrite_forward_source(textwrap.dedent(OPT_455))
    assert compiler._ast_fused_lm_head_fallback(hooked, "OPTForCausalLM") is None
    embed_out = OPT_455.replace("self.lm_head(", "self.embed_out(")
    assert compiler._ast_fused_lm_head_fallback(embed_out, "GPTNeoXForCausalLM") is not None
    assert compiler._ast_fused_lm_head_fallback(embed_out, "MoonshineForConditionalGeneration") is None
    unknown = OPT_455.replace("self.lm_head(", "self.cls(")
    assert compiler._ast_fused_lm_head_fallback(unknown, "X") is None

    class _CompositeHead(torch.nn.Module):
        pass

    class _Model:
        def __init__(self, config):
            self.lm_head = _CompositeHead(config)

    assert compiler._ast_fused_lm_head_fallback(OPT_455, "X", _Model) is None


def test_legacy_ce_requires_exact_shape():
    # Any change to the reduction / ignore_index / extra statement keeps the original forward.
    for old, new in (
        ("CrossEntropyLoss()", "CrossEntropyLoss(reduction=\"sum\")"),
        ("CrossEntropyLoss()", "CrossEntropyLoss(ignore_index=0)"),
        ("labels = labels.to(logits.device)", "labels = labels.to(logits.device)\n            labels[labels == 3] = -100"),
    ):
        assert rewrite_forward_source_spliced(MAMBA.replace(old, new)) == (None, None)


def test_guarded_scale_must_precede_softcap():
    src = COHERE_COMPASS.replace(
        "        if self.logit_scale is not None:",
        "        logits = torch.tanh(logits / 3.0) * 3.0\n        if self.logit_scale is not None:",
    )
    assert rewrite_forward_source_spliced(src) == (None, None)
