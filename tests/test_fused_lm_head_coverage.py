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

"""Fused CE coverage for transformers 4.x / 5.x forward shapes (CPU, source only)."""

import ast
import textwrap

import pytest
import torch

from unsloth_zoo import compiler
from unsloth_zoo.fused_losses.ast_rewriter import rewrite_forward_source

# transformers 4.57 GPT-Neo: casts around the loss call in the labels branch.
CAST_WRAPPED = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.transformer(input_ids)[0]
    lm_logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        labels = labels.to(lm_logits.device)
        lm_logits = lm_logits.to(torch.float32)
        loss = self.loss_function(lm_logits, labels, vocab_size=self.config.vocab_size, **kwargs)
        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)
    return (loss, lm_logits)
"""

# RecurrentGemma: tanh softcap between the head and the labels branch.
SOFTCAPPED = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.model(input_ids)[0]
    logits = self.lm_head(hidden_states)
    cap = self.config.logits_soft_cap
    logits = nn.functional.tanh(logits / cap) * cap
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""


def _branches(src):
    fn = ast.parse(src).body[0]
    outer = next(s for s in fn.body if isinstance(s, ast.If) and ast.unparse(s.test) == "labels is not None")
    inner = outer.body[0]
    return [ast.unparse(s) for s in inner.body], [ast.unparse(s) for s in inner.orelse], fn


def test_cast_wrapped_labels_branch_is_fused():
    new, cap = rewrite_forward_source(CAST_WRAPPED)
    assert new is not None and cap.logits_name == "lm_logits"
    unfused, fused, _ = _branches(new)
    assert fused[0].startswith("loss = unsloth_fused_lm_head_loss(hidden_states, self.lm_head, labels")
    # Fused branch keeps the loss cast, drops logits casts and the labels move (kernel does it).
    assert fused[1:] == ["loss = loss.to(hidden_states.dtype)", "lm_logits = EMPTY_LOGITS"]
    # UNSLOTH_RETURN_LOGITS=1 replays the original statements in order.
    assert unfused == [
        "lm_logits = self.lm_head(hidden_states)",
        "labels = labels.to(lm_logits.device)",
        "lm_logits = lm_logits.to(torch.float32)",
        "loss = self.loss_function(lm_logits, labels, vocab_size=self.config.vocab_size, **kwargs)",
        "lm_logits = lm_logits.to(hidden_states.dtype)",
        "loss = loss.to(hidden_states.dtype)",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        "        shift_logits = lm_logits[..., :-1, :]\n",
        "        if labels.dim() > 2:\n            labels = labels[..., 0]\n",
    ],
)
def test_non_cast_statement_in_labels_branch_still_bails(extra):
    src = CAST_WRAPPED.replace("        labels = labels.to(lm_logits.device)\n", extra)
    assert rewrite_forward_source(src) == (None, None)


def test_softcap_becomes_kernel_argument_after_cap_is_bound():
    new, cap = rewrite_forward_source(SOFTCAPPED)
    assert new is not None
    unfused, fused, fn = _branches(new)
    assert "logit_softcapping=cap" in fused[0]
    assert "logits = nn.functional.tanh(logits / cap) * cap" in unfused
    body = [ast.unparse(s) for s in fn.body]
    assert body.index("cap = self.config.logits_soft_cap") < next(
        i for i, s in enumerate(body) if s.startswith("if labels is not None")
    )


def test_mismatched_softcap_is_not_captured():
    src = SOFTCAPPED.replace("tanh(logits / cap) * cap", "tanh(logits / cap) * 2.0")
    assert rewrite_forward_source(src) == (None, None)


def test_canonical_forward_output_unchanged():
    canonical = """
def forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.model(input_ids)[0]
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
    return (loss, logits)
"""
    new, _ = rewrite_forward_source(canonical)
    assert "if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' or not _can_fuse_loss(self.loss_function):" in new
    assert "unsloth_fused_lm_head_loss(hidden_states, self.lm_head, labels, vocab_size=self.config.vocab_size, **kwargs)" in new


class _CompositeHead(torch.nn.Module):
    pass


class _CompositeHeadModel:
    def __init__(self, config):
        self.lm_head = _CompositeHead(config)


class _LinearHeadModel:
    def __init__(self, config):
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class _InheritedLinearHead(_LinearHeadModel):
    def __init__(self, config):
        super().__init__(config)


def test_head_kind_from_init_source():
    from unsloth_zoo.fused_losses.forward_install import _head_built_as_linear

    assert not _head_built_as_linear(_CompositeHeadModel, "lm_head")
    assert _head_built_as_linear(_LinearHeadModel, "lm_head")
    assert _head_built_as_linear(_InheritedLinearHead, "lm_head")
    assert _head_built_as_linear(object, "lm_head")  # unknown stays eligible


def test_roberta_style_heads_are_not_fused():
    from unsloth_zoo.fused_losses.forward_install import _head_built_as_linear

    roberta = pytest.importorskip("transformers.models.roberta.modeling_roberta")
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    assert not _head_built_as_linear(roberta.RobertaForCausalLM, "lm_head")
    assert _head_built_as_linear(llama.LlamaForCausalLM, "lm_head")


LM_LOGITS_REGEX_FORWARD = """    def forward(self, hidden_states, labels=None, **kwargs):
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return (loss, lm_logits)
"""


def test_lm_logits_name_is_fused_by_regex():
    out, ok = compiler.apply_fused_lm_head(LM_LOGITS_REGEX_FORWARD, "CTRLLMHeadModel")
    assert ok and "unsloth_fused_ce_loss" in out and "lm_logits" not in out


def test_lm_logits_rename_skipped_when_logits_name_is_taken():
    src = LM_LOGITS_REGEX_FORWARD.replace("return (loss, lm_logits)", "logits = lm_logits\n        return (loss, logits)")
    out, ok = compiler.apply_fused_lm_head(src, "CTRLLMHeadModel")
    assert not ok and "lm_logits" in out


def test_inline_head_scale_is_fused_with_scale():
    src = """    def forward(self, hidden_states, labels=None, **kwargs):
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * self.config.text_config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )
        return loss
"""
    out, ok = compiler.apply_fused_lm_head(src, "HyperCLOVAXVisionV2ForConditionalGeneration")
    assert ok
    assert "logit_scale_multiply = (self.config.text_config.logits_scaling)" in out


def test_unmatched_source_is_returned_byte_identical():
    src = textwrap.dedent(LM_LOGITS_REGEX_FORWARD).replace("self.loss_function(", "self.other_loss(")
    src = textwrap.indent(src, "    ")
    out, ok = compiler.apply_fused_lm_head(src, "X")
    assert not ok and out == src


def test_installer_leaves_composite_head_forward_alone():
    roberta = pytest.importorskip("transformers.models.roberta.modeling_roberta")
    from unsloth_zoo.fused_losses import forward_install

    forward_install.install_for_class(roberta.RobertaForCausalLM)
    assert "RobertaForCausalLM" not in forward_install.audit()["patched"]
    assert "unsloth_fused_lm_head_loss" not in roberta.RobertaForCausalLM.forward.__code__.co_names
