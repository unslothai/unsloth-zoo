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

"""Fused CE for position-aligned labels (seq2seq heads); fixtures are verbatim upstream loss regions."""

import inspect
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
from unsloth_zoo.fused_losses.forward_adapter import EMPTY_LOGITS, unsloth_fused_lm_head_loss

# transformers 5.18 BartForConditionalGeneration (also Bart, BigBirdPegasus, Blenderbot* 5.4+).
BART = """
    def forward(self, outputs=None, labels=None, **kwargs):
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return masked_lm_loss, lm_logits
"""

# transformers 5.18 MarianMTModel (MBart, Mvp, LED use the same one-line bias with vocab_size).
MARIAN = """
    def forward(self, outputs=None, labels=None, **kwargs):
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

        return masked_lm_loss, lm_logits
"""

# transformers 5.18 T5ForConditionalGeneration (MT5, UMT5, LongT5 alike).
T5 = """
    def forward(self, sequence_output=None, labels=None, **kwargs):
        if self.config.scale_decoder_outputs:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return loss, lm_logits
"""

# transformers 5.18 WhisperForConditionalGeneration.
WHISPER = """
    def forward(self, outputs=None, labels=None, **kwargs):
        lm_logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        return loss, lm_logits
"""

# transformers 5.18 TrOCRForCausalLM.
TROCR = """
    def forward(self, outputs=None, labels=None, **kwargs):
        logits = self.output_projection(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return loss, logits
"""

# transformers 5.x BartForCausalLM (MBart, PLBart, Marian, Pegasus, Blenderbot* wrappers alike).
BART_CAUSAL = """
    def forward(self, hidden_states=None, labels=None, logits_to_keep=0, **kwargs):
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return loss, logits
"""

# transformers 5.13.0 .. 5.14.1 MoonshineForConditionalGeneration.
MOONSHINE_513 = """
    def forward(self, outputs=None, labels=None, **kwargs):
        logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        return loss, logits
"""

# transformers 5.15+ CohereAsrForConditionalGeneration (Canary 5.17+ alike): biased head.
COHERE_ASR = """
    def forward(self, outputs=None, labels=None, **kwargs):
        logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_labels = kwargs.pop("shift_labels", labels)
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                shift_labels=shift_labels,
                **kwargs,
            )

        return loss, logits
"""

# transformers 4.55 .. 5.12.1 MoonshineForConditionalGeneration: stock shifts, so it must stay out.
MOONSHINE_CAUSAL = """
    def forward(self, outputs=None, labels=None, **kwargs):
        logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return loss, logits
"""

# name -> (source, input keyword, head, class name used for the compiler admission rule)
FIXTURES = {
    "bart": (BART, "outputs", "lm_head", "BartForConditionalGeneration"),
    "marian": (MARIAN, "outputs", "lm_head", "MarianMTModel"),
    "t5": (T5, "sequence_output", "lm_head", "T5ForConditionalGeneration"),
    "whisper": (WHISPER, "outputs", "proj_out", "WhisperForConditionalGeneration"),
    "trocr": (TROCR, "outputs", "output_projection", "TrOCRForCausalLM"),
    "bart_causal": (BART_CAUSAL, "hidden_states", "lm_head", "BartForCausalLM"),
    "moonshine_513": (MOONSHINE_513, "outputs", "proj_out", "MoonshineForConditionalGeneration"),
    "cohere_asr": (COHERE_ASR, "outputs", "proj_out", "CohereAsrForConditionalGeneration"),
}
BIASED = ("bart", "marian")
V, H = 37, 16


class _Out(tuple):
    @property
    def last_hidden_state(self):
        return self[0]


def _compile(src):
    from transformers.loss.loss_utils import ForCausalLMLoss

    ns = dict(
        torch = torch, os = os, CrossEntropyLoss = CrossEntropyLoss,
        unsloth_fused_lm_head_loss = unsloth_fused_lm_head_loss, EMPTY_LOGITS = EMPTY_LOGITS,
        ForCausalLMLoss = ForCausalLMLoss,
    )
    exec(textwrap.dedent(src), ns)
    return ns["forward"]


def _model():
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.manual_seed(0)
    return SimpleNamespace(
        lm_head = torch.nn.Linear(H, V, bias = False),
        proj_out = torch.nn.Linear(H, V, bias = True),
        output_projection = torch.nn.Linear(H, V, bias = False),
        final_logits_bias = torch.randn(1, V),
        model_dim = H,
        config = SimpleNamespace(vocab_size = V, decoder_vocab_size = V, scale_decoder_outputs = True),
        loss_function = ForCausalLMLoss,
    )


def _labels():
    torch.manual_seed(1)
    labels = torch.randint(0, V, (2, 9))
    labels[0, :3] = -100
    labels[1, -2:] = -100
    return labels


def _run(fn, model, name, labels, hidden = None):
    _, key, head, _ = FIXTURES[name]
    if hidden is None:
        torch.manual_seed(2)
        hidden = torch.randn(2, 9, H)
    hidden = hidden.detach().clone().requires_grad_(True)
    head_mod = getattr(model, head)
    for p in head_mod.parameters():
        p.grad = None
    value = _Out((hidden,)) if key == "outputs" else hidden
    loss, logits = fn(model, labels = labels, **{key: value})
    if loss is None:
        return None, logits, None, None
    loss.backward()
    grads = [p.grad.clone() for p in head_mod.parameters()]
    return loss.detach(), logits, hidden.grad.clone(), grads


@pytest.mark.parametrize("name", list(FIXTURES))
def test_rewrite_parses_and_hook_is_unchanged(name):
    src, _, head, _ = FIXTURES[name]
    new, cap = rewrite_forward_source_spliced(src)
    assert new is not None and cap.head_attr == head and cap.aligned_target
    assert "unsloth_fused_lm_head_loss(" in new
    # The import hook never takes these shapes.
    assert rewrite_forward_source(textwrap.dedent(src)) == (None, None)


@pytest.mark.parametrize("name", list(FIXTURES))
def test_compiler_admits_aligned_heads(name):
    src, _, _, cls = FIXTURES[name]
    assert compiler._ast_fused_lm_head_fallback(src, cls) is not None


def test_compiler_keeps_causal_proj_out_out():
    new, cap = rewrite_forward_source_spliced(MOONSHINE_CAUSAL)
    assert new is not None and not cap.aligned_target
    assert compiler._ast_fused_lm_head_fallback(MOONSHINE_CAUSAL, "MoonshineForConditionalGeneration") is None


@pytest.mark.parametrize("name", list(FIXTURES))
def test_fused_matches_original(name, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    src = FIXTURES[name][0]
    original, fused = _compile(src), _compile(rewrite_forward_source_spliced(src)[0])
    model, labels = _model(), _labels()
    ref = _run(original, model, name, labels)
    out = _run(fused, model, name, labels)
    assert out[1] is EMPTY_LOGITS
    torch.testing.assert_close(out[0], ref[0], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(out[2], ref[2], rtol = 1e-4, atol = 1e-6)
    for g, r in zip(out[3], ref[3]):
        torch.testing.assert_close(g, r, rtol = 1e-4, atol = 1e-6)

    # Generation path: the original logits, bias included.
    _, logits_ref, _, _ = _run(original, model, name, None)
    _, logits_new, _, _ = _run(fused, model, name, None)
    torch.testing.assert_close(logits_new, logits_ref)

    # UNSLOTH_RETURN_LOGITS=1 replays the original statements.
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    out = _run(fused, model, name, labels)
    torch.testing.assert_close(out[1], ref[1])
    torch.testing.assert_close(out[0], ref[0])


@pytest.mark.parametrize("name", ["bart", "t5", "whisper", "cohere_asr"])
def test_wrong_shift_is_detected(name, monkeypatch):
    # A causal shift on aligned targets must change the loss, so the parity test above can fail.
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    src = FIXTURES[name][0]
    new = rewrite_forward_source_spliced(src)[0]
    if "shift_labels=False" in new:
        wrong = new.replace("shift_labels=False", "shift_labels=True")
    else:
        wrong = new.replace("kwargs.pop('shift_labels', labels)", "kwargs.pop('shift_labels', None)")
    assert wrong != new
    model, labels = _model(), _labels()
    ref = _run(_compile(src), model, name, labels)[0]
    assert not torch.allclose(_run(_compile(wrong), model, name, labels)[0], ref, rtol = 1e-3)


@pytest.mark.parametrize("name", BIASED)
def test_dropped_bias_is_detected(name, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    src = FIXTURES[name][0]
    new = rewrite_forward_source_spliced(src)[0]
    dropped = new.replace(", logits_bias=self.final_logits_bias", "")
    assert dropped != new
    model, labels = _model(), _labels()
    ref = _run(_compile(src), model, name, labels)[0]
    assert not torch.allclose(_run(_compile(dropped), model, name, labels)[0], ref, rtol = 1e-3)


@pytest.mark.parametrize("name", BIASED)
def test_bias_promotion_under_bf16_autocast(name, monkeypatch):
    # Stock adds the fp32 buffer to bf16 logits (-> fp32); the kernel must do the same, not fold
    # the bias into the bf16 matmul.
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    src = FIXTURES[name][0]
    original, fused = _compile(src), _compile(rewrite_forward_source_spliced(src)[0])
    model, labels = _model(), _labels()
    with torch.no_grad():
        model.final_logits_bias.mul_(50)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        ref = _run(original, model, name, labels)
        out = _run(fused, model, name, labels)
    torch.testing.assert_close(out[0], ref[0], rtol = 1e-4, atol = 1e-4)


def test_trainable_bias_takes_the_exact_path(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    original, fused = _compile(BART), _compile(rewrite_forward_source_spliced(BART)[0])
    model, labels = _model(), _labels()
    model.final_logits_bias.requires_grad_(True)
    ref = _run(original, model, "bart", labels)
    out = _run(fused, model, "bart", labels)
    assert out[1] is not EMPTY_LOGITS
    torch.testing.assert_close(out[0], ref[0])


@pytest.mark.parametrize("edit", [
    ("CrossEntropyLoss()", "CrossEntropyLoss(label_smoothing=0.1)"),
    ("CrossEntropyLoss()", "CrossEntropyLoss(ignore_index=0)"),
    ("CrossEntropyLoss()", "CrossEntropyLoss(reduction='sum')"),
    ("labels.reshape(-1))", "labels[:, 1:].reshape(-1))"),
    ("lm_logits.view(-1, self.config.vocab_size)", "lm_logits[:, :-1].reshape(-1, self.config.vocab_size)"),
    ("            loss = loss_fct(", "            loss = 2 * loss_fct("),
])
def test_other_ce_variants_are_left_alone(edit):
    src = WHISPER.replace(*edit)
    assert src != WHISPER
    assert rewrite_forward_source_spliced(src) == (None, None)


@pytest.mark.parametrize("module, cls", [
    ("switch_transformers.modeling_switch_transformers", "SwitchTransformersForConditionalGeneration"),
    ("nllb_moe.modeling_nllb_moe", "NllbMoeForConditionalGeneration"),
    ("git.modeling_git", "GitForCausalLM"),
])
def test_extra_losses_are_left_alone(module, cls):
    mod = pytest.importorskip(f"transformers.models.{module}")
    source = inspect.getsource(getattr(mod, cls).forward)
    assert compiler._ast_fused_lm_head_fallback(source, cls) is None


def test_logits_bias_kernel_default_unchanged():
    from unsloth_zoo.fused_losses.cross_entropy_loss import compute_fused_ce_loss

    torch.manual_seed(0)
    x, w, y = torch.randn(10, H), torch.randn(V, H), torch.randint(0, V, (10,))
    a = compute_fused_ce_loss(x, w, None, y, shift_labels = False)[0]
    b = compute_fused_ce_loss(x, w, None, y, shift_labels = False, logits_bias = torch.zeros(1, V))[0]
    c = compute_fused_ce_loss(x, w, None, y, shift_labels = False, logits_bias = torch.ones(1, V) * torch.arange(V))[0]
    torch.testing.assert_close(a, CrossEntropyLoss()(x @ w.t(), y))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(c, CrossEntropyLoss()(x @ w.t() + torch.arange(V).float(), y))
