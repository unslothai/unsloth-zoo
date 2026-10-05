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

"""Compiler source rewrites must not hang, must not change sources they do not
fuse, and must not route a biased lm_head to the bias-less CCE kernel (CPU)."""

import multiprocessing
import os
import re
import textwrap
import types

import pytest
import torch

from unsloth_zoo import compiler

# transformers >= 5.10.1 DeepseekOcr2SamVisionSdpaAttention.forward (trimmed): an
# `if output_attentions:` guard with no `return super().forward(...)` after it.
DEEPSEEK_OCR2_SDPA = '''
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        if output_attentions:
            logger.warning_once(
                f"{self.__class__.__name__} does not support `output_attentions=True`. The returned attention weights will "
                "be `None`. If you want to get attention weights, please set `attn_implementation='eager'` when loading the model."
            )
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return attn_output, None
'''

# The loose-anchor shape only the fallback regex rewrites (space before the colon).
LOOSE_SHAPE = '''
    def forward(self, hidden_states, output_attentions=False):
        if output_attentions :
            logger.warning_once("eager")
            return super().forward(hidden_states=hidden_states)
        return hidden_states
'''

LEGACY_SHAPE = '''
    def forward(self, hidden_states, output_attentions=False):
        if output_attentions:
            logger.warning_once("eager")
            return super().forward(
                hidden_states=hidden_states,
            )
        return hidden_states
'''

OLD_FALLBACK = (
    r"if[ \t]+output_attentions[ \t]*:[^\n]*\n(?:[ \t]+[^\n]+\n)*?[ \t]+return[ \t]+"
    r"super\(\)\.forward\([^)]*\)"
)


def _gqa_child(source, queue):
    queue.put(compiler.replace_with_grouped_query_attention("M", source))


def _gqa_bounded(source, timeout = 60):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target = _gqa_child, args = (source, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail(f"replace_with_grouped_query_attention did not return within {timeout}s")
    return queue.get(timeout = 5)


def test_gqa_rewrite_terminates_on_unmatched_output_attentions_guard():
    assert _gqa_bounded(DEEPSEEK_OCR2_SDPA) == DEEPSEEK_OCR2_SDPA


@pytest.mark.parametrize("source", [LOOSE_SHAPE, LEGACY_SHAPE], ids = ["loose", "legacy"])
def test_gqa_rewrite_legacy_shapes_unchanged(source):
    out = _gqa_bounded(source)
    assert "raise RuntimeError('Unsloth: Not supported')" in out
    assert "super().forward" not in out
    if source is LOOSE_SHAPE:
        assert out == re.sub(
            OLD_FALLBACK,
            "if output_attentions: raise RuntimeError('Unsloth: Not supported')",
            source,
            flags = re.MULTILINE,
        )


# Mamba / FalconMamba / xLSTM (transformers <= 5.16.1) and Clvp: a non-VLM whose shifted CE
# line the Idefics normaliser rewrites, but which no pattern fuses.
UNFUSED_ONE_LINE_CE = '''
    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.backbone(input_ids)[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, logits)
'''

FUSED_ONE_LINE_CE = '''
    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.model(input_ids)[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, logits)
'''


def test_unfused_source_comes_back_byte_identical():
    out, fused = compiler.apply_fused_lm_head(UNFUSED_ONE_LINE_CE, "MambaForCausalLM")
    assert not fused
    assert out == UNFUSED_ONE_LINE_CE
    assert "text_config" not in out


def test_one_line_shifted_ce_still_fuses():
    out, fused = compiler.apply_fused_lm_head(FUSED_ONE_LINE_CE, "IdeficsLikeForCausalLM")
    assert fused
    assert "unsloth_fused_ce_loss" in out
    assert "text_config" not in out


# GraniteSpeech (transformers >= 5.10): the CE call spans lines, so patterns 1 and 3 cannot
# match; their regex used to run until its 1 s timeout on every compile.
MULTILINE_ALIGNED_CE = '''
    def forward(self, input_ids=None, decoder_input_ids=None, labels=None, **kwargs):
        outputs = self.model(input_ids, decoder_input_ids=decoder_input_ids)
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        return (masked_lm_loss, lm_logits)
'''


def test_ce_patterns_skip_regex_without_required_tail(monkeypatch):
    calls = []
    findall = compiler.regex.findall

    def counting_findall(pattern, *args, **kwargs):
        calls.append(pattern)
        return findall(pattern, *args, **kwargs)

    monkeypatch.setattr(compiler.regex, "findall", counting_findall)
    out, fused = compiler.apply_fused_lm_head(MULTILINE_ALIGNED_CE, "BartForConditionalGeneration")
    assert not fused
    assert out == MULTILINE_ALIGNED_CE
    assert calls == []


PATTERN_2_FORWARD = '''
    def forward(self, hidden_states, labels=None, **kwargs):
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return (loss, logits)
'''


def _route_frozen_head(source, bias):
    """Run the rewritten forward with stub kernels; returns which kernel took the loss."""
    new_source, fused = compiler.apply_fused_lm_head(source, "M")
    assert fused
    taken = []

    def fused_linear_cross_entropy(**kwargs):
        taken.append("cce")
        return torch.zeros(())

    def unsloth_fused_ce_loss(**kwargs):
        taken.append("fused")
        return torch.zeros(())

    def ForCausalLMLoss(*args, **kwargs):
        taken.append("loss_function")
        return torch.zeros(())

    ns = dict(
        torch = torch, os = os, EMPTY_LOGITS = torch.empty(0),
        UNSLOTH_ENABLE_CCE = True, HAS_CUT_CROSS_ENTROPY = True, UNSLOTH_COMPILE_DISABLE = True,
        fused_linear_cross_entropy = fused_linear_cross_entropy,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss,
    )
    exec(textwrap.dedent(new_source), ns)
    lm_head = torch.nn.Linear(8, 16, bias = bias).to(torch.float16).requires_grad_(False)
    self = types.SimpleNamespace(
        lm_head = lm_head,
        loss_function = ForCausalLMLoss,
        config = types.SimpleNamespace(vocab_size = 16),
    )
    ns["forward"](self, torch.randn(2, 5, 8, dtype = torch.float16), labels = torch.randint(0, 16, (2, 5)))
    return taken


@pytest.mark.parametrize("source", [PATTERN_2_FORWARD, FUSED_ONE_LINE_CE.replace(
    "hidden_states = self.model(input_ids)[0]\n", "hidden_states = input_ids\n"
).replace("input_ids=None", "input_ids")], ids = ["pattern2", "pattern1"])
def test_frozen_biased_head_skips_bias_less_cce(source, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    assert _route_frozen_head(source, bias = False) == ["cce"]
    assert _route_frozen_head(source, bias = True) == ["fused"]
