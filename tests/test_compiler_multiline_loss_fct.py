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

"""Multi-line masked-shift `loss_fct(...)` (Qwen2-Audio, Granite Speech) gets pattern 3 fused CE."""

import importlib
import inspect
import os
import textwrap
import time
import types

import pytest
import torch

from unsloth_zoo import compiler

# transformers 5.18 Qwen2AudioForConditionalGeneration.forward, head + loss region.
QWEN2_AUDIO = """    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        attention_mask = outputs.attention_mask
        labels = outputs.labels if outputs.labels is not None else labels

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        return loss, logits
"""

# transformers 5.10+ GraniteSpeechForConditionalGeneration.forward, head + loss region.
GRANITE_SPEECH = """    def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=0, **kwargs):
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        return loss, logits
"""


# The rewrite takes ~20 ms. This bound only has to catch the exponential backtracking #1571 fixed,
# which never finished, so it is generous and counts this process's CPU time: wall time on a
# shared runner also counts time spent descheduled, which once read 1.6 s for this call.
_REWRITE_CPU_BUDGET_S = 5.0


def _fused(source, name):
    start = time.process_time()
    new, fused = compiler.apply_fused_lm_head(source, name)
    spent = time.process_time() - start
    assert spent < _REWRITE_CPU_BUDGET_S, f"apply_fused_lm_head spent {spent:.2f}s of CPU on {name}"
    return new, fused


SOURCES = {"Qwen2AudioForConditionalGeneration": QWEN2_AUDIO, "GraniteSpeechForConditionalGeneration": GRANITE_SPEECH}


@pytest.mark.parametrize("name", list(SOURCES))
def test_multiline_masked_shift_ce_is_fused(name):
    new, fused = _fused(SOURCES[name], name)
    assert fused
    assert "unsloth_fused_ce_loss(" in new
    assert "_mask = attention_mask\n" in new
    assert "text_config" not in new


def test_rebinds_before_loss_are_hoisted_above_the_head():
    new, fused = _fused(QWEN2_AUDIO, "Qwen2AudioForConditionalGeneration")
    assert fused
    head = new.index("logits = self.lm_head(hidden_states) if os.environ")
    assert new.index("attention_mask = outputs.attention_mask") < head
    assert new.index("labels = outputs.labels if outputs.labels is not None else labels") < head


def test_rebind_that_reads_logits_is_not_hoisted():
    source = QWEN2_AUDIO.replace(
        "        attention_mask = outputs.attention_mask\n",
        "        attention_mask = outputs.attention_mask[:, : logits.shape[1]]\n",
    )
    assert compiler.apply_fused_lm_head(source, "Qwen2AudioForConditionalGeneration") == (source, False)


@pytest.mark.parametrize("module, cls", [
    ("transformers.models.qwen2_audio.modeling_qwen2_audio", "Qwen2AudioForConditionalGeneration"),
    ("transformers.models.granite_speech.modeling_granite_speech", "GraniteSpeechForConditionalGeneration"),
    ("transformers.models.granite_speech_plus.modeling_granite_speech_plus", "GraniteSpeechPlusForConditionalGeneration"),
])
def test_installed_forwards_are_fused(module, cls):
    try:
        klass = getattr(importlib.import_module(module), cls)
    except (ImportError, AttributeError):
        pytest.skip(f"{cls} not in this transformers")
    source = compiler.fixup_fused_lm_head(inspect.getsource(klass.forward))
    if "loss = loss_fct(\n" not in source:
        pytest.skip(f"{cls} does not use the multi-line call in this transformers")
    if "self.lm_head(" not in source:
        # transformers 4.x takes the logits from the inner language model: nothing local to fuse.
        pytest.skip(f"{cls} has no local lm_head in this transformers")
    _, fused = _fused(source, cls)
    if "self.lm_head(" not in source:
        # transformers 4.57 (and Qwen2-Audio 5.4 - 5.9) take `logits = outputs[0]` from the inner
        # language model: there is no local hidden state to fuse, so the rewriter must decline.
        assert not fused, f"{cls} has no local lm_head call, yet its forward was rewritten"
    else:
        assert fused


def test_inner_model_logits_are_not_fused():
    # Qwen2-Audio 5.4 - 5.9 takes logits from the inner model: no local `hidden_states` to fuse.
    source = QWEN2_AUDIO.replace(
        "        hidden_states = outputs.last_hidden_state\n        logits = self.lm_head(hidden_states)\n",
        "        logits = outputs.logits\n",
    )
    assert compiler.apply_fused_lm_head(source, "Qwen2AudioForConditionalGeneration")[1] is False


def _granite_loss(source, attention_mask):
    from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_ce_loss
    ns = dict(
        torch = torch, nn = torch.nn, os = os, EMPTY_LOGITS = torch.empty(0), UNSLOTH_ENABLE_CCE = False,
        HAS_CUT_CROSS_ENTROPY = False, UNSLOTH_COMPILE_DISABLE = True,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss, __DYNAMO__RECOMPILING__ = None,
    )
    exec(textwrap.dedent(source), ns)
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 8)
    model = types.SimpleNamespace(
        lm_head = torch.nn.Linear(8, 16, bias = False),
        language_model = lambda **kwargs: types.SimpleNamespace(last_hidden_state = hidden),
        config = types.SimpleNamespace(vocab_size = 16, text_config = types.SimpleNamespace(vocab_size = 16)),
    )
    labels = torch.randint(0, 16, (2, 6))
    return ns["forward"](model, attention_mask = attention_mask, labels = labels)[0]


@pytest.mark.parametrize("prefix", [0, 3])
def test_granite_speech_mask_longer_than_labels(prefix, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    new, fused = _fused(GRANITE_SPEECH, "GraniteSpeechForConditionalGeneration")
    assert fused
    mask = torch.ones(2, prefix + 6, dtype = torch.long)
    mask[0, -2:] = 0
    torch.testing.assert_close(_granite_loss(new, mask), _granite_loss(GRANITE_SPEECH, mask))
