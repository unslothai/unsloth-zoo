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
import time

import pytest

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


def _fused(source, name):
    start = time.time()
    new, fused = compiler.apply_fused_lm_head(source, name)
    assert time.time() - start < 0.5
    return new, fused


SOURCES = {"Qwen2AudioForConditionalGeneration": QWEN2_AUDIO, "GraniteSpeechForConditionalGeneration": GRANITE_SPEECH}


@pytest.mark.parametrize("name", list(SOURCES))
def test_multiline_masked_shift_ce_is_fused(name):
    new, fused = _fused(SOURCES[name], name)
    assert fused
    assert "unsloth_fused_ce_loss(" in new
    assert "mask                 = attention_mask," in new
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
    _, fused = _fused(source, cls)
    assert fused


def test_inner_model_logits_are_not_fused():
    # Qwen2-Audio 5.4 - 5.9 takes logits from the inner model: no local `hidden_states` to fuse.
    source = QWEN2_AUDIO.replace(
        "        hidden_states = outputs.last_hidden_state\n        logits = self.lm_head(hidden_states)\n",
        "        logits = outputs.logits\n",
    )
    assert compiler.apply_fused_lm_head(source, "Qwen2AudioForConditionalGeneration")[1] is False
