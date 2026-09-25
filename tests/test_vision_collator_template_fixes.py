# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Vision collator fixes found by running the notebooks' vision recipe on real processors.

1. Command A Vision: the generation prompt opens with <|START_THINKING|> while a rendered
   assistant turn opens with <|START_TEXT|>, so auto-detection fell back to a response marker
   that carried the user turn's terminator. An image placeholder sits inside that terminator
   ("<|END_TEXT|><|IMG_PATCH|><|END_OF_TURN_TOKEN|>"), the marker never matched, and every
   label was masked: training silently ran on nothing. The collator now also refuses a batch
   with no trainable token instead of training on it.
2. Dynamic-resolution processors (Nemotron Omni) return one pixel tensor per image when the
   sizes differ, and their outer BatchFeature fails to stack them.
3. Remote processors that never load chat_template.jinja (MiniMax-M3 VL), that name the image
   component differently (Step-3.7), or whose template renders a content list as its repr.

Hermetic CPU tests: a byte-level tokenizer built in memory, no network.
"""

from __future__ import annotations

import pytest
import torch

tokenizers = pytest.importorskip("tokenizers")

SPECIALS = [
    "<BOS>", "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>",
    "<|CHATBOT_TOKEN|>", "<|START_TEXT|>", "<|END_TEXT|>", "<|START_THINKING|>", "<|IMG_PATCH|>",
]

# Command A Vision shaped template: images go after the user text, the generation prompt
# opens a thinking block, a finished assistant turn opens a text block.
TEMPLATE = (
    "{{ '<BOS>' }}"
    "{% for m in messages %}"
    "{% if m['content'] is string %}{% set parts = [{'type': 'text', 'text': m['content']}] %}"
    "{% else %}{% set parts = m['content'] %}{% endif %}"
    "{% if m['role'] == 'user' %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>"
    "{% for p in parts %}{% if p['type'] == 'text' %}{{ p['text'] }}{% endif %}{% endfor %}<|END_TEXT|>"
    "{% for p in parts %}{% if p['type'] == 'image' %}<|IMG_PATCH|>{% endif %}{% endfor %}"
    "<|END_OF_TURN_TOKEN|>"
    "{% else %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_TEXT|>"
    "{% for p in parts %}{{ p['text'] }}{% endfor %}<|END_TEXT|><|END_OF_TURN_TOKEN|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>{% endif %}"
)


def _tokenizer(template = TEMPLATE):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    core = Tokenizer(models.BPE(vocab = vocab, merges = []))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object = core, bos_token = "<BOS>", eos_token = "<|END_OF_TURN_TOKEN|>",
                                  pad_token = "<BOS>")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS[1:]})
    tok.chat_template = template
    return tok


def _conversation(image = True):
    user = [{"type": "text", "text": "Write the LaTeX."}] + ([{"type": "image"}] if image else [])
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": "x^2 + y^2"}]}]


def test_marker_drops_the_user_terminator_when_the_generation_prompt_differs():
    from unsloth_zoo.dataset_utils import get_chat_template_parts
    instruction_part, response_part = get_chat_template_parts(_tokenizer())
    assert response_part == "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_TEXT|>"
    assert instruction_part == "<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>"


def test_detected_markers_supervise_the_answer_of_an_image_turn():
    from unsloth_zoo.dataset_utils import get_chat_template_parts, train_on_responses_only
    tok = _tokenizer()
    instruction_part, response_part = get_chat_template_parts(tok)
    mask = train_on_responses_only(None, instruction_part = instruction_part, response_part = response_part,
                                   tokenizer = tok, return_function = True)
    text = tok.apply_chat_template(_conversation(image = True), tokenize = False)
    assert "<|END_TEXT|><|IMG_PATCH|><|END_OF_TURN_TOKEN|>" in text
    ids = torch.tensor([tok(text, add_special_tokens = False)["input_ids"]])
    labels = mask({"input_ids": ids, "labels": ids.clone()})["labels"][0]
    supervised = tok.decode(ids[0][labels != -100].tolist())
    assert "x^2 + y^2" in supervised
    assert "Write the LaTeX" not in supervised


def _bare_collator():
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.ignore_index = -100
    return collator


def test_a_batch_with_no_trainable_token_raises_before_any_supervised_batch():
    collator = _bare_collator()
    with pytest.raises(ValueError, match = "no trainable"):
        collator._check_supervised(torch.full((2, 5), -100))


def test_after_a_supervised_batch_an_empty_example_only_warns():
    collator = _bare_collator()
    ok = torch.full((2, 5), -100)
    ok[:, 3] = 7
    collator._check_supervised(ok)
    collator._check_supervised(torch.full((2, 5), -100))
    partial = ok.clone()
    partial[1] = -100
    collator._check_supervised(partial)


def test_a_partially_supervised_first_batch_counts_as_supervised():
    collator = _bare_collator()
    partial = torch.full((2, 5), -100)
    partial[0, 3] = 7
    collator._check_supervised(partial)
    collator._check_supervised(torch.full((2, 5), -100))


def test_ragged_pixel_values_stay_a_list_and_text_fields_become_tensors():
    from unsloth_zoo.vision_utils import _tensorize_ragged_batch
    batch = _tensorize_ragged_batch({
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
        "pixel_values": [torch.zeros(3, 16, 32), torch.zeros(3, 24, 16)],
        "imgs_sizes": [(16, 32), (24, 16)],
    })
    assert torch.is_tensor(batch["input_ids"]) and batch["input_ids"].shape == (2, 3)
    assert torch.is_tensor(batch["attention_mask"])
    assert isinstance(batch["pixel_values"], list) and len(batch["pixel_values"]) == 2
    same = _tensorize_ragged_batch({"pixel_values": [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)]})
    assert torch.is_tensor(same["pixel_values"]) and same["pixel_values"].shape == (2, 3, 8, 8)


def test_processor_without_a_chat_template_adopts_its_tokenizers():
    from unsloth_zoo.vision_utils import _adopt_tokenizer_chat_template

    class _Processor:
        chat_template = None

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tok = _tokenizer()
    processor = _Processor(tok)
    assert _adopt_tokenizer_chat_template(processor)
    assert processor.chat_template == tok.chat_template
    # A processor that already has one keeps it.
    processor.chat_template = "{{ 'own' }}"
    assert not _adopt_tokenizer_chat_template(processor)
    assert processor.chat_template == "{{ 'own' }}"


def test_content_list_rendered_as_repr_is_detected():
    from unsloth_zoo.vision_utils import _renders_content_list_as_repr
    repr_template = (
        "{% for m in messages %}<|START_OF_TURN_TOKEN|>{{ m['content'] }}<|END_OF_TURN_TOKEN|>{% endfor %}"
    )
    tok = _tokenizer(repr_template)
    rendered = tok.apply_chat_template(_conversation(), tokenize = False)
    assert _renders_content_list_as_repr(rendered, "x^2 + y^2")
    clean = _tokenizer().apply_chat_template(_conversation(), tokenize = False)
    assert not _renders_content_list_as_repr(clean, "x^2 + y^2")


def test_processor_with_a_differently_named_image_component_is_accepted():
    from unsloth_zoo.vision_utils import _processor_takes_images

    class _StepLike:
        image_preprocessor = object()

        def __call__(self, text = None, images = None, return_tensors = None, **kwargs):
            return {}

    class _TextOnly:
        def __call__(self, text = None, **kwargs):
            return {}

    assert _processor_takes_images(_StepLike())
    assert not _processor_takes_images(_TextOnly())
