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

"""UnslothVisionDataCollator forwards chat_template_kwargs (Qwen3.8 enable_thinking, #1070)."""

from __future__ import annotations

import pytest
import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

REASONING = "Reasoning effort is set to xhigh."


class _FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        return 0


class _Qwen38LikeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = object()
        self.template_kwargs = []

    def apply_chat_template(self, messages, tokenize = False, add_generation_prompt = False, **kwargs):
        self.template_kwargs.append(kwargs)
        text = " ".join(m["content"][0]["text"] for m in messages)
        if kwargs.get("enable_thinking", True):
            text = REASONING + " " + text
        return text

    def __call__(self, text, padding = True, padding_side = "right", return_tensors = "pt", **kwargs):
        rows = [[i + 1 for i, _ in enumerate(t.split())] for t in text]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
        }


def _make_collator(chat_template_kwargs = None):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _Qwen38LikeProcessor()
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = True
    collator.pad_to_multiple_of = None
    collator.padding_token_ids = torch.tensor([0])
    collator.train_on_responses_only = None
    collator.assistant_single_content = False
    collator.image_size = 224
    collator.patch_size = 14
    collator.snap_to_patch_size = False
    collator.size_func = lambda x: x
    if chat_template_kwargs is not None:
        collator.chat_template_kwargs = dict(chat_template_kwargs)
    return collator


def _user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _example(layout, **extra):
    if layout == "messages":
        return {"messages": [_user("hi"), _assistant("ok")], "images": [], **extra}
    return {"prompt": [_user("hi")], "completion": [_assistant("ok")], "images": [], **extra}


@pytest.mark.parametrize("layout", ["messages", "prompt_completion"])
def test_default_passes_no_template_kwargs(layout):
    collator = _make_collator()
    collator([_example(layout)])
    assert collator.processor.template_kwargs
    assert all("enable_thinking" not in k for k in collator.processor.template_kwargs)


@pytest.mark.parametrize("layout", ["messages", "prompt_completion"])
@pytest.mark.parametrize("source", ["collator", "row"])
def test_enable_thinking_false_reaches_template(layout, source):
    kwargs = {"enable_thinking": False}
    if source == "collator":
        collator, example = _make_collator(kwargs), _example(layout)
    else:
        collator, example = _make_collator(), _example(layout, chat_template_kwargs = kwargs)
    collator([example])
    assert all(k.get("enable_thinking") is False for k in collator.processor.template_kwargs)


@pytest.mark.parametrize("layout", ["messages", "prompt_completion"])
def test_row_kwargs_override_collator(layout):
    collator = _make_collator({"enable_thinking": False})
    collator([_example(layout, chat_template_kwargs = {"enable_thinking": True})])
    assert all(k.get("enable_thinking") is True for k in collator.processor.template_kwargs)
