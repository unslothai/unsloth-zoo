# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""chat_template_kwargs forwarding in UnslothVisionDataCollator."""

from __future__ import annotations

from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
    _merge_chat_template_kwargs,
)


_QWEN38_REASONING_SYSTEM_PROMPT = "Reasoning effort is set to"


class _FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __init__(self, chat_template: str):
        self.chat_template = chat_template

    def convert_tokens_to_ids(self, tokens):
        return 0


class _Qwen38LikeProcessor:
    """Minimal processor whose template mirrors Qwen3.8 reasoning injection."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer(
            "{% if enable_thinking is not defined %}{% set enable_thinking = true %}{% endif %}"
            "{% if enable_thinking %}"
            "<|im_start|>system\nReasoning effort is set to xhigh."
            "{% endif %}"
            "{{ messages[0]['content'][0]['text'] }}"
        )
        self.image_processor = object()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        enable_thinking = kwargs.get("enable_thinking", True)
        text = messages[0]["content"][0]["text"]
        if enable_thinking:
            return (
                "<|im_start|>system\n"
                "Reasoning effort is set to xhigh. Please think carefully.\n"
                f"{text}"
            )
        return text

    def __call__(self, text, padding=True, padding_side="right", return_tensors="pt", add_special_tokens=False, **kwargs):
        import torch
        rows = [[1, 2, 3]]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def _make_collator(processor, chat_template_kwargs=None):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = processor
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = True
    collator.pad_to_multiple_of = None
    collator.image_size = 224
    collator.patch_size = 14
    collator.padding_token_ids = __import__("torch").tensor([0])
    collator.train_on_responses_only = None
    collator.assistant_single_content = False
    collator.snap_to_patch_size = False
    collator.size_func = lambda x: x
    collator.chat_template_kwargs = dict(chat_template_kwargs or {})
    return collator


def test_merge_chat_template_kwargs_prefers_example_override():
    merged = _merge_chat_template_kwargs(
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert merged == {"enable_thinking": True}


def test_render_chat_default_does_not_inject_enable_thinking():
    processor = _Qwen38LikeProcessor()
    collator = _make_collator(processor)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    rendered = collator._apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert _QWEN38_REASONING_SYSTEM_PROMPT in rendered


def test_render_chat_respects_explicit_enable_thinking_false():
    processor = _Qwen38LikeProcessor()
    collator = _make_collator(processor, chat_template_kwargs={"enable_thinking": False})
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    rendered = collator._apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert _QWEN38_REASONING_SYSTEM_PROMPT not in rendered
    assert rendered == "Hello"


def test_render_chat_respects_explicit_enable_thinking_true():
    processor = _Qwen38LikeProcessor()
    collator = _make_collator(processor, chat_template_kwargs={"enable_thinking": True})
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    rendered = collator._apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert _QWEN38_REASONING_SYSTEM_PROMPT in rendered


def test_per_example_chat_template_kwargs_override_collator():
    processor = _Qwen38LikeProcessor()
    collator = _make_collator(processor, chat_template_kwargs={"enable_thinking": True})
    example = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Train me"}]},
        ],
        "images": [],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    rendered = collator._apply_chat_template(
        example["messages"],
        example=example,
        tokenize=False,
        add_generation_prompt=False,
    )
    assert _QWEN38_REASONING_SYSTEM_PROMPT not in rendered
    assert rendered == "Train me"
    out = collator([example])
    assert "input_ids" in out
