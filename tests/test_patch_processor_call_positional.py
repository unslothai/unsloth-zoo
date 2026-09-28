# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""patch_processor_call must keep each processor's own positional order.

Qwen3-Omni and Qwen2.5-Omni processors take (text, images, videos, audio), so a
wrapper that rebinds positionals as (images, text, videos) moves the text into
the images slot.
"""

import pytest

from unsloth_zoo.tokenizer_utils import patch_processor_call


class _Base:
    image_processor = object()

    def apply_chat_template(self, conversation, tokenize = False, add_generation_prompt = False):
        return "|".join(m["content"] for m in conversation) + ("<gen>" if add_generation_prompt else "")


class _TextFirst(_Base):
    def __call__(self, text = None, images = None, videos = None, audio = None, **kwargs):
        return {"text": text, "images": images, "videos": videos, "audio": audio}


class _ImagesFirst(_Base):
    def __call__(self, images = None, text = None, videos = None, audio = None, **kwargs):
        return {"text": text, "images": images, "videos": videos, "audio": audio}


CONVERSATION = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("cls", [_TextFirst, _ImagesFirst])
def test_positional_args_keep_processor_order(cls):
    processor = patch_processor_call(cls())
    args = ("hello", "img", "vid") if cls is _TextFirst else ("img", "hello", "vid")
    assert processor(*args) == {"text": "hello", "images": "img", "videos": "vid", "audio": None}


@pytest.mark.parametrize("cls", [_TextFirst, _ImagesFirst])
def test_conversation_is_templated_positional_and_keyword(cls):
    processor = patch_processor_call(cls())
    positional = (CONVERSATION,) if cls is _TextFirst else (None, CONVERSATION)
    assert processor(*positional)["text"] == "hi<gen>"
    assert processor(text = CONVERSATION, add_generation_prompt = False)["text"] == "hi"
