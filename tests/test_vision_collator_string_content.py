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

"""Phi-4-reasoning-vision string-content templates; stub mirrors processing_phi4_visionr.py."""

from __future__ import annotations

import sys
import types

import pytest
import torch

pytest.importorskip("tokenizers")
from PIL import Image

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

_remote = types.ModuleType("processing_phi4_visionr_stub")
_remote.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
sys.modules[_remote.__name__] = _remote

# chat_template.jinja of microsoft/Phi-4-reasoning-vision-15B (system prompt shortened).
PHI4V_TEMPLATE = (
    "<|im_start|>system<|im_sep|>You are Phi.<|im_end|>"
    "{% for message in messages %}{% if (message['role'] == 'user') %}"
    "{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}"
    "{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>'}}"
    "{% generation %}{{message['content'] + '<|im_end|>'}}{% endgeneration %}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
)
LIST_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}<|im_sep|>"
    "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for p in m['content'] %}"
    "{% if p['type'] == 'image' %}<image>{% else %}{{ p['text'] }}{% endif %}{% endfor %}{% endif %}"
    "<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant<|im_sep|>{% endif %}"
)


def _tokenizer(template):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    core = Tokenizer(models.BPE(vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}, merges = []))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object = core, eos_token = "<|im_end|>", pad_token = "<|pad|>")
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_sep|>"]})
    tok.chat_template = template
    return tok


class _ImageProcessor:
    def __call__(self, images, return_tensors = "pt"):
        return {"pixel_values": torch.zeros(len(images), 4, 3),
                "pixel_attention_mask": torch.ones(len(images), 4, dtype = torch.long)}


def _make_processor(template, image_token = None):
    class Phi4VisionRProcessorStub:
        """Same __call__ contract as Phi4VisionRProcessor: `<image>` -> one -200 id, right padding."""

        def __init__(self):
            self.tokenizer = _tokenizer(template)
            self.image_processor = _ImageProcessor()
            self.chat_template = None
            if image_token is not None:
                self.image_token = image_token

        def apply_chat_template(self, conversation, **kwargs):
            return self.tokenizer.apply_chat_template(conversation, **kwargs)

        def __call__(self, text = None, images = None, padding = False, **kwargs):
            placeholder = getattr(self, "image_token", DEFAULT_IMAGE_TOKEN)
            rows = []
            for t in text:
                chunks = [self.tokenizer(c, add_special_tokens = False).input_ids for c in t.split(placeholder)]
                ids = []
                for i, c in enumerate(chunks):
                    if i:
                        ids.append(IMAGE_TOKEN_INDEX)
                    ids.extend(c)
                rows.append(torch.tensor(ids))
            n = max(len(r) for r in rows)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.stack([torch.cat([r, torch.full((n - len(r),), pad)]) for r in rows])
            mask = torch.stack([torch.cat([torch.ones(len(r)), torch.zeros(n - len(r))]) for r in rows]).long()
            out = {"input_ids": input_ids, "attention_mask": mask}
            if images is not None:
                out.update(self.image_processor([im for group in images for im in group]))
            return out

    Phi4VisionRProcessorStub.__module__ = _remote.__name__
    return Phi4VisionRProcessorStub()


class _Model:
    max_seq_length = 256

    def __init__(self):
        self.config = types.SimpleNamespace(vision_config = {"model_type": "siglip2_vision_model"})

    def get_input_embeddings(self):
        return types.SimpleNamespace(weight = torch.zeros(1, dtype = torch.float32))


def _example(answer):
    image = Image.new("RGB", (32, 32))
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Write the LaTeX."}, {"type": "image", "image": image}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]}


def _collator(processor, **kw):
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
    return UnslothVisionDataCollator(
        _Model(), processor, train_on_responses_only = True,
        instruction_part = "<|im_start|>user<|im_sep|>", response_part = "<|im_start|>assistant<|im_sep|>", **kw,
    )


def test_string_content_template_builds_collator_and_masks_image_sentinel():
    processor = _make_processor(PHI4V_TEMPLATE)
    collator = _collator(processor)
    answers = ["x^2 + y^2", "\\frac{a}{b} = c"]
    batch = collator([_example(a) for a in answers])
    ids, labels = batch["input_ids"], batch["labels"]
    assert (ids == IMAGE_TOKEN_INDEX).sum(dim = 1).tolist() == [1, 1]
    assert (labels[ids == IMAGE_TOKEN_INDEX] == -100).all()
    assert (labels >= -100).all() and not (labels == IMAGE_TOKEN_INDEX).any()
    tok = processor.tokenizer
    for row, answer in zip(labels, answers):
        assert tok.decode(row[row != -100]) == answer + "<|im_end|>"
    assert batch["pixel_values"].shape[0] == 2


def test_string_content_template_renders_list_content_with_placeholder():
    processor = _make_processor(PHI4V_TEMPLATE)
    _collator(processor)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = processor.apply_chat_template(msgs, tokenize = False, add_generation_prompt = True)
    assert prompt.endswith("<|im_start|>user<|im_sep|><image>\nDescribe.<|im_end|><|im_start|>assistant<|im_sep|>")
    assert isinstance(msgs[0]["content"], list)
    same = processor.apply_chat_template([{"role": "user", "content": "<image>\nDescribe."}], tokenize = False)
    assert "<image>\nDescribe." in same
    wrapped = processor.apply_chat_template
    _collator(processor)
    assert processor.apply_chat_template is wrapped


def test_placeholder_prefers_processor_image_token():
    processor = _make_processor(PHI4V_TEMPLATE.replace("You are Phi.", "S"), image_token = "<|image_pad|>")
    batch = _collator(processor)([_example("z")])
    assert (batch["input_ids"] == IMAGE_TOKEN_INDEX).sum() == 1


def test_list_content_template_is_not_wrapped():
    processor = _make_processor(LIST_TEMPLATE)
    original = processor.apply_chat_template
    _collator(processor)
    assert processor.apply_chat_template == original
    assert not getattr(processor.apply_chat_template, "_unsloth_string_content", False)


class _PicklableStringProcessor:
    image_token = DEFAULT_IMAGE_TOKEN

    def apply_chat_template(self, conversation, tokenize = False, **kwargs):
        return "".join(m["role"] + ":" + m["content"] + "\n" for m in conversation)


def test_patched_processor_is_picklable():
    import pickle
    from unsloth_zoo.vision_utils import _patch_string_content_chat_template
    processor = _PicklableStringProcessor()
    assert _patch_string_content_chat_template(processor)
    clone = pickle.loads(pickle.dumps(processor))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    assert clone.apply_chat_template(msgs) == processor.apply_chat_template(msgs) == "user:<image>\nHi\n"
    assert not _patch_string_content_chat_template(clone)
