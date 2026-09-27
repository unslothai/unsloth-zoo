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

"""Kimi K2.5 / K2.7 processors take `medias=` and one unpadded `text`, not `images=`.
The processor stub below keeps the remote KimiK25Processor.__call__ body. Hermetic CPU test."""

from __future__ import annotations

import types

import pytest
import torch

tokenizers = pytest.importorskip("tokenizers")
from PIL import Image

SPECIALS = ["<|im_user|>", "<|im_assistant|>", "<|im_middle|>", "<|im_end|>", "<|media_begin|>",
            "<|media_content|>", "<|media_pad|>", "<|media_end|>", "[PAD]"]

TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<|im_user|>user<|im_middle|>{% else %}<|im_assistant|>assistant<|im_middle|>{% endif %}"
    "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}<|media_begin|>image<|media_content|><|media_pad|><|media_end|>\n"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>{% endfor %}"
    "{% if add_generation_prompt %}<|im_assistant|>assistant<|im_middle|>{% endif %}"
)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    core = Tokenizer(models.BPE(vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}, merges = []))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object = core, pad_token = "[PAD]", eos_token = "<|im_end|>")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"
    return tok


class _MediaProcessor:
    """KimiK25VisionProcessor shaped: 14 px patches, one grid_thw row per image."""
    def preprocess(self, medias, return_tensors = None):
        from transformers.feature_extraction_utils import BatchFeature
        pixel_values, grid_thws = [], []
        for media in medias:
            assert media["type"] == "image"
            w, h = media["image"].size
            gh, gw = max(h // 14, 1), max(w // 14, 1)
            pixel_values.append(torch.full((gh * gw, 3, 14, 14), float(len(grid_thws))))
            grid_thws.append(torch.tensor([1, gh, gw]))
        return BatchFeature(data = {"pixel_values": torch.cat(pixel_values), "grid_thws": torch.stack(grid_thws)},
                            tensor_type = return_tensors)


def _processor_class():
    from transformers.feature_extraction_utils import BatchFeature

    class KimiK25Processor:
        # __call__ body as in the remote kimi_k25_processor.py (video handling dropped).
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.image_processor = self.media_processor = _MediaProcessor()
            self.chat_template = tokenizer.chat_template

        def __call__(self, messages = None, medias = None, text = None, return_tensors = "pt", **kwargs):
            if messages is None and (medias is None or text is None):
                raise ValueError("Provide either 'messages' or both 'medias' and 'text'")
            if medias is None:
                medias = [{"type": "image", "image": c["image"]} for m in messages if m["role"] == "user"
                          for c in m["content"] if isinstance(c, dict) and c.get("type") == "image"]
            preprocessed = self.media_processor.preprocess(medias, return_tensors = return_tensors)
            if text is None:
                text = self.tokenizer.apply_chat_template(messages, **kwargs)
            text_inputs = self.tokenizer(text, return_tensors = return_tensors)
            return BatchFeature(data = {**text_inputs, **preprocessed.data})

        def apply_chat_template(self, messages, **kwargs):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    return KimiK25Processor


def _examples():
    sizes = [(140, 28), (56, 42)]
    return [{"messages": [
        {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", s)},
                                     {"type": "text", "text": "Write the LaTeX."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "x^2" if i == 0 else "\\frac{a}{b} + c"}]}]}
        for i, s in enumerate(sizes)]


def _model():
    cfg = types.SimpleNamespace(vision_config = types.SimpleNamespace(patch_size = 14, image_size = 56),
                                torch_dtype = torch.bfloat16, dtype = torch.bfloat16)
    return types.SimpleNamespace(config = cfg, max_seq_length = 256,
                                 get_input_embeddings = lambda: types.SimpleNamespace(weight = torch.empty(0, dtype = torch.bfloat16)))


def test_collator_calls_a_medias_processor_and_masks_the_media_tokens():
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator
    tok = _tokenizer()
    processor = _processor_class()(tok)
    collator = UnslothVisionDataCollator(
        _model(), processor, train_on_responses_only = True,
        instruction_part = "<|im_user|>user<|im_middle|>", response_part = "<|im_assistant|>assistant<|im_middle|>")
    batch = collator(_examples())
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    # One <|media_pad|> per image (the model expands it), one grid_thw row per image.
    pad = tok.convert_tokens_to_ids("<|media_pad|>")
    assert (batch["input_ids"] == pad).sum(dim = 1).tolist() == [1, 1]
    assert batch["grid_thws"].shape == (2, 3)
    n_patches = int(batch["grid_thws"].prod(dim = 1).sum())
    assert batch["pixel_values"].shape[0] == n_patches
    assert batch["pixel_values"].dtype == torch.bfloat16
    # Image order follows example order.
    first = int(batch["grid_thws"][0].prod())
    assert float(batch["pixel_values"][:first].float().mean()) == 0.0
    assert float(batch["pixel_values"][first:].float().mean()) == 1.0
    labels = batch["labels"]
    for token in ("<|media_begin|>", "<|media_content|>", "<|media_pad|>", "<|media_end|>", "[PAD]"):
        assert not (labels == tok.convert_tokens_to_ids(token)).any(), token
    supervised = [tok.decode(batch["input_ids"][i][labels[i] != -100].tolist()) for i in range(2)]
    assert "x^2" in supervised[0] and "\\frac{a}{b} + c" in supervised[1]
    assert all("Write the LaTeX" not in s for s in supervised)
    # Left padding kept: the shorter row starts with pad and ends with a real token.
    assert batch["attention_mask"][:, -1].tolist() == [1, 1]


def test_patched_processor_keeps_the_remote_calls():
    from unsloth_zoo.vision_utils import patch_medias_processor
    tok = _tokenizer()
    processor = _processor_class()(tok)
    assert patch_medias_processor(processor)
    image = Image.new("RGB", (28, 28))
    text = tok.apply_chat_template(_examples()[0]["messages"][:1], tokenize = False)
    remote = processor(medias = [{"type": "image", "image": image}], text = text)
    standard = processor(text = [text], images = [image])
    assert torch.equal(remote["input_ids"], standard["input_ids"])
    assert torch.equal(remote["pixel_values"], standard["pixel_values"])
    assert torch.equal(remote["grid_thws"], standard["grid_thws"])
    with pytest.raises(ValueError, match = "1 `<\\|media_pad\\|>`"):
        processor(text = [text], images = [image, image])


def test_standard_processors_are_not_patched():
    from unsloth_zoo.vision_utils import patch_medias_processor

    class _Standard:
        def __call__(self, text = None, images = None, **kwargs):
            return None

    assert not patch_medias_processor(_Standard())
    assert "__wrapped__" not in vars(_Standard.__call__)
