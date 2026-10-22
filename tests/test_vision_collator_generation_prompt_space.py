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

"""Generation prompts ending in a space the tokenizer fuses into the answer (unslothai/notebooks#149).

PaddleOCR-VL renders "Assistant: " before the answer: the whole string tokenizes as
[':', ' Z'], generation as [':', ' '] then 'Z'. Hermetic CPU tests with a stub
SentencePiece-like tokenizer (a space fuses into the following word).
"""

from __future__ import annotations

import re

import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator, _generation_prompt_fuses_answer

PAD_ID = 0
_PIECE = re.compile(r" ?[A-Za-z]+|\n|:|<s>|</s>| ")


class _FakeTokenizer:
    pad_token_id = PAD_ID
    padding_side = "right"

    def __init__(self):
        self.vocab = {"<pad>": PAD_ID}

    def encode(self, text):
        assert "".join(_PIECE.findall(text)) == text, text
        return [self.vocab.setdefault(p, len(self.vocab)) for p in _PIECE.findall(text)]

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse[i] for i in ids)

    def __call__(self, text, add_special_tokens = False, **kwargs):
        return {"input_ids": self.encode(text)}


def _text(content):
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


class _FakeProcessor:
    def __init__(self, generation_prompt = "Assistant: "):
        self.tokenizer = _FakeTokenizer()
        self.generation_prompt = generation_prompt

    def apply_chat_template(self, messages, tokenize = False, add_generation_prompt = False,
                            continue_final_message = False, **kwargs):
        out = "<s>"
        for m in messages:
            if m["role"] == "user":
                out += "User: " + _text(m["content"]) + "\n"
            else:
                out += self.generation_prompt + _text(m["content"]) + "</s>"
        return out + (self.generation_prompt if add_generation_prompt else "")

    def __call__(self, text, padding = True, padding_side = "right", return_tensors = "pt",
                 add_special_tokens = False, **kwargs):
        rows = [self.tokenizer.encode(t) for t in text]
        width = max(len(r) for r in rows)

        def pad(r, v):
            return [v] * (width - len(r)) + r if padding_side == "left" else r + [v] * (width - len(r))
        return {
            "input_ids": torch.tensor([pad(r, PAD_ID) for r in rows]),
            "attention_mask": torch.tensor([pad([1] * len(r), 0) for r in rows]),
        }


def make_collator(processor, split):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = processor
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = True
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.padding_token_ids = torch.tensor([PAD_ID])
    collator.assistant_single_content = False
    collator.chat_template_kwargs = {}
    collator.split_generation_prompt = split
    # As train_on_responses_only installs it; the stub masks nothing beyond the collator's labels.
    collator.train_on_responses_only = lambda batch: {"labels": batch["labels"].clone()}
    return collator


def chat(*answers):
    messages = [{"role": "user", "content": [{"type": "text", "text": "OCR:"}]}]
    for i, answer in enumerate(answers):
        if i:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Again:"}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return {"messages": messages}


def trained_pieces(processor, out, row = 0):
    ids, labels = out["input_ids"][row].tolist(), out["labels"][row].tolist()
    first = next(i for i, label in enumerate(labels) if label != -100)
    return [processor.tokenizer.decode([t]) for t in ids[first - 2:first + 1]], labels[first - 1]


def test_probe_flags_only_a_fused_trailing_space():
    assert _generation_prompt_fuses_answer(_FakeProcessor("Assistant: "))
    assert not _generation_prompt_fuses_answer(_FakeProcessor("Assistant:"))
    assert not _generation_prompt_fuses_answer(_FakeProcessor("Assistant:\n"))


def test_answer_boundary_matches_generation():
    processor = _FakeProcessor()
    out = make_collator(processor, _generation_prompt_fuses_answer(processor))([chat("Zuh tee"), chat("Hi")])
    # Generation feeds the rendered generation prompt, then continues with the bare answer
    prompt = processor.apply_chat_template(chat()["messages"], add_generation_prompt = True)
    for row, answer in enumerate(["Zuh tee", "Hi"]):
        expected = processor.tokenizer.encode(prompt) + processor.tokenizer.encode(answer + "</s>")
        ids = [t for t, m in zip(out["input_ids"][row].tolist(), out["attention_mask"][row].tolist()) if m]
        assert ids == expected
    pieces, before = trained_pieces(processor, out)
    assert pieces == [":", " ", "Zuh"] and before == -100


def test_without_split_the_space_fuses():
    processor = _FakeProcessor()
    out = make_collator(processor, False)([chat("Zuh tee")])
    pieces = [processor.tokenizer.decode([t]) for t in out["input_ids"][0].tolist()]
    at = pieces.index("Assistant")
    assert pieces[at:at + 3] == ["Assistant", ":", " Zuh"]


def test_multi_turn_keeps_full_text_path():
    processor = _FakeProcessor()
    out = make_collator(processor, True)([chat("Zuh", "Hi")])
    text = processor.apply_chat_template(chat("Zuh", "Hi")["messages"])
    assert out["input_ids"][0].tolist() == processor.tokenizer.encode(text)


def test_full_sequence_training_keeps_full_text_path():
    processor = _FakeProcessor()
    collator = make_collator(processor, True)
    collator.train_on_responses_only = None
    out = collator([chat("Zuh")])
    text = processor.apply_chat_template(chat("Zuh")["messages"])
    assert out["input_ids"][0].tolist() == processor.tokenizer.encode(text)
