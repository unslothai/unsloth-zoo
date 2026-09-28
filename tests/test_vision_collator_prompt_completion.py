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

"""Prompt-completion collation: token type id routing.

Gemma 3n emits ``token_type_ids``, Gemma 4 ``mm_token_type_ids``; both must
move in lock-step with ``input_ids`` instead of staying a stale prompt-width
copy. Hermetic CPU tests with a stub whitespace processor.
"""

from __future__ import annotations

import pytest
import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

PAD_ID = 0
IMG_ID = 7
VOCAB = {"<img>": IMG_ID, "<img-200>": -200, "<patch>": 8, "<end>": 9, "a": 1, "b": 2, "x": 3, "y": 4, "z": 5, "w": 6}


class _FakeTokenizer:
    pad_token_id = PAD_ID
    padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, -1)
        return [VOCAB.get(t, -1) for t in tokens]


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeProcessor:
    def __init__(self, tt_key):
        self.tokenizer = _FakeTokenizer()
        self.feature_extractor = _FakeFeatureExtractor()
        self.tt_key = tt_key

    def __call__(self, text, padding=True, padding_side="right", return_tensors="pt",
                 add_special_tokens=False, **kwargs):
        rows = [[VOCAB[t] for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [PAD_ID] * (width - len(r))
            if padding_side == "left":
                ids.append(pad + r)
                mask.append([0] * len(pad) + [1] * len(r))
            else:
                ids.append(r + pad)
                mask.append([1] * len(r) + [0] * len(pad))
        out = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        }
        if self.tt_key is not None:
            # Mirrors ProcessorMixin.create_mm_token_type_ids: 0 text, 1 image
            out[self.tt_key] = (out["input_ids"] == IMG_ID).long()
        return out


def make_collator(tt_key, max_seq_length=None):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeProcessor(tt_key)
    collator.formatting_func = None
    collator.max_seq_length = max_seq_length
    collator.truncation = max_seq_length is not None
    collator.ignore_index = -100
    collator.completion_only_loss = True
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.padding_token_ids = torch.tensor([PAD_ID, IMG_ID])
    return collator


# Skewed lengths so both pad sides appear and the flush genuinely moves tokens
EXAMPLES = [
    {"prompt": "<img> a b", "completion": "x"},
    {"prompt": "a", "completion": "x y z w"},
]


def test_mm_token_type_ids_routed_through_pc_path():
    out = make_collator("mm_token_type_ids")(EXAMPLES)
    assert "token_type_ids" not in out
    mm = out["mm_token_type_ids"]
    # Pre-fix the stale copy kept the prompt-only width (2, 3)
    assert mm.shape == out["input_ids"].shape
    assert torch.equal(mm, (out["input_ids"] == IMG_ID).long())


def test_token_type_ids_still_routed():
    out = make_collator("token_type_ids")(EXAMPLES)
    assert "mm_token_type_ids" not in out
    tt = out["token_type_ids"]
    assert tt.shape == out["input_ids"].shape
    assert torch.equal(tt, (out["input_ids"] == IMG_ID).long())


def test_mm_token_type_ids_truncation_stays_aligned():
    out = make_collator("mm_token_type_ids", max_seq_length=4)(EXAMPLES)
    mm = out["mm_token_type_ids"]
    assert mm.shape == out["input_ids"].shape == (2, 4)
    assert torch.equal(mm, (out["input_ids"] == IMG_ID).long())


def test_no_type_ids_emitted_is_a_noop():
    out = make_collator(None)(EXAMPLES)
    assert "token_type_ids" not in out and "mm_token_type_ids" not in out
    assert out["input_ids"].shape == out["attention_mask"].shape


def _mask_token_a(batch):
    # Stands in for train_on_responses_only: excludes token "a", re-exposes everything else.
    ids = batch["input_ids"]
    return {"labels": torch.where(ids == VOCAB["a"], torch.full_like(ids, -100), ids)}


def test_train_on_responses_only_applies_to_pc_path():
    collator = make_collator(None)
    collator.completion_only_loss = False
    collator.train_on_responses_only = _mask_token_a
    out = collator(EXAMPLES)
    labels, ids = out["labels"], out["input_ids"]
    assert (labels[ids == VOCAB["a"]] == -100).all()
    assert (labels[ids == VOCAB["x"]] == VOCAB["x"]).all()


def test_train_on_responses_only_never_unmasks_pc_path():
    collator = make_collator(None)
    collator.ignore_index = -1
    collator.train_on_responses_only = _mask_token_a
    out = collator(EXAMPLES)
    labels, ids = out["labels"], out["input_ids"]
    completion = labels != -1
    assert completion.any()
    assert not completion[out["attention_mask"] == 0].any()
    assert not completion[ids == IMG_ID].any()
    assert not completion[ids == VOCAB["b"]].any()
    assert (labels != -100).all()


def test_train_on_responses_only_pc_path_raises_when_nothing_is_trained():
    import pytest
    collator = make_collator(None)
    collator.train_on_responses_only = lambda batch: {"labels": torch.full_like(batch["input_ids"], -100)}
    with pytest.raises(ValueError, match = "no trainable token"):
        collator(EXAMPLES)


class _ChatProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__(None)
        self.seen_images = []

    def apply_chat_template(self, messages, **kwargs):
        words = []
        for m in messages:
            for part in m["content"]:
                words.append("<img>" if part["type"] == "image" else part["text"])
        return " ".join(words)

    def __call__(self, text, **kwargs):
        self.seen_images.append(kwargs.get("images"))
        return super().__call__(text, **kwargs)


@pytest.mark.parametrize("prompt", [
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a"}]}],
    "<img> a",
])
def test_top_level_images_reach_processor(prompt):
    # TRL vision rows: images column + bare markers or plain text; both used to drop the images.
    from PIL import Image

    collator = make_collator(None)
    collator.processor = _ChatProcessor()
    collator.assistant_single_content = False
    image = Image.new("RGB", (32, 32))
    completion = [{"role": "assistant", "content": [{"type": "text", "text": "x"}]}]
    collator([{"images": [image], "prompt": prompt,
               "completion": completion if isinstance(prompt, list) else "x"}])
    assert collator.processor.seen_images[0] == [[image]]


def test_top_level_image_urls_use_guarded_fetch(monkeypatch):
    import io
    from PIL import Image
    import unsloth_zoo.vision_utils as vu

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format = "PNG")
    fetched = []
    def fake_fetch(url):
        fetched.append(url)
        return io.BytesIO(buf.getvalue())
    monkeypatch.setattr(vu, "fetch_remote_media_bytes", fake_fetch)

    collator = make_collator(None)
    collator.processor = _ChatProcessor()
    collator.assistant_single_content = False
    collator([{"images": ["https://example.com/a.png"], "prompt": "<img> a", "completion": "x"}])
    assert fetched == ["https://example.com/a.png"]
    assert isinstance(collator.processor.seen_images[0][0][0], Image.Image)


def test_none_image_entries_are_dropped_in_pc_path():
    collator = make_collator(None)
    collator.processor = _ChatProcessor()
    collator.assistant_single_content = False
    collator([{"images": [None], "prompt": "a", "completion": "x"}])
    assert collator.processor.seen_images[0] is None


def test_mixed_batch_keeps_one_image_slot_per_row_in_pc_path():
    from PIL import Image
    collator = make_collator(None)
    collator.processor = _ChatProcessor()
    collator.assistant_single_content = False
    image = Image.new("RGB", (32, 32))
    collator([
        {"images": [image], "prompt": "<img> a", "completion": "x"},
        {"images": [None], "prompt": "a", "completion": "x"},
    ])
    assert collator.processor.seen_images[0] == [[image], []]


def test_mixed_batch_keeps_one_image_slot_per_row_in_messages_path():
    from PIL import Image
    collator = make_collator(None)
    collator.processor = _ChatProcessor()
    collator.assistant_single_content = False
    image = Image.new("RGB", (32, 32))
    def msgs(with_image):
        user = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": "a"}]
        return [{"role": "user", "content": user}, {"role": "assistant", "content": [{"type": "text", "text": "x"}]}]
    collator([{"images": None, "messages": msgs(False)}, {"images": [image], "messages": msgs(True)}])
    assert collator.processor.seen_images[0] == [[], [image]]


def _image_collator(max_seq_length):
    collator = make_collator(None, max_seq_length = max_seq_length)
    collator.processor = _ChatProcessor()
    collator.assistant_single_content = False
    return collator


@pytest.mark.parametrize("marker", ["<img>", "<img-200>", "<patch>"])
def test_truncation_that_cuts_an_image_placeholder_raises(marker):
    # <img-200>: negative sentinel in input_ids (Phi-4-reasoning-vision); <patch>: only the
    # processor declares it, as `image_token_id` (Step-3.7 `<im_patch>`).
    from PIL import Image
    collator = _image_collator(2)
    collator.processor.image_token_id = VOCAB["<patch>"]
    with pytest.raises(ValueError, match = "max_seq_length = 2 truncated 1 image / audio placeholder"):
        collator([{"images": [Image.new("RGB", (32, 32))], "prompt": f"a b {marker}", "completion": "x"}])


def test_truncation_that_keeps_every_image_placeholder_passes():
    from PIL import Image
    out = _image_collator(3)([{"images": [Image.new("RGB", (32, 32))], "prompt": "<img> a b", "completion": "x y"}])
    assert out["input_ids"].tolist() == [[IMG_ID, 1, 2]]


def test_cut_media_delimiter_alone_passes_when_the_model_names_its_feature_tokens():
    # <end> is a known media token but not a feature slot: the model forward still aligns.
    from PIL import Image
    batch = [{"images": [Image.new("RGB", (32, 32))], "prompt": "a <img> <end>", "completion": "x"}]
    collator = _image_collator(2)
    collator.padding_token_ids = torch.tensor([PAD_ID, IMG_ID, VOCAB["<end>"]])
    with pytest.raises(ValueError, match = "truncated 1 image / audio placeholder"):
        collator(batch)
    collator._feature_token_ids = [IMG_ID]
    assert collator(batch)["input_ids"].tolist() == [[1, IMG_ID]]


def test_feature_token_ids_cover_every_config_spelling():
    import types
    from unsloth_zoo.vision_utils import _media_feature_token_ids
    llava_onevision = types.SimpleNamespace(image_token_index = 151646, video_token_index = 151647)
    phi4mm = types.SimpleNamespace(vision_config = types.SimpleNamespace(image_token_id = 200010),
                                   audio_config = types.SimpleNamespace(audio_token_id = 200011))
    assert _media_feature_token_ids(types.SimpleNamespace(config = llava_onevision)) == [151646, 151647]
    assert _media_feature_token_ids(types.SimpleNamespace(config = phi4mm)) == [200010, 200011]
    omni = types.SimpleNamespace(thinker_config = types.SimpleNamespace(
        image_token_index = 151655, video_token_index = 151656, audio_token_index = 151646))
    assert _media_feature_token_ids(types.SimpleNamespace(config = omni)) == [151646, 151655, 151656]
