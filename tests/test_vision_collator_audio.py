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

"""Audio support in UnslothVisionDataCollator (PRs #723 and follow-ups).

Hermetic CPU tests with stub processors, no model or network needed:

1. ``extract_audio_info``: inline arrays, HF Audio dicts, url/path content
   parts, payload-less parts raising instead of silently training text-only,
   and sampling-rate validation.
2. ``_extract_audio_for_example``: top-level dict / ndarray / flat list /
   list-of-clips columns, None fallback to inline audio, torch tensor
   conversion, and the mono-only guard.
3. ``_truncate_sequence_tensors``: per-token-key allowlist (audio feature
   tensors must never be sliced even on dimension collisions), left-padding
   aware slicing, and the audio-span truncation guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
    extract_audio_info,
)

AUDIO_ID = 9
PAD_ID = 0


class _FakeTokenizer:
    pad_token_id = PAD_ID
    audio_token = "<|audio|>"

    def __init__(self, padding_side="left"):
        self.padding_side = padding_side

    def convert_tokens_to_ids(self, tokens):
        table = {"<|audio|>": AUDIO_ID}
        if isinstance(tokens, str):
            return table.get(tokens, -1)
        return [table.get(t, -1) for t in tokens]


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeProcessor:
    def __init__(self, padding_side="left"):
        self.tokenizer = _FakeTokenizer(padding_side)
        self.feature_extractor = _FakeFeatureExtractor()


def make_collator(max_seq_length=4, padding_side="left"):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeProcessor(padding_side)
    collator.max_seq_length = max_seq_length
    collator.truncation = True
    return collator


def msgs(part):
    return [{"role": "user", "content": [part, {"type": "text", "text": "hi"}]}]


CLIP = np.zeros(16, dtype=np.float32)


# ---------------------------------------------------------------------------
# extract_audio_info
# ---------------------------------------------------------------------------

def test_inline_array():
    out = extract_audio_info(msgs({"type": "audio", "audio": CLIP}))
    assert len(out) == 1 and out[0] is CLIP


def test_inline_hf_dict_unwrapped():
    part = {"type": "audio", "audio": {"array": CLIP, "sampling_rate": 16000}}
    out = extract_audio_info(msgs(part), sampling_rate=16000)
    assert len(out) == 1 and out[0] is CLIP


def test_inline_url_and_path_resolved():
    for key in ("url", "path"):
        out = extract_audio_info(msgs({"type": "audio", key: "/tmp/a.wav"}))
        assert out == ["/tmp/a.wav"]


def test_inline_no_payload_raises():
    with pytest.raises(ValueError, match="cannot be loaded"):
        extract_audio_info(msgs({"type": "audio"}))


def test_inline_sampling_rate_mismatch_raises():
    part = {"type": "audio", "audio": {"array": CLIP, "sampling_rate": 44100}}
    with pytest.raises(ValueError, match="sampling_rate"):
        extract_audio_info(msgs(part), sampling_rate=16000)


def test_non_audio_parts_ignored():
    out = extract_audio_info(msgs({"type": "image", "image": "x.png"}))
    assert out == []


# ---------------------------------------------------------------------------
# _extract_audio_for_example
# ---------------------------------------------------------------------------

def test_top_level_dict_unwrapped():
    collator = make_collator()
    out = collator._extract_audio_for_example(
        {"audio": {"array": CLIP, "sampling_rate": 16000}}, [])
    assert len(out) == 1 and out[0] is CLIP


def test_top_level_dict_rate_mismatch_raises():
    collator = make_collator()
    with pytest.raises(ValueError, match="sampling_rate"):
        collator._extract_audio_for_example(
            {"audio": {"array": CLIP, "sampling_rate": 44100}}, [])


def test_top_level_flat_list_is_one_clip():
    collator = make_collator()
    flat = [0.0] * 16
    out = collator._extract_audio_for_example({"audio": flat}, [])
    assert len(out) == 1 and out[0] is flat


def test_top_level_list_of_clips():
    collator = make_collator()
    out = collator._extract_audio_for_example(
        {"audio": [CLIP, {"array": CLIP, "sampling_rate": 16000}]}, [])
    assert len(out) == 2 and out[0] is CLIP and out[1] is CLIP


def test_top_level_none_falls_back_to_inline():
    collator = make_collator()
    out = collator._extract_audio_for_example(
        {"audio": None}, msgs({"type": "audio", "audio": CLIP}))
    assert len(out) == 1 and out[0] is CLIP


def test_torch_tensor_converted_to_numpy():
    collator = make_collator()
    out = collator._extract_audio_for_example({"audio": torch.zeros(16)}, [])
    assert len(out) == 1 and isinstance(out[0], np.ndarray) and out[0].ndim == 1


def test_mono_torchaudio_tensor_squeezed():
    # torchaudio.load returns [channels, frames]; mono is (1, N)
    collator = make_collator()
    out = collator._extract_audio_for_example({"audio": torch.zeros(1, 16)}, [])
    assert len(out) == 1 and isinstance(out[0], np.ndarray)
    assert out[0].shape == (16,)


def test_stereo_raises():
    collator = make_collator()
    with pytest.raises(ValueError, match="mono"):
        collator._extract_audio_for_example({"audio": np.zeros((2, 16))}, [])


def test_top_level_dict_path_resolved():
    # datasets.Audio(decode=False) style payload: {"bytes": None, "path": ...}
    collator = make_collator()
    out = collator._extract_audio_for_example(
        {"audio": {"bytes": None, "path": "/tmp/a.wav"}}, [])
    assert out == ["/tmp/a.wav"]


def test_top_level_dict_no_payload_raises():
    collator = make_collator()
    with pytest.raises(ValueError, match="cannot be loaded"):
        collator._extract_audio_for_example({"audio": {"sampling_rate": 16000}}, [])


def test_top_level_list_dict_path_resolved():
    collator = make_collator()
    out = collator._extract_audio_for_example(
        {"audio": [{"path": "/tmp/a.wav"}, {"array": CLIP, "sampling_rate": 16000}]}, [])
    assert out[0] == "/tmp/a.wav" and out[1] is CLIP


def test_inline_audio_decode_false_dict_resolved():
    part = {"type": "audio", "audio": {"bytes": None, "path": "/tmp/a.wav"}}
    out = extract_audio_info(msgs(part), sampling_rate=16000)
    assert out == ["/tmp/a.wav"]


# ---------------------------------------------------------------------------
# _truncate_sequence_tensors
# ---------------------------------------------------------------------------

def _batch_left_padded():
    # seq_len 6, max_seq_length 4. Row 0 is short (2 left pads + 2 audio + 2
    # text tokens), row 1 is full length. input_features last dim deliberately
    # equals seq_len to prove the old shape-collision bug stays fixed.
    return {
        "input_ids": torch.tensor([[PAD_ID, PAD_ID, AUDIO_ID, AUDIO_ID, 5, 6],
                                   [1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 3, 3, 0, 0],
                                           [0, 0, 0, 0, 0, 0]]),
        "input_features": torch.zeros(2, 3, 6),
        "input_features_mask": torch.ones(2, 6),
    }


def test_truncation_left_padding_keeps_short_row_content():
    collator = make_collator(max_seq_length=4, padding_side="left")
    batch = collator._truncate_sequence_tensors(_batch_left_padded(), seq_len=6)
    # Short row keeps its content (audio span intact), not its padding
    assert batch["input_ids"][0].tolist() == [AUDIO_ID, AUDIO_ID, 5, 6]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1]
    # Long row truncates its tail
    assert batch["input_ids"][1].tolist() == [1, 2, 3, 4]
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["mm_token_type_ids"].shape == (2, 4)


def test_truncation_never_slices_audio_features():
    collator = make_collator(max_seq_length=4, padding_side="left")
    batch = collator._truncate_sequence_tensors(_batch_left_padded(), seq_len=6)
    assert batch["input_features"].shape == (2, 3, 6)
    assert batch["input_features_mask"].shape == (2, 6)


def test_truncation_right_padding_slices_head():
    collator = make_collator(max_seq_length=4, padding_side="right")
    batch = {
        "input_ids": torch.tensor([[AUDIO_ID, AUDIO_ID, 5, 6, PAD_ID, PAD_ID]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
    }
    batch = collator._truncate_sequence_tensors(batch, seq_len=6)
    assert batch["input_ids"][0].tolist() == [AUDIO_ID, AUDIO_ID, 5, 6]


def test_truncation_cutting_audio_span_raises():
    collator = make_collator(max_seq_length=3, padding_side="right")
    batch = {
        "input_ids": torch.tensor([[1, 2, AUDIO_ID, AUDIO_ID, AUDIO_ID, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
    }
    with pytest.raises(ValueError, match="audio tokens"):
        collator._truncate_sequence_tensors(batch, seq_len=6)
