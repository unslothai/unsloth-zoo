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
    _fix_audio_feature_extractor_padding_side,
    _is_audio_mapping,
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
    out = collator._extract_audio_for_example({"audio": [0.0] * 16}, [])
    assert len(out) == 1 and isinstance(out[0], np.ndarray)
    assert out[0].shape == (16,)


def test_top_level_list_of_path_strings_are_clips():
    collator = make_collator()
    out = collator._extract_audio_for_example({"audio": ["/tmp/a.wav", "/tmp/b.wav"]}, [])
    assert out == ["/tmp/a.wav", "/tmp/b.wav"]


def test_top_level_list_of_flat_list_clips():
    collator = make_collator()
    out = collator._extract_audio_for_example({"audio": [[0.0] * 16, [1.0] * 8]}, [])
    assert len(out) == 2
    assert out[0].shape == (16,) and out[1].shape == (8,)


def test_inline_nested_list_stereo_raises():
    # stereo serialized as nested Python lists must hit the mono guard too
    collator = make_collator()
    stereo = [[0.0] * 16, [1.0] * 16]
    with pytest.raises(ValueError, match="mono"):
        collator._extract_audio_for_example(
            {}, msgs({"type": "audio", "audio": stereo}))


def test_inline_nested_list_mono_squeezed():
    collator = make_collator()
    out = collator._extract_audio_for_example(
        {}, msgs({"type": "audio", "audio": [[0.0] * 16]}))
    assert len(out) == 1 and out[0].shape == (16,)


def test_inline_list_of_strings_clip_raises():
    # one content part is one clip; a list of paths inside it is user error
    collator = make_collator()
    with pytest.raises(ValueError, match="list of strings"):
        collator._extract_audio_for_example(
            {}, msgs({"type": "audio", "audio": ["/tmp/a.wav", "/tmp/b.wav"]}))


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
# _fix_audio_feature_extractor_padding_side
# ---------------------------------------------------------------------------

def test_left_padded_feature_extractor_reset_to_right():
    proc = _FakeProcessor()
    proc.feature_extractor.padding_side = "left"
    _fix_audio_feature_extractor_padding_side(proc)
    assert proc.feature_extractor.padding_side == "right"


def test_right_padded_feature_extractor_untouched():
    proc = _FakeProcessor()
    proc.feature_extractor.padding_side = "right"
    _fix_audio_feature_extractor_padding_side(proc)
    assert proc.feature_extractor.padding_side == "right"


def test_processor_without_feature_extractor_noop():
    class _TextOnly:
        pass
    _fix_audio_feature_extractor_padding_side(_TextOnly())


def test_feature_extractor_without_padding_side_noop():
    proc = _FakeProcessor()
    _fix_audio_feature_extractor_padding_side(proc)
    assert not hasattr(proc.feature_extractor, "padding_side")


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


# ---------------------------------------------------------------------------
# datasets >= 4 torchcodec AudioDecoder columns (unsloth/unsloth#7226)
#
# patch_torchcodec_audio_decoder grafts the mapping protocol onto AudioDecoder
# instead of making it a dict subclass, so the audio gates' isinstance(x, dict)
# checks used to reject it and drop the decoder into the raw-waveform catch-all.
# It then reached the feature extractor as an object array and blew up inside
# np.fft.rfft. These cover all three gates.
#
# _FakeAudioDecoder mirrors the real class: __getitem__ plus exactly the methods
# patch_torchcodec_audio_decoder grafts, not a dict subclass, and no ndim. These
# tests run in CI. The real-decoder tests below need datasets >= 4 + torchcodec
# and skip when either is missing.
# ---------------------------------------------------------------------------

DECODED = np.linspace(-0.5, 0.5, 32, dtype=np.float32)


class _FakeAudioDecoder:
    """Same surface as a patched datasets.features._torchcodec.AudioDecoder."""

    def __getitem__(self, key):
        if key == "array":
            return DECODED
        if key == "sampling_rate":
            return 16000
        raise KeyError(key)

    # exactly what patch_torchcodec_audio_decoder grafts
    def __contains__(self, key):
        return key in ("array", "sampling_rate")

    def __iter__(self):
        return iter(("array", "sampling_rate"))

    def keys(self):
        return ("array", "sampling_rate")

    def get(self, key, default=None):
        return self[key] if key in ("array", "sampling_rate") else default


def test_fake_decoder_is_not_a_dict():
    # The premise of the bug: the gates' isinstance check cannot see this object.
    decoder = _FakeAudioDecoder()
    assert not isinstance(decoder, dict)
    assert _is_audio_mapping(decoder)


def test_gate1_top_level_decoder_column_decoded():
    collator = make_collator()
    clips = collator._extract_audio_for_example({"audio": _FakeAudioDecoder()}, msgs({"type": "text", "text": "hi"}))
    assert len(clips) == 1
    assert clips[0].dtype == np.float32
    assert clips[0].dtype != object
    np.testing.assert_allclose(clips[0], DECODED)


def test_gate2_list_of_decoders_is_list_of_clips():
    # Guards the flat-samples branch too: a list of decoders must not be
    # collapsed into a single clip.
    collator = make_collator()
    clips = collator._extract_audio_for_example(
        {"audio": [_FakeAudioDecoder(), _FakeAudioDecoder()]}, msgs({"type": "text", "text": "hi"})
    )
    assert len(clips) == 2
    for clip in clips:
        assert clip.dtype == np.float32
        np.testing.assert_allclose(clip, DECODED)


def test_gate3_inline_decoder_message_content_decoded():
    out = extract_audio_info(msgs({"type": "audio", "audio": _FakeAudioDecoder()}))
    assert len(out) == 1
    assert out[0].dtype == np.float32
    np.testing.assert_allclose(out[0], DECODED)


def test_decoder_sampling_rate_still_validated():
    # The decoder must go through _resolve_audio_dict, not bypass its checks.
    with pytest.raises(ValueError, match="does not match the feature extractor"):
        extract_audio_info(msgs({"type": "audio", "audio": _FakeAudioDecoder()}), sampling_rate=24000)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(8, dtype=np.float32),          # bare ndarray
        [0.0, 1.0, 2.0],                        # flat list
        "clip.wav",                             # path string
        torch.zeros(8),                         # torch tensor
    ],
)
def test_non_mapping_audio_values_are_not_treated_as_mappings(value):
    # The capability check is looser than isinstance; make sure the ordinary
    # waveform/path payloads still fall through to their own branches.
    assert not _is_audio_mapping(value)


# --- the real decoder, when the optional deps are installed ----------------

def _real_decoder():
    datasets = pytest.importorskip("datasets", minversion="4.0.0")
    pytest.importorskip("torchcodec")
    try:
        from datasets.features._torchcodec import AudioDecoder
    except ImportError:
        pytest.skip("datasets.features._torchcodec.AudioDecoder unavailable")

    from unsloth_zoo.dataset_utils import patch_torchcodec_audio_decoder
    patch_torchcodec_audio_decoder()

    import io, struct, wave
    sr = 16000
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.25, sr // 4, endpoint=False))).astype(np.float32)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))
    return AudioDecoder(buffer.getvalue())


def test_real_decoder_is_not_a_dict_but_is_a_mapping():
    decoder = _real_decoder()
    assert not isinstance(decoder, dict)
    assert _is_audio_mapping(decoder)


@pytest.mark.parametrize("gate", ["top_level", "list", "inline"])
def test_real_decoder_decodes_to_float32_at_every_gate(gate):
    decoder = _real_decoder()
    collator = make_collator()
    if gate == "top_level":
        clips = collator._extract_audio_for_example({"audio": decoder}, msgs({"type": "text", "text": "hi"}))
    elif gate == "list":
        clips = collator._extract_audio_for_example({"audio": [decoder]}, msgs({"type": "text", "text": "hi"}))
    else:
        clips = extract_audio_info(msgs({"type": "audio", "audio": decoder}))
    assert len(clips) == 1
    assert clips[0].dtype == np.float32
    assert clips[0].dtype != object
    assert clips[0].ndim == 1
