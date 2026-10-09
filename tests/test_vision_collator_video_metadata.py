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


"""transformers 5 reads a decoded video's rate from video_metadata, not fps= (else 24 fps).
unslothai/unsloth#3357."""

import types

import pytest
import torch

import unsloth_zoo.vision_utils as vu


def _video(n):
    return torch.zeros(n, 3, 8, 8, dtype = torch.uint8)


def _processor(do_sample_frames = False, num_frames = None):
    return types.SimpleNamespace(
        video_processor = types.SimpleNamespace(
            do_sample_frames = do_sample_frames, num_frames = num_frames,
        )
    )


def test_rate_travels_in_video_metadata_on_transformers_5(monkeypatch):
    monkeypatch.setattr(vu, "_PROCESSOR_TAKES_VIDEO_METADATA", True)
    monkeypatch.setattr(vu, "_VIDEO_RATE_FROM_METADATA", True)
    videos = [[_video(4)], [_video(6), _video(2)]]
    out_videos, kwargs = vu.video_processor_kwargs(_processor(), videos, [2.0, 1.0, 4.0])
    assert out_videos is not None and [len(v) for v in out_videos] == [1, 2]
    assert "fps" not in kwargs  # a list fps= is rejected by transformers 5
    meta = kwargs["video_metadata"]
    assert [m["fps"] for m in meta] == [2.0, 1.0, 4.0]
    assert [m["total_num_frames"] for m in meta] == [4, 6, 2]
    assert meta[1]["frames_indices"] == list(range(6))
    assert meta[0]["duration"] == 2.0


def test_transformers_4_keeps_fps_for_qwen2_5_vl(monkeypatch):
    monkeypatch.setattr(vu, "_PROCESSOR_TAKES_VIDEO_METADATA", True)
    monkeypatch.setattr(vu, "_VIDEO_RATE_FROM_METADATA", False)
    _, kwargs = vu.video_processor_kwargs(_processor(), [[_video(4)], [_video(4)]], [2.0, 2.0])
    assert kwargs["fps"] == 2.0
    assert len(kwargs["video_metadata"]) == 2


def test_no_metadata_support_falls_back_to_fps(monkeypatch):
    monkeypatch.setattr(vu, "_PROCESSOR_TAKES_VIDEO_METADATA", False)
    videos = [[_video(4)]]
    out_videos, kwargs = vu.video_processor_kwargs(_processor(), videos, [2.0])
    assert out_videos is videos
    assert kwargs == {"fps": 2.0}


def test_fixed_count_sampler_gets_its_frames_from_us(monkeypatch):
    monkeypatch.setattr(vu, "_PROCESSOR_TAKES_VIDEO_METADATA", True)
    monkeypatch.setattr(vu, "_VIDEO_RATE_FROM_METADATA", True)
    long_video = torch.arange(10).view(10, 1, 1, 1).expand(10, 3, 2, 2)
    out_videos, kwargs = vu.video_processor_kwargs(
        _processor(do_sample_frames = True, num_frames = 4), [[long_video], [_video(3)]], [2.0, 2.0],
    )
    assert kwargs["do_sample_frames"] is False
    assert out_videos[0][0].shape[0] == 4
    assert out_videos[0][0][:, 0, 0, 0].tolist() == [0, 3, 6, 9]
    assert kwargs["video_metadata"][0]["frames_indices"] == [0, 3, 6, 9]
    assert kwargs["video_metadata"][0]["total_num_frames"] == 10
    assert out_videos[1][0].shape[0] == 3


def test_mismatched_rates_fall_back_to_fps(monkeypatch):
    monkeypatch.setattr(vu, "_PROCESSOR_TAKES_VIDEO_METADATA", True)
    _, kwargs = vu.video_processor_kwargs(_processor(), [[_video(4)], [_video(4)]], [2.0])
    assert kwargs == {"fps": 2.0}


def test_real_qwen2_5_vl_processor_gets_the_rate():
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
    )
    text = "<|vision_start|><|video_pad|><|vision_end|>hi"
    videos = [[torch.randint(0, 255, (4, 3, 56, 56), dtype = torch.uint8)]]
    videos, kwargs = vu.video_processor_kwargs(processor, videos, [1.0])
    out = processor(text = [text], videos = videos, return_tensors = "pt", **kwargs)
    assert out["second_per_grid_ts"].tolist() == [2.0]


def test_collator_puts_qwen2_5_vl_frames_at_their_real_rate():
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    name = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
    processor = transformers.AutoProcessor.from_pretrained(name)
    config = transformers.AutoConfig.from_pretrained(name)
    model = transformers.AutoModelForImageTextToText.from_config(config)
    collator = vu.UnslothVisionDataCollator(model, processor)
    frames = [Image.new("RGB", (56, 56), (i * 40, 0, 0)) for i in range(4)]
    row = {"messages": [
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": 1.0},
            {"type": "text", "text": "What changes?"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "It gets redder."}]},
    ]}
    batch = collator([row])
    # 4 frames at 1 fps, temporal_patch_size 2: each grid step spans 2 seconds, not 2 / 24.
    assert batch["second_per_grid_ts"].tolist() == [2.0]
