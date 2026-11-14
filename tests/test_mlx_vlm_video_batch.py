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

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime, not the tests/mlx_simulation torch shim", allow_module_level=True)

from unsloth_zoo.mlx.utils import (
    _extract_vlm_images,
    _extract_vlm_media,
    _extract_vlm_pc_media,
    _processor_vlm_inputs,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, **kwargs):
        return {"input_ids": np.ones((len(text), 4), dtype = np.int32)}


class _FakeVideoProcessor:
    """Records processor kwargs; videos land in batch the way Qwen-VL processors emit them."""

    def __init__(self, accepts_video = True):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = object()
        self.seen = None
        self._accepts_video = accepts_video

    def _call(self, text, padding, return_tensors, add_special_tokens, videos = None, fps = None, **kwargs):
        self.seen = {"videos": videos, "fps": fps, **kwargs}
        rows = len(text)
        return {
            "input_ids": np.ones((rows, 8), dtype = np.int32),
            "attention_mask": np.ones((rows, 8), dtype = np.int32),
            "pixel_values_videos": np.zeros((4, 8), dtype = np.float32) if videos else None,
            "video_grid_thw": np.array([[1, 2, 2]]) if videos else None,
        }

    def __call__(self, text = None, padding = None, return_tensors = None, add_special_tokens = None, videos = None, fps = None, **kwargs):
        if not self._accepts_video and videos is not None:
            raise AssertionError("videos kwarg must be filtered before reaching __call__")
        return self._call(text, padding, return_tensors, add_special_tokens, videos, fps, **kwargs)


class _NoVideoProcessor(_FakeVideoProcessor):
    def __init__(self):
        super().__init__(accepts_video = False)

    def __call__(self, text = None, padding = None, return_tensors = None, add_special_tokens = None):
        return self._call(text, padding, return_tensors, add_special_tokens, None, None)


def _video_messages(path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": path},
                {"type": "text", "text": "what happens?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "it rolls"}]},
    ]


def _bare_video_messages():
    return [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": "what happens?"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "it rolls"}]},
    ]


def test_video_part_payload_decodes_through_fetch_video(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    calls = []

    def fake_fetch_video(info, **kwargs):
        calls.append((info, kwargs))
        return np.zeros((4, 8, 8, 3), dtype = np.uint8), 1.5

    monkeypatch.setattr(vu, "fetch_video", fake_fetch_video)
    monkeypatch.setattr(
        vu, "process_vision_info", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pvi unused"))
    )

    images, videos, video_kwargs = _extract_vlm_media(
        {}, _video_messages("clips/a.mp4"), None,
    )
    assert images == []
    assert len(videos) == 1
    assert video_kwargs["fps"] == [1.5]
    assert calls and calls[0][0] == {"type": "video", "video": "clips/a.mp4"}


def test_top_level_videos_column_and_bare_placeholder(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 2.0))

    _, videos, video_kwargs = _extract_vlm_media(
        {"videos": "clips/b.mp4"}, _bare_video_messages(), None,
    )
    assert videos == [decoded]
    assert video_kwargs["fps"] == [2.0]


def test_singular_video_column_falls_back_for_bare_placeholder(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 2.0))

    _, videos, _ = _extract_vlm_media(
        {"video": "clips/c.mp4"}, _bare_video_messages(), None,
    )
    assert videos == [decoded]


def test_image_only_rows_are_unchanged():
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (8, 8))
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    images, videos, video_kwargs = _extract_vlm_media({"image": img}, messages, None)
    assert images == [img]
    assert videos == []
    assert video_kwargs == {}
    assert _extract_vlm_images({"image": img}, messages, None) == [img]


def test_no_media_row_stays_empty():
    images, videos, video_kwargs = _extract_vlm_media(
        {}, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], None,
    )
    assert images == [] and videos == [] and video_kwargs == {}


def test_processor_receives_nested_videos_and_rate_metadata():
    processor = _FakeVideoProcessor()
    _processor_vlm_inputs(
        processor, ["t1", "t2"], [[], []], 32,
        all_videos = [["clip-a"], []],
        all_video_kwargs = [{"fps": [1.0]}, {}],
    )
    assert processor.seen["videos"] == ["clip-a"]
    # transformers 5.x reads per-clip rates from video_metadata; 4.x takes fps=.
    rate = processor.seen.get("video_metadata") or processor.seen.get("fps")
    assert rate is not None


def test_video_only_row_does_not_need_images():
    processor = _FakeVideoProcessor()
    _processor_vlm_inputs(
        processor, ["t1"], [[]], 32,
        all_videos = [["clip-a"]],
        all_video_kwargs = None,
    )
    assert processor.seen["videos"] == ["clip-a"]


def test_processor_without_videos_param_raises_clear_error():
    processor = _NoVideoProcessor()
    with pytest.raises(ValueError, match = "does not accept video"):
        _processor_vlm_inputs(
            processor, ["t1"], [[]], 32,
            all_videos = [["clip-a"]],
        )


def test_pc_media_carries_videos(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 3.0))

    prompt = _video_messages("clips/p.mp4")
    _, videos, video_kwargs = _extract_vlm_pc_media({}, prompt, None, None)
    assert videos == [decoded]
    assert video_kwargs["fps"] == [3.0]


def test_pc_media_reads_top_level_videos_behind_bare_placeholders(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 1.0))

    prompt = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": "which way?"}],
        }
    ]
    _, videos, _ = _extract_vlm_pc_media(
        {"videos": "clips/a.mp4"}, prompt, [{"role": "assistant", "content": "right"}], None,
    )
    assert videos == [decoded]


def test_pc_media_video_keys_survive_an_images_column(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 1.0))

    prompt = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": "which way?"}],
        }
    ]
    _, videos, _ = _extract_vlm_pc_media(
        {"videos": "clips/a.mp4", "images": ["x.png"]}, prompt, None, None,
    )
    assert videos == [decoded]


def _collate_row(path, answer = "it rolls"):
    # The shape _adapt_for_mlx_vlm emits: bare placeholder in content, payload at top level.
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": "what happens?"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "videos": path,
    }


def test_collate_vlm_batch_reaches_the_processor_with_videos(monkeypatch):
    import unsloth_zoo.vision_utils as vu
    from unsloth_zoo.mlx.utils import _collate_vlm_batch

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 1.0))

    processor = _FakeVideoProcessor()
    items = [_collate_row("clips/a.mp4"), _collate_row("clips/b.mp4")]
    batch = _collate_vlm_batch(items, processor, 64, None)
    assert processor.seen["videos"] == [decoded, decoded]
    assert batch.inputs["pixel_values_videos"] is not None
    assert batch.inputs["video_grid_thw"] is not None
    assert batch.inputs["input_ids"].shape[0] == 2


def test_collate_vlm_batch_mixed_video_and_text_rows(monkeypatch):
    import unsloth_zoo.vision_utils as vu
    from unsloth_zoo.mlx.utils import _collate_vlm_batch

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 1.0))

    processor = _FakeVideoProcessor()
    items = [
        _collate_row("clips/a.mp4"),
        {"messages": [{"role": "user", "content": "plain question"},
                      {"role": "assistant", "content": "plain answer"}]},
        _collate_row("clips/c.mp4"),
    ]
    batch = _collate_vlm_batch(items, processor, 64, None)
    assert processor.seen["videos"] == [decoded, decoded]
    assert batch.inputs["input_ids"].shape[0] == 3


def test_all_video_payloads_in_one_message_are_collected(monkeypatch):
    """Regression: guards must not sit inside the part loop and drop later clips."""
    import unsloth_zoo.vision_utils as vu

    calls = []

    def fake_fetch_video(info, **kwargs):
        calls.append(info["video"])
        return np.zeros((2, 4, 4, 3), dtype = np.uint8), 1.0

    monkeypatch.setattr(vu, "fetch_video", fake_fetch_video)
    monkeypatch.setattr(
        vu, "process_vision_info",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("pvi unused")),
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": "clips/a.mp4"},
            {"type": "video", "video": "clips/b.mp4"},
            {"type": "text", "text": "compare them"},
        ],
    }]
    _, videos, _ = _extract_vlm_media({}, messages, None)
    assert calls == ["clips/a.mp4", "clips/b.mp4"]
    assert len(videos) == 2


def test_per_part_video_hints_reach_fetch_video(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    calls = []
    monkeypatch.setattr(
        vu, "fetch_video",
        lambda info, **k: calls.append(info) or (np.zeros((2, 4, 4, 3), dtype = np.uint8), 4.0),
    )

    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": "clips/a.mp4", "fps": 4.0, "nframes": 8}],
    }]
    _, videos, _ = _extract_vlm_media({}, messages, None)
    assert calls == [{"type": "video", "video": "clips/a.mp4", "fps": 4.0, "nframes": 8}]
    assert len(videos) == 1


def test_top_level_video_without_placeholder_is_dropped(monkeypatch):
    """`video` is a weak key like `image`: no bare placeholder, nothing to pair with."""
    import unsloth_zoo.vision_utils as vu

    monkeypatch.setattr(
        vu, "fetch_video",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not decode")),
    )
    text_only = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    _, videos, _ = _extract_vlm_media({"video": "clips/a.mp4"}, text_only, None)
    assert videos == []


def test_video_url_and_input_video_part_aliases(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    calls = []
    monkeypatch.setattr(
        vu, "fetch_video",
        lambda info, **k: calls.append(info["video"]) or (
            np.zeros((2, 4, 4, 3), dtype = np.uint8), 1.0
        ),
    )

    for part in (
        {"type": "video_url", "video_url": "clips/a.mp4"},
        {"type": "input_video", "input_video": "clips/b.mp4"},
    ):
        calls.clear()
        messages = [{"role": "user", "content": [part, {"type": "text", "text": "?"}]}]
        _, videos, _ = _extract_vlm_media({}, messages, None)
        assert len(calls) == 1 and len(videos) == 1


def test_messageless_pc_row_keeps_top_level_videos(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 1.0))

    images, videos, _ = _extract_vlm_pc_media(
        {"videos": "clips/a.mp4"}, None, None, None,
    )
    assert images == []
    assert videos == [decoded]


@pytest.mark.parametrize("model_type", ["glm4v", "ernie4_5_moe_vl"])
def test_video_refused_for_families_whose_embedder_drops_videos(model_type):
    processor = _FakeVideoProcessor()
    processor._unsloth_model_type = model_type
    with pytest.raises(ValueError, match = "video training is not supported"):
        _processor_vlm_inputs(processor, ["t1"], [[]], 32, all_videos = [["clip-a"]])


def test_video_accepted_for_qwen_vl_stamp():
    processor = _FakeVideoProcessor()
    processor._unsloth_model_type = "qwen2_5_vl"
    _processor_vlm_inputs(processor, ["t1"], [[]], 32, all_videos = [["clip-a"]])
    assert processor.seen["videos"] == ["clip-a"]


def test_mixed_image_and_video_batch_is_refused():
    processor = _FakeVideoProcessor()
    with pytest.raises(ValueError, match = "cannot mix images and videos"):
        _processor_vlm_inputs(
            processor, ["t1", "t2"], [["img"], []], 32,
            all_videos = [[], ["clip-a"]],
        )


def test_pc_video_decode_failure_is_raised_not_trained_as_text(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    def broken(info, **kwargs):
        raise FileNotFoundError(info["video"])

    monkeypatch.setattr(vu, "fetch_video", broken)
    prompt = [{"role": "user", "content": [{"type": "video", "video": "missing.mp4"}, {"type": "text", "text": "q"}]}]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]
    with pytest.raises(FileNotFoundError, match = "missing.mp4"):
        _extract_vlm_pc_media({}, prompt, completion, None)


def test_pc_bare_video_placeholder_without_a_clip_is_refused():
    prompt = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "q"}]}]
    with pytest.raises(ValueError, match = "video placeholder but no video"):
        _extract_vlm_pc_media({}, prompt, [{"role": "assistant", "content": "a"}], None)


@pytest.mark.parametrize("images", [None, []])
def test_messageless_pc_row_keeps_videos_beside_an_empty_images_column(monkeypatch, images):
    import unsloth_zoo.vision_utils as vu

    decoded = np.zeros((2, 4, 4, 3), dtype = np.uint8)
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (decoded, 1.0))
    got_images, videos, kwargs = _extract_vlm_pc_media({"images": images, "videos": ["clips/a.mp4"]}, None, None, None)
    assert got_images == [] and len(videos) == 1 and kwargs == {"fps": [1.0]}


@pytest.mark.parametrize("model_type,module", [("qwen3_vl", "qwen3_vl"), ("qwen3_5", "qwen3_5"), ("qwen2_5_vl", "qwen2_5_vl")])
def test_video_positions_match_mlx_vlm_rope_index(model_type, module):
    """Qwen3 processors put a timestamp between temporal patches; positions must equal mlx-vlm's own."""
    import importlib
    import types
    from unsloth_zoo.mlx.utils import _vlm_positions_for_compile

    language = importlib.import_module(f"mlx_vlm.models.{module}.language")
    start, end, video, image = 90, 91, 92, 93
    frame = [start] + [video] * 4 + [end]
    if model_type == "qwen2_5_vl":
        ids = [1, 2] + [start] + [video] * 8 + [end] + [5, 6]
    else:
        ids = [1, 2, 7, 8] + frame + [7, 9] + frame + [5, 6]
    ids = np.array([ids], dtype = np.int32)
    mask = np.ones_like(ids)
    grid = np.array([[2, 4, 4]], dtype = np.int32)
    config = {"model_type": model_type, "image_token_id": image, "video_token_id": video,
              "vision_start_token_id": start, "vision_config": {"spatial_merge_size": 2}}
    ours = _vlm_positions_for_compile(
        {"input_ids": ids, "attention_mask": mask, "video_grid_thw": grid}, config,
    )["position_ids"]
    owner = types.SimpleNamespace(config = types.SimpleNamespace(
        vision_config = types.SimpleNamespace(spatial_merge_size = 2),
        image_token_id = image, video_token_id = video, vision_start_token_id = start,
    ))
    ref, _ = language.LanguageModel.get_rope_index(owner, mx.array(ids), None, mx.array(grid), mx.array(mask))
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(ref))


def test_openai_style_video_url_is_unwrapped(monkeypatch):
    import unsloth_zoo.vision_utils as vu

    seen = []
    monkeypatch.setattr(vu, "fetch_video", lambda info, **k: (seen.append(info["video"]), (np.zeros((2, 4, 4, 3), np.uint8), 1.0))[1])
    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "clips/a.mp4"}}, {"type": "text", "text": "q"}]}]
    _, videos, _ = _extract_vlm_media({}, messages, None)
    assert seen == ["clips/a.mp4"] and len(videos) == 1


def test_video_rows_without_torch_name_the_install(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a))
    with pytest.raises(ImportError, match = "pip install torch torchvision"):
        _extract_vlm_media({}, _video_messages("clips/a.mp4"), None)


@pytest.mark.parametrize("part", [
    {"type": "video_url", "video_url": "clips/a.mp4"},
    {"type": "video_url", "video_url": {"url": "clips/a.mp4"}},
    {"type": "input_video", "input_video": "clips/a.mp4"},
    {"video_url": "clips/a.mp4"},
])
def test_video_aliases_are_canonical_before_the_template_renders(part):
    from unsloth_zoo.mlx.utils import _normalize_mlx_messages

    messages = [{"role": "user", "content": [part, {"type": "text", "text": "q"}]}]
    (message,) = _normalize_mlx_messages(messages, is_vlm = True)
    assert message["content"][0] == {"type": "video", "video": "clips/a.mp4"}
