from __future__ import annotations

import re

import numpy as np
import pytest
from pathlib import Path


mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)


class _FakeTokenizer:
    pad_token_id = 0
    unk_token_id = -1
    image_token = "<image>"

    _vocab = {
        "<image>": 200,
        "<|image_pad|>": 201,
    }

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self._vocab.get(token, self.unk_token_id) for token in tokens]
        return self._vocab.get(tokens, self.unk_token_id)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = object()
    chat_template = "{{ messages }}"

    def __call__(self, text, **_kwargs):
        rows = []
        masks = []
        for idx, _ in enumerate(text):
            if idx == 0:
                row = [101, 10, 200, 11, 0]
                mask = [1, 1, 1, 1, 0]
            else:
                row = [101, 12, 13, 0, 0]
                mask = [1, 1, 1, 0, 0]
            rows.append(row)
            masks.append(mask)
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }


class _ResponseMaskFilteringProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = object()
    chat_template = "{{ messages }}"

    def __call__(self, text, **_kwargs):
        rows = []
        masks = []
        for value in text:
            if "bad" in value:
                row = [101, 10, 0, 0]
                mask = [1, 1, 0, 0]
            else:
                row = [101, 12, 13, 0]
                mask = [1, 1, 1, 0]
            rows.append(row)
            masks.append(mask)
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }


class _VisionFeatureProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = object()
    chat_template = "{{ messages }}"

    def __call__(self, text, **_kwargs):
        rows = [[101, 200, 102, 0] for _ in text]
        masks = [[1, 1, 1, 0] for _ in text]
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
            "pixel_values": np.ones((len(text), 2), dtype=np.float32),
            "image_grid_thw": np.ones((len(text), 3), dtype=np.int32),
        }


class _PromptCompletionProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = object()
    chat_template = "{{ messages }}"

    def __call__(self, text, **_kwargs):
        rows = []
        masks = []
        for value in text:
            if value == "prompt":
                row = [101, 0, 0, 0]
                mask = [1, 0, 0, 0]
            else:
                row = [101, 102, 103, 0]
                mask = [1, 1, 1, 0]
            rows.append(row)
            masks.append(mask)
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }


class _ConversationalPromptCompletionProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = object()
    chat_template = "{{ messages }}"

    def __init__(self):
        self.images_seen = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if message.get("role") == "user":
                parts.append(f"USER:{content}")
            elif message.get("role") == "assistant":
                parts.append(f"ASSISTANT:{content}")
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n".join(parts)

    def __call__(self, text, images=None, **_kwargs):
        self.images_seen.append(images)
        rows = []
        masks = []
        for value in text:
            if value == "USER:Q\nASSISTANT:":
                row = [101, 102, 0]
                mask = [1, 1, 0]
            elif value == "A":
                row = [103, 0, 0]
                mask = [1, 0, 0]
            else:
                row = [999, 0, 0]
                mask = [1, 0, 0]
            rows.append(row)
            masks.append(mask)
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }


def _finalized_collate(*args, **kwargs):
    """Direct-collate tests exercise the production staged+finalize composition."""
    from unsloth_zoo.mlx.utils import _collate_vlm_batch, _finalize_vlm_batch
    result = _collate_vlm_batch(*args, **kwargs)
    if isinstance(result, tuple):
        staged, is_pc = result
        return _finalize_vlm_batch(staged), is_pc
    return _finalize_vlm_batch(result)


def test_vlm_collate_creates_sft_labels_and_masks_special_tokens():
    from unsloth_zoo.mlx.utils import (
        _collate_vlm_batch,
        _get_vlm_ignore_token_ids,
    )

    processor = _FakeProcessor()
    ignore_ids = _get_vlm_ignore_token_ids(
        processor=processor,
        config={"image_token_id": 200},
    )
    batch = _finalized_collate(
        [{"text": "first"}, {"text": "second"}],
        processor,
        max_seq_length=8,
        image_size=16,
        ignore_token_ids=ignore_ids,
    )

    assert "labels" in batch
    assert batch["input_ids"].tolist() == [
        [101, 10, 200, 11, 0],
        [101, 12, 13, 0, 0],
    ]
    assert batch["labels"].tolist() == [
        [101, 10, -100, 11, -100],
        [101, 12, 13, -100, -100],
    ]


def test_vlm_response_mask_reapplies_special_token_masks():
    from unsloth_zoo.mlx.utils import _apply_response_mask_to_vlm_batch

    batch = {
        "input_ids": mx.array([[101, 200, 13, 0]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 0]], dtype=mx.int32),
        "labels": mx.array([[101, -100, 13, -100]], dtype=mx.int32),
    }

    def mask_fn(_batch):
        return {"labels": [[-100, 200, 13, 0]]}

    out = _apply_response_mask_to_vlm_batch(
        batch,
        mask_fn,
        ignore_token_ids=[0, 200],
    )

    assert out["labels"].tolist() == [[-100, -100, 13, -100]]


def test_vlm_response_mask_preserves_existing_labels_like_cuda():
    from unsloth_zoo.mlx.utils import _apply_response_mask_to_vlm_batch

    batch = {
        "input_ids": mx.array([[101, 12, 13, 0]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 0]], dtype=mx.int32),
        "labels": mx.array([[-100, 777, -100, -100]], dtype=mx.int64),
    }

    def mask_fn(mask_batch):
        old_labels = mask_batch["labels"].tolist()
        return {"labels": [[-100, old_labels[0][1], old_labels[0][2], -100]]}

    out = _apply_response_mask_to_vlm_batch(batch, mask_fn, ignore_token_ids=[0])

    assert out["labels"].tolist() == [[-100, 777, -100, -100]]


def test_vlm_response_mask_drops_fully_masked_rows():
    from unsloth_zoo.mlx.utils import create_vlm_batches

    def mask_fn(batch):
        labels = []
        for row in batch["input_ids"]:
            if 10 in row:
                labels.append([-100] * len(row))
            else:
                labels.append([-100, 12, 13, 0])
        return {"labels": labels}

    batches = create_vlm_batches(
        dataset=[{"text": "bad"}, {"text": "good"}],
        processor=_ResponseMaskFilteringProcessor(),
        config={},
        batch_size=2,
        max_seq_length=8,
        response_mask_fn=mask_fn,
        dataset_order="sequential",
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [[101, 12, 13, 0]]
    assert batches[0]["labels"].tolist() == [[-100, 12, 13, -100]]


def test_vlm_response_mask_filters_before_batching_like_cuda():
    from unsloth_zoo.mlx.utils import create_vlm_batches

    def mask_fn(batch):
        labels = []
        for row in batch["input_ids"]:
            if 10 in row:
                labels.append([-100] * len(row))
            else:
                labels.append([-100, 12, 13, 0])
        return {"labels": labels}

    batches = create_vlm_batches(
        dataset=[{"text": "good-1"}, {"text": "bad"}, {"text": "good-2"}],
        processor=_ResponseMaskFilteringProcessor(),
        config={},
        batch_size=2,
        max_seq_length=8,
        response_mask_fn=mask_fn,
        dataset_order="sequential",
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [
        [101, 12, 13, 0],
        [101, 12, 13, 0],
    ]
    assert batches[0]["labels"].tolist() == [
        [-100, 12, 13, -100],
        [-100, 12, 13, -100],
    ]


def test_vlm_empty_eval_padding_keeps_forward_valid_with_vision_features():
    from unsloth_zoo.mlx.utils import create_vlm_batches

    class FakeWorld:
        def rank(self): return 1
        def size(self): return 2

    batches = create_vlm_batches(
        dataset=[{"text": "0"}, {"text": "1"}, {"text": "2"}],
        processor=_VisionFeatureProcessor(),
        config={"image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        dataset_order="sequential",
        comm_group=FakeWorld(),
        distributed_pad_mode="empty",
    )

    empty_batch = batches[1]
    assert empty_batch["input_ids"].tolist()[0] == [101, 200, 102, 0]
    assert empty_batch["attention_mask"].tolist()[0] == [1, 1, 1, 0]
    assert empty_batch["labels"].tolist()[0] == [-100, -100, -100, -100]
    assert empty_batch["pixel_values"].shape[0] == 1
    assert len(empty_batch["image_grid_thw"]) == 1


def test_vlm_streaming_response_mask_skips_fully_masked_rows():
    from unsloth_zoo.mlx.utils import iterate_vlm_training_batches

    class StreamingDataset:
        def __iter__(self):
            return iter([{"text": "bad"}, {"text": "good"}])

    def mask_fn(batch):
        labels = []
        for row in batch["input_ids"]:
            if 10 in row:
                labels.append([-100] * len(row))
            else:
                labels.append([-100, 12, 13, 0])
        return {"labels": labels}

    batches = iterate_vlm_training_batches(
        dataset=StreamingDataset(),
        processor=_ResponseMaskFilteringProcessor(),
        config={},
        batch_size=2,
        max_seq_length=8,
        response_mask_fn=mask_fn,
    )
    batch = next(batches)

    assert batch["input_ids"].tolist() == [[101, 12, 13, 0]]
    assert batch["labels"].tolist() == [[-100, 12, 13, -100]]


def test_vlm_response_mask_formats_each_filtered_row_once():
    from unsloth_zoo.mlx.utils import create_vlm_batches

    calls = []

    def formatting_func(item):
        calls.append(item["text"])
        return {"text": item["text"]}

    def mask_fn(batch):
        return {"labels": [[-100, 12, 13, 0] for _ in batch["input_ids"]]}

    batches = create_vlm_batches(
        dataset=[{"text": "good-1"}, {"text": "good-2"}],
        processor=_ResponseMaskFilteringProcessor(),
        config={},
        batch_size=2,
        max_seq_length=8,
        response_mask_fn=mask_fn,
        formatting_func=formatting_func,
        dataset_order="sequential",
    )

    assert calls == ["good-1", "good-2"]
    assert len(batches) == 1


def test_vlm_filter_caches_only_kept_formatted_rows():
    from unsloth_zoo.mlx.utils import _filter_trainable_vlm_indices

    def formatting_func(item):
        return {"text": item["text"]}

    def mask_fn(batch):
        labels = []
        for row in batch["input_ids"]:
            if 10 in row:
                labels.append([-100] * len(row))
            else:
                labels.append([-100, 12, 13, 0])
        return {"labels": labels}

    kept, removed, formatted_items, _supervision = _filter_trainable_vlm_indices(
        [{"text": "bad"}, {"text": "good"}],
        [0, 1],
        _ResponseMaskFilteringProcessor(),
        {},
        max_seq_length=8,
        image_size=16,
        response_mask_fn=mask_fn,
        formatting_func=formatting_func,
    )

    assert kept == [1]
    assert removed == 1
    assert formatted_items == {1: {"text": "good"}}


def test_vlm_prompt_completion_skips_response_mask_like_cuda():
    from unsloth_zoo.mlx.utils import create_vlm_batches

    def mask_fn(_batch):
        raise AssertionError("CUDA VLM prompt/completion returns before response masking")

    batches = create_vlm_batches(
        dataset=[{"prompt": "prompt", "completion": "completion"}],
        processor=_PromptCompletionProcessor(),
        config={},
        batch_size=1,
        max_seq_length=8,
        response_mask_fn=mask_fn,
        dataset_order="sequential",
    )

    assert batches[0]["labels"].tolist() == [[-100, 101, 102, 103]]


def test_vlm_prompt_completion_honors_completion_only_loss_false():
    from unsloth_zoo.mlx.utils import _collate_vlm_batch

    default_batch = _finalized_collate(
        [{"prompt": "prompt", "completion": "completion"}],
        _PromptCompletionProcessor(),
        max_seq_length=8,
        image_size=16,
    )
    batch = _finalized_collate(
        [{"prompt": "prompt", "completion": "completion"}],
        _PromptCompletionProcessor(),
        max_seq_length=8,
        image_size=16,
        completion_only_loss=False,
    )

    assert default_batch["labels"].tolist() == [[-100, 101, 102, 103]]
    assert batch["labels"].tolist() == [[101, 101, 102, 103]]


def test_vlm_prompt_completion_conversational_uses_cuda_prompt_split():
    from unsloth_zoo.mlx.utils import _collate_vlm_batch

    processor = _ConversationalPromptCompletionProcessor()
    batch = _finalized_collate(
        [{
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "A"}]}],
        }],
        processor,
        max_seq_length=8,
        image_size=16,
    )

    assert batch["input_ids"].tolist() == [[101, 102, 103]]
    assert batch["labels"].tolist() == [[-100, -100, 103]]


def test_vlm_prompt_completion_prefers_embedded_images_like_cuda():
    from unsloth_zoo.mlx.utils import _collate_vlm_batch

    processor = _ConversationalPromptCompletionProcessor()
    _finalized_collate(
        [{
            "image": "top-level",
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "embedded"},
                    {"type": "text", "text": "Q"},
                ],
            }],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "A"}]}],
        }],
        processor,
        max_seq_length=8,
        image_size=16,
    )

    assert processor.images_seen[0] == ["embedded"]


def test_vlm_prompt_completion_uses_top_level_image_for_bare_placeholder():
    from unsloth_zoo.mlx.utils import _extract_vlm_pc_images

    messages = [{"role": "user", "content": [{"type": "image"}]}]
    assert _extract_vlm_pc_images(
        {"image": "top-level"}, messages, [], image_size=16,
    ) == ["top-level"]


def test_vlm_collate_passes_studio_top_level_image_to_processor():
    from unsloth_zoo.mlx.utils import _collate_vlm_batch

    processor = _ConversationalPromptCompletionProcessor()
    _collate_vlm_batch(
        [{
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Q"},
                ],
            }],
            "image": "top-level",
        }],
        processor,
        max_seq_length=8,
        image_size=16,
    )

    assert processor.images_seen == [["top-level"]]


def test_vlm_top_level_images_key_still_wins_over_image_key():
    from unsloth_zoo.mlx.utils import _extract_vlm_images

    assert _extract_vlm_images(
        {"images": ["plural"], "image": "singular"},
        [],
        image_size=16,
    ) == ["plural"]


def test_vlm_top_level_image_key_requires_bare_image_placeholder():
    from unsloth_zoo.mlx.utils import _extract_vlm_images

    messages = [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]
    assert _extract_vlm_images({"image": "top-level"}, messages, image_size=16) == []
    assert _extract_vlm_images({"image": "top-level"}, [], image_size=16) == []


def test_vlm_top_level_image_key_rejects_mixed_bare_placeholders():
    from unsloth_zoo.mlx.utils import _extract_vlm_images

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "video"},
        ],
    }]
    with pytest.raises(ValueError, match="image, image_url or video"):
        _extract_vlm_images({"image": "top-level"}, messages, image_size=16)


def test_vlm_image_extraction_raises_process_errors_like_cuda(monkeypatch):
    import unsloth_zoo.vision_utils as vision_utils
    from unsloth_zoo.mlx.utils import _extract_vlm_images

    def fail_process_vision_info(*_args, **_kwargs):
        raise ValueError("bad image")

    monkeypatch.setattr(
        vision_utils,
        "process_vision_info",
        fail_process_vision_info,
    )

    with pytest.raises(ValueError, match="bad image"):
        _extract_vlm_images(
            [{"role": "user", "content": [{"type": "image"}]}],
            [{"role": "user", "content": [{"type": "image"}]}],
            image_size=16,
        )


def test_vlm_prompt_completion_top_level_image_errors_are_suppressed_like_cuda(monkeypatch):
    import unsloth_zoo.vision_utils as vision_utils
    from unsloth_zoo.mlx.utils import _extract_vlm_pc_images

    def fail_process_vision_info(*_args, **_kwargs):
        raise ValueError("bad top-level image")

    monkeypatch.setattr(
        vision_utils,
        "process_vision_info",
        fail_process_vision_info,
    )

    assert _extract_vlm_pc_images({"images": ["bad"]}, [], [], image_size=16) == []


def test_vlm_prompt_completion_top_level_images_use_cuda_process_shape(monkeypatch):
    import unsloth_zoo.vision_utils as vision_utils
    from unsloth_zoo.mlx.utils import _extract_vlm_pc_images

    seen = {}

    def fake_process_vision_info(conversations, **kwargs):
        seen["conversations"] = conversations
        seen["kwargs"] = kwargs
        return ["processed"], None, {"fps": []}

    monkeypatch.setattr(
        vision_utils,
        "process_vision_info",
        fake_process_vision_info,
    )

    assert _extract_vlm_pc_images({"images": ["raw"]}, [], [], image_size=16) == ["processed"]
    assert seen == {
        "conversations": [{"image": "raw"}],
        "kwargs": {"return_video_kwargs": True},
    }


def test_vlm_prompt_completion_message_rows_do_not_fallback_to_top_level_images(monkeypatch):
    import unsloth_zoo.vision_utils as vision_utils
    from unsloth_zoo.mlx.utils import _extract_vlm_pc_images

    def fake_process_vision_info(_conversations, **_kwargs):
        return None, None, {"fps": []}

    monkeypatch.setattr(
        vision_utils,
        "process_vision_info",
        fake_process_vision_info,
    )

    assert _extract_vlm_pc_images(
        {"images": ["top-level"]},
        [{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "A"}]}],
        image_size=16,
    ) == []


def test_vlm_render_falls_back_to_content_part_templates():
    from unsloth_zoo.mlx.utils import _render_vlm_messages

    class ContentPartProcessor:
        chat_template = "parts"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            if messages and all(isinstance(part, dict) and "type" in part for part in messages):
                return "parts:" + ",".join(part["type"] for part in messages)
            raise ValueError("expected content parts")

    rendered = _render_vlm_messages(
        ContentPartProcessor(),
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}],
    )

    assert rendered == "parts:image,text"


def test_vlm_render_falls_back_to_text_templates():
    from unsloth_zoo.mlx.utils import _render_vlm_messages

    class TextTemplateProcessor:
        chat_template = "text"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            if messages and all(isinstance(message.get("content"), str) for message in messages):
                return "|".join(message["content"] for message in messages)
            raise ValueError("expected text content")

    rendered = _render_vlm_messages(
        TextTemplateProcessor(),
        [
            {"role": "user", "content": [{"type": "text", "text": "Q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ],
    )

    assert rendered == "Q|A"


def test_vlm_processor_inputs_flattens_qwen_style_images():
    from unsloth_zoo.mlx.utils import _processor_vlm_inputs

    class QwenLikeProcessor:
        __module__ = "mlx_vlm.models.qwen3_vl.processing_qwen3_vl"

        def __init__(self):
            self.seen_images = None

        def __call__(self, text, images=None, **_kwargs):
            self.seen_images = images
            return {
                "input_ids": np.ones((len(text), 2), dtype=np.int32),
                "attention_mask": np.ones((len(text), 2), dtype=np.int32),
            }

    processor = QwenLikeProcessor()
    _processor_vlm_inputs(processor, ["a", "b"], [["img0"], ["img1"]], 8)

    assert processor.seen_images == ["img0", "img1"]


def test_vlm_processor_inputs_preserves_nested_image_processors():
    from unsloth_zoo.mlx.utils import _processor_vlm_inputs

    class PixtralLikeProcessor:
        __module__ = "mlx_vlm.models.pixtral.processing_pixtral"

        def __init__(self):
            self.seen_images = None

        def __call__(self, text, images=None, **_kwargs):
            self.seen_images = images
            return {
                "input_ids": np.ones((len(text), 2), dtype=np.int32),
                "attention_mask": np.ones((len(text), 2), dtype=np.int32),
            }

    processor = PixtralLikeProcessor()
    _processor_vlm_inputs(processor, ["a", "b"], [["img0"], ["img1", "img2"]], 8)

    assert processor.seen_images == [["img0"], ["img1", "img2"]]


@pytest.mark.parametrize(
    "module_name, expected",
    (
        ("mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl", ["img0", "img1"]),
        ("mlx_vlm.models.qwen3_5.processing_qwen3_vl", ["img0", "img1"]),
        ("mlx_vlm.models.gemma4.processing_gemma4", ["img0", "img1"]),
        ("mlx_vlm.models.gemma3.processing_gemma3", [["img0"], ["img1"]]),
        ("mlx_vlm.models.idefics3.processing_idefics3", [["img0"], ["img1"]]),
        ("mlx_vlm.models.deepseek_vl_v2.processing_deepsek_vl_v2", [["img0"], ["img1"]]),
        ("mlx_vlm.models.falcon_ocr.processing_falcon_ocr", [["img0"], ["img1"]]),
    ),
)
def test_vlm_processor_inputs_known_arch_image_layouts(module_name, expected):
    from unsloth_zoo.mlx.utils import _processor_vlm_inputs

    def call(self, text, images=None, **_kwargs):
        self.seen_images = images
        return {
            "input_ids": np.ones((len(text), 2), dtype=np.int32),
            "attention_mask": np.ones((len(text), 2), dtype=np.int32),
        }

    Processor = type("Processor", (), {"__module__": module_name, "__call__": call})
    processor = Processor()
    _processor_vlm_inputs(processor, ["a", "b"], [["img0"], ["img1"]], 8)

    assert processor.seen_images == expected


def test_vlm_resize_int_does_not_upscale_small_images():
    from PIL import Image

    from unsloth_zoo.mlx.utils import _resize_vlm_images

    image = Image.new("RGB", (512, 512))
    resized = _resize_vlm_images([image], 896)

    assert resized[0].size == (512, 512)


def test_vlm_resize_int_downscales_large_images_like_cuda_collator():
    from PIL import Image

    from unsloth_zoo.mlx.utils import _resize_vlm_images

    image = Image.new("RGB", (1024, 512))
    resized = _resize_vlm_images([image], 512)

    assert resized[0].size == (512, 256)


def test_vlm_processor_inputs_retries_duplicate_add_special_tokens():
    from unsloth_zoo.mlx.utils import _processor_vlm_inputs

    class PaddleLikeProcessor:
        __module__ = "mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl"

        def __init__(self):
            self.calls = []

        def __call__(self, text, images=None, **kwargs):
            self.calls.append(dict(kwargs))
            if "add_special_tokens" in kwargs:
                raise TypeError(
                    "got multiple values for keyword argument 'add_special_tokens'"
                )
            return {
                "input_ids": np.ones((len(text), 2), dtype=np.int32),
                "attention_mask": np.ones((len(text), 2), dtype=np.int32),
            }

    processor = PaddleLikeProcessor()
    _processor_vlm_inputs(processor, ["a"], [["img0"]], 8)

    assert "add_special_tokens" in processor.calls[0]
    assert "add_special_tokens" not in processor.calls[1]


def test_vlm_processor_inputs_retry_only_exact_pytorch_output_error():
    import torch

    from unsloth_zoo.mlx.utils import _call_vlm_processor

    class Batch(dict):
        pass

    calls = []
    def pytorch_only(*_args, return_tensors=None, **_kwargs):
        calls.append(return_tensors)
        if return_tensors != "pt":
            raise ValueError("Only returning PyTorch tensors is currently supported.")
        return Batch({
            "ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "nested": [torch.tensor([1.5], dtype=torch.float16), ("meta",)],
        })

    converted = _call_vlm_processor(
        pytorch_only, (), {"return_tensors": "np"}
    )
    assert calls == ["np", "pt"] and isinstance(converted, Batch)
    assert converted["ids"].dtype == np.int64
    assert converted["nested"][0].dtype == np.float16
    assert converted["nested"][1] == ("meta",)

    converted = _call_vlm_processor(
        pytorch_only, (), {"return_tensors": "mlx"}
    )
    # why: a sibling module can swap the util's `mx` for the simulation stub,
    # whose `array` is a factory rather than a type.
    from unsloth_zoo.mlx import utils as mlx_utils

    if "mlx_simulation" not in str(getattr(mlx_utils.mx, "__file__", "")):
        assert isinstance(converted["ids"], mx.array)
        assert converted["ids"].dtype == mx.int64

    native, native_calls = object(), []
    assert _call_vlm_processor(lambda **kw: native_calls.append(kw["return_tensors"]) or native, (), {"return_tensors": "np"}) is native
    assert native_calls == ["np"]
    unrelated = ValueError("unrelated")
    with pytest.raises(ValueError) as raised:
        _call_vlm_processor(lambda **_kwargs: (_ for _ in ()).throw(unrelated), (), {"return_tensors": "np"})
    assert raised.value is unrelated


def test_vlm_processor_multimodal_token_type_ownership():
    from unsloth_zoo.mlx.utils import _processor_vlm_inputs
    def call(self, text, **kwargs):
        self.request = kwargs.get("return_mm_token_type_ids")
        output = {"input_ids": np.ones((len(text), 2))}
        if self.native_types or self.request is True:
            output["token_type_ids"] = np.zeros((len(text), 2))
        return output
    def processor(model, name="Processor", base=object, native_types=False, inherit_call=False):
        attrs = {"__module__": f"mlx_vlm.models.{model}.processing", "native_types": native_types}
        if not inherit_call:
            attrs["__call__"] = call
        return type(name, (base,), attrs)()
    gemma3n = processor("gemma3n", "Gemma3nProcessor", native_types=True)
    relocated = processor("custom", "CustomGemma3nProcessor", type(gemma3n), True, True)
    owners = [processor("gemma3", "Gemma3NextProcessor")]
    for model in ("gemma3", "gemma4", "qwen3_vl", "qwen3_5"):
        native = processor(model)
        owners.extend((native, processor("custom", f"Custom{model}Processor", type(native), inherit_call=True)))
    owners.append(processor("custom", "Gemma3nFlagProcessor", type(gemma3n)))
    owners.append(processor("custom", "Gemma3nProcessor", type(gemma3n)))
    for candidate in owners:
        assert "token_type_ids" in _processor_vlm_inputs(candidate, ["x"], [[]], 8) and candidate.request is True
    for candidate in (gemma3n, relocated):
        assert "token_type_ids" in _processor_vlm_inputs(candidate, ["x"], [[]], 8) and candidate.request is None


def test_deepseek_ocr_loader_patches_removed_llama_flash_attention(monkeypatch):
    import sys
    import types

    from unsloth_zoo.mlx.loader import _patch_deepseek_ocr_transformers_import_compat

    llama_module = types.SimpleNamespace(LlamaAttention=object)
    package = types.ModuleType("transformers.models.llama")
    package.modeling_llama = llama_module
    monkeypatch.setitem(sys.modules, "transformers.models.llama", package)
    import transformers.utils.import_utils as import_utils
    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)

    _patch_deepseek_ocr_transformers_import_compat("deepseekocr")

    assert llama_module.LlamaFlashAttention2 is llama_module.LlamaAttention
    assert import_utils.is_torch_fx_available() is False


def test_deepseek_rendering_repairs_missing_image_token():
    from unsloth_zoo.mlx.utils import _render_vlm_messages

    class DeepseekProcessor:
        __module__ = "mlx_vlm.models.deepseekocr.processing_deepseekocr"
        image_token = "<image>"
        chat_template = "deepseek"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "question"

    text = _render_vlm_messages(
        DeepseekProcessor(),
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "question"}]}],
    )

    assert text == "<image>question"


def test_token_expansion_masks_inserted_label_positions():
    from unsloth_zoo.mlx.utils import _expand_token_runs

    input_ids = mx.array([[1, 200, 3, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)
    labels = mx.array([[1, -100, 3, -100]], dtype=mx.int32)

    expanded_ids, expanded_mask, expanded_labels = _expand_token_runs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        replacements_by_batch=(((1, 2, 200, 3),),),
        labels=labels,
    )

    assert expanded_ids.tolist() == [[1, 200, 200, 200, 3, 0]]
    assert expanded_mask.tolist() == [[1, 1, 1, 1, 1, 0]]
    assert expanded_labels.tolist() == [[1, -100, -100, -100, 3, -100]]


def test_mlx_trainer_does_not_attach_processor_for_loss_masking():
    trainer_source = (
        Path(__file__).resolve().parents[1]
        / "unsloth_zoo"
        / "mlx"
        / "trainer.py"
    ).read_text()

    assert "self.model._processor =" not in trainer_source
    assert "_get_vlm_ignore_token_ids(" in trainer_source


def test_text_only_vlm_wrapper_uses_text_training_path():
    from unsloth_zoo.mlx.utils import _is_vlm_model

    class TextOnlyVLMWrapper:
        _is_vlm_model = True
        _unsloth_text_only_vlm = True
        language_model = object()
        vision_tower = object()

    assert _is_vlm_model(TextOnlyVLMWrapper()) is False


def test_gemma3_vlm_cce_does_not_forward_outer_product_attention_mask():
    from types import SimpleNamespace

    from unsloth_zoo.mlx.utils import _unpack_embed_result

    embeds = mx.ones((1, 4, 8))
    outer_mask = mx.ones((1, 1, 4, 4), dtype=mx.int32)
    embed_result = SimpleNamespace(
        inputs_embeds=embeds,
        attention_mask_4d=outer_mask,
    )

    _merged, kwargs = _unpack_embed_result(
        embed_result,
        SimpleNamespace(config=SimpleNamespace(model_type="gemma3")),
    )

    assert "attention_mask_4d" not in kwargs


def test_non_gemma3_vlm_cce_keeps_embedder_attention_mask():
    from types import SimpleNamespace

    from unsloth_zoo.mlx.utils import _unpack_embed_result

    embeds = mx.ones((1, 4, 8))
    outer_mask = mx.ones((1, 1, 4, 4), dtype=mx.int32)
    embed_result = SimpleNamespace(
        inputs_embeds=embeds,
        attention_mask_4d=outer_mask,
    )

    _merged, kwargs = _unpack_embed_result(
        embed_result,
        SimpleNamespace(config=SimpleNamespace(model_type="gemma3n")),
    )

    assert kwargs["attention_mask_4d"] is outer_mask


def test_gemma_image_attention_mask_allows_bidirectional_image_block():
    from unsloth_zoo.mlx.utils import _build_gemma_image_attention_mask

    token_type_ids = mx.array([[0, 1, 1, 0]], dtype=mx.int32)
    mask = _build_gemma_image_attention_mask(token_type_ids)[0, 0].tolist()

    assert mask[0] == [True, False, False, False]
    assert mask[1] == [True, True, True, False]
    assert mask[2] == [True, True, True, False]
    assert mask[3] == [True, True, True, True]


def test_gemma3_vlm_hidden_stack_uses_image_mask_and_embed_scale():
    from types import SimpleNamespace

    from unsloth_zoo.mlx.utils import _forward_text_hidden_states

    class RecordingLayer:
        def __init__(self):
            self.seen_h = None
            self.seen_mask = None

        def __call__(self, h, mask, _cache):
            self.seen_h = h
            self.seen_mask = mask
            return h

    class IdentityNorm:
        weight = mx.ones((4,), dtype=mx.float32)

        def __call__(self, h):
            return h

    layer = RecordingLayer()
    stack = SimpleNamespace(
        config=SimpleNamespace(model_type="gemma3_text", hidden_size=4),
        embed_tokens=object(),
        layers=[layer],
        norm=IdentityNorm(),
        sliding_window_pattern=1,
        window_size=2,
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=stack))
    embeds = mx.ones((1, 4, 4), dtype=mx.float32)
    token_type_ids = mx.array([[0, 1, 1, 0]], dtype=mx.int32)

    out = _forward_text_hidden_states(
        model,
        mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        inputs_embeds=embeds,
        token_type_ids=token_type_ids,
    )

    assert mx.allclose(out, mx.full((1, 4, 4), 2.0))
    assert mx.allclose(layer.seen_h, mx.full((1, 4, 4), 2.0))
    assert layer.seen_mask[0, 0].tolist()[1] == [True, True, True, False]


class _LifecycleVLMRows:
    """Unsized replayable VLM source with consumption/epoch instrumentation."""

    def __init__(self, count=6):
        self.count, self.pulls, self.epochs = count, 0, []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        def _gen():
            for i in range(self.count):
                self.pulls += 1
                yield {"text": str(101 + i)}
        return _gen()


def _lazy_vlm(dataset, **kwargs):
    from unsloth_zoo.mlx.utils import iterate_vlm_training_batches
    options = dict(processor=_FakeProcessor(), config={}, batch_size=2, max_seq_length=8)
    return iterate_vlm_training_batches(dataset=dataset, **(options | kwargs))


def test_vlm_lazy_lifecycle_replay_oneshot_and_fast_forward():
    source = _LifecycleVLMRows(6)
    stream = _lazy_vlm(source)
    assert source.pulls == 0 and source.epochs == []      # construction-lazy
    first = next(stream)["input_ids"].tolist()
    assert source.pulls == 2 and source.epochs == [0]     # bounded first yield
    for _ in range(2):
        next(stream)
    assert next(stream)["input_ids"].tolist() == first    # replay restart
    assert source.epochs == [0, 1]

    rows = [{"text": str(101 + i)} for i in range(4)]
    one_shot = _lazy_vlm(iter(list(rows)))
    for _ in range(2):
        next(one_shot)
    with pytest.raises(RuntimeError, match="one-shot"):
        next(one_shot)

    consumed = []
    def counting():
        for row in rows:
            consumed.append(row)
            yield row
    with pytest.raises(RuntimeError, match="replayable"):
        next(_lazy_vlm(counting(), require_replayable=True))
    assert consumed == []                                  # rejected pre-consumption

    steady = _lazy_vlm(_LifecycleVLMRows(6))
    uninterrupted = [next(steady)["input_ids"].tolist() for _ in range(5)]
    resumed = _lazy_vlm(_LifecycleVLMRows(6), require_replayable=True)
    for _ in range(4):
        next(resumed)
    assert next(resumed)["input_ids"].tolist() == uninterrupted[4]


def test_vlm_lazy_declared_epochs_and_pre_consumption_rejections():
    exact = _lazy_vlm(_LifecycleVLMRows(4), expected_rows_per_pass=4)
    seen = [next(exact)["input_ids"].tolist() for _ in range(4)]
    assert seen[0] == seen[2]                              # deferred final + replay

    overrun, emitted = _lazy_vlm(_LifecycleVLMRows(4), expected_rows_per_pass=3), []
    with pytest.raises(ValueError, match="declared length"):
        while True:
            emitted.append(next(overrun))
    assert len(emitted) == 1                               # deferred final withheld

    underrun = _lazy_vlm(_LifecycleVLMRows(4), expected_rows_per_pass=5)
    for _ in range(2):
        next(underrun)
    with pytest.raises(ValueError, match="declared length"):
        next(underrun)

    def mask_fn(batch):
        return {"labels": [[-100] * len(r) if r[0] == 101 else list(r)
                           for r in batch["input_ids"]]}
    filtered = _lazy_vlm(_LifecycleVLMRows(4), expected_rows_per_pass=4,
                         processor=_ResponseMaskFilteringProcessor(),
                         response_mask_fn=mask_fn)
    with pytest.raises(ValueError, match="exactly one trainable"):
        for _ in range(4):
            next(filtered)

    class FakeWorld:
        def rank(self): return 0
        def size(self): return 2
    ddp_probe = _LifecycleVLMRows(4)
    with pytest.raises(ValueError, match="DDP training"):
        next(_lazy_vlm(ddp_probe, comm_group=FakeWorld()))
    assert ddp_probe.pulls == 0 and ddp_probe.epochs == []
    with pytest.raises(ValueError, match="torch_randperm"):
        next(_lazy_vlm(_LifecycleVLMRows(4), dataset_order="torch_randperm"))


def test_vlm_sized_index_routing_guard_and_cleanup():
    from unsloth_zoo.mlx.utils import create_vlm_batches, iterate_vlm_training_batches

    class SizedIterableMap:
        """Map-style dataset that ALSO iterates (must stay on the sized path)."""
        def __init__(self):
            self.rows = [{"text": str(101 + i)} for i in range(4)]
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx): return self.rows[idx]
        def __iter__(self): return iter(self.rows)


    # A sized iterable-map hybrid passes the DDP gate and batches via the sized path.
    sized_trainer = _vlm_trainer_shell_for(world_size=2, dataset=SizedIterableMap())
    _batches, sized_stream = sized_trainer._prepare_data(is_vlm=True)
    assert next(sized_stream)["input_ids"].shape[0] == 1
    knob_trainer = _vlm_trainer_shell_for(world_size=1, dataset=SizedIterableMap())
    knob_trainer.args.streaming_prefetch_batches = 1
    _b2, s2 = knob_trainer._prepare_data(is_vlm=True)  # notice path must not crash
    assert next(s2)["input_ids"].shape[0] == 1
    # A genuinely unsized source is rejected before consumption.
    lazy_probe = _LifecycleVLMRows(4)
    with pytest.raises(ValueError, match="DDP training"):
        _vlm_trainer_shell_for(world_size=2, dataset=lazy_probe)._prepare_data(is_vlm=True)
    assert lazy_probe.pulls == 0 and lazy_probe.epochs == []

    with pytest.raises(ValueError, match="__len__ and __getitem__"):
        create_vlm_batches(dataset=_LifecycleVLMRows(4), processor=_FakeProcessor(),
                           config={}, batch_size=2, max_seq_length=8)

    class LenOnlyRows(_LifecycleVLMRows):
        def __len__(self): return self.count
    assert next(_lazy_vlm(LenOnlyRows(4)))["input_ids"].shape[0] == 2

    import torch.utils.data as tud
    class TorchStyleRows(_LifecycleVLMRows, tud.IterableDataset):
        def __init__(self): _LifecycleVLMRows.__init__(self, 4)
        def __len__(self): return self.count
    assert next(_lazy_vlm(TorchStyleRows()))["input_ids"].shape[0] == 2

    class GetattrProxyRows(_LifecycleVLMRows):
        """Instance __getattr__ proxies must not classify as sized."""
        def __len__(self): return self.count
        def __getattr__(self, name):
            if name == "__getitem__":
                return lambda idx: {"text": "999"}
            raise AttributeError(name)
    proxied = GetattrProxyRows(4)
    assert next(_lazy_vlm(proxied))["input_ids"].shape[0] == 2
    assert proxied.pulls == 2                              # iterated, not indexed

    closed = []
    class RecordingCursor:
        def __init__(self, name, rows, explode_on_close=False):
            self._name, self._rows, self._explode = name, iter(rows), explode_on_close
        def __iter__(self): return self
        def __next__(self): return next(self._rows)
        def close(self):
            closed.append(self._name)
            if self._explode:
                raise RuntimeError("close exploded")
    class ClosingRows:
        def __init__(self): self.handed = 0
        def __iter__(self):
            self.handed += 1
            name = "serving" if self.handed == 1 else "cached"
            return RecordingCursor(name, [{"text": "101"}, {"text": "boom"}],
                                   explode_on_close=(name == "serving"))
    class ExplodingProcessor(_FakeProcessor):
        def __call__(self, text, **kwargs):
            if any("boom" in str(item) for item in text):
                raise RuntimeError("processor exploded")
            return super().__call__(text, **kwargs)
    stream = _lazy_vlm(ClosingRows(), processor=ExplodingProcessor(), batch_size=1,
                       require_replayable=True)
    next(stream)
    with pytest.raises(RuntimeError, match="processor exploded"):
        next(stream)
    stream.close()
    assert sorted(closed) == ["cached", "serving"]         # both closed, error kept


def test_vlm_host_label_authority_and_staged_finalize():
    import numpy as np
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import (
        _HostStagedVLMBatch, _collate_vlm_batch, _finalize_vlm_batch,
        _stage_vlm_label_mask_np, _vlm_inputs_host_valued,
        _RAW_INPUT_IDS_FOR_LABELS,
    )

    # Mask ignore ids, attention zeros and existing -100s; floats narrow as finalized.
    mask = _stage_vlm_label_mask_np(
        {"input_ids": np.array([[101, 200, 2, 7]]),
         "attention_mask": np.array([[1, 1, 0, 1]])},
        ignore_token_ids=[200])
    assert mask.tolist() == [[False, True, True, False]]
    fractional = _stage_vlm_label_mask_np(
        {"input_ids": [[7, 8, 9]], "attention_mask": [[1, 0.5, 0.99999999]]})
    assert fractional.tolist() == [[False, True, False]]

    # Non-integer id streams never stage: legacy path in sync, reject in producer.
    from unsloth_zoo.mlx.utils import _vlm_ids_integer_host
    assert _vlm_ids_integer_host({"input_ids": np.array([[1, 2]])}) is True
    assert _vlm_ids_integer_host(
        {"input_ids": np.array([[1.0, 2.0]], dtype=np.float64)}) is False
    assert _vlm_ids_integer_host({"input_ids": [[1, 2.5]]}) is False
    assert _vlm_ids_integer_host({"input_ids": [[1, True]]}) is False
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        _collate_vlm_batch(
            [{"text": "101", "images": [mx.array([1.0])]}],
            _FakeProcessor(), 8, None, reject_mlx_valued=True)
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        _collate_vlm_batch(
            [{"text": "101"}], _FakeProcessor(), 8, None,
            reject_mlx_valued=True,
            formatting_func=lambda item: {
                "text": item["text"], "images": [mx.array([1.0])],
            })
    class FloatProcessor(_FakeProcessor):
        def __call__(self, text, **kwargs):
            out = super().__call__(text, **kwargs)
            out["input_ids"] = np.asarray(out["input_ids"], dtype=np.float32)
            return out
    floaty = _collate_vlm_batch([{"text": "101"}], FloatProcessor(), 8, None)
    assert floaty.prefinalized is not None  # sync legacy route
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        _collate_vlm_batch([{"text": "101"}], FloatProcessor(), 8, None,
                           reject_mlx_valued=True)
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        _collate_vlm_batch(
            [{"prompt": "101", "completion": "102"}], FloatProcessor(), 8,
            None, reject_mlx_valued=True)

    # Finalized values ride the same converted ids as legacy, incl. int64 widening.
    plain = _finalized_collate([{"text": "101"}], _FakeProcessor(), 8, None)
    assert plain["labels"].dtype == plain["input_ids"].dtype
    assert _RAW_INPUT_IDS_FOR_LABELS not in plain
    pc = _finalized_collate(
        [{"prompt": "101", "completion": "102"}], _FakeProcessor(), 8, None)
    assert pc["labels"].dtype == mx.int64  # legacy completion branch widens
    off = _finalized_collate(
        [{"prompt": "101", "completion": "102"}], _FakeProcessor(), 8, None,
        completion_only_loss=False)
    assert off["labels"].dtype == off["input_ids"].dtype

    # MLX-returning processors flag host_valued=False, nested too; reject raises first.
    assert _vlm_inputs_host_valued(
        {"input_ids": np.array([[1]])}) is True
    assert _vlm_inputs_host_valued(
        {"pixel_values": {"tensor": mx.array([1])}}) is False

    class MxProcessor(_FakeProcessor):
        def __call__(self, text, **kwargs):
            out = super().__call__(text, **kwargs)
            return {k: mx.array(v) for k, v in out.items()}
    # Processor-owned MLX outputs stage opaquely: labels defer to the finalizer.
    opaque = _collate_vlm_batch([{"text": "101"}], MxProcessor(), 8, None,
                                reject_mlx_valued=True)
    assert opaque.host_valued is False and opaque.label_mask is None
    finalized_opaque = _finalize_vlm_batch(opaque)
    assert "labels" in finalized_opaque
    from unsloth_zoo.mlx.utils import _build_response_masked_vlm_batch as _brm
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        _brm([{"text": "101"}], _FakeProcessor(), {}, 8, None,
             response_mask_fn=lambda b: {"labels": b["input_ids"]},
             yield_host_staged=True)
    # Zero-touch iterator rejection: no set_epoch, no pull, no processor call.
    from unsloth_zoo.mlx.utils import _iterate_lazy_vlm_training_batches
    probe = _LifecycleVLMRows(4)
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        next(_iterate_lazy_vlm_training_batches(
            probe, _FakeProcessor(), {}, 2, 8,
            response_mask_fn=lambda b: {"labels": b["input_ids"]},
            yield_host_staged=True))
    assert probe.pulls == 0 and probe.epochs == []

    # Value-carrying closures finalize through the legacy pipeline unchanged.
    from unsloth_zoo.mlx.utils import _build_response_masked_vlm_batch
    def _value_closure(mask_batch):
        width = len(mask_batch["input_ids"][0])
        return {"labels": [[-100, 777] + [-100] * (width - 2)]}
    routed = _build_response_masked_vlm_batch(
        [{"text": "101"}], _FakeProcessor(), {}, 8, None,
        response_mask_fn=_value_closure,
    )
    assert routed["labels"].tolist()[0][1] == 777  # custom value preserved
    assert routed["labels"].dtype == mx.int64  # legacy value-pipeline dtype

    # The raw wide-id carrier survives finalize: the closure sees the uint32 id.
    class WideProcessor(_FakeProcessor):
        def __call__(self, text, **kwargs):
            out = super().__call__(text, **kwargs)
            ids = np.asarray(out["input_ids"], dtype=np.uint32)
            ids[0, 0] = np.uint32(2**32 - 100)
            out["input_ids"] = ids
            return out
    seen = {}
    def identity_closure(mask_batch):
        seen["first"] = int(mask_batch["input_ids"][0][0])
        return {"labels": [list(row) for row in mask_batch["input_ids"]]}
    _build_response_masked_vlm_batch(
        [{"text": "101"}], WideProcessor(), {}, 8, None,
        response_mask_fn=identity_closure)
    assert seen["first"] == 2**32 - 100


def _vlm_trainer_shell_for(dataset, world_size=1, prefetch=0):
    import types as _types
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        per_device_train_batch_size=1, max_seq_length=8, streaming=True,
        streaming_prefetch_batches=prefetch,
    )
    trainer.model = _types.SimpleNamespace(_config={})
    trainer.tokenizer = _FakeProcessor()
    trainer.processor = trainer.tokenizer
    trainer.train_dataset = dataset
    trainer.formatting_func = None
    trainer._batches = None
    trainer._distributed_initialized = True
    trainer._distributed_world = None
    trainer._distributed_world_size = world_size
    return trainer


def test_vlm_prefetch_identity_laziness_and_masked_rejection():
    from unsloth_zoo.mlx.utils import iterate_vlm_training_batches

    class ContentProcessor(_FakeProcessor):
        """Encodes each row's text so batch content tracks source order."""
        def __call__(self, text, **kwargs):
            out = super().__call__(text, **kwargs)
            ids = np.asarray(out["input_ids"]).copy()
            for row, value in enumerate(text):
                ids[row, 0] = int(str(value).split()[0])
            out["input_ids"] = ids
            return out

    def stream(source, **kwargs):
        return iterate_vlm_training_batches(
            dataset=source, processor=ContentProcessor(), config={},
            batch_size=2, max_seq_length=8, **kwargs)

    sync = [b["input_ids"].tolist()
            for b in (lambda it: [next(it) for _ in range(4)])(
                iter(stream(_LifecycleVLMRows(6))))]
    assert sync[0] != sync[1]  # content varies with source rows
    control = {}
    prefetched_iter = iter(stream(_LifecycleVLMRows(6), prefetch_batches=2,
                                  prefetch_control=control))
    prefetched = [next(prefetched_iter)["input_ids"].tolist() for _ in range(4)]
    assert prefetched == sync  # bit-for-bit consumer-visible sequence
    assert control["prefetcher"].close()

    # Trainer wiring: eligibility, control registration, and cleanup.
    shell_probe = _LifecycleVLMRows(6)
    trainer = _vlm_trainer_shell_for(shell_probe, prefetch=2)
    _b, shell_stream = trainer._prepare_data(is_vlm=True)
    assert trainer._mlx_prefetch_control.get("eligible") is True
    assert next(iter(shell_stream))["input_ids"].shape[0] == 1
    shell_pf = trainer._mlx_prefetch_control.get("prefetcher")
    assert shell_pf is not None
    trainer._active_batch_iter = shell_stream
    trainer._close_active_batch_iterator()
    assert trainer._mlx_prefetch_control.get("prefetcher") is None

    probe = _LifecycleVLMRows(6)
    lazy = iter(stream(probe, prefetch_batches=2))
    assert probe.pulls == 0  # construction-lazy at P>0
    next(lazy)
    assert probe.pulls >= 2

    masked_probe = _LifecycleVLMRows(4)
    with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
        next(iter(stream(masked_probe, prefetch_batches=1,
                         response_mask_fn=lambda b: {"labels": b["input_ids"]})))
    assert masked_probe.pulls == 0 and masked_probe.epochs == []


def test_vlm_prefetch_opaque_lazy_mx_processor_paths(monkeypatch):
    """Lazy processor-owned MLX graphs must cross the prefetch boundary.

    Regression: without the producer-side materialization barrier, consumer
    evaluation raises "There is no Stream ... in current thread". Label decisions
    for both opaque routes must run on the consumer thread only.
    """
    import threading

    import mlx.core as mx

    from unsloth_zoo.mlx import utils as U

    class LazyMxProcessor(_FakeProcessor):
        def __call__(self, text, **kwargs):
            out = super().__call__(text, **kwargs)
            rows, width = np.asarray(out["input_ids"]).shape
            return {
                "input_ids": mx.broadcast_to(mx.arange(1, width + 1), (rows, width)),
                "attention_mask": mx.broadcast_to(mx.array(1), (rows, width)),
            }

    label_threads = []
    legacy_masks = U._apply_vlm_label_masks

    def _spy(*args, **kwargs):
        label_threads.append(threading.current_thread())
        return legacy_masks(*args, **kwargs)

    monkeypatch.setattr(U, "_apply_vlm_label_masks", _spy)

    def stream(source, **kwargs):
        kwargs.setdefault("batch_size", 2)
        return U.iterate_vlm_training_batches(
            dataset=source, processor=LazyMxProcessor(), config={},
            max_seq_length=8, **kwargs)

    it = iter(stream(_LifecycleVLMRows(4), prefetch_batches=1))
    batch = next(it)
    mx.eval(batch["input_ids"], batch["labels"])  # consumer-side evaluation
    assert np.asarray(batch["input_ids"]).tolist() == [[1, 2, 3, 4, 5]] * 2
    it.close()

    class _PCRows:  # replayable unsized source
        def __iter__(self):
            return iter([{"prompt": "101", "completion": "102"},
                         {"prompt": "103", "completion": "104"}])

    def take(iterator, count):
        taken = []
        for _ in range(count):
            b = next(iterator)
            taken.append((np.asarray(b["input_ids"]).tolist(),
                          np.asarray(b["labels"]).tolist(), b["labels"].dtype))
        return taken

    for pc_loss in (None, False):
        sync_seq = take(iter(stream(
            _PCRows(), batch_size=1, completion_only_loss=pc_loss)), 2)
        pf_it = iter(stream(_PCRows(), batch_size=1, prefetch_batches=1,
                            completion_only_loss=pc_loss))
        assert take(pf_it, 2) == sync_seq  # parity incl. label dtypes
        pf_it.close()
    assert label_threads and set(label_threads) == {threading.main_thread()}

    # pc_opaque holds no live tokenizer: finalize works after it is torn down.
    class _PoisonedTokenizer:
        def __getattr__(self, name):
            raise AssertionError(f"finalize dereferenced tokenizer.{name}")

    proc = LazyMxProcessor()
    staged = U._collate_vlm_prompt_completion_batch(
        [{"prompt": "101", "completion": "102"}], proc, 8, None,
        reject_mlx_valued=True)
    assert staged.pc_opaque is not None
    proc.tokenizer = _PoisonedTokenizer()
    poisoned = U._finalize_vlm_batch(staged)
    assert np.asarray(poisoned["input_ids"]).tolist() == sync_seq[0][0]


def test_gemma3n_token_type_ownership_survives_module_relocation():
    """Gemma3n builds token types itself however its module is named."""
    from unsloth_zoo.mlx.utils import _vlm_processor_requests_mm_token_type_ids

    def processor(module, name="Gemma3nProcessor"):
        def __call__(self, *_args, **_kwargs):
            return {}
        return type(name, (object,), {"__module__": module, "__call__": __call__})()

    # Remote-code and single-file loads expose the marker only as a suffix of
    # the module name, never as a standalone path component.
    for module in (
        "transformers.models.gemma3n.processing_gemma3n",
        "google.gemma-3n-E4B-it.a1b2c3.processing_gemma3n",
        "processing_gemma3n",
        "mlx_vlm.models.gemma3n.processing",
    ):
        assert _vlm_processor_requests_mm_token_type_ids(processor(module)) is False

    # Gemma3 proper still asks the processor for the multimodal token types.
    assert _vlm_processor_requests_mm_token_type_ids(
        processor("transformers.models.gemma3.processing_gemma3", "Gemma3Processor")
    ) is True


# --- causality: a padding mask must never become the attention mask ---------


def _utils_mx():
    """The mlx the module under test is bound to (a sibling test may shim it)."""
    from unsloth_zoo.mlx import utils
    return utils.mx


def _run_loss(model_type, kv_shared=0):
    """Run the baseline loss and report what the model was handed."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.utils import make_vlm_baseline_loss_fn

    mx_ = _utils_mx()
    seen = {}
    ids = mx_.array([[1, 2, 3, 4]], dtype=mx_.int32)
    batch = {"input_ids": ids, "labels": ids,
             "attention_mask": mx_.ones(ids.shape, dtype=mx_.int32)}

    class _Model:
        config = SimpleNamespace(model_type=model_type)

        # Declared and never read, exactly as molmo_point declares it.
        def get_input_embeddings(self, input_ids=None, pixel_values=None, mask=None):
            return None

        # Keyword-only and required, as thirteen families declare it: omitting
        # it is a TypeError, so every case below also proves it is always sent.
        def __call__(self, inputs, pixel_values=None, *, mask, cache=None, **kw):
            seen["mask"], seen["cache"] = mask, cache
            return SimpleNamespace(
                logits=mx_.zeros((*inputs.shape, 16), dtype=mx_.float32))

    model = _Model()
    if kv_shared:
        model.language_model = SimpleNamespace(model=SimpleNamespace(
            first_kv_shared_layer_idx=15,
            config=SimpleNamespace(num_kv_shared_layers=kv_shared)))
    make_vlm_baseline_loss_fn(model=None, ignore_token_ids=[])(model, batch)
    return seen, batch


# mlx-vlm model_type values, not module names: FastVLM reports either.
_MASK_KEEPING_FAMILIES = (
    "gemma3", "paligemma", "llava_qwen2", "fastvlm",
    "falcon_ocr", "falcon_perception", "falcon-perception",
)


@pytest.mark.parametrize("model_type,keeps", [
    *[(family, True) for family in _MASK_KEEPING_FAMILIES],
    # mlx-vlm resolves the module case-insensitively, the config keeps its own.
    ("Gemma3", True), ("FALCON-PERCEPTION", True),
    ("qwen2_vl", False),
    ("molmo_point", False),
])
def test_baseline_loss_fn_sends_the_padding_mask_only_where_it_is_read(
        model_type, keeps):
    # Elsewhere it reaches the layers verbatim and replaces causality, letting
    # every supervised position read the token it is scored on.
    seen, batch = _run_loss(model_type)
    assert seen["mask"] is (batch["attention_mask"] if keeps else None)


def test_mask_keeping_allowlist_is_exactly_these_families_and_all_their_aliases():
    from unsloth_zoo.mlx.utils import _VLM_FAMILIES_KEEPING_FORWARDED_MASK as listed
    assert set(listed) == set(_MASK_KEEPING_FAMILIES)
    # A config keeps its model_type through mlx-vlm's remap, so an alias onto a
    # listed module has to be listed under its own spelling too.
    remap = pytest.importorskip("mlx_vlm.utils").MODEL_REMAPPING
    assert not {a for a, t in remap.items() if t in listed and a not in listed}


class _NoPadIdTokenizer(_FakeTokenizer):
    pad_token_id = None


class _LayoutProcessor(_FakeProcessor):
    """Emits one fixed layout, honouring `padding_side` only when told to.

    Instruct checkpoints ship `padding_side: left`; deepseek_vl_v2 and the
    falcon pair left-pad multimodal rows whichever side they are asked for.
    """

    def __init__(self, rows, masks, honours_side=False, tokenizer=None, **extras):
        self.rows, self.masks, self.extras = rows, masks, extras
        self.honours_side = honours_side
        self.tokenizer = tokenizer or _FakeTokenizer()

    def __call__(self, text, padding_side=None, **_kwargs):
        flush = self.honours_side and padding_side == "right"
        ids, mask = [], []
        for row, keep in zip(self.rows, self.masks):
            body = [t for t, k in zip(row, keep) if k]
            pad = [0] * (len(row) - len(body))
            ids.append(body + pad if flush else list(row))
            mask.append([1] * len(body) + pad if flush else list(keep))
        out = {"input_ids": np.array(ids, dtype=np.int32),
               "attention_mask": np.array(mask, dtype=np.int32)}
        out.update({key: np.asarray(value) for key, value in self.extras.items()})
        return out


# One row already flush, one left-padded, so a repair has something to move.
_RAGGED = ([[101, 10, 200, 11], [0, 0, 101, 200]],
           [[1, 1, 1, 1], [0, 0, 1, 1]])
_MARKS = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int32)


@pytest.mark.parametrize("honours_side", [True, False])
def test_collation_delivers_content_then_padding(honours_side):
    # Causality is all that excludes the pads once the mask is withheld, and it
    # excludes a trailing pad, not a leading one.
    batch = _finalized_collate(
        [{"text": "a"}] * 2, _LayoutProcessor(*_RAGGED, honours_side=honours_side),
        8, None)
    ids, mask = np.asarray(batch["input_ids"]), np.asarray(batch["attention_mask"])
    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert ids[1][:2].tolist() == [101, 200]


@pytest.mark.parametrize("key", ["mm_token_type_ids", "images_seq_mask"])
def test_collation_moves_token_aligned_sidecars_with_their_tokens(key):
    # gemma4 builds its bidirectional multimodal blocks from
    # `mm_token_type_ids`, the deepseek pair place image embeddings at
    # `images_seq_mask` indices; left behind, row 1's mark sits on a pad.
    batch = _finalized_collate(
        [{"text": "a"}] * 2,
        _LayoutProcessor(*_RAGGED, mm_token_type_ids=_MARKS,
                         images_seq_mask=_MARKS.astype(bool)), 8, None)
    ids, marked = np.asarray(batch["input_ids"]), np.asarray(batch[key]).astype(bool)
    assert marked.sum() == 2 and (ids[marked] == 200).all()
    assert marked[1].tolist() == [False, True, False, False]


@pytest.mark.parametrize("key", ["image_grid_hw", "some_unlisted_field"])
def test_collation_leaves_an_unlisted_field_alone(key):
    # Three images of two columns, in a batch of three rows two tokens wide:
    # per-image metadata shaped exactly like a per-token array. Recognising
    # sidecars by shape would compact row 1's to [5, 0], losing its height.
    grid = np.array([[2, 3], [4, 5], [6, 7]], dtype=np.int32)
    batch = _finalized_collate(
        [{"text": "a"}] * 3,
        _LayoutProcessor([[101, 200], [0, 200], [101, 200]],
                         [[1, 1], [0, 1], [1, 1]],
                         image_grid_hw=grid, some_unlisted_field=grid + 10),
        8, None)
    assert np.asarray(batch["input_ids"])[1].tolist() == [200, 0]
    assert np.asarray(batch[key]).tolist() == (grid if key == "image_grid_hw"
                                               else grid + 10).tolist()


@pytest.mark.parametrize("key", ["position_ids", "rope_deltas"])
def test_collation_refuses_a_processor_authored_layout_coordinate(key, monkeypatch):
    # Sidecars label tokens and travel with them; these are coordinates of the
    # layout the repair replaces. `rope_deltas` is injected into the pipeline's
    # own collection, so naming "position_ids" inline would fail this.
    from unsloth_zoo.mlx import utils

    monkeypatch.setattr(utils, "_VLM_WIDTH_GENERATED_KEYS",
                        tuple(utils._VLM_WIDTH_GENERATED_KEYS) + ("rope_deltas",))
    coords = np.tile(np.arange(4), (2, 1))
    with pytest.raises(ValueError, match=key):
        _finalized_collate([{"text": "a"}] * 2,
                           _LayoutProcessor(*_RAGGED, **{key: coords}), 8, None)


@pytest.mark.parametrize("honours_side", [True, False])
def test_collation_does_not_require_a_pad_id(honours_side):
    # falcon_ocr and falcon_perception ship no pad id and pad with 0, so
    # demanding one would refuse batches they collate fine, repair or not.
    batch = _finalized_collate(
        [{"text": "a"}] * 2,
        _LayoutProcessor(*_RAGGED, honours_side=honours_side,
                         tokenizer=_NoPadIdTokenizer()), 8, None)
    ids, mask = np.asarray(batch["input_ids"]), np.asarray(batch["attention_mask"])
    assert mask[:, 0].tolist() == [1, 1] and (ids[mask == 0] == 0).all()


# --- KV-shared layers borrow their K/V through the cache --------------------


def test_baseline_loss_fn_forwards_real_shared_kv_slots():
    # Slots, not placeholders, and one per producer: a list of Nones forwards
    # the same shape while every shared layer rebuilds K/V from its own
    # projections, and one slot repeated leaves them all reading the last.
    from unsloth_zoo.mlx.utils import _SharedKVSlot

    seen, _batch = _run_loss("gemma4", kv_shared=20)
    caches = seen["cache"]
    assert caches is not None and len(caches) == 15
    assert all(isinstance(c, _SharedKVSlot) for c in caches)
    assert len({id(c) for c in caches}) == 15
    keys, values = object(), object()
    # The producer stores through `update_and_fetch`, shared layers read `state`.
    assert caches[0].offset == 0 and caches[0].borrow() is None
    assert caches[0].update_and_fetch(keys, values) == (keys, values)
    assert caches[0].state == (keys, values)


# --- gemma3n: the ids-path embedding scale a merge leaves off ---------------


def _scale_probe(model_type, hidden_size=16):
    """Middle position carries a merged feature; the others are plain tokens."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils

    mx_ = _utils_mx()
    # Every position shares one token id, so an id-based mask cannot tell the
    # feature-carrying position from the plain ones -- only a difference can.
    ids = mx_.array([[7, 7, 7]], dtype=mx_.int32)
    raw = mx_.ones((1, 3, hidden_size), dtype=mx_.float32)

    class _Backbone:
        config = SimpleNamespace(model_type=model_type, hidden_size=hidden_size)
        def embed_tokens(self, _ids): return raw

    merged = mx_.concatenate(
        [raw[:, :1], mx_.full((1, 1, hidden_size), 9.0), raw[:, 2:]], axis=1)
    out = utils._apply_vlm_embed_scale(
        SimpleNamespace(language_model=SimpleNamespace(model=_Backbone())),
        ids, merged)
    return np.asarray(out)[0], hidden_size ** 0.5


@pytest.mark.parametrize("model_type,scaled", [
    ("gemma3n_text", True),
    # gemma3_text is compensated for in `_run_hidden_stack`, the route it takes,
    # so scaling it here would apply the factor twice.
    ("qwen2_vl", False), ("gemma3_text", False), ("gemma4_text", False),
])
def test_embed_scale_lifts_only_gemma3n_plain_tokens(model_type, scaled):
    out, scale = _scale_probe(model_type)
    assert out[0][0] == pytest.approx(scale if scaled else 1.0)
    assert out[2][0] == pytest.approx(scale if scaled else 1.0)
    # Merged features already carry the scaled magnitude; rescaling inflates.
    assert out[1][0] == pytest.approx(9.0)


# --- paligemma: a prefix-LM mask, not a padding outer product ---------------


def test_paligemma_mask_is_bidirectional_on_the_prefix_and_causal_on_the_suffix():
    """PaliGemma's own mask is an outer product of the padding mask, so with
    nothing padded the suffix being trained on reads the tokens after it."""
    from unsloth_zoo.mlx.loader import (
        _VLM_MODEL_FIXUPS, _fix_paligemma_multimodal_causal_mask,
        _paligemma_prefix_lm_mask,
    )

    # Loading a VLM has to apply the correction, not just define it.
    assert _fix_paligemma_multimodal_causal_mask in _VLM_MODEL_FIXUPS

    mx_ = _utils_mx()
    seq = 8
    # 0 marks the prefix, 1 the suffix: the reverse of Gemma3's convention.
    token_type_ids = mx_.array([[0, 0, 0, 0, 1, 1, 0, 0]], dtype=mx_.int32)
    padding = mx_.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=mx_.int32)
    m = np.asarray(
        _paligemma_prefix_lm_mask(token_type_ids, padding)
    ).reshape(-1, seq, seq)[0].astype(bool)

    q_idx, kv_idx = np.indices(m.shape)
    real = np.asarray(padding)[0] == 1
    prefix = (np.asarray(token_type_ids)[0] == 0) & real
    both_real = real[q_idx] & real[kv_idx]
    both_prefix = prefix[q_idx] & prefix[kv_idx]
    ahead = q_idx < kv_idx
    # The suffix is causal: outside the prefix nothing reads ahead. Left as an
    # outer product every one of these is visible.
    assert not m[ahead & ~both_prefix & both_real].any()
    # The prefix stays bidirectional, which is what PaliGemma is trained for.
    assert m[both_prefix].all()
    # The past is always visible, so this is a mask and not all-zeros.
    assert m[(q_idx >= kv_idx) & both_real].all()
    # And the pads are excluded, which the outer product did do and this must too.
    assert not m[~both_real].any()


def test_paligemma_mask_is_left_alone_without_token_types():
    """`Model.__call__` forwards no extra kwargs, so on the plain loss path the
    token types never arrive and there is no prefix boundary to derive."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.loader import (
        _paligemma_causal_mask_wrapper, _paligemma_replace_mask,
    )

    mx_ = _utils_mx()
    outer_product = mx_.ones((1, 1, 4, 4), dtype=mx_.bool_)
    padding = mx_.ones((1, 4), dtype=mx_.int32)

    untouched = _paligemma_replace_mask(
        SimpleNamespace(attention_mask_4d=outer_product), None, padding)
    assert untouched.attention_mask_4d is outer_product
    # Nor when upstream built no mask at all, e.g. a text-only batch.
    assert _paligemma_replace_mask(
        SimpleNamespace(attention_mask_4d=None),
        mx_.zeros((1, 4), dtype=mx_.int32), padding).attention_mask_4d is None
    # But it is replaced once the token types are there.
    replaced = _paligemma_replace_mask(
        SimpleNamespace(attention_mask_4d=outer_product),
        mx_.array([[0, 0, 1, 1]], dtype=mx_.int32), padding)
    assert replaced.attention_mask_4d is not outer_product
    assert not np.asarray(replaced.attention_mask_4d).reshape(4, 4)[2, 3]


def test_paligemma_embedder_wrapper_replaces_the_mask_it_wrapped():
    """The wrapper is what actually reaches a loaded model, so it has to hand the
    token types on rather than return upstream's mask untouched."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.loader import _paligemma_causal_mask_wrapper

    mx_ = _utils_mx()
    outer_product = mx_.ones((1, 1, 4, 4), dtype=mx_.bool_)
    seen = {}

    def original(_self, input_ids=None, pixel_values=None, mask=None, **kwargs):
        seen["kwargs"] = kwargs
        return SimpleNamespace(attention_mask_4d=outer_product)

    wrapped = _paligemma_causal_mask_wrapper(original)
    got = wrapped(object(), mx_.zeros((1, 4), dtype=mx_.int32), None,
                  mx_.ones((1, 4), dtype=mx_.int32),
                  token_type_ids=mx_.array([[0, 0, 1, 1]], dtype=mx_.int32))
    # Upstream still runs and still sees its kwargs.
    assert "token_type_ids" in seen["kwargs"]
    # And its all-visible mask does not survive: the suffix cannot read ahead.
    assert got.attention_mask_4d is not outer_product
    assert not np.asarray(got.attention_mask_4d).reshape(4, 4)[2, 3]


def test_paligemma_plain_loss_path_also_gets_the_causal_suffix():
    """Upstream's call embeds with a fixed three arguments, dropping the token
    types, so without threading them the `use_cce=False` path keeps leaking."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.loader import (
        _paligemma_call_wrapper, _paligemma_causal_mask_wrapper,
        _paligemma_pending_token_types,
    )

    mx_ = _utils_mx()
    outer_product = mx_.ones((1, 1, 4, 4), dtype=mx_.bool_)
    padding = mx_.ones((1, 4), dtype=mx_.int32)
    seen = {}

    def upstream_embed(_self, input_ids=None, pixel_values=None, mask=None, **kw):
        return SimpleNamespace(attention_mask_4d=outer_product)

    class _Model:
        get_input_embeddings = _paligemma_causal_mask_wrapper(upstream_embed)

        def __call__(self, input_ids, pixel_values=None, mask=None, **kwargs):
            # Upstream drops kwargs here, exactly as PaliGemma does.
            seen["mask"] = self.get_input_embeddings(
                input_ids, pixel_values, mask).attention_mask_4d
            return seen["mask"]

    _Model.__call__ = _paligemma_call_wrapper(_Model.__call__)
    model = _Model()
    model(mx_.zeros((1, 4), dtype=mx_.int32), None, padding,
          token_type_ids=mx_.array([[0, 0, 1, 1]], dtype=mx_.int32))

    assert seen["mask"] is not outer_product
    assert not np.asarray(seen["mask"]).reshape(4, 4)[2, 3]
    # And nothing is left pending once the call returns.
    assert _paligemma_pending_token_types() is None


class _FakePaligemma:
    """Upstream's shape: `__call__` embeds with a fixed three arguments, so the
    token types it was given never reach the embedder on their own."""

    outer_product = None          # set per test, so identity can be compared
    seen = None

    def get_input_embeddings(self, input_ids=None, pixel_values=None, mask=None,
                             **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(attention_mask_4d=self.outer_product)

    def __call__(self, input_ids, pixel_values=None, mask=None, **kwargs):
        import threading
        from unsloth_zoo.mlx.loader import _paligemma_pending_token_types
        if self.seen is not None:
            self.seen[threading.current_thread().name] = \
                _paligemma_pending_token_types()
        return self.get_input_embeddings(input_ids, pixel_values, mask)


def test_paligemma_install_puts_both_wrappers_on_the_class():
    """Removing either assignment leaves the matching production loss path
    unfixed, so the install itself has to be checked, not just the wrappers."""
    from unsloth_zoo.mlx.loader import _install_paligemma_causal_mask

    mx_ = _utils_mx()
    padding = mx_.ones((1, 4), dtype=mx_.int32)
    token_type_ids = mx_.array([[0, 0, 1, 1]], dtype=mx_.int32)

    class _Model(_FakePaligemma):
        outer_product = mx_.ones((1, 1, 4, 4), dtype=mx_.bool_)
        seen = {}

    assert _install_paligemma_causal_mask(_Model)
    model, ids = _Model(), mx_.zeros((1, 4), dtype=mx_.int32)

    # The plain path: the call wrapper has to carry the token types across.
    plain = model(ids, None, padding, token_type_ids=token_type_ids)
    assert not np.asarray(plain.attention_mask_4d).reshape(4, 4)[2, 3]
    # The cross-entropy path: the embedder wrapper receives them itself.
    direct = model.get_input_embeddings(
        ids, None, padding, token_type_ids=token_type_ids).attention_mask_4d
    assert not np.asarray(direct).reshape(4, 4)[2, 3]


def test_paligemma_token_types_do_not_cross_between_concurrent_callers():
    """One model can serve several callers at once; an attribute on the model
    would let one read another's batch."""
    import threading
    from unsloth_zoo.mlx.loader import (
        _install_paligemma_causal_mask, _paligemma_pending_token_types,
    )

    mx_ = _utils_mx()
    padding = mx_.ones((1, 4), dtype=mx_.int32)
    entered = threading.Barrier(2)

    class _Model(_FakePaligemma):
        outer_product = mx_.ones((1, 1, 4, 4), dtype=mx_.bool_)
        seen = {}

        def __call__(self, *args, **kwargs):
            entered.wait(timeout=5)      # both callers inside their own call
            return _FakePaligemma.__call__(self, *args, **kwargs)

    assert _install_paligemma_causal_mask(_Model)
    model = _Model()
    marks = {"a": mx_.array([[0, 0, 1, 1]], dtype=mx_.int32),
             "b": mx_.array([[0, 1, 1, 1]], dtype=mx_.int32)}
    threads = [
        threading.Thread(name=name, target=model,
                         args=(mx_.zeros((1, 4), dtype=mx_.int32), None, padding),
                         kwargs={"token_type_ids": mark})
        for name, mark in marks.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # Each caller saw its own token types, not the other's.
    for name, mark in marks.items():
        assert _Model.seen[name] is mark, f"{name} saw another caller's batch"
    assert _paligemma_pending_token_types() is None


# --- gemma3n: AltUp correction at batch sizes above one ---------------------


def _broken_altup(mx_, hidden=4, inputs=3):
    """A stand-in shaped like the AltUp whose correction assumes a batch of one."""
    from types import SimpleNamespace

    class _Coefs:  # callable like the real nn.Linear, with a weight to clip
        weight = mx_.zeros((inputs, hidden), dtype=mx_.float32)

        def __call__(self, modalities):
            return mx_.zeros((*modalities.shape[:-1], inputs), dtype=mx_.float32)

    class _Altup:
        config = SimpleNamespace(hidden_size=hidden, altup_num_inputs=inputs,
                                 altup_coef_clip=None, altup_active_idx=0)
        correction_coefs = _Coefs()

        def compute_router_modalities(self, x):
            return x

        def correct(self, predictions, activated):
            all_coefs = mx_.ones((*activated.shape[:-1], inputs), dtype=mx_.float32)
            innovation = activated - predictions[self.config.altup_active_idx]
            # The batch axis is treated as though it were last.
            all_coefs = all_coefs.transpose(2, 1, 0)
            corrected = innovation[None] * all_coefs[:, None]
            corrected += predictions
            return corrected.astype(activated.dtype)

    return _Altup


def test_gemma3n_altup_correction_survives_a_batch_larger_than_one():
    """Upstream transposes its coefficients as though the batch axis were last,
    which only broadcasts when the batch is one."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.loader import (
        _VLM_MODEL_FIXUPS, _altup_correct_handles_batch, _fix_gemma3n_altup_batch,
    )

    assert _fix_gemma3n_altup_batch in _VLM_MODEL_FIXUPS

    mx_ = _utils_mx()
    # Distinct batch and sequence, or transposing the two would look the same.
    hidden, inputs, seq, rows = 4, 3, 5, 2
    cls = _broken_altup(mx_, hidden, inputs)
    altup = cls()
    assert not _altup_correct_handles_batch(altup), "fixture should start broken"

    upstream_correct = cls.correct
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(altup=altup)])))
    assert _fix_gemma3n_altup_batch(model)
    assert _altup_correct_handles_batch(altup)

    # Rows are independent, so a batch must equal the rows corrected separately.
    rng = np.random.default_rng(0)
    activated = mx_.array(
        rng.normal(size=(rows, seq, hidden)).astype(np.float32))
    # Non-zero, or subtracting the active prediction would not be observable.
    predictions = mx_.array(
        rng.normal(size=(inputs, rows, seq, hidden)).astype(np.float32))
    both = np.asarray(altup.correct(predictions, activated))
    assert both.shape == (inputs, rows, seq, hidden)
    for row in range(rows):
        one = np.asarray(altup.correct(
            predictions[:, row:row + 1], activated[row:row + 1]))
        assert np.allclose(both[:, row:row + 1], one, atol=1e-5), \
            f"row {row} differs when corrected in a batch"

    # The formula stays upstream's. At a batch of one, where upstream is
    # already right, the replacement has to reproduce it exactly.
    one_act, one_pred = activated[:1], predictions[:, :1]
    assert np.allclose(np.asarray(altup.correct(one_pred, one_act)),
                       np.asarray(upstream_correct(altup, one_pred, one_act)),
                       atol=1e-6), "the replacement changed the correction itself"


def test_gemma3n_altup_patch_declines_a_release_that_is_already_correct():
    """Upstream repaired this in 0.5.0, so the backport must leave it alone."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.loader import (
        _altup_correct_handles_batch, _fix_gemma3n_altup_batch,
    )

    mx_ = _utils_mx()
    cls = _broken_altup(mx_)

    def fixed(self, predictions, activated):
        all_coefs = (self.correction_coefs(activated) + 1.0)
        all_coefs = all_coefs.transpose(2, 0, 1)[..., None]
        return (activated - predictions[0])[None] * all_coefs + predictions

    cls.correct = fixed
    altup = cls()
    assert _altup_correct_handles_batch(altup)
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(altup=altup)])))
    assert not _fix_gemma3n_altup_batch(model)
    assert cls.correct is fixed, "an already-correct release must be left alone"
# --- Audio input collation -------------------------------------------------


class _FakeGemmaAudioProcessor(_ConversationalPromptCompletionProcessor):
    """Gemma-style: one delimited soft-token run per clip."""
    tokenizer = type("_AudioTok", (_FakeTokenizer,), {
        "audio_token": "<audio_soft_token>",
        "_vocab": dict(_FakeTokenizer._vocab, **{"<audio_soft_token>": 300})})()
    feature_extractor = type("_Extractor", (), {"sampling_rate": 16000})()
    soft_tokens = 3
    def __init__(self, truncates=False):
        super().__init__()
        self.truncates = truncates
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        # Real templates emit a placeholder per audio part.
        marked = [dict(m, content=[
            {"type": "text", "text": "<audio>"} if p.get("type") == "audio" else p
            for p in m.get("content", [])
        ] if isinstance(m.get("content"), list) else m.get("content", ""))
            for m in messages]
        return super().apply_chat_template(marked, tokenize=tokenize,
                                           add_generation_prompt=add_generation_prompt)
    def __call__(self, text, audio=None, max_length=None, **_kwargs):
        # Delimited (301/302) as real templates do, so runs stay separable.
        rows = [[101] + sum(([301] + [300] * self.soft_tokens + [302]
                             for _ in range(v.count("<audio>"))), []) + [11]
                for v in text]
        width = max(len(r) for r in rows)
        cut = max_length if self.truncates else width
        pad = lambda r, v: r + [v] * (width - len(r))
        out = {
            "input_ids": np.array([pad(r, 0) for r in rows], np.int32)[:, :cut],
            "attention_mask": np.array(
                [pad([1] * len(r), 0) for r in rows], np.int32)[:, :cut],
        }
        if audio:  # equal-length clips; ragged batches are qualification work
            feats = np.stack([np.asarray(c) for c in audio]).astype(np.float32)
            out["input_features"] = feats
            out["input_features_mask"] = np.ones(feats.shape, dtype=bool)
        return out


class _FakeAudioDecoder:  # datasets 4.x: output rate fixed at construction
    def __init__(self, data, sample_rate=16000):
        self._s = type("_D", (), {"data": data, "sample_rate": sample_rate})()
    def get_all_samples(self):
        return self._s


def _audio_row(clip, text="hi", placeholder_only=False):
    part = {"type": "audio"} if placeholder_only else {"type": "audio", "audio": clip}
    content = [part] if placeholder_only or clip is not None else []
    return {"messages": [
        {"role": "user", "content": content + [{"type": "text", "text": text}]},
        {"role": "assistant", "content": "ok"}]}


def _qualify(monkeypatch, processor=None, version=None):
    from unsloth_zoo.mlx import utils as mlx_utils
    family = mlx_utils._audio_family_from_processor(
        processor or _FakeGemmaAudioProcessor())
    monkeypatch.setattr(mlx_utils, "_AUDIO_QUALIFIED_FAMILIES", {family: frozenset(
        {version or mlx_utils._installed_mlx_vlm_version()})})


@pytest.mark.parametrize("gate,message", [
    ({}, "not enabled for any model family"),
    ({"otherfam": frozenset({"9.9.9"})}, "is not supported for"),
    (None, "only been verified"),
])
def test_audio_gate_refuses_unverified_family_or_version(monkeypatch, gate, message):
    from unsloth_zoo.mlx import utils as mlx_utils
    if gate is None:  # this family, but a version nothing was probed on
        _qualify(monkeypatch, version="0.0.1-never")
    else:
        monkeypatch.setattr(mlx_utils, "_AUDIO_QUALIFIED_FAMILIES", gate)
    with pytest.raises(NotImplementedError, match=message):
        _finalized_collate([_audio_row(_CLIP)], _FakeGemmaAudioProcessor(), 16, None)


_MONO = np.full(10, 0.5, dtype=np.float32)
_STEREO = np.stack([np.zeros(10, np.float32), np.ones(10, np.float32)])
_CLIP = {"array": _MONO, "sampling_rate": 16000}


def _audio_hiding_plan_kwargs(**extra):
    dataset = [
        {"audio": [{"array": np.zeros(1600, dtype=np.float32),
                    "sampling_rate": 16000}],
         "text": "a"},
    ]
    return dict(
        dataset=dataset,
        processor=_FakeProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        formatting_func=lambda row: {"text": row["text"]},
        **extra,
    )


def test_batch_plan_gates_audio_a_formatter_hides_under_response_masking():
    """With a response mask the plan filters rows first, formatting there and
    collating with no formatter, so it needs the same gate."""
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    with pytest.raises((NotImplementedError, ValueError), match="audio"):
        _create_vlm_batch_plan(**_audio_hiding_plan_kwargs(
            response_mask_fn=lambda b: {"labels": b["input_ids"]},
        ))


def test_batch_plan_gates_audio_that_a_formatter_would_hide():
    """The plan formats at construction and collates with no formatter, so the
    collation gate never sees the audio the user actually supplied."""
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    dataset = [
        {"audio": [{"array": np.zeros(1600, dtype=np.float32),
                    "sampling_rate": 16000}],
         "text": "a"},
    ]
    with pytest.raises((NotImplementedError, ValueError), match="audio"):
        _create_vlm_batch_plan(
            dataset=dataset,
            processor=_FakeProcessor(),
            config={"image_size": 16, "image_token_id": 200},
            batch_size=1,
            max_seq_length=8,
            formatting_func=lambda row: {"text": row["text"]},
        )


# Channel-first stereo must average to the same mono waveform, never flatten
# into a 20-sample concatenation of both channels.
@pytest.mark.parametrize("clip", [_CLIP, _FakeAudioDecoder(_MONO),
                                  _FakeAudioDecoder(_STEREO)],
                         ids=["datasets3_mapping", "datasets4_decoder", "stereo"])
def test_audio_source_forms_reach_the_processor_as_mono(monkeypatch, clip):
    _qualify(monkeypatch)
    row = _audio_row(clip)
    feats = np.asarray(_finalized_collate([row], _FakeGemmaAudioProcessor(),
                                          16, None)["input_features"])
    assert feats.shape == (1, 10) and np.allclose(feats[0], 0.5)
    # A bare message list must reach extraction; a single dict rendering
    # without a placeholder must be refused, not trained on.
    bare = _finalized_collate([row["messages"]], _FakeGemmaAudioProcessor(), 16, None)
    assert np.asarray(bare["input_features"]).shape[0] == 1
    with pytest.raises(ValueError, match="0 placeholder run"):
        _finalized_collate([dict(row["messages"][0], text="hi")],
                           _FakeGemmaAudioProcessor(), 16, None)


def test_multiple_clips_keep_their_order_and_untyped_parts_count(monkeypatch):
    _qualify(monkeypatch)
    quiet = {"array": np.full(10, 0.25, np.float32), "sampling_rate": 16000}
    loud = {"array": np.full(10, 0.75, np.float32), "sampling_rate": 16000}
    row = {"messages": [{"role": "user", "content": [
        {"type": "audio", "audio": quiet},
        {"audio": loud},
        {"type": "text", "text": "hi"}]}, {"role": "assistant", "content": "ok"}]}
    feats = np.asarray(_finalized_collate([row], _FakeGemmaAudioProcessor(),
                                          32, None)["input_features"])
    # Clip order must survive extraction, or features pair with the wrong runs.
    assert np.allclose(feats[0], 0.25) and np.allclose(feats[1], 0.75)


# over_cap models a processor diverting truncation to its audio extractor.
@pytest.mark.parametrize("clip,message,max_len", [
    ({"path": "a.wav", "bytes": b""}, "datasets.Audio", 16),
    (_MONO, "must carry their sampling rate", 16),  # no rate: untrustworthy
    ({"array": _MONO, "sampling_rate": 8000}, "8000 Hz", 16),
    (None, "no audio data", 16),
    (_CLIP, "did not apply truncation", 3),
], ids=["undecoded", "no_rate", "rate_mismatch", "no_data", "over_cap"])
def test_unusable_audio_rows_are_rejected(monkeypatch, clip, message, max_len):
    processor = _FakeGemmaAudioProcessor()
    _qualify(monkeypatch, processor=processor)
    row = _audio_row(clip, placeholder_only=clip is None)
    with pytest.raises(ValueError, match=message):
        _finalized_collate([row], processor, max_len, None)


def test_fixed_budget_families_reject_shortened_runs(monkeypatch):
    class _Budgeted(_FakeGemmaAudioProcessor):
        audio_seq_length = 3

    processor = _Budgeted(truncates=True)
    _qualify(monkeypatch, processor=processor)
    # Truncation leaves 2 of 3 placeholders: one run survives, so only the
    # budget can tell the row was clipped.
    with pytest.raises(ValueError, match="merges 3 per clip"):
        _finalized_collate([_audio_row(_CLIP)], processor, 4, None)


def test_a_prompt_completion_combine_cannot_clip_a_fixed_budget_run(monkeypatch):
    """The combine truncates after the halves are checked, so it is its own
    chance to cut a run short. One run per clip still survives that, which is
    why the budget has to reach the post-combine check too."""
    class _Budgeted(_FakeGemmaAudioProcessor):
        audio_seq_length = 3

    processor = _Budgeted()
    _qualify(monkeypatch, processor=processor)
    # The halves are checked untruncated and pass; the combine then cuts the
    # run to 2 of 3, which still leaves one run for the count check to accept.
    with pytest.raises(ValueError, match="merges 3 per clip"):
        _finalized_collate([_PC_AUDIO_ROW], processor, 4, None,
                           return_prompt_completion=True)


def test_processors_taking_audio_pairs_are_accommodated(monkeypatch):
    """Some processors want (samples, rate) pairs and reject add_special_tokens."""
    class _PairsOnly(_FakeGemmaAudioProcessor):
        """Takes (samples, rate) pairs and no add_special_tokens, like phi4mm.

        It also exposes no sampling rate of its own, so the only source for the
        pair's rate is the one extraction verified on the clip.
        """
        feature_extractor = None
        seen = {}

        def __call__(self, text, audio=None, padding=True, return_tensors="np",
                     truncation=False, max_length=None):
            # Accepts neither add_special_tokens nor padding_side, like phi4mm.
            self.seen["pairs"] = [isinstance(c, tuple) for c in (audio or [])]
            if audio and not all(self.seen["pairs"]):
                raise ValueError("too many values to unpack (expected 2)")
            self.seen["rates"] = [r for _, r in (audio or [])]
            return _FakeGemmaAudioProcessor.__call__(
                self, text, [s for s, _ in (audio or [])], max_length)

    processor = _PairsOnly()
    _qualify(monkeypatch, processor=processor)
    batch = _finalized_collate([_audio_row(_CLIP)], processor, 16, None)
    assert "input_features" in batch
    # The paired form is retried with the rate extraction verified.
    assert processor.seen["pairs"] == [True] and processor.seen["rates"] == [16000]
    # Prompt/completion also sends padding_side: one-at-a-time recovery fails.
    pc_row = {"prompt": _audio_row(_CLIP)["messages"][:1],
              "completion": [{"role": "assistant", "content": "ok"}]}
    batch, is_pc = _finalized_collate([pc_row], processor, 16, None,
                                      return_prompt_completion=True)
    assert is_pc and "input_features" in batch



def test_placeholders_without_features_are_rejected(monkeypatch):
    _qualify(monkeypatch)
    # Pre-rendered text can keep the placeholder after the clip is gone.
    with pytest.raises(ValueError) as caught:
        _finalized_collate([{"raw": "x"}], _FakeGemmaAudioProcessor(), 16, None,
                           formatting_func=lambda _: {"text": "<audio>hi"})
    message = str(caught.value)
    assert "returned no audio features" in message
    # This processor is fine and never saw a clip, so the message has to reach
    # that cause and introduce the checkpoint as an example. Prose cannot be
    # checked for every way of blaming it; what is pinned is that the hedge
    # still introduces the checkpoint clause.
    assert "never received" in message
    assert "for instance the checkpoint" in message


def test_a_checkpoint_that_drops_its_audio_is_named_as_the_cause(monkeypatch):
    """An export whose preprocessor configuration omits its audio half accepts
    the audio argument and skips the audio, so this is where the user learns
    their checkpoint is the problem rather than their dataset."""
    class _Drops(_FakeGemmaAudioProcessor):
        def __call__(self, text, audio=None, max_length=None, **kwargs):
            out = super().__call__(text, audio, max_length, **kwargs)
            out.pop("input_features", None)
            out.pop("input_features_mask", None)
            return out

    processor = _Drops()
    _qualify(monkeypatch, processor=processor)
    audio_row = dict(_audio_row(None, placeholder_only=True), audio=_CLIP)
    with pytest.raises(ValueError) as caught:
        _finalized_collate([audio_row], processor, 16, None)
    message = str(caught.value)
    assert "preprocessor configuration" in message
    assert "audio feature extractor" in message
    # Actionable, not just diagnostic.
    assert "train on the text and image parts" in message
    # One message serves both callers, and the other one has a healthy
    # processor, so the checkpoint stays introduced as an example.
    assert "for instance the checkpoint" in message
    assert "never received" in message


def test_audio_placeholders_masked_and_mixed_batches_collate(monkeypatch):
    from unsloth_zoo.mlx.utils import _get_vlm_ignore_token_ids
    _qualify(monkeypatch)
    processor = _FakeGemmaAudioProcessor()
    audio_row = dict(_audio_row(None, placeholder_only=True), audio=_CLIP)
    batch = _finalized_collate(
        [audio_row, _audio_row(None, text="plain")], processor, 16, None,
        ignore_token_ids=_get_vlm_ignore_token_ids(processor=processor))
    ids, labels = np.asarray(batch["input_ids"]), np.asarray(batch["labels"])
    # The column feeds only the row carrying the placeholder.
    assert ids.shape[0] == 2 and np.asarray(batch["input_features"]).shape[0] == 1
    assert (ids[1] != 300).all() and (labels[ids == 300] == -100).all()


_PC_AUDIO_ROW = {
    "prompt": [{"role": "user", "content": [
        {"type": "audio", "audio": _CLIP},
        {"type": "text", "text": "transcribe"}]}],
    "completion": [{"role": "assistant", "content": "done"}],
}


def test_prompt_completion_audio_rides_the_prompt_half(monkeypatch):
    _qualify(monkeypatch)
    batch, is_pc = _finalized_collate([_PC_AUDIO_ROW], _FakeGemmaAudioProcessor(),
                                      16, None, return_prompt_completion=True)
    assert is_pc and "input_features" in batch
    # The host combine re-verifies after truncating: here the run is dropped.
    with pytest.raises(ValueError, match="0 placeholder run"):
        _finalized_collate([_PC_AUDIO_ROW], _FakeGemmaAudioProcessor(truncates=True),
                           2, None, return_prompt_completion=True)
    # Audio may not sit in the completion half, payload or not.
    for part in ({"type": "audio", "audio": _CLIP}, {"type": "audio"}):
        moved = {"prompt": [{"role": "user",
                             "content": [{"type": "text", "text": "q"}]}],
                 "completion": [{"role": "assistant", "content": [part]}]}
        with pytest.raises(ValueError, match="audio in the completion half"):
            _finalized_collate([moved], _FakeGemmaAudioProcessor(), 16, None,
                               return_prompt_completion=True)


def test_prompt_completion_refuses_audio_stated_as_spans(monkeypatch):
    """A span is an absolute coordinate into the half it was measured on, and
    the combine joins the halves before flushing and truncating the result.
    Letting it through would place the clip's audio on whatever landed there."""
    class _Spans(_FakeGemmaAudioProcessor):
        def __call__(self, text, audio=None, max_length=None, **kwargs):
            out = super().__call__(text, audio, max_length, **kwargs)
            out["audio_bounds"] = [
                np.array([[1, 4]] * v.count("<audio>"), np.int32) for v in text
            ]
            return out

    processor = _Spans()
    _qualify(monkeypatch, processor=processor)
    with pytest.raises(NotImplementedError,
                       match="spans, which prompt/completion collation can move"):
        _finalized_collate([_PC_AUDIO_ROW], processor, 16, None,
                           return_prompt_completion=True)
    # Only audio rows are refused; the format itself still works for this family.
    text_only = {"prompt": [{"role": "user", "content": "q"}],
                 "completion": [{"role": "assistant", "content": "a"}]}
    _finalized_collate([text_only], processor, 16, None,
                       return_prompt_completion=True)
    _finalized_collate([text_only, text_only], processor, 16, None,
                       return_prompt_completion=True)
    # The audio is on the later row: a first-row check would let this through.
    with pytest.raises(NotImplementedError,
                       match="spans, which prompt/completion collation can move"):
        _finalized_collate([text_only, _PC_AUDIO_ROW], processor, 16, None,
                           return_prompt_completion=True)


def test_the_deferred_combine_carries_the_fixed_budget_too(monkeypatch):
    """MLX-valued halves defer the combine to the consumer thread, so the
    budget has to survive the carrier as well -- a run the combine cuts short
    still leaves one run per clip, and the count check alone accepts it."""
    import mlx.core as current_mx
    if current_mx is not mx:
        pytest.skip("requires real MLX runtime without mlx_simulation monkeypatch")
    from unsloth_zoo.mlx.utils import _collate_vlm_batch, _finalize_vlm_batch

    class _BudgetedMLXValued(_FakeGemmaAudioProcessor):
        audio_seq_length = 3

        def __call__(self, text, audio=None, max_length=None, **_kw):
            return {k: mx.array(v)
                    for k, v in super().__call__(text, audio, max_length).items()}

    processor = _BudgetedMLXValued()
    _qualify(monkeypatch, processor=processor)
    staged, _ = _collate_vlm_batch([_PC_AUDIO_ROW], processor, 4, None,
                                   reject_mlx_valued=True,
                                   return_prompt_completion=True)
    assert staged.pc_audio == ([1], [300], 3)
    # The combine truncates to 4, cutting the run to 2 of 3.
    with pytest.raises(ValueError, match="merges 3 per clip"):
        _finalize_vlm_batch(staged)


def test_deferred_prompt_completion_staging_rechecks_audio_runs(monkeypatch):
    """Production must build the pc_audio carrier and honor it at finalize."""
    import mlx.core as current_mx
    if current_mx is not mx:
        pytest.skip("requires real MLX runtime without mlx_simulation monkeypatch")
    from unsloth_zoo.mlx.utils import _collate_vlm_batch, _finalize_vlm_batch

    class _MLXValued(_FakeGemmaAudioProcessor):
        def __call__(self, text, audio=None, max_length=None, **_kw):
            return {k: mx.array(v)
                    for k, v in super().__call__(text, audio, max_length).items()}

    processor = _MLXValued()
    _qualify(monkeypatch, processor=processor)
    staged, _ = _collate_vlm_batch([_PC_AUDIO_ROW], processor, 2, None,
                                   reject_mlx_valued=True,
                                   return_prompt_completion=True)
    # MLX-valued halves defer the combine, so the counts must ride pc_audio for
    # the post-combine check to be possible on the consumer thread. This family
    # has no fixed budget, hence the third slot is empty.
    assert staged.pc_opaque is not None
    assert staged.pc_audio == ([1], [300], None)
    # Truncation drops the run; the deferred check must catch it.
    with pytest.raises(ValueError, match="0 placeholder run"):
        _finalize_vlm_batch(staged)


def test_audio_merge_compacts_valid_features_per_row():
    """Padded rows must not spill their tails into the next row's placeholders."""
    import mlx.core as mx_
    from unsloth_zoo.mlx.utils import _compact_audio_features

    embeds = mx_.array([[[1.0], [2.0], [9.0]], [[3.0], [4.0], [5.0]]])
    padded = mx_.array([[False, False, True], [False, False, False]])
    # Row 0 keeps 2 of 3 positions, row 1 keeps 3: the compacted source is the
    # valid ones in row order, so the merge's running count lines up again.
    out = _compact_audio_features(embeds, padded, [2, 3])
    assert out.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    # Each row must come out exact: [2, 3] clips against [3, 2] rows sums to
    # five either way while everything after the first position is mispaired.
    with pytest.raises(ValueError, match="cannot be paired for this row"):
        _compact_audio_features(embeds, padded, [3, 2])
    with pytest.raises(ValueError, match="cannot be paired for this row"):
        _compact_audio_features(embeds, padded, [2, 4])


def test_qualified_families_carry_their_probed_requirements():
    from unsloth_zoo.mlx import utils as mlx_utils

    gate = mlx_utils._AUDIO_QUALIFIED_FAMILIES
    # Exact versions: a stray one would enable code no probe ran against.
    assert gate == {
        "gemma3n": frozenset({"0.4.4"}),
        "gemma4": frozenset({"0.4.4"}),
        "phi4mm": frozenset({"0.4.4"}),
        "minicpmo": frozenset({"0.4.4"}),
    }
    # Gemma 4 was probed only on a newer transformers; this env pins an older.
    gemma4_like = type("Proc", (), {})
    gemma4_like.__module__ = "mlx_vlm.models.gemma4.processing_gemma4"
    assert mlx_utils._audio_family_from_processor(gemma4_like()) == "gemma4"
    with pytest.raises(NotImplementedError, match="transformers"):
        mlx_utils._check_audio_family_gate(gemma4_like())
    mlx_utils._check_audio_transformers_floor("gemma3n")  # no floor recorded


class _Gemma4Extractor:
    """Gemma 4's mel framing and masking, in the two shapes mlx-vlm has shipped.

    Both releases pad waveforms to the longest in the batch and on to a multiple
    of 128 samples before framing. 0.5.0 added the reference's semicausal
    left-pad and its mask rule -- a frame is valid only when its whole analysis
    window is real audio -- where 0.4.4 subsamples a sample-level mask, which
    keeps frames a padded batch filled with silence.

    Frame counts and mask sums match the shipped extractors at every length
    asserted below. Mel values, dithering, normalisation and the 480,000-sample
    truncation are not reproduced; nothing here depends on them.
    """
    sampling_rate = 16000
    frame_length, hop_length, pad_multiple = 320, 160, 128

    def __init__(self, semicausal_pad, emit_mask=True):
        self.semicausal_pad = semicausal_pad
        self.emit_mask = emit_mask

    def __call__(self, waveforms, sampling_rate=None, return_attention_mask=True):
        lengths = [len(w) for w in waveforms]
        target = max(lengths)
        if target % self.pad_multiple:
            target = (target // self.pad_multiple + 1) * self.pad_multiple
        left = self.frame_length // 2 if self.semicausal_pad else 0
        frames = (target + left - (self.frame_length + 1)) // self.hop_length + 1
        out = {"input_features": np.zeros(
            (len(waveforms), max(frames, 0), 128), np.float32)}
        if self.emit_mask:
            offsets = np.arange(max(frames, 0)) * self.hop_length
            out["input_features_mask"] = np.stack([
                (offsets + self.frame_length < length + left) if left
                else (offsets < length)
                for length in lengths
            ])
        return out


class Gemma4Processor:  # named as mlx-vlm's, so the family resolves to gemma4
    tokenizer = type("_AudioTok", (_FakeTokenizer,), {
        "audio_token": "<|audio|>",
        "_vocab": dict(_FakeTokenizer._vocab, **{"<|audio|>": 300})})()

    def __init__(self, extractor):
        self.feature_extractor = extractor

    def _compute_audio_num_tokens(self, waveform, sampling_rate):
        raise AssertionError("the shipped duration rule must not be reached")

    def __call__(self, text=None, audio=None, **_kwargs):
        extracted = self.feature_extractor(audio, sampling_rate=16000)
        runs = [[101] + [300] * self._compute_audio_num_tokens(clip, 16000) + [11]
                for clip in audio]
        width = max(len(run) for run in runs)
        return {
            "input_ids": np.array(
                [run + [0] * (width - len(run)) for run in runs], np.int32),
            "attention_mask": np.array(
                [[1] * len(run) + [0] * (width - len(run)) for run in runs],
                np.int32),
            "input_features": extracted["input_features"],
            "input_features_mask": extracted["input_features_mask"],
        }


class _ForwardsTheHook(Gemma4Processor):
    """Keeps its own attributes, but writes that one hook through to another.

    The narrowest sharing there is: a probe watching any other attribute name
    sees an independent copy.
    """
    def __setattr__(self, name, value):
        origin = self.__dict__.get("_origin")
        if origin is not None and name == "_compute_audio_num_tokens":
            return setattr(origin, name, value)
        return object.__setattr__(self, name, value)

    def __delattr__(self, name):
        origin = self.__dict__.get("_origin")
        if origin is not None and name == "_compute_audio_num_tokens":
            return delattr(origin, name)
        return object.__delattr__(self, name)

    def __copy__(self):
        twin = type(self)(self.feature_extractor)
        object.__setattr__(twin, "_origin", self)
        return twin


@pytest.mark.parametrize("samples,reference,floor_044,mask_blind", [
    # 5.855 s: ceil(duration / 40 ms) says 147 and the reference agrees, but
    # 0.4.4's extractor emits one frame fewer -- copying the reference formula
    # would keep predicting a frame that extractor never produces.
    (93680, 147, 146, 147),
    # 2.245 s: both extractors emit 56; only the duration rule is over, at 57.
    (35920, 56, 56, 56),
    # Short enough that the reference's trailing frames are all padding, so a
    # count taken from the frame total instead of the mask says two.
    (769, 1, 1, 2),
])
def test_gemma4_audio_placeholders_track_the_installed_extractor(
        samples, reference, floor_044, mask_blind):
    from unsloth_zoo.mlx import utils as mlx_utils

    clip = np.zeros(samples, dtype=np.float32)
    counts = []
    for semicausal_pad in (True, False):
        processor = Gemma4Processor(_Gemma4Extractor(semicausal_pad))
        mlx_utils._repair_gemma4_audio_processor(processor)
        counts.append(processor._compute_audio_num_tokens(clip, 16000))
    assert counts == [reference, floor_044]

    # What a count that ignored the mask would produce, so the cases above that
    # differ from it show the mask is load-bearing.
    frames = _Gemma4Extractor(True)([clip])["input_features"].shape[1]
    assert mlx_utils._gemma4_audio_encoder_positions(
        np.ones(frames, dtype=bool)) == mask_blind
    # An extractor reporting no mask leaves the encoder without one too, so
    # every frame counts -- the same number.
    maskless = Gemma4Processor(_Gemma4Extractor(True, emit_mask=False))
    mlx_utils._repair_gemma4_audio_processor(maskless)
    assert maskless._compute_audio_num_tokens(clip, 16000) == mask_blind


@pytest.mark.parametrize("order", [(93680, 35920), (35920, 93680)])
@pytest.mark.parametrize("emit_mask", [True, False])
def test_padded_audio_batch_masks_each_clip_to_its_own_audio(order, emit_mask):
    """Batch padding must not hand a short clip frames that are silence."""
    from unsloth_zoo.mlx import utils as mlx_utils

    # The 0.4.4 mask rule this corrects, and an extractor reporting no mask at
    # all: padding looks like real audio either way once a batch is padded.
    extractor = _Gemma4Extractor(False, emit_mask=emit_mask)
    processor = Gemma4Processor(extractor)
    clips = [np.zeros(samples, np.float32) for samples in order]
    solo = [223 if samples == 35920 else 584 for samples in order]

    batched = dict(extractor(clips))
    as_collated = batched.get("input_features_mask")
    features = batched["input_features"]
    if emit_mask:
        # The short clip keeps two frames of pure padding, and how many it
        # keeps depends on what it was batched against.
        assert sorted(int(m.sum()) for m in as_collated) == [225, 584]

    corrected = mlx_utils._audio_repaired_processor(processor)
    repaired = mlx_utils._repair_audio_batch(batched, clips, corrected)
    counts = [corrected._compute_audio_num_tokens(clip, 16000) for clip in clips]
    mask = repaired["input_features_mask"]
    # Exact, not just the count: the frames kept must be that clip's own, at
    # its own offsets, or the encoder reads silence for real audio.
    assert mask.shape == (2, 584) and mask.dtype == np.dtype(bool)
    for row, keep in enumerate(solo):
        assert mask[row][:keep].all() and not mask[row][keep:].any()
    # Every row's surviving positions now match the run written for that clip,
    # and the features those positions index are still there.
    assert [mlx_utils._gemma4_audio_encoder_positions(row)
            for row in mask] == counts
    assert repaired["input_features"] is features

    # Half a contract: on a processor whose count was left alone, so is the
    # mask. A clip list that does not pair with the feature rows -- in either
    # direction -- is left alone too, so the collation check reports that
    # instead of a plausible-looking mask hiding it.
    three = [*clips, np.zeros(74000, np.float32)]
    left_alone = [
        mlx_utils._repair_audio_batch(dict(extractor(clips)), clips, processor),
        mlx_utils._repair_audio_batch(dict(extractor(clips)), three, corrected),
        mlx_utils._repair_audio_batch(dict(extractor(three)), clips, corrected),
    ]
    for untouched, source in zip(left_alone, (clips, clips, three)):
        expected = extractor(source).get("input_features_mask")
        assert (untouched.get("input_features_mask") is None if not emit_mask
                else untouched["input_features_mask"].tolist()
                == expected.tolist())

    # A mask that does not cover its features is not left to broadcast: no
    # other check compares the two.
    if emit_mask:
        for batch_clips in (clips, clips[:1]):
            ragged = dict(extractor(batch_clips))
            ragged["input_features_mask"] = (
                ragged["input_features_mask"][:, :1])
            with pytest.raises(ValueError, match="paired with their clips"):
                mlx_utils._repair_audio_batch(ragged, batch_clips, corrected)


def test_collation_repairs_the_audio_batch_before_the_model_sees_it():
    import pickle

    from unsloth_zoo.mlx import utils as mlx_utils

    processor = Gemma4Processor(_Gemma4Extractor(False))
    clips = [np.zeros(93680, np.float32), np.zeros(35920, np.float32)]
    inputs = mlx_utils._processor_vlm_inputs(
        processor, ["<|audio|>", "<|audio|>"], [[], []], 1024,
        all_audio=[[clips[0]], [clips[1]]],
    )
    assert [int(m.sum()) for m in inputs["input_features_mask"]] == [584, 223]
    assert inputs["input_features"].shape[:2] == (2, 584)
    # The shared processor is never touched: a corrected count on it would meet
    # an uncorrected mask in anything else using it, concurrently or after.
    assert "_compute_audio_num_tokens" not in vars(processor)
    assert not mlx_utils._audio_count_corrected(processor)
    pickle.loads(pickle.dumps(processor))


def test_the_corrected_count_rides_a_copy_and_only_for_its_family(monkeypatch):
    """The correction must not be observable through the shared processor."""
    from unsloth_zoo.mlx import utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_AUDIO_MIN_TRANSFORMERS", {})
    monkeypatch.setattr(mlx_utils, "_AUDIO_QUALIFIED_FAMILIES", {
        "gemma4": frozenset({mlx_utils._installed_mlx_vlm_version()}),
        "fakegemmaaudio": frozenset({mlx_utils._installed_mlx_vlm_version()}),
    })
    processor = Gemma4Processor(_Gemma4Extractor(True))
    assert mlx_utils._check_audio_family_gate(processor) == "gemma4"

    corrected = mlx_utils._audio_repaired_processor(processor)
    assert corrected is not processor and type(corrected) is type(processor)
    assert corrected.feature_extractor is processor.feature_extractor
    assert corrected._compute_audio_num_tokens(
        np.zeros(35920, dtype=np.float32), 16000) == 56
    # Built on the copy, not installed on the original and moved across.
    assert corrected._compute_audio_num_tokens.args[0] is corrected

    # Same family name, so the repair is reached, but not independently
    # copyable -- correcting either would correct the caller's own processor.
    # A copy that is the original, and one that keeps its own attributes but
    # writes just that hook through -- the narrowest sharing there is.
    for cls in (type("Gemma4Processor", (Gemma4Processor,),
                     {"__copy__": lambda self: self}),
                type("Gemma4Processor", (_ForwardsTheHook,), {})):
        for own in (lambda clip, rate: 7, None):
            shared = cls(_Gemma4Extractor(True))
            if own is not None:
                shared._compute_audio_num_tokens = own
            with pytest.raises(NotImplementedError, match="independent object"):
                mlx_utils._audio_repaired_processor(shared)
            assert not mlx_utils._audio_count_corrected(shared)
            # Refusing must leave the processor exactly as it was: whatever it
            # had of its own, and nothing where it had been inheriting.
            if own is not None:
                assert shared._compute_audio_num_tokens is own
            else:
                assert "_compute_audio_num_tokens" not in vars(shared)
    # The gate only decides support; nothing about the original changes, so a
    # concurrent user of it cannot pick up half the correction.
    assert "_compute_audio_num_tokens" not in vars(processor)
    assert not mlx_utils._audio_count_corrected(processor)

    other = _FakeGemmaAudioProcessor()
    assert mlx_utils._check_audio_family_gate(other) == "fakegemmaaudio"
    assert mlx_utils._audio_repaired_processor(other) is other


def test_gemma4_repair_refuses_a_processor_it_cannot_correct():
    """A renamed hook would silently restore the defect the gate says is fixed."""
    from unsloth_zoo.mlx import utils as mlx_utils

    renamed = type("_Renamed", (), {})()
    renamed.feature_extractor = _Gemma4Extractor(True)
    with pytest.raises(NotImplementedError, match="_compute_audio_num_tokens"):
        mlx_utils._repair_gemma4_audio_processor(renamed)

    no_extractor = Gemma4Processor(None)
    mlx_utils._repair_gemma4_audio_processor(no_extractor)
    with pytest.raises(NotImplementedError, match="audio feature extractor"):
        no_extractor._compute_audio_num_tokens(np.zeros(16000, np.float32), 16000)


def test_the_merge_correction_follows_the_family_not_its_spelling():
    """mlx-vlm lowercases model_type before selecting the module, so a
    differently spelled checkpoint loads the same family and passes the audio
    gate. Matching it exactly here would skip the correction for a model whose
    merge needs it, and the misalignment that follows is silent."""
    from unsloth_zoo.mlx.utils import audio_merge_patch_needed

    # Every casing, not a sample of them: a listed few can be satisfied by
    # matching those strings, which is the defect this guards against.
    import itertools
    for chars in itertools.product(*({c.lower(), c.upper()} for c in "gemma4")):
        spelling = "".join(chars)
        assert audio_merge_patch_needed({"model_type": spelling}), spelling
    # Families whose merge is already per row must still be left alone.
    for other in ("gemma3n", "phi4mm", "minicpmo", None):
        assert not audio_merge_patch_needed({"model_type": other}), other


def test_audio_merge_patch_is_held_and_restored_exactly():
    """Overlapping runs share the correction; the last one puts it back."""
    from unsloth_zoo.mlx.utils import (
        install_audio_merge_patch, remove_audio_merge_patch,
    )

    class _Model:
        def get_input_embeddings(self, *a, **k):
            return "class method"

    model = _Model()
    original = model.get_input_embeddings
    # Every caller taking a hold is told so, and releases exactly one.
    assert install_audio_merge_patch(model, 1) is True
    assert install_audio_merge_patch(model, 1) is True    # second holder
    assert remove_audio_merge_patch(model) is False       # first release
    assert model.get_input_embeddings.__name__ == "patched"
    assert remove_audio_merge_patch(model) is True        # last release
    assert model.get_input_embeddings() == original()
    assert remove_audio_merge_patch(model) is False

    # Someone else's instance-level wrapper must survive.
    model.get_input_embeddings = lambda *a, **k: "instance wrapper"
    install_audio_merge_patch(model, 1)
    remove_audio_merge_patch(model)
    assert model.get_input_embeddings() == "instance wrapper"



def test_patched_audio_merge_places_each_row_own_features():
    """The correction must actually re-pair features, not just wrap the call."""
    import mlx.core as current_mx
    if current_mx is not mx:
        pytest.skip("requires real MLX runtime without mlx_simulation monkeypatch")
    mx_ = mx
    from unsloth_zoo.mlx.utils import (
        install_audio_merge_patch, remove_audio_merge_patch,
    )

    embed_dim, token = 1, 7

    class _Tower:
        def __call__(self, features, padding):
            # Row 0 has one valid frame, row 1 two; the tail is what leaks.
            return features, mx_.array([[False, True], [False, False]])

    class _Model:
        audio_tower = _Tower()
        embed_audio = staticmethod(lambda x: x)

        def get_input_embeddings(self, input_ids=None, pixel_values=None, **kw):
            # Text embeddings carry the model dtype; audio arrives wider.
            return mx_.zeros((*input_ids.shape, embed_dim), dtype=mx_.bfloat16)

    model = _Model()
    install_audio_merge_patch(model, token)
    try:
        # Row 0 wants 1 placeholder, row 1 wants 2.
        features = mx_.array([[[1.0], [99.0]], [[2.0], [3.0]]], dtype=mx_.float32)
        ids = mx_.array([[token, 0], [token, token]])
        out = model.get_input_embeddings(
            input_ids=ids, input_features=features,
            input_features_mask=mx_.array([[True, False], [True, True]]),
        )
        # 99.0 is padding: it must not appear; row 1 keeps its own values.
        assert out.flatten().tolist() == [1.0, 0.0, 2.0, 3.0]
        # Merging must not widen the LM input dtype.
        assert out.dtype == mx_.bfloat16
    finally:
        remove_audio_merge_patch(model)


def test_phi4mm_token_ids_fall_back_to_the_mlx_vlm_defaults():
    """Phi-4-multimodal's config.json declares neither token index.

    Training threads the checkpoint mapping rather than the model's config
    object, so both arrive absent and the compile preparation cannot read them
    directly. mlx-vlm carries them as dataclass defaults.
    """
    from unsloth_zoo.mlx.utils import _phi4mm_token_ids

    config_cls = pytest.importorskip("mlx_vlm.models.phi4mm.config").ModelConfig
    fields = config_cls.__dataclass_fields__
    expected = (int(fields["image_token_index"].default),
                int(fields["audio_token_index"].default))

    # The real checkpoint's shape: model_type present, neither index declared.
    assert _phi4mm_token_ids({"model_type": "phi4mm"}) == expected
    # A config that does declare them wins over the defaults.
    assert _phi4mm_token_ids(
        {"image_token_index": -7, "audio_token_index": 11}
    ) == (-7, 11)
    # One present, one absent: only the missing side falls back, either way
    # round, so neither explicit value can be overwritten by the other's gap.
    assert _phi4mm_token_ids({"image_token_index": -7}) == (-7, expected[1])
    assert _phi4mm_token_ids({"audio_token_index": 11}) == (expected[0], 11)


def test_compile_preparation_finds_phi4mm_positions_without_token_indices():
    """The regression this guards is in the compile-preparation branch, not the
    helper: training threads the checkpoint mapping, which declares neither
    index, and `int(None)` raised there before the first step. Surviving the
    call is not enough -- the positions have to come out where the real marker
    ids sit, so resolving the wrong ids or swapping them is caught too."""
    from unsloth_zoo.mlx.utils import _prepare_vlm_batch_for_compile

    config_cls = pytest.importorskip("mlx_vlm.models.phi4mm.config").ModelConfig
    fields = config_cls.__dataclass_fields__
    image_id = int(fields["image_token_index"].default)
    audio_id = int(fields["audio_token_index"].default)

    def _prepared(config, ids):
        return _prepare_vlm_batch_for_compile({
            "input_ids": mx.array([ids], dtype=mx.int32),
            "attention_mask": mx.array([[1] * len(ids)], dtype=mx.int32),
        }, config)

    # The real checkpoint's config.json shape: model_type only.
    prepared = _prepared({"model_type": "phi4mm"}, [7, image_id, 8, audio_id, 9])
    assert prepared["image_token_positions"] == ((1,),)
    assert prepared["audio_token_spans"] == (((3, 4),),)
    # The markers are also resolved a second time, to rewrite the ids: a pair
    # corrupted only there would leave the metadata above intact.
    assert np.asarray(prepared["input_ids"]).tolist() == [
        [7, image_id, 8, audio_id, 9]
    ]
    # An explicit config wins at the call site too, not only in the helper, and
    # at both resolutions: the rewrite ignoring it would restore the defaults.
    declared = _prepared({"model_type": "phi4mm", "image_token_index": 7,
                          "audio_token_index": 9}, [7, image_id, 8, audio_id, 9])
    assert declared["image_token_positions"] == ((0,),)
    assert declared["audio_token_spans"] == (((4, 5),),)
    assert np.asarray(declared["input_ids"]).tolist() == [
        [7, image_id, 8, audio_id, 9]
    ]

# --- audio alignment from stated spans, for families whose run carries no id ---


def _bounds_inputs(bounds, width=8, rows=1, features=1):
    """A processor output shaped like MiniCPM-o's: spans, not identifiable runs.

    The ids are one repeated token, which is what a placeholder run looks like;
    tests that need a span landing on ordinary text overwrite them.
    """
    return {
        "input_ids": np.zeros((rows, width), dtype=np.int32),
        "attention_mask": np.ones((rows, width), dtype=np.int32),
        "audio_bounds": bounds,
        "audio_features": np.zeros((features, 80, 4), dtype=np.float32),
    }


def test_the_delimiters_around_an_audio_run_are_not_targets():
    """A stated span covers the run's interior only, so the delimiters the
    processor generated around it would otherwise stay supervised -- and they
    are generated the same way the run is."""
    from unsloth_zoo.mlx.utils import _get_vlm_ignore_token_ids

    class _Delimited(_FakeProcessor):
        tokenizer = type("_Tok", (_FakeTokenizer,), {
            "audio_start_id": 151697, "audio_end_id": 151699})()

    assert {151697, 151699} <= set(_get_vlm_ignore_token_ids(processor=_Delimited()))


def test_an_audio_part_canonicalizes_whichever_alias_it_uses(monkeypatch):
    """Every audio spelling ends up typed "audio", whether the caller wrote the
    type out or left it off. An untyped alias otherwise reaches neither the
    family gate nor the extractor, and one left under an explicit alias reaches
    both but renders no placeholder."""
    from unsloth_zoo.mlx.utils import (
        _extract_vlm_audio, _normalize_vlm_messages, _raw_row_has_audio,
    )

    processor = _FakeGemmaAudioProcessor()
    _qualify(monkeypatch, processor=processor)
    # A ramp rather than the shared constant clip, so a clip substituted for
    # the caller's cannot coincide with it.
    ramp = {"array": np.linspace(-0.5, 0.5, 16, dtype=np.float32),
            "sampling_rate": 16000}
    text = {"type": "text", "text": "hi"}
    for alias in ("audio", "audio_url", "input_audio"):
        for part in ({alias: ramp}, {"type": alias, alias: ramp}):
            # The clip takes the first, an interior and the last position on
            # both axes -- content entry within a message, and message within
            # the conversation -- past index 1 on each, and one layout carries
            # two clips, once across messages and once within one. An
            # implementation that reaches only some positions, ties the two
            # indices together, or stops after the first clip in a message or
            # in the row, fails one of these.
            for content in ([[dict(part), text]],
                            [[text, dict(part)]],
                            [[text, dict(part), text]],
                            [[text, text, dict(part)]],
                            [[dict(part), text], [text]],
                            [[text], [text, dict(part)]],
                            [[text], [dict(part), text], [text]],
                            [[text], [text], [text, dict(part)]],
                            [[dict(part), text], [text, dict(part)]],
                            [[dict(part), text, dict(part)]]):
                row = {"messages": [{"role": "user", "content": entries}
                                    for entries in content]}
                messages = _normalize_vlm_messages(row["messages"])
                # Normalization retypes in place: the conversation keeps its
                # shape and every clip keeps the coordinates it was given, so
                # a pass that relocated or dropped one would not survive here.
                at = [(index, position)
                      for index, entries in enumerate(content)
                      for position, entry in enumerate(entries) if alias in entry]
                assert [len(message["content"]) for message in messages] == \
                    [len(entries) for entries in content], (part, content)
                assert [(index, position)
                        for index, message in enumerate(messages)
                        for position, entry in enumerate(message["content"])
                        if alias in entry] == at, (part, content)
                for index, position in at:
                    # "audio", not the alias: Gemma 3n's template renders a
                    # placeholder for that spelling alone.
                    assert messages[index]["content"][position]["type"] == \
                        "audio", (part, content)
                assert _raw_row_has_audio(row) is True, (part, content)
                # ...and the caller's own clips reach the extractor, which
                # reads the payload from the key the caller used.
                clips = _extract_vlm_audio(row, messages, processor)
                assert len(clips) == len(at), (part, content)
                for clip in clips:
                    assert np.array_equal(
                        np.asarray(clip), ramp["array"]), (part, content)

    # A type outside the audio spellings is never rewritten.
    kept = _normalize_vlm_messages(
        [{"role": "user", "content": [{"type": "video", "video": "v.mp4"}]}])
    assert kept[0]["content"][0]["type"] == "video"


def test_a_fixed_budget_row_cannot_hide_a_short_run_behind_a_long_one():
    """Runs of the wrong lengths still total correctly and still start once per
    clip, so the total cannot stand in for the individual runs: a clip then
    spills into the next run and pairs audio with the wrong positions."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact_ids

    # Budget 3, three clips: runs of 3, 2 and 4 total 9 and begin three times,
    # which the total check and the run-count check both accept. The aligned
    # row comes first, so a check that reads only row 0 cannot pass.
    ids = np.array([[1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1],
                    [1, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1]])
    with pytest.raises(ValueError,
                       match=r"row 1 .*run\(s\) of length \[3, 2, 4\]"):
        _assert_audio_runs_intact_ids(
            ids, np.ones_like(ids), [3, 3], [7], None, budget=3)

    # One wrong run in each position in turn, so an implementation that skips
    # any single position accepts the row whose only bad run sits there. The
    # fourth case keeps both bad runs past a three-run prefix, which a check
    # reading only the first few runs would clear, and the last is uniformly
    # wrong, which a check comparing the runs to each other would clear.
    for lengths in ([2, 3, 3], [3, 2, 3], [3, 3, 2], [3, 3, 3, 2, 4], [4, 4]):
        row = [1]
        for length in lengths:
            row += [7] * length + [1]
        row = np.array([row])
        with pytest.raises(ValueError,
                           match=re.escape(f"length {lengths}")):
            _assert_audio_runs_intact_ids(
                row, np.ones_like(row), [len(lengths)], [7], None, budget=3)

    assert _assert_audio_runs_intact_ids(
        ids[:1], np.ones_like(ids[:1]), [3], [7], None, budget=3) == 3


def test_audio_bounds_are_read_per_row():
    from unsloth_zoo.mlx.utils import _audio_bounds_per_row

    assert _audio_bounds_per_row({"input_ids": np.zeros((1, 4))}) is None
    assert _audio_bounds_per_row(_bounds_inputs([[[1, 5]], [[2, 3], [4, 6]]])) == [
        [(1, 5)], [(2, 3), (4, 6)]
    ]


@pytest.mark.parametrize("bounds, counts, message", [
    ([[[1, 5]]], [2], "none may be dropped by truncation"),   # a clip lost its run
    ([[[1, 3], [4, 6]]], [1], "no extra may be added"),       # one clip, two spans
    ([[[3, 3]]], [1], "does not run forwards"),          # a run with no positions
    ([[[5, 3]]], [1], "does not run forwards"),          # ends before it starts
    ([[[1, 99]]], [1], "outside its 8 position"),        # truncation cut the run
    # A negative start slices empty, which reads as "every position attended".
    ([[[-1, 3]]], [1], "outside its 8 position"),
    # The mismatch is on row 1: a check reading only the first row misses it.
    ([[[1, 3]], [[1, 3]]], [1, 2], "row 1 carries 2 audio clip"),
])
def test_audio_bounds_reject_a_batch_that_cannot_be_aligned(bounds, counts, message):
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    with pytest.raises(ValueError, match=message):
        _assert_audio_runs_intact(
            _bounds_inputs(bounds, rows=len(counts), features=sum(counts)),
            counts, _FakeProcessor(), 8,
        )


def test_the_projection_after_the_audio_tower_is_frozen_too():
    """Freezing the encoder alone leaves the layer that projects its output into
    the language model trainable, which a full fine-tune would pull into the
    optimizer. Families name that layer differently."""
    from unsloth_zoo.mlx.utils import freeze_audio_modules

    class _Module:
        def __init__(self):
            self.frozen = False

        def freeze(self, recurse=False):
            self.frozen = True

    class _MiniCPMOish:
        def __init__(self):
            self.audio_tower = _Module()
            self.audio_projection_layer = _Module()

    model = _MiniCPMOish()
    assert set(freeze_audio_modules(model)) == {
        "audio_tower", "audio_projection_layer",
    }
    assert model.audio_tower.frozen and model.audio_projection_layer.frozen


def test_stated_spans_refuse_the_left_padding_repair():
    """The repair moves each row's tokens forward but re-derives nothing, so a
    stated span would keep naming its pre-repair position and land the clip's
    audio on whatever moved there. Both ends of a stale span can still be
    attended, so no later check would catch it."""
    from unsloth_zoo.mlx.utils import _right_pad_vlm_rows

    left_padded = _bounds_inputs([[[4, 6]]], width=8)
    left_padded["attention_mask"] = np.array([[0, 0, 1, 1, 1, 1, 1, 1]], np.int32)
    with pytest.raises(ValueError, match=r"row\(s\) \[0\] content-first"):
        _right_pad_vlm_rows(left_padded, _FakeProcessor())

    # Interior padding moves tokens just as leading padding does, so the refusal
    # covers it and must not describe the layout as merely left-padded.
    interior = _bounds_inputs([[[5, 7]]], width=8)
    interior["attention_mask"] = np.array([[1, 1, 0, 0, 1, 1, 1, 1]], np.int32)
    with pytest.raises(ValueError, match=r"row\(s\) \[0\] content-first"):
        _right_pad_vlm_rows(interior, _FakeProcessor())

    # Already content-first: the repair is a no-op and must not refuse.
    right_padded = _bounds_inputs([[[1, 3]]], width=8)
    right_padded["attention_mask"] = np.array([[1, 1, 1, 1, 1, 1, 0, 0]], np.int32)
    _right_pad_vlm_rows(right_padded, _FakeProcessor())

    # These processors report the field on every batch, empty when there is no
    # audio, so keying the refusal on the field would strand text-only rows.
    no_audio = _bounds_inputs([[], []], width=8, rows=2, features=0)
    no_audio["attention_mask"] = np.array([[0, 0, 1, 1, 1, 1, 1, 1],
                                           [0, 1, 1, 1, 1, 1, 1, 1]], np.int32)
    repaired = _right_pad_vlm_rows(no_audio, _FakeProcessor())
    assert np.asarray(repaired["attention_mask"])[:, 0].tolist() == [1, 1]

    # Only the text-only row moves. The audio row keeps its layout, so its spans
    # keep naming the same tokens and the batch is repairable.
    neighbour_moves = _bounds_inputs([[[1, 3]], []], width=8, rows=2)
    neighbour_moves["attention_mask"] = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
                                                  [0, 0, 1, 1, 1, 1, 1, 1]], np.int32)
    repaired = _right_pad_vlm_rows(neighbour_moves, _FakeProcessor())
    assert np.asarray(repaired["attention_mask"])[:, 0].tolist() == [1, 1]

    # The moving span-bearing row is not row 0, so checking only the first row's
    # spans would repair this batch and strand row 1's coordinates.
    later_row = _bounds_inputs([[[1, 3]], [[4, 6]]], width=8, rows=2, features=2)
    later_row["attention_mask"] = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
                                            [0, 0, 1, 1, 1, 1, 1, 1]], np.int32)
    with pytest.raises(ValueError, match=r"row\(s\) \[1\] content-first"):
        _right_pad_vlm_rows(later_row, _FakeProcessor())


def test_spans_that_do_not_name_a_placeholder_run_are_refused():
    """These processors pair the n-th opening delimiter with the n-th closing
    one, so a delimiter the rendered text carries itself shifts every later pair
    onto ordinary tokens. The shifted span still counts, fits and attends: only
    the tokens it names show it is wrong."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    shifted = _bounds_inputs([[[1, 4]]], width=8)
    # start delimiter, prefix, start delimiter, placeholder, end delimiter, ...
    shifted["input_ids"] = np.array([[9, 5, 9, 0, 8, 6, 6, 6]], dtype=np.int32)
    with pytest.raises(ValueError, match="does not name a placeholder run"):
        _assert_audio_runs_intact(shifted, [1], _FakeProcessor(), 8)

    # Two rows whose runs disagree: one of them is not the placeholder token.
    mixed = _bounds_inputs([[[1, 3]], [[1, 3]]], width=8, rows=2, features=2)
    mixed["input_ids"] = np.array([[9, 0, 0, 8, 6, 6, 6, 6],
                                   [9, 4, 4, 8, 6, 6, 6, 6]], dtype=np.int32)
    with pytest.raises(ValueError, match="does not name a placeholder run"):
        _assert_audio_runs_intact(mixed, [1, 1], _FakeProcessor(), 8)

    # Two spans in one row, the second naming a different repeated token: a
    # check comparing only each row's first span would accept this.
    within_row = _bounds_inputs([[[1, 3], [4, 6]]], width=8, features=2)
    within_row["input_ids"] = np.array([[9, 0, 0, 9, 4, 4, 8, 6]], dtype=np.int32)
    with pytest.raises(ValueError, match="does not name a placeholder run"):
        _assert_audio_runs_intact(within_row, [2], _FakeProcessor(), 8)

    # And the batch this is all protecting: two rows, two clips, one marker.
    valid = _bounds_inputs([[[1, 3]], [[1, 3], [4, 6]]], width=8, rows=2,
                           features=3)
    valid["input_ids"] = np.array([[9, 0, 0, 8, 6, 6, 6, 6],
                                   [9, 0, 0, 9, 0, 0, 8, 6]], dtype=np.int32)
    _assert_audio_runs_intact(valid, [1, 2], _FakeProcessor(), 8)


def test_clips_the_extractor_would_truncate_are_refused():
    """A clip past the extractor's window asks for positions whose audio the
    model never receives. Sizing the run from the waveform also leaves it a
    position or two above the embeddings on ordinary clips, which is upstream's
    own rounding with the whole clip present -- that must still be accepted, or
    every real batch is refused."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    class _Windowed(_FakeProcessor):
        audio_processor = type("_Ex", (), {"n_samples": 16, "sampling_rate": 16})()

    inputs = _bounds_inputs([[], [[1, 3]]], width=8, rows=2)
    with pytest.raises(ValueError, match="row 1 carries a 17-sample audio clip"):
        _assert_audio_runs_intact(inputs, [0, 1], _Windowed(), 8,
                                  clips=[[], [np.zeros(17, np.float32)]])

    # The second clip is the long one: checking only the first would miss it.
    two_clips = _bounds_inputs([[], [[1, 3], [4, 6]]], width=8, rows=2, features=2)
    with pytest.raises(ValueError, match="row 1 carries a 17-sample audio clip"):
        _assert_audio_runs_intact(
            two_clips, [0, 2], _Windowed(), 8,
            clips=[[], [np.zeros(16, np.float32), np.zeros(17, np.float32)]],
        )

    _assert_audio_runs_intact(inputs, [0, 1], _Windowed(), 8,
                              clips=[[], [np.zeros(16, np.float32)]])


def test_text_carrying_the_audio_delimiters_itself_is_refused(monkeypatch):
    """The processor pairs opening delimiters with closing ones across the whole
    row, so one the text brought itself shifts the pairing, and truncation can
    drop the generated span and leave the foreign one standing in for it at any
    length -- including exactly the right one. Nothing downstream separates
    them, so the text is refused before the processor expands it."""
    class _Delimited(_FakeGemmaAudioProcessor):
        tokenizer = type("_Tok", (_FakeGemmaAudioProcessor.tokenizer.__class__,), {
            "audio_start_id": 301, "audio_end_id": 302,
            "convert_ids_to_tokens": staticmethod(
                lambda i: {301: "<|audio_start|>", 302: "<|audio_end|>"}.get(i)),
        })()

    processor = _Delimited()
    _qualify(monkeypatch, processor=processor)
    # Either delimiter alone is enough to shift the pairing, so each is refused
    # on its own, not only a well-formed pair.
    for text, named in (("<|audio_start|>x<|audio_end|> hi", "<|audio_start|>"),
                        ("<|audio_start|> hi", "<|audio_start|>"),
                        ("<|audio_end|> hi", "<|audio_end|>")):
        row = _audio_row(_CLIP)
        row["messages"][0]["content"][1]["text"] = text
        with pytest.raises(ValueError, match=f"already contains {re.escape(named)}"):
            _finalized_collate([row], processor, 16, None)

    # The delimiter is on the second audio row: a check reading only the first
    # would let it through and shift that row's spans.
    later = _audio_row(_CLIP)
    later["messages"][0]["content"][1]["text"] = "<|audio_start|> hi"
    with pytest.raises(ValueError, match="row 1 already contains"):
        _finalized_collate([_audio_row(_CLIP), later], processor, 16, None)

    # The pairing is per row, so a row with no clip has none to disturb: a
    # text-only sibling may say whatever it likes beside an audio row.
    sibling = _audio_row(None, text="<|audio_start|> plain")
    _finalized_collate([_audio_row(_CLIP), sibling], processor, 16, None)


def test_ragged_image_metadata_survives_an_audio_batch():
    """An audio row may carry images too, and those rows were refused before
    this family was qualified. Their per-row image coordinates are ragged for
    the same reason the audio spans are, and the model indexes both by row."""
    from unsloth_zoo.mlx.utils import _to_mx_vlm_batch

    batch = _to_mx_vlm_batch({
        "input_ids": np.zeros((2, 8), dtype=np.int32),
        "audio_bounds": [np.zeros((0, 2), np.int32), np.array([[1, 3]])],
        "image_bound": [np.array([[1, 3], [4, 6]]), np.array([[2, 5]])],
        "tgt_sizes": [np.array([[4, 4], [4, 4]]), np.array([[8, 8]])],
    })
    assert [np.asarray(r).tolist() for r in batch["image_bound"]] == [
        [[1, 3], [4, 6]], [[2, 5]]
    ]
    assert [np.asarray(r).tolist() for r in batch["tgt_sizes"]] == [
        [[4, 4], [4, 4]], [[8, 8]]
    ]


def test_collation_hands_the_clip_lengths_to_the_window_check(monkeypatch):
    """The check can only see clips the collation passes it, so dropping that
    wiring would leave it live but blind."""
    class _Spans(_FakeGemmaAudioProcessor):
        audio_processor = type("_Ex", (), {"n_samples": 4, "sampling_rate": 16})()

        def __call__(self, text, audio=None, max_length=None, **kwargs):
            out = super().__call__(text, audio, max_length, **kwargs)
            out["audio_bounds"] = [
                np.array([[2, 5]] * v.count("<audio>"), np.int32) for v in text
            ]
            return out

    processor = _Spans()
    _qualify(monkeypatch, processor=processor)
    with pytest.raises(ValueError, match="audio clip"):
        _finalized_collate([_audio_row(_CLIP)], processor, 16, None)


def test_ids_the_run_path_rejects_are_rejected_on_the_bounds_path_too():
    """Reading rows and width off shape[0] and shape[1] would take a 3-D batch
    for an 8-wide one and pass it on with its extra axis intact."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    inputs = _bounds_inputs([[[1, 3]]], width=8)
    inputs["input_ids"] = np.zeros((1, 8, 1), dtype=np.int32)
    inputs.pop("attention_mask")
    with pytest.raises(ValueError, match="cannot verify audio alignment"):
        _assert_audio_runs_intact(inputs, [1], _FakeProcessor(), 8)


def test_audio_clips_are_grouped_per_row_when_the_processor_assigns_them():
    """A flat list leaves the row assignment to a marker count over the rendered
    text, which a chat template need not emit; every clip then lands on row 0.
    Grouping per row states the assignment instead of leaving it inferable."""
    from unsloth_zoo.mlx.utils import _format_vlm_audio_for_processor

    class _MiniCPMOProcessor(_FakeProcessor):  # matched by class name, as images are
        pass

    # A row with two clips: keeping only the first would drop one, and the span
    # count would then disagree with what the row carries.
    clips = [[], ["one", "two"]]
    assert _format_vlm_audio_for_processor(clips, _MiniCPMOProcessor()) == [
        [], ["one", "two"]
    ]
    assert _format_vlm_audio_for_processor(clips, _FakeProcessor()) == ["one", "two"]
    assert _format_vlm_audio_for_processor([[], []], _MiniCPMOProcessor()) is None


def test_collation_hands_the_processor_the_grouping_it_assigns_by(monkeypatch):
    """Whatever the helper decides is worth nothing if collation calls it without
    the processor: the payload silently reverts to flat and rows are guessed."""
    seen = {}

    class _MiniCPMOish(_FakeGemmaAudioProcessor):
        def __call__(self, text, audio=None, max_length=None, **kwargs):
            seen["audio"] = audio
            flat = [clip for row in audio for clip in row] if audio else audio
            return super().__call__(text, flat, max_length, **kwargs)

    processor = _MiniCPMOish()
    _qualify(monkeypatch, processor=processor)
    audio_row = dict(_audio_row(None, placeholder_only=True), audio=_CLIP)
    _finalized_collate([_audio_row(None, text="plain"), audio_row], processor, 16, None)
    assert [len(row) for row in seen["audio"]] == [0, 1]


def test_stated_spans_are_checked_instead_of_soft_token_runs():
    """The run interior need not be an audio-specific token, so counting runs
    would find none. A batch whose spans line up must still pass, and one whose
    spans have no payload behind them must not: the placeholders would occupy
    the sequence with nothing to merge into them."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    _assert_audio_runs_intact(_bounds_inputs([[[1, 5]]]), [1], _FakeProcessor(), 8)

    without_payload = _bounds_inputs([[[1, 5]]])
    without_payload.pop("audio_features")
    with pytest.raises(ValueError, match="audio"):
        _assert_audio_runs_intact(without_payload, [1], _FakeProcessor(), 8)

    # Present is not enough: the payload has to carry one entry per span, no
    # more and no fewer, or a clip's placeholders would be filled from another
    # clip's features.
    for features in (1, 3):
        with pytest.raises(ValueError, match="audio"):
            _assert_audio_runs_intact(
                _bounds_inputs([[[1, 3], [4, 6]]], features=features), [2],
                _FakeProcessor(), 8,
            )
    _assert_audio_runs_intact(
        _bounds_inputs([[[1, 3], [4, 6]]], features=2), [2], _FakeProcessor(), 8,
    )


def test_audio_span_positions_are_excluded_from_the_loss():
    """Placeholder positions are inputs, never text to predict, and they cannot
    be masked by token id when the marker is the unknown token. Each row is
    masked by its own spans: reusing row 0's would silently supervise one row's
    audio while masking another's text."""
    from unsloth_zoo.mlx.utils import _stage_vlm_label_mask_np

    inputs = _bounds_inputs([[[2, 5]], [], [[0, 1], [6, 8]]], width=8, rows=3)
    mask = _stage_vlm_label_mask_np(inputs)
    assert mask[0].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert mask[1].tolist() == [0] * 8
    assert mask[2].tolist() == [1, 0, 0, 0, 0, 0, 1, 1]


@pytest.mark.parametrize("bounds, expected", [
    # Rows holding different clip counts: stacking raises, and keeping only the
    # first row would tell every later row it has no audio while its
    # placeholders still occupy the sequence.
    ([np.array([[1, 3]]), np.array([[1, 3], [4, 7]])], [[[1, 3]], [[1, 3], [4, 7]]]),
    # A row without audio beside one with it: the empty row must stay empty and
    # stay a row, or the audio row's index moves.
    ([np.zeros((0, 2), np.int32), np.array([[1, 3]])], [[], [[1, 3]]]),
])
def test_ragged_audio_bounds_survive_conversion_to_mlx(bounds, expected):
    from unsloth_zoo.mlx.utils import _to_mx_vlm_batch

    batch = _to_mx_vlm_batch({
        "input_ids": np.zeros((2, 8), dtype=np.int32),
        "audio_bounds": bounds,
        # Nested ints, the shape MiniCPM-o reports: no array to stack, so this
        # rides the ordinary passthrough and needs no per-row handling.
        "audio_feature_lens": [[9], [9, 4]],
    })
    assert [np.asarray(r).tolist() for r in batch["audio_bounds"]] == expected
    assert batch["audio_feature_lens"] == [[9], [9, 4]]


def test_audio_spans_are_masked_on_the_deferred_label_path_too():
    """Processor output the host staging cannot claim decides its labels at
    finalize instead, and that decision has to exclude the spans as well. The
    response-mask path arrives here with labels already chosen, so masking only
    while deriving them would supervise audio under train_on_responses_only."""
    import mlx.core as current_mx
    if current_mx is not mx:
        pytest.skip("requires real MLX runtime without mlx_simulation monkeypatch")
    from unsloth_zoo.mlx.utils import _apply_vlm_label_masks

    batch = {
        "input_ids": mx.array([[5, 6, 7, 8, 9, 10]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 1, 1, 1]], dtype=mx.int32),
        "audio_bounds": [mx.array([[2, 5]], dtype=mx.int32)],
    }
    derived = np.asarray(_apply_vlm_label_masks(batch)).tolist()
    assert derived == [[5, 6, -100, -100, -100, 10]]

    supplied = mx.array([[-100, -100, 7, 8, 9, 10]], dtype=mx.int32)
    responses_only = np.asarray(_apply_vlm_label_masks(batch, supplied)).tolist()
    assert responses_only == [[-100, -100, -100, -100, -100, 10]]

    # Each row by its own spans: broadcasting row 0's would supervise row 1's
    # audio positions and mask text it should be learning.
    rows = {
        "input_ids": mx.array([[5, 6, 7, 8, 9, 10]] * 2, dtype=mx.int32),
        "attention_mask": mx.array([[1] * 6] * 2, dtype=mx.int32),
        "audio_bounds": [mx.array([[0, 2]], dtype=mx.int32),
                         mx.array([[4, 6]], dtype=mx.int32)],
    }
    assert np.asarray(_apply_vlm_label_masks(rows)).tolist() == [
        [-100, -100, 7, 8, 9, 10], [5, 6, 7, 8, -100, -100],
    ]


def test_an_empty_audio_payload_is_not_audio():
    """MiniCPM-o hands a text-only batch audio_features shaped (0, 80, 0).
    Calling that audio would route the run eagerly and abort a strict one."""
    from unsloth_zoo.mlx.utils import _vlm_batch_carries_audio

    assert not _vlm_batch_carries_audio({"audio_features": np.zeros((0, 80, 0), np.float32)})
    assert _vlm_batch_carries_audio({"audio_features": np.zeros((1, 80, 4), np.float32)})
    # A payload can arrive as a plain sequence too, and an empty one is no more
    # audio than an empty array is.
    assert not _vlm_batch_carries_audio({"input_audio_embeds": []})
    assert _vlm_batch_carries_audio({"input_audio_embeds": [object()]})


def test_audio_spans_are_checked_against_attended_positions():
    """A span covering padding is not placed on the tokens the model attends.
    Rows reach here content-first, so the check is per row over the positions
    the span names."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    # Row 1's span starts on content and runs off the end of it. Row 0 attends
    # everything, so a check reading row 0's mask sees nothing wrong, and one
    # reading only the span's first position sees nothing wrong either.
    straddling = _bounds_inputs([[[1, 3]], [[3, 6]]], width=8, rows=2, features=2)
    straddling["attention_mask"] = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 0, 0, 0, 0]], np.int32)
    with pytest.raises(ValueError, match="row 1 .* covering padding"):
        _assert_audio_runs_intact(straddling, [1, 1], _FakeProcessor(), 8)


def test_audio_bounds_require_a_row_per_tokenized_row():
    """Bounds are indexed per row by the model, so a count that disagrees with
    the tokenized rows would silently leave a row's audio unplaced. All three
    counts have to agree, in both directions."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    # Dataset rows and tokenized rows agree; only the bounds disagree, so a
    # check comparing just those two would pass this batch.
    inputs = _bounds_inputs([[[1, 3]]], width=8, rows=2, features=2)
    with pytest.raises(ValueError, match="tokenized row"):
        _assert_audio_runs_intact(inputs, [1, 1], _FakeProcessor(), 8)

    # Bounds and dataset rows agree; the tokenized rows do not.
    fewer_ids = _bounds_inputs([[[1, 3]], [[1, 3]]], width=8, rows=1, features=2)
    with pytest.raises(ValueError, match="tokenized row"):
        _assert_audio_runs_intact(fewer_ids, [1, 1], _FakeProcessor(), 8)

    # And the other direction: more bounds rows than the batch has.
    extra_bounds = _bounds_inputs([[[1, 3]], [[1, 3]], [[1, 3]]], width=8,
                                  rows=2, features=2)
    with pytest.raises(ValueError, match="tokenized row"):
        _assert_audio_runs_intact(extra_bounds, [1, 1], _FakeProcessor(), 8)

    # Bounds and tokenized rows agree; only the dataset rows do not, so a check
    # comparing just those two would accept this.
    extra_counts = _bounds_inputs([[[1, 3]], [[1, 3]]], width=8, rows=2, features=2)
    with pytest.raises(ValueError, match="dataset row"):
        _assert_audio_runs_intact(extra_counts, [1, 1, 1], _FakeProcessor(), 8)


def test_overlong_rows_are_refused_by_what_the_model_attends():
    """The run path caps each row's attended tokens. Capping the padded width
    instead would refuse a batch it accepts, since padding is not the row's own
    length."""
    from unsloth_zoo.mlx.utils import _assert_audio_runs_intact

    padded = _bounds_inputs([[[1, 3]]], width=12)
    padded["attention_mask"] = np.array([[1] * 6 + [0] * 6], dtype=np.int32)
    _assert_audio_runs_intact(padded, [1], _FakeProcessor(), 8)

    with pytest.raises(ValueError, match="row 0 is 12 tokens"):
        _assert_audio_runs_intact(_bounds_inputs([[[1, 3]]], width=12), [1],
                                  _FakeProcessor(), 8)

    # Row 1 is the one over the cap: reading row 0's mask every time would
    # accept it, since row 0 stays under.
    later_row = _bounds_inputs([[[1, 3]], [[1, 3]]], width=12, rows=2, features=2)
    later_row["attention_mask"] = np.array([[1] * 6 + [0] * 6, [1] * 12], np.int32)
    with pytest.raises(ValueError, match="row 1 is 12 tokens"):
        _assert_audio_runs_intact(later_row, [1, 1], _FakeProcessor(), 8)


# --- can this checkpoint take audio at all: behaviour, not configuration ---


class _ProbeProcessor:

    def __init__(self, needs_text=None):
        self.needs_text = needs_text
        self.calls = 0

    @staticmethod
    def _samples(audio):
        clip = audio[0][0] if isinstance(audio[0], (list, tuple)) else audio[0]
        return np.asarray(clip, dtype=np.float32)

    def __call__(self, text, audio=None, **kwargs):
        self.calls += 1
        if self.needs_text is not None and text[0] != self.needs_text:
            raise ValueError("this processor wants its own marker")
        return {"input_ids": np.zeros((1, 8), np.int32),
                "audio_features": self._samples(audio)[:16].copy()}


class _AudioModel:
    def __init__(self, names=("language_model", "audio_tower")):
        self._names = names

    def named_modules(self):
        return [(name, None) for name in self._names]


def test_the_mlx_vlm_boundary_reshapes_audio_a_processor_cannot_unpack():
    """Inference reaches the processor through mlx-vlm, which decodes audio to
    bare waveforms and refuses tuples, so a processor wanting (samples, rate)
    pairs is unreachable from the caller and has to be met at this boundary."""
    from unsloth_zoo.mlx.utils import _mlx_vlm_process_inputs_adapter

    class _Processor:
        # A rate mlx-vlm never consults, to catch a pairing that resolves the
        # rate for itself instead of using the one the samples are actually at.
        sampling_rate = 22050

    def pairs_only(processor, prompts, images=None, audio=None, **kwargs):
        calls.append(audio)
        if any(not isinstance(entry, tuple) for entry in audio or ()):
            raise ValueError("too many values to unpack (expected 2)")
        return {"input_audio_embeds": audio}

    calls = []
    patched = _mlx_vlm_process_inputs_adapter(pairs_only)
    clip = np.zeros(4, dtype=np.float32)
    out = patched(_Processor(), ["hi"], audio=[clip])
    # mlx-vlm resamples a decoded file to feature_extractor.sampling_rate and
    # falls back to 16 kHz, so that is the rate it assumes these are at.
    assert [(int(np.asarray(s).size), r)
            for s, r in out["input_audio_embeds"]] == [(4, 16000)]
    assert len(calls) == 2, "the pair shape is a retry, not the first guess"

    # ...and when mlx-vlm has a rate of its own, that one is used.
    class _Resampled(_Processor):
        feature_extractor = type("_FE", (), {"sampling_rate": 24000})()

    calls = []
    out = _mlx_vlm_process_inputs_adapter(pairs_only)(
        _Resampled(), ["hi"], audio=[clip])
    assert [r for _s, r in out["input_audio_embeds"]] == [24000]

    # A request carrying no audio must not reach for audio attributes at all:
    # processors expose some of them through properties that warn or raise.
    class _Hostile:
        @property
        def feature_extractor(self):
            raise AssertionError("a non-audio request asked for the audio rate")

    def text_only(processor, prompts, images=None, audio=None, **kwargs):
        return {"input_ids": prompts}

    assert _mlx_vlm_process_inputs_adapter(text_only)(
        _Hostile(), ["hi"])["input_ids"] == ["hi"]

    # Only a ValueError is a payload-shape refusal: another type carrying the
    # same word is the processor's own failure and is raised on the first call.
    seen = []

    def wrong_type(processor, prompts, images=None, audio=None, **kwargs):
        seen.append(audio)
        raise RuntimeError("cannot unpack, but not as a ValueError")

    with pytest.raises(RuntimeError, match="not as a ValueError"):
        _mlx_vlm_process_inputs_adapter(wrong_type)(
            _Processor(), ["hi"], audio=[clip])
    assert len(seen) == 1

    # Neither guard may be dropped: a refusal that is not about unpacking, and
    # an unpacking refusal with no audio to reshape, are both raised on the
    # first call rather than provoking a second in another shape.
    for label, message, audio in (("not an unpacking refusal", "some other complaint", [clip]),
                                  ("nothing to reshape", "too many values to unpack", [])):
        seen = []

        def refuses(processor, prompts, images=None, audio=None,
                    _message=message, _seen=seen, **kwargs):
            _seen.append(audio)
            raise ValueError(_message)

        with pytest.raises(ValueError, match=re.escape(message)):
            _mlx_vlm_process_inputs_adapter(refuses)(
                _Processor(), ["hi"], audio=audio)
        assert len(seen) == 1, label

    # Only the retried call is answered with the first refusal. A failure
    # while building the pairs says nothing about what the processor wanted,
    # so it is raised as itself.
    from unsloth_zoo.mlx.utils import _call_pairing_audio_on_refusal

    def always_refuses(call_kwargs):
        raise ValueError("too many values to unpack (expected 2)")

    def cannot_pair(_clips):
        raise RuntimeError("building the pairs failed on its own terms")

    with pytest.raises(RuntimeError, match="on its own terms"):
        _call_pairing_audio_on_refusal(
            always_refuses, {"audio": [clip]}, "audio", cannot_pair)

    # Everything else the caller passed survives the retry, or a mixed
    # image-and-audio request would silently lose its image.
    retries = []

    def records(processor, prompts, images=None, audio=None, **kwargs):
        retries.append({"images": images, "prompts": prompts, **kwargs})
        if any(not isinstance(entry, tuple) for entry in audio or ()):
            raise ValueError("too many values to unpack (expected 2)")
        return {"ok": True}

    _mlx_vlm_process_inputs_adapter(records)(
        _Processor(), ["hi"], images=["an image"], audio=[clip],
        padding_side="right")
    assert len(retries) == 2 and retries[0] == retries[1], retries

    def still_broken(processor, prompts, images=None, audio=None, **kwargs):
        if any(isinstance(entry, tuple) for entry in audio or ()):
            raise RuntimeError("the retry's error, which must not surface")
        raise ValueError("too many values to unpack (expected 2)")

    with pytest.raises(ValueError, match="too many values"):
        _mlx_vlm_process_inputs_adapter(still_broken)(
            _Processor(), ["hi"], audio=[clip])


def test_a_pair_taking_processor_is_not_reported_unusable():
    """Some processors take ``(samples, rate)`` pairs and reject a bare
    waveform by failing to unpack it. Reading that first refusal as the answer
    reports a working checkpoint as unusable, which is what Phi-4's
    remote-code processor did."""
    from unsloth_zoo.mlx.utils import (
        _AUDIO_PROBE_RATE, _AUDIO_PROBE_TONES, _audio_payload_as_pairs,
        _audio_probe_tone, audio_input_capability,
    )

    class _PairsOnly(_ProbeProcessor):
        def __init__(self):
            super().__init__()
            self.seen = []

        def __call__(self, text, audio=None, **kwargs):
            for entry in audio or ():
                samples, rate = entry  # a bare waveform raises here
                self.seen.append((len(samples), rate))
            return super().__call__(text, audio=audio, **kwargs)

    processor = _PairsOnly()
    verdict = audio_input_capability(_AudioModel(), processor)
    assert verdict.capable is True and verdict.processor_ok is True
    # The whole waveform arrives, with the rate it was built at alongside it.
    whole = len(_audio_probe_tone(_AUDIO_PROBE_RATE, _AUDIO_PROBE_TONES[0]))
    assert processor.seen and set(processor.seen) == {(whole, _AUDIO_PROBE_RATE)}

    # A clip that carries no rate of its own -- what a dataset column yields --
    # takes the rate the processor's extractor expects, which Phi-4 keeps on a
    # separately named audio extractor rather than on the processor itself.
    rated = type("_Rated", (_ProbeProcessor,), {
        "audio_processor": type("_Extractor", (), {"sampling_rate": 22050})(),
    })()
    bare = np.zeros(4, dtype=np.float32)
    assert _audio_payload_as_pairs([bare], rated)[0][1] == 22050


def test_a_checkpoint_that_drops_its_audio_is_not_capable():
    """The motivating defect: the export omits the audio half of its
    preprocessor configuration, so the processor takes the argument and skips
    the audio. Nothing raises and every field is well formed, so only the
    content differential separates it from a working checkpoint."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _Drops(_ProbeProcessor):
        def __call__(self, text, audio=None, **kwargs):
            self.calls += 1
            return {"input_ids": np.zeros((1, 8), np.int32)}

    verdict = audio_input_capability(_AudioModel(), _Drops())
    assert verdict.capable is False and verdict.processor_ok is False
    assert "no audio-dependent output" in verdict.reason
    # The model half is not what refused it; the audio modules are present.
    assert verdict.model_ok is True


def test_output_that_follows_only_the_clip_length_is_not_capable():
    """Both tones are the same duration, so a processor sizing its output from
    the length alone produces identical outputs and cannot be told from one
    that ignored the audio."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _LengthOnly(_ProbeProcessor):
        def __call__(self, text, audio=None, **kwargs):
            self.calls += 1
            return {"input_ids": np.zeros((1, len(self._samples(audio))), np.int32)}

    assert audio_input_capability(_AudioModel(), _LengthOnly()).capable is False


def test_a_processor_that_cannot_repeat_itself_is_not_capable():
    """The same tone twice must agree. A processor alternating between outputs
    would otherwise show a difference that says nothing about the audio."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _Alternates(_ProbeProcessor):
        def __call__(self, text, audio=None, **kwargs):
            self.calls += 1
            # A period that also makes the two tones' outputs differ, so only
            # the same-tone agreement check can refuse this.
            return {"input_ids": np.full((1, 8), self.calls % 3, np.int32)}

    assert audio_input_capability(_AudioModel(), _Alternates()).capable is False


def test_keying_on_the_buffer_instead_of_its_samples_is_not_capable():
    """Every call gets a freshly built buffer, so a processor keying on object
    identity sees four distinct clips -- the discarded warm-up and the three
    measured calls -- and cannot repeat itself. Reusing one buffer per tone in
    the probe would let this shape pass."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _KeysOnIdentity(_ProbeProcessor):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def __call__(self, text, audio=None, **kwargs):
            self.calls += 1
            clip = audio[0][0] if isinstance(audio[0], (list, tuple)) else audio[0]
            index = self.seen.setdefault(id(clip), len(self.seen))
            return {"input_ids": np.full((1, 8), index, np.int32)}

    assert audio_input_capability(_AudioModel(), _KeysOnIdentity()).capable is False


def test_a_processor_that_warms_up_on_its_first_call_is_still_capable():
    """One discarded call absorbs one-time state, so a real processor that
    differs only on its first call is not mistaken for an unrepeatable one."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _Warms(_ProbeProcessor):
        def __call__(self, text, audio=None, **kwargs):
            if self.calls == 0:
                self.calls += 1
                return {"input_ids": np.full((1, 8), 99, np.int32)}
            return super().__call__(text, audio, **kwargs)

    assert audio_input_capability(_AudioModel(), _Warms()).capable is True


def test_a_caller_supplied_text_answers_for_a_processor_that_needs_a_marker():
    """Which marker a processor needs is prompt-rendering knowledge, so the
    caller supplies candidates. A lone string is one candidate, not a sequence
    of characters -- the form the Studio consumer passes."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    processor = _ProbeProcessor(needs_text="<|audio|> transcribe")
    assert audio_input_capability(_AudioModel(), processor).capable is False
    verdict = audio_input_capability(
        _AudioModel(), _ProbeProcessor(needs_text="<|audio|> transcribe"),
        texts="<|audio|> transcribe",
    )
    assert verdict.capable is True


def test_a_processor_that_raises_answers_not_capable_without_escaping():
    """Totality: this runs at model-load time and may never fail a load."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _Raises(_ProbeProcessor):
        def __call__(self, text, audio=None, **kwargs):
            raise RuntimeError("no audio machinery here")

    verdict = audio_input_capability(_AudioModel(), _Raises())
    assert verdict.capable is False and "raised" in verdict.reason

    # A model that raises escapes the probe's own guards, so only the outer
    # one keeps this total.
    class _BrokenModel:
        def named_modules(self):
            raise RuntimeError("this model cannot be walked")

    broken = audio_input_capability(_BrokenModel(), _ProbeProcessor())
    assert broken.capable is False and "could not run" in broken.reason


def test_audio_modules_are_found_wherever_the_family_nests_them():
    """The families spell it five ways, so the walk keys on the name. The
    nested path is a layout a fixed attribute list would miss -- not a claim
    about where any one family puts it; Phi-4's pair sits at the top level."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    for names in (("audio_tower", "embed_audio"),
                  ("audio_tower", "audio_projection_layer"),
                  ("embed_tokens_extend.audio_embed.audio_encoder",)):
        verdict = audio_input_capability(_AudioModel(names), _ProbeProcessor())
        assert verdict.model_ok is True and verdict.capable is True, names
    absent = audio_input_capability(
        _AudioModel(("language_model", "vision_tower")), _ProbeProcessor())
    assert absent.model_ok is False and absent.capable is False


def test_without_a_model_the_check_can_refute_but_never_affirm():
    """A processor-only call still answers for an unloadable export, and says
    plainly that it did not look at a model."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    verdict = audio_input_capability(None, _ProbeProcessor())
    assert verdict.capable is False
    assert verdict.processor_ok is True and verdict.model_ok is None


def test_mlx_array_outputs_are_fingerprinted_by_their_own_values():
    """Real processors return MLX arrays, and exact integer outputs must not be
    collapsed by a lossy cast on the way to comparison."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _MXOut(_ProbeProcessor):
        def __call__(self, text, audio=None, **kwargs):
            self.calls += 1
            # Two values a float32 round-trip cannot tell apart.
            value = 16777216 + int(abs(self._samples(audio)[1]) > 0.05)
            return {"input_ids": mx.array([[value]], dtype=mx.int32)}

    assert audio_input_capability(_AudioModel(), _MXOut()).capable is True


def test_a_later_candidate_text_still_answers():
    """The contract takes a sequence, so a processor whose marker appears only
    in a later candidate must still be found capable."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    processor = _ProbeProcessor(needs_text="second <|audio|>")
    verdict = audio_input_capability(
        _AudioModel(), processor, texts=["first", "second <|audio|>"],
    )
    assert verdict.capable is True


def test_a_processor_taking_only_audios_is_called_with_that_keyword():
    """Families disagree on the keyword; resolving it wrongly would refuse a
    capable checkpoint for a reason that has nothing to do with its audio."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _AudiosOnly(_ProbeProcessor):
        def __call__(self, text, audios=None, **kwargs):
            self.calls += 1
            return {"input_ids": np.zeros((1, 8), np.int32),
                    "audio_features": self._samples(audios)[:16].copy()}

    assert audio_input_capability(_AudioModel(), _AudiosOnly()).capable is True


def test_even_a_base_exception_cannot_escape_the_capability_check():
    """It runs at model-load time, so nothing it touches may abort a load --
    including the interrupt-shaped exceptions a narrower guard would let past."""
    from unsloth_zoo.mlx.utils import audio_input_capability

    class _Interrupts:
        def named_modules(self):
            raise KeyboardInterrupt("interrupted mid-walk")

    verdict = audio_input_capability(_Interrupts(), _ProbeProcessor())
    assert verdict.capable is False and "could not run" in verdict.reason


def test_the_repeated_audio_placeholder_is_never_a_target(monkeypatch):
    """Phi-4 spells its placeholder `<|endoftext11|>` and declares it in neither
    the tokenizer attributes nor its checkpoint config, so the names the loss
    mask collects from do not reach it and every audio position was supervised.
    It is resolved the same way run counting resolves it."""
    from unsloth_zoo.mlx.utils import (
        _get_vlm_audio_soft_token_ids, _get_vlm_ignore_token_ids,
    )

    class _UndeclaredPlaceholder(_FakeProcessor):
        # Phi-4's shape: the placeholder is in the vocabulary and nowhere else.
        # No `audio_token` attribute, no config index -- only the spelling.
        tokenizer = type("_Tok", (_FakeTokenizer,), {
            "_vocab": dict(_FakeTokenizer._vocab, **{"<|endoftext11|>": 200011}),
        })()

    processor = _UndeclaredPlaceholder()
    assert not hasattr(processor.tokenizer, "audio_token")
    soft = _get_vlm_audio_soft_token_ids(processor)
    assert soft, "this fixture must have a resolvable placeholder"
    ignored = _get_vlm_ignore_token_ids(processor=processor) or []
    for token_id in soft:
        assert token_id in ignored, token_id
