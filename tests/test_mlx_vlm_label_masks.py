from __future__ import annotations

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
