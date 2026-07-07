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
    batch = _collate_vlm_batch(
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

    kept, removed, formatted_items = _filter_trainable_vlm_indices(
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

    default_batch = _collate_vlm_batch(
        [{"prompt": "prompt", "completion": "completion"}],
        _PromptCompletionProcessor(),
        max_seq_length=8,
        image_size=16,
    )
    batch = _collate_vlm_batch(
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
    batch = _collate_vlm_batch(
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
    _collate_vlm_batch(
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


def test_vlm_top_level_image_key_is_not_cuda_images_alias():
    from unsloth_zoo.mlx.utils import _extract_vlm_images

    assert _extract_vlm_images({"image": "top-level"}, [], image_size=16) == []


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
