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
