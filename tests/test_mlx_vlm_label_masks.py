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
