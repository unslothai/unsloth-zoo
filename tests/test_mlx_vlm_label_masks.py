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
