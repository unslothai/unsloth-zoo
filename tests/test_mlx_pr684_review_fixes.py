from __future__ import annotations

import inspect

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)


def _skip_if_mlx_core_was_replaced():
    import mlx.core as current_mx
    if current_mx is not mx:
        pytest.skip("requires real MLX runtime without mlx_simulation monkeypatch")


class _TinyTokenizer:
    pad_token_id = 2
    eos_token_id = 2
    unk_token_id = -1
    image_token_id = 200

    def encode(self, text):
        return [int(part) for part in str(text).split()]

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(item) for item in token]
        return {"<image>": 200, "<|image_pad|>": 201}.get(token, self.unk_token_id)


class _ContentProcessor:
    tokenizer = _TinyTokenizer()
    image_processor = object()

    def __call__(self, text, **_kwargs):
        rows = [[int(item), 200, 2] for item in text]
        masks = [[1, 1, 1] for _ in rows]
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }


def test_vlm_ignore_ids_exclude_pad_even_when_pad_is_eos():
    from unsloth_zoo.mlx.utils import _get_vlm_ignore_token_ids

    ids = _get_vlm_ignore_token_ids(
        processor=_ContentProcessor(),
        config={"pad_token_id": 2, "image_token_id": 200},
    )

    assert 200 in ids
    assert 2 not in ids


def test_vlm_label_mask_keeps_in_sequence_pad_eos_token():
    from unsloth_zoo.mlx.utils import _apply_vlm_label_masks

    batch = {
        "input_ids": mx.array([[101, 2, 200, 2]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 0]], dtype=mx.int32),
    }
    out = _apply_vlm_label_masks(
        batch,
        labels=batch["input_ids"],
        ignore_token_ids=[200],
    )

    assert out.tolist() == [[101, 2, -100, -100]]


def test_manual_weight_decay_accepts_scalar_lr_and_preserves_dtype():
    from mlx.utils import tree_flatten
    from unsloth_zoo.mlx.trainer import MLXTrainer

    class TinyModel:
        def __init__(self):
            self.params = {
                "layer": {
                    "weight": mx.array([10.0], dtype=mx.bfloat16),
                    "bias": mx.array([10.0], dtype=mx.bfloat16),
                },
                "norm": {"weight": mx.array([10.0], dtype=mx.float32)},
            }

        def trainable_parameters(self):
            return self.params

        def update(self, updates):
            def merge(dst, src):
                for key, value in src.items():
                    if isinstance(value, dict):
                        merge(dst[key], value)
                    else:
                        dst[key] = value
            merge(self.params, updates)

    class TinyOptimizer:
        learning_rate = 0.1

    model = TinyModel()
    grad = {
        "layer": {
            "weight": mx.array([1.0], dtype=mx.bfloat16),
            "bias": mx.array([1.0], dtype=mx.bfloat16),
        },
        "norm": {"weight": mx.array([1.0], dtype=mx.float32)},
    }
    trainer = object.__new__(MLXTrainer)
    trainer._manual_weight_decay = 0.1

    trainer._apply_manual_weight_decay(model, TinyOptimizer(), grad)
    flat = dict(tree_flatten(model.trainable_parameters()))

    assert flat["layer.weight"].dtype == mx.bfloat16
    assert flat["layer.weight"].item() < 10.0
    assert flat["layer.bias"].item() == pytest.approx(10.0)
    assert flat["norm.weight"].item() == pytest.approx(10.0)


def test_nf4_dense_zero_group_dequantizes_to_zero_without_epsilon_scale():
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.loader import _nf4_dense_dequantize_weight

    weight = mx.zeros((1, 4), dtype=mx.float32)
    out = _nf4_dense_dequantize_weight(weight, group_size=4)

    assert out.tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_ordered_text_batches_raise_clear_error_when_all_rows_drop():
    from unsloth_zoo.mlx.utils import create_ordered_batches

    with pytest.raises(ValueError, match="no trainable token sequences"):
        create_ordered_batches(
            dataset=[{"text": "1"}],
            tokenizer=_TinyTokenizer(),
            batch_size=1,
            max_seq_length=1,
            dataset_order="sequential",
        )


def test_ordered_text_torch_randperm_can_materialize_multiple_epochs():
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import create_ordered_batches

    batches = create_ordered_batches(
        dataset=[{"text": f"{i} {i + 10}"} for i in range(5)],
        tokenizer=_TinyTokenizer(),
        batch_size=1,
        max_seq_length=4,
        seed=None,
        dataset_order="torch_randperm",
        num_epochs=2,
    )

    first_epoch = [int(batch[0, 0].item()) for batch, _lengths, _labels in batches[:5]]
    second_epoch = [int(batch[0, 0].item()) for batch, _lengths, _labels in batches[5:]]
    assert len(batches) == 10
    assert sorted(first_epoch) == [0, 1, 2, 3, 4]
    assert sorted(second_epoch) == [0, 1, 2, 3, 4]
    assert first_epoch != second_epoch


def test_vlm_torch_randperm_seed_none_and_multi_epoch_batches():
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import create_vlm_batches

    batches = create_vlm_batches(
        dataset=[{"text": str(i)} for i in range(5)],
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        seed=None,
        dataset_order="torch_randperm",
        num_epochs=2,
    )

    first_epoch = [int(batch["input_ids"][0, 0].item()) for batch in batches[:5]]
    second_epoch = [int(batch["input_ids"][0, 0].item()) for batch in batches[5:]]
    assert len(batches) == 10
    assert sorted(first_epoch) == [0, 1, 2, 3, 4]
    assert sorted(second_epoch) == [0, 1, 2, 3, 4]
    assert first_epoch != second_epoch


def test_pr684_compiler_review_guards_are_present():
    import unsloth_zoo.compiler as compiler
    import unsloth_zoo.mlx.compile as mlx_compile

    compiler_source = inspect.getsource(compiler)
    mlx_compile_source = inspect.getsource(mlx_compile)

    assert (
        'self.loss_function.__name__.endswith("ForCausalLMLoss") '
        "and labels is not None and NOT_RETURN_LOGITS"
    ) in compiler_source
    assert '"weight" in norm' not in mlx_compile_source
    assert '"bias" in norm' not in mlx_compile_source
    assert 'getattr(norm, "weight", None)' in mlx_compile_source


def test_norm_output_cast_discovers_custom_norms_from_loaded_model():
    _skip_if_mlx_core_was_replaced()
    import mlx.nn as nn

    gemma3_text = pytest.importorskip("mlx_lm.models.gemma3_text")
    stablelm = pytest.importorskip("mlx_lm.models.stablelm")
    fastvlm_vision = pytest.importorskip("mlx_vlm.models.fastvlm.vision")
    import unsloth_zoo.mlx.trainer as trainer_mod

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = gemma3_text.RMSNorm(4)
            self.q_layernorm = stablelm.LayerNormPerHead(
                head_dim=4, num_heads=2, eps=1e-5
            )
            self.norm = fastvlm_vision.LayerNormChannel(num_features=4)

    trainer_mod._set_norm_output_cast_to_input_dtype(False)
    model = TinyModel()
    cases = [
        (model.input_layernorm, mx.ones((2, 4), dtype=mx.bfloat16)),
        (
            model.q_layernorm,
            mx.ones((1, 3, 2, 4), dtype=mx.bfloat16),
        ),
        (
            model.norm,
            mx.ones((1, 2, 2, 4), dtype=mx.bfloat16),
        ),
    ]

    norm_classes = trainer_mod._iter_norm_output_cast_classes(model)
    for norm, x in cases:
        assert type(norm) in norm_classes
        raw = norm(x)
        assert raw.dtype == mx.float32

    try:
        trainer_mod._set_norm_output_cast_to_input_dtype(True, model)
        for norm, x in cases:
            out = norm(x)
            assert out.dtype == x.dtype
    finally:
        trainer_mod._set_norm_output_cast_to_input_dtype(False)


def test_norm_output_cast_does_not_double_patch_inherited_norm_call():
    _skip_if_mlx_core_was_replaced()
    import mlx.nn as nn
    import unsloth_zoo.mlx.trainer as trainer_mod

    class CustomRMSNorm(nn.RMSNorm):
        pass

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = CustomRMSNorm(4)

    trainer_mod._set_norm_output_cast_to_input_dtype(False)
    model = TinyModel()
    x = mx.ones((2, 4), dtype=mx.bfloat16)

    try:
        trainer_mod._set_norm_output_cast_to_input_dtype(True, model)
        assert nn.RMSNorm in trainer_mod._NORM_OUTPUT_CAST_PATCHED_CLASSES
        assert CustomRMSNorm not in trainer_mod._NORM_OUTPUT_CAST_PATCHED_CLASSES
        assert model.input_layernorm(x).dtype == x.dtype
    finally:
        trainer_mod._set_norm_output_cast_to_input_dtype(False)

    assert nn.RMSNorm not in trainer_mod._NORM_OUTPUT_CAST_PATCHED_CLASSES
    assert CustomRMSNorm not in trainer_mod._NORM_OUTPUT_CAST_PATCHED_CLASSES
    assert not getattr(
        CustomRMSNorm.__call__,
        "_unsloth_norm_output_cast_wrapper",
        False,
    )
