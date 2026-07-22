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


def test_ordered_streaming_batches_drop_one_token_rows():
    from unsloth_zoo.mlx.utils import iterate_training_batches

    stream = ({"text": text} for text in ["1", "2 3"])
    batch, lengths, _labels = next(iterate_training_batches(
        stream, _TinyTokenizer(), 1, 4,
        dataset_order="sequential", append_eos=False,
    ))

    assert (batch.tolist()[0][:2], lengths.tolist()[0]) == ([2, 3], [0, 2])


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


def test_compiler_review_guards_are_present():
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

    class StableScale(nn.RMSNorm):
        def __call__(self, x):
            return x.astype(mx.float32)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = gemma3_text.RMSNorm(4)
            self.q_layernorm = stablelm.LayerNormPerHead(
                head_dim=4, num_heads=2, eps=1e-5
            )
            self.norm = fastvlm_vision.LayerNormChannel(num_features=4)
            self.scale = StableScale(4)

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
        (model.scale, mx.ones((2, 4), dtype=mx.bfloat16)),
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
    import unsloth_zoo.mlx.utils as utils_mod

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
        patched_classes = utils_mod.mlx_norm_output_cast_patched_classes()
        assert nn.RMSNorm in patched_classes
        assert CustomRMSNorm not in patched_classes
        assert model.input_layernorm(x).dtype == x.dtype

        state = utils_mod.snapshot_mlx_norm_output_cast_state(
            trainer_mod._iter_norm_output_cast_classes(model)
        )
        trainer_mod._set_norm_output_cast_to_input_dtype(False, model)
        utils_mod.restore_mlx_norm_output_cast_state(state)
        assert "__call__" not in CustomRMSNorm.__dict__
    finally:
        trainer_mod._set_norm_output_cast_to_input_dtype(False)

    patched_classes = utils_mod.mlx_norm_output_cast_patched_classes()
    assert nn.RMSNorm not in patched_classes
    assert CustomRMSNorm not in patched_classes
    assert not getattr(
        CustomRMSNorm.__call__,
        "_unsloth_norm_output_cast_wrapper",
        False,
    )


class _CountingProcessor(_ContentProcessor):
    def __init__(self):
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        return super().__call__(text, **kwargs)


def _digest_vlm_batches(batches):
    """Complete-pytree digest: sorted keys, per-array dtype/shape/values,
    non-array constants verbatim — the serialization the frozen goldens use."""
    out = []
    for batch in batches:
        entry = []
        for key in sorted(batch.keys()):
            value = batch[key]
            if isinstance(value, mx.array):
                def _tuplify(x):
                    return (
                        tuple(_tuplify(item) for item in x)
                        if isinstance(x, list) else x
                    )
                # dtype-native values: float drift must fail the oracle.
                entry.append((
                    key, str(value.dtype), tuple(value.shape),
                    _tuplify(value.tolist()),
                ))
            else:
                entry.append((key, "const", repr(value)))
        out.append(tuple(entry))
    return tuple(out)


class _MultiModalityStyleProcessor(_ContentProcessor):
    """Representative of prepare-time sequence expansion: negative-free
    placeholder ids are repeated by _expand_image_token_sequences for
    multi_modality model types, changing the final text width."""

    def __call__(self, text, **kwargs):
        rows = [[int(t), 250, 2] for t in text]
        return {
            "input_ids": np.array(rows, dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]] * len(rows), dtype=np.int32),
            "pixel_values": np.full((len(rows), 2), 0.5, dtype=np.float32),
        }


class _FakeWorld:
    def __init__(self, rank): self._rank = rank
    def rank(self): return self._rank
    def size(self): return 2


def test_finite_vlm_plan_reproduces_merged_main_goldens():
    """Independent oracle: complete-pytree digests frozen from the pre-plan
    eager builder on merged main. Two smoke modes cover the plain default
    epoch-replay path and the multi_modality prepare-time image-token
    expansion path; the full 11-mode golden matrix is retained as recorded
    validation evidence outside the repository."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    # Frozen full-pytree batch digests captured from the pre-plan eager VLM
    # builder on merged main (independent parity oracle). Values are
    # dtype-native; regenerate only from a tree WITHOUT the plan diff.
    GOLDENS = {
        "default_epoch_replay": ((('_unsloth_raw_input_ids_for_labels', 'mlx.core.int32', (1, 3), ((0, 200, 2),)), ('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((0, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((0, -100, 2),))), (('_unsloth_raw_input_ids_for_labels', 'mlx.core.int32', (1, 3), ((1, 200, 2),)), ('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((1, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((1, -100, 2),))), (('_unsloth_raw_input_ids_for_labels', 'mlx.core.int32', (1, 3), ((2, 200, 2),)), ('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((2, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((2, -100, 2),))), (('_unsloth_raw_input_ids_for_labels', 'mlx.core.int32', (1, 3), ((3, 200, 2),)), ('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((3, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((3, -100, 2),))), (('_unsloth_raw_input_ids_for_labels', 'mlx.core.int32', (1, 3), ((4, 200, 2),)), ('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((4, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((4, -100, 2),)))),
        "multi_modality_expansion": ((('attention_mask', 'mlx.core.int32', (1, 5), ((1, 1, 1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 5), ((0, 250, 250, 250, 2),)), ('labels', 'mlx.core.int64', (1, 5), ((0, -100, -100, -100, 2),)), ('pixel_values', 'mlx.core.float32', (1, 2), ((0.5, 0.5),))), (('attention_mask', 'mlx.core.int32', (1, 5), ((1, 1, 1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 5), ((1, 250, 250, 250, 2),)), ('labels', 'mlx.core.int64', (1, 5), ((1, -100, -100, -100, 2),)), ('pixel_values', 'mlx.core.float32', (1, 2), ((0.5, 0.5),)))),
    }
    cfg = {"image_size": 16, "image_token_id": 200}
    ds5 = [{"text": str(i)} for i in range(5)]
    cases = {
        "default_epoch_replay": (dict(dataset=ds5, batch_size=1, max_seq_length=8), _ContentProcessor(), cfg),
        "multi_modality_expansion": (dict(dataset=[{"text": str(i)} for i in range(2)], batch_size=1, max_seq_length=16, dataset_order="sequential"), _MultiModalityStyleProcessor(), {"image_size": 16, "image_token_id": 200, "image_token_index": 250, "num_image_tokens": 3, "model_type": "multi_modality"}),
    }
    assert sorted(cases) == sorted(GOLDENS)
    for name, (kwargs, processor, config) in cases.items():
        kwargs = dict(kwargs)
        dataset = kwargs.pop("dataset")
        plan = _create_vlm_batch_plan(
            dataset=dataset, processor=processor, config=config, **kwargs,
        )
        assert plan.visit_policy == "identity", name
        assert _digest_vlm_batches(plan.materialize_all()) == GOLDENS[name], name
        assert [
            plan.batch_index_for_visit(v) for v in range(2 * len(plan))
        ] == [v % len(plan) for v in range(2 * len(plan))], name


def test_checker_reaches_collective_without_materialization_on_fake_rank(monkeypatch):
    """The pad-slot rank has bad>0 metadata; the checker must reach the
    all_sum collective without any processor call or materialization even
    when the processor would fail — a failing rank must not strand peers."""
    _skip_if_mlx_core_was_replaced()
    import mlx.core as mx_core
    from unsloth_zoo.mlx.trainer import _check_vlm_all_masked
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    processor = _CountingProcessor()
    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(5)],
        processor=processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        comm_group=_FakeWorld(1),
        distributed_pad_mode="empty",
    )
    assert plan.supervision_counts(100) == (1, 2)  # pad slot counts bad

    def poisoned(self, *_a, **_k):
        raise AssertionError("checker must not invoke the processor")

    # Special methods resolve on the type: poison the class, not the instance.
    monkeypatch.setattr(_CountingProcessor, "__call__", poisoned)
    calls_before_check = processor.calls
    collective_calls = []
    real_all_sum = mx_core.distributed.all_sum

    def spy_all_sum(value, **kwargs):
        collective_calls.append(value.tolist())
        return value  # identity: single fake rank stands in for the sum

    monkeypatch.setattr(mx_core.distributed, "all_sum", spy_all_sum)
    try:
        _check_vlm_all_masked(
            plan, max_check=100, comm_group=_FakeWorld(1), world_size=2,
        )
    finally:
        monkeypatch.setattr(mx_core.distributed, "all_sum", real_all_sum)
    assert collective_calls == [[1, 2]]
    assert processor.calls == calls_before_check


    # No-materialization proof lives in the fake-rank collective test, which
    # covers this single-process property strictly (poisoned class __call__).
