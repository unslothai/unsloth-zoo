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
    eager builder on merged main (two smoke modes; full matrix recorded)."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

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
        assert [plan.batch_index_for_visit(v) for v in range(2 * len(plan))] \
            == [v % len(plan) for v in range(2 * len(plan))], name
        assert _digest_vlm_batches(plan.materialize_all()) == GOLDENS[name], name


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


    # The fake-rank collective test covers no-materialization more strictly.


def test_vlm_family_invariant_against_live_mx_compile_traces():
    """The serializer's central safety invariant checked against OBSERVED
    mx.compile cache behavior, with families computed BEFORE the compile walk
    (production survey-then-compile order). 'merge' pairs share one plannable
    family and one trace; 'split' pairs (one per value-encoding rule) produce
    two traces and two still-plannable families; 'guard' pairs are cases MLX
    keys apart that one family may absorb ONLY by being unplannable; leaves
    the compile walk rejects are unplannable too. The exhaustive adversarial
    matrix is retained as recorded validation evidence outside the repo."""
    _skip_if_mlx_core_was_replaced()
    import collections

    from unsloth_zoo.mlx.utils import (
        _vlm_batch_family as family,
        _vlm_family_is_plannable as plannable,
    )

    ids = mx.zeros((2, 3), dtype=mx.int32)
    Point = collections.namedtuple("Point", "a b")

    class _LyingList(list):
        def __iter__(self):
            return iter([])

    cases = [
        ("merge", {"x": [ids, 1]}, {"x": (ids, 1)}),
        ("merge", {"flag": True, "x": ids}, {"flag": 1, "x": ids}),
        ("merge", {"x": ids, "t": _LyingList([1])}, {"x": ids, "t": [1]}),
        ("split", {"x": ids, "y": 1}, {"y": 1, "x": ids}),
        ("split", {"x": mx.zeros((2, 4), dtype=mx.int32)},
                  {"x": mx.zeros((2, 5), dtype=mx.int32)}),
        ("split", {"x": ids.astype(mx.int16)}, {"x": ids}),
        ("split", {"x": ids, "y": 0.0}, {"x": ids, "y": -0.0}),
        ("split", {"x": ids, "y": "a"}, {"x": ids, "y": "b"}),
        ("split", {"x": ids, b"a": 0}, {"x": ids, b"b": 0}),
        ("split", {"x": ids, (1, "k"): 0}, {"x": ids, (2, "k"): 0}),
        ("guard", {"x": ids, "t": Point(1, 2)}, {"x": ids, "t": Point(3, 4)}),
    ]
    for kind, left, right in cases:
        fam_left, fam_right = family(left), family(right)
        traces = []
        # The probe body ignores its input: mx.compile keys on the walk.
        compiled = mx.compile(lambda d: (traces.append(1), mx.ones(1))[1])
        compiled(left)
        compiled(right)
        same_key = len(traces) == 1
        if fam_left == fam_right and plannable(fam_left):
            assert same_key, (kind, left, right)
        if kind == "merge":
            assert same_key and fam_left == fam_right and plannable(fam_left)
        elif kind == "split":
            # Distinct values stay ELIGIBLE while splitting: making common
            # constants unplannable would regress ordinary batches to eager.
            assert not same_key and fam_left != fam_right
            assert plannable(fam_left) and plannable(fam_right)
        else:
            assert not plannable(fam_left) and not plannable(fam_right)
    rejected = [
        {"x": ids, "n": np.zeros((1,))},
        {"x": ids, "big": 2 ** 63},
        collections.UserDict({"x": ids}),
    ]
    for tree in rejected:
        assert not plannable(family(tree))
        with pytest.raises(Exception):
            mx.compile(lambda d: mx.ones(1))(tree)
    edge = {"x": ids, "lo": -(2 ** 63), "hi": 2 ** 63 - 1}
    assert plannable(family(edge))
    mx.compile(lambda d: mx.ones(1))(edge)
    assert family({"t": Point(1, 2)}) != family({"t": (1, 2)})


class _VarWidthProcessor(_CountingProcessor):
    """Batch width AND structure follow content: different scheduled
    batches produce genuinely distinct families even after the text axis
    goes symbolic (odd-width batches carry an extra sidecar array)."""

    def __call__(self, text, **kwargs):
        self.calls += 1
        width = 3 + max(int(item) % 3 for item in text)
        rows = [([int(item), 200] + [2] * width)[:width] for item in text]
        masks = [[1] * width for _ in rows]
        batch = {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }
        if width % 2:
            batch["row_flags"] = np.ones((len(rows), 1), dtype=np.int32)
        return batch


def test_vlm_plan_survey_is_lazy_idempotent_per_index_and_cache_free():
    """ensure_descriptors() never runs at construction, stores each index's
    OWN family (the fixture makes families differ across indices), builds
    once per batch, invalidates rather than reuses the plan cache, and is
    idempotent with no further processor work."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import (
        _create_vlm_batch_plan,
        _vlm_batch_family,
    )

    processor = _VarWidthProcessor()
    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(6)],
        processor=processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
    )
    assert processor.calls == 0
    with pytest.raises(RuntimeError, match="ensure_descriptors"):
        plan.batch_family(0)
    cached_batch = plan.materialize(0)
    assert processor.calls == 1 and plan._mru is not None
    descriptors = plan.ensure_descriptors()
    # The pre-populated cache is invalidated, not consulted.
    assert processor.calls == 1 + len(plan) == 4
    assert plan._mru is None
    assert plan.ensure_descriptors() is descriptors
    assert processor.calls == 4
    from unsloth_zoo.mlx.utils import _vlm_width_survey

    for index in range(len(plan)):
        rebuilt = plan._build_batch(index)
        width, axes, padable, _forbidden = _vlm_width_survey(rebuilt)
        assert padable and plan._padable[index]
        assert plan.batch_width(index) == width
        assert descriptors[index] == _vlm_batch_family(
            rebuilt, symbolic_axes=axes,
        )
    # Widths merge symbolically; the structural sidecar still splits.
    assert len(set(descriptors)) == 2
    assert plan.batch_family(1) == descriptors[1]
    assert len(descriptors) == 3
    assert plan._padable == (True, True, True)
    assert plan.materialize(0)["input_ids"].tolist() == (
        cached_batch["input_ids"].tolist()
    )


def test_vlm_plan_survey_does_not_consume_preprocessing_rng():
    """The survey is RNG-neutral: an augmenting processor sees the SAME draws
    whether or not it ran, so enabling compile cannot shift the training data
    stream. Without the save/restore the surveyed run augments differently."""
    _skip_if_mlx_core_was_replaced()
    import random as _random

    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    class _AugmentingProcessor(_VarWidthProcessor):
        """Width depends only on the items (so shapes are call-order
        independent); the pixel content is a random augmentation draw."""

        def __call__(self, text, **kwargs):
            batch = super().__call__(text, **kwargs)
            batch["pixel_values"] = np.stack([
                np.random.randint(0, 255, size=(4,)) for _ in text
            ]).astype(np.float32)
            return batch

    def _stream(survey):
        np.random.seed(4321)
        _random.seed(4321)
        plan = _create_vlm_batch_plan(
            dataset=[{"text": str(i)} for i in range(6)],
            processor=_AugmentingProcessor(),
            config={"image_size": 16, "image_token_id": 200},
            batch_size=2,
            max_seq_length=8,
        )
        if survey:
            plan.ensure_descriptors()
        return [
            plan.materialize(index)["pixel_values"].tolist()
            for index in range(len(plan))
        ], (np.random.randint(0, 1 << 30), _random.random())

    unsurveyed, unsurveyed_tail = _stream(False)
    surveyed, surveyed_tail = _stream(True)
    assert surveyed == unsurveyed
    # The survey also leaves no offset behind for anything drawing afterwards.
    assert surveyed_tail == unsurveyed_tail


def test_vlm_plan_survey_releases_each_batch_before_the_next_build(monkeypatch):
    """One-at-a-time TENSOR ownership: every mx.array leaf of every built
    batch is tracked by weakref, all of the previous batch's tensors are
    already collected before the next build starts, and none survive the
    survey."""
    _skip_if_mlx_core_was_replaced()
    import gc
    import weakref

    from unsloth_zoo.mlx.utils import FiniteVLMBatchPlan, _create_vlm_batch_plan

    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(6)],
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
    )

    live = []
    inner_build = FiniteVLMBatchPlan._build_batch

    def _array_leaves(node):
        if isinstance(node, mx.array):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from _array_leaves(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                yield from _array_leaves(item)

    def tracked_build(self, index):
        gc.collect()
        assert [ref for ref in live if ref() is not None] == [], (
            "previous survey batch's tensors still alive at next build"
        )
        batch = inner_build(self, index)
        # Nest one array so the tracker binds nested leaves, not just top level.
        batch["nested"] = [{"probe": mx.zeros((2, 2), dtype=mx.int32)}]
        tracked = 0
        for leaf in _array_leaves(batch):
            live.append(weakref.ref(leaf))
            tracked += 1
        assert tracked >= 3
        return batch

    monkeypatch.setattr(FiniteVLMBatchPlan, "_build_batch", tracked_build)
    plan.ensure_descriptors()
    gc.collect()
    assert len(live) >= 2 * len(plan)
    assert all(ref() is None for ref in live)


def test_vlm_family_drift_check_fails_hard_with_location():
    """The runtime drift seam accepts a faithful rebuild against its OWN
    index and hard-fails on added keys, dtype drift, shape drift, key-order
    drift, and cross-index confusion, naming the batch and the
    divergence."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(6)],
        processor=_VarWidthProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
    )
    plan.ensure_descriptors()
    batch = plan.materialize(1)
    assert plan.check_family_drift(1, batch) is None
    with pytest.raises(RuntimeError, match=r"batch 1 drifted.*'extra'"):
        plan.check_family_drift(1, {**batch, "extra": 1})
    retyped = {**batch, "input_ids": batch["input_ids"].astype(mx.int16)}
    with pytest.raises(RuntimeError, match=r"'input_ids'.*int16"):
        plan.check_family_drift(1, retyped)
    # A width change alone is NOT drift (the text axis is symbolic), but an
    # inconsistent one (one array narrowed, siblings unchanged) is.
    narrowed = {**batch, "input_ids": batch["input_ids"][:, :-1]}
    with pytest.raises(RuntimeError, match=r"batch 1 drifted"):
        plan.check_family_drift(1, narrowed)
    uniformly_narrowed = {
        key: (
            value[:, :-1]
            if isinstance(value, mx.array)
            and value.ndim == 2
            and value.shape[1] == batch["input_ids"].shape[1]
            else value
        )
        for key, value in batch.items()
    }
    assert plan.check_family_drift(1, uniformly_narrowed) is None
    # Non-first leaves are checked too, so pinning to input_ids cannot pass.
    other_key = next(
        key for key, value in batch.items()
        if key != "input_ids" and isinstance(value, mx.array)
    )
    remasked = {**batch, other_key: batch[other_key].astype(mx.float16)}
    with pytest.raises(RuntimeError, match=rf"'{other_key}'.*float16"):
        plan.check_family_drift(1, remasked)
    reordered = dict(reversed(list(batch.items())))
    with pytest.raises(RuntimeError, match="drifted"):
        plan.check_family_drift(1, reordered)
    # Families genuinely differ, so a checker pinned wrong cannot pass.
    assert plan.batch_family(0) != plan.batch_family(1)
    with pytest.raises(RuntimeError, match="batch 0 drifted"):
        plan.check_family_drift(0, batch)


class _WidthOnlyProcessor(_CountingProcessor):
    """One family; widths 5 vs 40 stay distinct after event-width rounding."""

    def __call__(self, text, **kwargs):
        self.calls += 1
        width = 5 + 35 * (max(int(item) % 2 for item in text))
        rows = [([int(item), 200] + [2] * width)[:width] for item in text]
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(
                [[1] * width for _ in rows], dtype=np.int32,
            ),
        }


def _vlm_planner_fixtures(processor=None, rows=6):
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    return _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(rows)],
        processor=processor or _WidthOnlyProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
    )


def test_vlm_should_raise_decision_aborts_inside_the_coordinated_block():
    """A compile decision that mandates an abort raises during planning —
    inside the coordinated block — rather than surviving as a benign
    non-planning state that would strand peers at a later rank-local
    raise."""
    _skip_if_mlx_core_was_replaced()
    from types import SimpleNamespace

    from unsloth_zoo.mlx.trainer import _plan_single_process_vlm_shapes

    plan = _vlm_planner_fixtures(rows=2)
    with pytest.raises(RuntimeError, match="compile cannot be enabled"):
        _plan_single_process_vlm_shapes(
            plan, None,
            args=SimpleNamespace(compile_max_variants=None,
                                 gradient_accumulation_steps=1),
            total_steps=2, distributed_world_size=2,
            compile_policy=SimpleNamespace(mode="strict"),
            compile_decision=SimpleNamespace(
                enabled=False, should_raise=True, arch="tiny",
                reason="unsupported architecture",
            ),
        )


def test_vlm_shape_planning_follows_the_resolved_override_mode():
    """The RESOLVED mode decides: an arch override selecting strict under a
    best_effort base must abort on a shape-planning failure, and the reverse
    override must still degrade under a strict base."""
    _skip_if_mlx_core_was_replaced()
    from types import SimpleNamespace

    from unsloth_zoo.mlx.compile import (
        MLXVLMCompilePolicy,
        get_compile_qualification,
        resolve_training_compile,
    )
    from unsloth_zoo.mlx.trainer import _plan_single_process_vlm_shapes

    arch = "qwen2_vl"
    qualification = get_compile_qualification(arch)
    if qualification is None or not qualification.training_compile:
        pytest.skip(f"{arch} is not training-compile qualified here")

    class _UnstableFamilyProcessor(_CountingProcessor):
        """An opaque batch key makes the compile-key family unplannable."""

        def __call__(self, text, **kwargs):
            batch = dict(super().__call__(text, **kwargs))
            batch[object()] = 1
            return batch

    plan = _vlm_planner_fixtures(processor=_UnstableFamilyProcessor(), rows=2)
    args = SimpleNamespace(
        compile_max_variants=None, gradient_accumulation_steps=1,
    )

    strict_override = MLXVLMCompilePolicy(
        mode="best_effort", arch_overrides=((arch, "strict"),),
    )
    decision = resolve_training_compile(arch, policy=strict_override)
    # Qualified arch: the decision is enabled, so planning runs past the
    # should_raise and unqualified guards and reaches the failure branches.
    assert decision.enabled and decision.policy_mode == "strict"
    assert not decision.fallback_allowed and not decision.should_raise
    with pytest.raises(RuntimeError, match="not stable enough"):
        _plan_single_process_vlm_shapes(
            plan, None, args=args, total_steps=len(plan),
            distributed_world_size=1, compile_policy=strict_override,
            compile_decision=decision,
        )

    lenient_override = MLXVLMCompilePolicy(
        mode="strict", arch_overrides=((arch, "best_effort"),),
    )
    lenient_decision = resolve_training_compile(arch, policy=lenient_override)
    assert lenient_decision.enabled and lenient_decision.fallback_allowed
    _shape_plan, report, allowed, _frontier = _plan_single_process_vlm_shapes(
        plan, None, args=args, total_steps=len(plan),
        distributed_world_size=1, compile_policy=lenient_override,
        compile_decision=lenient_decision,
    )
    assert not allowed and report.reason == "vlm_unplannable_family"


def test_vlm_automatic_planning_stays_exact_below_ceiling():
    """Below the ceiling: canonical event widths, no budget compression."""
    _skip_if_mlx_core_was_replaced()
    from types import SimpleNamespace

    from unsloth_zoo.mlx.trainer import _plan_single_process_vlm_shapes

    class _SteppedWidthProcessor(_CountingProcessor):
        def __call__(self, text, **kwargs):  # 32-step widths: distinct signatures

            self.calls += 1
            width = 5 + 32 * max(int(item) for item in text)
            rows = [([int(item), 200] + [2] * width)[:width] for item in text]
            return {"input_ids": np.array(rows, dtype=np.int32),
                    "attention_mask": np.array(
                        [[1] * width for _ in rows], dtype=np.int32)}

    plan = _vlm_planner_fixtures(processor=_SteppedWidthProcessor(), rows=40)
    _shape_plan, report, allowed, _frontier = _plan_single_process_vlm_shapes(
        plan, None,
        args=SimpleNamespace(compile_max_variants=None,
                             gradient_accumulation_steps=1),
        total_steps=len(plan), distributed_world_size=1,
        compile_policy=SimpleNamespace(mode="strict"),
        compile_decision=SimpleNamespace(enabled=True),
    )
    assert allowed and report.action == "exact"
    assert report.raw_signatures == report.planned_signatures == 40
    assert (report.cap_selection, report.budget_satisfied) == ("exact", True)
    assert (report.padding_work_fraction, report.max_width_stretch) == (0.0, 1.0)


class _ProcessorNativeLimitProcessor:
    """Emits its own native width and honours ``max_length`` only when given."""

    def __init__(self, native_width=12):
        self.tokenizer = _TinyTokenizer()
        self.image_processor = object()
        self.native_width = native_width
        self.seen_max_length = []

    def __call__(self, text, **kwargs):
        self.seen_max_length.append(kwargs.get("max_length", "<absent>"))
        width = self.native_width
        if kwargs.get("max_length") is not None:
            width = min(width, int(kwargs["max_length"]))
        rows = [([int(item), 200] + [7] * width)[:width] for item in text]
        masks = [[1] * width for _ in rows]
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(masks, dtype=np.int32),
        }


def test_vlm_batches_keep_unbounded_max_seq_length_as_none():
    """``max_seq_length=None`` means "use the processor's own limit"."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan, create_vlm_batches

    processor = _ProcessorNativeLimitProcessor(native_width=12)
    batches = create_vlm_batches(
        dataset=[{"text": str(i)} for i in range(4)],
        processor=processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=None,
        dataset_order="sequential",
    )

    # The processor is never handed a cap, so nothing is truncated.
    assert set(processor.seen_max_length) == {"<absent>"}
    assert len(batches) == 2
    assert all(batch["input_ids"].shape == (2, 12) for batch in batches)

    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(4)],
        processor=_ProcessorNativeLimitProcessor(native_width=12),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=None,
        dataset_order="sequential",
    )
    assert plan.max_seq_length is None

    # A finite cap still coerces to int and still truncates.
    capped_processor = _ProcessorNativeLimitProcessor(native_width=12)
    capped = create_vlm_batches(
        dataset=[{"text": str(i)} for i in range(4)],
        processor=capped_processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=5,
        dataset_order="sequential",
    )
    assert set(capped_processor.seen_max_length) == {5}
    assert all(batch["input_ids"].shape == (2, 5) for batch in capped)


def test_vlm_plan_refreshes_visit_dependent_formatting_per_occurrence():
    """A stochastic or epoch-dependent formatting_func must refresh on every
    scheduled visit, the way the eager builder did when it formatted inside
    each batch build, while re-materialization stays free of user code."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    visits = {}

    def formatting_func(item):
        row = int(item["row"])
        visit = visits.get(row, 0)
        visits[row] = visit + 1
        return {"text": str(10 * visit + row)}

    plan = _create_vlm_batch_plan(
        dataset=[{"row": i} for i in range(3)],
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=6,
        dataset_order="sequential",
        formatting_func=formatting_func,
    )

    assert visits == {0: 2, 1: 2, 2: 2}
    heads = [int(plan[i]["input_ids"][0, 0].item()) for i in range(len(plan))]
    assert heads == [0, 1, 2, 10, 11, 12]

    plan.materialize_all()
    plan.ensure_descriptors()
    assert visits == {0: 2, 1: 2, 2: 2}
    assert [int(plan[i]["input_ids"][0, 0].item()) for i in range(len(plan))] == heads


def test_vlm_plan_formats_only_scheduled_rows_and_compacts_without_a_formatter():
    """Unscheduled rows never reach the formatter, and the plans that need no
    per-visit formatting keep storing one row per referenced dataset index."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    seen = []

    def formatting_func(item):
        seen.append(int(item["row"]))
        return {"text": str(item["row"])}

    plan = _create_vlm_batch_plan(
        dataset=[{"row": i} for i in range(6)],
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=4,
        dataset_order="sequential",
        formatting_func=formatting_func,
    )
    assert seen == [0, 1, 2, 3]
    assert len(plan.rows) == 4

    plain = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(3)],
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=6,
        dataset_order="sequential",
    )
    assert (len(plain), len(plain.rows)) == (6, 3)


class _AugProcessor:
    """Index-deterministic widths, globally-drawn pixel content."""

    def __init__(self):
        self.calls = 0
        self.tokenizer = type("T", (), {"pad_token_id": 7})()

    def __call__(self, text, **kwargs):
        self.calls += 1
        width = 4 if max(int(t) for t in text) < 2 else 40
        rows = [([int(t), 200] + [3] * width)[:width] for t in text]
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.ones((len(rows), width), dtype=np.int32),
            "labels": np.array(rows, dtype=np.int32),
            # Extent 6 never equals a text width, so batches stay padable.
            "pixel_values": np.stack([
                np.random.randint(0, 255, size=(6,)) for _ in text
            ]).astype(np.float32),
        }


def _make_plan():
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    np.random.seed(4321)
    return _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(6)],
        processor=_AugProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=64,
    )


def _install(plan):
    from unsloth_zoo.mlx import shape_guard as SG

    plan.ensure_descriptors()
    widths = plan.planned_event_widths()
    counts = {}
    for i in range(len(plan)):
        key = (plan.batch_family(i), widths[i], "full_step", len(plan.schedule[i]))
        counts[key] = counts.get(key, 0) + 1
    events = tuple(
        SG.TextShapeEvent(
            family=f, width=w, phase=p, frequency=n, local_batch_size=b,
        )
        for (f, w, p, b), n in counts.items()
    )
    plan.set_shape_plan(
        SG.select_text_shape_padding_budget(
            SG.build_text_shape_frontier(
                events, compile_scope=SG.FULL_STEP_SCOPE,
            ),
            exact_signature_threshold=SG.AUTOMATIC_TEXT_COMPILE_CEILING,
        ),
        widths,
    )
    return widths


def _pixels(batch):
    return np.asarray(batch["pixel_values"]).tolist()


def test_vlm_plan_is_excluded_from_eager_fallback_refetch():
    """The fallback refetch is restricted to plans whose rebuild is free.

    FiniteTextBatchPlan rebuilds from stored token ids and touches no RNG, so
    it keeps unpadding on fallback. FiniteVLMBatchPlan reruns the caller's
    processor, so it must reuse the materialized batch instead.
    """
    from unsloth_zoo.mlx import trainer as trainer_module
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, FiniteVLMBatchPlan

    assert FiniteTextBatchPlan in trainer_module._FINITE_BATCH_PLAN_TYPES
    assert FiniteVLMBatchPlan in trainer_module._FINITE_BATCH_PLAN_TYPES
    # Absent constant means the fallback still refetches every finite plan.
    refetchable = getattr(
        trainer_module,
        "_EAGER_REFETCHABLE_PLAN_TYPES",
        trainer_module._FINITE_BATCH_PLAN_TYPES,
    )
    assert FiniteTextBatchPlan in refetchable
    assert FiniteVLMBatchPlan not in refetchable


def test_vlm_fallback_refetch_would_shift_the_augmentation_stream():
    """Why the exclusion exists: the refetch is a second processor call whose
    draws offset every later batch, and the reuse it replaces is loss-safe."""
    # Reference: an eager-from-start run, which never surveys or materializes.
    plan = _make_plan()
    eager = [_pixels(plan[i]) for i in range(len(plan))]
    eager_tail = int(np.random.randint(0, 1 << 30))

    # What the removed refetch did: materialize, then re-index the same visit.
    plan = _install_and_get()
    calls_before = plan._processor.calls
    plan.materialize(0, phase="full_step")
    refetched = [_pixels(plan[0])]
    refetched += [_pixels(plan[i]) for i in range(1, len(plan))]
    refetch_tail = int(np.random.randint(0, 1 << 30))
    assert plan._processor.calls - calls_before == len(plan) + 1, (
        "the refetch is an extra processor call"
    )
    assert sum(a != b for a, b in zip(refetched, eager)) == len(eager), (
        "every batch from the fallback batch onward diverges from an eager run"
    )
    assert refetch_tail != eager_tail

    # What the fix does: reuse the materialized batch, drawing nothing extra.
    plan = _install_and_get()
    calls_before = plan._processor.calls
    reused = [_pixels(plan.materialize(0, phase="full_step"))]
    reused += [_pixels(plan[i]) for i in range(1, len(plan))]
    reuse_tail = int(np.random.randint(0, 1 << 30))
    assert plan._processor.calls - calls_before == len(plan)
    assert reused == eager, "reuse keeps the compiled run bit-identical to eager"
    assert reuse_tail == eager_tail


def _install_and_get():
    plan = _make_plan()
    _install(plan)
    return plan


def test_reused_padded_vlm_batch_is_loss_equivalent():
    """The reused batch carries planned padding; masking makes it free."""
    plan = _make_plan()
    widths = _install(plan)
    raw = [plan.batch_width(i) for i in range(len(plan))]
    assert widths != tuple(raw), "fixture must actually exercise padding"

    def masked_ce(batch):
        ids, lab = batch["input_ids"], batch["labels"]
        base = mx.arange(256, dtype=mx.float32).reshape(1, 1, 256)
        logits = ids.astype(mx.float32)[..., None] * 0.01 + base * 0.001
        mask = lab != -100
        safe = mx.where(mask, lab, mx.zeros_like(lab))
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(lp, safe[..., None], axis=-1)[..., 0]
        total = mx.sum(mx.where(mask, -picked, mx.zeros_like(picked)))
        return float(total.item()), int(mx.sum(mask).item())

    for index in range(len(plan)):
        padded = plan.materialize(index, phase="full_step")
        plan._mru = None
        unpadded = plan[index]
        assert masked_ce(padded) == masked_ce(unpadded), (
            f"batch {index}: planned padding changed the masked loss"
        )

class _NoPadTokenizer:
    pad_token_id = None
    eos_token_id = 2
    unk_token_id = -1
    image_token_id = 200

    def encode(self, text):
        return [int(part) for part in str(text).split()]

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(item) for item in token]
        return {"<image>": 200, "<|image_pad|>": 201}.get(token, self.unk_token_id)


class _UniformNoPadProcessor:
    """Uniform width, no tokenizer pad id: every planned endpoint would equal
    the raw width, so this plan looks like it never needs to pad."""

    image_processor = object()

    def __init__(self, width=32, narrowed=None):
        self.tokenizer = _NoPadTokenizer()
        self.width = width
        self.narrowed = narrowed
        self.training = False
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        width = self.narrowed if self.training else self.width
        rows = [([int(item), 200] + [2] * width)[:width] for item in text]
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array(
                [[1] * width for _ in rows], dtype=np.int32,
            ),
        }


def _plan_for(processor, rows=6):
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    return _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(rows)],
        processor=processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=None,
    )


def test_vlm_missing_pad_token_degrades_before_any_survey_work():
    """No pad id degrades to eager, and aborts under strict, before the survey
    runs, even when uniform widths mean the plan would never widen."""
    _skip_if_mlx_core_was_replaced()
    from types import SimpleNamespace

    from unsloth_zoo.mlx.trainer import _plan_single_process_vlm_shapes

    args = SimpleNamespace(
        compile_max_variants=None, gradient_accumulation_steps=1,
    )
    processor = _UniformNoPadProcessor()
    plan = _plan_for(processor)
    shape_plan, report, allowed, _frontier = _plan_single_process_vlm_shapes(
        plan, None, args=args, total_steps=len(plan),
        distributed_world_size=1,
        compile_policy=SimpleNamespace(mode="best_effort"),
        compile_decision=SimpleNamespace(enabled=True),
    )
    assert not allowed
    assert (report.action, report.reason) == ("eager", "vlm_pad_token_unavailable")
    assert shape_plan is None
    # The survey materializes every scheduled batch; a plan that can never be
    # compiled must not pay for it.
    assert processor.calls == 0

    strict_processor = _UniformNoPadProcessor()
    strict_plan = _plan_for(strict_processor)
    with pytest.raises(RuntimeError, match="without a tokenizer pad id"):
        _plan_single_process_vlm_shapes(
            strict_plan, None, args=args, total_steps=len(strict_plan),
            distributed_world_size=1,
            compile_policy=SimpleNamespace(mode="strict"),
            compile_decision=SimpleNamespace(enabled=True),
        )
    assert strict_processor.calls == 0


def test_vlm_uniform_width_plan_would_abort_mid_run_without_a_pad_id():
    """Why the guard cannot wait for "does the plan widen?": a later uniformly
    narrower rebuild passes the drift check, then cannot be widened back."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.shape_guard import FULL_STEP_SCOPE, phase_for_microstep

    processor = _UniformNoPadProcessor(width=32, narrowed=30)
    plan = _plan_for(processor)
    plan.ensure_descriptors()
    raw = tuple(plan.batch_width(i) for i in range(len(plan)))
    planned = plan.planned_event_widths()
    # The premise the reviewer relies on really does hold at survey time.
    assert raw == planned == (32,) * len(plan)

    # Training starts and the processor rebuilds every array uniformly narrower.
    processor.training = True
    narrowed = plan._build_batch(0)
    assert narrowed["input_ids"].shape[1] == 30
    # A uniform narrowing is deliberately NOT family drift.
    assert plan.check_family_drift(0, narrowed) is None

    from unsloth_zoo.mlx.utils import _finalize_vlm_batch_width

    with pytest.raises(ValueError, match="pad id is required to pad"):
        _finalize_vlm_batch_width(dict(narrowed), 32, plan.pad_token_id)
