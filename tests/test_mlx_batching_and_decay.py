from __future__ import annotations

import collections
import inspect
import json
import os

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


def test_ordered_text_fractional_num_epochs_builds_the_partial_pass():
    # int(num_epochs) rounded the requested row count down: 0 < epochs < 1 built
    # an empty plan, surfacing later as "No training batches created" and
    # blaming the dataset, and 1.5 built a single pass. Five rows at batch 1 is
    # 3 batches for half an epoch and 8 for one and a half.
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import create_ordered_batches

    def plan_for(num_epochs):
        return create_ordered_batches(
            dataset=[{"text": f"{i} {i + 10}"} for i in range(5)],
            tokenizer=_TinyTokenizer(),
            batch_size=1,
            max_seq_length=4,
            seed=None,
            dataset_order="torch_randperm",
            num_epochs=num_epochs,
        )

    assert len(plan_for(0.5)) == 3
    assert len(plan_for(1.5)) == 8
    # Whole counts are unchanged.
    assert len(plan_for(2)) == 10


def test_ordered_text_fractional_epochs_match_transformers_step_budget():
    # Golden values measured by running a real transformers.Trainer on the same
    # shapes: HF quantizes a fractional num_train_epochs to whole accumulation
    # windows and re-iterates the dataloader, so 0.5 epochs of 5 rows at batch 2
    # and accum 2 is one update over 4 rows, not a proportional 3 rows.
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import create_ordered_batches

    def rows_for(n_rows, batch_size, grad_accum, num_epochs):
        batches = create_ordered_batches(
            dataset=[{"text": f"{i} {i + 10}"} for i in range(n_rows)],
            tokenizer=_TinyTokenizer(),
            batch_size=batch_size,
            max_seq_length=4,
            seed=None,
            dataset_order="torch_randperm",
            num_epochs=num_epochs,
            grad_accum=grad_accum,
        )
        return sum(int(lengths.shape[0]) for _b, lengths, _l in batches)

    # (rows, batch, accum, epochs): rows consumed by transformers.Trainer
    assert rows_for(5, 2, 2, 0.5) == 4
    assert rows_for(5, 2, 2, 1.0) == 5
    assert rows_for(5, 2, 2, 1.5) == 9
    assert rows_for(10, 2, 2, 0.75) == 10
    assert rows_for(10, 2, 2, 1.5) == 18


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
    eager builder on merged main (two smoke modes; full matrix recorded).
    Re-frozen after the staged VLM finalizer stopped emitting the private
    raw-input-ids carrier; both digests still come from the eager builder."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    GOLDENS = {
        "default_epoch_replay": ((('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((0, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((0, -100, 2),))), (('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((1, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((1, -100, 2),))), (('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((2, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((2, -100, 2),))), (('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((3, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((3, -100, 2),))), (('attention_mask', 'mlx.core.int32', (1, 3), ((1, 1, 1),)), ('input_ids', 'mlx.core.int32', (1, 3), ((4, 200, 2),)), ('labels', 'mlx.core.int32', (1, 3), ((4, -100, 2),)))),
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


def _open_fd_count():
    """Count this process's open descriptors, or None where we cannot."""
    import os

    for probe in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(probe):
            try:
                return len(os.listdir(probe))
            except OSError:
                pass
    return None


def test_vlm_plan_does_not_pin_one_file_descriptor_per_row(tmp_path):
    """FILE ownership, the counterpart to the tensor-ownership survey above.

    ``PIL.Image.open`` leaves the file open until the raster is loaded, so a
    plan that stores one row per scheduled slot must not pin one descriptor
    per row: a few hundred rows would blow past the 256 soft limit that macOS,
    the platform MLX runs on, hands every process by default.
    """
    Image = pytest.importorskip("PIL.Image")
    import gc

    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    n_rows = 64
    paths = []
    for i in range(n_rows):
        path = tmp_path / f"row{i}.png"
        Image.fromarray(np.full((8, 8, 3), i % 251, dtype=np.uint8)).save(path)
        paths.append(str(path))

    class _LazyImageRows:
        """The textbook dataset idiom: open in __getitem__, decode later."""

        def __len__(self):
            return n_rows

        def __getitem__(self, index):
            return {
                "text": str(index % 7),
                "images": [Image.open(paths[index])],
            }

    gc.collect()
    before = _open_fd_count()
    if before is None:
        pytest.skip("no portable descriptor count on this platform")

    plan = _create_vlm_batch_plan(
        dataset=_LazyImageRows(),
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
    )
    gc.collect()
    growth = _open_fd_count() - before

    assert len(plan) == n_rows // 2
    assert growth <= 8, (
        f"the plan pinned {growth} descriptors for {n_rows} rows; a stored "
        f"row must not keep its image file open until materialization"
    )

    # Releasing the descriptor has to leave a row that still materializes.
    batch = plan[0]
    assert batch["input_ids"].shape[0] == 2


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


class _TailUnplannableProcessor(_ContentProcessor):
    """Makes exactly one surveyed batch unplannable, chosen by call order."""

    def __init__(self, poison_call):
        self.poison_call = poison_call
        self.calls = 0

    def __call__(self, text, **kwargs):
        batch = dict(super().__call__(text, **kwargs))
        self.calls += 1
        if self.calls == self.poison_call:
            batch[object()] = 1      # an unstable key: never plannable
        return batch


def test_vlm_admission_covers_only_the_batches_the_loop_visits():
    """A schedule whose length is not a multiple of the accumulation factor
    ends in micro-batches the loop never reaches. An unplannable family
    confined to that tail must not degrade the run to eager or abort strict
    mode, since no compiled call can ever see it; the same family inside the
    executed region still must."""
    _skip_if_mlx_core_was_replaced()
    from types import SimpleNamespace

    from unsloth_zoo.mlx.trainer import _plan_single_process_vlm_shapes

    # 10 scheduled batches against 2 steps x 4 accumulation = 8 visits, so
    # batches 8 and 9 are surveyed and never trained.
    args = SimpleNamespace(
        compile_max_variants=None, gradient_accumulation_steps=4,
    )

    def _plan(poison_call):
        return _plan_single_process_vlm_shapes(
            _vlm_planner_fixtures(
                processor=_TailUnplannableProcessor(poison_call), rows=10,
            ),
            None, args=args, total_steps=2, distributed_world_size=1,
            compile_policy=SimpleNamespace(mode="strict"),
            compile_decision=SimpleNamespace(enabled=True, policy_mode="strict"),
        )

    for poison_call in (9, 10):      # calls are 1-based: batches 8 and 9
        _shape_plan, report, allowed, _frontier = _plan(poison_call)
        assert allowed and report.reason != "vlm_unplannable_family"

    # Still enforced for a batch the loop does reach.
    with pytest.raises(RuntimeError, match="cannot plan VLM batch 3"):
        _plan(4)


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


class _RandomAugmentingProcessor(_ContentProcessor):
    """A processor whose augmentation draws from the process-global RNG.

    The drawn value is stamped into the produced ids, so a batch's contents
    reveal which position of the preprocessing stream built it. When ``log``
    is given, every call also records ``(rows, draw)`` so a test can tell which
    stretch of the stream built which split.
    """

    def __init__(self, log=None):
        self.log = log

    def __call__(self, text, **_kwargs):
        import random

        draw = random.random()
        stamp = 300 + int(draw * 1e9) % 997
        rows = [[int(item), 200, stamp, 2] for item in text]
        if self.log is not None:
            self.log.append(([str(item) for item in text], draw))
        return {
            "input_ids": np.array(rows, dtype=np.int32),
            "attention_mask": np.array([[1] * 4 for _ in rows], dtype=np.int32),
        }


def _train_stochastic_vlm(monkeypatch, tmp_path, *, resume_step=0,
                          eval_dataset=None, grad_accum=1, max_steps=4,
                          seed=3407, processor_log=None):
    """Run the real training loop and return the ids each micro-step saw.

    The model maths are stubbed; every data path (plan construction, eval
    batch construction, resume handling, batch fetch) stays real.
    """
    import os
    import random
    import mlx.nn as nn
    from mlx.utils import tree_map
    import unsloth_zoo.mlx.trainer as trainer_mod
    import unsloth_zoo.mlx.utils as utils_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    consumed = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4096, 4)
            self._config = {"image_size": 16, "image_token_id": 200,
                            "model_type": "tinyvlm"}
            self._hf_repo = None

        def __call__(self, input_ids, **_kwargs):
            return self.embed(input_ids)

        def train(self, mode=True):
            return self

        def load_weights(self, *_args, **_kwargs):
            return None

    processor = _RandomAugmentingProcessor(log=processor_log)
    output_dir = str(tmp_path)
    trainer = MLXTrainer(
        model=Model(),
        tokenizer=processor,
        train_dataset=[{"text": str(i)} for i in range(8)],
        eval_dataset=eval_dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=grad_accum,
            max_steps=max_steps,
            max_seq_length=16,
            seed=seed,
            output_dir=output_dir,
            report_to="none",
            logging_steps=10 ** 6,
            save_steps=0,
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            # Large enough that evaluation never fires inside the loop; the
            # eval BATCHES are still built before it.
            eval_steps=(10 ** 6 if eval_dataset is not None else 0),
        ),
        processor=processor,
    )
    trainer._is_vlm = True

    def fake_value_and_grad(model, _loss_fn):
        params = model.trainable_parameters()

        def wrapped(_model, batch_data, *_rest):
            consumed.append(batch_data["input_ids"].tolist())
            return (mx.array(1.0), mx.array(4)), tree_map(mx.zeros_like, params)

        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", fake_value_and_grad)
    monkeypatch.setattr(trainer_mod.nn, "value_and_grad", fake_value_and_grad)

    class _Optimizer:
        def __init__(self):
            self.learning_rate = mx.array(1e-4)
            self.state = {}

        def update(self, *_args, **_kwargs):
            return None

    trainer._build_optimizer = lambda _total_steps: _Optimizer()
    trainer.save_model = lambda *_args, **_kwargs: None
    trainer._install_neftune = lambda *_args, **_kwargs: None

    checkpoint = None
    if resume_step:
        checkpoint = os.path.join(output_dir, f"checkpoint-{resume_step}")
        os.makedirs(checkpoint, exist_ok=True)
        # Write the files a real checkpoint carries. The loaders below are
        # stubbed, but the trainer refuses a checkpoint whose resume state is
        # missing rather than silently restarting from step 0, so a directory
        # that only pretends to exist is no longer a usable fixture.
        mx.save_safetensors(
            os.path.join(checkpoint, "adapters.safetensors"),
            {"placeholder": mx.zeros((1,))},
        )
        mx.save_safetensors(
            os.path.join(checkpoint, "optimizer_state.safetensors"),
            {"placeholder": mx.zeros((1,))},
        )
        with open(os.path.join(checkpoint, "trainer_state.json"), "w") as f:
            json.dump({"global_step": resume_step}, f)
        monkeypatch.setattr(
            trainer_mod, "load_optimizer_state", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            trainer_mod, "load_trainer_state",
            lambda *a, **k: {"global_step": resume_step,
                             "train_loss_history": []},
        )

    # Warm one-time lazy imports (some draw from the global RNG when first
    # loaded) so only the processor's own draws are measured.
    utils_mod.create_vlm_batches(
        dataset=[{"text": "0"}, {"text": "1"}],
        processor=_RandomAugmentingProcessor(),
        config=trainer.model._config,
        batch_size=2,
        max_seq_length=16,
        dataset_order="sequential",
    )

    random.seed(seed)
    trainer.train(resume_from_checkpoint=checkpoint)
    return consumed


def test_vlm_resume_replays_the_skipped_preprocessing_stream(
    monkeypatch, tmp_path,
):
    """Resuming must hand the first resumed batch the augmentation an
    uninterrupted run used there, not the opening draw.

    The eager builder created every scheduled batch up front, so the skipped
    micro-batches had still run the processor; the lazy plan has to replay
    them or a stochastic preprocessing pipeline restarts its stream at the
    resume point.
    """
    _skip_if_mlx_core_was_replaced()

    full = _train_stochastic_vlm(
        monkeypatch, tmp_path / "full", grad_accum=2, max_steps=4,
    )
    resumed = _train_stochastic_vlm(
        monkeypatch, tmp_path / "resumed", grad_accum=2, max_steps=4,
        resume_step=2,
    )

    assert len(full) == 8 and len(resumed) == 4
    assert resumed == full[4:]


def test_vlm_eval_batches_leave_the_training_preprocessing_stream_alone(
    monkeypatch, tmp_path,
):
    """Building eval batches must not consume the draws the first training
    batch is owed. Eager training batches were built before the eval ones, so
    adding an eval split never moved the training augmentation stream."""
    _skip_if_mlx_core_was_replaced()

    without_eval = _train_stochastic_vlm(
        monkeypatch, tmp_path / "plain", max_steps=4,
    )
    with_eval = _train_stochastic_vlm(
        monkeypatch, tmp_path / "with_eval", max_steps=4,
        eval_dataset=[{"text": str(100 + i)} for i in range(4)],
    )

    assert len(without_eval) == 4
    assert with_eval == without_eval


def test_vlm_eval_splits_each_draw_their_own_augmentation_stretch(
    monkeypatch, tmp_path,
):
    """A dict ``eval_dataset`` must keep the preprocessing stream progressing
    ACROSS splits.

    Keeping eval builds out of the training stream is one preservation around
    the whole set of splits, not one per split: restoring between splits hands
    every split the identical snapshot, so an augmenting processor replays the
    same draw sequence for each of them instead of progressing the way
    sequential construction did. The training stream must still be untouched
    however many splits there are.
    """
    _skip_if_mlx_core_was_replaced()

    without_eval = _train_stochastic_vlm(
        monkeypatch, tmp_path / "plain", max_steps=4,
    )
    calls = []
    with_splits = _train_stochastic_vlm(
        monkeypatch, tmp_path / "splits", max_steps=4,
        eval_dataset={
            "alpha": [{"text": str(100 + i)} for i in range(4)],
            "beta": [{"text": str(200 + i)} for i in range(4)],
            "gamma": [{"text": str(300 + i)} for i in range(4)],
        },
        processor_log=calls,
    )

    # Evaluation never moves the training stream, whatever the split count.
    assert len(without_eval) == 4
    assert with_splits == without_eval

    per_split = {}
    for rows, draw in calls:
        head = int(rows[0])
        if head >= 100:
            per_split.setdefault(head // 100, []).append(draw)
    assert sorted(per_split) == [1, 2, 3]
    assert [len(draws) for draws in per_split.values()] == [2, 2, 2]

    # No split may replay a draw another split already consumed.
    eval_draws = [draw for draws in per_split.values() for draw in draws]
    assert len(set(eval_draws)) == len(eval_draws)


def test_vlm_plan_pins_each_visit_against_an_in_place_formatting_func():
    """A formatting_func that mutates and returns its argument must not let a
    later visit rewrite the row an earlier visit already stored, so every batch
    keeps the value the eager builder collated at that point in the schedule."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    dataset = [{"text": str(i)} for i in range(3)]

    def formatting_func(item):
        item["text"] = str(int(item["text"]) + 10)
        return item

    plan = _create_vlm_batch_plan(
        dataset=dataset,
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=6,
        dataset_order="sequential",
        formatting_func=formatting_func,
    )

    heads = [int(plan[i]["input_ids"][0, 0].item()) for i in range(len(plan))]
    assert heads == [10, 11, 12, 20, 21, 22]


def test_vlm_plan_reads_scheduled_rows_only_once_per_occurrence():
    """Unscheduled rows never reach the formatter or the dataset, and every
    scheduled slot gets exactly one read, the way the eager builder indexed
    the dataset while building each batch."""
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

    reads = []

    class _ReadCountingDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            reads.append(int(index))
            return {"text": str(index)}

    plain = _create_vlm_batch_plan(
        dataset=_ReadCountingDataset(),
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=6,
        dataset_order="sequential",
    )
    assert reads == [0, 1, 2, 3, 0, 1]
    assert (len(plain), len(plain.rows)) == (6, 6)
    plain.materialize_all()
    assert reads == [0, 1, 2, 3, 0, 1]


def test_vlm_plan_resamples_a_visit_dependent_dataset_per_occurrence():
    """A map-style dataset with stochastic __getitem__ augmentation must be
    re-read on every scheduled visit, so a revisited index does not replay the
    sample the first visit happened to draw."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    class _AugmentingDataset:
        def __init__(self, size):
            self.size = size
            self.visits = [0] * size

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            visit = self.visits[index]
            self.visits[index] += 1
            return {"text": str(10 * index + visit)}

    dataset = _AugmentingDataset(3)
    plan = _create_vlm_batch_plan(
        dataset=dataset,
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=6,
        dataset_order="sequential",
    )

    assert dataset.visits == [2, 2, 2]
    heads = [int(plan[i]["input_ids"][0, 0].item()) for i in range(len(plan))]
    assert heads == [0, 10, 20, 1, 11, 21]

    plan.materialize_all()
    assert dataset.visits == [2, 2, 2]
    assert [int(plan[i]["input_ids"][0, 0].item()) for i in range(len(plan))] == heads


def test_vlm_plan_pins_each_visit_against_a_nested_in_place_formatting_func():
    """An in-place formatting_func that reaches a nested container must not let
    a later visit rewrite an earlier stored visit, and pinning a visit must
    reference the row's payloads rather than duplicate them."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    payload = object()
    dataset = [
        {"holder": {"text": str(i)}, "media": [payload]} for i in range(3)
    ]

    def formatting_func(item):
        item["holder"]["text"] = str(int(item["holder"]["text"]) + 10)
        return {
            "text": item["holder"]["text"],
            "holder": item["holder"],
            "media": item["media"],
        }

    plan = _create_vlm_batch_plan(
        dataset=dataset,
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=6,
        dataset_order="sequential",
        formatting_func=formatting_func,
    )

    assert [row.item["holder"]["text"] for row in plan.rows] == [
        "10", "11", "12", "20", "21", "22",
    ]
    heads = [int(plan[i]["input_ids"][0, 0].item()) for i in range(len(plan))]
    assert heads == [10, 11, 12, 20, 21, 22]
    assert all(row.item["media"][0] is payload for row in plan.rows)


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


def _vlm_shape_plan(plan, mode, args=None):
    """Run the real VLM planner for one policy mode."""
    from types import SimpleNamespace

    from unsloth_zoo.mlx.trainer import _plan_single_process_vlm_shapes

    args = args or SimpleNamespace(
        compile_max_variants=None, gradient_accumulation_steps=1,
    )
    return _plan_single_process_vlm_shapes(
        plan, None, args=args, total_steps=len(plan),
        distributed_world_size=1,
        compile_policy=SimpleNamespace(mode=mode),
        compile_decision=SimpleNamespace(enabled=True, policy_mode=mode),
    )


class _WideningNoPadProcessor(_UniformNoPadProcessor):
    """One wider batch, so the shared rounded width really does widen the rest."""

    def __init__(self, widths=(40, 32, 32, 32, 32, 32)):
        super().__init__(width=widths[0])
        self.widths = widths
        self.row = 0

    def __call__(self, text, **kwargs):
        self.width = self.widths[self.row % len(self.widths)]
        self.row += 1
        return super().__call__(text, **kwargs)


def test_vlm_exact_plan_compiles_without_a_pad_token():
    """A pad id is required by the BUILT plan, not by the processor: when no
    admitted endpoint widens a batch, nothing is ever padded."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.shape_guard import FULL_STEP_SCOPE, phase_for_microstep

    for mode in ("best_effort", "strict"):
        processor = _UniformNoPadProcessor()
        plan = _plan_for(processor)
        shape_plan, report, allowed, _frontier = _vlm_shape_plan(plan, mode)
        assert allowed, mode
        assert (report.action, report.reason) == ("exact", "schedule_within_cap")
        assert plan.pad_token_id is None
        raw = tuple(plan.batch_width(index) for index in range(len(plan)))
        endpoints = tuple(
            shape_plan.endpoint_for(plan.batch_family(index), width)
            for index, width in enumerate(plan.planned_event_widths())
        )
        assert endpoints == raw == (32,) * len(plan)
        # The planned fetch the training loop makes must not need a pad id.
        batch = plan.materialize(0, phase=phase_for_microstep(
            FULL_STEP_SCOPE, 1, 0,
        ))
        assert batch["input_ids"].shape[1] == 32


def test_vlm_declined_only_plan_compiles_without_a_pad_token():
    """A sidecar sharing the text extent declines every batch, so the width
    finalizer returns them untouched and no pad id is consulted."""
    _skip_if_mlx_core_was_replaced()

    class _DeclinedNoPadProcessor(_UniformNoPadProcessor):
        def __call__(self, text, **kwargs):
            batch = super().__call__(text, **kwargs)
            width = batch["input_ids"].shape[1]
            batch["pixel_values"] = np.zeros(
                (len(batch["input_ids"]), width), dtype=np.float32,
            )
            return batch

    processor = _DeclinedNoPadProcessor()
    plan = _plan_for(processor)
    _shape_plan, report, allowed, _frontier = _vlm_shape_plan(plan, "strict")
    assert allowed
    assert report.action == "exact"
    assert not any(plan._padable)


def test_vlm_widening_plan_still_requires_a_pad_token():
    """Widening still needs a fill value, including the forbidden-width bump
    that widens batches whose raw widths are identical."""
    _skip_if_mlx_core_was_replaced()

    processor = _WideningNoPadProcessor()
    plan = _plan_for(processor)
    _shape_plan, report, allowed, _frontier = _vlm_shape_plan(
        plan, "best_effort",
    )
    assert not allowed
    assert (report.action, report.reason) == ("eager", "vlm_pad_token_unavailable")
    raw = tuple(plan.batch_width(index) for index in range(len(plan)))
    planned = plan.planned_event_widths()
    assert raw == (40, 32, 32, 32, 32, 32)
    # The five identical raw widths are bumped off a forbidden extent.
    assert planned == (40, 33, 33, 33, 33, 33)

    strict_plan = _plan_for(_WideningNoPadProcessor())
    with pytest.raises(RuntimeError, match="without a tokenizer pad id"):
        _vlm_shape_plan(strict_plan, "strict")


def test_vlm_planned_width_steps_above_the_surveyed_maximum_on_collision():
    """The forbidden-extent bump outranks the surveyed-maximum cap.

    Declining the capped width instead would split one shared endpoint back
    into one per raw width, so the overflow is the cheaper of the two; it
    still has to produce endpoints every member batch can actually reach.
    """
    _skip_if_mlx_core_was_replaced()

    class _ForeignExtentProcessor(_UniformNoPadProcessor):
        # The last two batches carry a sidecar whose extent equals the widest
        # surveyed text width, so the union forbids that width for everyone.
        _ROWS = ((20, 6), (32, 6), (20, 6), (32, 6), (20, 32), (20, 32))

        def __init__(self):
            super().__init__(width=20)
            self.tokenizer.pad_token_id = 7
            self.row = 0

        def __call__(self, text, **kwargs):
            self.width, sidecar = self._ROWS[self.row % len(self._ROWS)]
            self.row += 1
            batch = super().__call__(text, **kwargs)
            batch["pixel_values"] = np.zeros(
                (len(batch["input_ids"]), sidecar), dtype=np.float32,
            )
            return batch

    plan = _plan_for(_ForeignExtentProcessor())
    plan.ensure_descriptors()
    raw = tuple(plan.batch_width(index) for index in range(len(plan)))
    surveyed_max = max(
        width for width, padable in zip(raw, plan._padable) if padable
    )
    planned = plan.planned_event_widths()

    assert raw == (20, 32, 20, 32, 20, 20)
    assert surveyed_max == 32
    assert planned == (33,) * len(plan)
    assert all(width > surveyed_max for width in planned)
    # One shared endpoint, and every batch reaches it: above its own prepared
    # width and clear of every extent the pipeline does not pad.
    assert [
        int(
            plan.materialize(index, target_width=planned[index])
            ["input_ids"].shape[1]
        )
        for index in range(len(plan))
    ] == [33] * len(plan)


def test_vlm_plan_rejects_a_rebuild_that_leaves_its_frozen_endpoint():
    """Endpoints freeze at survey time, so a processor that rebuilds at a new
    width hard-fails on the planned fetch whether or not a pad id exists."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.shape_guard import FULL_STEP_SCOPE, phase_for_microstep

    phase = phase_for_microstep(FULL_STEP_SCOPE, 1, 0)
    for pad_token_id, rebuilt, message in (
        (0, 36, "is below this batch's prepared width"),
        (None, 30, "pad id is required to pad"),
    ):
        processor = _UniformNoPadProcessor(width=32, narrowed=rebuilt)
        processor.tokenizer.pad_token_id = pad_token_id
        plan = _plan_for(processor)
        _shape_plan, _report, allowed, _frontier = _vlm_shape_plan(
            plan, "best_effort",
        )
        assert allowed
        processor.training = True
        plan._mru = None
        with pytest.raises(ValueError, match=message):
            plan.materialize(0, phase=phase)

    # A uniform narrowing is not family drift, so nothing earlier catches it.
    processor = _UniformNoPadProcessor(width=32, narrowed=30)
    plan = _plan_for(processor)
    plan.ensure_descriptors()
    processor.training = True
    assert plan.check_family_drift(0, plan._build_batch(0)) is None
def test_vlm_drift_error_names_the_survey_for_privately_owned_draw_state():
    """The survey builds every batch through the processor once before
    training and preserves only the PROCESS RNGs around it, so a processor
    carrying draw state of its own reaches training one survey further along
    and drifts. There is no generic way to restore another object's state, so
    the message has to say where the shift came from: on its own it sends the
    user to a data pipeline that is fine on the eager path."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    class _PrivateCounterProcessor(_ContentProcessor):
        """Carries the reviewer's 'augmentation counter': private state that
        no RNG preservation reaches, advanced once per call."""

        def __init__(self):
            self.calls = 0

        def __call__(self, text, **kwargs):
            batch = dict(super().__call__(text, **kwargs))
            self.calls += 1
            if self.calls % 5 == 0:
                batch["extra_mask"] = np.ones((len(text), 2), dtype=np.int32)
            return batch

    processor = _PrivateCounterProcessor()
    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(4)],
        processor=processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
    )
    plan.ensure_descriptors()
    assert processor.calls == 4      # one private advance per scheduled batch

    rebuilt = plan._build_batch(0)
    # Training resumes the private stream past the survey, so batch 0 rebuilds
    # into a structure its own surveyed family never had.
    assert processor.calls == 5 and "extra_mask" in rebuilt
    with pytest.raises(RuntimeError) as excinfo:
        plan.check_family_drift(0, rebuilt)
    message = str(excinfo.value)
    assert "shape survey" in message and "compile disabled" in message


def test_vlm_plan_releases_the_previous_batch_before_building_the_next(
    monkeypatch,
):
    """A move to the next scheduled batch must not hold two batches at once.

    The training loop clears its own ``batch_data`` reference before every
    fetch, so the plan's most-recent-batch cache is the last thing that can
    keep the previous batch alive while the builder allocates the next one.
    Holding it across the build doubles the batch-side residency at every
    transition, which for large image batches is a peak-memory cost paid for
    nothing: the cached batch can no longer serve this fetch. A repeated
    fetch of the same batch still returns from the cache without rebuilding.
    """
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import FiniteVLMBatchPlan, _create_vlm_batch_plan

    processor = _CountingProcessor()
    plan = _create_vlm_batch_plan(
        dataset=[{"text": str(i)} for i in range(4)],
        processor=processor,
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=4,
        dataset_order="sequential",
    )

    alive = {"now": 0, "peak": 0}

    class _Marker:
        def __init__(self):
            alive["now"] += 1
            alive["peak"] = max(alive["peak"], alive["now"])

        def __del__(self):
            alive["now"] -= 1

    class _TrackedBatch(dict):
        """A batch whose lifetime is observable through refcounting."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.marker = _Marker()

    live_at_build_start = []
    build_batch = FiniteVLMBatchPlan._build_batch

    def _tracked_build(self, index, target_width=None):
        live_at_build_start.append(alive["now"])
        return _TrackedBatch(
            build_batch(self, index, target_width=target_width)
        )

    monkeypatch.setattr(FiniteVLMBatchPlan, "_build_batch", _tracked_build)

    batch_data = None
    for index in range(len(plan)):
        batch_data = None      # what the training loop does before a fetch
        batch_data = plan.materialize(index)
    del batch_data

    # No build ever started while a previously built batch was still held.
    assert live_at_build_start == [0, 0, 0, 0]
    assert alive["peak"] == 1

    # Releasing the cache early must not cost a rebuild for a repeated fetch.
    calls_before = processor.calls
    again = plan.materialize(len(plan) - 1)
    assert processor.calls == calls_before
    del again



def _reused_buffer_vlm_datasets():
    """Datasets whose __getitem__ hands back memory it later overwrites."""
    from PIL import Image

    class _SameArray:
        """Refills and returns one preallocated ndarray."""
        def __init__(self): self.buffer = np.zeros((4, 4, 3), np.uint8)
        def __len__(self): return 4
        def __getitem__(self, index):
            self.buffer[:] = index + 1
            return {"text": str(index), "images": [self.buffer]}

    class _SharedMemoryView:
        """Fresh ndarray per read, but a view over one reused byte buffer."""
        def __init__(self): self.raw = bytearray(4 * 4 * 3)
        def __len__(self): return 4
        def __getitem__(self, index):
            self.raw[:] = bytes([index + 1]) * len(self.raw)
            return {
                "text": str(index),
                "images": [np.frombuffer(memoryview(self.raw), np.uint8).reshape(4, 4, 3)],
            }

    class _MutatedImage:
        """Repaints and returns one PIL image."""
        def __init__(self): self.image = Image.new("RGB", (4, 4))
        def __len__(self): return 4
        def __getitem__(self, index):
            self.image.paste((index + 1,) * 3, (0, 0, 4, 4))
            return {"text": str(index), "images": [self.image]}

    class _TupleWrapped:
        """Reused ndarray delivered inside a tuple rather than a list."""
        def __init__(self): self.buffer = np.zeros((4, 4, 3), np.uint8)
        def __len__(self): return 4
        def __getitem__(self, index):
            self.buffer[:] = index + 1
            return {"text": str(index), "images": (self.buffer,)}

    return [_SameArray(), _SharedMemoryView(), _MutatedImage(), _TupleWrapped()]


def test_vlm_plan_reports_a_dataset_that_overwrites_a_media_buffer():
    """Every scheduled row is collected before any is processed, so a dataset
    that reuses one decode buffer leaves earlier rows holding the last image.
    Copying each payload would cost one image per scheduled slot rather than
    per row, so the plan probes content and reports it instead."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    for dataset in _reused_buffer_vlm_datasets():
        with pytest.raises(ValueError, match="changed after the row"):
            _create_vlm_batch_plan(
                dataset=dataset,
                processor=_ContentProcessor(),
                config={"image_size": 16, "image_token_id": 200},
                batch_size=2,
                max_seq_length=8,
                num_batches=2,
                dataset_order="sequential",
            )


def test_vlm_plan_rechecks_media_between_setup_and_the_batch_that_uses_it():
    """Batches are built on demand, so a payload mutated after setup is still
    live when the processor finally reads it. The eager builder had already
    converted every row to tensors, so the same mutation could not reach it."""
    _skip_if_mlx_core_was_replaced()
    import numpy as np
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    class _OwnArrayPerRow:
        """Distinct array per row, handed out by reference on every read."""
        def __init__(self):
            self.arrays = [np.full((4, 4, 3), i + 1, np.uint8) for i in range(4)]
        def __len__(self): return 4
        def __getitem__(self, index):
            return {"text": str(index), "images": [self.arrays[index]]}

    dataset = _OwnArrayPerRow()
    plan = _create_vlm_batch_plan(
        dataset=dataset,
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
        num_batches=2,
        dataset_order="sequential",
    )
    # Clean at setup: nothing aliases, so every batch builds.
    assert plan.materialize(0) is not None

    dataset.arrays[2][:] = 99
    with pytest.raises(ValueError, match="between trainer setup and the batch"):
        plan.materialize(1)
    # The untouched batch still builds, so the check is per row, not per plan.
    assert plan.materialize(0) is not None


def test_vlm_plan_keeps_referencing_media_payloads_it_does_not_own():
    """The probe must stay a probe: a dataset that returns its images by
    reference on every revisit keeps ONE object per row across all scheduled
    slots, because copying them would scale with steps rather than rows."""
    _skip_if_mlx_core_was_replaced()
    from PIL import Image

    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    dataset = [
        {"text": str(i), "images": [Image.new("RGB", (4, 4), (i + 1,) * 3)]}
        for i in range(3)
    ]
    plan = _create_vlm_batch_plan(
        dataset=dataset,
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=1,
        max_seq_length=8,
        num_batches=9,          # three full revisits of every row
        dataset_order="sequential",
    )

    assert len(plan.rows) == 9
    stored = [row.item["images"][0] for row in plan.rows]
    # Nine slots, still only the three images the dataset owns.
    assert len({id(image) for image in stored}) == 3
    assert all(image is dataset[i % 3]["images"][0] for i, image in enumerate(stored))


def test_vlm_plan_accepts_a_dataset_that_returns_a_fresh_image_per_read():
    """The probe must not fire on a dataset that decodes fresh, which is what
    HuggingFace `datasets` and a plain list of rows both do."""
    _skip_if_mlx_core_was_replaced()
    from PIL import Image

    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    class _FreshImages:
        def __len__(self): return 4
        def __getitem__(self, index):
            return {
                "text": str(index),
                "images": [Image.new("RGB", (4, 4), (index + 1,) * 3)],
            }

    plan = _create_vlm_batch_plan(
        dataset=_FreshImages(),
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
        num_batches=2,
        dataset_order="sequential",
    )
    plan.materialize_all()
    assert [int(np.asarray(row.item["images"][0]).reshape(-1)[0]) for row in plan.rows] \
        == [1, 2, 3, 4]


def test_vlm_media_probe_sees_a_change_that_a_sparse_sample_would_miss():
    # A sampled probe reads a fixed stride derived from the shape, so a
    # mutation that lands between samples is invisible to it. The digest reads
    # every byte, so a change that cancels under a sum still registers.
    Image = pytest.importorskip("PIL.Image")
    from unsloth_zoo.mlx.utils import _vlm_media_fingerprint

    original = np.random.RandomState(0).randint(
        0, 255, (64, 64, 3), dtype=np.uint8,
    )
    mutated = original.copy()
    flat = mutated.reshape(-1)
    stride = max(1, flat.size // 16)
    off_sample = next(i for i in range(flat.size) if i % stride)
    flat[off_sample] = (int(flat[off_sample]) + 7) % 256

    sampled = original.reshape(-1)[::stride][:16].tobytes()
    assert sampled == mutated.reshape(-1)[::stride][:16].tobytes(), (
        "the mutation must be invisible to a 16-sample probe, or this test "
        "is not exercising the gap it exists for"
    )
    assert _vlm_media_fingerprint(original) != _vlm_media_fingerprint(mutated)
    assert _vlm_media_fingerprint(original) == _vlm_media_fingerprint(
        original.copy()
    )
    # Same gap through PIL, whose probe used spread getpixel reads.
    assert _vlm_media_fingerprint(Image.fromarray(original)) \
        != _vlm_media_fingerprint(Image.fromarray(mutated))
    # A sum alone is blind to a change that cancels; the xor half is not.
    cancelling = original.copy().reshape(-1)
    cancelling[1] = (int(cancelling[1]) + 3) % 256
    cancelling[2] = (int(cancelling[2]) - 3) % 256
    assert _vlm_media_fingerprint(original) != _vlm_media_fingerprint(
        cancelling.reshape(original.shape)
    )


def test_vlm_media_probe_sees_an_in_place_byte_permutation():
    """Geometric augmentation rewrites a buffer with a permutation of its own
    bytes, which both a byte sum and a byte xor are blind to. The digest must
    be order sensitive or the plan accepts exactly the corruption it exists to
    report."""
    Image = pytest.importorskip("PIL.Image")
    torch = pytest.importorskip("torch")
    from unsloth_zoo.mlx.utils import _vlm_media_fingerprint

    rows, cols = np.mgrid[0:48, 0:48]
    base = np.stack(
        [(cols * 3 + rows).astype(np.uint8), (rows * 5).astype(np.uint8),
         ((cols ^ rows) * 2).astype(np.uint8)], axis=-1,
    )
    base = (base + np.random.RandomState(0).randint(0, 12, base.shape)).astype(
        np.uint8,
    )
    permutations = {
        "hflip": base[:, ::-1],
        "vflip": base[::-1],
        "rot90": np.rot90(base),
        "transpose": base.transpose(1, 0, 2),
        "roll_rows": np.roll(base, 7, axis=0),
        "channel_swap": base[..., ::-1],
    }
    for name, view in permutations.items():
        mutated = np.ascontiguousarray(view)
        assert np.array_equal(
            np.sort(base.reshape(-1)), np.sort(mutated.reshape(-1)),
        ), f"{name} must be a byte permutation, or it is not the gap under test"
        assert _vlm_media_fingerprint(base) != _vlm_media_fingerprint(mutated), name

    # Byte permutation for float payloads too: reversing an axis moves whole
    # 4-byte groups, so the byte multiset survives.
    tensor = torch.from_numpy(base.astype(np.float32) / 255.0)
    assert _vlm_media_fingerprint(tensor) != _vlm_media_fingerprint(
        torch.flip(tensor, dims=[1]).contiguous()
    )
    assert _vlm_media_fingerprint(Image.fromarray(base)) != _vlm_media_fingerprint(
        Image.fromarray(np.ascontiguousarray(base[:, ::-1]))
    )


def test_vlm_plan_reports_an_in_place_augmentation_over_a_reused_buffer():
    """The reachable form of the same gap: one decode buffer plus an in-place
    flip per read leaves every stored row aliasing the final image, and each
    read is a byte permutation of the last, so a commutative digest saw no
    change at all."""
    _skip_if_mlx_core_was_replaced()
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    class _FlipInPlace:
        """Augments one shared buffer, the way a single-sample or repeated-row
        dataset with in-place geometric augmentation does."""

        def __init__(self):
            self.buffer = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
            self.returned = []

        def __len__(self):
            return 6

        def __getitem__(self, index):
            self.buffer[:] = (
                self.buffer[:, ::-1] if index % 2 else self.buffer[::-1]
            ).copy()
            self.returned.append(self.buffer.copy())
            return {"text": str(index), "images": [self.buffer]}

    dataset = _FlipInPlace()
    with pytest.raises(ValueError, match="changed after the row"):
        _create_vlm_batch_plan(
            dataset=dataset,
            processor=_ContentProcessor(),
            config={"image_size": 16, "image_token_id": 200},
            batch_size=2,
            max_seq_length=8,
            num_batches=3,
            dataset_order="sequential",
        )
    # Every read really did hand back a permutation of the previous one, so
    # the sum and the xor were identical at every pin.
    sums = {int(np.asarray(image).sum()) for image in dataset.returned}
    assert len(dataset.returned) == 6 and len(sums) == 1


def test_vlm_plan_releases_image_handles_nested_in_a_tuple(tmp_path):
    # The snapshot walker traverses tuples, so the release and restore walkers
    # must too: a row shaped {"images": (Image.open(path),)} otherwise keeps
    # every descriptor open for the life of the plan.
    Image = pytest.importorskip("PIL.Image")
    from unsloth_zoo.mlx.utils import (
        _release_vlm_row_image_handles,
        _restore_vlm_row_image_handles,
    )

    paths = []
    for index in range(24):
        path = tmp_path / f"tuple_{index}.png"
        Image.fromarray(
            np.full((8, 8, 3), index + 1, dtype=np.uint8)
        ).save(path)
        paths.append(str(path))

    before = _open_fd_count()
    released = [
        _release_vlm_row_image_handles({"images": (Image.open(path),)})
        for path in paths
    ]
    held = _open_fd_count() - before
    assert held <= 2, (
        f"the plan pinned {held} descriptors for {len(paths)} tuple-nested "
        f"rows; a stored row must not keep its image file open"
    )

    restored = _restore_vlm_row_image_handles(released[5])
    assert isinstance(restored["images"], tuple)
    assert int(np.asarray(restored["images"][0]).reshape(-1)[0]) == 6

    # A namedtuple keeps its type rather than collapsing to a plain tuple.
    holder = collections.namedtuple("holder", "left right")
    row = {"media": holder(left=Image.open(paths[0]), right="caption")}
    out = _release_vlm_row_image_handles(row)
    assert type(out["media"]) is holder
    assert out["media"].right == "caption"
    back = _restore_vlm_row_image_handles(out)
    assert int(np.asarray(back["media"].left).reshape(-1)[0]) == 1

    # A tuple holding nothing releasable keeps its exact identity.
    plain = {"images": ("a", "b")}
    assert _release_vlm_row_image_handles(plain)["images"] is plain["images"]


def test_vlm_plan_reports_a_dataset_that_reuses_one_temporary_image_path(tmp_path):
    """Releasing a handle makes the FILE the payload, so a reused path aliases
    exactly like a reused decode buffer: every scheduled row re-opens the last
    sample. The eager builder decoded each handle before the next read, so this
    is only reachable once the plan collects every row first."""
    Image = pytest.importorskip("PIL.Image")
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    shared = tmp_path / "row.png"

    class _OneSharedPath:
        """Writes every sample to one scratch file, the way a dataset that
        renders or downloads to a fixed temporary path does."""

        def __init__(self):
            self.written = []

        def __len__(self):
            return 6

        def __getitem__(self, index):
            Image.fromarray(
                np.full((8, 8, 3), index + 1, dtype=np.uint8)
            ).save(shared)
            self.written.append(index + 1)
            return {"text": str(index), "images": [Image.open(str(shared))]}

    dataset = _OneSharedPath()
    with pytest.raises(ValueError, match="file backing a dataset image changed"):
        _create_vlm_batch_plan(
            dataset=dataset,
            processor=_ContentProcessor(),
            config={"image_size": 16, "image_token_id": 200},
            batch_size=2,
            max_seq_length=8,
            num_batches=3,
            dataset_order="sequential",
        )

    # Six distinct samples really did go through the one file, and the file now
    # holds only the last of them -- which is what every row would have read.
    assert dataset.written == [1, 2, 3, 4, 5, 6]
    assert int(np.asarray(Image.open(str(shared))).reshape(-1)[0]) == 6


def test_vlm_plan_materializes_per_row_image_files_unchanged(tmp_path):
    """The control for the check above: one file per sample is the normal
    shape, so it must plan, materialize and read back its own pixels."""
    Image = pytest.importorskip("PIL.Image")
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    paths = []
    for index in range(6):
        path = tmp_path / f"row_{index}.png"
        Image.fromarray(np.full((8, 8, 3), index + 1, dtype=np.uint8)).save(path)
        paths.append(str(path))

    class _PerRowPath:
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            return {"text": str(index), "images": [Image.open(paths[index])]}

    seen = []

    class _RecordingProcessor(_ContentProcessor):
        def __call__(self, text, images=None, **kwargs):
            for image in images or []:
                for one in (image if isinstance(image, (list, tuple)) else [image]):
                    seen.append(int(np.asarray(one.convert("RGB")).reshape(-1)[0]))
            return _ContentProcessor.__call__(self, text, **kwargs)

    plan = _create_vlm_batch_plan(
        dataset=_PerRowPath(),
        processor=_RecordingProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
        num_batches=3,
        dataset_order="sequential",
    )
    plan.materialize_all()

    assert seen == [1, 2, 3, 4, 5, 6]


def test_vlm_plan_reports_an_image_file_rewritten_after_the_plan_was_built(tmp_path):
    """The second half of the same window: a released row is only re-opened at
    materialize, so the file can also be rewritten after the plan is built."""
    Image = pytest.importorskip("PIL.Image")
    from unsloth_zoo.mlx.utils import _create_vlm_batch_plan

    paths = []
    for index in range(4):
        path = tmp_path / f"late_{index}.png"
        Image.fromarray(np.full((8, 8, 3), index + 1, dtype=np.uint8)).save(path)
        paths.append(str(path))

    class _PerRowPath:
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            return {"text": str(index), "images": [Image.open(paths[index])]}

    plan = _create_vlm_batch_plan(
        dataset=_PerRowPath(),
        processor=_ContentProcessor(),
        config={"image_size": 16, "image_token_id": 200},
        batch_size=2,
        max_seq_length=8,
        num_batches=2,
        dataset_order="sequential",
    )

    Image.fromarray(np.full((8, 8, 3), 99, dtype=np.uint8)).save(paths[2])
    with pytest.raises(ValueError, match="file backing a dataset image changed"):
        plan.materialize_all()
