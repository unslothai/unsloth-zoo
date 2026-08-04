# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""group_by_length on the MLX trainer (HF TrainingArguments parity).

Before this, MLX forced a choice between two bad options: the legacy default
length-sorts once (padding-efficient but the *same* order every epoch), while
dataset_order="torch_randperm" shuffles properly but pads to whatever lengths
happen to land together. HF's group_by_length gives both -- a fresh shuffle
each epoch, with similar lengths grouped inside it.

The ordering helper is a step-for-step port of transformers'
``get_length_grouped_indices``; ``test_matches_hf_reference_bit_for_bit``
transcribes that function and asserts equality over randomized inputs, so the
port is pinned to the real algorithm rather than to my reading of it.

CPU-pure: pure index math plus torch's RNG. No MLX, no weights, no Metal.
"""

from __future__ import annotations

import random

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    # Install-only (see tests/test_mlx_double_quant_reject.py): tearing the
    # shim down here would break later files that hold module references.
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _helper():
    from unsloth_zoo.mlx.utils import _length_grouped_order

    return _length_grouped_order


def _hf_reference(lengths, batch_size, generator, mega_batch_mult=None):
    """transformers.trainer_pt_utils.get_length_grouped_indices, transcribed."""
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches
    ]
    megabatch_maximums = [lengths[mb[0]] for mb in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = (
        megabatches[max_idx][0], megabatches[0][0],
    )
    return [i for mb in megabatches for i in mb]


# --- the ordering algorithm ---


def test_matches_hf_reference_bit_for_bit():
    """Randomized equality against HF's own algorithm across dataset sizes,
    batch sizes and seeds -- including the tiny-dataset mega_batch_mult clamp
    and the swap-largest-first step."""
    order = _helper()
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 400)
        batch_size = rng.randint(1, 16)
        seed = rng.randint(0, 10**6)
        lengths = [rng.randint(1, 2048) for _ in range(n)]

        generator = torch.Generator()
        generator.manual_seed(seed)
        expected = _hf_reference(lengths, batch_size, generator)

        assert order(lengths, batch_size, seed) == expected


def test_is_a_permutation_and_deterministic_per_seed():
    order = _helper()
    lengths = [random.Random(1).randint(1, 512) for _ in range(101)]
    a = order(lengths, 4, 1234)
    assert sorted(a) == list(range(len(lengths)))
    assert a == order(lengths, 4, 1234)
    assert a != order(lengths, 4, 1235)  # a different seed reshuffles


def test_longest_row_runs_first():
    """HF puts the largest batch first so an OOM surfaces immediately."""
    order = _helper()
    lengths = [10] * 60 + [9999]
    assert order(lengths, 4, 7)[0] == 60


def test_groups_similar_lengths_far_better_than_a_shuffle():
    """The point of the feature: padding waste should drop sharply versus a
    plain permutation of the same rows."""
    order = _helper()
    rng = random.Random(3)
    lengths = [rng.randint(1, 1024) for _ in range(512)]
    batch_size = 8

    def padding_cost(indices):
        total = 0
        for i in range(0, len(indices), batch_size):
            batch = indices[i : i + batch_size]
            widest = max(lengths[j] for j in batch)
            total += sum(widest - lengths[j] for j in batch)
        return total

    from unsloth_zoo.mlx.utils import _torch_randperm_order

    grouped = padding_cost(order(lengths, batch_size, 11))
    shuffled = padding_cost(_torch_randperm_order(len(lengths), 11))
    assert grouped < shuffled / 4


def test_empty_and_single_row_datasets():
    order = _helper()
    assert order([], 4, 1) == []
    assert order([5], 4, 1) == [0]


# --- config resolution ---


def _args(**kw):
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    return MLXTrainingConfig(**kw)


def test_resolver_returns_length_grouped_only_when_requested():
    from unsloth_zoo.mlx.trainer import _resolve_text_dataset_order

    assert _resolve_text_dataset_order(_args(group_by_length=True)) == "length_grouped"
    assert _resolve_text_dataset_order(_args()) == "default"
    assert _resolve_text_dataset_order(
        _args(preserve_dataset_order=True)
    ) == "sequential"
    assert _resolve_text_dataset_order(
        _args(dataset_order="torch_randperm")
    ) == "torch_randperm"


def test_resolver_preserves_explicit_none_dataset_order():
    """dataset_order=None must stay None so it still routes to the ordered
    plan (where it means sequential) rather than the default builder."""
    from unsloth_zoo.mlx.trainer import _resolve_text_dataset_order

    assert _resolve_text_dataset_order(_args(dataset_order=None)) is None


@pytest.mark.parametrize(
    "conflict", [
        dict(preserve_dataset_order=True),
        dict(dataset_order="sequential"),
        dict(dataset_order="torch_randperm"),
    ],
)
def test_conflicting_order_knobs_raise(conflict):
    """Each of these picks a concrete, different order. Letting one silently
    win would train on an order the caller never asked for."""
    from unsloth_zoo.mlx.trainer import _resolve_text_dataset_order

    with pytest.raises(ValueError, match="group_by_length"):
        _resolve_text_dataset_order(_args(group_by_length=True, **conflict))


def test_config_accepts_group_by_length_and_defaults_off():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    assert MLXTrainingConfig().group_by_length is False
    assert MLXTrainingConfig(group_by_length=True).group_by_length is True


def test_appended_field_keeps_the_optional_copy_suffix():
    from dataclasses import fields as dataclass_fields
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _MLX_CONFIG_OPTIONAL_COPY_FIELDS,
    )

    names = [f.name for f in dataclass_fields(MLXTrainingConfig) if f.init]
    assert "group_by_length" in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    tail = tuple(names[-len(_MLX_CONFIG_OPTIONAL_COPY_FIELDS):])
    assert tail == _MLX_CONFIG_OPTIONAL_COPY_FIELDS


# --- the two text paths must agree ---


def test_labeled_and_unlabeled_paths_produce_the_same_order():
    """The load-bearing invariant. `_create_labeled_batches` (the
    train_on_responses_only path) and `_create_ordered_text_plan` (the
    unlabeled path) order rows independently; if they disagree,
    train_on_responses_only masks a different stream than it trains on.

    Nothing pinned this before, which is how their accepted `dataset_order`
    vocabularies were able to drift apart.
    """
    from unsloth_zoo.mlx.utils import _length_grouped_order, _normalize_seed

    rng = random.Random(5)
    rows = [[0] * rng.randint(2, 300) for _ in range(233)]
    labeled_items = [(ids, [1] * len(ids)) for ids in rows]
    batch_size, seed = 4, 4242

    for epoch in range(3):
        # exactly what _create_labeled_batches._order_indices_for_epoch does
        labeled = _length_grouped_order(
            [len(item[0]) for item in labeled_items],
            batch_size,
            _normalize_seed(seed) + epoch,
        )
        # exactly what _create_ordered_text_plan.make_order does
        from unsloth_zoo.mlx.utils import _text_row_length

        unlabeled = _length_grouped_order(
            [_text_row_length(row) for row in rows],
            batch_size,
            _normalize_seed(seed) + epoch,
        )
        assert labeled == unlabeled


def test_row_length_handles_labeled_and_unlabeled_shapes():
    from unsloth_zoo.mlx.utils import _text_row_length

    assert _text_row_length([1, 2, 3]) == 3                 # unlabeled ids
    assert _text_row_length(([1, 2, 3], [-100, 2, 3])) == 3  # (ids, labels)


def test_order_reshuffles_across_epochs():
    """A fresh permutation per epoch is the whole difference from the legacy
    length-sorted default, which repeats one order forever."""
    from unsloth_zoo.mlx.utils import _length_grouped_order, _normalize_seed

    lengths = [random.Random(9).randint(1, 400) for _ in range(300)]
    base = _normalize_seed(1234)
    orders = [_length_grouped_order(lengths, 8, base + e) for e in range(3)]
    assert orders[0] != orders[1] != orders[2]


def test_eval_batches_are_never_length_grouped():
    """HF installs the length-grouped sampler on the train dataloader only.
    Regrouping eval as well (and reshuffling it per epoch) would move metrics
    around underneath the user for a knob that is about training throughput.
    """
    from unsloth_zoo.mlx.trainer import _resolve_text_dataset_order

    args = _args(group_by_length=True)
    assert _resolve_text_dataset_order(args, for_training=True) == "length_grouped"
    assert _resolve_text_dataset_order(args, for_training=False) == "default"


def test_eval_resolution_still_honors_the_other_order_knobs():
    from unsloth_zoo.mlx.trainer import _resolve_text_dataset_order

    assert _resolve_text_dataset_order(
        _args(preserve_dataset_order=True), for_training=False,
    ) == "sequential"
    assert _resolve_text_dataset_order(
        _args(dataset_order="torch_randperm"), for_training=False,
    ) == "torch_randperm"


def test_conflicts_raise_on_the_eval_path_too():
    """A contradictory config must fail the same way whichever path resolves
    it first, rather than depending on whether eval happens to run."""
    from unsloth_zoo.mlx.trainer import _resolve_text_dataset_order

    with pytest.raises(ValueError, match="group_by_length"):
        _resolve_text_dataset_order(
            _args(group_by_length=True, preserve_dataset_order=True),
            for_training=False,
        )

# --- unsupported paths must fail loudly, not silently mis-group ---


def test_vlm_rejects_length_grouping():
    """A VLM row's cost is dominated by image/audio tokens the text length does
    not see, so grouping on that length would not bound padding."""
    from unsloth_zoo.mlx.trainer import _reject_group_by_length

    with pytest.raises(ValueError, match="vision-language"):
        _reject_group_by_length("length_grouped", "vlm")


def test_streaming_rejects_length_grouping():
    """The mega-batch permutation needs the whole dataset up front."""
    from unsloth_zoo.mlx.trainer import _reject_group_by_length

    with pytest.raises(ValueError, match="requires a sized dataset"):
        _reject_group_by_length("length_grouped", "streaming")


@pytest.mark.parametrize("context", ["vlm", "streaming"])
@pytest.mark.parametrize(
    "order", ["default", "sequential", "torch_randperm", None],
)
def test_existing_orders_are_untouched_by_the_guard(order, context):
    """The guard must be a no-op for every pre-existing order, so VLM and
    streaming runs that never set group_by_length behave exactly as before."""
    from unsloth_zoo.mlx.trainer import _reject_group_by_length

    assert _reject_group_by_length(order, context) is None

# --- distributed (DDP) interaction ---


class _FakeWorld:
    """Minimal comm_group: _distributed_rank_size only calls rank()/size()."""

    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_every_rank_derives_the_identical_global_order():
    """DDP correctness hinges on this. Ranks do not exchange the sample order;
    each derives it locally and slices its share. If two ranks disagreed they
    would train on mismatched rows and the collectives would pair up the wrong
    gradients. The order depends only on (lengths, batch_size, seed), none of
    which is rank-dependent, and this pins that.
    """
    from unsloth_zoo.mlx.utils import _length_grouped_order

    lengths = [random.Random(2).randint(1, 900) for _ in range(257)]
    orders = [_length_grouped_order(lengths, 8, 4242) for _ in range(4)]
    assert all(o == orders[0] for o in orders)


def test_rank_slices_are_disjoint_and_cover_the_global_batch():
    from unsloth_zoo.mlx.utils import (
        _length_grouped_order,
        _rank_slice_distributed_batch,
        _distributed_global_batch_size,
    )

    world_size, local_batch = 4, 2
    lengths = [random.Random(4).randint(1, 500) for _ in range(128)]
    order = _length_grouped_order(lengths, local_batch * world_size, 7)
    global_batch = _distributed_global_batch_size(local_batch, _FakeWorld(0, world_size))
    assert global_batch == local_batch * world_size

    for start in range(0, len(order), global_batch):
        chunk = order[start : start + global_batch]
        slices = [
            _rank_slice_distributed_batch(
                chunk, local_batch, comm_group=_FakeWorld(r, world_size),
                pad_source=order,
            )
            for r in range(world_size)
        ]
        flat = [i for s in slices for i in s]
        # Every rank gets its own local_batch rows, and together they reconstruct
        # the global batch (a short tail is cycle-padded, so compare as a set).
        assert all(len(s) == local_batch for s in slices)
        assert set(chunk) <= set(flat)
        if len(chunk) == global_batch:
            assert sorted(flat) == sorted(chunk)


def test_length_grouping_survives_rank_sharding():
    """Grouping is only useful if it holds after DDP slices a global batch:
    each rank should still see a length-homogeneous local batch."""
    from unsloth_zoo.mlx.utils import (
        _length_grouped_order,
        _rank_slice_distributed_batch,
    )

    world_size, local_batch = 2, 4
    rng = random.Random(6)
    lengths = [rng.randint(1, 1024) for _ in range(512)]
    global_batch = local_batch * world_size
    order = _length_grouped_order(lengths, global_batch, 21)

    def local_padding(order_source):
        waste = 0
        for start in range(0, len(order_source), global_batch):
            chunk = order_source[start : start + global_batch]
            for r in range(world_size):
                s = _rank_slice_distributed_batch(
                    chunk, local_batch, comm_group=_FakeWorld(r, world_size),
                    pad_source=order_source,
                )
                if not s:
                    continue
                widest = max(lengths[i] for i in s)
                waste += sum(widest - lengths[i] for i in s)
        return waste

    from unsloth_zoo.mlx.utils import _torch_randperm_order

    assert local_padding(order) < local_padding(
        _torch_randperm_order(len(lengths), 21)
    ) / 4
