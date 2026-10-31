"""Finite MLX text plans store rows as int32 arrays; batches and consumers must match the tuple rows exactly."""

import pickle
import types
import tracemalloc

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type = ImportError)

from unsloth_zoo.mlx.utils import (  # noqa: E402
    FiniteTextBatchPlan,
    _FiniteTextRow,
    _compact_token_row,
    _create_ordered_text_plan,
    _create_tokenized_text_plan,
    _finite_text_rows,
)


def _legacy_row(row):
    return _FiniteTextRow(
        tuple(int(x) for x in row.input_ids),
        row.offset,
        None if row.labels is None else tuple(int(x) for x in row.labels),
    )


def _random_items(seed, count, labeled, max_len=90):
    rng = np.random.RandomState(seed)
    items = []
    for _ in range(count):
        length = int(rng.randint(2, max_len))
        ids = rng.randint(0, 151_000, size=length).tolist()
        if labeled:
            start = int(rng.randint(0, length))
            labels = [-100] * start + ids[start:]
            if rng.rand() < 0.2:
                labels[int(rng.randint(0, length))] = -100
            items.append((ids, labels))
        else:
            items.append((ids, int(rng.randint(0, 2))))
    return items


def _batch_values(batch):
    return [
        None if value is None else (str(np.array(value).dtype), np.array(value).tolist())
        for value in batch
    ]


def _plan_trace(plan, epochs=3):
    plan.configure_cce_compaction(True)
    trace = [sorted(map(repr, plan._cce_capacities.items()))]
    for visit in range(epochs * len(plan)):
        index = plan.batch_index_for_visit(visit)
        batch = plan.materialize(index)
        trace.append((index, plan.batch_width(index), _batch_values(plan.prepare_cce_batch(index, batch))))
    return trace


def _twin_plans(items, labeled, **kwargs):
    compact = _finite_text_rows(items, with_offsets=not labeled)
    legacy = tuple(_legacy_row(row) for row in compact)
    schedule = [tuple(range(start, start + 4)) for start in range(0, len(items) - 3, 4)]
    schedule.append((0, None, 1, None))
    def make(rows):
        return FiniteTextBatchPlan(
            rows, schedule, visit_policy="epoch_permute", visit_seed=3407, **kwargs,
        )

    return make(compact), make(legacy)


def test_compact_token_row_matches_int_coercion():
    row = _compact_token_row([1, 151_000, -100, 2 ** 31 - 1, -(2 ** 31)])
    assert isinstance(row, np.ndarray) and row.dtype == np.int32
    assert not row.flags.writeable
    assert row.tolist() == [1, 151_000, -100, 2 ** 31 - 1, -(2 ** 31)]
    assert _compact_token_row([1.9, True, np.int64(7)]).tolist() == [1, 1, 7]
    assert _compact_token_row(np.arange(4, dtype=np.int64)).dtype == np.int32
    assert _compact_token_row([]).shape == (0,)
    # Outside int32, non integer arrays and one-shot iterables keep the historical tuple form.
    assert _compact_token_row([2 ** 31]) == (2 ** 31,)
    assert _compact_token_row([2 ** 70]) == (2 ** 70,)
    assert _compact_token_row(np.array([1.5, 2.0])) == (1, 2)
    assert _compact_token_row(iter([3, 4])) == (3, 4)
    source = np.arange(3, dtype=np.int32)
    copied = _compact_token_row(source)
    source[0] = 99
    assert copied.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        _compact_token_row([float("nan")])


@pytest.mark.parametrize("labeled", [False, True])
@pytest.mark.parametrize("label_dtype", ["int64", "int32"])
@pytest.mark.parametrize("pad_to_multiple", [0, 32])
def test_compact_plan_batches_match_tuple_rows(labeled, label_dtype, pad_to_multiple):
    items = _random_items(7, 41, labeled)
    compact, legacy = _twin_plans(
        items, labeled, max_seq_length=64, pad_id=5, minimum_width=2,
        pad_to_multiple=pad_to_multiple, label_dtype=label_dtype,
    )
    assert isinstance(compact.rows[0].input_ids, np.ndarray)
    assert isinstance(legacy.rows[0].input_ids, tuple)
    assert _plan_trace(compact) == _plan_trace(legacy)


@pytest.mark.parametrize("labeled", [False, True])
def test_compact_cce_selection_matches_tuple_rows(labeled):
    items = _random_items(11, 24, labeled, max_len=200)
    compact, legacy = _twin_plans(items, labeled, max_seq_length=160, pad_id=0, minimum_width=2)
    for plan in (compact, legacy):
        # Force a capacity for every batch so the selection runs on unlabeled and fully supervised rows too.
        plan._cce_capacities = {
            (plan.batch_family(i), plan.batch_width(i)): 4096 for i in range(len(plan))
        }
    for index in range(len(compact)):
        left = compact.prepare_cce_batch(index, compact.materialize(index))
        right = legacy.prepare_cce_batch(index, legacy.materialize(index))
        assert len(left) == 4
        assert _batch_values(left) == _batch_values(right)


def test_supervision_counts_and_lengths_match_tuple_rows():
    items = _random_items(3, 30, True)
    compact, legacy = _twin_plans(items, True, max_seq_length=50, pad_id=0)
    for plan in (compact, legacy):
        plan.configure_cce_compaction(True)
    assert compact._cce_capacities == legacy._cce_capacities
    for a, b in zip(compact.rows, legacy.rows):
        assert len(a.input_ids) == len(b.input_ids)
        assert list(a.labels[1:50]).count(-100) == list(b.labels[1:50]).count(-100)


def test_compact_rows_compare_hash_and_pickle_like_tuple_rows():
    compact = _finite_text_rows([([1, 2, 3], [-100, 2, 3])])[0]
    legacy = _legacy_row(compact)
    assert compact == legacy and hash(compact) == hash(legacy)
    assert len({compact, legacy}) == 1
    assert compact != _FiniteTextRow((1, 2, 4), 0, (-100, 2, 3))
    items = _random_items(5, 20, True)
    plan, _ = _twin_plans(items, True, max_seq_length=64, pad_id=0)
    clone = pickle.loads(pickle.dumps(plan))
    assert clone.rows == plan.rows
    assert clone.rows[0].input_ids.dtype == np.int32
    assert _plan_trace(clone) == _plan_trace(plan)


def test_check_all_masked_reads_compact_labels():
    from unsloth_zoo.mlx.trainer import _check_all_masked

    masked = FiniteTextBatchPlan(
        _finite_text_rows([([1, 2, 3], [-100, -100, -100]), ([4, 5], [-100, -100])]),
        [(0, 1)], max_seq_length=8, pad_id=0,
    )
    with pytest.raises(ZeroDivisionError):
        _check_all_masked(masked)
    mixed = FiniteTextBatchPlan(
        _finite_text_rows([([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, -100])]),
        [(0, 1)], max_seq_length=8, pad_id=0,
    )
    _check_all_masked(mixed)


def test_plan_builders_store_compact_rows():
    from unsloth_zoo.mlx.trainer import _create_labeled_batches

    labeled = _create_tokenized_text_plan(_random_items(1, 12, True), 2, 64, seed=3407)
    tokenizer = types.SimpleNamespace(
        chat_template=None, eos_token_id=None, pad_token_id=0,
        encode=lambda text, add_special_tokens=True: [int(part) for part in str(text).split()],
    )
    texts = [{"text": " ".join(str(500 + i + j) for j in range(3 + i))} for i in range(6)]
    ordered = _create_ordered_text_plan(texts, tokenizer, 2, 64, seed=3407, dataset_order="sequential")

    def mask_fn(batch):
        return {"labels": [[-100] + list(ids[1:]) for ids in batch["input_ids"]]}

    response = _create_labeled_batches(
        dataset=texts, tokenizer=tokenizer, mask_fn=mask_fn, batch_size=2,
        max_seq_length=64, dataset_order="sequential", return_plan=True,
    )
    for plan in (labeled, ordered, response):
        row = plan.rows[0]
        assert isinstance(row.input_ids, np.ndarray) and row.input_ids.dtype == np.int32
        if row.labels is not None:
            assert isinstance(row.labels, np.ndarray) and row.labels.dtype == np.int32
    assert np.array(response[0][2]).dtype == np.int32


def test_compact_plan_memory_is_four_bytes_per_token():
    rng = np.random.RandomState(0)
    lengths = rng.randint(256, 2048, size=64)
    tokens = int(lengths.sum())
    items = [(rng.randint(300, 151_000, size=int(n)).tolist(), 0) for n in lengths]
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    rows = _finite_text_rows(items, with_offsets=True)
    retained = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert sum(row.input_ids.nbytes for row in rows) == 4 * tokens
    # Row objects and array headers only add a small per-row constant; tuples cost 8 bytes per token in pointers.
    assert retained < 4 * tokens + 400 * len(rows)
