"""A fresh streaming dataset has no `batch_size` to copy.

`train_on_responses_only` and the SFT tokenizer path both re-use the batch size
of the dataset they are about to map:

    trainer.train_dataset._ex_iterable.batch_size

That attribute only exists once a dataset has ALREADY been mapped, which is
when `_ex_iterable` is a `MappedExamplesIterable`. A dataset straight from
`load_dataset(..., streaming = True)` or `.to_iterable_dataset()` holds an
`ArrowExamplesIterable`, which has no `batch_size` at all, so reading it raised

    AttributeError: 'ArrowExamplesIterable' object has no attribute 'batch_size'

before the first map could run. Every streaming user hit this on the first
call, which is why it is a plain getattr with datasets' own default now.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth_zoo.dataset_utils import _iterable_batch_size  # noqa: E402


# ---- the shapes datasets actually produces --------------------------------

def test_a_fresh_streaming_dataset_has_no_batch_size():
    """The premise. If datasets starts shipping one, this fix is moot and the
    reasoning above needs revisiting rather than the test being deleted."""
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"a": [1, 2, 3]}).to_iterable_dataset()
    assert not hasattr(ds._ex_iterable, "batch_size"), (
        f"{type(ds._ex_iterable).__name__} now carries batch_size"
    )


def test_the_helper_answers_for_one_anyway():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"a": [1, 2, 3]}).to_iterable_dataset()
    assert _iterable_batch_size(ds) == 1000


def test_an_already_mapped_dataset_keeps_its_own():
    """The behaviour the original code was reaching for, preserved."""
    datasets = pytest.importorskip("datasets")
    ds = (datasets.Dataset.from_dict({"a": [1, 2, 3]})
          .to_iterable_dataset()
          .map(lambda b: b, batched = True, batch_size = 7))
    assert _iterable_batch_size(ds) == 7


def test_mapping_a_fresh_streaming_dataset_no_longer_raises():
    """End to end: the call that used to raise."""
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"a": [1, 2, 3]}).to_iterable_dataset()
    mapped = ds.map(lambda b: b, batched = True,
                    batch_size = _iterable_batch_size(ds))
    assert [row["a"] for row in mapped] == [1, 2, 3]


# ---- the helper on its own -------------------------------------------------

def test_the_default_matches_datasets_own():
    """A different default would silently change batching for every streaming
    user, which is a behaviour change wearing a bug fix's clothes."""
    datasets = pytest.importorskip("datasets")
    import inspect
    assert (inspect.signature(datasets.IterableDataset.map)
            .parameters["batch_size"].default) == 1000


@pytest.mark.parametrize("dataset", [None, object()])
def test_anything_without_the_attribute_gets_the_default(dataset):
    assert _iterable_batch_size(dataset) == 1000


def test_a_zero_or_none_batch_size_falls_back():
    """`batch_size = None` means "one batch" to datasets, but it reaches here
    only from a dataset that never set one; treating it as unset is what the
    old code effectively did by crashing instead."""
    class _Ex:
        batch_size = None

    class _DS:
        _ex_iterable = _Ex()

    assert _iterable_batch_size(_DS()) == 1000


def test_the_default_is_overridable():
    class _DS:
        pass

    assert _iterable_batch_size(_DS(), default = 32) == 32


# ---- no raw reads left -----------------------------------------------------

def test_no_call_site_reads_the_attribute_directly():
    """Four sites shared this bug. A fifth added later would reintroduce it."""
    src = (ROOT / "unsloth_zoo" / "dataset_utils.py").read_text(encoding="utf-8")
    assert "._ex_iterable.batch_size" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
