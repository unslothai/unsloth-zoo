"""Fresh streaming datasets (load_dataset(streaming = True), to_iterable_dataset()) have no batch_size until mapped, so
reads use getattr with datasets' default (1000) instead of raising AttributeError."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth_zoo.dataset_utils import _iterable_batch_size  # noqa: E402


# ---- the shapes datasets actually produces --------------------------------

def test_a_fresh_streaming_dataset_has_no_batch_size():
    """If datasets starts shipping a batch_size on fresh streaming datasets, the getattr fix is moot and the reasoning must be revisited, not the test deleted."""
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
    """A different default would silently change batching for every streaming user, so the fallback must stay datasets' own default (1000)."""
    datasets = pytest.importorskip("datasets")
    import inspect
    assert (inspect.signature(datasets.IterableDataset.map)
            .parameters["batch_size"].default) == 1000


@pytest.mark.parametrize("dataset", [None, object()])
def test_anything_without_the_attribute_gets_the_default(dataset):
    assert _iterable_batch_size(dataset) == 1000


def test_a_zero_or_none_batch_size_falls_back():
    """batch_size = None means "one batch" to datasets but only reaches here from a dataset that never set one, so it falls back to the default."""
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
