"""Tests patch_datasets_map_worker_death_retry in temporary_patches/misc.py.

A long-text corpus can have the kernel OOM-kill a `dataset_num_proc` worker,
which datasets turns into "One of the subprocesses has abruptly died during
map operation", killing the run inside SFTTrainer.__init__.

The retry must be narrow: a real exception raised by the map function has to
keep propagating, or this would hide genuine bugs behind a slow rerun.

A fake `datasets` module stands in for the real one, so no dataset is built
and no worker is forked.
"""

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"
_SRC = MISC.read_text(encoding = "utf-8")

WORKER_DIED = ("One of the subprocesses has abruptly died during map operation."
               "To debug the error, disable multiprocessing.")


def _load():
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_datasets_map_worker_death_retry":
            ns = {"inspect": inspect}
            exec(ast.get_source_segment(_SRC, node), ns)
            return ns[node.name]
    raise AssertionError("patch_datasets_map_worker_death_retry not found")


patch = _load()


def _make_fake_datasets(map_impl):
    """Install a fake `datasets` module whose Dataset.map is `map_impl`."""
    class Dataset:
        pass
    Dataset.map = map_impl
    mod = types.ModuleType("datasets")
    mod.Dataset = Dataset
    sys.modules["datasets"] = mod
    return Dataset


@pytest.fixture(autouse = True)
def _restore_datasets():
    saved = sys.modules.get("datasets")
    yield
    if saved is None: sys.modules.pop("datasets", None)
    else: sys.modules["datasets"] = saved


def test_retries_single_process_after_worker_death():
    calls = []

    def original(self, fn, num_proc = None, **kw):
        calls.append(num_proc)
        if num_proc and num_proc > 1:
            raise RuntimeError(WORKER_DIED)
        return "mapped"

    Dataset = _make_fake_datasets(original)
    assert patch() is True
    assert Dataset().map(lambda x: x, num_proc = 8) == "mapped"
    assert calls == [8, 1], "should retry exactly once, with num_proc=1"


def test_single_process_death_is_reraised():
    def original(self, fn, num_proc = None, **kw):
        raise RuntimeError(WORKER_DIED)

    Dataset = _make_fake_datasets(original)
    patch()
    with pytest.raises(RuntimeError, match = "abruptly died"):
        Dataset().map(lambda x: x, num_proc = 1)


def test_missing_num_proc_is_reraised():
    def original(self, fn, num_proc = None, **kw):
        raise RuntimeError(WORKER_DIED)

    Dataset = _make_fake_datasets(original)
    patch()
    with pytest.raises(RuntimeError, match = "abruptly died"):
        Dataset().map(lambda x: x)


def test_unrelated_runtime_error_is_untouched():
    def original(self, fn, num_proc = None, **kw):
        raise RuntimeError("something else entirely")

    Dataset = _make_fake_datasets(original)
    patch()
    with pytest.raises(RuntimeError, match = "something else entirely"):
        Dataset().map(lambda x: x, num_proc = 8)


def test_real_exception_from_map_fn_still_propagates():
    # The whole point: a genuine bug must not be retried into silence.
    def original(self, fn, num_proc = None, **kw):
        raise ValueError("tokenizer exploded")

    Dataset = _make_fake_datasets(original)
    patch()
    with pytest.raises(ValueError, match = "tokenizer exploded"):
        Dataset().map(lambda x: x, num_proc = 8)


def test_successful_map_is_passed_through_once():
    calls = []

    def original(self, fn, num_proc = None, **kw):
        calls.append(num_proc)
        return "fine"

    Dataset = _make_fake_datasets(original)
    patch()
    assert Dataset().map(lambda x: x, num_proc = 4) == "fine"
    assert calls == [4]


def test_positional_num_proc_is_retried():
    calls = []

    def original(self, fn, with_indices = False, num_proc = None, **kw):
        calls.append(num_proc)
        if num_proc and num_proc > 1:
            raise RuntimeError(WORKER_DIED)
        return "mapped"

    Dataset = _make_fake_datasets(original)
    patch()
    assert Dataset().map(lambda x: x, False, 6) == "mapped"
    assert calls == [6, 1]


def test_idempotent():
    def original(self, fn, num_proc = None, **kw):
        return "fine"

    Dataset = _make_fake_datasets(original)
    assert patch() is True
    first = Dataset.map
    assert patch() is None, "second apply must be a no-op"
    assert Dataset.map is first


def test_registered_as_a_temporary_patch():
    assert "TEMPORARY_PATCHES.append(patch_datasets_map_worker_death_retry)" in _SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
