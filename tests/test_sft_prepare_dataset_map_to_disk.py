# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""In-memory Dataset tokenizing writes a temp Arrow file: memory-mapped, same rows; CPU-only."""

import os
from types import SimpleNamespace

import pytest
from datasets import Dataset

import unsloth_zoo.dataset_utils as dataset_utils
from unsloth_zoo.dataset_utils import sft_prepare_dataset


class _Tokenizer:
    """Whitespace tokenizer: one id per word, enough for the text branch."""

    pad_token_id = 0
    bos_token = None
    eos_token = None
    chat_template = None

    def __call__(self, texts, truncation = False, max_length = None, add_special_tokens = True, **kwargs):
        single = isinstance(texts, str)
        rows = [[len(word) for word in text.split()] for text in ([texts] if single else texts)]
        if truncation and max_length:
            rows = [row[:max_length] for row in rows]
        out = {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}
        return {k: v[0] for k, v in out.items()} if single else out


def _prepare(dataset):
    trainer = SimpleNamespace(model = None, completion_only_loss = None, data_collator = None)
    args = SimpleNamespace(
        max_length = 64, dataset_text_field = "text", remove_unused_columns = True,
        packing_strategy = "bfd", dataset_num_proc = 1,
    )
    return sft_prepare_dataset(trainer, dataset, _Tokenizer(), args, False, None, "train")


def _rows():
    return Dataset.from_dict({"text": [f"row {i} " + "word " * (i % 7) for i in range(200)]})


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_utils, "_MAP_TO_DISK_DIR", None, raising = False)
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK_DIR", str(tmp_path))
    return tmp_path


def test_in_memory_dataset_is_tokenized_to_a_temp_file(fresh, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK", "1")
    out = _prepare(_rows())
    assert out.cache_files, "tokenized rows stayed in anonymous RAM"
    assert all(f["filename"].startswith(str(fresh)) for f in out.cache_files)
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK", "0")
    ref = _prepare(_rows())
    assert not ref.cache_files
    assert out.column_names == ref.column_names
    assert out.to_dict() == ref.to_dict()


def test_each_call_gets_its_own_file(fresh, monkeypatch):
    """A unique name per map: a second dataset is never answered from the first one's file."""
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK", "1")
    first = _prepare(_rows())
    other = Dataset.from_dict({"text": ["completely different text here"] * 50})
    second = _prepare(other)
    assert first.cache_files[0]["filename"] != second.cache_files[0]["filename"]
    assert second["input_ids"][0] == [10, 9, 4, 4]


def test_auto_leaves_small_datasets_in_memory(fresh, monkeypatch):
    monkeypatch.delenv("UNSLOTH_DATASET_MAP_TO_DISK", raising = False)
    assert not _prepare(_rows()).cache_files
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK_MIN_MB", "0")
    assert _prepare(_rows()).cache_files


def test_file_backed_and_iterable_datasets_are_left_alone(fresh, tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK", "1")
    path = tmp_path / "saved"
    _rows().save_to_disk(str(path))
    from datasets import load_from_disk

    on_disk = load_from_disk(str(path))
    assert dataset_utils._map_to_disk_kwargs(on_disk) == {}
    assert dataset_utils._map_to_disk_kwargs(_rows().to_iterable_dataset()) == {}


def test_not_enough_disk_keeps_ram(fresh, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK", "1")
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(total = 1, used = 1, free = 0))
    assert dataset_utils._map_to_disk_kwargs(_rows()) == {}
    assert not os.listdir(fresh)


def test_stale_dirs_of_exited_processes_are_swept(fresh, monkeypatch):
    """Windows keeps a still-mapped file at exit; the next run removes dirs whose process is gone."""
    import subprocess
    import sys

    monkeypatch.setenv("UNSLOTH_DATASET_MAP_TO_DISK", "1")
    monkeypatch.setattr(dataset_utils, "_SWEEP_STALE_MAP_DIRS", True, raising = False)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    stale = fresh / f"unsloth_dataset_{child.pid}_abc"
    stale.mkdir()
    (stale / "tokenized_old.arrow").write_bytes(b"x")
    live = fresh / f"unsloth_dataset_{os.getppid()}_abc"
    live.mkdir()
    unrelated = fresh / "unsloth_dataset_notapid"
    unrelated.mkdir()
    out = _prepare(_rows())
    assert out.cache_files
    assert not stale.exists(), "a dead process's dir was left behind"
    assert live.exists() and unrelated.exists()
    assert os.path.basename(os.path.dirname(out.cache_files[0]["filename"])).startswith(f"unsloth_dataset_{os.getpid()}_")
