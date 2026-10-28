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

"""`train_on_responses_only` on a `with_format("arrow")` split.

Under the arrow format rows and batches are pyarrow Tables, so the pretokenized
check, the masking map and the fully-masked label scan all misread them: the run
either crashed or kept rows with no response, while every other format drops
those rows. The arrow split must keep and drop exactly what the default format
does, print the same messages, and still come back as arrow. CPU-pure, offline.
"""

import contextlib
import io

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import train_on_responses_only

from test_pretokenized_bypass_collator import (
    INSTRUCTION_PART,
    RESPONSE_PART,
    ROW,
    USER_ID,
    StubTokenizer,
    StubTrainer,
)

NO_RESPONSE = [USER_ID, 10, 11, 12]
MIXED = [list(ROW), NO_RESPONSE, list(ROW), NO_RESPONSE, list(ROW)]


def _rows(rows):
    return Dataset.from_dict({
        "input_ids": rows,
        "attention_mask": [[1] * len(r) for r in rows],
    })


def _run(train, eval_dataset = None):
    trainer = StubTrainer(None, train)
    trainer.eval_dataset = eval_dataset
    trainer.processing_class = StubTokenizer()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    except Exception as error:
        return ("error", type(error).__name__, str(error)), out.getvalue()
    return trainer, out.getvalue()


def _unsloth_lines(text):
    return [line for line in text.splitlines() if line.startswith("Unsloth")]


def _plain(dataset):
    return dataset.with_format(None).to_dict()


def test_arrow_drops_the_same_rows_as_the_default_format():
    base, base_log = _run(_rows(MIXED))
    arrow, arrow_log = _run(_rows(MIXED).with_format("arrow"))
    assert not isinstance(arrow, tuple), arrow
    assert len(base.train_dataset) == 3
    assert _plain(arrow.train_dataset) == _plain(base.train_dataset)
    assert arrow.train_dataset.format["type"] == "arrow"
    assert _unsloth_lines(arrow_log) == _unsloth_lines(base_log)
    assert any("Removed 2 out of 5" in line for line in _unsloth_lines(arrow_log))


def test_arrow_eval_splits_match_the_default_format():
    base, base_log = _run(_rows(MIXED), {"a": _rows(MIXED), "b": _rows([list(ROW)])})
    arrow, arrow_log = _run(
        _rows(MIXED).with_format("arrow"),
        {"a": _rows(MIXED).with_format("arrow"), "b": _rows([list(ROW)]).with_format("arrow")},
    )
    assert not isinstance(arrow, tuple), arrow
    for key in ("a", "b"):
        assert _plain(arrow.eval_dataset[key]) == _plain(base.eval_dataset[key]), key
        assert arrow.eval_dataset[key].format["type"] == "arrow"
    assert _unsloth_lines(arrow_log) == _unsloth_lines(base_log)

    base, _ = _run(_rows(MIXED), _rows(MIXED))
    arrow, _ = _run(_rows(MIXED).with_format("arrow"), _rows(MIXED).with_format("arrow"))
    assert _plain(arrow.eval_dataset) == _plain(base.eval_dataset)
    assert arrow.eval_dataset.format["type"] == "arrow"


@pytest.mark.parametrize("columns", [None, ["input_ids", "attention_mask"]])
def test_arrow_keeps_its_column_selection(columns):
    train = _rows(MIXED)
    base, _ = _run(train.with_format(None, columns = columns))
    arrow, _ = _run(train.with_format("arrow", columns = columns))
    assert arrow.train_dataset.format["columns"] == base.train_dataset.format["columns"]
    assert arrow.train_dataset.format["output_all_columns"] == base.train_dataset.format["output_all_columns"]
    assert _plain(arrow.train_dataset) == _plain(base.train_dataset)


def test_an_all_masked_arrow_split_fails_like_the_default_format():
    base, _ = _run(_rows([NO_RESPONSE, NO_RESPONSE]))
    arrow, _ = _run(_rows([NO_RESPONSE, NO_RESPONSE]).with_format("arrow"))
    assert base[0] == "error" and "nothing to train on" in base[2]
    assert arrow == base
