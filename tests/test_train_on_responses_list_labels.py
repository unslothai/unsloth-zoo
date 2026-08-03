# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""``train_on_responses_only`` over a dataset that already has a ``labels`` column.

The masking function type-checks ``input_ids`` for tensor-vs-list:

    if type(input_ids_) is torch_Tensor:
        use_tensors = True
        input_ids_ = input_ids_.tolist()

but called ``.tolist()`` on ``labels`` unconditionally. Under
``datasets.map(batched = True)`` a ``labels`` column arrives as a plain list of
lists, which has no ``.tolist()``, so any dataset already carrying labels raised
``AttributeError: 'list' object has no attribute 'tolist'``.

CPU-pure and offline: the tokenizer is a local stub, no weights are loaded.
"""

import pytest
import torch
from datasets import Dataset

from unsloth_zoo.dataset_utils import train_on_responses_only


INSTRUCTION_PART = "<|user|>"
RESPONSE_PART = "<|assistant|>"

# Token ids: 1 = <|user|>, 2 = <|assistant|>, everything else is content.
USER_ID, ASSISTANT_ID = 1, 2
ROW = [USER_ID, 10, 11, ASSISTANT_ID, 20, 21]


class StubTokenizer:
    """Maps the two markers to single ids; any other text to per-character ordinals."""

    def __call__(self, text, add_special_tokens=False):
        class _Result:
            pass

        result = _Result()
        if text == INSTRUCTION_PART:
            result.input_ids = [USER_ID]
        elif text == RESPONSE_PART:
            result.input_ids = [ASSISTANT_ID]
        else:
            result.input_ids = [ord(c) for c in text]
        return result


def _masker():
    return train_on_responses_only(
        None,
        INSTRUCTION_PART,
        RESPONSE_PART,
        tokenizer=StubTokenizer(),
        return_function=True,
    )


def test_plain_list_labels_do_not_raise():
    """The regression: a labels column of plain lists must be accepted."""
    masker = _masker()

    try:
        out = masker({"input_ids": [list(ROW), list(ROW)], "labels": [list(ROW), list(ROW)]})
    except AttributeError as exception:  # pragma: no cover - the bug being fixed
        pytest.fail(f"plain-list labels raised AttributeError: {exception}")

    assert len(out["labels"]) == 2


def test_datasets_map_with_existing_labels_column():
    """End to end through datasets.map(batched=True), which is how the column
    actually reaches the masking function."""
    dataset = Dataset.from_dict(
        {"input_ids": [list(ROW), list(ROW)], "labels": [list(ROW), list(ROW)]}
    )
    masker = _masker()

    mapped = dataset.map(masker, batched=True)

    assert len(mapped) == 2
    for labels in mapped["labels"]:
        assert any(label != -100 for label in labels), "every label was masked away"


def test_tensor_labels_still_supported():
    """Guard the pre-existing path: tensor labels must keep working."""
    masker = _masker()

    out = masker(
        {"input_ids": torch.tensor([ROW, ROW]), "labels": torch.tensor([ROW, ROW])}
    )

    assert type(out["labels"]) is torch.Tensor
    assert out["labels"].shape == (2, len(ROW))


def test_existing_labels_are_the_source_of_truth():
    """When labels are supplied, unmasked positions must be copied from them, not
    re-read from input_ids."""
    sentinel_row = [900, 901, 902, 903, 904, 905]
    masker = _masker()

    out = masker({"input_ids": [list(ROW)], "labels": [list(sentinel_row)]})

    kept = [label for label in out["labels"][0] if label != -100]
    assert kept, "expected at least one unmasked position"
    assert all(label in sentinel_row for label in kept), (
        f"unmasked labels {kept} came from input_ids instead of the labels column"
    )


def test_no_labels_column_still_masks_from_input_ids():
    """Guard the other branch: with no labels column the function builds labels from
    input_ids as before."""
    masker = _masker()

    out = masker({"input_ids": [list(ROW)]})

    labels = out["labels"][0]
    assert len(labels) == len(ROW)
    assert any(label != -100 for label in labels)
