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

"""Packed rows (TRL seq_lengths) mask each document as if unpacked, incl. last_response_only."""

from datasets import Dataset

from unsloth_zoo.dataset_utils import train_on_responses_only

USER_ID, ASSISTANT_ID = 1, 2
DOC_A = [USER_ID, 10, ASSISTANT_ID, 20, USER_ID, 11, ASSISTANT_ID, 21, 22]
DOC_B = [USER_ID, 12, ASSISTANT_ID, 23, 24]


class StubTokenizer:
    def __call__(self, text, add_special_tokens = False):
        class _Result:
            pass

        result = _Result()
        result.input_ids = {"<|user|>": [USER_ID], "<|assistant|>": [ASSISTANT_ID]}.get(
            text, [ord(c) for c in text]
        )
        return result


def _masker(**kwargs):
    return train_on_responses_only(
        None, "<|user|>", "<|assistant|>", tokenizer = StubTokenizer(), return_function = True, **kwargs
    )


def _unpacked(masker, docs):
    return [t for row in masker({"input_ids": [list(d) for d in docs]})["labels"] for t in row]


def test_packed_rows_mask_like_their_documents():
    for kwargs in ({}, {"last_response_only": True}):
        masker = _masker(**kwargs)
        packed = Dataset.from_dict(
            {"input_ids": [DOC_A + DOC_B], "seq_lengths": [[len(DOC_A), len(DOC_B)]]}
        ).map(masker, batched = True)
        assert packed["labels"][0] == _unpacked(masker, [DOC_A, DOC_B]), kwargs


def test_last_response_only_keeps_every_documents_last_turn():
    masker = _masker(last_response_only = True)
    out = masker({"input_ids": [DOC_A + DOC_B], "seq_lengths": [[len(DOC_A), len(DOC_B)]]})
    kept = [t for t in out["labels"][0] if t != -100]
    assert kept == [21, 22, 23, 24]


def test_a_packed_row_with_none_labels_masks_from_input_ids():
    masker = _masker()
    out = masker(
        {"input_ids": [DOC_A + DOC_B], "labels": [None], "seq_lengths": [[len(DOC_A), len(DOC_B)]]}
    )
    assert out["labels"][0] == _unpacked(masker, [DOC_A, DOC_B])
