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

"""``mask_out_tokens`` on ``train_on_responses_only`` (unslothai/unsloth#6695).

Every route to the masking closure threads the argument through a different call,
and a route that drops it fails silently, so each is pinned with a control that
leaves the argument out: the list path (``datasets.map``), the tensor path (what
``UnslothVisionDataCollator.__call__`` hands over), the vision-collator path
(``train_on_responses_only(trainer, ...)``) and the MLX delegation. CPU-pure and
offline; one test builds a real ``PreTrainedTokenizerFast`` in memory.
"""

import sys
import types

import pytest
import torch
from datasets import Dataset

from unsloth_zoo.dataset_utils import train_on_responses_only


INSTRUCTION_PART = "<|user|>"
RESPONSE_PART = "<|assistant|>"
THINK_CLOSE = "</think>"

PAD_ID, USER_ID, ASSISTANT_ID, THINK_ID, IMAGE_ID = 0, 1, 2, 3, 4
MARKERS = {INSTRUCTION_PART: USER_ID, RESPONSE_PART: ASSISTANT_ID, THINK_CLOSE: THINK_ID}

# user turn | assistant turn: thinking, the closer, the answer.
ROW = [USER_ID, 10, 11, ASSISTANT_ID, 20, 21, THINK_ID, 22, 23]
ROW_THINK_POS = 6
ROW_RESPONSE = slice(4, 9)


class StubEncoding(dict):
    """Tokenizer output: a mapping with attribute access."""
    __getattr__ = dict.__getitem__


class StubTokenizer:
    """Markers are single ids wherever they appear; other text is per-character ordinals."""

    def __call__(self, text, add_special_tokens = False, **kwargs):
        if isinstance(text, (list, tuple)):
            ids = [self(t).input_ids for t in text]
            return StubEncoding(input_ids = ids, attention_mask = [[1] * len(r) for r in ids])
        ids = []
        rest = text
        while rest:
            for marker, tid in MARKERS.items():
                if rest.startswith(marker):
                    ids.append(tid)
                    rest = rest[len(marker):]
                    break
            else:
                ids.append(ord(rest[0]))
                rest = rest[1:]
        return StubEncoding(input_ids = ids, attention_mask = [1] * len(ids))


def _masker(**kwargs):
    return train_on_responses_only(
        None,
        INSTRUCTION_PART,
        RESPONSE_PART,
        tokenizer = StubTokenizer(),
        return_function = True,
        **kwargs,
    )


def _expected(row, masked_positions = ()):
    """Response-only labels for ``row`` with ``masked_positions`` at -100."""
    labels = [-100] * len(row)
    labels[ROW_RESPONSE] = row[ROW_RESPONSE]
    for position in masked_positions:
        labels[position] = -100
    return labels


# --------------------------------------------------------------------------- list path


def test_the_closer_inside_a_kept_response_is_masked():
    labels = _masker(mask_out_tokens = [THINK_CLOSE])({"input_ids": [list(ROW)]})["labels"][0]

    assert labels == _expected(ROW, masked_positions = [ROW_THINK_POS])


def test_without_the_argument_the_closer_is_trained():
    """Control: the same row with the argument left out trains the closer."""
    labels = _masker()({"input_ids": [list(ROW)]})["labels"][0]

    assert labels[ROW_THINK_POS] == THINK_ID
    assert labels == _expected(ROW)


def test_a_bare_string_means_a_list_of_one():
    as_string = _masker(mask_out_tokens = THINK_CLOSE)({"input_ids": [list(ROW)]})["labels"]
    as_list = _masker(mask_out_tokens = [THINK_CLOSE])({"input_ids": [list(ROW)]})["labels"]

    assert as_string == as_list


def test_every_occurrence_is_masked_across_turns():
    """A closer in each assistant turn, and one in the user turn (already -100)."""
    row = [USER_ID, 10, THINK_ID, ASSISTANT_ID, THINK_ID, 20, THINK_ID,
           USER_ID, 11, ASSISTANT_ID, 21, THINK_ID]

    labels = _masker(mask_out_tokens = [THINK_CLOSE])({"input_ids": [row]})["labels"][0]

    assert labels == [-100, -100, -100, -100, -100, 20, -100, -100, -100, -100, 21, -100]
    assert all(label == -100 for label, token in zip(labels, row) if token == THINK_ID)


def test_a_multi_token_sequence_matches_whole_and_does_not_overlap():
    """``"ab"`` tokenizes to ``[97, 98]``; a partial match is left alone."""
    a, b, c = ord("a"), ord("b"), ord("c")
    row = [USER_ID, 10, ASSISTANT_ID, a, b, a, b, a, c, 20]

    labels = _masker(mask_out_tokens = ["ab"])({"input_ids": [row]})["labels"][0]

    assert labels == [-100, -100, -100, -100, -100, -100, -100, a, c, 20]


def test_datasets_map_batched_is_the_text_sft_route():
    """Plain lists in, plain lists out."""
    dataset = Dataset.from_dict({"input_ids": [list(ROW), list(ROW)]})

    mapped = dataset.map(_masker(mask_out_tokens = [THINK_CLOSE]), batched = True)

    assert len(mapped) == 2
    for labels in mapped["labels"]:
        assert labels == _expected(ROW, masked_positions = [ROW_THINK_POS])


# ------------------------------------------------------------------------- tensor path


def test_tensor_batches_keep_their_type_and_their_existing_masks():
    """A collator's batch: padded tensors, pads already -100. The closer is matched on
    ``input_ids`` and masked in ``labels``; kept positions come from ``labels``."""
    row_a = ROW + [PAD_ID, PAD_ID]
    row_b = [USER_ID, 12, ASSISTANT_ID, 30, THINK_ID, 31, 32, PAD_ID, PAD_ID, PAD_ID, PAD_ID]
    sentinel_a = [900 + i if t != PAD_ID else -100 for i, t in enumerate(row_a)]
    sentinel_b = [800 + i if t != PAD_ID else -100 for i, t in enumerate(row_b)]
    batch = {
        "input_ids": torch.tensor([row_a, row_b]),
        "labels": torch.tensor([sentinel_a, sentinel_b]),
    }

    out = _masker(mask_out_tokens = [THINK_CLOSE])(batch)["labels"]

    assert type(out) is torch.Tensor
    assert out.dtype == torch.int64
    assert out.shape == (2, len(row_a))
    assert out[0].tolist() == [-100, -100, -100, -100, 904, 905, -100, 907, 908, -100, -100]
    assert out[1].tolist() == [-100, -100, -100, 803, -100, 805, 806, -100, -100, -100, -100]


def test_tensor_batches_without_the_argument_are_unchanged():
    """Control for the tensor path."""
    batch = {"input_ids": torch.tensor([ROW]), "labels": torch.tensor([ROW])}

    out = _masker()(batch)["labels"]

    assert type(out) is torch.Tensor
    assert out[0].tolist() == _expected(ROW)


# ----------------------------------------------------------------- vision collator path


class StubProcessor:
    """An image half plus the text tokenizer under ``.tokenizer``."""
    def __init__(self):
        self.image_processor = object()
        self.tokenizer = StubTokenizer()


class UnslothVisionDataCollator:
    """Name-matched; ``train_on_responses_only(trainer)`` installs the masker on it."""
    def __init__(self, processor):
        self.processor = processor
        self.image_processor = processor.image_processor
        self.train_on_responses_only = None


class StubTrainer:
    def __init__(self, collator):
        self.data_collator = collator
        self.train_dataset = None
        self.eval_dataset = None
        self.processing_class = collator.processor
        self.args = types.SimpleNamespace(packing = False, max_length = 2048)


def _collated_batch():
    """Right-padded batch as the vision collator builds it before masking."""
    row_a = [USER_ID, IMAGE_ID, IMAGE_ID, 10, ASSISTANT_ID, 20, THINK_ID, 21, PAD_ID]
    row_b = [USER_ID, IMAGE_ID, 11, ASSISTANT_ID, THINK_ID, 30, 31, 32, 33]
    labels_a = [t if t not in (PAD_ID, IMAGE_ID) else -100 for t in row_a]
    labels_b = [t if t not in (PAD_ID, IMAGE_ID) else -100 for t in row_b]
    return {
        "input_ids": torch.tensor([row_a, row_b]),
        "labels": torch.tensor([labels_a, labels_b]),
    }


def test_the_vision_collator_gets_a_masker_that_masks_the_closer():
    collator = UnslothVisionDataCollator(StubProcessor())
    trainer = StubTrainer(collator)

    out = train_on_responses_only(
        trainer, INSTRUCTION_PART, RESPONSE_PART, mask_out_tokens = [THINK_CLOSE]
    )

    assert out is trainer
    assert callable(collator.train_on_responses_only), "the collator was not configured"

    # the call __call__ makes on every batch
    labels = collator.train_on_responses_only(_collated_batch())["labels"]

    assert type(labels) is torch.Tensor
    assert labels[0].tolist() == [-100, -100, -100, -100, -100, 20, -100, 21, -100]
    assert labels[1].tolist() == [-100, -100, -100, -100, -100, 30, 31, 32, 33]


def test_the_vision_collator_without_the_argument_trains_the_closer():
    """Control for the vision-collator path."""
    collator = UnslothVisionDataCollator(StubProcessor())

    train_on_responses_only(StubTrainer(collator), INSTRUCTION_PART, RESPONSE_PART)
    labels = collator.train_on_responses_only(_collated_batch())["labels"]

    assert labels[0].tolist() == [-100, -100, -100, -100, -100, 20, THINK_ID, 21, -100]
    assert labels[1].tolist() == [-100, -100, -100, -100, THINK_ID, 30, 31, 32, 33]


# ------------------------------------------------------------------ MLX delegation


def test_the_mlx_delegation_forwards_the_argument(monkeypatch):
    """The MLXTrainer branch forwards keyword by keyword, and a missing keyword is
    accepted silently. ``unsloth_zoo.mlx.trainer`` is faked so this runs without MLX."""
    calls = []

    class MLXTrainer:
        pass

    def fake_mlx_train_on_responses_only(trainer, **kwargs):
        calls.append(kwargs)
        return trainer

    fake = types.ModuleType("unsloth_zoo.mlx.trainer")
    fake.MLXTrainer = MLXTrainer
    fake.train_on_responses_only = fake_mlx_train_on_responses_only
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.trainer", fake)
    trainer = MLXTrainer()

    out = train_on_responses_only(
        trainer, INSTRUCTION_PART, RESPONSE_PART, mask_out_tokens = [THINK_CLOSE]
    )

    assert out is trainer
    assert len(calls) == 1
    assert calls[0]["mask_out_tokens"] == [THINK_CLOSE]
    assert calls[0]["instruction_part"] == INSTRUCTION_PART
    assert calls[0]["response_part"] == RESPONSE_PART


# ------------------------------------------------------- a real tokenizer, in memory


def test_the_precompute_against_a_real_fast_tokenizer():
    """A real ``PreTrainedTokenizerFast`` built in memory: word-level vocab, the markers
    and ``</think>`` as added special tokens. No weights, no network."""
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "[PAD]", "hello", "there", "let", "me", "think",
             "the", "answer", "is", "four"]
    core = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token = "[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    core.add_special_tokens([
        AddedToken(marker, special = True)
        for marker in (INSTRUCTION_PART, RESPONSE_PART, THINK_CLOSE)
    ])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = core, unk_token = "[UNK]", pad_token = "[PAD]"
    )
    think_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    assert tokenizer(THINK_CLOSE, add_special_tokens = False).input_ids == [think_id]

    text = "<|user|> hello there <|assistant|> let me think </think> the answer is four"
    input_ids = tokenizer(text, add_special_tokens = False).input_ids
    assert input_ids.count(think_id) == 1
    assert tokenizer.unk_token_id not in input_ids

    masker = train_on_responses_only(
        None,
        INSTRUCTION_PART,
        RESPONSE_PART,
        tokenizer = tokenizer,
        return_function = True,
        mask_out_tokens = [THINK_CLOSE],
    )
    labels = masker({"input_ids": [input_ids]})["labels"][0]

    assert labels[input_ids.index(think_id)] == -100
    trained = tokenizer.decode([label for label in labels if label != -100])
    assert "the answer is four" in trained
    assert "let me think" in trained, "only the closer is masked"
    assert THINK_CLOSE not in trained
    assert "hello" not in trained and INSTRUCTION_PART not in trained
