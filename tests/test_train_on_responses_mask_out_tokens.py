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

"""mask_out_tokens on train_on_responses_only (unslothai/unsloth issue 6695): every route to the masking closure must
thread it, as a dropped one fails silently; covers list, tensor, vision-collator and MLX paths; CPU-only and offline."""

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


def test_a_multi_token_sequence_matches_whole_sequences_only():
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




def test_tensor_batches_keep_their_type_and_their_existing_masks():
    """A collator batch of padded tensors with pads already -100: the closer is matched on input_ids and masked in labels; kept positions come from labels."""
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




def test_the_mlx_delegation_forwards_the_argument(monkeypatch):
    """The MLXTrainer branch forwards keyword by keyword and a missing keyword is accepted silently; unsloth_zoo.mlx.trainer is faked so MLX is not needed."""
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




def test_the_precompute_against_a_real_fast_tokenizer():
    """A real PreTrainedTokenizerFast built in memory: word-level vocab with the markers and </think> as added special tokens; no weights, no network."""
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




def test_a_separate_space_token_before_the_closer_stays_trained():
    """Qwen3 and SmolLM3 encode ' </think>' as [space, </think>]; only the closer is masked, the space before it stays trained."""
    space = ord(" ")
    row = [USER_ID, 10, ASSISTANT_ID, 20, space, THINK_ID, 21]

    labels = _masker(mask_out_tokens = [THINK_CLOSE])({"input_ids": [row]})["labels"][0]

    assert labels == [-100, -100, -100, 20, space, -100, 21]


def test_a_merged_leading_space_form_is_still_matched():
    """BPE merges " ab" into one piece; that in-context form is masked too."""
    class MergingTokenizer(StubTokenizer):
        def __call__(self, text, add_special_tokens = False, **kwargs):
            if text == " ab":
                return StubEncoding(input_ids = [77], attention_mask = [1])
            return super().__call__(text, add_special_tokens, **kwargs)

    a, b = ord("a"), ord("b")
    row = [USER_ID, 10, ASSISTANT_ID, a, b, 20, 77, 21]
    masker = train_on_responses_only(
        None, INSTRUCTION_PART, RESPONSE_PART, tokenizer = MergingTokenizer(),
        return_function = True, mask_out_tokens = ["ab"],
    )

    labels = masker({"input_ids": [row]})["labels"][0]

    assert labels == [-100, -100, -100, -100, -100, 20, -100, 21]




def _real_collator(**kwargs):
    from test_vision_collator_string_content import PHI4V_TEMPLATE, _Model, _make_processor
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    processor = _make_processor(PHI4V_TEMPLATE)
    processor.tokenizer.add_tokens([THINK_CLOSE], special_tokens = True)
    collator = UnslothVisionDataCollator(
        _Model(), processor, train_on_responses_only = True,
        instruction_part = "<|im_start|>user<|im_sep|>",
        response_part = "<|im_start|>assistant<|im_sep|>", **kwargs,
    )
    return collator, processor.tokenizer.convert_tokens_to_ids(THINK_CLOSE)


@pytest.mark.parametrize("mask", [True, False])
def test_the_collator_constructor_forwards_the_argument(mask):
    from test_vision_collator_string_content import _example

    kwargs = {"mask_out_tokens": [THINK_CLOSE]} if mask else {}
    collator, think_id = _real_collator(**kwargs)
    batch = collator([_example("ab </think> cd"), _example("</think>xy")])

    input_ids, labels = batch["input_ids"], batch["labels"]
    at_closer = labels[input_ids == think_id]
    assert at_closer.numel() == 2
    if mask:
        assert (at_closer == -100).all()
    else:
        assert (at_closer == think_id).all()
    # the rest of the answer stays trained either way
    assert ((labels != -100) & (input_ids != think_id)).sum() > 0


def test_overlapping_occurrences_are_all_masked():
    a = ord("a")
    row = [USER_ID, 10, ASSISTANT_ID, a, a, a, a, 20]

    labels = _masker(mask_out_tokens = ["aaa"])({"input_ids": [row]})["labels"][0]

    assert labels == [-100, -100, -100, -100, -100, -100, -100, 20]


def test_a_collator_that_already_masks_refuses_mask_out_tokens():
    """Its masker was built without the argument; ignoring it silently would train the closer."""
    collator = UnslothVisionDataCollator(StubProcessor())
    collator.train_on_responses_only = lambda batch: batch
    trainer = StubTrainer(collator)

    assert train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART) is trainer
    with pytest.raises(ValueError, match = "mask_out_tokens"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART, mask_out_tokens = [THINK_CLOSE])


@pytest.mark.parametrize("strings", [
    ["a", "ab", "aba", "bc", THINK_CLOSE],            # 3 first ids: list.index scans
    ["a", "ab", "aba", "bc", "c", "d", THINK_CLOSE],  # 5 first ids: one set-membership pass
])
def test_matches_a_brute_force_scan_on_random_rows(strings):
    """Shared first ids, overlaps and truncated tails, against a naive per-position scan."""
    import random
    rng = random.Random(3407)
    a, b, c, d = ord("a"), ord("b"), ord("c"), ord("d")
    seqs = [StubTokenizer()(t).input_ids for t in strings]
    masker = _masker(mask_out_tokens = strings)
    for _ in range(200):
        row = [USER_ID, 10, ASSISTANT_ID] + [rng.choice([a, b, c, d, THINK_ID, 20]) for _ in range(rng.randint(0, 30))]
        expected = [-100] * 3 + row[3:]
        for seq in seqs:
            for i in range(len(row) - len(seq) + 1):
                if row[i : i + len(seq)] == seq:
                    expected[i : i + len(seq)] = [-100] * len(seq)
        assert masker({"input_ids": [row]})["labels"][0] == expected
