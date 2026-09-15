# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_untrained_tokens must accept any indexable train_dataset, not only a
datasets.Dataset (unsloth#2953).

Every other train_dataset use in that function needs only len() and [j]; the
untrained-token count was the single line that required .map, so a plain list of
rows reached it and died with AttributeError. TRL 1.x rejects a list before the
trainer is built, but TRL 0.22.x does not, so this guard is the only protection on
the older line.

The second half of this file pins the decorator. `fix_untrained_tokens` writes the
corrected rows back with `embedding_matrix[where_untrained] = ...`, an in-place
write to a leaf weight that requires grad, which torch refuses outside inference
mode. Introducing the helper above the function is therefore only safe if the
helper is defined above the `@_maybe_inference_mode` line rather than between the
decorator and the function it decorates.
"""

import itertools

import numpy as np
import pytest
import torch

from unsloth_zoo import tokenizer_utils


def _counter(vocab_size = 8):
    final_counts = np.zeros(vocab_size, dtype = np.int64)

    def mapping(examples):
        input_ids = examples["input_ids"]
        counter = np.fromiter(itertools.chain.from_iterable(input_ids), dtype = np.int32)
        np.add.at(final_counts, counter, 1)

    return final_counts, mapping


def test_count_input_ids_accepts_a_plain_list_of_rows():
    final_counts, mapping = _counter()
    rows = [{"input_ids" : [1, 2, 3]}, {"input_ids" : [2, 3, 4]}]

    tokenizer_utils._count_input_ids(rows, mapping)

    assert final_counts.tolist() == [0, 1, 2, 2, 1, 0, 0, 0]


def test_count_input_ids_skips_rows_without_input_ids():
    final_counts, mapping = _counter()

    tokenizer_utils._count_input_ids([{"text" : "hi"}], mapping)

    assert final_counts.sum() == 0


def test_count_input_ids_still_uses_map_when_it_exists():
    """The datasets path must not regress: batched .map with the progress desc."""
    seen = {}

    class _FakeDataset(list):
        def map(self, function, **kwargs):
            seen.update(kwargs)
            function({"input_ids" : [row["input_ids"] for row in self]})

    final_counts, mapping = _counter()
    tokenizer_utils._count_input_ids(_FakeDataset([{"input_ids" : [1, 1]}]), mapping)

    assert seen["batched"] is True
    assert seen["desc"] == "Counting untrained tokens"
    assert final_counts[1] == 2


def test_count_input_ids_works_on_a_real_dataset():
    datasets = pytest.importorskip("datasets")
    final_counts, mapping = _counter()

    tokenizer_utils._count_input_ids(
        datasets.Dataset.from_list([{"input_ids" : [1, 2]}, {"input_ids" : [2, 2]}]),
        mapping,
    )

    assert final_counts.tolist() == [0, 1, 3, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# The decorator must still land on fix_untrained_tokens
# ---------------------------------------------------------------------------

class _Tokenizer:
    chat_template = "<|extra_0|> in the template"

    def __len__(self):
        return 8

    def convert_ids_to_tokens(self, ids):
        return [f"<|extra_{i}|>" for i in range(len(ids))]


class _Model:
    def __init__(self, embedding):
        self._embedding = embedding

        class _Config:
            _name_or_path = "unsloth/tiny-test-model"

        self.config = _Config()

    def get_input_embeddings(self):
        return self._embedding

    def get_output_embeddings(self):
        return self._embedding


def _untrained_model():
    """Rows 4 and 5 are all zero, so both untrained indicators fire on them, and the
    weight is a trainable leaf, so the write-back needs inference mode."""
    embedding = torch.nn.Embedding(8, 4)
    with torch.no_grad():
        embedding.weight.fill_(1.0)
        embedding.weight[4].zero_()
        embedding.weight[5].zero_()
    assert embedding.weight.requires_grad and embedding.weight.is_leaf
    return _Model(embedding)


def test_fix_untrained_tokens_is_still_decorated():
    wrapper = tokenizer_utils.fix_untrained_tokens
    assert wrapper.__code__.co_name == "wrapper", (
        "fix_untrained_tokens lost @_maybe_inference_mode; a helper was most likely "
        "inserted between the decorator line and the function"
    )
    assert getattr(wrapper, "__wrapped__", None) is not None
    assert wrapper.__wrapped__.__code__.co_name == "fix_untrained_tokens"


def test_count_input_ids_did_not_steal_the_decorator():
    assert getattr(tokenizer_utils._count_input_ids, "__wrapped__", None) is None
    assert tokenizer_utils._count_input_ids.__code__.co_name == "_count_input_ids"


def test_fix_untrained_tokens_runs_under_inference_mode():
    seen = []
    original = tokenizer_utils._count_input_ids

    def spy(train_dataset, mapping):
        seen.append(torch.is_inference_mode_enabled())
        return original(train_dataset, mapping)

    tokenizer_utils._count_input_ids = spy
    try:
        model = _untrained_model()
        tokenizer_utils.fix_untrained_tokens(
            model, _Tokenizer(), [{"input_ids" : [1, 2, 4]}, {"input_ids" : [2, 3, 5]}],
        )
    finally:
        tokenizer_utils._count_input_ids = original

    assert seen == [True], "fix_untrained_tokens body did not run under inference mode"
    # The whole point of the inference mode: the untrained rows are written back in
    # place on a leaf that requires grad, which torch rejects otherwise.
    assert not torch.equal(model.get_input_embeddings().weight[4], torch.zeros(4))


def test_fix_untrained_tokens_end_to_end_on_a_list_dataset():
    model = _untrained_model()

    tokenizer_utils.fix_untrained_tokens(
        model, _Tokenizer(), [{"input_ids" : [1, 2, 4]}, {"input_ids" : [2, 3, 5]}],
    )

    weight = model.get_input_embeddings().weight
    assert torch.all(weight[:4] == 1.0)
    assert not torch.equal(weight[4], torch.zeros(4))
    assert not torch.equal(weight[5], torch.zeros(4))


def test_the_undecorated_function_cannot_write_the_rows_back():
    """Proof that the decorator is load-bearing and not cosmetic. Run outside
    inference mode the same body fails on a trainable weight: first on
    lm_head_bad.numpy(), and if that is detached, on the in-place write back."""
    model = _untrained_model()

    with pytest.raises(RuntimeError, match = "requires grad|in-place operation"):
        tokenizer_utils.fix_untrained_tokens.__wrapped__(
            model, _Tokenizer(), [{"input_ids" : [1, 2, 4]}, {"input_ids" : [2, 3, 5]}],
        )


def test_count_input_ids_accepts_rows_whose_ids_are_arrays():
    """A row read back from a collator or a torch dataset holds a tensor or an
    ndarray, not a list. The counting path only chains and counts, so both work,
    and pinning it keeps the fallback honest about what it accepts."""
    final_counts, mapping = _counter()

    tokenizer_utils._count_input_ids(
        [{"input_ids" : np.array([1, 2])}, {"input_ids" : torch.tensor([2, 3])}], mapping,
    )

    assert final_counts.tolist() == [0, 1, 2, 1, 0, 0, 0, 0]


def test_count_input_ids_accepts_a_generator_of_rows():
    final_counts, mapping = _counter()

    tokenizer_utils._count_input_ids(iter([{"input_ids" : [5]}, {"input_ids" : [5, 6]}]), mapping)

    assert final_counts.tolist() == [0, 0, 0, 0, 0, 2, 1, 0]


def test_count_input_ids_on_an_empty_list_never_calls_the_mapping():
    called = []
    tokenizer_utils._count_input_ids([], lambda examples: called.append(examples))
    assert called == []
