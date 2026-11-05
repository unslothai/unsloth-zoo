# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_untrained_tokens runs in UnslothSFTTrainer.__init__ before TRL tokenizes, so train_dataset is usually raw.
Rows of a raw datasets.Dataset have no "input_ids", so the row scans must not fetch (and decode) them, and the
token count must not index a column that does not exist."""

import os
import sys

import numpy as np
import pytest
import torch

datasets = pytest.importorskip("datasets")

sys.path.insert(0, os.path.dirname(__file__))
from test_fix_untrained_tokens_list_dataset import _Tokenizer, _untrained_model  # noqa: E402

from unsloth_zoo import tokenizer_utils  # noqa: E402


@pytest.fixture
def image_decodes(monkeypatch):
    calls = []
    original = datasets.features.Image.decode_example

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(datasets.features.Image, "decode_example", counting)
    return calls


def _image_dataset(n = 600):
    from PIL import Image
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    return datasets.Dataset.from_dict({
        "image" : [image] * n,
        "text"  : [f"row {i}" for i in range(n)],
    }).cast_column("image", datasets.Image())


def _frozen_untrained_model():
    model = _untrained_model()
    model.get_input_embeddings().weight.requires_grad_(False)
    return model


class _NoTemplateTokenizer(_Tokenizer):
    chat_template = None

    def decode(self, ids):
        return " ".join(map(str, ids))


def test_raw_media_dataset_is_not_decoded_by_the_scans(image_decodes):
    tokenizer_utils.fix_untrained_tokens(_frozen_untrained_model(), _NoTemplateTokenizer(), _image_dataset())
    assert len(image_decodes) == 0


def test_raw_media_dataset_is_not_decoded_on_the_not_trainable_path(image_decodes):
    """The chat template holds an untrained token and the embeddings are frozen: main re-scans 250+250+2000+2000 rows,
    finds no input_ids, and returns. Head returns the same way without reading a row."""
    model = _frozen_untrained_model()
    before = model.get_input_embeddings().weight.clone()
    tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), _image_dataset())
    assert len(image_decodes) == 0
    assert torch.equal(model.get_input_embeddings().weight, before)


def test_raw_text_dataset_with_trainable_embeddings_does_not_raise_keyerror():
    """Chat template holds an untrained token and embed/lm_head are trainable (full finetune, modules_to_save):
    main counted with examples["input_ids"] on a dataset without that column and raised KeyError."""
    model = _untrained_model()
    tokenizer_utils.fix_untrained_tokens(
        model, _Tokenizer(), datasets.Dataset.from_list([{"text" : "hello"}, {"text" : "world"}]),
    )
    weight = model.get_input_embeddings().weight
    # Same as a dataset whose rows never contain the untrained ids: unseen rows are set to zero.
    assert torch.all(weight[:4] == 1.0)
    assert torch.equal(weight[4:6], torch.zeros(2, 4))


def test_custom_transform_that_adds_input_ids_is_still_scanned():
    dataset = datasets.Dataset.from_list([{"text" : "a"}, {"text" : "b"}])
    dataset.set_transform(lambda batch: {"input_ids" : [[1, 4] for _ in batch["text"]]})
    with pytest.raises(ValueError, match = "Untrained tokens"):
        tokenizer_utils.fix_untrained_tokens(_frozen_untrained_model(), _NoTemplateTokenizer(), dataset)


@pytest.mark.parametrize("rows", [
    [{"input_ids" : [1, 2, 4]}, {"input_ids" : [2, 3, 5]}],
    [{"input_ids" : [1, 2, 3]}, {"input_ids" : [2, 3, 1]}],
])
def test_tokenized_dataset_matches_the_list_path(rows):
    expected = _untrained_model()
    tokenizer_utils.fix_untrained_tokens(expected, _Tokenizer(), rows)
    model = _untrained_model()
    tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), datasets.Dataset.from_list(rows))
    assert torch.equal(model.get_input_embeddings().weight, expected.get_input_embeddings().weight)


def test_rows_lack_input_ids_only_for_plain_datasets_without_the_column():
    raw = datasets.Dataset.from_list([{"text" : "a"}])
    assert tokenizer_utils._rows_lack_input_ids(raw)
    assert tokenizer_utils._rows_lack_input_ids(raw.with_format("torch"))
    assert not tokenizer_utils._rows_lack_input_ids(raw.with_transform(lambda b: b))
    assert not tokenizer_utils._rows_lack_input_ids(datasets.Dataset.from_list([{"input_ids" : [1]}]))
    assert not tokenizer_utils._rows_lack_input_ids([{"text" : "a"}])


def _old(where, input_ids):
    return any(item in where for item in input_ids)


def _new(where, input_ids):
    return not where.isdisjoint(input_ids)


@pytest.mark.parametrize("input_ids", [
    [1, 2, 3], [1, 4], [], np.array([1, 5]), np.array([1, 2]), [np.int64(4)], [np.int32(1)],
    torch.tensor([1, 4]), [torch.tensor(4)], (4,), [4.0], [True, 4],
])
def test_isdisjoint_matches_any_in(input_ids):
    where = frozenset([4, 5])
    assert _old(where, input_ids) == _new(where, input_ids)


@pytest.mark.parametrize("input_ids", [[[1, 4]], [1, [4]], [np.array([4])]])
def test_isdisjoint_raises_like_any_in(input_ids):
    where = frozenset([4, 5])
    errors = []
    for check in (_old, _new):
        try:
            errors.append(("ok", check(where, input_ids)))
        except Exception as error:
            errors.append(type(error))
    assert errors[0] == errors[1]
