# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""Offline tests for unsloth_zoo.dataset_utils.mask_thinking_tokens (no network)."""

import importlib.util
import os

import pytest
import torch

# Load dataset_utils.py directly from its file so the test is self-contained and
# does not run unsloth_zoo/__init__.py (which pulls the full stack).
_DATASET_UTILS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth_zoo", "dataset_utils.py",
)
_spec = importlib.util.spec_from_file_location("_unsloth_dataset_utils_offline", _DATASET_UTILS_PATH)
_dataset_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dataset_utils)
mask_thinking_tokens = _dataset_utils.mask_thinking_tokens


class FakeTokenizer:
    """Minimal stand-in: encodes the think token to a fixed id sequence."""

    def __init__(self, think_token, think_ids):
        self._think_token = think_token
        self._think_ids = list(think_ids)

    def __call__(self, text, add_special_tokens=False):
        ids = self._think_ids if text == self._think_token else [0, 1]
        return {"input_ids": ids}


def _mask_fn(think_token, think_ids):
    tok = FakeTokenizer(think_token, think_ids)
    return mask_thinking_tokens(
        trainer = None,
        think_token = think_token,
        tokenizer = tok,
        return_function = True,
    )


def test_single_token_masked():
    fn = _mask_fn("</think>", [42])
    out = fn({"input_ids": [[5, 42, 7, 8]], "labels": [[5, 42, 7, 8]]})
    assert out["labels"] == [[5, -100, 7, 8]]


def test_no_match_leaves_labels_untouched():
    fn = _mask_fn("</think>", [42])
    out = fn({"input_ids": [[5, 6, 7]], "labels": [[5, 6, 7]]})
    assert out["labels"] == [[5, 6, 7]]


def test_multi_token_think_masks_full_span():
    fn = _mask_fn("</think>", [42, 43])
    out = fn({"input_ids": [[1, 42, 43, 9]], "labels": [[1, 42, 43, 9]]})
    assert out["labels"] == [[1, -100, -100, 9]]


def test_multi_token_partial_not_masked():
    fn = _mask_fn("</think>", [42, 43])
    # 42 not followed by 43 must NOT be masked; the real span must be.
    out = fn({"input_ids": [[42, 9, 42, 43]], "labels": [[42, 9, 42, 43]]})
    assert out["labels"] == [[42, 9, -100, -100]]


def test_multiple_occurrences_all_masked():
    fn = _mask_fn("</think>", [42])
    out = fn({"input_ids": [[42, 1, 42, 2]], "labels": [[42, 1, 42, 2]]})
    assert out["labels"] == [[-100, 1, -100, 2]]


def test_preserves_existing_minus_100():
    # Mirrors composing after train_on_responses_only: prompt already -100,
    # we only additionally mask the </think> token.
    fn = _mask_fn("</think>", [42])
    out = fn({"input_ids": [[5, 6, 42, 7]], "labels": [[-100, -100, 42, 7]]})
    assert out["labels"] == [[-100, -100, -100, 7]]


def test_derives_labels_when_absent():
    fn = _mask_fn("</think>", [42])
    out = fn({"input_ids": [[5, 42, 7]]})
    assert out["labels"] == [[5, -100, 7]]


def test_input_ids_not_mutated_when_deriving_labels():
    fn = _mask_fn("</think>", [42])
    examples = {"input_ids": [[5, 42, 7]]}
    fn(examples)
    assert examples["input_ids"] == [[5, 42, 7]]


def test_tensor_inputs_round_trip():
    fn = _mask_fn("</think>", [42])
    out = fn({
        "input_ids": torch.tensor([[5, 42, 7]]),
        "labels": torch.tensor([[5, 42, 7]]),
    })
    assert isinstance(out["labels"], torch.Tensor)
    assert out["labels"].tolist() == [[5, -100, 7]]


def test_empty_think_token_raises():
    tok = FakeTokenizer("</think>", [])
    with pytest.raises(ValueError):
        mask_thinking_tokens(trainer = None, think_token = "</think>", tokenizer = tok, return_function = True)


def test_list_input_tensor_labels_preserve_tensor():
    # input_ids is a list but labels is a tensor -> output stays a tensor.
    fn = _mask_fn("</think>", [42])
    out = fn({"input_ids": [[5, 42, 7]], "labels": torch.tensor([[5, 42, 7]])})
    assert isinstance(out["labels"], torch.Tensor)
    assert out["labels"].tolist() == [[5, -100, 7]]


def test_dataset_as_trainer_raises():
    # Passing a Dataset (has .map, no train/eval datasets) must error, not no-op.
    class FakeDataset:
        def map(self, *args, **kwargs):
            return self

    tok = FakeTokenizer("</think>", [42])
    with pytest.raises(TypeError):
        mask_thinking_tokens(trainer = FakeDataset(), tokenizer = tok)


def test_no_tokenizer_or_trainer_raises():
    with pytest.raises(ValueError):
        mask_thinking_tokens(trainer = None, tokenizer = None)
