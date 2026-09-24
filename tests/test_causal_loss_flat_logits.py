# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The patched ForCausalLM loss accepts flat (tokens, vocab) logits like the stock loss.

Remote inclusionAI/Ling-2.6-flash computes its multi-token-prediction loss as
self.loss_function(logits.view(-1, vocab), labels.view(-1), vocab). Stock
ForCausalLMLoss flattens anyway, but the Unsloth kernel unpacked (batch, seq, vocab)
and raised "not enough values to unpack (expected 3, got 2)" on the first step.
"""

from __future__ import annotations

import pytest
import torch


def _fast_ce_3d(logits, labels, n_items = None, **kw):
    batch, seq_len, vocab = logits.shape  # same contract as unsloth's fast kernel
    assert labels.shape == (batch, seq_len)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab).float(), labels.reshape(-1), ignore_index = -100, reduction = "sum",
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    return loss / n_items


@pytest.fixture
def patched_loss():
    lu = pytest.importorskip("transformers.loss.loss_utils")
    from unsloth_zoo import loss_utils as zoo_loss
    stock = lu.ForCausalLMLoss
    saved = dict(lu.LOSS_MAPPING)
    try:
        zoo_loss.patch_loss_functions(_fast_ce_3d, torch_compile = False)
        yield stock, lu.LOSS_MAPPING["ForCausalLM"]
    finally:
        lu.LOSS_MAPPING.clear()
        lu.LOSS_MAPPING.update(saved)


def _inputs(batch = 2, seq = 7, vocab = 11):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(batch, seq, vocab, generator = g)
    labels = torch.randint(0, vocab, (batch, seq), generator = g)
    labels[0, :2] = -100
    return logits, labels, vocab


def test_flat_logits_match_the_stock_loss(patched_loss):
    stock, unsloth = patched_loss
    logits, labels, vocab = _inputs()
    flat_logits, flat_labels = logits.reshape(-1, vocab), labels.reshape(-1)
    expected = stock(flat_logits, flat_labels, vocab)
    got = unsloth(flat_logits, flat_labels, vocab)
    torch.testing.assert_close(got, expected)


def test_batched_logits_are_unchanged(patched_loss):
    stock, unsloth = patched_loss
    logits, labels, vocab = _inputs()
    torch.testing.assert_close(unsloth(logits, labels, vocab), stock(logits, labels, vocab))


def test_num_items_in_batch_is_honoured_for_flat_logits(patched_loss):
    stock, unsloth = patched_loss
    logits, labels, vocab = _inputs()
    flat_logits, flat_labels = logits.reshape(-1, vocab), labels.reshape(-1)
    n = torch.tensor(5)
    torch.testing.assert_close(
        unsloth(flat_logits, flat_labels, vocab, num_items_in_batch = n),
        stock(flat_logits, flat_labels, vocab, num_items_in_batch = n),
    )
