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

"""Patched ForCausalLM loss accepts flat (tokens, vocab) logits like stock (Ling-2.6-flash MTP head)."""

from __future__ import annotations

import pytest
import torch


def _fast_ce_3d(logits, labels, n_items = None, **kw):
    batch, seq_len, vocab = logits.shape
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


def test_flat_logits_with_batched_labels_keep_row_boundaries(patched_loss):
    """The last token of a row predicts nothing, not the next row's first token."""
    stock, unsloth = patched_loss
    logits, labels, vocab = _inputs()
    flat_logits = logits.reshape(-1, vocab)
    expected = stock(flat_logits, labels, vocab)
    torch.testing.assert_close(expected, stock(logits, labels, vocab))
    torch.testing.assert_close(unsloth(flat_logits, labels, vocab), expected)


@pytest.mark.parametrize("flat", [False, True])
def test_a_custom_ignore_index_matches_the_stock_loss(patched_loss, flat):
    stock, unsloth = patched_loss
    logits, labels, vocab = _inputs()
    labels = labels.masked_fill(labels == -100, -1)
    if flat:
        logits, labels = logits.reshape(-1, vocab), labels.reshape(-1)
    for n in (None, torch.tensor(5)):
        torch.testing.assert_close(
            unsloth(logits, labels, vocab, num_items_in_batch = n, ignore_index = -1),
            stock(logits, labels, vocab, num_items_in_batch = n, ignore_index = -1),
        )
