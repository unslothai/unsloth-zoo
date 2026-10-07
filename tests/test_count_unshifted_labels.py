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

"""A head marked `_unsloth_counts_unshifted_labels` (aligned decoder, encoder-decoder) averages every
non-ignored label, so the GA count must skip the causal labels[..., 1:] shift. CPU only."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_loss_normalization_contract import _fake_trainer, _loss_utils  # noqa: E402

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class AlignedForCausalLM(nn.Module):
    """BartForCausalLM-style aligned CE: labels already line up with logits, no shift."""

    def __init__(self, vocab = 11, hidden = 6):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias = False)

    def forward(self, input_ids, labels = None, attention_mask = None, **kwargs):
        logits = self.lm_head(self.embed(input_ids))
        n_items = kwargs.get("num_items_in_batch", None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            reduction = "sum" if n_items is not None else "mean",
        )
        return loss / n_items if n_items is not None else loss


class MarkedAlignedForCausalLM(AlignedForCausalLM):
    _unsloth_counts_unshifted_labels = True


def _batches():
    g = torch.Generator().manual_seed(3)
    out = []
    for lengths in ((1, 4), (6, 2), (3, 3)):
        ids = torch.randint(0, 11, (2, 6), generator = g)
        labels = ids.clone()
        mask = torch.ones_like(ids)
        for row, n in enumerate(lengths):
            labels[row, n:] = -100
            mask[row, n:] = 0
        out.append({"input_ids": ids, "labels": labels, "attention_mask": mask})
    return out


def _count(model, batches, accepts = True, **kw):
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    return mod._unsloth_get_batch_samples(_fake_trainer(model, accepts, **kw), iter(batches), len(batches))[1]


def test_marked_head_counts_every_label():
    batches = _batches()
    expected = sum(int((b["labels"] != -100).sum()) for b in batches)
    assert int(_count(MarkedAlignedForCausalLM(), batches)) == expected == 19


def test_unmarked_head_keeps_the_shifted_count():
    batches = _batches()
    shifted = sum(int(((b["labels"][..., 1:] != -100) & (b["attention_mask"][..., 1:] != 0)).sum()) for b in batches)
    assert int(_count(AlignedForCausalLM(), batches)) == shifted == 13


def test_instance_marker_counts_too():
    model = AlignedForCausalLM()
    model._unsloth_counts_unshifted_labels = True
    assert int(_count(model, _batches())) == 19


def test_marked_head_is_ga_invariant():
    batches = _batches()
    model = MarkedAlignedForCausalLM()
    full = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    model(**full).backward()
    reference = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none = True)
    count = _count(model, batches)
    for b in batches:
        model(**b, num_items_in_batch = count).backward()
    torch.testing.assert_close(model.lm_head.weight.grad, reference)


def test_marked_head_still_respects_the_loss_kwargs_gate():
    assert _count(MarkedAlignedForCausalLM(), _batches(), accepts = False) is None


def test_marked_single_column_labels_are_countable():
    batch = {"input_ids": torch.tensor([[1], [2]]), "labels": torch.tensor([[1], [-100]])}
    assert int(_count(MarkedAlignedForCausalLM(), [batch])) == 1


def test_marker_installed_after_a_first_count_is_seen():
    batches = _batches()
    model = AlignedForCausalLM()
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    trainer = _fake_trainer(model, True)
    assert int(mod._unsloth_get_batch_samples(trainer, iter(batches), 3)[1]) == 13
    type(model)._unsloth_counts_unshifted_labels = True
    try:
        assert int(mod._unsloth_get_batch_samples(trainer, iter(batches), 3)[1]) == 19
    finally:
        del type(model)._unsloth_counts_unshifted_labels


def test_a_replaced_forward_refreshes_the_kwargs_cache():
    class NoKwargsForCausalLM(AlignedForCausalLM):
        _unsloth_counts_unshifted_labels = True

        def forward(self, input_ids, labels = None, attention_mask = None):
            return super().forward(input_ids, labels, attention_mask)

    batches = _batches()
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    model = NoKwargsForCausalLM()
    trainer = _fake_trainer(model, True)
    assert mod._unsloth_get_batch_samples(trainer, iter(batches), 3)[1] is None
    original = NoKwargsForCausalLM.forward
    NoKwargsForCausalLM.forward = AlignedForCausalLM.forward
    try:
        assert int(mod._unsloth_get_batch_samples(trainer, iter(batches), 3)[1]) == 19
    finally:
        NoKwargsForCausalLM.forward = original
