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

"""All labels -100 (e.g. truncated response, batch size 1) must give loss 0 and zero grads, not NaN."""

import pytest
import torch

from unsloth_zoo.fused_losses.cross_entropy_loss import (
    compute_fused_ce_loss,
    unsloth_count_aware_cross_entropy,
    unsloth_fused_ce_loss,
)

H, V = 16, 32


def _inputs(device = "cpu"):
    g = torch.Generator(device = device).manual_seed(0)
    hidden = torch.randn(2, 8, H, device = device, generator = g, requires_grad = True)
    weight = torch.randn(V, H, device = device, generator = g, requires_grad = True)
    labels = torch.full((2, 8), -100, device = device)
    return hidden, weight, labels


def _assert_zero_loss_and_grads(loss, *params):
    loss.backward()
    assert loss.item() == 0.0
    for p in params:
        assert torch.isfinite(p.grad).all() and not p.grad.any()


@pytest.mark.parametrize("n_items", [None, 0, torch.tensor(0)])
def test_compute_fused_ce_loss_all_masked(n_items):
    hidden, weight, labels = _inputs()
    loss, _ = compute_fused_ce_loss(hidden, weight, None, labels, n_items = n_items)
    _assert_zero_loss_and_grads(loss, hidden, weight)


def test_count_aware_cross_entropy_all_masked_with_a_zero_count():
    logits = torch.randn(16, V, requires_grad = True)
    loss = unsloth_count_aware_cross_entropy(logits, torch.full((16,), -100), torch.tensor([0]), shift = False)
    _assert_zero_loss_and_grads(loss, logits)


def test_count_aware_cross_entropy_without_a_count_stays_the_stock_mean():
    logits = torch.randn(16, V)
    labels = torch.full((16,), -100)
    stock = torch.nn.functional.cross_entropy(logits, labels)
    ours = unsloth_count_aware_cross_entropy(logits, labels, None, shift = False)
    torch.testing.assert_close(ours, stock, equal_nan = True)


@pytest.mark.parametrize("n_items", [None, torch.tensor(0)])
def test_chunked_fused_ce_all_masked(n_items):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden, weight, labels = _inputs(device)
    loss = unsloth_fused_ce_loss(None, hidden, weight, None, labels, n_items = n_items, torch_compile = False)
    _assert_zero_loss_and_grads(loss, hidden, weight)


def test_partly_masked_batches_keep_their_normalisation():
    hidden, weight, labels = _inputs()
    labels[0, :4] = torch.randint(0, V, (4,))
    ours, _ = compute_fused_ce_loss(hidden, weight, None, labels, n_items = torch.tensor(7.0))
    logits = (hidden.float() @ weight.float().t())[..., :-1, :].reshape(-1, V)
    target = labels[..., 1:].reshape(-1)
    ref = torch.nn.functional.cross_entropy(logits, target, ignore_index = -100, reduction = "sum") / 7.0
    torch.testing.assert_close(ours, ref)
