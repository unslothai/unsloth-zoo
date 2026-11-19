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

"""unsloth_label_smoothing_override vs an fp64 F.cross_entropy(label_smoothing) oracle."""

import pytest
import torch

from unsloth_zoo.fused_losses.cross_entropy_loss import (
    unsloth_fused_ce_loss,
    unsloth_label_smoothing_override,
)

V, H = 41, 12


def _inputs():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(3, 9, H, dtype = torch.float64, generator = g).requires_grad_()
    w = torch.randn(V, H, dtype = torch.float64, generator = g).requires_grad_()
    labels = torch.randint(0, V, (3, 9), generator = g)
    labels[0, :5] = -100
    labels[1] = -100
    return h, w, labels


def _oracle(h, w, labels, eps, divisor):
    logits = (h @ w.T)[:, :-1].reshape(-1, V)
    return torch.nn.functional.cross_entropy(
        logits, labels[:, 1:].reshape(-1), label_smoothing = eps, reduction = "sum"
    ) / divisor


@pytest.mark.parametrize("eps", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("divisor", [7, 30])
def test_override_matches_torch_label_smoothing(eps, divisor):
    h, w, labels = _inputs()
    with unsloth_label_smoothing_override(eps, torch.tensor(divisor)) as state:
        loss = unsloth_fused_ce_loss(None, h, w, None, labels, n_items = torch.tensor(999), torch_compile = False)
    assert state.engaged == 1 and not state.active
    ref = _oracle(h, w, labels, eps, divisor)
    torch.testing.assert_close(loss.double(), ref, rtol = 1e-6, atol = 1e-6)
    grads = torch.autograd.grad(loss, (h, w))
    ref_grads = torch.autograd.grad(ref, (h, w))
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g.double(), r, rtol = 1e-5, atol = 1e-7)


def test_no_override_outside_block():
    h, w, labels = _inputs()
    with unsloth_label_smoothing_override(0.5, torch.tensor(3)):
        pass
    loss = unsloth_fused_ce_loss(None, h, w, None, labels, torch_compile = False)
    n = (labels[:, 1:] != -100).sum()
    torch.testing.assert_close(loss.double(), _oracle(h, w, labels, 0.0, n), rtol = 1e-6, atol = 1e-6)


def test_override_rejects_bad_epsilon_and_reentry():
    with pytest.raises(ValueError):
        with unsloth_label_smoothing_override(1.5, 1):
            pass
    with unsloth_label_smoothing_override(0.1, 1):
        with pytest.raises(RuntimeError):
            with unsloth_label_smoothing_override(0.1, 1):
                pass
