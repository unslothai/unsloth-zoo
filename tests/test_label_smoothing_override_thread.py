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

"""unsloth_label_smoothing_override applies only to fused CE calls on the thread that opened it."""

import threading

import torch

from unsloth_zoo.fused_losses.cross_entropy_loss import (
    unsloth_fused_ce_loss,
    unsloth_label_smoothing_override,
)

V, H = 41, 12


def _inputs():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 9, H, dtype = torch.float64, generator = g)
    w = torch.randn(V, H, dtype = torch.float64, generator = g)
    labels = torch.randint(0, V, (2, 9), generator = g)
    labels[0, :4] = -100
    return h, w, labels


def _plain_loss_on_other_thread(h, w, labels):
    out = {}

    def run():
        out["loss"] = unsloth_fused_ce_loss(None, h, w, None, labels, torch_compile = False).detach()

    t = threading.Thread(target = run)
    t.start()
    t.join()
    return out["loss"]


def test_other_thread_keeps_plain_loss_during_override():
    h, w, labels = _inputs()
    alone = _plain_loss_on_other_thread(h, w, labels)
    with unsloth_label_smoothing_override(0.5, torch.tensor(3)) as state:
        during = _plain_loss_on_other_thread(h, w, labels)
        assert state.engaged == 0
        own = unsloth_fused_ce_loss(None, h, w, None, labels, torch_compile = False).detach()
        assert state.engaged == 1
    torch.testing.assert_close(during, alone, rtol = 0, atol = 0)
    assert not torch.allclose(own, alone)
