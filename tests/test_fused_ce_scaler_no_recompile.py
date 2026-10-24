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

"""A changing fp16 GradScaler scale must not recompile the fused CE chunk loop.

GradScaler.get_scale() is a Python float. Passed straight into the compiled
accumulate_chunk, dynamo guarded on its value (``scaling == 65536.0``), so each
overflow halving and each growth step built a new graph, and after the
recompile limit the chunk loop ran eager.
"""

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

import unsloth_zoo.fused_losses.cross_entropy_loss as ce

V, H = 256, 48


def _run(scaling, bsz = 2, qlen = 40, n_chunks = 4, seed = 0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(bsz, qlen, H, generator = g).requires_grad_(True)
    weight = torch.randn(V, H, generator = g) * 0.05
    labels = torch.randint(0, V, (bsz, qlen), generator = g)
    loss = ce.unsloth_fused_ce_loss(None, hidden, weight, None, labels, scaling = scaling, n_chunks = n_chunks)
    # The trainer backpropagates the GradScaler-scaled loss.
    (loss * (1.0 if scaling is None else scaling)).backward()
    return loss.detach(), hidden.grad


def _unique_graphs():
    return counters["stats"].get("unique_graphs", 0)


@pytest.fixture(autouse = True)
def _fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


def test_scale_changes_reuse_one_graph():
    if ce._FUSED_CE_COMPILE_SUPPORTED is False:
        pytest.skip("fused CE compile unsupported here")
    _run(65536.0)
    first = _unique_graphs()
    assert first >= 1
    for scale in (32768.0, 16384.0, 8192.0, 131072.0, 3.0):
        _run(scale)
    assert _unique_graphs() == first, dict(counters["stats"])


def test_scaled_gradient_is_unchanged():
    # GradScaler scales are powers of two, so the scaled gradient is exactly scale times the
    # scaling=1 gradient, and the returned loss is the unscaled one.
    loss_1, grad_1 = _run(1.0, seed = 1)
    for scale in (65536.0, 1024.0, 0.5):
        loss_s, grad_s = _run(scale, seed = 1)
        assert torch.equal(loss_s, loss_1)
        assert torch.equal(grad_s / scale, grad_1)


def test_tensor_and_none_scaling_still_work():
    loss_n, grad_n = _run(None, seed = 2)
    loss_t, grad_t = _run(torch.tensor(1.0), seed = 2)
    torch.testing.assert_close(loss_t, loss_n)
    torch.testing.assert_close(grad_t, grad_n)
