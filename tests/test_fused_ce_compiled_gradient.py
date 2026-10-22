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

"""Compiled fused CE must give eager's gradient (torch 2.11 traced it to zeros)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_ce_loss


def _loss_and_grad(fn, h0, W, labels):
    h = h0.clone().requires_grad_(True)
    loss = fn(h * 1.0, W, labels)  # an intermediate, as the final norm's output is
    loss.backward()
    return loss.detach(), h.grad


def _fused(x, W, labels):
    return unsloth_fused_ce_loss(None, x, W, None, labels)


def test_compiled_fused_ce_gradient_matches_eager():
    g = torch.Generator(device = "cuda").manual_seed(0)
    h0 = torch.randn(2, 64, 32, device = "cuda", generator = g)
    W = torch.randn(128, 32, device = "cuda", generator = g)
    labels = torch.randint(0, 128, (2, 64), device = "cuda", generator = g)
    loss_e, grad_e = _loss_and_grad(_fused, h0, W, labels)
    torch._dynamo.reset()
    loss_c, grad_c = _loss_and_grad(torch.compile(_fused), h0, W, labels)
    assert grad_e.norm() > 0
    torch.testing.assert_close(loss_c, loss_e)
    torch.testing.assert_close(grad_c, grad_e)
