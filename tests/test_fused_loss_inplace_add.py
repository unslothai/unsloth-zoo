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

"""MoE forwards on transformers 4.x do `loss += router_aux_loss_coef * aux_loss` on the fused loss."""
import torch

from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_ce_loss


def test_aux_loss_can_be_added_in_place():
    torch.manual_seed(0)
    weight = torch.randn(64, 16, requires_grad = True)
    hidden = torch.randn(2, 8, 16, requires_grad = True)
    labels = torch.randint(0, 64, (2, 8))
    aux = torch.tensor(0.5, requires_grad = True)
    loss = unsloth_fused_ce_loss(
        trainer = None, hidden_states = hidden, lm_head_weight = weight, lm_head_bias = None,
        labels = labels, mask = None, n_items = None, scaling = None,
    )
    loss += 0.01 * aux
    loss.backward()

    ref_hidden = hidden.detach().clone().requires_grad_()
    ref = torch.nn.functional.cross_entropy((ref_hidden[:, :-1] @ weight.detach().T).reshape(-1, 64), labels[:, 1:].reshape(-1))
    (ref + 0.01 * aux.detach()).backward()
    torch.testing.assert_close(loss.detach(), (ref + 0.01 * aux).detach())
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(aux.grad, torch.tensor(0.01))
