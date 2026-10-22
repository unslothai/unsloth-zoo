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

"""A batch split over GPUs and gradient accumulation must train exactly like the whole batch on one GPU."""

import pytest
import torch

from unsloth_zoo.rl_replacements import grpo_compute_loss

WORLD, GA, ROWS, T, H, V = 2, 2, 2, 7, 5, 11
N = WORLD * GA * ROWS
LENGTHS = [7, 2, 5, 0, 6, 1, 7, 3]  # uneven token counts per micro-batch, one fully masked row

g = torch.Generator().manual_seed(0)
X = torch.randn(N, T, H, generator = g, dtype = torch.float64)
IDS = torch.randint(0, V, (N, T), generator = g)
MASK = (torch.arange(T)[None] < torch.tensor(LENGTHS)[:, None]).double()
ADV = torch.randn(N, generator = g, dtype = torch.float64)
OLD = 0.3 * torch.randn(N, T, generator = g, dtype = torch.float64)
REF = 0.2 * torch.randn(N, T, generator = g, dtype = torch.float64)
W0 = 0.5 * torch.randn(V, H, generator = g, dtype = torch.float64)


def _grad(rows, loss_type, epsilon_high, accumulation, processes):
    w = torch.nn.Parameter(W0.clone())
    new = torch.log_softmax(X[rows] @ w.T, -1).gather(-1, IDS[rows].unsqueeze(-1)).squeeze(-1)
    loss, *_ = grpo_compute_loss(
        new.detach() + REF[rows], new, new.detach() + OLD[rows], None, IDS[rows], MASK[rows], 0.04, ADV[rows],
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = epsilon_high, max_completion_length = T,
        num_items_in_batch = MASK.sum(),  # TRL gathers it over every rank and the whole generation batch
        num_processes = processes, current_gradient_accumulation_steps = accumulation,
        steps_per_generation = accumulation,
    )
    loss.backward()
    return w.grad


@pytest.mark.parametrize("loss_type, epsilon_high", [
    ("grpo", 0.2), ("bnpo", 0.2), ("dr_grpo", 0.2), ("dapo", 0.28), ("sapo", 0.2), ("luspo", 0.2), ("cispo", 5.0),
])
def test_split_batch_matches_one_gpu(loss_type, epsilon_high):
    whole = _grad(list(range(N)), loss_type, epsilon_high, accumulation = 1, processes = 1)
    per_rank = []
    for rank in range(WORLD):
        micro = [list(range((rank * GA + m) * ROWS, (rank * GA + m + 1) * ROWS)) for m in range(GA)]
        per_rank.append(sum(_grad(rows, loss_type, epsilon_high, GA, WORLD) for rows in micro))
    split = torch.stack(per_rank).mean(0)  # DDP averages gradients over ranks
    torch.testing.assert_close(split, whole, rtol = 1e-12, atol = 1e-14)


def test_bnpo_without_global_token_count_keeps_the_micro_batch_mean():
    w = torch.nn.Parameter(W0.clone())
    new = torch.log_softmax(X[:2] @ w.T, -1).gather(-1, IDS[:2].unsqueeze(-1)).squeeze(-1)
    loss, *_ = grpo_compute_loss(
        new.detach() + REF[:2], new, new.detach(), None, IDS[:2], MASK[:2], 0.0, ADV[:2],
        loss_type = "bnpo", num_items_in_batch = None, current_gradient_accumulation_steps = 2,
    )
    per_token = -ADV[:2, None] * torch.ones_like(new)  # ratio 1 at old = new, beta 0
    expected = (per_token * MASK[:2]).sum() / MASK[:2].sum() / 2
    assert loss.item() == pytest.approx(expected.item(), rel = 1e-12)
