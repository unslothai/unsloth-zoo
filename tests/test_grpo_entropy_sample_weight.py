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

"""Zero-variance exclusion (sample_weight == 0) must drop a completion from the entropy bonus too."""
import pytest
import torch

import unsloth_zoo.rl_replacements as rr


def _kwargs(sample_weight, loss_type = "dapo"):
    return dict(
        loss_type = loss_type, num_items_in_batch = torch.tensor(12.0), num_processes = 1,
        current_gradient_accumulation_steps = 1, steps_per_generation = 1, sample_weight = sample_weight,
    )


@pytest.mark.parametrize("trl_version", ["1.8.0", "1.15.0"])
def test_fully_excluded_batch_gets_no_entropy_gradient(trl_version):
    mask = torch.ones((2, 3), dtype = torch.long)
    logits = torch.randn(2, 3, 4, generator = torch.Generator().manual_seed(0), requires_grad = True)
    logp = logits.log_softmax(-1)
    entropies = -(logp.exp() * logp).sum(-1)
    kwargs = _kwargs(torch.zeros(2))
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.1, kwargs, trl_version = trl_version)
    bonus = rr.UnslothGRPOEntropyBonus.apply(entropies, weights, 1.0, None)
    bonus.backward()
    assert bonus.item() == 0.0
    assert logits.grad.abs().max().item() == 0.0


def test_partial_exclusion_matches_masked_mean():
    gen = torch.Generator().manual_seed(3)
    mask = (torch.rand(4, 5, generator = gen) > 0.2).long()
    entropies = torch.rand(4, 5, generator = gen, dtype = torch.float64)
    sample_weight = torch.tensor([0.0, 0.1, 0.0, 0.05])
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.3, _kwargs(sample_weight), trl_version = "1.15.0")
    kept = mask.double() * (sample_weight > 0).double().unsqueeze(-1)
    want = -0.3 * (entropies * kept).sum() / kept.sum().clamp(min = 1.0)
    torch.testing.assert_close((weights.double() * entropies).sum(), want)
    assert (weights[sample_weight == 0] == 0).all()


def test_no_sample_weight_is_unchanged():
    gen = torch.Generator().manual_seed(4)
    mask = (torch.rand(3, 6, generator = gen) > 0.3).long()
    kwargs = _kwargs(None)
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.2, kwargs, trl_version = "1.15.0")
    want = mask.to(torch.float32) * (-0.2 / mask.to(torch.float32).sum().clamp(min = 1.0) / 1)
    assert torch.equal(weights, want)


def test_excluded_rows_do_not_set_the_entropy_quantile():
    # High-entropy excluded completions would otherwise take the top quantile and then be dropped.
    seen = []

    class Trainer:
        def get_high_entropy_mask(self, entropies, mask, threshold):
            seen.append(mask.clone())
            return mask.bool()

    entropies = torch.rand(3, 4)
    mask = torch.ones(3, 4)
    sample_weight = torch.tensor([1.0, 0.0, 2.0])
    rr.grpo_high_entropy_mask(Trainer(), entropies, mask, 0.8, sample_weight = sample_weight)
    assert torch.equal(seen[-1], torch.tensor([[1.0] * 4, [0.0] * 4, [1.0] * 4]))
    rr.grpo_high_entropy_mask(Trainer(), entropies, mask, 0.8)
    assert torch.equal(seen[-1], mask)


@pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
def test_legacy_bonus_of_kept_rows_matches_running_them_alone(loss_type):
    # TRL 1.8 / 1.9 divide the bonus by the policy normaliser; excluded rows must not dilute it.
    gen = torch.Generator().manual_seed(5)
    mask = torch.ones((2, 6), dtype = torch.long)
    entropies = torch.rand(2, 6, generator = gen, dtype = torch.float64)
    # sample_weight as _unsloth_grpo_sample_weight builds it: 1 / kept tokens, 0 for excluded rows.
    kwargs = _kwargs(torch.tensor([1.0 / 6, 0.0]), loss_type)
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.2, kwargs, trl_version = "1.8.0")
    alone = dict(_kwargs(None, loss_type), num_items_in_batch = torch.tensor(6.0))
    want = rr.grpo_entropy_bonus_weights(mask[:1], None, 0.2, alone, trl_version = "1.8.0")
    torch.testing.assert_close((weights[:1].double() * entropies[:1]).sum(), (want.double() * entropies[:1]).sum())
    assert (weights[1] == 0).all()
    # No exclusion: identical to the num_items_in_batch normaliser.
    full = rr.grpo_entropy_bonus_weights(mask, None, 0.2, _kwargs(torch.full((2,), 1.0 / 12), loss_type), trl_version = "1.8.0")
    torch.testing.assert_close(full, rr.grpo_entropy_bonus_weights(mask, None, 0.2, _kwargs(None, loss_type), trl_version = "1.8.0"))
