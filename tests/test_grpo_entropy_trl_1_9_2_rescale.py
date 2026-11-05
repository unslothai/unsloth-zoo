"""TRL 1.9.1 and 1.9.2 (trl#6024) divide the cispo / dapo / vespo entropy term by the policy normalizer rescaled to one accumulation
window (num_items_in_batch / num_processes * GA / steps_per_generation); 1.8.0, 1.9.0 and eval do not rescale."""
import pytest
import torch

import unsloth_zoo.rl_replacements as rr


def _setup(loss_type, ga, spg):
    gen = torch.Generator().manual_seed(11)
    mask = (torch.rand(4, 6, generator = gen) > 0.3).long()
    entropies = torch.rand(4, 6, generator = gen, dtype = torch.float64)
    kwargs = dict(
        loss_type = loss_type, num_items_in_batch = torch.tensor(40.0), num_processes = 2,
        steps_per_generation = spg, current_gradient_accumulation_steps = ga,
    )
    return mask, entropies, kwargs


@pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
@pytest.mark.parametrize("ga, spg", [(2, 4), (4, 2), (1, 8), (3, 3)])
@pytest.mark.parametrize("version", ["1.9.1", "1.9.2"])
def test_bonus_weights_rescale_to_window_on_trl_1_9_2(loss_type, ga, spg, version):
    mask, entropies, kwargs = _setup(loss_type, ga, spg)
    # TRL 1.9.1 / 1.9.2 grpo_trainer.py L3141-3144 and L3171, transcribed.
    normalizer = 40.0 / 2 * ga / spg
    want = -0.3 * (entropies * mask).sum() / normalizer
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.3, kwargs, trl_version = version)
    torch.testing.assert_close((weights.double() * entropies).sum(), want)


@pytest.mark.parametrize("version", ["1.8.0", "1.9.0"])
def test_bonus_weights_keep_no_rescale_before_1_9_1(version):
    mask, entropies, kwargs = _setup("dapo", 2, 4)
    want = -0.3 * (entropies * mask).sum() / (40.0 / 2)
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.3, kwargs, trl_version = version)
    torch.testing.assert_close((weights.double() * entropies).sum(), want)


def test_bonus_weights_do_not_rescale_in_eval_on_trl_1_9_2():
    mask, entropies, kwargs = _setup("dapo", 2, 4)
    kwargs["steps_per_generation"] = None  # grpo_accumulated_loss passes None outside training
    want = -0.3 * (entropies * mask).sum() / (40.0 / 2)
    weights = rr.grpo_entropy_bonus_weights(mask, None, 0.3, kwargs, trl_version = "1.9.2", training = False)
    torch.testing.assert_close((weights.double() * entropies).sum(), want)


def test_sample_weight_branch_matches_policy_rescale_on_trl_1_9_2():
    mask, entropies, kwargs = _setup("dapo", 2, 4)
    sample_weight = torch.full((4,), 1.0 / 40.0)
    kwargs["sample_weight"] = sample_weight
    plain = {k: v for k, v in kwargs.items() if k != "sample_weight"}
    got = rr.grpo_entropy_bonus_weights(mask, None, 0.3, kwargs, trl_version = "1.9.2")
    want = rr.grpo_entropy_bonus_weights(mask, None, 0.3, plain, trl_version = "1.9.2")
    torch.testing.assert_close(got.double(), want.double())
