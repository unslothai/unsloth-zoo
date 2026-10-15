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

"""TRL top_entropy_quantile masking and the fixed entropy bonus against TRL's own _compute_loss.

TRL's GRPOTrainer._compute_loss runs unmodified on a stand-in trainer whose logprob pass returns
fixed tensors. Unsloth's side is grpo_compute_loss through UnslothEfficientGRPO plus
UnslothGRPOEntropyBonus, the way grpo_accumulated_loss assembles them. Loss values and the
gradients reaching the logprobs and the entropies must match in float64.
"""
import inspect
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

trl_grpo = pytest.importorskip("trl.trainer.grpo_trainer")
GRPOTrainer = trl_grpo.GRPOTrainer
HAS_BONUS = "entropy_coef" in {f.name for f in __import__("dataclasses").fields(trl_grpo.GRPOConfig)}


class _Accelerator:
    num_processes = 1
    sync_gradients = True
    scaler = None

    def pad_across_processes(self, tensor, dim = 0, pad_index = 0):
        return tensor

    def gather(self, tensor):
        return tensor

    def gather_for_metrics(self, tensor):
        return tensor

    def reduce(self, tensor, reduction = "sum"):
        return tensor


def _trl_trainer(quantile, coef, loss_type, beta, logps, entropies, ga):
    trainer = SimpleNamespace()
    trainer.accelerator = _Accelerator()
    trainer.top_entropy_quantile = quantile
    trainer.entropy_coef = coef
    trainer.use_adaptive_entropy = False
    trainer._entropy_bonus_enabled = coef != 0.0
    trainer.loss_type = loss_type
    trainer.beta = beta
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.28
    trainer.importance_sampling_level = "token"
    trainer.off_policy_mask_threshold = None
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer.aux_loss_enabled = False
    trainer.max_completion_length = logps.shape[1]
    trainer.current_gradient_accumulation_steps = ga
    trainer.model = SimpleNamespace(training = True)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.args = SimpleNamespace(
        delta = None, use_bias_correction_kl = False, steps_per_generation = ga,
        top_entropy_quantile = quantile, entropy_coef = coef, use_adaptive_entropy = False,
    )
    # TRL < 1.7 returns (logps, entropies), later ones add the MoE aux loss.
    returned = (logps, entropies, None) if "aux_loss" in inspect.getsource(GRPOTrainer._compute_loss) else (logps, entropies)
    trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: returned
    trainer.get_high_entropy_mask = GRPOTrainer.get_high_entropy_mask.__get__(trainer)
    return trainer


def _inputs(rows, cols, seed):
    gen = torch.Generator().manual_seed(seed)
    logps = (-torch.rand(rows, cols, generator = gen, dtype = torch.float64) * 3).requires_grad_(True)
    old = logps.detach() + torch.randn(rows, cols, generator = gen, dtype = torch.float64) * 0.1
    ref = logps.detach() + torch.randn(rows, cols, generator = gen, dtype = torch.float64) * 0.1
    # Rounded so the quantile threshold lands on ties.
    entropies = (torch.rand(rows, cols, generator = gen, dtype = torch.float64) * 4).round(decimals = 1).requires_grad_(True)
    mask = (torch.rand(rows, cols, generator = gen) > 0.25).long()
    mask[0] = 0  # a fully masked row
    advantages = torch.randn(rows, generator = gen, dtype = torch.float64)
    return logps, old, ref, entropies, mask, advantages


def _trl(quantile, coef, loss_type, beta, ga, data):
    logps, old, ref, entropies, mask, advantages = data
    trainer = _trl_trainer(quantile, coef, loss_type, beta, logps, entropies, ga)
    rows, cols = logps.shape
    inputs = {
        "prompt_ids": torch.zeros(rows, 1, dtype = torch.long),
        "prompt_mask": torch.ones(rows, 1, dtype = torch.long),
        "completion_ids": torch.zeros(rows, cols, dtype = torch.long),
        "completion_mask": mask,
        "advantages": advantages,
        "old_per_token_logps": old,
        "ref_per_token_logps": ref,
        "num_items_in_batch": mask.sum(),
    }
    loss = GRPOTrainer._compute_loss(trainer, None, inputs)
    return loss, torch.autograd.grad(loss, [logps, entropies], allow_unused = True)


def _unsloth(quantile, coef, loss_type, beta, ga, data):
    logps, old, ref, entropies, mask, advantages = data
    trainer = _trl_trainer(quantile, coef, loss_type, beta, logps, entropies, ga)
    kwargs = dict(
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28,
        max_completion_length = logps.shape[1], current_gradient_accumulation_steps = ga,
        num_items_in_batch = mask.sum(), steps_per_generation = ga, num_processes = 1,
        importance_sampling_level = "token",
    )
    entropy_mask = None
    if quantile < 1.0:
        entropy_mask = trainer.get_high_entropy_mask(entropies.detach(), mask, 1 - quantile)
        kwargs["entropy_mask"] = entropy_mask
    loss, *_ = rr.grpo_compute_loss(ref, logps, old, None, torch.zeros_like(mask), mask, beta, advantages, **kwargs)
    if coef != 0.0:
        weights = rr.grpo_entropy_bonus_weights(mask, entropy_mask, coef, kwargs).to(torch.float64)
        loss = loss + rr.UnslothGRPOEntropyBonus.apply(entropies, weights, 1.0, None)
    return loss, torch.autograd.grad(loss, [logps, entropies], allow_unused = True)


CASES = [
    (0.2, 0.0, "grpo", 0.0, 1),
    (0.2, 0.0, "dapo", 0.04, 2),
    (0.0, 0.0, "bnpo", 0.0, 1),
    (0.5, 0.0, "dr_grpo", 0.04, 4),
]
BONUS_CASES = [
    (1.0, 0.01, "grpo", 0.0, 1),
    (1.0, 0.3, "dapo", 0.04, 2),
    (0.2, 0.05, "grpo", 0.04, 4),
    (0.0, 0.1, "bnpo", 0.0, 1),
]


def _check(case, seed):
    data = _inputs(rows = 5, cols = 9, seed = seed)
    want_loss, want_grads = _trl(*case, data)
    got_loss, got_grads = _unsloth(*case, data)
    torch.testing.assert_close(got_loss.to(torch.float64), want_loss.to(torch.float64), rtol = 1e-6, atol = 1e-9)
    for got, want in zip(got_grads, want_grads):
        if want is None:
            assert got is None or float(got.abs().max()) == 0.0
            continue
        torch.testing.assert_close(got.to(torch.float64), want, rtol = 1e-6, atol = 1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_entropy_mask_matches_trl(case, seed):
    _check(case, seed)


@pytest.mark.skipif(not HAS_BONUS, reason = "this TRL has no entropy bonus (TRL < 1.8)")
@pytest.mark.parametrize("case", BONUS_CASES)
@pytest.mark.parametrize("seed", [0, 3])
def test_entropy_bonus_matches_trl(case, seed):
    _check(case, seed)


@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "cispo", "bnpo"])
def test_bonus_weights_follow_trl_1_8_and_1_9_for_token_sum_losses(loss_type):
    # TRL 1.8.0 / 1.9.0 GRPOTrainer._compute_loss, transcribed: cispo / dapo / vespo divided the
    # entropy sum by the policy normalizer; the rest took the masked mean over the GA steps.
    gen = torch.Generator().manual_seed(11)
    mask = (torch.rand(4, 6, generator = gen) > 0.3).long()
    entropy_mask = torch.rand(4, 6, generator = gen) > 0.5
    entropies = torch.rand(4, 6, generator = gen, dtype = torch.float64)
    kwargs = dict(
        loss_type = loss_type, num_items_in_batch = torch.tensor(40.0), num_processes = 2,
        steps_per_generation = 4, current_gradient_accumulation_steps = 2,
    )
    effective = (mask * entropy_mask).double()
    if loss_type in ("cispo", "dapo"):
        normalizer = 40.0 / 2
        want = (entropies * effective).sum() / normalizer
    else:
        want = (entropies * effective).sum() / effective.sum().clamp(min = 1.0) / 2
    for version in ("1.8.0", "1.9.0"):
        weights = rr.grpo_entropy_bonus_weights(mask, entropy_mask, 0.3, kwargs, trl_version = version)
        torch.testing.assert_close((weights.double() * entropies).sum(), -0.3 * want)
    weights = rr.grpo_entropy_bonus_weights(mask, entropy_mask, 0.3, kwargs, trl_version = "1.10.0")
    torch.testing.assert_close(
        (weights.double() * entropies).sum(),
        -0.3 * (entropies * effective).sum() / effective.sum().clamp(min = 1.0) / 2,
    )


def test_bonus_weights_do_not_divide_by_accumulation_in_eval():
    # TRL scales the bonus by current_gradient_accumulation_steps only when model.training.
    gen = torch.Generator().manual_seed(5)
    mask = (torch.rand(3, 7, generator = gen) > 0.3).long()
    for trl_version in ("1.10.0", "1.8.0"):
        kwargs = dict(loss_type = "grpo", current_gradient_accumulation_steps = 4, num_items_in_batch = torch.tensor(9.0))
        train = rr.grpo_entropy_bonus_weights(mask, None, 0.2, kwargs, trl_version = trl_version)
        evaluation = rr.grpo_entropy_bonus_weights(mask, None, 0.2, kwargs, trl_version = trl_version, training = False)
        torch.testing.assert_close(evaluation, train * 4)
        torch.testing.assert_close(
            evaluation.sum(), torch.tensor(-0.2, dtype = evaluation.dtype),
        )
    source = inspect.getsource(rr.grpo_accumulated_loss)
    assert "training = _training" in source


def test_bonus_gradient_ignores_grad_output_like_the_policy_term():
    entropies = torch.rand(2, 3, dtype = torch.float64, requires_grad = True)
    weights = torch.rand(2, 3, dtype = torch.float64)
    out = rr.UnslothGRPOEntropyBonus.apply(entropies, weights, 4.0, None)
    (grad,) = torch.autograd.grad(out * 7.0, [entropies])
    torch.testing.assert_close(grad, weights * 4.0)
    out = rr.UnslothGRPOEntropyBonus.apply(entropies, weights, 1.0, 3.0)
    (grad,) = torch.autograd.grad(out * 0.5, [entropies])
    torch.testing.assert_close(grad, weights * 1.5)


def test_grpo_accumulated_loss_wires_mask_and_bonus():
    source = inspect.getsource(rr.grpo_accumulated_loss)
    assert "trainer.get_high_entropy_mask(new_entropies.detach(), completion_mask, 1 - _entropy_quantile)" in source
    assert 'kwargs["entropy_mask"] = entropy_mask' in source
    assert "_UnslothGRPOEntropyBonus.apply(" in source
    # PrefixGrouper returns no entropies, so it must stand down when they are needed.
    assert "and not _need_entropy" in source
    assert "entropy_mask = kwargs.get(\"entropy_mask\", None)" in inspect.getsource(rr.grpo_compute_loss)


def test_quantile_and_fixed_bonus_no_longer_warn(caplog):
    import logging
    trainer = SimpleNamespace(
        args = SimpleNamespace(top_entropy_quantile = 0.2, entropy_coef = 0.1, use_adaptive_entropy = False),
        get_high_entropy_mask = lambda *args: None,
    )
    with caplog.at_level(logging.WARNING, logger = "unsloth_zoo.log"):
        rr._warn_unsupported_grpo_options(trainer)
    assert caplog.records == []
