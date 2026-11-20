# SPDX-License-Identifier: AGPL-3.0-or-later
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""GRPO num_iterations > 1: the first loss pass hands its detached logps back as pi_old (ratio exactly 1),
and the later passes, given that pi_old, match TRL's loss on a separately scored pi_old."""

import sys
from pathlib import Path

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_zero_advantage_skip as h  # noqa: E402


_ADV = torch.tensor([0.5, -0.5, 1.5, -1.5, 0.2, -0.2])
CASES = [
    ("grpo", "token", 0.0),
    ("bnpo", "token", 0.0),
    ("dr_grpo", "token", 0.0),
    ("dapo", "token", 0.0),
    ("dapo", "token", 0.04),
    ("grpo", "token", 0.04),
    ("dapo", "sequence", 0.0),  # GSPO
    ("grpo", "sequence", 0.04),
]


@pytest.fixture(autouse = True)
def _log_prob_route(monkeypatch):
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)


def _call(monkeypatch, model, adv, old, store, loss_type, level, beta, skip, is_mode = "token_truncate", vllm_is = True):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "1" if skip else "0")
    model.zero_grad(set_to_none = True)
    # Scored first: _score_old swaps UnslothEfficientGRPO.apply itself.
    ref = h._score_old(h._Model()) - 0.1 if beta else None
    trainer = h._trainer(model, beta = beta, is_mode = is_mode, use_vllm = vllm_is, correction = vllm_is)
    seen = []

    def spy(new, *args):
        seen.append(new.detach().clone())
        return h._APPLY(new, *args)
    monkeypatch.setattr(rr.UnslothEfficientGRPO, "apply", spy)
    mask = (h._IDS[:, -h._COMPLETION:] != 0).long()
    kwargs = dict(
        loss_type = loss_type, importance_sampling_level = level, sampling_per_token_logps = h._sampling(),
        max_completion_length = 8, num_items_in_batch = mask.sum().float(), current_gradient_accumulation_steps = 2,
        epsilon_low = h.EPS_LOW, epsilon_high = h.EPS_HIGH,
    )
    if store:
        kwargs.update(old_logps_from_new = True, store_old_logps = True)
    out = rr.grpo_accumulated_loss(trainer, h._IDS, (h._IDS != 0).long(), h._COMPLETION, mask, adv, old, ref, **kwargs)
    out[0].backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return out, grads, seen, trainer


def _step(model, scale = 0.05):
    """Stands in for the optimizer step between iterations: the policy moves, pi_old must not."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator = g, dtype = p.dtype))


@pytest.mark.parametrize("loss_type, level, beta", CASES)
def test_first_pass_ratio_is_exactly_one_and_pi_old_is_kept(monkeypatch, loss_type, level, beta):
    model = h._Model()
    out, grads, seen, trainer = _call(monkeypatch, model, _ADV, None, True, loss_type, level, beta, skip = False)
    stored = trainer._unsloth_grpo_stored_old_logps
    (new,) = seen
    assert not stored.requires_grad and stored.shape == new.shape
    assert torch.equal(stored, new)
    mask = out[6].bool()
    coef_1 = out[5]
    assert torch.equal(coef_1[mask] if level == "token" else coef_1, torch.ones_like(coef_1[mask] if level == "token" else coef_1))
    # Same loss and gradient as TRL's iteration 1, whose separately scored pi_old equals pi_theta.
    ref, ref_grads, _, _ = _call(monkeypatch, h._Model(), _ADV, h._score_old(h._Model()), False, loss_type, level, beta, skip = False)
    torch.testing.assert_close(out[0], ref[0], rtol = 1e-6, atol = 1e-9)
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-5, atol = 1e-9)


@pytest.mark.parametrize("loss_type, level, beta", CASES)
def test_later_pass_uses_the_kept_pi_old(monkeypatch, loss_type, level, beta):
    model = h._Model()
    _, _, _, trainer = _call(monkeypatch, model, _ADV, None, True, loss_type, level, beta, skip = False)
    stored = trainer._unsloth_grpo_stored_old_logps
    pi_old_scored = h._score_old(h._Model())   # TRL's separate no-grad forward of the rollout weights
    _step(model)
    head, head_grads, seen, trainer2 = _call(monkeypatch, model, _ADV, stored, False, loss_type, level, beta, skip = False)
    assert not hasattr(trainer2, "_unsloth_grpo_stored_old_logps")
    moved = h._Model()
    moved.load_state_dict(model.state_dict())
    base, base_grads, _, _ = _call(monkeypatch, moved, _ADV, pi_old_scored, False, loss_type, level, beta, skip = False)
    torch.testing.assert_close(head[0], base[0], rtol = 1e-6, atol = 1e-9)
    for name in head_grads:
        torch.testing.assert_close(head_grads[name], base_grads[name], rtol = 1e-5, atol = 1e-9)
    # The ratio is pi_theta / kept pi_old, not 1: the iteration actually corrects.
    (new,) = seen
    mask = head[6].bool()
    if level == "token":
        torch.testing.assert_close(head[5][mask], torch.exp(new - stored)[mask].to(head[5].dtype))
        assert (head[5][mask] - 1).abs().max() > 1e-4


def test_later_pass_matches_the_float64_trl_oracle(monkeypatch):
    """Iteration 2 on the kept pi_old vs TRL 1.18 _compute_loss written out in float64."""
    model = h._Model()
    _, _, _, trainer = _call(monkeypatch, model, _ADV, None, True, "dapo", "token", 0.0, skip = False)
    stored = trainer._unsloth_grpo_stored_old_logps
    _step(model)
    head, _, seen, _ = _call(monkeypatch, model, _ADV, stored, False, "dapo", "token", 0.0, skip = False)
    (new,) = seen
    mask = head[6].to(torch.float64)
    sampling = rr.align_logprobs_with_mask(h._sampling(), head[6]) if hasattr(rr, "align_logprobs_with_mask") else h._sampling()
    want = h._trl_loss(
        new.to(torch.float64), stored.to(torch.float64), sampling.to(torch.float64), _ADV.to(torch.float64), mask,
        "dapo", "token", "token_truncate", ga = 2, num_items = mask.sum(), spg = 2,
    )
    torch.testing.assert_close(head[0].to(torch.float64), want, rtol = 2e-6, atol = 1e-7)


def test_skipped_zero_advantage_rows_keep_no_pi_old(monkeypatch):
    adv = h._ADV   # rows 0, 1, 4, 5 have zero advantage and are not scored
    model = h._Model()
    _, _, seen, trainer = _call(monkeypatch, model, adv, None, True, "dapo", "token", 0.0, skip = True)
    stored = trainer._unsloth_grpo_stored_old_logps
    assert stored.shape[0] == 6
    kept, dropped = torch.tensor([2, 3]), torch.tensor([0, 1, 4, 5])
    assert torch.isnan(stored[dropped]).all()
    assert torch.equal(stored[kept], seen[0])
    pi_old_scored = h._score_old(h._Model())
    _step(model)
    head, head_grads, _, _ = _call(monkeypatch, model, adv, stored, False, "dapo", "token", 0.0, skip = True)
    moved = h._Model()
    moved.load_state_dict(model.state_dict())
    base, base_grads, _, _ = _call(monkeypatch, moved, adv, pi_old_scored, False, "dapo", "token", 0.0, skip = False)
    assert torch.isfinite(head[0])
    torch.testing.assert_close(head[0], base[0], rtol = 1e-6, atol = 1e-9)
    for name in head_grads:
        assert torch.isfinite(head_grads[name]).all()
        torch.testing.assert_close(head_grads[name], base_grads[name], rtol = 1e-5, atol = 1e-9)
    # IS metrics: the dropped rows stay unscored (NaN), as on the first pass.
    assert torch.isnan(head[3][dropped]).all() and torch.isnan(head[4][dropped]).all()
    torch.testing.assert_close(head[3][kept], base[3][kept], rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("case", ["no_flag", "old_given"])
def test_nothing_is_kept_unless_asked_on_a_from_new_pass(monkeypatch, case):
    model = h._Model()
    old = h._score_old(h._Model()) if case == "old_given" else None
    store = case != "no_flag"
    # Without the flag pi_old None is TRL's aligned single-iteration path (IS off), unchanged.
    _, _, _, trainer = _call(monkeypatch, model, _ADV, old, store, "dapo", "token", 0.0, skip = False,
                             vllm_is = case != "no_flag")
    assert not hasattr(trainer, "_unsloth_grpo_stored_old_logps")


def test_store_helper():
    old = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad = True)
    assert not rr.grpo_old_logps_store(old).requires_grad
    assert torch.equal(rr.grpo_old_logps_store(old), old.detach())
    full = rr.grpo_old_logps_store(old, torch.tensor([1, 3]), 4)
    assert full.shape == (4, 2)
    assert torch.equal(full[[1, 3]], old.detach())
    assert torch.isnan(full[[0, 2]]).all()
