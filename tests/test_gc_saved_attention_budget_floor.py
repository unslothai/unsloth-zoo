# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Automatic saved attention budget: step 1's peak is below step 2's (AdamW moments come after
activations are freed), so trusting it overcommitted step 2 past the card (full finetuning,
trained embeddings / lm_head); the per-step floor lifts the expected peak until a step with
moments is measured."""
import pathlib

import pytest
import torch

SA = pytest.importorskip("unsloth_zoo.saved_attention")

GiB = 1 << 30
CAPACITY = 40 * GiB
LAYERS = 32


def test_saved_attention_module_is_this_tree():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(SA.__file__).resolve() == root / "unsloth_zoo" / "saved_attention.py"


@pytest.fixture
def allocator(monkeypatch):
    sim = {"base": 0, "peak": 0, "allocated": 0}

    def live():
        return sum(b.live for b in SA._BUDGETS.values())

    def peak(device):
        sim["peak"] = max(sim["peak"], sim["base"] + live())
        return sim["peak"]

    monkeypatch.setattr(SA, "_device_peak", peak)
    monkeypatch.setattr(SA, "_device_capacity", lambda device: CAPACITY)
    monkeypatch.setattr(SA, "_device_allocated", lambda device: sim["allocated"] + live(), raising = False)
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [True])
    monkeypatch.setattr(SA, "_BUDGETS", {})
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
    SA.SAVED_ATTENTION_STATS.clear()
    return sim, peak


def _forward(sim, peak, query, estimate):
    """One step's forward: every layer asks to keep its attention output."""
    frames = []
    for _ in range(LAYERS):
        frame = SA.SavedAttentionFrame(auto = True)
        if SA._auto_admit(frame, [query]):
            budget = SA._BUDGETS[query.device]
            SA._observe(budget, query.device)
            budget.live += estimate
            frame.kept.append((query.device, estimate))
        frames.append(frame)
    return frames


def _backward(sim, peak, frames, query):
    peak(query.device)
    for frame in reversed(frames):
        SA._auto_release(frame, True)


def test_first_optimizer_step_does_not_overcommit_step_2(allocator):
    sim, peak = allocator
    # 32K tokens, 32 heads, head_dim 128, bf16: about 0.25 GiB kept per layer.
    query = torch.empty((1, 32, 32768, 128), dtype = torch.bfloat16, device = "meta")
    estimate = SA._bytes(query) + (query.numel() // query.shape[-1]) * 4
    model, grads, moments, activations = 8 * GiB, 8 * GiB, 16 * GiB, 12 * GiB

    # Step 1: model only at the start; activations to 20 GiB in the backward; nothing kept (unmeasured).
    sim["allocated"] = sim["base"] = model
    frames = _forward(sim, peak, query, estimate)
    sim["base"] = model + activations
    _backward(sim, peak, frames, query)
    assert SA._BUDGETS[query.device].live == 0
    # First optimizer.step: activations freed, grads + new AdamW moments: 32 GiB.
    sim["base"] = model + grads + moments
    peak(query.device)

    # Step 2 runs with the moments resident: its own peak is 36 GiB before anything is kept.
    sim["allocated"] = sim["base"] = model + moments
    frames = _forward(sim, peak, query, estimate)
    budget = SA._BUDGETS[query.device]
    step2_peak = model + moments + activations + budget.live
    margin = max(SA._AUTO_MARGIN_BYTES, int(CAPACITY * SA._AUTO_MARGIN_FRACTION))
    assert step2_peak <= CAPACITY - margin, (
        f"kept {budget.live / GiB:.2f} GiB in step 2: peak {step2_peak / GiB:.2f} of {CAPACITY / GiB:.0f} GiB"
    )
    sim["base"] = model + moments + activations
    _backward(sim, peak, frames, query)
    sim["base"] = model + grads + moments
    peak(query.device)

    # Step 3: step 2's peak has been measured with the moments; keeping resumes inside the budget.
    sim["allocated"] = sim["base"] = model + moments
    frames = _forward(sim, peak, query, estimate)
    budget = SA._BUDGETS[query.device]
    assert budget.live > 0
    assert model + moments + activations + budget.live <= CAPACITY - margin
    for frame in frames:
        SA._auto_release(frame, False)


def test_steady_floor_keeps_as_before(allocator):
    """No state appears between steps: the floor does not move and admission is unchanged."""
    sim, peak = allocator
    query = torch.empty((1, 32, 32768, 128), dtype = torch.bfloat16, device = "meta")
    estimate = SA._bytes(query) + (query.numel() // query.shape[-1]) * 4
    kept = []
    for _ in range(3):
        sim["allocated"] = sim["base"] = 8 * GiB
        frames = _forward(sim, peak, query, estimate)
        kept.append(SA._BUDGETS[query.device].live)
        sim["base"] = 20 * GiB
        _backward(sim, peak, frames, query)
    margin = max(SA._AUTO_MARGIN_BYTES, int(CAPACITY * SA._AUTO_MARGIN_FRACTION))
    assert kept[0] == 0  # unmeasured first step
    assert kept[1] == kept[2] > 0
    assert 20 * GiB + kept[1] <= CAPACITY - margin
