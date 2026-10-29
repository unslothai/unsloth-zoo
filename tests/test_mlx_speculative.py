# Unsloth Zoo - Utilities for Unsloth
# Pin MLXTrainer's batch padding to match mlx-lm's iterate_batches:
# `1 + _PAD_MULTIPLE * ceil(L / _PAD_MULTIPLE)`.
#
# mlx-lm (mlx_lm/tuner/trainer.py:158) pads to `1 + 32*ceil(max_len/32)`;
# default_loss then slices [:, :-1] / [:, 1:]. unsloth_zoo's
# create_text_batches previously dropped the `+1`, leaving the input one
# token short after the autoregressive shift, which shifted small fixtures
# into a different convergence basin (probe 31 = 67% vs probe 33-37 = 40-53%,
# see danielhanchen/unsloth-staging-2).

import random
from collections import Counter
from math import prod
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
from unsloth_zoo.mlx.speculative import (
    DraftController, NgramProposer, RoundPlan, RowPlan, RowState, _Calibration, _CycleCosts, _Ema, _upper,
)


def test_ngram_prefers_longer_then_newer_match():
    prompt = [9, 1, 2, 3, 4, 50, 7, 1, 2, 3, 4, 60, 8, 1, 2, 3, 4, 70]
    proposer = NgramProposer(prompt, min_ngram = 4)
    assert proposer.propose([5, 1, 2, 3, 4], 2) == [70]
    assert proposer.propose([7, 1, 2, 3, 4], 3) == [60, 8, 1]
    assert proposer.propose([2, 3, 4], 3) == []
    assert proposer.propose([7, 1, 2, 3, 4], 0) == []
    assert NgramProposer([1, 2, 3, 4], min_ngram = 4).propose([1, 2, 3, 4], 5) == []


class _Machine:
    """Synthetic costs: batching a plain step is nearly free, verify width costs per row."""

    def __init__(self, *, step = 1.0, draft_slope = 0.2, verify_slope = 0.03, overhead = 1.2):
        self.step, self.draft_slope, self.verify_slope, self.overhead = step, draft_slope, verify_slope, overhead

    def plain(self, batch):
        return self.step * (1 + 0.15 * (batch - 1))

    def verify(self, batch, width):
        return self.plain(batch) * self.overhead + self.step * self.verify_slope * (width - 1) * batch

    def draft(self, depth):
        return self.step * self.draft_slope * depth


def _run(controller, machine, rows, *, rounds, seed = 0, states = None, report = None, copies = 12):
    """``rows``: per row (draft rates, copy rate or None). Returns the choice emitting most recent tokens."""
    rng = random.Random(seed)
    states = states if states is not None else []
    states[:] = [
        RowState(controller.new_reply(), can_draft = bool(draft), copy_available = copies if copy else 0)
        for draft, copy in rows
    ]
    chosen, emitted_total, seconds_total = [], 0, 0.0
    for index in range(rounds):
        plan = controller.plan(states)
        if plan.kind == "plain":
            seconds, emitted, choice = machine.plain(len(rows)) * plan.length, plan.length * len(rows), "plain"
            controller.record_plain(states, plan.length, seconds, machine.plain(len(rows)))
        else:
            accepted, drafting = [], 0.0
            for row, (draft, copy) in zip(plan.rows, rows):
                rates = draft if row.source == "draft" else [copy or 0.0] * row.length
                count = 0
                while count < row.length and rng.random() < rates[count]:
                    count += 1
                accepted.append(count)
                if row.source == "draft":
                    drafting += machine.draft(row.length)
            # A fused round overlaps drafting with building the verify; a split one does not.
            seconds = machine.verify(len(rows), plan.width) + drafting * (1.0 if plan.split else 0.8)
            controller.record_round(plan, states, accepted, seconds = seconds, draft_seconds = drafting)
            emitted, choice = sum(accepted) + len(rows), tuple((row.source, row.length) for row in plan.rows)
        if index >= rounds // 2:
            emitted_total, seconds_total = emitted_total + emitted, seconds_total + seconds
        chosen.append((choice, emitted))
    if report is not None:
        report["rate"], report["choices"] = emitted_total / seconds_total, {choice for choice, _ in chosen}
    return sum((Counter({choice: emitted}) for choice, emitted in chosen[-200:]), Counter()).most_common(1)[0][0]


def _throughput(choice, draft, machine):
    if choice == "plain":
        return 1.0 / machine.plain(1)
    ((_, depth),) = choice
    expected = sum(prod(draft[:k]) for k in range(depth + 1))
    return expected / (machine.verify(1, depth + 1) + machine.draft(depth))


@pytest.mark.parametrize(
    "draft, slope",
    [([0.9, 0.7, 0.3, 0.2], 0.2), ([0.95, 0.9, 0.85, 0.8], 0.1), ([0.3, 0.2, 0.1, 0.1], 0.6)],
)
def test_controller_finds_best_depth_including_plain(draft, slope):
    machine = _Machine(draft_slope = slope)
    best = max(["plain"] + [(("draft", d),) for d in range(1, 5)], key = lambda c: _throughput(c, draft, machine))
    for seed in range(3):
        choice = _run(DraftController(max_depth = 4, can_copy = False), machine, [(draft, None)],
                      rounds = 1500, seed = seed, report = (report := {}))
        assert _throughput(choice, draft, machine) >= 0.95 * _throughput(best, draft, machine)
        # Probes and split rounds included, the second half runs near the optimum.
        assert report["rate"] >= 0.9 * _throughput(best, draft, machine)


def test_controller_prefers_plain_decoding_over_width_one_rounds():
    # Depth 1 beats a one-token round (1.2 + 0.3) but loses to pipelined decoding.
    machine = _Machine(step = 0.02, draft_slope = 0.3, overhead = 1.2)
    controller = DraftController(max_depth = 1, can_copy = False)
    state = [RowState(controller.new_reply())]
    controller.record_round(controller.plan(state), state, [0], seconds = 0.03, draft_seconds = 0.006)
    assert controller.plan(state).kind == "plain"
    assert _run(controller, machine, [([0.45], None)], rounds = 800) == "plain"
    hopeless, states = DraftController(max_depth = 1, can_copy = False), []
    _run(hopeless, machine, [([0.05], None)], rounds = 300, states = states)
    assert hopeless.plan(states).length == hopeless.max_window


def test_a_round_must_clearly_beat_plain_decoding_at_its_running_step():
    for margin, choice in ((1.0, (("draft", 1),)), (1.05, "plain")):
        assert _run(controller := DraftController(max_depth = 1, can_copy = False, plain_margin = margin), _Machine(draft_slope = 0.8), [([1.0], None)], rounds = 800) == choice
    for window in ((8, 80.0, 2.0), (1, 9.0)):
        controller.record_plain([RowState(controller.new_reply())], *window)
    assert controller.plain_cost[1].value == pytest.approx(1.0 + controller.cost_alpha)
    assert (cold := DraftController(max_depth = 1, can_copy = False)).record_plain([RowState(cold.new_reply())], 1, 9.0) is None and not cold.plain_cost and (short := DraftController(max_depth = 0, can_copy = False, min_window = 1)).plan([RowState(short.new_reply())]).length == 3


def test_controller_follows_cost_drift():
    draft = [0.9, 0.7, 0.4, 0.2]
    controller = DraftController(max_depth = 4, can_copy = False)
    ((source, depth),) = _run(controller, _Machine(draft_slope = 0.05), [(draft, None)], rounds = 800)
    assert source == "draft" and depth >= 2
    assert _run(controller, _Machine(draft_slope = 0.9), [(draft, None)], rounds = 1500, seed = 1) == "plain"


def test_controller_copies_only_while_copies_hold():
    machine = _Machine(draft_slope = 0.3)
    ((source, length),) = _run(DraftController(max_depth = 2, max_copy = 12), machine, [([0.6, 0.3], 0.97)], rounds = 1500)
    assert source == "copy" and length >= 8
    poor, states = DraftController(max_depth = 2, max_copy = 12), []
    choice = _run(poor, machine, [([0.6, 0.3], 0.05)], rounds = 1500, states = states)
    assert choice == "plain" or choice[0][0] != "copy"
    assert poor.copy_acceptance.value < 0.3
    assert states[0].stats.backoff["copy"] > poor._base_backoff


def test_controller_prefers_free_copy_over_costlier_drafter():
    # The drafter accepts slightly more, but its forward makes the round slower.
    machine = _Machine(draft_slope = 0.8)
    choice = _run(DraftController(max_depth = 4, max_copy = 12), machine, [([0.85] * 4, 0.8)], rounds = 1500)
    assert choice != "plain" and choice[0][0] == "copy"


def test_controller_drafts_alone_but_not_in_a_wide_batch():
    machine = _Machine(draft_slope = 0.1, verify_slope = 0.12)
    controller = DraftController(max_depth = 3, can_copy = False)
    alone = _run(controller, machine, [([0.8, 0.6, 0.4], None)], rounds = 1000)
    assert alone == (("draft", 2),)
    assert _run(controller, machine, [([0.8, 0.6, 0.4], None)] * 8, rounds = 1000, seed = 1) == "plain"
    assert _run(fixed := DraftController(max_depth = 3, can_copy = False, fixed_depth = True), machine, [([0.8, 0.6, 0.4], None)] * 8, rounds = 1000, seed = 1, report = (report := {})) == (("draft", 3),) * 8 and "plain" not in report["choices"]
    assert _run(DraftController(max_depth = 3, fixed_depth = True), machine, [([0.8, 0.6, 0.4], 0.95), ([0.8, 0.6, 0.4], None), ([0.8, 0.6, 0.4], 0.1)], rounds = 200) == (("copy", 12), ("draft", 3), ("draft", 3))
    assert [controller.plain_cost[b].value for b in (1, 8)] == pytest.approx([machine.plain(1), machine.plain(8)])


def test_a_draft_probe_shrinks_to_the_deepest_depth_the_credit_affords():
    controller = DraftController(max_depth = 8, can_copy = False)
    _run(controller, _Machine(draft_slope = 0.3, verify_slope = 0.3, overhead = 1.7), [([0.5, 0.2] + [0.1] * 6, None)], rounds = 400)
    controller._credit = 1.5
    assert controller.plan([RowState(controller.new_reply())]).rows == (RowPlan("draft", 2),)


def test_fused_rounds_learn_how_much_drafting_they_hide():
    controller = DraftController(max_depth = 2, can_copy = False, split_every = 4)
    state = [RowState(controller.new_reply())]
    plan = RoundPlan("round", rows = (RowPlan("draft", 2),), split = True)
    controller.record_round(plan, state, [1], seconds = 1.5, draft_seconds = 0.5)
    fused = RoundPlan("round", rows = (RowPlan("draft", 2),))
    for _ in range(40):
        controller.record_round(fused, state, [1], seconds = 1.2)
    assert (controller.verify_cost[1][3].value, controller.draft_cost[2].value) == pytest.approx((1.0, 0.5))
    assert controller.round_seconds(fused, state) == pytest.approx(1.2, rel = 0.01)
    assert controller._split_due(fused).split
    controller.record_round(fused, state, [1], seconds = 1.9, catch_up_seconds = 0.7)
    assert controller.round_seconds(fused, state) == pytest.approx(1.2, rel = 0.01)


def test_a_round_running_first_use_work_updates_no_cost_and_is_measured_again():
    controller = DraftController(max_depth = 2, can_copy = False)
    state = [RowState(controller.new_reply())]
    plan = controller.plan(state)
    controller.record_round(plan, state, [1], seconds = 9.0, draft_seconds = 3.0, first_use = True)
    controller.record_round(RoundPlan("round", rows = (RowPlan("draft", 2),)), state, [1], seconds = 9.0, first_use = True)
    assert controller.verify_cost == {} and controller.draft_cost[2].value is None and controller.hidden_drafting == {}
    assert controller.draft_acceptance[0].count == 3 and controller.steps == 4 and controller._measured_at == {}
    assert controller.plan(state) == plan and plan.split
    controller.record_round(plan, state, [1], seconds = 1.5, draft_seconds = 0.5)
    assert (controller.verify_cost[1][3].value, controller.draft_cost[2].value) == pytest.approx((1.0, 0.5))
    assert controller._measured_at == {(1, "round", 3): 6}


def test_a_first_use_spike_at_each_width_does_not_pin_a_shallow_width():
    # 27B 4-bit on flat verify kernels, in plain steps: each width's first block from 3 rows on pays a 7x compile.
    table, seen, draft = [1.19, 1.29, 1.48, 1.51, 1.53, 1.62, 1.71], set(), [0.85] * 7
    machine = _Machine()
    machine.draft = lambda depth: 0.19
    machine.verify = lambda batch, width: table[width - 2] * (1.0 if width < 3 or width in seen or seen.add(width) else 7.0)
    choice = _run(DraftController(max_depth = 7, can_copy = False), machine, [(draft, None)], rounds = 800, report = (report := {}))
    deepest = sum(prod(draft[:k]) for k in range(8)) / (table[-1] + 0.19)
    assert choice[0][1] >= 6 and report["rate"] >= 0.93 * deepest


def test_timings_resist_one_off_work():
    # The first two keep the faster; later, 100 counts as 2 and 0.1 in full.
    for samples, value in (((5.0, 1.0, 100.0), 4 / 3), ((1.0, 5.0), 1.0), ((1.0, 1.0, 0.1), 0.7)):
        timing = _Ema()
        for sample in samples:
            timing.update_time(sample, 0.2)
        assert (timing.value, timing.count) == pytest.approx((value, len(samples)))
    controller = DraftController(max_depth = 7, can_copy = False)
    controller.verify_cost[1] = {2: _Ema(1.0, 2), 4: _Ema(1.2, 2), 6: _Ema(9.0), 8: _Ema(1.5)}
    # A width timed once counts no more than the confirmed widths predict.
    assert [value for _, value in controller._measured(controller.verify_cost[1])] == pytest.approx([1.0, 1.2, 1.4, 1.5])
    # And a round at it, or drafting a depth timed once, is timed apart again, so its second sample confirms or replaces it.
    controller.draft_cost.update({1: _Ema(0.2, 2), 3: _Ema(0.2), 7: _Ema(0.2, 2)})
    rows = [RowState(controller.new_reply())]
    assert [controller._measured_parts(RoundPlan("round", rows = (RowPlan("draft", depth),)), rows) for depth in (1, 3, 7)] == [True, False, False]


def test_a_trimmed_round_verifies_the_width_its_drafts_pay_for():
    controller = DraftController(max_depth = 7, can_copy = False, full_every = 4)
    def costs(values):
        curve = controller.cycle_cost[1] = _CycleCosts()
        for width, seconds in values:  # twice: a width counts once observed twice
            curve.observe(width, seconds)
            curve.observe(width, seconds)
    costs([(w, 1.0 + 0.5 * (w - 1)) for w in range(1, 9)])
    rows, plan = [RowState(controller.new_reply())], RoundPlan("round", rows = (RowPlan("draft", 7),), split = True, trim = True)
    width = lambda confidence: controller.trim(plan, rows, [confidence]).width
    # A doubtful third draft ends the verify before it, confident drafts all verify, doubtful ones keep one; the fourth is a full round.
    assert [width([0.99, 0.99, 0.2] + [0.99] * 4), width([0.99] * 7), width([0.2] * 7), width([0.2] * 7)] == [3, 8, 1, 8]
    # When width costs nothing, every draft is worth verifying.
    costs([(w, 1.0) for w in range(1, 9)])
    assert width([0.2] * 7) == 8
    # Accounted, the verified positions calibrate, and the drafting is that of the block drafted, not of the rows verified.
    controller.record_round(RoundPlan("round", rows = (RowPlan("draft", 3),), split = True, trim = True), rows, [3], seconds = 2.0,
                            draft_seconds = 0.4, confidences = [[0.9] * 7])
    assert [len(records) for records in controller.calibration._records] == [1, 1, 1, 0, 0, 0, 0]
    assert controller.draft_cost[7].value == pytest.approx(0.4) and controller.draft_cost[3].value is None
    # So is a row's that had no confidences; the fused rounds' hidden drafting is no discount for a round that waited for it.
    controller.record_round(RoundPlan("round", rows = (RowPlan("draft", 3),), split = True, trim = True), rows, [3], seconds = 2.0,
                            draft_seconds = 0.6, confidences = [None])
    assert controller.draft_cost[7].count == 2 and controller.draft_cost[3].value is None
    costs([(w, 5.0 + 0.1 * (w - 1)) for w in range(1, 9)])
    assert width([0.99, 0.6] + [0.2] * 5) == 4


def test_drafter_confidences_are_calibrated_against_acceptance():
    import math
    calibration, rng = _Calibration(3), random.Random(17)
    offsets = [-0.2, -0.8, -1.3]
    probability = lambda p, pos: 1 / (1 + math.exp(-0.55 * math.log(p / (1-p)) - offsets[pos]))
    for _ in range(2400):
        confidence, accepted = [rng.uniform(0.3, 0.999) for _ in offsets], 0
        while accepted < 3 and rng.random() < probability(confidence[accepted], accepted):
            accepted += 1
        calibration.observe(confidence, 3, accepted)
    assert len(set(calibration.beta)) == 1
    for confidence in ([0.4, 0.8, 0.95], [0.95, 0.4, 0.8]):
        survival, expected = 1.0, 1.0
        for pos, p in enumerate(confidence):
            survival *= probability(p, pos)
            expected += survival
        assert calibration.expected(confidence) == pytest.approx(expected, abs = 0.09)
    cold = _Calibration(3)
    for _ in range(8):
        cold.observe([0.99, 0.99, 0.99], 3, 0)
    assert cold.beta[2] == cold.beta[0] < 1.0
    assert cold.alpha[0] < 0 == cold.alpha[1] == cold.alpha[2]
    assert cold.expected([0.99] * 3) < 2.0
    before = [list(r) for r in cold._records]
    cold.observe([float("nan"), float("inf"), -0.1], 3, 3)
    assert [list(r) for r in cold._records] == before
    assert 1 < cold.expected([float("nan"), 1.0, 0.0]) <= 4
    fresh = _Calibration(2)
    assert fresh.expected([1.1, 0.9]) == pytest.approx(2.33)
    assert fresh.expected([0.9, -0.1]) == pytest.approx(2.53)


def test_completed_cycle_costs_keep_width_coverage_and_clip_stalls():
    curve = _CycleCosts()
    for width, seconds in ((2, 12.0), (8, 24.0), (2, 12.0)):
        curve.observe(width, seconds)
    assert not curve.ready
    curve.observe(8, 24.0)
    assert curve.ready and curve.seconds(5) == 18.0
    curve.observe(8, 1e6)
    assert curve.seconds(8) == pytest.approx(32.0)
    before = dict(curve.points)
    for bad in (0, -1, float("nan"), float("inf")):
        curve.observe(8, bad)
    assert curve.points == before
    for _ in range(64):
        curve.observe(4, 18.0)
    assert not curve.ready and curve.points == {4: (18.0, 64)}
    curve.observe(8, 24.0)
    curve.observe(8, 24.0)
    assert curve.ready and curve.seconds(6) == 21.0
    curve.points = {2: (23.0, 2), 4: (21.0, 6), 8: (28.0, 8)}
    assert [curve.seconds(w) for w in (2, 4, 6, 8)] == [21.5, 21.5, 24.75, 28.0]


def test_short_budgets_never_plan_empty_drafts():
    controller = DraftController(max_depth = 3, can_copy = False)
    for budgets in ([1], [1, 10], [0, 1]):
        states = [RowState(controller.new_reply(), remaining = budget) for budget in budgets]
        plan = controller.plan(states)
        if plan.kind == "round":
            assert all(row.length for row in plan.rows if row.source == "draft")
            controller.record_round(plan, states, [0] * len(states), seconds = 1.0, draft_seconds = 0.1)


def test_copy_probe_displaces_a_winning_drafter():
    # Two perfect copy tokens tie a perfect depth-2 draft yet skip its forward, but the copy prior scores below the drafter, so only a probe finds that out.
    controller = DraftController(max_depth = 2, max_copy = 12)
    _run(controller, _Machine(draft_slope = 0.05), [([1.0, 1.0], 1.0)], rounds = 400, copies = 2)
    assert controller.copy_acceptance.count > 1


def test_plain_is_interrupted_only_by_a_worthwhile_copy():
    controller = DraftController(max_depth = 0, max_copy = 16)
    state = [RowState(controller.new_reply(), can_draft = False)]
    controller.record_plain(state, 1, 1.0, 1.0)
    assert not controller.interrupts_plain(state)
    state[0].copy_available = 16
    for width, seconds in ((2, 1.1), (17, 4.0)):
        controller.record_round(RoundPlan("round", rows = (RowPlan("copy", width - 1),), split = True),
                                state, [width - 1], seconds = seconds)
    for rate, credit, interrupts in ((0.01, 0.0, False), (0.01, 100.0, True), (0.5, 0.0, True)):
        state[0].stats.copy.value, state[0].stats.probe_at["copy"], controller._credit = rate, 0, credit
        assert controller.interrupts_plain(state) == interrupts
    fresh = DraftController(max_depth = 0, max_copy = 16)
    unmeasured = [RowState(fresh.new_reply(), can_draft = False, copy_available = 16)]
    fresh.record_plain(unmeasured, 1, 1.0, 1.0)
    fresh.verify_cost[1] = {2: _Ema(3.0), 17: _Ema(12.0)}
    assert fresh.interrupts_plain(unmeasured)


def test_mixed_depth_rounds_keep_flat_drafting_costs():
    # Parallel-block drafters cost the same at any depth; a mixed round must not skew that.
    controller = DraftController(max_depth = 4, can_copy = False)
    states = [RowState(controller.new_reply()) for _ in range(2)]
    for depth in (4, 1):
        split = RoundPlan("round", rows = (RowPlan("draft", depth),) * 2, split = True)
        controller.record_round(split, states, [0, 0], seconds = 1.2, draft_seconds = 0.2)
    mixed = RoundPlan("round", rows = (RowPlan("draft", 1), RowPlan("draft", 4)))
    controller.record_round(mixed, states, [0, 0], seconds = 1.2)
    assert controller.draft_cost[1].value == pytest.approx(0.1)
    assert controller.draft_cost[4].value == pytest.approx(0.1)


def test_drafter_recovers_after_copies_won():
    controller, machine = DraftController(max_depth = 4, max_copy = 12), _Machine(draft_slope = 0.05)
    _run(controller, machine, [([1.0, 1.0, 0.0, 0.0], 1.0)], rounds = 1000, copies = 2)
    assert _run(controller, machine, [([1.0] * 4, 1.0)], rounds = 3000, copies = 2) == (("draft", 4),)



def test_unlucky_rounds_do_not_end_drafting():
    _run(controller := DraftController(max_depth = 3, can_copy = False), machine := _Machine(draft_slope = 0.3), [([0.7] * 3, None)], rounds = 1000)
    _run(controller, machine, [([0.0] * 3, None)], rounds = 80)  # one reply's rejected tail, then the next reply's rejected first round
    assert _run(controller, machine, [([0.0] * 3, None)], rounds = 1, states = (states := [])) and max(controller.score(controller._round_at(width, states), states) for width in range(2, 5)) > controller.score(RoundPlan("plain", 1), states)


def test_plain_must_beat_drafting_at_the_rates_the_loads_samples_allow():
    # Depth 2 at rates 0.8 runs 1.13x plain at these costs, at 0.4 0.72x: four samples of 0.4 do not rule 0.8 out, forty do.
    for optimistic, first in ((True, "round"), (False, "plain")):
        controller = DraftController(max_depth = 2, can_copy = False, **({} if optimistic else {"optimism": 0.0}))
        controller._warmup, controller._next_probe, controller.plain_cost[1], controller.draft_acceptance, controller.verify_cost[1], controller.draft_cost = [], float("inf"), _Ema(1.0), [_Ema(0.4, 4) for _ in range(2)], {w: _Ema(1.7 + 0.03 * (w - 1)) for w in (2, 3)}, {d: _Ema(0.2 * d) for d in (1, 2)}
        choose = lambda states: controller._choose([RoundPlan("plain", 8)] + [controller._round_at(w, states) for w in (2, 3)], states, 1)[0].kind
        assert choose(states := [RowState(controller.new_reply())]) == first
        controller.draft_acceptance[:] = [_Ema(0.4, 40) for _ in range(2)]  # samples the load gathers while this reply runs
        assert choose(states) == "plain"
    assert _upper(0.0, 8, 2.0) == pytest.approx(1 / 3) and 0.4 < _upper(0.4, 4000, 2.0) < 0.42


def test_a_fresh_loads_first_reply_drafts_on_its_own_accepted_rounds():
    # Flash-Next's first rounds and plain window, in plain steps; with a cheaper head, unmeasured copies get one probe.
    for copies, draft, first in ((0, 0.70, RowPlan("draft", 3)), (16, 0.05, RowPlan("copy", 4)), (16, 0.05, RowPlan("draft", 3))):
        controller = DraftController(max_depth = 3, can_copy = bool(copies))
        if first.source == "draft" and copies:
            controller.copy_acceptance = _Ema(0.7, controller.seed_weight)
        state = [RowState(controller.new_reply(), copy_available = copies)]
        for accepted, seconds, drafting in ((3, 1.96 + draft, draft), (1, 1.80, 0.18)):
            controller.record_round(controller.plan(state), state, [accepted], seconds = seconds, draft_seconds = drafting)
        assert controller.plan(state).kind == "plain"
        controller.record_plain(state, 8, 8.0, 1.0)
        assert controller.plan(state).rows == (first,)


@pytest.mark.parametrize("draft, floor", [([1.0], 0.97), ([0.5] * 16, 0.97)])
def test_wide_batch_settles_on_plain_without_endless_probes(draft, floor):
    # At B=16 (plain step 2, width-2 verify 4.3) depth 1 is a close loser probed in bursts longer than the probe interval; depth 16 leaves sixteen stale widths.
    step = 2 / 3.25
    machine, report = _Machine(step = step, draft_slope = 0.01 / step, verify_slope = 1.9 / (16 * step)), {}
    controller = DraftController(max_depth = len(draft), can_copy = False, explore_fraction = 0.05)
    assert _run(controller, machine, [(draft, None)] * 16, rounds = 500, report = report) == "plain"
    assert report["rate"] >= floor * 16 / machine.plain(16)


def test_each_row_probes_its_own_sources():
    controller, states = DraftController(max_depth = 2, max_copy = 12), []
    _run(controller, _Machine(draft_slope = 0.05), [([1.0, 1.0], None), ([1.0, 1.0], 1.0)],
         rounds = 400, copies = 2, states = states)
    assert states[1].stats.copy.count > 1


def test_drafter_recovers_while_plain_wins():
    # Rejected drafts score the drafter below failing copies while plain wins, yet perfect drafts still become the fastest choice.
    controller = DraftController(max_depth = 2, max_copy = 12)
    machine = _Machine(draft_slope = 0.1, overhead = 1.1)
    _run(controller, machine, [([0.0, 0.0], 1e-9)], rounds = 500, copies = 2)
    assert _run(controller, machine, [([1.0, 1.0], 1e-9)], rounds = 3000, copies = 2)[0][0] == "draft"


def test_rows_switch_to_drafting_together():
    # Either row drafting alone widens the verify for both; only both drafting pays for it.
    controller = DraftController(max_depth = 4, max_copy = 1)
    states = [RowState(controller.new_reply(), copy_available = 1) for _ in range(2)]
    for depth in range(4, 0, -1):
        drafts = RoundPlan("round", rows = (RowPlan("draft", depth),) * 2, split = True)
        controller.record_round(drafts, states, [depth] * 2, seconds = 1.2 + 0.62 * depth, draft_seconds = 0.02 * depth)
    for state in states:
        state.stats.copy.value = 1.0
        state.stats.draft = [_Ema(1.0) for _ in range(4)]
        state.stats.probe_at = {"copy": 1 << 30, "draft": 1 << 30}
    controller.record_plain(states, 1, 5.0, 5.0)
    assert [row.source for row in controller.plan(states).rows] == ["draft", "draft"]
    states[1].stats.draft = [_Ema(0.0) for _ in range(4)]
    states[1].copy_available = 0
    assert [row.source == "draft" for row in controller._round_at(5, states).rows] == [True, False]


def test_lone_drafter_can_trail_the_ratio_order():
    controller = DraftController(max_depth = 1, can_copy = False)
    controller.verify_cost[2], controller.draft_cost[1], controller.catch_up_cost = {2: _Ema(1.0)}, _Ema(0.2), _Ema(0.1)
    states = [RowState(controller.new_reply(), catch_up = n) for n in (8, 0)]
    for state, rate in zip(states, (0.9, 0.1)):
        state.stats.draft, state.stats.draft_seen = [_Ema(rate)], [True]
    assert [row.source == "draft" for row in controller._round_at(2, states).rows] == [False, True]


def test_hidden_drafting_picks_the_best_set_of_unequal_drafters():
    # Drafting up to the hidden time is free, so two cheap rows beat the best ratio.
    controller = DraftController(max_depth = 1, can_copy = False)
    rows = [RowState(controller.new_reply(), catch_up = catch_up) for catch_up in (1, 0, 0)]
    for state, rate in zip(rows, (0.6, 0.45, 0.45)):
        state.stats.draft[0].value = rate
    controller.plain_cost[4] = _Ema(1.0)
    controller.verify_cost[4] = {2: _Ema(1.0)}
    controller.draft_cost[1], controller.catch_up_cost = _Ema(1.0), _Ema(0.2)
    controller.hidden_drafting[4] = _Ema(2.0)
    assert [row.source for row in controller._round_at(2, rows).rows] == ["none", "draft", "draft"]



def test_probes_pay_their_measured_regret_within_the_budget():
    # Plain wins and widths past 8 cost 2.5 times what narrower widths predict: probes may cost 1%.
    machine, report = _Machine(verify_slope = 0.1), {}
    machine.verify = lambda batch, width, verify = machine.verify: verify(batch, width) * (2.5 if width > 8 else 1.0)
    controller = DraftController(max_depth = 16, max_copy = 12, explore_fraction = 0.01)
    assert _run(controller, machine, [([0.3] * 16, 0.2), ([0.3] * 16, None)] * 2, rounds = 600, report = report) == "plain"
    assert report["rate"] >= 0.987 * 4 / machine.plain(4) and DraftController(max_depth = 15)._warmup == [15, 7, 3, 1]



def _init_attributes(cls) -> set[str]:
    """Every ``self.<name> = ...`` target in ``cls.__init__``."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    names = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        for target in targets:
            for leaf in ast.walk(target) if target is not None else ():
                if isinstance(leaf, ast.Attribute) and isinstance(leaf.value, ast.Name) and leaf.value.id == "self":
                    names.add(leaf.attr)
    return names


def _bare_engine(cls, **attrs):
    """Hand-built engine skips __init__ (imports mlx_vlm) and sets each attribute to None, so a flag added later (issue 1871's _invariant) takes its default path."""
    engine = object.__new__(cls)
    for name in _init_attributes(cls):
        setattr(engine, name, None)
    for name, value in attrs.items():
        setattr(engine, name, value)
    return engine


def test_the_bare_engine_has_every_attribute_init_sets():
    from unsloth_zoo.mlx.speculative import SpeculativeEngine

    names = _init_attributes(SpeculativeEngine)
    # The flags _verify branches on: if the reader stops finding them, the helper is reading nothing.
    assert {"_exact", "_invariant", "lm", "cache", "_verify_forward"} <= names
    engine = _bare_engine(SpeculativeEngine)
    assert all(getattr(engine, name) is None for name in names)


def test_committed_cache_submission_defers_only_a_pending_fold_replay():
    from unsloth_zoo.mlx.speculative import _committed_eval_state

    class Cache:
        def __init__(self, state, pending = None):
            self.state = state
            if pending is not None:
                self._unsloth_pending = (pending, object(), object())

    first, replay, last = object(), object(), object()
    assert _committed_eval_state(Cache([first, replay, last], replay)) == [first, last]
    assert _committed_eval_state(Cache([first, last], replay)) == [first, last]
    assert _committed_eval_state(Cache((first, last))) == (first, last)
    assert _committed_eval_state(Cache(replay, replay)) == []
    assert _committed_eval_state(Cache(first, replay)) is first


def test_only_the_verify_forward_is_built_as_a_verify_block():
    import importlib
    from unsloth_zoo.mlx.speculative import SpeculativeEngine

    # _verify's from .inference import resolves via sys.modules; the package attribute can go stale after shim tests purge and restore unsloth_zoo.mlx.
    inference = importlib.import_module("unsloth_zoo.mlx.inference")

    marked = lambda: getattr(inference._VERIFY_STATE, "attention", False)
    seen = []
    engine = _bare_engine(SpeculativeEngine, _kwargs = dict, _hidden = lambda out: None)
    engine._verify_forward = lambda lm, inputs, cache: seen.append(marked()) or (type("Out", (), {"logits": None}), None)
    engine._verify(None)
    assert seen == [True] and not marked()


@pytest.mark.parametrize("history_only", [True, False])
def test_a_row_kept_off_the_draft_never_drafts_and_pipelines_as_its_processors_allow(history_only, monkeypatch):
    import contextlib

    import mlx.core as mx
    from unsloth_zoo.mlx.speculative import EngineRow, SpeculativeEngine, _Row

    processor = lambda tokens, logits: logits
    processor.history_only = history_only
    controller = DraftController(max_depth = 1, can_copy = False)
    rows = [_Row(EngineRow(None, 1, [1, 2], processors = [processor], drafts = drafts), controller) for drafts in (True, False)]
    assert [row.speculates for row in rows] == [history_only, False]
    # Pipelining waits for a processor that reads more than the history, not for a row off the draft.
    queued = []
    step = lambda inputs, ahead = 0: queued.append(ahead) or (mx.array([3, 3]), None, None)
    record = SimpleNamespace(record_plain = lambda *_a: None, interrupts_plain = lambda _s: False)
    # No stream work happens here; the CPU shim implements none.
    monkeypatch.setattr(mx, "stream", lambda _s: contextlib.nullcontext())
    engine = _bare_engine(SpeculativeEngine, _step = step, _stream = None, controller = record)
    engine._rows, engine._states = rows, lambda: []
    list(engine._plain(3, []))
    assert (1 in queued) == history_only


def test_a_context_drafter_drafts_the_next_block_ahead_until_a_push_overtakes_it():
    import mlx.core as mx
    from unsloth_zoo.mlx.speculative import ContextDrafter, ContextRow
    sizes, drafter, row = [], object.__new__(ContextDrafter), ContextRow([])
    drafter.max_lag, drafter.max_depth, drafter._selector = 64, 3, None
    # Each draft encodes its block's anchor and context length.
    drafter._draft = lambda pending, context, cache, size, sampler, dtype: sizes.append(size) or mx.array([[100 * pending + 10 * context.shape[1] + i for i in range(size - 1)]])
    drafter.push(row, [1, 2], mx.zeros((2, 4)))
    drafter.prefetch([row, None])
    assert row.ready and drafter.draft([], [row, None], 2, []).tolist() == [[220, 221], [0, 0]] and sizes == [4] and not row.ready
    drafter.push(row, [3], mx.zeros((1, 4)))
    drafter.prefetch([row])
    drafter.push(row, [4], mx.zeros((1, 4)))
    assert drafter.draft([], [row], 2, []).tolist() == [[410, 411]] and sizes == [4, 4, 3]


def test_a_context_drafter_drafts_from_the_block_that_moves_a_long_backlog_into_its_cache():
    import mlx.core as mx
    from unsloth_zoo.mlx.speculative import ContextDrafter, ContextRow
    sizes, drafter, row = [], object.__new__(ContextDrafter), ContextRow([])
    drafter.max_lag, drafter.max_depth, drafter._selector = 64, 3, None
    drafter._draft = lambda pending, context, cache, size, sampler, dtype: sizes.append(size) or mx.array([[100 * pending + 10 * context.shape[1] + i for i in range(size - 1)]])
    drafter.push(row, [1], mx.zeros((63, 4)))
    assert row.ready and row.prefetched is None and not sizes and not row.flushed
    # A prompt past the backlog drafts its first round at once, from the full block that moved it.
    drafter.push(row, [5], mx.zeros((64, 4)))
    assert row.ready and not row.features and sizes == [4] and row.flushed
    assert drafter.draft([], [row], 3, []).tolist() == [[1770, 1771, 1772]] and sizes == [4] and not row.flushed


def test_a_context_drafter_serves_a_prefetched_block_of_the_requested_depth_unsliced():
    import mlx.core as mx
    from unsloth_zoo.mlx.speculative import ContextDrafter, ContextRow
    drafter = object.__new__(ContextDrafter)
    drafter.max_lag, drafter.max_depth, drafter._selector = 64, 3, None
    drafter._draft = lambda pending, context, cache, size, sampler, dtype: mx.arange(size - 1)[None]
    def ready():
        row = ContextRow([])
        drafter.push(row, [1, 2], mx.zeros((2, 4)))
        drafter.prefetch([row])
        row.prefetched_confidence = mx.full((1, 3), 0.5)
        return row
    row = ready()
    block, confidence = row.prefetched, row.prefetched_confidence
    assert drafter.draft([], [row], 3, []) is block and drafter.confidence == [confidence]
    row = ready()
    block, confidence = row.prefetched, row.prefetched_confidence
    served = drafter.draft([], [row], 2, [])
    assert served is not block and served.tolist() == block[:, :2].tolist()
    assert drafter.confidence[0] is not confidence and drafter.confidence[0].shape == (1, 2)
    first, second = ready(), ready()
    assert drafter.draft([], [first, second], 3, []).tolist() == [[0, 1, 2], [0, 1, 2]]


def test_drafter_layers_submit_their_output_when_built_in_eval_mode(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx import inference

    class Layer(nn.Module):
        def __init__(self, mlp = True):
            super().__init__()
            self.self_attn = nn.Linear(4, 4)
            if mlp:
                self.mlp = nn.Linear(4, 4)

        def __call__(self, x):
            return self.mlp(self.self_attn(x)) if hasattr(self, "mlp") else self.self_attn(x)

    model, submitted = nn.Module(), []
    model.layers = [Layer(), Layer(mlp = False), Layer()]
    model.eval()
    monkeypatch.setattr(inference.mx, "async_eval", lambda *arrays: submitted.extend(arrays))
    x = mx.ones((1, 4))
    expected = [layer(x) for layer in model.layers]  # before reclassing
    assert inference.submitted_drafter_layers(model) == 2 and type(model.layers[1]) is Layer
    outputs = [layer(x) for layer in model.layers]
    assert all(mx.array_equal(out, want).item() for out, want in zip(outputs, expected))
    assert len(submitted) == 2 and submitted[0] is outputs[0] and submitted[1] is outputs[2]
    # Training submits nothing.
    model.layers[0].train()
    model.layers[0](x)
    assert len(submitted) == 2
    # Inside a function transformation, where submitting is refused, the layer still returns its output.
    monkeypatch.setattr(inference.mx, "async_eval", lambda *arrays: (_ for _ in ()).throw(ValueError("transform")))
    assert mx.array_equal(model.layers[2](x), expected[2]).item()


def test_prepared_plans_preserve_controller_state_and_choices():
    import copy
    import pickle
    controller = DraftController(max_depth = 3, min_window = 3, probe_every = 5, stale_after = 8)
    controller.trims = True
    states = [RowState(controller.new_reply(), copy_available = 6), RowState(controller.new_reply())]
    controller.cycle_cost[2] = curve = _CycleCosts()
    curve.observe(2, 0.03)
    curve.observe(2, 0.03)
    curve.observe(4, 0.04)
    curve.observe(4, 0.04)
    baseline, original = copy.deepcopy((controller, states))
    for index in range(160):
        before = pickle.dumps((controller, states))
        prepared = controller.prepare_plan(states)
        assert pickle.dumps((controller, states)) == before
        for c, s in ((controller, states), (baseline, original)):
            c.observe_cycle(RoundPlan("round", rows = (RowPlan("draft", 3),) * 2, trim = True), s, 0.04)
        actual, expected = controller.plan(states, prepared), baseline.plan(original)
        assert (actual is prepared.plan) == (prepared.cycle_ready == controller._cycle_ready(states))
        assert actual == expected
        assert pickle.dumps((controller, states)) == pickle.dumps((baseline, original))
        for c, s in ((controller, states), (baseline, original)):
            if actual.kind == "round":
                c.record_round(actual, s, [min(r.length, (index + i) % 4) for i, r in enumerate(actual.rows)],
                               seconds = 0.04, draft_seconds = 0.006)
            else:
                c.record_plain(s, actual.length, 0.02 * actual.length, 0.02)
    assert controller.cycle_cost[2].points[4][1] == 64
