import random
from collections import Counter
from math import prod

import pytest

pytest.importorskip("mlx.core")
from unsloth_zoo.mlx.speculative import DraftController, NgramProposer, RoundPlan, RowPlan, RowState, _Ema


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
            controller.record_plain(states, plan.length, seconds)
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


def test_short_budgets_never_plan_empty_drafts():
    controller = DraftController(max_depth = 3, can_copy = False)
    for budgets in ([1], [1, 10], [0, 1]):
        states = [RowState(controller.new_reply(), remaining = budget) for budget in budgets]
        plan = controller.plan(states)
        if plan.kind == "round":
            assert all(row.length for row in plan.rows if row.source == "draft")
            controller.record_round(plan, states, [0] * len(states), seconds = 1.0, draft_seconds = 0.1)


def test_copy_probe_displaces_a_winning_drafter():
    # Two perfect copy tokens tie a perfect depth-2 draft on tokens and skip its forward, but
    # the copy prior scores below the drafter, so only a probe can find that out.
    controller = DraftController(max_depth = 2, max_copy = 12)
    _run(controller, _Machine(draft_slope = 0.05), [([1.0, 1.0], 1.0)], rounds = 400, copies = 2)
    assert controller.copy_acceptance.count > 1


def test_plain_is_interrupted_only_by_a_worthwhile_copy():
    controller = DraftController(max_depth = 0, max_copy = 16)
    state = [RowState(controller.new_reply(), can_draft = False)]
    controller.record_plain(state, 1, 1.0)
    assert not controller.interrupts_plain(state)
    state[0].copy_available = 16
    for width, seconds in ((2, 1.1), (17, 4.0)):
        controller.record_round(RoundPlan("round", rows = (RowPlan("copy", width - 1),), split = True),
                                state, [width - 1], seconds = seconds)
    for rate, credit, interrupts in ((0.01, 0.0, False), (0.01, 100.0, True), (0.5, 0.0, True)):
        state[0].stats.copy.value, state[0].stats.probe_at["copy"], controller._credit = rate, 0, credit
        assert controller.interrupts_plain(state) == interrupts


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

@pytest.mark.parametrize("draft, floor", [([1.0], 0.97), ([0.5] * 16, 0.97)])
def test_wide_batch_settles_on_plain_without_endless_probes(draft, floor):
    # Plain step 2 and a width-2 verify of 4.3 at B=16: depth 1 is a close loser, probed in
    # bursts longer than a probe interval; depth 16 leaves sixteen hopeless stale widths.
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
    # Rejected drafts leave the drafter scored below failing copies while plain wins, yet
    # perfect drafts become the fastest choice.
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
    controller.record_plain(states, 1, 5.0)
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

