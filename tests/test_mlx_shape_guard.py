from __future__ import annotations

import itertools
import random

import pytest

from unsloth_zoo.mlx import shape_guard
from unsloth_zoo.mlx.shape_guard import (
    AUTOMATIC_TEXT_COMPILE_CEILING,
    DDP_LOCAL_GRAD_SCOPE,
    FULL_STEP_SCOPE,
    TextShapeEvent,
    build_text_shape_frontier,
    materialize_text_shape_frontier,
    phase_for_microstep,
    plan_text_shape_buckets,
    plan_text_shape_padding_budget,
    resolve_compile_max_variants,
    select_text_shape_padding_budget,
)


def _event(family, width, phase="p", frequency=1, batch_size=1):
    return TextShapeEvent(family, width, phase, frequency, batch_size)


def _partition_options(events):
    by_family = {}
    for event in events:
        width_data = by_family.setdefault(event.family, {}).setdefault(
            event.width, {}
        )
        width_data[event.phase] = width_data.get(event.phase, 0) + event.weight
    options = []
    for family in sorted(by_family, key=shape_guard._family_key):
        widths = sorted(by_family[family])
        family_options = []
        for cuts in range(1 << max(0, len(widths) - 1)):
            starts = [0] + [i + 1 for i in range(len(widths) - 1) if cuts & (1 << i)]
            ends = starts[1:] + [len(widths)]
            cost = 0
            slots = 0
            endpoints = []
            for start, end in zip(starts, ends):
                endpoint = widths[end - 1]
                phases = set()
                for width in widths[start:end]:
                    for phase, weight in by_family[family][width].items():
                        phases.add(phase)
                        cost += weight * (endpoint * endpoint - width * width)
                slots += len(phases)
                endpoints.append(endpoint)
            family_options.append((slots, cost, tuple(endpoints)))
        options.append(family_options)
    return options


def _brute_force(events, cap):
    raw = {
        (FULL_STEP_SCOPE, event.phase, event.family, event.width)
        for event in events
    }
    if len(raw) <= cap:
        return 0, len(raw), ()
    best = None
    for choices in itertools.product(*_partition_options(events)):
        slots = sum(choice[0] for choice in choices)
        if slots > cap:
            continue
        candidate = (
            sum(choice[1] for choice in choices),
            slots,
            tuple(choice[2] for choice in choices),
        )
        if best is None or candidate < best:
            best = candidate
    return best


def _planned_endpoints(plan):
    return tuple(
        tuple(dict.fromkeys(endpoint for _width, endpoint in mapping))
        for _family, mapping in plan.endpoint_maps
    )


def test_compile_phases_match_actual_argument_structures():
    assert [
        len({phase_for_microstep(FULL_STEP_SCOPE, accum, i) for i in range(12)})
        for accum in (1, 2, 4)
    ] == [1, 2, 3]
    assert [
        len({phase_for_microstep(DDP_LOCAL_GRAD_SCOPE, accum, i) for i in range(12)})
        for accum in (1, 2, 4)
    ] == [1, 2, 2]


def test_exact_catalog_preserves_every_width_without_padding():
    events = [
        _event(("text", 1, "int32", None), 13, "single"),
        _event(("text", 1, "int32", None), 29, "single"),
        _event(("text", 1, "int32", "labels:int64"), 13, "single"),
    ]
    plan = plan_text_shape_buckets(
        events, cap=3, compile_scope=FULL_STEP_SCOPE,
    )

    assert plan.report.action == "exact"
    assert plan.report.raw_signatures == plan.report.planned_signatures == 3
    assert all(plan.endpoint_for(event.family, event.width) == event.width for event in events)
    assert plan.padding_cost == 0


def test_phase_aware_plan_uses_minimum_quadratic_padding():
    family = ("text", 2, "int32", "labels:int64")
    events = [
        _event(family, 10, "none_no_update", frequency=3, batch_size=2),
        _event(family, 10, "tree_no_update", frequency=2, batch_size=2),
        _event(family, 11, "none_no_update", frequency=4, batch_size=2),
        _event(family, 30, "tree_no_update", frequency=1, batch_size=2),
    ]
    plan = plan_text_shape_buckets(
        events, cap=3, compile_scope=FULL_STEP_SCOPE,
    )

    assert plan.report.action == "bucket"
    assert plan.report.planned_signatures == 3
    assert plan.endpoint_for(family, 10) == 11
    assert plan.endpoint_for(family, 11) == 11
    assert plan.endpoint_for(family, 30) == 30
    assert all(plan.allows(event.family, event.width, event.phase) for event in events)


def test_irreducible_families_and_planner_limits_select_eager(monkeypatch):
    irreducible = plan_text_shape_buckets(
        [_event(("a",), 2), _event(("b",), 3)],
        cap=1,
        compile_scope=FULL_STEP_SCOPE,
    )
    assert (irreducible.report.action, irreducible.report.reason) == (
        "eager", "irreducible_signatures",
    )

    monkeypatch.setattr(shape_guard, "MAX_WIDTHS_PER_FAMILY", 2)
    too_many = plan_text_shape_buckets(
        [_event(("a",), width) for width in (2, 3, 4)],
        cap=1,
        compile_scope=FULL_STEP_SCOPE,
    )
    assert (too_many.report.action, too_many.report.reason) == (
        "eager", "too_many_widths",
    )

    monkeypatch.setattr(shape_guard, "MAX_WIDTHS_PER_FAMILY", 2_048)
    monkeypatch.setattr(shape_guard, "MAX_PLANNER_WORK", 1)
    over_work = plan_text_shape_buckets(
        [_event(("a",), width) for width in (2, 3, 4)],
        cap=1,
        compile_scope=FULL_STEP_SCOPE,
    )
    assert (over_work.report.action, over_work.report.reason) == (
        "eager", "planner_work_limit",
    )


@pytest.mark.parametrize("value", [True, False, 0, 257, 1.5, "32"])
def test_variant_cap_rejects_non_integer_or_out_of_range_values(value):
    with pytest.raises(ValueError, match="compile_max_variants"):
        resolve_compile_max_variants(value)
    assert resolve_compile_max_variants(None) == 128


def test_report_is_bounded_and_deterministic_across_event_order():
    events = [
        _event(("tail", 1), 17, "a"),
        _event(("main", 2), 10, "a", frequency=2),
        _event(("main", 2), 11, "b", frequency=3),
        _event(("tail", 1), 19, "a"),
    ]
    forward = plan_text_shape_buckets(
        events, cap=3, compile_scope=FULL_STEP_SCOPE,
    )
    reverse = plan_text_shape_buckets(
        reversed(events), cap=3, compile_scope=FULL_STEP_SCOPE,
    )

    assert forward.report.to_dict() == reverse.report.to_dict()
    assert forward.endpoint_maps == reverse.endpoint_maps
    report = forward.report.to_dict()
    assert set(report) == {
        "action", "reason", "cap", "compile_scope", "raw_signatures",
        "planned_signatures", "raw_widths", "planned_endpoints",
        "padding_tokens", "original_tokens", "lazy_batches",
        "configured_cap", "effective_cap", "cap_selection",
        "padding_work_fraction", "max_width_stretch", "budget_satisfied",
        "padding_fraction",
    }
    assert len(report["planned_endpoints"]) <= report["cap"]


def test_randomized_small_plans_match_exhaustive_optima():
    rng = random.Random(3407)
    for _ in range(350):
        events = []
        for family_index in range(rng.randint(1, 2)):
            family = ("family", family_index, rng.randint(1, 3))
            widths = sorted(rng.sample(range(2, 24), rng.randint(1, 4)))
            for width in widths:
                phases = rng.sample(("a", "b", "c"), rng.randint(1, 2))
                for phase in phases:
                    events.append(
                        _event(
                            family,
                            width,
                            phase,
                            frequency=rng.randint(1, 3),
                            batch_size=rng.randint(1, 2),
                        )
                    )
        raw_count = len({(event.phase, event.family, event.width) for event in events})
        cap = rng.randint(1, raw_count)
        expected = _brute_force(events, cap)
        actual = plan_text_shape_buckets(
            events, cap=cap, compile_scope=FULL_STEP_SCOPE,
        )

        if expected is None:
            assert actual.report.action == "eager"
            continue
        expected_cost, expected_slots, expected_endpoints = expected
        if raw_count <= cap:
            assert actual.report.action == "exact"
            assert actual.padding_cost == 0
        else:
            assert actual.report.action == "bucket"
            assert actual.padding_cost == expected_cost
            assert actual.report.planned_signatures == expected_slots
            assert _planned_endpoints(actual) == expected_endpoints
        assert len(actual.planned_catalog) <= cap
        for event in events:
            assert actual.endpoint_for(event.family, event.width) >= event.width


def test_padding_budget_keeps_small_schedules_exact_without_frontier(monkeypatch):
    def unexpected_frontier(*_args, **_kwargs):
        raise AssertionError("small exact schedules must bypass frontier work")

    monkeypatch.setattr(shape_guard, "_build_global_frontier", unexpected_frontier)
    events = [_event(("text",), width) for width in range(10, 42)]

    plan = plan_text_shape_padding_budget(events, compile_scope=FULL_STEP_SCOPE)

    assert plan.report.cap_selection == "exact"
    assert plan.report.configured_cap == AUTOMATIC_TEXT_COMPILE_CEILING
    assert plan.report.effective_cap == plan.report.cap == 32
    assert plan.report.action == "exact"
    assert plan.report.padding_work_fraction == 0.0
    assert plan.report.max_width_stretch == 1.0
    assert plan.report.budget_satisfied is True


def test_padding_budget_uses_one_frontier_and_adapts_to_width_distribution(monkeypatch):
    original = shape_guard._build_global_frontier
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(shape_guard, "_build_global_frontier", counted)
    clustered = [_event(("text",), width) for width in range(100, 164)]
    irregular = [_event(("text",), 10 + width * width) for width in range(64)]

    clustered_plan = plan_text_shape_padding_budget(
        clustered, compile_scope=FULL_STEP_SCOPE,
    )
    irregular_plan = plan_text_shape_padding_budget(
        irregular, compile_scope=FULL_STEP_SCOPE,
    )

    assert calls == [1, 1]
    assert 1 <= clustered_plan.report.effective_cap < AUTOMATIC_TEXT_COMPILE_CEILING
    assert clustered_plan.report.budget_satisfied is True
    assert irregular_plan.report.effective_cap > clustered_plan.report.effective_cap
    assert irregular_plan.report.budget_satisfied is True


def test_padding_budget_exact_boundaries_and_explicit_caps_remain_deterministic():
    boundary = [_event(("text",), 100, "p0", frequency=3)]
    boundary.extend(
        _event(("text",), 150, f"p{phase}") for phase in range(32)
    )
    over_stretch = [_event(("text",), 100, "p0", frequency=3)]
    over_stretch.extend(
        _event(("text",), 151, f"p{phase}") for phase in range(32)
    )

    accepted = plan_text_shape_padding_budget(
        boundary, compile_scope=FULL_STEP_SCOPE,
    )
    rejected = plan_text_shape_padding_budget(
        over_stretch, compile_scope=FULL_STEP_SCOPE,
    )

    assert accepted.report.effective_cap == 32
    assert accepted.padding_cost * 100 == sum(
        event.weight * event.width * event.width for event in boundary
    ) * 5
    assert accepted.report.max_width_stretch == 1.5
    assert rejected.report.effective_cap == 33
    for cap in (1, 256):
        fixed = plan_text_shape_buckets(
            [_event(("fixed",), 10), _event(("fixed",), 20)],
            cap=cap,
            compile_scope=FULL_STEP_SCOPE,
        )
        assert fixed.report.cap_selection == "fixed"
        assert fixed.report.configured_cap == fixed.report.effective_cap == cap


def test_padding_budget_ceiling_is_bounded_when_no_point_meets_stretch():
    events = [_event(("text",), 2**index) for index in range(129)]

    plan = plan_text_shape_padding_budget(events, compile_scope=FULL_STEP_SCOPE)

    assert plan.report.action == "bucket"
    assert plan.report.cap_selection == "ceiling"
    assert plan.report.effective_cap == plan.report.cap == 128
    assert plan.report.planned_signatures == 128
    assert plan.report.budget_satisfied is False
    assert plan.report.max_width_stretch > 1.5


def test_padding_budget_keeps_stretch_feasible_long_tail_frontier():
    widths = (8, 13, 21, 34, 55) + tuple(
        random.Random(6994).sample(range(96, 513), 139)
    )
    events = [_event(("text",), width, "single") for width in widths]

    plan = plan_text_shape_padding_budget(
        events, compile_scope=FULL_STEP_SCOPE,
    )

    assert plan.report.effective_cap == plan.report.planned_signatures == 21
    assert plan.report.padding_work_fraction == pytest.approx(
        0.04942145401598708,
    )
    assert plan.report.max_width_stretch == pytest.approx(1.34375)
    assert plan.report.budget_satisfied is True
    assert all(
        plan.endpoint_for(event.family, event.width) >= event.width
        for event in events
    )


def test_padding_budget_is_independent_of_event_order():
    events = [
        _event(("a",), 10 + index * index, "a", frequency=1 + index % 3)
        for index in range(40)
    ] + [
        _event(("b",), 20 + index * 3, "b", frequency=2)
        for index in range(20)
    ]

    forward = plan_text_shape_padding_budget(
        events, compile_scope=FULL_STEP_SCOPE,
    )
    reverse = plan_text_shape_padding_budget(
        reversed(events), compile_scope=FULL_STEP_SCOPE,
    )

    assert forward.report.to_dict() == reverse.report.to_dict()
    assert forward.endpoint_maps == reverse.endpoint_maps


def test_shared_cap_materialization_preserves_both_padding_budgets():
    events = [
        _event(("core",), width, frequency=frequency)
        for width, frequency in ((26, 13), (30, 1), (53, 11), (70, 15))
    ]
    events.extend(
        _event(("singleton", index), 150) for index in range(29)
    )
    frontier = build_text_shape_frontier(
        events, compile_scope=FULL_STEP_SCOPE,
    )
    local = select_text_shape_padding_budget(frontier)

    assert local.report.effective_cap == 31
    assert local.report.budget_satisfied is True
    shared = materialize_text_shape_frontier(
        frontier,
        cap=32,
        cap_selection=local.report.cap_selection,
    )

    assert shared.report.effective_cap == shared.report.cap == 32
    assert shared.report.planned_signatures <= 32
    assert shared.report.padding_work_fraction <= 0.05
    assert shared.report.max_width_stretch <= 1.5
    assert shared.report.budget_satisfied is True
