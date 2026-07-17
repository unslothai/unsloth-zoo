# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Pure planning for bounded finite-text ``mx.compile`` signatures.

The cap here counts application-visible callable signatures. It is not an MLX
compiler-cache count and is not an estimate of Metal resources.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from fractions import Fraction
import heapq
import hashlib


AUTOMATIC_TEXT_COMPILE_CEILING = 128
DEFAULT_TEXT_COMPILE_MAX_VARIANTS = AUTOMATIC_TEXT_COMPILE_CEILING
SMALL_EXACT_SIGNATURE_THRESHOLD = 32
MAX_PADDING_WORK_PERCENT = 5
MAX_WIDTH_STRETCH_NUMERATOR = 3
MAX_WIDTH_STRETCH_DENOMINATOR = 2
MAX_COMPILE_VARIANTS = 256
_MAX_EXACT_WIDTHS_PER_FAMILY = 2_048
_MAX_EXACT_PLANNER_WORK = 16_000_000

FULL_STEP_SCOPE = "full_step"
DDP_LOCAL_GRAD_SCOPE = "ddp_local_grad"


@dataclass(frozen=True)
class TextShapeEvent:
    """One or more scheduled occurrences of a complete text batch shape."""

    family: tuple
    width: int
    phase: str
    frequency: int = 1
    local_batch_size: int = 1

    def __post_init__(self):
        if type(self.family) is not tuple:
            raise ValueError("family must be a tuple")
        try:
            hash(self.family)
        except TypeError as exc:
            raise ValueError("family must be hashable") from exc
        if type(self.width) is not int or self.width < 1:
            raise ValueError("width must be a positive integer")
        if type(self.phase) is not str or not self.phase:
            raise ValueError("phase must be a non-empty string")
        if type(self.frequency) is not int or self.frequency < 1:
            raise ValueError("frequency must be a positive integer")
        if type(self.local_batch_size) is not int or self.local_batch_size < 1:
            raise ValueError("local_batch_size must be a positive integer")

    @property
    def weight(self):
        return self.frequency * self.local_batch_size


@dataclass(frozen=True)
class TextShapeGuardReport:
    action: str
    reason: str
    cap: int
    compile_scope: str
    raw_signatures: int
    planned_signatures: int | None
    raw_widths: int
    planned_endpoints: tuple[tuple[str, tuple[int, ...]], ...] = ()
    padding_tokens: int = 0
    original_tokens: int = 0
    lazy_batches: bool = True
    configured_cap: int | None = None
    effective_cap: int | None = None
    cap_selection: str = "fixed"
    padding_work_fraction: float = 0.0
    max_width_stretch: float = 1.0
    budget_satisfied: bool = True

    def __post_init__(self):
        if self.configured_cap is None:
            object.__setattr__(self, "configured_cap", self.cap)
        if self.effective_cap is None:
            object.__setattr__(self, "effective_cap", self.cap)

    @property
    def padding_fraction(self):
        return (
            self.padding_tokens / self.original_tokens
            if self.original_tokens else 0.0
        )

    def to_dict(self):
        result = asdict(self)
        result["planned_endpoints"] = {
            family: list(endpoints)
            for family, endpoints in self.planned_endpoints
        }
        result["padding_fraction"] = self.padding_fraction
        return result


@dataclass(frozen=True)
class TextShapePlan:
    """An immutable exact, bucketed, or eager planning result."""

    report: TextShapeGuardReport
    raw_catalog: frozenset[tuple]
    planned_catalog: frozenset[tuple]
    endpoint_maps: tuple[tuple[tuple, tuple[tuple[int, int], ...]], ...] = ()
    padding_cost: int = 0

    def endpoint_for(self, family, width):
        for candidate, mapping in self.endpoint_maps:
            if candidate == family:
                for raw_width, endpoint in mapping:
                    if raw_width == width:
                        return endpoint
                break
        return width

    def signature_for(self, family, width, phase):
        return (
            self.report.compile_scope,
            phase,
            family,
            self.endpoint_for(family, width),
        )

    def allows(self, family, width, phase):
        return self.signature_for(family, width, phase) in self.planned_catalog


class _PlannerLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _TextShapeProblem:
    compile_scope: str
    family_data: dict
    raw_catalog: frozenset[tuple]
    sorted_families: tuple[tuple, ...]
    original_tokens: int
    raw_work: int


@dataclass(frozen=True)
class TextShapeFrontier:
    """Bounded exact or fallback planning state for a finite text schedule."""

    _problem: _TextShapeProblem
    _points: tuple[tuple[int, int, tuple], ...] = ()
    failure_reason: str | None = None
    _stretch_points: tuple[tuple[int, int, tuple], ...] = ()
    _fallback_points: tuple[tuple[int, int, tuple], ...] = ()

    @property
    def raw_signatures(self):
        return len(self._problem.raw_catalog)


class _FamilyBudgetPlanner:
    """Incremental exact stretch-feasible DP for one shape family."""

    def __init__(self, width_data, max_padding_cost, work):
        self.width_data = width_data
        self.widths = tuple(sorted(width_data))
        if len(self.widths) > _MAX_EXACT_WIDTHS_PER_FAMILY:
            raise _PlannerLimit("too_many_widths")
        self.max_padding_cost = max_padding_cost
        self.work = work
        self.width_weights = {
            width: sum(width_data[width].values()) for width in self.widths
        }
        self.max_slots = sum(
            len(width_data[width]) for width in self.widths
        )
        self.irreducible_slots = len({
            phase
            for phases in width_data.values()
            for phase in phases
        })
        self.layers = [[(0, ())] + [None] * len(self.widths)]
        self.options = {}

    def extend(self, target_slots):
        target_slots = min(target_slots, self.max_slots)
        while len(self.layers) <= target_slots:
            slots = len(self.layers)
            layer = [None] * (len(self.widths) + 1)
            for end in range(1, len(self.widths) + 1):
                endpoint = self.widths[end - 1]
                segment_phases = set()
                segment_weight = 0
                segment_squares = 0
                for start in range(end - 1, -1, -1):
                    self.work[0] += 1
                    if self.work[0] > _MAX_EXACT_PLANNER_WORK:
                        raise _PlannerLimit("planner_work_limit")
                    width = self.widths[start]
                    if (
                        endpoint * MAX_WIDTH_STRETCH_DENOMINATOR
                        > width * MAX_WIDTH_STRETCH_NUMERATOR
                    ):
                        break
                    phase_weights = self.width_data[width]
                    segment_phases.update(phase_weights)
                    width_weight = self.width_weights[width]
                    segment_weight += width_weight
                    segment_squares += width_weight * width * width
                    segment_slots = len(segment_phases)
                    if segment_slots > slots:
                        break
                    previous = self.layers[slots - segment_slots][start]
                    if previous is None:
                        continue
                    segment_cost = (
                        endpoint * endpoint * segment_weight - segment_squares
                    )
                    candidate_cost = previous[0] + segment_cost
                    if candidate_cost > self.max_padding_cost:
                        continue
                    candidate = (
                        candidate_cost,
                        previous[1] + (endpoint,),
                    )
                    current = layer[end]
                    if current is None or candidate < current:
                        layer[end] = candidate
            self.layers.append(layer)
            if layer[-1] is not None:
                self.options[slots] = layer[-1]


@dataclass
class _FallbackSegment:
    family_index: int
    start: int
    end: int
    endpoint: int
    phases: frozenset
    weight: int
    squares: int
    cost: int
    previous: int | None
    next: int | None
    active: bool = True


def resolve_compile_max_variants(value=None):
    if value is None:
        return DEFAULT_TEXT_COMPILE_MAX_VARIANTS
    if type(value) is not int or not 1 <= value <= MAX_COMPILE_VARIANTS:
        raise ValueError(
            "compile_max_variants must be an integer from 1 through "
            f"{MAX_COMPILE_VARIANTS}"
        )
    return value


def phase_for_microstep(compile_scope, gradient_accumulation_steps, index):
    """Return the actual compiled argument structure at one microstep."""

    if (
        type(gradient_accumulation_steps) is not int
        or gradient_accumulation_steps < 1
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if type(index) is not int or index < 0:
        raise ValueError("microstep index must be a non-negative integer")
    position = index % gradient_accumulation_steps
    if compile_scope == FULL_STEP_SCOPE:
        if gradient_accumulation_steps == 1:
            return "single"
        if position == 0:
            return "none_no_update"
        if position == gradient_accumulation_steps - 1:
            return "tree_update"
        return "tree_no_update"
    if compile_scope == DDP_LOCAL_GRAD_SCOPE:
        if gradient_accumulation_steps == 1 or position == 0:
            return "none"
        return "tree"
    raise ValueError(f"unsupported compile scope: {compile_scope!r}")


def _family_key(family):
    representation = repr(family)
    digest = hashlib.blake2b(
        representation.encode("utf-8"), digest_size=8,
    ).hexdigest()
    return digest, representation


def _family_labels(families):
    return {
        family: f"{_family_key(family)[0]}:{index}"
        for index, family in enumerate(sorted(families, key=_family_key))
    }


def _report(
    *,
    action,
    reason,
    cap,
    compile_scope,
    raw_catalog,
    family_data,
    planned_catalog=None,
    endpoints=(),
    padding_tokens=0,
    original_tokens=0,
    configured_cap=None,
    cap_selection="fixed",
    padding_work=0,
    raw_work=0,
    max_width_stretch=1.0,
    budget_satisfied=True,
):
    labels = _family_labels(family_data)
    planned_endpoints = tuple(
        (labels[family], tuple(values))
        for family, values in endpoints
    )
    return TextShapeGuardReport(
        action=action,
        reason=reason,
        cap=cap,
        compile_scope=compile_scope,
        raw_signatures=len(raw_catalog),
        planned_signatures=(
            None if planned_catalog is None else len(planned_catalog)
        ),
        raw_widths=sum(len(widths) for widths in family_data.values()),
        planned_endpoints=planned_endpoints,
        padding_tokens=padding_tokens,
        original_tokens=original_tokens,
        configured_cap=cap if configured_cap is None else configured_cap,
        effective_cap=cap,
        cap_selection=cap_selection,
        padding_work_fraction=(padding_work / raw_work if raw_work else 0.0),
        max_width_stretch=max_width_stretch,
        budget_satisfied=budget_satisfied,
    )


def _collect_problem(events, compile_scope):
    if compile_scope not in (FULL_STEP_SCOPE, DDP_LOCAL_GRAD_SCOPE):
        raise ValueError(f"unsupported compile scope: {compile_scope!r}")
    family_data = {}
    raw_catalog = set()
    original_tokens = 0
    raw_work = 0
    for event in tuple(events):
        if not isinstance(event, TextShapeEvent):
            raise TypeError("events must contain TextShapeEvent values")
        phase_weights = family_data.setdefault(event.family, {}).setdefault(
            event.width, {}
        )
        phase_weights[event.phase] = phase_weights.get(event.phase, 0) + event.weight
        raw_catalog.add((compile_scope, event.phase, event.family, event.width))
        original_tokens += event.weight * event.width
        raw_work += event.weight * event.width * event.width
    return _TextShapeProblem(
        compile_scope=compile_scope,
        family_data=family_data,
        raw_catalog=frozenset(raw_catalog),
        sorted_families=tuple(sorted(family_data, key=_family_key)),
        original_tokens=original_tokens,
        raw_work=raw_work,
    )


def _family_options(width_data, cap, work, *, include_stretch=False):
    """Return unrestricted and optionally stretch-feasible exact options."""

    widths = tuple(sorted(width_data))
    if len(widths) > _MAX_EXACT_WIDTHS_PER_FAMILY:
        raise _PlannerLimit("too_many_widths")
    states = [dict() for _ in range(len(widths) + 1)]
    states[0][0] = (0, ())
    stretch_states = [dict() for _ in range(len(widths) + 1)]
    if include_stretch:
        stretch_states[0][0] = (0, ())
    for end in range(1, len(widths) + 1):
        endpoint = widths[end - 1]
        segment_phases = set()
        segment_weight = 0
        segment_squares = 0
        for start in range(end - 1, -1, -1):
            width = widths[start]
            for phase, weight in width_data[width].items():
                segment_phases.add(phase)
                segment_weight += weight
                segment_squares += weight * width * width
            segment_slots = len(segment_phases)
            segment_cost = endpoint * endpoint * segment_weight - segment_squares
            for previous_slots, (previous_cost, previous_endpoints) in states[start].items():
                work[0] += 1
                if work[0] > _MAX_EXACT_PLANNER_WORK:
                    raise _PlannerLimit("planner_work_limit")
                slots = previous_slots + segment_slots
                if slots > cap:
                    continue
                candidate = (
                    previous_cost + segment_cost,
                    previous_endpoints + (endpoint,),
                )
                current = states[end].get(slots)
                if current is None or candidate < current:
                    states[end][slots] = candidate
            if (
                include_stretch
                and endpoint * MAX_WIDTH_STRETCH_DENOMINATOR
                <= width * MAX_WIDTH_STRETCH_NUMERATOR
            ):
                for previous_slots, (
                    previous_cost, previous_endpoints,
                ) in stretch_states[start].items():
                    work[0] += 1
                    if work[0] > _MAX_EXACT_PLANNER_WORK:
                        raise _PlannerLimit("planner_work_limit")
                    slots = previous_slots + segment_slots
                    if slots > cap:
                        continue
                    candidate = (
                        previous_cost + segment_cost,
                        previous_endpoints + (endpoint,),
                    )
                    current = stretch_states[end].get(slots)
                    if current is None or candidate < current:
                        stretch_states[end][slots] = candidate
    return states[-1], stretch_states[-1]


def _irreducible_signature_count(problem):
    return sum(
        len({
            phase
            for phases in problem.family_data[family].values()
            for phase in phases
        })
        for family in problem.sorted_families
    )


def _build_global_frontier(problem, cap, *, include_stretch=False):
    """Build unrestricted and stretch-feasible exact padding frontiers."""

    irreducible = _irreducible_signature_count(problem)
    if irreducible > cap:
        raise _PlannerLimit("irreducible_signatures")

    work = [0]
    family_options = []
    for family in problem.sorted_families:
        options, stretch_options = _family_options(
            problem.family_data[family],
            cap,
            work,
            include_stretch=include_stretch,
        )
        if not options:
            raise _PlannerLimit("irreducible_signatures")
        family_options.append((options, stretch_options))

    states = {0: (0, ())}
    stretch_states = {0: (0, ())} if include_stretch else {}
    for options, stretch_options in family_options:
        next_states = {}
        for used, (cost, choices) in states.items():
            for slots, (family_cost, endpoints) in options.items():
                work[0] += 1
                if work[0] > _MAX_EXACT_PLANNER_WORK:
                    raise _PlannerLimit("planner_work_limit")
                total = used + slots
                if total > cap:
                    continue
                candidate = (cost + family_cost, choices + (endpoints,))
                current = next_states.get(total)
                if current is None or candidate < current:
                    next_states[total] = candidate
        states = next_states
        if include_stretch:
            next_stretch_states = {}
            for used, (cost, choices) in stretch_states.items():
                for slots, (family_cost, endpoints) in stretch_options.items():
                    work[0] += 1
                    if work[0] > _MAX_EXACT_PLANNER_WORK:
                        raise _PlannerLimit("planner_work_limit")
                    total = used + slots
                    if total > cap:
                        continue
                    candidate = (cost + family_cost, choices + (endpoints,))
                    current = next_stretch_states.get(total)
                    if current is None or candidate < current:
                        next_stretch_states[total] = candidate
            stretch_states = next_stretch_states
    if not states:
        raise _PlannerLimit("irreducible_signatures")
    return states, stretch_states


def _build_padding_budget_frontier(problem, cap, *, stop_at_first):
    """Build exact stretch-feasible points in increasing signature count."""

    max_padding_cost = (
        problem.raw_work * MAX_PADDING_WORK_PERCENT // 100
    )
    work = [0]
    planners = [
        _FamilyBudgetPlanner(
            problem.family_data[family], max_padding_cost, work,
        )
        for family in problem.sorted_families
    ]
    irreducible = sum(planner.irreducible_slots for planner in planners)
    if irreducible > cap:
        raise _PlannerLimit("irreducible_signatures")

    points = []
    for total_slots in range(irreducible, cap + 1):
        for planner in planners:
            other_minimum = irreducible - planner.irreducible_slots
            planner.extend(total_slots - other_minimum)

        states = {0: (0, ())}
        for planner in planners:
            next_states = {}
            for used, (cost, choices) in states.items():
                for slots, (family_cost, endpoints) in planner.options.items():
                    total = used + slots
                    if total > total_slots:
                        continue
                    work[0] += 1
                    if work[0] > _MAX_EXACT_PLANNER_WORK:
                        raise _PlannerLimit("planner_work_limit")
                    candidate_cost = cost + family_cost
                    if candidate_cost > max_padding_cost:
                        continue
                    candidate = (
                        candidate_cost,
                        choices + (endpoints,),
                    )
                    current = next_states.get(total)
                    if current is None or candidate < current:
                        next_states[total] = candidate
            states = next_states
        if total_slots not in states:
            continue
        cost, choices = states[total_slots]
        points.append((total_slots, cost, choices))
        if stop_at_first:
            break
    return tuple(points)


def _global_combination_work_bound(problem, cap):
    """Conservative transition bound for global family combination."""

    family_ranges = []
    for family in problem.sorted_families:
        width_data = problem.family_data[family]
        minimum = len({
            phase for phases in width_data.values() for phase in phases
        })
        maximum = min(
            cap, sum(len(phases) for phases in width_data.values()),
        )
        family_ranges.append((minimum, maximum))
    irreducible = sum(minimum for minimum, _ in family_ranges)
    work = 0
    for total in range(irreducible, cap + 1):
        used_min = used_max = 0
        for minimum, maximum in family_ranges:
            used_count = max(0, min(used_max, total) - used_min + 1)
            option_count = max(0, min(maximum, total) - minimum + 1)
            work += used_count * option_count
            used_min += minimum
            used_max = min(total, used_max + maximum)
    return work


def _padding_budget_work_bound(problem, cap):
    """Bound the slot-major DP scans before allocating its state layers."""

    irreducible = _irreducible_signature_count(problem)
    work = _global_combination_work_bound(problem, cap)
    for family in problem.sorted_families:
        width_data = problem.family_data[family]
        widths = tuple(sorted(width_data))
        if len(widths) > _MAX_EXACT_WIDTHS_PER_FAMILY:
            return _MAX_EXACT_PLANNER_WORK + 1
        family_minimum = len({
            phase for phases in width_data.values() for phase in phases
        })
        family_maximum = sum(len(phases) for phases in width_data.values())
        layers = min(cap - (irreducible - family_minimum), family_maximum)
        scans = 0
        for end, endpoint in enumerate(widths):
            first = bisect_left(widths, (endpoint * 2 + 2) // 3, 0, end + 1)
            scans += end - first + 1
            if first:
                scans += 1
        work += max(0, layers) * scans
        if work > _MAX_EXACT_PLANNER_WORK:
            return work
    return work


def _unrestricted_work_bound(problem, cap):
    """Conservative bound for the lazy unrestricted ceiling planner."""

    work = _global_combination_work_bound(problem, cap)
    for family in problem.sorted_families:
        width_data = problem.family_data[family]
        width_count = len(width_data)
        if width_count > _MAX_EXACT_WIDTHS_PER_FAMILY:
            return _MAX_EXACT_PLANNER_WORK + 1
        maximum = min(
            cap, sum(len(phases) for phases in width_data.values()),
        )
        work += width_count * (width_count + 1) // 2 * maximum
        if work > _MAX_EXACT_PLANNER_WORK:
            return work
    return work


def _fallback_choices(problem, segments):
    choices = [[] for _ in problem.sorted_families]
    active = sorted(
        (segment for segment in segments.values() if segment.active),
        key=lambda segment: (segment.family_index, segment.start),
    )
    for segment in active:
        choices[segment.family_index].append(segment.endpoint)
    return tuple(tuple(endpoints) for endpoints in choices)


def _build_bounded_fallback_frontier(problem, cap):
    """Build deterministic adjacent-merge snapshots without exact DP."""

    if _irreducible_signature_count(problem) > cap:
        raise _PlannerLimit("irreducible_signatures")
    segments = {}
    next_identifier = 0
    heap = []
    family_widths = tuple(
        tuple(sorted(problem.family_data[family]))
        for family in problem.sorted_families
    )

    def push_candidate(left_identifier):
        left = segments[left_identifier]
        if not left.active or left.next is None:
            return
        right_identifier = left.next
        right = segments[right_identifier]
        if not right.active or right.family_index != left.family_index:
            return
        phases = left.phases | right.phases
        reduction = len(left.phases) + len(right.phases) - len(phases)
        weight = left.weight + right.weight
        squares = left.squares + right.squares
        cost = right.endpoint * right.endpoint * weight - squares
        delta = cost - left.cost - right.cost
        violates_stretch = (
            right.endpoint * MAX_WIDTH_STRETCH_DENOMINATOR
            > family_widths[left.family_index][left.start]
            * MAX_WIDTH_STRETCH_NUMERATOR
        )
        priority = (
            1 if violates_stretch else 0,
            0 if reduction else 1,
            Fraction(delta, reduction) if reduction else Fraction(0, 1),
            delta,
            left.family_index,
            left.start,
            right.end,
            left_identifier,
            right_identifier,
        )
        heapq.heappush(heap, priority)

    previous_identifier = None
    previous_family = None
    for family_index, family in enumerate(problem.sorted_families):
        for position, width in enumerate(sorted(problem.family_data[family])):
            phase_weights = problem.family_data[family][width]
            weight = sum(phase_weights.values())
            identifier = next_identifier
            next_identifier += 1
            previous = (
                previous_identifier if previous_family == family_index else None
            )
            segments[identifier] = _FallbackSegment(
                family_index=family_index,
                start=position,
                end=position,
                endpoint=width,
                phases=frozenset(phase_weights),
                weight=weight,
                squares=weight * width * width,
                cost=0,
                previous=previous,
                next=None,
            )
            if previous is not None:
                segments[previous].next = identifier
                push_candidate(previous)
            previous_identifier = identifier
            previous_family = family_index

    current_slots = len(problem.raw_catalog)
    current_cost = 0
    points = []
    recorded_slots = set()
    while True:
        if current_slots <= cap and current_slots not in recorded_slots:
            points.append((
                current_slots,
                current_cost,
                _fallback_choices(problem, segments),
            ))
            recorded_slots.add(current_slots)
        while heap:
            *_, left_identifier, right_identifier = heapq.heappop(heap)
            left = segments[left_identifier]
            right = segments[right_identifier]
            if (
                left.active
                and right.active
                and left.next == right_identifier
                and right.previous == left_identifier
            ):
                break
        else:
            break

        phases = left.phases | right.phases
        reduction = len(left.phases) + len(right.phases) - len(phases)
        weight = left.weight + right.weight
        squares = left.squares + right.squares
        cost = right.endpoint * right.endpoint * weight - squares
        delta = cost - left.cost - right.cost
        new_identifier = next_identifier
        next_identifier += 1
        segments[new_identifier] = _FallbackSegment(
            family_index=left.family_index,
            start=left.start,
            end=right.end,
            endpoint=right.endpoint,
            phases=phases,
            weight=weight,
            squares=squares,
            cost=cost,
            previous=left.previous,
            next=right.next,
        )
        left.active = right.active = False
        if left.previous is not None:
            segments[left.previous].next = new_identifier
            push_candidate(left.previous)
        if right.next is not None:
            segments[right.next].previous = new_identifier
        push_candidate(new_identifier)
        current_slots -= reduction
        current_cost += delta

    if current_slots != _irreducible_signature_count(problem):
        raise AssertionError("fallback planner did not reach its structural minimum")
    if not points:
        raise _PlannerLimit("irreducible_signatures")
    return tuple(points)


def _select_fallback_point(problem, points):
    valid = [point for point in points if _point_within_budgets(problem, point)]
    if valid:
        return min(valid, key=lambda point: (point[0], point[1], point[2]))
    return min(points, key=lambda point: (point[1], point[0], point[2]))


def _eager_plan(
    problem, reason, cap, *, configured_cap=None, cap_selection="fixed",
):
    report = _report(
        action="eager",
        reason=reason,
        cap=cap,
        compile_scope=problem.compile_scope,
        raw_catalog=problem.raw_catalog,
        family_data=problem.family_data,
        original_tokens=problem.original_tokens,
        configured_cap=configured_cap,
        cap_selection=cap_selection,
        raw_work=problem.raw_work,
        budget_satisfied=False,
    )
    return TextShapePlan(report, problem.raw_catalog, frozenset())


def _exact_plan(problem, cap, *, configured_cap=None, cap_selection="fixed"):
    endpoints = tuple(
        (family, tuple(sorted(problem.family_data[family])))
        for family in problem.sorted_families
    )
    report = _report(
        action="exact",
        reason="schedule_within_cap",
        cap=cap,
        compile_scope=problem.compile_scope,
        raw_catalog=problem.raw_catalog,
        family_data=problem.family_data,
        planned_catalog=problem.raw_catalog,
        endpoints=endpoints,
        original_tokens=problem.original_tokens,
        configured_cap=configured_cap,
        cap_selection=cap_selection,
        raw_work=problem.raw_work,
    )
    return TextShapePlan(report, problem.raw_catalog, problem.raw_catalog)


def _choice_metrics(problem, choices):
    endpoint_lookup = {}
    max_numerator = 1
    max_denominator = 1
    stretch_within_budget = True
    for index, family in enumerate(problem.sorted_families):
        family_endpoints = choices[index]
        for width in sorted(problem.family_data[family]):
            endpoint = next(value for value in family_endpoints if width <= value)
            if endpoint < width:
                raise AssertionError("endpoint mapping truncated a sequence")
            endpoint_lookup[(family, width)] = endpoint
            if endpoint * max_denominator > max_numerator * width:
                max_numerator, max_denominator = endpoint, width
            if (
                endpoint * MAX_WIDTH_STRETCH_DENOMINATOR
                > width * MAX_WIDTH_STRETCH_NUMERATOR
            ):
                stretch_within_budget = False
    return (
        endpoint_lookup,
        max_numerator / max_denominator,
        stretch_within_budget,
    )


def _point_within_budgets(problem, point):
    _slots, padding_cost, choices = point
    return (
        padding_cost * 100
        <= problem.raw_work * MAX_PADDING_WORK_PERCENT
        and _choice_metrics(problem, choices)[2]
    )


def _materialize_choice(
    problem,
    slots,
    padding_cost,
    choices,
    *,
    cap,
    configured_cap=None,
    cap_selection="fixed",
):
    endpoints = tuple(
        (family, choices[index])
        for index, family in enumerate(problem.sorted_families)
    )
    endpoint_lookup, max_width_stretch, stretch_within_budget = (
        _choice_metrics(problem, choices)
    )
    endpoint_maps = tuple(
        (
            family,
            tuple(
                (width, endpoint_lookup[(family, width)])
                for width in sorted(problem.family_data[family])
            ),
        )
        for family in problem.sorted_families
    )
    planned_catalog = frozenset({
        (scope, phase, family, endpoint_lookup[(family, width)])
        for scope, phase, family, width in problem.raw_catalog
    })
    if len(planned_catalog) != slots or len(planned_catalog) > cap:
        raise AssertionError("planned signature catalog exceeds its cap")

    padding_tokens = 0
    reconstructed_cost = 0
    for family in problem.sorted_families:
        for width, phase_weights in problem.family_data[family].items():
            endpoint = endpoint_lookup[(family, width)]
            weight = sum(phase_weights.values())
            padding_tokens += (endpoint - width) * weight
            reconstructed_cost += (endpoint * endpoint - width * width) * weight
    if reconstructed_cost != padding_cost:
        raise AssertionError("planner padding cost reconstruction failed")

    work_within_budget = (
        padding_cost * 100
        <= problem.raw_work * MAX_PADDING_WORK_PERCENT
    )
    budget_satisfied = work_within_budget and stretch_within_budget
    is_exact = planned_catalog == problem.raw_catalog and padding_cost == 0
    report = _report(
        action="exact" if is_exact else "bucket",
        reason="schedule_within_cap" if is_exact else "schedule_exceeds_cap",
        cap=cap,
        compile_scope=problem.compile_scope,
        raw_catalog=problem.raw_catalog,
        family_data=problem.family_data,
        planned_catalog=planned_catalog,
        endpoints=endpoints,
        padding_tokens=padding_tokens,
        original_tokens=problem.original_tokens,
        configured_cap=configured_cap,
        cap_selection=cap_selection,
        padding_work=padding_cost,
        raw_work=problem.raw_work,
        max_width_stretch=max_width_stretch,
        budget_satisfied=budget_satisfied,
    )
    return TextShapePlan(
        report=report,
        raw_catalog=problem.raw_catalog,
        planned_catalog=planned_catalog,
        endpoint_maps=() if is_exact else endpoint_maps,
        padding_cost=padding_cost,
    )


def build_text_shape_frontier(events, *, compile_scope):
    """Build bounded automatic planning state for ``events``."""

    problem = _collect_problem(events, compile_scope)
    if not problem.raw_catalog:
        return TextShapeFrontier(problem, failure_reason="empty_schedule")
    if len(problem.raw_catalog) <= SMALL_EXACT_SIGNATURE_THRESHOLD:
        return TextShapeFrontier(problem)
    try:
        fallback_points = _build_bounded_fallback_frontier(
            problem, AUTOMATIC_TEXT_COMPILE_CEILING,
        )
    except _PlannerLimit as exc:
        return TextShapeFrontier(problem, failure_reason=str(exc))
    fallback_frontier = TextShapeFrontier(
        problem, _fallback_points=fallback_points,
    )
    fallback_point = _select_fallback_point(problem, fallback_points)
    exact_target = (
        fallback_point[0]
        if _point_within_budgets(problem, fallback_point)
        else AUTOMATIC_TEXT_COMPILE_CEILING
    )
    if (
        _padding_budget_work_bound(problem, exact_target)
        > _MAX_EXACT_PLANNER_WORK
    ):
        return fallback_frontier
    try:
        stretch_points = _build_padding_budget_frontier(
            problem,
            AUTOMATIC_TEXT_COMPILE_CEILING,
            stop_at_first=True,
        )
        if stretch_points:
            return TextShapeFrontier(
                problem,
                _stretch_points=stretch_points,
            )
        if (
            _unrestricted_work_bound(
                problem, AUTOMATIC_TEXT_COMPILE_CEILING,
            )
            > _MAX_EXACT_PLANNER_WORK
        ):
            return fallback_frontier
        states, _ = _build_global_frontier(
            problem, AUTOMATIC_TEXT_COMPILE_CEILING,
        )
    except _PlannerLimit as exc:
        if str(exc) in ("planner_work_limit", "too_many_widths"):
            return fallback_frontier
        return TextShapeFrontier(problem, failure_reason=str(exc))
    points = tuple(
        (slots, cost, choices)
        for slots, (cost, choices) in sorted(states.items())
    )
    return TextShapeFrontier(problem, points)


def materialize_text_shape_frontier(
    frontier,
    *,
    cap,
    cap_selection="padding_budget",
):
    """Materialize the minimum-padding local plan under a shared auto cap."""

    if not isinstance(frontier, TextShapeFrontier):
        raise TypeError("frontier must be a TextShapeFrontier")
    if type(cap) is not int or not 1 <= cap <= AUTOMATIC_TEXT_COMPILE_CEILING:
        raise ValueError(
            "automatic effective cap must be an integer from 1 through "
            f"{AUTOMATIC_TEXT_COMPILE_CEILING}"
        )
    problem = frontier._problem
    if frontier.failure_reason is not None:
        return _eager_plan(
            problem,
            frontier.failure_reason,
            cap,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="fallback",
        )
    if len(problem.raw_catalog) <= cap:
        return _exact_plan(
            problem,
            cap,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection=cap_selection,
        )
    if frontier._fallback_points:
        candidates = [
            point for point in frontier._fallback_points if point[0] <= cap
        ]
        if not candidates:
            return _eager_plan(
                problem,
                "irreducible_signatures",
                cap,
                configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
                cap_selection="fallback",
            )
        slots, padding_cost, choices = min(
            candidates,
            key=lambda point: (point[1], point[0], point[2]),
        )
        return _materialize_choice(
            problem,
            slots,
            padding_cost,
            choices,
            cap=cap,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="fallback",
        )
    if frontier._stretch_points:
        budget_points = frontier._stretch_points
        if budget_points[-1][0] < cap:
            if (
                _padding_budget_work_bound(problem, cap)
                <= _MAX_EXACT_PLANNER_WORK
            ):
                try:
                    budget_points = _build_padding_budget_frontier(
                        problem, cap, stop_at_first=False,
                    )
                except _PlannerLimit:
                    budget_points = frontier._stretch_points
        candidates = [point for point in budget_points if point[0] <= cap]
        if candidates:
            slots, padding_cost, choices = min(
                candidates,
                key=lambda point: (point[1], point[0], point[2]),
            )
            return _materialize_choice(
                problem,
                slots,
                padding_cost,
                choices,
                cap=cap,
                configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
                cap_selection=cap_selection,
            )
    points = frontier._points
    if not points:
        try:
            states, _ = _build_global_frontier(problem, cap)
        except _PlannerLimit as exc:
            return _eager_plan(
                problem,
                str(exc),
                cap,
                configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
                cap_selection="fallback",
            )
        points = tuple(
            (slots, cost, choices)
            for slots, (cost, choices) in sorted(states.items())
        )
    candidates = [point for point in points if point[0] <= cap]
    if not candidates:
        return _eager_plan(
            problem,
            "irreducible_signatures",
            cap,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="fallback",
        )
    slots, padding_cost, choices = min(
        candidates,
        key=lambda point: (point[1], point[0], point[2]),
    )
    return _materialize_choice(
        problem,
        slots,
        padding_cost,
        choices,
        cap=cap,
        configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
        cap_selection=cap_selection,
    )


def select_text_shape_padding_budget(frontier):
    """Select the smallest exact-frontier point within both policy budgets."""

    if not isinstance(frontier, TextShapeFrontier):
        raise TypeError("frontier must be a TextShapeFrontier")
    problem = frontier._problem
    if frontier.failure_reason is not None:
        return _eager_plan(
            problem,
            frontier.failure_reason,
            AUTOMATIC_TEXT_COMPILE_CEILING,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="fallback",
        )
    if len(problem.raw_catalog) <= SMALL_EXACT_SIGNATURE_THRESHOLD:
        return _exact_plan(
            problem,
            len(problem.raw_catalog),
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="exact",
        )
    if frontier._fallback_points:
        slots, padding_cost, choices = _select_fallback_point(
            problem, frontier._fallback_points,
        )
        return _materialize_choice(
            problem,
            slots,
            padding_cost,
            choices,
            cap=slots,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="fallback",
        )

    selected = None
    for point in frontier._stretch_points:
        slots, padding_cost, choices = point
        if _point_within_budgets(problem, point):
            selected = point
            break
    if selected is not None:
        slots, padding_cost, choices = selected
        return _materialize_choice(
            problem,
            slots,
            padding_cost,
            choices,
            cap=slots,
            configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
            cap_selection="padding_budget",
        )

    slots, padding_cost, choices = min(
        frontier._points,
        key=lambda point: (point[1], point[0], point[2]),
    )
    return _materialize_choice(
        problem,
        slots,
        padding_cost,
        choices,
        cap=slots,
        configured_cap=AUTOMATIC_TEXT_COMPILE_CEILING,
        cap_selection="ceiling",
    )


def plan_text_shape_padding_budget(events, *, compile_scope):
    """Build one frontier and select the deterministic automatic text cap."""

    return select_text_shape_padding_budget(
        build_text_shape_frontier(events, compile_scope=compile_scope)
    )


def plan_text_shape_buckets(events, *, cap, compile_scope):
    """Choose the minimum-cost upward endpoint map under a signature cap."""

    cap = resolve_compile_max_variants(cap)
    problem = _collect_problem(events, compile_scope)
    if not problem.raw_catalog:
        return _eager_plan(problem, "empty_schedule", cap)
    if len(problem.raw_catalog) <= cap:
        return _exact_plan(problem, cap)
    try:
        states, _stretch_states = _build_global_frontier(problem, cap)
    except _PlannerLimit as exc:
        return _eager_plan(problem, str(exc), cap)
    slots, (padding_cost, choices) = min(
        states.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    return _materialize_choice(
        problem, slots, padding_cost, choices, cap=cap,
    )


__all__ = (
    "AUTOMATIC_TEXT_COMPILE_CEILING",
    "DEFAULT_TEXT_COMPILE_MAX_VARIANTS",
    "SMALL_EXACT_SIGNATURE_THRESHOLD",
    "MAX_PADDING_WORK_PERCENT",
    "MAX_WIDTH_STRETCH_NUMERATOR",
    "MAX_WIDTH_STRETCH_DENOMINATOR",
    "MAX_COMPILE_VARIANTS",
    "FULL_STEP_SCOPE",
    "DDP_LOCAL_GRAD_SCOPE",
    "TextShapeEvent",
    "TextShapeFrontier",
    "TextShapeGuardReport",
    "TextShapePlan",
    "resolve_compile_max_variants",
    "phase_for_microstep",
    "build_text_shape_frontier",
    "materialize_text_shape_frontier",
    "select_text_shape_padding_budget",
    "plan_text_shape_buckets",
    "plan_text_shape_padding_budget",
)
