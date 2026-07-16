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

from dataclasses import asdict, dataclass
import hashlib


DEFAULT_TEXT_COMPILE_MAX_VARIANTS = 32
MAX_COMPILE_VARIANTS = 256
MAX_WIDTHS_PER_FAMILY = 2_048
MAX_PLANNER_WORK = 16_000_000

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
    )


def _family_options(width_data, cap, work):
    """Return minimum ``(cost, endpoints)`` for every exact slot count."""

    widths = tuple(sorted(width_data))
    if len(widths) > MAX_WIDTHS_PER_FAMILY:
        raise _PlannerLimit("too_many_widths")
    states = [dict() for _ in range(len(widths) + 1)]
    states[0][0] = (0, ())
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
                if work[0] > MAX_PLANNER_WORK:
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
    return states[-1]


def _eager_plan(reason, cap, compile_scope, raw_catalog, family_data, original_tokens):
    report = _report(
        action="eager",
        reason=reason,
        cap=cap,
        compile_scope=compile_scope,
        raw_catalog=raw_catalog,
        family_data=family_data,
        original_tokens=original_tokens,
    )
    return TextShapePlan(report, frozenset(raw_catalog), frozenset())


def plan_text_shape_buckets(events, *, cap, compile_scope):
    """Choose the minimum-cost upward endpoint map under a signature cap."""

    cap = resolve_compile_max_variants(cap)
    if compile_scope not in (FULL_STEP_SCOPE, DDP_LOCAL_GRAD_SCOPE):
        raise ValueError(f"unsupported compile scope: {compile_scope!r}")
    events = tuple(events)
    if not events:
        return _eager_plan("empty_schedule", cap, compile_scope, (), {}, 0)

    family_data = {}
    raw_catalog = set()
    original_tokens = 0
    for event in events:
        if not isinstance(event, TextShapeEvent):
            raise TypeError("events must contain TextShapeEvent values")
        phase_weights = family_data.setdefault(event.family, {}).setdefault(
            event.width, {}
        )
        phase_weights[event.phase] = phase_weights.get(event.phase, 0) + event.weight
        raw_catalog.add((compile_scope, event.phase, event.family, event.width))
        original_tokens += event.weight * event.width

    sorted_families = tuple(sorted(family_data, key=_family_key))
    if len(raw_catalog) <= cap:
        raw_endpoints = tuple(
            (family, tuple(sorted(family_data[family])))
            for family in sorted_families
        )
        report = _report(
            action="exact",
            reason="schedule_within_cap",
            cap=cap,
            compile_scope=compile_scope,
            raw_catalog=raw_catalog,
            family_data=family_data,
            planned_catalog=raw_catalog,
            endpoints=raw_endpoints,
            original_tokens=original_tokens,
        )
        return TextShapePlan(
            report,
            frozenset(raw_catalog),
            frozenset(raw_catalog),
        )

    irreducible = sum(
        len({phase for phases in family_data[family].values() for phase in phases})
        for family in sorted_families
    )
    if irreducible > cap:
        return _eager_plan(
            "irreducible_signatures", cap, compile_scope,
            raw_catalog, family_data, original_tokens,
        )

    work = [0]
    family_options = []
    try:
        for family in sorted_families:
            options = _family_options(family_data[family], cap, work)
            if not options:
                return _eager_plan(
                    "irreducible_signatures", cap, compile_scope,
                    raw_catalog, family_data, original_tokens,
                )
            family_options.append((family, options))

        states = {0: (0, ())}
        for _family, options in family_options:
            next_states = {}
            for used, (cost, choices) in states.items():
                for slots, (family_cost, endpoints) in options.items():
                    work[0] += 1
                    if work[0] > MAX_PLANNER_WORK:
                        raise _PlannerLimit("planner_work_limit")
                    total = used + slots
                    if total > cap:
                        continue
                    candidate = (cost + family_cost, choices + (endpoints,))
                    current = next_states.get(total)
                    if current is None or candidate < current:
                        next_states[total] = candidate
            states = next_states
    except _PlannerLimit as exc:
        return _eager_plan(
            str(exc), cap, compile_scope,
            raw_catalog, family_data, original_tokens,
        )

    if not states:
        return _eager_plan(
            "irreducible_signatures", cap, compile_scope,
            raw_catalog, family_data, original_tokens,
        )
    slots, (padding_cost, choices) = min(
        states.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    endpoints = tuple(
        (family, choices[index])
        for index, family in enumerate(sorted_families)
    )
    endpoint_maps = []
    endpoint_lookup = {}
    for family, family_endpoints in endpoints:
        mapping = []
        for width in sorted(family_data[family]):
            endpoint = next(value for value in family_endpoints if width <= value)
            if endpoint < width:
                raise AssertionError("endpoint mapping truncated a sequence")
            mapping.append((width, endpoint))
            endpoint_lookup[(family, width)] = endpoint
        endpoint_maps.append((family, tuple(mapping)))

    planned_catalog = {
        (scope, phase, family, endpoint_lookup[(family, width)])
        for scope, phase, family, width in raw_catalog
    }
    if len(planned_catalog) != slots or len(planned_catalog) > cap:
        raise AssertionError("planned signature catalog exceeds its cap")

    padding_tokens = 0
    reconstructed_cost = 0
    for family in sorted_families:
        for width, phase_weights in family_data[family].items():
            endpoint = endpoint_lookup[(family, width)]
            weight = sum(phase_weights.values())
            padding_tokens += (endpoint - width) * weight
            reconstructed_cost += (endpoint * endpoint - width * width) * weight
    if reconstructed_cost != padding_cost:
        raise AssertionError("planner padding cost reconstruction failed")

    report = _report(
        action="bucket",
        reason="schedule_exceeds_cap",
        cap=cap,
        compile_scope=compile_scope,
        raw_catalog=raw_catalog,
        family_data=family_data,
        planned_catalog=planned_catalog,
        endpoints=endpoints,
        padding_tokens=padding_tokens,
        original_tokens=original_tokens,
    )
    return TextShapePlan(
        report=report,
        raw_catalog=frozenset(raw_catalog),
        planned_catalog=frozenset(planned_catalog),
        endpoint_maps=tuple(endpoint_maps),
        padding_cost=padding_cost,
    )


__all__ = (
    "DEFAULT_TEXT_COMPILE_MAX_VARIANTS",
    "MAX_COMPILE_VARIANTS",
    "MAX_WIDTHS_PER_FAMILY",
    "MAX_PLANNER_WORK",
    "FULL_STEP_SCOPE",
    "DDP_LOCAL_GRAD_SCOPE",
    "TextShapeEvent",
    "TextShapeGuardReport",
    "TextShapePlan",
    "resolve_compile_max_variants",
    "phase_for_microstep",
    "plan_text_shape_buckets",
)
