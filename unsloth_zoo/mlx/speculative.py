# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Speculative decoding for MLX generation: draft sources and the policy choosing between them."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Sequence

import mlx.core as mx

from .generate import SamplingParams

__all__ = [
    "DraftController",
    "EngineRow",
    "NgramProposer",
    "ReplyStats",
    "RoundPlan",
    "RowPlan",
    "RowState",
    "SpeculativeEngine",
    "StepOutput",
]


class NgramProposer:
    """Prompt lookup: continue the prompt from where the reply's last tokens occur in it.

    Only the prompt is indexed. A reply repeating itself is a poor predictor of what it
    says next, while a reply quoting the prompt (code under edit, a document being
    summarised) keeps quoting it.
    """

    def __init__(
        self,
        prompt_tokens: Sequence[int],
        *,
        min_ngram: int = 4,
        max_extension: int = 8,
        max_candidates: int = 32,
    ):
        if min_ngram < 1:
            raise ValueError("min_ngram must be at least 1")
        self.prompt = [int(token) for token in prompt_tokens]
        self.min_ngram = int(min_ngram)
        self.max_extension = int(max_extension)
        self.max_candidates = int(max_candidates)
        self._index: dict[tuple[int, ...], list[int]] = {}
        n = self.min_ngram
        for end in range(n, len(self.prompt)):
            self._index.setdefault(tuple(self.prompt[end - n : end]), []).append(end)

    def propose(self, history: Sequence[int], limit: int) -> list[int]:
        """Up to ``limit`` tokens following the best prompt match of ``history``'s tail."""
        n = self.min_ngram
        if limit <= 0 or len(history) < n:
            return []
        candidates = self._index.get(tuple(int(token) for token in history[-n:]))
        if not candidates:
            return []
        best, best_extension = None, -1
        for end in reversed(candidates[-self.max_candidates :]):
            extension = 0
            while (
                extension < self.max_extension
                and end - n - 1 - extension >= 0
                and len(history) - n - 1 - extension >= 0
                and self.prompt[end - n - 1 - extension]
                == int(history[len(history) - n - 1 - extension])
            ):
                extension += 1
            # Newest wins ties: candidates are walked newest first.
            if extension > best_extension:
                best, best_extension = end, extension
        return self.prompt[best : best + limit]


@dataclass(frozen = True)
class RowPlan:
    source: Literal["none", "copy", "draft"] = "none"
    length: int = 0


@dataclass(frozen = True)
class RoundPlan:
    """A plain window of ``length`` steps, or one verify round over ``rows``.

    ``split`` asks the engine to evaluate drafts and catch-up before building the verify, so
    the round's components can be timed apart.
    """

    kind: Literal["plain", "round"]
    length: int = 0
    rows: tuple[RowPlan, ...] = ()
    split: bool = False

    @property
    def width(self) -> int:
        return 1 + max((row.length for row in self.rows), default = 0)


class _Ema:
    """Running mean until ``1 / alpha`` samples, then exponential; a prior counts as one sample."""

    __slots__ = ("value", "count")

    def __init__(self, prior: float | None = None):
        self.value = prior
        self.count = 0 if prior is None else 1

    def update(self, sample: float, alpha: float) -> None:
        self.count += 1
        rate = max(alpha, 1.0 / self.count)
        self.value = sample if self.value is None else self.value + rate * (sample - self.value)


class ReplyStats:
    """One reply's acceptance, seeded from what the load has seen so far."""

    def __init__(self, draft: list[_Ema], copy: _Ema, backoff: int):
        self.draft = [_Ema(ema.value) for ema in draft]
        self.draft_seen = [ema.count > 1 for ema in draft]
        self.copy = _Ema(copy.value)
        self.tokens = 0
        # Per source: the doubling back-off, and the token count to retry an unused source at.
        self.backoff = {"copy": backoff, "draft": backoff}
        self.probe_at = {"copy": 0, "draft": 0}

    def draft_rate(self, position: int) -> float:
        # A position nobody has reached is assumed no better than the one before it.
        if position == 0 or self.draft_seen[position]:
            return self.draft[position].value
        return min(self.draft[position].value, self.draft_rate(position - 1))

    def expected(self, row: RowPlan) -> float:
        expected, reach = 1.0, 1.0
        for position in range(row.length):
            reach *= self.draft_rate(position) if row.source == "draft" else self.copy.value
            expected += reach
        return expected


@dataclass
class RowState:
    """A row as the engine sees it at a round boundary."""

    stats: ReplyStats
    can_draft: bool = True
    copy_available: int = 0
    remaining: int | None = None
    catch_up: int = 0


def _bucket(batch: int) -> int:
    return 1 if batch <= 1 else min(16, 1 << (batch - 1).bit_length())


class DraftController:
    """Chooses each round by expected tokens per second, measured on this machine.

    A round verifies ``[pending, draft...]`` for every row of the batch at one width; each
    row's draft comes from the drafter, from a copy of its prompt, or from nothing. It
    competes with a window of plain decoding. Acceptance is kept per reply (conditional per
    drafted position, one rate for copies), seeded from the load's aggregate. Time is kept per
    component, since the components move independently: verify by batch size and width,
    drafting per row by depth, drafter catch-up per token, plain decoding per step by batch
    size. A candidate is worth its expected emitted tokens over its expected seconds, so a
    drafter forward or a catch-up is paid for only where it earns more than a free copy.

    Plain decoding is timed as the pipelined decode it is, so a drafter that loses to ordinary
    decoding at some batch size converges to ordinary decoding there. Unmeasured entries are
    extrapolated; close or stale rivals are re-measured on a token clock.
    """

    def __init__(
        self,
        *,
        max_depth: int,
        max_copy: int = 16,
        can_copy: bool = True,
        acceptance_alpha: float = 0.05,
        cost_alpha: float = 0.2,
        hysteresis: float = 1.03,
        probe_margin: float = 1.15,
        probe_every: int = 64,
        probe_rounds: int = 3,
        split_every: int = 8,
        stale_after: int = 1024,
        max_stale_after: int = 16384,
        min_window: int = 8,
        max_window: int = 256,
        probe_backoff: int = 32,
        max_probe_backoff: int = 2048,
    ):
        self.max_depth = max(0, int(max_depth))
        self.max_copy = max(0, int(max_copy)) if can_copy else 0
        self.max_width = 1 + max(self.max_depth, self.max_copy)
        self.acceptance_alpha = acceptance_alpha
        self.cost_alpha = cost_alpha
        self.hysteresis = hysteresis
        self.probe_margin = probe_margin
        self.probe_every = probe_every
        self.probe_rounds = probe_rounds
        self.split_every = split_every
        self._since_split = 0
        self.stale_after = stale_after
        self.max_stale_after = max_stale_after
        self._stale_gap: dict[tuple, int] = {}
        self.min_window = min_window
        self.max_window = max_window
        self._base_backoff = probe_backoff
        self._max_backoff = max_probe_backoff
        self.draft_acceptance = [_Ema(self._PRIOR_ACCEPTANCE) for _ in range(self.max_depth)]
        self.copy_acceptance = _Ema(self._PRIOR_ACCEPTANCE)
        self.verify_cost: dict[int, dict[int, _Ema]] = {}
        self.plain_cost: dict[int, _Ema] = {}
        self.draft_cost = {depth: _Ema() for depth in range(1, self.max_depth + 1)}
        self.catch_up_cost = _Ema()
        self.tokens = 0
        # Probes and staleness run on sequence positions, so a wide batch probes no more often.
        self.steps = 0.0
        self._measured_at: dict[tuple, int] = {}
        self._next_probe = probe_every
        self._warmup = list(range(self.max_depth, 0, -1))
        self._current: tuple | None = None
        self._window = min_window
        self._probing_sources: dict[int, str] = {}
        self._probe: tuple | None = None
        self._probe_left = 0

    _PRIOR_ACCEPTANCE = 0.7
    # Priors until measured, relative to a plain step: a round pays a sync more than a plain
    # step, a verified row a little, a drafted token a drafter forward, a catch-up token less.
    _ROUND_OVERHEAD = 1.2
    _VERIFY_SLOPE = 0.03
    _DRAFT_SLOPE = 0.25
    _CATCH_UP = 0.1
    _COPY_PROBE = 4

    def new_reply(self) -> ReplyStats:
        return ReplyStats(self.draft_acceptance, self.copy_acceptance, self._base_backoff)

    def _plain_step(self, bucket: int) -> float:
        measured = {b: ema.value for b, ema in self.plain_cost.items() if ema.value is not None}
        if bucket in measured:
            return measured[bucket]
        if not measured:
            return 1.0
        nearest = min(measured, key = lambda b: abs(b.bit_length() - bucket.bit_length()))
        return measured[nearest]

    @staticmethod
    def _extrapolate(points: list[tuple[int, float]], key: int, slope_floor: float) -> float:
        below = [point for point in points if point[0] <= key]
        above = [point for point in points if point[0] >= key]
        if below and above:
            (k0, t0), (k1, t1) = below[-1], above[0]
            return t0 if k1 == k0 else t0 + (t1 - t0) * (key - k0) / (k1 - k0)
        slope = slope_floor
        if len(points) >= 2:
            (k0, t0), (k1, t1) = points[0], points[-1]
            slope = max(slope, (t1 - t0) / (k1 - k0))
        anchor = below[-1] if below else above[0]
        return max(anchor[1] + slope * (key - anchor[0]), 1e-9)

    def _measured(self, table: dict[int, _Ema]) -> list[tuple[int, float]]:
        return [(key, ema.value) for key, ema in sorted(table.items()) if ema.value is not None]

    def _verify_seconds(self, bucket: int, width: int) -> float:
        step = self._plain_step(bucket)
        slope = step * self._VERIFY_SLOPE * bucket
        points = self._measured(self.verify_cost.get(bucket, {}))
        if not points:
            return step * self._ROUND_OVERHEAD + slope * (width - 1)
        return self._extrapolate(points, width, slope)

    def _draft_seconds(self, depth: int) -> float:
        points = self._measured(self.draft_cost)
        slope = self._plain_step(1) * self._DRAFT_SLOPE
        return slope * depth if not points else self._extrapolate(points, depth, slope * 0.1)

    def _catch_up_seconds(self) -> float:
        value = self.catch_up_cost.value
        return self._plain_step(1) * self._CATCH_UP if value is None else value

    def _row_seconds(self, row: RowPlan, state: RowState) -> float:
        if row.source != "draft":
            return 0.0
        return self._draft_seconds(row.length) + state.catch_up * self._catch_up_seconds()

    def round_seconds(self, plan: RoundPlan, rows: Sequence[RowState], width: int | None = None) -> float:
        seconds = self._verify_seconds(_bucket(len(rows)), width or plan.width)
        return seconds + sum(self._row_seconds(row, state) for row, state in zip(plan.rows, rows))

    def score(self, plan: RoundPlan, rows: Sequence[RowState], width: int | None = None) -> float:
        if plan.kind == "plain":
            return len(rows) / self._plain_step(_bucket(len(rows)))
        expected = sum(state.stats.expected(row) for row, state in zip(plan.rows, rows))
        return expected / self.round_seconds(plan, rows, width)

    @staticmethod
    def _cap(state: RowState, length: int) -> int:
        return length if state.remaining is None else max(0, min(length, state.remaining - 1))

    def _round_at(self, width: int, rows: Sequence[RowState]) -> RoundPlan:
        # Drafting rows add tokens and their own forward to one shared verify, so the best set is
        # a prefix by tokens gained per second. Scored at the nominal width, so the first row to
        # draft does not carry the whole widening alone.
        base, options = [], []
        for i, state in enumerate(rows):
            copy = self._cap(state, min(state.copy_available, self.max_copy, width - 1))
            base.append(RowPlan("copy", copy) if copy else RowPlan())
            depth = self._cap(state, min(self.max_depth, width - 1))
            if state.can_draft and depth:
                draft = RowPlan("draft", depth)
                gain = state.stats.expected(draft) - state.stats.expected(base[i])
                options.append((gain, self._row_seconds(draft, state), i, draft))
        tokens = sum(state.stats.expected(row) for row, state in zip(base, rows))
        seconds = self._verify_seconds(_bucket(len(rows)), width)
        best = RoundPlan("round", rows = tuple(base))
        best_score = tokens / seconds
        if best.width == 1 and options:
            # A round needs a drafter; the least harmful lone one need not lead the ratio order.
            gain, cost, i, draft = max(options, key = lambda option: (tokens + option[0]) / (seconds + option[1]))
            best, best_score = RoundPlan("round", rows = best.rows[:i] + (draft,) + best.rows[i + 1 :]), (tokens + gain) / (seconds + cost)
        choice = base
        for gain, cost, i, draft in sorted(options, key = lambda option: option[0] / option[1], reverse = True):
            choice[i] = draft
            tokens, seconds = tokens + gain, seconds + cost
            if tokens / seconds > best_score:
                best, best_score = RoundPlan("round", rows = tuple(choice)), tokens / seconds
        return best

    @staticmethod
    def _identity(plan: RoundPlan, bucket: int) -> tuple:
        return (bucket, "plain") if plan.kind == "plain" else (bucket, "round", plan.width)

    def plan(self, rows: Sequence[RowState]) -> RoundPlan:
        """The next round for these rows, or a plain window."""
        bucket = _bucket(len(rows))
        room = min((state.remaining for state in rows if state.remaining is not None), default = 1 << 30)
        plain = RoundPlan("plain", max(1, min(self._window, room)))
        self._probing_sources = {}
        if self._warmup:
            depth = self._warmup[0]
            warmup = tuple(
                RowPlan("draft", self._cap(state, depth))
                if state.can_draft and self._cap(state, depth) else RowPlan()
                for state in rows
            )
            if any(row.source == "draft" for row in warmup):
                return RoundPlan("round", rows = warmup, split = True)
        if bucket not in self.plain_cost:
            return RoundPlan("plain", max(1, min(self.min_window, room)))

        rounds = {}
        for width in range(2, self.max_width + 1):
            candidate = self._round_at(width, rows)
            if candidate.width > 1:
                rounds.setdefault(candidate.width, candidate)
        candidates = [plain] + list(rounds.values())
        chosen, probing = self._choose(candidates, rows, bucket)
        if not probing:
            chosen = self._with_source_probes(chosen, rows)
        if chosen.kind == "round" and (probing or not self._measured_parts(chosen, rows)):
            chosen = replace(chosen, split = True)
        return self._split_due(chosen)

    def _split_due(self, plan: RoundPlan) -> RoundPlan:
        if plan.kind == "round" and not plan.split and self._since_split + 1 >= self.split_every:
            return replace(plan, split = True)
        return plan

    def _measured_parts(self, plan: RoundPlan, rows: Sequence[RowState]) -> bool:
        verify = self.verify_cost.get(_bucket(len(rows)), {}).get(plan.width)
        if verify is None or verify.value is None:
            return False
        for row, state in zip(plan.rows, rows):
            if row.source == "draft":
                if self.draft_cost[row.length].value is None:
                    return False
                if state.catch_up and self.catch_up_cost.value is None:
                    return False
        return True

    def _choose(self, candidates, rows, bucket) -> tuple[RoundPlan, bool]:
        if self._probe_left > 0:
            # A probe runs a few rounds: one sample moves a rarely drafted position too little.
            probe = next((p for p in candidates if self._identity(p, bucket) == self._probe), None)
            if probe is not None:
                self._probe_left -= 1
                return self._shorten(probe), True
            self._probe_left = 0
        best = max(candidates, key = lambda plan: self.score(plan, rows))
        current = next((p for p in candidates if self._identity(p, bucket) == self._current), None)
        if current is not None and self.score(best, rows) < self.score(current, rows) * self.hysteresis:
            best = current
        if self.steps >= self._next_probe:
            self._next_probe = self.steps + self.probe_every
            top = self.score(best, rows)

            def stale(plan):
                identity = self._identity(plan, bucket)
                gap = self._stale_gap.get(identity, self.stale_after)
                return self.steps - self._measured_at.get(identity, -gap) >= gap

            rivals = [
                plan for plan in candidates
                if plan is not best and (self.score(plan, rows) * self.probe_margin >= top or stale(plan))
            ]
            if rivals:
                probe = min(rivals, key = lambda plan: self._measured_at.get(self._identity(plan, bucket), -1))
                self._probe, self._probe_left = self._identity(probe, bucket), self.probe_rounds - 1
                if self.score(probe, rows) * self.probe_margin < top:
                    # A rival that keeps losing is re-measured exponentially less often.
                    gap = self._stale_gap.get(self._probe, self.stale_after)
                    self._stale_gap[self._probe] = min(2 * gap, self.max_stale_after)
                return self._shorten(probe), True
        identity = self._identity(best, bucket)
        if best.kind == "plain":
            self._window = min(self._window * 2, self.max_window) if identity == self._current else self.min_window
            best = RoundPlan("plain", min(self._window, best.length))
        else:
            self._window = self.min_window
        self._current = identity
        return best, False

    def _shorten(self, plan: RoundPlan) -> RoundPlan:
        return RoundPlan("plain", min(self.min_window, plan.length)) if plan.kind == "plain" else plan

    def _available(self, state: RowState, source: str) -> int:
        if source == "copy":
            return self._cap(state, min(state.copy_available, self.max_copy, self._COPY_PROBE))
        # Full depth: a shallower probe never re-measures the deeper positions.
        return self._cap(state, self.max_depth) if state.can_draft else 0

    def _due(self, state: RowState, source: str) -> bool:
        return bool(self._available(state, source)) and state.stats.tokens >= state.stats.probe_at[source]

    def _with_source_probes(self, plan: RoundPlan, rows: Sequence[RowState]) -> RoundPlan:
        # A source that loses gets no samples to win back with, whatever beat it, so each row
        # retries its unused sources, exponentially less often while they keep losing.
        base = list(plan.rows) if plan.kind == "round" else [RowPlan()] * len(rows)
        probing = {}
        for i, state in enumerate(rows):
            due = [
                source for source in ("copy", "draft")
                if base[i].source != source and self._due(state, source)
            ]
            if due:
                source = min(due, key = lambda source: state.stats.probe_at[source])
                base[i] = RowPlan(source, self._available(state, source))
                probing[i] = source
        if not probing:
            return plan
        self._probing_sources = probing
        return RoundPlan("round", rows = tuple(base))

    def interrupts_plain(self, rows: Sequence[RowState]) -> bool:
        """Whether copies now on offer are worth ending a plain window early for."""
        if any(self._due(state, "copy") for state in rows):
            return True
        if not any(state.copy_available for state in rows):
            return False
        copying = [RowState(state.stats, False, state.copy_available, state.remaining) for state in rows]
        plain = self.score(RoundPlan("plain", 1), rows)
        widest = 1 + min(self.max_copy, max(state.copy_available for state in rows))
        for width in range(2, widest + 1):
            copies = self._round_at(width, copying)
            if copies.width > 1 and self.score(copies, rows) > plain:
                return True
        return False

    def record_plain(self, rows: Sequence[RowState], steps: int, seconds: float) -> None:
        if steps <= 0:
            return
        bucket = _bucket(len(rows))
        self.plain_cost.setdefault(bucket, _Ema()).update(seconds / steps, self.cost_alpha)
        for state in rows:
            state.stats.tokens += steps
        self.tokens += steps * len(rows)
        self.steps += steps
        self._measured_at[(bucket, "plain")] = self.steps

    def record_round(
        self,
        plan: RoundPlan,
        rows: Sequence[RowState],
        accepted: Sequence[int],
        *,
        seconds: float,
        draft_seconds: float = 0.0,
        catch_up_seconds: float = 0.0,
    ) -> None:
        """Account a finished round; ``accepted[i]`` counts row ``i``'s accepted drafts.

        ``seconds`` is the whole round. The drafting and catch-up shares are read only from a
        split round; a fused round rescales the current estimates to its total instead.
        """
        bucket = _bucket(len(rows))
        alpha = self.acceptance_alpha
        for i, (row, state, count) in enumerate(zip(plan.rows, rows, accepted)):
            count = max(0, min(int(count), row.length))
            stats = state.stats
            for position in range(min(count + 1, row.length)):
                sample = 1.0 if position < count else 0.0
                if row.source == "draft":
                    stats.draft[position].update(sample, alpha)
                    stats.draft_seen[position] = True
                    self.draft_acceptance[position].update(sample, alpha)
                elif row.source == "copy":
                    stats.copy.update(sample, alpha)
                    self.copy_acceptance.update(sample, alpha)
            if row.source in stats.backoff:
                probed = i in self._probing_sources
                backoff = stats.backoff[row.source]
                stats.backoff[row.source] = min(backoff * 2, self._max_backoff) if probed else self._base_backoff
            stats.tokens += count + 1
            self.tokens += count + 1
            if row.source in stats.probe_at:
                stats.probe_at[row.source] = stats.tokens + stats.backoff[row.source]
        self._probing_sources = {}

        depths = [row.length for row in plan.rows if row.source == "draft" and row.length]
        catch_up = sum(state.catch_up for row, state in zip(plan.rows, rows) if row.source == "draft")
        if plan.split:
            self._since_split = 0
            verify_seconds = max(seconds - draft_seconds - catch_up_seconds, 1e-9)
        else:
            self._since_split += 1
            verify_seconds = self._verify_seconds(bucket, plan.width)
            drafts = sum(self._draft_seconds(depth) for depth in depths)
            replay = catch_up * self._catch_up_seconds()
            scale = seconds / (verify_seconds + drafts + replay)
            verify_seconds *= scale
            catch_up_seconds = replay * scale
        self.verify_cost.setdefault(bucket, {}).setdefault(plan.width, _Ema()).update(
            verify_seconds, self.cost_alpha
        )
        if depths:
            # A split round only apportions drafting time it can attribute: one depth for
            # every drafting row. A fused round moves each depth by the round's own scale.
            if not plan.split:
                for depth in set(depths):
                    self.draft_cost[depth].update(self._draft_seconds(depth) * scale, self.cost_alpha)
            elif len(set(depths)) == 1:
                self.draft_cost[depths[0]].update(draft_seconds / len(depths), self.cost_alpha)
            if self._warmup and self._warmup[0] >= max(depths):
                self._warmup.pop(0)
        if catch_up:
            self.catch_up_cost.update(catch_up_seconds / catch_up, self.cost_alpha)
        self.steps += (sum(min(int(a), row.length) for a, row in zip(accepted, plan.rows)) + len(rows)) / len(rows)
        self._measured_at[self._identity(plan, bucket)] = self.steps


# Speculative decoding engine over an mlx-vlm model: verify rounds and plain windows for B rows.


_U64 = (1 << 64) - 1


def _position_seed(seed: int, position: int) -> int:
    # splitmix64 of (seed, position): draws for different positions of one reply are independent.
    x = (seed + (position + 1) * 0x9E3779B97F4A7C15) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    return x ^ (x >> 31)


class _RowSampler:
    """Draws the token at absolute position ``p`` from a key of (seed, p), so a draw depends only on
    its logits and position: rejected positions consume nothing, and rounds, plain windows and
    batch membership leave the random stream unchanged."""

    def __init__(self, params: SamplingParams):
        self.temperature = params.temperature
        self.seed = params.seed
        self.stages = []
        if self.temperature == 0:
            return
        if self.seed is None:
            self.seed = int(mx.random.randint(0, 1 << 31).item())
        from mlx_lm import sample_utils

        if 0 < params.top_p < 1:
            self.stages.append(lambda x: sample_utils.apply_top_p(x, params.top_p))
        if params.min_p:
            self.stages.append(lambda x: sample_utils.apply_min_p(x, params.min_p, 1))
        if params.top_k > 0:
            self.stages.append(lambda x: sample_utils.apply_top_k(x, params.top_k))

    def __call__(self, logprobs: mx.array, position: int) -> mx.array:
        """Tokens for ``logprobs`` rows ``[n, vocab]`` at positions ``position .. position + n - 1``."""
        if self.temperature == 0:
            return mx.argmax(logprobs, axis = -1)
        for stage in self.stages:
            logprobs = stage(logprobs)
        keys = mx.stack([mx.random.key(_position_seed(self.seed, position + i)) for i in range(logprobs.shape[0])])
        scale = 1 / self.temperature
        return mx.vmap(lambda row, key: mx.random.categorical(row * scale, key = key))(logprobs, keys)


@dataclass(eq = False)
class EngineRow:
    """A prefilled reply joining the engine.

    ``cache`` holds the prompt; ``pending`` is the last sampled token, already emitted but not yet
    forwarded; ``emitted`` counts the tokens emitted so far, ``pending`` included, and is the
    position of the next draw.
    """

    cache: list
    pending: int
    prompt: Sequence[int]
    sampling: SamplingParams = field(default_factory = SamplingParams)
    max_tokens: int | None = None
    emitted: int = 1
    stop_tokens: frozenset = frozenset()
    processors: Sequence[Callable] = ()
    rope_delta: int = 0
    uid: Any = None


@dataclass(frozen = True)
class StepOutput:
    """One row's tokens from a step, with its reply's running draft counters."""

    uid: Any
    tokens: list[int]
    finished: bool
    draft_n: int
    draft_n_accepted: int


class _Row:
    def __init__(self, row: EngineRow, controller: DraftController):
        self.uid = row.uid
        self.pending = int(row.pending)
        self.tokens = [int(token) for token in row.prompt] + [self.pending]
        self.emitted = int(row.emitted)
        self.max_tokens = row.max_tokens
        self.stop_tokens = frozenset(row.stop_tokens)
        self.processors = list(row.processors)
        self.rope_delta = int(row.rope_delta)
        self.sample = _RowSampler(row.sampling)
        self.proposer = NgramProposer(row.prompt)
        self.state = RowState(controller.new_reply(), can_draft = False)
        self.finished = self.pending in self.stop_tokens or self.remaining == 0
        self.draft_n = 0
        self.draft_n_accepted = 0

    @property
    def remaining(self) -> int | None:
        return None if self.max_tokens is None else max(0, self.max_tokens - self.emitted)

    def logprobs(self, logits: mx.array) -> mx.array:
        """``logits [1, vocab]`` for the next position, through this row's processors."""
        if self.processors:
            history = mx.array(self.tokens)
            for processor in self.processors:
                logits = processor(history, logits)
        return logits - mx.logsumexp(logits, axis = -1, keepdims = True)

    def take(self, tokens: Sequence[int]) -> list[int]:
        """Emit ``tokens`` up to a stop token or the budget; the last one taken becomes pending."""
        taken = []
        for token in tokens:
            if self.finished:
                break
            taken.append(int(token))
            self.emitted += 1
            self.finished = int(token) in self.stop_tokens or self.remaining == 0
        if taken:
            self.tokens.extend(taken)
            self.pending = taken[-1]
        return taken


def _language_call(family: str):
    try:
        module = __import__(f"mlx_vlm.models.{family}.language", fromlist = ["LanguageModel"])
    except ImportError:
        return None
    return module.LanguageModel.__call__


class SpeculativeEngine:
    """Decodes B rows with the controller's choice each step: a verify round at one width, each
    row verifying its own proposal and committing its own accepted count, or a window of
    ordinary decode steps. Rows join and leave only between steps, when every row holds exactly
    one pending token and the caches hold everything before it."""

    def __init__(self, model, controller: DraftController):
        from mlx_vlm.speculative.common import generation_stream, verify_forward

        self.lm = getattr(model, "language_model", model)
        self.controller = controller
        self._verify_forward = verify_forward
        self._stream = generation_stream
        # qwen3_5's exact verifier matches one-token decoding bitwise, alone and batched, and passing
        # a capture list keeps the forward off the per-row paths that replace cache objects.
        call = type(self.lm).__call__
        self._exact = call is _language_call("qwen3_5")
        # Families whose verify forward was measured to commit ragged acceptance correctly.
        self._batch_rounds = self._exact or call is _language_call("gemma4")
        self._mrope = hasattr(self.lm, "_rope_deltas")
        self._rows: list[_Row] = []
        self.cache: list | None = None
        self.round_refusal: str | None = None

    @property
    def rows(self) -> list[Any]:
        return [row.uid for row in self._rows]

    def add(self, row: EngineRow) -> None:
        if any(existing.uid == row.uid for existing in self._rows):
            raise ValueError(f"row {row.uid!r} is already in the engine")
        if not self._rows:
            self.cache = list(row.cache)
        elif len(self._rows) == 1:
            self.cache = [type(batch).merge([batch, new]) for batch, new in zip(self.cache, row.cache)]
        else:
            for batch, new in zip(self.cache, row.cache):
                batch.extend(type(new).merge([new]))
        self._rows.append(_Row(row, self.controller))

    def remove(self, uid) -> None:
        keep = [i for i, row in enumerate(self._rows) if row.uid != uid]
        if len(keep) == len(self._rows):
            raise KeyError(uid)
        self._rows = [self._rows[i] for i in keep]
        if not keep:
            self.cache = None
        elif len(keep) == 1:
            # A single-row batch cache sends some families down per-row paths that replace it.
            self.cache = [batch.extract(keep[0]) for batch in self.cache]
        else:
            indices = mx.array(keep)
            for batch in self.cache:
                batch.filter(indices)

    def step(self) -> list[StepOutput]:
        """One round or plain window, one output per row. Finished rows leave the engine."""
        if not self._rows:
            return []
        states = self._states()
        if self._rounds_allowed():
            plan = self.controller.plan(states)
        else:
            room = min((row.remaining for row in self._rows if row.remaining is not None), default = None)
            plan = RoundPlan("plain", self.controller.max_window if room is None else min(room, self.controller.max_window))
        emitted = self._round(plan, states) if plan.kind == "round" else None
        if emitted is None:
            emitted = self._plain(max(1, plan.length), states)
        out = [StepOutput(row.uid, tokens, row.finished, row.draft_n, row.draft_n_accepted) for row, tokens in zip(self._rows, emitted)]
        for row in [row for row in self._rows if row.finished]:
            self.remove(row.uid)
        return out

    def _rounds_allowed(self) -> bool:
        return self.round_refusal is None and (len(self._rows) == 1 or self._batch_rounds)

    def _states(self) -> list[RowState]:
        for row in self._rows:
            state = row.state
            state.remaining = row.remaining
            state.copy_available = (
                0 if row.processors else len(row.proposer.propose(row.tokens, self.controller.max_copy))
            )
        return [row.state for row in self._rows]

    def _position_kwargs(self) -> dict:
        if not self._mrope:
            return {}
        return {"rope_deltas": mx.array([[row.rope_delta] for row in self._rows])}

    def _forward(self, inputs: mx.array) -> mx.array:
        return self.lm(inputs, cache = self.cache, **self._position_kwargs()).logits

    def _verify(self, inputs: mx.array):
        if self._exact:
            out = self.lm(
                inputs, cache = self.cache, capture_layer_ids = [], speculative_verify = True, **self._position_kwargs()
            )
            return out.logits, out.gdn_states
        out, transaction = self._verify_forward(self.lm, inputs, self.cache, **self._position_kwargs())
        return out.logits, transaction

    def _sample_step(self, logits: mx.array, ahead: int = 0) -> mx.array:
        # ``ahead``: tokens already sampled for each row but not taken yet.
        return mx.concatenate(
            [row.sample(row.logprobs(logits[i : i + 1, -1]), row.emitted + ahead) for i, row in enumerate(self._rows)]
        )

    def _plain(self, length: int, states: list[RowState]) -> list[list[int]]:
        rows = self._rows
        emitted = [[] for _ in rows]
        # Processors read the history, so a row with them needs each token before the next step.
        pipelined = not any(row.processors for row in rows)
        start = time.perf_counter()
        with mx.stream(self._stream):
            tokens = self._sample_step(self._forward(mx.array([[row.pending] for row in rows])))
        steps = 0
        while True:
            steps += 1
            last = steps >= length or all(row.finished for row in rows)
            upcoming = None
            if not last and pipelined:
                with mx.stream(self._stream):
                    upcoming = self._sample_step(self._forward(tokens[:, None]), ahead = 1)
                mx.async_eval(upcoming)
            for i, token in enumerate(tokens.tolist()):
                emitted[i].extend(rows[i].take([token]))
            if last:
                break
            if upcoming is None:
                with mx.stream(self._stream):
                    upcoming = self._sample_step(self._forward(tokens[:, None]))
            elif self.controller.interrupts_plain(self._states()):
                # The next step is already queued, so it becomes the window's last.
                length = steps + 1
            tokens = upcoming
        self.controller.record_plain(states, steps, time.perf_counter() - start)
        return emitted

    def _round(self, plan: RoundPlan, states: list[RowState]) -> list[list[int]] | None:
        rows, width = self._rows, plan.width
        proposals = [
            row.proposer.propose(row.tokens, spec.length) if spec.source == "copy" else []
            for row, spec in zip(rows, plan.rows)
        ]
        inputs = mx.array(
            [[row.pending, *proposal] + [row.pending] * (width - 1 - len(proposal)) for row, proposal in zip(rows, proposals)]
        )
        start = time.perf_counter()
        entries = list(self.cache)
        with mx.stream(self._stream):
            logits, transaction = self._verify(inputs)
        if any(now is not before for now, before in zip(self.cache, entries)):
            self.cache[:] = entries
            transaction.abort()
            self.round_refusal = f"{type(self.lm).__name__} replaced its cache objects during a verify forward"
            return None
        try:
            with mx.stream(self._stream):
                targets = []
                for i, (row, proposal) in enumerate(zip(rows, proposals)):
                    if row.processors:
                        logprobs = row.logprobs(logits[i : i + 1, 0])
                    else:
                        logprobs = logits[i, : len(proposal) + 1]
                        logprobs = logprobs - mx.logsumexp(logprobs, axis = -1, keepdims = True)
                    targets.append(row.sample(logprobs, row.emitted))
            mx.eval(targets)
            targets = [target.tolist() for target in targets]
            accepted = []
            for proposal, target in zip(proposals, targets):
                count = 0
                while count < len(proposal) and proposal[count] == target[count]:
                    count += 1
                accepted.append(count)
            transaction.commit([count + 1 for count in accepted])
            if len(set(accepted)) > 1:
                # A ragged commit grows BatchKVCache.left_padding in place, but qwen3_5 memoizes decode pads by that array's identity.
                for entry in self.cache:
                    padding = getattr(entry, "left_padding", None)
                    if isinstance(padding, mx.array) and not hasattr(entry, "metadata_revision"):
                        entry.left_padding = padding + 0
        except BaseException:
            transaction.abort()
            raise
        emitted = []
        for row, proposal, target, count in zip(rows, proposals, targets, accepted):
            row.draft_n += len(proposal)
            row.draft_n_accepted += count
            emitted.append(row.take(proposal[:count] + [target[count]]))
        self.controller.record_round(plan, states, accepted, seconds = time.perf_counter() - start)
        return emitted
