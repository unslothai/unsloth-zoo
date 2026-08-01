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

"""Regression tests: pair count is the accumulation weight, not the token count.

The preference losses return the PAIR count as their second value, because a
pair-mean loss must weight each micro-batch by pairs when accumulating -- that
is a correctness property, pinned here so it cannot drift.

The trainer also consumed that same value as a token count, so trained_tokens
and tokens_per_second reported pairs. _mlx_supervised_token_count feeds a
separate telemetry accumulator instead, leaving the weight alone.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


def _batch(spans, width=32):
    """A (2B, width) batch with the given [response_start, seq_end) rows."""
    import mlx.core as mx

    rows = len(spans)
    return (
        mx.zeros((rows, width), dtype=mx.int32),
        mx.array(spans),
        None,
    )


# --------------------------------------------------------------------------
# The telemetry counter.
# --------------------------------------------------------------------------

def test_counts_the_loss_span_of_every_row():
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count

    # 2 pairs: chosen rows then rejected rows, 8 supervised tokens each.
    spans = [[10, 18], [10, 18], [10, 18], [10, 18]]
    assert _mlx_supervised_token_count(_batch(spans)) == 32


def test_counts_ragged_spans_independently():
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count

    spans = [[10, 18], [5, 20], [10, 17], [5, 12]]
    assert _mlx_supervised_token_count(_batch(spans)) == 8 + 15 + 7 + 7


def test_empty_span_contributes_nothing():
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count

    assert _mlx_supervised_token_count(_batch([[9, 9], [4, 10]])) == 6


def test_unlabeled_sft_excludes_the_unshifted_first_input_token():
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count

    assert _mlx_supervised_token_count(_batch([[0, 6], [0, 4]])) == 8


def test_labeled_sft_counts_only_shifted_unmasked_targets():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count

    batch = mx.zeros((2, 6), dtype=mx.int32)
    lengths = mx.array([[1, 5], [2, 6]])
    labels = mx.array([
        [-100, 10, -100, 12, 13, 14],
        [-100, 20, 21, -100, 23, 24],
    ])
    assert _mlx_supervised_token_count((batch, lengths, labels)) == 6


@pytest.mark.parametrize("bad", [None, (), (1,), ("x", None)])
def test_returns_none_when_lengths_are_unavailable(bad):
    """No lengths to count -> caller falls back to the loss fn's own value."""
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count

    assert _mlx_supervised_token_count(bad) is None


# --------------------------------------------------------------------------
# The accumulation weight must stay the PAIR count.
# --------------------------------------------------------------------------

def _run_loss(make_fn, monkeypatch):
    """Drive a loss fn end-to-end; the pair-count return is what is under test.

    The shim's nn.losses.cross_entropy defaults to reduction="mean" and is not
    shape-faithful, so patch in a per-token tensor as the existing preference
    tests do.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import torch

    monkeypatch.setattr(
        nn.losses, "cross_entropy",
        lambda logits, targets, **kw: torch.zeros(tuple(targets.shape)),
    )

    B, L, V = 2, 12, 16
    batch = mx.zeros((2 * B, L), dtype=mx.int32)
    lengths = mx.array([[4, 10]] * (2 * B))

    class _Model:
        def __call__(self, x):
            return mx.zeros((x.shape[0], x.shape[1], V))

    _loss, weight = make_fn(_Model(), batch, lengths)
    return B, int(weight.item()), lengths


def test_orpo_weight_is_the_pair_count_not_tokens(monkeypatch):
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count
    from unsloth_zoo.mlx.utils import make_orpo_loss_fn

    B, weight, lengths = _run_loss(make_orpo_loss_fn(beta=0.1), monkeypatch)
    assert weight == B, "ORPO must weight microbatches by PAIRS"

    supervised = _mlx_supervised_token_count((None, lengths, None))
    assert supervised == 4 * 6                       # 4 rows x 6 tokens
    assert weight != supervised, "pairs and tokens must not be interchangeable"


def test_dpo_weight_is_the_pair_count_not_tokens(monkeypatch):
    from unsloth_zoo.mlx.trainer import _mlx_supervised_token_count
    from unsloth_zoo.mlx.utils import make_dpo_loss_fn

    B, weight, lengths = _run_loss(
        make_dpo_loss_fn(beta=0.1, reference_free=True), monkeypatch,
    )
    assert weight == B, "DPO must weight microbatches by PAIRS"
    assert weight != _mlx_supervised_token_count((None, lengths, None))


# --------------------------------------------------------------------------
# The trainer must read the telemetry counter, not the weight.
# --------------------------------------------------------------------------

def test_trainer_reports_tokens_from_the_telemetry_accumulator():
    """trained_tokens / tokens_per_second must come from real_tokens."""
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "real_tokens" in src, "telemetry accumulator missing"
    assert "tok_count = int(metric_real.item())" in src, (
        "tok_count must come from the real-token accumulator"
    )
    assert "trained_tokens += tok_count" in src
    assert "tokens_sec = tok_count / train_time" in src
    # The loss weighting must still divide by n_tokens (pairs).
    assert "(metric_losses / metric_tokens)" in src
    assert "metric_tokens = self._distributed_all_sum(n_tokens" in src


def test_zero_token_guard_uses_the_supervised_span_count():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "supervised_toks = toks if _real is None else" in src
    assert "global_supervised_toks = self._distributed_all_sum(" in src
    assert "if int(global_supervised_toks.item()) == 0:" in src


def test_abandoned_accumulation_window_drops_real_token_telemetry():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    start = src.index("grad_accum_state = None      # abandon the partial window")
    end = src.index("pending_time = 0", start)
    abandoned = src[start:end]
    assert "pending_losses = 0" in abandoned
    assert "pending_n_tokens = 0" in abandoned
    assert "pending_real_tokens = 0" in abandoned
    assert "pending_steps = 0" in abandoned


def test_weighted_loss_denominator_stays_the_accumulation_weight():
    """_train_loss_token_total is the denominator of the weighted average loss.

    _train_loss_token_sum accumulates loss * n_tokens, so the running total must
    accumulate the same weight. Feeding it the real-token count instead makes
    avg_loss and the checkpointed totals disagree with the numerator.
    """
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "weight_count = int(metric_tokens.item())" in src
    assert "self._train_loss_token_total += weight_count" in src
    assert "self._train_loss_token_total += tok_count" not in src, (
        "the weighted-loss denominator must not use the telemetry counter"
    )
