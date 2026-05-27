# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# MLX CCE target-classification coverage that runs on non-Apple-Silicon
# hosts via the simulation shim. The companion file
# tests/test_mlx_runtime_cce_compile.py gates on real Metal and skips
# under the shim, leaving the pure-Python validation, in-vocab
# ignore_index precedence, and logit_softcap fallback paths without
# Linux CI coverage. This file fills those gaps.

from __future__ import annotations

import math

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    try:
        from mlx_simulation import simulate_mlx_on_torch
    except ImportError:
        pytest.skip("mlx_simulation suite not on sys.path", allow_module_level=False)
    simulate_mlx_on_torch()


def _expected_valid_loss(vocab_size: int) -> float:
    # hidden=ones, weight=ones, vocab=V: each logit is dim, lse=log(V)+dim,
    # target_logit=dim, so loss = log(V).
    return math.log(float(vocab_size))


# ----------------------------------------------------------------------
# Pure-Python validation: shape, length, zero-token mismatch
# ----------------------------------------------------------------------

def test_runtime_cce_zero_tokens_with_non_empty_targets_raises():
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.zeros((0, 16), dtype=mx.float32)
    weight = mx.zeros((32, 16), dtype=mx.float32)
    targets = mx.array([0, 1, 2], dtype=mx.int32)

    with pytest.raises(ValueError, match="hidden has 0 tokens"):
        runtime_cce(hidden, weight, targets)


def test_runtime_cce_rejects_non_flat_targets():
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.zeros((4, 16), dtype=mx.float32)
    weight = mx.zeros((32, 16), dtype=mx.float32)
    targets_2d = mx.zeros((4, 1), dtype=mx.int32)
    targets_scalar = mx.array(0, dtype=mx.int32)

    with pytest.raises(ValueError, match="flat 1D vector"):
        runtime_cce(hidden, weight, targets_2d)
    with pytest.raises(ValueError, match="flat 1D vector"):
        runtime_cce(hidden, weight, targets_scalar)


def test_runtime_cce_rejects_target_length_mismatch():
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
    )
    hidden = mx.zeros((4, 16), dtype=mx.float32)
    weight = mx.zeros((32, 16), dtype=mx.float32)
    targets_wrong_len = mx.zeros((3,), dtype=mx.int32)

    with pytest.raises(ValueError, match="targets length does not match"):
        runtime_cce(hidden, weight, targets_wrong_len)


# ----------------------------------------------------------------------
# In-vocab ignore_index must take precedence over invalid classification
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ignore_index", [0, 5, 31])
def test_in_vocab_ignore_index_zero_loss_not_nan(ignore_index):
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=ignore_index,
        chunk_size=16,
    )
    vocab_size = 32
    hidden = mx.ones((3, 16), dtype=mx.float32)
    weight = mx.ones((vocab_size, 16), dtype=mx.float32)
    valid_other = (ignore_index + 1) % vocab_size
    targets = mx.array(
        [0 if ignore_index != 0 else 1, ignore_index, valid_other],
        dtype=mx.int32,
    )

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)

    assert losses[1].item() == pytest.approx(0.0)
    assert not math.isnan(losses[1].item())
    assert losses[0].item() == pytest.approx(_expected_valid_loss(vocab_size), rel=1e-5)
    assert losses[2].item() == pytest.approx(_expected_valid_loss(vocab_size), rel=1e-5)


def test_in_vocab_ignore_index_does_not_poison_other_rows():
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=10,
        chunk_size=16,
    )
    vocab_size = 32
    hidden = mx.ones((4, 16), dtype=mx.float32)
    weight = mx.ones((vocab_size, 16), dtype=mx.float32)
    targets = mx.array([3, 10, 33, 7], dtype=mx.int32)

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)

    assert losses[0].item() == pytest.approx(_expected_valid_loss(vocab_size), rel=1e-5)
    assert losses[1].item() == pytest.approx(0.0)
    assert math.isnan(losses[2].item())
    assert losses[3].item() == pytest.approx(_expected_valid_loss(vocab_size), rel=1e-5)


# ----------------------------------------------------------------------
# logit_softcap > 0 must preserve NaN poisoning for invalid labels
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bad_target", [-1, 32])
def test_softcap_invalid_label_poisons_loss(bad_target):
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
        logit_softcap=30.0,
    )
    hidden = mx.ones((3, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([0, bad_target, -100], dtype=mx.int32)

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)

    assert math.isfinite(losses[0].item())
    assert math.isnan(losses[1].item())
    assert losses[2].item() == pytest.approx(0.0)


def test_softcap_valid_labels_remain_finite():
    import mlx.core as mx

    from unsloth_zoo.mlx.cce import make_chunked_cross_entropy_loss

    runtime_cce, _ = make_chunked_cross_entropy_loss(
        ignore_index=-100,
        chunk_size=16,
        logit_softcap=30.0,
    )
    hidden = mx.ones((2, 16), dtype=mx.float32)
    weight = mx.ones((32, 16), dtype=mx.float32)
    targets = mx.array([0, 1], dtype=mx.int32)

    losses = runtime_cce(hidden, weight, targets)
    mx.eval(losses)

    assert math.isfinite(losses[0].item())
    assert math.isfinite(losses[1].item())
