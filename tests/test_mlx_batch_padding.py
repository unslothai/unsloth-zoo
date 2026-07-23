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

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _mlx_lm_padded_len(max_len, pad_to=32):
    """mlx-lm's iterate_batches padding (mlx_lm/tuner/trainer.py:158)."""
    return 1 + pad_to * ((max_len + pad_to - 1) // pad_to)


def _zoo_padded_len(max_len, pad_multiple=32):
    """Reproduce the in-source rule from create_text_batches so we can
    assert it stays aligned with mlx-lm without standing up the full
    tokenizer + dataset pipeline.
    """
    return 1 + ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple


@pytest.mark.parametrize(
    "max_len, expected",
    [
        # Inside the first 32-token bucket -> 33.
        (1, 33),
        (14, 33),  # the probe fixture's TRAIN_TEXT length
        (31, 33),
        (32, 33),
        # Second bucket -> 65.
        (33, 65),
        (63, 65),
        (64, 65),
        # Third bucket -> 97.
        (65, 97),
        # Larger buckets.
        (97, 129),
        (128, 129),
        (129, 161),
    ],
)
def test_zoo_padding_matches_mlx_lm(max_len, expected):
    """Zoo's pad rule must equal mlx-lm's pad rule, value-for-value."""
    assert _zoo_padded_len(max_len) == expected
    assert _mlx_lm_padded_len(max_len) == expected
    assert _zoo_padded_len(max_len) == _mlx_lm_padded_len(max_len)


def test_padding_formula_includes_plus_one():
    """All width callers share _finite_text_pad_width; lock its +1 contract."""
    from unsloth_zoo.mlx import trainer
    from unsloth_zoo.mlx.utils import _finite_text_pad_width as width

    m = trainer._PAD_MULTIPLE
    assert [
        width(m, pad_to_multiple=m, max_seq_length=10_000),
        width(m - 1, pad_to_multiple=m, max_seq_length=10_000),
        width(m + 1, pad_to_multiple=m, max_seq_length=10_000),
        width(m + 1, pad_to_multiple=m, max_seq_length=20),
        width(0, pad_to_multiple=m, minimum_width=2, max_seq_length=64),
        width(5, max_seq_length=64),
        width(100, max_seq_length=64),
    ] == [1 + m, 1 + m, 1 + 2 * m, 20, 2, 5, 64]


def test_pad_multiple_constant_still_32():
    """mlx-lm uses pad_to=32; we must too."""
    from unsloth_zoo.mlx import trainer
    assert trainer._PAD_MULTIPLE == 32
