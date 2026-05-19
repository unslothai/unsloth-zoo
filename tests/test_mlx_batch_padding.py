# Unsloth Zoo - Utilities for Unsloth
# Pin MLXTrainer's batch padding to match mlx-lm's iterate_batches
# semantics: pad to `1 + _PAD_MULTIPLE * ceil(L / _PAD_MULTIPLE)`.
#
# Why this matters:
# mlx-lm's tuner trainer (`mlx_lm/tuner/trainer.py:158`) pads each
# batch to `1 + 32 * ceil(max_len / 32)`. The default loss then
# slices `inputs = batch[:, :-1]` / `targets = batch[:, 1:]`, so the
# effective per-position-attention length is `32 * ceil(max_len/32)`.
# unsloth_zoo's `create_text_batches` previously rounded WITHOUT the
# `+1` (just `32 * ceil(max_len/32)`), which dropped one token of
# input length after the autoregressive shift, putting the trainer
# one token shy of mlx-lm. On a single-row LoRA memorization fixture
# against gemma-3-270m-it, the one-token gap moved the run into a
# different convergence basin (probe 31 manual loop = 10/15 = 67%
# vs probe 33-37 MLXTrainer = 6-8/15 = 40-53% on paired seeds, see
# danielhanchen/unsloth-staging-2).

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


def test_source_padding_formula_includes_plus_one():
    """Guard against a future refactor that drops the +1 again."""
    import inspect
    from unsloth_zoo.mlx import trainer

    src = inspect.getsource(trainer)
    # The exact line we care about. If someone rewrites the formula
    # they must preserve the +1 contract or add a new test alongside.
    needle = "1 + ((max_len + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE"
    assert needle in src, (
        f"create_text_batches must use `{needle}` to match mlx-lm's "
        "1 + pad_to*ceil(L/pad_to). Dropping the +1 leaves the input "
        "one token shorter than mlx-lm after the autoregressive shift "
        "and changes the convergence basin on small fixtures."
    )


def test_pad_multiple_constant_still_32():
    """mlx-lm uses pad_to=32; we must too."""
    from unsloth_zoo.mlx import trainer
    assert trainer._PAD_MULTIPLE == 32
