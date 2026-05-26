# Unsloth Zoo - Utilities for Unsloth
# Pin `make_baseline_loss_fn` source so the labels=None fast path stays
# byte-for-byte equivalent to mlx_lm.tuner.trainer.default_loss.
#
# Why pin source rather than run a numerical comparison: the test
# harness uses a torch-based MLX shim that doesn't faithfully reproduce
# MLX's autodiff graph or its rounding; an apples-to-apples numerical
# parity check requires a real MLX runtime (Apple Silicon), so it's
# done in the Round BP probe matrix on
# danielhanchen/unsloth-staging-2. Locally we guard against future
# refactors silently re-introducing the divergent code patterns
# (fp32-cast mask, mx.where(safe_targets), _safe_token_denominator)
# that mlx_lm.tuner.trainer.default_loss does NOT do.

from __future__ import annotations

import inspect
import re
import textwrap

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _labels_none_block():
    """Return CODE LINES (comments stripped) for the labels=None fast path."""
    from unsloth_zoo.mlx import utils
    src = inspect.getsource(utils.make_baseline_loss_fn)
    m = re.search(
        r"if labels is None:\s*\n(.*?)(?:# labels-aware path|else:\s*\n)",
        src,
        flags=re.DOTALL,
    )
    assert m, "make_baseline_loss_fn must keep a `labels is None` fast path"
    raw = textwrap.dedent(m.group(1))
    # Strip whole-line comments so test assertions on code don't trip on
    # explanatory prose like "no safe_targets mx.where" in docstrings.
    code_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    return "\n".join(code_lines)


def test_no_fp32_mask_cast_in_fast_path():
    """The labels=None path must NOT cast the bool mask to fp32. mlx-lm's
    default_loss multiplies the cross-entropy result by a raw bool mask;
    casting to fp32 produces a different MLX autodiff graph and shifts
    gradients by ~1e-2 per step on small fixtures."""
    block = _labels_none_block()
    # Anything that would cast a mask to fp32: `.astype(mx.float32)` or
    # `astype(float32)` immediately on a `mask` / `length_mask` name.
    bad_patterns = (
        r"length_mask\.astype\(mx\.float32\)",
        r"mask\s*=\s*[^=]*\.astype\(mx\.float32\)",
    )
    for pat in bad_patterns:
        assert not re.search(pat, block), (
            f"labels=None fast path must not contain `{pat}`; "
            "matches mlx-lm requires a bool mask."
        )


def test_no_safe_targets_where_in_fast_path():
    """The labels=None path has no -100 to worry about, so no mx.where
    is needed on targets. The where node was empirically the cause of
    step-2 loss divergence vs mlx-lm CLI (Round BO probe_31 vs probe_37)."""
    block = _labels_none_block()
    assert "mx.where" not in block, (
        "labels=None fast path must not include mx.where on targets; "
        "mlx-lm's default_loss does not call mx.where."
    )
    assert "safe_targets" not in block, (
        "labels=None fast path must use `targets` directly; "
        "`safe_targets = mx.where(...)` belongs only to the labels-aware path."
    )


def test_no_safe_token_denominator_in_fast_path():
    """mlx-lm's default_loss divides by raw `ntoks` (int). The fast path
    must match that to preserve the autodiff graph; the safety wrapper
    `_safe_token_denominator` introduces a fp32 cast + maximum() that
    changes rounding in MLX."""
    block = _labels_none_block()
    assert "_safe_token_denominator" not in block, (
        "labels=None fast path must divide by raw ntoks for mlx-lm parity. "
        "_safe_token_denominator is fine on the labels-aware path."
    )


def test_fast_path_returns_ce_and_ntoks_in_that_order():
    """Match the (loss, ntoks) return signature mlx-lm uses; the test
    pins return-order so a future refactor doesn't accidentally swap."""
    block = _labels_none_block()
    # Look for a `return X, Y` somewhere in the fast path. The variable
    # names are loose (mlx-lm uses `ce`; zoo previously used `loss`),
    # but the order matters.
    m = re.search(r"return\s+(\w+),\s*(\w+)", block)
    assert m, "labels=None fast path must return a (loss, ntoks) tuple"
    loss_name, ntoks_name = m.group(1), m.group(2)
    assert ntoks_name in ("ntoks", "n_toks", "ntokens"), (
        f"second return value name should look like a token count, got "
        f"{ntoks_name!r}"
    )


def test_labels_aware_path_still_uses_safe_targets():
    """The labels-aware path (train_on_responses_only) DOES need
    `safe_targets` and the fp32 mask because labels can contain -100."""
    from unsloth_zoo.mlx import utils
    src = inspect.getsource(utils.make_baseline_loss_fn)
    # The labels-aware path lives after the fast path's `return`. Look
    # at the full source to verify the machinery still exists somewhere.
    assert "mx.where" in src, (
        "make_baseline_loss_fn must still call mx.where on the labels-aware path"
    )
    assert "safe_targets" in src, "labels-aware path must keep safe_targets"
