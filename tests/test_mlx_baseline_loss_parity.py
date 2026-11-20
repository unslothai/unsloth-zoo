# Unsloth Zoo - Utilities for Unsloth
# Pin `make_baseline_loss_fn` source so the labels=None fast path stays
# byte-for-byte equivalent to mlx_lm.tuner.trainer.default_loss.
#
# Source is pinned rather than numerically compared because the torch-based
# MLX shim doesn't reproduce MLX's autodiff graph / rounding (real numerical
# parity needs Apple Silicon, done in the Round BP probe matrix on
# danielhanchen/unsloth-staging-2). Locally we guard against refactors
# re-introducing the divergent patterns (fp32-cast mask, mx.where(safe_targets),
# _safe_token_denominator) that default_loss does NOT do.

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
    code_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    return "\n".join(code_lines)


def test_no_fp32_mask_cast_in_fast_path():
    """labels=None path must not cast the bool mask to fp32: mlx-lm's default_loss multiplies by the raw mask, and fp32 changes the MLX autodiff graph."""
    block = _labels_none_block()
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
    """labels=None path has no -100 targets, so mx.where is unneeded there; the where node caused step-2 loss divergence vs mlx-lm CLI."""
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
    """Fast path must divide by raw ntoks (int) like mlx-lm default_loss to keep the autodiff graph;
    _safe_token_denominator adds an fp32 cast and maximum() that changes MLX rounding"""
    block = _labels_none_block()
    assert "_safe_token_denominator" not in block, (
        "labels=None fast path must divide by raw ntoks for mlx-lm parity. "
        "_safe_token_denominator is fine on the labels-aware path."
    )


def test_fast_path_returns_ce_and_ntoks_in_that_order():
    """Pins the (loss, ntoks) return order used by mlx-lm so a refactor cannot silently swap the two values."""
    block = _labels_none_block()
    # Only the return order is pinned; the names are loose (mlx-lm uses ce, zoo has used loss).
    m = re.search(r"return\s+(\w+),\s*(\w+)", block)
    assert m, "labels=None fast path must return a (loss, ntoks) tuple"
    loss_name, ntoks_name = m.group(1), m.group(2)
    assert ntoks_name in ("ntoks", "n_toks", "ntokens"), (
        f"second return value name should look like a token count, got "
        f"{ntoks_name!r}"
    )


def test_labels_aware_path_still_uses_safe_targets():
    """The labels-aware path (train_on_responses_only) needs safe_targets and the fp32 mask because labels can contain -100."""
    from unsloth_zoo.mlx import utils
    src = inspect.getsource(utils.make_baseline_loss_fn)
    # This path lives after the fast path's `return`, so read the full source.
    assert "mx.where" in src, (
        "make_baseline_loss_fn must still call mx.where on the labels-aware path"
    )
    assert "safe_targets" in src, "labels-aware path must keep safe_targets"
