# Unsloth Zoo - Utilities for Unsloth
# A strict mlx load that rejects q_norm/k_norm means the installed mlx-lm/mlx-vlm
# is too old (or regressed, e.g. 0.31.3) for a QK-norm arch (gemma4 / qwen3_5).
# Surface a clear, actionable error instead of the raw mlx ValueError; dropping
# those weights would break the model.

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


_TESTER_MSG = (
    "Received 140 parameters not in model: "
    "language_model.model.layers.15.self_attn.k_norm.weight, "
    "language_model.model.layers.15.self_attn.q_norm.weight"
)


def test_qk_norm_mismatch_raises_actionable_error():
    from unsloth_zoo.mlx.loader import _raise_if_qk_norm_version_gap

    with pytest.raises(ValueError) as exc:
        _raise_if_qk_norm_version_gap("gemma4", _TESTER_MSG, ValueError("orig"))
    msg = str(exc.value)
    assert "mlx-lm" in msg and "0.31.3" in msg and "gemma4" in msg


def test_qwen3_5_q_norm_also_caught():
    from unsloth_zoo.mlx.loader import _raise_if_qk_norm_version_gap

    with pytest.raises(ValueError):
        _raise_if_qk_norm_version_gap(
            "qwen3_5", "Received 5 parameters not in model: model.layers.3.self_attn.q_norm.weight",
            ValueError("orig"),
        )


def test_non_qk_norm_mismatch_passes_through():
    from unsloth_zoo.mlx.loader import _raise_if_qk_norm_version_gap

    # A different extra-weight mismatch (handled by the known gemma4 filter) or an
    # unrelated error must NOT be swallowed by the QK-norm guard.
    _raise_if_qk_norm_version_gap(
        "gemma4", "Received 4 parameters not in model: per_layer_model_projection.scales",
        ValueError("orig"),
    )
    _raise_if_qk_norm_version_gap("llama", "some unrelated value error", ValueError("orig"))
