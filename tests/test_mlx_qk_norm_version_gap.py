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


def test_kv_sharing_dead_tail_falls_through_to_strict_false():
    # A gemma4 KV-sharing checkpoint's shared tail carries DEAD k_proj/v_proj/
    # k_norm (those layers reuse an earlier layer's K/V), so mlx-lm 0.31.3 rejects
    # exactly that dead tail: k_proj + v_proj + k_norm, never q_norm. Dropping it
    # via the registered strict=False fallback is safe, so the guard must NOT
    # raise here (else it regresses a working gemma4_text load - mlx-lm #1242).
    from unsloth_zoo.mlx.loader import (
        _KNOWN_MLX_LM_STRICT_FALLBACKS,
        _message_matches_known_fallback,
        _raise_if_qk_norm_version_gap,
    )

    msg = (
        "Received 126 parameters not in model: "
        "language_model.model.layers.24.self_attn.k_norm.weight, "
        "language_model.model.layers.24.self_attn.k_proj.weight, "
        "language_model.model.layers.24.self_attn.v_proj.weight"
    )
    # Guard must not raise on the dead shared-KV tail.
    _raise_if_qk_norm_version_gap("gemma4_text", msg, ValueError("orig"))
    # ... and the message must still match the strict=False fallback that loads it.
    rule = _KNOWN_MLX_LM_STRICT_FALLBACKS["gemma4_text"]
    assert _message_matches_known_fallback(msg, rule)


def test_active_layer_k_norm_and_q_norm_still_raises():
    # A genuine QK-norm version gap rejects q_norm/k_norm on active layers WITHOUT
    # the paired KV-sharing k_proj/v_proj - dropping these would break the model,
    # so the guard must still raise even though k_norm is present.
    from unsloth_zoo.mlx.loader import _raise_if_qk_norm_version_gap

    msg = (
        "Received 8 parameters not in model: "
        "model.layers.7.self_attn.k_norm.weight, "
        "model.layers.7.self_attn.q_norm.weight"
    )
    with pytest.raises(ValueError):
        _raise_if_qk_norm_version_gap("gemma4_text", msg, ValueError("orig"))


def test_non_qk_norm_mismatch_passes_through():
    from unsloth_zoo.mlx.loader import _raise_if_qk_norm_version_gap

    # A different extra-weight mismatch (handled by the known gemma4 filter) or an
    # unrelated error must NOT be swallowed by the QK-norm guard.
    _raise_if_qk_norm_version_gap(
        "gemma4", "Received 4 parameters not in model: per_layer_model_projection.scales",
        ValueError("orig"),
    )
    _raise_if_qk_norm_version_gap("llama", "some unrelated value error", ValueError("orig"))
