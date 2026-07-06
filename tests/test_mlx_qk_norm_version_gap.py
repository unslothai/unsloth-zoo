# Unsloth Zoo - Utilities for Unsloth
# Strict mlx load rejecting q_norm/k_norm = mlx-lm/mlx-vlm too old for a QK-norm
# arch; the guard must raise a clear error instead of the raw mlx ValueError.

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
            "qwen3_5",
            "Received 5 parameters not in model: model.layers.3.self_attn.q_norm.weight",
            ValueError("orig"),
        )


def test_kv_sharing_dead_tail_falls_through_to_strict_false():
    # Dead KV-sharing tail (k_proj+v_proj+k_norm, never q_norm) is safe to drop
    # via the strict=False fallback: the guard must NOT raise (mlx-lm #1242).
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
    _raise_if_qk_norm_version_gap("gemma4_text", msg, ValueError("orig"))
    # The message must still match the strict=False fallback that loads it.
    rule = _KNOWN_MLX_LM_STRICT_FALLBACKS["gemma4_text"]
    assert _message_matches_known_fallback(msg, rule)


def test_active_layer_k_norm_and_q_norm_still_raises():
    # Active-layer q_norm/k_norm without paired k_proj/v_proj = genuine gap:
    # must raise even though k_norm is present.
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

    # Other extra-weight mismatches / unrelated errors must pass through.
    _raise_if_qk_norm_version_gap(
        "gemma4",
        "Received 4 parameters not in model: per_layer_model_projection.scales",
        ValueError("orig"),
    )
    _raise_if_qk_norm_version_gap(
        "llama", "some unrelated value error", ValueError("orig")
    )


def test_vlm_retry_qk_norm_mismatch_raises_actionable_error():
    # First strict VLM load fails on the allow-listed per_layer_model_projection
    # extras (so the code enters the filtered retry); the retry then hits an
    # older-mlx-vlm q_norm/k_norm mismatch, which must surface the actionable
    # version-gap error rather than escaping as the raw strict-load ValueError.
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_with_extra_weight_filter

    calls = {"n": 0}
    first = (
        "Received 4 parameters not in model: "
        "language_model.model.per_layer_model_projection.scales, "
        "language_model.model.per_layer_model_projection.biases"
    )
    retry = (
        "Received 8 parameters not in model: "
        "language_model.model.layers.15.self_attn.k_norm.weight, "
        "language_model.model.layers.15.self_attn.q_norm.weight"
    )

    def fake_vlm_load(model_name, **kwargs):
        calls["n"] += 1
        raise ValueError(first if calls["n"] == 1 else retry)

    with pytest.raises(ValueError) as exc:
        _load_mlx_vlm_with_extra_weight_filter("some/model", "gemma4", fake_vlm_load, {}, hf_token=None)
    msg = str(exc.value)
    assert "mlx-lm" in msg and "0.31.3" in msg and "gemma4" in msg
    assert calls["n"] == 2  # it must have entered the filtered retry
