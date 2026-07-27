# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    assert "mlx-lm" in msg and "q_norm" in msg and "gemma4" in msg
    assert "unsloth-zoo" in msg and "mlx-vlm" in msg and "mlx-audio" in msg
    assert "Studio users should rerun installer/repair" in msg
    assert "strict=False" in msg
    assert "0.31.3 broke" not in msg and "!=0.31.3" not in msg


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
    # First load fails on allow-listed extras (entering the filtered retry); the
    # retry then hits a q_norm/k_norm gap, which must surface the actionable error.
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
    assert "mlx-lm" in msg and "q_norm" in msg and "gemma4" in msg
    assert "unsloth-zoo" in msg and "mlx-audio" in msg and "!=0.31.3" not in msg
    assert calls["n"] == 2  # it must have entered the filtered retry


def test_vlm_retry_filters_only_owned_shared_kv(monkeypatch):
    import threading, types

    import mlx.nn as nn
    import unsloth_zoo.mlx.loader as loader

    def unused(self, key):
        layer = int(key.removeprefix("language_model.model.layers.").split(".", 1)[0])
        return layer < len(self.model.layers) and self.model.layers[layer].self_attn.is_kv_shared_layer

    class LanguageModel:
        __module__ = "mlx_vlm.models.gemma4.language"
        _is_unused_shared_kv_weight = unused

    language = LanguageModel()
    language.model = types.SimpleNamespace(
        layers=[types.SimpleNamespace(
            self_attn=types.SimpleNamespace(is_kv_shared_layer=value)
        ) for value in (False, True)]
    )
    model = types.SimpleNamespace(language_model=language)
    expected_hash = "e859b50eb966089f8966a38ca2e69c3b68ccf48c5b4a264fbc01f415be60003f"
    assert loader._GEMMA4_UNUSED_SHARED_KV_TOKEN_SHA256 == expected_hash
    fingerprint = loader._source_token_sha256(loader._safe_getsource(unused))
    monkeypatch.setattr(loader, "_GEMMA4_UNUSED_SHARED_KV_TOKEN_SHA256", fingerprint)
    prefix = "language_model.model.layers."
    known = [f"{prefix}1.self_attn.{name}" for name in ("k_norm.weight", "k_proj.scales", "v_proj.biases")]
    assert all(loader._gemma4_unused_shared_kv_weight(model, key) for key in known)
    bad = tuple(f"{prefix}{suffix}" for suffix in (
        "-1.self_attn.k_proj.weight", "01.self_attn.k_proj.weight",
        "1.self_attn.q_proj.weight", "1.self_attn.k_proj.garbage",
    ))
    assert not any(loader._gemma4_unused_shared_kv_weight(model, key) for key in bad)

    monkeypatch.setattr(loader, "_GEMMA4_UNUSED_SHARED_KV_TOKEN_SHA256", "unknown")
    assert not loader._gemma4_unused_shared_kv_weight(model, known[0])
    monkeypatch.setattr(loader, "_GEMMA4_UNUSED_SHARED_KV_TOKEN_SHA256", fingerprint)
    LanguageModel.__name__ = "Other"
    assert not loader._gemma4_unused_shared_kv_weight(model, known[0])
    LanguageModel.__name__, LanguageModel.__module__ = "LanguageModel", "foreign"
    assert not loader._gemma4_unused_shared_kv_weight(model, known[0])
    LanguageModel.__module__ = "mlx_vlm.models.gemma4.language"
    Inherited = type("LanguageModel", (LanguageModel,), {"__module__": LanguageModel.__module__})
    inherited = Inherited()
    inherited.model = language.model
    assert not loader._gemma4_unused_shared_kv_weight(
        types.SimpleNamespace(language_model=inherited), known[0]
    )

    initial = "Received parameters not in model: " + ", ".join(known)
    state = {"attempt": 0, "weights": [(key, object()) for key in known], "concurrent": True}
    seen, foreign = [], []

    def strict_load(_self, weights, strict=True):
        keys = [key for key, _ in weights]
        seen.append((keys, strict))
        if weights:
            raise ValueError("Received parameters not in model: " + ", ".join(keys))
        return "loaded"

    def foreign_load():
        try:
            nn.Module.load_weights(model, state["weights"])
        except ValueError as error:
            foreign.append(str(error))

    def vlm_load(_model_name, **_kwargs):
        state["attempt"] += 1
        if state["attempt"] == 1:
            raise ValueError(initial)
        if state.pop("concurrent", False):
            thread = threading.Thread(target=foreign_load)
            thread.start()
            thread.join()
        return nn.Module.load_weights(model, state["weights"])

    monkeypatch.setattr(nn.Module, "load_weights", strict_load)

    def partial(*_args, **_kwargs):
        raise ValueError("parameters not in model: self_attn.k_proj")

    with pytest.raises(ValueError, match="k_proj"):
        loader._load_mlx_vlm_with_extra_weight_filter("model", "gemma4", partial, {})
    assert nn.Module.load_weights is strict_load
    native = loader._load_mlx_vlm_with_extra_weight_filter(
        "model", "gemma4", lambda *_a, **_k: "native", {})
    assert native == "native"
    assert loader._load_mlx_vlm_with_extra_weight_filter("model", "gemma4", vlm_load, {}) == "loaded"
    assert seen == [(known, True), ([], True)]
    assert known[0] in foreign[0]
    assert nn.Module.load_weights is strict_load
