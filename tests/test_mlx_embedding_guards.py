# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Sentence-transformers layout shim, pooling-mode detection, batch advisory.

Pure dict/string/arithmetic logic, so this runs under the torch shim on Linux CI
rather than skipping. The numeric behaviour lives in
test_mlx_embedding_pooling.py, which needs real MLX.

The layout shim matters because Qwen/Qwen3-Embedding-0.6B -- the obvious model for
this feature -- does not load without it: sentence-transformers stores the inner
transformer at the repo root, so all 310 keys arrive without the ``model.`` prefix
mlx_lm expects and loading fails with "Received 310 parameters not in model".
"""

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _embedding():
    import unsloth_zoo.mlx.embedding as embedding
    return embedding


# --- pooling-mode detection --------------------------------------------------
def test_pooling_map_matches_cuda_exactly():
    """Same map as unsloth/models/sentence_transformer.py:587-599, so a checkpoint
    resolves to the same mode on both backends."""
    expected = {
        "pooling_mode_cls_token": "cls",
        "pooling_mode_mean_tokens": "mean",
        "pooling_mode_max_tokens": "max",
        "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
        "pooling_mode_weightedmean_tokens": "weightedmean",
        "pooling_mode_lasttoken": "lasttoken",
    }
    assert _embedding().SENTENCE_TRANSFORMERS_POOLING_MAP == expected


def test_qwen3_embedding_resolves_to_lasttoken():
    """Its real 1_Pooling/config.json. Defaulting to mean would pool the wrong thing."""
    config = {
        "word_embedding_dimension": 1024,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": True,
        "include_prompt": True,
    }
    assert _embedding().read_pooling_mode(config) == "lasttoken"


@pytest.mark.parametrize("key,mode", [
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_mean_tokens", "mean"),
    ("pooling_mode_max_tokens", "max"),
    ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len"),
    ("pooling_mode_weightedmean_tokens", "weightedmean"),
    ("pooling_mode_lasttoken", "lasttoken"),
])
def test_every_mode_is_detectable(key, mode):
    assert _embedding().read_pooling_mode({key: True}) == mode


def test_missing_or_empty_config_falls_back_to_mean():
    e = _embedding()
    assert e.read_pooling_mode(None) == "mean"
    assert e.read_pooling_mode({}) == "mean"
    assert e.read_pooling_mode({"pooling_mode_mean_tokens": False}) == "mean"


# --- layout shim -------------------------------------------------------------
ST_KEYS = [
    "embed_tokens.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "norm.weight",
]
HF_KEYS = ["model." + k for k in ST_KEYS]


def test_sentence_transformers_layout_is_detected():
    e = _embedding()
    assert e.is_sentence_transformers_layout(ST_KEYS) is True
    assert e.is_sentence_transformers_layout(HF_KEYS) is False
    assert e.is_sentence_transformers_layout([]) is False


def test_remap_restores_the_model_prefix():
    e = _embedding()
    remapped = e.remap_sentence_transformer_weights({k: object() for k in ST_KEYS})
    assert sorted(remapped) == sorted(HF_KEYS)


def test_remap_preserves_values_identically():
    """Values must pass through untouched -- only keys change."""
    e = _embedding()
    original = {k: f"tensor-{i}" for i, k in enumerate(ST_KEYS)}
    remapped = e.remap_sentence_transformer_weights(original)
    for key, value in original.items():
        assert remapped["model." + key] is value


def test_remap_is_a_noop_on_plain_hf_layout():
    e = _embedding()
    original = {k: object() for k in HF_KEYS}
    assert e.remap_sentence_transformer_weights(original) == original


def test_remap_leaves_lm_head_at_top_level():
    e = _embedding()
    remapped = e.remap_sentence_transformer_weights(
        {"layers.0.mlp.down_proj.weight": 1, "lm_head.weight": 2}
    )
    assert "model.layers.0.mlp.down_proj.weight" in remapped
    assert "lm_head.weight" in remapped
    assert "model.lm_head.weight" not in remapped


def test_remap_does_not_double_prefix():
    e = _embedding()
    mixed = {"model.norm.weight": 1, "layers.0.mlp.down_proj.weight": 2}
    remapped = e.remap_sentence_transformer_weights(mixed)
    assert not any(k.startswith("model.model.") for k in remapped)


# --- batch-size advisory -----------------------------------------------------
def test_estimate_reproduces_the_measured_points():
    """Fitted on measured Qwen3-0.6B steps: 4->4.76, 8->7.92, 16->14.24 GB."""
    e = _embedding()
    for batch, measured in ((4, 4.76), (8, 7.92), (16, 14.24)):
        assert abs(e.estimate_peak_gb(batch) - measured) < 0.6, (
            f"batch={batch}: estimate {e.estimate_peak_gb(batch):.2f} GB vs measured {measured}"
        )


def test_estimate_extrapolates_to_the_measured_batch32_point():
    """Measured once at 26.46 GB (via swap) on a 16 GB machine."""
    assert abs(_embedding().estimate_peak_gb(32) - 26.46) < 1.0


def test_recommendation_fits_within_the_safety_budget():
    e = _embedding()
    for ram in (16, 32, 64, 128):
        batch, _ = e.recommend_batch_size(ram)
        assert batch >= 1
        assert e.estimate_peak_gb(batch) <= ram * e.MEMORY_SAFETY_FRACTION


def test_recommendation_grows_with_ram_and_shrinks_with_seq_len():
    e = _embedding()
    assert e.recommend_batch_size(64)[0] > e.recommend_batch_size(16)[0]
    assert e.recommend_batch_size(16, seq_len=1024)[0] < e.recommend_batch_size(16, seq_len=512)[0]
    assert e.recommend_batch_size(16, hidden=2048)[0] < e.recommend_batch_size(16, hidden=1024)[0]


def test_small_machines_get_a_warning_not_an_exception():
    """Advisory only: oversubscription here degrades into swap, not breakage, so a
    hard refusal would be wrong."""
    e = _embedding()
    batch, warning = e.recommend_batch_size(8)
    assert batch >= 1
    assert warning and "batch_size" in warning
    tiny_batch, tiny_warning = e.recommend_batch_size(1)
    assert tiny_batch == 1 and tiny_warning


def test_comfortable_machine_gets_no_warning():
    _, warning = _embedding().recommend_batch_size(64)
    assert warning == ""
