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

"""Real-runtime check that GGUF staging restores the per-expert HF tensor names a real
mlx-lm MoE model was built from: llama.cpp cannot map stacked ``switch_mlp``."""

import json
import os

import pytest
try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    _METAL = mx.metal.is_available()
except Exception:
    pytest.skip("requires mlx", allow_module_level=True)
metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

CONFIG = dict(
    model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
    moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    rms_norm_eps=1e-6, vocab_size=128, num_experts=4, num_experts_per_tok=2,
    decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True, rope_theta=1e6,
    tie_word_embeddings=False, max_position_embeddings=512,
)


def _stage_merged_moe_model(path, shards=1):
    from mlx_lm.models import qwen3_moe

    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(CONFIG))
    weights = {n: t.astype(mx.bfloat16) for n, t in tree_flatten(model.parameters())}
    mx.eval(*weights.values())
    path.mkdir(parents=True, exist_ok=True)
    names, weight_map = sorted(weights), {}
    for shard in range(shards):
        part = {n: weights[n] for i, n in enumerate(names) if i % shards == shard}
        file = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        mx.save_safetensors(str(path / file), part, metadata={"format": "mlx"})
        weight_map.update({n: file for n in part})
    index = json.dumps({"weight_map": weight_map})
    (path / "model.safetensors.index.json").write_text(index)
    return model, weights


@metal_only
def test_moe_staging_restores_names_mlx_sanitize_maps_back_bitwise(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    path = tmp_path / "merged"
    model, staged = _stage_merged_moe_model(path)
    stacked_names = sorted(name for name in staged if ".switch_mlp." in name)
    assert len(stacked_names) == 3 * CONFIG["num_hidden_layers"]
    split = mutils._prepare_moe_gguf_export_directory(path, model=model)
    assert split == len(stacked_names)

    rewritten = mx.load(str(path / "model-00001-of-00001.safetensors"))
    assert not any(".switch_mlp." in name for name in rewritten)
    assert "model.layers.0.mlp.experts.3.gate_proj.weight" in rewritten
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(rewritten)

    # Replaying mlx-lm's own HF -> MLX map must land the rewritten names back on the
    # tensors the model was saved from.
    restored = model.sanitize(dict(rewritten))
    for name in stacked_names:
        assert _bytes_identical(restored[name], staged[name]), name


@metal_only
def test_moe_staging_holds_no_more_shards_open_as_a_checkpoint_is_split(
    monkeypatch, tmp_path
):
    import unsloth_zoo.mlx.utils as mutils

    def descriptor_growth(shards):
        path = tmp_path / f"merged-{shards}"
        model, _ = _stage_merged_moe_model(path, shards=shards)
        files = sorted(path.glob("*.safetensors"))
        peak = before = len(os.listdir("/dev/fd"))
        real_load = mutils.mx.load

        def counting_load(*args, **kwargs):
            nonlocal peak
            loaded = real_load(*args, **kwargs)
            peak = max(peak, len(os.listdir("/dev/fd")))
            return loaded

        monkeypatch.setattr(mutils.mx, "load", counting_load)
        assert mutils._plan_mlx_moe_gguf_rewrite(model, files)[0]
        monkeypatch.undo()
        return peak - before

    # A slope, not a ceiling: fixed cost cancels, so only retention that scales with
    # the checkpoint shows. Small checkpoint first, so warm-up lands in the baseline.
    assert descriptor_growth(2) >= descriptor_growth(32)


def _bytes_identical(actual, expected):
    """The same tensor as stored: dtype, shape and bits, not numeric equality."""
    return (actual.dtype == expected.dtype and actual.shape == expected.shape
            and bool(mx.all(mx.view(actual, mx.uint8) == mx.view(expected, mx.uint8))))
