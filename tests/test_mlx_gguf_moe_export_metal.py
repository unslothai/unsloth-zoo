"""Real-runtime check that GGUF staging restores the per-expert tensor names a real
mlx-lm MoE model was built from: llama.cpp cannot map MLX's stacked ``switch_mlp``
parameters, so the staged directory has to carry the HF names."""

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
    # exact tensors the model was saved from.
    restored = model.sanitize(dict(rewritten))
    for name in stacked_names:
        assert bool(mx.array_equal(restored[name], staged[name])), name


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
        assert mutils._plan_mlx_moe_expert_unstacking(model, files)
        monkeypatch.undo()
        return peak - before

    # A slope, not a ceiling: fixed loading and sampling cost cancels, so only
    # retention that scales with the checkpoint shows. The small checkpoint goes
    # first so warm-up lands in the baseline.
    assert descriptor_growth(2) >= descriptor_growth(32)
