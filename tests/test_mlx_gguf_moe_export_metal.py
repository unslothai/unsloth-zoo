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


STEP3P5_CONFIG = dict(
    model_type="step3p5", vocab_size=256, hidden_size=64, intermediate_size=128,
    moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=4, num_attention_groups=1, moe_num_experts=4, moe_top_k=2,
    rms_norm_eps=1e-5, rope_theta=1e4, head_dim=16, max_position_embeddings=512,
    share_expert_dim=32, moe_layers_enum="0,1", tie_word_embeddings=False,
)


@metal_only
def test_moe_staging_recovers_names_a_real_sanitizer_only_renamed(tmp_path):
    """step3p5 keeps its experts fused in HF and is renamed, never un-stacked."""
    import unsloth_zoo.mlx.utils as mutils
    from mlx_lm.models import step3p5

    model = step3p5.Model(step3p5.ModelArgs.from_dict(STEP3P5_CONFIG))
    staged = {n: t.astype(mx.bfloat16) for n, t in tree_flatten(model.parameters())}
    mx.eval(*staged.values())
    path = tmp_path / "merged"
    path.mkdir()
    mx.save_safetensors(str(path / "model.safetensors"), staged, {"format": "mlx"})

    rewritten = mutils._prepare_moe_gguf_export_directory(path, model=model)
    # Every MLX-only name, plus the norms whose stored value is one above the
    # checkpoint convention llama.cpp reads.
    assert rewritten == len([n for n in staged if ".mlp." in n or "norm" in n])

    rewritten = mx.load(str(path / "model.safetensors"))
    assert "model.layers.0.moe.gate_proj.weight" in rewritten
    assert "model.layers.0.share_expert.down_proj.weight" in rewritten
    # The expert tensors stay fused: llama.cpp reads them in exactly this shape.
    assert rewritten["model.layers.0.moe.gate_proj.weight"].ndim == 3
    restored = model.sanitize(dict(rewritten))
    for name, tensor in staged.items():
        assert bool(mx.array_equal(restored[name], tensor)), name


KIMI_LINEAR_CONFIG = dict(
    model_type="kimi_linear", vocab_size=256, hidden_size=64, intermediate_size=128,
    moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=4, num_experts=4, first_k_dense_replace=0, rms_norm_eps=1e-5,
    rope_theta=1e4, max_position_embeddings=512, head_dim=16, kv_lora_rank=16,
    num_experts_per_token=2, num_shared_experts=1, qk_nope_head_dim=16,
    qk_rope_head_dim=16, v_head_dim=16, model_max_length=512,
    tie_word_embeddings=False, routed_scaling_factor=1.0,
    linear_attn_config={"kda_layers": [1], "short_conv_kernel_size": 4,
                        "head_dim": 16, "num_heads": 4},
)


def _stage_real_model(path, module, config, shards=1):
    """Write what save_merged_model produces for one real mlx-lm model."""
    model = module.Model(module.ModelArgs.from_dict(config))
    staged = {n: t.astype(mx.bfloat16) for n, t in tree_flatten(model.parameters())}
    mx.eval(*staged.values())
    path.mkdir(parents=True, exist_ok=True)
    names, weight_map = sorted(staged), {}
    for shard in range(shards):
        part = {n: staged[n] for i, n in enumerate(names) if i % shards == shard}
        file = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        mx.save_safetensors(str(path / file), part, metadata={"format": "mlx"})
        weight_map.update({n: file for n in part})
    index = json.dumps({"weight_map": weight_map})
    (path / "model.safetensors.index.json").write_text(index)
    return model, staged


def _restores_bitwise(model, path, staged):
    """Assert mlx-lm's own sanitize maps the rewritten directory back exactly.

    Also checks the index still names the shard each tensor is really in.
    """
    rewritten = {}
    index = json.loads((path / "model.safetensors.index.json").read_text())
    for file in sorted(path.glob("*.safetensors")):
        held = mx.load(str(file))
        for name in held:
            assert index["weight_map"].get(name) == file.name, name
        rewritten.update(held)
    assert set(index["weight_map"]) == set(rewritten)

    restored = model.sanitize(dict(rewritten))
    # Exactly the checkpoint, not a superset: a tensor left behind under its MLX
    # name is invisible to the round-trip, because the sanitizer overwrites it
    # from the one that replaced it, and llama.cpp rejects what it cannot map.
    assert set(restored) == set(staged)
    for name, tensor in staged.items():
        assert bool(mx.array_equal(restored[name], tensor)), name
    return rewritten


@metal_only
def test_moe_staging_restores_a_layout_a_real_sanitizer_moved(tmp_path):
    """Kimi Linear renames its KDA convolutions and swaps two of their axes."""
    import unsloth_zoo.mlx.utils as mutils
    from mlx_lm.models import kimi_linear

    path = tmp_path / "merged"
    model, staged = _stage_real_model(path, kimi_linear, KIMI_LINEAR_CONFIG)

    assert mutils._prepare_moe_gguf_export_directory(path, model=model)

    rewritten = _restores_bitwise(model, path, staged)
    moved = rewritten["model.layers.0.self_attn.k_conv1d.weight"]
    staged_conv = staged["model.layers.0.self_attn.k_conv.conv.weight"]
    assert tuple(moved.shape) == (staged_conv.shape[0], 1, staged_conv.shape[1])
    # The whole MoE block is relocated, not only the expert tensors, and only
    # the converter-side spellings of all of it are mappable.
    for leaf in ("experts.3.w2.weight", "gate.weight", "gate.e_score_correction_bias",
                 "shared_experts.down_proj.weight"):
        assert f"model.layers.1.block_sparse_moe.{leaf}" in rewritten
    assert not any(
        ".mlp." in name or ".switch_mlp." in name or "_conv.conv." in name
        for name in rewritten
    )
