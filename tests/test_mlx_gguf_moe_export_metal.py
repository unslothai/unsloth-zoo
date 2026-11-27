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

"""Real-runtime: GGUF staging restores per-expert HF names; llama.cpp cannot map stacked switch_mlp."""

import json
import os
from pathlib import Path

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


def _static_sanitize_family(loader):
    """Resolved at runtime: llava_onevision exists only from mlx-vlm 0.7.0, yet pyproject admits 0.6.4."""
    import importlib
    import inspect
    import pkgutil
    import mlx_vlm.models

    preferred = ("llava_onevision", "granite_vision", "florence2", "deepseek_vl_v2")
    installed = sorted(name for _, name, _ in pkgutil.iter_modules(mlx_vlm.models.__path__))
    for name in (*preferred, *(n for n in installed if n not in preferred)):
        try:
            importlib.import_module(f"mlx_vlm.models.{name}.{name}")
        except Exception:
            continue
        # Use loader._resolve_mlx_vlm_model_class, the class _ensure_native_vlm_weight_names installs on.
        resolved = loader._resolve_mlx_vlm_model_class(name)
        if resolved is not None and isinstance(
            inspect.getattr_static(resolved, "sanitize", None), staticmethod
        ):
            return name, resolved
    return None, None


@metal_only
def test_native_name_sanitizer_preserves_vlm_expert_export(monkeypatch, tmp_path):
    import inspect
    from types import SimpleNamespace
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model
    from unsloth_zoo.mlx import loader, utils

    static_type, StaticModel = _static_sanitize_family(loader)
    if StaticModel is None:
        pytest.skip("no installed mlx-vlm family declares sanitize as a staticmethod")

    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(
        tie_word_embeddings=False, num_hidden_layers=2, num_experts=2,
    ))
    source = {}
    for layer in range(2):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        source[f"{prefix}.gate_up_proj"] = mx.arange(48).reshape(2, 6, 4).astype(mx.float32) + layer * 100
        source[f"{prefix}.down_proj"] = mx.arange(24).reshape(2, 4, 3).astype(mx.float32) + layer * 100
    staged = model.sanitize(dict(source))
    static_model = StaticModel.__new__(StaticModel)
    owners = [model, static_model]
    vocabulary = set(utils._mlx_sanitizer_vocabulary(owners))
    def exported(directory):
        directory.mkdir()
        mx.save_safetensors(str(directory / "model.safetensors"), staged)
        assert utils._prepare_moe_gguf_export_directory(directory, model=model) > 0
        return mx.load(str(directory / "model.safetensors"))

    expected = exported(tmp_path / "before")
    assert any(name.endswith("experts.gate_up_proj") for name in expected)

    monkeypatch.setattr(Model, "sanitize", Model.sanitize)
    monkeypatch.setattr(StaticModel, "sanitize", inspect.getattr_static(StaticModel, "sanitize"))
    loader._ensure_native_vlm_weight_names("qwen3_5_moe")
    loader._ensure_native_vlm_weight_names(static_type)
    for order in (owners, owners[::-1]):
        assert vocabulary == set(utils._mlx_sanitizer_vocabulary(order))
    assert utils._mlx_sanitizer_writes_in_place(utils._mlx_moe_sanitizers(model)[0])
    actual = exported(tmp_path / "after")
    assert actual.keys() == expected.keys()
    for name in actual:
        assert mx.array_equal(actual[name], expected[name]).item()
        assert mx.array_equal(actual[name], source[name]).item()


@metal_only
def test_vlm_fused_expert_export_restores_every_checkpoint_name(tmp_path):
    from types import SimpleNamespace
    import mlx.nn as nn
    from unsloth_zoo.mlx import utils

    Model = pytest.importorskip("mlx_vlm.models.gemma4.gemma4").Model
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.audio_tower = None
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(num_experts=2),
        vision_config=SimpleNamespace(use_clipped_linears=False),
    )
    source = {}
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}.experts"
        source[f"{prefix}.gate_up_proj"] = mx.arange(48).reshape(2, 6, 4).astype(mx.float32) + layer * 100
        source[f"{prefix}.down_proj"] = mx.arange(24).reshape(2, 4, 3).astype(mx.float32) + layer * 100
    mx.save_safetensors(str(tmp_path / "model.safetensors"), model.sanitize(dict(source)))
    assert utils._prepare_moe_gguf_export_directory(tmp_path, model=model) > 0
    restored = mx.load(str(tmp_path / "model.safetensors"))
    assert restored.keys() == source.keys()
    for name in source:
        assert mx.array_equal(restored[name], source[name]).item()


@pytest.mark.parametrize("source_form", ["converted", "unsanitized_layout", "extra_key"])
def test_native_name_sanitizer_keeps_converted_norms(source_form):
    import mlx.nn as nn
    from mlx_simulation import mlx_is_simulated

    # A sibling module can install the torch shim mid-session; it has no nn.RMSNorm.
    if mlx_is_simulated() or "mlx_simulation" in str(getattr(nn, "__file__", "")):
        pytest.skip("needs real MLX, the torch shim is installed")
    from unsloth_zoo.mlx import loader

    class Reshifting(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(2)
            self.dt_bias = mx.zeros((2,))
            self.proj = nn.Linear(3, 2)
            self.head = nn.Linear(2, 2, bias=False)

        def sanitize(self, weights):
            out = {}
            for key, value in weights.items():
                if key.startswith("mtp.") or key == "head.weight":
                    continue
                if key == "proj.weight" and value.shape != (2, 3):
                    value = value.T
                if key.endswith("norm.weight"):
                    value += 1.0
                if key == "dt_bias":
                    value = value * 2.0
                out[key] = value
            return out

    Reshifting.sanitize = loader._NativeVLMWeightSanitizer(Reshifting.__dict__["sanitize"])
    # (w + 1) - 1 rounds away 1.0078125 in bfloat16, so only a restore is exact.
    norm = mx.array([1.0078125, -0.5], dtype=mx.bfloat16)
    proj = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    source = {
        "norm.weight": norm,
        "dt_bias": mx.array([0.5, 0.25]),
        "proj.weight": proj.T if source_form == "unsanitized_layout" else proj,
        "proj.bias": mx.zeros((2,)),
        "proj.scales": mx.ones((2, 1)),
        "head.weight": mx.ones((2, 2)),
    }
    if source_form == "extra_key":
        source["mtp.0.norm.weight"] = mx.array([0.5, 0.25])
    expected = {
        "norm.weight": mx.array(norm) if source_form == "converted" else norm + 1.0,
        "dt_bias": mx.array([1.0, 0.5]),
        "proj.weight": proj,
        "proj.bias": mx.zeros((2,)),
        "proj.scales": mx.ones((2, 1)),
    }

    sanitized = Reshifting().sanitize(source)

    assert sanitized.keys() == expected.keys()
    for key, value in expected.items():
        assert mx.array_equal(sanitized[key], value).item(), key

    # GGUF export must still measure the shift it converts back.
    from unsloth_zoo.mlx import utils as mlx_utils
    model = Reshifting()
    offsets = mlx_utils._mlx_measure_norm_offsets(
        lambda probe: mlx_utils._mlx_sanitize_probe(model, probe),
        {key: mx.array(value) for key, value in source.items()},
    )
    assert offsets == {"norm.weight": 1.0}


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

    # Replaying mlx-lm's HF-to-MLX map must land rewritten names back on the tensors it was saved from.
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

    # Slopes, not ceilings: fixed cost cancels; a small checkpoint first puts warm-up in the baseline.
    assert descriptor_growth(2) >= descriptor_growth(32)


def _bytes_identical(actual, expected):
    """The same tensor as stored: dtype, shape and bits, not numeric equality."""
    return (actual.dtype == expected.dtype and actual.shape == expected.shape
            and bool(mx.all(mx.view(actual, mx.uint8) == mx.view(expected, mx.uint8))))


def _tiny_inkling2(model_type):
    ink = pytest.importorskip("mlx_vlm.models.inkling2.inkling2")
    from mlx.utils import tree_map
    from mlx_vlm.models.inkling2.config import ModelConfig

    loop = {"loop_config": {"core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8, "is_varloop": False}} if model_type == "inkling2_loop" else {}
    text = dict(model_type=f"{model_type}_text", vocab_size=256, hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, head_dim=32, swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=32,
                sliding_window_size=8, local_layer_ids=[0, 1, 3, 4], d_rel=4, rel_extent=8, sconv_kernel_size=4,
                intermediate_size=64, n_routed_experts=4, num_experts_per_tok=2, n_layers_per_attn_res_block=2, **loop)
    vision = dict(hidden_size=64, num_hidden_layers=1, num_attention_heads=2, projection_hidden_dim=64, text_hidden_size=128)
    config = {"model_type": model_type, "text_config": text, "vision_config": vision}
    model = ink.Model(ModelConfig.from_dict(json.loads(json.dumps(config))))
    if loop and "message_queries" not in model.language_model.model:
        pytest.skip(reason="Inkling 2 Loop needs mlx-vlm >= 0.8.1")
    mx.random.seed(0)
    model.update(tree_map(lambda value: mx.random.normal(value.shape) * 0.05, model.parameters()))
    return model, config, bool(loop)


@metal_only
@pytest.mark.parametrize("model_type", ["inkling2", "inkling2_loop"])
def test_inkling2_export_is_the_source_layout_mlx_vlm_sanitizes_back_bitwise(tmp_path, model_type):
    """llama.cpp's Inkling 2 converter reads only the source checkpoint names, which mlx-vlm's sanitize maps from."""
    from unsloth_zoo.mlx.utils import _restore_mlx_gguf_source_layout

    model, config, loop = _tiny_inkling2(model_type)
    params = dict(tree_flatten(model.parameters()))

    # Up, then everything else, then gate projections: the fused expert tensor lands in the last shard and reads the first.
    def shard_of(k):
        return 1 if "up_proj" in k else 3 if "gate_proj" in k else 2
    shards = {f"model-0000{i}-of-00003.safetensors": {k: v for k, v in params.items() if shard_of(k) == i} for i in (1, 2, 3)}
    for shard, tensors in shards.items():
        mx.save_safetensors(str(tmp_path / shard), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: s for s, t in shards.items() for k in t}}))
    (tmp_path / "config.json").write_text(json.dumps(config))

    count = _restore_mlx_gguf_source_layout(tmp_path)
    located = {k: file.name for file in sorted(tmp_path.glob("*.safetensors")) for k in mx.load(str(file))}
    exported = {k: v for file in sorted(tmp_path.glob("*.safetensors")) for k, v in mx.load(str(file)).items()}
    assert count == len(exported) and json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"] == located
    mlx_only = ("self_attn.", "switch_mlp.", "layernorm", "message_queries", "embed_tokens", ".conv.", "gate_weight", "e_score",
                "_input.query", "static_weight", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "attn.r_proj",
                "attn.gate_proj", "attn.sink_proj", "q_scale", "rel_scale", "rel_proj")
    assert not [k for k in exported if not k.startswith(("model.llm.", "model.visual."))]
    assert not [k for k in exported if k.startswith("model.llm.") and any(s in k for s in mlx_only)]
    assert not [k for k in exported if k.startswith("model.visual.") and any(s in k for s in ("layernorm", "lambda_", ".fc1.", ".fc2."))]
    assert "model.llm.layers.1.mlp.experts.w13_weight" in exported and "model.llm.layers.2.attn.k_sconv.static_w" in exported
    assert ("model.llm.middle_cycle_cross_pass_kv_queries" in exported) == bool(loop)
    # The converter squeezes axis 1 of the [channels, 1, kernel] source convolutions; mlx-vlm accepts either layout.
    for name in ("attn_sconv", "mlp_sconv", "attn.k_sconv", "attn.v_sconv"):
        assert exported[f"model.llm.layers.0.{name}.weight"].shape[1:] == (1, 4), name
    if loop:  # two queries per core layer, as the converter checks
        assert exported["model.llm.middle_cycle_cross_pass_queries"].shape == exported["model.llm.middle_cycle_cross_pass_kv_queries"].shape == (4, 128)
    assert {"model.visual.backbone.ln_pre.weight", "model.visual.proj_in.weight", "model.visual.proj_out.weight"} <= exported.keys()

    restored = model.sanitize(exported)
    assert restored.keys() == params.keys()
    for name, value in params.items():
        assert _bytes_identical(restored[name], value), name


@metal_only
def test_inkling2_save_pretrained_gguf_hands_the_converter_dense_source_tensors(monkeypatch, tmp_path):
    """Through save_pretrained_gguf: a quantized Inkling 2 reached the converter in mlx-vlm names with packed experts."""
    pytest.importorskip("torch")
    import mlx.nn as nn
    from unsloth_zoo import llama_cpp
    from unsloth_zoo.mlx import utils

    model, config, _ = _tiny_inkling2("inkling2")
    shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and "vision" not in p and m.weight.shape[-1] % 64 == 0)
    assert any(type(m).__name__ == "QuantizedSwitchLinear" for _, m in model.named_modules())
    model._config = config
    seen = {}

    def fake_convert_to_gguf(**kwargs):
        seen.update({k: v for f in Path(kwargs["input_folder"]).glob("*.safetensors") for k, v in mx.load(str(f)).items()})
        output = Path(f"{kwargs['model_name']}.{kwargs['quantization_type'].upper()}.gguf")
        output.write_bytes(b"GGUF")
        return [str(output)], False

    converter = str(tmp_path / "convert_hf_to_gguf.py")
    monkeypatch.setattr(llama_cpp, "check_llama_cpp", lambda folder: (str(tmp_path / "llama-quantize"), converter))
    monkeypatch.setattr(llama_cpp, "_download_convert_hf_to_gguf", lambda: (converter, None, None))
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", fake_convert_to_gguf)
    tokenizer = type("Tokenizer", (), {"save_pretrained": lambda self, path: None})()
    utils.save_pretrained_gguf(model, tokenizer, tmp_path / "out", quantization_method="bf16")

    assert not [k for k in seen if not k.startswith(("model.llm.", "model.visual."))]
    assert not [k for k, v in seen.items() if v.dtype == mx.uint32 or k.endswith((".scales", ".biases"))]
    restored = model.sanitize(seen)
    assert {k: v.shape for k, v in restored.items()} == shapes
