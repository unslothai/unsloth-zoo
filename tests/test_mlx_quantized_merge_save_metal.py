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

"""merged_4bit really yields 4-bit via the public save_pretrained_merged; pre-quantized bases, merged_16bit and push_to_hub_merged must not change."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)

import mlx.core as mx
import mlx.nn as nn

from unsloth_zoo.mlx.loader import FastMLXModel

QUANTIZED_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
FP_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(autouse=True)
def _require_real_metal():
    import mlx.core as _mx
    if not (getattr(_mx, "metal", None) and _mx.metal.is_available()
            and _mx.default_device() == _mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


def _saved_artifact(model, tokenizer, method):
    """Save through the public entry point and report what landed on disk."""
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method=method)
        config = json.load(open(os.path.join(directory, "config.json")))
        weight_bytes = sum(
            os.path.getsize(os.path.join(directory, name))
            for name in os.listdir(directory)
            if name.endswith(".safetensors")
        )
    return config.get("quantization") or config.get("quantization_config"), weight_bytes


def _load_and_adapt(model_name, **load_kwargs):
    model, tokenizer = FastMLXModel.from_pretrained(
        model_name, max_seq_length=64, **load_kwargs)
    if not load_kwargs.get("full_finetuning"):
        model = FastMLXModel.get_peft_model(
            model, r=8, lora_alpha=16, lora_dropout=0,
            use_gradient_checkpointing=False,
        )
    return model, tokenizer


def test_merged_4bit_quantizes_a_16bit_lora_model():
    model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_4bit")

    assert quantization == {"group_size": 64, "bits": 4, "mode": "affine"}, (
        f"merged_4bit wrote no quantization metadata: {quantization}"
    )
    # ~988MB unquantized -> ~474MB at 4-bit; assert well under the fp16 size.
    assert weight_bytes < 700e6, f"weights still {weight_bytes / 1e6:.1f}MB"


def test_merged_4bit_quantizes_a_full_finetuned_model():
    model, tokenizer = _load_and_adapt(FP_MODEL, full_finetuning=True)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_4bit")

    assert quantization == {"group_size": 64, "bits": 4, "mode": "affine"}
    # The checkpoint dtype comes from the config (bfloat16), not the float32 training, so this holds to the 16-bit LoRA bound.
    assert weight_bytes < 700e6, f"weights still {weight_bytes / 1e6:.1f}MB"


def test_full_finetuned_and_lora_merges_agree_on_size():
    """Same base and grid: a size gap between a float32 full-finetune and a 16-bit LoRA run means a training dtype leaked into the checkpoint."""
    lora_model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    _, lora_bytes = _saved_artifact(lora_model, tokenizer, "merged_4bit")
    del lora_model

    ft_model, ft_tokenizer = _load_and_adapt(FP_MODEL, full_finetuning=True)
    _, ft_bytes = _saved_artifact(ft_model, ft_tokenizer, "merged_4bit")

    assert abs(ft_bytes - lora_bytes) < 0.02 * lora_bytes, (
        f"full_finetuning wrote {ft_bytes / 1e6:.1f}MB vs "
        f"{lora_bytes / 1e6:.1f}MB for the 16-bit LoRA merge"
    )


def test_merged_4bit_artifact_has_no_float32_tensors():
    """float32 scales/embeddings are the tell: packed weights are uint32, so check the float dtype, not whether 32 appears in the name."""
    model, tokenizer = _load_and_adapt(FP_MODEL, full_finetuning=True)
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method="merged_4bit")
        float32 = []
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".safetensors"):
                continue
            for key, value in mx.load(os.path.join(directory, name)).items():
                if value.dtype == mx.float32:
                    float32.append(key)
    assert not float32, f"float32 tensors in a 4-bit artifact: {float32[:8]}"


def test_quantized_save_reloads_and_runs():
    """The artifact has to be loadable, not merely smaller."""
    model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    ids = mx.array([[3, 11, 19, 27, 35, 43, 51, 59]])

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained_merged(directory, tokenizer, save_method="merged_4bit")
        reloaded, _ = FastMLXModel.from_pretrained(directory, max_seq_length=64)
        logits = reloaded(ids)
        logits = getattr(logits, "logits", logits)
        mx.eval(logits)

    assert bool(mx.all(mx.isfinite(logits)).item()), "reloaded model produced non-finite logits"


def test_already_quantized_merged_4bit_is_unchanged():
    """An already-4bit base must not be re-quantized; its metadata passes through verbatim and mlx-community configs may omit mode, so check the grid only."""
    model, tokenizer = _load_and_adapt(QUANTIZED_MODEL)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_4bit")
    assert quantization is not None
    assert quantization["bits"] == 4
    assert quantization["group_size"] == 64
    assert quantization.get("mode", "affine") == "affine"
    assert weight_bytes < 700e6


def test_merged_16bit_still_writes_full_precision():
    """merged_16bit must never acquire quantization metadata."""
    model, tokenizer = _load_and_adapt(QUANTIZED_MODEL)
    quantization, weight_bytes = _saved_artifact(model, tokenizer, "merged_16bit")
    assert quantization is None, f"merged_16bit gained quantization: {quantization}"
    assert weight_bytes > 700e6


def test_push_to_hub_merged_does_not_quantize():
    """push_to_hub_merged calls save_merged_model with default dequantize=False, so inferring quantize from that flag would 4-bit every unquantized push."""
    from unsloth_zoo.mlx.utils import save_merged_model

    model, tokenizer = _load_and_adapt(
        FP_MODEL, load_in_16bit=True, load_in_4bit=False)
    with tempfile.TemporaryDirectory() as directory:
        # Exactly the call push_to_hub_merged makes.
        save_merged_model(model, tokenizer, directory)
        config = json.load(open(os.path.join(directory, "config.json")))
    quantization = config.get("quantization") or config.get("quantization_config")
    assert quantization is None, (
        f"push_to_hub_merged's save path quantized unexpectedly: {quantization}"
    )


# Ternary Bonsai 2 packs, from mlx-vlm and (0.32+) mlx-lm.
_HADAMARD_PACKS = pytest.mark.parametrize("pack_module", [
    "mlx_vlm.models.prism_hadamard_qwen35.prism_hadamard_qwen35", "mlx_lm.models.prism_hadamard_qwen35",
], ids=["mlx_vlm", "mlx_lm"])


@pytest.mark.parametrize("config_on", ["_config", "config"])
@_HADAMARD_PACKS
def test_merged_16bit_unpacks_hadamard_layers_to_the_dense_layers_they_compute(tmp_path, config_on, pack_module):
    pack = pytest.importorskip(pack_module)
    rotate = getattr(pack, "hadamard_transform", None) or pack.hadamard_rotate
    from unsloth_zoo.mlx.utils import LoRAHadamardLinear, save_merged_model
    mx.random.seed(5)

    def packed(cls, rows):
        layer = cls(1024, rows, 512)
        layer.signs = mx.where(mx.random.uniform(shape=(1024,)) < 0.5, -1.0, 1.0)
        rotated = rotate(mx.random.normal((rows, 1024)) * 0.05, 512, layer.signs)
        layer.weight, scales, biases = mx.quantize(rotated, group_size=128, bits=2)
        layer.scales, layer.biases = scales.astype(mx.float16), biases.astype(mx.float16)
        return layer

    model = nn.Module()
    model.proj = packed(pack.HadamardQuantizedLinear, 256)
    model.embed = packed(pack.HadamardQuantizedEmbedding, 300)
    adapted = LoRAHadamardLinear.from_base(packed(pack.HadamardQuantizedLinear, 128), r=4, scale=2.0)
    adapted.lora_b = mx.random.normal(adapted.lora_b.shape) * 0.05
    model.adapted = adapted
    x, ids = mx.random.normal((3, 1024)), mx.array([0, 17, 299])
    forward = lambda: [model.proj(x), model.embed(ids), model.embed.as_linear(x), model.adapted(x)]
    want = forward()
    with pytest.raises(ValueError, match="merged_16bit"):
        adapted.fuse(dequantize=False)
    config = {"model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5", "modules": [],
              "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
              "text_config": {"hidden_size": 1024}, "image_token_id": 7}
    if config_on == "_config":
        model._config = config
    else:  # mlx_vlm.load exposes it on `model.config` only
        model.config = type("Config", (), {"to_dict": lambda self: dict(config)})()
    save_merged_model(model, type("Tokenizer", (), {"save_pretrained": lambda self, path: None})(),
                      tmp_path, dequantize=True)
    assert json.loads((tmp_path / "config.json").read_text()) == {
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 1024}, "image_token_id": 7}
    assert [type(m) for m in (model.proj, model.embed, model.adapted)] == [nn.Linear, nn.Embedding, nn.Linear]
    for expected, actual in zip(want, forward()):
        expected, actual = expected.astype(mx.float32), actual.astype(mx.float32)
        assert (mx.abs(actual - expected).max() / mx.abs(expected).max()).item() < 2e-3


def _nvfp4_supports_pre_quant():
    import inspect
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedLinear

    return "pre_quant_scale" in inspect.signature(ScaledQuantizedLinear.__init__).parameters


def _nvfp4_scaled_module(kind, scale_name, *, with_pre_quant=False, pre_2d=False, scalar_gs=False, seed=7):
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedLinear, ScaledQuantizedSwitchLinear

    if with_pre_quant and not _nvfp4_supports_pre_quant():
        pytest.skip("pre_quant_scale first ships in mlx-vlm 0.8.0; 0.7.8-0.7.10 constructors take three arguments")
    mx.random.seed(seed)
    if kind == "switch":
        base = QuantizedSwitchLinear(32, 16, 4, bias=False, group_size=16, bits=4, mode="nvfp4")
        gs = mx.array(6.5, dtype=mx.float32) if scalar_gs else mx.array([0.125, 7.5, 0.25, 4.5], dtype=mx.float32)
        if with_pre_quant:
            pre1d = mx.linspace(0.25, 4.0, 32, dtype=mx.float32)
            pre = mx.stack([pre1d * (0.5 + 0.5 * i) for i in range(4)]) if pre_2d else pre1d
            return ScaledQuantizedSwitchLinear(base, gs, scale_name, pre_quant_scale=pre)
        return ScaledQuantizedSwitchLinear(base, gs, scale_name)
    base = nn.QuantizedLinear(32, 16, bias=False, group_size=16, bits=4, mode="nvfp4")
    gs = mx.array(6.5, dtype=mx.float32)
    if with_pre_quant:
        pre = mx.linspace(0.25, 4.0, 32, dtype=mx.float32)
        return ScaledQuantizedLinear(base, gs, scale_name, pre_quant_scale=pre)
    return ScaledQuantizedLinear(base, gs, scale_name)


def _scaled_nvfp4_model(seed=7):
    has_pre = _nvfp4_supports_pre_quant()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = _nvfp4_scaled_module("dense", "weight_global_scale", with_pre_quant=has_pre, seed=seed)
            self.mlp = nn.Module()
            self.mlp.switch_mlp = nn.Module()
            self.mlp.switch_mlp.gate_proj = _nvfp4_scaled_module("switch", "weight_scale_2", seed=seed + 1)
            self.mlp.switch_mlp.up_proj = _nvfp4_scaled_module("switch", "weight_global_scale", with_pre_quant=has_pre, pre_2d=has_pre, seed=seed + 2)

    class _ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = [_Block()]

    return _ToyModel()


def test_scaled_nvfp4_linear_reports_unpacked_input_width():
    nvfp4 = pytest.importorskip("mlx_vlm.quantization.nvfp4")
    from unsloth_zoo.mlx.loader import _feeds_text_width, _semantic_dims

    packed = nn.QuantizedLinear.from_linear(nn.Linear(32, 32, bias=False), 16, 4, mode="nvfp4")
    head = nn.Module()
    head.proj = nvfp4.ScaledQuantizedLinear(packed, mx.array(0.5), "weight_scale_2")
    assert _semantic_dims(head.proj) == (32, 32)
    # A packed width read as the input width would make a text-width head look like a projector.
    assert not _feeds_text_width("aux_head", head, 32)


def test_scaled_nvfp4_default_targets_and_adapter_reload(tmp_path):
    nvfp4 = pytest.importorskip("mlx_vlm.quantization.nvfp4")
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear as VlmQuantizedSwitchLinear
    ScaledQuantizedLinear, ScaledQuantizedSwitchLinear = nvfp4.ScaledQuantizedLinear, nvfp4.ScaledQuantizedSwitchLinear
    from unsloth_zoo.mlx.loader import (
        _apply_lora_at_paths, _mlx_lora_spec_for_module, _mlx_lora_type_specs, _validate_mlx_adapter_base,
    )
    from unsloth_zoo.mlx.utils import (
        LoRAScaledQuantizedLinear, LoRAScaledQuantizedSwitchLinear,
        _extract_mlx_lora_parameters, iter_mlx_lora_modules, save_lora_adapters,
    )

    qq = nn.QQLinear(32, 16, group_size=16, bits=4, mode="nvfp4")
    qq.quantize()
    qq_scaled = ScaledQuantizedLinear(qq, mx.array(1.5, dtype=mx.float32), "weight_scale_2")
    assert not LoRAScaledQuantizedLinear.supports(qq_scaled)
    assert _mlx_lora_spec_for_module(qq_scaled, _mlx_lora_type_specs()) is None

    model = _scaled_nvfp4_model()
    adapted = FastMLXModel.get_peft_model(model, r=4, lora_alpha=8, use_gradient_checkpointing=False)
    wrapped = dict(iter_mlx_lora_modules(adapted))
    assert isinstance(wrapped["model.layers.0.self_attn.q_proj"], LoRAScaledQuantizedLinear)
    assert isinstance(wrapped["model.layers.0.mlp.switch_mlp.gate_proj"], LoRAScaledQuantizedSwitchLinear)
    assert isinstance(wrapped["model.layers.0.mlp.switch_mlp.up_proj"], LoRAScaledQuantizedSwitchLinear)
    assert _extract_mlx_lora_parameters(adapted)[:2] == (4, 2.0)

    wrapped["model.layers.0.self_attn.q_proj"].lora_b = mx.ones((4, 16)) * 0.25
    wrapped["model.layers.0.mlp.switch_mlp.gate_proj"].lora_b = mx.ones((4, 16, 4)) * 0.25
    wrapped["model.layers.0.mlp.switch_mlp.up_proj"].lora_b = mx.ones((4, 16, 4)) * -0.2
    save_lora_adapters(adapted, tmp_path)
    adapter_cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert adapter_cfg["base_resolved_quantization_map_supports_scaled_nvfp4"] is True
    assert adapter_cfg["base_resolved_quantization_map"]["model.layers.0.mlp.switch_mlp.gate_proj"]["mode"] == "nvfp4"
    saved_adapter = mx.load(str(tmp_path / "adapters.safetensors"))
    assert not any(k.endswith(("weight_scale_2", "weight_global_scale", "pre_quant_scale")) for k in saved_adapter)

    fresh = _scaled_nvfp4_model()
    _validate_mlx_adapter_base(fresh, adapter_cfg)
    _apply_lora_at_paths(fresh, adapter_cfg["unsloth_mlx_lora_module_paths"], adapter_cfg, str(tmp_path / "adapters.safetensors"))
    fresh.load_weights(str(tmp_path / "adapters.safetensors"), strict=False)
    x_d = mx.random.normal((2, 32)).astype(mx.bfloat16)
    x_sw = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
    idx = mx.array([[1, 3], [2, 1]])
    assert mx.allclose(fresh.model.layers[0].self_attn.q_proj(x_d), wrapped["model.layers.0.self_attn.q_proj"](x_d))
    assert mx.allclose(fresh.model.layers[0].mlp.switch_mlp.gate_proj(x_sw, idx), wrapped["model.layers.0.mlp.switch_mlp.gate_proj"](x_sw, idx))
    assert mx.allclose(fresh.model.layers[0].mlp.switch_mlp.up_proj(x_sw, idx), wrapped["model.layers.0.mlp.switch_mlp.up_proj"](x_sw, idx))
    assert isinstance(fresh.model.layers[0].self_attn.q_proj.linear, ScaledQuantizedLinear)
    assert isinstance(fresh.model.layers[0].mlp.switch_mlp.gate_proj.linear, ScaledQuantizedSwitchLinear)

    mixed = _scaled_nvfp4_model()
    mixed.model.layers[0].mlp.switch_mlp.down_proj = VlmQuantizedSwitchLinear(64, 16, 4, bias=False)
    legacy_cfg = {
        "base_resolved_quantization_map": {
            "model.layers.0.mlp.switch_mlp.down_proj": {"bits": 4, "group_size": 64, "mode": "affine"},
        },
        "base_resolved_quantization_map_supports_switch": True,
    }
    _validate_mlx_adapter_base(mixed, legacy_cfg)
    with pytest.raises(ValueError, match="unexpected quantized modules"):
        _validate_mlx_adapter_base(mixed, {**legacy_cfg, "base_resolved_quantization_map_supports_scaled_nvfp4": True})


@pytest.mark.parametrize("kind", ["switch", "dense"])
@pytest.mark.parametrize("scale_name", ["weight_scale_2", "weight_global_scale"])
@pytest.mark.parametrize("with_pre_quant", [False, True])
def test_scaled_nvfp4_lora_forward_grad_and_fuse(kind, scale_name, with_pre_quant):
    nvfp4 = pytest.importorskip("mlx_vlm.quantization.nvfp4")
    from mlx_vlm.models.switch_layers import SwitchLinear as VlmSwitchLinear
    ScaledQuantizedLinear, ScaledQuantizedSwitchLinear = nvfp4.ScaledQuantizedLinear, nvfp4.ScaledQuantizedSwitchLinear
    from unsloth_zoo.mlx.loader import _mlx_lora_from_base, _mlx_lora_type_specs

    base = _nvfp4_scaled_module(
        kind, scale_name, with_pre_quant=with_pre_quant,
        pre_2d=(kind == "switch" and with_pre_quant),
        scalar_gs=(kind == "switch" and scale_name == "weight_global_scale" and not with_pre_quant),
        seed=11,
    )
    orig_scale = mx.array(base.global_scale)
    scale = 4.0
    lora = _mlx_lora_from_base(base, {"rank": 4, "scale": scale, "dropout": 0.0, "use_dora": False}, specs=_mlx_lora_type_specs())
    mx.random.seed(99)
    lora.lora_b = mx.random.normal(lora.lora_b.shape) * 0.5
    mx.eval(lora.parameters())

    w_deq = mx.dequantize(base.weight, base.scales, group_size=16, bits=4, mode="nvfp4", dtype=mx.float32)
    gs_b = orig_scale.astype(mx.float32)
    if gs_b.size > 1:
        gs_b = gs_b.reshape(*w_deq.shape[:-2], -1, 1)
    w_eff = w_deq / gs_b if scale_name == "weight_global_scale" else w_deq * gs_b
    pre_b = getattr(base, "pre_quant_scale", None)
    if pre_b is not None:
        pre_b = pre_b.astype(mx.float32)
        if pre_b.ndim > 1:
            pre_b = pre_b.reshape(*pre_b.shape[:-1], 1, pre_b.shape[-1])
        w_eff = w_eff * pre_b

    if kind == "switch":
        x = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
        idx = mx.array([[1, 3], [2, 1]])
        base_y = base(x, idx)
        expected_delta = scale * mx.gather_mm(
            mx.gather_mm(x, lora.lora_a.swapaxes(-1, -2), rhs_indices=idx),
            lora.lora_b.swapaxes(-1, -2),
            rhs_indices=idx,
        )
        actual_y = lora(x, idx)
        assert mx.allclose(actual_y, base_y + expected_delta.astype(x.dtype), atol=1e-4, rtol=1e-4)

        def loss_fn(a, b, inp):
            lora.lora_a, lora.lora_b = a, b
            return lora(inp, idx).astype(mx.float32).sum()

        g_a, g_b, g_x = mx.grad(loss_fn, argnums=(0, 1, 2))(lora.lora_a, lora.lora_b, x)
        mx.eval(g_a, g_b, g_x)
        assert mx.array_equal(g_a[0], mx.zeros_like(g_a[0])) and mx.array_equal(g_b[0], mx.zeros_like(g_b[0]))
        assert all(float(mx.abs(g).max()) > 0 for g in (g_a[1], g_a[3], g_b[1], g_b[3], g_x))

        dense = lora.fuse(dequantize=True)
        assert type(dense) is VlmSwitchLinear
        assert mx.allclose(dense(x, idx), actual_y, atol=5e-2, rtol=2e-2)

        requant = lora.fuse(dequantize=False)
        assert type(requant) is ScaledQuantizedSwitchLinear
        assert requant._scale_name == scale_name and mx.array_equal(requant.global_scale, orig_scale)

        target_eff = w_eff + (scale * lora.lora_b) @ lora.lora_a
        target_pre = target_eff / pre_b if pre_b is not None else target_eff
        target_pre = target_pre * gs_b if scale_name == "weight_global_scale" else target_pre / gs_b
        ref_sw = VlmSwitchLinear(32, 16, 4, bias=False)
        ref_sw.weight = target_pre
        ref_mod = ScaledQuantizedSwitchLinear(
            ref_sw.to_quantized(group_size=16, bits=4, mode="nvfp4"), orig_scale, scale_name,
            *( (base.pre_quant_scale,) if with_pre_quant else () ),
        )
        ref_y = ref_mod(x, idx)
        req_y = requant(x, idx)
        mx.eval(ref_y, req_y, base_y)
        assert mx.allclose(req_y, ref_y, atol=1e-2, rtol=1e-2)
        assert float(mx.abs(req_y - base_y).max()) > 0.5
    else:
        x = mx.random.normal((3, 32)).astype(mx.bfloat16)
        base_y = base(x)
        expected_delta = scale * ((x @ lora.lora_a) @ lora.lora_b)
        actual_y = lora(x)
        assert mx.allclose(actual_y, base_y + expected_delta.astype(x.dtype), atol=1e-4, rtol=1e-4)

        def loss_fn(a, b, inp):
            lora.lora_a, lora.lora_b = a, b
            return lora(inp).astype(mx.float32).sum()

        g_a, g_b, g_x = mx.grad(loss_fn, argnums=(0, 1, 2))(lora.lora_a, lora.lora_b, x)
        mx.eval(g_a, g_b, g_x)
        assert all(float(mx.abs(g).max()) > 0 for g in (g_a, g_b, g_x))

        dense = lora.fuse(dequantize=True)
        assert type(dense) is nn.Linear
        assert mx.allclose(dense(x), actual_y, atol=5e-2, rtol=2e-2)

        requant = lora.fuse(dequantize=False)
        assert type(requant) is ScaledQuantizedLinear
        assert requant._scale_name == scale_name and mx.array_equal(requant.global_scale, orig_scale)

        target_eff = w_eff + (scale * lora.lora_b.T) @ lora.lora_a.T
        target_pre = target_eff / pre_b if pre_b is not None else target_eff
        target_pre = target_pre * gs_b if scale_name == "weight_global_scale" else target_pre / gs_b
        ref_lin = nn.Linear(32, 16, bias=False)
        ref_lin.weight = target_pre
        ref_mod = ScaledQuantizedLinear(
            nn.QuantizedLinear.from_linear(ref_lin, 16, 4, mode="nvfp4"), orig_scale, scale_name,
            *( (base.pre_quant_scale,) if with_pre_quant else () ),
        )
        ref_y = ref_mod(x)
        req_y = requant(x)
        mx.eval(ref_y, req_y, base_y)
        assert mx.allclose(req_y, ref_y, atol=1e-2, rtol=1e-2)
        assert float(mx.abs(req_y - base_y).max()) > 0.5


@pytest.mark.parametrize("scale_name", ["weight_scale_2", "weight_global_scale"])
def test_scaled_nvfp4_merged_4bit_widens_a_saturated_global_scale(scale_name):
    nvfp4 = pytest.importorskip("mlx_vlm.quantization.nvfp4")
    from unsloth_zoo.mlx.loader import _mlx_lora_from_base, _mlx_lora_type_specs

    mx.random.seed(5)
    w_eff = mx.random.normal((16, 32))
    amax = mx.abs(w_eff).max()
    # ModelOpt sets weight_scale_2 = amax / (6 * 448): the largest group already sits at the NVFP4 limit.
    gs = amax / 2688.0 if scale_name == "weight_scale_2" else 2688.0 / amax
    dense = nn.Linear(32, 16, bias=False)
    dense.weight = w_eff / gs if scale_name == "weight_scale_2" else w_eff * gs
    base = nvfp4.ScaledQuantizedLinear(nn.QuantizedLinear.from_linear(dense, 16, 4, mode="nvfp4"), gs, scale_name)
    lora = _mlx_lora_from_base(base, {"rank": 4, "scale": 1.0, "dropout": 0.0, "use_dora": False}, specs=_mlx_lora_type_specs())
    lora.lora_b = mx.random.normal(lora.lora_b.shape) * 4.0
    x = mx.random.normal((3, 32))
    expected = lora.fuse(dequantize=True)(x)
    merged = lora.fuse(dequantize=False)
    mx.eval(expected, merged.parameters())
    assert not mx.array_equal(merged.global_scale, gs)
    rel = mx.abs(merged(x) - expected).max() / mx.abs(expected).max()
    assert rel.item() < 0.1


def test_scaled_nvfp4_save_merged_model_roundtrip(tmp_path):
    nvfp4 = pytest.importorskip("mlx_vlm.quantization.nvfp4")
    from mlx_vlm.models.switch_layers import SwitchLinear as VlmSwitchLinear
    ScaledQuantizedLinear, ScaledQuantizedSwitchLinear = nvfp4.ScaledQuantizedLinear, nvfp4.ScaledQuantizedSwitchLinear
    from mlx_vlm.utils import load_model_weights
    from unsloth_zoo.mlx.utils import save_merged_model

    def _skeleton():
        s = nn.Module()
        s.model = nn.Module()
        b = nn.Module()
        b.dense = nn.Linear(32, 16, bias=False)
        b.switch = VlmSwitchLinear(32, 16, 4, bias=False)
        s.model.layers = [b]
        return s

    model = nn.Module()
    model.model = nn.Module()
    blk = nn.Module()
    has_pre = _nvfp4_supports_pre_quant()
    blk.dense = _nvfp4_scaled_module("dense", "weight_global_scale", with_pre_quant=has_pre, seed=21)
    blk.switch = _nvfp4_scaled_module("switch", "weight_scale_2", seed=22)
    model.model.layers = [blk]
    model._config = {"model_type": "test_nvfp4", "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}}
    FastMLXModel.get_peft_model(model, r=4, lora_alpha=16, target_modules=["dense", "switch"], use_gradient_checkpointing=False)
    mx.random.seed(33)
    blk.dense.lora_b = mx.random.normal((4, 16)) * 0.5
    blk.switch.lora_b = mx.random.normal((4, 16, 4)) * 0.5
    x_d = mx.random.normal((2, 32)).astype(mx.bfloat16)
    x_s = mx.random.normal((2, 1, 32)).astype(mx.bfloat16)
    idx = mx.array([[1, 3], [2, 1]])
    base_d, base_s = blk.dense.linear(x_d), blk.switch.linear(x_s, idx)
    ref_d = blk.dense.fuse(dequantize=False)(x_d)
    ref_s = blk.switch.fuse(dequantize=False)(x_s, idx)
    mx.eval(base_d, base_s, ref_d, ref_s)

    tok = type("Tokenizer", (), {"save_pretrained": lambda self, path: None})()
    save_merged_model(model, tok, tmp_path / "merged_4bit", dequantize=False)
    cfg = json.loads((tmp_path / "merged_4bit" / "config.json").read_text())
    assert cfg["quantization"] == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    saved = mx.load(str(tmp_path / "merged_4bit" / "model.safetensors"))
    expected_keys = {
        "model.layers.0.dense.weight_global_scale",
        "model.layers.0.switch.weight_scale_2",
    }
    if has_pre:
        expected_keys.add("model.layers.0.dense.pre_quant_scale")
    assert expected_keys <= set(saved)

    reloaded = load_model_weights(_skeleton(), cfg, saved, tmp_path / "merged_4bit", strict=True)
    reloaded._config = cfg
    rblk = reloaded.model.layers[0]
    assert type(rblk.dense) is ScaledQuantizedLinear
    assert type(rblk.switch) is ScaledQuantizedSwitchLinear
    q_d, q_s = rblk.dense(x_d), rblk.switch(x_s, idx)
    mx.eval(q_d, q_s)
    assert mx.allclose(q_d, ref_d, atol=1e-2, rtol=1e-2)
    assert mx.allclose(q_s, ref_s, atol=1e-2, rtol=1e-2)
    assert float(mx.abs(q_d - base_d).max()) > 1.0
    assert float(mx.abs(q_s - base_s).max()) > 1.0

    save_merged_model(reloaded, tok, tmp_path / "merged_16bit", dequantize=True)
    dense_cfg = json.loads((tmp_path / "merged_16bit" / "config.json").read_text())
    assert "quantization" not in dense_cfg
    assert type(rblk.dense) is nn.Linear and type(rblk.switch) is VlmSwitchLinear
    assert mx.allclose(rblk.dense(x_d), q_d, atol=5e-2, rtol=2e-2)
    assert mx.allclose(rblk.switch(x_s, idx), q_s, atol=5e-2, rtol=2e-2)
