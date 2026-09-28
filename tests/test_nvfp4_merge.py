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

"""16bit merge of a compressed-tensors NVFP4 (and mixed NVFP4 + FP8) base: weight_packed is dequantized, LoRA folded in."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("float8_e4m3fn unavailable", allow_module_level=True)

from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class _FakeLoraStats:
    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None
    alpha: float = 1.0
    module: object = None


def _nvfp4(out_f, in_f, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (out_f, in_f // 2), generator=g, dtype=torch.uint8)
    scale = (torch.rand(out_f, in_f // 16, generator=g) * 3 + 0.25) * torch.linspace(0.5, 2.0, in_f // 16)
    return packed, scale.to(torch.float8_e4m3fn), torch.tensor([37.5])


def _reference(packed, scale, global_scale):
    ct = pytest.importorskip("compressed_tensors.compressors.nvfp4.base")
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    # compressed-tensors >= 0.19 reads the weight args from the scheme instead of inferring them.
    args = QuantizationArgs(num_bits=4, type="float", strategy="tensor_group", group_size=16)
    return ct.NVFP4PackedCompressor.decompress(
        {"weight_packed": packed, "weight_scale": scale, "weight_global_scale": global_scale},
        QuantizationScheme(targets=["Linear"], weights=args),
    )["weight"].float()


def _fp8_channel(W):
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    return (W / scale).to(torch.float8_e4m3fn), scale.to(torch.bfloat16)


def _merge(tmp_path, shards, lora, order):
    from unsloth_zoo.saving_utils import (
        _collect_fp8_weight_keys, _drop_resolved_fp8_scales_after_rewrite, _merge_and_overwrite_lora,
    )
    names = [p.name for p in shards]
    prerewrite = _collect_fp8_weight_keys(str(tmp_path), names)
    total, seen = 0, set()
    for name in order:
        count, keys = _merge_and_overwrite_lora(
            save_directory=str(tmp_path), filename=name, lora_weights=lora, output_dtype=torch.bfloat16,
            model_class_name="Qwen3ForCausalLM", base_model_is_quantized=True, quant_type="fp8",
        )
        total += count
        seen.update(keys)
    _merge.seen = seen
    _drop_resolved_fp8_scales_after_rewrite(str(tmp_path), names, prerewrite)
    out = {}
    for p in shards:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            out.update({k: f.get_tensor(k) for k in f.keys()})
    return total, out


def test_dequantize_matches_compressed_tensors():
    from unsloth_zoo.saving_utils import _nvfp4_dequantize
    packed, scale, gs = _nvfp4(48, 96, 0)
    # compressed-tensors returns bf16 of the same fp32 arithmetic.
    assert torch.equal(_nvfp4_dequantize(packed, scale, gs).to(torch.bfloat16), _reference(packed, scale, gs).to(torch.bfloat16))


def test_mixed_nvfp4_and_fp8_merge_with_lora(tmp_path):
    torch.manual_seed(1)
    packed, scale, gs = _nvfp4(64, 96, 1)
    from unsloth_zoo.saving_utils import _nvfp4_dequantize
    W_mlp = _nvfp4_dequantize(packed, scale, gs)
    assert torch.equal(W_mlp.to(torch.bfloat16), _reference(packed, scale, gs).to(torch.bfloat16))
    W_attn = torch.randn(32, 96) * 0.1
    q_attn, s_attn = _fp8_channel(W_attn)
    A1, B1 = torch.randn(4, 96) * 0.02, torch.randn(64, 4) * 0.02
    A2, B2 = torch.randn(4, 96) * 0.02, torch.randn(32, 4) * 0.02
    shard = tmp_path / "model.safetensors"
    save_file({
        "model.layers.0.mlp.gate_proj.weight_packed": packed,
        "model.layers.0.mlp.gate_proj.weight_scale": scale,
        "model.layers.0.mlp.gate_proj.weight_global_scale": gs,
        "model.layers.0.mlp.gate_proj.input_global_scale": torch.tensor([3.0]),
        "model.layers.0.self_attn.q_proj.weight": q_attn,
        "model.layers.0.self_attn.q_proj.weight_scale": s_attn,
        "model.norm.weight": torch.ones(96, dtype=torch.bfloat16),
    }, str(shard), metadata={"format": "pt"})
    lora = defaultdict(_FakeLoraStats)
    lora["model.layers.0.mlp.gate_proj"] = _FakeLoraStats(lora_A=A1, lora_B=B1, alpha=2.0)
    lora["model.layers.0.self_attn.q_proj"] = _FakeLoraStats(lora_A=A2, lora_B=B2, alpha=1.0)
    count, out = _merge(tmp_path, [shard], lora, [shard.name])
    assert count == 2
    # The Step-7 sanity check counts LoRAs backed by the merged keys; a NVFP4 layer must count once.
    from unsloth_zoo.saving_utils import _count_backed_lora_modules
    assert _count_backed_lora_modules(lora, _merge.seen, "Qwen3ForCausalLM", False) == 2
    assert set(out) == {"model.layers.0.mlp.gate_proj.weight", "model.layers.0.self_attn.q_proj.weight", "model.norm.weight"}
    mlp = out["model.layers.0.mlp.gate_proj.weight"]
    assert mlp.dtype == torch.bfloat16 and mlp.shape == (64, 96)
    expected = (W_mlp + 2.0 * B1 @ A1).to(torch.bfloat16)
    assert torch.equal(mlp, expected)
    assert torch.allclose(out["model.layers.0.self_attn.q_proj.weight"].float(), W_attn + B2 @ A2, atol=0.05)


@pytest.mark.parametrize("scales_first", [False, True])
def test_nvfp4_scales_in_another_shard(tmp_path, scales_first):
    packed, scale, gs = _nvfp4(32, 64, 2)
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    weights = {"model.layers.0.mlp.up_proj.weight_packed": packed, "model.norm.weight": torch.ones(64, dtype=torch.bfloat16)}
    scales = {"model.layers.0.mlp.up_proj.weight_scale": scale, "model.layers.0.mlp.up_proj.weight_global_scale": gs}
    save_file(scales if scales_first else weights, str(s1), metadata={"format": "pt"})
    save_file(weights if scales_first else scales, str(s2), metadata={"format": "pt"})
    _, out = _merge(tmp_path, [s1, s2], defaultdict(_FakeLoraStats), [s1.name, s2.name])
    assert set(out) == {"model.layers.0.mlp.up_proj.weight", "model.norm.weight"}
    assert torch.equal(out["model.layers.0.mlp.up_proj.weight"], _reference(packed, scale, gs).to(torch.bfloat16))


@pytest.mark.parametrize("fmt", ["nvfp4-pack-quantized", "mixed-precision"])
def test_nvfp4_checkpoints_are_detected_for_dequant(tmp_path, fmt):
    from unsloth_zoo.saving_utils import check_model_quantization_status
    groups = {"group_1": {"format": "nvfp4-pack-quantized", "targets": ["Linear"],
                          "weights": {"num_bits": 4, "type": "float", "group_size": 16}}}
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "compressed-tensors", "format": fmt, "config_groups": groups}}))
    assert check_model_quantization_status(str(tmp_path)) == (True, "fp8")


@pytest.mark.parametrize("global_shape", [(1,), (3,), (3, 1)])
def test_fused_expert_nvfp4_merges_per_expert(tmp_path, global_shape):
    # A 3-D fused-expert weight_packed (E, rows, cols / 2) must merge like three separate 2-D weights,
    # with or without per-expert global scales, even when no LoRA targets the experts.
    from unsloth_zoo.saving_utils import _nvfp4_dequantize

    experts = [_nvfp4(32, 64, seed) for seed in range(3)]
    packed = torch.stack([p for p, _, _ in experts])
    scale = torch.stack([s for _, s, _ in experts])
    gs = torch.tensor([37.5, 12.0, 3.25][: global_shape[0]]).reshape(global_shape)
    per_expert = [gs.reshape(-1)[i if gs.numel() > 1 else 0].reshape(1) for i in range(3)]
    expected = torch.stack([_nvfp4_dequantize(p, s, g) for (p, s, _), g in zip(experts, per_expert)])
    assert torch.equal(_nvfp4_dequantize(packed, scale, gs), expected)

    shard = tmp_path / "model.safetensors"
    save_file({
        "model.layers.0.mlp.experts.gate_up_proj.weight_packed": packed,
        "model.layers.0.mlp.experts.gate_up_proj.weight_scale": scale,
        "model.layers.0.mlp.experts.gate_up_proj.weight_global_scale": gs,
        "model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
    }, str(shard), metadata={"format": "pt"})
    count, out = _merge(tmp_path, [shard], defaultdict(_FakeLoraStats), [shard.name])
    assert set(out) == {"model.layers.0.mlp.experts.gate_up_proj.weight", "model.norm.weight"}
    assert torch.equal(out["model.layers.0.mlp.experts.gate_up_proj.weight"], expected.to(torch.bfloat16))


def test_nvfp4_global_scale_count_mismatch_is_refused():
    from unsloth_zoo.saving_utils import _nvfp4_dequantize

    packed, scale, _ = _nvfp4(32, 64, 0)
    with pytest.raises(RuntimeError, match="global scales"):
        _nvfp4_dequantize(packed[None].expand(3, -1, -1), scale[None].expand(3, -1, -1), torch.ones(2))


def test_non_nvfp4_packed_weights_are_refused():
    from unsloth_zoo.saving_utils import _nvfp4_dequantize

    packed, scale, gs = _nvfp4(32, 64, 0)
    # int4 pack-quantized: int32 packs and a bf16 group scale under the same weight_packed name.
    with pytest.raises(RuntimeError, match="is not NVFP4"):
        _nvfp4_dequantize(torch.zeros(32, 8, dtype=torch.int32), torch.ones(32, 1, dtype=torch.bfloat16), gs)
    # A (N, 1) scale against a (1, N) packed row would broadcast to (1, N, 2N) instead of failing.
    with pytest.raises(RuntimeError, match="is not NVFP4"):
        _nvfp4_dequantize(packed[:1], scale.reshape(-1, 1), gs)


def test_missing_nvfp4_global_scale_is_refused():
    from unsloth_zoo.saving_utils import _nvfp4_dequantize

    packed, scale, _ = _nvfp4(32, 64, 0)
    with pytest.raises(RuntimeError, match="no weight_global_scale"):
        _nvfp4_dequantize(packed, scale, None)
