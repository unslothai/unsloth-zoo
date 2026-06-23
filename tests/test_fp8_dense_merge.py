# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""FP8 dense merge-to-16bit (unslothai/unsloth#4919).

Pins the rewrite path: FP8 dequantizes to 16bit, scale keys drop, and a missing
scale raises instead of corrupting silently (previously written straight back).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import importlib.machinery
import sys
import types

import pytest
import torch

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("float8_e4m3fn unavailable", allow_module_level=True)

# unsloth_zoo / transformers import bitsandbytes at import time (a 4bit-merge path these
# FP8 tests never reach). Stub it only when absent so the suite runs in a CPU-only core
# env; a real install is left untouched. The stub carries a valid __spec__ so
# transformers' importlib.util.find_spec("bitsandbytes") probe returns a spec instead of
# raising on a spec-less module (and then reports unavailable via missing dist metadata).
try:
    import bitsandbytes  # noqa: F401
except Exception:
    def _stub_module(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return mod
    _bnb = _stub_module("bitsandbytes")
    _bnb.__version__ = "0.0.0"
    _bnb_nn = _stub_module("bitsandbytes.nn")
    class _Linear4bit:  # placeholder; isinstance checks just return False here
        pass
    _bnb_nn.Linear4bit = _Linear4bit
    _bnb.nn = _bnb_nn
    sys.modules.setdefault("bitsandbytes", _bnb)
    sys.modules.setdefault("bitsandbytes.nn", _bnb_nn)

from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class _FakeLoraStats:
    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None
    alpha: float = 1.0
    module: object = None


def _fp8_quant_channel(W):
    """Per-channel (out, 1) FP8 quant like compressed-tensors fp8-dynamic."""
    max_fp8 = 448.0
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / max_fp8
    q = (W / scale).clamp(-max_fp8, max_fp8).to(torch.float8_e4m3fn)
    return q, scale.to(torch.bfloat16)


def _fp8_quant_row_1d(W):
    """Per-row 1-D scale (fbgemm_fp8 / static channel quant)."""
    max_fp8 = 448.0
    amax = W.abs().amax(dim=1).clamp_min(1e-12)  # 1-D, len == out_features
    scale = amax / max_fp8
    q = (W / scale.unsqueeze(1)).clamp(-max_fp8, max_fp8).to(torch.float8_e4m3fn)
    return q, scale.to(torch.bfloat16)


def _fp8_quant_block(W, bs=128):
    """DeepSeek-style (ceil(rows/bs), ceil(cols/bs)) block scale (weight_scale_inv)."""
    max_fp8 = 448.0
    rows, cols = W.shape
    srows, scols = -(-rows // bs), -(-cols // bs)
    Wpad = torch.zeros(srows * bs, scols * bs, dtype=torch.float32)
    Wpad[:rows, :cols] = W
    blocks = Wpad.reshape(srows, bs, scols, bs)
    scale = (blocks.abs().amax(dim=(1, 3)) / max_fp8).clamp_min(1e-12)
    q = (blocks / scale.unsqueeze(-1).unsqueeze(1)).clamp(-max_fp8, max_fp8)
    q = q.reshape(srows * bs, scols * bs)[:rows, :cols].to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def _write_shard(path, weights):
    save_file(weights, str(path), metadata={"format": "pt"})


def test_fp8_dense_dequantizes_and_drops_scales(tmp_path):
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(0)
    W_real = torch.randn(64, 128, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_quant_channel(W_real)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": W_fp8,
        "model.layers.0.mlp.down_proj.weight_scale": scale,
        "model.layers.0.mlp.down_proj.input_scale": torch.ones(1, dtype=torch.float32),
        "lm_head.weight": torch.randn(8, 8, dtype=torch.bfloat16),  # not FP8, passthrough
    })

    count, seen = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats),
        output_dtype=torch.bfloat16,
        model_class_name="Qwen2ForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # No FP8 left, no dangling scale companions.
        assert all(f.get_tensor(k).dtype != torch.float8_e4m3fn for k in keys)
        assert not any(k.endswith(("weight_scale", "weight_scale_inv", "input_scale")) for k in keys)
        # Dequantized weight matches the real value within FP8 round-trip tolerance.
        merged = f.get_tensor("model.layers.0.mlp.down_proj.weight").float()
        assert torch.allclose(merged, W_real, atol=0.05)
        # bf16 passthrough tensor preserved.
        assert "lm_head.weight" in keys


def test_fp8_dense_merges_lora_on_dequantized_weight(tmp_path):
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(1)
    out_f, in_f, r = 32, 48, 4
    W_real = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_quant_channel(W_real)
    A = torch.randn(r, in_f, dtype=torch.float32) * 0.02
    B = torch.randn(out_f, r, dtype=torch.float32) * 0.02
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.self_attn.q_proj.weight": W_fp8,
        "model.layers.0.self_attn.q_proj.weight_scale": scale,
    })

    lora = defaultdict(_FakeLoraStats)
    lora["model.layers.0.self_attn.q_proj"] = _FakeLoraStats(lora_A=A, lora_B=B, alpha=1.0)
    count, seen = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=lora,
        output_dtype=torch.bfloat16,
        model_class_name="Qwen2ForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
    )

    assert count == 1
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        merged = f.get_tensor("model.layers.0.self_attn.q_proj.weight").float()
    expected = (W_real + B @ A).to(torch.bfloat16).float()
    assert torch.allclose(merged, expected, atol=0.05)


def test_fp8_dense_missing_scale_raises(tmp_path):
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    W_fp8, _ = _fp8_quant_channel(torch.randn(16, 16) * 0.1)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {"model.layers.0.mlp.up_proj.weight": W_fp8})  # no companion scale

    with pytest.raises(RuntimeError, match="weight_scale"):
        _merge_and_overwrite_lora(
            save_directory=str(tmp_path),
            filename="model.safetensors",
            lora_weights=defaultdict(_FakeLoraStats),
            output_dtype=torch.bfloat16,
            model_class_name="Qwen2ForCausalLM",
            base_model_is_quantized=True,
            quant_type="fp8",
        )


def test_fp8_quant_config_detection():
    from unsloth_zoo.saving_utils import _is_fp8_quant_config

    assert _is_fp8_quant_config({"quant_method": "compressed-tensors", "format": "float-quantized"})
    assert _is_fp8_quant_config({"quant_method": "fp8"})
    assert _is_fp8_quant_config({"quant_method": "fbgemm_fp8"})
    assert _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8}}},
    })
    # Non-FP8 schemes must NOT be reclassified.
    assert not _is_fp8_quant_config({"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"})
    assert not _is_fp8_quant_config({"quant_method": "mxfp4"})
    # Microscaling FP8 (mxfp8) carries its own block scales, not dense FP8.
    assert not _is_fp8_quant_config({"quant_method": "mxfp8"})
    assert not _is_fp8_quant_config({"quant_method": "gptq", "bits": 4})
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "int", "num_bits": 8}}},
    })
    # 4-bit float formats (e.g. NVFP4) also use type "float"; not dense FP8.
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4}}},
    })
    # compressed-tensors MXFP8: 8-bit float but microscaling; the MX marker excludes it.
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "format": "mxfp8-quantized",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8}}},
    })
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "format": "mxfp8"}}},
    })
    # Malformed num_bits (null / non-int) must not crash; treated as not-FP8.
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": None}}},
    })
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": "eight"}}},
    })


def test_fp8_dense_per_channel_2d_scale_ignores_block_size(tmp_path):
    """A 2-D per-channel (out, 1) scale dequantizes correctly even with a config
    weight_block_size that does not tile the weight (it must not be applied)."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(7)
    W_real = torch.randn(64, 128, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_quant_channel(W_real)  # scale is 2-D (64, 1)
    assert scale.ndim == 2 and scale.shape == (64, 1)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": W_fp8,
        "model.layers.0.mlp.down_proj.weight_scale": scale,
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats),
        output_dtype=torch.float32,
        model_class_name="Qwen2ForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
        weight_block_size=(128, 128),  # would corrupt rows if blindly applied
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        merged = f.get_tensor("model.layers.0.mlp.down_proj.weight").float()
    assert torch.allclose(merged, W_real, atol=0.05)


def test_fp8_dense_scale_grid_mismatch_with_block_size_raises(tmp_path):
    """A 2-D scale grid that neither exact-divides the weight nor ceil-matches the configured
    block size must raise, not silently dequantize with inferred tiles."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    # 130x256 weight, block (128,128) -> matching grid would be (2,2). A (3,2) grid matches
    # neither exact division (130 % 3 != 0) nor the block-size ceil grid.
    W_fp8 = (torch.randn(130, 256) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn)
    bad_scale = torch.rand(3, 2, dtype=torch.float32) + 0.5
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": W_fp8,
        "model.layers.0.mlp.down_proj.weight_scale_inv": bad_scale,
    })

    with pytest.raises(RuntimeError, match="not tiled"):
        _merge_and_overwrite_lora(
            save_directory=str(tmp_path),
            filename="model.safetensors",
            lora_weights=defaultdict(_FakeLoraStats),
            output_dtype=torch.float32,
            model_class_name="Qwen2ForCausalLM",
            base_model_is_quantized=True,
            quant_type="fp8",
            weight_block_size=(128, 128),
        )


def _run_cross_shard_merge(tmp_path, weight_in, scale_in, process_order):
    """Helper: write an FP8 weight and its scale into separate shards, run the per-shard
    rewrite in the given order, then the orchestrator's order-independent scale cleanup."""
    from unsloth_zoo.saving_utils import (
        _merge_and_overwrite_lora, _drop_resolved_fp8_scales_after_rewrite,
    )
    torch.manual_seed(5)
    W_real = torch.randn(64, 128, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_quant_channel(W_real)  # 2-D (64, 1) scale
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    contents = {
        s1: {"model.layers.1.mlp.up_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16)},
        s2: {},
    }
    contents[weight_in]["model.layers.0.mlp.down_proj.weight"] = W_fp8
    contents[scale_in]["model.layers.0.mlp.down_proj.weight_scale"] = scale
    for path, tensors in ((s1, contents[s1]), (s2, contents[s2])):
        _write_shard(path, tensors)

    for fn in process_order:
        _merge_and_overwrite_lora(
            save_directory=str(tmp_path), filename=fn,
            lora_weights=defaultdict(_FakeLoraStats), output_dtype=torch.bfloat16,
            model_class_name="Qwen2ForCausalLM", base_model_is_quantized=True, quant_type="fp8",
        )
    _drop_resolved_fp8_scales_after_rewrite(str(tmp_path), [s1.name, s2.name])

    keys = set()
    merged = None
    for path in (s1, s2):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys |= set(f.keys())
            if "model.layers.0.mlp.down_proj.weight" in f.keys():
                merged = f.get_tensor("model.layers.0.mlp.down_proj.weight")
    # Weight dequantized via cross-shard scale; orphaned scale dropped; other tensor kept.
    assert merged is not None and merged.dtype == torch.bfloat16
    assert torch.allclose(merged.float(), W_real, atol=0.05)
    assert "model.layers.0.mlp.down_proj.weight_scale" not in keys
    assert "model.layers.1.mlp.up_proj.weight" in keys


def test_fp8_dense_scale_in_sibling_shard_weight_first(tmp_path):
    """Weight in shard 1, scale in shard 2, processed weight-first."""
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    _run_cross_shard_merge(tmp_path, weight_in=s1, scale_in=s2, process_order=(s1.name, s2.name))


def test_fp8_dense_scale_in_sibling_shard_scale_first(tmp_path):
    """Scale in shard 1, weight in shard 2, processed scale-first: the order-independent
    cleanup must still drop the orphaned scale (per-shard processing alone cannot)."""
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    _run_cross_shard_merge(tmp_path, weight_in=s2, scale_in=s1, process_order=(s1.name, s2.name))


def test_fp8_fused_expert_3d_dequantizes(tmp_path):
    """A 3-D fused MoE expert FP8 tensor dequantizes per-expert, not hard-fails on rank 3."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(8)
    n_experts, out_f, in_f = 3, 32, 48
    W_real = torch.randn(n_experts, out_f, in_f, dtype=torch.float32) * 0.1
    q = [_fp8_quant_channel(W_real[e]) for e in range(n_experts)]
    W_fp8 = torch.stack([w for w, _ in q], dim=0)
    scale = torch.stack([s for _, s in q], dim=0)  # (E, out, 1)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.experts.gate_up_proj": W_fp8,
        "model.layers.0.mlp.experts.gate_up_proj.weight_scale": scale,
        "model.layers.0.self_attn.q_proj.weight": _fp8_quant_channel(torch.randn(16, 16) * 0.1)[0],
        "model.layers.0.self_attn.q_proj.weight_scale": _fp8_quant_channel(torch.randn(16, 16) * 0.1)[1],
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats),  # attention-only / no expert LoRA
        output_dtype=torch.float32,
        model_class_name="Qwen3MoeForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert all(f.get_tensor(k).dtype != torch.float8_e4m3fn for k in keys)
        assert not any(k.endswith(("weight_scale", "weight_scale_inv")) for k in keys)
        merged = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj").float()
    assert merged.shape == (n_experts, out_f, in_f)
    assert torch.allclose(merged, W_real, atol=0.05)


def test_fp8_dense_one_d_row_scale_dequantizes(tmp_path):
    """A 1-D per-row weight_scale (fbgemm / static FP8) must dequantize, not raise."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(2)
    W_real = torch.randn(64, 128, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_quant_row_1d(W_real)
    assert scale.ndim == 1
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": W_fp8,
        "model.layers.0.mlp.down_proj.weight_scale": scale,
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats),
        output_dtype=torch.bfloat16,
        model_class_name="Qwen2ForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert all(f.get_tensor(k).dtype != torch.float8_e4m3fn for k in keys)
        assert not any(k.endswith(("weight_scale", "weight_scale_inv")) for k in keys)
        merged = f.get_tensor("model.layers.0.mlp.down_proj.weight").float()
        assert torch.allclose(merged, W_real, atol=0.05)


def test_fp8_dense_block_scale_partial_block(tmp_path):
    """Block-quant with a dim not a block multiple: weight_block_size dequantizes
    correctly (inferring rows//srows is wrong)."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(3)
    W_real = torch.randn(130, 256, dtype=torch.float32) * 0.1  # 130 not a multiple of 128
    W_fp8, scale_inv = _fp8_quant_block(W_real, bs=128)
    assert scale_inv.shape == (2, 2)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": W_fp8,
        "model.layers.0.mlp.down_proj.weight_scale_inv": scale_inv,
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats),
        output_dtype=torch.float32,
        model_class_name="Qwen2ForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
        weight_block_size=(128, 128),
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        merged = f.get_tensor("model.layers.0.mlp.down_proj.weight").float()
    assert torch.allclose(merged, W_real, atol=0.05)


def test_fp8_dense_path_refuses_moe_expert_lora(tmp_path):
    """The dense FP8 rewrite cannot fuse per-expert adapters, so it refuses MoE-expert LoRA
    (the dispatcher routes those to the in-place quant-aware path instead; see
    test_moe_merge_legacy_mixtral_cpu)."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora_fp8

    W_fp8, scale = _fp8_quant_channel(torch.randn(32, 48) * 0.1)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.experts.0.gate_proj.weight": W_fp8,
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale": scale,
    })
    lora = defaultdict(_FakeLoraStats)
    lora["model.layers.0.mlp.experts"] = _FakeLoraStats(
        lora_A=torch.randn(4, 48) * 0.02, lora_B=torch.randn(32, 4) * 0.02, alpha=1.0,
    )
    with pytest.raises(RuntimeError, match="MoE experts"):
        _merge_and_overwrite_lora_fp8(
            save_directory=str(tmp_path),
            filename="model.safetensors",
            lora_weights=lora,
            output_dtype=torch.bfloat16,
            model_class_name="Qwen3MoeForCausalLM",
        )


def test_fp8_fused_moe_gate_up_lora_refused(tmp_path):
    """Merging LoRA onto a fused 3-D FP8 gate_up_proj expert tensor must refuse rather than
    merge raw FP8 values (which silently loses the delta)."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    n_experts, inter, hidden, r = 2, 8, 12, 4
    gate_up = (torch.randn(n_experts, 2 * inter, hidden) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.gate_up_proj_scale_inv": torch.ones(n_experts, 1, 1, dtype=torch.float32),
    })
    lora = defaultdict(_FakeLoraStats)
    lora["model.layers.0.mlp.experts.base_layer"] = _FakeLoraStats(
        lora_A=torch.randn(n_experts * r, hidden) * 0.02,
        lora_B=torch.randn(2 * inter, n_experts * r) * 0.02, alpha=1.0,
    )
    with pytest.raises(RuntimeError, match="FP8 fused MoE"):
        _merge_and_overwrite_lora(
            save_directory=str(tmp_path), filename="model.safetensors",
            lora_weights=lora, output_dtype=torch.bfloat16,
            model_class_name="GptOssForCausalLM",
            base_model_is_quantized=True, quant_type="fp8",
        )


def test_fp8_moe_expert_lora_routes_to_inplace_not_dense(tmp_path):
    """The dispatcher must NOT send FP8 + MoE-expert LoRA to the dense rewrite (which would
    raise); it falls through to the in-place quant-aware path which keeps FP8."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_requant_blockwise, _fp8_dequant_blockwise

    out_f, in_f, r = 32, 48, 4
    W_hp = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_requant_blockwise(W_hp, (out_f, in_f), torch.float32)  # per-tensor-ish block
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.experts.0.gate_proj.weight": W_fp8,
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": scale,
    })
    lora = defaultdict(_FakeLoraStats)
    lora["model.layers.0.mlp.experts.base_layer"] = _FakeLoraStats(
        lora_A=torch.randn(r, in_f) * 0.02, lora_B=torch.randn(2 * out_f, r) * 0.02, alpha=1.0,
    )
    # Must not raise the dense "MoE experts" error; routed to the in-place quant-aware path.
    _merge_and_overwrite_lora(
        save_directory=str(tmp_path), filename="model.safetensors",
        lora_weights=lora, output_dtype=torch.bfloat16,
        model_class_name="Qwen3MoeForCausalLM",
        base_model_is_quantized=True, quant_type="fp8",
    )
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        # FP8 is preserved (in-place quant-aware merge), not dequantized to 16bit.
        assert f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight").dtype == torch.float8_e4m3fn


def test_fp8_e5m2_dequantizes(tmp_path):
    """float8_e5m2 weights must be dequantized (scale applied), not passed through raw."""
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("float8_e5m2 unavailable")
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(11)
    W_real = torch.randn(32, 64, dtype=torch.float32) * 0.1
    amax = W_real.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 57344.0).to(torch.float32)  # e5m2 max ~57344
    q = (W_real / scale).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    ref = q.to(torch.float32) * scale  # the correct dequantization
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": q,
        "model.layers.0.mlp.down_proj.weight_scale": scale,
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path), filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats), output_dtype=torch.float32,
        model_class_name="Qwen2ForCausalLM", base_model_is_quantized=True, quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        merged = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        assert merged.dtype not in (torch.float8_e5m2, torch.float8_e4m3fn)
        merged = merged.float()
    # Matches the scaled dequantization; raw pass-through (no scale) would not.
    assert torch.allclose(merged, ref, atol=1e-4)


def test_fp8_fused_expert_underscore_scale(tmp_path):
    """Fused expert FP8 with <key>_scale_inv naming (no .weight suffix) is found and
    dequantized, not a missing-scale error."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(12)
    n_experts, out_f, in_f = 2, 16, 24
    W_real = torch.randn(n_experts, out_f, in_f, dtype=torch.float32) * 0.1
    q = [_fp8_quant_channel(W_real[e]) for e in range(n_experts)]
    W_fp8 = torch.stack([w for w, _ in q], dim=0)
    scale = torch.stack([s for _, s in q], dim=0)  # (E, out, 1)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.experts.gate_up_proj": W_fp8,
        "model.layers.0.mlp.experts.gate_up_proj_scale_inv": scale,  # underscore naming
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path), filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats), output_dtype=torch.float32,
        model_class_name="Qwen3MoeForCausalLM", base_model_is_quantized=True, quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert "model.layers.0.mlp.experts.gate_up_proj_scale_inv" not in keys  # scale dropped
        merged = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj").float()
    assert torch.allclose(merged, W_real, atol=0.05)


def test_strip_fp8_suffix():
    from unsloth_zoo.saving_utils import _strip_fp8_suffix
    assert _strip_fp8_suffix("unsloth/GLM-5.2-FP8") == "unsloth/GLM-5.2"
    assert _strip_fp8_suffix("unsloth/Llama-3.1-8B-Instruct-FP8-Dynamic") == "unsloth/Llama-3.1-8B-Instruct"
    assert _strip_fp8_suffix("unsloth/Llama-3.1-8B-Instruct-FP8-Block") == "unsloth/Llama-3.1-8B-Instruct"
    assert _strip_fp8_suffix("org/Model-FP8-Static") == "org/Model"
    assert _strip_fp8_suffix("org/model-fp8") == "org/model"
    # No marker -> None (do not touch 16bit names).
    assert _strip_fp8_suffix("unsloth/GLM-5.2") is None
    assert _strip_fp8_suffix("unsloth/Llama-3.1-8B-Instruct") is None


def _write_16bit_model(path):
    path.mkdir(parents=True, exist_ok=True)
    save_file({"w": torch.zeros(2, 2)}, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text('{"model_type": "llama"}')


def test_resolve_fp8_16bit_sibling_local(tmp_path):
    from unsloth_zoo.saving_utils import _resolve_fp8_16bit_sibling
    # Existing non-quantized 16bit sibling -> resolved (merge onto full precision).
    _write_16bit_model(tmp_path / "GLM-5.2")
    got = _resolve_fp8_16bit_sibling(str(tmp_path / "GLM-5.2-FP8"))
    assert got is not None and str(got).endswith("GLM-5.2") and "FP8" not in str(got)
    # No sibling on disk -> None (fall back to FP8 dequant).
    assert _resolve_fp8_16bit_sibling(str(tmp_path / "Nonexistent-FP8")) is None
    # Plain 16bit name (no FP8 marker) -> None (nothing to resolve).
    assert _resolve_fp8_16bit_sibling(str(tmp_path / "GLM-5.2")) is None


def test_resolve_fp8_16bit_sibling_quantized_sibling_ignored(tmp_path):
    from unsloth_zoo.saving_utils import _resolve_fp8_16bit_sibling
    # A sibling that is itself quantized must NOT be used as the 16bit base.
    base = tmp_path / "GLM-5.2"
    base.mkdir(parents=True)
    save_file({"w": torch.zeros(2, 2)}, str(base / "model.safetensors"), metadata={"format": "pt"})
    (base / "config.json").write_text(
        '{"model_type": "llama", "quantization_config": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4"}}'
    )
    assert _resolve_fp8_16bit_sibling(str(tmp_path / "GLM-5.2-FP8")) is None


def test_fp8_keeps_non_companion_scale_tensors(tmp_path):
    """Tensors whose names merely end in `_scale`/`_scale_inv` but are NOT companions of
    an FP8 weight (logit_scale, router per_expert_scale, ...) must survive the rewrite."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(3)
    W_real = torch.randn(16, 32, dtype=torch.float32) * 0.1
    W_fp8, scale = _fp8_quant_channel(W_real)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.mlp.down_proj.weight": W_fp8,
        "model.layers.0.mlp.down_proj.weight_scale": scale,          # real FP8 companion -> dropped
        "model.layers.0.mlp.router.per_expert_scale": torch.randn(8, dtype=torch.bfloat16),
        "model.embed_scale": torch.randn(4, dtype=torch.bfloat16),
        "logit_scale": torch.tensor([2.5], dtype=torch.float32),
        "model.layers.0.attn.k_scale_inv": torch.randn(2, dtype=torch.bfloat16),
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path), filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats), output_dtype=torch.bfloat16,
        model_class_name="Qwen2ForCausalLM", base_model_is_quantized=True, quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        # FP8 companion dropped, weight dequantized.
        assert "model.layers.0.mlp.down_proj.weight_scale" not in keys
        assert f.get_tensor("model.layers.0.mlp.down_proj.weight").dtype == torch.bfloat16
        # Unrelated *_scale / *_scale_inv tensors preserved (not FP8 companions).
        for k in ("model.layers.0.mlp.router.per_expert_scale", "model.embed_scale",
                  "logit_scale", "model.layers.0.attn.k_scale_inv"):
            assert k in keys, f"{k} was wrongly dropped"


def test_fp8_preserves_non_fp8_buffer_dtypes(tmp_path):
    """Non-FP8, non-merged buffers (int64/bool/float32) must keep their dtype, not be cast
    to output_dtype like a dequantized FP8 weight would be."""
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora

    torch.manual_seed(4)
    W_fp8, scale = _fp8_quant_channel(torch.randn(16, 16, dtype=torch.float32) * 0.1)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {
        "model.layers.0.self_attn.q_proj.weight": W_fp8,
        "model.layers.0.self_attn.q_proj.weight_scale": scale,
        "model.rotary_emb.inv_freq": torch.randn(8, dtype=torch.float32),
        "model.layers.0.attn.bias_mask": torch.zeros(4, dtype=torch.bool),
        "model.position_ids": torch.arange(8, dtype=torch.int64),
    })

    _merge_and_overwrite_lora(
        save_directory=str(tmp_path), filename="model.safetensors",
        lora_weights=defaultdict(_FakeLoraStats), output_dtype=torch.bfloat16,
        model_class_name="Qwen2ForCausalLM", base_model_is_quantized=True, quant_type="fp8",
    )

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        # FP8 weight dequantized to output_dtype.
        assert f.get_tensor("model.layers.0.self_attn.q_proj.weight").dtype == torch.bfloat16
        # Buffers keep their original dtypes.
        assert f.get_tensor("model.rotary_emb.inv_freq").dtype == torch.float32
        assert f.get_tensor("model.layers.0.attn.bias_mask").dtype == torch.bool
        ids = f.get_tensor("model.position_ids")
        assert ids.dtype == torch.int64
        assert torch.equal(ids, torch.arange(8, dtype=torch.int64))
