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

Before the fix `_merge_and_overwrite_lora` wrote FP8 weights straight back for
dense compressed-tensors / finegrained_fp8 models: weight_scale was never
applied, companion scales were left dangling, and no error was raised. These
tests pin the rewrite path: FP8 weights dequantize to 16bit, companion scale
keys are dropped, and a missing scale raises instead of corrupting silently.
"""

from __future__ import annotations

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


def _fp8_quant_channel(W):
    """Per-channel (out, 1) FP8 quant like compressed-tensors fp8-dynamic."""
    max_fp8 = 448.0
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / max_fp8
    q = (W / scale).clamp(-max_fp8, max_fp8).to(torch.float8_e4m3fn)
    return q, scale.to(torch.bfloat16)


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
    assert not _is_fp8_quant_config({"quant_method": "gptq", "bits": 4})
    assert not _is_fp8_quant_config({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "int", "num_bits": 8}}},
    })
