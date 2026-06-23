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

"""CPU regression for legacy-Mixtral MoE merge (#5403).

transformers v5 fuses Mixtral experts in-memory but still writes them unfused on
disk; the merge must map those disk keys back to the mlp.experts LoRA.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import torch
from safetensors.torch import load_file, save_file

from unsloth_zoo.saving_utils import (
    LoraStats,
    _MOE_MERGE_STATE,
    _merge_and_overwrite_lora,
    _reset_moe_merge_state,
)


SEED = 5403


@dataclass
class _InnerMoE:
    num_experts: int


def _build_legacy_shard(path, num_layers, num_experts, hidden, intermediate, dtype=torch.float32):
    torch.manual_seed(SEED)
    tensors = {}
    for L in range(num_layers):
        for e in range(num_experts):
            p = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            tensors[f"{p}.w1.weight"] = torch.randn(intermediate, hidden, dtype=dtype)  # gate
            tensors[f"{p}.w3.weight"] = torch.randn(intermediate, hidden, dtype=dtype)  # up
            tensors[f"{p}.w2.weight"] = torch.randn(hidden, intermediate, dtype=dtype)  # down
    save_file(tensors, path)
    return tensors


def _make_lora_weights(num_layers, num_experts, rank_per, hidden, intermediate, alpha, dtype=torch.float32):
    """Standard PEFT-0.19 fused layout: gate_up lora on .base_layer, down on the experts module."""
    torch.manual_seed(SEED + 1)
    TR = num_experts * rank_per
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu = torch.randn(TR, hidden, dtype=dtype) * 0.05
        B_gu = torch.randn(2 * intermediate, TR, dtype=dtype) * 0.05
        A_dn = torch.randn(TR, intermediate, dtype=dtype) * 0.05
        B_dn = torch.randn(hidden, TR, dtype=dtype) * 0.05
        lw[prefix + ".base_layer"] = LoraStats(_InnerMoE(num_experts), A_gu, B_gu, alpha)
        lw[prefix] = LoraStats(_InnerMoE(num_experts), A_dn, B_dn, alpha)
    return lw


def _delta(A, B, alpha, expert_idx, num_experts):
    r = A.shape[0] // num_experts
    s, e = expert_idx * r, (expert_idx + 1) * r
    a = A[s:e].to(torch.float64)
    b = B[:, s:e].to(torch.float64)
    return alpha * (b @ a)  # standard layout: (out, H)


def test_legacy_mixtral_w1w2w3_experts_are_merged(tmp_path):
    num_layers, num_experts, rank_per = 2, 4, 4
    # 2*intermediate != hidden keeps gate_up_proj shape-distinct.
    hidden, intermediate = 12, 8
    alpha = 2.0
    path = str(tmp_path / "model.safetensors")

    base = _build_legacy_shard(path, num_layers, num_experts, hidden, intermediate)
    lw = _make_lora_weights(num_layers, num_experts, rank_per, hidden, intermediate, alpha)

    _reset_moe_merge_state()
    result = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=lw,
        output_dtype=torch.float32,
        model_class_name="MixtralForCausalLM",
    )
    count = result[0] if isinstance(result, tuple) else result
    merged = load_file(path)

    # Every per-expert tensor changed; nothing fell back.
    n_expert = sum(1 for k in base if ".experts." in k)
    n_changed = sum(
        1 for k in base
        if (base[k].float() - merged[k].float()).abs().max().item() > 1e-6
    )
    assert n_changed == n_expert
    assert _MOE_MERGE_STATE["fallback"] == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    # gate_up + down per layer.
    assert count == num_layers * 2

    # Disk layout stays unfused.
    assert not any("gate_up_proj" in k for k in merged)

    # Numeric check against the analytic LoRA delta per expert.
    max_err = 0.0
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu, B_gu = lw[prefix + ".base_layer"].lora_A, lw[prefix + ".base_layer"].lora_B
        A_dn, B_dn = lw[prefix].lora_A, lw[prefix].lora_B
        for e in range(num_experts):
            dp = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            gu_delta = _delta(A_gu, B_gu, alpha, e, num_experts)  # (2I, H)
            ref_w1 = base[f"{dp}.w1.weight"].to(torch.float64) + gu_delta[:intermediate]
            ref_w3 = base[f"{dp}.w3.weight"].to(torch.float64) + gu_delta[intermediate:]
            ref_w2 = base[f"{dp}.w2.weight"].to(torch.float64) + _delta(A_dn, B_dn, alpha, e, num_experts)
            for ref, key in ((ref_w1, "w1"), (ref_w3, "w3"), (ref_w2, "w2")):
                got = merged[f"{dp}.{key}.weight"].to(torch.float64)
                max_err = max(max_err, (got - ref).abs().max().item())
    assert max_err < 1e-4, f"legacy Mixtral merge delta error too large: {max_err:.2e}"
    _reset_moe_merge_state()


def test_legacy_mixtral_gate_up_proj_keyed_adapter_is_merged(tmp_path):
    """gate_up LoRA keyed on .gate_up_proj (no .base_layer wrapper) is still found by the legacy w1/w3 path."""
    num_layers, num_experts, rank_per = 2, 4, 4
    # 2*intermediate != hidden keeps gate_up_proj shape-distinct.
    hidden, intermediate = 12, 8
    alpha = 2.0
    path = str(tmp_path / "model.safetensors")

    base = _build_legacy_shard(path, num_layers, num_experts, hidden, intermediate)

    torch.manual_seed(SEED + 7)
    TR = num_experts * rank_per
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu = torch.randn(TR, hidden, dtype=torch.float32) * 0.05
        B_gu = torch.randn(2 * intermediate, TR, dtype=torch.float32) * 0.05
        A_dn = torch.randn(TR, intermediate, dtype=torch.float32) * 0.05
        B_dn = torch.randn(hidden, TR, dtype=torch.float32) * 0.05
        # gate_up keyed on .gate_up_proj instead of .base_layer.
        lw[prefix + ".gate_up_proj"] = LoraStats(_InnerMoE(num_experts), A_gu, B_gu, alpha)
        lw[prefix] = LoraStats(_InnerMoE(num_experts), A_dn, B_dn, alpha)

    _reset_moe_merge_state()
    result = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=lw,
        output_dtype=torch.float32,
        model_class_name="MixtralForCausalLM",
    )
    count = result[0] if isinstance(result, tuple) else result
    merged = load_file(path)

    n_expert = sum(1 for k in base if ".experts." in k)
    n_changed = sum(
        1 for k in base
        if (base[k].float() - merged[k].float()).abs().max().item() > 1e-6
    )
    # Without the .gate_up_proj fallback the w1/w3 deltas would be dropped.
    assert n_changed == n_expert
    assert _MOE_MERGE_STATE["fallback"] == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    assert count == num_layers * 2

    max_err = 0.0
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu, B_gu = lw[prefix + ".gate_up_proj"].lora_A, lw[prefix + ".gate_up_proj"].lora_B
        A_dn, B_dn = lw[prefix].lora_A, lw[prefix].lora_B
        for e in range(num_experts):
            dp = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            gu_delta = _delta(A_gu, B_gu, alpha, e, num_experts)
            ref_w1 = base[f"{dp}.w1.weight"].to(torch.float64) + gu_delta[:intermediate]
            ref_w3 = base[f"{dp}.w3.weight"].to(torch.float64) + gu_delta[intermediate:]
            ref_w2 = base[f"{dp}.w2.weight"].to(torch.float64) + _delta(A_dn, B_dn, alpha, e, num_experts)
            for ref, key in ((ref_w1, "w1"), (ref_w3, "w3"), (ref_w2, "w2")):
                got = merged[f"{dp}.{key}.weight"].to(torch.float64)
                max_err = max(max_err, (got - ref).abs().max().item())
    assert max_err < 1e-4, f"gate_up_proj keyed legacy merge delta error too large: {max_err:.2e}"
    _reset_moe_merge_state()


def test_legacy_mixtral_down_proj_keyed_adapter_is_merged(tmp_path):
    """down LoRA keyed on .down_proj (PEFT target_parameters for 3D MoE) is still found by the legacy w2 path."""
    num_layers, num_experts, rank_per = 2, 4, 4
    hidden, intermediate = 12, 8
    alpha = 2.0
    path = str(tmp_path / "model.safetensors")

    base = _build_legacy_shard(path, num_layers, num_experts, hidden, intermediate)

    torch.manual_seed(SEED + 9)
    TR = num_experts * rank_per
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu = torch.randn(TR, hidden, dtype=torch.float32) * 0.05
        B_gu = torch.randn(2 * intermediate, TR, dtype=torch.float32) * 0.05
        A_dn = torch.randn(TR, intermediate, dtype=torch.float32) * 0.05
        B_dn = torch.randn(hidden, TR, dtype=torch.float32) * 0.05
        lw[prefix + ".base_layer"] = LoraStats(_InnerMoE(num_experts), A_gu, B_gu, alpha)
        # down keyed on .down_proj instead of the experts module.
        lw[prefix + ".down_proj"] = LoraStats(_InnerMoE(num_experts), A_dn, B_dn, alpha)

    _reset_moe_merge_state()
    result = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=lw,
        output_dtype=torch.float32,
        model_class_name="MixtralForCausalLM",
    )
    count = result[0] if isinstance(result, tuple) else result
    merged = load_file(path)

    n_expert = sum(1 for k in base if ".experts." in k)
    n_changed = sum(
        1 for k in base
        if (base[k].float() - merged[k].float()).abs().max().item() > 1e-6
    )
    # Without the .down_proj fallback the w2 deltas would be dropped.
    assert n_changed == n_expert
    assert _MOE_MERGE_STATE["fallback"] == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    assert count == num_layers * 2

    max_err = 0.0
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu, B_gu = lw[prefix + ".base_layer"].lora_A, lw[prefix + ".base_layer"].lora_B
        A_dn, B_dn = lw[prefix + ".down_proj"].lora_A, lw[prefix + ".down_proj"].lora_B
        for e in range(num_experts):
            dp = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            gu_delta = _delta(A_gu, B_gu, alpha, e, num_experts)
            ref_w1 = base[f"{dp}.w1.weight"].to(torch.float64) + gu_delta[:intermediate]
            ref_w3 = base[f"{dp}.w3.weight"].to(torch.float64) + gu_delta[intermediate:]
            ref_w2 = base[f"{dp}.w2.weight"].to(torch.float64) + _delta(A_dn, B_dn, alpha, e, num_experts)
            for ref, key in ((ref_w1, "w1"), (ref_w3, "w3"), (ref_w2, "w2")):
                got = merged[f"{dp}.{key}.weight"].to(torch.float64)
                max_err = max(max_err, (got - ref).abs().max().item())
    assert max_err < 1e-4, f"down_proj keyed legacy merge delta error too large: {max_err:.2e}"
    _reset_moe_merge_state()


def test_legacy_mixtral_both_fused_keys_resolve_num_experts(tmp_path):
    """gate_up on .gate_up_proj AND down on .down_proj, with no .base_layer / bare .experts
    key. Neither matches the #5410 num_experts override, so num_experts must come from the
    wrapped module (lora_stats), not the shard key scan, or the fused rank slices are wrong."""
    num_layers, num_experts, rank_per = 1, 4, 4
    hidden, intermediate = 12, 8
    alpha = 2.0
    path = str(tmp_path / "model.safetensors")

    base = _build_legacy_shard(path, num_layers, num_experts, hidden, intermediate)

    torch.manual_seed(SEED + 13)
    TR = num_experts * rank_per
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu = torch.randn(TR, hidden, dtype=torch.float32) * 0.05
        B_gu = torch.randn(2 * intermediate, TR, dtype=torch.float32) * 0.05
        A_dn = torch.randn(TR, intermediate, dtype=torch.float32) * 0.05
        B_dn = torch.randn(hidden, TR, dtype=torch.float32) * 0.05
        # Only fused-parameter keys; no .base_layer or bare .experts to seed moe_num_experts.
        lw[prefix + ".gate_up_proj"] = LoraStats(_InnerMoE(num_experts), A_gu, B_gu, alpha)
        lw[prefix + ".down_proj"] = LoraStats(_InnerMoE(num_experts), A_dn, B_dn, alpha)

    _reset_moe_merge_state()
    result = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=lw,
        output_dtype=torch.float32,
        model_class_name="MixtralForCausalLM",
    )
    count = result[0] if isinstance(result, tuple) else result
    merged = load_file(path)

    n_expert = sum(1 for k in base if ".experts." in k)
    n_changed = sum(
        1 for k in base
        if (base[k].float() - merged[k].float()).abs().max().item() > 1e-6
    )
    assert n_changed == n_expert
    assert _MOE_MERGE_STATE["fallback"] == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    assert count == num_layers * 2

    max_err = 0.0
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu, B_gu = lw[prefix + ".gate_up_proj"].lora_A, lw[prefix + ".gate_up_proj"].lora_B
        A_dn, B_dn = lw[prefix + ".down_proj"].lora_A, lw[prefix + ".down_proj"].lora_B
        for e in range(num_experts):
            dp = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            gu_delta = _delta(A_gu, B_gu, alpha, e, num_experts)
            ref_w1 = base[f"{dp}.w1.weight"].to(torch.float64) + gu_delta[:intermediate]
            ref_w3 = base[f"{dp}.w3.weight"].to(torch.float64) + gu_delta[intermediate:]
            ref_w2 = base[f"{dp}.w2.weight"].to(torch.float64) + _delta(A_dn, B_dn, alpha, e, num_experts)
            for ref, key in ((ref_w1, "w1"), (ref_w3, "w3"), (ref_w2, "w2")):
                got = merged[f"{dp}.{key}.weight"].to(torch.float64)
                max_err = max(max_err, (got - ref).abs().max().item())
    assert max_err < 1e-4, f"both-fused-keys legacy merge delta error too large: {max_err:.2e}"
    _reset_moe_merge_state()


def test_legacy_mixtral_fp8_shard_is_quant_aware(tmp_path):
    """FP8 legacy Mixtral shard must dequantise -> merge -> requantise and rewrite the scale, not merge raw float8."""
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
        _fp8_dequant_blockwise,
        _fp8_requant_blockwise,
    )

    num_layers, num_experts, rank_per = 1, 2, 4
    # 2*intermediate != hidden keeps gate_up_proj shape-distinct.
    hidden, intermediate = 12, 8
    alpha = 2.0
    path = str(tmp_path / "model.safetensors")

    # FP8 shard with a single block scale; merging raw float8 (the bug) drops the delta.
    torch.manual_seed(SEED + 11)
    hp = {}     # dequantised base each merge starts from
    tensors = {}
    for L in range(num_layers):
        for e in range(num_experts):
            p = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            for w_name, shape in (("w1", (intermediate, hidden)),
                                  ("w3", (intermediate, hidden)),
                                  ("w2", (hidden, intermediate))):
                W_hp = torch.randn(*shape, dtype=torch.float32)
                W_fp8, scale = _fp8_requant_blockwise(W_hp, shape, torch.float32)
                tensors[f"{p}.{w_name}.weight"] = W_fp8
                tensors[f"{p}.{w_name}.weight_scale_inv"] = scale
                hp[f"{p}.{w_name}.weight"] = _fp8_dequant_blockwise(W_fp8, scale)
    save_file(tensors, path)

    # LoRA deltas sized to a clear fraction of the base so a dropped delta is unambiguous.
    TR = num_experts * rank_per
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu = torch.randn(TR, hidden, dtype=torch.float32) * 0.3
        B_gu = torch.randn(2 * intermediate, TR, dtype=torch.float32) * 0.3
        A_dn = torch.randn(TR, intermediate, dtype=torch.float32) * 0.3
        B_dn = torch.randn(hidden, TR, dtype=torch.float32) * 0.3
        lw[prefix + ".base_layer"] = LoraStats(_InnerMoE(num_experts), A_gu, B_gu, alpha)
        lw[prefix] = LoraStats(_InnerMoE(num_experts), A_dn, B_dn, alpha)

    _reset_moe_merge_state()
    result = _merge_and_overwrite_lora(
        save_directory=str(tmp_path),
        filename="model.safetensors",
        lora_weights=lw,
        output_dtype=torch.float32,
        model_class_name="MixtralForCausalLM",
        base_model_is_quantized=True,
        quant_type="fp8",
    )
    count = result[0] if isinstance(result, tuple) else result
    merged = load_file(path)

    # Data stays float8 and scales were rewritten.
    for k, v in merged.items():
        if k.endswith(".weight"):
            assert v.dtype == torch.float8_e4m3fn, f"{k} should stay FP8, got {v.dtype}"
    assert _MOE_MERGE_STATE["fallback"] == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    assert count == num_layers * 2

    # Requantised weights must track dequant base + delta; a raw-float8 merge loses the delta.
    max_err = 0.0
    min_delta_ratio = 1.0
    for L in range(num_layers):
        prefix = f"model.layers.{L}.mlp.experts"
        A_gu, B_gu = lw[prefix + ".base_layer"].lora_A, lw[prefix + ".base_layer"].lora_B
        A_dn, B_dn = lw[prefix].lora_A, lw[prefix].lora_B
        for e in range(num_experts):
            dp = f"model.layers.{L}.block_sparse_moe.experts.{e}"
            gu_delta = _delta(A_gu, B_gu, alpha, e, num_experts)
            exp_deltas = {
                "w1": gu_delta[:intermediate],
                "w3": gu_delta[intermediate:],
                "w2": _delta(A_dn, B_dn, alpha, e, num_experts),
            }
            for w_name, exp_delta in exp_deltas.items():
                base_dq = hp[f"{dp}.{w_name}.weight"].to(torch.float64)
                ref = base_dq + exp_delta
                scale = merged[f"{dp}.{w_name}.weight_scale_inv"]
                got = _fp8_dequant_blockwise(merged[f"{dp}.{w_name}.weight"], scale).to(torch.float64)
                denom = ref.abs().max().item() or 1.0
                max_err = max(max_err, (got - ref).abs().max().item() / denom)
                # Fraction of the intended delta that landed on disk.
                got_delta = got - base_dq
                exp_norm = exp_delta.norm().item() or 1.0
                ratio = (got_delta.norm().item()) / exp_norm
                min_delta_ratio = min(min_delta_ratio, ratio)
    assert max_err < 0.1, f"FP8 legacy merge delta error too large: {max_err:.2e}"
    assert min_delta_ratio > 0.8, (
        f"LoRA delta not applied through the quant-aware path (ratio={min_delta_ratio:.2f})"
    )
    _reset_moe_merge_state()
