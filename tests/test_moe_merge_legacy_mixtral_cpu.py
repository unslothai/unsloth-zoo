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

transformers v5 fuses Mixtral experts (w1+w3 -> gate_up_proj, w2 -> down_proj) in
memory, but save_pretrained still writes them unfused on disk as
block_sparse_moe.experts.N.w1/w2/w3.weight. The merge must map those disk keys
back to the mlp.experts LoRA so the expert deltas are not silently dropped.
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
    hidden, intermediate = 16, 8
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

    # Every per-expert tensor changed; nothing was skipped via fallback.
    n_expert = sum(1 for k in base if ".experts." in k)
    n_changed = sum(
        1 for k in base
        if (base[k].float() - merged[k].float()).abs().max().item() > 1e-6
    )
    assert n_changed == n_expert
    assert _MOE_MERGE_STATE["fallback"] == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    # 2 fused LoRA modules per layer (gate_up + down) were counted.
    assert count == num_layers * 2

    # Disk layout stays unfused (drop-in for the original checkpoint).
    assert not any("gate_up_proj" in k for k in merged)

    # Numeric correctness against the analytic LoRA delta per expert.
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
