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

"""CPU end-to-end regression for per-expert MoE merge (#5410)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from unsloth_zoo.saving_utils import (
    LoraStats,
    _MOE_MERGE_STATE,
    _detect_moe_lora_layout,
    _merge_moe_down_proj_expert,
    _merge_moe_gate_expert,
    _merge_moe_up_expert,
    _reset_moe_merge_state,
    _resolve_num_experts_from_lora_stats,
)


SEED = 5410


@dataclass
class _InnerMoE:
    num_experts: int


@dataclass
class _OuterParamWrapper:
    base_layer: object


def _build_synthetic_layer(num_experts, rank_per, hidden, intermediate, layout, alpha, dtype=torch.float32):
    torch.manual_seed(SEED)
    TR = num_experts * rank_per
    fused_gate_up = torch.randn(num_experts, 2 * intermediate, hidden, dtype=dtype)
    fused_down    = torch.randn(num_experts, hidden, intermediate, dtype=dtype)
    if layout == "swapped":
        A_gu = torch.randn(TR, 2 * intermediate, dtype=dtype) * 0.05
        B_gu = torch.randn(hidden, TR, dtype=dtype) * 0.05
        A_dn = torch.randn(TR, hidden, dtype=dtype) * 0.05
        B_dn = torch.randn(intermediate, TR, dtype=dtype) * 0.05
    elif layout == "standard":
        A_gu = torch.randn(TR, hidden, dtype=dtype) * 0.05
        B_gu = torch.randn(2 * intermediate, TR, dtype=dtype) * 0.05
        A_dn = torch.randn(TR, intermediate, dtype=dtype) * 0.05
        B_dn = torch.randn(hidden, TR, dtype=dtype) * 0.05
    else:
        raise ValueError(layout)
    return fused_gate_up, fused_down, A_gu, B_gu, A_dn, B_dn


def _analytic_gate_up_delta(A, B, alpha, expert_idx, num_experts, role, layout, I, H):
    r = A.shape[0] // num_experts
    s, e = expert_idx * r, (expert_idx + 1) * r
    a = A[s:e].to(torch.float64); b = B[:, s:e].to(torch.float64)
    if layout == "swapped":
        half = a[:, :I] if role == "gate" else a[:, I:]
        return alpha * (b @ half).T
    half = b[:I, :] if role == "gate" else b[I:, :]
    return alpha * (half @ a)


def _analytic_down_delta(A, B, alpha, expert_idx, num_experts, layout):
    r = A.shape[0] // num_experts
    s, e = expert_idx * r, (expert_idx + 1) * r
    a = A[s:e].to(torch.float64); b = B[:, s:e].to(torch.float64)
    if layout == "swapped":
        return alpha * (b @ a).T
    return alpha * (b @ a)


@pytest.mark.parametrize("layout", ["swapped", "standard"])
def test_per_layer_merge_round_trip(layout):
    num_layers, num_experts, rank_per = 2, 4, 4
    hidden, intermediate = 12, 8
    alpha = 8.0
    dtype = torch.float32

    _reset_moe_merge_state()
    total_expected_apply = 0
    max_err = 0.0
    for layer in range(num_layers):
        fused_gu, fused_dn, A_gu, B_gu, A_dn, B_dn = _build_synthetic_layer(
            num_experts, rank_per, hidden, intermediate, layout, alpha, dtype
        )
        stats_gu = LoraStats(module=_InnerMoE(num_experts), lora_A=A_gu, lora_B=B_gu, alpha=alpha)
        stats_dn = LoraStats(module=_InnerMoE(num_experts), lora_A=A_dn, lora_B=B_dn, alpha=alpha)

        for ei in range(num_experts):
            gate_disk = fused_gu[ei, :intermediate, :].clone()
            up_disk   = fused_gu[ei, intermediate:, :].clone()
            down_disk = fused_dn[ei].clone()

            gate_out = _merge_moe_gate_expert(gate_disk, stats_gu, ei, num_experts, dtype)
            up_out   = _merge_moe_up_expert  (up_disk,   stats_gu, ei, num_experts, dtype)
            down_out = _merge_moe_down_proj_expert(down_disk, stats_dn, ei, num_experts, dtype)

            gate_ref = (fused_gu[ei, :intermediate, :].to(torch.float64)
                        + _analytic_gate_up_delta(A_gu, B_gu, alpha, ei, num_experts, "gate", layout, intermediate, hidden)).to(dtype)
            up_ref   = (fused_gu[ei, intermediate:, :].to(torch.float64)
                        + _analytic_gate_up_delta(A_gu, B_gu, alpha, ei, num_experts, "up",   layout, intermediate, hidden)).to(dtype)
            down_ref = (fused_dn[ei].to(torch.float64)
                        + _analytic_down_delta(A_dn, B_dn, alpha, ei, num_experts, layout)).to(dtype)

            for out, ref in ((gate_out, gate_ref), (up_out, up_ref), (down_out, down_ref)):
                err = (out.cpu() - ref.cpu()).abs().max().item()
                if err > max_err: max_err = err
                total_expected_apply += 1

    assert max_err < 1e-4, f"merge delta error too large: {max_err:.2e}"
    assert _MOE_MERGE_STATE["applied"]   == total_expected_apply
    assert _MOE_MERGE_STATE["attempted"] == total_expected_apply
    assert _MOE_MERGE_STATE["fallback"]  == 0
    assert _MOE_MERGE_STATE["first_error"] is None
    _reset_moe_merge_state()


def test_unrecognised_layout_records_fallback_and_first_error():
    _reset_moe_merge_state()
    num_experts, rank_per, intermediate, hidden = 4, 4, 8, 12
    TR = num_experts * rank_per
    W = torch.randn(intermediate, hidden)
    A = torch.randn(TR, hidden + 7); B = torch.randn(hidden, TR)
    stats = LoraStats(module=_InnerMoE(num_experts), lora_A=A, lora_B=B, alpha=1.0)
    out = _merge_moe_gate_expert(W.clone(), stats, 0, num_experts, torch.float32)
    assert torch.equal(out.cpu(), W)
    assert _MOE_MERGE_STATE["fallback"] >= 1
    err = _MOE_MERGE_STATE["first_error"]
    assert err is not None and err["role"] == "gate"
    assert err["lora_A_shape"] == (TR, hidden + 7)
    _reset_moe_merge_state()


def test_resolver_walks_outer_wrapper_chain():
    """Walks past outer ParamWrapper (.module=None) to inner num_experts."""
    outer = _OuterParamWrapper(base_layer=_InnerMoE(num_experts=128))
    stats = LoraStats(module=outer, lora_A=None, lora_B=None, alpha=0.0)
    assert _resolve_num_experts_from_lora_stats(stats, fallback=-1) == 128


def test_resolver_terminates_on_self_cycle():
    class SelfCycle: pass
    sc = SelfCycle(); sc.base_layer = sc
    stats = LoraStats(module=sc, lora_A=None, lora_B=None, alpha=0.0)
    assert _resolve_num_experts_from_lora_stats(stats, fallback=42) == 42


def test_detector_is_stable_against_non_divisor_num_experts():
    num_experts, rank_per, intermediate, hidden = 128, 4, 8, 12
    TR = num_experts * rank_per
    A = torch.empty(TR, hidden); B = torch.empty(2 * intermediate, TR)
    layout, _ = _detect_moe_lora_layout(A, B, num_experts=17, out_dim=2*intermediate, in_dim=hidden)
    assert layout == "unknown"
