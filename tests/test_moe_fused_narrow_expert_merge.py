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

"""CPU regression for the fused-3D MoE merge on narrow-expert layouts.

In a narrow-expert MoE (Gemma-4 26B-A4B, Qwen3.5-35B-A3B) gate_up is
(2*intermediate, hidden) with 2*intermediate < hidden. The shape-magnitude
heuristic assumes 2*intermediate > hidden, so it mislabels these as transposed
and drops the LoRA delta. These tests feed a narrow-expert standard layout (no
hint, and the WRONG is_transposed=True hint) and require the delta to still be
applied -- the LoRA-dimension layout must win.
"""

from __future__ import annotations

import pytest
import torch

from unsloth_zoo.saving_utils import (
    LoraStats,
    _MOE_MERGE_STATE,
    _merge_moe_fused_down_proj_expert,
    _merge_moe_fused_gate_up_expert,
    _reset_moe_merge_state,
)

SEED = 5410
NUM_EXPERTS, RANK, ALPHA = 4, 2, 8.0
# Narrow-expert ratios (2*intermediate < hidden), scaled down from real models.
GEOMS = {"gemma4_26B_A4B": (16, 3), "qwen3_5_35B_A3B": (18, 4)}  # (hidden, intermediate)


def _build(hidden, intermediate, dtype=torch.float32):
    torch.manual_seed(SEED)
    TR = NUM_EXPERTS * RANK
    twoI = 2 * intermediate
    gate_up_W = torch.randn(NUM_EXPERTS, twoI, hidden, dtype=dtype)          # (E, 2I, H)
    down_W    = torch.randn(NUM_EXPERTS, hidden, intermediate, dtype=dtype)  # (E, H, I)
    # standard PEFT fused: lora_A=(E*r, in), lora_B=(out, E*r)
    A_gu = torch.randn(TR, hidden, dtype=dtype) * 0.05
    B_gu = torch.randn(twoI, TR, dtype=dtype) * 0.05
    A_dn = torch.randn(TR, intermediate, dtype=dtype) * 0.05
    B_dn = torch.randn(hidden, TR, dtype=dtype) * 0.05
    return gate_up_W, down_W, A_gu, B_gu, A_dn, B_dn


def _reference(W, A, B, alpha):
    out = W.clone().to(torch.float64)
    for e in range(NUM_EXPERTS):
        s, t = e * RANK, (e + 1) * RANK
        out[e] += alpha * (B[:, s:t].to(torch.float64) @ A[s:t, :].to(torch.float64))
    return out


# is_transposed=True is the wrong value the magnitude heuristic produces for these.
@pytest.mark.parametrize("model", sorted(GEOMS))
@pytest.mark.parametrize("hint", [None, True])
def test_fused_narrow_expert_merge_applies_delta(model, hint):
    hidden, intermediate = GEOMS[model]
    gate_up_W, down_W, A_gu, B_gu, A_dn, B_dn = _build(hidden, intermediate)
    stats_gu = LoraStats(module=None, lora_A=A_gu, lora_B=B_gu, alpha=ALPHA)
    stats_dn = LoraStats(module=None, lora_A=A_dn, lora_B=B_dn, alpha=ALPHA)
    kw = {} if hint is None else {"is_transposed": hint}

    _reset_moe_merge_state()
    gu_out = _merge_moe_fused_gate_up_expert(gate_up_W.clone(), stats_gu, torch.float32, **kw)
    dn_out = _merge_moe_fused_down_proj_expert(down_W.clone(), stats_dn, torch.float32, **kw)

    gu_ref = _reference(gate_up_W, A_gu, B_gu, ALPHA)
    dn_ref = _reference(down_W, A_dn, B_dn, ALPHA)
    gu_err = (gu_out.cpu().to(torch.float64) - gu_ref).abs().max().item()
    dn_err = (dn_out.cpu().to(torch.float64) - dn_ref).abs().max().item()

    assert _MOE_MERGE_STATE["fallback"] == 0, _MOE_MERGE_STATE["first_error"]
    assert _MOE_MERGE_STATE["applied"] == 2
    assert gu_err < 1e-4 and dn_err < 1e-4, f"gate_up={gu_err:.2e} down={dn_err:.2e}"
    # delta must actually be applied, not the base written through
    assert not torch.equal(gu_out.cpu(), gate_up_W)
    assert not torch.equal(dn_out.cpu(), down_W)
    _reset_moe_merge_state()
