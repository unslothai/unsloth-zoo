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

"""PEFT 3D-ParamWrapper layout drift canary (#5410). Fires if PEFT
introduces a third layout. CPU only."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model


# 3D MoE fused parameter (num_experts, 2*intermediate, hidden).
NUM_EXPERTS    = 4
INTERMEDIATE   = 8
HIDDEN         = 12
TWO_INTER      = 2 * INTERMEDIATE
PER_EXPERT_R   = 4
TOTAL_RANK     = NUM_EXPERTS * PER_EXPERT_R


class _ToyMoE(nn.Module):
    num_experts = NUM_EXPERTS

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, TWO_INTER, HIDDEN))

    def forward(self, x):
        return torch.einsum("bh,eih->bei", x, self.gate_up_proj)


def _peft_supports_target_parameters() -> bool:
    try:
        LoraConfig(r=1, target_parameters=["dummy"])
        return True
    except TypeError:
        return False
    except Exception:
        return True


@pytest.mark.skipif(not _peft_supports_target_parameters(),
                    reason="PEFT < 0.18 lacks target_parameters")
def test_paramwrapper_lora_shape_is_one_of_two_known_layouts():
    torch.manual_seed(0)
    base = _ToyMoE()

    cfg_kwargs = dict(r=PER_EXPERT_R, lora_alpha=PER_EXPERT_R * 2, lora_dropout=0.0, bias="none")
    try:
        cfg = LoraConfig(target_parameters=["gate_up_proj"], **cfg_kwargs)
    except TypeError:
        pytest.skip("Installed PEFT does not accept target_parameters yet")

    try:
        peft_model = get_peft_model(base, cfg)
    except Exception as e:
        pytest.skip(f"PEFT failed to wrap fused 3D param on this build: {e}")

    lora_A = lora_B = None
    for name, p in peft_model.named_parameters():
        if name.endswith("lora_A.default") or name.endswith("lora_A.default.weight"):
            lora_A = p
        elif name.endswith("lora_B.default") or name.endswith("lora_B.default.weight"):
            lora_B = p

    assert lora_A is not None and lora_B is not None, (
        f"lora_A / lora_B not found in named_parameters: "
        f"{[n for n, _ in peft_model.named_parameters()]}"
    )

    A_shape, B_shape = tuple(lora_A.shape), tuple(lora_B.shape)
    swapped  = ((TOTAL_RANK, TWO_INTER), (HIDDEN,    TOTAL_RANK))
    standard = ((TOTAL_RANK, HIDDEN),    (TWO_INTER, TOTAL_RANK))
    observed = (A_shape, B_shape)
    layout = "swapped" if observed == swapped else "standard" if observed == standard else "unknown"

    assert layout != "unknown", (
        f"PEFT layout drift: peft={peft.__version__} A={A_shape} B={B_shape}; "
        f"expected swapped={swapped} or standard={standard}. Update "
        f"_detect_moe_lora_layout + merge math (#5410)."
    )
    assert A_shape[0] // NUM_EXPERTS == PER_EXPERT_R


def test_zoo_detector_classifies_both_known_layouts():
    from unsloth_zoo.saving_utils import _detect_moe_lora_layout
    A = torch.empty(TOTAL_RANK, TWO_INTER); B = torch.empty(HIDDEN, TOTAL_RANK)
    assert _detect_moe_lora_layout(A, B, NUM_EXPERTS, TWO_INTER, HIDDEN) == ("swapped", PER_EXPERT_R)
    A = torch.empty(TOTAL_RANK, HIDDEN);    B = torch.empty(TWO_INTER, TOTAL_RANK)
    assert _detect_moe_lora_layout(A, B, NUM_EXPERTS, TWO_INTER, HIDDEN) == ("standard", PER_EXPERT_R)
