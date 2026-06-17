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

"""End-to-end LoRA merge correctness for per-expert MoE models.

Qwen3-MoE / GLM4-MoE / GraniteMoE store experts as separate 2D Linear modules, so
each expert projection merges like a dense layer. The two cases that matter:

  * full     -- expert + attention LoRA merges correctly per expert.
  * attn_only -- the user's core worry: when NO LoRA targets the experts, every
                 expert tensor (and shared experts / router) must stay
                 byte-identical to the base.
"""

from __future__ import annotations

import pytest

import _merge_e2e_helpers as H


# qwen3_moe == Qwen3-30B-A3B family; glm4_moe == GLM-4.x MoE / GLM-4.7-Flash family.
# Both store experts as separate 2D Linear modules. (GraniteMoe uses fused parallel
# expert modules, not 2D Linears, so it belongs with the fused-MoE suite.)
PER_EXPERT = ["qwen3_moe", "glm4_moe"]


def _skip_if_missing(family):
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")


@pytest.mark.parametrize("family", PER_EXPERT)
def test_moe_per_expert_full(family, tmp_path):
    """Experts + attention adapted: every adapted expert projection matches the
    reference; nothing else changes."""
    _skip_if_missing(family)
    try:
        n_adapted, _ = H.run_case(family, "full", str(tmp_path))
    except H.KeyResolutionError as e:
        # Some arches serialize experts fused-3D on newer transformers while PEFT
        # adapts them per-expert (or vice versa); the tiny-config key layout then
        # differs from this reference. The fused-3D merge path is covered by the
        # fused suite + the real-model sweep.
        pytest.skip(f"{family}: expert key layout differs on this transformers "
                    f"version ({e}); covered by the fused suite / real-model sweep")
    # full targets attention (4 per layer) + 3 expert projections * num_experts
    assert n_adapted >= 4


@pytest.mark.parametrize("family", PER_EXPERT)
def test_moe_per_expert_attention_only_keeps_experts_byte_identical(family, tmp_path):
    """No expert LoRA -> all expert/router/shared-expert tensors byte-identical.

    assert_merge_correct already enforces byte-identity for every non-adapted
    tensor, so a clean run here proves the MoE merge does not touch experts when
    no adapter targets them.
    """
    _skip_if_missing(family)
    n_adapted, n_pass = H.run_case(family, "attn_only", str(tmp_path))
    assert n_adapted >= 1 and n_pass >= 1
