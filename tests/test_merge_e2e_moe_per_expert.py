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

"""Per-expert MoE: experts are 2D Linears merged like dense; attn_only keeps them byte-identical."""

from __future__ import annotations

import pytest

import _merge_e2e_helpers as H


# qwen3_moe ~ Qwen3-30B-A3B; glm4_moe ~ GLM-4.7-Flash. (GraniteMoe is fused -> fused suite.)
PER_EXPERT = ["qwen3_moe", "glm4_moe"]


def _skip_if_missing(family):
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")


@pytest.mark.parametrize("family", PER_EXPERT)
def test_moe_per_expert_full(family, tmp_path):
    """Every adapted expert projection matches the reference; nothing else changes."""
    _skip_if_missing(family)
    try:
        n_adapted, _ = H.run_case(family, "full", str(tmp_path))
    except H.KeyResolutionError as e:
        # Expert key layout differs by transformers version (fused-3D vs per-expert); fused suite covers it.
        pytest.skip(f"{family}: expert key layout differs on this transformers "
                    f"version ({e}); covered by the fused suite / real-model sweep")
    # attention + expert projections
    assert n_adapted >= 4


@pytest.mark.parametrize("family", PER_EXPERT)
def test_moe_per_expert_attention_only_keeps_experts_byte_identical(family, tmp_path):
    """No expert LoRA: expert, router and shared-expert tensors stay byte-identical."""
    _skip_if_missing(family)
    n_adapted, n_pass = H.run_case(family, "attn_only", str(tmp_path))
    assert n_adapted >= 1 and n_pass >= 1
