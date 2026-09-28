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

"""End-to-end LoRA merge-to-16bit correctness for dense decoder models.

Llama / Qwen3 / Mistral / Gemma2 via the real merge: adapted == base + scale*(B@A),
untargeted tensors byte-identical. Covers full / attn-only / mlp-only, per-module
alpha+rank, bf16, and forced multi-shard.
"""

from __future__ import annotations

import pytest
import torch

import _merge_e2e_helpers as H


DENSE = ["llama", "qwen3", "mistral", "gemma2"]


def _skip_if_missing(family):
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")


@pytest.mark.parametrize("family", DENSE)
@pytest.mark.parametrize("scenario", ["full", "attn_only", "mlp_only"])
def test_dense_merge(family, scenario, tmp_path):
    _skip_if_missing(family)
    n_adapted, n_pass = H.run_case(family, scenario, str(tmp_path))
    assert n_adapted >= 1 and n_pass >= 1


@pytest.mark.parametrize("family", DENSE)
def test_dense_merge_bf16(family, tmp_path):
    """bf16 output: pass-through stays byte-identical, adapted within bf16 tol."""
    _skip_if_missing(family)
    H.run_case(family, "full", str(tmp_path), dtype=torch.bfloat16)


def test_dense_per_module_alpha_rank_pattern(tmp_path):
    """Per-module alpha/rank: the merge must read scale per module, not a single
    global lora_alpha/r."""
    _skip_if_missing("llama")
    H.run_case(
        "llama", "full", str(tmp_path),
        alpha_pattern={"q_proj": 4, "down_proj": 64},
        rank_pattern={"q_proj": 4, "down_proj": 16},
    )


def test_dense_forced_multishard(tmp_path):
    """Tiny max_shard_size forces multiple shards + index, catching stale-shard /
    index-mismatch bugs."""
    _skip_if_missing("llama")
    H.run_case("llama", "full", str(tmp_path), max_shard_size="40KB")
