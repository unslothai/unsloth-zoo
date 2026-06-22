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

"""Expert-count resolution precedence for per-expert MoE merges.

A w1/w3/w2 checkpoint can split one layer's experts across shards; a low-index shard
(experts 0..15 of 64) reports 16, which drives the per-expert LoRA slicing stride. These
tests pin that the authoritative module/fused-LoRA count always wins and the shard header
may only RAISE a missing/too-low count, never lower it.
"""

from __future__ import annotations

import types

from unsloth_zoo.saving_utils import _resolve_moe_num_experts_with_header


PREFIX = "model.layers.0.feed_forward.experts"
SCHEME = ("w1", "w3", "w2")


def _header(prefix, indices, scheme=SCHEME):
    """Build a fake shard header containing per-expert tensors for ``indices``."""
    keys = {}
    for i in indices:
        for name in scheme:
            keys[f"{prefix}.{i}.{name}.weight"] = object()
    return keys


class _Stats:
    """Minimal stand-in for the LoRA stats object the merge passes around."""
    def __init__(self, module=None, lora_A=None, lora_B=None, rank=None):
        self.module = module
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.rank = rank


def test_partial_low_index_shard_does_not_lower_module_count():
    # Live module exposes the true count (64); this shard only holds experts 0..15.
    module = types.SimpleNamespace(num_experts=64)
    stats = _Stats(module=module)
    moe_num_experts = {}
    header = _header(PREFIX, range(16))

    n = _resolve_moe_num_experts_with_header(
        PREFIX, stats, moe_num_experts, header, SCHEME,
    )
    assert n == 64, f"shard-local 16 must not override module 64 (got {n})"
    assert moe_num_experts[PREFIX] == 64


def test_header_seeds_count_when_module_absent_single_shard():
    # No module, no fused LoRA to derive from: a down_proj-only adapter would
    # otherwise leave experts 1..N-1 unmerged. The full shard header supplies 64.
    stats = _Stats(module=None)
    moe_num_experts = {}
    header = _header(PREFIX, range(64))

    n = _resolve_moe_num_experts_with_header(
        PREFIX, stats, moe_num_experts, header, SCHEME,
    )
    assert n == 64, f"header must seed the count when nothing else resolves (got {n})"
    assert moe_num_experts[PREFIX] == 64


def test_header_raises_a_degenerate_derived_count():
    # Per-expert LoRA derives a degenerate 1 (total_rank == rank); the header,
    # which sees all 64 experts in this shard, must raise it.
    import torch

    stats = _Stats(
        module=None,
        lora_A=torch.zeros((8, 16)),   # total_rank == rank -> derived count 1
        lora_B=torch.zeros((16, 8)),
        rank=8,
    )
    moe_num_experts = {}
    header = _header(PREFIX, range(64))

    n = _resolve_moe_num_experts_with_header(
        PREFIX, stats, moe_num_experts, header, SCHEME,
    )
    assert n == 64, f"header must raise a degenerate derived count of 1 (got {n})"


def test_fused_lora_shape_count_not_lowered_by_partial_shard():
    # No module, but a fused LoRA encodes 64 experts (total_rank // rank == 64).
    # A partial shard reporting 16 must not lower it.
    import torch

    stats = _Stats(
        module=None,
        lora_A=torch.zeros((64 * 8, 16)),  # 64 experts * rank 8
        lora_B=torch.zeros((16, 64 * 8)),
        rank=8,
    )
    moe_num_experts = {}
    header = _header(PREFIX, range(16))  # partial low-index shard

    n = _resolve_moe_num_experts_with_header(
        PREFIX, stats, moe_num_experts, header, SCHEME,
    )
    assert n == 64, f"fused-LoRA count 64 must survive a partial 16-expert shard (got {n})"


def test_no_per_expert_tensors_returns_none():
    # Header has no per-expert tensors for this prefix and nothing else resolves.
    stats = _Stats(module=None)
    moe_num_experts = {}
    header = {"model.layers.0.self_attn.q_proj.weight": object()}

    n = _resolve_moe_num_experts_with_header(
        PREFIX, stats, moe_num_experts, header, SCHEME,
    )
    assert n is None
