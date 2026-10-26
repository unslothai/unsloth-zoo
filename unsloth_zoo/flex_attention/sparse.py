# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared switches and engagement counters for the sparse-attention fast paths
(DeepSeek-V3.2 / GLM-MoE-DSA / MiniMax-M3 / DeepSeek-V4)."""

import os

import torch

__all__ = [
    "SPARSE_ATTN_STATS",
    "sparse_skip_full_enabled",
    "record_sparse_attn",
    "reset_sparse_attn_stats",
]

# {kind: {model_type: count}}; not updated while torch.compile traces (no guard on the int).
SPARSE_ATTN_STATS = {}


def sparse_skip_full_enabled():
    """UNSLOTH_SPARSE_ATTN_SKIP_FULL=0 restores the indexer when every key is selected anyway."""
    return os.environ.get("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "1") != "0"


def record_sparse_attn(kind, model_type):
    if torch.compiler.is_compiling():
        return
    counts = SPARSE_ATTN_STATS.setdefault(kind, {})
    counts[model_type] = counts.get(model_type, 0) + 1


def reset_sparse_attn_stats():
    SPARSE_ATTN_STATS.clear()
