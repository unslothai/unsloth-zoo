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
    "sparse_flex_enabled",
    "sparse_indexer_chunked_enabled",
    "record_sparse_attn",
    "reset_sparse_attn_stats",
    "additive_bias_present",
]

# {kind: {model_type: count}}; not updated while torch.compile traces (no guard on the int).
SPARSE_ATTN_STATS = {}
# (device, dtype, shape) -> whether the last eager additive mask of that shape carried finite biases.
_BIAS_SEEN = {}


def sparse_skip_full_enabled():
    """UNSLOTH_SPARSE_ATTN_SKIP_FULL=0 restores the indexer when every key is selected anyway."""
    return os.environ.get("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "1") != "0"


def sparse_flex_enabled():
    """UNSLOTH_SPARSE_ATTN_FLEX=0 keeps upstream dense attention above the selection budget."""
    return os.environ.get("UNSLOTH_SPARSE_ATTN_FLEX", "1") != "0"


def sparse_indexer_chunked_enabled():
    """UNSLOTH_SPARSE_INDEXER_CHUNKED=0 keeps upstream's unchunked indexer scoring above the budget."""
    return os.environ.get("UNSLOTH_SPARSE_INDEXER_CHUNKED", "1") != "0"


def record_sparse_attn(kind, model_type):
    if torch.compiler.is_compiling():
        return
    counts = SPARSE_ATTN_STATS.setdefault(kind, {})
    counts[model_type] = counts.get(model_type, 0) + 1


def reset_sparse_attn_stats():
    SPARSE_ATTN_STATS.clear()


def additive_bias_present(mask, visible):
    """Whether an additive float mask holds finite nonzero entries (not just 0 / masked), which flex adds
    through a score_mod (5-15% slower). Eager this is a host sync. Under CUDA graph capture a sync invalidates
    the capture, and flipping the answer would need a kernel compiled during capture, so the eager warm-up's
    answer for the same shape is reused: a bias in any warm-up call of that shape keeps the bias; with none,
    no bias is asserted on device at every replay (a finite bias raises instead of being dropped); a shape
    never run eagerly keeps the bias."""
    key = (mask.device, mask.dtype, tuple(mask.shape))
    if mask.is_cuda and torch.cuda.is_current_stream_capturing():
        if _BIAS_SEEN.get(key) is False:
            torch._assert_async(
                ~(visible & (mask != 0)).any(), "Unsloth: finite attention-mask bias in a CUDA graph captured without one",
            )
            return False
        return True
    found = bool((visible & (mask != 0)).any())
    # Sticky: one same-shape call with a bias keeps every captured call of that shape on the bias kernel.
    _BIAS_SEEN[key] = found or _BIAS_SEEN.get(key, False)
    return found
