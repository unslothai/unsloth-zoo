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

"""Generic bnb 4-bit experts route stays out of compiled regions except Unsloth's per-layer block compile
(UNSLOTH_DECODE_COMPILE set while tracing), avoiding a graph break per MoE layer."""

import pytest
import torch

mb = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
from unsloth_zoo.temporary_patches import moe_utils as mu
from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE


def _backend(self, hidden_states, top_k_index, top_k_weights):
    return hidden_states * top_k_weights.sum(-1, keepdim = True) + top_k_index.sum()


def _resolver_with_file_checks():
    import os
    os.path.normpath("/tmp/./x")  # what the real resolver does; Dynamo cannot trace it
    return _backend


def _breaks(in_block):
    def caller(x, idx, w):
        return mb._forward_generic_experts_eagerly(_resolver_with_file_checks, None, x * 2, idx, w) + 1

    torch._dynamo.reset()
    previous_flag, previous_cache = UNSLOTH_DECODE_COMPILE[0], mu._CACHED_FORWARD_MOE_BACKEND
    UNSLOTH_DECODE_COMPILE[0], mu._CACHED_FORWARD_MOE_BACKEND = in_block, _backend
    try:
        x, idx, w = torch.randn(4, 8), torch.zeros(4, 2, dtype = torch.long), torch.rand(4, 2)
        explain = torch._dynamo.explain(caller)(x, idx, w)
        expected = _backend(None, x * 2, idx, w) + 1
        torch.testing.assert_close(torch.compile(caller, backend = "eager")(x, idx, w), expected)
        torch.testing.assert_close(caller(x, idx, w), expected)
    finally:
        UNSLOTH_DECODE_COMPILE[0], mu._CACHED_FORWARD_MOE_BACKEND = previous_flag, previous_cache
        torch._dynamo.reset()
    return explain.graph_break_count


def test_traced_inside_block_compile():
    assert _breaks(in_block = True) == 0


def test_still_disabled_outside_block_compile():
    assert _breaks(in_block = False) >= 1
