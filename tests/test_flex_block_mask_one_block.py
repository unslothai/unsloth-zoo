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

# torch 2.10's dynamic create_block_mask graph gave block counts of 2 for one-block lengths (gpt-oss
# illegal memory access); compiled_create_block_mask must match the eager builder there.
from __future__ import annotations

import pytest
import torch

NAMES = ("kv_num_blocks", "q_num_blocks", "full_kv_num_blocks", "full_q_num_blocks")


def _utils():
    from unsloth_zoo.flex_attention import utils
    if not utils.HAS_FLEX_ATTENTION:
        pytest.skip("no flex attention")
    return utils


def test_one_block_lengths_build_eagerly(monkeypatch):
    utils = _utils()
    monkeypatch.setattr(utils, "_ONE_BLOCK_MISCOMPILE", True, raising = False)
    BS = utils.FLEX_ATTENTION_BLOCK_SIZE
    assert not utils._compile_block_mask(4, 64, BS, BS)
    assert not utils._compile_block_mask(None, None, 1, 4096)
    assert utils._compile_block_mask(4, 64, BS + 1, BS + 1)
    assert utils._compile_block_mask(64, 64, 1, 1 << 20)
    # The cap counts the block-padded mask (8 x 64 x 128 x 32768).
    assert utils._compile_block_mask(8, 64, 1, 32768)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex attention block masks need CUDA")
def test_compiled_block_mask_matches_eager_after_going_dynamic():
    utils = _utils()
    from torch.nn.attention.flex_attention import create_block_mask as eager_create_block_mask
    torch._dynamo.reset()

    def causal(b, h, q, kv):
        return q >= kv

    # Two long lengths make the compiled builder dynamic; the short ones then hit the one-block case.
    for L in (512, 700, 100, 64, 128, 129, 300):
        got = utils.compiled_create_block_mask(causal, 4, 64, L, L, device = "cuda")
        ref = eager_create_block_mask(causal, 4, 64, L, L, device = "cuda")
        for name in NAMES:
            a, b = getattr(got, name), getattr(ref, name)
            if a is None and b is None:
                continue
            assert torch.equal(a, b), f"L={L} {name}: {a.flatten()[:4].tolist()} vs {b.flatten()[:4].tolist()}"


def test_fixed_torch_keeps_the_compiled_builder(monkeypatch):
    """torch 2.13+ does not miscompile one-block lengths: no dense O(M x N) mask there."""
    utils = _utils()
    monkeypatch.setattr(utils, "_ONE_BLOCK_MISCOMPILE", False, raising = False)
    BS = utils.FLEX_ATTENTION_BLOCK_SIZE
    assert utils._compile_block_mask(4, 64, BS, BS)
    assert utils._compile_block_mask(None, None, 1, 4096)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex attention block masks need CUDA")
def test_block_mask_inside_an_outer_compile_matches_eager():
    """A compiled caller must not trace the one-block build into its own (miscompiled) graph."""
    utils = _utils()
    from torch.nn.attention.flex_attention import create_block_mask as eager_create_block_mask
    torch._dynamo.reset()

    def causal(b, h, q, kv):
        return q >= kv

    def outer(x):
        m = utils.compiled_create_block_mask(causal, 4, 64, x.shape[0], x.shape[0], device = "cuda")
        return tuple(getattr(m, name) for name in NAMES)

    compiled = torch.compile(outer)  # automatic dynamic: the second length recompiles dynamic
    try:
        for L in (512, 700, 100, 64, 128, 129, 300):
            got = compiled(torch.empty(L, device = "cuda"))
            ref = eager_create_block_mask(causal, 4, 64, L, L, device = "cuda")
            for name, a in zip(NAMES, got):
                b = getattr(ref, name)
                if a is None and b is None:
                    continue
                assert torch.equal(a, b), f"L={L} {name}: {a.flatten()[:4].tolist()} vs {b.flatten()[:4].tolist()}"
    finally:
        torch._dynamo.reset()
