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

# compiled_create_block_mask must match the eager builder when a sequence fits one sparse block.
# torch 2.10's dynamic create_block_mask graph returned kv_num_blocks / q_num_blocks = 2 there
# (an unmasked R0_BLOCK = 2 reduction over one block, reading past the buffer), which crashed
# gpt-oss LoRA training with an illegal memory access on the first batch of <= 128 tokens.
from __future__ import annotations

import pytest
import torch

NAMES = ("kv_num_blocks", "q_num_blocks", "full_kv_num_blocks", "full_q_num_blocks")


def _utils():
    from unsloth_zoo.flex_attention import utils
    if not utils.HAS_FLEX_ATTENTION:
        pytest.skip("no flex attention")
    return utils


def test_one_block_lengths_build_eagerly():
    utils = _utils()
    BS = utils.FLEX_ATTENTION_BLOCK_SIZE
    assert not utils._compile_block_mask(4, 64, BS, BS)
    assert not utils._compile_block_mask(None, None, 1, 4096)
    assert utils._compile_block_mask(4, 64, BS + 1, BS + 1)
    # A one-block query against a huge key length keeps the compiled (O(blocks) memory) builder.
    assert utils._compile_block_mask(64, 64, 1, 1 << 20)
    # The cap counts the block-padded mask: 8 x 64 x 128 x 32768 elements, not 8 x 64 x 1 x 32768.
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
