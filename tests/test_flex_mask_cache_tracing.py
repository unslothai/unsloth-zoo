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

"""A cold per-device mask cache must not mutate Python state in a Flex subgraph."""

import ast
import functools
import inspect
from pathlib import Path
from textwrap import dedent

import pytest
import torch

_flex = pytest.importorskip("torch.nn.attention.flex_attention")
create_block_mask, flex_attention = _flex.create_block_mask, _flex.flex_attention


MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"


def _namespace():
    # Execute the actual helpers and nested factories without importing unrelated GPU patches.
    source = MISC.read_text(encoding = "utf-8")
    namespace = {"torch": torch, "functools": functools}
    names = {"_warm_mask_device_cache", "_warm_mask_mod_devices", "_patch_block_mask_to",
             "padding_mask_wrapper", "packed_sequence_mask_wrapper"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(dedent(ast.get_source_segment(source, node)), namespace)
    missing = names - namespace.keys()
    assert not missing, missing
    return namespace


def _factory(kind):
    return _namespace()[kind + "_mask_wrapper"]


def _mask(kind):
    if kind == "padding":
        return torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype = torch.bool)
    return torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])


@pytest.mark.parametrize("kind", ["padding", "packed_sequence"])
def test_cold_mask_cache_is_traceable(kind):
    source = _mask(kind)
    mask = _factory(kind)(source)
    cache = inspect.getclosurevars(mask).nonlocals["_cache"]

    def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
        return (q_idx >= kv_idx) & mask(batch_idx, head_idx, q_idx, kv_idx)

    block_mask = create_block_mask(causal_mask, 1, None, 8, 8, device = "cpu")
    # Cold miss, as on a model-parallel device the BlockMask was not built on.
    cache.clear()
    torch.manual_seed(0)
    # Forward only: newer torch has no FlexAttention backward on CPU.
    q, k, v = [torch.randn(1, 2, 8, 16) for _ in range(3)]
    out = torch.compile(flex_attention, backend = "eager", fullgraph = True)(
        q, k, v, block_mask = block_mask,
    )

    indices = torch.arange(8)
    allowed = indices[:, None] >= indices[None, :]
    if kind == "padding":
        allowed = allowed & source[0, None, :]
    else:
        allowed = allowed & (source[0, :, None] == source[0, None, :])
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = allowed)
    torch.testing.assert_close(out, expected)
    assert not cache


@pytest.mark.parametrize("kind", ["padding", "packed_sequence"])
def test_eager_mask_still_populates_the_device_cache(kind):
    source = _mask(kind)
    mask = _factory(kind)(source)
    cache = inspect.getclosurevars(mask).nonlocals["_cache"]
    cache.clear()
    zero = torch.zeros((), dtype = torch.long)
    assert mask(zero, zero, zero, zero)
    assert source.device in cache


@pytest.mark.parametrize("kind", ["padding", "packed_sequence"])
def test_block_mask_move_warms_the_target_device(kind, monkeypatch):
    # Inductor cannot lower the cold copy, so moving the BlockMask must make it.
    namespace = _namespace()
    monkeypatch.setattr(_flex.BlockMask, "to", _flex.BlockMask.to)
    namespace["_patch_block_mask_to"]()
    namespace["_patch_block_mask_to"]()
    source = _mask(kind)
    mask = namespace[kind + "_mask_wrapper"](source)
    cache = inspect.getclosurevars(mask).nonlocals["_cache"]
    masks = (lambda b, h, q, kv: q >= kv, mask)

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        return masks[0](batch_idx, head_idx, q_idx, kv_idx) & masks[1](batch_idx, head_idx, q_idx, kv_idx)

    def offset_mask(batch_idx, head_idx, q_idx, kv_idx):
        return and_mask(batch_idx, head_idx, q_idx, kv_idx)

    block_mask = create_block_mask(offset_mask, 1, None, 8, 8, device = "cpu")
    assert list(cache) == [torch.device("cpu")]
    moved = block_mask.to("meta")
    assert moved.kv_num_blocks.is_meta
    assert cache[torch.device("meta")].is_meta
    assert cache[torch.device("cpu")] is source
    block_mask.to("cpu")
    assert len(cache) == 2
