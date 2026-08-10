# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Head-aware device map planner: introspection and arithmetic.

These tests use meta-device models only, so they need no GPU and no weights.
They pin the parts that must stay model agnostic: finding the output head,
finding the decoder block class, honouring tied embeddings, the headroom
arithmetic, and refusing rather than silently spilling when a request cannot
fit.
"""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import (
    DeviceMapInfeasible,
    logit_headroom_bytes,
    plan_device_map,
    resolve_head_width,
    resolve_no_split_classes,
    resolve_output_head,
)

_GiB = 1024 ** 3


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden, bias = False)


class _Tiny(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, hidden = 64, vocab = 512, layers = 8, tie = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias = False)
        if tie:
            self.lm_head.weight = self.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def _meta(**kw):
    with torch.device("meta"):
        return _Tiny(**kw)


def test_finds_head_and_block_class():
    model = _meta()
    name, head = resolve_output_head(model)
    assert name == "lm_head"
    assert head is model.lm_head
    assert resolve_head_width(model, head) == 512
    assert "_Block" in resolve_no_split_classes(model)


def test_headroom_grows_with_rows_and_vocab():
    small = logit_headroom_bytes(1000, 128, softcapped = False)
    wide = logit_headroom_bytes(200000, 128, softcapped = False)
    assert wide > small
    more_rows = logit_headroom_bytes(200000, 256, softcapped = False)
    assert more_rows > wide


def test_headroom_retained_term_scales_with_total_rows():
    # The retained term is what makes the backward pass expensive: it depends on
    # the total number of rows, not on the chunk size, so chunking alone cannot
    # bound it.
    a = logit_headroom_bytes(200000, 128, retained_rows = 1024)
    b = logit_headroom_bytes(200000, 128, retained_rows = 4096)
    assert b > a


def test_single_device_returns_none_shaped_plan():
    model = _meta()
    plan = plan_device_map(model, max_memory = {0: "8GiB"})
    assert plan is None or set(plan.device_map.values()) == {0}


def test_two_devices_place_every_module_on_a_gpu():
    model = _meta(layers = 8)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert plan is not None
    assert set(plan.device_map.values()) <= {0, 1}
    assert all(isinstance(v, int) for v in plan.device_map.values()), "no cpu or disk spill"
    assert plan.head_device in (0, 1)


def test_tied_embedding_lands_with_the_head():
    model = _meta(tie = True)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert plan is not None
    assert plan.device_map["embed_tokens"] == plan.head_device


def test_refuses_instead_of_spilling_when_it_cannot_fit():
    model = _meta(hidden = 256, vocab = 4096, layers = 16)
    with pytest.raises(DeviceMapInfeasible):
        plan_device_map(
            model,
            max_memory = {0: "1MiB", 1: "1MiB"},
            headroom_bytes = 8 * _GiB,
        )
