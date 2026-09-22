# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""A block appended to the decoder list but missing from ``_no_split_modules`` stays whole.

Remote MoE ports such as inclusionAI/Ling-2.6-flash keep a multi-token-prediction
layer at ``model.layers[-1]`` and declare only the decoder layer as atomic. The
planner then spread that layer's attention, norms and experts over several cards,
and because accelerate only moves the inputs of a placement unit, the causal mask
reached the attention on another card and the first forward failed with
"Expected all tensors to be on the same device".
"""

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import plan_device_map, resolve_no_split_classes

H = 64
KiB = 1024


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(H, H, bias = False)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias = False)
        self.o_proj = nn.Linear(H, H, bias = False)


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(H, H, bias = False) for _ in range(4)])


class _MTPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enorm = nn.LayerNorm(H)
        self.eh_proj = nn.Linear(2 * H, H, bias = False)
        self.attention = _Attention()
        self.mlp = _Experts()


class _WithMTP(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, vocab = 512, layers = 4, mtp = True):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, H)
        blocks = [_Block() for _ in range(layers)]
        self.layers = nn.ModuleList(blocks + ([_MTPLayer()] if mtp else []))
        self.norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, vocab, bias = False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def _meta(**kw):
    with torch.device("meta"):
        return _WithMTP(**kw)


def _plan(model):
    # float32: embedding 128 KiB, each block 16 KiB, the MTP layer ~128 KiB, head 128 KiB.
    # Card 0 holds the embedding and the four blocks with 48 KiB to spare, so an
    # in-order walk that may descend into the MTP layer puts its first children there.
    return plan_device_map(
        model,
        max_memory = {0: 240 * KiB, 1: 300 * KiB},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
        prefer_head_device = 1,
    )


def test_the_undeclared_sibling_block_is_detected():
    assert resolve_no_split_classes(_meta()) == ["_Block", "_MTPLayer"]


def test_the_mtp_layer_is_placed_on_one_card():
    plan = _plan(_meta())
    devices = {v for k, v in plan.device_map.items() if k == "layers.4" or k.startswith("layers.4.")}
    assert devices and len(devices) == 1, plan.device_map
    assert "layers.4" in plan.device_map


def test_models_without_an_extra_block_plan_as_before():
    model = _meta(mtp = False)
    assert resolve_no_split_classes(model) == ["_Block"]


def test_an_empty_declaration_still_means_nothing_is_atomic():
    model = _meta()
    model._no_split_modules = []
    assert resolve_no_split_classes(model) == []


def test_an_explicit_override_is_left_alone():
    plan = plan_device_map(
        _meta(),
        max_memory = {0: 240 * KiB, 1: 300 * KiB},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
        prefer_head_device = 1,
        no_split_module_classes = ["_Block"],
    )
    assert "_MTPLayer" not in plan.no_split_module_classes
