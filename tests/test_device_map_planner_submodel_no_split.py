# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Blocks declared on a sub-model, not the root (transformers 4.x composites like Mistral3), stay atomic (unsloth#2864)."""

import pytest
import torch.nn as nn

from unsloth_zoo.device_map_planner import plan_device_map, resolve_no_split_classes

H = 64
KiB = 1024


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias = False)
        self.mlp = nn.Linear(H, H, bias = False)


class _TextModel(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, vocab, layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, H)
        self.layers = nn.ModuleList([_Block() for _ in range(layers)])
        self.norm = nn.LayerNorm(H)


class _Composite(nn.Module):
    # What transformers 4.x leaves on a Mistral3ForConditionalGeneration instance.
    _no_split_modules = []

    def __init__(self, vocab = 512, layers = 6):
        super().__init__()
        self.language_model = _TextModel(vocab, layers)
        self.lm_head = nn.Linear(H, vocab, bias = False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.language_model.embed_tokens


def test_sub_model_blocks_are_collected():
    assert resolve_no_split_classes(_Composite()) == ["_Block"]


def test_root_declaring_nothing_with_plain_children_stays_empty():
    model = _Composite()
    model.language_model._no_split_modules = []
    assert resolve_no_split_classes(model) == []


def test_split_lands_on_block_boundaries():
    model = _Composite()
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = int(total * 0.6) + 64 * KiB
    plan = plan_device_map(
        model,
        max_memory = {0: budget, 1: budget},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
        prefer_head_device = 1,
    )
    devices = set(plan.device_map.values())
    assert devices == {0, 1}
    assert not any(".layers." in k and k.count(".") > 2 for k in plan.device_map), plan.device_map


def test_mistral3_keeps_decoder_layers_whole():
    transformers = pytest.importorskip("transformers")
    from accelerate import init_empty_weights
    cfg = transformers.Mistral3Config(
        text_config = dict(model_type = "mistral", vocab_size = 128, hidden_size = 32,
                           intermediate_size = 64, num_hidden_layers = 2,
                           num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8),
        vision_config = dict(model_type = "pixtral", hidden_size = 32, intermediate_size = 64,
                             num_hidden_layers = 2, num_attention_heads = 4, image_size = 32,
                             patch_size = 16, head_dim = 8),
    )
    with init_empty_weights():
        model = transformers.Mistral3ForConditionalGeneration(cfg)
    classes = resolve_no_split_classes(model)
    assert "MistralDecoderLayer" in classes and "PixtralAttentionLayer" in classes
