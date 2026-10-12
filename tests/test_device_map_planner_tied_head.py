# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""A tied output head is sized once on every transformers version.

A meta model built under ``init_empty_weights`` is not tied by identity, so
accelerate's ``compute_module_sizes`` (the path on transformers 4) used to
charge the head a second time. Real loads share the weight.

Meta-device models only: no GPU, no weights, no network.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config

import unsloth_zoo.device_map_planner as planner

_COMMON = dict(
    vocab_size = 1024,
    hidden_size = 64,
    intermediate_size = 256,
    num_hidden_layers = 8,
    num_attention_heads = 4,
    num_key_value_heads = 2,
)


def _config(cls, tied):
    extra = dict(head_dim = 16) if cls is Qwen3Config else {}
    config = cls(tie_word_embeddings = tied, **_COMMON, **extra)
    config.torch_dtype = torch.bfloat16
    return config


def _true_bytes(config):
    """Unique parameter + buffer bytes of a real (tied when configured) model."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.bfloat16)
    tensors = {id(t): t for t in list(model.parameters()) + list(model.buffers())}
    return sum(t.numel() * t.element_size() for t in tensors.values()), model


CASES = [(cls, tied) for cls in (Qwen3Config, LlamaConfig) for tied in (True, False)]


@pytest.mark.parametrize("cls,tied", CASES, ids = [f"{c.__name__}-tied{t}" for c, t in CASES])
def test_planned_size_matches_real_parameter_bytes(cls, tied):
    config = _config(cls, tied)
    true_total, real = _true_bytes(config)
    assert (real.lm_head.weight is real.model.embed_tokens.weight) == tied

    meta, quantizer, _ = planner.build_meta_model("org/unused", config = config, dtype = torch.bfloat16)
    sizes = planner._compute_module_sizes(meta, quantizer)
    head_bytes = _COMMON["vocab_size"] * _COMMON["hidden_size"] * 2
    assert sizes[""] == true_total
    assert sizes["model.embed_tokens"] == head_bytes
    assert sizes.get("lm_head", 0) == (0 if tied else head_bytes)


def _plan(model, budget, quantizer = None):
    try:
        plan = planner.plan_block_swap(
            model = model, hf_quantizer = quantizer,
            max_memory = {0: budget}, reserve_bytes = 0, safety_bytes = 0, headroom_bytes = 0,
        )
    except planner.DeviceMapInfeasible:
        return None
    return plan.layers, plan.swapped_bytes


@pytest.mark.parametrize("tied", [True, False])
def test_block_swap_plans_meta_like_real_model(tied):
    """The meta model must plan exactly like the loaded one, budget for budget."""
    config = _config(Qwen3Config, tied)
    true_total, real = _true_bytes(config)
    meta, quantizer, _ = planner.build_meta_model("org/unused", config = config, dtype = torch.bfloat16)
    step = max(1, true_total // 64)
    plans = []
    for budget in range(step, 2 * true_total, step):
        expected = _plan(real, budget)
        assert _plan(meta, budget, quantizer) == expected, budget
        plans.append(expected)
    # The sweep crossed from infeasible through swapping to fully resident.
    assert None in plans and (0, 0) in plans
    assert any(p is not None and p[0] > 0 for p in plans)


def test_head_already_tied_by_identity_is_not_dropped_twice():
    config = _config(Qwen3Config, True)
    true_total, real = _true_bytes(config)
    sizes = planner._compute_module_sizes(real)
    assert sizes[""] == true_total
    assert sizes["model.embed_tokens"] == _COMMON["vocab_size"] * _COMMON["hidden_size"] * 2


@pytest.mark.parametrize("tied", [True, False])
def test_multimodal_tied_head_sized_from_text_config(tied):
    """A VLM declares the tie on its text config; the head is still counted once."""
    from transformers import AutoModelForImageTextToText, Gemma3Config
    config = Gemma3Config(
        text_config = dict(
            vocab_size = 512, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
            num_attention_heads = 2, num_key_value_heads = 1, head_dim = 32,
        ),
        vision_config = dict(
            hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2, num_attention_heads = 2,
            image_size = 32, patch_size = 16,
        ),
        mm_tokens_per_image = 4,
        tie_word_embeddings = tied,
    )
    config.text_config.tie_word_embeddings = tied
    real = AutoModelForImageTextToText.from_config(config, torch_dtype = torch.bfloat16)
    assert (real.get_output_embeddings().weight is real.get_input_embeddings().weight) == tied
    tensors = {id(t): t for t in list(real.parameters()) + list(real.buffers())}
    true_total = sum(t.numel() * t.element_size() for t in tensors.values())

    meta, quantizer, _ = planner.build_meta_model("org/unused", config = config, dtype = torch.bfloat16)
    assert planner._compute_module_sizes(meta, quantizer)[""] == true_total


def test_outer_config_untie_keeps_the_head_charged():
    """Outer tie_word_embeddings = False with a tied text config (Qwen2.5-Omni Thinker shape): the head
    stays charged, so the plan can only overestimate, never drop a resident head."""
    from transformers import AutoModelForImageTextToText, Gemma3Config
    config = Gemma3Config(
        text_config = dict(
            vocab_size = 512, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
            num_attention_heads = 2, num_key_value_heads = 1, head_dim = 32,
        ),
        vision_config = dict(
            hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2, num_attention_heads = 2,
            image_size = 32, patch_size = 16,
        ),
        mm_tokens_per_image = 4,
        tie_word_embeddings = False,
    )
    config.text_config.tie_word_embeddings = True
    real = AutoModelForImageTextToText.from_config(config, torch_dtype = torch.bfloat16)
    head = real.get_output_embeddings().weight
    tensors = {id(t): t for t in list(real.parameters()) + list(real.buffers())}
    unique_total = sum(t.numel() * t.element_size() for t in tensors.values())
    head_bytes = head.numel() * head.element_size()
    charged_total = unique_total + (head_bytes if head is real.get_input_embeddings().weight else 0)

    meta, quantizer, _ = planner.build_meta_model("org/unused", config = config, dtype = torch.bfloat16)
    assert planner._compute_module_sizes(meta, quantizer)[""] == charged_total
