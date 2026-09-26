# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vision towers stay on one card: their forward adds pos_embed.weight to patch_embed output."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from accelerate import init_empty_weights

import unsloth_zoo.device_map_planner as planner


def _qwen3_5_moe(layers = 8, depth = 4, out_hidden_size = 64):
    if not hasattr(transformers, "Qwen3_5MoeConfig"):
        pytest.skip("transformers has no Qwen3.5 MoE")
    config = transformers.Qwen3_5MoeConfig(
        text_config = dict(
            vocab_size = 512, hidden_size = 64, num_hidden_layers = layers,
            num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
            linear_key_head_dim = 16, linear_value_head_dim = 16,
            linear_num_key_heads = 2, linear_num_value_heads = 4,
            moe_intermediate_size = 32, shared_expert_intermediate_size = 32,
            num_experts = 4, num_experts_per_tok = 2,
            layer_types = ["linear_attention", "full_attention"] * (layers // 2),
        ),
        vision_config = dict(
            depth = depth, hidden_size = 64, intermediate_size = 128, num_heads = 4,
            out_hidden_size = out_hidden_size, num_position_embeddings = 64, patch_size = 4,
        ),
    )
    return _build(transformers.Qwen3_5MoeForConditionalGeneration, config)


def _qwen3_vl():
    config = transformers.Qwen3VLConfig(
        text_config = dict(
            vocab_size = 512, hidden_size = 64, intermediate_size = 128,
            num_hidden_layers = 8, num_attention_heads = 4, num_key_value_heads = 2,
            head_dim = 16, rope_scaling = dict(rope_type = "default", mrope_section = [4, 2, 2]),
        ),
        vision_config = dict(
            depth = 4, hidden_size = 64, intermediate_size = 128, num_heads = 4,
            out_hidden_size = 64, num_position_embeddings = 64, patch_size = 4,
            deepstack_visual_indexes = [1, 2],
        ),
    )
    return _build(transformers.Qwen3VLForConditionalGeneration, config)


def _gemma3():
    config = transformers.Gemma3Config(
        text_config = dict(
            vocab_size = 512, hidden_size = 64, intermediate_size = 128,
            num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1,
            head_dim = 32,
        ),
        vision_config = dict(
            hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
            num_attention_heads = 2, image_size = 32, patch_size = 16,
        ),
        mm_tokens_per_image = 4,
    )
    return _build(transformers.Gemma3ForConditionalGeneration, config)


def _build(cls, config):
    config.dtype = torch.bfloat16
    with init_empty_weights():
        model = cls._from_config(config)
    return model.eval()


def _plan(model, fraction, **kw):
    total = planner._compute_module_sizes(model)[""]
    budget = int(total * fraction)
    return planner.plan_device_map(
        model, max_memory = {0: budget, 1: budget}, headroom_bytes = 0, **kw
    )


def _devices(plan, prefix):
    return {d for k, d in plan.device_map.items() if k == prefix or k.startswith(prefix + ".")}


@pytest.mark.parametrize("build, tower", [
    (_qwen3_5_moe, "model.visual"),
    (_qwen3_vl, "model.visual"),
    (_gemma3, "model.vision_tower"),
])
def test_only_the_non_text_tower_is_detected(build, tower):
    assert planner._sub_model_towers(build()) == [tower]


@pytest.mark.parametrize("build", [_qwen3_5_moe, _qwen3_vl])
def test_the_vision_tower_lands_on_one_device(build):
    model = build()
    plan = _plan(model, 0.6)
    assert len(_devices(plan, "model.visual")) == 1, plan.device_map
    assert "model.visual" in plan.device_map, plan.device_map
    assert any("placed whole" in n for n in plan.notes)
    if build is _qwen3_5_moe:
        assert _devices(plan, "model.language_model.layers") == {0, 1}


def test_a_tower_larger_than_a_card_keeps_its_embeddings_with_its_first_block():
    model = _qwen3_5_moe(layers = 2, depth = 12)
    sizes = planner._compute_module_sizes(model)
    budget = int(sizes[""] * 0.55)
    assert sizes["model.visual"] > budget
    plan = _plan(model, 0.55)
    loose = {
        plan.device_map[k] for k in (
            "model.visual.patch_embed.proj", "model.visual.pos_embed",
            "model.visual.rotary_pos_emb", "model.visual.blocks.0",
        )
    }
    assert len(loose) == 1, plan.device_map
    assert _devices(plan, "model.visual.blocks") == {0, 1}
    assert any("split at its blocks" in n for n in plan.notes)


def test_a_tower_that_cannot_be_kept_together_still_plans_as_before():
    # Loose parts plus one block exceed a card; each unit alone fits.
    model = _qwen3_5_moe(layers = 4, depth = 2, out_hidden_size = 1024)
    plan = _plan(model, 0.52)
    old = _plan(model, 0.52, no_split_module_classes = planner.resolve_no_split_classes(model))
    assert plan.device_map == old.device_map
    assert any("split per unit" in n for n in plan.notes)


def test_an_explicit_no_split_override_keeps_the_per_unit_split():
    model = _qwen3_5_moe()
    plan = _plan(model, 0.6, no_split_module_classes = planner.resolve_no_split_classes(model))
    assert _devices(plan, "model.visual") == {0, 1}
    assert not any("sub-model towers" in n for n in plan.notes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU next to the CPU")
@pytest.mark.parametrize("inputs_on", ["cpu", "cuda:0"])
def test_a_whole_tower_runs_with_inputs_on_another_device(inputs_on):
    from accelerate import dispatch_model
    config = _qwen3_vl().config
    config.dtype = torch.float32
    model = transformers.Qwen3VLForConditionalGeneration._from_config(config).eval()
    plan = _plan(model, 0.6)
    device_map = {k: (0 if d == 0 else "cpu") for k, d in plan.device_map.items()}
    model = dispatch_model(model, device_map = device_map, main_device = "cpu")
    grid = torch.tensor([[1, 4, 4]], device = inputs_on)
    with torch.no_grad():
        model.model.visual(torch.randn(16, 96, device = inputs_on), grid_thw = grid)
