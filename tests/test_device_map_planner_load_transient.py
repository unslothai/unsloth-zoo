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

"""The device map planner keeps room for the tensors transformers 5 builds while
it merges checkpoint tensors into one parameter (per-expert weights into an
expert stack, gate and up into gate_up).

transformers materialises every source tensor on the target card and stacks
them there, so a card needs one merged tensor above its loaded weights, and in
practice two to five times that because the allocator cannot reuse the holes
the sources leave. MiniMax-M3 16-bit ran out of memory mid-load on exactly this
while its steady-state plan fitted. Meta-device models only, no GPU needed.
"""
import importlib.util
import sys
import types

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import (
    _compute_module_sizes,
    _load_transient_by_unit,
    _merged_parameter_patterns,
    _split_units,
    plan_device_map,
    resolve_no_split_classes,
)

_HAS_CONVERSION_MAPPING = importlib.util.find_spec("transformers.conversion_mapping") is not None
needs_conversion_mapping = pytest.mark.skipif(
    not _HAS_CONVERSION_MAPPING,
    reason = "transformers 4.x loads checkpoints without merging converters",
)

_HIDDEN, _INTER, _EXPERTS, _LAYERS = 64, 128, 8, 6
# One layer's gate_up stack in bf16: experts x (gate + up) x intermediate x hidden.
_GATE_UP_BYTES = _EXPERTS * 2 * _INTER * _HIDDEN * 2


@pytest.fixture(autouse = True)
def _default_allocator(monkeypatch):
    # unsloth_zoo turns expandable segments on at import; pin the multiple.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising = False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising = False)


def _meta_mixtral(layers = _LAYERS):
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        hidden_size = _HIDDEN, intermediate_size = _INTER, num_local_experts = _EXPERTS,
        num_experts_per_tok = 2, num_hidden_layers = layers, num_attention_heads = 4,
        num_key_value_heads = 2, vocab_size = 256, max_position_embeddings = 64,
    )
    config.torch_dtype = torch.bfloat16
    config.dtype = torch.bfloat16
    with torch.device("meta"):
        return MixtralForCausalLM(config).to(torch.bfloat16)


def _meta_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size = _HIDDEN, intermediate_size = _INTER, num_hidden_layers = _LAYERS,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256,
        max_position_embeddings = 64,
    )
    config.torch_dtype = torch.bfloat16
    config.dtype = torch.bfloat16
    with torch.device("meta"):
        return LlamaForCausalLM(config).to(torch.bfloat16)


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden, bias = False)


class _Tiny(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, hidden = 64, vocab = 512, layers = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def _meta_tiny():
    with torch.device("meta"):
        return _Tiny()


def _units(model):
    sizes = _compute_module_sizes(model)
    return _split_units(model, resolve_no_split_classes(model), sizes), sizes[""]


def _same_placement(a, b):
    return (
        a.device_map == b.device_map
        and a.weight_bytes == b.weight_bytes
        and a.activation_reserve_by_device == b.activation_reserve_by_device
        and a.head_device == b.head_device
    )


def _layer_devices(plan):
    return {d for name, d in plan.device_map.items() if ".layers." in name}


def test_merged_expert_stacks_are_sized_per_layer():
    model = _meta_mixtral()
    units, _ = _units(model)
    transient = _load_transient_by_unit(model, units)
    if not _HAS_CONVERSION_MAPPING:
        # transformers 4.x loads the experts as they are stored: nothing to reserve.
        assert transient == {}
        return
    layers = {u: n for u, n in transient.items() if ".layers." in u}
    assert len(layers) == _LAYERS
    assert set(layers.values()) == {_GATE_UP_BYTES}


@pytest.mark.parametrize("build", [_meta_tiny, _meta_llama])
def test_models_that_merge_nothing_plan_exactly_as_before(build):
    model = build()
    units, total = _units(model)
    assert _load_transient_by_unit(model, units) == {}
    for budgets in (
        {0: total, 1: total},
        {0: total // 2 + 4096, 1: total // 2 + 4096},
        {0: total // 3, 1: total, 2: total // 3},
    ):
        before = plan_device_map(
            model, max_memory = budgets, headroom_bytes = 0, reserve_load_transient = False,
        )
        after = plan_device_map(model, max_memory = budgets, headroom_bytes = 0)
        assert _same_placement(before, after)
        assert after.notes == before.notes
        assert after.load_transient_by_device == {}


@needs_conversion_mapping
def test_a_merging_model_with_room_to_spare_keeps_its_placement():
    model = _meta_mixtral()
    _, total = _units(model)
    budgets = {0: total, 1: total}
    before = plan_device_map(
            model, max_memory = budgets, headroom_bytes = 0, reserve_load_transient = False,
        )
    after = plan_device_map(model, max_memory = budgets, headroom_bytes = 0)
    assert _same_placement(before, after)
    assert any(note.startswith("load transient: 3x") for note in after.notes)


@needs_conversion_mapping
def test_cards_holding_merged_experts_keep_room_to_merge():
    # With no activation reserve the in-order walk fills cuda:0 to the brim, and
    # the first expert stack built there has nowhere to go.
    model = _meta_mixtral()
    units, total = _units(model)
    layer = max(n for u, n in units if ".layers." in u)
    # Room for 3x on both cards plus a layer of packing slack.
    per_card = total // 2 + 3 * _GATE_UP_BYTES + layer
    budgets = {0: per_card, 1: per_card}
    plan = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
    )
    free = plan.free_bytes
    for d in _layer_devices(plan):
        assert free[d] >= 3 * _GATE_UP_BYTES, (d, free[d], 3 * _GATE_UP_BYTES)
    assert plan.load_transient_by_device == {d: 3 * _GATE_UP_BYTES for d in _layer_devices(plan)}

    # The same request without the load transient packs cuda:0 past that point,
    # which is the map that ran out of memory while loading.
    unguarded = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        reserve_load_transient = False,
    )
    assert unguarded.free_bytes[0] < 3 * _GATE_UP_BYTES


@needs_conversion_mapping
def test_expandable_segments_ask_for_more_room_first(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    model = _meta_mixtral()
    _, total = _units(model)
    plan = plan_device_map(
        model, max_memory = {0: 2 * total, 1: 2 * total}, headroom_bytes = 0,
        activation_reserve_bytes = 0,
    )
    assert any(note.startswith("load transient: 5x") for note in plan.notes)
    for d in _layer_devices(plan):
        assert plan.free_bytes[d] >= 5 * _GATE_UP_BYTES


@needs_conversion_mapping
def test_the_largest_multiple_that_fits_is_kept(monkeypatch):
    # Slack for 3x on both cards but not 5x.
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    model = _meta_mixtral()
    _, total = _units(model)
    per_card = (total + 8 * _GATE_UP_BYTES) // 2
    plan = plan_device_map(
        model, max_memory = {0: per_card, 1: per_card}, headroom_bytes = 0,
        activation_reserve_bytes = 0,
    )
    assert any(note.startswith("load transient: 3x") for note in plan.notes)
    for d in _layer_devices(plan):
        assert plan.free_bytes[d] >= 3 * _GATE_UP_BYTES


@needs_conversion_mapping
def test_a_room_no_placement_can_keep_leaves_the_old_plan_and_says_so():
    model = _meta_mixtral()
    _, total = _units(model)
    # Room for the weights and half a merged tensor per card, no more.
    budgets = {0: total // 2 + _GATE_UP_BYTES // 2, 1: total // 2 + _GATE_UP_BYTES // 2}
    before = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        reserve_load_transient = False,
    )
    after = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
    )
    assert _same_placement(before, after)
    assert after.load_transient_by_device == {}
    assert any("loading may run out of memory" in note for note in after.notes)


def test_only_converters_that_merge_on_the_card_count(monkeypatch):
    class MergeModulelist:
        pass

    class Concatenate:
        pass

    class Transpose:
        pass

    def converter(targets, operations, force_cpu = False):
        return types.SimpleNamespace(
            target_patterns = targets, operations = operations, force_cpu = force_cpu,
        )

    conversions = [
        converter([".experts.gate_up_proj"], [MergeModulelist(), Concatenate()]),
        converter([r"\1.gate_up_proj.weight"], [Concatenate()]),
        converter([".ngram_embedding.weight"], [Concatenate()], force_cpu = True),
        converter([".o_proj.weight"], [Transpose()]),
        converter(["renamed"], []),
    ]
    fake = types.ModuleType("transformers.conversion_mapping")
    fake.get_model_conversion_mapping = lambda model, key_mapping = None, hf_quantizer = None: conversions
    monkeypatch.setitem(sys.modules, "transformers.conversion_mapping", fake)
    patterns = _merged_parameter_patterns(nn.Linear(1, 1))
    names = [
        "model.layers.3.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.ngram_embedding.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]
    matched = [n for n in names if any(p.search(n) for p in patterns)]
    assert matched == names[:2]


def test_a_transformers_without_conversion_mapping_reserves_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers.conversion_mapping", None)
    model = _meta_tiny()
    units, _ = _units(model)
    assert _merged_parameter_patterns(model) == []
    assert _load_transient_by_unit(model, units) == {}


def test_a_floor_left_on_a_card_the_merge_moved_off_does_not_rule_out_a_fit(monkeypatch):
    # Floors only grew while re-planning, so one kept on a card whose merging layer had
    # moved elsewhere made 3x look infeasible here and the plan fell back to 1x.
    import unsloth_zoo.device_map_planner as planner

    with torch.device("meta"):
        model = _Tiny(layers = 10)
    layer = 64 * 64 * 4
    monkeypatch.setattr(
        planner, "_load_transient_by_unit",
        lambda model, units, hf_quantizer = None: {u: 8192 for u, _ in units if u == "layers.6"},
    )
    budgets = {0: 8 * layer, 1: 8 * layer, 2: 3 * layer, 3: 9 * layer}
    plan = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        free_space_policy = "head_max",
    )
    device = plan.device_map["layers.6"]
    assert plan.load_transient_by_device == {device: 3 * 8192}
    assert plan.free_bytes[device] >= 3 * 8192


def test_the_packers_place_merging_units_where_their_transient_fits(monkeypatch):
    # Validating a finished packing could only reject it: here the first weight-only fit puts
    # the merging layer on a card without room for 3x, and 3x was dropped although another
    # packing keeps it. The transient is now part of the packing itself.
    import unsloth_zoo.device_map_planner as planner

    with torch.device("meta"):
        model = _Tiny(layers = 10)
    layer = 64 * 64 * 4
    monkeypatch.setattr(
        planner, "_load_transient_by_unit",
        lambda model, units, hf_quantizer = None: {u: 8192 for u, _ in units if u == "layers.1"},
    )
    budgets = {0: 7 * layer, 1: 3 * layer, 2: 9 * layer, 3: 9 * layer}
    plan = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        free_space_policy = "head_max",
    )
    device = plan.device_map["layers.1"]
    assert plan.load_transient_by_device == {device: 3 * 8192}
    assert budgets[device] - plan.weight_bytes[device] >= 3 * 8192


def test_a_merge_into_a_pinned_unit_keeps_its_room_on_the_head_card(monkeypatch):
    # Pinned units skip the packers, whose peaks started at zero, so a converter merging into
    # lm_head was reported as reserved on a head card the packing had filled to the limit.
    import unsloth_zoo.device_map_planner as planner

    with torch.device("meta"):
        model = _Tiny(layers = 10)
    layer = 64 * 64 * 4
    monkeypatch.setattr(
        planner, "_load_transient_by_unit",
        lambda model, units, hf_quantizer = None: {u: 8192 for u, _ in units if u == "lm_head"},
    )
    budgets = {0: 16 * layer, 1: 20 * layer}
    plan = plan_device_map(
        model, max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        prefer_head_device = 0,
    )
    assert plan.device_map["lm_head"] == 0
    assert plan.load_transient_by_device == {0: 3 * 8192}
    assert budgets[0] - plan.weight_bytes[0] >= 3 * 8192


def test_the_pretrained_helper_forwards_the_opt_out(monkeypatch):
    import unsloth_zoo.device_map_planner as planner

    seen = {}

    def build_meta_model(name, trust_remote_code = False, **config_kwargs):
        seen["config_kwargs"] = config_kwargs
        return object(), None, None

    def plan(model, **kwargs):
        seen["reserve_load_transient"] = kwargs.get("reserve_load_transient")
        return None

    monkeypatch.setattr(planner, "build_meta_model", build_meta_model)
    monkeypatch.setattr(planner, "plan_device_map", plan)
    planner.plan_device_map_for_pretrained(
        "some/model", max_memory = {0: 1 << 30, 1: 1 << 30}, reserve_load_transient = False,
    )
    assert seen == {"config_kwargs": {}, "reserve_load_transient": False}


def test_ernie_vl_fuse_and_split_experts_count_as_merging():
    # Ernie 4.5-VL stacks and concatenates its text and vision experts on the card in one op.
    configuration = pytest.importorskip("transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe")
    modeling = pytest.importorskip("transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe")
    from accelerate import init_empty_weights

    config = configuration.Ernie4_5_VLMoeConfig(
        text_config = dict(
            hidden_size = 256, intermediate_size = 128, moe_intermediate_size = [32, 16],
            moe_num_experts = 4, num_hidden_layers = 2, num_attention_heads = 2,
            num_key_value_heads = 1, vocab_size = 256, moe_k = 2,
        ),
        vision_config = dict(hidden_size = 32, depth = 1, intermediate_size = 64, num_heads = 2),
    )
    with init_empty_weights():
        model = modeling.Ernie4_5_VLMoeForConditionalGeneration(config)
    patterns = _merged_parameter_patterns(model)
    merged = {n for n, _ in model.named_parameters() if any(p.search(n) for p in patterns)}
    for branch in ("text_moe", "vision_moe"):
        for stack in ("gate_up_proj", "down_proj"):
            assert f"model.language_model.layers.1.mlp.{branch}.experts.{stack}" in merged


def test_an_op_built_around_a_concatenate_counts_as_merging(monkeypatch):
    class Concatenate:
        pass

    class FuseAndPermute:
        def __init__(self):
            self.concat_op = Concatenate()
            self.dim = 0

    class Transpose:
        def __init__(self):
            self.dim = 0

    conversions = [
        types.SimpleNamespace(target_patterns = [".qkv.weight"], operations = [FuseAndPermute()], force_cpu = False),
        types.SimpleNamespace(target_patterns = [".o_proj.weight"], operations = [Transpose()], force_cpu = False),
    ]
    fake = types.ModuleType("transformers.conversion_mapping")
    fake.get_model_conversion_mapping = lambda model, key_mapping = None, hf_quantizer = None: conversions
    monkeypatch.setitem(sys.modules, "transformers.conversion_mapping", fake)
    patterns = _merged_parameter_patterns(nn.Linear(1, 1))
    names = ["vision.blocks.0.attn.qkv.weight", "model.layers.0.self_attn.o_proj.weight"]
    assert [n for n in names if any(p.search(n) for p in patterns)] == names[:1]
