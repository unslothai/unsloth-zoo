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

"""Step-3.7-Flash reads `moe.gate.weight` directly, so a Linear4bit router fails with
"mat1 and mat2 shapes cannot be multiplied (512x4096 and 1x589824)"."""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES


ROUTER_NAMES = [
    "model.layers.3.moe.gate",
    "model.layers.3.mlp.gate",
    "model.layers.3.block_sparse_moe.gate",
    "layers.0.moe.gate",
]
EXPERT_NAMES = [
    "model.layers.3.moe.gate_proj",
    "model.layers.3.moe.up_proj",
    "model.layers.3.moe.down_proj",
    "model.layers.3.mlp.gate_proj",
    "model.layers.3.share_expert.gate_proj",
    "model.layers.3.self_attn.q_proj",
]


def _should_convert_module():
    module = pytest.importorskip("transformers.quantizers.quantizers_utils")
    function = getattr(module, "should_convert_module", None)
    if function is None:
        pytest.skip(reason="transformers < 5.0 has no should_convert_module")
    return function


def test_moe_gate_is_in_the_skip_list():
    assert "moe.gate" in SKIP_QUANTIZATION_MODULES


@pytest.mark.parametrize("name", ROUTER_NAMES)
def test_router_under_moe_block_is_skipped(name):
    should_convert_module = _should_convert_module()
    assert not should_convert_module(name, SKIP_QUANTIZATION_MODULES), name


@pytest.mark.parametrize("name", EXPERT_NAMES)
def test_neighbouring_projections_still_convert(name):
    should_convert_module = _should_convert_module()
    assert should_convert_module(name, SKIP_QUANTIZATION_MODULES), name


def test_legacy_match_reaches_no_router_and_no_expert():
    def legacy_skip(full_name, keys):
        return any((key + "." in full_name) or (key == full_name) for key in keys)

    for name in ROUTER_NAMES:
        if name == "moe.gate":
            assert legacy_skip(name, SKIP_QUANTIZATION_MODULES), name
        else:
            assert not legacy_skip(name, SKIP_QUANTIZATION_MODULES), name
    for name in EXPERT_NAMES:
        assert not legacy_skip(name, SKIP_QUANTIZATION_MODULES), name


class _MoELinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

    def forward(self, x, expert_id):
        return nn.functional.linear(x.float(), self.weight[expert_id].float())


class _MoEBlock(nn.Module):
    def __init__(self, hidden = 16, inter = 8, num_experts = 4):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias = False)
        self.up_proj = _MoELinear(num_experts, hidden, inter)
        self.gate_proj = _MoELinear(num_experts, hidden, inter)
        self.down_proj = _MoELinear(num_experts, inter, hidden)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _MoEBlock()
        self.share_expert = nn.Sequential()
        self.share_expert.gate_proj = nn.Linear(16, 8, bias = False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


def test_replace_with_bnb_linear_leaves_the_router_alone():
    _should_convert_module()
    bnb = pytest.importorskip("bitsandbytes")
    integrations = pytest.importorskip("transformers.integrations.bitsandbytes")
    from transformers import BitsAndBytesConfig

    model = _Model()
    config = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = list(SKIP_QUANTIZATION_MODULES))
    model = integrations.replace_with_bnb_linear(
        model, modules_to_not_convert = list(SKIP_QUANTIZATION_MODULES), quantization_config = config,
    )
    if isinstance(model, tuple):
        model = model[0]
    for layer in model.layers:
        assert type(layer.moe.gate) is nn.Linear, type(layer.moe.gate)
        assert isinstance(layer.share_expert.gate_proj, bnb.nn.Linear4bit)
        assert isinstance(layer.moe.up_proj.weight, nn.Parameter)


def test_the_4x_matcher_gains_a_suffix_match_for_moe_gate_only():
    from unsloth_zoo.patching_utils import (
        _add_suffix_match_to_bnb_skip, _BNB_SKIP_MATCH, _BNB_SUFFIX_MATCH_KEYS,
    )

    patched = _add_suffix_match_to_bnb_skip(
        "def matches(current_key_name_str, modules_to_not_convert):\n"
        "    return any(" + _BNB_SKIP_MATCH + " for key in modules_to_not_convert)\n"
    )
    namespace = {"_BNB_SUFFIX_MATCH_KEYS": _BNB_SUFFIX_MATCH_KEYS}
    exec(patched, namespace)
    matches = namespace["matches"]
    assert matches("model.layers.3.moe.gate", ["moe.gate"]) is True
    assert matches("model.layers.3.moe.gate_proj", ["moe.gate"]) is False
    assert matches("model.layers.3.moe.gate.weight", ["moe.gate"]) is True
    # 4.x saves list these while storing the router packed; matching them would break the reload.
    assert matches("model.layers.3.mlp.gate", ["mlp.gate", "router", "block_sparse_moe.gate"]) is False
    assert matches("model.layers.3.feed_forward.router", ["router"]) is False
    assert _add_suffix_match_to_bnb_skip("nothing here") == "nothing here"


def test_patched_4x_replace_with_bnb_linear_leaves_the_router_alone():
    bnb = pytest.importorskip("bitsandbytes")
    integrations = pytest.importorskip("transformers.integrations.bitsandbytes")
    if not hasattr(integrations, "_replace_with_bnb_linear"):
        pytest.skip(reason="transformers >= 5.0 has no _replace_with_bnb_linear")
    import unsloth_zoo.patching_utils  # noqa: F401
    from transformers import BitsAndBytesConfig

    skip = list(SKIP_QUANTIZATION_MODULES)
    model = integrations.replace_with_bnb_linear(
        _Model(), modules_to_not_convert = skip,
        quantization_config = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = skip),
    )
    for layer in model.layers:
        assert type(layer.moe.gate) is nn.Linear, type(layer.moe.gate)
        assert isinstance(layer.share_expert.gate_proj, bnb.nn.Linear4bit)


def test_patched_4x_keeps_packing_an_mlp_gate_router():
    bnb = pytest.importorskip("bitsandbytes")
    integrations = pytest.importorskip("transformers.integrations.bitsandbytes")
    if not hasattr(integrations, "_replace_with_bnb_linear"):
        pytest.skip(reason="transformers >= 5.0 has no _replace_with_bnb_linear")
    import unsloth_zoo.patching_utils  # noqa: F401
    from transformers import BitsAndBytesConfig

    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.gate = nn.Linear(16, 4, bias = False)
    skip = list(SKIP_QUANTIZATION_MODULES)
    model = integrations.replace_with_bnb_linear(
        model, modules_to_not_convert = skip,
        quantization_config = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = skip),
    )
    assert isinstance(model.layers[0].mlp.gate, bnb.nn.Linear4bit)
