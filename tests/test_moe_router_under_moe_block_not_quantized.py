# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""A MoE router that lives under a `moe` block must stay out of 4-bit.

stepfun-ai/Step-3.7-Flash keeps its router at `layers.N.moe.gate` and reads
its weight directly (`hidden @ self.gate.weight.t()`), so a Linear4bit there
hands the matmul packed bytes: "mat1 and mat2 shapes cannot be multiplied
(512x4096 and 1x589824)". The skip list already covered `mlp.gate` and
`block_sparse_moe.gate`; this pins `moe.gate` and checks that the suffix
does not swallow the `moe.gate_proj` / `moe.up_proj` expert stacks next to it.
"""
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
    """transformers' own matcher, or a skip.

    `quantizers_utils` exists on transformers 4.x too, so importorskip on the
    module resolves there and the attribute access then raises AttributeError:
    `should_convert_module` first ships in transformers 5.0. Gate on the symbol.
    """
    module = pytest.importorskip("transformers.quantizers.quantizers_utils")
    function = getattr(module, "should_convert_module", None)
    if function is None:
        pytest.skip("transformers < 5.0 has no should_convert_module")
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
    """What transformers 4.x really does with this entry.

    `_replace_with_bnb_linear` there tests `(key + "." in current_key_name_str)
    or (key == current_key_name_str)` on the UNPADDED path, so a pattern can
    only match something BELOW the key, never a leaf: `moe.gate` does not reach
    `model.layers.3.moe.gate`, exactly as the pre-existing `mlp.gate` and
    `block_sparse_moe.gate` entries do not. The entry is inert on 4.x and the
    router is still quantized there; what matters is that it is inert in the
    SAFE direction, i.e. it never captures an expert or attention projection.
    Copied from transformers 4.57.6 integrations/bitsandbytes.py."""
    def legacy_skip(full_name, keys):
        return any((key + "." in full_name) or (key == full_name) for key in keys)

    for name in ROUTER_NAMES:
        if name == "moe.gate":   # a root-level router is an exact-equality hit
            assert legacy_skip(name, SKIP_QUANTIZATION_MODULES), name
        else:
            assert not legacy_skip(name, SKIP_QUANTIZATION_MODULES), name
    for name in EXPERT_NAMES:
        assert not legacy_skip(name, SKIP_QUANTIZATION_MODULES), name


class _MoELinear(nn.Module):
    """Step-3.7's per-expert stack: one 3-D weight, indexed by expert id."""
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
    _should_convert_module()   # the entry only reaches a leaf on transformers >= 5.0
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
        # Expert stacks are bare parameters: not a Linear, so untouched here.
        assert isinstance(layer.moe.up_proj.weight, nn.Parameter)


def test_the_4x_matcher_gains_a_suffix_match():
    """On transformers 4.x the bitsandbytes skip check is `key + "." in path or key == path`, so
    a dotted leaf entry never matched the leaf itself. The rewritten check also takes the
    entry as a suffix."""
    from unsloth_zoo.patching_utils import _add_suffix_match_to_bnb_skip, _BNB_SKIP_MATCH

    snippet = (
        "            if not any(\n"
        "                " + _BNB_SKIP_MATCH + " for key in modules_to_not_convert\n"
        "            ):\n"
    )
    patched = _add_suffix_match_to_bnb_skip(snippet)
    assert 'endswith("." + key)' in patched
    namespace = {}
    exec("def matches(current_key_name_str, modules_to_not_convert):\n    return any(" + _BNB_SKIP_MATCH + ' or current_key_name_str.endswith("." + key)' + " for key in modules_to_not_convert)", namespace)
    assert namespace["matches"]("model.layers.3.moe.gate", ["moe.gate"]) is True
    assert namespace["matches"]("model.layers.3.moe.gate_proj", ["moe.gate"]) is False
    assert namespace["matches"]("model.layers.3.moe.gate.weight", ["moe.gate"]) is True   # parent prefix, as before
    assert _add_suffix_match_to_bnb_skip("nothing here") == "nothing here"
