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

"""get_model_type returns the vision name, so a branch on the top level one is dead: the
bound falls back to 32 and Mllama loses its cross-attn gates on text layers 33 and 38.
"""
import pytest
import torch

from unsloth_zoo.empty_model import (
    QWEN_VL_MERGED_QKV_TYPES,
    extract_vision_layers,
    get_model_layer_counts,
    get_model_type,
)


class Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def mllama(text_layers = 40, vision_layers = 32, global_layers = 8, cross = (3, 33, 38)):
    return Cfg(
        model_type = "mllama",
        text_config = Cfg(
            model_type = "mllama_text_model",
            num_hidden_layers = text_layers,
            cross_attention_layers = list(cross),
        ),
        vision_config = Cfg(
            model_type = "mllama_vision_model",
            num_hidden_layers = vision_layers,
            num_global_layers = global_layers,
        ),
    )


def iterate_to(config):
    counts = get_model_layer_counts(config)
    return max(counts.values()) if isinstance(counts, dict) else counts


@pytest.mark.parametrize("top, vision, expected", [
    ("mllama",         "mllama_vision_model",   "mllama_vision_model"),
    ("gemma3",         "siglip_vision_model",   "siglip_vision_model"),
    ("gemma4",         "gemma4_vision",         "gemma4_vision"),
    ("gemma4_unified", "gemma4_unified_vision", "gemma4_unified_vision"),
    ("qwen3_vl",       "qwen3_vl_vision",       "qwen3_vl_vision"),
    ("qwen2_5_vl",     "qwen2_5_vl",            "qwen2_5_vl"),
])
def test_get_model_type_prefers_the_vision_name(top, vision, expected):
    cfg = Cfg(model_type = top, vision_config = Cfg(model_type = vision))
    assert get_model_type(cfg) == expected


@pytest.mark.parametrize("vision_name", ["mllama", "mllama_vision_model"])
def test_mllama_counts_under_both_spellings(vision_name):
    cfg = mllama()
    cfg.vision_config.model_type = vision_name
    assert get_model_layer_counts(cfg) == {
        "text_layers": 40, "vision_layers": 32, "global_layers": 8,
    }


@pytest.mark.parametrize("vision_name", ["gemma3", "siglip_vision_model"])
def test_gemma3_counts_under_both_spellings(vision_name):
    cfg = Cfg(
        model_type = "gemma3",
        text_config = Cfg(num_hidden_layers = 62),
        vision_config = Cfg(model_type = vision_name, num_hidden_layers = 27),
    )
    assert get_model_layer_counts(cfg) == {"text_layers": 62, "vision_layers": 27}


@pytest.mark.parametrize("vision_name", [
    "gemma4", "gemma4_vision", "gemma4_unified", "gemma4_unified_vision",
])
def test_gemma4_counts_under_every_spelling(vision_name):
    cfg = Cfg(
        model_type = "gemma4",
        text_config = Cfg(num_hidden_layers = 60),
        vision_config = Cfg(model_type = vision_name, num_hidden_layers = 27),
    )
    assert get_model_layer_counts(cfg) == {"text_layers": 60, "vision_layers": 27}


@pytest.mark.parametrize("vision_name", ["qwen3_vl", "qwen3_vl_vision"])
def test_qwen3_vl_counts_under_both_spellings(vision_name):
    cfg = Cfg(
        model_type = "qwen3_vl",
        num_hidden_layers = 36,
        vision_config = Cfg(model_type = vision_name, depth = 27, deepstack_depth = 3),
    )
    assert get_model_layer_counts(cfg) == {
        "text_layers": 36, "vision_layers": 27, "deepstack_layers": 3,
    }


def test_qwen3_vl_vision_still_takes_the_merged_qkv_path():
    # renamed in transformers 5; dropping out splits a qkv HF keeps merged.
    assert "qwen3_vl" in QWEN_VL_MERGED_QKV_TYPES
    assert "qwen3_vl_vision" in QWEN_VL_MERGED_QKV_TYPES
    assert "qwen2_5_vl" in QWEN_VL_MERGED_QKV_TYPES


def test_plain_causal_lm_still_returns_an_int():
    assert get_model_layer_counts(Cfg(model_type = "llama", num_hidden_layers = 16)) == 16


@pytest.mark.parametrize("model_type", ["mllama", "gemma3", "gemma4"])
def test_missing_sub_configs_do_not_raise(model_type):
    # A vision model_type with no text_config used to raise AttributeError.
    assert isinstance(get_model_layer_counts(Cfg(model_type = model_type)), dict)


def test_unknown_vision_model_is_untouched():
    cfg = Cfg(
        model_type = "smolvlm",
        num_hidden_layers = 24,
        vision_config = Cfg(model_type = "smolvlm_vision", num_hidden_layers = 27),
    )
    assert get_model_layer_counts(cfg) == 24


def test_cross_attention_gates_above_layer_31_are_extracted():
    cross = [3, 8, 13, 18, 23, 28, 33, 38]
    cfg = mllama(text_layers = 40, cross = cross)

    root = torch.nn.Module()
    root.config = cfg
    root.model = torch.nn.Module()
    lm = torch.nn.Module()
    lm.model = torch.nn.Module()
    lm.model.layers = torch.nn.ModuleList()
    for i in range(40):
        layer = torch.nn.Module()
        if i in cross:
            layer.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))
            layer.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        lm.model.layers.append(layer)
    root.model.language_model = lm

    seen = {}
    state_dict, quant_state_dict = {}, {}

    def record(prefix, kk, sd, proj, slice_weights = True, slice_indices = None):
        seen[prefix] = True

    extract_vision_layers(root, state_dict, quant_state_dict, record)

    found = set(seen) | set(state_dict)
    gates = {k for k in found if "cross_attn" in k}
    assert len(gates) == 2 * len(cross), sorted(gates)
    for layer in (33, 38):
        assert any(f".layers.{layer}.cross_attn_attn_gate" in k for k in gates)
        assert any(f".layers.{layer}.cross_attn_mlp_gate" in k for k in gates)


def test_iteration_bound_covers_the_deepest_layer():
    assert iterate_to(mllama(text_layers = 100)) == 100
    assert iterate_to(mllama(text_layers = 40)) == 40
