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

"""Gemma-3 vLLM -> HF rebuild against both SigLIP tower layouts.

vLLM names SigLIP weights `vision_tower.vision_model.encoder...`. transformers 4.x nests
the same way, transformers 5 flattened embeddings / encoder / post_layernorm onto
SiglipVisionModel. The installed transformers gives one layout for real; the other is
produced by rebuilding the tower in the opposite shape, so both run on any version.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Gemma3Config, Gemma3ForConditionalGeneration  # noqa: E402


class _NestedSiglipTower(torch.nn.Module):
    """transformers 4.x shape: SiglipVisionModel.vision_model.{embeddings,encoder,post_layernorm}."""

    def __init__(self, flat_tower):
        super().__init__()
        self.config = flat_tower.config
        self.vision_model = flat_tower

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)


class _FlatSiglipTower(torch.nn.Module):
    """transformers 5 shape: embeddings / encoder / post_layernorm directly on the tower."""

    def __init__(self, nested_tower):
        super().__init__()
        inner = nested_tower.vision_model
        self.config = nested_tower.config
        for name, child in inner.named_children():
            setattr(self, name, child)
        self._inner = [inner]

    def forward(self, *args, **kwargs):
        return self._inner[0](*args, **kwargs)


def _is_flat(tower):
    return not isinstance(getattr(tower, "vision_model", None), torch.nn.Module)


def _tiny_config():
    config = Gemma3Config(
        text_config = dict(
            vocab_size = 96, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
            num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
            max_position_embeddings = 128, sliding_window = 16,
        ),
        vision_config = dict(
            hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
            num_attention_heads = 2, image_size = 28, patch_size = 14, vision_use_head = False,
        ),
        mm_tokens_per_image = 4,
        image_token_index = 95, boi_token_index = 93, eoi_token_index = 94,
    )
    config.architectures = ["Gemma3ForConditionalGeneration"]
    return config


def _set_layout(model, flat):
    tower = model.model.vision_tower
    if _is_flat(tower) == flat:
        return
    model.model.vision_tower = _FlatSiglipTower(tower) if flat else _NestedSiglipTower(tower)


@pytest.fixture
def layout_patch(monkeypatch, request):
    """Force every Gemma3ForConditionalGeneration built during the test to one tower layout."""
    flat = request.param
    original_init = Gemma3ForConditionalGeneration.__init__

    def __init__(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        _set_layout(self, flat)

    monkeypatch.setattr(Gemma3ForConditionalGeneration, "__init__", __init__)
    return flat


def _vllm_style_state_dict(model):
    """HF weights renamed the way _get_vllm_state_dict names them (nested SigLIP tower)."""
    flat = _is_flat(model.model.vision_tower)
    out = {}
    for key, value in model.state_dict().items():
        if flat and key.startswith("model.vision_tower."):
            key = "model.vision_tower.vision_model." + key[len("model.vision_tower."):]
        out[key] = value.detach().clone()
    return out


def _rebuild(monkeypatch, config, quant_state_dict):
    from unsloth_zoo import vllm_utils
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda index = 0: torch.device("cpu"))
    return vllm_utils.convert_vllm_to_huggingface(
        quant_state_dict, copy.deepcopy(config), dtype = torch.float32, is_vision_model = True,
    )


def _reference(config):
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(copy.deepcopy(config)).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02)
    return model


@pytest.mark.parametrize("layout_patch", [False, True], ids = ["nested_4x", "flat_5x"], indirect = True)
def test_gemma3_vllm_rebuild_matches_hf(monkeypatch, layout_patch):
    config = _tiny_config()
    reference = _reference(config)
    assert _is_flat(reference.model.vision_tower) == layout_patch

    rebuilt = _rebuild(monkeypatch, config, _vllm_style_state_dict(reference))

    expected = reference.state_dict()
    got = rebuilt.state_dict()
    vision_keys = [k for k in expected if k.startswith("model.vision_tower.")]
    assert vision_keys and all(("vision_model" in k) != layout_patch for k in vision_keys)
    assert sorted(set(expected) - set(got)) == []
    for key, value in expected.items():
        assert got[key].shape == value.shape, key
        assert torch.equal(got[key].float(), value.float()), key

    torch.manual_seed(1)
    pixel_values = torch.randn(1, 3, 28, 28)
    input_ids = torch.tensor([[2, 93] + [95] * 4 + [94, 10, 11, 12]])
    with torch.no_grad():
        ref_out = reference(input_ids = input_ids, pixel_values = pixel_values).logits
        new_out = rebuilt(input_ids = input_ids, pixel_values = pixel_values).logits
    torch.testing.assert_close(new_out, ref_out)


@pytest.mark.parametrize("layout_patch", [False, True], ids = ["nested_4x", "flat_5x"], indirect = True)
def test_gemma3_vision_census_is_clean(monkeypatch, layout_patch):
    from unsloth_zoo.empty_model import vision_tower_census, create_empty_model
    config = _tiny_config()
    reference = _reference(config)
    rebuilt = _rebuild(monkeypatch, config, _vllm_style_state_dict(reference))
    _, meta_model, _, _ = create_empty_model(copy.deepcopy(config), torch.float32, is_vision_model = True)
    tower = "model.vision_tower" if layout_patch else "model.vision_tower.vision_model"
    assert vision_tower_census(rebuilt, meta_model, [tower]) == []
    tower_module = rebuilt.model.vision_tower
    assert getattr(tower_module, "vision_model", None) is not tower_module


@pytest.mark.parametrize("layout_patch", [False, True], ids = ["nested_4x", "flat_5x"], indirect = True)
def test_vision_name_mapping_follows_built_module(layout_patch):
    from unsloth_zoo.empty_model import vllm_vision_name_to_hf, align_vision_tower_names
    model = Gemma3ForConditionalGeneration(_tiny_config())
    name = "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    expected = name.replace(".vision_model.", ".") if layout_patch else name
    assert vllm_vision_name_to_hf(name, model) == expected
    # A parent that does not own the next segment is never rewritten (Idefics3 / Mllama style).
    other = "model.vision_model.encoder.layers.0.layer_norm1.weight"
    assert vllm_vision_name_to_hf(other, model) == other
    qsd, names, flat = align_vision_tower_names(
        model, {name: torch.zeros(1)}, ["model.vision_tower.vision_model.encoder.layers.{kk}.mlp.fc1"],
    )
    assert list(qsd) == [expected]
    assert flat == (["model.vision_tower"] if layout_patch else [])
    assert names == [("model.vision_tower.encoder.layers.{kk}.mlp.fc1" if layout_patch
                      else "model.vision_tower.vision_model.encoder.layers.{kk}.mlp.fc1")]


def test_vision_census_reports_unassigned_and_wrong_shape():
    from unsloth_zoo.empty_model import vision_tower_census
    config = _tiny_config()
    reference = Gemma3ForConditionalGeneration(config)
    broken = Gemma3ForConditionalGeneration(config)
    tower = broken.model.vision_tower
    inner = getattr(tower, "vision_model", tower)
    inner.post_layernorm.weight = torch.nn.Parameter(torch.ones(3))
    path = "model.vision_tower" if inner is tower else "model.vision_tower.vision_model"
    problems = vision_tower_census(broken, reference, [path])
    assert any("post_layernorm.weight" in name and "shape" in problem for name, problem in problems)
    inner.self_ref = inner  # a module registered as its own child
    problems = vision_tower_census(broken, reference, [path])
    assert any(problem == "module aliases itself" for _, problem in problems)
