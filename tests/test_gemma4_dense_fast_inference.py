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

"""Dense Gemma-4 (E2B / E4B) with fast_inference.

vLLM profiled the audio tower Unsloth never feeds, and with Unsloth's compiled transformers
audio modules in place that aborted the process. A text_only load must also not share a
vLLM AOT compile artifact with a multimodal run (vllm-project/vllm#50891). With audio off in
vLLM, the training model's audio tower has to come from the checkpoint.
"""
import inspect
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

import unsloth_zoo.vllm_utils as vllm_utils
from unsloth_zoo.vllm_utils import (
    _get_multimodal_engine_args,
    _load_gemma4_audio_from_checkpoint,
)


def _gemma4(audio = True, vision = True):
    return SimpleNamespace(
        model_type = "gemma4",
        text_config = SimpleNamespace(model_type = "gemma4_text"),
        vision_config = SimpleNamespace() if vision else None,
        audio_config = SimpleNamespace() if audio else None,
    )


def test_vision_load_turns_audio_off():
    args = _get_multimodal_engine_args(_gemma4(), is_vision_model = True)
    assert args == {"limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0}}


def test_vision_load_without_audio_is_unchanged():
    qwen = SimpleNamespace(model_type = "qwen2_5_vl", vision_config = SimpleNamespace())
    assert _get_multimodal_engine_args(qwen, True) == {"limit_mm_per_prompt": {"image": 1, "video": 0}}
    assert _get_multimodal_engine_args(_gemma4(audio = False), True) == {
        "limit_mm_per_prompt": {"image": 1, "video": 0}
    }


def test_text_only_gemma4_runs_language_model_only():
    # text_only hands load_vllm the text config, which has no vision / audio sub-config.
    text_config = SimpleNamespace(model_type = "gemma4_text")
    for config in (text_config, _gemma4()):
        args = _get_multimodal_engine_args(config, is_vision_model = False)
        # language_model_only is in vLLM's AOT cache key; zeroed limits alone are not.
        assert args["language_model_only"] is True
        assert args["limit_mm_per_prompt"]["audio"] == 0


def test_text_models_get_no_multimodal_args():
    assert _get_multimodal_engine_args(SimpleNamespace(model_type = "llama"), False) == {}
    qwen = SimpleNamespace(model_type = "qwen3_5", vision_config = SimpleNamespace())
    assert _get_multimodal_engine_args(qwen, False) == {}


def test_load_vllm_uses_the_helper():
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "engine_args.update(_get_multimodal_engine_args(config, is_vision_model))" in source
    assert 'engine_args["limit_mm_per_prompt"] = {"image": 1, "video": 0}' not in source


class _Audio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias = False)
        self.norm = torch.nn.LayerNorm(4)


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = _Audio()
        self.embed_audio = torch.nn.Module()
        # create_empty_model's 1-wide placeholder
        self.embed_audio.embedding_projection = torch.nn.Linear(4, 1, bias = False)
        self.language_model = torch.nn.Linear(4, 4, bias = False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()


def test_audio_tower_is_read_from_the_checkpoint(tmp_path):
    torch.manual_seed(0)
    ckpt = {
        "model.audio_tower.proj.weight": torch.randn(4, 4),
        "model.audio_tower.norm.weight": torch.randn(4),
        "model.audio_tower.norm.bias": torch.randn(4),
        "model.embed_audio.embedding_projection.weight": torch.randn(6, 4),
        "model.language_model.weight": torch.randn(4, 4),
    }
    path = str(tmp_path / "model.safetensors")
    save_file(ckpt, path)
    model = _Model()
    lm_before = model.model.language_model.weight.detach().clone()
    loaded = _load_gemma4_audio_from_checkpoint(
        model, _gemma4(), weight_map = {k: path for k in ckpt}
    )
    assert loaded == 4
    params = dict(model.named_parameters())
    for key in ckpt:
        if key.startswith("model.language_model"): continue
        assert torch.equal(params[key], ckpt[key]), key
        assert not params[key].requires_grad
    proj = model.model.embed_audio.embedding_projection
    assert (proj.out_features, proj.in_features) == (6, 4)
    # Only the audio prefixes are touched; the rest is shared with vLLM.
    assert torch.equal(model.model.language_model.weight, lm_before)


def test_unreadable_checkpoint_drops_the_audio_tower():
    model = _Model()
    loaded = _load_gemma4_audio_from_checkpoint(model, _gemma4(), weight_map = {})
    assert loaded == 0
    assert model.model.audio_tower is None and model.model.embed_audio is None
    assert model.model.language_model is not None


def test_no_audio_config_is_a_no_op():
    model = _Model()
    assert _load_gemma4_audio_from_checkpoint(model, _gemma4(audio = False), weight_map = {}) == 0
    assert model.model.audio_tower is not None


def test_convert_loads_audio_on_the_vision_path():
    source = inspect.getsource(vllm_utils.convert_vllm_to_huggingface)
    assert "_load_gemma4_audio_from_checkpoint(new_model, config)" in source
