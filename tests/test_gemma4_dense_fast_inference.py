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
    assert "_load_gemma4_audio_from_checkpoint(" in source
    assert 'getattr(quant_state_dict, "_unsloth_checkpoint_source", None)' in source


def test_state_dict_carries_vllms_checkpoint_source_as_an_attribute():
    source = inspect.getsource(vllm_utils._get_vllm_state_dict)
    assert 'getattr(model_config, "revision", None)' in source
    assert 'getattr(load_config, "download_dir", None)' in source
    assert "quant_state_dict._unsloth_checkpoint_source = checkpoint_source" in source
    # An OrderedDict attribute is not a key, so nothing iterating the tensors sees it.
    from collections import OrderedDict

    quant_state_dict = OrderedDict(w = torch.zeros(1))
    quant_state_dict._unsloth_checkpoint_source = ("org/repo", "rev", "/cache")
    assert list(quant_state_dict) == ["w"]


def _two_revision_hub(tmp_path, monkeypatch):
    """A fake Hub cache holding `main` and `rev-b` snapshots with different audio weights."""
    import huggingface_hub

    snapshots, calls = {}, []
    for revision, seed in (("main", 0), ("rev-b", 1)):
        torch.manual_seed(seed)
        tensors = {
            "model.audio_tower.proj.weight": torch.randn(4, 4),
            "model.audio_tower.norm.weight": torch.randn(4),
            "model.audio_tower.norm.bias": torch.randn(4),
            "model.embed_audio.embedding_projection.weight": torch.randn(6, 4),
        }
        folder = tmp_path / revision
        folder.mkdir()
        save_file(tensors, str(folder / "model.safetensors"))
        snapshots[revision] = (folder, tensors)

    def fake_download(repo_id, filename, revision = None, cache_dir = None, local_files_only = False, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision, "cache_dir": cache_dir})
        folder, _ = snapshots[revision or "main"] if (revision or "main") in snapshots else (None, None)
        path = None if folder is None else folder / filename
        if path is None or not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return snapshots, calls


def test_requested_revision_and_cache_dir_are_honoured(tmp_path, monkeypatch):
    snapshots, calls = _two_revision_hub(tmp_path, monkeypatch)
    model = _Model()
    loaded = _load_gemma4_audio_from_checkpoint(
        model, _gemma4(), checkpoint_source = ("org/gemma-4-e2b", "rev-b", "/custom/cache"),
    )
    assert loaded == 4
    assert calls and all(
        c["repo_id"] == "org/gemma-4-e2b" and c["revision"] == "rev-b" and c["cache_dir"] == "/custom/cache"
        for c in calls
    )
    params = dict(model.named_parameters())
    for key, value in snapshots["rev-b"][1].items():
        assert torch.equal(params[key], value), key
        assert not torch.equal(params[key], snapshots["main"][1][key]), key


def test_default_snapshot_is_not_used_for_another_revision(tmp_path, monkeypatch):
    _, calls = _two_revision_hub(tmp_path, monkeypatch)
    model = _Model()
    loaded = _load_gemma4_audio_from_checkpoint(
        model, _gemma4(), checkpoint_source = ("org/gemma-4-e2b", "rev-missing", None),
    )
    # rev-missing is not cached: drop the tower rather than read main's weights.
    assert loaded == 0
    assert model.model.audio_tower is None
    assert calls and all(c["revision"] == "rev-missing" for c in calls)


def test_config_commit_hash_pins_the_snapshot_without_a_source(tmp_path, monkeypatch):
    snapshots, calls = _two_revision_hub(tmp_path, monkeypatch)
    config = _gemma4()
    config._name_or_path = "org/gemma-4-e2b"
    config._commit_hash = "rev-b"
    model = _Model()
    assert _load_gemma4_audio_from_checkpoint(model, config) == 4
    assert calls and all(c["revision"] == "rev-b" for c in calls)
    assert torch.equal(
        model.model.audio_tower.proj.weight, snapshots["rev-b"][1]["model.audio_tower.proj.weight"]
    )


def _quantized_audio_checkpoint(tmp_path, packed = True, siblings = True, shape = None):
    torch.manual_seed(0)
    ckpt = {
        "model.audio_tower.norm.weight": torch.randn(4),
        "model.audio_tower.norm.bias": torch.randn(4),
        "model.embed_audio.embedding_projection.weight": torch.randn(6, 4),
    }
    if packed:
        # bitsandbytes 4-bit: (out * in / 2, 1) uint8 plus its quant state
        ckpt["model.audio_tower.proj.weight"] = torch.randint(0, 255, (8, 1), dtype = torch.uint8)
    else:
        ckpt["model.audio_tower.proj.weight"] = torch.randn(*(shape or (4, 4)))
    if siblings:
        ckpt["model.audio_tower.proj.weight.absmax"] = torch.rand(1)
        ckpt["model.audio_tower.proj.weight.quant_map"] = torch.rand(16)
        ckpt["model.audio_tower.proj.weight.quant_state.bitsandbytes__nf4"] = torch.zeros(8, dtype = torch.uint8)
    path = str(tmp_path / "model.safetensors")
    save_file(ckpt, path)
    return {k: path for k in ckpt}


def _assert_nothing_installed(model, before, loaded):
    assert loaded == 0
    assert model.model.audio_tower is None and model.model.embed_audio is None
    assert torch.equal(model.model.language_model.weight, before)


def test_bnb_packed_audio_tower_is_never_installed(tmp_path):
    model = _Model()
    before = model.model.language_model.weight.detach().clone()
    # Keep references: the tower must not have been written before it was dropped.
    audio = model.model.audio_tower
    proj_before = audio.proj.weight.detach().clone()
    loaded = _load_gemma4_audio_from_checkpoint(model, _gemma4(), weight_map = _quantized_audio_checkpoint(tmp_path))
    _assert_nothing_installed(model, before, loaded)
    assert audio.proj.weight.dtype == torch.float32 and torch.equal(audio.proj.weight, proj_before)


def test_integer_audio_weight_without_quant_state_is_refused(tmp_path):
    model = _Model()
    before = model.model.language_model.weight.detach().clone()
    weight_map = _quantized_audio_checkpoint(tmp_path, packed = True, siblings = False)
    _assert_nothing_installed(model, before, _load_gemma4_audio_from_checkpoint(model, _gemma4(), weight_map = weight_map))


def test_quant_state_siblings_alone_are_refused(tmp_path):
    model = _Model()
    before = model.model.language_model.weight.detach().clone()
    weight_map = _quantized_audio_checkpoint(tmp_path, packed = False, siblings = True)
    _assert_nothing_installed(model, before, _load_gemma4_audio_from_checkpoint(model, _gemma4(), weight_map = weight_map))


def test_mismatched_audio_shape_is_refused(tmp_path):
    model = _Model()
    before = model.model.language_model.weight.detach().clone()
    weight_map = _quantized_audio_checkpoint(tmp_path, packed = False, siblings = False, shape = (8, 2))
    _assert_nothing_installed(model, before, _load_gemma4_audio_from_checkpoint(model, _gemma4(), weight_map = weight_map))
