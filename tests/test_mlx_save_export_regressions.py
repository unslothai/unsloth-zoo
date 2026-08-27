# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fast regressions for MLX save/export parity fixes: the contracts behind
the real save / GGUF export bugs, without downloading or converting models.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from copy import copy
from pathlib import Path

import pytest


AutoProcessor = load_model = load_processor = nn = skip_multimodal_module = None
_to_mx_array = None


def _test_bound_load_processor(model_path, **kwargs):
    return AutoProcessor.from_pretrained(model_path, **kwargs)


def _test_bound_load_model(paths, weights):
    return load_model(paths, weights)


def _test_bound_vlm_load(model_path, paths=None, weights=None, **kwargs):
    if paths is not None:
        load_model(paths, weights)
    return load_processor(model_path, **kwargs)


def _test_make_streaming_detokenizer(processor):
    detokenizer = copy(processor.detokenizer)
    detokenizer.reset()
    return detokenizer


def _test_naive_detokenizer_init(self, tokenizer):
    self._tokenizer, self._tokens = tokenizer, []


def _test_naive_detokenizer_reset(self):
    self._tokens = []


def _test_legacy_projector_load(paths, weights):
    skip_vision = True
    config = {"quantization": {}}

    def get_class_predicate(p, m):
        if skip_multimodal_module(p) and skip_vision:
            return False
        if p in config["quantization"]:
            return config["quantization"][p]
        if not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.size % 64 != 0:
            return False
        return f"{p}.scales" in weights

    return nn.quantize(paths, class_predicate=get_class_predicate)


def _test_aware_projector_load(paths, weights):
    skip_vision = True

    def get_class_predicate(path, module):
        if skip_multimodal_module(path) and skip_vision and f"{path}.scales" not in weights:
            return False
        return f"{path}.scales" in weights

    return nn.quantize(paths, class_predicate=get_class_predicate)


def _test_lfm_projector_init(self, config):
    self.projector_use_layernorm = config.projector_use_layernorm
    self.layer_norm = lambda x: ("normalized", x)


def _test_lfm_projector_call(self, x):
    return self.layer_norm(x) if self.projector_use_layernorm else x


def _test_minicpmo_legacy_vision(self, pixel_values, tgt_sizes):
    dtype = self.language_model.model.embed_tokens.weight.dtype
    return _to_mx_array(pixel_values, dtype=dtype)


def _test_minicpmo_fixed_vision(self, pixel_values, tgt_sizes):
    dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
    return _to_mx_array(pixel_values, dtype=dtype)


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_torch_shim():
    pytest.importorskip("torch")
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def test_vlm_config_save_uses_vlm_helper_and_preserves_quantization_config(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.utils as mutils

    calls = {}
    fake_vlm_utils = types.ModuleType("mlx_vlm.utils")

    def fake_save_config(config, path):
        calls["config"] = config
        calls["path"] = Path(path)
        Path(path).write_text(json.dumps(config), encoding="utf-8")

    fake_vlm_utils.save_config = fake_save_config
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_vlm_utils)

    config = {
        "model_type": "gemma3",
        "vision_config": {"hidden_size": 8},
        "quantization": {"group_size": 64, "bits": 4},
    }
    mutils._save_mlx_config(config, tmp_path / "config.json", is_vlm=True)

    assert calls["path"] == tmp_path / "config.json"
    assert calls["config"]["quantization"] == config["quantization"]
    assert calls["config"]["quantization_config"] == config["quantization"]
    assert "quantization_config" not in config


def test_merged_16bit_save_fully_dequantizes_model(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = {"fuse": [], "dequantize": 0}

    class LoRALinear:
        def fuse(self, dequantize=False):
            calls["fuse"].append(dequantize)
            return "fused-linear"

    class Model:
        _config = {
            "model_type": "llama",
            "tie_word_embeddings": False,
            "quantization": {"bits": 4},
            "nested": {"quantization_config": {"bits": 4}},
        }

        def eval(self):
            calls["eval"] = True

        def named_modules(self):
            return [("layers.0.self_attn.q_proj", LoRALinear())]

        def update_modules(self, modules):
            calls["updated"] = modules

    class Tokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            calls["tokenizer_path"] = Path(path)

    fake_mlx_lm_utils = types.ModuleType("mlx_lm.utils")

    def fake_dequantize_model(model):
        calls["dequantize"] += 1
        return model

    def fake_save_model(path, model, donate_model=False):
        Path(path).mkdir(parents=True, exist_ok=True)
        calls["donate_model"] = donate_model

    def fake_save_config(config, path):
        calls["saved_config"] = config
        Path(path).write_text(json.dumps(config), encoding="utf-8")

    fake_mlx_lm_utils.dequantize_model = fake_dequantize_model
    fake_mlx_lm_utils.save_model = fake_save_model
    fake_mlx_lm_utils.save_config = fake_save_config
    fake_mlx_lm_utils.create_model_card = lambda path, hf_repo: None
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake_mlx_lm_utils)

    fake_mlx_utils = types.ModuleType("mlx.utils")
    fake_mlx_utils.tree_unflatten = dict
    monkeypatch.setitem(sys.modules, "mlx.utils", fake_mlx_utils)

    mutils.save_merged_model(Model(), Tokenizer(), tmp_path, dequantize=True)

    assert calls["eval"] is True
    assert calls["fuse"] == [True]
    assert calls["dequantize"] == 1
    assert calls["donate_model"] is False
    assert "quantization" not in calls["saved_config"]
    assert "quantization_config" not in calls["saved_config"]["nested"]


def test_bound_gguf_save_filters_cuda_only_kwargs(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    def fake_save_pretrained_gguf(
        model,
        tokenizer,
        save_directory,
        quantization_method="fast_quantized",
        **kwargs,
    ):
        calls["tokenizer"] = tokenizer
        calls["save_directory"] = Path(save_directory)
        calls["quantization_method"] = quantization_method
        calls["kwargs"] = kwargs

    monkeypatch.setattr(mutils, "save_pretrained_gguf", fake_save_pretrained_gguf)
    tokenizer = object()
    model = types.SimpleNamespace(_tokenizer=tokenizer)

    loader._mlx_save_pretrained_gguf(
        model,
        tmp_path,
        quantization_method="not_quantized",
        first_conversion="f16",
        maximum_memory_usage=0.5,
        temporary_location="/tmp/ignored",
    )

    assert calls == {
        "tokenizer": tokenizer,
        "save_directory": tmp_path,
        "quantization_method": "not_quantized",
        "kwargs": {"first_conversion": "f16"},
    }


def test_bound_gguf_push_filters_kwargs(monkeypatch):
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    def fake_push_to_hub_gguf(
        model,
        tokenizer,
        save_directory,
        repo_id,
        quantization_method="fast_quantized",
        **kwargs,
    ):
        calls["tokenizer"] = tokenizer
        calls["save_directory"] = save_directory
        calls["repo_id"] = repo_id
        calls["quantization_method"] = quantization_method
        calls["kwargs"] = kwargs

    monkeypatch.setattr(mutils, "push_to_hub_gguf", fake_push_to_hub_gguf)
    tokenizer = object()
    model = types.SimpleNamespace(_tokenizer=tokenizer)

    loader._mlx_push_to_hub_gguf(
        model,
        "org/model",
        quantization_method="q8_0",
        first_conversion="bf16",
        token="hf_token",
        private=True,
        maximum_memory_usage=0.5,
        temporary_location="/tmp/ignored",
    )

    assert calls == {
        "tokenizer": tokenizer,
        "save_directory": "org/model",
        "repo_id": "org/model",
        "quantization_method": "q8_0",
        "kwargs": {
            "first_conversion": "bf16",
            "token": "hf_token",
            "private": True,
        },
    }


def test_text_generate_honors_do_sample_false(monkeypatch):
    import mlx_lm
    import mlx_lm.sample_utils as sample_utils
    import torch
    from transformers.tokenization_utils_base import to_py_obj
    import unsloth_zoo.mlx.loader as loader

    calls = {}

    class Tokenizer:
        bos_token = None
        eos_token_ids = {2}

        def encode(self, prompt, add_special_tokens=True):
            return [1, 2, 3]

    def fake_make_sampler(**kwargs):
        calls["sampler"] = kwargs
        return "sampler"

    def fake_stream_generate(_model, tokenizer, prompt, max_tokens=None, **kwargs):
        calls["prompt"] = prompt
        calls["max_tokens"] = max_tokens
        calls["stream_sampler"] = kwargs["sampler"]
        calls["eos_during_stream"] = set(tokenizer.eos_token_ids)
        yield types.SimpleNamespace(token=9, finish_reason=None)
        yield types.SimpleNamespace(token=5, finish_reason="stop")

    monkeypatch.setattr(sample_utils, "make_sampler", fake_make_sampler)
    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)

    tokenizer = Tokenizer()
    model = types.SimpleNamespace(_tokenizer=tokenizer, _is_vlm_model=False)
    out = loader._mlx_generate(
        model,
        input_ids=[[0, 1, 2, 0]],
        attention_mask=[[0, 1, 1, 0]],
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        top_k=32,
        eos_token_id=5,
        max_length=4,
    )

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    assert out.tolist() == [[1, 2, 9, 5]]
    assert out.shape == (1, 4)
    assert out[:, 2:].tolist() == [[9, 5]]
    assert to_py_obj(out) == [[1, 2, 9, 5]]
    assert calls["sampler"] == {
        "temp": 0.0,
        "top_p": 0.0,
        "min_p": 0.0,
        "top_k": 0,
    }
    assert calls["prompt"] == [1, 2]
    assert calls["max_tokens"] == 2
    assert calls["stream_sampler"] == "sampler"
    assert calls["eos_during_stream"] == {5}
    assert tokenizer.eos_token_ids == {2}


def test_mlx_generate_output_numpy_fallback_without_torch(monkeypatch):
    import builtins
    import numpy as np
    import unsloth_zoo.mlx.loader as loader

    # torch is installed here (the sim needs it), so force `import torch` to fail
    # inside _mlx_generate_output to cover the numpy int64 fallback branch. Test
    # both a missing torch (ImportError) and an installed-but-broken torch
    # (OSError) -- the broadened except must degrade to numpy in both cases.
    real_import = builtins.__import__

    def failing_import(exc):
        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise exc
            return real_import(name, *args, **kwargs)
        return _fake_import

    for exc in (ImportError("no torch"), OSError("broken torch native lib")):
        monkeypatch.setattr(builtins, "__import__", failing_import(exc))
        out = loader._mlx_generate_output([1, 2], [9, 5])
        monkeypatch.undo()
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int64
        assert out.shape == (1, 4)
        assert out.tolist() == [[1, 2, 9, 5]]
        assert out[:, 2:].tolist() == [[9, 5]]


def test_tokenizer_wrapper_chat_template_return_dict_expands_for_generate():
    import unsloth_zoo.mlx.loader as loader

    class InnerTokenizer:
        def __call__(self, *args, **kwargs):
            return {"called": True}

        def apply_chat_template(self, *args, tokenize=True, **kwargs):
            if tokenize and kwargs.get("return_dict", False):
                return {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                }
            return [1, 2, 3] if tokenize else "rendered"

    class TokenizerWrapper:
        def __init__(self):
            self._tokenizer = InnerTokenizer()

        def apply_chat_template(self, *args, tokenize=True, **kwargs):
            return [1, 2, 3] if tokenize else "rendered"

    tokenizer = TokenizerWrapper()
    loader._patch_mlx_tokenizer_call(tokenizer)

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=True,
        return_dict=True,
    )

    def expand_generate_inputs(**kwargs):
        return kwargs

    assert expand_generate_inputs(**encoded) == {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
    }
    assert encoded.to("cpu")["input_ids"] == [1, 2, 3]
    assert tokenizer.apply_chat_template([], tokenize=False, return_dict=True) == "rendered"
    assert tokenizer("hi") == {"called": True}


def test_vlm_prompt_patch_preserves_model_specific_media_markers(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader

    marker, state = "<|image_1|>Describe it.", {"result": None}
    def original(*_args, **_kwargs):
        if isinstance(state["result"], Exception):
            raise state["result"]
        return marker if state["result"] is None else state["result"]
    monkeypatch.setattr(prompt_utils, "apply_chat_template", original, raising=False)
    monkeypatch.setattr(prompt_utils, "_get_role_content", lambda item: (item["role"], item["content"]), raising=False)
    monkeypatch.setattr(prompt_utils, "get_chat_template", lambda *_a, **_k: "fallback", raising=False)
    monkeypatch.setattr(prompt_utils, "MODEL_CONFIG", {"phi3_v": object()}, raising=False)
    monkeypatch.setattr(loader, "_vlm_prompt_utils_patched", False)
    monkeypatch.setattr(loader, "_original_vlm_apply_chat_template", None)
    for name in ("mlx_vlm.chat", "mlx_vlm.generate", "mlx_vlm.generate.dispatch", "mlx_vlm.generate.ar", "mlx_vlm.server", "mlx_vlm.evals.utils"):
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], "apply_chat_template", original, raising=False)
    loader._ensure_vlm_prompt_utils_patched()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}]}]
    render = lambda value, model_type="phi3_v", **kwargs: prompt_utils.apply_chat_template(object(), {"model_type": model_type}, value, **kwargs)

    assert render([{"role": "user", "content": [{"type": "audio"}]}], num_audios=2) == "fallback"
    assert render(messages, num_images=2) == "fallback"
    assert render(messages, model_type="unknown", num_images=1) == "fallback"
    assert render(messages + [{"role": "user", "content": "Again."}], num_images=1) == "fallback"
    nested = [{"role": "user", "content": [{"type": "group", "content": messages[0]["content"]}]}]
    assert render(nested, num_images=1) == "fallback"
    nested_text = [{"type": "group", "content": [{"type": "text", "text": "Policy"}]}]
    assert render([{"role": "system", "content": nested_text}] + messages, num_images=1) == "fallback"
    assert render([{"role": "User", "content": messages[0]["content"]}], num_images=1) == "fallback"
    tool_calls = [{"function": {"name": "inspect", "arguments": '{"detail":"full"}'}}]
    normalized_calls = [{"function": {"name": "inspect", "arguments": {"detail": "full"}}}]
    monkeypatch.setattr(prompt_utils, "_normalize_tool_call_arguments", lambda message: {**message, **({"tool_calls": normalized_calls} if "tool_calls" in message else {})}, raising=False)
    tool_messages = [{**messages[0], "tool_calls": tool_calls}]
    assert render(tool_messages, num_images=1, return_messages=True) == [{**messages[0], "tool_calls": normalized_calls}]
    assert tool_messages[0]["tool_calls"] == tool_calls
    assert render([{"role": "user", "content": [{"type": "IMAGE"}, {"type": "TEXT", "text": "Describe it."}]}], num_images=1) == "fallback"
    for video_type in ("video", "input_video", "video_url"):
        assert render([{"role": "user", "content": [{"type": video_type}]}]) == "fallback"
    assert render(messages, num_images=1, video="clip.mp4") == "fallback"
    assert render(messages, num_images=1, return_messages=True) == messages
    for state["result"] in ("", ValueError("rejected")):
        assert render(messages, num_images=1) == "fallback"
    monkeypatch.setattr(prompt_utils, "extract_text_from_content", lambda content: content, raising=False)
    monkeypatch.setattr(prompt_utils, "get_message_json", lambda *_args, **_kwargs: "anchored", raising=False)
    string_messages = ["Plain.", {"role": "user", "content": "Plain dict."}, {"role": "user", "content": "Describe.", "name": "owner"}, {"role": "assistant", "content": "", "tool_calls": tool_calls}]
    expected = ["anchored", "anchored", {**string_messages[2], "content": "anchored"}, {**string_messages[3], "content": "anchored", "tool_calls": normalized_calls}]
    assert render(string_messages, num_images=1, return_messages=True) == expected


def test_vlm_prompt_patch_rebinds_every_loaded_mlx_vlm_alias(monkeypatch):
    """No loaded mlx-vlm module may keep the original chat-template callable.

    The hard-coded list only force-imports mlx-vlm's entry points, so aliases in
    modules that are already loaded -- `mlx_vlm` itself re-exports this one --
    survive the patch and render multi-turn prompts the old way.
    """
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader

    def original(*_args, **_kwargs):
        return "original"

    monkeypatch.setattr(prompt_utils, "apply_chat_template", original, raising=False)
    monkeypatch.setattr(loader, "_vlm_prompt_utils_patched", False)
    monkeypatch.setattr(loader, "_original_vlm_apply_chat_template", None)

    # Aliases mlx-vlm really holds: the package re-export and a submodule that
    # any `import mlx_vlm` already pulls in through mlx_vlm/trainer/__init__.py.
    aliased = ("mlx_vlm", "mlx_vlm.trainer.datasets")
    for name in aliased:
        module = sys.modules.get(name) or types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(module, "apply_chat_template", original, raising=False)

    # A same-named alias outside mlx-vlm must stay untouched.
    outsider = types.ModuleType("mlx_vlm_extension")
    outsider.apply_chat_template = original
    monkeypatch.setitem(sys.modules, outsider.__name__, outsider)

    loader._ensure_vlm_prompt_utils_patched()

    patched = prompt_utils.apply_chat_template
    assert patched is not original
    stale = [name for name in aliased if sys.modules[name].apply_chat_template is original]
    assert stale == [], f"stale mlx-vlm chat-template aliases: {stale}"
    assert outsider.apply_chat_template is original


def _install_qwen_prompt_patch(monkeypatch, prompt_utils, loader, **overrides):
    attrs = {
        "apply_chat_template": lambda *_args, **_kwargs: "count-rendered",
        "get_message_json": lambda _model, text, role="user", **_kwargs: {
            "role": role, "content": [{"type": "text", "text": text}],
        },
        "_get_role_content": lambda item: (item["role"], item["content"]),
        "extract_text_from_content": lambda content: content,
        "MODEL_CONFIG": {"qwen3_omni_moe": object()},
        **overrides,
    }
    for name, value in attrs.items():
        monkeypatch.setattr(prompt_utils, name, value, raising=False)
    monkeypatch.setattr(loader, "_vlm_prompt_utils_patched", False)
    monkeypatch.setattr(loader, "_original_vlm_apply_chat_template", None)
    loader._ensure_vlm_prompt_utils_patched()


def _render_qwen(prompt_utils, prompt, processor=None, **kwargs):
    processor = object() if processor is None else processor
    config = {"model_type": "qwen3_omni_moe"}
    return prompt_utils.apply_chat_template(processor, config, prompt, **kwargs)


def test_vlm_prompt_patch_matches_published_model_type_case_insensitively(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader
    configured = "nemotronh_nano_omni_reasoning_v3"
    calls = []
    def original(_processor, config, prompt, **kwargs):
        model_type = config["model_type"]
        calls.append((model_type, prompt, kwargs.get("num_audios")))
        return "configured" if model_type in prompt_utils.MODEL_CONFIG else "text-only"
    _install_qwen_prompt_patch(monkeypatch, prompt_utils, loader,
                               apply_chat_template=original,
                               MODEL_CONFIG={configured: object()})
    rendered = prompt_utils.apply_chat_template(object(), {
        "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
    }, "Transcribe this audio.", num_audios=1)
    assert rendered == "configured"
    assert calls == [(configured, "Transcribe this audio.", 1)]


def test_vlm_prompt_patch_places_counted_qwen3_omni_audio_before_text(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader
    rendered_messages = []
    def render(_processor, messages, _add_generation_prompt, **_kwargs):
        rendered_messages.append(messages)
        return "structured-rendered"
    _install_qwen_prompt_patch(
        monkeypatch, prompt_utils, loader, get_chat_template=render,
    )
    result = _render_qwen(prompt_utils, "Transcribe the audio into text.", num_audios=1)
    assert result == "structured-rendered"
    assert [x["type"] for x in rendered_messages[0][0]["content"]] == ["audio", "text"]


def test_vlm_prompt_patch_preserves_qwen3_omni_video_with_audio(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader
    def video_message(_model_type, text, **kwargs):
        video = {"type": "video", "video": kwargs["video"], "fps": kwargs["fps"]}
        return {"role": "user", "content": [video, {"type": "text", "text": text}]}
    _install_qwen_prompt_patch(
        monkeypatch, prompt_utils, loader, get_message_json=video_message,
    )
    messages = _render_qwen(prompt_utils, "Describe both inputs.",
                            return_messages=True, num_audios=1,
                            video="clip.mp4", fps=2)
    assert [x["type"] for x in messages[0]["content"]] == ["video", "audio", "text"]
    assert messages[0]["content"][0] == {"type": "video", "video": "clip.mp4", "fps": 2}


def test_vlm_prompt_patch_honors_qwen3_omni_media_suppression(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader
    _install_qwen_prompt_patch(monkeypatch, prompt_utils, loader)
    text = "Describe the input."
    counted = {"role": "user", "content": text}
    cases = (
        (text, {"skip_audio_token": True}, "user", ["image", "text"]),
        (text, {"skip_image_token": True}, "user", ["audio", "text"]),
        (text, {"role": "assistant"}, "assistant", ["text"]),
        (counted, {"skip_audio_token": True}, "user", ["image", "text"]),
        (counted, {"skip_image_token": True}, "user", ["audio", "text"]),
    )
    for prompt, options, role, expected_types in cases:
        message = _render_qwen(prompt_utils, prompt, return_messages=True,
                               num_images=1, num_audios=1, **options)[0]
        assert message["role"] == role
        assert [item["type"] for item in message["content"]] == expected_types


def test_vlm_prompt_patch_preserves_structured_qwen3_omni_media_order(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader

    rendered_messages = []

    def counted_message(_model, text, role="user", **kwargs):
        videos = [{"type": "video", "video": kwargs["video"]}] if kwargs.get("video") else []
        return {"role": role, "content": videos + [{"type": "text", "text": text}]}

    _install_qwen_prompt_patch(
        monkeypatch, prompt_utils, loader,
        get_chat_template=(
            lambda _processor, messages, _add_generation_prompt, **_kwargs:
                rendered_messages.append(messages) or "structured-rendered"
        ),
        get_message_json=counted_message,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_video", "video": "clip.mp4"},
                {"type": "video"},
                {"type": "image", "image": "frame.png"},
                {"type": "image"},
                {"type": "input_audio", "audio": "first.wav"},
                {"audio": "key-only.wav"},
                {"type": "audio"},
                {
                    "type": "group",
                    "content": [
                        {"type": "input_audio", "audio": "nested.wav"},
                    ],
                },
                {"type": "text", "text": "Transcribe the audio into text."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image": "second.png"},
                {"type": "input_audio", "audio": "second.wav"},
                {"type": "text", "text": "Continue."},
            ],
        },
    ]
    result = _render_qwen(prompt_utils, messages, num_images=4, num_audios=5)

    plain_prompt = [
        {"role": "user", "content": "First."},
        messages[1],
    ]
    plain = _render_qwen(
        prompt_utils,
        plain_prompt,
        return_messages=True,
        num_images=2,
        num_audios=2,
    )
    key_only_prompt = [
        {
            "role": "user",
            "content": [
                {"audio": "only.wav", "metadata": {"source": "fixture"}},
                {"audio_url": "second.wav"},
                {"type": "text", "text": "Keep me."},
            ],
        },
    ]
    key_only = _render_qwen(
        prompt_utils,
        key_only_prompt,
        return_messages=True,
        num_audios=2,
    )
    complete_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe."},
                {"type": "audio"},
            ],
        },
    ]
    complete = _render_qwen(
        prompt_utils,
        complete_prompt,
        return_messages=True,
        num_audios=1,
    )
    nested_prompt = [
        {
            "role": "user",
            "content": (
                messages[0]["content"][4:5]
                + messages[0]["content"][7:]
            ),
        },
    ]
    nested = _render_qwen(
        prompt_utils,
        nested_prompt,
        return_messages=True,
        num_audios=2,
    )

    assert [item["type"] for item in plain[0]["content"]] == [
        "image",
        "audio",
        "text",
    ]
    assert plain[0]["content"][-1]["text"] == "First."
    assert plain[1] == messages[1]
    assert key_only == key_only_prompt
    assert [item["type"] for item in complete[0]["content"]] == [
        "audio",
        "text",
    ]
    assert [item["type"] for item in nested[0]["content"]] == [
        "input_audio",
        "audio",
        "group",
        "text",
    ]
    assert _render_qwen(
        prompt_utils,
        plain_prompt,
        return_messages=True,
        num_images=1,
        num_audios=1,
    ) == plain_prompt
    assert _render_qwen(
        prompt_utils,
        plain_prompt,
        return_messages=True,
        num_images=2,
        num_audios=2,
        skip_image_token=True,
        skip_audio_token=True,
    ) == plain_prompt

    counted = {"role": "user", "content": "Transcribe the audio."}
    for prompt in (counted, [counted]):
        _render_qwen(prompt_utils, prompt, num_audios=1)
    conversation = [
        {"role": "system", "content": "Follow instructions."},
        {"role": "HuMaN", "content": "Describe all inputs."},
        {"role": "assistant", "content": "Ready."},
    ]
    anchored = _render_qwen(prompt_utils, conversation, return_messages=True,
                            num_images=1, num_audios=1, video="clip.mp4")

    assert result == "structured-rendered"
    assert rendered_messages[0][0]["content"] == (
        messages[0]["content"][:4]
        + [{"type": "image"}]
        + messages[0]["content"][4:7]
        + [{"type": "audio"}]
        + messages[0]["content"][7:]
    )
    assert rendered_messages[0][1] == messages[1]
    assert len(rendered_messages) == 3
    for rendered in rendered_messages[1:]:
        assert [item["type"] for item in rendered[0]["content"]] == ["audio", "text"]
    assert [[item["type"] for item in turn["content"]] for turn in anchored] == [
        ["text"], ["video", "image", "audio", "text"], ["text"],
    ]


def test_vlm_prompt_patch_uses_qwen3_omni_native_non_thinking_template(monkeypatch):
    import mlx_vlm.prompt_utils as prompt_utils
    import unsloth_zoo.mlx.loader as loader
    native_calls = []
    class Processor:
        def apply_chat_template(
            self, _messages, *, tokenize, add_generation_prompt, **kwargs,
        ):
            native_calls.append((tokenize, kwargs))
            return "native-rendered"
    _install_qwen_prompt_patch(
        monkeypatch, prompt_utils, loader,
        get_chat_template=lambda *_args, **_kwargs: "generic-rendered",
    )
    prompt = "Transcribe the audio into text."
    result = _render_qwen(prompt_utils, prompt, Processor(), num_audios=1)
    explicit_result = _render_qwen(prompt_utils, prompt, Processor(),
                                   num_audios=1, enable_thinking=True)
    tokenized_result = _render_qwen(prompt_utils, prompt, Processor(),
                                    num_audios=1, tokenize=True)
    assert (result, explicit_result, tokenized_result) == ("native-rendered",) * 3
    assert native_calls == [(False, {}), (False, {"enable_thinking": True}), (True, {})]


def test_vlm_generate_hf_kwargs(monkeypatch):
    import torch
    from transformers.tokenization_utils_base import to_py_obj
    import unsloth_zoo.mlx.loader as loader

    fake_mlx_vlm = types.ModuleType("mlx_vlm")
    calls = []

    def fake_stream_generate(_model, _processor, _prompt, max_tokens=None, **batch):
        calls.append((max_tokens, batch))
        return iter(())

    fake_mlx_vlm.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)

    model = types.SimpleNamespace(
        _tokenizer=types.SimpleNamespace(tokenizer=object()),
        _is_vlm_model=True,
        config=types.SimpleNamespace(eos_token_id=None),
    )
    out = loader._mlx_generate(
        model,
        input_ids=[1, 2],
        attention_mask=[1, 1],
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=1,
    )

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    assert out.tolist() == [[1, 2]]
    assert out.shape == (1, 2)
    assert to_py_obj(out) == [[1, 2]]
    assert calls[0][0] == 1
    assert tuple(calls[0][1]["input_ids"].shape) == (1, 2)
    assert tuple(calls[0][1]["mask"].shape) == (1, 2)
    assert calls[0][1]["temperature"] == 0.0
    assert "top_p" not in calls[0][1]


def test_bound_save_pretrained_defaults_to_full_save_without_lora(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    def fake_save_pretrained_merged(model, tokenizer, save_directory, **kwargs):
        calls["tokenizer"] = tokenizer
        calls["save_directory"] = Path(save_directory)
        calls["kwargs"] = kwargs

    monkeypatch.setattr(mutils, "collect_mlx_lora_adapter_tensors", lambda model: {})
    monkeypatch.setattr(mutils, "save_pretrained_merged", fake_save_pretrained_merged)

    tokenizer = object()
    model = types.SimpleNamespace(_tokenizer=tokenizer)
    loader._mlx_save_pretrained_merged(
        model,
        tmp_path,
        safe_serialization=True,
        save_peft_format=True,
        save_embedding_layers="auto",
        path_initial_model_for_weight_conversion="/tmp/base",
        token="hf_token",
    )

    assert calls == {
        "tokenizer": tokenizer,
        "save_directory": tmp_path,
        "kwargs": {"save_method": "merged_16bit", "token": "hf_token"},
    }


def test_adapter_bnb_base_remap_defaults_to_mlx_4bit_quantization(
    monkeypatch,
    tmp_path,
):
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unsloth/tinyllama-bnb-4bit",
            }
        ),
        encoding="utf-8",
    )

    calls = {}
    original_from_pretrained = loader.FastMLXModel.from_pretrained

    class StopAdapterLoad(RuntimeError):
        pass

    # Accept allow_patterns: from_pretrained's config download now forwards it,
    # so the shim must match the real _download signature.
    def fake_download(model_name, revision=None, allow_patterns=None):
        assert model_name == str(adapter_dir)
        return adapter_dir

    def fake_recursive_from_pretrained(model_name, **kwargs):
        calls["model_name"] = model_name
        calls["kwargs"] = kwargs
        raise StopAdapterLoad("Unsloth: stop after recursive adapter base load")

    monkeypatch.setattr(mlx_lm_utils, "_download", fake_download)
    monkeypatch.setattr(
        loader.FastMLXModel,
        "from_pretrained",
        staticmethod(fake_recursive_from_pretrained),
    )

    with pytest.raises(StopAdapterLoad):
        original_from_pretrained(str(adapter_dir))

    assert calls["model_name"] == "unsloth/tinyllama"
    assert calls["kwargs"]["load_in_4bit"] is False
    assert calls["kwargs"]["mlx_quantization_config"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }


def test_lora_push_uses_lora_adapter_hub_path(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    class Model:
        def named_modules(self):
            return [("layers.0.q_proj", types.SimpleNamespace(fuse=lambda: None))]

        def trainable_parameters(self):
            return {}

    class Tokenizer:
        def save_pretrained(self, path):
            calls["tokenizer_path"] = Path(path)

    def fake_save_lora_adapters(model, save_directory):
        calls["adapter_dir"] = Path(save_directory)

    def fake_push_lora_adapters_to_hub(
        save_directory,
        **kwargs,
    ):
        calls["hub_dir"] = Path(save_directory)
        calls["hub_kwargs"] = kwargs

    monkeypatch.setattr(
        mutils,
        "collect_mlx_lora_adapter_tensors",
        lambda model: {"layers.0.q_proj.lora_a": object()},
    )
    monkeypatch.setattr(mutils, "iter_mlx_lora_modules", lambda model: [])
    monkeypatch.setattr(mutils, "save_lora_adapters", fake_save_lora_adapters)
    monkeypatch.setattr(
        mutils,
        "_push_lora_adapters_to_hub",
        fake_push_lora_adapters_to_hub,
    )
    monkeypatch.setattr(
        mutils,
        "push_to_hub_merged",
        lambda *args, **kwargs: pytest.fail("push_to_hub_merged should not run"),
    )

    mutils.save_pretrained_merged(
        Model(),
        Tokenizer(),
        tmp_path,
        save_method="lora",
        push_to_hub=True,
        token="hf_token",
        private=True,
    )

    assert calls["adapter_dir"] == tmp_path
    assert calls["hub_dir"] == tmp_path
    assert calls["hub_kwargs"]["repo_id"] is None
    assert calls["hub_kwargs"]["token"] == "hf_token"
    assert calls["hub_kwargs"]["private"] is True


def _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils):
    import torch

    monkeypatch.setattr(
        mutils.mx,
        "transpose",
        lambda tensor, axes=None, **kwargs: tensor.permute(*axes)
        if axes is not None
        else tensor.permute(*reversed(range(tensor.ndim))),
    )
    monkeypatch.setattr(mutils.mx, "all", torch.all)


@pytest.mark.parametrize(
    ("mlx_name", "hf_name"),
    [
        ("audio_tower.weight", "model.audio_tower.weight"),
        ("vision_tower.weight", "model.vision_tower.weight"),
        ("embed_audio.weight", "model.embed_audio.weight"),
        ("embed_vision.weight", "model.embed_vision.weight"),
        # The projector shares the encoder's namespace; a converter that takes only the
        # canonical name drops it, and the mmproj it writes then holds the encoder alone
        # and is refused at load for the missing projector tensor.
        ("vision_adapter.fc1.weight", "model.vision_adapter.fc1.weight"),
        ("vision_adapter.fc2.weight", "model.vision_adapter.fc2.weight"),
        ("vision_projection.weight", "model.vision_projection.weight"),
    ],
)
def test_vlm_gguf_candidates_prefer_canonical_model_namespace(mlx_name, hf_name):
    import unsloth_zoo.mlx.utils as mutils
    candidates = mutils._vlm_gguf_name_candidates(mlx_name)
    assert candidates[0] == hf_name
    assert candidates.index(hf_name) < candidates.index(mlx_name)


def test_vlm_rewrite_restores_namespace_with_conv1d_layout(monkeypatch):
    import torch
    import unsloth_zoo.mlx.utils as mutils

    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class StripNamespaceAndTransposeConv1d:
        @staticmethod
        def sanitize(weights):
            return {
                name.removeprefix("model."): mutils.mx.transpose(tensor, (0, 2, 1))
                for name, tensor in weights.items()
            }

    mlx_layout = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    new_name, hf_layout, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight",
        mlx_layout,
        [(StripNamespaceAndTransposeConv1d, None)],
    )

    assert changed is True
    assert new_name == "model.audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"
    assert tuple(hf_layout.shape) == (2, 4, 3)
    assert mutils._mlx_arrays_match(
        mutils.mx.transpose(hf_layout, (0, 2, 1)),
        mlx_layout,
    )


def test_vlm_rewrite_prefers_hf_alias_before_current_name(monkeypatch):
    import torch
    import unsloth_zoo.mlx.utils as mutils

    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class QwenSanitizer:
        @staticmethod
        def sanitize(weights):
            renamed = {}
            for name, tensor in weights.items():
                if name.startswith("visual."):
                    name = f"vision_tower.{name[len('visual.'):]}"
                renamed[name] = tensor
            return renamed

    tensor = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.patch_embed.proj.weight",
        tensor,
        [(QwenSanitizer, None)],
    )

    assert changed is True
    assert new_name == "visual.patch_embed.proj.weight"
    assert mutils._mlx_arrays_match(new_tensor, tensor)


_MTP_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

_MTP_SHIFTED_KEYS = (
    "model.layers.0.input_layernorm.weight",
    "model.layers.1.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
    "model.norm.weight",
)


class _MtpGatedNormSanitizer:
    """Shaped like the MTP families': an MTP shard or a still-unsanitized
    conv1d marks the source as HF convention, which one tensor cannot show."""

    def __init__(self, src_path):
        self._src_path = str(src_path)

    def sanitize(self, weights):
        shift = any("mtp." in key for key in weights) or any(
            "conv1d.weight" in key and value.shape[-1] != 1
            for key, value in weights.items()
        )
        sanitized = {}
        for key, value in weights.items():
            if "mtp." in key:
                continue
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if shift and value.ndim == 1 and key.endswith(_MTP_NORM_SUFFIXES):
                value = value + 1.0
            sanitized[key] = value
        return sanitized


def _mtp_source_weights(mx, *, gate):
    """``gate`` selects what puts the source in HF convention, if anything."""
    weights = {key: mx.array([0.5, 0.25]) for key in _MTP_SHIFTED_KEYS}
    weights.update({
        # Norm-named but unshifted: catches a correction keyed on the name.
        "model.layers.0.linear_attn.norm.weight": mx.array([0.5, 0.25]),
        "vision_tower.blocks.0.norm1.weight": mx.array([0.5, 0.25]),
        # 1-D float but not a norm: catches a blanket 1-D correction.
        "model.layers.0.linear_attn.dt_bias": mx.array([0.5, 0.25]),
        "model.layers.0.self_attn.q_proj.weight": mx.array([[0.5, 0.25]]),
    })
    if gate == "mtp":
        weights["mtp.0.input_layernorm.weight"] = mx.array([0.5, 0.25])
    elif gate == "conv1d":
        weights["model.layers.0.linear_attn.conv1d.weight"] = mx.zeros((2, 1, 4))
    return weights


def _write_weights(mx, directory, weights):
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(directory / "model.safetensors"), weights)


@pytest.mark.parametrize("gate", ["mtp", "conv1d", None])
def test_norm_offsets_follow_the_source_checkpoint_gate(tmp_path, gate):
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    src = tmp_path / "src"
    _write_weights(mx, src, _mtp_source_weights(mx, gate=gate))

    offsets = mutils._mlx_sanitizer_norm_offsets(_MtpGatedNormSanitizer(src))

    expected = dict.fromkeys(_MTP_SHIFTED_KEYS, 1.0) if gate else {}
    assert offsets == expected


@pytest.mark.parametrize("gate", ["mtp", "conv1d", None])
def test_gguf_export_restores_the_hf_norm_convention(tmp_path, gate):
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    src = tmp_path / "src"
    source_weights = _mtp_source_weights(mx, gate=gate)
    _write_weights(mx, src, source_weights)
    model = _MtpGatedNormSanitizer(src)

    export = tmp_path / "export"
    merged = model.sanitize(dict(source_weights))
    _write_weights(mx, export, merged)
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    rewritten = mutils._prepare_mlx_gguf_export_directory(
        export, model=model, replay_sanitizers=False
    )

    assert rewritten == (len(_MTP_SHIFTED_KEYS) if gate else 0)
    exported = mx.load(str(export / "model.safetensors"))
    assert sorted(exported) == sorted(merged)
    for key, value in exported.items():
        # conv1d is relayouted rather than offset.
        expected = source_weights[key]
        if "conv1d.weight" in key:
            expected = merged[key]
        assert mutils._mlx_arrays_match(value, expected), key


def test_gguf_export_does_not_subtract_a_replayed_offset_twice(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class RenamingNormSanitizer:
        """Renames and shifts, so the single-tensor replay recovers the value
        on its own and the measured offset must not be applied again."""

        def __init__(self, src_path):
            self._src_path = str(src_path)

        def sanitize(self, weights):
            sanitized = {}
            for key, value in weights.items():
                key = key.replace("model.", "renamed.", 1)
                if value.ndim == 1 and key.endswith(_MTP_NORM_SUFFIXES):
                    value = value + 1.0
                sanitized[key] = value
            return sanitized

    src = tmp_path / "src"
    source_weights = {
        "model.layers.0.input_layernorm.weight": mx.array([0.5, 0.25]),
        "model.layers.0.linear_attn.dt_bias": mx.array([0.5, 0.25]),
    }
    _write_weights(mx, src, source_weights)
    model = RenamingNormSanitizer(src)

    export = tmp_path / "export"
    _write_weights(mx, export, model.sanitize(dict(source_weights)))
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    mutils._prepare_mlx_gguf_export_directory(export, model=model)

    exported = mx.load(str(export / "model.safetensors"))
    assert mutils._mlx_arrays_match(
        exported["renamed.layers.0.input_layernorm.weight"],
        source_weights["model.layers.0.input_layernorm.weight"],
    )
    assert mutils._mlx_arrays_match(
        exported["renamed.layers.0.linear_attn.dt_bias"],
        source_weights["model.layers.0.linear_attn.dt_bias"],
    )


class _KeyGatedNormSanitizer:
    """Shaped like the Qwen3.5 family's: the RMSNorm shift is decided by the
    pre-sanitize key, so it survives a checkpoint that was already converted.

    ``shift`` is parametrized rather than fixed at 1.0 so the tests can tell a
    measured offset apart from the 1.0 the export falls back to.
    """

    def __init__(self, src_path, shift=1.0):
        self._src_path = str(src_path)
        self._shift = shift

    def sanitize(self, weights):
        sanitized = {}
        for original_key, value in weights.items():
            key = original_key
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model."):]
            if (
                value.ndim == 1
                and key.endswith(_MTP_NORM_SUFFIXES)
                and not original_key.startswith("language_model.")
            ):
                value = value + self._shift
            sanitized[key] = value
        return sanitized


@pytest.mark.parametrize("source_convention", ["hf", "mlx"])
def test_gguf_export_unshifts_an_already_converted_source(tmp_path, source_convention):
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    hf_weights = {
        "model.language_model.layers.0.input_layernorm.weight": mx.array([0.5, 0.25]),
        "model.language_model.norm.weight": mx.array([0.125, 0.75]),
        "model.language_model.layers.0.linear_attn.dt_bias": mx.array([0.5, 0.25]),
    }
    model_of = _KeyGatedNormSanitizer

    src = tmp_path / "src"
    if source_convention == "hf":
        _write_weights(mx, src, hf_weights)
    else:
        # An mlx-community style checkpoint: already sanitized, so reloading it
        # shifts nothing and the measurement comes back bare.
        _write_weights(mx, src, model_of(src).sanitize(dict(hf_weights)))
    model = model_of(src)

    if source_convention == "mlx":
        assert mutils._mlx_sanitizer_norm_offsets(model) == {}

    export = tmp_path / "export"
    _write_weights(mx, export, model.sanitize(dict(hf_weights)))
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    mutils._prepare_mlx_gguf_export_directory(export, model=model)

    exported = mx.load(str(export / "model.safetensors"))
    for key, expected in hf_weights.items():
        assert key in exported, key
        assert mutils._mlx_arrays_match(exported[key], expected), key


def test_gguf_export_removes_a_measured_offset_that_is_not_one(tmp_path):
    # The export threads the MEASURED offset into the replay candidates, and a
    # hardcoded ``- 1`` reproduces every 1.0 case. Pin a shift that is not 1.0:
    # there the wrong constant makes no candidate round-trip, so the tensor
    # keeps its MLX name and never reaches the export.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    hf_weights = {
        "model.language_model.layers.0.input_layernorm.weight": mx.array([0.5, 0.25]),
        "model.language_model.norm.weight": mx.array([0.125, 0.75]),
    }
    src = tmp_path / "src"
    _write_weights(mx, src, hf_weights)
    model = _KeyGatedNormSanitizer(src, shift=0.5)

    assert mutils._mlx_sanitizer_norm_offsets(model) == {
        "language_model.model.layers.0.input_layernorm.weight": 0.5,
        "language_model.model.norm.weight": 0.5,
    }

    export = tmp_path / "export"
    _write_weights(mx, export, model.sanitize(dict(hf_weights)))
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    mutils._prepare_mlx_gguf_export_directory(export, model=model)

    exported = mx.load(str(export / "model.safetensors"))
    for key, expected in hf_weights.items():
        assert key in exported, key
        assert mutils._mlx_arrays_match(exported[key], expected), key


class _SynthesizedScaleSanitizer:
    """Shaped like mlx-vlm's Inkling: the sanitizer CREATES 1-D scale vectors
    that the source checkpoint never held, filled with ones, under names that
    are real model parameters. They are constant and non-zero, so a measurement
    that only looks at its own output reads them as an added offset."""

    def __init__(self, src_path):
        self._src_path = str(src_path)

    def sanitize(self, weights):
        import unsloth_zoo.mlx.utils as mutils

        out = dict(weights)
        for key in [k for k in out if k.endswith("switch_mlp.gate_proj.weight")]:
            prefix = key[: -len("gate_proj.weight")]
            ones = mutils.mx.ones((out[key].shape[0],))
            out.setdefault(prefix + "gate_scale", ones)
            out.setdefault(prefix + "out_scale", ones)
        return out


_SWITCH_MLP = "language_model.model.layers.0.mlp.switch_mlp."


def test_norm_offsets_ignore_a_sanitizer_created_constant(tmp_path):
    # A created tensor is not an offset: nothing in the source was shifted to
    # produce it, so subtracting it annihilates a real parameter.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    src = tmp_path / "src"
    _write_weights(mx, src, {_SWITCH_MLP + "gate_proj.weight": mx.zeros((2, 3))})

    assert mutils._mlx_sanitizer_norm_offsets(_SynthesizedScaleSanitizer(src)) == {}


@pytest.mark.parametrize("replay_sanitizers", [True, False])
def test_gguf_export_keeps_sanitizer_created_scales(tmp_path, replay_sanitizers):
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    src = tmp_path / "src"
    _write_weights(mx, src, {_SWITCH_MLP + "gate_proj.weight": mx.zeros((2, 3))})
    model = _SynthesizedScaleSanitizer(src)

    export = tmp_path / "export"
    merged = {
        _SWITCH_MLP + "gate_proj.weight": mx.zeros((2, 3)),
        _SWITCH_MLP + "gate_scale": mx.array([1.0, 1.0]),
        _SWITCH_MLP + "out_scale": mx.array([1.0, 1.0]),
    }
    _write_weights(mx, export, merged)
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    mutils._prepare_mlx_gguf_export_directory(
        export, model=model, replay_sanitizers=replay_sanitizers
    )

    exported = mx.load(str(export / "model.safetensors"))
    for key, expected in merged.items():
        assert mutils._mlx_arrays_match(exported[key], expected), key


def test_norm_offsets_ignore_a_non_additive_transform(tmp_path):
    # ``A_log = -exp(A_log)`` maps a zeroed probe to a constant -1.0. Reading
    # that as an offset would shift a real state-space parameter by +1.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class NegExpSanitizer:
        def __init__(self, src_path):
            self._src_path = str(src_path)

        def sanitize(self, weights):
            return {
                key: (-mx.exp(value) if key.endswith("A_log") else value)
                for key, value in weights.items()
            }

    src = tmp_path / "src"
    _write_weights(mx, src, {
        "model.layers.0.mixer.A_log": mx.array([0.5, 0.25]),
        "model.layers.0.input_layernorm.weight": mx.array([0.5, 0.25]),
    })

    assert mutils._mlx_sanitizer_norm_offsets(NegExpSanitizer(src)) == {}


def test_norm_offsets_do_not_mutate_the_model(tmp_path):
    # Real sanitizers write to ``self``: mlx-vlm's phi4mm caches its LoRA
    # weights, and mlx-lm's gemma3_text drops the lm_head submodule for a tied
    # checkpoint. Measuring must not leave the caller holding a model rebuilt
    # from the all-zero probe.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class SelfMutatingSanitizer(dict):
        def __init__(self, src_path):
            super().__init__({"lm_head": "the real lm_head module"})
            self._src_path = str(src_path)
            self.cached_weights = {"trained": "weights"}

        def sanitize(self, weights):
            self.cached_weights = dict(weights)
            self.pop("lm_head", None)
            return {
                key: (value + 1.0 if value.ndim == 1 else value)
                for key, value in weights.items()
            }

    src = tmp_path / "src"
    _write_weights(mx, src, {"model.norm.weight": mx.array([0.5, 0.25])})
    model = SelfMutatingSanitizer(src)

    assert mutils._mlx_sanitizer_norm_offsets(model) == {"model.norm.weight": 1.0}

    assert model.cached_weights == {"trained": "weights"}
    assert model["lm_head"] == "the real lm_head module"


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
def test_already_converted_recovery_is_exact_in_low_precision(tmp_path, dtype_name):
    # The recovery accepts ``tensor - 1.0`` only if replaying the sanitizer
    # reproduces the stored tensor EXACTLY. ``(t - c) + c == t`` is not a
    # floating-point identity, but the stored tensor is itself ``source + c`` in
    # this dtype, and across every finite bfloat16 and float16 value the only
    # sources that fail are |source| = 258 and 2050. So the exact check is right
    # and must not be loosened into a tolerance.
    #
    # These values span the range the shifting families occupy (the published
    # Qwen3-Next input_layernorm runs -0.154 to 0.781). Above |source| = 1 the
    # low bit is gone before Unsloth sees the checkpoint, so the recovery
    # returns what mlx-lm's load left, which is what inference runs on too.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    dtype = getattr(mx, dtype_name)
    source = mx.array([-0.953125, -0.7, 0.046875, 0.25, 0.5, 0.78125], dtype=dtype)
    hf_weights = {"model.language_model.norm.weight": source}

    src = tmp_path / "src"
    # An mlx-community style checkpoint: already sanitized on disk, so the
    # measurement is empty and only the replay can recover the shift.
    _write_weights(mx, src, _KeyGatedNormSanitizer(src).sanitize(dict(hf_weights)))
    model = _KeyGatedNormSanitizer(src)
    assert mutils._mlx_sanitizer_norm_offsets(model) == {}

    export = tmp_path / "export"
    _write_weights(mx, export, model.sanitize(dict(hf_weights)))
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    assert mutils._prepare_mlx_gguf_export_directory(export, model=model) == 1

    exported = mx.load(str(export / "model.safetensors"))
    recovered = exported["model.language_model.norm.weight"]
    assert mutils._mlx_arrays_match(recovered, source), recovered


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_norm_offsets_reject_a_non_finite_constant(tmp_path, bad):
    # The probe hands zeros to a third-party sanitizer, so a division by a
    # weight it zeroed comes back inf or NaN. NaN fails every comparison, so an
    # unguarded spread check reads it as a constant and the export writes it
    # into the real tensor.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class DividingSanitizer:
        def __init__(self, src_path):
            self._src_path = str(src_path)

        def sanitize(self, weights):
            out = dict(weights)
            out["model.layers.0.normed"] = (
                weights["model.layers.0.normed"] / weights["model.layers.0.scale"]
            )
            return out

    numerator = 0.0 if bad == "nan" else 4.0
    weights = {
        "model.layers.0.scale": mx.array([2.0, 2.0]),
        "model.layers.0.normed": mx.array([numerator, numerator]),
    }
    src = tmp_path / "src"
    _write_weights(mx, src, weights)
    model = DividingSanitizer(src)

    assert mutils._mlx_sanitizer_norm_offsets(model) == {}

    export = tmp_path / "export"
    _write_weights(mx, export, weights)
    (export / "config.json").write_text(json.dumps({"model_type": "stub"}))

    assert mutils._prepare_mlx_gguf_export_directory(
        export, model=model, replay_sanitizers=False
    ) == 0
    exported = mx.load(str(export / "model.safetensors"))
    for key, expected in weights.items():
        assert mutils._mlx_arrays_match(exported[key], expected), key


def test_norm_offsets_fail_closed_when_the_model_cannot_be_copied(tmp_path):
    # Falling back to the live instance would sanitize the model this exists to
    # protect. Unmeasurable is the safe answer: it is what the export did before
    # any of this landed.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class Uncopyable:
        def __init__(self, src_path):
            self._src_path = str(src_path)
            self.cached_weights = {"trained": "weights"}

        def __copy__(self):
            raise TypeError("this model cannot be shallow-copied")

        def sanitize(self, weights):
            self.cached_weights = dict(weights)
            return {
                key: (value + 1.0 if value.ndim == 1 else value)
                for key, value in weights.items()
            }

    src = tmp_path / "src"
    _write_weights(mx, src, {"model.norm.weight": mx.array([0.5, 0.25])})
    model = Uncopyable(src)

    assert mutils._mlx_sanitizer_norm_offsets(model) is None
    assert model.cached_weights == {"trained": "weights"}


def test_norm_offsets_survive_a_zero_length_tensor(tmp_path):
    # mx.min refuses a zero-size reduce. Letting that raise would discard every
    # offset measured alongside it and silently drop the whole correction.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class ShiftEveryNorm:
        def __init__(self, src_path):
            self._src_path = str(src_path)

        def sanitize(self, weights):
            return {
                key: (value + 1.0 if key.endswith("norm.weight") else value)
                for key, value in weights.items()
            }

    src = tmp_path / "src"
    _write_weights(mx, src, {
        "model.norm.weight": mx.array([0.5, 0.25]),
        "model.layers.0.empty_bias": mx.zeros((0,)),
    })

    assert mutils._mlx_sanitizer_norm_offsets(ShiftEveryNorm(src)) == {
        "model.norm.weight": 1.0,
    }


def test_norm_offsets_replay_once_when_nothing_is_shifted(tmp_path):
    # The confirming replay only runs when the first one found something to
    # confirm. Sanitizers that shift nothing are the common case, and some of
    # them dequantize the whole checkpoint on the way through.
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    calls = []

    class CountingPassthrough:
        def __init__(self, src_path):
            self._src_path = str(src_path)

        def sanitize(self, weights):
            calls.append(len(weights))
            return dict(weights)

    src = tmp_path / "src"
    _write_weights(mx, src, {"model.norm.weight": mx.array([0.5, 0.25])})

    assert mutils._mlx_sanitizer_norm_offsets(CountingPassthrough(src)) == {}
    assert len(calls) == 1


class _StopAfterExportPrep(Exception):
    """Ends save_pretrained_gguf once the export prep has been observed."""


@pytest.mark.parametrize("is_vlm", [True, False])
def test_gguf_save_prepares_the_export_directory_for_every_model(
    monkeypatch, tmp_path, is_vlm
):
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    def fake_prepare(path, model=None, replay_sanitizers=True):
        calls["path"] = Path(path)
        calls["model"] = model
        calls["replay_sanitizers"] = replay_sanitizers
        raise _StopAfterExportPrep

    monkeypatch.setattr(mutils, "_is_vlm_model", lambda _model: is_vlm)
    monkeypatch.setattr(
        mutils, "save_merged_model", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(mutils, "_prepare_mlx_gguf_export_directory", fake_prepare)

    model = object()
    with pytest.raises(_StopAfterExportPrep):
        mutils.save_pretrained_gguf(model, object(), tmp_path / "out")

    assert calls["model"] is model
    assert calls["replay_sanitizers"] is is_vlm


def test_vlm_rewrite_handles_same_name_layout_transforms(monkeypatch):
    import torch
    import unsloth_zoo.mlx.utils as mutils

    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class SameNameConvSanitizer:
        @staticmethod
        def sanitize(weights):
            return {
                name: mutils.mx.transpose(tensor, (0, 2, 3, 1))
                for name, tensor in weights.items()
            }

    mlx_layout = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    new_name, hf_layout, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.patch_embed.proj.weight",
        mlx_layout,
        [(SameNameConvSanitizer, None)],
    )

    assert changed is True
    assert new_name == "vision_tower.patch_embed.proj.weight"
    assert tuple(hf_layout.shape) == (2, 5, 3, 4)
    assert mutils._mlx_arrays_match(
        mutils.mx.transpose(hf_layout, (0, 2, 3, 1)),
        mlx_layout,
    )


def test_vlm_rewrite_restores_alias_without_transposing_unrelated_rank3(monkeypatch):
    import torch
    import unsloth_zoo.mlx.utils as mutils

    calls = 0
    monkeypatch.setattr(mutils.mx, "transpose", pytest.fail)

    class StripModelNamespace:
        @staticmethod
        def sanitize(weights):
            nonlocal calls
            calls += 1
            (name, tensor), = weights.items()
            return {name.removeprefix("model."): tensor}

    tensor = torch.zeros(2, 3, 4)
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.position_embedding",
        tensor,
        [(StripModelNamespace, None)],
    )

    assert calls == 1
    assert changed is True
    assert new_name == "model.vision_tower.position_embedding"
    assert new_tensor is tensor
    assert not mutils._has_vlm_gguf_rewrite_candidate("language_model.weight", tensor)


def test_mlx_arrays_match_checks_2d_tensor_values(monkeypatch):
    import torch
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setattr(mutils.mx, "all", torch.all)

    assert mutils._mlx_arrays_match(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    )
    assert not mutils._mlx_arrays_match(
        torch.zeros(2, 3),
        torch.ones(2, 3),
    )


def test_vlm_sanitizer_replay_uses_real_model_instances():
    import unsloth_zoo.mlx.utils as mutils

    class VisionTower:
        def sanitize(self, weights):
            assert "vision_tower.proj.weight" in weights
            return {"visual.proj.weight": weights["vision_tower.proj.weight"]}

    class Model:
        def __init__(self):
            self.vision_tower = VisionTower()

        def sanitize(self, weights):
            return self.vision_tower.sanitize(weights)

    model = Model()
    pipelines = mutils._get_mlx_vlm_model_sanitize_pipelines(model)

    assert pipelines[0][0][0] is model
    assert mutils._apply_mlx_vlm_sanitizers(
        pipelines[0],
        {"vision_tower.proj.weight": "tensor"},
    ) == {"visual.proj.weight": "tensor"}


def test_repair_degraded_vlm_processor_rebuilds_from_sidecar_configs(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.loader as loader

    class FakeProcessor:
        def __init__(self, image_processor, tokenizer, chat_template=None):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

    fake_processing = types.ModuleType("mlx_vlm.models.glm_ocr.processing")
    fake_processing.FakeProcessor = FakeProcessor
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models.glm_ocr.processing",
        fake_processing,
    )

    image_processor = object()
    monkeypatch.setattr(
        loader,
        "_build_vlm_image_processor_from_config",
        lambda model_path, processor_config, preprocessor_config, model_type=None: (
            image_processor
        ),
    )

    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "FakeProcessor"}),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "FakeImageProcessor"}),
        encoding="utf-8",
    )

    tokenizer = types.SimpleNamespace(
        chat_template=None,
        save_pretrained=lambda path: None,
    )
    degraded = types.SimpleNamespace(
        tokenizer=tokenizer,
        chat_template="{{ messages }}",
    )

    repaired = loader._repair_degraded_vlm_processor(
        degraded,
        tmp_path,
        "glm_ocr",
    )

    assert isinstance(repaired, FakeProcessor)
    assert repaired.image_processor is image_processor
    assert repaired.tokenizer is tokenizer
    assert repaired.chat_template == "{{ messages }}"
    assert tokenizer.chat_template == "{{ messages }}"


def test_processor_loader_is_call_scoped_and_preserves_failure_policy(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.loader as loader

    native, calls = object(), []

    class FakeAutoProcessor:
        error = ValueError("Unrecognized processing class")

        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            calls.append(kwargs["trust_remote_code"])
            raise cls.error

    monkeypatch.setitem(globals(), "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(loader, "_ensure_vlm_detokenizer_copy", lambda: None)
    monkeypatch.setattr(loader, "_load_declared_mlx_vlm_processor", lambda *_a, **_k: native)
    (tmp_path / "config.json").write_text('{"model_type":"native"}', encoding="utf-8")
    monkeypatch.setitem(globals(), "load_processor", _test_bound_load_processor)
    scoped = loader._bind_mlx_vlm_processor_loader(_test_bound_vlm_load)
    trusted = loader._bind_mlx_vlm_processor_loader(
        _test_bound_vlm_load, allow_remote_code=True
    )
    assert scoped(tmp_path) is native and trusted(tmp_path) is native
    assert calls == [False, True] and load_processor is _test_bound_load_processor
    assert AutoProcessor is FakeAutoProcessor
    FakeAutoProcessor.error = RuntimeError("unrelated")
    with pytest.raises(RuntimeError, match="unrelated"):
        scoped(tmp_path)


def test_legacy_detokenizer_copy_is_reset_and_inherited_native_copy_wins(monkeypatch):
    import unsloth_zoo.mlx.loader as loader

    module_name = "mlx_vlm.tokenizer_utils"
    base = type(
        "StreamingDetokenizer",
        (),
        {"__module__": module_name, "__slots__": ("text", "tokens", "offset")},
    )
    legacy = type(
        "NaiveStreamingDetokenizer",
        (base,),
        {
            "__module__": module_name,
            "__init__": _test_naive_detokenizer_init,
            "reset": _test_naive_detokenizer_reset,
            "text": property(lambda self: ""),
        },
    )
    module = types.SimpleNamespace(
        __name__=module_name,
        StreamingDetokenizer=base,
        NaiveStreamingDetokenizer=legacy,
        make_streaming_detokenizer=_test_make_streaming_detokenizer,
    )
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)
    source_hash = loader._source_token_sha256(loader._safe_getsource(_test_make_streaming_detokenizer))
    monkeypatch.setattr(loader, "_MLX_VLM_DETOKENIZER_COPY_TOKEN_SHA256", source_hash)
    loader._bind_mlx_vlm_processor_loader(_test_bound_vlm_load)
    original = legacy(object())
    original._tokens.append(1)
    assert copy(original)._tokens == [] and original._tokens == [1]
    del legacy.__copy__
    base.__copy__ = lambda self: self
    loader._bind_mlx_vlm_processor_loader(_test_bound_vlm_load)
    inherited = legacy(object())
    assert copy(inherited) is inherited
    assert "__copy__" not in legacy.__dict__


def test_quantized_projector_binding_is_call_scoped_and_fail_closed(monkeypatch):
    import unsloth_zoo.mlx.loader as loader

    paths = ["multi_modal_projector.quantized", "multi_modal_projector.dense", "vision.quantized"]
    weights = {f"{paths[0]}.scales": object(), f"{paths[2]}.scales": object()}
    skip = lambda path: path.startswith(("multi_modal_projector", "vision"))
    module = types.SimpleNamespace(to_quantized=True, weight=types.SimpleNamespace(size=64))
    quantize = lambda values, class_predicate: [class_predicate(path, module) for path in values]
    monkeypatch.setitem(globals(), "nn", types.SimpleNamespace(quantize=quantize))
    monkeypatch.setitem(globals(), "skip_multimodal_module", skip)
    monkeypatch.setitem(globals(), "load_model", _test_legacy_projector_load)
    original = _test_bound_load_model
    scoped = loader._bind_mlx_vlm_quantized_projector_loader(original)
    assert scoped(paths, weights) == [True, False, False]
    assert load_model is _test_legacy_projector_load and skip_multimodal_module is skip
    monkeypatch.setitem(globals(), "load_model", _test_aware_projector_load)
    assert loader._bind_mlx_vlm_quantized_projector_loader(original) is original
    monkeypatch.setitem(globals(), "load_model", _test_legacy_projector_load)
    monkeypatch.setitem(globals(), "AutoProcessor", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: "processor"))
    monkeypatch.setitem(globals(), "load_processor", _test_bound_load_processor)
    processor_bound = loader._bind_mlx_vlm_processor_loader(_test_bound_vlm_load)
    assert loader._bind_mlx_vlm_quantized_projector_loader(processor_bound)(
        "model", paths=paths, weights=weights
    ) == "processor"


def test_lfm_disabled_projector_norm_is_loader_gated(monkeypatch):
    import unsloth_zoo.mlx.loader as loader

    module_name = "mlx_vlm.models.lfm2_vl.lfm2_vl"
    Projector = type(
        "Lfm2VlMultiModalProjector", (), {"__module__": module_name,
        "__init__": _test_lfm_projector_init, "__call__": _test_lfm_projector_call},
    )
    module = types.ModuleType(module_name)
    module.Lfm2VlMultiModalProjector = Projector
    monkeypatch.setitem(sys.modules, module_name, module)
    init_hash = loader._source_token_sha256(loader._safe_getsource(_test_lfm_projector_init))
    call_hash = loader._source_token_sha256(loader._safe_getsource(_test_lfm_projector_call))
    original = Projector.__init__
    incompatible_contracts = (
        ("unknown", call_hash, Projector.__name__),
        (init_hash, "unknown", Projector.__name__),
        (init_hash, call_hash, "OtherProjector"),
    )
    for init_token, call_token, class_name in incompatible_contracts:
        monkeypatch.setattr(loader, "_LFM2_PROJECTOR_INIT_TOKEN_SHA256", init_token)
        monkeypatch.setattr(loader, "_LFM2_PROJECTOR_CALL_TOKEN_SHA256", call_token)
        Projector.__name__ = class_name
        loader._bind_mlx_vlm_quantized_projector_loader(lambda: None, model_type="lfm2_vl")
        assert Projector.__init__ is original

    Projector.__name__ = "Lfm2VlMultiModalProjector"
    monkeypatch.setattr(loader, "_LFM2_PROJECTOR_INIT_TOKEN_SHA256", init_hash)
    monkeypatch.setattr(loader, "_LFM2_PROJECTOR_CALL_TOKEN_SHA256", call_hash)
    loader._bind_mlx_vlm_quantized_projector_loader(lambda: None, model_type="lfm2_vl")
    disabled = Projector(types.SimpleNamespace(projector_use_layernorm=False))
    enabled = Projector(types.SimpleNamespace(projector_use_layernorm=True))
    assert not hasattr(disabled, "layer_norm") and enabled("x") == ("normalized", "x")


def test_minicpmo_mlx_sanitize_is_complete_and_loader_gated(monkeypatch):
    import unsloth_zoo.mlx.loader as loader

    class MiniCPM:
        __module__ = "mlx_vlm.models.minicpmo.minicpmo"
        def sanitize(self, weights):
            output = {}
            for key, value in weights.items():
                for source, target in zip(
                    ("llm.", "vpm.", "apm."),
                    ("language_model.", "vision_tower.", "audio_tower."),
                ):
                    if key.startswith(source):
                        key = target + key[len(source) :]
                        break
                else:
                    if not key.startswith(("resampler.", "audio_projection_layer.")):
                        continue
                if key == "resampler.attn.in_proj_weight":
                    output.update({
                        f"resampler.attn.{name}_proj.weight": part
                        for name, part in zip(("q", "k", "v"), value)
                    })
                elif key.endswith("embeddings.patch_embedding.weight"):
                    output[key] = ("vision-layout", value)
                elif key.endswith("audio_tower.conv1.weight"):
                    output[key] = ("audio-layout", value)
                elif key != "language_model.lm_head.weight":
                    output[key] = value
            return output

    sanitize_weights = lambda _model, weights: weights

    def affected_load_model(model, weights):
        return sanitize_weights(model, weights)

    load_source = "def load_model(model, weights):\n    weights = sanitize_weights(model, weights)\n    return weights\n"
    sources = {affected_load_model: load_source}
    original_getsource = loader._safe_getsource

    def getsource(obj):
        return sources[obj] if obj in sources else original_getsource(obj)

    expected_hash = loader._source_token_sha256(getsource(MiniCPM.sanitize))
    monkeypatch.setattr(loader, "_MINICPM_SANITIZE_TOKEN_SHA256", expected_hash)
    monkeypatch.setattr(loader, "_safe_getsource", getsource)
    monkeypatch.setattr(loader, "_resolve_mlx_vlm_model_class", lambda _: MiniCPM)
    utils = types.ModuleType("mlx_vlm.utils")
    utils.load_model = affected_load_model
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)

    original = MiniCPM.sanitize
    loader._ensure_minicpmo_mlx_sanitize("minicpmo")
    assert MiniCPM.sanitize.__wrapped__ is original
    language, vision, audio = object(), object(), object()
    values = {
        "language_model.model.norm.weight": language,
        "vision_tower.embeddings.patch_embedding.weight": vision,
        "audio_tower.conv1.weight": audio,
        "resampler.attn.in_proj_weight": ("q", "k", "v"),
        "language_model.lm_head.weight": object(),
    }
    result = MiniCPM().sanitize(values)
    assert result["language_model.model.norm.weight"] is language and result["vision_tower.embeddings.patch_embedding.weight"] == ("vision-layout", vision)
    assert result["audio_tower.conv1.weight"] == ("audio-layout", audio) and "language_model.lm_head.weight" not in result
    assert tuple(result[f"resampler.attn.{name}_proj.weight"] for name in ("q", "k", "v")) == ("q", "k", "v")
    with pytest.raises(ValueError, match="mixes source and MLX tower names"):
        MiniCPM().sanitize({"llm.x": 1, "language_model.x": 2})
    assert MiniCPM().sanitize({"language_model.x": 1}) == {}
    loader._ensure_minicpmo_mlx_sanitize("minicpmo")
    assert MiniCPM.sanitize.__wrapped__ is original


def test_minicpmo_vision_dtype_adapter_is_scoped_and_fail_closed(monkeypatch):
    import unsloth_zoo.mlx.loader as loader

    convert = lambda value, dtype=None: (value, dtype)
    owner = "mlx_vlm.models.minicpmo.minicpmo"
    functions = (
        (convert, "_to_mx_array"),
        (_test_minicpmo_legacy_vision, "get_vision_embedding"),
        (_test_minicpmo_fixed_vision, "get_vision_embedding"),
    )
    for function, name in functions:
        monkeypatch.setattr(function, "__module__", owner)
        monkeypatch.setattr(function, "__name__", name)
    monkeypatch.setitem(globals(), "_to_mx_array", convert)
    fingerprint = loader._source_token_sha256(
        loader._safe_getsource(_test_minicpmo_legacy_vision)
    )
    monkeypatch.setattr(loader, "_MINICPM_LEGACY_VISION_TOKEN_SHA256", fingerprint)
    language = types.SimpleNamespace(
        model=types.SimpleNamespace(
            embed_tokens=types.SimpleNamespace(weight=types.SimpleNamespace(dtype="uint32"))
        )
    )
    vision = types.SimpleNamespace(
        embeddings=types.SimpleNamespace(
            patch_embedding=types.SimpleNamespace(weight=types.SimpleNamespace(dtype="float16"))
        )
    )
    model = types.SimpleNamespace(language_model=language, vision_tower=vision)
    adapted = loader._minicpmo_vision_dtype_adapter(_test_minicpmo_legacy_vision)
    assert _test_minicpmo_legacy_vision(model, "pixels", None) == ("pixels", "uint32")
    assert adapted(model, "pixels", None) == ("pixels", "float16")
    assert _to_mx_array is convert and adapted.__wrapped__ is _test_minicpmo_legacy_vision
    assert loader._minicpmo_vision_dtype_adapter(adapted) is adapted
    assert loader._minicpmo_vision_dtype_adapter(_test_minicpmo_fixed_vision) is _test_minicpmo_fixed_vision
    monkeypatch.setattr(_test_minicpmo_legacy_vision, "__module__", "foreign")
    assert loader._minicpmo_vision_dtype_adapter(_test_minicpmo_legacy_vision) is _test_minicpmo_legacy_vision


def test_read_json_file_returns_empty_for_missing_or_malformed_files(tmp_path):
    import unsloth_zoo.mlx.loader as loader

    assert loader._read_json_file(tmp_path / "missing.json") == {}

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")

    assert loader._read_json_file(malformed) == {}


def test_read_json_file_does_not_swallow_unexpected_errors(monkeypatch, tmp_path):
    import builtins
    import unsloth_zoo.mlx.loader as loader

    def fail_open(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(builtins, "open", fail_open)

    with pytest.raises(RuntimeError, match="unexpected"):
        loader._read_json_file(tmp_path / "config.json")


def test_get_model_config_extracts_dataclass_configs():
    import unsloth_zoo.mlx.utils as mutils

    @dataclasses.dataclass
    class VisionConfig:
        hidden_size: int

    @dataclasses.dataclass
    class ModelConfig:
        model_type: str
        vision_config: VisionConfig
        scales: tuple[int, int]

    model = types.SimpleNamespace(
        config=ModelConfig(
            model_type="glm_ocr",
            vision_config=VisionConfig(hidden_size=16),
            scales=(1, 2),
        )
    )

    assert mutils._get_model_config(model) == {
        "model_type": "glm_ocr",
        "vision_config": {"hidden_size": 16},
        "scales": [1, 2],
    }


def test_get_model_config_prefers_copied_raw_config():
    import unsloth_zoo.mlx.utils as mutils

    raw_config = {"model_type": "qwen3", "nested": {"values": [1]}}
    model = types.SimpleNamespace(
        _config=raw_config,
        config=types.SimpleNamespace(to_dict=lambda: {"model_type": "wrong"}),
    )

    extracted = mutils._get_model_config(model)
    extracted["nested"]["values"].append(2)

    assert extracted["model_type"] == "qwen3"
    assert raw_config["nested"]["values"] == [1]


def test_has_vision_config_handles_nested_and_malformed_configs():
    import unsloth_zoo.mlx.utils as mutils

    assert not mutils._has_vision_config(None)
    assert not mutils._has_vision_config({"thinker_config": "bad"})
    assert mutils._has_vision_config({"vision_config": {}})
    assert mutils._has_vision_config({"thinker_config": {"vision_config": {}}})


def test_save_merged_model_detects_nested_vlm_config(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    class Model:
        _config = {
            "model_type": "glm_ocr",
            "thinker_config": {"vision_config": {"hidden_size": 8}},
        }

        def eval(self):
            pass

        def named_modules(self):
            return []

    class Tokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    fake_mlx_lm_utils = types.ModuleType("mlx_lm.utils")
    fake_mlx_lm_utils.dequantize_model = lambda model: model
    fake_mlx_lm_utils.save_model = lambda path, model, donate_model=False: Path(
        path
    ).mkdir(parents=True, exist_ok=True)
    fake_mlx_lm_utils.create_model_card = lambda path, hf_repo: None
    fake_mlx_lm_utils.save_config = lambda config, path: pytest.fail(
        "text save_config should not run"
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake_mlx_lm_utils)

    fake_mlx_utils = types.ModuleType("mlx.utils")
    fake_mlx_utils.tree_unflatten = dict
    monkeypatch.setitem(sys.modules, "mlx.utils", fake_mlx_utils)

    monkeypatch.setattr(mutils, "_is_vlm_model", lambda model: False)

    def fake_save_mlx_config(config, config_path, *, is_vlm=False):
        calls["is_vlm"] = is_vlm
        calls["config"] = config

    monkeypatch.setattr(mutils, "_save_mlx_config", fake_save_mlx_config)

    mutils.save_merged_model(Model(), Tokenizer(), tmp_path)

    assert calls["is_vlm"] is True
    assert calls["config"]["thinker_config"]["vision_config"]["hidden_size"] == 8


def test_prepare_mlx_gguf_export_directory_writes_nextn_config_without_tensors(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.utils as mutils

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "glm_ocr",
                "vision_config": {},
                "text_config": {
                    "num_hidden_layers": 16,
                    "num_nextn_predict_layers": 1,
                    "mtp_num_hidden_layers": 1,
                    "nextn_predict_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mutils, "_get_transformer_layers", lambda model: [object()] * 16)
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [],
    )

    rewritten = mutils._prepare_mlx_gguf_export_directory(tmp_path, model=object())

    assert rewritten == 0
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert "num_nextn_predict_layers" not in updated["text_config"]
    assert "mtp_num_hidden_layers" not in updated["text_config"]
    assert "nextn_predict_layers" not in updated["text_config"]


def test_prepare_mlx_gguf_export_directory_ignores_malformed_thinker_config(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.utils as mutils

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "glm_ocr",
                "thinker_config": "bad",
                "text_config": {"num_hidden_layers": 16},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mutils,
        "_get_transformer_layers",
        lambda model: [object()] * 16,
    )
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [],
    )

    assert mutils._prepare_mlx_gguf_export_directory(tmp_path, model=object()) == 0


def test_copy_source_sidecars_preserves_image_processor_metadata(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    for name in (
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.txt",
        "custom_processing.py",
        "config.json",
        "README.md",
        ".gitattributes",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "pytorch_model.bin",
    ):
        (src / name).write_text(name, encoding="utf-8")
    (dst / "preprocessor_config.json").write_text("existing", encoding="utf-8")

    copied = mutils._copy_source_sidecars(src, dst)

    assert copied == 6
    assert (dst / "preprocessor_config.json").read_text(encoding="utf-8") == "existing"
    for name in (
        "processor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.txt",
        "custom_processing.py",
    ):
        assert (dst / name).read_text(encoding="utf-8") == name
    for skipped in (
        "config.json",
        "README.md",
        ".gitattributes",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "pytorch_model.bin",
    ):
        assert not (dst / skipped).exists()


def test_copy_source_sidecars_ignores_non_directory_source(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    src = tmp_path / "model.safetensors"
    dst = tmp_path / "dst"
    src.write_text("weights", encoding="utf-8")
    dst.mkdir()

    assert mutils._copy_source_sidecars(src, dst) == 0
    assert list(dst.iterdir()) == []


def test_save_pretrained_gguf_anchors_patcher_to_checked_llama_cpp_root(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.llama_cpp as llama_cpp
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))

    llama_root = tmp_path / "llama.cpp"
    llama_root.mkdir()
    converter = llama_root / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    quantizer = llama_root / "llama-quantize"
    quantizer.write_text("# quantizer", encoding="utf-8")

    calls = {}

    def fake_save_merged_model(model, tokenizer, path, dequantize=False):
        calls["dequantize"] = dequantize
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text(
            json.dumps(
                {
                    "mtp_num_hidden_layers": 1,
                    "unsloth_fixed_mtp": True,
                    "num_hidden_layers": 24,
                }
            ),
            encoding="utf-8",
        )

    def fake_download_convert_hf_to_gguf():
        calls["scripts_dir"] = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
        patched = llama_root / "unsloth_convert_hf_to_gguf.py"
        patched.write_text("# patched converter", encoding="utf-8")
        return str(patched), {"Qwen3ForCausalLM"}, {"Gemma3ForConditionalGeneration"}

    def fake_convert_to_gguf(**kwargs):
        calls["convert_kwargs"] = kwargs
        calls["convert_config"] = json.loads(
            (Path(kwargs["input_folder"]) / "config.json").read_text(encoding="utf-8")
        )
        output = Path(
            f"{kwargs['model_name']}.{kwargs['quantization_type'].upper()}.gguf"
        )
        output.write_bytes(b"GGUF")
        return [str(output)], False

    monkeypatch.setattr(mutils, "save_merged_model", fake_save_merged_model)
    monkeypatch.setattr(mutils, "_is_vlm_model", lambda model: False)
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "unused"))
    monkeypatch.setattr(
        llama_cpp,
        "check_llama_cpp",
        lambda llama_cpp_folder: (str(quantizer), str(converter)),
    )
    monkeypatch.setattr(
        llama_cpp,
        "install_llama_cpp",
        lambda llama_cpp_folder: pytest.fail("install_llama_cpp should not run"),
    )
    monkeypatch.setattr(
        llama_cpp,
        "_download_convert_hf_to_gguf",
        fake_download_convert_hf_to_gguf,
    )
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", fake_convert_to_gguf)
    monkeypatch.setattr(
        llama_cpp,
        "quantize_gguf",
        lambda **kwargs: pytest.fail("quantize_gguf should not run"),
    )

    old_scripts_dir = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
    model = types.SimpleNamespace(_hf_repo="org/TestModel")
    out = tmp_path / "out"
    mutils.save_pretrained_gguf(
        model,
        tokenizer=object(),
        save_directory=out,
        quantization_method="not_quantized",
        first_conversion="f16",
    )

    assert calls["dequantize"] is True
    assert calls["scripts_dir"] == str(llama_root)
    assert calls["convert_kwargs"]["converter_location"] == str(
        llama_root / "unsloth_convert_hf_to_gguf.py"
    )
    assert calls["convert_kwargs"]["supported_text_archs"] == {"Qwen3ForCausalLM"}
    assert calls["convert_config"]["mtp_num_hidden_layers"] == 1
    assert calls["convert_config"]["unsloth_fixed_mtp"] is True
    assert (out / "TestModel.F16.gguf").read_bytes() == b"GGUF"
    assert os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR") == old_scripts_dir


def _write_single_gguf(path):
    """An unsplit conversion, in convert_to_gguf's (files, is_vlm) return shape."""
    Path(path).write_bytes(b"GGUF")
    return [str(path)], False


@pytest.mark.parametrize(
    "platform_name, install_behavior, expect_macos_helper",
    [
        ("darwin", "prebuilt_ok", False),    # prebuilt-first: no clone/compile
        ("darwin", "apt_get_error", True),   # prebuilt unavailable -> macOS cmake+Metal helper
        ("linux", "prebuilt_ok", False),     # unchanged Linux path
    ],
)
def test_gguf_install_fallback_prefers_prebuilt_then_macos_helper(
    monkeypatch, tmp_path, platform_name, install_behavior, expect_macos_helper
):
    """When llama.cpp is missing, the install fallback must try the shared
    install_llama_cpp() (prebuilt-first) on every platform, and only drop to the
    macOS cmake+Metal source helper when that prebuilt path hit the apt-get
    failure that is macOS-specific."""
    import unsloth_zoo.llama_cpp as llama_cpp
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(mutils.sys, "platform", platform_name)

    llama_root = tmp_path / "llama.cpp"
    llama_root.mkdir()
    converter = llama_root / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    quantizer = llama_root / "llama-quantize"
    quantizer.write_text("# quantizer", encoding="utf-8")
    (llama_root / "unsloth_convert_hf_to_gguf.py").write_text("# patched", encoding="utf-8")

    calls = []
    check_state = {"n": 0}
    gpu_support_seen = {"value": None}

    def fake_check(folder):
        # First probe fails (forces the install fallback); the re-probe after the
        # macOS helper succeeds.
        check_state["n"] += 1
        calls.append("check")
        if check_state["n"] == 1:
            raise RuntimeError("llama.cpp not found")
        return (str(quantizer), str(converter))

    def fake_install(folder, gpu_support=False):
        calls.append("install_llama_cpp")
        gpu_support_seen["value"] = gpu_support
        if install_behavior == "prebuilt_ok":
            return (str(quantizer), str(converter))
        # Mirror the real macOS-only source-build failure (no apt-get).
        raise RuntimeError(
            "[FAIL] Unsloth: apt-get does not exist? Is this NOT a Linux / Mac based computer?"
        )

    def fake_macos(folder):
        calls.append("_install_llama_cpp_macos")

    monkeypatch.setattr(
        mutils, "save_merged_model",
        lambda model, tokenizer, path, dequantize=False: Path(path).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(mutils, "_is_vlm_model", lambda model: False)
    monkeypatch.setattr(mutils, "_install_llama_cpp_macos", fake_macos)
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(llama_root))
    monkeypatch.setattr(llama_cpp, "check_llama_cpp", fake_check)
    monkeypatch.setattr(llama_cpp, "install_llama_cpp", fake_install)
    monkeypatch.setattr(
        llama_cpp, "_download_convert_hf_to_gguf",
        lambda: (str(llama_root / "unsloth_convert_hf_to_gguf.py"), {"Qwen3ForCausalLM"}, set()),
    )
    monkeypatch.setattr(
        llama_cpp, "convert_to_gguf",
        lambda **kw: _write_single_gguf(
            f"{kw['model_name']}.{kw['quantization_type'].upper()}.gguf"
        ),
    )
    monkeypatch.setattr(llama_cpp, "quantize_gguf", lambda **kw: None)

    model = types.SimpleNamespace(_hf_repo="org/TestModel")
    mutils.save_pretrained_gguf(
        model,
        tokenizer=object(),
        save_directory=tmp_path / "out",
        quantization_method="not_quantized",
        first_conversion="f16",
    )

    # Prebuilt-first is attempted on every platform.
    assert "install_llama_cpp" in calls
    # Export only needs the CPU-only llama-quantize, so gpu_support=False on every
    # platform. On macOS this still resolves the universal unslothai/llama.cpp
    # Metal bundle (same archive from the CPU selector), and the Metal source build
    # is handled by the macOS helper below, not by this flag.
    assert gpu_support_seen["value"] is False
    # The macOS source helper is reached only on the darwin apt-get path.
    assert ("_install_llama_cpp_macos" in calls) == expect_macos_helper


def test_gguf_install_fallback_reraises_non_aptget_runtimeerror(monkeypatch, tmp_path):
    """A non-apt-get RuntimeError from install_llama_cpp must propagate, not get
    silently swallowed into the macOS source build."""
    import unsloth_zoo.llama_cpp as llama_cpp
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    monkeypatch.setattr(mutils.sys, "platform", "darwin")

    monkeypatch.setattr(
        mutils, "save_merged_model",
        lambda model, tokenizer, path, dequantize=False: Path(path).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(mutils, "_is_vlm_model", lambda model: False)
    monkeypatch.setattr(
        mutils, "_install_llama_cpp_macos",
        lambda folder: pytest.fail("_install_llama_cpp_macos must not run for an unrelated error"),
    )
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "llama.cpp"))
    monkeypatch.setattr(
        llama_cpp, "check_llama_cpp",
        lambda folder: (_ for _ in ()).throw(RuntimeError("not found")),
    )
    monkeypatch.setattr(
        llama_cpp, "install_llama_cpp",
        lambda folder, gpu_support=False: (_ for _ in ()).throw(RuntimeError("disk full while downloading prebuilt")),
    )

    model = types.SimpleNamespace(_hf_repo="org/TestModel")
    with pytest.raises(RuntimeError, match="disk full"):
        mutils.save_pretrained_gguf(
            model,
            tokenizer=object(),
            save_directory=tmp_path / "out",
            quantization_method="not_quantized",
            first_conversion="f16",
        )


def test_push_to_hub_gguf_forwards_export_options(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    class FakeHfApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, exist_ok=True, private=None):
            calls["repo"] = {
                "repo_id": repo_id,
                "exist_ok": exist_ok,
                "private": private,
            }

        def update_repo_settings(self, **kwargs):
            calls["update_repo_settings"] = kwargs

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
            calls["upload"] = {
                "path_or_fileobj": Path(path_or_fileobj),
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
            }

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    def fake_save_pretrained_gguf(
        model,
        tokenizer,
        save_directory,
        quantization_method="fast_quantized",
        first_conversion=None,
        token=None,
        imatrix_file=None,
    ):
        calls["save"] = {
            "quantization_method": quantization_method,
            "first_conversion": first_conversion,
            "imatrix_file": imatrix_file,
            "token": token,
        }
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "model.F16.gguf").write_bytes(b"GGUF")

    monkeypatch.setattr(mutils, "save_pretrained_gguf", fake_save_pretrained_gguf)

    mutils.push_to_hub_gguf(
        model=object(),
        tokenizer=object(),
        save_directory=tmp_path,
        repo_id="org/model",
        quantization_method="not_quantized",
        first_conversion="f16",
        token="hf_token",
        private=True,
        imatrix_file="/path/to/imatrix.dat",
    )

    assert calls["save"] == {
        "quantization_method": "not_quantized",
        "first_conversion": "f16",
        "imatrix_file": "/path/to/imatrix.dat",
        "token": "hf_token",
    }
    assert calls["token"] == "hf_token"
    assert calls["repo"] == {
        "repo_id": "org/model",
        "exist_ok": True,
        "private": True,
    }
    assert calls["upload"]["path_in_repo"] == "model.F16.gguf"


def test_macos_helper_reclones_non_source_dir(monkeypatch, tmp_path):
    # A stale prebuilt install (binaries + marker, no CMakeLists.txt) left in the
    # llama.cpp folder must be replaced before the macOS source build, or cmake runs
    # against a directory with no CMakeLists.txt and the source fallback fails. The
    # prebuilt-first export path reaches this helper exactly that way on macOS.
    import subprocess
    import unsloth_zoo.mlx.utils as mutils

    import unsloth_zoo.llama_cpp as lcpp

    # The helper only deletes a recognised managed prebuilt install (marker present)
    # that lives in a safe-to-delete location, so anchor UNSLOTH_HOME at tmp_path.
    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(tmp_path), raising=False)

    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    (folder / "llama-quantize").write_text("broken prebuilt binary")
    (folder / "UNSLOTH_PREBUILT_INFO.json").write_text("{}")

    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(list(cmd))
        if list(cmd[:2]) == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "CMakeLists.txt").write_text("# source tree")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda: 2))

    mutils._install_llama_cpp_macos(str(folder))

    assert any(list(c[:2]) == ["git", "clone"] for c in cmds), "expected a re-clone of the non-source dir"
    assert (folder / "CMakeLists.txt").is_file(), "folder should be a source tree after the re-clone"


def test_macos_helper_refuses_unmanaged_non_source_dir(monkeypatch, tmp_path):
    # A non-source directory that is NOT a recognised Unsloth prebuilt install
    # (no UNSLOTH_PREBUILT_INFO.json marker) must never be deleted -- a caller may
    # point UNSLOTH_LLAMA_CPP_PATH at a directory full of their own files. The
    # helper must raise instead of wiping it, mirroring the generic installer's
    # _is_safe_to_delete / prebuilt-marker guard.
    import subprocess
    import unsloth_zoo.mlx.utils as mutils
    import unsloth_zoo.llama_cpp as lcpp

    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(tmp_path), raising=False)

    folder = tmp_path / "user_data"
    folder.mkdir()
    (folder / "important.txt").write_text("precious user file")  # no marker, no CMakeLists

    def fake_run(cmd, *a, **k):
        if list(cmd[:2]) == ["git", "clone"]:
            pytest.fail("must not re-clone (and therefore delete) an unmanaged directory")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda: 2))

    with pytest.raises(RuntimeError, match="will not be removed"):
        mutils._install_llama_cpp_macos(str(folder))

    # The user's directory and its contents must be left fully intact.
    assert folder.is_dir()
    assert (folder / "important.txt").read_text() == "precious user file"


def test_macos_helper_keeps_existing_source_tree(monkeypatch, tmp_path):
    # A real source checkout (CMakeLists.txt present) is kept and rebuilt, never
    # re-cloned -- only non-source dirs are replaced.
    import subprocess
    import unsloth_zoo.mlx.utils as mutils

    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    (folder / "CMakeLists.txt").write_text("# existing source tree")

    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda: 2))

    mutils._install_llama_cpp_macos(str(folder))

    assert not any(list(c[:2]) == ["git", "clone"] for c in cmds), "must not re-clone an existing source tree"
    assert (folder / "CMakeLists.txt").is_file()


def _stub_gguf_export(monkeypatch, tmp_path):
    """Stub llama.cpp and the merge so save_pretrained_gguf runs with no model or binaries, and
    record "merged", "quantize" (llama-quantize's kwargs) and "imatrix_bytes" into the result."""
    import unsloth_zoo.llama_cpp as llama_cpp
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))

    llama_root = tmp_path / "llama.cpp"
    llama_root.mkdir()
    (llama_root / "convert_hf_to_gguf.py").write_text("# converter", encoding="utf-8")
    quantizer = llama_root / "llama-quantize"
    quantizer.write_text("# quantizer", encoding="utf-8")

    calls = {}

    def fake_save_merged_model(model, tokenizer, path, dequantize=False):
        calls["merged"] = True
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")

    def fake_convert_to_gguf(**kwargs):
        output = Path(f"{kwargs['model_name']}.{kwargs['quantization_type'].upper()}.gguf")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"GGUF")
        return [str(output)], False

    def fake_quantize_gguf(**kwargs):
        calls["quantize"] = kwargs
        # The imatrix only has to exist while llama-quantize runs, so read it here.
        imatrix = kwargs.get("imatrix")
        calls["imatrix_bytes"] = Path(imatrix).read_bytes() if imatrix else None
        Path(kwargs["output_gguf"]).write_bytes(b"GGUF")

    monkeypatch.setattr(mutils, "save_merged_model", fake_save_merged_model)
    monkeypatch.setattr(mutils, "_is_vlm_model", lambda model: False)
    monkeypatch.setattr(
        llama_cpp, "check_llama_cpp", lambda folder: (str(quantizer), str(llama_root / "convert_hf_to_gguf.py"))
    )
    monkeypatch.setattr(
        llama_cpp,
        "_download_convert_hf_to_gguf",
        lambda: (str(llama_root / "convert_hf_to_gguf.py"), None, None),
    )
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", fake_convert_to_gguf)
    monkeypatch.setattr(llama_cpp, "quantize_gguf", fake_quantize_gguf)
    return calls


def _export(save_directory, **kwargs):
    import unsloth_zoo.mlx.utils as mutils

    model = types.SimpleNamespace(_hf_repo="unsloth/TestModel")
    mutils.save_pretrained_gguf(model, object(), save_directory, **kwargs)

# Both export paths glob save_directory/*.gguf, so a copy left there would be uploaded and
# reported as an exported model.
@pytest.mark.parametrize(
    "source_name, resolved_name, inside",
    [
        ("imatrix_unsloth.dat", "imatrix_unsloth.dat", False),
        ("imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf", False),  # llama.cpp rejects .gguf_file
        ("imatrix_unsloth.gguf", "imatrix_unsloth.gguf", True),  # already .gguf, and in the way
    ],
)
def test_the_imatrix_copy_lands_outside_the_export_directory(monkeypatch, tmp_path, source_name,
                                                             resolved_name, inside):
    calls = _stub_gguf_export(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    source = (out if inside else tmp_path) / source_name
    source.write_bytes(b"IMAT")

    _export(out, quantization_method="iq2_xxs", imatrix_file=str(source))

    used = Path(calls["quantize"]["imatrix"])
    assert calls["quantize"]["quant_type"] == "iq2_xxs"
    assert used.name == resolved_name and used != source
    assert calls["imatrix_bytes"] == b"IMAT"
    assert out not in used.parents
    assert source.exists(), "the caller's file must be copied, not moved"
    left_behind = [source_name] if inside else []
    assert sorted(p.name for p in out.rglob("*.gguf")) == sorted(["TestModel.IQ2_XXS.gguf"] + left_behind)


# The caller's own imatrix may sit in save_directory; we must not delete it, but both the
# completion summary and push_to_hub_gguf glob save_directory/*.gguf, so it must not be
# reported or uploaded as though this export had produced it.
def test_an_imatrix_inside_the_export_directory_is_not_reported_or_uploaded(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    _stub_gguf_export(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    source = out / "imatrix_unsloth.gguf"
    source.write_bytes(b"IMAT")

    _export(out, quantization_method="iq2_xxs", imatrix_file=str(source))

    reported = [f.name for f in mutils._exported_gguf_files(out, str(source))]
    assert reported == ["TestModel.IQ2_XXS.gguf"]
    assert source.exists(), "the caller's file must never be deleted"


# An imatrix named like a file the export writes is destroyed by it: convert_to_gguf and
# llama-quantize overwrite by name, and the intermediate is deleted afterwards. Resolution copies
# it out first, so without this guard the export SUCCEEDS while eating the caller's input.
@pytest.mark.parametrize("source_name", ["TestModel.IQ2_XXS.gguf", "TestModel.BF16.gguf"])
def test_an_imatrix_named_like_an_output_is_refused_before_anything_is_written(
    monkeypatch, tmp_path, source_name
):
    calls = _stub_gguf_export(monkeypatch, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    source = out / source_name
    source.write_bytes(b"MY_IMATRIX")

    with pytest.raises(RuntimeError, match="would overwrite it"):
        _export(out, quantization_method="iq2_xxs", imatrix_file=str(source))

    assert source.read_bytes() == b"MY_IMATRIX", "the caller's imatrix must survive"
    assert "merged" not in calls, "must refuse before paying for the merge"


def test_exported_gguf_files_without_an_imatrix_lists_everything(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    (tmp_path / "a.gguf").write_bytes(b"A")
    (tmp_path / "b.gguf").write_bytes(b"B")

    assert [f.name for f in mutils._exported_gguf_files(tmp_path)] == ["a.gguf", "b.gguf"]
    # A path that no longer exists must not crash the listing -- samefile raises on a dead path.
    assert [f.name for f in mutils._exported_gguf_files(tmp_path, str(tmp_path / "gone.gguf"))] == \
        ["a.gguf", "b.gguf"]


def test_is_same_file_asks_the_filesystem_then_falls_back(tmp_path):
    """The identity test must survive a path the filesystem spells differently, or one now gone."""
    import unsloth_zoo.mlx.utils as mutils

    real = tmp_path / "imatrix.gguf"
    real.write_bytes(b"IMAT")
    other = tmp_path / "quant.gguf"
    other.write_bytes(b"Q")

    assert mutils._is_same_file(real, str(real))
    # One file, two names: only the filesystem knows. Comparing the strings would miss it, and
    # the imatrix would be uploaded as a model.
    link = tmp_path / "linked.gguf"
    link.symlink_to(real)
    assert mutils._is_same_file(link, str(real))
    assert not mutils._is_same_file(real, other)
    # Both gone: samefile raises OSError, so the normcase/NFC fallback decides.
    gone = tmp_path / "gone.gguf"
    assert mutils._is_same_file(gone, str(gone))
    assert not mutils._is_same_file(gone, tmp_path / "also-gone.gguf")


# "BF16" is spelled either way: the converter lowercases, so these checks have to as well.
@pytest.mark.parametrize("quantization_method", ["not_quantized", "bf16", "BF16"])
def test_a_direct_conversion_drops_the_imatrix_instead_of_resolving_it(
    monkeypatch, tmp_path, quantization_method
):
    # bf16/f16/f32 never reach llama-quantize, so resolving risks a Hub failure for nothing.
    import unsloth_zoo.llama_cpp as llama_cpp

    calls = _stub_gguf_export(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(
        llama_cpp, "resolve_imatrix_file", lambda imatrix_file, **k: seen.setdefault("arg", imatrix_file)
    )

    with pytest.warns(UserWarning, match="ignoring imatrix_file"):
        _export(tmp_path / "out", quantization_method=quantization_method, imatrix_file=True)

    # Resolution was reached with None, so no repo lookup happened. quant_type is normalized before
    # either comparison, so "BF16" is a direct conversion like the others and llama-quantize is not
    # reached at all -- it no longer degenerates into a BF16 -> BF16 requantize.
    assert seen["arg"] is None
    assert "quantize" not in calls
    assert calls.get("quantize", {}).get("imatrix") is None


# The drop guard predicts whether llama-quantize will run. It has to agree with the condition that
# actually gates it, or the imatrix is discarded from a run that then goes ahead without one. These
# spellings differ only in case/whitespace from the intermediate, which llama-quantize accepts.
@pytest.mark.parametrize("first_conversion", ["Q4_K_M", " q4_k_m ", "q4_k_m"])
def test_a_case_variant_intermediate_does_not_silently_discard_the_imatrix(
    monkeypatch, tmp_path, first_conversion
):
    import unsloth_zoo.llama_cpp as llama_cpp

    calls = _stub_gguf_export(monkeypatch, tmp_path)
    monkeypatch.setattr(
        llama_cpp, "resolve_imatrix_file",
        lambda imatrix_file, **k: None if imatrix_file is None else "/tmp/imatrix.gguf",
    )

    with pytest.warns(UserWarning, match="ignoring imatrix_file"):
        _export(
            tmp_path / "out", quantization_method="q4_k_m",
            first_conversion=first_conversion, imatrix_file=True,
        )

    # Target and intermediate are the same quant however it is spelled, so the warning is honest:
    # llama-quantize really is skipped. It must not be reached with the imatrix thrown away.
    assert "quantize" not in calls


# Left: needs an imatrix. Right: quantizes without one. Measured, not read off the ftype defaults:
# `llama-quantize --dry-run <bf16.gguf> /dev/null <quant>` on a real Llama-3.2-1B prints
# "will require an imatrix!" for exactly the left column.
#
# iq3_xs is on the LEFT despite its ftype defaulting to IQ3_S, because the attention Q/K overrides
# in llama_tensor_get_type_impl promote to IQ3_XXS with no has_imatrix check. A real export fails on
# blk.0.attn_k.weight -- the first block -- so leaving it off the list buys nothing and costs the
# user the whole merge and BF16 conversion first, which is the failure this guard exists to prevent.
@pytest.mark.parametrize(
    "quant, required",
    [(q, True) for q in (
        "iq1_s", "iq1_m", "iq1_xs", "iq1_xxs", "iq1_xxxs",
        "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_xs", "q2_k_s",
    )] + [(q, False) for q in (
        "iq3_s", "iq3_m", "iq4_nl", "iq4_xs",
        "q2_k", "q3_k_s", "tq1_0", "tq2_0", "q4_k_m", "q8_0", "bf16", "f16",
    )],
)
def test_quant_requires_imatrix_matches_llama_cpp(quant, required):
    import unsloth_zoo.llama_cpp as llama_cpp

    assert llama_cpp.quant_requires_imatrix(quant) is required
    assert llama_cpp.quant_requires_imatrix(quant.upper()) is required


# llama.cpp's tensor_requires_imatrix rejects every one of these outright (src/llama-quant.cpp).
@pytest.mark.parametrize("quant", ["iq1_s", "iq2_xxs", "iq2_m", "iq3_xxs", "iq3_xs", "q2_k_s"])
def test_imatrix_only_quants_are_refused_before_the_merge(monkeypatch, tmp_path, quant):
    calls = _stub_gguf_export(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="importance matrix"):
        _export(tmp_path / "out", quantization_method=quant)

    assert "merged" not in calls, "must fail before paying for the merge and conversion"
    assert not (tmp_path / "out").exists(), "a refused export must not leave a directory behind"


# The counterpart: each of these quantizes without an imatrix, so the export must run rather than
# be refused. llama.cpp may still warn that quality suffers -- that is its call to make, not ours.
@pytest.mark.parametrize("quant", ["iq4_xs", "iq4_nl", "iq3_s", "iq3_m", "q2_k", "q4_k_m"])
def test_quants_not_refused_up_front(monkeypatch, tmp_path, quant):
    calls = _stub_gguf_export(monkeypatch, tmp_path)

    _export(tmp_path / "out", quantization_method=quant)

    assert calls["quantize"]["quant_type"] == quant
    assert calls["quantize"]["imatrix"] is None


def _fake_hub(monkeypatch, tmp_path, upstream_name, hosted_by="unsloth/TestModel-GGUF"):
    """Stand in for the Hub: only `hosted_by` exists, and only it ships `upstream_name`."""
    cached = tmp_path / "cache" / upstream_name
    cached.parent.mkdir(exist_ok=True)
    cached.write_bytes(b"UPSTREAM")
    seen = {"looked_up": []}

    class FakeHfApi:
        def __init__(self, token=None):
            seen["token"] = token

        def list_repo_files(self, repo_id):
            seen["looked_up"].append(repo_id)
            if repo_id != hosted_by:
                raise RuntimeError("404")
            return ["config.json", upstream_name]

    def fake_download(repo_id, filename, token=None):
        seen["downloaded"] = {"repo_id": repo_id, "filename": filename, "token": token}
        return str(cached.parent / filename)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeHfApi
    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    return seen


# Both upstream spellings must resolve: .dat is the classic llama.cpp format, .gguf_file the
# GGUF one the Hub would otherwise list as a model.
@pytest.mark.parametrize(
    "upstream_name, expected_local",
    [
        ("imatrix_unsloth.dat", "imatrix_unsloth.dat"),
        ("imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf"),
        # Some repos publish the GGUF imatrix under its plain name, without the .gguf_file guard
        # (unsloth/Qwen3.8-27B-GGUF does). llama-quantize --imatrix reads it just the same.
        ("imatrix_unsloth.gguf", "imatrix_unsloth.gguf"),
    ],
)
def test_imatrix_file_true_resolves_the_upstream_gguf_repo(monkeypatch, tmp_path, upstream_name,
                                                          expected_local):
    import unsloth_zoo.llama_cpp as llama_cpp

    seen = _fake_hub(monkeypatch, tmp_path, upstream_name)

    resolved = llama_cpp.resolve_imatrix_file(
        True, dest_dir=str(tmp_path / "dest"), token="hf_token",
        repo_candidates=["unsloth/Missing-GGUF", "unsloth/TestModel-GGUF"],
    )

    assert seen["looked_up"] == ["unsloth/Missing-GGUF", "unsloth/TestModel-GGUF"]
    assert seen["token"] == "hf_token"
    # The download must be authenticated too, and aimed at the repo that actually had the file.
    assert seen["downloaded"] == {
        "repo_id": "unsloth/TestModel-GGUF", "filename": upstream_name, "token": "hf_token",
    }
    assert Path(resolved).name == expected_local
    assert Path(resolved).read_bytes() == b"UPSTREAM"


def test_gguf_export_resolves_the_imatrix_from_the_models_own_repo(monkeypatch, tmp_path):
    """The automatic path end to end: model -> candidate repos -> download -> llama-quantize."""
    import unsloth_zoo.mlx.utils as mutils

    calls = _stub_gguf_export(monkeypatch, tmp_path)
    seen = _fake_hub(monkeypatch, tmp_path, "imatrix_unsloth.dat")

    mutils.save_pretrained_gguf(
        types.SimpleNamespace(_hf_repo="mlx-community/TestModel-4bit"), object(), tmp_path / "out",
        quantization_method="iq2_xxs", imatrix_file=True, token="hf_token",
    )

    # The 4bit repackaging is tried first, then the base model it was quantized from.
    assert seen["looked_up"] == ["unsloth/TestModel-4bit-GGUF", "unsloth/TestModel-GGUF"]
    assert seen["token"] == "hf_token"
    assert seen["downloaded"]["repo_id"] == "unsloth/TestModel-GGUF"
    assert seen["downloaded"]["token"] == "hf_token"
    assert Path(calls["quantize"]["imatrix"]).name == "imatrix_unsloth.dat"
    assert calls["imatrix_bytes"] == b"UPSTREAM"


def _unauthorized(self, repo_id):
    raise PermissionError("401 Unauthorized")


@pytest.mark.parametrize(
    "list_repo_files, expected",
    [
        # Nothing upstream: the error has to name the repo it looked in.
        (lambda self, repo_id: ["config.json"], "unsloth/TestModel-GGUF"),
        # A bad token or an outage must not read as "this model has no imatrix".
        (_unauthorized, "401 Unauthorized"),
    ],
)
def test_upstream_resolution_failure_names_its_cause(monkeypatch, tmp_path, list_repo_files, expected):
    import unsloth_zoo.llama_cpp as llama_cpp

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = type(
        "FakeHfApi", (), {"__init__": lambda self, token=None: None, "list_repo_files": list_repo_files}
    )
    fake_hub.hf_hub_download = lambda **kwargs: pytest.fail("nothing to download")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(RuntimeError, match=expected):
        llama_cpp.resolve_imatrix_file(
            True, dest_dir=str(tmp_path / "dest"), repo_candidates=["unsloth/TestModel-GGUF"]
        )


def test_missing_imatrix_path_is_rejected(tmp_path):
    import unsloth_zoo.llama_cpp as llama_cpp

    with pytest.raises(FileNotFoundError):
        llama_cpp.resolve_imatrix_file(str(tmp_path / "absent.dat"), dest_dir=str(tmp_path))


# resolve_imatrix_file is exported, so a caller may hand it a dest_dir that already holds the file
# -- Studio does. shutil.copyfile raises SameFileError on that; there is simply nothing to copy.
def test_an_imatrix_already_in_dest_dir_is_used_in_place(tmp_path):
    import unsloth_zoo.llama_cpp as llama_cpp

    source = tmp_path / "imatrix_unsloth.dat"
    source.write_bytes(b"IMAT")

    resolved = llama_cpp.resolve_imatrix_file(str(source), dest_dir=str(tmp_path))

    assert Path(resolved).samefile(source)
    assert source.read_bytes() == b"IMAT"


@pytest.mark.parametrize(
    "repo, expected",
    [
        ("unsloth/Qwen3.5-0.8B", ["unsloth/Qwen3.5-0.8B-GGUF"]),
        ("Qwen/Qwen3.5-0.8B", ["unsloth/Qwen3.5-0.8B-GGUF"]),
        ("unsloth/Qwen3.5-0.8B-GGUF", ["unsloth/Qwen3.5-0.8B-GGUF"]),
        (None, []),
        # A repackaged repo is tried verbatim first, then peeled one marker at a time down to the
        # base the imatrix is published against. -MTP survives: it names a different model.
        ("mlx-community/Llama-3.2-1B-Instruct-4bit",
         ["unsloth/Llama-3.2-1B-Instruct-4bit-GGUF", "unsloth/Llama-3.2-1B-Instruct-GGUF"]),
        ("mlx-community/Qwen3.5-2B-MLX-8bit",
         ["unsloth/Qwen3.5-2B-MLX-8bit-GGUF", "unsloth/Qwen3.5-2B-MLX-GGUF",
          "unsloth/Qwen3.5-2B-GGUF"]),
        ("mlx-community/Qwen3.5-9B-MTP-4bit",
         ["unsloth/Qwen3.5-9B-MTP-4bit-GGUF", "unsloth/Qwen3.5-9B-MTP-GGUF"]),
        ("mlx-community/Qwen3-8B-4bit-AWQ",
         ["unsloth/Qwen3-8B-4bit-AWQ-GGUF", "unsloth/Qwen3-8B-4bit-GGUF",
          "unsloth/Qwen3-8B-GGUF"]),
        ("some-org/Model-GPTQ-Int4",
         ["unsloth/Model-GPTQ-Int4-GGUF", "unsloth/Model-GPTQ-GGUF", "unsloth/Model-GGUF"]),
        ("unsloth/Qwen3-8B-unsloth-bnb-4bit",
         ["unsloth/Qwen3-8B-unsloth-bnb-4bit-GGUF", "unsloth/Qwen3-8B-unsloth-bnb-GGUF",
          "unsloth/Qwen3-8B-unsloth-GGUF", "unsloth/Qwen3-8B-GGUF"]),
        # mlx-community publishes the bit width both ways; -4-bit is as common as -4bit and is
        # what this exporter sees most (e.g. mlx-community/Mistral-7B-Instruct-v0.2-4-bit).
        ("mlx-community/Mistral-7B-Instruct-v0.2-4-bit",
         ["unsloth/Mistral-7B-Instruct-v0.2-4-bit-GGUF", "unsloth/Mistral-7B-Instruct-v0.2-GGUF"]),
        ("mlx-community/Llama-3.2-3B-Instruct-8-bit",
         ["unsloth/Llama-3.2-3B-Instruct-8-bit-GGUF", "unsloth/Llama-3.2-3B-Instruct-GGUF"]),
        ("some-org/Model-float16", ["unsloth/Model-float16-GGUF", "unsloth/Model-GGUF"]),
    ],
)
def test_imatrix_repo_candidates_map_onto_the_unsloth_gguf_namespace(repo, expected):
    import unsloth_zoo.mlx.utils as mutils

    model = types.SimpleNamespace(_hf_repo=repo)
    assert mutils._gguf_imatrix_repo_candidates(model) == expected


def test_imatrix_repo_candidates_fall_back_to_the_config_name(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    # A local checkpoint on its own names no upstream repo.
    assert mutils._gguf_imatrix_repo_candidates(types.SimpleNamespace(_hf_repo=str(tmp_path))) == []

    model = types.SimpleNamespace(
        _hf_repo=None, config=types.SimpleNamespace(_name_or_path="Qwen/Qwen3.5-0.8B")
    )
    assert mutils._gguf_imatrix_repo_candidates(model) == ["unsloth/Qwen3.5-0.8B-GGUF"]

    # Both sources contribute, in order, without duplicates.
    model = types.SimpleNamespace(
        _hf_repo="mlx-community/Qwen3.5-0.8B-4bit",
        config=types.SimpleNamespace(_name_or_path="Qwen/Qwen3.5-0.8B"),
    )
    assert mutils._gguf_imatrix_repo_candidates(model) == [
        "unsloth/Qwen3.5-0.8B-4bit-GGUF", "unsloth/Qwen3.5-0.8B-GGUF",
    ]

    # A local directory skips _hf_repo, and MLX keeps a text config as a dict, so _config is the
    # only route left to the upstream repo.
    model = types.SimpleNamespace(
        _hf_repo=str(tmp_path), _config={"_name_or_path": "Qwen/Qwen3.5-0.8B"}
    )
    assert mutils._gguf_imatrix_repo_candidates(model) == ["unsloth/Qwen3.5-0.8B-GGUF"]


@pytest.mark.parametrize(
    "binding, target, destination, forwarded",
    [
        # The credential travels with the imatrix: resolving one reads a Hub repo.
        ("_mlx_save_pretrained_gguf", "save_pretrained_gguf", "out",
         {"imatrix_file": True, "token": "hf_token"}),
        ("_mlx_push_to_hub_gguf", "push_to_hub_gguf", "org/model",
         {"imatrix_file": "/path/to/imatrix.dat"}),
    ],
)
def test_bound_gguf_apis_forward_imatrix_file(monkeypatch, tmp_path, binding, target, destination, forwarded):
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    calls = {}
    monkeypatch.setattr(
        mutils,
        target,
        lambda model, tokenizer, save_directory, *rest, quantization_method=None, repo_id=None,
        **kwargs: calls.update(kwargs),
    )

    destination = str(tmp_path) if destination == "out" else destination
    model = types.SimpleNamespace(_tokenizer=object())
    getattr(loader, binding)(model, destination, quantization_method="iq2_xxs", **forwarded)

    assert calls == forwarded


# Every binding that filters kwargs must name what it dropped, not just the GGUF save path.
@pytest.mark.parametrize(
    "binding, target, args",
    [
        ("_mlx_save_pretrained_gguf", "save_pretrained_gguf", ("out",)),
        ("_mlx_save_pretrained_merged", "save_pretrained_merged", ("out",)),
        ("_mlx_push_to_hub_gguf", "push_to_hub_gguf", ("org/model",)),
        ("_mlx_push_to_hub", "save_pretrained_merged", ("org/model",)),
    ],
)
def test_dropped_kwargs_are_announced(monkeypatch, tmp_path, binding, target, args):
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setattr(mutils, target, lambda *a, **kw: None)
    monkeypatch.setattr(mutils, "collect_mlx_lora_adapter_tensors", lambda model: {})
    model = types.SimpleNamespace(_tokenizer=object())
    args = tuple(str(tmp_path / a) if a == "out" else a for a in args)

    with pytest.warns(UserWarning, match="maximum_memory_usage"):
        getattr(loader, binding)(model, *args, maximum_memory_usage=0.5)
def _run_macos_helper_capturing_pip(monkeypatch, folder):
    """Run the macOS helper against a ready source tree, returning the pip argv."""
    import subprocess
    import unsloth_zoo.mlx.utils as mutils

    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda: 2))

    mutils._install_llama_cpp_macos(str(folder))

    return [c for c in cmds if "pip" in c and "install" in c]


def _make_source_tree_with_gguf_py(folder):
    (folder / "gguf-py").mkdir(parents=True)
    (folder / "CMakeLists.txt").write_text("# source tree")


def test_macos_helper_refuses_pip_install_from_untrusted_checkout(monkeypatch, tmp_path):
    # `pip install <dir>` runs that directory's build backend, so a checkout we
    # neither manage nor were pointed at must never be installed from. gguf comes
    # from the package index instead; conversion itself is unaffected either way,
    # since the converter loads its own sibling gguf-py.
    import unsloth_zoo.llama_cpp as lcpp

    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(tmp_path / "unsloth_home"), raising=False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)

    folder = tmp_path / "untrusted" / "llama.cpp"
    _make_source_tree_with_gguf_py(folder)

    pip_cmds = _run_macos_helper_capturing_pip(monkeypatch, folder)

    assert pip_cmds, "expected the helper to install converter deps"
    installed = pip_cmds[0]
    assert not any(str(folder) in arg for arg in installed), \
        f"must not pip install from an untrusted checkout: {installed}"
    assert "gguf" in installed


def test_macos_helper_installs_gguf_py_from_managed_checkout(monkeypatch, tmp_path):
    # The normal path is unchanged: the managed ~/.unsloth checkout still gets its
    # in-tree gguf-py installed so gguf stays in sync with llama.cpp.
    import unsloth_zoo.llama_cpp as lcpp

    home = tmp_path / "unsloth_home"
    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(home), raising=False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)

    folder = home / "llama.cpp"
    _make_source_tree_with_gguf_py(folder)

    pip_cmds = _run_macos_helper_capturing_pip(monkeypatch, folder)

    assert pip_cmds, "expected the helper to install converter deps"
    assert any(str(folder / "gguf-py") in arg for arg in pip_cmds[0]), pip_cmds[0]


def test_macos_helper_installs_gguf_py_from_operator_named_checkout(monkeypatch, tmp_path):
    # An operator who explicitly points UNSLOTH_LLAMA_CPP_PATH at their own
    # checkout has vouched for it, so the in-tree gguf-py is still used.
    import unsloth_zoo.llama_cpp as lcpp

    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(tmp_path / "unsloth_home"), raising=False)

    folder = tmp_path / "my_llama_cpp"
    _make_source_tree_with_gguf_py(folder)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(folder))

    pip_cmds = _run_macos_helper_capturing_pip(monkeypatch, folder)

    assert pip_cmds, "expected the helper to install converter deps"
    assert any(str(folder / "gguf-py") in arg for arg in pip_cmds[0]), pip_cmds[0]


# --- _is_trusted_local_llama_cpp_dir path semantics -------------------------
# `pip install <dir>` runs that directory's build backend, so the containment
# check that guards it has to be exact. These cover the ways a naive prefix
# comparison goes wrong.

def _trusted(monkeypatch, folder, home, env_value=None):
    import unsloth_zoo.mlx.utils as mutils
    import unsloth_zoo.llama_cpp as lcpp

    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(home), raising=False)
    if env_value is None:
        monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)
    else:
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(env_value))
    return mutils._is_trusted_local_llama_cpp_dir(str(folder))


def test_trusted_dir_accepts_managed_checkout(monkeypatch, tmp_path):
    home = tmp_path / ".unsloth"
    assert _trusted(monkeypatch, home / "llama.cpp", home) is True
    assert _trusted(monkeypatch, home, home) is True


def test_trusted_dir_rejects_prefix_sibling(monkeypatch, tmp_path):
    # "~/.unsloth-evil" shares a string prefix with "~/.unsloth" but is not inside
    # it. A startswith() without the separator would trust it.
    home = tmp_path / ".unsloth"
    evil = tmp_path / ".unsloth-evil"
    evil.mkdir(parents=True)
    assert _trusted(monkeypatch, evil, home) is False
    assert _trusted(monkeypatch, evil / "llama.cpp", home) is False


def test_trusted_dir_rejects_symlink_escaping_the_managed_root(monkeypatch, tmp_path):
    # A symlink sitting inside ~/.unsloth that points out of it must not launder
    # an untrusted directory into the trusted set.
    home = tmp_path / ".unsloth"
    home.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = home / "llama.cpp"
    link.symlink_to(outside)
    assert _trusted(monkeypatch, link, home) is False


def test_trusted_dir_rejects_cwd_relative_checkout(monkeypatch, tmp_path):
    # The case the whole guard exists for: a llama.cpp that just happens to be in
    # the working directory is not something anyone vouched for.
    home = tmp_path / ".unsloth"
    home.mkdir(parents=True)
    cwd = tmp_path / "cwd"
    (cwd / "llama.cpp").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    assert _trusted(monkeypatch, "llama.cpp", home) is False
    assert _trusted(monkeypatch, os.path.join(".", "llama.cpp"), home) is False
    # An operator who names that same directory does get the local install.
    assert _trusted(monkeypatch, "llama.cpp", home, env_value=cwd / "llama.cpp") is True


def test_trusted_dir_accepts_operator_named_checkout(monkeypatch, tmp_path):
    # How Unsloth Studio configures this: it exports UNSLOTH_LLAMA_CPP_PATH.
    home = tmp_path / ".unsloth"
    studio = tmp_path / "StudioHome" / "llama.cpp"
    studio.mkdir(parents=True)
    assert _trusted(monkeypatch, studio, home, env_value=studio) is True
    assert _trusted(monkeypatch, studio / "gguf-py", home, env_value=studio) is True
    # Whitespace is stripped, matching how Studio itself reads the variable.
    assert _trusted(monkeypatch, studio, home, env_value=f"  {studio}  ") is True
    # An empty or blank value must not trust anything.
    assert _trusted(monkeypatch, tmp_path / "other", home, env_value="") is False
    assert _trusted(monkeypatch, tmp_path / "other", home, env_value="   ") is False


def test_trusted_dir_handles_a_root_trusted_path(monkeypatch, tmp_path):
    # os.path.join(parent, "") keeps a root parent as "/" rather than "//", which
    # a bare parent + os.sep would produce and never match.
    home = tmp_path / ".unsloth"
    assert _trusted(monkeypatch, tmp_path / "anywhere", home, env_value=os.sep) is True


def test_trusted_dir_is_case_insensitive_on_windows_style_paths(monkeypatch):
    # Only reached on macOS today, but the comparison should not quietly depend on
    # that. Drive letter and directory case must not change the verdict.
    import ntpath
    import types
    import unsloth_zoo.mlx.utils as mutils
    import unsloth_zoo.llama_cpp as lcpp

    fake_os = types.SimpleNamespace(
        path=types.SimpleNamespace(
            realpath=ntpath.normpath,
            normcase=ntpath.normcase,
            join=ntpath.join,
        ),
        sep="\\",
        environ={},
    )
    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", r"C:\Users\Dan\.unsloth", raising=False)
    monkeypatch.setattr(mutils, "os", fake_os)

    assert mutils._is_trusted_local_llama_cpp_dir(r"c:\users\dan\.UNSLOTH\llama.cpp") is True
    assert mutils._is_trusted_local_llama_cpp_dir("C:/Users/Dan/.unsloth/llama.cpp") is True
    assert mutils._is_trusted_local_llama_cpp_dir(r"C:\Users\Dan\.unsloth-evil") is False
    assert mutils._is_trusted_local_llama_cpp_dir(r"D:\Users\Dan\.unsloth\llama.cpp") is False


def test_trusted_dir_never_raises_on_bad_input(monkeypatch, tmp_path):
    # A bad path must degrade to "untrusted" (which still installs gguf from the
    # index), never take down an export with an unexpected exception.
    home = tmp_path / ".unsloth"
    for bad in ("", None, "/tmp/a\x00b"):
        assert _trusted(monkeypatch, bad, home) is False


def test_macos_helper_defaults_to_the_managed_checkout(monkeypatch, tmp_path):
    # The default used to be a CWD-relative "llama.cpp", so a no-arg call built and
    # pip installed out of the process working directory.
    import subprocess
    import unsloth_zoo.mlx.utils as mutils
    import unsloth_zoo.llama_cpp as lcpp

    managed = tmp_path / ".unsloth" / "llama.cpp"
    managed.mkdir(parents=True)
    (managed / "CMakeLists.txt").write_text("# source tree")
    monkeypatch.setattr(lcpp, "UNSLOTH_HOME", str(tmp_path / ".unsloth"), raising=False)
    monkeypatch.setattr(lcpp, "LLAMA_CPP_DEFAULT_DIR", str(managed), raising=False)

    cwd = tmp_path / "cwd"
    (cwd / "llama.cpp").mkdir(parents=True)
    monkeypatch.chdir(cwd)

    cmds = []

    def fake_run(cmd, *a, **k):
        cmds.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda: 2))

    mutils._install_llama_cpp_macos()

    assert cmds, "expected the helper to run build commands"
    assert any(str(managed) in " ".join(str(p) for p in c) for c in cmds), \
        f"no-arg call should target the managed checkout: {cmds}"
    assert not any(list(c[:2]) == ["git", "clone"] for c in cmds), \
        "the managed source tree already exists, so nothing should be cloned"


def test_trusted_dir_matches_every_llama_cpp_default_dir_spelling(monkeypatch, tmp_path):
    # Why nothing changes for real users: save_pretrained_gguf only ever passes
    # LLAMA_CPP_DEFAULT_DIR, so every spelling of that variable must read as
    # trusted, or an export quietly stops using the user's own gguf-py.
    home = tmp_path / ".unsloth"
    home.mkdir(parents=True)
    custom = tmp_path / "custom" / "llama.cpp"
    custom.mkdir(parents=True)

    spellings = [
        None,                                   # unset
        str(custom),
        str(custom) + os.sep,
        f"  {custom}  ",                        # LLAMA_CPP_DEFAULT_DIR does not strip
        f"\t{custom}\n",
        os.path.join(str(tmp_path), "custom", ".", "llama.cpp"),
        os.path.join(str(tmp_path), "custom", "..", "custom", "llama.cpp"),
        str(tmp_path / "not_created_yet" / "llama.cpp"),
    ]
    for value in spellings:
        if value is None:
            monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)
            folder = str(home / "llama.cpp")
        else:
            monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", value)
            folder = value  # exactly what LLAMA_CPP_DEFAULT_DIR would hold
        assert _trusted(monkeypatch, folder, home, env_value=value) is True, \
            f"UNSLOTH_LLAMA_CPP_PATH={value!r} should stay trusted"
