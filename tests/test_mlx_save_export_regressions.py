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

"""Fast regressions for MLX save/export parity fixes.

These tests cover the contracts behind the real save / GGUF export bugs without
downloading or converting large models.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_torch_shim():
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


def test_vlm_rewrite_skips_untransformable_text_tensors():
    import torch
    import unsloth_zoo.mlx.utils as mutils

    calls = 0

    class CountingSanitizer:
        @staticmethod
        def sanitize(weights):
            nonlocal calls
            calls += 1
            return weights

    tensor = torch.zeros(2, 3)
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "language_model.model.layers.0.mlp.gate_proj.weight",
        tensor,
        [(CountingSanitizer, None)],
    )

    assert calls == 0
    assert changed is False
    assert new_name == "language_model.model.layers.0.mlp.gate_proj.weight"
    assert new_tensor is tensor


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
        lambda model_path, processor_config, preprocessor_config: image_processor,
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


def test_prepare_vlm_gguf_export_directory_writes_nextn_config_without_tensors(
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

    rewritten = mutils._prepare_vlm_gguf_export_directory(tmp_path, model=object())

    assert rewritten == 0
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert "num_nextn_predict_layers" not in updated["text_config"]
    assert "mtp_num_hidden_layers" not in updated["text_config"]
    assert "nextn_predict_layers" not in updated["text_config"]


def test_prepare_vlm_gguf_export_directory_ignores_malformed_thinker_config(
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

    assert mutils._prepare_vlm_gguf_export_directory(tmp_path, model=object()) == 0


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

    def fake_download_convert_hf_to_gguf():
        calls["scripts_dir"] = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
        patched = llama_root / "unsloth_convert_hf_to_gguf.py"
        patched.write_text("# patched converter", encoding="utf-8")
        return str(patched), {"Qwen3ForCausalLM"}, {"Gemma3ForConditionalGeneration"}

    def fake_convert_to_gguf(**kwargs):
        calls["convert_kwargs"] = kwargs
        output = Path(
            f"{kwargs['model_name']}.{kwargs['quantization_type'].upper()}.gguf"
        )
        output.write_bytes(b"GGUF")

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
    assert (out / "TestModel.F16.gguf").read_bytes() == b"GGUF"
    assert os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR") == old_scripts_dir


def test_push_to_hub_gguf_forwards_first_conversion(monkeypatch, tmp_path):
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
    ):
        calls["save"] = {
            "quantization_method": quantization_method,
            "first_conversion": first_conversion,
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
    )

    assert calls["save"] == {
        "quantization_method": "not_quantized",
        "first_conversion": "f16",
    }
    assert calls["token"] == "hf_token"
    assert calls["repo"] == {
        "repo_id": "org/model",
        "exist_ok": True,
        "private": True,
    }
    assert calls["upload"]["path_in_repo"] == "model.F16.gguf"
