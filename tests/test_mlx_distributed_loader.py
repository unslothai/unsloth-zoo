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

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


class _FakeGroup:
    def __init__(self, size=2, rank=0, name="group"):
        self._size = size
        self._rank = rank
        self.name = name

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _PipelineInner:
    def __init__(self):
        self.pipeline_calls = []

    def pipeline(self, group):
        self.pipeline_calls.append(group)


class _BothModel:
    def __init__(self):
        self.shard_calls = []
        self.model = _PipelineInner()

    def shard(self, group):
        self.shard_calls.append(group)


class _TensorOnlyModel:
    def __init__(self):
        self.shard_calls = []

    def shard(self, group):
        self.shard_calls.append(group)

    def parameters(self):
        return {}


def _write_config(tmp_path, config):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_apply_mlx_distributed_sharding_modes_and_guards():
    from unsloth_zoo.mlx.loader import (
        _apply_mlx_distributed_sharding,
        _mlx_active_distributed_groups,
    )

    tensor_group, pipeline_group = _FakeGroup(name="tensor"), _FakeGroup(name="pipeline")

    tensor_model = _BothModel()
    assert _apply_mlx_distributed_sharding(tensor_model, tensor_group=tensor_group, model_name="fake") == "tensor"
    assert tensor_model.shard_calls == [tensor_group]

    pipeline_model = _BothModel()
    assert _apply_mlx_distributed_sharding(pipeline_model, pipeline_group=pipeline_group, model_name="fake") == "pipeline"
    assert pipeline_model.model.pipeline_calls == [pipeline_group]

    with pytest.raises(ValueError, match="either pipeline_group or tensor_group"):
        _mlx_active_distributed_groups(pipeline_group, tensor_group)
    with pytest.raises(ValueError, match="tensor parallelism"):
        _apply_mlx_distributed_sharding(object(), tensor_group=tensor_group, model_name="fake")


def test_load_mlx_lm_distributed_pipeline_filters_quant_shards(monkeypatch, tmp_path):
    import mlx_lm.utils as mlx_lm_utils
    from unsloth_zoo.mlx.loader import _load_mlx_lm_distributed

    model_path = _write_config(tmp_path, {"model_type": "llama"})
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.1.weight": "model-00002-of-00004.safetensors",
                    "model.layers.1.scales": "model-00003-of-00004.safetensors",
                    "model.layers.1.biases": "model-00004-of-00004.safetensors",
                    "model.layers.1.bias": "model-00005-of-00005.safetensors",
                }
            }
        )
    )
    for shard in (
        "model-00001-of-00005.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model-00005-of-00005.safetensors",
    ):
        (model_path / shard).write_text(shard)

    events, load_paths = [], []

    class _PipelineModel:
        def __init__(self):
            self.model = _PipelineInner()

        def parameters(self):
            if self.model.pipeline_calls:
                return {"model.layers.1.weight": object()}
            return {"model.layers.0.weight": object(), "model.layers.1.weight": object()}

    def _download(_repo, *args, **kwargs):
        events.append(("download", tuple(kwargs.get("allow_patterns") or ())))
        return model_path

    def _load_model(path, lazy=False, strict=True, model_config=None):
        path = Path(path)
        load_paths.append(path)
        events.append(("load", sorted(p.name for p in path.glob("*.safetensors"))))
        final_load = any(path.glob("*.safetensors"))
        return _PipelineModel(), {
            "model_type": "llama",
            "eos_token_id": 3 if final_load else 2,
        }

    monkeypatch.setattr(mlx_lm_utils, "_download", _download)
    monkeypatch.setattr(mlx_lm_utils, "load_model", _load_model)
    monkeypatch.setattr(mlx_lm_utils, "load_tokenizer", lambda *_a, **_k: types.SimpleNamespace(name="tok"))

    model, tokenizer, config = _load_mlx_lm_distributed(
        "fake/repo",
        "llama",
        {"return_config": True},
        pipeline_group=_FakeGroup(name="pipeline"),
    )

    local_shards = (
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model-00005-of-00005.safetensors",
    )
    assert ("download", local_shards) in events
    assert ("load", list(local_shards)) in events
    assert all(not path.exists() for path in load_paths)
    assert config["eos_token_id"] == 3
    assert config["model_type"] == "llama"
    assert model._unsloth_mlx_distributed_parallel_mode == "pipeline"
    assert not hasattr(model, "_unsloth_mlx_distributed_snapshot_view")


def test_load_mlx_lm_distributed_tensor_uses_strict_fallback(monkeypatch, tmp_path):
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _load_mlx_lm_distributed

    calls = []
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *_a, **_k: _write_config(tmp_path, {"model_type": "llama"}))
    monkeypatch.setattr(mlx_lm_utils, "load_model", lambda *_a, **_k: (_TensorOnlyModel(), {}))

    def _strict_load(_model_name, _model_type, _mlx_load, mlx_load_kwargs, *, hf_token=None):
        calls.append((dict(mlx_load_kwargs), hf_token))
        return _TensorOnlyModel(), types.SimpleNamespace(name="tok"), {"model_type": "llama"}

    monkeypatch.setattr(loader, "_load_mlx_lm_with_strict_fallback", _strict_load)

    model, tokenizer, config = _load_mlx_lm_distributed(
        "fake/repo",
        "llama",
        {"return_config": True},
        tensor_group=_FakeGroup(name="tensor"),
        hf_token="token",
    )

    assert calls == [({"return_config": True, "lazy": True}, "token")]
    assert model.shard_calls
    assert config["model_type"] == "llama"


def test_load_mlx_vlm_distributed_delegates_to_mlx_vlm_sharded_load(monkeypatch, tmp_path):
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_distributed
    calls = []
    model_dir = _write_config(tmp_path, {"model_type": "raw"})
    messages = ["The model does not support pipeline parallelism", "Model type kimi_k25 not supported", "Unsupported model type kimi_k25", "checkpoint exploded"]

    class _FakeVLM:
        pass

    def sharded_load(repo, *, tensor_group=None, pipeline_group=None):
        patched_type = json.loads((Path(repo) / "config.json").read_text()).get("model_type")
        calls.append((repo, tensor_group, pipeline_group, patched_type))
        if pipeline_group is not None:
            raise ValueError(messages.pop(0))
        return _FakeVLM(), types.SimpleNamespace(name="processor")

    vlm_utils = types.ModuleType("mlx_vlm.utils")
    vlm_utils.get_model_path = lambda repo, revision=None: model_dir
    vlm_utils.sharded_load = sharded_load
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", vlm_utils)
    tensor_group, pipeline_group = _FakeGroup(name="tensor"), _FakeGroup(name="pipeline")
    model, _processor = _load_mlx_vlm_distributed("fake/vlm", "qwen3_vl_moe", tensor_group=tensor_group, config_override_data={"model_type": "patched"})
    for _ in range(3):
        with pytest.raises(ValueError, match=r"Unsloth: 'fake/vlm'.*kimi_k25.*pipeline"):
            _load_mlx_vlm_distributed("fake/vlm", "kimi_k25", pipeline_group=pipeline_group)
    with pytest.raises(ValueError, match="checkpoint exploded") as exc_info:
        _load_mlx_vlm_distributed("fake/vlm", "kimi_k25", pipeline_group=pipeline_group)
    assert calls[0][3] == "patched"
    assert Path(calls[0][0]).exists()
    assert model._unsloth_mlx_config_view_paths == [str(calls[0][0])]
    model._unsloth_mlx_config_view_finalizers[0]()
    assert not Path(calls[0][0]).exists()
    assert "Unsloth:" not in str(exc_info.value)
    assert calls[0][1:3] == (tensor_group, None)
    assert all(call[1:3] == (None, pipeline_group) for call in calls[1:])


def test_from_pretrained_distributed_vlm_passes_override_without_temp_view(monkeypatch, tmp_path):
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import FastMLXModel

    config = {"model_type": "raw", "vision_config": {}, "architectures": ["DeepSeekOCRForCausalLM"], "auto_map": {"x": "y"}}
    model_path, calls = _write_config(tmp_path, config), []
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *_a, **_k: model_path)
    monkeypatch.setattr(loader, "_materialize_mlx_vlm_config_data", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("first temp view")))
    monkeypatch.setattr(loader, "_load_mlx_vlm_distributed", lambda *_a, config_override_data=None, **_k: (calls.append(config_override_data), (types.SimpleNamespace(), types.SimpleNamespace(tokenizer=object())))[1])
    for name in ("install_mlx_compile_patches", "_ensure_vlm_prompt_utils_patched", "_convert_mlx_dtype", "_patch_mixed_precision_set_dtype", "_fix_gemma4_kv_sharing", "_fix_gemma3_vision_post_layernorm_eps", "_fix_gemma3_vision_attention_fp32_sdpa", "_fix_gemma3_vision_encoder_fp32_layernorm", "_fix_gemma3_vision_post_layernorm_fp32", "_fix_gemma3_vision_mlp_fp32_activation", "_fix_gemma3_language_mlp_fp32_activation", "_fix_gemma3_multimodal_image_feature_scale"):
        monkeypatch.setattr(loader, name, lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_repair_degraded_vlm_processor", lambda processor, *_a, **_k: processor)
    monkeypatch.setattr(loader, "_infer_snapshot_commit", lambda *_a, **_k: "commit")
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(load=lambda *_a, **_k: None))

    FastMLXModel.from_pretrained("fake/vlm", text_only=False, tensor_group=_FakeGroup(), load_in_4bit=False)

    assert calls == [{k: v for k, v in config.items() if k != "auto_map"} | {"model_type": "deepseekocr"}]


def _patch_fast_mlx_text_load(monkeypatch, tmp_path, config):
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mlx_utils

    model_path = _write_config(tmp_path, config)
    calls = []

    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *_a, **_k: model_path)
    monkeypatch.setattr(loader, "_ensure_safe_text_wrapper_sanitize", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_convert_mlx_dtype", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_keep_norm_parameters_float32", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_patch_mixed_precision_set_dtype", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_patch_mlx_saving", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "_infer_snapshot_commit", lambda *_a, **_k: "commit")
    monkeypatch.setattr(mlx_utils, "normalize_mlx_chat_template", lambda tokenizer, **_k: tokenizer)

    def _distributed_load(_model_name, _model_type, _load_kwargs, *, pipeline_group=None, tensor_group=None, hf_token=None):
        model = _BothModel()
        loader._apply_mlx_distributed_sharding(
            model,
            pipeline_group=pipeline_group,
            tensor_group=tensor_group,
            model_name=_model_name,
        )
        calls.append((pipeline_group, tensor_group, hf_token, model))
        return model, types.SimpleNamespace(name="tok"), dict(config)

    monkeypatch.setattr(loader, "_load_mlx_lm_distributed", _distributed_load)
    return calls


def test_from_pretrained_uses_distributed_loader(monkeypatch, tmp_path):
    from unsloth_zoo.mlx.loader import FastMLXModel

    group = _FakeGroup(name="tensor")
    calls = _patch_fast_mlx_text_load(
        monkeypatch,
        tmp_path,
        {"model_type": "llama"},
    )

    with pytest.warns(UserWarning, match="default load_in_4bit=True"):
        model, tokenizer = FastMLXModel.from_pretrained("fake/repo", text_only=True, tensor_group=group)

    assert calls[0][1] is group
    assert model.shard_calls == [group]
