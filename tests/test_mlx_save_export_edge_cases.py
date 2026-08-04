"""Edge cases for MLX save / GGUF export beyond the regression suite.

Covers tensor-match ranks, sanitizer proxy replay, the real shard-rewrite
loop, sidecar and JSON edges, concurrent exports, and converter placement.
Runs on the tests/mlx_simulation torch shim like the rest of the MLX suite.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import types
from pathlib import Path, PureWindowsPath

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_torch_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils):
    """Route mutils.mx tensor helpers to real torch ops."""
    def _transpose(tensor, axes):
        return tensor.permute(*axes)

    def _all(value):
        return torch.all(value)

    monkeypatch.setattr(mutils.mx, "transpose", _transpose)
    monkeypatch.setattr(mutils.mx, "all", _all)


# --- Group 1: _mlx_arrays_match ---


def test_arrays_match_rank3_checks_values(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    same = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert mutils._mlx_arrays_match(same, same.clone()) is True
    different = same.clone()
    different[1, 2, 3] += 1.0
    assert mutils._mlx_arrays_match(same, different) is False


def test_arrays_match_rank6_checks_values(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    same = torch.zeros(1, 2, 1, 2, 1, 2)
    assert mutils._mlx_arrays_match(same, torch.zeros(1, 2, 1, 2, 1, 2)) is True
    assert mutils._mlx_arrays_match(same, torch.ones(1, 2, 1, 2, 1, 2)) is False


def test_arrays_match_rank1_and_dtype_mismatch(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    a = torch.tensor([1.0, 2.0, 3.0])
    assert mutils._mlx_arrays_match(a, torch.tensor([1.0, 2.0, 3.0])) is True
    assert mutils._mlx_arrays_match(a, torch.tensor([1.0, 2.0, 4.0])) is False
    # Equal values across dtypes compare equal elementwise; pin that.
    assert mutils._mlx_arrays_match(a, torch.tensor([1, 2, 3])) is True


def test_arrays_match_shape_mismatch_short_circuits(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils

    def _boom(*args, **kwargs):
        raise AssertionError("mx.all must not run on shape mismatch")

    monkeypatch.setattr(mutils.mx, "all", _boom)
    assert mutils._mlx_arrays_match(torch.zeros(2, 3), torch.zeros(3, 2)) is False


def test_arrays_match_identity_shortcut_skips_value_check(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils

    def _boom(*args, **kwargs):
        raise AssertionError("mx.all must not run for identical objects")

    monkeypatch.setattr(mutils.mx, "all", _boom)
    t = torch.zeros(2, 2)
    assert mutils._mlx_arrays_match(t, t) is True


def test_arrays_match_scalar_pair_uses_python_equality():
    import unsloth_zoo.mlx.utils as mutils

    assert mutils._mlx_arrays_match(1.5, 1.5) is True
    assert mutils._mlx_arrays_match(1.5, 2.5) is False


def test_arrays_match_backend_error_returns_false(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils

    def _boom(*args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(mutils.mx, "all", _boom)
    assert mutils._mlx_arrays_match(torch.zeros(2), torch.zeros(2)) is False


def test_arrays_match_nan_tensors_do_not_match(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    nan = torch.tensor([float("nan"), 1.0])
    # NaN != NaN elementwise, so a clone never value-matches; identity does.
    assert mutils._mlx_arrays_match(nan, nan.clone()) is False
    assert mutils._mlx_arrays_match(nan, nan) is True


# --- Group 2: _rewrite_mlx_vlm_tensor_for_gguf branches ---


def test_rewrite_handles_5d_layout_transform(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class Conv3dSanitizer:
        @staticmethod
        def sanitize(weights):
            # HF (O, C, T, H, W) -> MLX (O, T, H, W, C): invert of (0,4,1,2,3)
            return {
                name: tensor.permute(0, 2, 3, 4, 1)
                for name, tensor in weights.items()
            }

    mlx_tensor = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(
        2, 3, 4, 5, 6
    )
    name = "vision_tower.patch_embed.proj.weight"
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        name, mlx_tensor, [(Conv3dSanitizer, None)]
    )
    assert changed is True
    assert new_tensor.shape == (2, 6, 3, 4, 5)
    assert torch.equal(new_tensor.permute(0, 2, 3, 4, 1), mlx_tensor)


def test_rewrite_handles_1d_floating_offset_candidate(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class OffsetSanitizer:
        @staticmethod
        def sanitize(weights):
            return {name: tensor + 1 for name, tensor in weights.items()}

    mlx_tensor = torch.tensor([3.0, 4.0, 5.0])
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.pos_ids", mlx_tensor, [(OffsetSanitizer, None)]
    )
    assert changed is True
    assert torch.equal(new_tensor, mlx_tensor - 1)


def test_rewrite_integer_1d_tensor_is_left_alone(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class OffsetSanitizer:
        @staticmethod
        def sanitize(weights):
            return {name: tensor + 1 for name, tensor in weights.items()}

    mlx_tensor = torch.tensor([3, 4, 5])
    name, tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "language_model.token_ids", mlx_tensor, [(OffsetSanitizer, None)]
    )
    assert changed is False
    assert tensor is mlx_tensor


def test_rewrite_multi_pipeline_falls_through_to_matching_pipeline(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class RaisingSanitizer:
        @staticmethod
        def sanitize(weights):
            raise RuntimeError("first pipeline rejects everything")

    class ConvSanitizer:
        @staticmethod
        def sanitize(weights):
            return {
                name: tensor.permute(0, 2, 3, 1)
                for name, tensor in weights.items()
            }

    mlx_tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    pipelines = [[(RaisingSanitizer, None)], [(ConvSanitizer, None)]]
    new_name, new_tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.patch_embed.weight", mlx_tensor, pipelines
    )
    assert changed is True
    assert new_tensor.shape == (2, 5, 3, 4)


def test_rewrite_sanitizer_exception_returns_unchanged(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class RaisingSanitizer:
        @staticmethod
        def sanitize(weights):
            raise RuntimeError("boom")

    mlx_tensor = torch.zeros(2, 3, 4, 5)
    name, tensor, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "vision_tower.patch_embed.weight", mlx_tensor, [(RaisingSanitizer, None)]
    )
    assert (name, changed) == ("vision_tower.patch_embed.weight", False)
    assert tensor is mlx_tensor


def test_rewrite_model_prefixed_alias_families(monkeypatch):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)

    class RenamingSanitizer:
        @staticmethod
        def sanitize(weights):
            return {
                name.replace("model.visual.", "model.vision_tower."): tensor
                for name, tensor in weights.items()
            }

    tensor = torch.zeros(3, dtype=torch.float32)

    def fake_match(actual, expected):
        return True

    monkeypatch.setattr(mutils, "_mlx_arrays_match", fake_match)
    new_name, _, changed = mutils._rewrite_mlx_vlm_tensor_for_gguf(
        "model.vision_tower.embed.bias", tensor, [(RenamingSanitizer, None)]
    )
    assert changed is True
    assert new_name == "model.visual.embed.bias"


# --- Group 3: _MlxVlmSanitizeProxy (2-param sanitize path) ---


def test_two_param_sanitize_receives_config_proxy():
    import unsloth_zoo.mlx.utils as mutils

    seen = {}

    class InstanceStyleSanitizer:
        def sanitize(self, weights):
            seen["config"] = self.config
            seen["args"] = self.args
            scale = self.config["scale"]
            return {name: tensor * scale for name, tensor in weights.items()}

    config = {"scale": 2}
    out = mutils._call_mlx_vlm_sanitize(
        InstanceStyleSanitizer, config, {"w": torch.tensor([1.0, 2.0])}
    )
    assert seen["config"] is config
    assert seen["args"] is config
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0]))


def test_one_param_staticmethod_sanitize_skips_proxy():
    import unsloth_zoo.mlx.utils as mutils

    class StaticSanitizer:
        @staticmethod
        def sanitize(weights):
            return {f"renamed.{name}": tensor for name, tensor in weights.items()}

    out = mutils._call_mlx_vlm_sanitize(
        StaticSanitizer, {"unused": True}, {"w": torch.zeros(1)}
    )
    assert list(out) == ["renamed.w"]


def test_sanitize_missing_method_returns_weights_unchanged():
    import unsloth_zoo.mlx.utils as mutils

    class NoSanitize:
        pass

    weights = {"w": torch.zeros(1)}
    assert mutils._call_mlx_vlm_sanitize(NoSanitize, {}, weights) is weights


# --- Group 4: _prepare_vlm_gguf_export_directory end-to-end (real shards) ---


class _ConvAndRenameSanitizer:
    """HF -> MLX replay: visual. -> vision_tower. plus 4D OIHW -> OHWI."""

    @staticmethod
    def sanitize(weights):
        out = {}
        for name, tensor in weights.items():
            new_name = name.replace("visual.", "vision_tower.")
            if tensor.ndim == 4:
                tensor = tensor.permute(0, 2, 3, 1)
            out[new_name] = tensor
        return out


class _StripModelNamespaceSanitizer:
    """HF -> MLX replay: strip the canonical top-level model namespace."""

    @staticmethod
    def sanitize(weights):
        return {
            name[len("model."):] if name.startswith("model.") else name: tensor
            for name, tensor in weights.items()
        }


def _write_shard(path, tensors):
    from safetensors.torch import save_file
    save_file(tensors, str(path))


def _read_shard(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def _export_dir_with_shards(tmp_path, config, shards, index=None):
    out = tmp_path / "staging"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for filename, tensors in shards.items():
        _write_shard(out / filename, tensors)
    if index is not None:
        (out / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
    return out


def test_prepare_export_rewrites_real_shards_and_index(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [[(_ConvAndRenameSanitizer, None)]],
    )

    conv_mlx = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    text = torch.ones(8, 8)
    out = _export_dir_with_shards(
        tmp_path,
        config={"model_type": "fake_vlm"},
        shards={
            "model-00001-of-00002.safetensors": {
                "vision_tower.patch_embed.weight": conv_mlx,
            },
            "model-00002-of-00002.safetensors": {
                "language_model.layers.0.mlp.weight": text,
            },
        },
        index={
            "metadata": {},
            "weight_map": {
                "vision_tower.patch_embed.weight": "model-00001-of-00002.safetensors",
                "language_model.layers.0.mlp.weight": "model-00002-of-00002.safetensors",
            },
        },
    )

    rewritten = mutils._prepare_vlm_gguf_export_directory(out)
    assert rewritten == 1

    shard1 = _read_shard(out / "model-00001-of-00002.safetensors")
    assert list(shard1) == ["visual.patch_embed.weight"]
    # OHWI (2,3,4,5) -> OIHW (2,5,3,4)
    assert tuple(shard1["visual.patch_embed.weight"].shape) == (2, 5, 3, 4)
    assert torch.equal(
        shard1["visual.patch_embed.weight"].permute(0, 2, 3, 1), conv_mlx
    )

    shard2 = _read_shard(out / "model-00002-of-00002.safetensors")
    assert torch.equal(shard2["language_model.layers.0.mlp.weight"], text)

    index_data = json.loads(
        (out / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index_data["weight_map"] == {
        "language_model.layers.0.mlp.weight": "model-00002-of-00002.safetensors",
        "visual.patch_embed.weight": "model-00001-of-00002.safetensors",
    }


def test_prepare_export_restores_stripped_multimodal_namespaces_and_index(
    monkeypatch,
    tmp_path,
):
    import unsloth_zoo.mlx.utils as mutils
    monkeypatch.setattr(mutils, "mx", sys.modules["mlx.core"])
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [[(_StripModelNamespaceSanitizer, None)]],
    )

    tensors = {
        "audio_tower.layers.0.ffw.input_max": torch.tensor([1.0]),
        "audio_tower.layers.0.ffw.input_min": torch.tensor([-1.0]),
        "audio_tower.layers.0.ffw.output_max": torch.tensor([2.0]),
        "audio_tower.layers.0.ffw.output_min": torch.tensor([-2.0]),
        "vision_tower.encoder.layers.0.weight": torch.arange(6).reshape(2, 3),
        "embed_audio.weight": torch.arange(8).reshape(2, 4),
        "embed_vision.weight": torch.arange(10).reshape(2, 5),
    }
    unrelated_name = "multi_modal_projector.linear.weight"
    unrelated_tensor = torch.arange(24).reshape(2, 3, 4)
    shard_name = "model-00001-of-00001.safetensors"
    weight_map = {name: shard_name for name in (*tensors, unrelated_name)}
    out = _export_dir_with_shards(
        tmp_path,
        config={"model_type": "fake_vlm"},
        shards={shard_name: {**tensors, unrelated_name: unrelated_tensor}},
        index={"metadata": {"total_size": 123}, "weight_map": weight_map},
    )

    assert mutils._prepare_vlm_gguf_export_directory(out) == len(tensors)

    shard = _read_shard(out / shard_name)
    expected_names = {f"model.{name}" for name in tensors}
    assert set(shard) == expected_names | {unrelated_name}
    for old_name, original in tensors.items():
        assert torch.equal(shard[f"model.{old_name}"], original)
    assert torch.equal(shard[unrelated_name], unrelated_tensor)

    index_data = json.loads(
        (out / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index_data["metadata"] == {"total_size": 123}
    assert index_data["weight_map"] == {
        **{f"model.{name}": shard_name for name in tensors},
        unrelated_name: shard_name,
    }


def test_prepare_export_duplicate_rewrite_name_raises(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [[(_ConvAndRenameSanitizer, None)]],
    )

    conv = torch.zeros(2, 3, 4, 5)
    out = _export_dir_with_shards(
        tmp_path,
        config={"model_type": "fake_vlm"},
        shards={
            "model-00001-of-00001.safetensors": {
                # Both names land on visual.embed.weight after the rewrite.
                "vision_tower.embed.weight": conv,
                "visual.embed.weight": conv.permute(0, 3, 1, 2).contiguous(),
            },
        },
    )
    with pytest.raises(RuntimeError, match="duplicate tensor name"):
        mutils._prepare_vlm_gguf_export_directory(out)


def test_prepare_export_missing_index_is_fine(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [[(_ConvAndRenameSanitizer, None)]],
    )

    out = _export_dir_with_shards(
        tmp_path,
        config={"model_type": "fake_vlm"},
        shards={
            "model.safetensors": {
                "vision_tower.embed.weight": torch.zeros(2, 3, 4, 5),
            },
        },
    )
    assert mutils._prepare_vlm_gguf_export_directory(out) == 1
    assert not (out / "model.safetensors.index.json").exists()


def test_prepare_export_no_shards_returns_zero(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [[(_ConvAndRenameSanitizer, None)]],
    )
    out = _export_dir_with_shards(
        tmp_path, config={"model_type": "fake_vlm"}, shards={}
    )
    assert mutils._prepare_vlm_gguf_export_directory(out) == 0


def test_prepare_export_unicode_shard_filename(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils
    _patch_mlx_tensor_helpers_for_torch(monkeypatch, mutils)
    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [[(_ConvAndRenameSanitizer, None)]],
    )

    out = _export_dir_with_shards(
        tmp_path,
        config={"model_type": "fake_vlm"},
        shards={
            "模型-shard.safetensors": {
                "vision_tower.embed.weight": torch.zeros(2, 3, 4, 5),
            },
        },
    )
    assert mutils._prepare_vlm_gguf_export_directory(out) == 1
    assert list(_read_shard(out / "模型-shard.safetensors")) == [
        "visual.embed.weight"
    ]


def _nextn_model(num_layers):
    layer = types.SimpleNamespace()
    inner = types.SimpleNamespace(layers=[layer] * num_layers)
    language_model = types.SimpleNamespace(model=inner)
    return types.SimpleNamespace(language_model=language_model)


def test_nextn_decrement_branch_keeps_partial_layers(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    # 2 of the 3 advertised speculative layers exist: decrement, do not pop.
    config = {
        "text_config": {"num_hidden_layers": 4, "num_nextn_predict_layers": 3},
    }
    changed = mutils._sync_gguf_nextn_layer_config(config, _nextn_model(6))
    assert changed is True
    assert config["text_config"]["num_nextn_predict_layers"] == 2


def test_nextn_pop_branch_when_no_speculative_layers_exported(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    config = {
        "text_config": {"num_hidden_layers": 4, "num_nextn_predict_layers": 1},
    }
    changed = mutils._sync_gguf_nextn_layer_config(config, _nextn_model(4))
    assert changed is True
    assert "num_nextn_predict_layers" not in config["text_config"]


def test_nextn_skips_when_exported_fewer_than_hidden_layers():
    import unsloth_zoo.mlx.utils as mutils

    config = {
        "text_config": {"num_hidden_layers": 6, "num_nextn_predict_layers": 1},
    }
    assert mutils._sync_gguf_nextn_layer_config(config, _nextn_model(4)) is False
    assert config["text_config"]["num_nextn_predict_layers"] == 1


def test_nextn_skips_when_all_advertised_layers_exported():
    import unsloth_zoo.mlx.utils as mutils

    config = {
        "text_config": {"num_hidden_layers": 4, "num_nextn_predict_layers": 2},
    }
    assert mutils._sync_gguf_nextn_layer_config(config, _nextn_model(6)) is False
    assert config["text_config"]["num_nextn_predict_layers"] == 2


def test_nextn_handles_language_and_thinker_nested_configs():
    import unsloth_zoo.mlx.utils as mutils

    config = {
        "language_config": {"num_hidden_layers": 4, "mtp_num_hidden_layers": 2},
        "thinker_config": {
            "text_config": {"num_hidden_layers": 4, "nextn_predict_layers": 2},
        },
    }
    changed = mutils._sync_gguf_nextn_layer_config(config, _nextn_model(4))
    assert changed is True
    assert "mtp_num_hidden_layers" not in config["language_config"]
    assert "nextn_predict_layers" not in config["thinker_config"]["text_config"]


# --- Group 5: _copy_source_sidecars filesystem edges ---


def test_sidecars_src_equals_dst_copies_nothing(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    assert mutils._copy_source_sidecars(tmp_path, tmp_path) == 0


def test_sidecars_follow_file_symlinks_and_skip_broken_ones(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    real = tmp_path / "real_preprocessor.json"
    real.write_text('{"size": 224}', encoding="utf-8")
    (src / "preprocessor_config.json").symlink_to(real)
    (src / "broken_link.json").symlink_to(tmp_path / "does_not_exist.json")

    assert mutils._copy_source_sidecars(src, dst) == 1
    assert json.loads(
        (dst / "preprocessor_config.json").read_text(encoding="utf-8")
    ) == {"size": 224}
    assert not (dst / "broken_link.json").exists()


def test_sidecars_skip_suffixless_and_uppercase_suffix_files(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "LICENSE").write_text("MIT", encoding="utf-8")
    # Suffix matching is exact lowercase; HF sidecars are lowercase by convention.
    (src / "PREPROCESSOR_CONFIG.JSON").write_text("{}", encoding="utf-8")
    assert mutils._copy_source_sidecars(src, dst) == 0


def test_sidecars_unicode_names_and_spaces_in_paths(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    src = tmp_path / "my model dir"
    dst = tmp_path / "out dir"
    src.mkdir()
    dst.mkdir()
    (src / "tokenizer_配置.json").write_text('{"ok": true}', encoding="utf-8")
    assert mutils._copy_source_sidecars(src, dst) == 1
    assert (dst / "tokenizer_配置.json").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="permission checks no-op as root")
def test_sidecars_unreadable_source_propagates_loudly(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    secret = src / "processor_config.json"
    secret.write_text("{}", encoding="utf-8")
    secret.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            mutils._copy_source_sidecars(src, dst)
    finally:
        secret.chmod(0o644)


def test_sidecars_windows_style_path_object_does_not_crash():
    import unsloth_zoo.mlx.utils as mutils

    # On POSIX a PureWindowsPath coerces to a missing relative path: return 0.
    assert mutils._copy_source_sidecars(PureWindowsPath("C:/no/such"), "out") == 0


# --- Group 6: _read_json_file non-dict payloads + processor repair robustness ---


@pytest.mark.parametrize("payload", ["[]", '"just a string"', "null", "3.14"])
def test_read_json_file_non_dict_payload_yields_dict(tmp_path, payload):
    """Non-object JSON must degrade to {}; callers .get() on the result."""
    import unsloth_zoo.mlx.loader as loader

    sidecar = tmp_path / "processor_config.json"
    sidecar.write_text(payload, encoding="utf-8")
    result = loader._read_json_file(sidecar)
    assert isinstance(result, dict)


def test_repair_survives_non_dict_sidecar_json(tmp_path):
    import unsloth_zoo.mlx.loader as loader

    (tmp_path / "processor_config.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("null", encoding="utf-8")
    processor = types.SimpleNamespace(image_processor=None, tokenizer=None)
    repaired = loader._repair_degraded_vlm_processor(
        processor, str(tmp_path), "fake_type"
    )
    assert repaired is processor


def test_read_json_file_invalid_utf8_returns_empty(tmp_path):
    import unsloth_zoo.mlx.loader as loader

    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00garbage")
    assert loader._read_json_file(bad) == {}


def test_read_json_file_directory_path_returns_empty(tmp_path):
    import unsloth_zoo.mlx.loader as loader

    assert loader._read_json_file(tmp_path) == {}


@pytest.mark.skipif(os.geteuid() == 0, reason="permission checks no-op as root")
def test_read_json_file_permission_error_returns_empty(tmp_path):
    import unsloth_zoo.mlx.loader as loader

    locked = tmp_path / "locked.json"
    locked.write_text("{}", encoding="utf-8")
    locked.chmod(0o000)
    try:
        assert loader._read_json_file(locked) == {}
    finally:
        locked.chmod(0o644)


def test_read_json_file_accepts_str_and_path(tmp_path):
    import unsloth_zoo.mlx.loader as loader

    sidecar = tmp_path / "ok.json"
    sidecar.write_text('{"a": 1}', encoding="utf-8")
    assert loader._read_json_file(str(sidecar)) == {"a": 1}
    assert loader._read_json_file(sidecar) == {"a": 1}


# --- Group 7: _get_model_config fallback ladder ---


def test_get_model_config_to_dict_returning_non_dict_falls_through():
    import unsloth_zoo.mlx.utils as mutils

    @dataclasses.dataclass
    class Args:
        model_type: str = "llama"

    class WeirdConfig:
        def to_dict(self):
            return ["not", "a", "dict"]

    model = types.SimpleNamespace(config=WeirdConfig(), args=Args())
    assert mutils._get_model_config(model) == {"model_type": "llama"}


def test_get_model_config_non_dataclass_args_yields_empty():
    import unsloth_zoo.mlx.utils as mutils

    model = types.SimpleNamespace(args=types.SimpleNamespace(model_type="x"))
    assert mutils._get_model_config(model) == {}


def test_get_model_config_no_sources_yields_empty():
    import unsloth_zoo.mlx.utils as mutils

    assert mutils._get_model_config(object()) == {}


def test_get_model_config_slotted_dataclass():
    import unsloth_zoo.mlx.utils as mutils

    @dataclasses.dataclass(**({"slots": True} if sys.version_info >= (3, 10) else {}))
    class SlottedConfig:
        model_type: str = "gemma3"
        hidden_size: int = 16

    model = types.SimpleNamespace(config=SlottedConfig())
    assert mutils._get_model_config(model) == {
        "model_type": "gemma3",
        "hidden_size": 16,
    }


def test_get_model_config_non_dict_private_config_falls_through():
    import unsloth_zoo.mlx.utils as mutils

    @dataclasses.dataclass
    class Config:
        model_type: str = "qwen3"

    model = types.SimpleNamespace(_config="not a dict", config=Config())
    assert mutils._get_model_config(model) == {"model_type": "qwen3"}


# --- Group 8: _save_mlx_config routing and quantization key handling ---


def _capture_save_config(monkeypatch, module_name):
    calls = {}
    fake = types.ModuleType(module_name)

    def fake_save_config(config, path):
        calls["config"] = config
        calls["path"] = Path(path)

    fake.save_config = fake_save_config
    monkeypatch.setitem(sys.modules, module_name, fake)
    return calls


def test_save_config_vlm_clobbers_stale_quantization_config(monkeypatch, tmp_path):
    """Live MLX `quantization` wins over a stale HF `quantization_config`."""
    import unsloth_zoo.mlx.utils as mutils

    calls = _capture_save_config(monkeypatch, "mlx_vlm.utils")
    config = {
        "vision_config": {},
        "quantization": {"bits": 4},
        "quantization_config": {"bits": 8, "stale": True},
    }
    mutils._save_mlx_config(config, tmp_path / "config.json", is_vlm=True)
    assert calls["config"]["quantization_config"] == {"bits": 4}
    # Caller's dict must stay untouched.
    assert config["quantization_config"] == {"bits": 8, "stale": True}


def test_save_config_text_routes_to_mlx_lm_without_mirroring(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    lm_calls = _capture_save_config(monkeypatch, "mlx_lm.utils")
    vlm_calls = _capture_save_config(monkeypatch, "mlx_vlm.utils")
    config = {"model_type": "llama", "quantization": {"bits": 4}}
    mutils._save_mlx_config(config, tmp_path / "config.json", is_vlm=False)
    assert "config" in lm_calls and "config" not in vlm_calls
    assert "quantization_config" not in lm_calls["config"]


def test_save_config_vlm_without_quantization_key_adds_nothing(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = _capture_save_config(monkeypatch, "mlx_vlm.utils")
    config = {"vision_config": {}, "quantization_config": {"bits": 8}}
    mutils._save_mlx_config(config, tmp_path / "config.json", is_vlm=True)
    assert calls["config"]["quantization_config"] == {"bits": 8}
    assert "quantization" not in calls["config"]


# --- Group 9: _is_vlm_model real body ---


def test_is_vlm_explicit_flag_overrides_structure():
    import unsloth_zoo.mlx.utils as mutils

    vlm_shaped = types.SimpleNamespace(
        language_model=object(), vision_tower=object(), _is_vlm_model=False
    )
    assert mutils._is_vlm_model(vlm_shaped) is False
    text_shaped = types.SimpleNamespace(_is_vlm_model=True)
    assert mutils._is_vlm_model(text_shaped) is True
    falsy_flag = types.SimpleNamespace(language_model=object(), _is_vlm_model=0)
    assert mutils._is_vlm_model(falsy_flag) is False


@pytest.mark.parametrize(
    "attr",
    [
        "vision_tower",
        "vision_model",
        "vision_encoder",
        "visual",
        "multi_modal_projector",
        "audio_tower",
    ],
)
def test_is_vlm_detects_each_modal_attribute(attr):
    import unsloth_zoo.mlx.utils as mutils

    model = types.SimpleNamespace(language_model=object(), **{attr: object()})
    assert mutils._is_vlm_model(model) is True


def test_is_vlm_requires_language_model():
    import unsloth_zoo.mlx.utils as mutils

    assert mutils._is_vlm_model(types.SimpleNamespace(vision_tower=object())) is False
    assert mutils._is_vlm_model(types.SimpleNamespace(language_model=object())) is False
    assert mutils._is_vlm_model(object()) is False


# --- Group 10: save_merged_model contracts ---


def _install_fake_lm_save(monkeypatch):
    calls = {"dequantize": 0}
    fake_mlx_lm_utils = types.ModuleType("mlx_lm.utils")

    def fake_dequantize_model(model):
        calls["dequantize"] += 1
        return model

    def fake_save_model(path, model, donate_model=False):
        Path(path).mkdir(parents=True, exist_ok=True)

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
    return calls


class _Tokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_merged_save_tied_embeddings_dequantizes_and_strips(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = _install_fake_lm_save(monkeypatch)

    class Model:
        _config = {
            "model_type": "gemma3_text",
            "tie_word_embeddings": True,
            "quantization": {"bits": 4},
        }

        def eval(self):
            pass

        def named_modules(self):
            return []

    mutils.save_merged_model(Model(), _Tokenizer(), tmp_path, dequantize=True)
    assert calls["dequantize"] == 1
    assert "quantization" not in calls["saved_config"]
    assert calls["saved_config"]["tie_word_embeddings"] is True


def test_merged_save_empty_fused_linears_skips_update(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    _install_fake_lm_save(monkeypatch)

    class Model:
        # No update_modules attribute: the no-LoRA path must never call it.
        _config = {"model_type": "llama"}

        def eval(self):
            pass

        def named_modules(self):
            return [("plain", types.SimpleNamespace())]

    mutils.save_merged_model(Model(), _Tokenizer(), tmp_path, dequantize=False)


def test_merged_save_without_private_config_uses_config_object(
    monkeypatch, tmp_path
):
    import unsloth_zoo.mlx.utils as mutils

    calls = _install_fake_lm_save(monkeypatch)

    @dataclasses.dataclass
    class Config:
        model_type: str = "llama"

    class Model:
        config = Config()

        def eval(self):
            pass

        def named_modules(self):
            return []

    mutils.save_merged_model(Model(), _Tokenizer(), tmp_path, dequantize=True)
    assert calls["dequantize"] == 1
    assert calls["saved_config"]["model_type"] == "llama"


def test_merged_save_keeps_quantization_when_not_dequantizing(
    monkeypatch, tmp_path
):
    import unsloth_zoo.mlx.utils as mutils

    calls = _install_fake_lm_save(monkeypatch)

    class Model:
        _config = {"model_type": "llama", "quantization": {"bits": 4}}

        def eval(self):
            pass

        def named_modules(self):
            return []

    mutils.save_merged_model(Model(), _Tokenizer(), tmp_path, dequantize=False)
    assert calls["dequantize"] == 0
    assert calls["saved_config"]["quantization"] == {"bits": 4}


# --- Group 11: save_pretrained_gguf first_conversion derivation + quantize step ---


def _gguf_export_scaffold(monkeypatch, tmp_path):
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
        calls["merge_count"] = calls.get("merge_count", 0) + 1
        Path(path).mkdir(parents=True, exist_ok=True)

    def fake_download():
        patched = llama_root / "unsloth_convert_hf_to_gguf.py"
        patched.write_text("# patched", encoding="utf-8")
        return str(patched), {"LlamaForCausalLM"}, set()

    def fake_convert_to_gguf(**kwargs):
        calls["convert_kwargs"] = kwargs
        calls["convert_count"] = calls.get("convert_count", 0) + 1
        output = Path(
            f"{kwargs['model_name']}.{kwargs['quantization_type'].upper()}.gguf"
        )
        output.write_bytes(b"GGUF-intermediate")

    def fake_quantize_gguf(**kwargs):
        # "quantize_kwargs" keeps the last call (pre-existing single-quant
        # assertions); "quantize_calls" accumulates for multi-quant exports.
        calls["quantize_kwargs"] = kwargs
        calls.setdefault("quantize_calls", []).append(kwargs)
        Path(kwargs["output_gguf"]).write_bytes(b"GGUF-final")

    monkeypatch.setattr(mutils, "save_merged_model", fake_save_merged_model)
    monkeypatch.setattr(mutils, "_is_vlm_model", lambda model: False)
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "unused"))
    monkeypatch.setattr(
        llama_cpp,
        "check_llama_cpp",
        lambda llama_cpp_folder: (str(quantizer), str(converter)),
    )
    monkeypatch.setattr(
        llama_cpp, "_download_convert_hf_to_gguf", fake_download
    )
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", fake_convert_to_gguf)
    monkeypatch.setattr(llama_cpp, "quantize_gguf", fake_quantize_gguf)
    return mutils, calls


def test_gguf_default_first_conversion_quantizes_through_bf16(
    monkeypatch, tmp_path
):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    model = types.SimpleNamespace(_hf_repo="org/EdgeModel")
    mutils.save_pretrained_gguf(
        model,
        tokenizer=object(),
        save_directory=out,
        quantization_method="fast_quantized",  # q8_0, no first_conversion given
    )
    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    assert calls["quantize_kwargs"]["quant_type"] == "q8_0"
    assert not (out / "EdgeModel.BF16.gguf").exists()
    assert (out / "EdgeModel.Q8_0.gguf").read_bytes() == b"GGUF-final"


def test_gguf_default_first_conversion_not_quantized_skips_quantizer(
    monkeypatch, tmp_path
):
    import unsloth_zoo.llama_cpp as llama_cpp

    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    monkeypatch.setattr(
        llama_cpp,
        "quantize_gguf",
        lambda **kwargs: pytest.fail("quantize_gguf must not run for bf16 target"),
    )
    out = tmp_path / "out"
    model = types.SimpleNamespace(_hf_repo="org/EdgeModel")
    mutils.save_pretrained_gguf(
        model,
        tokenizer=object(),
        save_directory=out,
        quantization_method="not_quantized",
    )
    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    assert (out / "EdgeModel.BF16.gguf").exists()


# --- Group 11b: multi-quant export (CUDA parity, unsloth/save.py:1323) ---


def _export(mutils, out, quantization_method, **kwargs):
    return mutils.save_pretrained_gguf(
        types.SimpleNamespace(_hf_repo="org/EdgeModel"),
        tokenizer=object(),
        save_directory=out,
        quantization_method=quantization_method,
        **kwargs,
    )


def test_gguf_list_produces_every_quant_from_one_conversion(monkeypatch, tmp_path):
    """A list is the documented CUDA contract; before this it raised
    TypeError: unhashable type: 'list' inside the alias lookup."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["q4_k_m", "q8_0", "q5_k_m"])

    # The expensive steps run once; each extra quant only costs its own pass.
    assert calls["merge_count"] == 1
    assert calls["convert_count"] == 1
    assert [c["quant_type"] for c in calls["quantize_calls"]] == [
        "q4_k_m", "q8_0", "q5_k_m",
    ]
    for name in ("Q4_K_M", "Q8_0", "Q5_K_M"):
        assert (out / f"EdgeModel.{name}.gguf").exists()
    # Every quant is produced from the shared intermediate, which is scratch.
    assert {c["input_gguf"] for c in calls["quantize_calls"]} == {
        str(out / "EdgeModel.BF16.gguf")
    }
    assert not (out / "EdgeModel.BF16.gguf").exists()


def test_gguf_tuple_accepted_and_aliases_resolve_per_element(monkeypatch, tmp_path):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ("fast_quantized", "quantized"))
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q8_0", "q4_k_m"]


def test_gguf_list_keeps_intermediate_when_it_is_also_requested(monkeypatch, tmp_path):
    """bf16 alongside a k-quant is a requested artifact, not scratch -- the
    single-quant path deleted the intermediate unconditionally."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["bf16", "q4_k_m"])

    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    # bf16 comes straight from convert_hf_to_gguf; only q4_k_m is quantized.
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q4_k_m"]
    assert (out / "EdgeModel.BF16.gguf").exists()
    assert (out / "EdgeModel.Q4_K_M.gguf").exists()


def test_gguf_list_deduplicates_preserving_order(monkeypatch, tmp_path):
    """Repeats would re-run llama-quantize onto an identical output path."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["q4_k_m", "q8_0", "q4_k_m", "quantized"])
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q4_k_m", "q8_0"]


def test_gguf_rejects_bad_type_before_the_expensive_merge(monkeypatch, tmp_path):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    with pytest.raises(TypeError, match="must be a string"):
        _export(mutils, out, 4)
    with pytest.raises(TypeError, match="every quantization_method entry"):
        _export(mutils, out, ["q4_k_m", 4])
    with pytest.raises(ValueError, match="empty list"):
        _export(mutils, out, [])
    # Argument errors must not cost a LoRA merge or a GGUF conversion.
    assert "merge_count" not in calls
    assert "convert_count" not in calls


def test_gguf_single_string_behaviour_is_unchanged(monkeypatch, tmp_path):
    """Regression guard: the scalar path must keep its exact old semantics."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, "q4_k_m")
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q4_k_m"]
    assert (out / "EdgeModel.Q4_K_M.gguf").exists()
    assert not (out / "EdgeModel.BF16.gguf").exists()


def test_gguf_explicit_first_conversion_still_honored_for_a_list(monkeypatch, tmp_path):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["q4_k_m", "q8_0"], first_conversion="f32")
    assert calls["convert_kwargs"]["quantization_type"] == "f32"
    assert {c["input_gguf"] for c in calls["quantize_calls"]} == {
        str(out / "EdgeModel.F32.gguf")
    }
    assert not (out / "EdgeModel.F32.gguf").exists()


def test_gguf_list_produces_full_precision_targets_that_are_not_the_intermediate(
    monkeypatch, tmp_path
):
    """f16 alongside a k-quant must still be emitted. llama-quantize produces
    f16/bf16/f32 too, so a full-precision target is only free when it IS the
    intermediate -- exempting all of them silently drops the request."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["f16", "q4_k_m"])

    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["f16", "q4_k_m"]
    assert (out / "EdgeModel.F16.gguf").exists()
    assert (out / "EdgeModel.Q4_K_M.gguf").exists()
    # bf16 was only scratch here - it was never requested.
    assert not (out / "EdgeModel.BF16.gguf").exists()


def test_gguf_two_full_precision_targets_both_land(monkeypatch, tmp_path):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["bf16", "f16"])

    # bf16 is the intermediate and is requested, so only f16 needs a pass.
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["f16"]
    assert (out / "EdgeModel.BF16.gguf").exists()
    assert (out / "EdgeModel.F16.gguf").exists()


def test_gguf_keeps_intermediate_when_nothing_was_quantized(monkeypatch, tmp_path):
    """A full-precision target reached via an explicit first_conversion runs no
    quantize pass, so the converted file IS the export and must survive."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, "not_quantized", first_conversion="f16")

    assert "quantize_calls" not in calls
    assert (out / "EdgeModel.F16.gguf").exists()


# --- Group 11c: quant-spec normalization ------------------------------------
#
# Output filenames are derived as `{base}.{quant.upper()}.gguf`, so two spellings
# of one type name are the SAME artifact on disk. Every downstream decision
# (which intermediate to convert to, which targets still need a llama-quantize
# pass, whether the intermediate is scratch) compares quant names as strings, so
# they must all see one canonical spelling. llama.cpp's own side already works
# this way: unsloth_zoo/llama_cpp.py:2174 `check_quantization_type` lowercases
# before validating.


def test_gguf_case_variant_target_is_the_same_artifact(monkeypatch, tmp_path):
    """`["BF16", "q4_k_m"]` is the llama.cpp spelling of `["bf16", "q4_k_m"]`.

    Un-normalized it produced an in-place `llama-quantize BF16.gguf BF16.gguf`
    pass and then deleted the requested BF16 artifact as if it were scratch.
    """
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["BF16", "q4_k_m"])

    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    # BF16 IS the intermediate: it must not be re-quantized onto its own path.
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q4_k_m"]
    for c in calls["quantize_calls"]:
        assert c["input_gguf"] != c["output_gguf"]
    assert (out / "EdgeModel.BF16.gguf").exists()
    assert (out / "EdgeModel.Q4_K_M.gguf").exists()


def test_gguf_case_variant_first_conversion_matches_requested_target(
    monkeypatch, tmp_path
):
    """`first_conversion` is the other half of the same spec and needs the same
    normalization: spelled `"BF16"` against a requested `"bf16"` it both ran an
    in-place quantize pass and deleted the requested artifact."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["bf16", "q4_k_m"], first_conversion="BF16")

    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q4_k_m"]
    for c in calls["quantize_calls"]:
        assert c["input_gguf"] != c["output_gguf"]
    assert (out / "EdgeModel.BF16.gguf").exists()


def test_gguf_dedup_is_case_insensitive_and_strips(monkeypatch, tmp_path):
    """Dedup exists because repeats re-run llama-quantize onto an identical
    output path -- which is exactly what case/whitespace variants also do."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["Q4_K_M", "q4_k_m", " q4_k_m ", "Q8_0"])
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q4_k_m", "q8_0"]


def test_gguf_case_variant_alias_resolves(monkeypatch, tmp_path):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["Fast_Quantized"])
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["q8_0"]


def test_gguf_rejects_non_string_first_conversion_before_the_merge(
    monkeypatch, tmp_path
):
    """Same "reject before the expensive merge" contract the quant list has."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    with pytest.raises(TypeError, match="first_conversion"):
        _export(mutils, out, ["q4_k_m"], first_conversion=["bf16"])
    assert "merge_count" not in calls
    assert "convert_count" not in calls


# --- Group 11d: singleton list is a list, not a scalar ----------------------


def test_gguf_singleton_list_full_precision_is_still_emitted(monkeypatch, tmp_path):
    """Upstream runs a llama-quantize pass for every requested method that is not
    the intermediate, with no full-precision exemption
    (unslothai/unsloth unsloth/save.py:2067-2070). The scalar carve-out here is a
    pre-PR MLX behaviour; a singleton LIST is the new list contract and must
    emit what was asked for."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["f16"], first_conversion="bf16")

    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["f16"]
    assert (out / "EdgeModel.F16.gguf").exists()
    # bf16 was never requested, so the intermediate is scratch.
    assert not (out / "EdgeModel.BF16.gguf").exists()


def test_gguf_scalar_full_precision_keeps_its_pre_pr_behaviour(monkeypatch, tmp_path):
    """The scalar form must NOT change: pre-PR a lone full-precision string ran
    no quantize pass whatever `first_conversion` said."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, "f16", first_conversion="bf16")

    assert "quantize_calls" not in calls
    assert (out / "EdgeModel.BF16.gguf").exists()


# --- Group 11e: intermediate must not down-convert an f32 request -----------


def test_gguf_f32_request_converts_directly_to_f32(monkeypatch, tmp_path):
    """bf16 and f16 both embed exactly in f32, so an f32 intermediate is never a
    precision regression for the other targets -- but a bf16 intermediate makes
    the requested F32.gguf f32-shaped bf16 data."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["f32", "f16"])

    assert calls["convert_kwargs"]["quantization_type"] == "f32"
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["f16"]
    assert (out / "EdgeModel.F32.gguf").exists()
    assert (out / "EdgeModel.F16.gguf").exists()


def test_gguf_f32_request_alongside_a_kquant_still_converts_to_f32(
    monkeypatch, tmp_path
):
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["f32", "q4_k_m"])

    assert calls["convert_kwargs"]["quantization_type"] == "f32"
    assert {c["input_gguf"] for c in calls["quantize_calls"]} == {
        str(out / "EdgeModel.F32.gguf")
    }
    assert (out / "EdgeModel.F32.gguf").exists()
    assert (out / "EdgeModel.Q4_K_M.gguf").exists()


def test_gguf_without_f32_the_intermediate_stays_bf16(monkeypatch, tmp_path):
    """Deliberate non-change. f16 and bf16 are not orderable -- bf16 has the
    wider exponent, f16 the longer mantissa -- so promoting a requested f16 to
    the intermediate would CLIP the range of a bf16 checkpoint. Only f32, which
    represents all three exactly, is a safe promotion."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["f16", "q4_k_m"])
    assert calls["convert_kwargs"]["quantization_type"] == "bf16"


def test_gguf_two_full_precision_no_f32_keeps_bf16_intermediate(
    monkeypatch, tmp_path
):
    """Companion to the above for the all-full-precision set."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["bf16", "f16"])
    assert calls["convert_kwargs"]["quantization_type"] == "bf16"


def test_gguf_explicit_first_conversion_beats_the_f32_promotion(
    monkeypatch, tmp_path
):
    """An explicit `first_conversion` is always honoured; the derivation only
    runs when it is None (unslothai/unsloth unsloth/save.py:1953)."""
    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    out = tmp_path / "out"
    _export(mutils, out, ["f32", "q4_k_m"], first_conversion="bf16")
    assert calls["convert_kwargs"]["quantization_type"] == "bf16"
    assert [c["quant_type"] for c in calls["quantize_calls"]] == ["f32", "q4_k_m"]


def test_mlx_model_method_forwards_a_quant_list(monkeypatch, tmp_path):
    """The user-facing hop: model.save_pretrained_gguf(...) in loader.py."""
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mutils

    seen = {}
    monkeypatch.setattr(
        mutils, "save_pretrained_gguf",
        lambda model, tokenizer, save_directory, **kw: seen.update(kw),
    )
    model = types.SimpleNamespace(_tokenizer=object())
    loader._mlx_save_pretrained_gguf(
        model, tmp_path / "out", quantization_method=["q4_k_m", "q8_0"],
    )
    assert seen["quantization_method"] == ["q4_k_m", "q8_0"]


def test_push_to_hub_gguf_forwards_a_quant_list(monkeypatch, tmp_path):
    mutils, _ = _gguf_export_scaffold(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(
        mutils, "save_pretrained_gguf",
        lambda model, tokenizer, save_directory, **kw: seen.update(kw),
    )
    monkeypatch.setattr(
        mutils, "HfApi", None, raising=False,
    )
    uploaded = {}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, **k):
            uploaded.update(k)

        def upload_large_folder(self, **k):
            uploaded.update(k)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(HfApi=_FakeApi),
    )
    mutils.push_to_hub_gguf(
        types.SimpleNamespace(_hf_repo="org/EdgeModel"),
        tokenizer=object(),
        save_directory=tmp_path / "out",
        repo_id="org/EdgeModel",
        quantization_method=["q4_k_m", "q8_0"],
    )
    assert seen["quantization_method"] == ["q4_k_m", "q8_0"]


# --- Group 12: concurrency over the patcher env mutation ---


def test_concurrent_gguf_exports_serialize_env_mutation(monkeypatch, tmp_path):
    import unsloth_zoo.llama_cpp as llama_cpp

    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)

    state = {"inside": 0, "max_inside": 0, "env_values": []}
    state_lock = threading.Lock()

    def slow_download():
        with state_lock:
            state["inside"] += 1
            state["max_inside"] = max(state["max_inside"], state["inside"])
            state["env_values"].append(
                os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
            )
        import time
        time.sleep(0.01)
        with state_lock:
            state["inside"] -= 1
        patched = tmp_path / "llama.cpp" / "unsloth_convert_hf_to_gguf.py"
        patched.write_text("# patched", encoding="utf-8")
        return str(patched), {"LlamaForCausalLM"}, set()

    monkeypatch.setattr(llama_cpp, "_download_convert_hf_to_gguf", slow_download)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising=False)

    errors = []

    def export(i):
        try:
            out = tmp_path / f"out{i}"
            model = types.SimpleNamespace(_hf_repo=f"org/Model{i}")
            mutils.save_pretrained_gguf(
                model,
                tokenizer=object(),
                save_directory=out,
                quantization_method="not_quantized",
                first_conversion="f16",
            )
        except Exception as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=export, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # The module lock serializes every patcher invocation.
    assert state["max_inside"] == 1
    expected = str(tmp_path / "llama.cpp")
    assert state["env_values"] == [expected] * 6
    assert os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR") is None


def test_gguf_export_respects_preexisting_scripts_dir_override(
    monkeypatch, tmp_path
):
    import unsloth_zoo.llama_cpp as llama_cpp

    mutils, calls = _gguf_export_scaffold(monkeypatch, tmp_path)
    seen = {}

    def fake_download():
        seen["env"] = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
        patched = tmp_path / "llama.cpp" / "unsloth_convert_hf_to_gguf.py"
        patched.write_text("# patched", encoding="utf-8")
        return str(patched), {"LlamaForCausalLM"}, set()

    monkeypatch.setattr(llama_cpp, "_download_convert_hf_to_gguf", fake_download)
    custom = str(tmp_path / "user_override")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", custom)

    model = types.SimpleNamespace(_hf_repo="org/EdgeModel")
    mutils.save_pretrained_gguf(
        model,
        tokenizer=object(),
        save_directory=tmp_path / "out",
        quantization_method="not_quantized",
        first_conversion="f16",
    )
    # The wrapper must NOT clobber a user-set override, and must keep it after.
    assert seen["env"] == custom
    assert os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR") == custom


# --- Group 13: converter placement contract (monolith vs package layout) ---

_PACKAGE_ENTRYPOINT = b"""\
#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
    print_registered_models,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--outtype", default="f16")
    parser.add_argument("--vocab-only", action="store_true")
    return parser.parse_args()
"""

_PACKAGE_INIT_PY = b"""\
from __future__ import annotations
from .base import ModelBase, ModelType


TEXT_MODEL_MAP: dict[str, str] = {
    "LlamaForCausalLM": "llama",
    "Qwen3ForCausalLM": "qwen",
}


MMPROJ_MODEL_MAP: dict[str, str] = {
    "Gemma3ForConditionalGeneration": "gemma",
}


def load_all_models() -> None:
    pass


def get_model_class(name, mmproj=False):
    return ModelBase
"""

_PACKAGE_BASE_PY = b"""\
import gguf
from enum import IntEnum


class ModelType(IntEnum):
    TEXT = 0
    MMPROJ = 1


class ModelBase:
    _model_classes = {ModelType.TEXT: {}, ModelType.MMPROJ: {}}

    def prepare_metadata(self, vocab_only):
        total_params, shared_params, expert_params, expert_count = (0, 0, 0, 0)

        self.metadata = gguf.Metadata.load(self.metadata_override, self.dir_model_card, self.model_name, total_params)

        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id
"""

_PACKAGE_QWEN_PY = b"""\
from .base import ModelBase


class Qwen2MoeModel(ModelBase):
    def set_gguf_parameters(self):
        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        return n_experts
"""

_MONOLITH = b"""\
import argparse
import gguf
from enum import IntEnum


class ModelType(IntEnum):
    TEXT = 0
    MMPROJ = 1


class ModelBase:
    _model_classes = {ModelType.TEXT: {"LlamaForCausalLM": object}, ModelType.MMPROJ: {}}

    def prepare_metadata(self):
        self.metadata = gguf.Metadata.load(override, card, name, params)

        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--outtype", default="f16")
    parser.add_argument("--vocab-only", action="store_true")
    return parser.parse_args()
"""


def _package_tree(root):
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_bytes(_PACKAGE_ENTRYPOINT)
    conv = root / "conversion"
    conv.mkdir()
    (conv / "__init__.py").write_bytes(_PACKAGE_INIT_PY)
    (conv / "base.py").write_bytes(_PACKAGE_BASE_PY)
    (conv / "qwen.py").write_bytes(_PACKAGE_QWEN_PY)
    return root


def test_package_layout_patched_script_lands_beside_conversion(
    tmp_path, monkeypatch
):
    import unsloth_zoo.llama_cpp as llama_cpp

    root = _package_tree(tmp_path / "custom_llama_cpp")
    default_dir = tmp_path / "default_dir"
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(default_dir))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(root))
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, _ = llama_cpp._download_convert_hf_to_gguf(
            "edge_package_placement"
        )
    finally:
        llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()

    # Patched entrypoint must sit beside conversion/ for sibling imports.
    assert Path(patched_path).parent == root
    assert "LlamaForCausalLM" in text_archs
    assert not default_dir.exists() or not list(default_dir.glob("*.py"))


def test_monolith_layout_patched_script_stays_in_default_dir(
    tmp_path, monkeypatch
):
    import unsloth_zoo.llama_cpp as llama_cpp

    monkeypatch.setitem(sys.modules, "gguf", types.ModuleType("gguf"))
    root = tmp_path / "old_llama_cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_bytes(_MONOLITH)
    default_dir = tmp_path / "default_dir"
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(default_dir))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(root))
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        result = llama_cpp._download_convert_hf_to_gguf("edge_monolith_placement")
    finally:
        llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()

    patched_path = result[0] if isinstance(result, tuple) else result
    # Backward-compat guard: monolith placement stays in the default dir.
    assert Path(patched_path).parent == default_dir
    assert (root / "convert_hf_to_gguf.py").read_bytes() == _MONOLITH


# --- Group 14: path-shape portability ---


def test_save_config_accepts_str_and_path_targets(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    calls = _capture_save_config(monkeypatch, "mlx_lm.utils")
    mutils._save_mlx_config({"a": 1}, str(tmp_path / "config.json"), is_vlm=False)
    assert calls["path"] == tmp_path / "config.json"


def test_prepare_export_accepts_string_path(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    monkeypatch.setattr(
        mutils,
        "_build_mlx_vlm_sanitize_pipelines",
        lambda config, model=None: [],
    )
    out = _export_dir_with_shards(tmp_path, config={"model_type": "t"}, shards={})
    assert mutils._prepare_vlm_gguf_export_directory(str(out)) == 0


def test_repair_with_backslash_path_string_returns_processor_unchanged():
    import unsloth_zoo.mlx.loader as loader

    processor = types.SimpleNamespace(image_processor=None)
    # A Windows-style string is a missing path on POSIX; return unchanged.
    repaired = loader._repair_degraded_vlm_processor(
        processor, "C:\\models\\fake", "fake_type"
    )
    assert repaired is processor


# --- Group 15: review follow-ups ---


def test_push_to_hub_gguf_positional_token_stays_token(monkeypatch, tmp_path):
    """Pre-existing positional callers must keep binding token as token."""
    import unsloth_zoo.mlx.utils as mutils

    calls = {}

    class FakeHfApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, repo_id, exist_ok=True, private=None):
            calls["repo_id"] = repo_id

        def update_repo_settings(self, **kwargs):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
            calls.setdefault("uploads", []).append(path_in_repo)

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
        calls["first_conversion"] = first_conversion
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "model.F16.gguf").write_bytes(b"GGUF")

    monkeypatch.setattr(mutils, "save_pretrained_gguf", fake_save_pretrained_gguf)

    mutils.push_to_hub_gguf(
        types.SimpleNamespace(),
        object(),
        tmp_path / "out",
        "org/repo",
        "not_quantized",
        "hf_secret_token",
    )
    assert calls["token"] == "hf_secret_token"
    assert calls["first_conversion"] is None
    assert calls["uploads"] == ["model.F16.gguf"]


def test_resolve_processor_class_follows_model_remapping(monkeypatch):
    import unsloth_zoo.mlx.loader as loader

    class RemappedProcessor:
        pass

    fake_vlm_utils = types.ModuleType("mlx_vlm.utils")
    fake_vlm_utils.MODEL_REMAPPING = {"alias_type": "real_type"}
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_vlm_utils)
    fake_processing = types.ModuleType("mlx_vlm.models.real_type.processing")
    fake_processing.RemappedProcessor = RemappedProcessor
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.real_type.processing", fake_processing
    )
    # The alias package does not exist in real mlx-vlm; the permissive shim
    # would auto-create it, so block those imports to mirror production.
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.alias_type.processing", None)
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.alias_type.processing_alias_type", None
    )

    resolved = loader._resolve_mlx_vlm_processor_class(
        "alias-type", "RemappedProcessor"
    )
    assert resolved is RemappedProcessor


def test_sanitize_pipelines_built_from_submodule_only_sanitizers():
    import unsloth_zoo.mlx.utils as mutils

    class VisionTower:
        def sanitize(self, weights):
            return weights

    tower = VisionTower()
    wrapper = types.SimpleNamespace(vision_tower=tower)
    pipelines = mutils._get_mlx_vlm_model_sanitize_pipelines(wrapper)
    assert pipelines == [[(tower, None)]]


def test_sanitize_pipelines_keep_wrapper_first_when_present():
    import unsloth_zoo.mlx.utils as mutils

    class Sanitizing:
        def sanitize(self, weights):
            return weights

    wrapper = Sanitizing()
    tower = Sanitizing()
    wrapper.vision_tower = tower
    pipelines = mutils._get_mlx_vlm_model_sanitize_pipelines(wrapper)
    assert pipelines == [
        [(wrapper, None)],
        [(wrapper, None), (tower, None)],
    ]


def test_image_processor_rebuild_uses_mlx_vlm_module_class(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.loader as loader

    class EdgeImageProcessor:
        def __init__(self, size=None):
            self.size = size

    fake_processing = types.ModuleType("mlx_vlm.models.edge_type.processing")
    fake_processing.EdgeImageProcessor = EdgeImageProcessor
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.edge_type.processing", fake_processing
    )

    built = loader._build_vlm_image_processor_from_config(
        tmp_path,
        {},
        {"image_processor_type": "EdgeImageProcessor", "size": 224},
        "edge_type",
    )
    assert isinstance(built, EdgeImageProcessor)
    assert built.size == 224
