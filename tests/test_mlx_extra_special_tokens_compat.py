# Unsloth Zoo - Utilities for Unsloth
# Coerce list extra_special_tokens only when transformers would otherwise crash.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


@pytest.fixture
def base_init():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    saved = PreTrainedTokenizerBase.__init__
    yield PreTrainedTokenizerBase
    PreTrainedTokenizerBase.__init__ = saved


def test_list_preserved_when_init_supports_lists(base_init):
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens
    seen = {}
    base_init.__init__ = lambda self, **kw: seen.update(kw)
    _coerce_list_extra_special_tokens()
    base_init.__init__(object(), extra_special_tokens=["<a>", "<b>"])
    assert seen["extra_special_tokens"] == ["<a>", "<b>"]


def test_list_coerced_only_when_init_crashes(base_init):
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens
    seen = {}

    def crashing_init(self, **kw):
        if isinstance(kw.get("extra_special_tokens"), list):
            raise AttributeError("'list' object has no attribute 'keys'")
        seen.update(kw)

    base_init.__init__ = crashing_init
    _coerce_list_extra_special_tokens()
    base_init.__init__(object(), extra_special_tokens=["<a>", "<b>"])
    assert seen["extra_special_tokens"] == {}

    seen.clear()
    base_init.__init__(object(), extra_special_tokens={"image_token": "<img>"})
    assert seen["extra_special_tokens"] == {"image_token": "<img>"}


def test_unrelated_attributeerror_not_swallowed(base_init):
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens

    def boom_init(self, **kw):
        raise AttributeError("unrelated boom")

    base_init.__init__ = boom_init
    _coerce_list_extra_special_tokens()
    with pytest.raises(AttributeError, match="unrelated"):
        base_init.__init__(object(), extra_special_tokens=["<a>"])


def test_idempotent_and_guard_shared(base_init):
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens
    base_init.__init__ = lambda self, **kw: None
    _coerce_list_extra_special_tokens()
    wrapped = base_init.__init__
    assert getattr(wrapped, "_unsloth_extra_special_tokens_patched", False) is True
    _coerce_list_extra_special_tokens()
    assert base_init.__init__ is wrapped


def test_materialize_normalizes_list_extra_special_tokens_sidecar(
    tmp_path, monkeypatch
):
    import transformers.models.auto.tokenization_auto as tokenization_auto
    from unsloth_zoo.mlx.loader import (
        _materialize_mlx_vlm_config_override,
        _normalize_tokenizer_config_extra_special_tokens,
    )
    source_dir = tmp_path / "snapshot"
    source_dir.mkdir()
    original = {
        "tokenizer_class": "TokenizersBackend",
        "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "additional_special_tokens": ["<existing>", "<|im_start|>"],
        "model_specific_special_tokens": {"image_token": "<|image_pad|>"},
    }
    monkeypatch.setattr(
        tokenization_auto, "tokenizer_class_from_name", lambda _name: None
    )
    (source_dir / "tokenizer_config.json").write_text(
        json.dumps(original),
        encoding="utf-8",
    )
    config_data = {"model_type": "qwen3_vl"}
    assert _materialize_mlx_vlm_config_override(str(source_dir), config_data) == (
        str(source_dir),
        config_data,
    )
    load_path, returned_config = _materialize_mlx_vlm_config_override(
        str(source_dir),
        config_data,
        normalize_tokenizer_config=True,
        supports_list_extra_special_tokens=False,
    )
    assert load_path != str(source_dir)
    assert returned_config is config_data
    assert json.loads(
        (source_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    ) == original
    patched_config = json.loads(
        (Path(load_path) / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert patched_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert patched_config["extra_special_tokens"] == {"image_token": "<|image_pad|>"}
    assert patched_config["additional_special_tokens"] == [
        "<existing>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    assert _normalize_tokenizer_config_extra_special_tokens(
        original,
        supports_list_extra_special_tokens=True,
    ) == (
        original,
        False,
    )
    monkeypatch.setattr(
        tokenization_auto, "tokenizer_class_from_name", lambda _name: object
    )
    assert _materialize_mlx_vlm_config_override(
        str(source_dir),
        config_data,
        normalize_tokenizer_config=True,
        supports_list_extra_special_tokens=True,
    ) == (
        str(source_dir),
        config_data,
    )
    patched_without_model_specific, changed = (
        _normalize_tokenizer_config_extra_special_tokens(
            {"extra_special_tokens": ["<legacy>", "<legacy>"]},
            supports_list_extra_special_tokens=False,
        )
    )
    assert changed is True
    assert patched_without_model_specific["extra_special_tokens"] == {}
    assert patched_without_model_specific["additional_special_tokens"] == ["<legacy>"]
    valid_tokenizer_config = {
        "extra_special_tokens": {"image_token": "<|image_pad|>"}
    }
    assert _normalize_tokenizer_config_extra_special_tokens(
        valid_tokenizer_config
    ) == (
        valid_tokenizer_config,
        False,
    )


def test_materialize_probe_ignores_installed_coercion_patch(base_init, tmp_path):
    import unsloth_zoo.mlx.loader as loader

    def crashing_init(self, **kw):
        if isinstance(kw.get("extra_special_tokens"), list):
            raise AttributeError("'list' object has no attribute 'keys'")

    base_init.__init__ = crashing_init
    loader._coerce_list_extra_special_tokens()
    source_dir = tmp_path / "snapshot"
    source_dir.mkdir()
    (source_dir / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<legacy>"]}),
        encoding="utf-8",
    )

    load_path, _ = loader._materialize_mlx_vlm_config_override(
        str(source_dir),
        {"model_type": "qwen3_vl"},
        normalize_tokenizer_config=True,
    )

    assert load_path != str(source_dir)


def test_normalizes_tokenizers_backend_only_when_unavailable(monkeypatch):
    import transformers.models.auto.tokenization_auto as tokenization_auto
    from unsloth_zoo.mlx.loader import _normalize_tokenizer_config_backend_class

    config = {"tokenizer_class": "TokenizersBackend"}
    assert _normalize_tokenizer_config_backend_class(
        config, backend_class_available=False
    ) == ({"tokenizer_class": "PreTrainedTokenizerFast"}, True)
    assert _normalize_tokenizer_config_backend_class(
        config, backend_class_available=True
    ) == (config, False)
    unrelated = {"tokenizer_class": "CustomTokenizer"}
    assert _normalize_tokenizer_config_backend_class(
        unrelated, backend_class_available=False
    ) == (unrelated, False)

    def unavailable(_name):
        raise RuntimeError("resolver unavailable")

    monkeypatch.setattr(
        tokenization_auto,
        "tokenizer_class_from_name",
        unavailable,
    )
    assert _normalize_tokenizer_config_backend_class(config) == (config, False)


def test_temporary_patch_exposes_original_init_for_probe(
    base_init,
    monkeypatch,
):
    from unsloth_zoo.mlx.loader import _tokenizer_supports_list_extra_special_tokens

    def crashing_init(self, **kw):
        if isinstance(kw.get("extra_special_tokens"), list):
            raise AttributeError("'list' object has no attribute 'keys'")

    package_name = "_unsloth_misc_patch_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = []
    common = types.ModuleType(f"{package_name}.common")
    common.TEMPORARY_PATCHES = []
    common.torch_compile = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    common._torch_compile = common.torch_compile
    utils = types.ModuleType(f"{package_name}.utils")
    for name in (
        "patch_function",
        "process_output_options",
        "process_return",
        "raise_error",
    ):
        setattr(utils, name, lambda *args, **kwargs: None)
    for name in (
        "KWARGS_TYPE",
        "ImageInput",
        "PreTokenizedInput",
        "TextInput",
        "Cache",
        "StaticCache",
        "HybridCache",
        "Unpack",
    ):
        setattr(utils, name, object)
    utils._get_unique_storage_name = lambda *args, **kwargs: "unused"
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.common", common)
    monkeypatch.setitem(sys.modules, f"{package_name}.utils", utils)
    misc_path = (
        Path(__file__).resolve().parents[1]
        / "unsloth_zoo"
        / "temporary_patches"
        / "misc.py"
    )
    spec = importlib.util.spec_from_file_location(f"{package_name}.misc", misc_path)
    misc = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, f"{package_name}.misc", misc)
    spec.loader.exec_module(misc)

    base_init.__init__ = crashing_init
    misc.patch_tokenizer_extra_special_tokens()

    assert hasattr(base_init.__init__, "_unsloth_original_init")
    assert _tokenizer_supports_list_extra_special_tokens() is False
