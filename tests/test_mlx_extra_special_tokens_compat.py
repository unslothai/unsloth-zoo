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


def test_materializes_only_missing_vlm_processor_geometry(tmp_path):
    import unsloth_zoo.mlx.loader as loader
    source = tmp_path / "snapshot"
    source.mkdir()
    original = {"processor_class": "LlavaProcessor", "patch_size": None}
    (source / "processor_config.json").write_text(json.dumps(original), encoding="utf-8")
    (source / "config.json").write_text(json.dumps({"model_type": "raw"}), encoding="utf-8")
    model = {"model_type": "llava", "vision_config": {"patch_size": 14}, "vision_feature_select_strategy": "default"}
    view, _ = loader._materialize_mlx_vlm_config_override(
        str(source), model, config_override_data=model, normalize_processor_geometry=True,
    )
    assert json.loads((Path(view) / "config.json").read_text()) == model
    patched = json.loads((Path(view) / "processor_config.json").read_text(encoding="utf-8"))
    assert patched == {**original, "patch_size": 14, "vision_feature_select_strategy": "default"}
    assert json.loads((source / "processor_config.json").read_text(encoding="utf-8")) == original
    second_view = tmp_path / "second_view"
    second_view.mkdir()
    loaded, _ = loader._load_mlx_vlm_with_config_views(
        lambda: (type("Model", (), {})(), object()), [view, second_view],
    )
    for finalizer in loaded._unsloth_mlx_config_view_finalizers:
        finalizer()
    assert not Path(view).exists() and not second_view.exists()
    failed_view = tmp_path / "failed_view"
    failed_view.mkdir()
    with pytest.raises(RuntimeError):
        loader._load_mlx_vlm_with_config_views(
            lambda: (_ for _ in ()).throw(RuntimeError("load failed")), [failed_view],
        )
    assert not failed_view.exists()

    existing = {"patch_size": 16, "vision_feature_select_strategy": "full"}
    assert loader._normalize_vlm_processor_geometry(existing, model) == (existing, False)


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


def test_mlx_vlm_bpe_incomplete_utf8_fallback(monkeypatch):
    pytest.importorskip("mlx.core")
    from unsloth_zoo.mlx.loader import _ensure_mlx_vlm_bpe_utf8_fallback, _mlx_vlm_bpe_needs_utf8_fallback
    tokenizer_utils = types.ModuleType("mlx_vlm.tokenizer_utils")
    tokenizer_utils._remove_space = lambda text: text[1:] if text.startswith(" ") else text
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.tokenizer_utils", tokenizer_utils)
    class StrictBPEStreamingDetokenizer:
        _byte_decoder = {"Ã": 0xC3, "¿": 0xBF, " ": 32}
        make_byte_decoder = classmethod(lambda cls: None)
        def reset(self):
            self.offset, self._unflushed, self.text, self.tokens = 0, "", "", []
        def add_token(self, token, skip_special_token_ids=[]):
            value = self.tokenmap[token]
            if token in (2, 3, 4):
                encoding = "synthetic-codec" if token == 2 else "utf-8"
                self.failure = UnicodeDecodeError(encoding, b"\xc3", 0, 1, "unexpected end of data")
                if token == 3:
                    self.text = "changed"
                if token == 4:
                    self._unflushed = "changed"
                raise self.failure
            if self._byte_decoder[value[0]] == 32:
                bytearray(self._byte_decoder[char] for char in self._unflushed).decode("utf-8")
                self._unflushed = value
            else:
                self._unflushed += value
    strict = StrictBPEStreamingDetokenizer
    tokenizer_utils.BPEStreamingDetokenizer = strict
    byte_chars = {byte: char for char, byte in strict._byte_decoder.items()}
    def make_detokenizer(tokenmap=None):
        detokenizer = object.__new__(strict)
        detokenizer.trim_space = False
        detokenizer.tokenmap = tokenmap or [byte_chars[0xC3], byte_chars[32] + "x", byte_chars[32], byte_chars[32], byte_chars[32]]
        detokenizer.reset()
        return detokenizer
    _ensure_mlx_vlm_bpe_utf8_fallback()
    detokenizer = make_detokenizer()
    detokenizer.add_token(0)
    detokenizer.add_token(1)
    assert (detokenizer.text, detokenizer._unflushed) == ("\ufffd", byte_chars[32] + "x")
    assert not _mlx_vlm_bpe_needs_utf8_fallback(strict)
    detokenizer = make_detokenizer({123: byte_chars[0xBF], 220: byte_chars[32], 2: byte_chars[32]})
    for token in (123, 220):
        detokenizer.add_token(token)
    assert (detokenizer.text, detokenizer._unflushed) == ("\ufffd", byte_chars[32])
    detokenizer = make_detokenizer()
    detokenizer.add_token(0)
    with pytest.raises(UnicodeDecodeError, match="synthetic-codec"):
        detokenizer.add_token(2)
    for token in (3, 4):
        detokenizer = make_detokenizer()
        detokenizer.add_token(0)
        with pytest.raises(UnicodeDecodeError) as raised:
            detokenizer.add_token(token)
        assert raised.value is detokenizer.failure
    def incomplete_safe_add_token(self, token, skip_special_token_ids=[]):
        value = self.tokenmap[token]
        if token == 0 or self._unflushed == byte_chars[0xC3]:
            self._unflushed = value
            return
        bytearray(self._byte_decoder[char] for char in self._unflushed).decode("utf-8")
    monkeypatch.setattr(strict, "add_token", incomplete_safe_add_token)
    assert _mlx_vlm_bpe_needs_utf8_fallback(strict)
