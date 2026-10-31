# SPDX-License-Identifier: AGPL-3.0-only

"""Mega-cache saves twice in one process against torch's real CacheArtifactManager (CPU only, no compile)."""

import importlib.util
import json
from pathlib import Path

import pytest

_cache = pytest.importorskip("torch.compiler._cache")
Manager = _cache.CacheArtifactManager

_COMPILE_CACHE_PATH = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "compile_cache.py"


@pytest.fixture
def cache(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("zoo_compile_cache_repeated_save", _COMPILE_CACHE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module._mega_cache_supported():
        pytest.skip("mega-cache needs torch >= 2.7")
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE_DIR", str(tmp_path / "mega"))
    monkeypatch.setattr(module, "_environment_fingerprint", lambda: {"test": "repeated-save"})
    module._STATE.update(armed = True, loaded_key = "repeated-save", model_keys = [])
    with Manager.with_fresh_cache():
        yield module, tmp_path / "mega" / "repeated-save"


def _published_keys(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    artifacts = Manager.deserialize((directory / manifest["bundle"]).read_bytes())
    return sorted(artifact.key for artifact in artifacts["autotune"])


def _record(key):
    Manager.record_artifact("autotune", key, {"timing": 1.0})


def test_second_save_in_one_process_keeps_the_first_saves_artifacts(cache):
    module, directory = cache
    _record("a")
    _record("b")
    assert module.megacache_save()
    assert _published_keys(directory) == ["a", "b"]
    _record("c")
    assert module.megacache_save()
    assert _published_keys(directory) == ["a", "b", "c"]


def test_save_after_an_outside_serialize_keeps_everything(cache):
    module, directory = cache
    _record("a")
    Manager.serialize()
    _record("b")
    assert module.megacache_save()
    assert _published_keys(directory) == ["a", "b"]
