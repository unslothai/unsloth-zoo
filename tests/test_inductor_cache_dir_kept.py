# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""patch_torch_compile keeps a user / Studio TORCHINDUCTOR_CACHE_DIR, drops only the compiled cache folder. CPU only."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")


def _run_patch_torch_compile():
    from unsloth_zoo.patching_utils import patch_torch_compile
    patch_torch_compile(debug = False)


def test_user_inductor_cache_dir_is_kept(monkeypatch, tmp_path):
    chosen = str(tmp_path / "persistent_inductor")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", chosen)
    _run_patch_torch_compile()
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == chosen


def test_inductor_resolves_the_kept_directory(monkeypatch, tmp_path):
    chosen = str(tmp_path / "persistent_inductor")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", chosen)
    _run_patch_torch_compile()
    from torch._inductor.runtime.cache_dir_utils import cache_dir
    assert os.path.realpath(cache_dir()) == os.path.realpath(chosen)


def test_compiled_cache_folder_is_still_dropped(monkeypatch):
    from unsloth_zoo import patching_utils
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", patching_utils.UNSLOTH_COMPILE_LOCATION)
    _run_patch_torch_compile()
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") != patching_utils.UNSLOTH_COMPILE_LOCATION


def test_compiled_cache_folder_from_env_is_dropped(monkeypatch, tmp_path):
    folder = str(tmp_path / "ucc")
    os.makedirs(folder)
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", folder)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", folder + os.sep + "." + os.sep)
    _run_patch_torch_compile()
    assert os.path.realpath(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")) != os.path.realpath(folder)


def test_compiled_cache_folder_differing_in_case_is_dropped(monkeypatch, tmp_path):
    # Windows: case-insensitive, and realpath keeps a missing folder's spelling.
    folder = str(tmp_path / "UCC_missing")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", folder)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", folder.lower())
    monkeypatch.setattr(os.path, "normcase", lambda p: p.lower())
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


def test_empty_value_is_dropped(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "")
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


def test_unset_stays_unset(monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising = False)
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
