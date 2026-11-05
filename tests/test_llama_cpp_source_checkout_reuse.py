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

"""install_llama_cpp(just_clone_repo=True) on an intact, unbuilt source checkout (LoRA GGUF export) must reuse it:
no build-tool / package-manager probes, no pip unless a converter dep is missing, and no refusal under UNSLOTH_AUTO_INSTALL=0."""

from __future__ import annotations

import importlib
import os

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")


def _make_source_checkout(root, with_lora = True):
    for d in ("src", "ggml", "common", os.path.join("gguf-py", "gguf")):
        os.makedirs(os.path.join(root, d), exist_ok = True)
    names = ["convert_hf_to_gguf.py"] + (["convert_lora_to_gguf.py"] if with_lora else [])
    for name in names:
        with open(os.path.join(root, name), "w") as f: f.write("# stub\n")
    return str(root)


class _Calls(list):
    pass


@pytest.fixture
def probes(monkeypatch, tmp_path):
    calls = _Calls()
    monkeypatch.chdir(tmp_path)  # no ./llama.cpp in CWD
    monkeypatch.delenv("UNSLOTH_AUTO_INSTALL", raising = False)

    def rec(kind, ret = None):
        def f(*a, **k):
            calls.append((kind, a[0] if a else None))
            return ret
        return f

    monkeypatch.setattr(llama_cpp, "check_build_requirements", rec("check_build_requirements", ([], "debian")))
    monkeypatch.setattr(llama_cpp, "do_we_need_sudo", rec("do_we_need_sudo", False))
    monkeypatch.setattr(llama_cpp, "install_package", rec("install_package"))
    monkeypatch.setattr(llama_cpp, "check_pip", rec("check_pip", "pip"))
    monkeypatch.setattr(llama_cpp, "try_execute", rec("try_execute"))
    monkeypatch.setattr(llama_cpp, "try_execute_with_auto_install", rec("try_execute_with_auto_install"))
    monkeypatch.setattr(llama_cpp, "_maybe_install_llama_cpp_prebuilt", rec("prebuilt"))
    # Converter deps all present, independent of this venv; a test can mark one missing.
    missing = set()
    calls.missing = missing
    calls.queried = queried = []
    def find_spec(name, *a):
        queried.append(name)
        return None if name in missing else object()
    monkeypatch.setattr(llama_cpp.importlib.util, "find_spec", find_spec)
    return calls


def _kinds(calls):
    return [k for k, _ in calls]


@pytest.mark.parametrize("is_windows", [False, True])
def test_intact_unbuilt_checkout_runs_no_probes(probes, tmp_path, monkeypatch, is_windows):
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", is_windows)
    folder = _make_source_checkout(tmp_path / "llama.cpp-source")
    for _ in range(3):
        assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    # Converter deps present, so nothing at all runs.
    assert probes == []
    # Import names (protobuf imports as google.protobuf), not pip names.
    assert set(probes.queried) == {"gguf", "google.protobuf", "sentencepiece", "mistral_common"}


def test_missing_python_dep_still_pip_installs(probes, tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    folder = _make_source_checkout(tmp_path / "llama.cpp-source")
    probes.missing.add("mistral_common")
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    assert _kinds(probes) == ["check_pip", "try_execute"]
    assert "install gguf protobuf sentencepiece mistral_common" in probes[1][1]

    probes.clear()
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    assert probes == []


def test_auto_install_off_reuses_checkout(probes, tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    folder = _make_source_checkout(tmp_path / "llama.cpp-source")
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    assert probes == []


def test_checkout_without_lora_converter_keeps_old_path(probes, tmp_path):
    folder = _make_source_checkout(tmp_path / "llama.cpp-source", with_lora = False)
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    assert _kinds(probes) == ["check_build_requirements", "check_pip", "try_execute"]


def test_full_build_unchanged(probes, tmp_path):
    folder = _make_source_checkout(tmp_path / "llama.cpp-source")
    # just_clone_repo False still goes to the prebuilt / build path.
    with pytest.raises(Exception):
        llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = False)
    assert _kinds(probes)[:2] == ["prebuilt", "check_build_requirements"]


def test_corrupted_checkout_still_repaired(probes, tmp_path, monkeypatch):
    folder = str(tmp_path / "llama.cpp-broken")
    os.makedirs(folder)
    with open(os.path.join(folder, "convert_lora_to_gguf.py"), "w") as f: f.write("# stub\n")
    monkeypatch.setattr(llama_cpp, "_is_safe_to_delete", lambda p: True)
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    assert not os.path.exists(folder)  # wiped, then (mocked) re-cloned
    assert _kinds(probes) == ["check_build_requirements", "try_execute_with_auto_install", "check_pip", "try_execute"]
    assert "git clone" in probes[1][1]
