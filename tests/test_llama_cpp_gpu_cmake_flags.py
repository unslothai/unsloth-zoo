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

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_gpu_flags", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llama_cpp = _load_llama_cpp_module()


def _fake_torch(hip, gfx = "gfx1100:sramecc+:xnack-", cuda_available = True):
    gfxs = gfx if isinstance(gfx, list) else [gfx]
    cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        device_count = lambda: len(gfxs),
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = gfxs[i]),
        get_device_capability = lambda i: (9, 0),
    )
    return types.SimpleNamespace(version = types.SimpleNamespace(hip = hip, cuda = None if hip else "12.8"), cuda = cuda)


def _source_build(monkeypatch, tmp_path, torch_stub, gpu_support, windows = False):
    folder = tmp_path / "llama.cpp"
    for d in ("src", "ggml", "common"):
        (folder / d).mkdir(parents = True)
    calls = {"check": 0}

    def fake_check(llama_cpp_folder = None):
        calls["check"] += 1
        if calls["check"] == 1:
            raise RuntimeError("no binaries yet")
        return "llama-quantize", "convert_hf_to_gguf.py"

    commands = []
    monkeypatch.setattr(llama_cpp, "torch", torch_stub)
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", windows)
    monkeypatch.setattr(llama_cpp, "check_llama_cpp", fake_check)
    monkeypatch.setattr(llama_cpp, "_auto_install_enabled", lambda: True)
    monkeypatch.setattr(llama_cpp, "_maybe_install_llama_cpp_prebuilt", lambda *a, **k: None)
    monkeypatch.setattr(llama_cpp, "check_build_requirements", lambda: ([], "debian"))
    monkeypatch.setattr(llama_cpp, "do_we_need_sudo", lambda system_type = None: False)
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "pip")
    monkeypatch.setattr(llama_cpp, "_is_cmake_only_llama_cpp", lambda folder: True)
    monkeypatch.setattr(llama_cpp, "_find_lib_path", lambda name: None)
    monkeypatch.setattr(llama_cpp, "try_execute", lambda command, *a, **k: commands.append(command) or "")
    if windows:
        monkeypatch.setattr(llama_cpp, "_find_visual_studio", lambda: ("Visual Studio 17 2022", None))
        monkeypatch.setattr(llama_cpp, "_find_openssl_root", lambda: None)
        monkeypatch.setattr(
            llama_cpp.subprocess, "run",
            lambda argv, **k: commands.append(argv) or types.SimpleNamespace(returncode = 1, stdout = "", stderr = ""),
        )
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    try:
        llama_cpp.install_llama_cpp(llama_cpp_folder = str(folder), gpu_support = gpu_support)
    except RuntimeError:
        pass  # Windows stub fails configure after recording it
    configure = [c for c in commands if (c[0] if isinstance(c, list) else c.split()[0]) == "cmake"
                 and "--build" not in c]
    assert configure, commands
    return configure[0]


def test_rocm_source_build_uses_hip(monkeypatch, tmp_path):
    clang = tmp_path / "rocm" / "llvm" / "bin" / "clang"
    clang.parent.mkdir(parents = True)
    clang.write_text("")
    cmd = _source_build(monkeypatch, tmp_path, _fake_torch(hip = "7.1"), gpu_support = True)
    assert "-DGGML_HIP=ON" in cmd
    assert "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" in cmd
    assert "-DGPU_TARGETS=gfx1100" in cmd
    assert f"-DCMAKE_HIP_COMPILER={clang}" in cmd
    assert "GGML_CUDA" not in cmd


def test_rocm_mixed_arch_host_targets_every_card(monkeypatch, tmp_path):
    torch_stub = _fake_torch(hip = "7.1", gfx = ["gfx1100", "gfx1030:xnack-", "gfx1100"])
    cmd = _source_build(monkeypatch, tmp_path, torch_stub, gpu_support = True)
    assert "'-DGPU_TARGETS=gfx1100;gfx1030'" in cmd


def test_rocm_without_visible_device_still_builds_hip(monkeypatch, tmp_path):
    cmd = _source_build(monkeypatch, tmp_path, _fake_torch(hip = "7.1", cuda_available = False), gpu_support = True)
    assert "-DGGML_HIP=ON" in cmd
    assert "GPU_TARGETS" not in cmd and "CMAKE_HIP_COMPILER" not in cmd


@pytest.mark.parametrize("hip,gpu_support,expected", [
    (None, True, "-DGGML_CUDA=ON"),
    (None, False, "-DGGML_CUDA=OFF"),
    ("7.1", False, "-DGGML_CUDA=OFF"),
])
def test_non_rocm_or_cpu_builds_unchanged(monkeypatch, tmp_path, hip, gpu_support, expected):
    cmd = _source_build(monkeypatch, tmp_path, _fake_torch(hip = hip), gpu_support = gpu_support)
    assert cmd.endswith(f"-DBUILD_SHARED_LIBS=OFF {expected}")


def test_windows_keeps_single_cuda_argv_item(monkeypatch, tmp_path):
    cmd = _source_build(monkeypatch, tmp_path, _fake_torch(hip = "7.1"), gpu_support = True, windows = True)
    assert isinstance(cmd, list)
    assert "-DGGML_CUDA=ON" in cmd
    assert not any("HIP" in a for a in cmd)
