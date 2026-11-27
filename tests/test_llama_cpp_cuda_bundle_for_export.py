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

"""A CPU-oriented llama.cpp install (gpu_support=False, the GGUF export path) on a Linux CUDA
host installs the fork's CUDA bundle when its CUDA backend sees a GPU, so llama-cli / llama-server
-ngl offload; otherwise the CPU bundle, as before. Every download and GPU probe is faked."""

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason = "shell-script fake binaries")

TAG = "b9585-mix-abc1234"
CUDA_OLDER = f"app-{TAG}-linux-x64-cuda12-older.tar.gz"
CUDA_PORTABLE = f"app-{TAG}-linux-x64-cuda12-portable.tar.gz"
CPU = f"app-{TAG}-linux-x64-cpu.tar.gz"
SOURCE = f"llama.cpp-source-{TAG}.tar.gz"
MANIFEST = {"artifacts": [
    {"asset_name": CUDA_OLDER, "install_kind": "linux-cuda", "runtime_line": "cuda12",
     "coverage_class": "older", "supported_sms": ["70", "75", "80", "86", "89"], "min_sm": 70, "max_sm": 89, "rank": 10},
    {"asset_name": CUDA_PORTABLE, "install_kind": "linux-cuda", "runtime_line": "cuda12",
     "coverage_class": "portable", "supported_sms": ["70", "75", "80", "86", "89", "90", "100", "120"],
     "min_sm": 70, "max_sm": 120, "rank": 30},
    {"asset_name": f"app-{TAG}-linux-x64-rocm-gfx110X.tar.gz", "install_kind": "linux-rocm"},
    {"asset_name": CPU, "install_kind": "linux-cpu"},
    {"asset_name": f"app-{TAG}-windows-x64-cpu.zip", "install_kind": "windows-cpu"},
    {"asset_name": f"app-{TAG}-windows-x64-cuda12-older.zip", "install_kind": "windows-cuda", "runtime_line": "cuda12",
     "coverage_class": "older", "supported_sms": ["70", "75", "80", "86", "89"], "min_sm": 70, "max_sm": 89, "rank": 10},
]}
ASSETS = {a["asset_name"]: f"https://example.invalid/{a['asset_name']}" for a in MANIFEST["artifacts"]}
ASSETS[SOURCE] = f"https://example.invalid/{SOURCE}"

QUANTIZE = "#!/bin/sh\necho 'usage: llama-quantize [--help] model-f32.gguf'\nexit 0\n"
NO_DEVICE = "#!/bin/sh\necho 'Available devices:'\necho '  (none)'\nexit 0\n"
T4 = "#!/bin/sh\necho 'Available devices:'\necho '  CUDA0: Tesla T4 (15095 MiB, 14900 MiB free)'\nexit 0\n"
# Sees the GPU only once libcudart.so.12 sits next to it, like libggml-cuda.so's RUNPATH $ORIGIN.
T4_IF_RUNTIME = (
    "#!/bin/sh\necho 'Available devices:'\n"
    'if [ -e "$(dirname "$0")/libcudart.so.12" ]; then echo "  CUDA0: Tesla T4 (15095 MiB, 14900 MiB free)";'
    ' else echo "  (none)"; fi\nexit 0\n'
)


def _load():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("llama_cpp_cuda_bundle_under_test", root / "unsloth_zoo" / "llama_cpp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod(monkeypatch):
    for name in ("UNSLOTH_LLAMA_FORCE_COMPILE", "UNSLOTH_LLAMA_TAG", "UNSLOTH_LLAMA_CPP_OFFLINE", "UNSLOTH_OFFLINE"):
        monkeypatch.delenv(name, raising = False)
    module = _load()
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module, "IS_WINDOWS", False)
    monkeypatch.setattr(module, "try_execute", lambda *a, **k: "")
    monkeypatch.setattr(module, "check_pip", lambda: "pip")
    # No host CUDA runtime leaks in unless a test provides one.
    monkeypatch.setattr(module, "_cuda_runtime_candidate_dirs", lambda: [], raising = False)
    return module


def _add(tar, name, data, mode = 0o644):
    raw = data.encode()
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    info.mode = mode
    tar.addfile(info, io.BytesIO(raw))


def _wire(mod, monkeypatch, tmp_path, cli_script):
    """Serve fake fork assets: CUDA bundles whose llama-cli prints `cli_script`, the CPU bundle, the source."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for asset in ASSETS:
        if asset == SOURCE:
            with tarfile.open(fixtures / asset, "w:gz") as tar:
                for name in ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "conversion/__init__.py",
                             "gguf-py/gguf/tensor_mapping.py"):
                    _add(tar, f"llama.cpp-{TAG}/{name}", "# stub\n")
            continue
        if not asset.endswith(".tar.gz"):
            continue
        with tarfile.open(fixtures / asset, "w:gz") as tar:
            _add(tar, "llama-quantize", QUANTIZE, 0o755)
            _add(tar, "llama-cli", cli_script if "-cuda" in asset else NO_DEVICE, 0o755)
            _add(tar, "libggml-base.so", "not elf")
            if "-cuda" in asset:
                _add(tar, "libggml-cuda.so", "not elf")
    checksums = {"artifacts": {name: {"sha256": mod._sha256_file(str(fixtures / name))}
                               for name in ASSETS if (fixtures / name).exists()}}
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release",
                        lambda releases_api = None: (TAG, ASSETS) if releases_api == mod.LLAMA_CPP_PUBLISHED_RELEASES_API else None)
    monkeypatch.setattr(mod, "_fetch_release_json_asset",
                        lambda assets, name: MANIFEST if "manifest" in name else checksums)
    downloaded = []
    def fake_download(url, dest_path):
        name = url.rsplit("/", 1)[1]
        downloaded.append(name)
        with open(fixtures / name, "rb") as fr, open(dest_path, "wb") as fw:
            fw.write(fr.read())
    monkeypatch.setattr(mod, "_download_archive", fake_download)
    return downloaded


def _installed_asset(folder):
    return json.load(open(os.path.join(folder, "UNSLOTH_PREBUILT_INFO.json")))["asset"]


def test_export_install_on_a_cuda_host_gets_the_cuda_bundle(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: ("cuda", 75, "cuda12"))
    downloaded = _wire(mod, monkeypatch, tmp_path, T4)
    folder = str(tmp_path / "llama.cpp")
    quantizer, _ = mod._install_llama_cpp_prebuilt(folder, gpu_support = False)
    assert _installed_asset(folder) == CUDA_OLDER
    assert quantizer == os.path.join(folder, "llama-quantize")
    assert os.path.isfile(os.path.join(folder, "libggml-cuda.so"))
    assert CPU not in downloaded


def test_cuda_bundle_without_a_visible_device_falls_back_to_the_cpu_bundle(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: ("cuda", 75, "cuda12"))
    downloaded = _wire(mod, monkeypatch, tmp_path, NO_DEVICE)
    folder = str(tmp_path / "llama.cpp")
    assert mod._install_llama_cpp_prebuilt(folder, gpu_support = False) is not None
    assert _installed_asset(folder) == CPU
    # Both CUDA candidates were tried first, the CPU bundle last; nothing CUDA was activated.
    assert [d for d in downloaded if d != SOURCE] == [CUDA_OLDER, CUDA_PORTABLE, CPU]
    assert not os.path.exists(os.path.join(folder, "libggml-cuda.so"))
    assert not [e for e in os.listdir(tmp_path) if e.startswith(".llama_cpp_prebuilt_")]


def test_missing_cuda_runtime_is_linked_in_from_the_torch_wheels(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: ("cuda", 75, "cuda12"))
    _wire(mod, monkeypatch, tmp_path, T4_IF_RUNTIME)
    wheel_lib = tmp_path / "site" / "nvidia" / "cu12" / "lib"
    wheel_lib.mkdir(parents = True)
    for lib in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        (wheel_lib / lib).write_text("runtime")
    monkeypatch.setattr(mod, "_cuda_runtime_candidate_dirs", lambda: [str(wheel_lib)], raising = False)
    folder = str(tmp_path / "llama.cpp")
    assert mod._install_llama_cpp_prebuilt(folder, gpu_support = False) is not None
    assert _installed_asset(folder) == CUDA_OLDER
    for lib in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        link = os.path.join(folder, lib)
        assert os.path.islink(link) and os.path.realpath(link) == str(wheel_lib / lib)


def test_existing_folder_merge_keeps_runtime_links_as_links(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: ("cuda", 75, "cuda12"))
    _wire(mod, monkeypatch, tmp_path, T4_IF_RUNTIME)
    wheel_lib = tmp_path / "wheel"
    wheel_lib.mkdir()
    for lib in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        (wheel_lib / lib).write_text("runtime")
    monkeypatch.setattr(mod, "_cuda_runtime_candidate_dirs", lambda: [str(wheel_lib)], raising = False)
    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    (folder / "user_file.txt").write_text("keep")
    assert mod._install_llama_cpp_prebuilt(str(folder), gpu_support = False) is not None
    assert (folder / "user_file.txt").read_text() == "keep"
    assert os.path.islink(folder / "libcublas.so.12")


@pytest.mark.parametrize("target", [None, ("rocm", "gfx1100")])
def test_non_cuda_linux_hosts_keep_the_cpu_bundle(mod, monkeypatch, tmp_path, target):
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: target)
    downloaded = _wire(mod, monkeypatch, tmp_path, T4)
    folder = str(tmp_path / "llama.cpp")
    assert mod._install_llama_cpp_prebuilt(folder, gpu_support = False) is not None
    assert _installed_asset(folder) == CPU
    assert [d for d in downloaded if d != SOURCE] == [CPU]


def test_windows_cuda_host_export_install_is_unchanged(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: ("cuda", 75, "cuda12"))
    staged = []
    def fake_stage(folder, tag, asset_name, asset_url, **kwargs):
        staged.append((asset_name, kwargs.get("require_cuda_device", False)))
        raise RuntimeError("stop")
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", lambda releases_api = None: (TAG, ASSETS))
    monkeypatch.setattr(mod, "_fetch_release_json_asset", lambda assets, name: MANIFEST if "manifest" in name else {})
    monkeypatch.setattr(mod, "_stage_prebuilt_install", fake_stage)
    assert mod._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = False) is None
    assert staged == [(f"app-{TAG}-windows-x64-cpu.zip", False)]


def test_explicit_gpu_request_still_keeps_its_cuda_bundle_without_a_device(mod, monkeypatch, tmp_path):
    # gpu_support=True is unchanged: no CPU substitute, the CUDA bundle stays (with a warning).
    monkeypatch.setattr(mod, "_detect_gpu_target", lambda: ("cuda", 75, "cuda12"))
    downloaded = _wire(mod, monkeypatch, tmp_path, NO_DEVICE)
    folder = str(tmp_path / "llama.cpp")
    assert mod._install_llama_cpp_prebuilt(folder, gpu_support = True) is not None
    assert _installed_asset(folder) == CUDA_OLDER
    assert CPU not in downloaded
