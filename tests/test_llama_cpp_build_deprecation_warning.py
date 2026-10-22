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

"""A compiler `... is deprecated` warning must not fail a successful llama.cpp source build.

GCC 16 (Fedora 44) warns `-Wdeprecated-enum-enum-conversion` in tools/mtmd (unsloth#5832).
Runs a real cmake build of a stand-in checkout whose source triggers such a warning.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("cmake") and shutil.which("c++")),
    reason = "needs cmake and a C++ compiler on Linux/macOS",
)

_SOURCE = """
[[deprecated]] int old_api() { return 0; }
int main() { return old_api(); }
"""


def _fake_checkout(root):
    for d in ("src", "ggml", "common"):
        os.makedirs(os.path.join(root, d))
    with open(os.path.join(root, "Makefile"), "w") as f:
        f.write("$(error Build system changed: the Makefile build has been replaced by CMake)\n")
    with open(os.path.join(root, "main.cpp"), "w") as f:
        f.write(_SOURCE)
    lines = [
        "cmake_minimum_required(VERSION 3.14)",
        "project(fake CXX)",
        "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)",
    ]
    lines += [f"add_executable({t} main.cpp)" for t in llama_cpp.LLAMA_CPP_TARGETS]
    with open(os.path.join(root, "CMakeLists.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def test_deprecation_warning_does_not_fail_cmake_build(tmp_path, monkeypatch):
    folder = str(tmp_path / "llama.cpp")
    _fake_checkout(folder)

    def fake_check(llama_cpp_folder = folder):
        quantizer = os.path.join(llama_cpp_folder, "llama-quantize")
        if not os.path.exists(quantizer):
            raise RuntimeError("no quantizer")
        return quantizer, os.path.join(llama_cpp_folder, "convert_hf_to_gguf.py")

    monkeypatch.setattr(llama_cpp, "check_llama_cpp", fake_check)
    monkeypatch.setattr(llama_cpp, "_maybe_install_llama_cpp_prebuilt", lambda *a, **k: None)
    monkeypatch.setattr(llama_cpp, "check_build_requirements", lambda: ([], "debian"))
    monkeypatch.setattr(llama_cpp, "do_we_need_sudo", lambda *a, **k: False)
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "true")
    monkeypatch.setattr(llama_cpp, "_find_lib_path", lambda *a, **k: None)
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")

    quantizer, _ = llama_cpp.install_llama_cpp(llama_cpp_folder = folder, gpu_support = False)
    assert os.path.exists(quantizer)
