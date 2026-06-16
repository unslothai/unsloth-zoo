# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# Coverage for the DiffusionGemma visual-server subprocess env builder
# (visual_engine._build_subprocess_env): the bundled CUDA runtime must reach the
# child's library path so it loads the CUDA backend instead of falling back to CPU
# (Windows unsloth#6273, Linux/WSL #6303), and torch/lib must not shadow the C++
# binary's system libs on Linux.

from __future__ import annotations

import os

from unsloth_zoo.diffusion_studio import visual_engine as V


def test_linux_prepends_nvidia_wheels_and_excludes_torch_lib(monkeypatch):
    monkeypatch.setattr(V, "_bundled_cuda_lib_dirs", lambda: ["/wheel/nvidia/cu13/lib"])
    monkeypatch.setattr(V, "_torch_cuda_lib_dir", lambda: "/py/torch/lib")
    env = V._build_subprocess_env(
        "/opt/dg/llama-diffusion-gemma-visual-server",
        gpu = "3", maxtok = 4096,
        base_env = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/sys/cuda"},
        os_name = "posix",
    )
    ld = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert ld[0] == "/opt/dg"                      # binary dir first
    assert "/wheel/nvidia/cu13/lib" in ld          # bundled CUDA runtime present
    assert ld[-1] == "/sys/cuda"                    # inherited path kept, last
    assert all("torch" not in p for p in ld)       # torch/lib excluded on Linux
    assert env["PATH"] == "/usr/bin"               # PATH untouched on Linux
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["NGL"] == "99"
    assert env["MAXTOK"] == "4096"


# Note: paths here are colon-free so the assertions hold regardless of the host's
# os.pathsep (on a Linux CI host os.pathsep is ":", which would split a "C:\..." path).
def test_windows_prepends_torch_lib_to_path_only(monkeypatch):
    monkeypatch.setattr(V, "_bundled_cuda_lib_dirs", lambda: ["/wheel/nvidia/cu13/lib"])
    monkeypatch.setattr(V, "_torch_cuda_lib_dir", lambda: "/py/torch/lib")
    env = V._build_subprocess_env(
        "/dg/llama-diffusion-gemma-visual-server.exe",
        gpu = "0", maxtok = 0,
        base_env = {"PATH": "/winpath"},
        os_name = "nt",
    )
    assert env["PATH"].startswith("/py/torch/lib" + os.pathsep)         # torch/lib first on PATH
    assert "/wheel/nvidia/cu13/lib" not in env.get("LD_LIBRARY_PATH", "")  # not forced on Windows
    assert env["MAXTOK"] == "0"


def test_windows_without_torch_cudart_leaves_path_untouched(monkeypatch):
    monkeypatch.setattr(V, "_torch_cuda_lib_dir", lambda: None)
    env = V._build_subprocess_env(
        "/dg/srv.exe", base_env = {"PATH": "/winpath"}, os_name = "nt",
    )
    assert env["PATH"] == "/winpath"


def test_bundled_cuda_lib_dirs_finds_nvidia_next_to_install(tmp_path, monkeypatch):
    fake_file = tmp_path / "unsloth_zoo" / "diffusion_studio" / "visual_engine.py"
    fake_file.parent.mkdir(parents = True)
    fake_file.write_text("")
    for pkg in ("cu13", "cudnn"):
        (tmp_path / "nvidia" / pkg / "lib").mkdir(parents = True)
    monkeypatch.setattr(V, "__file__", str(fake_file))
    dirs = V._bundled_cuda_lib_dirs()
    assert str(tmp_path / "nvidia" / "cu13" / "lib") in dirs
    assert str(tmp_path / "nvidia" / "cudnn" / "lib") in dirs
    assert all("torch" not in d for d in dirs)
