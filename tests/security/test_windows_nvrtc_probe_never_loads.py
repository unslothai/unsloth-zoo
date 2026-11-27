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

"""The Windows nvrtc PATH check must not load a DLL by name: LoadLibrary's standard search includes the current
folder, so a planted nvrtc-builtins DLL there would run on `import unsloth_zoo`. Simulated on any OS."""

import ctypes
import os

from unsloth_zoo import windows_nvrtc

NAME = "nvrtc-builtins64_128.dll"


def _fake_torch_lib(tmp_path):
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents = True)
    (lib / NAME).write_bytes(b"MZ")
    return str(lib)


def test_probe_never_loads_a_dll(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, *a, **k: loaded.append(name), raising = False)
    cwd = tmp_path / "shared_cwd"
    cwd.mkdir()
    (cwd / NAME).write_bytes(b"MZ planted")
    monkeypatch.chdir(cwd)
    lib = _fake_torch_lib(tmp_path)
    environ = {"PATH": str(tmp_path / "empty")}
    monkeypatch.setattr(os, "environ", {**os.environ, **environ})
    windows_nvrtc.expose_torch_lib_to_nvrtc(platform = "win32", torch_lib_dir = lambda: lib, environ = environ)
    assert loaded == [], f"a DLL was loaded by bare name during the check: {loaded}"


def test_on_disk_lookup_matches_the_search_order(tmp_path, monkeypatch):
    on_path = tmp_path / "on_path"
    on_path.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / NAME).write_bytes(b"MZ")
    monkeypatch.chdir(cwd)
    env = {"PATH": str(tmp_path / "nothing") + os.pathsep + str(on_path)}
    # only in the current folder: not counted, so torch\lib is put on PATH as when nothing finds it
    assert windows_nvrtc._loads_by_name(NAME, environ = env) is False
    (on_path / NAME).write_bytes(b"MZ")
    assert windows_nvrtc._loads_by_name(NAME, environ = env) is True
    sysroot = tmp_path / "Windows"
    (sysroot / "System32").mkdir(parents = True)
    (sysroot / "System32" / NAME).write_bytes(b"MZ")
    assert windows_nvrtc._loads_by_name(NAME, environ = {"PATH": "", "SystemRoot": str(sysroot)}) is True


def test_unfound_builtins_still_put_torch_lib_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lib = _fake_torch_lib(tmp_path)
    environ = {"PATH": str(tmp_path / "x")}
    monkeypatch.setattr(os, "environ", {**os.environ, **environ, "PATH": environ["PATH"]})
    assert windows_nvrtc.expose_torch_lib_to_nvrtc(platform = "win32", torch_lib_dir = lambda: lib, environ = environ)
    assert environ["PATH"].split(os.pathsep)[0] == lib
