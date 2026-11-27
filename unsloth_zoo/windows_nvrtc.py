# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Windows: let NVRTC find the nvrtc-builtins DLL that ships in torch\\lib.

NVRTC loads ``nvrtc-builtins64_<ver>.dll`` by name with its own LoadLibrary call, which searches PATH but not
the directories torch registers through ``os.add_dll_directory``. On a torch build where that name is not
otherwise reachable, every NVRTC / jiterator kernel fails with "failed to open nvrtc-builtins64_<ver>.dll"
even though the DLL is in torch\\lib. Prepending torch\\lib to PATH before the first such kernel fixes it.
Runs before torch is imported and never imports it."""

from __future__ import annotations

import glob
import os
import sys
from typing import Callable, Optional

__all__ = ["expose_torch_lib_to_nvrtc"]


def _torch_lib_dir() -> Optional[str]:
    """torch\\lib of the installed torch, found without importing torch."""
    try:
        from importlib.util import find_spec

        spec = find_spec("torch")
    except (ImportError, ValueError):
        return None
    for root in (spec.submodule_search_locations or ()) if spec else ():
        lib = os.path.join(root, "lib")
        if os.path.isdir(lib):
            return lib
    return None


def _loads_by_name(name: str, environ = os.environ) -> bool:
    """Whether a bare-name LoadLibrary, as NVRTC's own call does, finds ``name`` in the executable's folder, the
    system folders or PATH. Looked up on disk, never loaded: a load probe also searches the current folder, so an
    import from a folder another account can write would run whatever DLL of that name was left there."""
    dirs = [os.path.dirname(exe) for exe in (sys.executable, getattr(sys, "_base_executable", None)) if exe]
    root = environ.get("SystemRoot") or environ.get("windir")
    if root:
        dirs += [os.path.join(root, "System32"), os.path.join(root, "System"), root]
    dirs += [entry for entry in environ.get("PATH", "").split(os.pathsep) if entry]
    for folder in dirs:
        try:
            if os.path.isfile(os.path.join(os.path.expandvars(folder.strip('"')), name)):
                return True
        except (OSError, ValueError, TypeError):
            continue
    return False


def expose_torch_lib_to_nvrtc(
    platform: str = sys.platform,
    torch_lib_dir: Callable[[], Optional[str]] = _torch_lib_dir,
    loads_by_name: Callable[[str], bool] = _loads_by_name,
    environ = os.environ,
) -> bool:
    """Prepend torch\\lib to PATH when it ships an nvrtc-builtins DLL that a bare-name load cannot find.
    Returns True when PATH changed. Never raises: a failed check leaves PATH alone."""
    if platform != "win32":
        return False
    try:
        lib = torch_lib_dir()
        if not lib:
            return False
        builtins = sorted(glob.glob(os.path.join(lib, "nvrtc-builtins64_*.dll")))
        if not builtins:
            return False
        path = environ.get("PATH", "")
        norm = os.path.normcase(os.path.normpath(lib))
        if any(
            os.path.normcase(os.path.normpath(entry)) == norm
            for entry in path.split(os.pathsep)
            if entry
        ):
            return False
        if all(loads_by_name(os.path.basename(dll)) for dll in builtins):
            return False
        environ["PATH"] = lib + (os.pathsep + path if path else "")
        return True
    except Exception:
        return False
