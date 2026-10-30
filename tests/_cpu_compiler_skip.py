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

"""Run a test's child script and skip only when a missing C++ compiler ended it.

CPU Inductor needs a C++ compiler (cl on Windows, g++ / clang++ elsewhere); AMD Windows boxes often have none. The
child runs its script through runpy (the script itself is untouched, so `from __future__` imports still work) under an
excepthook that judges only the exception that ends the run, following its __cause__ chain and the inner_exception
Dynamo / Inductor wrap the compiler error in. It writes one ASCII marker to a file only it knows, so nothing the script prints can fake or
outlive it. A missing compiler is InvalidCxxCompiler (Linux, torch 2.13+) or torch 2.9 - 2.12's Windows
RuntimeError("Compiler: <cl> is not found.").
"""

import os
import subprocess
import sys

import pytest

MARKER_ENV = "UNSLOTH_TEST_COMPILER_SKIP_MARKER"

_BOOT = r'''
import os, runpy, sys

def _missing_compiler(e):
    if type(e).__name__ == "InvalidCxxCompiler":
        return True
    message = str(e).strip()
    return isinstance(e, RuntimeError) and message.startswith("Compiler:") and message.endswith("is not found.")

def _hook(tp, exc, tb, _previous = sys.excepthook):
    seen, todo = set(), [exc]
    while todo:
        e = todo.pop(0)
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if _missing_compiler(e):
            text = (type(e).__name__ + ": " + str(e).strip().splitlines()[0])[:300]
            with open(os.environ["''' + MARKER_ENV + r'''"], "w", encoding = "ascii", errors = "backslashreplace") as f:
                f.write(type(e).__name__ + "\n" + text)
            break
        todo += [e.__cause__, getattr(e, "inner_exception", None)]
    _previous(tp, exc, tb)

sys.excepthook = _hook
path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(path, run_name = "__main__")
'''


def missing_compiler(type_name, message):
    """The parent's re-check of the marker (same rule as the child's hook)."""
    if type_name == "InvalidCxxCompiler":
        return True
    message = message.split(": ", 1)[-1].strip()
    return type_name == "RuntimeError" and message.startswith("Compiler:") and message.endswith("is not found.")


def run_child(path, args, env, timeout = 1500):
    """Run `path` with `args`; pytest.skip when a missing C++ compiler ended it, else return the CompletedProcess."""
    marker = f"{path}.compiler_skip"
    if os.path.exists(marker):
        os.remove(marker)
    env = dict(env, **{MARKER_ENV: marker})
    proc = subprocess.run([sys.executable, "-c", _BOOT, str(path), *args], capture_output = True, text = True,
                          env = env, timeout = timeout)
    if proc.returncode != 0 and os.path.exists(marker):
        with open(marker, encoding = "ascii") as f:
            type_name, _, text = f.read().partition("\n")
        if missing_compiler(type_name, text):
            pytest.skip(f"CPU Inductor has no C++ compiler here ({text})")
    return proc
