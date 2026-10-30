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

"""Skip a failed CPU subprocess check when CPU Inductor has no C++ compiler here.

CPU Inductor needs a C++ compiler (cl on Windows, g++ / clang++ elsewhere); AMD Windows boxes often have none. A
failed child run skips only when it did not run on CUDA, its own output names a missing compiler, and one trivial
compile in a fresh process with the same environment confirms it. Any other failure still fails.
"""

import re
import subprocess
import sys

import pytest

_PROBE = "import torch; torch.compile(lambda x: x + 1, backend = 'inductor')(torch.ones(2))"
# Linux / torch 2.13+: InvalidCxxCompiler. torch 2.9 - 2.12 on Windows: RuntimeError("Compiler: cl is not found.").
_MISSING = re.compile(r"InvalidCxxCompiler|Compiler: .* is not found\.")
# Everything that changes whether the probe compiles, or how it fails.
_KEY_VARS = ("CXX", "CC", "PATH", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_SUPPRESS_ERRORS")
_probes = {}


def missing_compiler(returncode, output):
    return returncode != 0 and _MISSING.search(output) is not None


def _probe(env):
    key = (sys.executable,) + tuple(sorted(
        (k, v) for k, v in env.items() if k.upper() in _KEY_VARS or k.upper().startswith("TORCHINDUCTOR_")
    ))
    if key not in _probes:
        try:
            proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output = True, text = True,
                                  errors = "replace", env = env, timeout = 600)
            _probes[key] = (proc.returncode, proc.stdout + proc.stderr)
        except subprocess.TimeoutExpired:
            _probes[key] = (0, "")
    return _probes[key]


def skip_if_cpu_inductor_missing(proc, env, args = ()):
    """Skip only a failed non-CUDA child whose own failure is a missing compiler that a fresh probe confirms."""
    if proc.returncode == 0 or "cuda" in args:
        return
    if not missing_compiler(proc.returncode, (proc.stdout or "") + (proc.stderr or "")):
        return
    returncode, output = _probe(env)
    if missing_compiler(returncode, output):
        reason = [x.strip() for x in output.splitlines() if _MISSING.search(x)][-1]
        pytest.skip(f"CPU Inductor has no C++ compiler here ({reason[:200]})")
