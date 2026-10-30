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

"""Skip a failed subprocess check when CPU Inductor has no C++ compiler here.

CPU Inductor needs a C++ compiler (cl on Windows, g++ / clang++ elsewhere); AMD Windows boxes often have none. Only
after a child run fails, one trivial compile in a fresh process with the same environment decides: the test skips
only when that probe fails for a missing compiler. A run that needs no compiler still asserts, and a failure on a host
that has one still fails.
"""

import re
import subprocess
import sys

import pytest

_PROBE = "import torch; torch.compile(lambda x: x + 1, backend = 'inductor')(torch.ones(2))"
# Linux / torch 2.13+: InvalidCxxCompiler. torch 2.9 - 2.12 on Windows: RuntimeError("Compiler: cl is not found.").
_MISSING = re.compile(r"InvalidCxxCompiler|Compiler: .* is not found\.")
_probes = {}


def missing_compiler(returncode, output):
    return returncode != 0 and _MISSING.search(output) is not None


def _probe(env):
    key = (sys.executable,) + tuple(sorted(
        (k, v) for k, v in env.items() if k.upper() in ("CXX", "CC", "PATH") or k.upper().startswith("TORCHINDUCTOR_")
    ))
    if key not in _probes:
        try:
            proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output = True, text = True,
                                  errors = "replace", env = env, timeout = 600)
            _probes[key] = (proc.returncode, (proc.stdout + proc.stderr)[-4000:])
        except subprocess.TimeoutExpired:
            _probes[key] = (0, "")
    return _probes[key]


def skip_if_cpu_inductor_missing(env):
    returncode, output = _probe(env)
    if missing_compiler(returncode, output):
        reason = [x.strip() for x in output.splitlines() if _MISSING.search(x)][-1]
        pytest.skip(f"CPU Inductor has no C++ compiler here ({reason[:200]})")
