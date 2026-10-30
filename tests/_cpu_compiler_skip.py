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

"""Skip the CPU subprocess checks up front when CPU Inductor has no C++ compiler here.

CPU Inductor needs a C++ compiler (cl on Windows, g++ / clang++ elsewhere); AMD Windows boxes often have none. One
trivial compile runs in a fresh process; the checks skip only when it fails with a missing compiler. The checks
themselves run unchanged, so no failure of theirs is ever reinterpreted.
"""

import json
import os
import re
import subprocess
import sys

import pytest

# The probe reports its own failure as one JSON line, so torch's hint footers and colours never matter.
_PROBE = r"""
import json, sys
try:
    import torch
    torch.compile(lambda x: x + 1)(torch.ones(2))
except BaseException as e:
    chain, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append([type(e).__name__, str(e)])
        e = e.__cause__ or getattr(e, "inner_exception", None) or e.__context__
    print("PROBE " + json.dumps(chain))
    sys.exit(3)
"""
# Linux / torch 2.13+: InvalidCxxCompiler, raised directly or wrapped in an InductorError / BackendCompilerFailed.
# torch 2.9 - 2.12 on Windows: RuntimeError("Compiler: cl is not found.").
_WINDOWS = re.compile(r"^Compiler: .* is not found\.$")
# Everything that changes whether the probe compiles.
_KEY_NAMES = ("CXX", "CC", "PATH")
_KEY_PREFIXES = ("TORCH",)
_probes = {}


def missing_compiler(returncode, stdout):
    """True only when the probe ran its except branch and an exception in the chain is a missing compiler."""
    lines = [x for x in (stdout or "").splitlines() if x.startswith("PROBE ")]
    if returncode != 3 or not lines:
        return False
    try:
        chain = json.loads(lines[-1][len("PROBE "):])
    except ValueError:
        return False
    return any(name == "InvalidCxxCompiler" or (name == "RuntimeError" and _WINDOWS.match(msg.strip()))
               for name, msg in chain)


def _probe(env):
    key = (sys.executable,) + tuple(sorted((k, v) for k, v in env.items()
                                           if k.upper() in _KEY_NAMES or k.upper().startswith(_KEY_PREFIXES)))
    if key not in _probes:
        try:
            proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output = True, text = True,
                                  errors = "replace", env = env, timeout = 600)
            _probes[key] = (proc.returncode, proc.stdout)
        except subprocess.TimeoutExpired:
            _probes[key] = (None, "")
    return _probes[key]


def skip_without_cpu_inductor(env = None):
    """Skip when a trivial CPU compile fails with a missing C++ compiler; any other result runs the check."""
    returncode, stdout = _probe(dict(os.environ) if env is None else env)
    if missing_compiler(returncode, stdout):
        pytest.skip("CPU Inductor has no C++ compiler here (a trivial torch.compile failed with a missing compiler)")
