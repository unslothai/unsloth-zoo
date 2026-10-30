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

CPU Inductor needs a C++ compiler (cl on Windows, g++ / clang++ elsewhere); AMD Windows boxes often have none. A
fresh process asks Inductor for that compiler; the checks skip only when it reports a missing one. The checks
themselves run unchanged, so no failure of theirs is ever reinterpreted.
"""

import json
import os
import re
import subprocess
import sys

import pytest

# The probe asks Inductor for its C++ compiler directly, so an inherited suppress_errors / compile-disable setting
# cannot let it pass through an eager fallback; torch without cpp_builder falls back to a trivial compile. It reports
# its own failure as one JSON line, so torch's hint footers and colours never matter. "stage" is "import" when torch
# or Inductor itself could not be imported (no CPU Inductor here at all).
_PROBE = r"""
import json, sys
stage = "import"
try:
    import torch
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
    except ImportError:
        get_cpp_compiler = None
    stage = "compiler"
    if get_cpp_compiler is not None:
        get_cpp_compiler()
    else:
        torch.compile(lambda x: x + 1)(torch.ones(2))
except BaseException as e:
    chain, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append([type(e).__name__, str(e)])
        e = e.__cause__ or getattr(e, "inner_exception", None) or e.__context__
    print("PROBE " + json.dumps({"stage": stage, "chain": chain}))
    sys.exit(3)
"""
# Linux / torch 2.13+: InvalidCxxCompiler, raised directly or wrapped in an InductorError / BackendCompilerFailed.
# torch 2.9 - 2.12 on Windows: RuntimeError("Compiler: cl is not found.").
_WINDOWS = re.compile(r"^Compiler: .* is not found\.$")
_probes = {}


def _report(stdout):
    """The probe's JSON report, or None when there is none or it is not the expected shape."""
    lines = [x for x in (stdout or "").splitlines() if x.startswith("PROBE ")]
    if not lines:
        return None
    try:
        report = json.loads(lines[-1][len("PROBE "):])
    except ValueError:
        return None
    if not isinstance(report, dict) or not isinstance(report.get("chain"), list):
        return None
    return report


def missing_compiler(returncode, stdout):
    """True only when the probe ran its except branch and an exception in the chain is a missing compiler."""
    report = _report(stdout) if returncode == 3 else None
    if report is None or report.get("stage") != "compiler":
        return False
    for link in report["chain"]:
        if not (isinstance(link, list) and len(link) == 2 and all(isinstance(x, str) for x in link)):
            continue
        name, msg = link
        if name == "InvalidCxxCompiler" or (name == "RuntimeError" and _WINDOWS.match(msg.strip())):
            return True
    return False


def _probe(env):
    # The child gets the whole env and this cwd (relative CXX / PATH entries resolve against it), so both are
    # in the key; pytest's per-test bookkeeping variables are not, so a session probes once.
    key = (sys.executable, os.getcwd()) + tuple(sorted((k, v) for k, v in env.items() if not k.startswith("PYTEST_")))
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
        pytest.skip("CPU Inductor has no C++ compiler here (Inductor reported a missing compiler)")
