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

"""The up-front CPU Inductor probe skips only on a missing C++ compiler."""

import json
import os

import pytest

import _cpu_compiler_skip as skip


def _report(*chain):
    return "PROBE " + json.dumps([list(x) for x in chain]) + "\n"


@pytest.mark.parametrize("chain", [
    [("InductorError", "InvalidCxxCompiler: No working C++ compiler found"), ("InvalidCxxCompiler", "No working")],
    [("BackendCompilerFailed", "backend='inductor' raised:"), ("InvalidCxxCompiler", "No working C++ compiler")],
    [("InductorError", "RuntimeError: Compiler: cl is not found."), ("RuntimeError", "Compiler: cl is not found.")],
])
def test_a_missing_compiler_skips(chain, monkeypatch):
    monkeypatch.setattr(skip, "_probe", lambda env: (3, _report(*chain)))
    with pytest.raises(pytest.skip.Exception, match = "no C\\+\\+ compiler"):
        skip.skip_without_cpu_inductor({})


@pytest.mark.parametrize("result", [
    (0, ""),  # compiles
    (3, _report(("ImportError", "cannot import name 'InvalidCxxCompiler'"))),  # names it, is not it
    (3, _report(("AssertionError", "InvalidCxxCompiler was expected"))),
    (3, _report(("RuntimeError", "Compiler: cl is not found. Retrying"))),
    (1, "PROBE " + json.dumps([["InvalidCxxCompiler", ""]])),  # not the probe's own except branch
    (3, "PROBE not json"),
    (None, ""),  # timed out
])
def test_any_other_probe_result_runs_the_check(result, monkeypatch):
    monkeypatch.setattr(skip, "_probe", lambda env: result)
    skip.skip_without_cpu_inductor({})


def test_the_probe_is_cached_per_compile_environment(monkeypatch):
    calls = []

    class _Proc:
        returncode, stdout = 0, ""

    monkeypatch.setattr(skip, "_probes", {})
    monkeypatch.setattr(skip.subprocess, "run", lambda *a, **k: calls.append(k["env"]) or _Proc())
    for env in ({"CXX": "g++"}, {"CXX": "g++", "HOME": "/x"}, {"CXX": "clang++"}, {"CXX": "g++", "TORCH_LOGS": "+x"}):
        skip.skip_without_cpu_inductor(env)
    assert [e.get("CXX") for e in calls] == ["g++", "clang++", "g++"], calls


def test_a_real_probe_with_a_bogus_compiler_skips():
    pytest.importorskip("torch")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("TORCH")}
    env["CXX"] = os.path.join(os.sep, "nonexistent", "cxx-for-test")
    returncode, stdout = skip._probe(env)
    if returncode == 0:
        pytest.skip("this torch compiles without the configured C++ compiler, so there is nothing to classify")
    assert skip.missing_compiler(returncode, stdout), stdout[-2000:]
