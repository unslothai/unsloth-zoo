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

"""The no-C++-compiler probe behind the GPT-OSS decode subprocess checks."""

import os
import subprocess
import sys

import pytest
from _pytest.outcomes import Skipped

import _cpu_compiler_skip as probe


@pytest.mark.parametrize("returncode, output, missing", [
    (1, "torch._inductor.exc.InductorError: RuntimeError: Compiler: cl is not found.", True),
    (1, "torch._inductor.exc.InvalidCxxCompiler: No working C++ compiler found in torch._inductor.config.cpp.cxx", True),
    (1, "Traceback (most recent call last):\nValueError: boom", False),
    (0, "", False),
    (0, "warning: InvalidCxxCompiler", False),
])
def test_missing_compiler(returncode, output, missing):
    assert probe.missing_compiler(returncode, output) is missing


_CHILD_MISSING = "torch._inductor.exc.InductorError: RuntimeError: Compiler: cl is not found."


def _child(returncode, stderr):
    return subprocess.CompletedProcess(["child"], returncode, "", stderr)


def _stub_probe(monkeypatch, returncode, output):
    monkeypatch.setattr(probe, "_probes", {})
    calls = []

    def run(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess(a, returncode, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("returncode, output, skips", [
    (1, "InductorError: RuntimeError: Compiler: C:/\u00fcbersetzer/cl.exe is not found.", True),
    (1, "RuntimeError: out of memory", False),
    (0, "", False),
])
def test_only_a_probe_that_finds_no_compiler_skips(monkeypatch, returncode, output, skips):
    _stub_probe(monkeypatch, returncode, output)
    if skips:
        with pytest.raises(Skipped, match = r"no C\+\+ compiler here \(InductorError"):
            probe.skip_if_cpu_inductor_missing(_child(1, _CHILD_MISSING), dict(os.environ), ("cpu",))
    else:
        assert probe.skip_if_cpu_inductor_missing(_child(1, _CHILD_MISSING), dict(os.environ), ("cpu",)) is None


def test_an_unrelated_child_failure_fails_on_a_host_without_a_compiler(monkeypatch):
    # The probe would confirm a missing compiler, but the child failed for another reason: no skip.
    calls = _stub_probe(monkeypatch, 1, _CHILD_MISSING)
    child = _child(1, "Traceback (most recent call last):\nAssertionError: tokens differ")
    assert probe.skip_if_cpu_inductor_missing(child, dict(os.environ), ("cpu",)) is None
    assert calls == []


def test_a_cuda_child_never_skips(monkeypatch):
    calls = _stub_probe(monkeypatch, 1, _CHILD_MISSING)
    assert probe.skip_if_cpu_inductor_missing(_child(1, _CHILD_MISSING), dict(os.environ), ("cuda",)) is None
    assert calls == []


def test_a_passing_child_never_probes(monkeypatch):
    calls = _stub_probe(monkeypatch, 1, _CHILD_MISSING)
    assert probe.skip_if_cpu_inductor_missing(_child(0, _CHILD_MISSING), dict(os.environ), ("cpu",)) is None
    assert calls == []


def test_the_probe_reads_its_whole_output(monkeypatch):
    # A long shutdown tail after the compiler error must not push it out of what is classified.
    _stub_probe(monkeypatch, 1, _CHILD_MISSING + "\n" + "W1030 shutdown noise\n" * 2000)
    with pytest.raises(Skipped, match = r"no C\+\+ compiler here"):
        probe.skip_if_cpu_inductor_missing(_child(1, _CHILD_MISSING), dict(os.environ), ("cpu",))


@pytest.mark.parametrize("name", ["TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_SUPPRESS_ERRORS"])
def test_the_probe_cache_tells_compile_switches_apart(monkeypatch, name):
    calls = _stub_probe(monkeypatch, 0, "")
    env = {k: v for k, v in os.environ.items() if k != name}
    probe._probe(env)
    probe._probe(dict(env, **{name: "1"}))
    assert len(calls) == 2


def test_a_missing_compiler_is_found_by_a_real_probe(monkeypatch):
    # Windows torch reads CXX as well (get_cpp_compiler: os.environ.get("CXX", "cl")), so this runs everywhere.
    monkeypatch.setattr(probe, "_probes", {})
    env = {k: v for k, v in os.environ.items() if k.upper() not in probe._KEY_VARS[3:]}
    env["CXX"] = os.path.join(os.sep, "nonexistent", "g++")
    with pytest.raises(Skipped, match = r"no C\+\+ compiler here"):
        probe.skip_if_cpu_inductor_missing(_child(1, _CHILD_MISSING), env, ("cpu",))
