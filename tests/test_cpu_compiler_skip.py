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


@pytest.mark.parametrize("returncode, output, skips", [
    (1, "InductorError: RuntimeError: Compiler: C:/\u00fcbersetzer/cl.exe is not found.", True),
    (1, "RuntimeError: out of memory", False),
    (0, "", False),
])
def test_only_a_probe_that_finds_no_compiler_skips(monkeypatch, returncode, output, skips):
    monkeypatch.setattr(probe, "_probes", {})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, returncode, output, ""))
    if skips:
        with pytest.raises(Skipped, match = r"no C\+\+ compiler here \(InductorError"):
            probe.skip_if_cpu_inductor_missing(dict(os.environ))
    else:
        assert probe.skip_if_cpu_inductor_missing(dict(os.environ)) is None


@pytest.mark.skipif(sys.platform == "win32", reason = "Inductor uses cl on Windows, whatever CXX says")
def test_a_missing_compiler_is_found_by_a_real_probe(monkeypatch):
    monkeypatch.setattr(probe, "_probes", {})
    env = dict(os.environ, CXX = os.path.join(os.sep, "nonexistent", "g++"))
    with pytest.raises(Skipped, match = r"no C\+\+ compiler here"):
        probe.skip_if_cpu_inductor_missing(env)
