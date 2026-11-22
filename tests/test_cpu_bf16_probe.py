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

"""CPU bf16 speed probe (no /proc/cpuinfo: Windows, macOS): runs in a child process, so an illegal-instruction crash
falls back to fp32 instead of killing the caller, and it runs once per process."""

import subprocess
import warnings

import pytest


@pytest.fixture
def runtime(monkeypatch):
    from unsloth_zoo.cpu import runtime
    monkeypatch.delenv("UNSLOTH_CPU_BF16", raising = False)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset())  # as on Windows
    runtime.cpu_bf16_supported.cache_clear()
    yield runtime
    runtime.cpu_bf16_supported.cache_clear()


def _spoof(monkeypatch, result):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if isinstance(result, BaseException):
            raise result
        return subprocess.CompletedProcess(command, result[0], stdout = result[1], stderr = "")
    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("returncode", [132, -4, 0xC000001D])  # SIGILL as bash / POSIX / Windows report it
def test_crashing_probe_falls_back_to_fp32(runtime, monkeypatch, returncode):
    calls = _spoof(monkeypatch, (returncode, ""))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert runtime.cpu_bf16_supported() is False
    assert any("illegal instruction" in str(w.message) for w in caught)
    assert runtime.cpu_bf16_supported() is False
    assert len(calls) == 1  # cached: the probe runs at most once per process


@pytest.mark.parametrize("result, expected", [
    ((0, "UNSLOTH_BF16_FAST=1\n"), True),
    ((0, "UNSLOTH_BF16_FAST=0\n"), False),
    ((0, ""), False),
    ((1, "Traceback ..."), False),
    (subprocess.TimeoutExpired("python", 180), False),
    (OSError("no interpreter"), False),
])
def test_probe_result_and_failures(runtime, monkeypatch, result, expected):
    calls = _spoof(monkeypatch, result)
    assert runtime.cpu_bf16_supported() is expected
    assert runtime.cpu_bf16_supported() is expected
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-2] == "-c" and "timeout" in kwargs and kwargs["timeout"] > 0


def test_readable_cpu_flags_never_spawn_the_probe(monkeypatch):
    """Linux (/proc/cpuinfo) keeps its flag check: no child process."""
    from unsloth_zoo.cpu import runtime
    monkeypatch.delenv("UNSLOTH_CPU_BF16", raising = False)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset({"avx512_bf16", "fpu"}))
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    calls = _spoof(monkeypatch, (0, "UNSLOTH_BF16_FAST=0\n"))
    runtime.cpu_bf16_supported.cache_clear()
    try:
        assert runtime.cpu_bf16_supported() is True and calls == []
    finally:
        runtime.cpu_bf16_supported.cache_clear()


def test_real_probe_child_answers():
    """The child really runs here and returns a bool without touching this process."""
    from unsloth_zoo.cpu import runtime
    assert runtime._bf16_autocast_is_fast() in (True, False)
