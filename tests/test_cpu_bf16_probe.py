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


@pytest.fixture(autouse = True)
def probe_cache_dir(monkeypatch, tmp_path):
    """Every test gets its own verdict cache, never the user's ~/.cache/unsloth."""
    directory = tmp_path / "probe_cache"
    monkeypatch.setenv("UNSLOTH_CPU_PROBE_CACHE_DIR", str(directory))
    monkeypatch.delenv("UNSLOTH_CPU_PROBE_CACHE", raising = False)
    return directory


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
        if command[-2:-1] != ["-c"]:  # not the probe (macOS reads its CPU name with sysctl)
            return subprocess.CompletedProcess(command, 0, stdout = "", stderr = "")
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


# The verdict persists on disk: each new process (every `import unsloth` on a CPU-only host, every Studio worker)
# would otherwise pay a second torch import plus the benchmark.

FAST = "UNSLOTH_BF16_RATIO=0.4000\nUNSLOTH_BF16_FAST=1\n"
SLOW = "UNSLOTH_BF16_RATIO=40.0000\nUNSLOTH_BF16_FAST=0\n"


def _new_process(runtime):
    runtime.cpu_bf16_supported.cache_clear()


def _cache_file(probe_cache_dir):
    return probe_cache_dir / "cpu_bf16_probe.json"


@pytest.mark.parametrize("stdout, expected", [(FAST, True), (SLOW, False)])
def test_verdict_persists_across_processes(runtime, monkeypatch, probe_cache_dir, stdout, expected):
    calls = _spoof(monkeypatch, (0, stdout))
    assert runtime.cpu_bf16_supported() is expected
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is expected
    assert len(calls) == 1
    assert _cache_file(probe_cache_dir).is_file()


def test_illegal_instruction_persists_and_still_warns(runtime, monkeypatch):
    calls = _spoof(monkeypatch, (0xC000001D, ""))
    with warnings.catch_warnings(record = True):
        warnings.simplefilter("always")
        assert runtime.cpu_bf16_supported() is False
    _new_process(runtime)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert runtime.cpu_bf16_supported() is False
    assert any("illegal instruction" in str(w.message) for w in caught)
    assert len(calls) == 1


@pytest.mark.parametrize("result", [
    subprocess.TimeoutExpired("python", 180),
    OSError("no interpreter"),
    (1, "Traceback ..."),
    (0, ""),
])
def test_unsettled_results_are_not_cached(runtime, monkeypatch, probe_cache_dir, result):
    calls = _spoof(monkeypatch, result)
    assert runtime.cpu_bf16_supported() is False
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is False
    assert len(calls) == 2
    assert not _cache_file(probe_cache_dir).exists()


@pytest.mark.parametrize("stdout, expected", [
    ("UNSLOTH_BF16_RATIO=1.6000\nUNSLOTH_BF16_FAST=0\n", False),  # a native CPU on a loaded host
    ("UNSLOTH_BF16_RATIO=1.2000\nUNSLOTH_BF16_FAST=1\n", True),
    ("UNSLOTH_BF16_FAST=1\n", True),  # no ratio
    ("UNSLOTH_BF16_RATIO=nan?\nUNSLOTH_BF16_FAST=1\n", True),
])
def test_close_or_unknown_ratio_answers_but_is_not_cached(runtime, monkeypatch, probe_cache_dir, stdout, expected):
    calls = _spoof(monkeypatch, (0, stdout))
    assert runtime.cpu_bf16_supported() is expected
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is expected
    assert len(calls) == 2
    assert not _cache_file(probe_cache_dir).exists()


def test_timeout_then_answer_is_cached(runtime, monkeypatch):
    _spoof(monkeypatch, subprocess.TimeoutExpired("python", 180))
    assert runtime.cpu_bf16_supported() is False
    _new_process(runtime)
    calls = _spoof(monkeypatch, (0, FAST))
    assert runtime.cpu_bf16_supported() is True
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is True
    assert len(calls) == 1


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"x": 1}', ""])
def test_corrupt_cache_reprobes_and_is_replaced(runtime, monkeypatch, probe_cache_dir, content):
    import json
    probe_cache_dir.mkdir()
    _cache_file(probe_cache_dir).write_text(content)
    calls = _spoof(monkeypatch, (0, FAST))
    assert runtime.cpu_bf16_supported() is True
    assert len(calls) == 1
    assert isinstance(json.loads(_cache_file(probe_cache_dir).read_text()), dict)
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is True and len(calls) == 1


def test_tampered_entry_reprobes(runtime, monkeypatch, probe_cache_dir):
    import json
    calls = _spoof(monkeypatch, (0, FAST))
    runtime.cpu_bf16_supported()
    path = _cache_file(probe_cache_dir)
    data = json.loads(path.read_text())
    path.write_text(json.dumps({key: {"fast": "yes"} for key in data}))
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is True
    assert len(calls) == 2


def test_changed_torch_or_interpreter_reprobes(runtime, monkeypatch):
    import sys
    import torch
    calls = _spoof(monkeypatch, (0, FAST))
    runtime.cpu_bf16_supported()
    monkeypatch.setattr(torch, "__version__", "0.0.1+test", raising = False)
    _new_process(runtime)
    runtime.cpu_bf16_supported()
    assert len(calls) == 2
    monkeypatch.setattr(sys, "executable", sys.executable + "_other")
    _new_process(runtime)
    runtime.cpu_bf16_supported()
    assert len(calls) == 3
    _new_process(runtime)
    runtime.cpu_bf16_supported()  # the new key is cached too
    assert len(calls) == 3


def test_changed_cpu_reprobes(runtime, monkeypatch):
    calls = _spoof(monkeypatch, (0, FAST))
    runtime.cpu_bf16_supported()
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64-test")
    _new_process(runtime)
    runtime.cpu_bf16_supported()
    assert len(calls) == 2


@pytest.mark.parametrize("override, expected", [("0", False), ("1", True)])
def test_override_wins_over_cached_verdict(runtime, monkeypatch, override, expected):
    calls = _spoof(monkeypatch, (0, SLOW if expected else FAST))
    assert runtime.cpu_bf16_supported() is (not expected)
    monkeypatch.setenv("UNSLOTH_CPU_BF16", override)
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is expected
    assert len(calls) == 1


def test_opt_out_env_probes_every_process(runtime, monkeypatch, probe_cache_dir):
    monkeypatch.setenv("UNSLOTH_CPU_PROBE_CACHE", "0")
    calls = _spoof(monkeypatch, (0, FAST))
    runtime.cpu_bf16_supported()
    _new_process(runtime)
    runtime.cpu_bf16_supported()
    assert len(calls) == 2
    assert not probe_cache_dir.exists()


def test_unwritable_cache_dir_still_answers(runtime, monkeypatch, tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    monkeypatch.setenv("UNSLOTH_CPU_PROBE_CACHE_DIR", str(blocker / "sub"))  # a file as parent: even root cannot mkdir
    calls = _spoof(monkeypatch, (0, FAST))
    assert runtime.cpu_bf16_supported() is True
    _new_process(runtime)
    assert runtime.cpu_bf16_supported() is True
    assert len(calls) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["a_file"]


def test_default_location_is_the_unsloth_user_cache(runtime, monkeypatch, tmp_path):
    import os
    monkeypatch.delenv("UNSLOTH_CPU_PROBE_CACHE_DIR")
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path), 1))
    assert runtime._probe_cache_path() == os.path.join(str(tmp_path), ".cache", "unsloth", "cpu_bf16_probe.json")


def test_linux_flags_path_writes_no_cache(monkeypatch, probe_cache_dir):
    from unsloth_zoo.cpu import runtime
    monkeypatch.delenv("UNSLOTH_CPU_BF16", raising = False)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset({"avx512_bf16"}))
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    runtime.cpu_bf16_supported.cache_clear()
    try:
        assert runtime.cpu_bf16_supported() is True
    finally:
        runtime.cpu_bf16_supported.cache_clear()
    assert not probe_cache_dir.exists()


_LOAD_RUNTIME = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_zoo_cpu_runtime", sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)  # by file: the probe cache needs neither torch nor the rest of unsloth_zoo
"""


def _runtime_file():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "unsloth_zoo", "cpu", "runtime.py")


_STORE_WORKER = _LOAD_RUNTIME + """
path, tag = sys.argv[2], sys.argv[3]
for i in range(40):
    runtime._probe_cache_store(path, tag + str(i), i % 2 == 0, False)
"""


def test_concurrent_writers_never_tear_the_file(probe_cache_dir):
    import json
    import os
    import sys
    path = str(_cache_file(probe_cache_dir))
    workers = [subprocess.Popen([sys.executable, "-c", _STORE_WORKER, _runtime_file(), path, tag]) for tag in "abcd"]
    reads = 0
    while any(w.poll() is None for w in workers):
        if os.path.exists(path):
            with open(path, encoding = "utf-8") as f:
                assert isinstance(json.load(f), dict)  # os.replace: a reader never sees a partial write
            reads += 1
    assert all(w.wait() == 0 for w in workers)
    data = json.loads(open(path, encoding = "utf-8").read())
    assert 1 <= len(data) <= 64 and all(isinstance(v["fast"], bool) for v in data.values())
    assert not [n for n in os.listdir(str(probe_cache_dir)) if n.endswith(".tmp")]


_REAL_WORKER = _LOAD_RUNTIME + """
runtime.cpu_flags = lambda: frozenset()
print("VERDICT=" + str(runtime.cpu_bf16_supported()))
"""


def test_two_real_first_runs_share_one_valid_entry(probe_cache_dir):
    """Two processes probing at once with an empty cache: each answers, the file holds at most one valid entry (a ratio
    too close to the threshold on a loaded host is not stored), and a later process reuses a stored verdict."""
    import json
    import os
    import sys
    env = dict(os.environ)
    env.pop("UNSLOTH_CPU_BF16", None)

    def run_all(count):
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _REAL_WORKER, _runtime_file()], env = env, stdout = subprocess.PIPE, text = True,
            ) for _ in range(count)
        ]
        return [p.communicate(timeout = 300)[0].strip().splitlines()[-1] for p in procs]
    for verdict in run_all(2):
        assert verdict in ("VERDICT=True", "VERDICT=False")
    path = _cache_file(probe_cache_dir)
    if not path.exists():
        pytest.skip("both ratios were too close to the threshold to cache on this (loaded) host")
    data = json.loads(path.read_text())
    assert len(data) == 1
    (entry,) = data.values()
    assert run_all(1) == ["VERDICT=" + str(entry["fast"])]
