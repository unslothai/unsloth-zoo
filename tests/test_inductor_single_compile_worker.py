# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth/_gpu_init.py forces TORCHINDUCTOR_COMPILE_THREADS=1 plus UNSLOTH_FORCE_SINGLE_COMPILE_WORKER on cgroup pinned
GPUs, as compile workers cannot enumerate one; patch_torch_compile and the options dict must not undo it; CPU only."""

from __future__ import annotations

import os

import pytest

from unsloth_zoo.temporary_patches.common import determine_compile_threads


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    """Both variables are process global, so they are cleared per test; determine_compile_threads is lru_cached and must have its cache cleared to observe them."""
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", raising = False)
    determine_compile_threads.cache_clear()
    yield
    determine_compile_threads.cache_clear()


def _available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def _run_patch_torch_compile():
    from unsloth_zoo.patching_utils import patch_torch_compile
    patch_torch_compile(debug = False)


def test_pop_is_skipped_when_single_worker_is_forced(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "1")
    _run_patch_torch_compile()
    assert os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") == "1"


def test_pop_still_happens_without_the_sentinel(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "8")
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


@pytest.mark.parametrize("sentinel", ["0", "", "auto", "true"])
def test_only_the_exact_sentinel_value_keeps_the_variable(monkeypatch, sentinel):
    """"auto" is the opt-out knob's own default, so it must not read as forcing."""
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", sentinel)
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


def test_determine_compile_threads_honours_the_sentinel(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    assert determine_compile_threads() == 1


def test_determine_compile_threads_unchanged_when_unset(monkeypatch):
    import sys
    threads = determine_compile_threads()
    if sys.platform == "win32":
        assert threads == 1
    else:
        assert threads == min(32, max(4, _available_cpus()))


def test_determine_compile_threads_ignores_other_values(monkeypatch):
    """A request for 4 still gets the auto detected value; do not widen that."""
    import sys
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "4")
    threads = determine_compile_threads()
    if sys.platform == "win32":
        assert threads == 1
    else:
        assert threads == min(32, max(4, _available_cpus()))


def test_the_env_var_alone_does_not_force_a_single_worker(monkeypatch):
    """Gated on UNSLOTH_FORCE_SINGLE_COMPILE_WORKER, not TORCHINDUCTOR_COMPILE_THREADS, which vLLM sets to 1 at import
    (vllm/env_override.py) and would silently cut ordinary hosts to one worker."""
    import sys
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.delenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", raising = False)
    threads = determine_compile_threads()
    if sys.platform == "win32":
        assert threads == 1
    else:
        assert threads == min(32, max(4, _available_cpus())), (
            "TORCHINDUCTOR_COMPILE_THREADS=1 alone forced a single compile worker; "
            "vLLM sets that variable on every import, so this would hit everyone"
        )


def test_vllm_really_does_set_the_variable_unconditionally():
    """Pin the premise against the installed vLLM rather than trusting it."""
    import importlib.util, pathlib
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        import pytest as _pytest
        _pytest.skip("vLLM not installed")
    override = pathlib.Path(spec.origin).parent / "env_override.py"
    if not override.exists():
        import pytest as _pytest
        _pytest.skip("vllm/env_override.py not present in this version")
    assert 'TORCHINDUCTOR_COMPILE_THREADS' in override.read_text()


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason = "no CPU affinity on this platform")
def test_determine_compile_threads_follows_cpu_affinity_not_the_host_total(monkeypatch):
    """A taskset / cpuset / Slurm / k8s limited process gets workers for its own CPUs, not os.cpu_count()."""
    monkeypatch.setattr(os, "cpu_count", lambda: 192)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(6)))
    assert determine_compile_threads() == 6
    determine_compile_threads.cache_clear()
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
    assert determine_compile_threads() == 4  # floor kept
    determine_compile_threads.cache_clear()
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)))
    assert determine_compile_threads() == 32  # cap kept


def test_determine_compile_threads_falls_back_to_cpu_count_without_affinity(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "platform", "darwin")  # macOS: no sched_getaffinity, and not the Windows early return
    monkeypatch.delattr(os, "sched_getaffinity", raising = False)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    assert determine_compile_threads() == 12
    determine_compile_threads.cache_clear()
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert determine_compile_threads() == 4


def test_compile_options_use_the_affinity_count_under_taskset():
    """End to end under a real taskset: the Inductor options dict carries the pinned CPU count."""
    import shutil, subprocess, sys, pathlib
    if sys.platform != "linux" or shutil.which("taskset") is None:
        pytest.skip("needs Linux taskset")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 6:
        pytest.skip("needs at least 6 CPUs to pin to")
    pinned = ",".join(str(c) for c in cpus[:6])
    repo = pathlib.Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in ("TORCHINDUCTOR_COMPILE_THREADS", "UNSLOTH_FORCE_SINGLE_COMPILE_WORKER")}
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import unsloth_zoo, unsloth_zoo.temporary_patches.common as c;"
        "print(unsloth_zoo.__file__);"
        "print(c.get_torch_compile_options()['compile_threads'])"
    )
    out = subprocess.run(["taskset", "-c", pinned, sys.executable, "-c", code], env = env, cwd = str(repo),
                         capture_output = True, text = True, timeout = 600)
    lines = out.stdout.strip().splitlines()
    assert len(lines) >= 2, out.stdout[-2000:] + out.stderr[-4000:]
    assert pathlib.Path(lines[-2]).resolve().is_relative_to(repo), lines[-2]
    assert int(lines[-1]) == 6
