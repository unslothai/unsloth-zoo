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
        assert threads == min(32, max(4, os.cpu_count()))


def test_determine_compile_threads_ignores_other_values(monkeypatch):
    """A request for 4 still gets the auto detected value; do not widen that."""
    import sys
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "4")
    threads = determine_compile_threads()
    if sys.platform == "win32":
        assert threads == 1
    else:
        assert threads == min(32, max(4, os.cpu_count()))


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
        assert threads == min(32, max(4, os.cpu_count())), (
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
