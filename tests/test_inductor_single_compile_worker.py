# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The single compile worker forcing survives patch_torch_compile and the options dict.

``unsloth/_gpu_init.py`` fingerprints the cgroup pinned GPU case
(``docker --gpus '"device=N"'``) and sets both ``TORCHINDUCTOR_COMPILE_THREADS=1``
and the ``UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1`` sentinel. It has to, because
Inductor's compile worker subprocesses cannot enumerate a GPU that the cgroup
pinned, and raise from ``torch/_inductor/runtime/triton_helpers.py``:

    RuntimeError: Could not find an active GPU backend

Two consumers in this repo used to undo that:

1. ``patch_torch_compile``'s non debug branch unconditionally popped
   ``TORCHINDUCTOR_COMPILE_THREADS``, letting the worker pool respawn.
2. ``determine_compile_threads`` returned the cpu count, and
   ``get_torch_compile_options`` puts that into the Inductor options dict,
   which takes precedence over the env var.

Both tests here are CPU only. They set and clear the env vars themselves so
they are valid on a bare metal host, which is the case that must stay unchanged.
"""

from __future__ import annotations

import os

import pytest

from unsloth_zoo.temporary_patches.common import determine_compile_threads


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    """Both variables are process global, so never leak them into another test.

    ``determine_compile_threads`` is ``lru_cache``d, which is correct in
    production (``_gpu_init`` sets the variable at ``import unsloth``, long
    before any compile options are built) but means a test must clear the cache
    to observe the env var at all."""
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
    """The control. Nothing on a bare metal host sets the sentinel, so the
    pre-existing cleanup must be untouched there."""
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "8")
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


@pytest.mark.parametrize("sentinel", ["0", "", "auto", "true"])
def test_only_the_exact_sentinel_value_keeps_the_variable(monkeypatch, sentinel):
    """``_gpu_init`` writes the literal "1". Anything else, including the "auto"
    default of the opt out knob, must not be read as forcing."""
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", sentinel)
    _run_patch_torch_compile()
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


def test_determine_compile_threads_honours_the_sentinel(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    assert determine_compile_threads() == 1


def test_determine_compile_threads_unchanged_when_unset(monkeypatch):
    """The control for the options dict half. Without the env var the auto
    detection from https://github.com/unslothai/unsloth-zoo/pull/187 stands."""
    import sys
    threads = determine_compile_threads()
    if sys.platform == "win32":
        assert threads == 1
    else:
        assert threads == min(32, max(4, os.cpu_count()))


def test_determine_compile_threads_ignores_other_values(monkeypatch):
    """A user asking for 4 workers gets the auto detected value, not 4. That is
    pre-existing behaviour and this change must not widen it."""
    import sys
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "4")
    threads = determine_compile_threads()
    if sys.platform == "win32":
        assert threads == 1
    else:
        assert threads == min(32, max(4, os.cpu_count()))


def test_the_env_var_alone_does_not_force_a_single_worker(monkeypatch):
    """The regression guard, and the whole reason this is gated on our own
    sentinel rather than on TORCHINDUCTOR_COMPILE_THREADS.

    vLLM sets TORCHINDUCTOR_COMPILE_THREADS="1" unconditionally at import
    (vllm/env_override.py). Reading that variable directly would drop every vLLM
    user on an ordinary host from the auto detected worker count to 1, which is a
    large and completely silent compile slowdown. Measured before the gate was
    narrowed: 32 on main, 1 on this branch, with nothing forced."""
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
    """The premise of the test above, pinned against the installed vLLM rather
    than taken on trust. If upstream ever stops doing this the gate is still
    correct, so this is informational and skips when vLLM is absent."""
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
