# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FlashInfer decision has to reach the vLLM WORKER, not just this process.

Measured on vLLM 0.29.0:

- vLLM re-resolves the attention backend inside the worker
  (``platforms/cuda.py::get_attn_backend_cls``). ``VllmConfig`` carries only the
  request, so with ``backend=None`` every process re-runs the scan itself.
- On a default single-GPU run the EngineCore is its own child process
  (``VLLM_ENABLE_V1_MULTIPROCESSING`` defaults to True).
- ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to ``fork``, but
  ``utils/system_utils.py::_maybe_force_spawn`` overrides it to ``spawn`` when
  ``cuda_is_initialized()``, under WSL, under Ray, and for ``vllm serve``.

Unsloth always has CUDA initialised before ``load_vllm`` (the model is already on
the GPU), so the forced-spawn branch is the normal case, not the exotic one.
Measured directly: vLLM logs ``Overriding VLLM_WORKER_MULTIPROC_METHOD to
'spawn' ... Reasons: CUDA is initialized``.

``sys.modules`` is NOT inherited across spawn, so the import block alone is a
no-op there. ``VLLM_ATTENTION_BACKEND`` was removed in vLLM 0.13.0, so the env
var is not an option either. The engine arg is, because it travels inside the
config that gets pickled into the child.

These tests are CPU only and need no GPU. The multiprocessing ones use a
synthetic module rather than the real flashinfer, so they run anywhere.
"""

from __future__ import annotations

import importlib
import inspect
import multiprocessing as mp
import sys

import pytest

import unsloth_zoo.vllm_utils as vllm_utils


@pytest.fixture(autouse = True)
def _restore():
    vllm_utils._UNSLOTH_BLOCKED_FLASHINFER_MODULES.clear()
    before = vllm_utils._UNSLOTH_FLASHINFER_UNUSABLE
    yield
    vllm_utils._UNSLOTH_FLASHINFER_UNUSABLE = before
    vllm_utils._unblock_flashinfer_import()
    vllm_utils._UNSLOTH_BLOCKED_FLASHINFER_MODULES.clear()


# ------------------------------------------------- the inheritance asymmetry

def _child_probe(q):
    import importlib.util
    import os
    q.put({
        "find_spec_none": importlib.util.find_spec("unsloth_spawn_canary") is None,
        "env_seen": os.environ.get("UNSLOTH_SPAWN_CANARY"),
    })


def _make_canary(tmp_path):
    (tmp_path / "unsloth_spawn_canary.py").write_text("VALUE = 1\n")
    return str(tmp_path)


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_sys_modules_block_is_inherited_only_by_fork(tmp_path, monkeypatch, method):
    """The crux. os.environ crosses both start methods; sys.modules crosses only
    fork. This is why the block alone cannot carry the decision to the worker."""
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable on this platform")

    monkeypatch.syspath_prepend(_make_canary(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_SPAWN_CANARY", "set-in-parent")
    assert importlib.util.find_spec("unsloth_spawn_canary") is not None

    sys.modules["unsloth_spawn_canary"] = None       # the same block shape
    try:
        ctx = mp.get_context(method)
        q = ctx.Queue()
        p = ctx.Process(target = _child_probe, args = (q,))
        p.start()
        result = q.get(timeout = 120)
        p.join(timeout = 60)
    finally:
        sys.modules.pop("unsloth_spawn_canary", None)

    assert result["env_seen"] == "set-in-parent", "env should cross both methods"
    if method == "fork":
        assert result["find_spec_none"] is True, "fork should inherit the block"
    else:
        assert result["find_spec_none"] is False, (
            "spawn inherited the sys.modules block, which contradicts the "
            "premise of the engine-arg fix"
        )


# ------------------------------------------------- the durable channel

def test_engine_arg_is_set_only_when_flashinfer_was_rejected():
    """An explicit backend is a HARD PIN: vLLM raises on an unsupported value
    instead of falling through. Setting it unconditionally would break hosts
    where FlashInfer works, and ROCm, which never reaches the blocker."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert 'if _UNSLOTH_FLASHINFER_UNUSABLE:' in source
    guard = source.index("if _UNSLOTH_FLASHINFER_UNUSABLE:")
    assign = source.index('engine_args["attention_backend"] = "FLASH_ATTN"')
    assert guard < assign, "the assignment is not behind the guard"


def test_engine_arg_is_set_before_the_signature_filter():
    """Ordering matters twice over: the filter is what makes this safe on an
    older vLLM with no such argument, by deleting the key instead of raising."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assign = source.index('engine_args["attention_backend"] = "FLASH_ATTN"')
    # Specifically the EngineArgs filter. An earlier `good_keys` in this function
    # filters CompilationConfig and is not the one that protects this key.
    filt = source.index(
        "good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs)"
    )
    assert assign < filt, "the key would not be validated against EngineArgs"


def test_the_flag_defaults_to_false():
    """Import time must not pin anything."""
    assert vllm_utils._UNSLOTH_FLASHINFER_UNUSABLE is False


def test_the_flag_is_reset_per_call():
    """Otherwise one rejected call pins FLASH_ATTN for every later load_vllm in
    the process, including after the user installs nvcc."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "global _UNSLOTH_FLASHINFER_UNUSABLE" in source
    assert "_UNSLOTH_FLASHINFER_UNUSABLE = False" in source


def test_both_blocking_arms_also_set_the_flag():
    """The sys.modules block and the engine arg must never disagree, or the
    parent and the worker choose different backends."""
    source = inspect.getsource(vllm_utils.load_vllm)
    blocks = source.count("_block_flashinfer_import()")
    flags = source.count("_UNSLOTH_FLASHINFER_UNUSABLE = True")
    assert blocks == flags, (
        f"{blocks} call(s) block the import but {flags} set the engine-arg flag; "
        f"every block must also pin the backend for the worker"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason = "needs vLLM installed",
)
def test_installed_vllm_accepts_the_argument_or_the_filter_drops_it():
    """Forwards and backwards compatibility, checked against whatever is here."""
    from vllm import EngineArgs
    accepted = "attention_backend" in inspect.signature(EngineArgs).parameters
    # Either it is accepted, or load_vllm's existing filter removes it. Both are
    # fine; what would not be fine is passing an argument that raises.
    assert accepted or True
    if not accepted:
        source = inspect.getsource(vllm_utils.load_vllm)
        assert "del engine_args[key]" in source
