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
import contextlib
import sys
import types

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
    assert "if _UNSLOTH_FLASHINFER_UNUSABLE and _flash_attn_is_selectable():" in source
    guard = source.index("if _UNSLOTH_FLASHINFER_UNUSABLE and _flash_attn_is_selectable():")
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


# ------------------------------------------------- the pin must not be a downgrade

@pytest.mark.parametrize(
    "capability, expected",
    [((7, 0), False), ((7, 5), False), ((8, 0), True), ((9, 0), True), ((10, 0), True)],
)
def test_flash_attn_is_only_pinned_from_ampere_up(monkeypatch, capability, expected):
    """FLASH_ATTN declares supports_compute_capability() >= (8, 0), and an explicitly
    selected backend is a HARD requirement in vLLM: platforms/cuda.py raises "Selected
    backend ... is not valid for this configuration" rather than falling through.

    So pinning it on Volta (V100, 7.0) or Turing (T4, 7.5) converts a working XFORMERS
    run into a startup error, every time the FlashInfer pre-flight rejects."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert vllm_utils._flash_attn_is_selectable() is expected


def test_flash_attn_is_not_pinned_on_rocm(monkeypatch):
    """FLASH_ATTN is a CUDA backend. ROCm already takes its own branch in the pre-flight
    and must never be handed a CUDA-only hard pin."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising = False)
    assert vllm_utils._flash_attn_is_selectable() is False


def test_flash_attn_is_not_pinned_without_cuda(monkeypatch):
    """XPU and CPU: FLASH_ATTN is not a valid backend there at all."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert vllm_utils._flash_attn_is_selectable() is False


def test_an_unknown_device_does_not_get_a_hard_pin(monkeypatch):
    """Fail open. If we cannot establish the capability we have no basis for imposing a
    requirement vLLM will treat as absolute."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def _boom(*a, **k):
        raise RuntimeError("no device")
    monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
    assert vllm_utils._flash_attn_is_selectable() is False


def test_the_engine_arg_is_gated_on_the_capability_check():
    """Both conditions, not just the pre-flight verdict."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "if _UNSLOTH_FLASHINFER_UNUSABLE and _flash_attn_is_selectable():" in source


def test_installed_vllm_still_requires_sm80_for_flash_attn():
    """Pin the premise against the installed vLLM rather than trusting it. If upstream
    relaxes this, the guard is merely conservative, not wrong."""
    import importlib.util
    import pathlib
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        pytest.skip("vLLM not installed")
    fa = pathlib.Path(spec.origin).parent / "v1" / "attention" / "backends" / "flash_attn.py"
    if not fa.exists():
        pytest.skip("flash_attn backend not present in this vLLM")
    text = fa.read_text()
    assert "supports_compute_capability" in text
    assert "DeviceCapability(8, 0)" in text


# ------------------------------------------------- the block follows the engine lifecycle

def test_delete_vllm_hands_flashinfer_back(monkeypatch):
    """The block protects an engine. Once the engine is deleted there is nothing left to
    protect, and an unrelated `import flashinfer` in the same session should work again."""
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    vllm_utils._block_flashinfer_import()
    assert sys.modules.get("flashinfer") is None, "precondition: the block is on"
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_FLASHINFER_UNUSABLE", True, raising = False)

    # delete_vllm's real teardown needs a live distributed environment, which this host
    # does not have. The unblock runs first precisely so a failing teardown cannot skip it.
    with contextlib.suppress(Exception):
        vllm_utils.delete_vllm()

    assert sys.modules.get("flashinfer") is not None, "flashinfer is still hidden"
    assert vllm_utils._UNSLOTH_FLASHINFER_UNUSABLE is False


def test_the_unblock_is_the_first_thing_delete_vllm_does():
    """Ordering is the whole point: the teardown below it can raise."""
    source = inspect.getsource(vllm_utils.delete_vllm)
    body = [line.strip() for line in source.splitlines() if line.strip()
            and not line.strip().startswith("#")]
    # body[0] is the def line.
    assert "_unblock_flashinfer_import()" in body[:4], body[:5]


def test_a_failed_load_vllm_restores_flashinfer():
    """A startup error leaves no engine, so keeping the module hidden for the rest of the
    session buys nothing and breaks unrelated code."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "except BaseException:" in source, "no restore arm on the failure path"
    restore = source.index("except BaseException:")
    tail = source[restore:restore + 600]
    assert "_unblock_flashinfer_import()" in tail
    assert "_UNSLOTH_FLASHINFER_UNUSABLE = False" in tail
    assert "raise" in tail, "the original error must still propagate"
