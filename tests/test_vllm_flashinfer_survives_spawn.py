# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FlashInfer decision must reach the vLLM worker: on vLLM 0.29.0 spawn is forced once CUDA is initialised,
sys.modules is not inherited, and VLLM_ATTENTION_BACKEND was removed in 0.13.0, so the engine arg, pickled into the
child config, is the option; tests are CPU-only."""

from __future__ import annotations

import importlib
import importlib.util
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
    """os.environ crosses both fork and spawn but sys.modules crosses only fork, so the import block alone cannot carry the decision to the worker."""
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
    """An explicit attention backend is a hard pin that vLLM enforces, so it is set only when FlashInfer was rejected, never on hosts where it works or on ROCm."""
    source = inspect.getsource(vllm_utils.load_vllm)
    guard = source.index("if _UNSLOTH_FLASHINFER_UNUSABLE:")
    for backend in ('"FLASH_ATTN"', '"TRITON_MLA"'):
        assign = source.index('engine_args["attention_backend"] = %s' % backend)
        assert guard < assign, "%s is not behind the pre-flight verdict" % backend


def test_engine_arg_is_set_before_the_signature_filter():
    """The engine arg must be set before the EngineArgs signature filter, which deletes unsupported keys on an older vLLM instead of raising."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assign = source.index('engine_args["attention_backend"] = "FLASH_ATTN"')
    # The EngineArgs good_keys filter is what protects this key; an earlier good_keys in the function filters CompilationConfig and does not.
    filt = source.index(
        "good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs)"
    )
    assert assign < filt, "the key would not be validated against EngineArgs"


def test_the_flag_defaults_to_false():
    """Import time must not pin anything."""
    assert vllm_utils._UNSLOTH_FLASHINFER_UNUSABLE is False


def test_the_flag_is_reset_per_call():
    """The flag must reset per call, or one rejected load pins FLASH_ATTN for every later load_vllm in the process, even after nvcc is installed."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "global _UNSLOTH_FLASHINFER_UNUSABLE" in source
    assert "_UNSLOTH_FLASHINFER_UNUSABLE = False" in source


def test_both_blocking_arms_also_set_the_flag():
    """The sys.modules block and the engine arg must never disagree, or parent and worker choose different backends;
    each block arm also sets _UNSLOTH_FLASHINFER_UNUSABLE."""
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
    # Either EngineArgs accepts attention_backend or load_vllm's existing filter removes it; passing an argument that raises is the only bad outcome.
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
    """FLASH_ATTN needs compute capability 8.0+, and an explicit backend is a hard requirement, so pinning it on V100 or
    T4 makes a working XFORMERS run fail at startup."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert vllm_utils._flash_attn_is_selectable() is expected


def test_flash_attn_is_not_pinned_on_rocm(monkeypatch):
    """FLASH_ATTN is CUDA-only; ROCm takes its own pre-flight branch and must never receive this hard pin."""
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
    """Fail open: if the device capability cannot be established there is no basis for a requirement that vLLM treats as absolute."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def _boom(*a, **k):
        raise RuntimeError("no device")
    monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
    assert vllm_utils._flash_attn_is_selectable() is False


def test_the_engine_arg_is_gated_on_the_capability_check():
    """The non-MLA pin still needs the device check: FLASH_ATTN requires sm_80+."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assign = source.index('engine_args["attention_backend"] = "FLASH_ATTN"')
    guard = source.rindex("elif ", 0, assign)
    assert "_flash_attn_is_selectable()" in source[guard:assign]


def test_installed_vllm_still_requires_sm80_for_flash_attn():
    """Checks the installed vLLM still requires sm80 for FLASH_ATTN; if upstream relaxes this, the guard is merely conservative, not wrong."""
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
    """The block protects a live engine; once delete_vllm removes it, flashinfer must import again for unrelated code in the same session."""
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    vllm_utils._block_flashinfer_import()
    assert sys.modules.get("flashinfer") is None, "precondition: the block is on"
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_FLASHINFER_UNUSABLE", True, raising = False)

    # delete_vllm's real teardown needs a live distributed environment this host lacks; the unblock runs first so a failing teardown cannot skip it.
    with contextlib.suppress(Exception):
        vllm_utils.delete_vllm()

    assert sys.modules.get("flashinfer") is not None, "flashinfer is still hidden"
    assert vllm_utils._UNSLOTH_FLASHINFER_UNUSABLE is False


def test_the_unblock_is_the_first_thing_delete_vllm_does():
    """delete_vllm must unblock flashinfer before any step that can raise (ownership bookkeeping may come first); the
    distributed teardown below can raise and the module must not stay hidden."""
    source = inspect.getsource(vllm_utils.delete_vllm)
    assert source.index("_unblock_flashinfer_import()") < source.index(
        "from vllm.distributed.parallel_state import"
    ), "the teardown runs before the restore"


def test_a_failed_load_vllm_restores_flashinfer():
    """A failed load_vllm must restore flashinfer on the except BaseException path: a startup error leaves no engine, so keeping the block buys nothing."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "except BaseException:" in source, "no restore arm on the failure path"
    restore = source.index("except BaseException:")
    tail = source[restore:restore + 1200]
    assert "_unblock_flashinfer_import()" in tail
    assert "_UNSLOTH_FLASHINFER_UNUSABLE = False" in tail
    assert "raise" in tail, "the original error must still propagate"


def test_return_args_does_not_leave_flashinfer_blocked():
    """load_vllm with return_args=True builds no engine, so nothing reaches delete_vllm to lift the block; the FLASH_ATTN pin still travels in engine_args."""
    source = inspect.getsource(vllm_utils.load_vllm)
    marker = source.index("if return_args:")
    window = source[marker:marker + 1400]
    assert "_unblock_flashinfer_import()" in window, "the dry-run exit leaks the block"
    assert "_UNSLOTH_FLASHINFER_UNUSABLE = False" in window
    unblock = window.index("_unblock_flashinfer_import()")
    assert unblock < window.index("return engine_args"), "restore must precede the return"


# ------------------------------------------------- MLA models need an MLA backend

def test_an_mla_config_is_not_pinned_to_flash_attn(monkeypatch):
    """Under MLA, platforms/cuda.py's priority list holds only MLA backends and validate_configuration rejects an
    is_mla() mismatch, so a pinned FLASH_ATTN on DeepSeek-V2/V3 fails startup."""
    monkeypatch.delenv("VLLM_MLA_DISABLE", raising = False)
    deepseek = types.SimpleNamespace(kv_lora_rank = 512)
    assert vllm_utils._config_uses_mla(deepseek) is True

    # Also when it sits on a nested text_config, which is how the multimodal ones carry it.
    nested = types.SimpleNamespace(text_config = types.SimpleNamespace(kv_lora_rank = 512))
    assert vllm_utils._config_uses_mla(nested) is True


def test_an_ordinary_config_is_still_eligible_for_the_pin(monkeypatch):
    monkeypatch.delenv("VLLM_MLA_DISABLE", raising = False)
    llama = types.SimpleNamespace(hidden_size = 4096, num_attention_heads = 32)
    assert vllm_utils._config_uses_mla(llama) is False
    nested = types.SimpleNamespace(text_config = types.SimpleNamespace(hidden_size = 4096))
    assert vllm_utils._config_uses_mla(nested) is False


def test_vllm_mla_disable_turns_the_path_off(monkeypatch):
    """With MLA disabled vLLM takes the ordinary path, so the pin is valid again."""
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    assert vllm_utils._config_uses_mla(types.SimpleNamespace(kv_lora_rank = 512)) is False


def test_an_unreadable_config_is_assumed_to_be_mla(monkeypatch):
    """An unreadable config is assumed to be MLA: a false positive only lets vLLM pick its own backend, while a false negative is a startup error."""
    monkeypatch.delenv("VLLM_MLA_DISABLE", raising = False)

    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError("config is not readable")

    assert vllm_utils._config_uses_mla(_Hostile()) is True


def test_the_pin_is_gated_on_both_the_device_and_the_config():
    """Device capability alone is not enough: an MLA model on an sm_80+ host must not get the non-MLA FLASH_ATTN pin."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assign = source.index('engine_args["attention_backend"] = "FLASH_ATTN"')
    window = source[source.index("if _UNSLOTH_FLASHINFER_UNUSABLE:"):assign]
    assert "_config_uses_mla(config)" in window, "the MLA check does not gate the pin"
    assert "_flash_attn_is_selectable()" in window


def test_an_mla_model_gets_an_mla_backend_not_nothing(monkeypatch):
    """The engine arg is the only exclusion that survives into a spawned EngineCore; dropping it for MLA lets the worker
    re-probe FlashInfer and pick FLASHINFER_MLA, reintroducing the missing-nvcc failure."""
    source = inspect.getsource(vllm_utils.load_vllm)
    marker = source.index("if _UNSLOTH_FLASHINFER_UNUSABLE:")
    window = source[marker:marker + 500]
    assert '"TRITON_MLA"' in window, "MLA models are left with no durable exclusion"
    assert '"FLASH_ATTN"' in window
    assert window.index("_config_uses_mla(config)") < window.index('"TRITON_MLA"')


def test_triton_mla_is_a_real_backend_with_no_nvcc_requirement():
    """Checks the premise against the installed vLLM: TRITON_MLA is pure Triton, accepts every compute capability, and
    is the fallback once FlashInfer MLA backends are excluded."""
    import pathlib
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        pytest.skip("vLLM not installed")
    root = pathlib.Path(spec.origin).parent
    registry = root / "v1" / "attention" / "backends" / "registry.py"
    if not registry.is_file():
        pytest.skip("backend registry not at the expected path")
    assert "TRITON_MLA" in registry.read_text(), "TRITON_MLA is not a known backend name"

    triton_mla = root / "v1" / "attention" / "backends" / "mla" / "triton_mla.py"
    if triton_mla.is_file():
        text = triton_mla.read_text()
        capability = text.index("def supports_compute_capability")
        assert "return True" in text[capability:capability + 200], (
            "TRITON_MLA now restricts compute capability, so the pin needs re-checking"
        )


def test_the_dry_run_unblock_is_skipped_when_the_arg_was_filtered():
    """On an older vLLM without attention_backend in EngineArgs the filter drops the key, so unblocking would leave no
    exclusion and vLLM could pick FlashInfer again; a hidden module is the lesser harm."""
    source = inspect.getsource(vllm_utils.load_vllm)
    marker = source.index("if return_args:")
    window = source[marker:marker + 1400]
    guard = window.index('"attention_backend" in engine_args')
    unblock = window.index("_unblock_flashinfer_import()")
    assert guard < unblock, "the unblock is not gated on the argument surviving"
    assert "not _UNSLOTH_FLASHINFER_UNUSABLE or" in window, (
        "a run that never blocked anything should still unblock harmlessly"
    )


def test_the_filter_runs_before_the_dry_run_exit():
    """The gate reads engine_args AFTER filtering, so the order matters."""
    source = inspect.getsource(vllm_utils.load_vllm)
    assert source.index("good_keys = inspect.signature") < source.index("if return_args:")


# ------------------------------------------------- the block is owned, not global state

def test_a_second_failed_load_keeps_a_live_engine_s_block(monkeypatch):
    """The block is process-wide: a later failed load_vllm must not lift the block a running engine still needs for its lazy FlashInfer imports."""
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    vllm_utils._block_flashinfer_import()
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_FLASHINFER_UNUSABLE", True, raising = False)
    # One engine is alive and owns the block.
    monkeypatch.setattr(vllm_utils, "_UNSLOTH_FLASHINFER_BLOCK_OWNERS", 1, raising = False)

    source = inspect.getsource(vllm_utils.load_vllm)
    arm = source.index("except BaseException:")
    window = source[arm:arm + 900]
    guard = window.index("_UNSLOTH_FLASHINFER_BLOCK_OWNERS == 0")
    assert guard < window.index("_unblock_flashinfer_import()"), (
        "the failure arm unblocks without checking for a live owner"
    )
    assert sys.modules.get("flashinfer") is None, "precondition: still blocked"


def test_delete_vllm_only_unblocks_for_the_last_engine():
    source = inspect.getsource(vllm_utils.delete_vllm)
    assert "_UNSLOTH_FLASHINFER_BLOCK_OWNERS -= 1" in source, "ownership is never released"
    guard = source.index("_UNSLOTH_FLASHINFER_BLOCK_OWNERS == 0")
    assert guard < source.index("_unblock_flashinfer_import()")


def test_a_successful_load_takes_ownership():
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "_UNSLOTH_FLASHINFER_BLOCK_OWNERS += 1" in source, (
        "a successful engine never registers its dependence on the block"
    )
    # Only when something was actually blocked.
    take = source.index("_UNSLOTH_FLASHINFER_BLOCK_OWNERS += 1")
    assert "if _UNSLOTH_FLASHINFER_UNUSABLE:" in source[take - 200:take]


def test_the_dry_run_exit_also_respects_ownership():
    source = inspect.getsource(vllm_utils.load_vllm)
    marker = source.index("if return_args:")
    window = source[marker:marker + 1400]
    assert "_UNSLOTH_FLASHINFER_BLOCK_OWNERS == 0" in window, (
        "a dry run can still strip a live engine's block"
    )


def test_the_reprobe_unblock_also_respects_owners():
    """The top of load_vllm lifts a previous block so a new nvcc can be re-probed; with a live engine built under it,
    that strips its protection, and a healthy second load never reinstates it."""
    source = inspect.getsource(vllm_utils.load_vllm)
    marker = source.index("_no_flashinfer = os.environ.get")
    window = source[marker:marker + 400]
    unblock = window.index("_unblock_flashinfer_import()")
    assert "_UNSLOTH_FLASHINFER_BLOCK_OWNERS == 0" in window[:unblock], (
        "the re-probe unblock ignores live owners"
    )


def test_every_unblock_site_is_owner_gated():
    """Three exits in load_vllm reach _unblock_flashinfer_import (re-probe, dry run, failure arm); all three must agree or the weakest one wins."""
    source = inspect.getsource(vllm_utils.load_vllm)
    sites = [i for i in range(len(source))
             if source.startswith("_unblock_flashinfer_import()", i)]
    assert len(sites) == 3, "unexpected number of unblock sites: %d" % len(sites)
    for i in sites:
        preceding = source[max(0, i - 700):i]
        assert "_UNSLOTH_FLASHINFER_BLOCK_OWNERS == 0" in preceding, (
            "an unblock site at offset %d is not owner-gated" % i
        )
