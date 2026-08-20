# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Regression guards for upstream-pinned symbols in the MLX / Apple-Silicon /
accelerator-dispatch lanes of unsloth_zoo.

Each test cites the zoo commit that introduced or repaired the symbol it
covers, so a future refactor that renames or silently drops the symbol
fails loudly here. Tests are designed to run on Linux+CUDA via the
``tests/mlx_simulation`` shim and on Apple Silicon natively; CUDA-only
APIs are not exercised directly so the suite is CPU-runnable in CI.
"""

from __future__ import annotations

import contextlib
import sys
import types
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. device_type.device_synchronize / device_empty_cache / device_is_bf16_supported
#    must tolerate a partial torch.xpu build that exposes is_available() but
#    lacks the specific call (synchronize / empty_cache / is_bf16_supported).
#
#    Covers commits:
#      - 35dc451 Guard XPU empty_cache call against partial torch.xpu builds
#      - e08c1df Guard XPU synchronize call against partial torch.xpu builds
#      - 2564f39 Route GGUF merge cache flushes and MoE expert merges
#                through active backend (introduced device_empty_cache)
#      - d631837 Route VLM GGUF mmproj bf16 check through active backend
#                (introduced device_is_bf16_supported)
#
#    The existing test_backend_device_helpers.py covers the happy path; this
#    test pins the PARTIAL-BUILD case where torch.xpu.is_available is True
#    but the specific symbol is missing.
# ---------------------------------------------------------------------------

def test_xpu_partial_build_all_three_helpers_silent_no_op():
    """All three device_type helpers must no-op (not AttributeError) on a
    torch.xpu module that lacks synchronize / empty_cache / is_bf16_supported.
    The hasattr-then-call pattern is the exact regression net for the
    e08c1df / 35dc451 / d631837 partial-build crashes seen in the GGUF
    merge and VLM mmproj export paths.
    """
    from unsloth_zoo import device_type as dt

    class PartialXpu:
        """A torch.xpu that knows is_available but nothing else.

        Reflects the upstream IPEX dev build where torch.xpu.is_available is
        True but synchronize / empty_cache / is_bf16_supported are not yet
        wired in. Pre-fix, this raised AttributeError mid-GGUF-export.
        """
        def is_available(self):
            return True

    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False

    with mock.patch.object(dt, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", PartialXpu(), create=True):
        # None of these may raise. The whole regression class is "raises
        # AttributeError because the partial xpu build is missing one of
        # the three call names".
        dt.device_synchronize()
        dt.device_empty_cache()
        assert dt.device_is_bf16_supported() is False


# ---------------------------------------------------------------------------
# 2. saving_utils._active_merge_device() must take NO positional args and
#    cascade cuda -> xpu -> mps -> cpu.
#
#    Covers commit:
#      - fd58aa1 saving_utils: route LoRA merge through accelerator-family probe
#      - 70b93ad fix(mlx): migrate deprecated mx.metal memory APIs + restore
#                device-agnostic LoRA merge
#
#    The pre-fix signature was _active_merge_device(W) which (a) silently
#    dropped MPS, (b) propagated W.device.index across families. This
#    pin asserts the no-arg shape AND the MPS-wins-when-only-mps branch
#    which the previous DEVICE_TYPE_TORCH-only routing dropped.
# ---------------------------------------------------------------------------

def test_active_merge_device_mps_branch_pinned():
    """_active_merge_device() returns "mps" on Apple Silicon (no cuda/xpu).
    This is the exact regression that broke the MLX backend's on-host LoRA
    merge when the helper still routed through DEVICE_TYPE_TORCH.
    """
    from unsloth_zoo.saving_utils import _active_merge_device

    _active_merge_device.cache_clear()
    try:
        # No required positional args. Pre-fix took W; signature change
        # alone would crash every callsite if reverted.
        import inspect
        sig = inspect.signature(_active_merge_device)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert required == [], (
            "_active_merge_device() must take no required args; the "
            "pre-fix W-arg signature silently propagated device.index "
            "across accelerator families."
        )

        # Spoof: only MPS available. The cuda-only cascade pre-fix dropped
        # this branch entirely; this assertion is the canary.
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            xpu_ctx = (
                mock.patch.object(torch.xpu, "is_available", return_value=False)
                if hasattr(torch, "xpu") else _NullCtx()
            )
            mps_stub = types.SimpleNamespace(is_available=lambda: True)
            mps_ctx = (
                mock.patch.object(torch.backends.mps, "is_available", return_value=True)
                if hasattr(torch.backends, "mps")
                else mock.patch.object(torch.backends, "mps", mps_stub, create=True)
            )
            with xpu_ctx, mps_ctx:
                _active_merge_device.cache_clear()
                assert _active_merge_device() == "mps"
    finally:
        _active_merge_device.cache_clear()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# 3. MoE-expert _active_merge_device() callsites in saving_utils.py.
#
#    Covers commit:
#      - 2564f39 (introduced)
#      - fd58aa1 (refactored to no-arg helper)
#
#    Pre-fix the five MoE expert helpers (_merge_moe_gate_expert,
#    _merge_moe_up_expert, _merge_moe_down_proj_expert,
#    _merge_moe_fused_gate_up_expert, _merge_moe_fused_down_proj_expert)
#    fell back to CPU on XPU due to hardcoded .to("cuda", ...). This pin
#    asserts those callsites still go through the helper.
# ---------------------------------------------------------------------------

def test_moe_expert_merges_call_active_merge_device():
    """The five MoE-expert merge helpers must route their .to(...) calls
    through _active_merge_device(). A regression to a hardcoded "cuda" or
    DEVICE_TYPE_TORCH inside any one of them silently drops MPS/XPU
    placement and was the exact 2564f39 bug class.

    After unsloth-zoo#647 the gate / up wrappers delegate to a unified
    helper ``_merge_moe_gate_or_up_expert``; the check follows that
    delegation by inspecting the union of each entry-point's source and
    the source of any sibling ``_merge_moe_*`` helper it explicitly
    forwards to.
    """
    import inspect
    import re
    import unsloth_zoo.saving_utils as su

    targets = [
        "_merge_moe_gate_expert",
        "_merge_moe_up_expert",
        "_merge_moe_down_proj_expert",
        "_merge_moe_fused_gate_up_expert",
        "_merge_moe_fused_down_proj_expert",
    ]
    _helper_call_re = re.compile(r"\b(_merge_moe_[A-Za-z0-9_]+)\(")

    def _effective_source(name: str, seen: set) -> str:
        """Return the entry-point's source plus the source of any
        sibling ``_merge_moe_*`` helper it explicitly forwards to.
        One-level follow is enough: zoo never chains wrapper -> wrapper
        -> helper, and the implementations all live in saving_utils."""
        if name in seen:
            return ""
        seen.add(name)
        fn = getattr(su, name, None)
        if fn is None:
            return ""
        src = inspect.getsource(fn)
        callees = set(_helper_call_re.findall(src)) - {name}
        for callee in callees:
            src += "\n" + _effective_source(callee, seen)
        return src

    for name in targets:
        fn = getattr(su, name, None)
        assert fn is not None, (
            f"{name} missing; the MoE-expert merge dispatch surface "
            "shrank without notice — see commit 2564f39."
        )
        src = _effective_source(name, set())
        assert "_active_merge_device(" in src, (
            f"{name} (and any sibling _merge_moe_* it delegates to) no "
            "longer routes through _active_merge_device(). That regresses "
            "2564f39 + fd58aa1: hardcoded 'cuda' breaks Intel XPU and "
            "Apple MPS LoRA merge."
        )
        assert '.to("cuda"' not in src and ".to('cuda'" not in src, (
            f"{name} (or the helper it delegates to) hardcodes "
            ".to('cuda', ...) again — same regression class as commit "
            "2564f39."
        )


# ---------------------------------------------------------------------------
# 4. mx.metal memory APIs migrated to the modern non-namespaced form.
#
#    Covers commit:
#      - 70b93ad fix(mlx): migrate deprecated mx.metal memory APIs +
#                restore device-agnostic LoRA merge
#
#    The deprecated form (mx.metal.set_memory_limit / .set_cache_limit)
#    prints a warning every training run; the modern form is
#    mx.set_memory_limit / mx.set_cache_limit / mx.set_wired_limit.
#    The MLX shim exposes both, so this test pins the trainer source.
# ---------------------------------------------------------------------------

def test_mlx_trainer_uses_modern_memory_apis_only():
    """unsloth_zoo.mlx.trainer must call the non-namespaced memory APIs
    (mx.set_memory_limit, mx.set_cache_limit, mx.set_wired_limit). The
    namespaced mx.metal.set_* forms are deprecated upstream and reverting
    to them resurrects the per-run deprecation warning that 70b93ad fixed.
    """
    import importlib.util
    import pathlib

    pkg_root = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    )
    # The MLX path was promoted from a flat module (mlx_trainer.py) to a
    # subpackage (mlx/trainer.py) in e6d8f7f. Accept either layout so the
    # test survives the rename.
    candidates = [pkg_root / "mlx" / "trainer.py", pkg_root / "mlx_trainer.py"]
    mlx_trainer_path = next((c for c in candidates if c.is_file()), None)
    assert mlx_trainer_path is not None, (
        f"Neither {candidates[0]} nor {candidates[1]} exists; the MLX "
        f"trainer module was relocated again. Update this test's path "
        f"candidates."
    )
    src = mlx_trainer_path.read_text()

    # The deprecated forms must NOT appear.
    assert "mx.metal.set_memory_limit" not in src, (
        "Deprecated mx.metal.set_memory_limit call resurfaced; "
        "regresses commit 70b93ad."
    )
    assert "mx.metal.set_cache_limit" not in src, (
        "Deprecated mx.metal.set_cache_limit call resurfaced; "
        "regresses commit 70b93ad."
    )

    # The modern forms must appear.
    for modern in ("mx.set_memory_limit", "mx.set_cache_limit", "mx.set_wired_limit"):
        assert modern in src, f"Expected modern API {modern} missing from {mlx_trainer_path.name}"


# ---------------------------------------------------------------------------
# 5. Apple-Silicon stub injection on __init__ (3 sub-bugs from 2053539).
#
#    Covers commit:
#      - 2053539 fix(mlx): repair stub injection on Apple Silicon (3 sub-bugs)
#
#    Sub-bugs:
#      a. Inverted gate: stubs were inside `if not _SKIP_GPU_INIT:`. Fix
#         moved them under `if _SKIP_GPU_INIT:`.
#      b. Wrong function name: install_*_stub vs the real inject_into_sys_modules.
#      c. _Noop.__call__ silently returned None — fix raises NotImplementedError.
# ---------------------------------------------------------------------------

def test_apple_silicon_stub_injection_entrypoints_pinned():
    """Sub-bugs (a) and (b) of commit 2053539. The init module must gate
    stub injection on `if _SKIP_GPU_INIT:` (NOT the negated form) and call
    inject_into_sys_modules (NOT install_*_stub).
    """
    import importlib.util
    import pathlib

    init_path = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    ) / "__init__.py"
    src = init_path.read_text()

    # Sub-bug (b): the real entry point is inject_into_sys_modules.
    assert "inject_into_sys_modules" in src, (
        "Stub injection entry point inject_into_sys_modules vanished from "
        "unsloth_zoo/__init__.py — regresses commit 2053539 sub-bug (b)."
    )
    # Pre-fix names that must NOT come back.
    assert "install_triton_stub" not in src
    assert "install_bitsandbytes_stub" not in src

    # Sub-bug (a): the gate must be positive `if _SKIP_GPU_INIT:` not
    # `if not _SKIP_GPU_INIT:` around the injection block. We look for the
    # exact positive line.
    assert "if _SKIP_GPU_INIT:" in src, (
        "Apple-Silicon stub-injection gate flipped — regresses commit "
        "2053539 sub-bug (a)."
    )


def test_stub_noop_call_raises_not_returns_none():
    """Sub-bug (c) of 2053539. _Noop.__call__ must raise NotImplementedError
    so a stray `bnb.functional.quantize_4bit(weight, ...)` on Apple Silicon
    crashes loudly rather than silently producing None that corrupts the
    downstream tensor pipeline. __bool__ and hasattr probes must still work.
    """
    from unsloth_zoo.stubs import triton_stub, bitsandbytes_stub

    for mod in (triton_stub, bitsandbytes_stub):
        noop = mod._Noop("test.symbol")
        with pytest.raises(NotImplementedError, match="test.symbol"):
            noop()
        # Optional-feature probes still work:
        assert bool(noop) is False  # __bool__ pass-through
        sub = noop.some_attr        # attribute chaining returns another _Noop
        assert sub is not noop
        with pytest.raises(NotImplementedError, match="test.symbol.some_attr"):
            sub()


# Reads sys.modules rather than importing either package: both stubs seed sys.modules
# directly, so an import would surface the regression as an ImportError instead of as a
# wrong verdict below.
_GATE_PROBE = """
import json, sys
if {hide_bitsandbytes}:
    # Simulate a host with no bitsandbytes regardless of what is really installed,
    # so this case cannot fail on a machine that happens to have the real one.
    import importlib.util
    _find_spec = importlib.util.find_spec
    importlib.util.find_spec = (
        lambda name, *a, **k: None if name == "bitsandbytes"
        else _find_spec(name, *a, **k)
    )
import unsloth_zoo  # runs the stub-injection gate
bnb = sys.modules.get("bitsandbytes")
triton = sys.modules.get("triton")
print(json.dumps({{
    "bnb_stubbed": bool(getattr(bnb, "IS_UNSLOTH_STUB", False)),
    "triton_stubbed": "triton_stub" in (getattr(triton, "__file__", "") or ""),
}}))
"""


@pytest.mark.parametrize("bitsandbytes_installed", [True, False])
def test_the_stub_gate_shadows_no_real_bitsandbytes_but_always_stubs_triton(
    tmp_path, bitsandbytes_installed,
):
    """Run the real gate in a fresh interpreter and look at what it produced.

    Shadowing a real bitsandbytes made every bnb-quantized checkpoint unloadable. Triton
    ships no macOS wheel to prefer, so it is stubbed either way.
    """
    import json
    import os
    import pathlib
    import subprocess

    import unsloth_zoo

    env = dict(os.environ)
    # Forces the skip-GPU-init branch on a non-MLX host too, so this runs everywhere.
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    path = [str(pathlib.Path(unsloth_zoo.__file__).parent.parent)]
    if bitsandbytes_installed:
        fake = tmp_path / "bitsandbytes"
        fake.mkdir()
        (fake / "__init__.py").write_text("__version__ = '0.0.0-fake'\n")
        path.insert(0, str(tmp_path))
    env["PYTHONPATH"] = os.pathsep.join(path)

    run = subprocess.run(
        [sys.executable, "-c",
         _GATE_PROBE.format(hide_bitsandbytes = not bitsandbytes_installed)],
        capture_output = True, text = True, env = env, timeout = 300,
    )
    # No skip on failure: the gate runs during `import unsloth_zoo`, so a crash in the
    # code under test and an unusable environment produce the same traceback.
    assert run.returncode == 0, f"gate probe failed:\n{run.stderr[-2000:]}"
    result = json.loads(run.stdout.strip().splitlines()[-1])

    assert result["bnb_stubbed"] is not bitsandbytes_installed, (
        "the stub shadowed a real bitsandbytes install"
        if bitsandbytes_installed else
        "bitsandbytes was left unstubbed with none installed"
    )
    assert result["triton_stubbed"], "triton stubbing became conditional"


def test_real_bitsandbytes_available_distinguishes_a_stub_from_a_real_module():
    from unsloth_zoo.stubs import bitsandbytes_stub

    real = types.ModuleType("bitsandbytes")
    stub_like = types.ModuleType("bitsandbytes")
    stub_like.IS_UNSLOTH_STUB = True

    with mock.patch.dict(sys.modules, {"bitsandbytes": real}):
        assert bitsandbytes_stub.real_bitsandbytes_available() is True
    with mock.patch.dict(sys.modules, {"bitsandbytes": stub_like}):
        assert bitsandbytes_stub.real_bitsandbytes_available() is False


@pytest.mark.parametrize("installed", [True, False])
def test_real_bitsandbytes_available_locates_without_importing(installed):
    """Detection must not import bitsandbytes: that pulls in torch, ~1s, on hosts whose
    whole reason for skipping GPU init is to avoid it."""
    import builtins

    from unsloth_zoo.stubs import bitsandbytes_stub

    # A real distribution produces a loader-bearing spec; a loaderless one is a namespace
    # package, which is not an install (see the namespace test below).
    spec = types.SimpleNamespace(loader = object()) if installed else None
    real_import = builtins.__import__

    def guarded_import(name, *a, **kw):
        # Catches a plain `import bitsandbytes` too, not just importlib.import_module.
        assert not name.startswith("bitsandbytes"), f"must not import {name}"
        return real_import(name, *a, **kw)

    saved = {k: v for k, v in sys.modules.items() if k.startswith("bitsandbytes")}
    for k in saved:
        del sys.modules[k]
    try:
        with mock.patch.object(
            bitsandbytes_stub.importlib.util, "find_spec", return_value = spec
        ) as find_spec, mock.patch.object(builtins, "__import__", guarded_import):
            assert bitsandbytes_stub.real_bitsandbytes_available() is installed
        find_spec.assert_called_once_with("bitsandbytes")
        assert not [k for k in sys.modules if k.startswith("bitsandbytes")], (
            "detection left bitsandbytes resident, so it imported rather than located"
        )
    finally:
        sys.modules.update(saved)


def test_a_namespace_bitsandbytes_directory_is_not_a_real_install(tmp_path):
    """A bare `bitsandbytes/` directory with no __init__.py -- what a half-removed install
    leaves on sys.path.

    find_spec answers with a loaderless namespace spec, so treating "found" as "installed"
    stands aside for a package that imports to nothing, leaving the caller with neither a
    wheel nor the stub.
    """
    from unsloth_zoo.stubs import bitsandbytes_stub

    (tmp_path / "bitsandbytes").mkdir()
    saved = {k: v for k, v in sys.modules.items() if k.startswith("bitsandbytes")}
    for k in saved:
        del sys.modules[k]
    # A regular package later on the path beats a namespace portion, so an installed
    # wheel on this runner would mask the case under test.
    saved_path = list(sys.path)
    sys.path[:] = [str(tmp_path)] + [p for p in sys.path if "site-packages" not in p]
    try:
        spec = bitsandbytes_stub.importlib.util.find_spec("bitsandbytes")
        assert spec is not None and spec.loader is None, "not a namespace package"
        assert bitsandbytes_stub.real_bitsandbytes_available() is False
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k.startswith("bitsandbytes")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _mlx_loader():
    """The MLX loader, or skip: importing it pulls in mlx.core, and this file also runs in
    the Linux upstream-regression lane, which installs no mlx."""
    pytest.importorskip("mlx")
    from unsloth_zoo.mlx import loader

    return loader


def test_bitsandbytes_is_stubbed_only_reports_the_unsloth_stub():
    loader = _mlx_loader()

    stubbed = types.ModuleType("bitsandbytes")
    stubbed.IS_UNSLOTH_STUB = True

    with mock.patch.dict(sys.modules, {"bitsandbytes": stubbed}):
        assert loader._bitsandbytes_is_stubbed() is True
    with mock.patch.dict(sys.modules, {"bitsandbytes": types.ModuleType("bitsandbytes")}):
        assert loader._bitsandbytes_is_stubbed() is False
    saved = sys.modules.pop("bitsandbytes", None)
    try:
        assert loader._bitsandbytes_is_stubbed() is False
    finally:
        if saved is not None:
            sys.modules["bitsandbytes"] = saved


@contextlib.contextmanager
def _bnb_modules(entries):
    """Run with exactly ``entries`` as the resident bitsandbytes modules."""
    loader = _mlx_loader()

    saved = {k: v for k, v in sys.modules.items() if k.startswith("bitsandbytes")}
    saved_real = loader._REAL_BITSANDBYTES_MODULES
    saved_meta = list(sys.meta_path)
    for k in saved:
        del sys.modules[k]
    sys.modules.update(entries)
    try:
        yield
    finally:
        for k in [k for k in sys.modules if k.startswith("bitsandbytes")]:
            del sys.modules[k]
        sys.modules.update(saved)
        loader._REAL_BITSANDBYTES_MODULES = saved_real
        sys.meta_path[:] = saved_meta


def test_lifting_the_bnb_stub_exposes_the_real_wheel_then_puts_the_stub_back():
    """The swap must lift the stub, cache what the block imported, and restore."""
    loader = _mlx_loader()

    stub = types.ModuleType("bitsandbytes")
    stub.IS_UNSLOTH_STUB = True
    real = types.ModuleType("bitsandbytes")
    real_ops = types.ModuleType("bitsandbytes._ops")

    class _BnbFinder:  # the swap filters meta_path by this class name
        pass

    finder = _BnbFinder()

    with _bnb_modules({"bitsandbytes": stub, "bitsandbytes.nn": types.ModuleType("bitsandbytes.nn")}):
        loader._REAL_BITSANDBYTES_MODULES = {}
        sys.meta_path.insert(0, finder)
        with loader._lifted_bitsandbytes_stub():
            assert "bitsandbytes" not in sys.modules, "the stub was not lifted"
            assert "bitsandbytes.nn" not in sys.modules, "a stub submodule survived"
            assert finder not in sys.meta_path, (
                "the stub's finder still serves bitsandbytes submodules, so the real "
                "wheel would come back half stubbed"
            )
            sys.modules["bitsandbytes"] = real          # stands in for the real import
            sys.modules["bitsandbytes._ops"] = real_ops  # ... and its operator module
        assert sys.modules["bitsandbytes"] is stub, "the stub was not put back"
        assert "bitsandbytes.nn" in sys.modules, "a stub submodule was not put back"
        assert finder in sys.meta_path, "the stub's finder was not put back"
        # Cached so a second call need not re-import it, which re-registers its torch
        # operators and raises. Caching the package root alone would leave the operator
        # submodules to be imported again.
        assert loader._REAL_BITSANDBYTES_MODULES["bitsandbytes"] is real
        assert loader._REAL_BITSANDBYTES_MODULES["bitsandbytes._ops"] is real_ops

        # Second call: the cached real modules are reinstated rather than re-imported.
        with loader._lifted_bitsandbytes_stub():
            assert sys.modules["bitsandbytes"] is real
            assert sys.modules["bitsandbytes._ops"] is real_ops
        assert sys.modules["bitsandbytes"] is stub


def test_lifting_the_bnb_stub_is_a_no_op_when_no_stub_is_resident():
    """Evicting a resident real wheel makes the next import re-register its torch
    operators, which raises."""
    loader = _mlx_loader()

    real = types.ModuleType("bitsandbytes")
    ops = types.ModuleType("bitsandbytes._ops")

    late_finder = types.SimpleNamespace()
    late_submodule = types.ModuleType("bitsandbytes.functional")

    with _bnb_modules({"bitsandbytes": real, "bitsandbytes._ops": ops}):
        before_meta = list(sys.meta_path)
        with loader._lifted_bitsandbytes_stub():
            assert sys.modules["bitsandbytes"] is real, "the real wheel was evicted"
            assert sys.modules["bitsandbytes._ops"] is ops
            assert sys.meta_path == before_meta, (
                "meta_path was disturbed with no stub to lift"
            )
            # A dequant runs for minutes; another thread can install a finder or finish a
            # submodule import meanwhile.
            sys.meta_path.insert(0, late_finder)
            sys.modules["bitsandbytes.functional"] = late_submodule
        assert sys.modules["bitsandbytes"] is real, "the real wheel was evicted on exit"
        assert sys.modules["bitsandbytes._ops"] is ops
        assert late_finder in sys.meta_path, (
            "a finder installed during the block was clobbered by a meta_path restore "
            "that had nothing to restore"
        )
        assert sys.modules.get("bitsandbytes.functional") is late_submodule, (
            "a submodule whose import completed during the block was evicted by a "
            "snapshot restore that had nothing to restore"
        )


def test_lifting_the_bnb_stub_keeps_a_wheel_first_imported_inside_the_block():
    """The cold path: detection never imports, so a dequant can be the first consumer and
    a restore that runs anyway drops the wheel the block just imported."""
    loader = _mlx_loader()

    real = types.ModuleType("bitsandbytes")
    ops = types.ModuleType("bitsandbytes._ops")

    with _bnb_modules({}):
        assert not [k for k in sys.modules if k.startswith("bitsandbytes")]
        with loader._lifted_bitsandbytes_stub():
            sys.modules["bitsandbytes"] = real      # stands in for the first real import
            sys.modules["bitsandbytes._ops"] = ops
        assert sys.modules.get("bitsandbytes") is real, (
            "the wheel imported inside the block was evicted on exit"
        )
        assert sys.modules.get("bitsandbytes._ops") is ops


def test_the_dequant_sees_the_real_wheel_not_the_stub(monkeypatch, tmp_path):
    """The dequant must run inside the lift, not merely have one available: otherwise it
    imports whatever is resident, which on a stubbed host is the stub."""
    import transformers

    loader = _mlx_loader()

    stub = types.ModuleType("bitsandbytes")
    stub.IS_UNSLOTH_STUB = True
    real = types.ModuleType("bitsandbytes")
    seen = {}

    class _FakeModel:
        config = types.SimpleNamespace()

        def dequantize(self):
            return self

        def save_pretrained(self, *a, **k):
            pass

    class _FakeModelLoader:
        @staticmethod
        def from_pretrained(*a, **k):
            # Its own key: the dequantization is this call, so a later tokenizer load
            # seeing the real module must not stand in for it.
            seen["model_load"] = sys.modules.get("bitsandbytes")
            return _FakeModel()

    class _FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(*a, **k):
            seen["tokenizer_load"] = sys.modules.get("bitsandbytes")
            return _FakeModel()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeModelLoader)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizerLoader)
    monkeypatch.setattr(loader.tempfile, "mkdtemp", lambda **k: str(tmp_path))

    with _bnb_modules({"bitsandbytes": stub}):
        # The lift reinstates cached real modules, standing in for the real import.
        loader._REAL_BITSANDBYTES_MODULES = {"bitsandbytes": real}
        loader._dequantize_bnb_to_tempdir(
            "some/repo", token = None, trust_remote_code = False,
        )
        assert seen["model_load"] is real, (
            "the dequant ran against the stub, so it never entered the lift"
        )
        assert sys.modules["bitsandbytes"] is stub, "the stub was not put back"


def test_lifting_the_bnb_stub_restores_after_the_block_raises():
    loader = _mlx_loader()

    stub = types.ModuleType("bitsandbytes")
    stub.IS_UNSLOTH_STUB = True

    with _bnb_modules({"bitsandbytes": stub}):
        loader._REAL_BITSANDBYTES_MODULES = {}
        with pytest.raises(ImportError):
            with loader._lifted_bitsandbytes_stub():
                raise ImportError("no real wheel here")
        assert sys.modules["bitsandbytes"] is stub, "the stub was lost on the error path"


def test_lifting_the_bnb_stub_keeps_a_finder_installed_while_the_block_ran():
    """The stubbed path must leave sys.meta_path alone apart from the stub's own finder.

    The block is a multi-GB dequant running for minutes, so restoring a whole snapshot
    taken at entry silently drops whatever another thread installed in between.
    """
    loader = _mlx_loader()

    stub = types.ModuleType("bitsandbytes")
    stub.IS_UNSLOTH_STUB = True

    class _BnbFinder:  # the swap filters meta_path by this class name
        pass

    stub_finder = _BnbFinder()
    late_finder = types.SimpleNamespace()

    with _bnb_modules({"bitsandbytes": stub}):
        loader._REAL_BITSANDBYTES_MODULES = {}
        sys.meta_path.insert(0, stub_finder)
        with loader._lifted_bitsandbytes_stub():
            assert stub_finder not in sys.meta_path, "the stub finder was not lifted"
            sys.meta_path.insert(0, late_finder)   # another thread, mid-dequant
        assert late_finder in sys.meta_path, "a concurrently installed finder was dropped"
        assert stub_finder in sys.meta_path, "the stub finder was not put back"


# ---------------------------------------------------------------------------
# 6. mlx_loader rejects full_finetuning against a pre-quantized repo.
#
#    Covers commit:
#      - 7d2bb95 fix(mlx): reject full_finetuning against pre-quantized
#                repos loudly
#
#    Without this guard, the CCE backward returns mx.zeros for quantized
#    weight grads, so the user "trains" but most of the model never
#    updates. The detection helper is _get_existing_mlx_quantization.
# ---------------------------------------------------------------------------

def test_get_existing_mlx_quantization_detects_both_keys():
    """The detection helper must recognise BOTH the 'quantization' (MLX
    native) and 'quantization_config' (HF style) keys. A regression that
    only checks one silently re-enables the full_finetuning-on-quantized
    foot-gun that 7d2bb95 closed.
    """
    # Import the helper without triggering the heavy mlx_loader import
    # chain on the GPU-free harness. We pull the function directly.
    # Layout was promoted from mlx_loader.py (flat) to mlx/loader.py
    # (subpackage) in e6d8f7f. Try both so the test survives the rename.
    import importlib.util
    import pathlib
    pkg_loc = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    )
    candidates = [pkg_loc / "mlx" / "loader.py", pkg_loc / "mlx_loader.py"]
    loader_path = next((c for c in candidates if c.is_file()), None)
    assert loader_path is not None, (
        f"Neither {candidates[0]} nor {candidates[1]} exists; the MLX "
        f"loader module was relocated again. Update this test's path "
        f"candidates."
    )
    src = loader_path.read_text()

    # The function must check BOTH key names; otherwise repos saved by
    # mlx-lm (key "quantization") OR by HF transformers ("quantization_config")
    # slip through the guard.
    assert "config_data.get(\"quantization\"" in src, (
        "_get_existing_mlx_quantization no longer checks 'quantization' "
        "key — regresses commit 7d2bb95."
    )
    assert "config_data.get(\"quantization_config\"" in src, (
        "_get_existing_mlx_quantization no longer checks "
        "'quantization_config' key — regresses commit 7d2bb95."
    )


# ---------------------------------------------------------------------------
# 7. target_modules='all-linear' must collect EVERY nn.Linear name.
#
#    Covers commit:
#      - 7f8b0ca fix(mlx): make target_modules='all-linear' actually mean
#                every nn.Linear
#
#    Pre-fix, "all-linear" was silently rewritten to None and collapsed to
#    the canonical 7-name list. For Qwen3.5 that dropped the GatedDelta
#    in_proj_* and out_proj from LoRA targeting entirely.
# ---------------------------------------------------------------------------

def test_collect_all_linear_target_names_finds_qkv_and_moe():
    """_collect_all_linear_target_names must discover fused-QKV names
    (qkv_proj), GatedDelta projections (in_proj_a, in_proj_b, in_proj_qkv,
    in_proj_z, out_proj), vision tower fused linears, and MoE routers /
    experts — not just the canonical 7. Walks a fake model whose
    named_modules emits the names we care about so we don't need real MLX.
    """
    # Both packages, not just mlx. The helper reaches _mlx_lora_type_specs(), which does
    # `from mlx_lm.models.switch_layers import ...` and `from mlx_lm.tuner.lora import ...`,
    # and _collect_all_linear_target_names wraps the whole walk in a blanket
    # `except Exception: return []` so that LoRA setup never raises. Guarding on mlx alone
    # therefore let this test RUN against an mlx with no usable mlx_lm beside it, where the
    # helper returns [] for a missing-dependency reason and the assertion below reports it
    # as `assert {...} <= set()` -- indistinguishable from the coverage regression this test
    # exists to catch, and pointing at the wrong file.
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    from unsloth_zoo.mlx_loader import _collect_all_linear_target_names, _mlx_lora_base_types
    import mlx.nn as nn

    # Prove the prerequisite BEFORE asserting on the output. An empty type tuple makes every
    # isinstance() below false, so the helper returns nothing no matter how correct it is;
    # that is a broken environment, not dropped targeting, and it must not be reported as one.
    try:
        base_types = _mlx_lora_base_types()
    except Exception as exc:  # noqa: BLE001 - mirrors the helper's own blanket catch
        pytest.skip(f"MLX LoRA base types unavailable ({exc!r}); the helper would return [] "
                    f"for a reason unrelated to all-linear targeting")
    # Both of these are degraded-environment conditions, not regressions, so they SKIP.
    # Failing here would redden CI for a partial or stubbed MLX that says nothing about
    # whether all-linear targeting is correct -- which is the bug this whole guard had.
    if not base_types:
        pytest.skip("_mlx_lora_base_types() resolved to an empty tuple, so nothing can match "
                    "the isinstance walk and the helper returns [] regardless of its logic")
    if not (isinstance(nn.Linear, type) and issubclass(nn.Linear, tuple(base_types))):
        pytest.skip(f"mlx.nn.Linear is not among the types the helper matches on "
                    f"({base_types}); a stand-in MLX left in sys.modules by another test "
                    f"cannot exercise the walk")

    class FakeQwen3p5:
        """Minimal model whose named_modules() exposes the leaves that
        triggered the pre-fix silent collapse. Real mlx.nn.Linear types
        are required because the helper's isinstance check uses them.
        """
        def named_modules(self):
            yield ("model.layers.0.self_attn.q_proj", nn.Linear(4, 4))
            yield ("model.layers.0.self_attn.k_proj", nn.Linear(4, 4))
            yield ("model.layers.0.self_attn.v_proj", nn.Linear(4, 4))
            yield ("model.layers.0.self_attn.o_proj", nn.Linear(4, 4))
            yield ("model.layers.0.mlp.gate_proj", nn.Linear(4, 4))
            yield ("model.layers.0.mlp.up_proj",   nn.Linear(4, 4))
            yield ("model.layers.0.mlp.down_proj", nn.Linear(4, 4))
            # GatedDelta projections — the exact 7f8b0ca regression class.
            yield ("model.layers.0.gated_delta.in_proj_qkv", nn.Linear(4, 4))
            yield ("model.layers.0.gated_delta.in_proj_z",   nn.Linear(4, 4))
            yield ("model.layers.0.gated_delta.out_proj",    nn.Linear(4, 4))
            # MoE router + expert — fused QKV — vision tower (numeric leaves
            # are skipped, the *suffix* name is what gets returned).
            yield ("model.layers.0.moe.router", nn.Linear(4, 4))
            yield ("model.layers.0.moe.experts.0.w1", nn.Linear(4, 4))
            yield ("vision_tower.layers.0.attn.qkv", nn.Linear(4, 4))

    names = set(_collect_all_linear_target_names(FakeQwen3p5()))
    canonical = {"q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"}
    # Canonical 7 still resolve. Spelled with a message because the bare form renders as
    # `assert {...} <= set()`, which says nothing about which of the two failure modes it is.
    assert canonical <= names, (
        f"all-linear lost canonical names {sorted(canonical - names)}; got {sorted(names)}. "
        f"An EMPTY result here means the walk matched nothing at all -- check the MLX stack "
        f"before suspecting the targeting logic."
    )
    # Plus the extras the pre-fix collapse dropped.
    extras = {"in_proj_qkv", "in_proj_z", "out_proj",
              "router", "w1", "qkv"}
    missing = extras - names
    assert not missing, (
        f"all-linear missed {sorted(missing)} — regresses commit 7f8b0ca; "
        "the silent-collapse-to-canonical-7 bug would skip these layers."
    )


# ---------------------------------------------------------------------------
# 8. patch_gated_delta routes training (state=None) through the efficient
#    custom-VJP path, not the kernel.
#
#    Covers commit:
#      - 46866ce fix(mlx): correct GatedDeltaNet VJP mask handling +
#                actually run it
#
#    Pre-fix patched_gated_delta_update fell through to gated_delta_kernel
#    on Metal (the default use_kernel=True branch), making the custom VJP
#    dead code. The fix unconditionally routes training calls
#    (state is None on entry) through gated_delta_ops_efficient.
# ---------------------------------------------------------------------------

def test_patch_gated_delta_routes_training_through_efficient_path():
    """Pin the routing predicate in patch_gated_delta. The patched
    function MUST call gated_delta_ops_efficient when state is None
    (training entry), even if use_kernel=True and mlx says metal is
    available. Pre-fix the kernel branch shadowed the custom VJP.
    """
    import importlib.util
    import pathlib
    pkg_loc = importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    src = (pathlib.Path(pkg_loc) / "gated_delta_vjp.py").read_text()

    # The training-call routing line is the regression-net.
    # The fix added `is_training_call = state is None` and then the
    # unconditional `if is_training_call: return gated_delta_ops_efficient(...)`
    # branch BEFORE the kernel branch. Both must be present.
    assert "is_training_call" in src, (
        "patch_gated_delta dropped the is_training_call gate; "
        "regresses commit 46866ce — custom VJP becomes dead code under "
        "use_kernel=True."
    )
    assert "gated_delta_ops_efficient" in src
    # And the training branch must come before the kernel fallthrough.
    idx_eff = src.find("if is_training_call:")
    idx_kernel = src.find("gated_delta_kernel(")
    assert idx_eff != -1 and idx_kernel != -1
    assert idx_eff < idx_kernel, (
        "The training-call branch must precede the gated_delta_kernel "
        "fallthrough so the custom VJP actually runs (commit 46866ce)."
    )
