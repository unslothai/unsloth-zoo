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

"""Regression guards for upstream-pinned MLX / Apple-Silicon / accelerator symbols, each citing its zoo commit; run on
Linux+CUDA via the tests/mlx_simulation shim or natively on Apple Silicon; CPU-runnable in CI"""

from __future__ import annotations

import contextlib
import sys
import types
from unittest import mock

import pytest
import torch


# Pins the partial torch.xpu build case (is_available True, symbol missing) for 35dc451, e08c1df, 2564f39 and d631837;
# happy path is in test_backend_device_helpers.py.

def test_xpu_partial_build_all_three_helpers_silent_no_op():
    """The three device_type helpers must no-op, not raise AttributeError, on a torch.xpu lacking synchronize /
    empty_cache / is_bf16_supported; these partial builds crashed GGUF merge and VLM mmproj export (e08c1df, 35dc451,
    d631837)."""
    from unsloth_zoo import device_type as dt

    class PartialXpu:
        """Mimics the upstream IPEX dev build where torch.xpu.is_available is True but synchronize / empty_cache /
        is_bf16_supported are missing; pre-fix this raised AttributeError mid-GGUF-export."""
        def is_available(self):
            return True

    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False

    with mock.patch.object(dt, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", PartialXpu(), create=True):
        dt.device_synchronize()
        dt.device_empty_cache()
        assert dt.device_is_bf16_supported() is False


# _active_merge_device() takes no args and cascades cuda, xpu, mps, cpu; pre-fix _active_merge_device(W) silently
# dropped MPS and leaked W.device.index across families (fd58aa1, 70b93ad).

def test_active_merge_device_mps_branch_pinned():
    """_active_merge_device() must return 'mps' on Apple Silicon with no cuda or xpu; routing through DEVICE_TYPE_TORCH broke the MLX on-host LoRA merge."""
    from unsloth_zoo.saving_utils import _active_merge_device

    _active_merge_device.cache_clear()
    try:
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


# Pre-fix the five MoE expert helpers fell back to CPU on XPU via hardcoded .to('cuda', ...); 2564f39 introduced the routing and fd58aa1 made the helper no-arg.

def test_moe_expert_merges_call_active_merge_device():
    """Five MoE expert merge helpers must route .to() through _active_merge_device(); hardcoded cuda or
    DEVICE_TYPE_TORCH drops MPS/XPU (2564f39); after unsloth-zoo 647 gate/up delegate to
    _merge_moe_gate_or_up_expert, so the check also follows the sibling _merge_moe_* helpers they forward to."""
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
        """Returns the entry point's source plus any sibling _merge_moe_* helper it forwards to; one level is enough
        since zoo never chains wrapper to wrapper to helper."""
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


# Deprecated mx.metal.set_memory_limit / .set_cache_limit warn every training run; the MLX shim exposes both forms, so
# the test pins the trainer source (70b93ad).

def test_mlx_trainer_uses_modern_memory_apis_only():
    """unsloth_zoo.mlx.trainer must use mx.set_memory_limit / set_cache_limit / set_wired_limit; the deprecated
    mx.metal.set_* forms bring back the per-run warning fixed in 70b93ad."""
    import importlib.util
    import pathlib

    pkg_root = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    )
    # MLX trainer moved from flat mlx_trainer.py to mlx/trainer.py in e6d8f7f; accept either layout so the test survives the rename.
    candidates = [pkg_root / "mlx" / "trainer.py", pkg_root / "mlx_trainer.py"]
    mlx_trainer_path = next((c for c in candidates if c.is_file()), None)
    assert mlx_trainer_path is not None, (
        f"Neither {candidates[0]} nor {candidates[1]} exists; the MLX "
        f"trainer module was relocated again. Update this test's path "
        f"candidates."
    )
    src = mlx_trainer_path.read_text()

    assert "mx.metal.set_memory_limit" not in src, (
        "Deprecated mx.metal.set_memory_limit call resurfaced; "
        "regresses commit 70b93ad."
    )
    assert "mx.metal.set_cache_limit" not in src, (
        "Deprecated mx.metal.set_cache_limit call resurfaced; "
        "regresses commit 70b93ad."
    )

    for modern in ("mx.set_memory_limit", "mx.set_cache_limit", "mx.set_wired_limit"):
        assert modern in src, f"Expected modern API {modern} missing from {mlx_trainer_path.name}"


# Commit 2053539 repaired Apple Silicon stub injection: gate was inverted, install_*_stub name was wrong, and
# _Noop.__call__ silently returned None instead of raising.

def test_apple_silicon_stub_injection_entrypoints_pinned():
    """Commit 2053539 sub-bugs (a) and (b): init must gate stub injection on 'if _SKIP_GPU_INIT:' (not the negated form)
    and call inject_into_sys_modules, not install_*_stub."""
    import importlib.util
    import pathlib

    init_path = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    ) / "__init__.py"
    src = init_path.read_text()

    assert "inject_into_sys_modules" in src, (
        "Stub injection entry point inject_into_sys_modules vanished from "
        "unsloth_zoo/__init__.py — regresses commit 2053539 sub-bug (b)."
    )
    # Pre-fix names that must NOT come back.
    assert "install_triton_stub" not in src
    assert "install_bitsandbytes_stub" not in src

    # Sub-bug (a): the injection block must sit under positive 'if _SKIP_GPU_INIT:', not the negated 'if not _SKIP_GPU_INIT:' gate.
    assert "if _SKIP_GPU_INIT:" in src, (
        "Apple-Silicon stub-injection gate flipped — regresses commit "
        "2053539 sub-bug (a)."
    )


def test_stub_noop_call_raises_not_returns_none():
    """Commit 2053539 sub-bug (c): _Noop.__call__ must raise NotImplementedError, not return None, so a stray bnb call
    crashes loudly; __bool__ and hasattr must still work."""
    from unsloth_zoo.stubs import triton_stub, bitsandbytes_stub

    for mod in (triton_stub, bitsandbytes_stub):
        noop = mod._Noop("test.symbol")
        with pytest.raises(NotImplementedError, match="test.symbol"):
            noop()
        assert bool(noop) is False
        sub = noop.some_attr
        assert sub is not noop
        with pytest.raises(NotImplementedError, match="test.symbol.some_attr"):
            sub()


# Reads sys.modules, not imports: both stubs seed sys.modules, so importing would surface the regression as an ImportError instead of a wrong verdict.
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
    """Runs the real gate in a fresh interpreter: shadowing a real bitsandbytes makes bnb-quantized checkpoints
    unloadable; triton is always stubbed (no macOS wheel)."""
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
    # No skip on failure: the gate runs during import unsloth_zoo, so a crash in the code under test and a broken environment look the same.
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
    """Detection must not import bitsandbytes: importing it pulls in torch, on hosts that skip GPU init precisely to avoid that."""
    import builtins

    from unsloth_zoo.stubs import bitsandbytes_stub

    # A real distribution has a loader-bearing spec; a loaderless spec means a namespace package, which is not an install.
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
    """A bare bitsandbytes/ dir with no __init__.py gives a loaderless namespace spec, not an install; treating it as found leaves neither wheel nor stub."""
    from unsloth_zoo.stubs import bitsandbytes_stub

    (tmp_path / "bitsandbytes").mkdir()
    saved = {k: v for k, v in sys.modules.items() if k.startswith("bitsandbytes")}
    for k in saved:
        del sys.modules[k]
    # A regular package later on sys.path beats a namespace portion, so an installed wheel on this runner would mask the case under test.
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
    """Returns the MLX loader or skips: importing it pulls in mlx.core, and this file also runs in the Linux upstream lane, which installs no mlx."""
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
        # Cache every bitsandbytes submodule, not just the root: re-importing one re-registers its torch operators and raises.
        assert loader._REAL_BITSANDBYTES_MODULES["bitsandbytes"] is real
        assert loader._REAL_BITSANDBYTES_MODULES["bitsandbytes._ops"] is real_ops

        with loader._lifted_bitsandbytes_stub():
            assert sys.modules["bitsandbytes"] is real
            assert sys.modules["bitsandbytes._ops"] is real_ops
        assert sys.modules["bitsandbytes"] is stub


def test_lifting_the_bnb_stub_is_a_no_op_when_no_stub_is_resident():
    """Evicting a resident real bitsandbytes wheel makes the next import re-register its torch operators, which raises."""
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
            # A dequant runs for minutes, so another thread may install a finder or finish a submodule import meanwhile; the lift must not drop those.
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
    """Cold path: detection never imports, so a dequant can be the first consumer; a restore that runs anyway would drop the wheel the block just imported."""
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
    """The dequant must run inside the lift, not just have one available; otherwise it imports whatever is resident, which on a stubbed host is the stub."""
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
            # Dequantization gets its own key, since it is this call; a later tokenizer load seeing the real module must not stand in for it.
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
    """Leave sys.meta_path alone except the stub's own finder: restoring an entry snapshot would silently drop finders other threads install during the dequant."""
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


# mlx_loader rejects full_finetuning on pre-quantized repos (7d2bb95): without it the CCE backward returns mx.zeros for
# quantized weight grads, so most of the model never updates.

def test_get_existing_mlx_quantization_detects_both_keys():
    """Must detect both 'quantization' (MLX native) and 'quantization_config' (HF style) keys; checking only one
    re-enables the full_finetuning foot-gun closed by 7d2bb95."""
    # Reads the file rather than importing it, to avoid the heavy mlx_loader import chain; the layout moved to
    # mlx/loader.py in e6d8f7f, so both paths are tried.
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

    # The function must check both key names, or repos saved by mlx-lm ('quantization') or HF transformers ('quantization_config') slip past the guard.
    assert "config_data.get(\"quantization\"" in src, (
        "_get_existing_mlx_quantization no longer checks 'quantization' "
        "key — regresses commit 7d2bb95."
    )
    assert "config_data.get(\"quantization_config\"" in src, (
        "_get_existing_mlx_quantization no longer checks "
        "'quantization_config' key — regresses commit 7d2bb95."
    )


# target_modules='all-linear' must collect every nn.Linear name (7f8b0ca); pre-fix it was rewritten to None, collapsing
# to the canonical 7 and dropping GatedDelta in_proj_* and out_proj for Qwen3.5.

def test_collect_all_linear_target_names_finds_qkv_and_moe():
    """_collect_all_linear_target_names must find fused-QKV, GatedDelta projections, vision tower and MoE router/expert
    linears, not just the canonical 7; walks a fake model's named_modules()."""
    # Needs both mlx and mlx_lm: the helper's blanket except returns [] when mlx_lm is missing, which would read as a
    # coverage regression instead of a missing dependency.
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    from unsloth_zoo.mlx_loader import _collect_all_linear_target_names, _is_mlx_lora_base
    import mlx.nn as nn

    # If a plain Linear is not a LoRA base the helper returns [] regardless: a broken env, not a regression.
    try:
        linear_is_base = _is_mlx_lora_base(nn.Linear(4, 4))
    except Exception as exc:  # noqa: BLE001 - mirrors the helper's own blanket catch
        pytest.skip(f"MLX LoRA base types unavailable ({exc!r}); the helper would return [] "
                    f"for a reason unrelated to all-linear targeting")
    # Skip, not fail: a stubbed MLX says nothing about all-linear targeting (the bug this guard had).
    if not linear_is_base:
        pytest.skip("mlx.nn.Linear is not a LoRA base here; a stand-in MLX left in "
                    "sys.modules by another test cannot exercise the walk")

    class FakeQwen3p5:
        """Minimal model whose named_modules() exposes the leaves that triggered the pre-fix silent collapse; real mlx.nn.Linear is needed for the isinstance check."""
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
            # MoE router and expert, fused QKV and vision tower leaves; numeric path segments are skipped and the suffix name is what gets returned.
            yield ("model.layers.0.moe.router", nn.Linear(4, 4))
            yield ("model.layers.0.moe.experts.0.w1", nn.Linear(4, 4))
            yield ("vision_tower.layers.0.attn.qkv", nn.Linear(4, 4))

    names = set(_collect_all_linear_target_names(FakeQwen3p5()))
    canonical = {"q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"}
    # Canonical 7 must still resolve; an empty result means the walk matched nothing, so check the MLX stack before the targeting logic.
    assert canonical <= names, (
        f"all-linear lost canonical names {sorted(canonical - names)}; got {sorted(names)}. "
        f"An EMPTY result here means the walk matched nothing at all -- check the MLX stack "
        f"before suspecting the targeting logic."
    )
    extras = {"in_proj_qkv", "in_proj_z", "out_proj",
              "router", "w1", "qkv"}
    missing = extras - names
    assert not missing, (
        f"all-linear missed {sorted(missing)} — regresses commit 7f8b0ca; "
        "the silent-collapse-to-canonical-7 bug would skip these layers."
    )


# Covers 46866ce: pre-fix patched_gated_delta_update fell through to gated_delta_kernel on Metal (default use_kernel=True), leaving the custom VJP dead code.

def test_patch_gated_delta_routes_training_through_efficient_path():
    """patch_gated_delta must call gated_delta_ops_efficient when state is None (training entry), even with
    use_kernel=True on Metal; pre-fix the kernel branch shadowed the custom VJP."""
    import importlib.util
    import pathlib
    pkg_loc = importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    src = (pathlib.Path(pkg_loc) / "gated_delta_vjp.py").read_text()

    # The fix added is_training_call = state is None and an unconditional early return to gated_delta_ops_efficient
    # before the kernel branch; both must be present.
    assert "is_training_call" in src, (
        "patch_gated_delta dropped the is_training_call gate; "
        "regresses commit 46866ce — custom VJP becomes dead code under "
        "use_kernel=True."
    )
    assert "gated_delta_ops_efficient" in src
    idx_eff = src.find("if is_training_call:")
    idx_kernel = src.find("gated_delta_kernel(")
    assert idx_eff != -1 and idx_kernel != -1
    assert idx_eff < idx_kernel, (
        "The training-call branch must precede the gated_delta_kernel "
        "fallthrough so the custom VJP actually runs (commit 46866ce)."
    )
