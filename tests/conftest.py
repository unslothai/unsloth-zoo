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

"""pytest conftest: GPU-free harness + MLX simulation suite path.

Combines two pieces of test setup:

1. GPU-free harness for the CPU-only tests already in this directory
   (LoRA extractor shape parity, registration coverage, dtype helpers).
   ``unsloth_zoo.__init__`` calls ``device_type.get_device_type()`` at
   import time, which raises ``NotImplementedError`` on CI runners
   without CUDA / XPU / HIP visible. We pre-load the real
   ``unsloth_zoo.device_type`` under a temporarily-True
   ``torch.cuda.is_available()`` so the @cache permanently captures
   ``"cuda"`` and the package import chain succeeds. When a real
   accelerator IS available the pre-load is skipped and the real
   detection runs.

2. ``tests/`` is added to ``sys.path`` so the bundled MLX-on-torch
   simulation suite can ``from mlx_simulation import ...``. The shim is
   opt-in test infrastructure: it activates only when a test calls
   ``simulate_mlx_on_torch()`` and never touches production imports of
   ``unsloth_zoo``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


def _has_real_accelerator() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:
        pass
    try:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            return True
    except Exception:
        pass
    return False


def _preload_real_device_type(
    package: str = "unsloth_zoo",
    prereqs: tuple = ("utils",),
) -> bool:
    """Pre-load the REAL ``<package>.device_type`` module under a
    temporarily-mocked ``torch.cuda.is_available()`` so its
    ``DEVICE_TYPE = get_device_type()`` initialization succeeds without
    a real accelerator. Returns True on success; returns False if
    torch is not importable at all (the security-audit CI job runs
    tests/security/ without installing torch, and those tests don't
    need the preload), or if the target package isn't installed.

    Parameterised so the same harness works for both ``unsloth_zoo``
    (where ``utils.py`` defines ``Version`` before ``device_type``
    consumes it) and ``unsloth`` (which has no such prereq).
    """
    target = f"{package}.device_type"
    if target in sys.modules:
        return True
    pkg_spec = importlib.util.find_spec(package)
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return False
    pkg_path = pkg_spec.submodule_search_locations[0]

    import os

    skeleton_already = package in sys.modules
    if not skeleton_already:
        pkg_mod = types.ModuleType(package)
        pkg_mod.__path__ = [pkg_path]
        pkg_mod.__spec__ = pkg_spec
        pkg_mod.__package__ = package
        sys.modules[package] = pkg_mod

    try:
        for prereq in prereqs:
            full = f"{package}.{prereq}"
            if full in sys.modules:
                continue
            prereq_path = os.path.join(pkg_path, f"{prereq}.py")
            prereq_spec = importlib.util.spec_from_file_location(full, prereq_path)
            prereq_mod = importlib.util.module_from_spec(prereq_spec)
            sys.modules[full] = prereq_mod
            try:
                prereq_spec.loader.exec_module(prereq_mod)
            except ModuleNotFoundError as exc:
                # Tests that don't need torch (e.g. the tests/security
                # subtree which only exercises scanner regex tables and
                # subprocess invocations) shouldn't be blocked by the
                # device-type preload when torch isn't installed. Pop
                # the half-built modules and bail out gracefully.
                if "torch" in str(exc):
                    sys.modules.pop(full, None)
                    if not skeleton_already:
                        sys.modules.pop(package, None)
                    return False
                raise

        device_type_path = os.path.join(pkg_path, "device_type.py")
        dt_spec = importlib.util.spec_from_file_location(target, device_type_path)
        dt_mod = importlib.util.module_from_spec(dt_spec)
        sys.modules[target] = dt_mod

        import torch
        _orig_is_avail = torch.cuda.is_available
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]
        try:
            dt_spec.loader.exec_module(dt_mod)
        finally:
            torch.cuda.is_available = _orig_is_avail
    finally:
        if not skeleton_already:
            sys.modules.pop(package, None)

    return True


def _install_device_type_stub(name: str) -> None:
    """Last-resort stub when the real preload can't run (no torch / no
    package installed). Matches the surface ``unsloth`` and ``unsloth_zoo``
    consumers read at import time."""
    stub = types.ModuleType(name)
    stub.DEVICE_TYPE = "cuda"
    stub.DEVICE_TYPE_TORCH = "cuda"
    stub.DEVICE_COUNT = 1
    stub.ALLOW_PREQUANTIZED_MODELS = False
    stub.is_hip = lambda: False
    stub.get_device_type = lambda: "cuda"
    stub.get_device_count = lambda: 1
    stub.device_synchronize = lambda *a, **k: None
    stub.device_empty_cache = lambda *a, **k: None
    stub.device_is_bf16_supported = lambda *a, **k: False
    sys.modules[name] = stub


def _patch_torch_cuda_for_import() -> None:
    """Stub torch.cuda.* calls made at IMPORT time on CPU-only CI runners.

    Covers mem_get_info (used by temporary_patches/*), get_device_capability
    (compiler.py:87, loss_utils.py:39 -- gates cut_cross_entropy on Ampere+),
    and get_device_properties. Return (8, 0) so Ampere-gated imports proceed;
    the cut_cross_entropy import itself is try/except wrapped.
    """
    try:
        import torch  # type: ignore
        import torch.cuda.memory as _cuda_memory  # type: ignore
        _cuda_memory.mem_get_info = lambda *a, **k: (0, 80 * 1024 ** 3)
    except Exception:
        return
    try:
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)  # type: ignore[assignment]
    except Exception:
        pass
    try:
        class _StubDeviceProps:
            major = 8
            minor = 0
            total_memory = 80 * 1024 ** 3
            multi_processor_count = 108
            name = "stub"
        torch.cuda.get_device_properties = lambda *a, **k: _StubDeviceProps()  # type: ignore[assignment]
    except Exception:
        pass


if not _has_real_accelerator():
    if not _preload_real_device_type("unsloth_zoo", prereqs=("utils",)):
        _install_device_type_stub("unsloth_zoo.device_type")
    # NOTE: we deliberately do NOT stub ``unsloth.device_type`` here.
    # Doing so makes ``import unsloth`` succeed on CPU-only CI, which
    # then runs ``unsloth/_gpu_init.py:_patch_trl_trainer()`` and
    # rebinds ``trl.trainer.sft_trainer.SFTTrainer`` /
    # ``transformers.models.ministral.MinistralAttention`` to Unsloth's
    # compiled wrappers. ``inspect.getsource(...)`` on those classes
    # then returns the wrapper source, which masks upstream and causes
    # zoo's drift detectors (test_MinistralAttention_forward_signature,
    # test_unsloth_rl_trainer_*) to fail. The cost is that the
    # ``test_unsloth_trainer_exec_marker`` smoke test fails on CPU-only
    # runners; that failure exists on main too and tracks a separate
    # ``unsloth.device_type`` consumer that needs its own CPU fallback.
    _patch_torch_cuda_for_import()


_TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


# 3. Apply upstream-drift fixes (triton CompiledKernel attrs, vLLM rename,
#    peft transformers_weight_conversion shim, etc.) by triggering
#    ``import unsloth``. Fixes live on ``unsloth/import_fixes.py`` and run
#    at unsloth import time; zoo no longer carries a copy. Security-only
#    test suites without unsloth installed keep passing -- ImportError is
#    swallowed below.

def _apply_upstream_import_fixes_for_tests() -> None:
    # Let `import unsloth` succeed on a CPU-only CI runner. The flag is
    # honoured by unsloth's get_device_type (returns "cuda" sentinel) and
    # by PatchFastRL / _patch_trl_trainer (early-return so trl.SFTTrainer
    # stays pristine for downstream inspect.getsource drift detectors).
    # Production hosts with a real accelerator skip both branches.
    import os
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    try:
        import unsloth  # noqa: F401  # runs unsloth/import_fixes.py
    except Exception:
        # unsloth missing (security-only suites) or import failed; drift
        # detectors will surface any pathology the patches would mask.
        pass


_apply_upstream_import_fixes_for_tests()


# Xet health state isolation: the ladder persists per-machine verdicts next to the HF cache, so
# without this fixture a suite of SIMULATED failures would demote the real machine and later tests
# would start on HTTP and contradict their own assertions.

import pytest as _pytest


@_pytest.fixture(autouse = True)
def _isolate_xet_health_state(tmp_path_factory, monkeypatch):
    state_dir = tmp_path_factory.mktemp("xet_health_home")
    monkeypatch.setenv("HF_HOME", str(state_dir))
    try:
        from unsloth_zoo import hf_xet_health

        hf_xet_health.clear_xet_health()
        yield
        hf_xet_health.clear_xet_health()
    except Exception:
        # Module absent (partial checkout / security-only suite): nothing to isolate.
        yield


# grpo_accumulated_loss sets UNSLOTH_RETURN_HIDDEN_STATES=1 and only resets it on its way out, so
# a test that stops it part way (test_grpo_autotune_per_step raises _Stop from inside it) leaves the
# flag on for the rest of its xdist worker. Every patched causal LM forward then returns hidden
# states and loss=None, and test_gpt_oss_router_logits_capture failed on
# `'NoneType' object has no attribute 'backward'` whenever it followed one of those on a worker.
import os as _os

_HIDDEN_STATES_FLAG = "UNSLOTH_RETURN_HIDDEN_STATES"


@_pytest.fixture(autouse = True)
def _restore_hidden_states_flag():
    before = _os.environ.get(_HIDDEN_STATES_FLAG)
    yield
    if before is None:
        _os.environ.pop(_HIDDEN_STATES_FLAG, None)
    else:
        _os.environ[_HIDDEN_STATES_FLAG] = before


# UNSLOTH_COMPILE_DISABLE only switches off torch.compile, and since #1866 the fused losses and
# maybe_compile honour it, so a test of the compile path itself sees no compiled call and no
# fallback wrapper under it. Unsloth's Core zoo legs run this suite with the switch set, which is
# the only place these tests run. Tests of the compile machinery opt in with
# `pytestmark = pytest.mark.usefixtures("compile_enabled")`: the switch is cleared for the test and
# every flag the modules froze from it at import is turned back on, then restored.
# Modules copy the switch with `from ...common import UNSLOTH_COMPILE_DISABLE`, so every loaded
# unsloth_zoo module carrying one of these names is patched, not a hand-kept list.
_COMPILE_SWITCH_FLAGS = {
    "UNSLOTH_COMPILE_DISABLE": False,
    "UNSLOTH_COMPILE_DISABLE_PARTIAL": False,
    "_COMPILE": True,  # fused_losses.chunked_logprob keeps the inverse
}
_COMPILE_SWITCH_MODULES = (
    "unsloth_zoo.temporary_patches.common",
    "unsloth_zoo.temporary_patches.utils",
    "unsloth_zoo.compiler",
    "unsloth_zoo.fused_losses.cross_entropy_loss",
    "unsloth_zoo.fused_losses.chunked_logprob",
)


@_pytest.fixture
def compile_enabled(monkeypatch):
    import importlib
    import sys

    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    for name in _COMPILE_SWITCH_MODULES:
        importlib.import_module(name)
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "unsloth_zoo" or name.startswith("unsloth_zoo.")):
            continue
        for flag, enabled in _COMPILE_SWITCH_FLAGS.items():
            if isinstance(getattr(module, flag, None), bool):
                monkeypatch.setattr(module, flag, enabled)


# A test that swaps a module into sys.modules and does not swap it back breaks whatever
# imports that name later IN THE SAME PROCESS. Serially that is often invisible, because
# the victim happens to sort before the polluter and never sees it; under pytest-xdist a
# worker can take them the other way round and the victim fails for a reason that has
# nothing to do with it.
#
# That is not hypothetical. test_vllm_to_hf_conversion installed a bitsandbytes.functional
# carrying only dequantize_4bit and left it there, so every later
# `from bitsandbytes.functional import QuantState` -- which is what
# temporary_patches/moe_utils_bnb4bit.py does at import time -- failed with
#
#     cannot import name 'QuantState' from 'bitsandbytes.functional' (unknown location)
#
# Six tests in test_moe_bnb4bit_per_expert_conversions.py failed that way under -n 4 while
# passing serially, because m sorts before v.
#
# Narrow on purpose. It watches the names that have actually been swapped here rather than
# all of sys.modules, because reloading a module under test is a legitimate and common
# thing to do in this suite and a blanket check would flag all of it. The remedy is always
# the same and is already used elsewhere in these files: monkeypatch.setitem, which puts
# the original back.
_GUARDED_MODULES = ("bitsandbytes", "bitsandbytes.functional", "bitsandbytes.nn")

# Names that nothing in the installed environment provides. A top-level `moe_utils`
# only ever resolves because something put a compile-cache directory on sys.path.
# Leaving one behind makes the bare `from moe_utils import ...` in every generated
# MoE module resolve to it, which is invisible because that import is deliberately
# swallowed -- the module loads reporting success with its backend names undefined.
# That is how test_compiled_cache_collective.py's recovery test failed on main while
# passing when run alone.
#
# Flagged only when the copy came out of a pytest temp directory, and that
# qualification is load-bearing rather than caution. Compiling a MoE architecture
# imports moe_utils out of the real unsloth_compiled_cache as ordinary production
# behaviour, and that copy stays valid for the rest of the session; a rule without
# the qualification fails test_compiler_dynamic_exec.py for doing its job. A copy
# read from a tmp_path_factory directory is the opposite: it outlives the directory
# and is a different module from the one the next test means to import.
_GUARDED_BARE_MODULES = ("moe_utils",)


def _is_from_pytest_tmp(mod):
    import pathlib

    path = getattr(mod, "__file__", None)
    if not path:
        return False
    # Matched on the "pytest-of-<user>" element tmp_path_factory always creates,
    # rather than on a basetemp looked up from the config: --basetemp overrides it,
    # xdist gives each worker its own, and a stale entry can outlive the fixture
    # that made it. The directory name is the stable part.
    return any(
        part.startswith("pytest-of-") for part in pathlib.Path(path).parts
    )
_MODULE_SNAPSHOT_KEY = "_unsloth_guarded_module_snapshot"


def _is_module_stub(mod):
    """A real module has a file behind it; these substitutes are bare ModuleType.

    That is also why the resulting ImportError says "(unknown location)".
    """
    return mod is not None and getattr(mod, "__file__", None) is None


def _is_unsloth_stub(mod):
    """unsloth_zoo installs its own bitsandbytes stub where no real wheel exists, which
    is every Apple Silicon machine (stubs/bitsandbytes_stub.py). The finder mints
    submodules on first import, so the first test to import anything that touches
    bitsandbytes.nn looks like a polluter while doing nothing wrong: the stub is
    process wide, deliberate and idempotent, and putting it "back" would mean deleting
    a module the rest of the session expects. Every stub module carries the marker,
    finder-minted ones included.
    """
    return bool(getattr(mod, "IS_UNSLOTH_STUB", False))


def pytest_runtest_setup(item):
    import sys as _sys

    setattr(item, _MODULE_SNAPSHOT_KEY, {
        n: _sys.modules.get(n)
        for n in _GUARDED_MODULES + _GUARDED_BARE_MODULES
    })


@_pytest.hookimpl(trylast = True)
def pytest_runtest_teardown(item, nextitem):
    """Checked here rather than in an autouse fixture, and that is not a style choice.

    Fixtures tear down in reverse order of setup, and _isolate_xet_health_state above
    requests monkeypatch, which pulls monkeypatch's setup earlier than any fixture
    defined after it. A fixture-based check therefore ran BEFORE the test's own
    monkeypatch had put sys.modules back, and reported every correct test as a leak.
    A trylast teardown hook runs after fixture finalisation, which is the point.
    """
    import sys as _sys

    before = getattr(item, _MODULE_SNAPSHOT_KEY, None)
    if before is None:
        return
    leaked = []
    for name, was in before.items():
        now = _sys.modules.get(name)
        if now is was:
            continue
        if name in _GUARDED_BARE_MODULES:
            # See _GUARDED_BARE_MODULES: for these the import itself can be the
            # swap, so the exemption below does not apply. Narrowed to the copies
            # that go stale, which are the ones a later test can be misled by.
            if _is_from_pytest_tmp(now):
                leaked.append(name)
            continue
        if was is None and (not _is_module_stub(now) or _is_unsloth_stub(now)):
            # Nothing was there and the test imported the real thing, or the zoo's own
            # bitsandbytes stub. Either is an import, not a swap, and flagging it would
            # fail every test that touches the package.
            continue
        leaked.append(name)
    if leaked:
        # Naming the file is the whole diagnosis for the bare names: whether the copy
        # is a leak or the compile folder doing its job is a question about where it
        # was read from, and without this the report is the same either way.
        where = ", ".join(
            f"{n} from {getattr(_sys.modules.get(n), '__file__', None)!r}"
            for n in leaked
        )
        raise AssertionError(
            f"{item.nodeid} replaced {leaked} in sys.modules and did not put it back "
            f"({where}), "
            f"so every later test in this process imports the substitute. Use "
            f"monkeypatch.setitem(sys.modules, ...), which restores on teardown, or"
            f"save and restore the entry by hand where the import itself installs it."
        )


# Where real mlx is installed, it and the torch-backed shim would share one sys.modules,
# so a session's results depended on which test modules it collected
# (unslothai/unsloth#8138). Each test module runs under the mlx modules it was imported
# with, as in a process of its own, and the session's come back after it. Inert without
# real mlx.
_PLATFORM_ATTRS = ("system", "machine", "_orig_system_for_mlx_shim", "_orig_machine_for_mlx_shim")
_MISSING = object()


def _in_mlx_world(name):
    top, _, rest = name.partition(".")
    if top in ("mlx", "mlx_lm", "mlx_vlm", "mlx_simulation") or name == "unsloth_zoo.gated_delta_vjp":
        return True
    return top in ("unsloth", "unsloth_zoo") and "mlx" in rest


def _is_shim(obj):
    if not isinstance(obj, types.ModuleType):
        return type(obj).__module__.startswith("mlx_simulation.")
    return "mlx_simulation" in f"{obj.__name__} {getattr(obj, '__file__', '') or ''}"


class _RealMlxKeeper:
    """Hands back the real-mlx modules a world hid instead of importing them again: a
    second mlx.core import aborts the interpreter (nanobind refuses to register its
    types twice), and a second unsloth_zoo.mlx.utils patches mlx_vlm classes the first
    one already patched. Never under the shim, which gets copies of its own."""

    def __init__(self):
        self.modules = {}

    def find_spec(self, name, path = None, target = None):
        if name not in self.modules or _is_shim(sys.modules.get("mlx.core")):
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return self.modules[spec.name][0]

    def exec_module(self, module):
        module.__spec__ = self.modules[module.__name__][1]


class _MlxModuleWorlds:
    def __init__(self):
        self.keeper = _RealMlxKeeper()
        sys.meta_path.insert(0, self.keeper)
        self.base = self._capture()
        self.worlds = {}
        self.active = None

    def _capture(self):
        import platform

        return {
            "modules": {n: m for n, m in sys.modules.items() if _in_mlx_world(n)},
            "finders": [f for f in sys.meta_path if _is_shim(f)],
            "platform": {a: getattr(platform, a, _MISSING) for a in _PLATFORM_ATTRS},
        }

    def _install(self, world):
        import platform

        if not _is_shim(sys.modules.get("mlx.core")):
            for name, module in list(sys.modules.items()):
                if _in_mlx_world(name) and module is not None and not _is_shim(module):
                    self.keeper.modules.setdefault(name, (module, module.__spec__))
        current = [n for n in sys.modules if _in_mlx_world(n)]
        for name in current:
            del sys.modules[name]
        sys.modules.update(world["modules"])
        simulated = _is_shim(sys.modules.get("mlx.core"))
        # Parent attributes too, or `import a.b as m` and `from a.b import x` see two copies.
        for name in sorted(set(current) | set(world["modules"]), key = lambda n: n.count(".")):
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if name in sys.modules:
                setattr(parent, child, sys.modules[name])
            elif name in self.keeper.modules and not simulated:
                setattr(parent, child, self.keeper.modules[name][0])
            elif child in vars(parent):
                delattr(parent, child)
        sys.meta_path[:] = world["finders"] + [f for f in sys.meta_path if not _is_shim(f)]
        for attr, value in world["platform"].items():
            if value is not _MISSING:
                setattr(platform, attr, value)
            elif hasattr(platform, attr):
                delattr(platform, attr)

    def _world_for(self, module, world):
        """A module that never installs the shim also sees every real mlx, mlx_lm and mlx_vlm
        module loaded so far: zoo caches what it finds in sys.modules (_resolved_bindings),
        so hiding them there decides differently from the real session."""
        try:
            installs_shim = "simulate_mlx_on_torch(" in module.path.read_text(encoding = "utf-8")
        except (AttributeError, OSError):
            installs_shim = True
        if installs_shim:
            return world
        shared = {
            name: kept for name, (kept, _) in self.keeper.modules.items()
            if name.partition(".")[0] in ("mlx", "mlx_lm", "mlx_vlm")
        }
        return {**world, "modules": {**shared, **world["modules"]}}

    @_pytest.hookimpl(hookwrapper = True)
    def pytest_make_collect_report(self, collector):
        if not isinstance(collector, _pytest.Module):
            yield
            return
        self._install(self._world_for(collector, self.base))
        try:
            yield
        finally:
            self.worlds[collector.nodeid] = self._capture()
            self._install(self.base)

    @_pytest.hookimpl(tryfirst = True)
    def pytest_runtest_setup(self, item):
        module = item.getparent(_pytest.Module)
        key = module.nodeid if module is not None else None
        if key != self.active:
            self._install(self._world_for(module, self.worlds.get(key, self.base)))
            self.active = key

    @_pytest.hookimpl(trylast = True)
    def pytest_runtest_teardown(self, item, nextitem):
        # trylast: after module-scoped fixtures are finalised (see the leak guard above).
        if nextitem is None or nextitem.getparent(_pytest.Module) is not item.getparent(_pytest.Module):
            # Kept for a later visit (interleaved node IDs): a session fixture that installed
            # the shim here will not run again.
            if self.active is not None:
                self.worlds[self.active] = self._capture()
            self._install(self.base)
            self.active = None


def pytest_configure(config):
    if importlib.util.find_spec("mlx") is not None:
        config.pluginmanager.register(_MlxModuleWorlds(), "unsloth-mlx-module-worlds")
