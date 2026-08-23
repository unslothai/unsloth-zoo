# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Make the bundled ``flash-linear-attention`` (fla) kernels importable as
top-level ``fla`` so users do not need ``pip install flash-linear-attention``.

Qwen3.5 / Qwen3.6 / Qwen3-Next gated-deltanet models use fla's Triton kernels
when ``is_flash_linear_attention_available()`` is true, else a several-times
slower pure-PyTorch path. This registers the pruned ``unsloth_zoo/_vendored/fla``
snapshot into ``sys.modules`` as a real, walkable ``fla`` and reports availability.

Precedence / escape hatches:
  * ``UNSLOTH_DISABLE_VENDORED_FLA=1`` -> never inject the vendored copy. With no
    other fla present this keeps the pure-torch path; a separately installed fla is
    left untouched and still used (the flag scopes to the vendored injection only).
  * Version-aware auto-detection: a user-installed ``fla`` that is strictly newer
    than the vendored snapshot is used instead (a newer upstream supersedes ours);
    an equal or older install is shadowed by the vendored kernels, which carry
    post-0.5.1 backports. ``UNSLOTH_FORCE_VENDORED_FLA=1`` forces the vendored copy
    even over a newer install (rarely needed now that selection is automatic).
  * Only injects when torch >= 2.7, triton >= 3.3 and CUDA are available (the
    requirements of the vendored fla-core 0.5.1 kernels); otherwise the
    pure-torch fallback is left untouched.
  * ``UNSLOTH_DISABLE_HOPPER_FLA_BWD=1`` -> on Hopper with Triton in
    [3.4.0, 3.7.1), disable fla entirely (vendored *and* installed) and force the
    pure-torch gated-delta path. This is a correctness switch, not a source
    preference, so it is checked before the two flags above. The vendored
    chunk_bwd_dqkwg already steps around the miscompiled BK=64 tile (fla #640), so
    this is only for users who want the belt-and-braces fallback.
"""

__all__ = [
    "patch_vendor_fla",
    "fla_unavailable_reason",
]

import os
import sys
import functools
import inspect
import threading
import importlib
import importlib.util

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
    logger,
)

# Marker set on the vendored top-level module so we can tell our own injection
# apart from a user-installed fla.
_VENDORED_MARK = "_UNSLOTH_VENDORED_FLA"

# Eagerly registered so compile_fla_no_autotune's walk_packages and transformers'
# `from fla... import ...` resolve to the vendored tree. Importing gated_delta_rule
# transitively pulls fla.ops.common/cp/utils and fla.utils.
_EXPORT_SUBMODULES = ("fla.modules", "fla.ops", "fla.ops.gated_delta_rule")

# Gated-deltanet modeling modules that bind the fla symbols as module globals at
# import time (and set them to None when fla was unavailable).
_REPAIR_MODELING = ("qwen3_5", "qwen3_5_moe", "qwen3_next")

# Models whose fla imports are fully covered by the vendored exports. olmo_hybrid
# also imports ShortConvolution, which is not vendored, so its probe must answer
# False (keep the pure-torch path) or its modeling module crashes on import.
_VENDOR_COVERED_MODELS = frozenset(_REPAIR_MODELING)

# Every gated-deltanet consumer, not just the vendor-covered ones. Used by the
# UNSLOTH_DISABLE_HOPPER_FLA_BWD path and the purge path below: the extra models
# never bind the *vendored* kernels, but they do bind a user-installed fla's, which
# carries the same #640 miscompile, so they must be unbound too.
#
# olmo_hybrid (transformers >= 5.3) is the only one: it imports
# chunk_gated_delta_rule alongside ShortConvolution. Kimi Linear deliberately is
# NOT here -- transformers ships no `kimi_linear` model (only `kimi_k25`), the
# weights run through trust_remote_code as `transformers_modules...modeling_kimi`,
# and that code calls fla's KDA ops (chunk_kda / fused_recurrent_kda), which have
# their own kernels and never reach chunk_bwd_dqkwg. Listing it would be a name
# that can never resolve.
_GATED_DELTA_MODELING = _REPAIR_MODELING + ("olmo_hybrid",)

# The gated-deltanet consumers the vendored snapshot cannot serve, so
# _repair_already_imported_modeling can never rebind them onto the fixed kernels.
# If one of them was imported while a user-installed fla was live and we then purge
# that install, its module global still points at the unpatched kernel.
_UNCOVERED_GATED_DELTA = tuple(
    pkg for pkg in _GATED_DELTA_MODELING if pkg not in _VENDOR_COVERED_MODELS
)

# Minimum versions declared by fla-core 0.5.1.
_MIN_TORCH = "2.7"
_MIN_TRITON = "3.3"
# The version of the bundled fla-core snapshot. A user-installed fla is used
# instead of the vendored one only when it is strictly newer than this; an equal
# or older install is shadowed by the vendored kernels, which carry post-0.5.1
# correctness backports (Blackwell / Hopper). Kept in sync with
# unsloth_zoo/_vendored/fla/__init__.py (guarded by test_vendored_tree_layout).
_VENDORED_FLA_VERSION = "0.5.1"


def _flag(name):
    return os.environ.get(name, "0") == "1"


def _restore_env(name, previous):
    """Restore an env var to a snapshotted value (``None`` means it was unset)."""
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _vendored_fla_dir():
    # This file lives at unsloth_zoo/temporary_patches/fla_vendor.py
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(here)
    return os.path.join(pkg_root, "_vendored", "fla")


def _version_at_least(value, minimum):
    try:
        from packaging import version
        # Compare base versions so dev/nightly/pre-release builds still satisfy
        # the minimum: version.parse orders 2.7.0.dev... / 3.3.0a0 *below* the
        # release, which would wrongly reject a valid 2.7 nightly.
        parsed = version.parse(str(value).split("+")[0])
        return version.parse(parsed.base_version) >= version.parse(minimum)
    except Exception:
        return False


def _version_strictly_after(value, threshold):
    """True if ``value`` parses to a release strictly greater than ``threshold``.
    Base-version comparison, so a dev/nightly of the same release (e.g. 0.5.1.devN)
    is not counted as newer than 0.5.1."""
    try:
        from packaging import version
        parsed = version.parse(str(value).split("+")[0])
        return version.parse(parsed.base_version) > version.parse(threshold)
    except Exception:
        return False


def _hopper_dqkwg_suspect(torch_mod, triton_mod):
    """True on Hopper with triton in [3.4.0, 3.7.1), the range in which fla's gated
    ``chunk_bwd_dqkwg`` is miscompiled (fla #640).

    This no longer gates injection. The vendored kernel steps the block width away
    from the miscompiled BK=64 tile (see ops/common/chunk_o.py), so the vendored
    tree is safe to use on these hosts and does so by default. The predicate now
    answers two narrower questions: whether to prefer the vendored copy over a
    user-installed fla (only ours carries the tile fix), and whether
    ``UNSLOTH_DISABLE_HOPPER_FLA_BWD=1`` should force the pure-torch path.
    Mirrors upstream's exact constants (full version parse, not base_version).

    Every visible CUDA device is probed, not just device 0: on a mixed host a
    model can be placed on a nonzero Hopper card (e.g. cuda:0 Ada, cuda:1 H100),
    and a device-0-only check would report that setup as safe. If any visible GPU
    would hit the bug we answer True for the whole process.
    """
    try:
        from packaging import version
        v = version.parse(str(triton_mod.__version__).split("+")[0])
        if not (version.parse("3.4.0") <= v < version.parse("3.7.1")):
            return False
        # The Hopper miscompile is NVIDIA-specific. On a ROCm build a card can
        # report capability major 9 (e.g. AMD Instinct) without being Hopper, so
        # only the bare major==9 signal is gated on a CUDA (non-HIP) build; a name
        # that literally says "NVIDIA H" is unambiguous either way.
        is_nvidia = getattr(getattr(torch_mod, "version", None), "hip", None) is None
        try:
            count = int(torch_mod.cuda.device_count())
        except Exception:
            count = 0
        for i in range(count):
            try:
                name = torch_mod.cuda.get_device_name(i)
            except Exception:
                name = ""
            try:
                major = torch_mod.cuda.get_device_capability(i)[0]
            except Exception:
                major = -1
            if "NVIDIA H" in name or (is_nvidia and major == 9):
                return True
        return False
    except Exception:
        return False


def _torch_triton_cuda_supported():
    """The vendored fla-core 0.5.1 kernels need torch >= 2.7, triton >= 3.3, CUDA."""
    # The snapshot uses runtime-evaluated PEP 604 annotations (e.g. `int | None`
    # in fla/utils/_device.py), which raise at import on Python 3.9; skip the
    # injection outright instead of importing, failing and rolling back.
    if sys.version_info < (3, 10):
        return False
    try:
        import torch
        if not _version_at_least(torch.__version__, _MIN_TORCH):
            return False
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        import triton
        if not _version_at_least(triton.__version__, _MIN_TRITON):
            return False
    except Exception:
        return False
    # Hopper with triton in [3.4.0, 3.7.1) used to be excluded here, because fla's
    # gated chunk_bwd_dqkwg raised outright on that combination (fla #640) and the
    # whole model had to fall back to transformers' pure-torch gated-delta path.
    # The vendored kernel now avoids the miscompiled BK=64 tile instead of refusing
    # to run, so those hosts keep the Triton fast path. Users who want the old
    # conservative behaviour set UNSLOTH_DISABLE_HOPPER_FLA_BWD=1, which is handled
    # in patch_vendor_fla (it must also disable a user-installed fla, which this
    # boolean cannot express).
    return True


def _hopper_dqkwg_suspect_here():
    """``_hopper_dqkwg_suspect`` for the live interpreter, or False if unknowable."""
    try:
        import torch
        import triton
    except Exception:
        return False
    return _hopper_dqkwg_suspect(torch, triton)


@functools.lru_cache(maxsize=None)
def _device_index_is_hopper(index):
    """Whether one CUDA device index is an NVIDIA Hopper (SM90).

    ``_hopper_dqkwg_suspect_here`` deliberately answers for the whole process (any
    visible Hopper counts), which is right for deciding *whether to install* the
    workaround. Deciding whether a given call needs it is a per-tensor question:
    fla #640 is Hopper-only, so narrowing the tile for a tensor on an Ada or
    Blackwell card in the same box would cost speed for nothing. HIP excluded, as
    an AMD Instinct reports capability major 9 without being Hopper.

    Returns None when the device cannot be inspected, so callers can fall back to
    the process-wide answer rather than fail open into the miscompile.
    """
    try:
        import torch
        if getattr(getattr(torch, "version", None), "hip", None) is not None:
            return False
        if index is None:
            index = torch.cuda.current_device()
        if torch.cuda.get_device_capability(index)[0] == 9:
            return True
        return "NVIDIA H" in torch.cuda.get_device_name(index)
    except Exception:
        return None


def _tensor_on_hopper(x):
    """Whether a call on this tensor needs the #640 workaround.

    Unknown devices fall back to True: the wrapper is only installed at all when
    ``_hopper_dqkwg_suspect_here()`` already said some visible GPU is Hopper, so on
    a host we cannot inspect the safe answer is to keep the narrower tile. Being
    wrong that way costs speed; being wrong the other way corrupts gradients.
    """
    try:
        if x is None or not x.is_cuda:
            return True
        answer = _device_index_is_hopper(x.device.index)
        return True if answer is None else bool(answer)
    except Exception:
        return True


# Set once when UNSLOTH_DISABLE_HOPPER_FLA_BWD turns the fast kernels off, so
# unsloth's loader can explain *why* the slow path was chosen instead of blaming
# "no CUDA / torch < 2.7 / triton < 3.3", none of which is true on an H100.
_FLA_DISABLED_REASON = None


def fla_unavailable_reason():
    """A user-facing explanation of why Unsloth disabled fla's gated-delta kernels,
    or ``None`` when they were not disabled. Read by unsloth's model loader."""
    return _FLA_DISABLED_REASON


def _vendored_injection_supported():
    """Whether ``patch_vendor_fla`` would actually inject the vendored kernels.

    Exposed so tests can gate their subprocess assertions on the exact same
    production support check (Python >= 3.10, torch/triton minimums, CUDA)
    instead of a looser mirror that would fail rather than skip on unsupported
    hosts. Hopper in the fla #640 Triton range is no longer excluded: the vendored
    chunk_bwd_dqkwg steps around the miscompiled tile, so it injects there too.
    """
    return _torch_triton_cuda_supported()


def _vendored_already_injected():
    mod = sys.modules.get("fla")
    return mod is not None and getattr(mod, _VENDORED_MARK, False) is True


def _should_defer_to_installed_fla():
    """True if a user-installed (non-vendored) fla should be used instead of the
    vendored snapshot.

    We defer only when the installed fla is *strictly newer* than the vendored
    version: a newer upstream supersedes our copy, while an equal or older install
    is shadowed by the vendored kernels (which carry post-0.5.1 backports). A
    deliberate install whose version cannot be read is respected rather than
    shadowed. ``UNSLOTH_FORCE_VENDORED_FLA`` overrides this to force the vendored
    copy even over a newer install."""
    mod = sys.modules.get("fla")
    if mod is not None:
        if getattr(mod, _VENDORED_MARK, False) is True:
            return False  # our own vendored copy, not a user install
        ver = getattr(mod, "__version__", None)
    else:
        try:
            spec = importlib.util.find_spec("fla")
        except Exception:
            return False
        if spec is None:
            return False
        ver = None
        try:
            import importlib.metadata as _md
            for dist in ("flash-linear-attention", "fla-core", "fla"):
                try:
                    ver = _md.version(dist)
                    break
                except Exception:
                    continue
        except Exception:
            ver = None
    if ver is None:
        # Importable but version unknown: respect the user's deliberate install
        # rather than shadowing something we cannot assess.
        return True
    return _version_strictly_after(ver, _VENDORED_FLA_VERSION)


def _neutralize_tilelang_backend_probe():
    """Permanently make the pruned TileLang backend unavailable without importing
    the external ``tilelang``.

    The vendored ``TileLangBackend`` overrides ``is_available()`` to ``import
    tilelang`` (catching only ``ImportError``), and both backend registration
    (``can_use``) and the dispatch loop evaluate ``is_available()`` *before* the
    ``FLA_TILELANG=0`` ``is_enabled()`` gate. A broken/ABI-incompatible installed
    tilelang that raises a non-``ImportError`` on import would therefore abort the
    injection (during registration) and every later gated-delta dispatch. The
    pruned snapshot dropped the tilelang kernels, so the backend can never serve a
    call anyway; override the probe to a plain ``False``.

    Must be called while ``import tilelang`` is shadowed (see ``_inject_vendored_fla``)
    so importing the backend module here cannot raise. Best effort.
    """
    try:
        from fla.ops.common.backends.tilelang import TileLangBackend
        TileLangBackend.is_available = classmethod(lambda cls: False)
        try:
            # can_use is @cache and may have memoized a probe from registration.
            TileLangBackend.can_use.cache_clear()
        except Exception:
            pass
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: could not neutralize vendored tilelang backend: {e}")


def _neutralize_intracard_backend_probe():
    """Permanently make the pruned IntraCard CP backend unavailable.

    The vendored snapshot drops ``fla.ops.common.intracard_cp``, but
    ``IntraCardCPBackend.is_available()`` still returns ``True`` unconditionally.
    Dispatch checks ``is_available() and is_enabled()`` per call, so a user who
    flips ``FLA_INTRACARD_CP=1`` after import would route varlen inference into
    ``chunk_gated_delta_rule_fwd_h`` and hit ``ModuleNotFoundError`` on the pruned
    module. Forcing the env flag off is not enough (it is user-flippable), so
    override the probe to ``False`` like the TileLang backend. Best effort.
    """
    try:
        from fla.ops.common.backends.intracard import IntraCardCPBackend
        IntraCardCPBackend.is_available = classmethod(lambda cls: False)
        try:
            IntraCardCPBackend.can_use.cache_clear()
        except Exception:
            pass
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: could not neutralize vendored intracard backend: {e}")


def _blackwell_import_device(torch_mod):
    """A Blackwell CUDA device index to make current during the vendored import,
    or ``None`` when no switch is needed.

    The vendored ``fla.utils`` freezes ``IS_NVIDIA_BLACKWELL`` (and the Blackwell
    autotune configs / tl.dot workaround derived from it) at import time from the
    *current* device's capability. On a mixed host (e.g. cuda:0 Ada, cuda:1 B200)
    importing while cuda:0 is current wrongly disables the Blackwell-pinned configs
    for kernels that later launch on the B200, reintroducing the corruption the
    backports guard against. If any visible device is Blackwell (capability major
    10/12) but the current one is not, point the import at the Blackwell device.
    """
    try:
        if not torch_mod.cuda.is_available():
            return None
        current = torch_mod.cuda.current_device()
        if torch_mod.cuda.get_device_capability(current)[0] in (10, 12):
            return None  # already Blackwell-current
        for index in range(torch_mod.cuda.device_count()):
            if torch_mod.cuda.get_device_capability(index)[0] in (10, 12):
                return index
    except Exception:
        return None
    return None


def _inject_vendored_fla():
    """Register the vendored fla tree into sys.modules under the name ``fla``.

    Bootstraps ``fla`` as a real package whose ``__path__`` points at the
    vendored directory, then eagerly imports the exported subpackages so the
    whole tree (fla.ops.gated_delta_rule, fla.ops.common(.*), fla.ops.cp(.*),
    fla.ops.utils, fla.modules, fla.utils) is registered. Python's normal
    FileFinder resolves every submodule and the internal ``from fla...`` absolute
    imports against this ``__path__``.

    Returns ``(injected, replaced_real)``. ``replaced_real`` is True when a real
    (non-vendored) fla was purged to make room for the vendored tree (only under
    ``UNSLOTH_FORCE_VENDORED_FLA``); callers use it to rebind already-imported
    modeling modules whose kernel globals still point at the old install.
    """
    vendored_dir = _vendored_fla_dir()
    init_path = os.path.join(vendored_dir, "__init__.py")
    if not os.path.isfile(init_path):
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: vendored fla missing at {init_path}; keeping pure-torch path.")
        return False, False

    # The pruned snapshot drops the TileLang kernels (backends/tilelang/chunk_bwd
    # and parallel_attn_*) and the IntraCard CP impl (ops/common/intracard_cp), so
    # force their backend flags off. Otherwise the 'common' dispatch would route a
    # gated chunk_bwd_dqkwg to TileLang (on by default whenever an external
    # tilelang is installed) and hit ModuleNotFoundError. Set only for our injected
    # tree; a deferred-to real fla install never reaches here. Snapshot the prior
    # values so a failed injection does not leave a user's real fla with these
    # backends disabled for the rest of the process (restored in the rollback).
    prev_tilelang = os.environ.get("FLA_TILELANG")
    prev_intracard = os.environ.get("FLA_INTRACARD_CP")
    os.environ["FLA_TILELANG"] = "0"
    os.environ["FLA_INTRACARD_CP"] = "0"

    # Snapshot then purge any pre-existing fla* modules (e.g. a real install we
    # are shadowing under UNSLOTH_FORCE_VENDORED_FLA) so imports resolve to the
    # vendored tree rather than stale cached modules.
    saved = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "fla" or k.startswith("fla.")
    }
    replaced_real = any(
        getattr(m, _VENDORED_MARK, False) is not True for m in saved.values()
    )
    for k in saved:
        del sys.modules[k]

    spec = importlib.util.spec_from_file_location(
        "fla", init_path, submodule_search_locations=[vendored_dir],
    )
    fla_mod = importlib.util.module_from_spec(spec)
    setattr(fla_mod, _VENDORED_MARK, True)
    sys.modules["fla"] = fla_mod

    # Importing fla.ops.gated_delta_rule transitively registers the common
    # backends, whose TileLangBackend.is_available() does `import tilelang`
    # (catching only ImportError) and is probed before the FLA_TILELANG=0 gate. A
    # broken/incompatible installed tilelang that raises a non-ImportError would
    # abort this injection during registration (and every later dispatch). Shadow
    # the external tilelang with None (a clean ImportError) across the import so
    # registration cannot raise, permanently neutralize the probe, then restore
    # tilelang so a real, working install stays importable for any non-fla use.
    _tl_sentinel = object()
    _tl_prev = sys.modules.get("tilelang", _tl_sentinel)
    _tl_shadow = _tl_prev is _tl_sentinel or _tl_prev is None

    # Make a Blackwell device current for the import so fla.utils freezes
    # IS_NVIDIA_BLACKWELL (and its pinned autotune configs) correctly on a mixed
    # host where cuda:0 is not the Blackwell card the model runs on.
    _bw_dev = _bw_prev = None
    try:
        import torch as _torch_bw
        _bw_dev = _blackwell_import_device(_torch_bw)
    except Exception:
        _bw_dev = None
    try:
        try:
            if _bw_dev is not None:
                _bw_prev = _torch_bw.cuda.current_device()
                _torch_bw.cuda.set_device(_bw_dev)
            if _tl_shadow:
                sys.modules["tilelang"] = None
            spec.loader.exec_module(fla_mod)
            for sub in _EXPORT_SUBMODULES:
                importlib.import_module(sub)
            _neutralize_tilelang_backend_probe()
            _neutralize_intracard_backend_probe()
        finally:
            if _bw_prev is not None:
                try:
                    _torch_bw.cuda.set_device(_bw_prev)
                except Exception:
                    pass
            if _tl_shadow:
                if _tl_prev is _tl_sentinel:
                    sys.modules.pop("tilelang", None)
                else:
                    sys.modules["tilelang"] = _tl_prev
    except Exception as e:
        # Roll back a partial injection and restore whatever we purged, including
        # the backend env flags so a shadowed real fla is left exactly as we
        # found it.
        for k in list(sys.modules):
            if k == "fla" or k.startswith("fla."):
                sys.modules.pop(k, None)
        sys.modules.update(saved)
        _restore_env("FLA_TILELANG", prev_tilelang)
        _restore_env("FLA_INTRACARD_CP", prev_intracard)
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: failed injecting vendored fla ({e}); keeping pure-torch path.")
        return False, False
    return True, replaced_real


def _vendored_availability_probe():
    """Availability answer while the vendored (pruned) fla is the active one.

    Modeling modules call this once at import time and then ``from fla import``
    the kernels, so answer True only for callers the pruned exports fully cover;
    an uncovered model (olmo_hybrid needs ShortConvolution) keeps its pure-torch
    fallback instead of crashing on the import. Non-modeling callers get True.
    """
    try:
        caller = sys._getframe(1).f_globals.get("__name__", "")
    except Exception:
        caller = ""
    if caller.startswith("transformers.models."):
        parts = caller.split(".")
        return len(parts) > 2 and parts[2] in _VENDOR_COVERED_MODELS
    return True


def _unavailable_probe():
    """Availability answer when fla must not be used on this host at all.

    Unlike ``_vendored_availability_probe`` this is deliberately not caller-aware:
    the #640 miscompile hits every gated-deltanet model, so every caller has to see
    False and take its pure-torch fallback.
    """
    return False


def _patch_is_available(probe=None):
    """Replace transformers' cached availability probe.

    The probe is @lru_cache and keys on dist metadata that a vendored package
    lacks, so we clear the cache and replace the callable outright. Modeling
    modules bind the name lazily (after this runs), so replacement is enough.

    ``probe`` defaults to the vendored answer; the Hopper opt-out passes
    ``_unavailable_probe`` to force the pure-torch path instead.
    """
    if probe is None:
        probe = _vendored_availability_probe
    try:
        import transformers.utils.import_utils as iu
    except Exception:
        return False
    original = getattr(iu, "is_flash_linear_attention_available", None)
    try:
        iu.is_flash_linear_attention_available.cache_clear()
    except Exception:
        pass
    iu.is_flash_linear_attention_available = probe
    # Re-exporting namespaces (e.g. ``transformers.utils`` on versions that alias
    # it, or any transformers.* module that did ``from ...import_utils import
    # is_flash_linear_attention_available`` before this ran) still hold the
    # original cached callable, so public callers there would keep seeing False.
    # Rebind every transformers.* namespace that points at that exact object.
    if original is not None:
        for name, mod in list(sys.modules.items()):
            if mod is None or name == "transformers.utils.import_utils":
                continue
            if not (name == "transformers" or name.startswith("transformers.")):
                continue
            # Read via __dict__, not getattr: getattr fires transformers' lazy
            # __getattr__, which imports optional deps like torchvision and crashes.
            mod_dict = getattr(mod, "__dict__", None)
            if not isinstance(mod_dict, dict):
                continue
            if mod_dict.get("is_flash_linear_attention_available") is original:
                try:
                    setattr(mod, "is_flash_linear_attention_available", probe)
                except Exception:
                    pass
    return True


def _repair_already_imported_modeling(force_rebind=False):
    """Rebind fla globals on modeling modules imported before injection.

    If a gated-deltanet modeling module was imported while fla was unavailable it
    holds ``chunk_gated_delta_rule = fused_recurrent_gated_delta_rule =
    FusedRMSNormGated = None``. Rebind those to the vendored kernels.

    When ``force_rebind`` is set (``UNSLOTH_FORCE_VENDORED_FLA`` just replaced a
    real fla install), those globals are non-``None`` but still point at the old
    user kernels, so the None-only check misses them; rebind by module identity
    (anything not already the vendored callable) so the escape hatch takes hold.
    """
    fused_rms = chunk_fn = fused_recurrent_fn = None
    loaded = False
    for pkg in _REPAIR_MODELING:
        modname = f"transformers.models.{pkg}.modeling_{pkg}"
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        needs = (
            getattr(mod, "chunk_gated_delta_rule", "MISSING") is None
            or getattr(mod, "fused_recurrent_gated_delta_rule", "MISSING") is None
            or getattr(mod, "FusedRMSNormGated", "MISSING") is None
        )
        if not needs and not force_rebind:
            continue
        if not loaded:
            try:
                from fla.modules import FusedRMSNormGated
                from fla.ops.gated_delta_rule import (
                    chunk_gated_delta_rule,
                    fused_recurrent_gated_delta_rule,
                )
                fused_rms = FusedRMSNormGated
                chunk_fn = chunk_gated_delta_rule
                fused_recurrent_fn = fused_recurrent_gated_delta_rule
                loaded = True
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(f"Unsloth: could not load vendored fla symbols for repair: {e}")
                return
        # Already bound to the vendored kernels (e.g. an idempotent re-run): skip.
        if (
            getattr(mod, "chunk_gated_delta_rule", None) is chunk_fn
            and getattr(mod, "fused_recurrent_gated_delta_rule", None) is fused_recurrent_fn
            and getattr(mod, "FusedRMSNormGated", None) is fused_rms
        ):
            continue
        setattr(mod, "FusedRMSNormGated", fused_rms)
        setattr(mod, "chunk_gated_delta_rule", chunk_fn)
        setattr(mod, "fused_recurrent_gated_delta_rule", fused_recurrent_fn)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: rebound vendored fla kernels onto {modname}.")


def _disable_already_imported_gated_delta(packages=_GATED_DELTA_MODELING, why="UNSLOTH_DISABLE_HOPPER_FLA_BWD"):
    """Unbind the miscompiled chunk kernel on gated-delta modeling modules that were
    imported before this ran.

    Mirror image of ``_repair_already_imported_modeling``. A modeling module
    imported while some fla was available holds a live ``chunk_gated_delta_rule``,
    and ``Qwen3NextGatedDeltaNet.__init__`` reads it as ``chunk_gated_delta_rule or
    torch_chunk_gated_delta_rule``, so forcing the availability probe False is not
    enough on its own. Setting the global to ``None`` makes every layer built after
    this point pick the pure-torch path.

    Deliberately narrow: only ``chunk_gated_delta_rule`` is unbound. fla #640 is a
    backward-pass bug in the chunked kernel; ``fused_recurrent_gated_delta_rule``
    (decode) and ``FusedRMSNormGated`` are unaffected and stay fast.
    """
    for pkg in packages:
        modname = f"transformers.models.{pkg}.modeling_{pkg}"
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        if getattr(mod, "chunk_gated_delta_rule", None) is None:
            continue
        try:
            setattr(mod, "chunk_gated_delta_rule", None)
        except Exception:
            continue
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                f"Unsloth: unbound fla chunk_gated_delta_rule on {modname} ({why})."
            )


_INSTALLED_FLA_PATCH_MARK = "_unsloth_hopper_dqkwg_patched"

# Per-thread override for the installed-fla patch below. NOT a module global that
# gets saved/mutated/restored around the call: torch's autograd engine runs one
# worker thread per device ("The engine operates by having a single worker thread
# per work queue, and every work queue is pinned to a specific device",
# torch/csrc/autograd/engine.cpp), so a single ``.backward()`` over a model sharded
# across two GPUs already executes two Python backward bodies concurrently, and
# every ATen op / Triton launch inside them drops the GIL
# (``pybind11::gil_scoped_release`` in the generated bindings,
# ``Py_BEGIN_ALLOW_THREADS`` in Triton's launcher). A save/restore of a module
# global therefore genuinely interleaves: one call restores the guard flag to True
# while the other is still inside, which resurrects the RuntimeError mid-backward,
# or leaves the tile override stuck on for the rest of the process. A
# threading.local carries the override on the calling thread only, so no lock is
# needed and multi-GPU backward stays parallel.
_installed_fla_tls = threading.local()


def _installed_fla_forcing_small_tile():
    return getattr(_installed_fla_tls, "force_small_tile", False)


def _patch_installed_fla_dqkwg():
    """Apply the BK=64 workaround to a *user-installed* fla, in place.

    When a deliberate fla-core install is present on an affected Hopper host we
    would otherwise have to shadow it with the vendored snapshot to get the tile
    fix, silently downgrading whatever newer upstream the user chose. Patching
    their copy instead keeps their kernels and fixes only the miscompiled tile.

    Mechanism, chosen so it does not depend on the installed fla's source layout
    (which differs across versions): wrap ``chunk_bwd_dqkwg`` and steer the two
    module globals it reads. Both are plain ``LOAD_GLOBAL`` reads out of
    ``fla.ops.common.chunk_o.__dict__`` on every call, so rebinding the module
    attribute is enough.

      * ``IS_NVIDIA_HOPPER`` is set to False *once and permanently*. Inside
        ``chunk_bwd_dqkwg`` that global is read only by the blanket guard; the
        Hopper autotune restriction (``NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER
        else ...``) is a module constant frozen into the kernel's config list at
        import, so a later rebind cannot move it. Setting it permanently rather
        than per call is what makes this wrapper thread safe (see
        ``_installed_fla_tls``).
      * ``check_shared_mem`` is replaced *once and permanently* by a shim that
        answers False while the calling thread has the small-tile override set,
        and delegates to fla's real (``@cache``d) implementation otherwise. That
        makes the function's own arithmetic pick ``CONST_TILING = 32``, hence
        ``BK = 32``, only for the calls that would otherwise land on 64.

    ``BV`` drops to 32 alongside ``BK`` on that path, because both derive from the
    same ``CONST_TILING``. That costs some speed for head dims 33..64 only, and
    the vendored copy (which edits the source directly) keeps the wider ``BV``.

    Returns True if the installed fla is patched, including when a previous call
    already patched it: this runs once at import and again from TEMPORARY_PATCHES,
    and reporting idempotence as failure would make the second call shadow the
    user's fla with the vendored snapshot. Best effort: any real failure leaves the
    installed fla untouched and the caller falls back to shadowing it.
    """
    try:
        import triton
        import fla.ops.common.chunk_o as chunk_o
    except Exception:
        return False

    fn = getattr(chunk_o, "chunk_bwd_dqkwg", None)
    if fn is None:
        return False
    if getattr(fn, _INSTALLED_FLA_PATCH_MARK, False):
        return True  # already patched by an earlier call; idempotent success
    if not all(hasattr(chunk_o, a) for a in ("IS_NVIDIA_HOPPER", "check_shared_mem")):
        return False  # unrecognised layout; do not guess

    original = fn

    # Every in-tree fla caller passes `g=` / `k=` by keyword, but the signature is
    # (q, k, v, do, h, dh, w=None, g=None, ...), so a positional caller is possible.
    # Resolve the positions once; if the signature cannot be read, treat every call
    # as potentially gated and take the safe tile (costs speed, never correctness).
    try:
        _params = list(inspect.signature(original).parameters)
        _k_pos, _g_pos = _params.index("k"), _params.index("g")
    except Exception:
        _k_pos = _g_pos = None

    def _arg(name, pos, args, kwargs, missing):
        if name in kwargs:
            return kwargs[name]
        if pos is not None and len(args) > pos:
            return args[pos]
        return missing

    _MISSING = object()

    real_check_shared_mem = chunk_o.check_shared_mem
    if not getattr(real_check_shared_mem, _INSTALLED_FLA_PATCH_MARK, False):
        def _check_shared_mem_shim(arch="none", tensor_idx=0):
            if _installed_fla_forcing_small_tile():
                return False
            return real_check_shared_mem(arch, tensor_idx)

        setattr(_check_shared_mem_shim, _INSTALLED_FLA_PATCH_MARK, True)
        _check_shared_mem_shim.__wrapped__ = real_check_shared_mem
        chunk_o.check_shared_mem = _check_shared_mem_shim
    else:
        real_check_shared_mem = getattr(
            real_check_shared_mem, "__wrapped__", real_check_shared_mem,
        )

    # Permanent: the guard is the only runtime reader of this global inside
    # chunk_bwd_dqkwg, and on this host it would only ever refuse to run.
    chunk_o.IS_NVIDIA_HOPPER = False

    @functools.wraps(original)
    def _patched(*args, **kwargs):
        g = _arg("g", _g_pos, args, kwargs, _MISSING)
        k = _arg("k", _k_pos, args, kwargs, _MISSING)
        if g is None:
            return original(*args, **kwargs)  # ungated: the miscompile cannot fire
        if not _tensor_on_hopper(k):
            # The wrapper is installed process-wide because *some* visible GPU is
            # Hopper, but #640 is Hopper-only. A call on an Ada / Ampere / Blackwell
            # card in the same box must keep its normal tiling, or K=33..64 would
            # narrow both BK and BV on every backward for no reason.
            return original(*args, **kwargs)
        try:
            idx = k.device.index
            if real_check_shared_mem('hopper', idx):
                const_tiling = 128
            elif real_check_shared_mem('ada', idx):
                const_tiling = 64
            else:
                const_tiling = 32
            bad_tile = min(max(triton.next_power_of_2(k.shape[-1]), 16), const_tiling) == 64
        except Exception:
            bad_tile = True  # unknown shape/args: take the safe tile
        previous = _installed_fla_forcing_small_tile()
        _installed_fla_tls.force_small_tile = bad_tile or previous
        try:
            return original(*args, **kwargs)
        finally:
            _installed_fla_tls.force_small_tile = previous

    setattr(_patched, _INSTALLED_FLA_PATCH_MARK, True)
    chunk_o.chunk_bwd_dqkwg = _patched

    # fla/ops/gated_delta_rule/chunk.py does `from fla.ops.common.chunk_o import
    # chunk_bwd_dqkwg` at import, so that module global still holds the original.
    # Rebind every fla module pointing at it, the same way _patch_is_available
    # rebinds transformers' probe.
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "fla" or name.startswith("fla.")):
            continue
        mod_dict = getattr(mod, "__dict__", None)
        if not isinstance(mod_dict, dict):
            continue
        if mod_dict.get("chunk_bwd_dqkwg") is original:
            try:
                setattr(mod, "chunk_bwd_dqkwg", _patched)
            except Exception:
                pass
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth: patched the installed fla's chunk_bwd_dqkwg for the Hopper "
            "BK=64 miscompile (fla #640); keeping your fla install."
        )
    return True


def _mark_fla_disabled_hopper():
    global _FLA_DISABLED_REASON
    if _FLA_DISABLED_REASON is not None:
        return
    try:
        import triton
        triton_version = triton.__version__
    except Exception:
        triton_version = "?"
    _FLA_DISABLED_REASON = (
        "Unsloth: gated-deltanet (linear attention) fast kernels are DISABLED on this GPU\n"
        "because UNSLOTH_DISABLE_HOPPER_FLA_BWD=1 is set. Triton "
        f"{triton_version} on Hopper\n"
        "(H100 / H200 / H20) miscompiles flash-linear-attention's gated-delta backward\n"
        "pass (fla issue #640), so training falls back to the slower pure-PyTorch path.\n"
        '  To use the fast kernels with a Triton that has the fix: pip install -U "triton>=3.7.1"\n'
        "  To use them on this Triton: unset UNSLOTH_DISABLE_HOPPER_FLA_BWD. Unsloth's\n"
        "  bundled kernels already step around the miscompiled block size."
    )
    if UNSLOTH_ENABLE_LOGGING:
        logger.warning(_FLA_DISABLED_REASON)


def _transformers_uses_availability_probe():
    """Whether this Transformers still selects gated-delta kernels via the
    ``is_flash_linear_attention_available()`` probe + module globals.

    Transformers PR #47630 ("[Kernels] Refactor all linear attn models & native
    kernels fallback", merged after v5.14.1) stops using that probe: the modeling
    files now carry
    ``@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")``,
    which resolves the implementation with ``importlib.import_module`` at
    decoration time and freezes it into a closure. There is then no probe to answer
    False and no module global to unbind.

    Detected by the *presence of the new mechanism*, not the absence of the old
    name. Two traps make the obvious check wrong:

      * ``is_flash_linear_attention_available`` still exists in
        ``transformers/utils/import_utils.py`` after #47630 (it is merely unused by
        the modeling files), so ``hasattr`` on it is True in both layouts.
      * ``_patch_is_available`` assigns the attribute unconditionally, so once it
        has run the old name is present even on a Transformers that never had it.
        Any caller must therefore evaluate this BEFORE ``_patch_is_available``.
    """
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return True  # cannot tell; assume the old layout and stay quiet
    return not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback")


def _warn_hopper_optout_degraded():
    """Say so loudly when UNSLOTH_DISABLE_HOPPER_FLA_BWD cannot force pure torch.

    Silence would be the dangerous outcome: the user set a *correctness* switch and
    would reasonably believe gated-delta training had moved off the miscompiled
    kernel. On a post-#47630 Transformers neither lever we pull steers kernel
    selection, so the switch cannot do what it says.

    Deliberately makes no claim about the resulting gradients. The caller falls
    through to the normal path, which patches or shadows fla so the kernel the
    decorator resolves carries the tile fix, but whether that succeeds depends on
    the install it finds -- and asserting "your gradients are correct" here would
    be exactly the wrong thing to say if it did not.

    Note it must not advertise ``triton>=3.7.1`` as a way to reach the pure-PyTorch
    path either. Upgrading Triton makes ``_hopper_dqkwg_suspect_here()`` False, so
    this whole block is skipped and the fast kernels are used -- correct gradients,
    but still not the fallback the user asked for. On this layout there is no lever
    on our side that forces pure torch; only the absence of an importable ``fla``
    does that, since the decorator falls back when its import fails.
    """
    logger.warning(
        "Unsloth: UNSLOTH_DISABLE_HOPPER_FLA_BWD=1 could not force the pure-PyTorch\n"
        "path. This Transformers selects gated-deltanet kernels through the\n"
        "kernel-hub decorator (transformers#47630) rather than\n"
        "is_flash_linear_attention_available, so there is no availability probe to\n"
        "disable and no module global to unbind.\n"
        "Unsloth is instead making the fla that decorator resolves one that avoids\n"
        "the miscompiled block size (fla #640), which is what the opt-out was\n"
        "protecting you from.\n"
        'Installing a fixed Triton (pip install -U "triton>=3.7.1") also removes the\n'
        "miscompile, but note neither route gives you the pure-PyTorch path on this\n"
        "Transformers: that decorator only falls back when fla cannot be imported\n"
        "at all."
    )


def patch_vendor_fla(phase=None):
    """Register the bundled fla kernels and advertise availability.

    Idempotent; safe to call at import time and again from TEMPORARY_PATCHES.
    """
    # Correctness switch, checked before the source-preference flags below. On
    # Hopper + Triton [3.4.0, 3.7.1) *every* fla on the host has the #640 backward
    # miscompile except our vendored copy, which steps around it. A user who does
    # not want to rely on that opts out here and gets transformers' pure-torch
    # gated-delta path. Bailing out of injection is not enough on its own: an
    # installed fla stays importable, so transformers' own availability probe would
    # answer True and bind the unpatched kernels (unslothai/unsloth#5276).
    optout_degraded = False
    if _flag("UNSLOTH_DISABLE_HOPPER_FLA_BWD") and _hopper_dqkwg_suspect_here():
        # Sample the layout BEFORE _patch_is_available, which assigns the probe
        # attribute unconditionally and would otherwise make every Transformers
        # look like the old one.
        if _transformers_uses_availability_probe():
            _mark_fla_disabled_hopper()
            _patch_is_available(_unavailable_probe)
            _disable_already_imported_gated_delta()
            return
        # Post-#47630 Transformers: the kernel-hub decorator resolves fla with
        # importlib at decoration time, so neither the availability probe nor the
        # module globals steer kernel selection and the opt-out cannot reach the
        # pure-PyTorch path. Do NOT return here. Returning would leave whatever fla
        # the decorator resolves -- a user's unpatched install -- serving the BK=64
        # backward, making the safety switch worse than not setting it. Fall
        # through to the normal path instead: it patches an installed fla in place
        # and otherwise injects the vendored snapshot, either of which guarantees
        # the fla the decorator resolves carries the tile fix. Deliberately no
        # _mark_fla_disabled_hopper() -- fla is not disabled on this path, and
        # claiming otherwise would mislead unsloth's loader message.
        _warn_hopper_optout_degraded()
        optout_degraded = True

    if _flag("UNSLOTH_DISABLE_VENDORED_FLA") and not optout_degraded:
        # Scope is the vendored injection only: a user's own fla install is left as
        # found so Transformers' native availability probe still governs it.
        #
        # Skipped when the Hopper opt-out landed in its degraded mode: that flag is
        # a source *preference* ("prefer your fla over ours"), while
        # UNSLOTH_DISABLE_HOPPER_FLA_BWD is a *correctness* switch, and returning
        # here would leave the kernel-hub decorator resolving an unpatched BK=64
        # install. Correctness outranks the preference, so the protection path runs.
        return

    replaced_real = False
    if not _vendored_already_injected():
        force = _flag("UNSLOTH_FORCE_VENDORED_FLA")
        if not force and _should_defer_to_installed_fla():
            # A newer (or unversioned deliberate) user install is present; use it.
            # On an affected Hopper host it carries the #640 miscompile, which is
            # what left a pip-installed fla-core raising mid-backward
            # (unslothai/unsloth#5276). Patch their copy in place rather than
            # shadowing it, so they keep the upstream they deliberately chose. If
            # the patch cannot be applied, fall through to the vendored snapshot,
            # which has the fix compiled in.
            if not _hopper_dqkwg_suspect_here():
                return
            if _patch_installed_fla_dqkwg():
                # Deliberately no _patch_is_available() here: their fla is a real
                # install, so transformers' native probe already finds it, and our
                # vendored probe would wrongly narrow it to the vendor-covered
                # models. Rebinding chunk_bwd_dqkwg on the fla modules is enough,
                # because chunk_gated_delta_rule looks it up as a module global at
                # call time, so modeling modules imported earlier pick it up too.
                return
        if not _torch_triton_cuda_supported():
            # Cannot run the Triton kernels here; leave the pure-torch fallback.
            return
        injected, replaced_real = _inject_vendored_fla()
        if not injected:
            return
        if replaced_real and _hopper_dqkwg_suspect_here():
            # A real fla install was just purged on a host where its
            # chunk_bwd_dqkwg carries the #640 miscompile.
            # _repair_already_imported_modeling rebinds the vendor-covered Qwen
            # packages onto the fixed vendored kernels, but the snapshot cannot
            # serve olmo_hybrid (it prunes ShortConvolution), so a
            # module of theirs imported before this ran would keep calling the
            # purged install's unpatched kernel and silently corrupt dk/dg. Unbind
            # it and let them take transformers' pure-torch gated-delta path.
            _disable_already_imported_gated_delta(
                packages=_UNCOVERED_GATED_DELTA, why="fla #640; vendored tree cannot serve this model",
            )

    _patch_is_available()
    _repair_already_imported_modeling(force_rebind=replaced_real)


TEMPORARY_PATCHES.append(patch_vendor_fla)

# Run once at import so the vendored fla is registered as early as possible
# (before any gated-deltanet modeling module is imported). Re-run later via
# TEMPORARY_PATCHES once transformers is fully initialised.
#
# Setting UNSLOTH_VENDORED_FLA_NO_AUTORUN=1 suppresses only this import-time run,
# not the TEMPORARY_PATCHES pass or an explicit patch_vendor_fla() call. Tests
# import this module purely to read the support gate; without the guard that
# import would inject fla into their own interpreter as a side effect.
if not _flag("UNSLOTH_VENDORED_FLA_NO_AUTORUN"):
    try:
        patch_vendor_fla()
    except Exception as _e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: early vendored-fla injection deferred: {_e}")
