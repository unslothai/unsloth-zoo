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
  * ``UNSLOTH_DISABLE_VENDORED_FLA=1`` -> do nothing (keep the pure-torch path).
  * A real, importable, version-compatible ``fla`` already present -> defer to
    it (do not shadow a deliberate user install), unless
    ``UNSLOTH_FORCE_VENDORED_FLA=1``.
  * Only injects when torch >= 2.7, triton >= 3.3 and CUDA are available (the
    requirements of the vendored fla-core 0.5.1 kernels); otherwise the
    pure-torch fallback is left untouched.
"""

__all__ = [
    "patch_vendor_fla",
]

import os
import sys
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

# Minimum versions declared by fla-core 0.5.1.
_MIN_TORCH = "2.7"
_MIN_TRITON = "3.3"
# Transformers accepts fla >= 0.2.2 for is_flash_linear_attention_available.
_MIN_FLA = "0.2.2"


def _flag(name):
    return os.environ.get(name, "0") == "1"


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


def _torch_triton_cuda_supported():
    """The vendored fla-core 0.5.1 kernels need torch >= 2.7, triton >= 3.3, CUDA."""
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
    return True


def _vendored_already_injected():
    mod = sys.modules.get("fla")
    return mod is not None and getattr(mod, _VENDORED_MARK, False) is True


def _real_fla_present_and_compatible():
    """True if a user-installed (non-vendored) importable fla >= _MIN_FLA exists."""
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
        # Importable but version unknown: respect the user's deliberate install.
        return True
    return _version_at_least(ver, _MIN_FLA)


def _inject_vendored_fla():
    """Register the vendored fla tree into sys.modules under the name ``fla``.

    Bootstraps ``fla`` as a real package whose ``__path__`` points at the
    vendored directory, then eagerly imports the exported subpackages so the
    whole tree (fla.ops.gated_delta_rule, fla.ops.common(.*), fla.ops.cp(.*),
    fla.ops.utils, fla.modules, fla.utils) is registered. Python's normal
    FileFinder resolves every submodule and the internal ``from fla...`` absolute
    imports against this ``__path__``.
    """
    vendored_dir = _vendored_fla_dir()
    init_path = os.path.join(vendored_dir, "__init__.py")
    if not os.path.isfile(init_path):
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: vendored fla missing at {init_path}; keeping pure-torch path.")
        return False

    # The pruned snapshot drops the TileLang kernels (backends/tilelang/chunk_bwd
    # and parallel_attn_*) and the IntraCard CP impl (ops/common/intracard_cp), so
    # force their backend flags off. Otherwise the 'common' dispatch would route a
    # gated chunk_bwd_dqkwg to TileLang (on by default whenever an external
    # tilelang is installed) and hit ModuleNotFoundError. Set only for our injected
    # tree; a deferred-to real fla install never reaches here.
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
    for k in saved:
        del sys.modules[k]

    spec = importlib.util.spec_from_file_location(
        "fla", init_path, submodule_search_locations=[vendored_dir],
    )
    fla_mod = importlib.util.module_from_spec(spec)
    setattr(fla_mod, _VENDORED_MARK, True)
    sys.modules["fla"] = fla_mod
    try:
        spec.loader.exec_module(fla_mod)
        for sub in _EXPORT_SUBMODULES:
            importlib.import_module(sub)
    except Exception as e:
        # Roll back a partial injection and restore whatever we purged.
        for k in list(sys.modules):
            if k == "fla" or k.startswith("fla."):
                sys.modules.pop(k, None)
        sys.modules.update(saved)
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: failed injecting vendored fla ({e}); keeping pure-torch path.")
        return False
    return True


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


def _patch_is_available():
    """Replace transformers' cached availability probe.

    The probe is @lru_cache and keys on dist metadata that a vendored package
    lacks, so we clear the cache and replace the callable outright. Modeling
    modules bind the name lazily (after this runs), so replacement is enough.
    """
    try:
        import transformers.utils.import_utils as iu
    except Exception:
        return False
    try:
        iu.is_flash_linear_attention_available.cache_clear()
    except Exception:
        pass
    iu.is_flash_linear_attention_available = _vendored_availability_probe
    return True


def _repair_already_imported_modeling():
    """Rebind fla globals on modeling modules imported before injection.

    If a gated-deltanet modeling module was imported while fla was unavailable it
    holds ``chunk_gated_delta_rule = fused_recurrent_gated_delta_rule =
    FusedRMSNormGated = None``. Rebind those to the vendored kernels.
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
        if not needs:
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
        setattr(mod, "FusedRMSNormGated", fused_rms)
        setattr(mod, "chunk_gated_delta_rule", chunk_fn)
        setattr(mod, "fused_recurrent_gated_delta_rule", fused_recurrent_fn)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: rebound vendored fla kernels onto {modname}.")


def patch_vendor_fla(phase=None):
    """Register the bundled fla kernels and advertise availability.

    Idempotent; safe to call at import time and again from TEMPORARY_PATCHES.
    """
    if _flag("UNSLOTH_DISABLE_VENDORED_FLA"):
        return

    if not _vendored_already_injected():
        force = _flag("UNSLOTH_FORCE_VENDORED_FLA")
        if not force and _real_fla_present_and_compatible():
            # A deliberate user install is present; defer to it entirely.
            return
        if not _torch_triton_cuda_supported():
            # Cannot run the Triton kernels here; leave the pure-torch fallback.
            return
        if not _inject_vendored_fla():
            return

    _patch_is_available()
    _repair_already_imported_modeling()


TEMPORARY_PATCHES.append(patch_vendor_fla)

# Run once at import so the vendored fla is registered as early as possible
# (before any gated-deltanet modeling module is imported). Re-run later via
# TEMPORARY_PATCHES once transformers is fully initialised.
try:
    patch_vendor_fla()
except Exception as _e:
    if UNSLOTH_ENABLE_LOGGING:
        logger.warning(f"Unsloth: early vendored-fla injection deferred: {_e}")
