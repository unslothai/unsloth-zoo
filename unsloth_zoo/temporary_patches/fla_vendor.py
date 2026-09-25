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

"""Register the bundled fla (flash-linear-attention) snapshot as top-level ``fla``.

Env: ``UNSLOTH_DISABLE_VENDORED_FLA=1`` never injects (an installed fla is untouched);
an installed fla strictly newer than the snapshot wins unless ``UNSLOTH_FORCE_VENDORED_FLA=1``;
``UNSLOTH_DISABLE_HOPPER_FLA_BWD=1`` forces pure torch on Hopper + Triton [3.4.0, 3.7.1) (fla #640).
Injects only with torch >= 2.7, triton >= 3.3 and CUDA.
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

_VENDORED_MARK = "_UNSLOTH_VENDORED_FLA"

_EXPORT_SUBMODULES = ("fla.modules", "fla.ops", "fla.ops.gated_delta_rule")

# Modeling modules binding fla symbols as globals at import (None when unavailable).
_REPAIR_MODELING = ("qwen3_5", "qwen3_5_moe", "qwen3_next")

# olmo_hybrid also needs ShortConvolution (not vendored), so it is not covered.
_VENDOR_COVERED_MODELS = frozenset(_REPAIR_MODELING)

# All gated-delta consumers; olmo_hybrid can bind an installed fla's #640 kernel.
# Kimi Linear absent: remote code on KDA ops, never reaches chunk_bwd_dqkwg.
_GATED_DELTA_MODELING = _REPAIR_MODELING + ("olmo_hybrid",)

_UNCOVERED_GATED_DELTA = tuple(
    pkg for pkg in _GATED_DELTA_MODELING if pkg not in _VENDOR_COVERED_MODELS
)

# Minimum versions declared by fla-core 0.5.1.
_MIN_TORCH = "2.7"
_MIN_TRITON = "3.3"
# Kept in sync with _vendored/fla/__init__.py (test_vendored_tree_layout).
_VENDORED_FLA_VERSION = "0.5.1"


def _flag(name):
    return os.environ.get(name, "0") == "1"


def _restore_env(name, previous):
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _vendored_fla_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(here)
    return os.path.join(pkg_root, "_vendored", "fla")


def _version_at_least(value, minimum):
    try:
        from packaging import version
        # base_version: 2.7 nightlies / pre-releases must satisfy the minimum.
        parsed = version.parse(str(value).split("+")[0])
        return version.parse(parsed.base_version) >= version.parse(minimum)
    except Exception:
        return False


def _version_strictly_after(value, threshold):
    """``value`` > ``threshold`` by base version (0.5.1.devN is not newer)."""
    try:
        from packaging import version
        parsed = version.parse(str(value).split("+")[0])
        return version.parse(parsed.base_version) > version.parse(threshold)
    except Exception:
        return False


def _hopper_dqkwg_suspect(torch_mod, triton_mod):
    """Any visible GPU is Hopper with triton in [3.4.0, 3.7.1) (fla #640 miscompile).
    Probes every device: a mixed host may run the model on a nonzero Hopper card."""
    try:
        from packaging import version
        v = version.parse(str(triton_mod.__version__).split("+")[0])
        if not (version.parse("3.4.0") <= v < version.parse("3.7.1")):
            return False
        # ROCm Instinct also reports major 9: trust bare major==9 only on CUDA.
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
    # The snapshot uses runtime PEP 604 annotations, which fail on Python 3.9.
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
    # Hopper + #640 Triton is not excluded: the vendored kernel avoids the BK=64 tile.
    return True


# RDNA1 without dot instructions: Triton still emits v_dot2, LLVM aborts (FDOT2).
_NO_DOT_INSTRUCTION_GFX = ("gfx1010", "gfx1013")


def _gpu_lacks_dot_instructions(torch_mod=None):
    """ROCm and any visible GPU is RDNA1 without dot instructions; unreadable -> False."""
    try:
        if torch_mod is None:
            import torch as torch_mod
        if getattr(getattr(torch_mod, "version", None), "hip", None) is None:
            return False
        if not torch_mod.cuda.is_available():
            return False
        for i in range(int(torch_mod.cuda.device_count())):
            props = torch_mod.cuda.get_device_properties(i)
            arch = str(getattr(props, "gcnArchName", "") or "").split(":", 1)[0].strip().lower()
            if arch in _NO_DOT_INSTRUCTION_GFX:
                return True
    except Exception:
        return False
    return False


def _mark_fla_disabled_no_dot_instructions():
    global _FLA_DISABLED_REASON
    if _FLA_DISABLED_REASON is not None:
        return
    _FLA_DISABLED_REASON = (
        "Unsloth: gated-deltanet (linear attention) fast kernels are DISABLED on this GPU.\n"
        "RDNA1 (gfx1010 / gfx1013, e.g. RX 5700 XT) has no dot instructions, and Triton\n"
        "compiles flash-linear-attention's kernels to them anyway, so the process would\n"
        "abort inside LLVM (\"Cannot select: AMDGPUISD::FDOT2\"). Training uses the slower\n"
        "pure-PyTorch gated-delta path instead."
    )
    if UNSLOTH_ENABLE_LOGGING:
        logger.warning(_FLA_DISABLED_REASON)


# transformers' pure-torch l2norm reduces in the input dtype; in fp16 it overflows to
# inf and gives NaN grads in eager mode (RX 5700 XT). fla reduces in float32.
_L2NORM_FP32_MARK = "_unsloth_fp32_l2norm"


def _fp32_l2norm(x, dim = -1, eps = 1e-6):
    """``l2norm`` with the reduction in float32, result in the input dtype: what fla's kernel does."""
    import torch

    xf = x.float()
    inv_norm = torch.rsqrt((xf * xf).sum(dim = dim, keepdim = True) + eps)
    return (xf * inv_norm).to(x.dtype)


setattr(_fp32_l2norm, _L2NORM_FP32_MARK, True)


# unsloth's compiler copies l2norm into unsloth_compiled_module_*, which is what runs.
_UNSLOTH_COMPILED_MODULE_PREFIX = "unsloth_compiled_module"


def _l2norm_modules(packages = None):
    """transformers' gated-delta modeling modules plus unsloth's compiled copies."""
    if packages is None:
        packages = _GATED_DELTA_MODELING
    names = [f"transformers.models.{pkg}.modeling_{pkg}" for pkg in packages]
    names += sorted(
        name for name in list(sys.modules)
        if name.startswith(_UNSLOTH_COMPILED_MODULE_PREFIX) and name not in names
    )
    return names


def _patch_l2norm_fp32_on_torch_path(packages = None):
    """Rebind ``l2norm`` to the float32 version on imported gated-delta modules. Idempotent."""
    patched = []
    for modname in _l2norm_modules(packages):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        current = getattr(mod, "l2norm", None)
        if current is None or getattr(current, _L2NORM_FP32_MARK, False):
            continue
        try:
            setattr(mod, "l2norm", _fp32_l2norm)
        except Exception:
            continue
        patched.append(modname)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: {modname}.l2norm now reduces in float32 (pure-torch gated delta, float16 safe).")
    return patched


def _hopper_dqkwg_suspect_here():
    try:
        import torch
        import triton
    except Exception:
        return False
    return _hopper_dqkwg_suspect(torch, triton)


@functools.lru_cache(maxsize=None)
def _device_index_is_hopper(index):
    """Whether a CUDA device is NVIDIA Hopper; None if unknown. HIP -> False."""
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
    """Whether a call on ``x`` needs the #640 tile; unknown -> True (slower, never corrupt)."""
    try:
        if x is None or not x.is_cuda:
            return True
        answer = _device_index_is_hopper(x.device.index)
        return True if answer is None else bool(answer)
    except Exception:
        return True


# Why fla was disabled, for unsloth's loader message.
_FLA_DISABLED_REASON = None


def fla_unavailable_reason():
    """A user-facing explanation of why Unsloth disabled fla's gated-delta kernels,
    or ``None`` when they were not disabled. Read by unsloth's model loader."""
    return _FLA_DISABLED_REASON


def _vendored_injection_supported():
    """The exact production support gate, so tests skip on unsupported hosts."""
    return _torch_triton_cuda_supported()


def _vendored_already_injected():
    mod = sys.modules.get("fla")
    return mod is not None and getattr(mod, _VENDORED_MARK, False) is True


def _should_defer_to_installed_fla():
    """Use an installed fla only if strictly newer than the vendored one, or unversioned."""
    mod = sys.modules.get("fla")
    if mod is not None:
        if getattr(mod, _VENDORED_MARK, False) is True:
            return False
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
        return True
    return _version_strictly_after(ver, _VENDORED_FLA_VERSION)


def _neutralize_tilelang_backend_probe():
    """Force the pruned TileLang backend unavailable; a broken tilelang can raise a
    non-ImportError in its probe. Call while ``tilelang`` is shadowed."""
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
    """Force the pruned IntraCard CP backend unavailable (FLA_INTRACARD_CP is user-flippable)."""
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
    """Blackwell device to make current during import, else None: fla.utils freezes
    IS_NVIDIA_BLACKWELL from the current device at import."""
    try:
        if not torch_mod.cuda.is_available():
            return None
        current = torch_mod.cuda.current_device()
        if torch_mod.cuda.get_device_capability(current)[0] in (10, 12):
            return None
        for index in range(torch_mod.cuda.device_count()):
            if torch_mod.cuda.get_device_capability(index)[0] in (10, 12):
                return index
    except Exception:
        return None
    return None


def _inject_vendored_fla():
    """Register the vendored tree as ``fla``. Returns ``(injected, replaced_real)``;
    ``replaced_real`` means a real fla was purged."""
    vendored_dir = _vendored_fla_dir()
    init_path = os.path.join(vendored_dir, "__init__.py")
    if not os.path.isfile(init_path):
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: vendored fla missing at {init_path}; keeping pure-torch path.")
        return False, False

    # Pruned snapshot lacks TileLang / IntraCard CP; disable them (restored on rollback).
    prev_tilelang = os.environ.get("FLA_TILELANG")
    prev_intracard = os.environ.get("FLA_INTRACARD_CP")
    os.environ["FLA_TILELANG"] = "0"
    os.environ["FLA_INTRACARD_CP"] = "0"

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

    # Shadow tilelang during import so a broken install cannot abort registration.
    _tl_sentinel = object()
    _tl_prev = sys.modules.get("tilelang", _tl_sentinel)
    _tl_shadow = _tl_prev is _tl_sentinel or _tl_prev is None

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
    """True only for callers the pruned exports cover (olmo_hybrid needs ShortConvolution)."""
    try:
        caller = sys._getframe(1).f_globals.get("__name__", "")
    except Exception:
        caller = ""
    if caller.startswith("transformers.models."):
        parts = caller.split(".")
        return len(parts) > 2 and parts[2] in _VENDOR_COVERED_MODELS
    return True


def _unavailable_probe():
    """Not caller-aware: every gated-delta model must take the fallback."""
    return False


def _patch_is_available(probe=None):
    """Replace transformers' lru_cached fla probe (the vendored copy lacks dist metadata)."""
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
    # Rebind re-exports still holding the original cached probe.
    if original is not None:
        for name, mod in list(sys.modules.items()):
            if mod is None or name == "transformers.utils.import_utils":
                continue
            if not (name == "transformers" or name.startswith("transformers.")):
                continue
            # __dict__, not getattr: lazy __getattr__ imports optional deps and crashes.
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
    ``force_rebind`` also replaces non-None globals pointing at a purged real fla."""
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
    """Unbind ``chunk_gated_delta_rule`` on imported gated-delta modules; layers read
    ``chunk or torch_chunk``. Only chunk: #640 is backward-only."""
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

# threading.local, not a saved/restored global: autograd runs one backward thread per
# device and drops the GIL, so a global save/restore interleaves across GPUs.
_installed_fla_tls = threading.local()


def _installed_fla_forcing_small_tile():
    return getattr(_installed_fla_tls, "force_small_tile", False)


def _patch_installed_fla_dqkwg():
    """Apply the #640 BK=64 workaround to a user-installed fla in place.

    Sets ``IS_NVIDIA_HOPPER`` False permanently (only the guard reads it) and shims
    ``check_shared_mem`` to answer False under the thread-local override, giving BK=32.
    Returns True if patched (idempotent); False leaves fla untouched.
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
        return True
    if not all(hasattr(chunk_o, a) for a in ("IS_NVIDIA_HOPPER", "check_shared_mem")):
        return False  # unrecognised layout; do not guess

    original = fn

    # g/k may be positional; an unreadable signature always takes the safe tile.
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

    chunk_o.IS_NVIDIA_HOPPER = False

    @functools.wraps(original)
    def _patched(*args, **kwargs):
        g = _arg("g", _g_pos, args, kwargs, _MISSING)
        k = _arg("k", _k_pos, args, kwargs, _MISSING)
        if g is None:
            return original(*args, **kwargs)
        if not _tensor_on_hopper(k):
            # #640 is Hopper-only; keep normal tiling on other cards.
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
            bad_tile = True
        previous = _installed_fla_forcing_small_tile()
        _installed_fla_tls.force_small_tile = bad_tile or previous
        try:
            return original(*args, **kwargs)
        finally:
            _installed_fla_tls.force_small_tile = previous

    setattr(_patched, _INSTALLED_FLA_PATCH_MARK, True)
    chunk_o.chunk_bwd_dqkwg = _patched

    # gated_delta_rule/chunk.py imported the original by name; rebind it.
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
    """Whether transformers still selects kernels via the availability probe (pre-#47630).
    Must run before ``_patch_is_available``, which creates that attribute."""
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return True
    return not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback")


def _warn_hopper_optout_degraded():
    """Warn that UNSLOTH_DISABLE_HOPPER_FLA_BWD cannot force pure torch post-#47630."""
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


# transformers#47630 asks fla for recurrent_gated_delta_rule, which fla never exported
# (it is fused_recurrent_...), so decode silently ran the torch fallback.
_MISSING_GATED_DELTA_ALIASES = {
    "recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
}


def _alias_missing_gated_delta_names():
    """Add missing gated-delta names to the live fla; never replaces existing ones."""
    module = sys.modules.get("fla.ops.gated_delta_rule")
    if module is None:
        try:
            module = importlib.import_module("fla.ops.gated_delta_rule")
        except Exception:
            return ()

    added = []
    for wanted, source in _MISSING_GATED_DELTA_ALIASES.items():
        if getattr(module, wanted, None) is not None:
            continue
        implementation = getattr(module, source, None)
        if implementation is None:
            continue
        setattr(module, wanted, implementation)
        exported = getattr(module, "__all__", None)
        if isinstance(exported, list) and wanted not in exported:
            exported.append(wanted)
        added.append(wanted)

    if added and UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: aliased {', '.join(added)} onto fla.ops.gated_delta_rule "
            f"so the Triton decode kernel is reachable."
        )
    return tuple(added)


# Kernel-hub decorated wrapper -> fla kernel. causal_conv1d is not vendored.
_KERNEL_HUB_DECORATED = {
    "torch_chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule": "recurrent_gated_delta_rule",
}


def _resolved_implementation(wrapper):
    """The callable a kernel-hub wrapper will actually dispatch to, or None."""
    for cell in getattr(wrapper, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value):
            return value
    return None


def _live_gated_delta_kernel(name):
    """The kernel ``name`` currently resolves to on the live fla, or None."""
    module = sys.modules.get("fla.ops.gated_delta_rule")
    return getattr(module, name, None) if module is not None else None


def _repair_kernel_hub_closures(packages=_REPAIR_MODELING):
    """Re-apply kernel-hub decorators frozen before the live fla existed (post-#47630).
    Re-decorate rather than patch cell_contents: the wrapper also closes over param names."""
    try:
        from transformers.integrations.hub_kernels import (
            use_kernel_func_from_hub_with_fallback,
        )
    except Exception:
        return ()

    repaired = []
    for package in packages:
        module = sys.modules.get(f"transformers.models.{package}.modeling_{package}")
        if module is None:
            continue
        for attribute, kernel in _KERNEL_HUB_DECORATED.items():
            wrapper = getattr(module, attribute, None)
            original = getattr(wrapper, "__wrapped__", None)
            if original is None:
                continue
            current = _resolved_implementation(wrapper)
            live = _live_gated_delta_kernel(kernel)
            if live is not None and current is live:
                continue
            try:
                rebuilt = use_kernel_func_from_hub_with_fallback(kernel, "fla")(original)
            except Exception:
                continue
            if _resolved_implementation(rebuilt) is current:
                continue
            # Stale kernel, no live replacement: the fallback beats a purged install.
            setattr(module, attribute, rebuilt)
            repaired.append(f"{package}.{attribute}")

    if repaired and UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: re-resolved the fla kernels on {', '.join(repaired)}, which "
            f"were bound before the live fla was in place."
        )
    return tuple(repaired)


def _force_kernel_hub_fallback(packages=_GATED_DELTA_MODELING):
    """RDNA1: bind kernel-hub wrappers (incl. compiled copies) to their torch fallback."""
    try:
        from transformers.integrations.hub_kernels import (  # noqa: F401
            use_kernel_func_from_hub_with_fallback,
        )
    except Exception:
        return ()

    forced = []
    names = [f"transformers.models.{package}.modeling_{package}" for package in packages]
    names += sorted(
        name for name in list(sys.modules)
        if name.startswith(_UNSLOTH_COMPILED_MODULE_PREFIX) and name not in names
    )
    for modname in names:
        module = sys.modules.get(modname)
        if module is None:
            continue
        for attribute in _KERNEL_HUB_DECORATED:
            wrapper = getattr(module, attribute, None)
            original = getattr(wrapper, "__wrapped__", None)
            if original is None:
                continue
            if wrapper is original:
                continue
            setattr(module, attribute, original)
            forced.append(f"{modname}.{attribute}")

    if forced and UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: forced the fla kernels on {', '.join(forced)} to the pure-torch "
            f"fallback (no dot instructions on this GPU)."
        )
    return tuple(forced)


def patch_vendor_fla(phase=None):
    """Register the bundled fla kernels and advertise availability.

    Idempotent; safe to call at import time and again from TEMPORARY_PATCHES.
    """
    try:
        return _patch_vendor_fla(phase)
    finally:
        if _gpu_lacks_dot_instructions():
            # RDNA1: never make fla reachable; force the torch fallback, no alias/repair.
            try:
                _force_kernel_hub_fallback()
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(
                        f"Unsloth: could not force fla kernel closures to the pure-torch fallback: {e}"
                    )
        else:
            # In finally so every early return gets the alias; repair resolves through it.
            try:
                _alias_missing_gated_delta_names()
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(f"Unsloth: could not alias gated-delta decode name: {e}")
            try:
                _repair_kernel_hub_closures()
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(f"Unsloth: could not re-resolve fla kernel closures: {e}")


def _patch_vendor_fla(phase=None):
    # Hopper opt-out must also block an installed fla via the probe (unslothai/unsloth#5276).
    optout_degraded = False
    # RDNA1: any reachable fla aborts in LLVM, so never inject, on either layout.
    if _gpu_lacks_dot_instructions():
        _mark_fla_disabled_no_dot_instructions()
        # Sample layout before _patch_is_available creates the probe attribute.
        if _transformers_uses_availability_probe():
            _patch_is_available(_unavailable_probe)
        # post-#47630: _force_kernel_hub_fallback in the finally handles decorators.
        _disable_already_imported_gated_delta(why="no dot instructions on this GPU (RDNA1)")
        # The fp16 torch path needs a float32 l2norm.
        _patch_l2norm_fp32_on_torch_path()
        return
    if _flag("UNSLOTH_DISABLE_HOPPER_FLA_BWD") and _hopper_dqkwg_suspect_here():
        # Sample layout before _patch_is_available creates the probe attribute.
        if _transformers_uses_availability_probe():
            _mark_fla_disabled_hopper()
            _patch_is_available(_unavailable_probe)
            _disable_already_imported_gated_delta()
            return
        # Post-#47630 pure torch is unreachable; fall through so the resolved fla is
        # patched or vendored, never an unpatched BK=64 install.
        _warn_hopper_optout_degraded()
        optout_degraded = True

    if _flag("UNSLOTH_DISABLE_VENDORED_FLA") and not optout_degraded:
        # Correctness (the Hopper opt-out) outranks this source preference.
        return

    replaced_real = False
    if not _vendored_already_injected():
        force = _flag("UNSLOTH_FORCE_VENDORED_FLA")
        if not force and _should_defer_to_installed_fla():
            # Newer user fla: patch it in place on #640 hosts (unslothai/unsloth#5276).
            if not _hopper_dqkwg_suspect_here():
                return
            if _patch_installed_fla_dqkwg():
                # No _patch_is_available: the native probe finds a real install.
                return
        if not _torch_triton_cuda_supported():
            return
        injected, replaced_real = _inject_vendored_fla()
        if not injected:
            return
        if replaced_real and _hopper_dqkwg_suspect_here():
            # Purged real fla on a #640 host: vendor cannot serve olmo_hybrid, unbind it.
            _disable_already_imported_gated_delta(
                packages=_UNCOVERED_GATED_DELTA, why="fla #640; vendored tree cannot serve this model",
            )

    _patch_is_available()
    _repair_already_imported_modeling(force_rebind=replaced_real)


TEMPORARY_PATCHES.append(patch_vendor_fla)

# Early import-time run; UNSLOTH_VENDORED_FLA_NO_AUTORUN=1 skips only this one.
if not _flag("UNSLOTH_VENDORED_FLA_NO_AUTORUN"):
    try:
        patch_vendor_fla()
    except Exception as _e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: early vendored-fla injection deferred: {_e}")
