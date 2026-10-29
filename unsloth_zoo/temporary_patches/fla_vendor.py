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
``UNSLOTH_DISABLE_HOPPER_FLA_BWD=1`` forces pure torch on Hopper + Triton [3.4.0, 3.7.1) (fla #640);
``UNSLOTH_GDN_BACKEND=flashqla`` sends supported chunk_gated_delta_rule calls to FlashQLA (``pip install flash-qla``);
on SM120 only with ``UNSLOTH_FLASHQLA_ALLOW_SM120=1`` (QwenLM/FlashQLA#35);
``UNSLOTH_KDA_SAFE_GATE=0`` keeps GLM-5.3-Flash's bounded KDA gates off fla's safe_gate path.
On ROCm gfx11 float32 chunk_gated_delta_rule calls take transformers' pure-torch reference
(fla's float32 chunk kernels fault there); bfloat16 / float16 and CUDA are untouched.
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
_REPAIR_MODELING = ("qwen3_5", "qwen3_5_moe", "qwen3_next", "qwen4_exp")

# Kimi delta attention consumers; they reach fla only through kernel-hub wrappers.
_KDA_MODELING = ("glm5_next", "kimi_linear")

# Modules whose kernel-hub wrappers are re-resolved against the live fla.
_HUB_REPAIR_MODELING = _REPAIR_MODELING + _KDA_MODELING

# olmo_hybrid is not covered: it has not been validated on the vendored tree.
_VENDOR_COVERED_MODELS = frozenset(_HUB_REPAIR_MODELING)

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
# inf and gives NaN grads in eager mode (RX 5700 XT). fla reduces in float32. Up to at least
# transformers 5.5 torch_chunk_gated_delta_rule calls it before casting to float32, so any host
# that takes the pure-torch path in float16 (no Triton/CUDA, the Hopper opt-out, RDNA1) needs it;
# later releases cast first, where this is a no-op in effect.
_L2NORM_FP32_MARK = "_unsloth_fp32_l2norm"


def _fp32_l2norm(x, dim = -1, eps = 1e-6):
    """``l2norm`` with the reduction in at least float32, result in the input dtype: what fla's kernel does."""
    import torch

    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
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


def _withhold_kda_on_rocm():
    """ROCm: Triton's AMD pipeline pass fails compiling the KDA kernels (gfx1151, triton 3.6), so
    fla.ops.kda stays unimportable and the kernel-hub wrappers keep the torch fallback."""
    try:
        import torch
        if getattr(torch.version, "hip", None) is None:
            return False
    except Exception:
        return False
    sys.modules["fla.ops.kda"] = None
    return True


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
            _withhold_kda_on_rocm()
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
    """True only for callers the vendored tree covers (olmo_hybrid is not validated on it)."""
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


# GLM-5.3-Flash (glm5_next) bounds its KDA forget gate: Glm5NextTextForgetGate returns
# config.linear_lower_bound * sigmoid(.), in [linear_lower_bound, 0] (-5 in the released config).
# For such gates fla's chunk_kda(safe_gate=True) computes the 16x16 intra-chunk diagonal blocks on
# tensor cores instead of token by token (1.2x KDA fwd + bwd on A100 at 64 heads, T = 8192).
# Kimi-Linear's gate is unbounded (-exp(A_log) * softplus(.)), so only glm5_next is patched: each
# Glm5NextTextLinearAttention records config.linear_lower_bound at __init__, and while every
# glm5_next layer built so far shares one bound in fla's accepted [-5, 0), the fla kernel behind
# glm5_next's kernel-hub chunk_kimi_delta_attention becomes functools.partial(chunk_kda,
# safe_gate = True, lower_bound = bound). The call path keeps its frames (no new graph breaks).
# Only on the vendored tree (it carries fla #1242 and #1554, without which the safe_gate dg error
# is 5e-2 to 2.6e-1), Triton >= 3.6 (bf16x3 backward dots) and SM80+ (T4 is slower with it).
# UNSLOTH_KDA_SAFE_GATE=0 disables (read when a layer is built).
_GLM5_NEXT_MODULE = "transformers.models.glm5_next.modeling_glm5_next"
_KDA_SAFE_GATE_MARK = "_unsloth_kda_safe_gate"
_KDA_SAFE_GATE_INNER = "_unsloth_kda_inner"
_KDA_SAFE_GATE_RANGE = (-5.0, 0.0)  # fla chunk_kda only accepts lower_bound in [-5, 0)
# bound: the engaged lower bound or None; bounds: every glm5_next layer's; layers: engaged builds.
_kda_safe_gate_state = {"bound": None, "bounds": set(), "layers": 0}


def _kda_safe_gate_hardware_ok():
    """Vendored fla live, Triton >= 3.6, CUDA (not ROCm) and every visible GPU SM80+."""
    if not _vendored_already_injected():
        return False
    try:
        import torch
        from fla.utils import TRITON_ABOVE_3_6_0
        if not TRITON_ABOVE_3_6_0 or getattr(torch.version, "hip", None) is not None:
            return False
        n = torch.cuda.device_count()
        return n > 0 and all(torch.cuda.get_device_capability(i) >= (8, 0) for i in range(n))
    except Exception:
        return False


def _hub_implementation_cell(wrapper):
    """The ``implementation`` closure cell of a kernel-hub fallback wrapper, or None."""
    code = getattr(wrapper, "__code__", None)
    for name, cell in zip(getattr(code, "co_freevars", ()), getattr(wrapper, "__closure__", None) or ()):
        if name == "implementation":
            return cell
    return None


def _glm5_modules():
    names = [_GLM5_NEXT_MODULE] + sorted(
        name for name in list(sys.modules)
        if name.startswith(_UNSLOTH_COMPILED_MODULE_PREFIX) and "glm5_next" in name
    )
    return [sys.modules[n] for n in names if sys.modules.get(n) is not None]


def _apply_glm5_kda_safe_gate(bound):
    """Point glm5_next's chunk_kimi_delta_attention at chunk_kda with (bound) or without (None) safe_gate."""
    applied = 0
    for module in _glm5_modules():
        wrapper = getattr(module, "chunk_kimi_delta_attention", None)
        cell = _hub_implementation_cell(wrapper)
        if cell is None:
            continue
        try:
            current = cell.cell_contents
        except ValueError:
            continue
        base = getattr(current, _KDA_SAFE_GATE_INNER, current)
        # Only the fla kernel: the torch fallback (wrapper.__wrapped__) has no safe_gate.
        if base is getattr(wrapper, "__wrapped__", None) or "safe_gate" not in inspect.signature(base).parameters:
            continue
        if bound is None:
            new = base
        elif getattr(current, _KDA_SAFE_GATE_MARK, None) == bound:
            new = current
        else:
            new = functools.partial(base, safe_gate = True, lower_bound = bound)
            setattr(new, _KDA_SAFE_GATE_MARK, bound)
            setattr(new, _KDA_SAFE_GATE_INNER, base)
        if new is not current:
            cell.cell_contents = new
        applied += 1
    return applied


def _glm5_record_kda_bound(bound):
    """Called from Glm5NextTextLinearAttention.__init__ with config.linear_lower_bound."""
    state = _kda_safe_gate_state
    valid = not isinstance(bound, bool) and isinstance(bound, (int, float))
    low, high = _KDA_SAFE_GATE_RANGE
    state["bounds"].add(float(bound) if valid else None)
    (only,) = state["bounds"] if len(state["bounds"]) == 1 else (None,)
    enabled = (
        only is not None and low <= only < high
        and os.environ.get("UNSLOTH_KDA_SAFE_GATE", "1") != "0"
        and _kda_safe_gate_hardware_ok()
    )
    state["bound"] = only if enabled else None
    if _apply_glm5_kda_safe_gate(state["bound"]) and enabled:
        state["layers"] += 1
        if UNSLOTH_ENABLE_LOGGING and state["layers"] == 1:
            logger.info(f"Unsloth: GLM-5.3-Flash KDA gates are bounded below by {only}; using fla safe_gate.")


def _install_glm5_kda_safe_gate():
    """Make Glm5NextTextLinearAttention.__init__ record its bound. The modeling module is imported
    here (~0.1 s) once transformers' modeling_utils is loaded, so the patch never depends on
    whether the model class was imported before or after this runs."""
    module = sys.modules.get(_GLM5_NEXT_MODULE)
    if module is None:
        if "transformers.modeling_utils" not in sys.modules:
            return False
        try:
            if importlib.util.find_spec("transformers.models.glm5_next") is None:
                return False
            module = importlib.import_module(_GLM5_NEXT_MODULE)
        except Exception:
            return False
    cls = getattr(module, "Glm5NextTextLinearAttention", None)
    if cls is None:
        return False
    if not getattr(cls.__init__, _KDA_SAFE_GATE_MARK, False):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def __init__(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            _glm5_record_kda_bound(getattr(config, "linear_lower_bound", None))

        setattr(__init__, _KDA_SAFE_GATE_MARK, True)
        cls.__init__ = __init__
    # A repair may have rebuilt the wrapper: re-apply the current choice to it.
    _apply_glm5_kda_safe_gate(_kda_safe_gate_state["bound"])
    return True


# Kernel-hub decorated wrapper -> (fla module, kernel). causal_conv1d is not vendored.
_KERNEL_HUB_DECORATED = {
    "torch_chunk_gated_delta_rule": ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    "torch_recurrent_gated_delta_rule": ("fla.ops.gated_delta_rule", "recurrent_gated_delta_rule"),
    "chunk_kimi_delta_attention": ("fla.ops.kda", "chunk_kda"),
    "recurrent_kimi_delta_attention": ("fla.ops.kda", "fused_recurrent_kda"),
}


def _resolved_implementation(wrapper):
    """The callable a kernel-hub wrapper will actually dispatch to, or None."""
    for cell in getattr(wrapper, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value):
            # glm5_next's safe_gate partial: report the fla kernel it wraps.
            return getattr(value, _KDA_SAFE_GATE_INNER, value)
    return None


def _decorated_kernel_name(wrapper, default):
    code = getattr(wrapper, "__code__", None)
    closure = getattr(wrapper, "__closure__", None) or ()
    if code is None:
        return default
    for name, cell in zip(code.co_freevars, closure):
        if name != "func_name":
            continue
        try:
            value = cell.cell_contents
        except ValueError:
            return default
        return value if isinstance(value, str) and value else default
    return default


def _live_fla_kernel(module_name, name):
    """The kernel ``name`` currently resolves to on the live fla, or None."""
    module = sys.modules.get(module_name)
    return getattr(module, name, None) if module is not None else None


def _repair_kernel_hub_closures(packages=_HUB_REPAIR_MODELING):
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
        for attribute, (kernel_module, kernel) in _KERNEL_HUB_DECORATED.items():
            wrapper = getattr(module, attribute, None)
            original = getattr(wrapper, "__wrapped__", None)
            if original is None:
                continue
            # Keep the model's own name (qwen4_exp uses "fused_recurrent_gated_delta_rule").
            kernel = _decorated_kernel_name(wrapper, kernel)
            current = _resolved_implementation(wrapper)
            live = _live_fla_kernel(kernel_module, kernel)
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


_GDN_BACKEND_MARK = "_unsloth_gdn_backend"
_FLASHQLA_ARCHES = ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1))
_flashqla_state = {"module": None, "capability": None, "failed": False, "calls": 0, "skipped": set()}


def _gdn_backend():
    return os.environ.get("UNSLOTH_GDN_BACKEND", "").strip().lower()


def _flashqla_version(flash_qla):
    """Version of the imported flash_qla: the distribution's only if it owns the imported file
    (a checkout earlier on sys.path can shadow an installed flash-qla)."""
    version = getattr(flash_qla, "__version__", None)
    if version:
        return version
    try:
        import importlib.metadata as _md
        dist = _md.distribution("flash-qla")
        origin = os.path.realpath(flash_qla.__file__)
        if any(os.path.realpath(dist.locate_file(f)) == origin for f in dist.files or ()):
            return dist.version
    except Exception:
        pass
    return None


def _load_flashqla():
    """flash_qla, or None. Its import picks kernels for the current device and raises on others."""
    if _flashqla_state["module"] is not None:
        return _flashqla_state["module"]
    if _flashqla_state["failed"]:
        return None
    try:
        import torch
        if getattr(torch.version, "hip", None) is not None or not torch.cuda.is_available():
            raise RuntimeError("FlashQLA needs an NVIDIA GPU")
        capability = tuple(torch.cuda.get_device_capability())
        if capability not in _FLASHQLA_ARCHES:
            raise RuntimeError(f"FlashQLA does not support compute capability {capability}")
        # Its SM120 forward can return garbage (errors ~1e17) on some drivers with grouped value
        # heads, varlen and padded alike, unfixed upstream (QwenLM/FlashQLA#35).
        if capability[0] == 12 and os.environ.get("UNSLOTH_FLASHQLA_ALLOW_SM120", "0") != "1":
            raise RuntimeError(
                "its SM120 forward can return wrong outputs on some drivers (QwenLM/FlashQLA#35); "
                "set UNSLOTH_FLASHQLA_ALLOW_SM120=1 to use it anyway"
            )
        import flash_qla
        # 0.1.1's id()-keyed tensor cache can reuse stale chunk indices for packed batches (fixed
        # in 0.1.2); 0.1.0 lacks state_v_first. An unversioned build cannot be told apart.
        installed = _flashqla_version(flash_qla)
        if installed is None or not _version_at_least(installed, "0.1.2"):
            raise RuntimeError(f"flash-qla >= 0.1.2 is required (found {installed or 'an unversioned build'})")
    except Exception as e:
        _flashqla_state["failed"] = True
        logger.warning(
            f"Unsloth: UNSLOTH_GDN_BACKEND=flashqla is set but FlashQLA is unavailable ({e}); "
            f"using the fla Triton kernels. Install it with `pip install flash-qla`."
        )
        return None
    _flashqla_state["module"] = flash_qla
    _flashqla_state["capability"] = capability
    return flash_qla


def _flashqla_unsupported(q, k, v, g, beta, initial_state, cu_seqlens, cp_context, extra):
    """Why FlashQLA cannot serve this call, or None. Mirrors fla's own FlashQLA backend
    verifier (fla/ops/gated_delta_rule/backends/flash_qla.py)."""
    import torch
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return "not CUDA"
    capability = tuple(torch.cuda.get_device_capability(q.device))
    if capability != _flashqla_state["capability"]:
        return f"compute capability {capability} differs from the one FlashQLA was loaded for"
    if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        return "dtype"
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        return "head dim is not 128"
    if v.shape[2] % k.shape[2] != 0:
        return "value heads not a multiple of key heads"
    if cu_seqlens is not None and q.shape[0] != 1:
        return "cu_seqlens with batch > 1"
    if cp_context is not None:
        return "context parallel"
    if extra.get("use_gate_in_kernel") or extra.get("use_beta_sigmoid_in_kernel") or extra.get("allow_neg_eigval"):
        return "in-kernel gate / beta sigmoid / negative eigenvalues"
    if "transpose_state_layout" in extra or extra.get("head_first"):
        return "deprecated layout argument"
    if capability[0] == 12:
        if q.dtype == torch.float16:
            return "float16 on SM120"
        needs_backward = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)
        )
        if needs_backward:
            # FlashQLA < 0.1.3 has no SM120 backward kernel.
            try:
                from flash_qla.ops.gated_delta_rule import chunk as _chunk
                if getattr(_chunk, "fused_gdr_bwd", None) is None:
                    return "no backward kernel on SM120"
            except Exception:
                return "no backward kernel on SM120"
    return None


def _make_flashqla_chunk_gated_delta_rule(original):
    """Dispatch eligible calls to FlashQLA, everything else to ``original``. functools.wraps
    keeps ``inspect.signature`` on the original: transformers' kernel-hub wrapper filters
    kwargs by it."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def chunk_gated_delta_rule(*args, **kwargs):
        flash_qla = _flashqla_state["module"]
        if flash_qla is None:
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return original(*args, **kwargs)
        a = bound.arguments
        extra = dict(a.get("kwargs", {}))
        # Named parameters of the original land in `a`, not in its **kwargs: the verifier sees both.
        for name in (
            "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval",
            "head_first", "transpose_state_layout",
        ):
            if name in a:
                extra[name] = a[name]
        q, k, v, g, beta = (a.get(name) for name in ("q", "k", "v", "g", "beta"))
        initial_state = a.get("initial_state")
        cu_seqlens = a.get("cu_seqlens")
        reason = _flashqla_unsupported(
            q, k, v, g, beta, initial_state, cu_seqlens, a.get("cp_context"), extra,
        )
        if reason is not None:
            if UNSLOTH_ENABLE_LOGGING and reason not in _flashqla_state["skipped"]:
                _flashqla_state["skipped"].add(reason)
                logger.info(f"Unsloth: FlashQLA skipped ({reason}); using fla.")
            return original(*args, **kwargs)
        # A toolchain can accept a call and still fail to build its kernel (fp16 on SM100 with
        # TileLang 0.1.12 and nvcc 13.1): fall back to fla, and stop retrying that dtype.
        dtype = getattr(q, "dtype", None)
        failed_key = (_flashqla_state["capability"], dtype)
        failed_calls = _flashqla_state.setdefault("failed_calls", set())
        if failed_key in failed_calls:
            return original(*args, **kwargs)
        _flashqla_state["calls"] += 1
        try:
            return flash_qla.chunk_gated_delta_rule(
                q = q,
                k = k,
                v = v,
                g = g,
                beta = beta,
                scale = a.get("scale"),
                initial_state = initial_state,
                output_final_state = a.get("output_final_state", False),
                use_qk_l2norm_in_kernel = a.get("use_qk_l2norm_in_kernel", False),
                cu_seqlens = cu_seqlens,
                state_v_first = a.get("state_v_first", False),
            )
        except Exception as e:
            import torch
            if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                raise
            failed_calls.add(failed_key)
            # A build failure dumps the generated source: report its first compiler error.
            lines = [line.strip() for line in str(e).splitlines() if line.strip()]
            message = next((line for line in lines if ": error:" in line), lines[0] if lines else "")
            logger.warning(
                f"Unsloth: FlashQLA failed on {dtype} ({type(e).__name__}: {message[:200]}); "
                f"using the fla Triton kernels for {dtype}."
            )
        # Outside the handler: the traceback, and the failed FlashQLA buffers its frames
        # hold, are released before fla allocates its own.
        return original(*args, **kwargs)

    setattr(chunk_gated_delta_rule, _GDN_BACKEND_MARK, "flashqla")
    return chunk_gated_delta_rule


def _install_gdn_backend():
    """``UNSLOTH_GDN_BACKEND=flashqla``: route the live fla's chunk_gated_delta_rule through
    FlashQLA where it applies. Unset: nothing is touched. Idempotent."""
    if _gdn_backend() != "flashqla":
        return False
    module = sys.modules.get("fla.ops.gated_delta_rule")
    original = getattr(module, "chunk_gated_delta_rule", None) if module is not None else None
    if original is None:
        return False
    if getattr(original, _GDN_BACKEND_MARK, None) is not None:
        return True
    if _load_flashqla() is None:
        return False
    try:
        import torch
        wrapped = torch.compiler.disable(_make_flashqla_chunk_gated_delta_rule(original))
    except Exception:
        wrapped = _make_flashqla_chunk_gated_delta_rule(original)
    setattr(wrapped, _GDN_BACKEND_MARK, "flashqla")
    module.chunk_gated_delta_rule = wrapped
    # Pre-#47630 modeling modules bound the original as a global at import.
    for modname in _l2norm_modules(_GATED_DELTA_MODELING):
        mod = sys.modules.get(modname)
        if mod is not None and getattr(mod, "chunk_gated_delta_rule", None) is original:
            setattr(mod, "chunk_gated_delta_rule", wrapped)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: gated-delta chunk kernel routed through FlashQLA (UNSLOTH_GDN_BACKEND=flashqla).")
    return True


# ROCm gfx11 (RDNA3 / RDNA3.5): fla's chunked gated-delta kernels built for float32 inputs
# fault on gfx1151 (Radeon 8060S, Triton 3.6 / ROCm 7.13): chunk_gated_delta_rule_fwd_kernel_h_blockdim64
# hits an illegal memory access and prepare_wy_repr_bwd_kernel hangs, with every config that fits
# in 64 KB of LDS. The same kernels in bfloat16 / float16 run and match an fp64 reference, so only
# float32 calls leave fla: they take transformers' pure-torch chunked reference. Never on CUDA.
_ROCM_FP32_TORCH_PATH_GFX = "gfx11"
_ROCM_FP32_MARK = "_unsloth_rocm_fp32_torch_path"
_rocm_fp32_state = {"arch": {}, "reference": None, "calls": 0, "logged": False}


def _hip_gfx_arch(torch_mod, index):
    """gcnArchName of HIP device ``index`` without feature flags ("gfx1151"), or "" if unknown."""
    try:
        props = torch_mod.cuda.get_device_properties(index)
        return str(getattr(props, "gcnArchName", "") or "").split(":", 1)[0].strip().lower()
    except Exception:
        return ""


def _rocm_has_fp32_faulting_gpu(torch_mod=None):
    """ROCm and some visible GPU is gfx11; CUDA, CPU or unreadable -> False."""
    try:
        if torch_mod is None:
            import torch as torch_mod
        if getattr(getattr(torch_mod, "version", None), "hip", None) is None:
            return False
        if not torch_mod.cuda.is_available():
            return False
        return any(
            _hip_gfx_arch(torch_mod, i).startswith(_ROCM_FP32_TORCH_PATH_GFX)
            for i in range(int(torch_mod.cuda.device_count()))
        )
    except Exception:
        return False


def _rocm_fp32_on_faulting_gpu(*tensors):
    """Any float32 tensor among ``tensors`` sits on a ROCm gfx11 device."""
    import torch
    if getattr(torch.version, "hip", None) is None:
        return False
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.device.type != "cuda":
            continue
        index = t.device.index if t.device.index is not None else torch.cuda.current_device()
        arch = _rocm_fp32_state["arch"].get(index)
        if arch is None:
            arch = _rocm_fp32_state["arch"][index] = _hip_gfx_arch(torch, index)
        if arch.startswith(_ROCM_FP32_TORCH_PATH_GFX):
            return True
    return False


def _torch_reference_chunk_gated_delta_rule():
    """transformers' pure-torch ``torch_chunk_gated_delta_rule`` (the kernel-hub fallback), unwrapped."""
    if _rocm_fp32_state["reference"] is not None:
        return _rocm_fp32_state["reference"]
    fn = None
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule as fn
    except Exception:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as fn
        except Exception:
            try:
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule as fn
            except Exception:
                fn = None
    if fn is not None:
        _rocm_fp32_state["reference"] = inspect.unwrap(fn)
        return _rocm_fp32_state["reference"]
    raise RuntimeError("Unsloth: transformers has no torch_chunk_gated_delta_rule to run float32 gated delta on ROCm gfx11.")


def _rocm_fp32_chunk_gated_delta_rule(
    q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
    use_qk_l2norm_in_kernel = False, use_beta_sigmoid_in_kernel = False, allow_neg_eigval = False,
    state_v_first = False, cu_seqlens = None, cu_seqlens_cpu = None, cp_context = None, **kwargs,
):
    """fla's ``chunk_gated_delta_rule`` contract computed by transformers' torch reference."""
    import torch
    if "transpose_state_layout" in kwargs:
        # fla's deprecated name for state_v_first, with fla's conflict check.
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        state_v_first = kwargs.pop("transpose_state_layout")
    if cp_context is not None or kwargs.get("use_gate_in_kernel") or kwargs.get("head_first"):
        raise RuntimeError(
            "Unsloth: float32 gated delta on ROCm gfx11 runs in pure torch, which does not support "
            "context parallel, use_gate_in_kernel or head_first. Use bfloat16 or float16."
        )
    reference = _torch_reference_chunk_gated_delta_rule()
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    groups = v.shape[2] // q.shape[2]
    if groups > 1:
        q = q.repeat_interleave(groups, dim = 2)
        k = k.repeat_interleave(groups, dim = 2)
    if use_qk_l2norm_in_kernel:
        q = _fp32_l2norm(q)
        k = _fp32_l2norm(k)
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5
    # The reference multiplies q by head_dim ** -0.5 itself.
    q = q * (scale * head_dim ** 0.5)
    if initial_state is not None and state_v_first:
        initial_state = initial_state.transpose(-1, -2)

    def run(qs, ks, vs, gs, bs, h0):
        return reference(
            qs, ks, vs, gs, bs, initial_state = h0, output_final_state = output_final_state,
            use_qk_l2norm_in_kernel = False,
        )

    if cu_seqlens is None:
        o, final_state = run(q, k, v, g, beta, initial_state)
    else:
        bounds = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
        outs, states = [], []
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            h0 = initial_state[i : i + 1] if initial_state is not None else None
            if end <= start:
                states.append(h0.float() if h0 is not None else q.new_zeros(
                    1, v.shape[2], head_dim, v.shape[-1], dtype = torch.float32))
                continue
            o_i, h_i = run(*(t[:, start:end] for t in (q, k, v, g, beta)), h0)
            outs.append(o_i)
            states.append(h_i)
        o = torch.cat(outs, dim = 1) if outs else v.new_zeros(v.shape)
        final_state = torch.cat(states, dim = 0) if output_final_state else None
    if final_state is not None and state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return o, final_state


def _make_rocm_fp32_chunk_gated_delta_rule(original):
    """float32 on a ROCm gfx11 GPU -> torch reference; everything else -> ``original``, untouched."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def chunk_gated_delta_rule(*args, **kwargs):
        try:
            a = signature.bind(*args, **kwargs).arguments
        except TypeError:
            return original(*args, **kwargs)
        if not _rocm_fp32_on_faulting_gpu(a.get("q"), a.get("k"), a.get("v")):
            return original(*args, **kwargs)
        _rocm_fp32_state["calls"] += 1
        if not _rocm_fp32_state["logged"]:
            _rocm_fp32_state["logged"] = True
            logger.info(
                "Unsloth: float32 gated-delta (linear attention) runs in pure torch on this AMD GPU: "
                "fla's float32 kernels fault on gfx11. bfloat16 / float16 keep the fast kernels."
            )
        extra = dict(a.pop("kwargs", {}))
        return _rocm_fp32_chunk_gated_delta_rule(**a, **extra)

    setattr(chunk_gated_delta_rule, _ROCM_FP32_MARK, "torch")
    return chunk_gated_delta_rule


def _make_rocm_fp32_guard(original, what):
    """Refuse float32 on ROCm gfx11 before any kernel launches: a direct caller of fla's internals
    (unsloth's opaque gated-delta op self-checks through them) gets an exception, not a dead GPU."""

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        tensors = list(args[:3]) + [kwargs.get(name) for name in ("q", "k", "v")]
        if _rocm_fp32_on_faulting_gpu(*tensors):
            raise RuntimeError(
                f"Unsloth: fla {what} does not run float32 on ROCm gfx11 (its kernels fault); "
                f"call fla.ops.gated_delta_rule.chunk_gated_delta_rule, which uses pure torch for float32."
            )
        return original(*args, **kwargs)

    setattr(guarded, _ROCM_FP32_MARK, "guard")
    return guarded


def _install_rocm_fp32_torch_path(torch_mod=None):
    """ROCm with a gfx11 GPU: route the live fla's float32 chunk_gated_delta_rule calls to the
    torch reference and guard its float32 internals. Anything else: nothing is touched. Idempotent."""
    if not _rocm_has_fp32_faulting_gpu(torch_mod):
        return False
    package = sys.modules.get("fla.ops.gated_delta_rule")
    chunk_mod = sys.modules.get("fla.ops.gated_delta_rule.chunk")
    original = getattr(package, "chunk_gated_delta_rule", None) if package is not None else None
    if original is None:
        return False
    if getattr(original, _ROCM_FP32_MARK, None) is not None:
        return True
    try:
        import torch
        wrapped = torch.compiler.disable(_make_rocm_fp32_chunk_gated_delta_rule(original))
    except Exception:
        wrapped = _make_rocm_fp32_chunk_gated_delta_rule(original)
    setattr(wrapped, _ROCM_FP32_MARK, "torch")
    package.chunk_gated_delta_rule = wrapped
    if chunk_mod is not None:
        if getattr(chunk_mod, "chunk_gated_delta_rule", None) is original:
            chunk_mod.chunk_gated_delta_rule = wrapped
        for name in ("chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_bwd"):
            fn = getattr(chunk_mod, name, None)
            if fn is not None and getattr(fn, _ROCM_FP32_MARK, None) is None:
                setattr(chunk_mod, name, _make_rocm_fp32_guard(fn, name))
    # Modeling modules (and unsloth's compiled copies) that bound the original at import.
    for modname in _l2norm_modules(_GATED_DELTA_MODELING):
        mod = sys.modules.get(modname)
        if mod is not None and getattr(mod, "chunk_gated_delta_rule", None) is original:
            setattr(mod, "chunk_gated_delta_rule", wrapped)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: float32 gated-delta calls on ROCm gfx11 take the pure-torch path.")
    return True


_NO_FLA_HUB_MARK = "_unsloth_rdna1_no_fla"


def _block_fla_hub_decorator(packages=_GATED_DELTA_MODELING + _KDA_MODELING):
    """RDNA1: later kernel-hub decorations bind the torch function; the decorator resolves fla when applied, and unsloth's compiler re-applies it after the last patch phase."""
    try:
        from transformers.integrations import hub_kernels
    except Exception:
        return False
    original = getattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", None)
    if original is None:
        return False
    if getattr(original, _NO_FLA_HUB_MARK, False):
        patched = original
    else:
        @functools.wraps(original)
        def patched(func_name, package, *args, **kwargs):
            if package == "fla":
                return lambda torch_function: torch_function
            return original(func_name, package, *args, **kwargs)
        setattr(patched, _NO_FLA_HUB_MARK, True)
        hub_kernels.use_kernel_func_from_hub_with_fallback = patched
    holders = ["transformers.integrations"] + _l2norm_modules(packages)
    for modname in holders:
        module = sys.modules.get(modname)
        if module is None or modname != "transformers.integrations" and (
            "use_kernel_func_from_hub_with_fallback" not in vars(module)
        ):
            continue
        try:
            module.use_kernel_func_from_hub_with_fallback = patched
        except Exception:
            continue
    return True


def _force_kernel_hub_fallback(packages=_GATED_DELTA_MODELING + _KDA_MODELING):
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
        # Whichever way the kernels were resolved, the pure-torch fallback is what runs when
        # they are not, and it must not overflow in float16. Only that fallback calls l2norm.
        try:
            _patch_l2norm_fp32_on_torch_path()
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: could not make the pure-torch l2norm reduce in float32: {e}")
        if _gpu_lacks_dot_instructions():
            # RDNA1: never make fla reachable; force the torch fallback, no alias/repair.
            try:
                _block_fla_hub_decorator()
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
                _install_rocm_fp32_torch_path()
            except Exception as e:
                logger.warning(f"Unsloth: could not route float32 gated delta to pure torch on ROCm gfx11: {e}")
            try:
                _install_gdn_backend()
            except Exception as e:
                logger.warning(f"Unsloth: could not enable UNSLOTH_GDN_BACKEND: {e}")
            try:
                _repair_kernel_hub_closures()
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(f"Unsloth: could not re-resolve fla kernel closures: {e}")
            # After the repair, which may rebuild glm5_next's chunk_kimi_delta_attention.
            try:
                _install_glm5_kda_safe_gate()
            except Exception as e:
                if UNSLOTH_ENABLE_LOGGING:
                    logger.warning(f"Unsloth: could not route GLM-5.3-Flash KDA to fla safe_gate: {e}")


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
