# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Opt-in `kernels-community/sonic-moe` backend for the MoE expert GEMMs.

Selected only by ``UNSLOTH_MOE_BACKEND=sonic_moe``; never auto-selected, so with the
variable unset ``select_moe_backend()`` is byte-for-byte what it was before.

This module must stay import-safe with no CUDA, no `kernels`, no `cutlass` and no
sonic-moe: nothing at module scope touches the kernel, and every probe is wrapped.

Which GEMM goes where, and why (B200 / sm100, E=256 H=2048 I=512 topk=8, bf16,
median of 15 timed batches; see `scripts/sonic_probe.py`, `scripts/sonic_probe2.py`):

    op                      shape                 torch      sonic     ratio
    gate_up forward         [M,H]x[E,H,2I]        0.248 ms   0.190 ms   1.31x   -> sonic
    down forward            [M,I]x[E,I,H]         0.149 ms   0.156 ms   0.96x   -> sonic (env)
    dX  (through gate_up)   [M,2I]x[E,2I,H]       0.262 ms   0.192 ms   1.36x   -> sonic
    dH  (through down)      [M,H]x[E,H,I]         0.132 ms   0.156 ms   0.85x   -> torch
    LoRA A  (H -> r=8)      [M,H]x[E,H,r]         0.053 ms   0.159 ms   0.33x   -> torch

dH and the rank-r LoRA GEMMs are regressions, so they keep ``torch._grouped_mm``.

The fused ``gemm_gated`` (gate_up GEMM + SwiGLU in one kernel) runs on the HF weight
with NO copy (`_SonicGatedZeroCopy`). What the first integration measured, and why it
no longer applies (B200, kernels-community/sonic-moe v2 = quack 0.6.1):

  1. "concat_layout=("B",) gives garbage, rel err 1.41". It does not: quack rewrites a
     concat operand's NON-contiguous dim as interleaved. On the K-major view
     ``W.transpose(-2, -1)`` of the HF [E, 2I, H] storage that dim is 2I and the result
     matches an interleaved reference to bf16 rounding (rel 1.7e-3). Its PRE-activation
     comes back interleaved; compared against a chunked reference that is rel sqrt(2)
     = 1.41, which is the number that was seen. A contiguous [E, H, 2I] B instead gets
     its contraction dim interleaved (1.41 against both orders). dX passes concat on
     the [E, 2I, H] storage (its K is then interleaved); "C" in concat_layout
     interleaves a row-major C's M dim (rel 0.19), so the LoRA delta is produced
     interleaved instead. Details at `_SonicGatedZeroCopy`.
  2. "A cached interleaved copy is a FULL DUPLICATE of gate_up" (1.07 GiB per layer on
     Qwen3.5-35B-A3B, 43 GiB over 40 layers). Not needed any more; the legacy copy
     path stays behind ``UNSLOTH_MOE_SONIC_ZERO_COPY=0`` (plus a
     ``UNSLOTH_MOE_SONIC_CACHE_GIB`` budget) and is the automatic fallback when the
     installed build has no working concat_layout or the weight is not K-major.
  3. "gemm_gated(tuned=True) is nondeterministically wrong". Reproduced, and it is the
     config, not the shape: every sm100 config with tile_m=128 and cluster_m=2 returns
     different bytes for identical calls (post-activation rel err ~1.0), and the CLC
     (dynamic persistent) scheduler intermittently leaves output tiles unwritten. The
     autotuner keys on shapes and stride classes only and picks by speed, so it can
     land on either. Nothing autotunes by default any more (``UNSLOTH_MOE_SONIC_GEMM_TUNED=1``
     re-enables it for the plain GEMMs only): every sonic call runs one pinned
     config (`_pinned_config`: 256x256, cluster 2x1, static scheduler), which was
     correct and bitwise reproducible on every shape, op and repeat tested.

``UNSLOTH_MOE_SONIC_GATED=0`` turns the fusion off (plain GEMM + eager SwiGLU).
"""
import os
import weakref
import sys
import threading
import torch
import torch.nn.functional as F

__all__ = [
    "sonic_moe_available",
    "sonic_moe_unavailable_reason",
    "forward_sonic_moe",
    "reset_sonic_moe_state",
]

# sm90 (Hopper) and sm100 (Blackwell) are the architectures sonic-moe builds kernels
# for. Anything else must not reach it.
_SUPPORTED_MAJORS = (9, 10)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

_SONIC_MODULE = None
_SONIC_REASON = None
_SONIC_PROBED = False


def _log(message):
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print(message)


def reset_sonic_moe_state():
    """Forget the availability probe (tests; a device change)."""
    global _SONIC_MODULE, _SONIC_REASON, _SONIC_PROBED, _SONIC_DONE, _GEMM_INTERFACE, _CONCAT_SUPPORTED
    global _HOME_CAPABILITY
    _SONIC_MODULE = None
    _SONIC_REASON = None
    _SONIC_PROBED = False
    _SONIC_DONE = False
    _GEMM_INTERFACE = None
    _CONCAT_SUPPORTED = None
    _DEVICE_OK.clear()
    _HOME_CAPABILITY = None
    _INTERLEAVE_CACHE.clear()
    _CONFIG_CACHE.clear()
    _PINNED_CONFIG.clear()


def _apply_cutlass_dsl_shim():
    """sonic-moe declares an unpinned `nvidia-cutlass-dsl` and still references
    `cutlass.cute.core.ThrMma` / `ThrCopy`, which moved up to `cutlass.cute` in 4.6.x.
    Re-alias the classes back onto `cutlass.cute.core`.

    Only *classes* are aliased. Aliasing functions and submodules too makes
    `isinstance(x, cute.core.<name>)` raise "arg 2 must be a type", a worse failure
    than the one being fixed.
    """
    import inspect
    import cutlass.cute as _cute

    core = _cute.core
    for name in dir(_cute):
        if name.startswith("_") or hasattr(core, name):
            continue
        obj = getattr(_cute, name)
        if inspect.isclass(obj):
            setattr(core, name, obj)


def _install_sonic_module_path(module):
    """Make the kernel importable by its internal module id from a subprocess.

    sonic-moe autotunes in worker subprocesses that re-import the kernel by the
    synthetic module name `kernels.get_kernel` gave it (e.g. `_sonic_moe_cuda_83d1d6e`).
    That name is not on `sys.path` for a fresh interpreter, so every worker dies on
    import and the autotuner silently reports 76/76 failed configs and falls back to a
    slow default. Drop a symlink named after the module next to the compile cache and
    put it on PYTHONPATH so the workers can find it.
    """
    try:
        name = module.__name__.rsplit(".", 1)[-1]
        build_dir = os.path.dirname(os.path.abspath(module.__file__))
        root = os.path.join(
            os.path.abspath(os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")),
            "sonic_moe_modpath",
        )
        os.makedirs(root, exist_ok=True)
        link = os.path.join(root, name)
        if not os.path.exists(link):
            os.symlink(build_dir, link)
        existing = os.environ.get("PYTHONPATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if root not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([root] + parts)
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception as exc:                                  # never fatal
        _log(f"Unsloth: sonic-moe module path setup skipped ({exc}).")


def _device_is_supported():
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major not in _SUPPORTED_MAJORS:
        return False, f"compute capability {major}.{minor} is not sm90/sm100"
    return True, None


# CUDA device index -> whether sonic may run there. The kernel and the concat_layout
# probes ran on ONE card (the current device at load); only cards of that exact compute
# capability are trusted. Filled for every visible card at load, outside any trace, so a
# compiled region only reads it.
_DEVICE_OK = {}
_HOME_CAPABILITY = None


def _record_device_capabilities():
    global _HOME_CAPABILITY
    try:
        _HOME_CAPABILITY = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
        for index in range(torch.cuda.device_count()):
            _DEVICE_OK[index] = tuple(torch.cuda.get_device_capability(index)) == _HOME_CAPABILITY
    except Exception as exc:
        _log(f"Unsloth: sonic-moe device capability scan failed ({exc}).")


def _device_ok(device):
    """Whether `device` is a CUDA card with the probed card's compute capability."""
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _DEVICE_OK:
        if torch.compiler.is_compiling() or _HOME_CAPABILITY is None:
            return False
        try:
            _DEVICE_OK[index] = tuple(torch.cuda.get_device_capability(index)) == _HOME_CAPABILITY
        except Exception:
            _DEVICE_OK[index] = False
    return _DEVICE_OK[index]


def _probe_sonic(module):
    """One tiny real call. The import succeeding does not mean the kernel runs."""
    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.ones((8, 16), device=device, dtype=torch.bfloat16)
    b = torch.ones((2, 16, 16), device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4, 8], device=device, dtype=torch.int32)
    out = module.functional.gemm(a, b, cu_seqlens_m=cu, tuned=False)
    if out.shape != (8, 16):
        raise RuntimeError(f"sonic-moe probe returned {tuple(out.shape)}, expected (8, 16)")
    # The low-overhead path reaches internals `functional.gemm` hides, so probe it
    # too and make sure it agrees exactly (all-ones inputs: every config gives 16).
    fast = _sonic_gemm(module, a, b, cu)
    if fast.shape != out.shape or not torch.equal(fast, out):
        raise RuntimeError("sonic-moe fast call path disagrees with functional.gemm")


_SONIC_DONE = False
_SONIC_LOCK = threading.RLock()


def get_sonic_module():
    """The sonic-moe kernel module, or None. Probed once, never raises."""
    global _SONIC_DONE
    if _SONIC_DONE:
        return _SONIC_MODULE
    # Other threads wait for the first probe; the probing thread re-enters (custom-op probe path).
    with _SONIC_LOCK:
        if _SONIC_PROBED:  # finished by another thread, or this thread's probe re-entering
            return _SONIC_MODULE
        try:
            return _probe_sonic_module_once()
        finally:
            _SONIC_DONE = True


def _probe_sonic_module_once():
    global _SONIC_MODULE, _SONIC_REASON, _SONIC_PROBED
    _SONIC_PROBED = True

    if os.environ.get("UNSLOTH_DISABLE_MOE_SONIC", "0") == "1":
        _SONIC_REASON = "disabled by UNSLOTH_DISABLE_MOE_SONIC=1"
        return None
    ok, why = _device_is_supported()
    if not ok:
        _SONIC_REASON = why
        return None
    try:
        _apply_cutlass_dsl_shim()
    except Exception as exc:
        _SONIC_REASON = f"nvidia-cutlass-dsl shim failed: {exc}"
        return None
    try:
        from kernels import get_kernel
    except Exception as exc:
        _SONIC_REASON = f"`kernels` is not importable: {exc}"
        return None
    try:
        version = int(os.environ.get("UNSLOTH_MOE_SONIC_VERSION", "2"))
        module = get_kernel("kernels-community/sonic-moe", version=version)
    except Exception as exc:
        _SONIC_REASON = f"could not load kernels-community/sonic-moe: {exc}"
        return None
    _install_sonic_module_path(module)
    # Published before the probe: with UNSLOTH_MOE_SONIC_FASTCALL=0 the probe goes through the
    # custom op, whose eager body calls get_sonic_module() again.
    _SONIC_MODULE = module
    try:
        _probe_sonic(module)
    except Exception as exc:
        _SONIC_MODULE = None
        _SONIC_REASON = f"sonic-moe probe call failed: {exc}"
        return None

    _SONIC_REASON = None
    _record_device_capabilities()
    _log("Unsloth: sonic-moe MoE backend is available.")
    _concat_layout_supported(module)
    try:
        # Register the custom ops now: registering them while Dynamo traces the
        # first compiled forward is a graph break.
        _pinned_ops()
    except Exception as exc:
        _log(f"Unsloth: sonic-moe custom op registration failed ({exc}).")
    return module


def sonic_moe_available() -> bool:
    return get_sonic_module() is not None


def sonic_moe_unavailable_reason():
    get_sonic_module()
    return _SONIC_REASON


def _env_flag(name, default):
    return os.environ.get(name, default) == "1"


# ---------------------------------------------------------------------------
# Low-overhead call path
#
# `module.functional.gemm` is a thin Python wrapper over `gemm_out`, which is a
# `torch.library.custom_op` with EIGHTEEN arguments. Going through that op costs
# 0.13-0.15 ms of HOST time per call on this box (torch 2.13), essentially all of
# it in `torch._library.utils.fill_defaults` walking the 18-entry schema and in
# `_c_check_aliasing_constraint` re-checking every argument. Measured host-only
# enqueue cost per call, E=256 H=2048 I=512, median of 7 x 300 calls
# (`scripts/sonic_host_profile.py`):
#
#     level                                       M=512     M=4096    M=16384
#     functional.gemm (public entry)             0.1716 ms  0.1841 ms  0.1810 ms
#     torch.ops ... gemm_out (custom op)         0.1557     0.1657     0.1689
#     gemm_tuned(...)            (no custom op)  0.0358     0.0379     0.0377
#     gemm_tuned.fn(..., config) (no autotuner)  0.0192     0.0231     0.0201
#     quack.gemm.gemm            (no fn body)    0.0162     0.0209     0.0165
#     compiled_fn(raw tvm-ffi args)              0.0052     0.0059     0.0055
#     torch._grouped_mm (reference)              0.0178     0.0202     0.0184
#
# So the custom-op boundary is ~75% of sonic's host cost and the autotuner's
# per-call key (which `str()`s every tensor's shape, stride and dtype) is another
# ~10%. Neither is inside the kernel. Calling `gemm_tuned.fn` directly with a
# config resolved once per (shape, stride, dtype) brings the per-call host cost
# from 0.18 ms to 0.029 ms, i.e. within 0.011 ms of `torch._grouped_mm`. At 120
# MoE GEMMs per step that is the difference between +19 ms and +1.3 ms of host
# time on the critical path, against a ~7 ms GPU saving.
#
# Every level above was verified bit-identical (rel max err 0.000e+00) to the
# public entry point at every M.
#
# Under `torch.compiler.is_compiling()` the same pinned call goes through a
# registered, fake-implemented custom op (`_pinned_ops`), so tracing sees one op
# and runs the same config as eager.
# ---------------------------------------------------------------------------

_GEMM_INTERFACE = None
_CONFIG_CACHE = {}
# Bounded only so a pathological variable-length workload cannot grow it without
# limit; entries are one small object each and a fixed-shape model uses 2-3.
_CONFIG_CACHE_MAX = 1024


def _gemm_interface(module):
    """The live `quack.gemm_interface` module object backing `module.functional.gemm`.

    Resolved off the function itself, never re-imported by name: a second import
    under a different synthetic package name would get its own Autotuner instances
    and re-tune every shape.
    """
    global _GEMM_INTERFACE
    if _GEMM_INTERFACE is None:
        _GEMM_INTERFACE = sys.modules[module.functional.gemm.__module__]
    return _GEMM_INTERFACE


def _shape_key(t):
    # Mirrors Autotuner.__call__'s key: shape, strides normalized to {0, 1, other}
    # and dtype. Anything the autotuner would distinguish, this distinguishes.
    return (tuple(t.shape),
            tuple(s if s in (0, 1) else 2 for s in t.stride()),
            t.dtype)


_PINNED_CONFIG = {}


def _pinned_config(module, device):
    """The one GemmConfig every non-autotuned sonic call uses: the kernel's default,
    with the dynamic (CLC) tile scheduler turned off on sm100.

    Measured on the v2 build (quack 0.6.1), B200, varlen_m, NaN-poisoned outputs:
      * every config with tile_m=128 and cluster_m=2 makes gemm_gated return
        different bytes for identical calls (post-activation rel err ~1.0);
      * the CLC scheduler intermittently leaves output tiles unwritten: 128x64 at
        M=32768 every time, and the default 256x256 cluster 2x1 in 2 of 30
        back-to-back gemm_gated calls at E=32 H=2880 I=2880 M=16384;
      * 256x256 cluster 2x1 with the static persistent scheduler: 0 wrong and
        bitwise identical over every shape, op and repeat tested, and within 2% of
        the fastest correct config at the Qwen3 shape.
    """
    cap = torch.cuda.get_device_capability(device)[0]
    cfg = _PINNED_CONFIG.get(cap)
    if cfg is None:
        import dataclasses
        cfg = _gemm_interface(module).default_config(device)
        if cap == 10 and getattr(cfg, "is_dynamic_persistent", False):
            cfg = dataclasses.replace(cfg, is_dynamic_persistent=False)
        _PINNED_CONFIG[cap] = cfg
    return cfg


def _pinned_gemm_eager(module, A, B, cu_seqlens_m, concat_b):
    out = torch.empty((A.shape[0], B.shape[-1]), dtype=A.dtype, device=A.device)
    if out.numel() == 0:
        return out
    if A.numel() == 0:
        # K=0: the kernel rejects it; the empty contraction is a zero matrix.
        return out.zero_()
    extra = {"concat_layout": ("B",)} if concat_b else {}
    _gemm_interface(module).gemm_tuned.fn(
        A, B, out, cu_seqlens_m=cu_seqlens_m, dynamic_scheduler=False,
        config=_pinned_config(module, A.device), **extra,
    )
    return out


def _pinned_gated_eager(module, A, B, C, cu_seqlens_m, concat_b):
    M, two_i = A.shape[0], B.shape[-1]
    preact = torch.empty((M, two_i), dtype=A.dtype, device=A.device)
    postact = torch.empty((M, two_i // 2), dtype=A.dtype, device=A.device)
    if M == 0:
        return preact, postact
    extra = {"concat_layout": ("B",)} if concat_b else {}
    _gemm_interface(module).gemm_gated_tuned.fn(
        A, B, preact, postact, C, None, "swiglu", cu_seqlens_m, None, False,
        config=_pinned_config(module, A.device), **extra,
    )
    return preact, postact


_PINNED_OPS = None


def _pinned_ops():
    """(gemm, gemm_gated) as torch.library custom ops over the pinned-config calls, so
    torch.compile traces them without a graph break and runs the same config as eager.
    Registered on first use; a re-import reuses the registered ops."""
    global _PINNED_OPS
    if _PINNED_OPS is not None:
        return _PINNED_OPS
    from typing import Optional, Tuple
    try:
        gemm_op = torch.ops.unsloth_sonic.gemm
        gated_op = torch.ops.unsloth_sonic.gemm_gated
    except (AttributeError, RuntimeError):

        @torch.library.custom_op("unsloth_sonic::gemm", mutates_args=())
        def gemm_op(A: torch.Tensor, B: torch.Tensor, cu_seqlens_m: torch.Tensor,
                    concat_b: bool) -> torch.Tensor:
            return _pinned_gemm_eager(get_sonic_module(), A, B, cu_seqlens_m, concat_b)

        @gemm_op.register_fake
        def _(A, B, cu_seqlens_m, concat_b):
            return A.new_empty((A.shape[0], B.shape[-1]))

        @torch.library.custom_op("unsloth_sonic::gemm_gated", mutates_args=())
        def gated_op(A: torch.Tensor, B: torch.Tensor, C: Optional[torch.Tensor],
                     cu_seqlens_m: torch.Tensor, concat_b: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            return _pinned_gated_eager(get_sonic_module(), A, B, C, cu_seqlens_m, concat_b)

        @gated_op.register_fake
        def _(A, B, C, cu_seqlens_m, concat_b):
            return (A.new_empty((A.shape[0], B.shape[-1])),
                    A.new_empty((A.shape[0], B.shape[-1] // 2)))

    _PINNED_OPS = (gemm_op, gated_op)
    return _PINNED_OPS


def _sonic_gemm(module, A, B, cu_seqlens_m, concat_b=False):
    """grouped A @ B[e] on sonic, pinned config (see `_pinned_config`).

    Calls `gemm_tuned.fn` directly: the public `functional.gemm` custom op with 18
    arguments costs ~0.15 ms of host time per call (see above). Under torch.compile
    the same pinned call goes through `_pinned_ops()`.

    `concat_b=True` passes ``concat_layout=("B",)``: the kernel reads B's
    non-contiguous dim as interleaved [g0, u0, g1, u1, ...] over a chunked
    [gate | up] buffer, with no copy (see `_SonicGatedZeroCopy`).

    ``UNSLOTH_MOE_SONIC_GEMM_TUNED=1`` restores the autotuned plain (non-concat)
    calls of the first integration. concat calls never autotune: the Autotuner's
    key ignores concat_layout, so it would reuse the plain layout's winner.
    """
    tune = (not concat_b) and _env_flag("UNSLOTH_MOE_SONIC_GEMM_TUNED", "0")
    compiling = torch.compiler.is_compiling()
    if not tune:
        if compiling or not _env_flag("UNSLOTH_MOE_SONIC_FASTCALL", "1"):
            return _pinned_ops()[0](A, B, cu_seqlens_m, concat_b)
        return _pinned_gemm_eager(module, A, B, cu_seqlens_m, concat_b)
    if compiling or not _env_flag("UNSLOTH_MOE_SONIC_FASTCALL", "1"):
        return module.functional.gemm(A, B, cu_seqlens_m=cu_seqlens_m, tuned=True)
    try:
        gi = _gemm_interface(module)
        tuned = gi.gemm_tuned
        out = torch.empty((A.shape[0], B.shape[-1]), dtype=A.dtype, device=A.device)
        if out.numel() == 0:
            return out
        if A.numel() == 0:
            out.zero_()
            return out
        key = (_shape_key(A), _shape_key(B), _shape_key(out), _shape_key(cu_seqlens_m))
        config = _CONFIG_CACHE.get(key)
        if config is None:
            # Cold: let the Autotuner resolve (and disk-cache) the config, then
            # remember what it picked. `dynamic_scheduler=False` is passed
            # explicitly so the autotune key is byte-identical to the one
            # `gemm_out` would have produced -- otherwise the first call would
            # re-run the whole 76-config sweep under a new key.
            tuned(A, B, out, cu_seqlens_m=cu_seqlens_m, dynamic_scheduler=False)
            if len(_CONFIG_CACHE) < _CONFIG_CACHE_MAX:
                _CONFIG_CACHE[key] = tuned.best_config.all_kwargs()["config"]
            return out
        tuned.fn(A, B, out, cu_seqlens_m=cu_seqlens_m,
                 dynamic_scheduler=False, config=config)
        return out
    except Exception as exc:
        _log(f"Unsloth: sonic-moe fast call path failed ({exc}); using functional.gemm.")
        return module.functional.gemm(A, B, cu_seqlens_m=cu_seqlens_m, tuned=True)


def _cu_seqlens_from_offsets(offsets):
    """sonic groups by cumulative OFFSETS [E+1] starting at 0; torch._grouped_mm takes
    cumulative ENDS [E]. One cat, no device sync."""
    zero = torch.zeros(1, dtype=torch.int32, device=offsets.device)
    return torch.cat([zero, offsets.to(torch.int32)])


# Cached interleaved gate_up weights for the fused gemm_gated path.
# Keyed by storage pointer + layout; the entry also records the autograd version
# counter and the shape, so an in-place edit, a realloc or a reshape invalidates it.
_INTERLEAVE_CACHE = {}


def _interleave_budget_bytes():
    try:
        return float(os.environ.get("UNSLOTH_MOE_SONIC_CACHE_GIB", "0")) * (1024 ** 3)
    except ValueError:
        return 0.0


def _cached_bytes():
    return sum(v[3].numel() * v[3].element_size() for v in _INTERLEAVE_CACHE.values())


def _drop_interleaved(key, ref):
    hit = _INTERLEAVE_CACHE.get(key)
    if hit is not None and hit[4] is ref:
        del _INTERLEAVE_CACHE[key]


@torch.compiler.disable
def _interleaved_gate_up(weight):
    """`weight` [E, H, 2I] chunked [gate | up] -> [E, H, 2I] interleaved
    [g0, u0, g1, u1, ...], which is what sonic's gemm_gated consumes.

    Cached, because the permutation is a full copy of the expert stack. Returns None
    when the copy would not fit the configured budget. Runs eagerly: Dynamo would not keep
    the cache insert, so a compiled caller would rebuild the copy on every call.
    """
    # Not id(weight): the provider hands out a fresh transpose view every call, so an
    # id key never hits and every forward re-copied the stack (and kept the stale
    # copy until the budget ran out). The storage + layout identifies the weight.
    key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()), weight.dtype)
    stamp = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    # data_ptr is reused once a model is freed: tie the entry to the owning tensor's lifetime.
    owner = weight._base if weight._base is not None else weight
    hit = _INTERLEAVE_CACHE.get(key)
    if (hit is not None and hit[4]() is owner and hit[:3] == stamp[:3]
            and hit[3].dtype == weight.dtype):
        return hit[3]

    nbytes = weight.numel() * weight.element_size()
    budget = _interleave_budget_bytes()
    if hit is not None:
        del _INTERLEAVE_CACHE[key]
    if budget <= 0 or _cached_bytes() + nbytes > budget:
        return None

    two_i = weight.shape[-1]
    half = two_i // 2
    perm = torch.empty(two_i, dtype=torch.long, device=weight.device)
    perm[0::2] = torch.arange(0, half, device=weight.device)
    perm[1::2] = torch.arange(half, two_i, device=weight.device)
    with torch.no_grad():
        interleaved = weight[:, :, perm].contiguous()
    _INTERLEAVE_CACHE[key] = (stamp[0], stamp[1], stamp[2], interleaved,
                              weakref.ref(owner, lambda ref, k = key: _drop_interleaved(k, ref)))
    return interleaved


def _interleave_columns(tensor):
    """Chunked [gate | up] -> interleaved along the LAST dim. Differentiable."""
    two_i = tensor.shape[-1]
    half = two_i // 2
    gate, up = tensor[..., :half], tensor[..., half:]
    return torch.stack((gate, up), dim=-1).reshape(*tensor.shape[:-1], two_i)


# ---------------------------------------------------------------------------
# Zero-copy fused gate_up GEMM + SwiGLU (concat_layout=("B",))
#
# quack's sm90/sm100 GEMMs rewrite every operand named in `concat_layout` with
# `concat_to_interleave(t, 1 - t.leading_dim)`: the NON-contiguous dim of size 2N is
# split into (2, N) with strides (N * s, s), so logical column 2j reads row j of the
# first half and 2j + 1 reads row j of the second half. No data moves.
#
#   * gate_up forward: B must be K-major, i.e. `W.transpose(-2, -1)` of the HF
#     [E, 2I, H] storage (H at stride 1). Then the N (= 2I) dim is the one that gets
#     interleaved and the kernel sees exactly the interleaved weight it wants.
#     A contiguous [E, H, 2I] (N-major) B has K as its non-contiguous dim, so concat
#     interleaves the CONTRACTION dim instead: wrong numbers.
#   * The pre-activation it stores is therefore INTERLEAVED (col 2j gate_j, 2j + 1
#     up_j); the post-activation is in natural order.
#   * dX = d_pre @ W: B is the HF storage [E, 2I, H] itself (N-major), whose
#     non-contiguous dim is K = 2I, so concat_layout=("B",) interleaves K to match the
#     interleaved d_pre. Same buffer, no copy.
#   * A residual C (the separated-LoRA delta) must be in the interleaved order too.
#     "C" in concat_layout would interleave C's non-contiguous dim, which for a
#     row-major [M, 2I] C is M, so instead the LoRA up-projection's columns
#     ([E, r, 2I], tiny) are interleaved and the delta comes out interleaved.
#
# Comparing that interleaved pre-activation against a chunked reference gives a
# relative error of sqrt(2) = 1.41 (two independent vectors of equal norm), which is
# the "garbage" the first integration measured.
# ---------------------------------------------------------------------------

_CONCAT_SUPPORTED = None


def _zero_copy_requested():
    return os.environ.get("UNSLOTH_MOE_SONIC_ZERO_COPY", "1") != "0"


def _gated_requested():
    # "unsafe" is the old literal opt-in; keep it meaning "on".
    return os.environ.get("UNSLOTH_MOE_SONIC_GATED", "1") in ("1", "unsafe")


def _sonic_gated(module, A, B, C, cu_seqlens_m, concat_b=True):
    """(preact, postact) = gemm_gated(A, B [+ C], swiglu), pinned config, no autotuner.

    concat_b: B is the K-major [E, H, 2I] view of chunked storage and
    concat_layout=("B",) reads it interleaved; else B is already interleaved.
    preact comes back interleaved, postact in natural order.
    """
    if torch.compiler.is_compiling() or not _env_flag("UNSLOTH_MOE_SONIC_FASTCALL", "1"):
        return _pinned_ops()[1](A, B, C, cu_seqlens_m, concat_b)
    return _pinned_gated_eager(module, A, B, C, cu_seqlens_m, concat_b)


try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

if triton is not None:

    @triton.jit
    def _swiglu_bwd_interleaved_kernel(PRE, GRAD, OUT, n_pairs, BLOCK: tl.constexpr):
        # PRE / OUT: [M, 2I] interleaved (g0, u0, g1, u1, ...); GRAD: [M, I]. One pair
        # per lane, fp32 math, one rounding on store: same formula as the torch path.
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_pairs
        g = tl.load(PRE + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(PRE + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(GRAD + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu = g * sig
        d_gate = go * u * (sig + silu * (1.0 - sig))
        d_up = go * silu
        tl.store(OUT + 2 * offs, d_gate.to(OUT.dtype.element_ty), mask=mask)
        tl.store(OUT + 2 * offs + 1, d_up.to(OUT.dtype.element_ty), mask=mask)


def _swiglu_backward_interleaved(preact, grad_post):
    """d(silu(g) * u) for an interleaved preact; returns d_preact interleaved, fp32 math.

    One Triton pass on CUDA (no fp32 temporaries: the torch form below holds ~7 fp32
    [M, I] buffers at once, +540 MiB transient at M=32768 I=768); torch elsewhere.
    """
    if (
        triton is not None
        and preact.is_cuda
        and preact.dim() == 2
        and _env_flag("UNSLOTH_MOE_SONIC_TRITON_DGATED", "1")
    ):
        preact_c = preact.contiguous()
        grad_c = grad_post.contiguous()
        out = torch.empty_like(preact_c)
        n_pairs = grad_c.numel()
        if n_pairs:
            BLOCK = 1024
            _swiglu_bwd_interleaved_kernel[(triton.cdiv(n_pairs, BLOCK),)](
                preact_c, grad_c, out, n_pairs, BLOCK=BLOCK,
            )
        return out
    gate = preact[..., 0::2].float()
    up = preact[..., 1::2].float()
    sig = torch.sigmoid(gate)
    silu = gate * sig
    grad_post_f = grad_post.float()
    d_gate = grad_post_f * up * (sig + silu * (1.0 - sig))
    d_up = grad_post_f * silu
    return torch.stack((d_gate, d_up), dim=-1).reshape(preact.shape).to(preact.dtype)


_ALIGNED_ATTR = "_unsloth_sonic_zc_aligned"


def _is_k_major_gate_up(weight, source=None):
    """[E, H, 2I] view with H at stride 1 and TMA-aligned strides: the layout
    concat_layout=("B",) reinterprets as interleaved without a copy.

    `source` is the base weight tensor the view comes from; the eager base-pointer
    verdict is cached on it, because Dynamo cannot trace storage_offset()."""
    # 2I even, and I a multiple of 8 so the [M, I] post-activation and [M, 2I]
    # pre-activation rows are 16-byte aligned for TMA.
    if weight.dim() != 3 or weight.shape[-1] % 16 != 0:
        return False
    if weight.stride(-2) != 1:
        return False
    esize = weight.element_size()
    if (weight.stride(-1) * esize) % 16 != 0 or (weight.stride(0) * esize) % 16 != 0:
        return False
    # The base pointer must be 16-byte aligned too. Allocations are 256-byte aligned,
    # so it is the storage offset that matters. Dynamo cannot trace storage_offset()
    # (not even aten.sym_storage_offset), so the eager verdict is stored as an
    # attribute on the source tensor (Dynamo guards on it: setting it recompiles) and a
    # compiled region only reads it. No eager verdict yet -> decline zero-copy.
    if torch.compiler.is_compiling():
        return source is not None and getattr(source, _ALIGNED_ATTR, None) is True
    aligned = (weight.storage_offset() * esize) % 16 == 0
    if isinstance(source, torch.Tensor):
        try:
            setattr(source, _ALIGNED_ATTR, aligned)
        except Exception:
            pass
    return aligned


def _concat_layout_supported(module):
    """Whether this sonic / quack build takes concat_layout on gemm + gemm_gated and
    gets it right. One tiny deterministic probe, cached; any failure -> False.
    Probed eagerly by get_sonic_module(), so a compiled region only reads the flag."""
    global _CONCAT_SUPPORTED
    if _CONCAT_SUPPORTED is not None:
        return _CONCAT_SUPPORTED
    if torch.compiler.is_compiling():
        return False
    ok = False
    try:
        import inspect
        gi = _gemm_interface(module)
        for fn in (gi.gemm_gated_tuned.fn, gi.gemm_tuned.fn, gi.gemm_act, gi.gemm):
            if "concat_layout" not in inspect.signature(fn).parameters:
                raise RuntimeError(f"{fn.__name__} has no concat_layout")
        device = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=device).manual_seed(0)
        E, H, inter, rows = 2, 64, 64, (40, 24)
        W = (torch.randn(E, 2 * inter, H, device=device, generator=gen) / 8).to(torch.bfloat16)
        X = torch.randn(sum(rows), H, device=device, generator=gen).to(torch.bfloat16)
        cu = torch.tensor([0, rows[0], sum(rows)], device=device, dtype=torch.int32)
        Wt = W.transpose(-2, -1)
        preact, postact = _sonic_gated(module, X, Wt, None, cu)
        ref = torch.cat([X[:rows[0]].float() @ Wt[0].float(), X[rows[0]:].float() @ Wt[1].float()])
        ref_post = F.silu(ref[:, :inter]) * ref[:, inter:]
        dpre = torch.randn(sum(rows), 2 * inter, device=device, generator=gen).to(torch.bfloat16)
        dx = _sonic_gemm(module, dpre, W, cu, concat_b=True)
        dpre_c = torch.cat([dpre[:, 0::2], dpre[:, 1::2]], dim=-1).float()
        ref_dx = torch.cat([dpre_c[:rows[0]] @ W[0].float(), dpre_c[rows[0]:] @ W[1].float()])

        def _rel(a, b):
            return ((a.float() - b).norm() / b.norm()).item()

        errs = (_rel(preact, _interleave_columns(ref)), _rel(postact, ref_post), _rel(dx, ref_dx))
        if not all(e < 2e-2 for e in errs):
            raise RuntimeError(f"concat_layout probe mismatch, rel errs {errs}")
        ok = True
    except Exception as exc:
        _log(f"Unsloth: sonic-moe zero-copy gate_up unavailable ({exc}); "
             f"falling back to the interleaved copy / unfused path.")
        ok = False
    _CONCAT_SUPPORTED = ok
    return ok


class _SonicGatedZeroCopy(torch.autograd.Function):
    """post = swiglu(X @ W_gate_up [+ residual]) with NO interleaved weight copy.

    `weight_provider()` returns the K-major [E, H, 2I] view of the HF [E, 2I, H]
    storage; concat_layout=("B",) makes the kernel read it interleaved. `residual`
    (LoRA delta) must already be interleaved. Saves only the interleaved preact.
    dX reuses the same storage through concat_layout=("B",) on its N-major
    transpose; d_residual is d_preact, interleaved like the residual.
    """

    @staticmethod
    def forward(ctx, inputs, residual, offsets, cu_seqlens, weight_provider, recompute):
        module = get_sonic_module()
        weight = weight_provider()
        with torch.no_grad():
            preact, post = _sonic_gated(
                module, inputs.contiguous(), weight,
                None if residual is None else residual.contiguous(), cu_seqlens,
            )
        ctx.has_residual = residual is not None
        if recompute:
            ctx.weight_provider = weight_provider
            ctx.save_for_backward(preact, cu_seqlens)
        else:
            ctx.weight_provider = None
            ctx.save_for_backward(preact, cu_seqlens, weight)
        return post

    @staticmethod
    def backward(ctx, grad_post):
        if ctx.weight_provider is not None:
            preact, cu_seqlens = ctx.saved_tensors
            weight = ctx.weight_provider()
        else:
            preact, cu_seqlens, weight = ctx.saved_tensors
        with torch.no_grad():
            d_pre = _swiglu_backward_interleaved(preact, grad_post)
            grad_input = _sonic_gemm(
                get_sonic_module(), d_pre, weight.transpose(-2, -1), cu_seqlens, concat_b=True
            )
        return grad_input, (d_pre if ctx.has_residual else None), None, None, None, None


class _SonicGroupedMM(torch.autograd.Function):
    """out = grouped(inputs, weight_provider()) with sonic; dX by the chosen backend.

    The base expert weight is FROZEN (checked by the caller), so there is no dW and
    backward only produces dX. `recompute` mirrors `_base_grouped_mm`: rebuild the
    weight in backward instead of pinning the dense stack.
    """

    @staticmethod
    def forward(ctx, inputs, offsets, cu_seqlens, weight_provider, recompute, sonic_dgrad):
        module = get_sonic_module()
        weight = weight_provider()
        with torch.no_grad():
            out = _sonic_gemm(module, inputs.contiguous(), weight, cu_seqlens)
        ctx.sonic_dgrad = sonic_dgrad
        if recompute:
            ctx.weight_provider = weight_provider
            ctx.save_for_backward(offsets, cu_seqlens)
        else:
            ctx.weight_provider = None
            ctx.save_for_backward(offsets, cu_seqlens, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weight_provider is not None:
            offsets, cu_seqlens = ctx.saved_tensors
            weight = ctx.weight_provider()
        else:
            offsets, cu_seqlens, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        with torch.no_grad():
            # Transposed VIEW, never a copy: sonic reads it directly and is
            # bit-identical to the contiguous form here (measured), while making it
            # contiguous would copy the whole expert stack every step.
            weight_t = weight.transpose(-2, -1)
            if ctx.sonic_dgrad:
                grad_input = _sonic_gemm(
                    get_sonic_module(), grad_output, weight_t, cu_seqlens
                )
            else:
                from .moe_utils import _grouped_mm_with_backward_fix
                grad_input = _grouped_mm_with_backward_fix(grad_output, weight_t, offsets)
        return grad_input, None, None, None, None, None


class _SonicGatedGroupedMM(torch.autograd.Function):
    """post = swiglu(grouped(inputs, W_interleaved) + residual), fused in one kernel.

    `residual` carries the separated-LoRA delta into the PRE-activation sum, which is
    where it has to land; it is already in interleaved column order. dResidual is just
    d_preact, so the LoRA chain's own autograd is untouched.
    """

    @staticmethod
    def forward(ctx, inputs, residual, cu_seqlens, weight_interleaved, offsets):
        module = get_sonic_module()
        with torch.no_grad():
            preact, post = _sonic_gated(
                module, inputs.contiguous(), weight_interleaved,
                None if residual is None else residual.contiguous(), cu_seqlens,
                concat_b=False,
            )
        ctx.has_residual = residual is not None
        ctx.save_for_backward(preact, cu_seqlens, weight_interleaved, offsets)
        return post

    @staticmethod
    def backward(ctx, grad_post):
        preact, cu_seqlens, weight_interleaved, offsets = ctx.saved_tensors
        with torch.no_grad():
            d_pre = _swiglu_backward_interleaved(preact, grad_post)
            grad_input = _sonic_gemm(
                get_sonic_module(), d_pre, weight_interleaved.transpose(-2, -1), cu_seqlens
            )
        return grad_input, (d_pre if ctx.has_residual else None), None, None, None


def _is_quantized_source(source):
    """bnb 4-bit / fp8 / any non-plain-float expert weight must not reach sonic."""
    param = source
    seen = set()
    while param is not None and id(param) not in seen:
        seen.add(id(param))
        if type(param).__name__ in ("Params4bit", "Params8bit", "Int8Params"):
            return True
        nxt = getattr(param, "base_layer", None)
        if nxt is None or nxt is param:
            break
        param = nxt
    if getattr(source, "quant_state", None) is not None:
        return True
    return False


def _sonic_eligible(self, hidden_states):
    """(ok, reason). Anything unhandled falls back to grouped_mm, never to a crash."""
    if not sonic_moe_available():
        return False, f"sonic-moe unavailable ({sonic_moe_unavailable_reason()})"
    if not _device_ok(hidden_states.device):
        return False, f"device {hidden_states.device} is not the compute capability sonic was probed on"
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        return False, f"dtype {hidden_states.dtype} is not bf16/fp16"
    if not (hasattr(self, "gate_up_proj") and hasattr(self, "down_proj")):
        return False, "experts do not use the gate_up_proj/down_proj layout"
    cls = self.__class__.__name__
    if "GptOss" in cls:
        return False, "GptOss uses a clamped non-SwiGLU activation"
    from .moe_utils import _act_fn_is_silu, _gate_up_is_interleaved, _uses_own_apply_gate
    if _gate_up_is_interleaved(self):
        return False, "gate_up is stored interleaved"
    if _uses_own_apply_gate(self):
        return False, "experts apply their own gate"
    act = getattr(self, "act_fn", None)
    # Exact SiLU types only (F.silu, nn.SiLU, SiLUActivation): the fused kernel always
    # computes SiLU, and a name match let e.g. nn.Hardswish through.
    if act is not None and not _act_fn_is_silu(act):
        return False, f"activation {type(act).__name__} is not SiLU"
    if getattr(self, "gate_up_proj_bias", None) is not None:
        return False, "gate_up_proj_bias is unsupported"
    if getattr(self, "down_proj_bias", None) is not None:
        return False, "down_proj_bias is unsupported"
    for name in ("gate_up_proj", "down_proj"):
        source = getattr(self, name)
        if _is_quantized_source(source):
            return False, f"{name} is quantized"
        base = source
        while hasattr(base, "base_layer"):
            base = base.base_layer
        if isinstance(base, torch.Tensor):
            if base.requires_grad:
                # Base weights train -> a dW sonic does not produce here.
                return False, f"{name} base weight requires grad"
            if base.dtype not in _SUPPORTED_DTYPES:
                return False, f"{name} base dtype {base.dtype} is not bf16/fp16"
            if base.dim() != 3:
                return False, f"{name} base is not a 3D expert stack"
    return True, None


def _base_tensor(source):
    """The plain tensor under a PEFT wrapper chain, or None."""
    base = source
    while hasattr(base, "base_layer"):
        base = base.base_layer
    return base if isinstance(base, torch.Tensor) else None


_WARNED_FALLBACK = set()


def forward_sonic_moe(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """MoE forward with sonic-moe for the expert GEMMs it actually wins on.

    Same signature, routing, separated-LoRA algebra, gate-gradient identity and
    scatter-add as `forward_native_grouped_mm`; only the two frozen base GEMMs and
    the dX backward change. Anything it cannot handle falls back to that function.
    """
    from .moe_utils import (
        forward_native_grouped_mm,
        _grouped_mm_with_backward_fix,
        _should_use_separated_lora,
        _has_lora_adapters,
        _extract_lora_weights,
        _get_base_weight,
        _moe_recompute_enabled,
        _moe_gategrad_enabled,
        _MoEGateGradIdentity,
        _pad_lora_rank_for_grouped_mm,
        combine_permuted_moe_outputs,
        preprocess_weight,
        take_moe_lora_stash,
        weighted_unpermute,
    )

    ok, reason = _sonic_eligible(self, hidden_states)
    if not ok:
        if reason not in _WARNED_FALLBACK:
            _WARNED_FALLBACK.add(reason)
            _log(f"Unsloth: sonic-moe backend falling back to grouped_mm: {reason}.")
        return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

    module = get_sonic_module()
    use_sonic_down = _env_flag("UNSLOTH_MOE_SONIC_DOWN", "1")
    use_sonic_dh = _env_flag("UNSLOTH_MOE_SONIC_DH", "0")
    use_gated = _gated_requested()

    is_2d_input = hidden_states.dim() == 2
    if is_2d_input:
        sequence_length, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Routing: identical to forward_native_grouped_mm.
    from .moe_utils import moe_routing_sort
    _, sorted_indices, offsets = moe_routing_sort(top_k_index, self.num_experts)
    token_indices = sorted_indices // top_k_index.shape[-1]
    permuted_input = hidden_states[token_indices]
    cu_seqlens = _cu_seqlens_from_offsets(offsets)

    use_separated_lora = _should_use_separated_lora()
    gate_up_lora = None
    _stashed = take_moe_lora_stash(self, "gate_up_proj")
    if _stashed is not None:
        gate_up_lora = _stashed[:3]
    elif use_separated_lora and _has_lora_adapters(self.gate_up_proj):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts, experts_module=self
        )

    model_type = getattr(self, "_unsloth_model_type", None)
    gate_up_source = self.gate_up_proj

    def gate_up_provider(_src=gate_up_source, _mt=model_type, _h=hidden_dim,
                         _dt=hidden_states.dtype, _mod=self):
        return preprocess_weight(
            _get_base_weight(_src, _dt), "gate_up", _h, _mt, experts_module=_mod
        )

    # Fused gate_up + SwiGLU: zero-copy concat_layout on the K-major view when this
    # build supports it (default), else the cached interleaved duplicate (only with a
    # UNSLOTH_MOE_SONIC_CACHE_GIB budget), else the unfused GEMM + eager SwiGLU.
    gated_mode = None
    interleaved = None
    if use_gated:
        if (
            _zero_copy_requested()
            and _concat_layout_supported(module)
            and _is_k_major_gate_up(gate_up_provider(), _base_tensor(gate_up_source))
        ):
            gated_mode = "zero_copy"
        elif _interleave_budget_bytes() > 0 and not torch.compiler.is_compiling():
            # The cache is only filled eagerly; a compiled frame takes the unfused path.
            interleaved = _interleaved_gate_up(gate_up_provider())
            if interleaved is not None:
                gated_mode = "duplicate"

    # LoRA delta: ((X @ A) @ B) * scaling on torch._grouped_mm, which beats sonic 3:1
    # at rank 8. For zero-copy the up-projection's [E, r, 2I] columns are interleaved
    # (a tiny copy, differentiable) so the delta lands in the kernel's column order.
    lora_delta = None
    lora_scaling = None
    if gate_up_lora is not None:
        first_weight, second_weight, lora_scaling = gate_up_lora
        first_weight = first_weight.to(permuted_input.dtype).contiguous()
        second_weight = second_weight.to(permuted_input.dtype)
        if gated_mode == "zero_copy":
            second_weight = _interleave_columns(second_weight)
        second_weight = second_weight.contiguous()
        first_weight, second_weight = _pad_lora_rank_for_grouped_mm(
            first_weight, second_weight, permuted_input
        )
        lora_out = _grouped_mm_with_backward_fix(
            permuted_input, first_weight, offsets
        ).contiguous()
        if second_weight.shape[-1] % 8 != 0:
            pad = 8 - (second_weight.shape[-1] % 8)
            padded = F.pad(second_weight, (0, pad)).contiguous()
            lora_delta = _grouped_mm_with_backward_fix(lora_out, padded, offsets)[:, :-pad]
        else:
            lora_delta = _grouped_mm_with_backward_fix(lora_out, second_weight, offsets)

    if gated_mode is not None and gated_mode not in _WARNED_FALLBACK:
        _WARNED_FALLBACK.add(gated_mode)
        _log(f"Unsloth: sonic-moe fused gate_up + SwiGLU ({gated_mode}).")
    if gated_mode == "zero_copy":
        residual = None if lora_delta is None else lora_delta * lora_scaling
        inter = _SonicGatedZeroCopy.apply(
            permuted_input, residual, offsets, cu_seqlens, gate_up_provider,
            _moe_recompute_enabled(gate_up_source),
        )
    elif gated_mode == "duplicate":
        # Legacy: fused kernel on a cached interleaved COPY of gate_up. The residual
        # must be in the kernel's interleaved column order too.
        residual = None
        if lora_delta is not None:
            residual = _interleave_columns(lora_delta * lora_scaling)
        inter = _SonicGatedGroupedMM.apply(
            permuted_input, residual, cu_seqlens, interleaved, offsets
        )
    else:
        mm1_out = _SonicGroupedMM.apply(
            permuted_input, offsets, cu_seqlens, gate_up_provider,
            _moe_recompute_enabled(gate_up_source), True,
        )
        if lora_delta is not None:
            mm1_out = mm1_out + lora_delta * lora_scaling
        gate, up = mm1_out.chunk(2, dim=-1)
        inter = F.silu(gate) * up

    # Gate-gradient identity: byte-for-byte the grouped_mm policy.
    _gategrad = (
        _moe_gategrad_enabled()
        and getattr(self, "down_proj_bias", None) is None
        and top_k_weights.requires_grad
        and inter.dtype != torch.float16
    )
    permuted_weights = None
    if _gategrad:
        raw_weights = top_k_weights.reshape(-1)[sorted_indices]
        if raw_weights.dtype == torch.float16:
            raw_weights = raw_weights.to(torch.float32)
        eps = 1e-12
        floored = torch.where(
            raw_weights >= 0, raw_weights.clamp(min=eps), raw_weights.clamp(max=-eps)
        )
        permuted_weights = raw_weights + (floored - raw_weights).detach()
        inter = _MoEGateGradIdentity.apply(inter, permuted_weights)

    down_lora = None
    _stashed = take_moe_lora_stash(self, "down_proj")
    if _stashed is not None:
        down_lora = _stashed[:3]
    elif use_separated_lora and _has_lora_adapters(self.down_proj):
        down_lora = _extract_lora_weights(
            self.down_proj, num_experts=self.num_experts, experts_module=self
        )

    down_source = self.down_proj

    def down_provider(_src=down_source, _mt=model_type, _h=hidden_dim,
                      _dt=hidden_states.dtype, _mod=self):
        return preprocess_weight(
            _get_base_weight(_src, _dt), "down", _h, _mt, experts_module=_mod
        )

    if use_sonic_down:
        mm2_out = _SonicGroupedMM.apply(
            inter, offsets, cu_seqlens, down_provider,
            _moe_recompute_enabled(down_source), use_sonic_dh,
        )
    else:
        mm2_out = _grouped_mm_with_backward_fix(inter, down_provider(), offsets)

    if down_lora is not None:
        first_weight, second_weight, scaling = down_lora
        first_weight = first_weight.to(inter.dtype).contiguous()
        second_weight = second_weight.to(inter.dtype).contiguous()
        first_weight, second_weight = _pad_lora_rank_for_grouped_mm(first_weight, second_weight, inter)
        lora_out = _grouped_mm_with_backward_fix(inter, first_weight, offsets).contiguous()
        lora_delta_down = _grouped_mm_with_backward_fix(lora_out, second_weight, offsets)
        mm2_out = mm2_out + lora_delta_down * scaling

    # Weighted combine: the same deterministic unpermute + sum as
    # forward_native_grouped_mm (an index_add_ here is bf16 atomics, so the output
    # changed bit-for-bit from call to call).
    if _gategrad:
        permuted_weights = permuted_weights.detach()
    else:
        permuted_weights = top_k_weights.reshape(-1)[sorted_indices]
    num_tokens = batch_size * sequence_length
    final_hidden_states = weighted_unpermute(
        mm2_out, sorted_indices, permuted_weights, num_tokens, top_k_index.shape[-1],
        out_dtype=hidden_states.dtype,
    )
    if final_hidden_states is None:
        mm2_out = mm2_out * permuted_weights.unsqueeze(-1)
        final_hidden_states = combine_permuted_moe_outputs(
            mm2_out, sorted_indices, num_tokens, top_k_index.shape[-1],
            out_dtype=hidden_states.dtype,
        )

    if is_2d_input:
        return final_hidden_states
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
