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

The fused ``gemm_gated`` (gate_up GEMM + SwiGLU in one kernel, 0.291 -> 0.188 ms) is
implemented but DEFAULTS OFF, for three measured reasons:

  1. It needs an INTERLEAVED gate/up weight (silu(even_cols) * odd_cols) while
     Qwen/torch store the CHUNKED layout. sonic's ``concat_layout=("B",)`` advertises
     the chunked layout but produces garbage here: rel err 1.41 for both stride
     orders, so interleaving has to be a real column permutation.
  2. A cached interleaved copy is a FULL DUPLICATE of the gate_up stack: 1.07 GiB per
     layer on Qwen3.5-35B-A3B, 43 GiB over its 40 layers. Permuting per step instead
     moves ~2 GiB per layer per step, far more than the 0.1 ms it saves.
  3. ``gemm_gated(store_preact=True, tuned=True)`` is WRONG on some inputs, and
     nondeterministically so. At E=96 H=768 I=384 M=6144 the post-activation has rel
     err ~1.0; two IDENTICAL back-to-back calls disagree with each other by 0.76 in
     the pre-activation, i.e. the kernel returns raced / uninitialized memory. The
     failure is not a function of the weight shape alone, nor of (shape, M): a probe
     run at the exact shape and the exact row count can pass and the very next real
     call at the same shape and row count still returns garbage. That is why there is
     no self-probe here -- a nondeterministic kernel cannot be gated by sampling it.
     ``tuned=False`` is correct but slow, and ``store_preact=False`` is correct but
     forces a second gate_up GEMM in backward, which erases the win. Plain ``gemm`` is
     exact (rel err 0.00000) on every shape, row count and routing tested.

The gated code below is therefore NOT REACHABLE by any boolean flag. It runs only with
the literal ``UNSLOTH_MOE_SONIC_GATED=unsafe`` plus a non-zero
``UNSLOTH_MOE_SONIC_CACHE_GIB``, and warns on every engagement. Do not enable it until
the upstream kernel is fixed; it is kept so the integration is ready when that happens.
"""
import os
import sys
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
    global _SONIC_MODULE, _SONIC_REASON, _SONIC_PROBED, _GEMM_INTERFACE
    _SONIC_MODULE = None
    _SONIC_REASON = None
    _SONIC_PROBED = False
    _GEMM_INTERFACE = None
    _INTERLEAVE_CACHE.clear()
    _CONFIG_CACHE.clear()


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


def _probe_sonic(module):
    """One tiny real call. The import succeeding does not mean the kernel runs."""
    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.ones((8, 16), device=device, dtype=torch.bfloat16)
    b = torch.ones((2, 16, 16), device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4, 8], device=device, dtype=torch.int32)
    out = module.functional.gemm(a, b, cu_seqlens_m=cu, tuned=True)
    if out.shape != (8, 16):
        raise RuntimeError(f"sonic-moe probe returned {tuple(out.shape)}, expected (8, 16)")
    # The low-overhead path reaches internals `functional.gemm` hides, so probe it
    # too and make sure it agrees exactly. Its own try/except would otherwise hide
    # a permanently-broken fast path behind a silent per-call fallback.
    fast = _sonic_gemm(module, a, b, cu)
    if fast.shape != out.shape or not torch.equal(fast, out):
        raise RuntimeError("sonic-moe fast call path disagrees with functional.gemm")


def get_sonic_module():
    """The sonic-moe kernel module, or None. Probed once, never raises."""
    global _SONIC_MODULE, _SONIC_REASON, _SONIC_PROBED
    if _SONIC_PROBED:
        return _SONIC_MODULE
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
    try:
        _probe_sonic(module)
    except Exception as exc:
        _SONIC_REASON = f"sonic-moe probe call failed: {exc}"
        return None

    _SONIC_MODULE = module
    _SONIC_REASON = None
    _log("Unsloth: sonic-moe MoE backend is available.")
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
# The custom op is still used verbatim when `torch.compiler.is_compiling()`, so
# anything that traces this code still sees a registered, fake-implemented op.
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


def _sonic_gemm(module, A, B, cu_seqlens_m):
    """`module.functional.gemm(A, B, cu_seqlens_m=..., tuned=True)` without the
    custom-op boundary and without the per-call autotune key.

    Same output tensor, same kernel, same config. Falls back to the public entry
    point under torch.compile tracing or if anything at all goes wrong.
    """
    if not _env_flag("UNSLOTH_MOE_SONIC_FASTCALL", "1") or torch.compiler.is_compiling():
        return module.functional.gemm(A, B, cu_seqlens_m=cu_seqlens_m, tuned=True)
    try:
        gi = _gemm_interface(module)
        tuned = gi.gemm_tuned
        out = torch.empty((A.shape[0], B.shape[-1]), dtype=A.dtype, device=A.device)
        if out.numel() == 0:
            return out
        if A.numel() == 0:
            # K=0: the kernel rejects it; the empty contraction is a zero matrix.
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
# Keyed by id(tensor); the entry records the storage pointer, the autograd version
# counter and the shape, so an in-place edit, a realloc or a reshape invalidates it.
_INTERLEAVE_CACHE = {}


def _interleave_budget_bytes():
    try:
        return float(os.environ.get("UNSLOTH_MOE_SONIC_CACHE_GIB", "0")) * (1024 ** 3)
    except ValueError:
        return 0.0


def _cached_bytes():
    return sum(v[3].numel() * v[3].element_size() for v in _INTERLEAVE_CACHE.values())


def _interleaved_gate_up(weight):
    """`weight` [E, H, 2I] chunked [gate | up] -> [E, H, 2I] interleaved
    [g0, u0, g1, u1, ...], which is what sonic's gemm_gated consumes.

    Cached, because the permutation is a full copy of the expert stack. Returns None
    when the copy would not fit the configured budget.
    """
    key = id(weight)
    stamp = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    hit = _INTERLEAVE_CACHE.get(key)
    if hit is not None and hit[:3] == stamp[:3] and hit[3].dtype == weight.dtype:
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
    _INTERLEAVE_CACHE[key] = (stamp[0], stamp[1], stamp[2], interleaved)
    return interleaved


def _interleave_columns(tensor):
    """Chunked [gate | up] -> interleaved along the LAST dim. Differentiable."""
    two_i = tensor.shape[-1]
    half = two_i // 2
    gate, up = tensor[..., :half], tensor[..., half:]
    return torch.stack((gate, up), dim=-1).reshape(*tensor.shape[:-1], two_i)


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
            preact, post = module.functional.gemm_gated(
                inputs.contiguous(),
                weight_interleaved,
                C=None if residual is None else residual.contiguous(),
                activation="swiglu",
                cu_seqlens_m=cu_seqlens,
                store_preact=True,
                tuned=True,
            )
        ctx.has_residual = residual is not None
        ctx.save_for_backward(preact, cu_seqlens, weight_interleaved, offsets)
        return post

    @staticmethod
    def backward(ctx, grad_post):
        preact, cu_seqlens, weight_interleaved, offsets = ctx.saved_tensors
        with torch.no_grad():
            gate = preact[..., 0::2].float()
            up = preact[..., 1::2].float()
            sig = torch.sigmoid(gate)
            silu = gate * sig
            grad_post_f = grad_post.float()
            d_gate = grad_post_f * up * (sig + silu * (1.0 - sig))
            d_up = grad_post_f * silu
            d_pre = torch.stack((d_gate, d_up), dim=-1).reshape(preact.shape).to(preact.dtype)
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
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        return False, f"dtype {hidden_states.dtype} is not bf16/fp16"
    if not (hasattr(self, "gate_up_proj") and hasattr(self, "down_proj")):
        return False, "experts do not use the gate_up_proj/down_proj layout"
    cls = self.__class__.__name__
    if "GptOss" in cls:
        return False, "GptOss uses a clamped non-SwiGLU activation"
    from .moe_utils import _gate_up_is_interleaved, _uses_own_apply_gate
    if _gate_up_is_interleaved(self):
        return False, "gate_up is stored interleaved"
    if _uses_own_apply_gate(self):
        return False, "experts apply their own gate"
    act = getattr(self, "act_fn", None)
    if act is not None:
        name = getattr(act, "__name__", type(act).__name__).lower()
        if "silu" not in name and "swish" not in name:
            return False, f"activation {name} is not SiLU"
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
        preprocess_weight,
        take_moe_lora_stash,
    )

    ok, reason = _sonic_eligible(self, hidden_states)
    if not ok:
        if reason not in _WARNED_FALLBACK:
            _WARNED_FALLBACK.add(reason)
            _log(f"Unsloth: sonic-moe backend falling back to grouped_mm: {reason}.")
        return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

    module = get_sonic_module()
    sonic_gemm = module.functional.gemm
    use_sonic_down = _env_flag("UNSLOTH_MOE_SONIC_DOWN", "1")
    use_sonic_dh = _env_flag("UNSLOTH_MOE_SONIC_DH", "0")
    # Not a boolean: the fused kernel is nondeterministically wrong (docstring
    # reason 3), so it takes a literal opt-in nobody sets by accident.
    use_gated = os.environ.get("UNSLOTH_MOE_SONIC_GATED", "") == "unsafe"

    is_2d_input = hidden_states.dim() == 2
    if is_2d_input:
        sequence_length, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Routing: identical to forward_native_grouped_mm.
    flat_top_k = top_k_index.view(-1)
    from .moe_utils import count_tokens_per_expert
    num_tokens_per_expert = count_tokens_per_expert(flat_top_k, self.num_experts, torch.int32)
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[-1]
    permuted_input = hidden_states[token_indices]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
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

    # LoRA delta, unchanged: ((X @ A) @ B) * scaling on torch._grouped_mm, which beats
    # sonic 3:1 at rank 8.
    lora_delta = None
    lora_scaling = None
    if gate_up_lora is not None:
        first_weight, second_weight, lora_scaling = gate_up_lora
        first_weight = first_weight.to(permuted_input.dtype).contiguous()
        second_weight = second_weight.to(permuted_input.dtype).contiguous()
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

    interleaved = _interleaved_gate_up(gate_up_provider()) if use_gated else None
    if interleaved is not None and "gated" not in _WARNED_FALLBACK:
        _WARNED_FALLBACK.add("gated")
        print(
            "Unsloth: UNSLOTH_MOE_SONIC_GATED=unsafe -- using sonic gemm_gated, which "
            "returns nondeterministically wrong results on this build (identical calls "
            "disagree by ~0.76 in the pre-activation). Training on this is unsound.",
            file=sys.stderr,
        )
    if interleaved is not None:
        # Fused gate_up GEMM + SwiGLU. The residual must be in the kernel's
        # interleaved column order too.
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

    if _gategrad:
        mm2_out = mm2_out * permuted_weights.detach().unsqueeze(-1)
    else:
        permuted_weights = top_k_weights.reshape(-1)[sorted_indices]
        mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    final_hidden_states.index_add_(0, token_indices, mm2_out.to(hidden_states.dtype))

    if is_2d_input:
        return final_hidden_states
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
