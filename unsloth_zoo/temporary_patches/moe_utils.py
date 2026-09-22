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
import torch
import torch.nn.functional as F
import contextlib
import json
import os
import shutil
import stat
import tempfile
import sys
import warnings
import importlib
import importlib.util
from typing import Optional, Tuple
from torch.autograd import Function
from unsloth_zoo.mlx import is_mlx_available

UNSLOTH_COMPILE_LOCATION = os.environ.get(
    "UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache"
)

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit
    # unsloth_zoo installs a permissive bitsandbytes stub wherever the real package is
    # absent, macOS arm64 among others, and every attribute of that stub resolves to a
    # placeholder object rather than a class. `isinstance(x, Params4bit)` against one
    # raises TypeError: isinstance() arg 2 must be a type, so a non-class Params4bit has
    # to count as no bitsandbytes at all. On a real install this is just True.
    HAS_BNB = isinstance(Params4bit, type)
    if not HAS_BNB:
        Params4bit = None
except Exception:
    # Not just ImportError: a bitsandbytes mismatched with torch fails its own import with AttributeError.
    HAS_BNB = False
    Params4bit = None

if not isinstance(Params4bit, type):
    # A bitsandbytes that imports but does not expose Params4bit as a class, which is what
    # the macOS build does, makes every `isinstance(param, Params4bit)` below raise
    # TypeError instead of answering False. Probe the object, not the platform or the
    # version, and treat that install as no bitsandbytes at all: there is no 4-bit expert
    # path without the class.
    HAS_BNB = False
    Params4bit = None


def _get_compile_location() -> str:
    return os.path.abspath(
        os.environ.get("UNSLOTH_COMPILE_LOCATION", UNSLOTH_COMPILE_LOCATION)
    )


def _log_info(message: str):
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        print(message)


def _warn_without_raising(message):
    """warnings.warn, except that it cannot be the thing that fails the import.

    install_to_cache() runs at import and simplefilter("error") would turn this
    into a raise, for a condition already handled by not using the file.
    """
    try:
        warnings.warn(message)
    except Exception:
        print(message)


def _read_file_bytes(path):
    """The file's bytes, or None when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _replace_with_copy(current_file, destination):
    """Copy over `destination` by replacing the path, not by writing into it.

    shutil.copy() opens the destination, so a read-only one refuses it, and it
    is not atomic while nothing locks this file and every rank installs it at
    import. os.replace() needs only the directory and is atomic.
    """
    directory = os.path.dirname(destination) or "."
    descriptor, temporary = tempfile.mkstemp(
        prefix = f".{os.path.basename(destination)}.", suffix = ".tmp", dir = directory,
    )
    try:
        # Through the DESCRIPTOR, not by reopening the name, or the copy and the
        # mode follow that name somewhere else; this directory is writable by
        # whoever would plant a copy here. Same shape as
        # compiler._replace_compiled_cache_file.
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            with open(current_file, "rb") as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            # mkstemp is owner-only; keep the mode shutil.copy() would have left.
            mode = stat.S_IMODE(os.stat(current_file).st_mode)
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), mode)
            else:
                os.chmod(temporary, mode)
        os.replace(temporary, destination)
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def install_to_cache(source_path, destination_filename=None):
    """Copy a file into unsloth_compiled_cache so compiled modules can use it.

    Returns whether the copy is known to match `source_path`. Runs at import
    time, so an unwritable directory is not fatal; a destination that exists and
    does NOT match is reported, since _load_cached_moe_utils_module() execs it.
    """
    compile_location = _get_compile_location()
    if not os.path.exists(compile_location):
        try:
            os.makedirs(compile_location)
        except:
            pass

    current_file = os.path.abspath(source_path)
    if destination_filename is None:
        destination_filename = os.path.basename(current_file)

    destination = os.path.abspath(os.path.join(compile_location, destination_filename))

    if current_file == destination:
        return True

    current = _read_file_bytes(current_file)
    if current is not None and _read_file_bytes(destination) == current:
        # Already installed. This runs on every `import unsloth_zoo`, on every
        # rank, and copies this whole module, so without the compare the steady
        # state was replacing a file with an identical copy of itself.
        return True

    try:
        _replace_with_copy(current_file, destination)
    except Exception as replace_error:
        # A directory we cannot add a temp file to can still hold a destination
        # we can write through, so the plain copy is worth one attempt. It gives
        # up the atomicity above, so the readback below is what keeps a partial
        # copy from being used.
        try:
            shutil.copy(current_file, destination)
        except Exception as copy_error:
            _log_info(
                f"Unsloth: Could not install {destination}: "
                f"{replace_error}; {copy_error}"
            )

    current = _read_file_bytes(current_file)
    installed = _read_file_bytes(destination)
    if installed is None or current is None:
        # Nothing landed, or there is nothing to compare it against. Either way
        # no compiled module gets to use it.
        return False
    if installed == current:
        return True
    _warn_without_raising(
        f"Unsloth: {destination} does not match {current_file} and could not be "
        "replaced, so it will not be used. Delete it to restore the compiled "
        "cache copy."
    )
    return False


install_to_cache(__file__, "moe_utils.py")

_CACHED_FORWARD_MOE_BACKEND = None
_CACHED_MOE_UTILS_MODULE = None


_WARNED_STALE_CACHE = set()


def _cached_copy_is_current(cache_file, current_file) -> bool:
    """Whether the compiled cache holds byte for byte what this module is.

    They diverge when the copy could not be written, leaving an OLDER
    unsloth_zoo that every caller below would prefer, silently installing that
    release's patches over this one's. Prefer this module, and say so once.
    """
    cached = _read_file_bytes(cache_file)
    current = _read_file_bytes(current_file)
    if cached is None or current is None:
        return False
    if cached == current:
        return True

    if cache_file not in _WARNED_STALE_CACHE:
        _WARNED_STALE_CACHE.add(cache_file)
        logger = _moe_utils_logger()
        if logger is not None:
            logger.warning(
                f"Unsloth: {cache_file} is from a different version of unsloth_zoo and "
                f"could not be refreshed, so it is being ignored. Delete that directory "
                f"if MoE behaviour looks stale."
            )
    return False


def _remove_cached_bytecode(source_file):
    """Drop the pyc beside a cache copy we are about to execute.

    The comparison covers the .py only, and CPython runs an unchecked-hash pyc
    without consulting it, so a pyc we cannot remove means using our own defs.
    """
    try:
        bytecode_location = importlib.util.cache_from_source(source_file)
    except NotImplementedError:
        return True
    # A symlinked `__pycache__` sends this unlink outside the cache. Refusing
    # costs the cache copy and falls back to our own definitions, which beats
    # deleting somebody else's file. sys.pycache_prefix is the user's own.
    if not getattr(sys, "pycache_prefix", None):
        try:
            if os.path.islink(os.path.dirname(bytecode_location)):
                return False
        except OSError:
            return False
    try:
        os.remove(bytecode_location)
    except FileNotFoundError:
        pass
    except OSError:
        return not os.path.isfile(bytecode_location)
    return True


def cached_copy_is_importable(directory) -> bool:
    """Whether a generated module may be allowed to import moe_utils from here.

    A path that exists and is not a real directory is never trusted: zipimport
    is a default sys.path hook, so a ZIP named `unsloth_compiled_cache` passed
    every isfile check and served moe_utils out of the archive.

    True when there is no copy at all; the caller runs
    compiler._reject_shadowing_import_candidates first, which refuses a
    moe_utils package, extension or sourceless pyc. Matching bytes are necessary
    and NOT sufficient, hence the name: a bare import prefers the pyc, so it is
    dropped here and one that cannot be dropped means no.
    """
    try:
        # Exists but is not a directory, not merely "is not a directory": a path
        # that does not exist yet imports nothing and stays trusted, which is
        # what a cache folder looks like before it is created.
        if os.path.exists(directory) and not os.path.isdir(directory):
            return False
        cache_file = os.path.abspath(os.path.join(directory, "moe_utils.py"))
    except Exception:
        return False
    current_file = os.path.abspath(__file__)
    if cache_file == current_file or not os.path.isfile(cache_file):
        return True
    if not _cached_copy_is_current(cache_file, current_file):
        return False
    return _remove_cached_bytecode(cache_file)


def _load_cached_moe_utils_module():
    global _CACHED_MOE_UTILS_MODULE

    cache_file = os.path.abspath(os.path.join(_get_compile_location(), "moe_utils.py"))
    current_file = os.path.abspath(__file__)
    if not os.path.isfile(cache_file) or cache_file == current_file:
        return None
    if not _cached_copy_is_current(cache_file, current_file):
        _CACHED_MOE_UTILS_MODULE = None
        return None

    # The cache copy is only ever a copy of this file, so bytes that differ are
    # bytes install_to_cache() did not put there, and exec_module() below runs
    # whatever is in the file. Use this module's own definitions instead, which
    # is what every caller falls back to anyway.
    cached_bytes = _read_file_bytes(cache_file)
    if cached_bytes is None or cached_bytes != _read_file_bytes(current_file):
        return None

    try:
        module_name = "unsloth_cached_moe_utils"
        module = sys.modules.get(module_name, None)
        if module is not None and os.path.abspath(getattr(module, "__file__", "")) == cache_file:
            _CACHED_MOE_UTILS_MODULE = module
            return module

        if not _remove_cached_bytecode(cache_file):
            return None

        spec = importlib.util.spec_from_file_location(module_name, cache_file)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            # The bytes compared above, not a fresh read of the path. exec_module
            # reopens the file, and the comparison and that open are two moments:
            # a writer with access to the shared cache replaces the file between
            # them and its bytes run as unsloth_cached_moe_utils having matched
            # nothing. compiler.py's loader closed the same window; this one is
            # the copy of it that lives here.
            exec(compile(cached_bytes, cache_file, "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        _CACHED_MOE_UTILS_MODULE = module
        return module
    except Exception:
        return None


def get_forward_moe_backend():
    """Resolve forward_moe_backend from the compiled cache copy, else the local def."""
    global _CACHED_FORWARD_MOE_BACKEND
    module = _load_cached_moe_utils_module()
    if module is not None and hasattr(module, "forward_moe_backend"):
        _CACHED_FORWARD_MOE_BACKEND = module.forward_moe_backend
        return _CACHED_FORWARD_MOE_BACKEND

    _CACHED_FORWARD_MOE_BACKEND = forward_moe_backend
    return _CACHED_FORWARD_MOE_BACKEND

# Grouped MM wrapper around torch._grouped_mm; native backward works correctly.


def _grouped_mm_with_backward_fix(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Grouped matmul; passes the weight as a transposed view (no copy) when safe.

    Forcing weight.contiguous() copies the frozen base stack (~805 MB for gate_up on Qwen3-30B,
    ~57% of MoE GPU time) every step. torch._grouped_mm takes the non-contiguous view directly,
    but some CUDA builds silently miscompute it (pytorch/pytorch#186365), so we only skip the
    copy when a one-time probe proves the view path matches the contiguous one; else we keep the
    always-correct copy. Falls back to a per-group matmul when the device has no
    torch._grouped_mm, and on the 16-byte stride error. Bit-exact vs the always-contiguous
    path in forward and backward.
    """
    inputs = inputs.contiguous()
    # The Triton backend is picked precisely when the probe says no, yet its separated
    # LoRA delta still routed here. torch 2.8 hard-raises unless `dprops->major == 9`
    # (Blas.cpp, sm90_only) and 2.6/2.7 have no `_grouped_mm`, so a LoRA MoE died on
    # every card but an H100; 2.9 falls back internally. Probe is cached: one read.
    if not _check_torch_grouped_mm_supported():
        return _manual_grouped_mm(inputs, weight, offsets)
    if not _transposed_view_grouped_mm_is_safe():
        weight = weight.contiguous()   # #186365: view path unproven on this build -> safe copy
    try:
        return torch._grouped_mm(inputs, weight, offs=offsets)
    except RuntimeError as exc:
        if "strides should be multiple of 16 bytes" not in str(exc):
            raise
    weight = weight.contiguous()
    try:
        return torch._grouped_mm(inputs, weight, offs=offsets)
    except RuntimeError as exc:
        if "strides should be multiple of 16 bytes" not in str(exc):
            raise
        return _manual_grouped_mm(inputs, weight, offsets)


def _grouped_matmul_loop(inputs, weight, offsets, bounds = None):
    """out[s:e] = inputs[s:e] @ weight[g], group by group. No autograd.

    `bounds` is the already-decoded group ends: the signature packs the offsets into
    its own transfer, so re-reading them here was a second stream synchronization per
    grouped matmul, over every layer's base and LoRA projections.
    """
    outputs = []
    start = 0
    if bounds is None: bounds = offsets.detach().cpu().tolist()
    for expert_idx, end in enumerate(bounds):
        if start < end:
            outputs.append(torch.matmul(inputs[start:end], weight[expert_idx]))
        start = end
    if outputs:
        return torch.cat(outputs, dim=0)
    return inputs.new_empty((0, weight.shape[-1]))


# Projection strides for the routing signature. A CHECKSUM, not an identity: four
# projections leave a `hidden - 4` null space, so rows differing within it project
# alike; an exact identity means copying the whole `[T, hidden]` input, the 512MB
# transient this file avoids. A swap escapes only by matching ALL of these AND the
# norm below, each term weighted by the row index.
_SIGNATURE_STRIDES = (12.9898, 78.233, 43.7585, 96.4271)

# Row 2-norm, carried beside the projections. Not linear in the row, so that null
# space does not reach it: rows differing within it must also match norms to swap unseen.
_SIGNATURE_EXTRA = 1

# Trailing checksum entries of a packed signature, not group boundaries. Named because
# offsets are read back by slicing it off, and a bare `[:-1]` read checksums as experts.
_SIGNATURE_WIDTH = len(_SIGNATURE_STRIDES) + _SIGNATURE_EXTRA


def _routing_signature(inputs, offsets):
    """Offsets plus an order-sensitive checksum of the expert-sorted rows.

    Offsets are only the routing histogram, so a replay swapping two tokens between
    equal-sized experts leaves them identical while `inputs` holds a different
    sequence; the checksum moves with the rows, so the swap is visible. Bit-cast into
    the offsets vector so the whole thing is one device sync, the one the group loop
    already pays.

    Rows reduce through a fixed pseudo-random projection, not a sum: a sum ignores
    WHERE a row's values sit, so `[1, 0]` and `[0, 1]` reduced alike and a swap across
    an expert boundary went unseen, letting the guard accept gradients from a different
    routing. The projection derives from the hidden size alone, so forward and
    recompute build the same one without carrying state.

    The matvec runs in the input's own dtype, not an FP32 copy: upcasting on the way in
    materialised a whole `[routed_tokens, hidden]` transient, 512MB at 32K by 4096, to
    produce one number per row -- an OOM risk on exactly the memory-constrained runs
    this fallback exists for. cuBLAS accumulates a half-precision gemv in FP32 anyway.
    """
    inputs = inputs.detach()
    hidden = inputs.shape[-1]
    # Autocast off, explicitly: `mv`/`mm` are on the autocast lower-precision list and
    # only ONE of the two calls sees it (forward with the caller's autocast live,
    # backward with it disabled), so FP32 inputs hashed bf16 one side and fp32 the
    # other and every backward raised the routing error below on unchanged routing.
    with torch.autocast(device_type = inputs.device.type, enabled = False):
        weights = torch.linspace(
            1.0, 2.0, hidden, device = inputs.device, dtype = torch.float32)
        # Irrational stride: a linear ramp alone sums the same under a reversal.
        strides = torch.tensor(
            _SIGNATURE_STRIDES, device = inputs.device, dtype = torch.float32)
        weights = torch.sin(
            weights.unsqueeze(1) * strides).to(inputs.dtype)
        # Several projections, not one: a single dot maps each row to one scalar, so
        # rows orthogonal to it hash like the zero row (`[p[1], -p[0], 0, ...]` and
        # zeros both give exactly 0) and swapping that pair across an expert boundary
        # survived the check.
        rows = (inputs @ weights).float()
        # Plus the norm, not linear in the row, so the projections' shared null space
        # does not reach it. `vector_norm` accumulates in fp32 from the input's own
        # dtype; `(x.float() * x.float()).sum(...)` held three concurrent 512MB
        # temporaries at the 32K-by-4096 shape this fallback targets. Promoted, not
        # pinned to fp32: `vector_norm` refuses a dtype that narrows its input.
        norm = torch.linalg.vector_norm(
            inputs, dim = -1, keepdim = True,
            dtype = torch.promote_types(inputs.dtype, torch.float32))
        rows = torch.cat((rows, norm.to(rows.dtype)), -1)
        ramp = torch.arange(
            1, rows.shape[0] + 1, device = rows.device, dtype = rows.dtype)
        checksum = (rows * ramp.unsqueeze(1)).sum(0)
    packed = torch.cat((
        offsets.detach().reshape(-1).to(torch.int64),
        checksum.view(torch.int32).to(torch.int64),
    ))
    return packed.cpu().tolist()


class _ManualGroupedMM(torch.autograd.Function):
    """The loop above, saving what torch._grouped_mm saves and nothing else.

    A Function rather than plain autograd because the naive loop tapes every per-group
    SLICE, whose shape is the group size the router decides. Non-reentrant checkpointing
    replays the forward and compares saved metadata, so any routing difference between
    the two passes surfaces as

        CheckpointError: Recomputed values ... have different metadata
        saved: torch.Size([38, 8])  recomputed: torch.Size([39, 8])

    one row apart, in a hundred groups at once. `torch._grouped_mm` never shows that:
    it is one op saving the whole `[T, K]` input, so a drop-in has to save the same
    shape-stable set (inputs, weight, offsets) and rebuild the slices in backward.

    This does not make routing deterministic, and does not pretend to: it restores the
    numerics the fused path already has on an H100.
    """
    @staticmethod
    def forward(ctx, inputs, weight, offsets):
        # A plain list, NOT a tensor: non-reentrant checkpointing swaps the saved
        # TENSORS for the replay's, so a saved `offsets` reports the replay's routing,
        # never the forward's. This copy survives, and backward compares the two.
        ctx.forward_routing = routing = _routing_signature(inputs, offsets)
        ctx.save_for_backward(inputs, weight, offsets)
        with torch.no_grad():
            return _grouped_matmul_loop(
                inputs, weight, offsets, routing[:-_SIGNATURE_WIDTH])

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, offsets = ctx.saved_tensors
        routing = _routing_signature(inputs, offsets)
        if routing != ctx.forward_routing:
            # Shape-stable saves would let this through silently, and pairing the
            # original `grad_output` with the replay's partition is a gradient for a
            # routing that never produced the loss. Louder than CheckpointError.
            were = ctx.forward_routing[:-_SIGNATURE_WIDTH]
            now = routing[:-_SIGNATURE_WIDTH]
            how = ("the same experts in a different order"
                   if were == now else f"expert ends {were} then {now}")
            raise RuntimeError(
                "Unsloth: the MoE router assigned tokens differently in the "
                f"activation-checkpoint replay than in the forward ({how}), so "
                "the gradients would belong to a routing that never produced "
                "the loss. Turn gradient checkpointing off for this run, or "
                "make the router deterministic."
            )
        bounds = routing[:-_SIGNATURE_WIDTH]
        need_x, need_w, _ = ctx.needs_input_grad
        grad_output = grad_output.contiguous()
        # `backward` runs OUTSIDE the forward's autocast, so `grad_output` can be bf16
        # while the saved tensors are still fp32 and the first matmul raises. Aligned
        # by hand, not with `torch.amp.custom_fwd/custom_bwd`, whose `device_type` is
        # fixed at class definition: this fallback also runs on CPU and XPU.
        compute_dtype = grad_output.dtype
        grad_inputs = torch.zeros_like(inputs) if need_x else None
        grad_weight = torch.zeros_like(weight) if need_w else None
        # Write each group straight into its slice when no cast is due. The assignment
        # below allocates a temporary and copies it in; `out=` does neither, and there
        # are two per group -- 256 on a 128-expert layer, 37% of this loop measured.
        # A cast needs the temporary anyway, and a non-contiguous destination would put
        # the copy back, so both keep the plain path.
        # Grad mode first: it is off for an ordinary backward and ON only under
        # `create_graph = True`, where `out=` raises "functions with out=... arguments
        # don't support automatic differentiation". The fused path double-backwards
        # fine, so the fallback has to as well; the plain branch is differentiable.
        direct = (
            not torch.is_grad_enabled()
            and inputs.dtype == weight.dtype == compute_dtype
            and (grad_inputs is None or grad_inputs.is_contiguous())
            and (grad_weight is None or grad_weight.is_contiguous())
        )
        start = 0
        for expert_idx, end in enumerate(bounds):
            if start < end:
                g = grad_output[start:end]
                if need_x:
                    w = weight[expert_idx]
                    if direct:
                        torch.matmul(g, w.transpose(-2, -1),
                                     out = grad_inputs[start:end])
                    else:
                        grad_inputs[start:end] = (
                            g @ w.to(compute_dtype).transpose(-2, -1)
                        ).to(inputs.dtype)
                if need_w:
                    if direct:
                        torch.matmul(inputs[start:end].transpose(-2, -1), g,
                                     out = grad_weight[expert_idx])
                    else:
                        x = inputs[start:end].to(compute_dtype)
                        grad_weight[expert_idx] = (
                            x.transpose(-2, -1) @ g).to(weight.dtype)
            start = end
        return grad_inputs, grad_weight, None


def _manual_grouped_mm(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Differentiable grouped matmul fallback for torch._grouped_mm alignment gaps.

    Not compilable, and marked so: group boundaries come off a tensor, so `start < end`
    is a data-dependent branch Dynamo cannot guard, and a tensorized rewrite is worse
    (gathering `weight[expert_id]` per row materializes `[T, K, N]`; masking runs every
    expert over every row). `torch.compiler.disable` makes a compiled caller break
    cleanly here rather than abort mid-trace; under `fullgraph = True` a break is still
    fatal, which is what the eager fallback in `temporary_patches/utils.py` is for.
    """
    # Grad off means `apply` builds no autograd node, so nothing will ever read the
    # signature: skip the Function rather than pay a `[T, hidden]` projection per call.
    # Covers `inference_mode` and `no_grad` alike.
    #
    # KNOWN GAP, deliberately not closed here. A REENTRANT checkpoint also runs its
    # loss-producing forward with grad off and its replay with grad ON, so the replay
    # compares its own routing against itself and a reroute passes silently. Not
    # fixable from inside this function: `ctx` cannot cross a reentrant boundary, and a
    # side channel keyed on the weight fails too because
    # `_canonical_lora_weights_for_grouped_mm` rebuilds a contiguous tensor per call.
    # It needs `unsloth_checkpoint` to announce its region and call order. The
    # non-reentrant path is unaffected: its forward runs with grad on, so the forward
    # ctx survives and only its SAVED TENSORS are swapped.
    if not torch.is_grad_enabled():
        return _grouped_matmul_loop(inputs, weight, offsets)
    return _ManualGroupedMM.apply(inputs, weight, offsets)


if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _manual_grouped_mm = torch.compiler.disable(_manual_grouped_mm)


# Recompute-in-backward for the frozen base expert GEMM: the dequantized bf16 stack
# is rebuilt from the 4-bit Params4bit in backward (dX only; the base is frozen and
# LoRA is a separate additive grouped_mm) instead of being pinned on the tape. Output
# is unchanged. See _moe_recompute_enabled for the pin-vs-recompute policy.


def _base_is_recomputable(source) -> bool:
    """True iff the base expert weight can be rebuilt in backward (frozen and
    grouped-mm capable). A trainable or unsupported base must use the pinned path."""
    try:
        if not _should_use_separated_lora():          # merged LoRA folds the delta into base
            return False
        if not _check_torch_grouped_mm_supported():
            return False
        param = source
        while hasattr(param, "base_layer"):
            param = param.base_layer
        if HAS_BNB and Params4bit is not None and isinstance(param, Params4bit):
            if getattr(param, "quant_state", None) is None:
                return False
            return not param.requires_grad
        if isinstance(param, torch.Tensor):
            return (not param.requires_grad) and param.dtype in (
                torch.bfloat16, torch.float16, torch.float32,
            )
    except Exception:
        return False
    return False


def _moe_recompute_default(prefer_memory: bool = False) -> bool:
    """Pin-vs-recompute decision independent of the source weight.

    Adaptive by default: pin (return False) inside a gradient-checkpoint recompute
    pass, where the stack is rebuilt immediately before the layer's own backward so
    the pin is momentary and cheap; otherwise recompute (return True), so a
    non-checkpointed forward does not hold every layer's dense stack across the whole
    backward. UNSLOTH_MOE_RECOMPUTE overrides it: "1" forces recompute (max memory
    saving), "0" forces pinning (max speed for memory-rich runs).

    ``prefer_memory`` biases the no-override case toward recompute even inside a GC
    recompute pass. It is set for bases whose pinned form is a large dense dequant of
    a compressed weight (bnb 4-bit MoE experts): there the "momentary" pin still
    materializes the full bf16 expert stack the 4-bit storage exists to avoid, which
    can be several GiB per layer on large MoEs, so recompute is the better default
    when the model is already memory-constrained (4-bit + gradient checkpointing).

    The GC branch reads a thread-local; under torch.compile of the MoE forward that
    read can be traced away and frozen at the first trace, so the adaptive choice may
    not re-evaluate per GC pass. That only trades memory for speed (pin and recompute
    are dX-identical), never correctness, and the grouped GEMM path generally runs
    eager anyway; set UNSLOTH_MOE_RECOMPUTE explicitly to pin the choice if needed."""
    override = os.environ.get("UNSLOTH_MOE_RECOMPUTE")
    if override == "1":
        return True
    if override == "0":
        return False
    if prefer_memory:
        return True
    try:
        from unsloth_zoo.gradient_checkpointing import in_gradient_checkpoint_recompute
        return not in_gradient_checkpoint_recompute()
    except Exception:
        return True  # safe default: recompute rather than pin across a full backward


def _source_pins_large_dequant(source) -> bool:
    """True iff pinning this base means holding a large dense dequant of a compressed
    weight (a frozen bnb 4-bit MoE expert). For these the pinned path materializes the
    full bf16 expert stack, so recompute is the memory-preserving default under
    gradient checkpointing; a plain bf16/fp16/fp32 base pins its own storage (no
    extra dequant) and keeps the speed-oriented adaptive policy."""
    if not (HAS_BNB and Params4bit is not None):
        return False
    try:
        param = source
        while hasattr(param, "base_layer"):
            param = param.base_layer
        return isinstance(param, Params4bit) and getattr(param, "quant_state", None) is not None
    except Exception:
        return False


def _moe_recompute_enabled(source) -> bool:
    """Whether to recompute the dequantized base stack in backward (True) or pin it
    for reuse (False). Only a frozen, grouped-mm-capable base can be recomputed; for
    everything else the pinned eager path is used. A bnb 4-bit base prefers recompute
    even under gradient checkpointing so the momentary pin never holds the full bf16
    expert dequant (see _source_pins_large_dequant)."""
    return _base_is_recomputable(source) and _moe_recompute_default(
        prefer_memory = _source_pins_large_dequant(source)
    )


class _GroupedMMRecompute(torch.autograd.Function):
    """grouped_mm(inputs, W) for a frozen W from weight_provider(): saves only offsets and rebuilds
    W in backward (dX only) instead of pinning the dense stack."""

    @staticmethod
    def forward(ctx, inputs, offsets, weight_provider):
        ctx.weight_provider = weight_provider
        ctx.save_for_backward(offsets)   # inputs is unused in backward (frozen base -> dX only)
        with torch.no_grad():
            out = _grouped_mm_with_backward_fix(inputs, weight_provider(), offsets)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (offsets,) = ctx.saved_tensors
        with torch.no_grad():
            weight_t = ctx.weight_provider().transpose(-2, -1).contiguous()
            grad_input = _grouped_mm_with_backward_fix(grad_output.contiguous(), weight_t, offsets)
        return grad_input, None, None


def _base_grouped_mm(inputs, offsets, weight_provider, recompute):
    """recompute -> rebuild W in backward; else the prior eager grouped_mm."""
    if recompute:
        return _GroupedMMRecompute.apply(inputs, offsets, weight_provider)
    return _grouped_mm_with_backward_fix(inputs, weight_provider(), offsets)


_GROUPED_GEMM_AVAILABLE = None
_TORCH_GROUPED_MM_AVAILABLE = hasattr(torch, "_grouped_mm")

# GPU support for torch._grouped_mm, verified via runtime probe.
_TORCH_GROUPED_MM_SUPPORTED = None


def _check_torch_grouped_mm_supported():
    """Check torch._grouped_mm support on the current GPU; a runtime probe is the only reliable check."""
    global _TORCH_GROUPED_MM_SUPPORTED
    if _TORCH_GROUPED_MM_SUPPORTED is not None: return _TORCH_GROUPED_MM_SUPPORTED

    if not _TORCH_GROUPED_MM_AVAILABLE:
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    # Typed device, not a bare index: an int resolves to the default accelerator, losing this branch.
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu", torch.xpu.current_device())
    else:
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    try:
        # Dummy call verifies real support (symbol may exist but hardware unsupported, e.g. < H100).
        dtype = torch.float16

        # 1 expert, 1 token, dim 8 (safe alignment).
        x = torch.ones((1, 8), device=device, dtype=dtype)
        w = torch.ones((1, 8, 8), device=device, dtype=dtype)
        offs = torch.tensor([1], device=device, dtype=torch.int32)

        torch._grouped_mm(x, w, offs=offs)
        del x, w, offs
        _TORCH_GROUPED_MM_SUPPORTED = True
    except Exception:
        _TORCH_GROUPED_MM_SUPPORTED = False

    return _TORCH_GROUPED_MM_SUPPORTED


# Some CUDA builds silently miscompute torch._grouped_mm for a transposed bf16 view preceded by a
# broadcast op (pytorch/pytorch#186365, Blackwell + torch 2.11/2.13). This probe checks the view
# matches the contiguous copy so _grouped_mm_with_backward_fix can skip the copy only when safe.
_TRANSPOSED_VIEW_GROUPED_MM_SAFE = None


def _transposed_view_grouped_mm_is_safe():
    global _TRANSPOSED_VIEW_GROUPED_MM_SAFE
    if _TRANSPOSED_VIEW_GROUPED_MM_SAFE is not None:
        return _TRANSPOSED_VIEW_GROUPED_MM_SAFE

    safe = False
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu", torch.xpu.current_device())
        else:
            device = None
        if _TORCH_GROUPED_MM_AVAILABLE and device is not None:
            E, N, K, M = 4, 64, 32, 32
            # local generator: never touch the process-wide RNG (manual_seed would shift training)
            gen = torch.Generator(device=device).manual_seed(0)
            A = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=gen)
            w = torch.randn(E, N, K, dtype=torch.bfloat16, device=device, generator=gen)
            w_t = w.transpose(-2, -1)
            w_tc = w_t.contiguous()
            per = M // E
            offs = torch.arange(per, M + 1, per, dtype=torch.int32, device=device)[:E]
            offs[-1] = M
            ok, ref = True, None
            for _ in range(6):
                row_wise_max = A.abs().amax(dim=-1, keepdim=True)
                _ = A / (row_wise_max / 448.0)     # the #186365 trigger (result discarded)
                r_view = torch._grouped_mm(A, w_t, offs=offs)
                r_contig = torch._grouped_mm(A, w_tc, offs=offs)
                if (r_view - r_contig).abs().max().item() > 1e-2:   # view disagrees with contiguous
                    ok = False; break
                if ref is None:
                    ref = r_view
                elif (r_view - ref).abs().max().item() > 1e-2:      # view not stable across calls
                    ok = False; break
            safe = ok
    except Exception:
        safe = False   # anything unexpected -> keep the safe contiguous copy

    _TRANSPOSED_VIEW_GROUPED_MM_SAFE = safe
    return safe


_TRITON_ALLOCATOR_INITIALIZED = False
_PERSISTENT_BUFFER = None
_original_peft_get_peft_model = None


def _init_triton_allocator():
    """Initialize a persistent Triton allocator to avoid per-call allocation overhead."""
    global _TRITON_ALLOCATOR_INITIALIZED, _PERSISTENT_BUFFER
    if _TRITON_ALLOCATOR_INITIALIZED: return

    try:
        import triton

        # Persistent buffer that grows as needed, avoiding per-kernel allocations.
        def persistent_alloc_fn(size: int, alignment: int, stream):
            global _PERSISTENT_BUFFER
            # Round up to nearest 128 bytes for alignment / fewer reallocations.
            rounded_size = ((size + 128 - 1) // 128) * 128

            if (
                _PERSISTENT_BUFFER is None
                or _PERSISTENT_BUFFER.numel() * _PERSISTENT_BUFFER.element_size()
                < rounded_size
            ):
                # 10% headroom; uint8 for raw byte storage.
                _PERSISTENT_BUFFER = torch.empty(
                    int(rounded_size * 1.1), device="cuda", dtype=torch.uint8
                )
                _PERSISTENT_BUFFER.__hibernate__ = {"type": "ignore"}
            return _PERSISTENT_BUFFER

        triton.set_allocator(persistent_alloc_fn)
        triton._unsloth_allocator_set = True
        _TRITON_ALLOCATOR_INITIALIZED = True
    except Exception:
        pass


def _check_grouped_gemm_available():
    """Check if Unsloth grouped GEMM kernels are available."""
    if os.environ.get("UNSLOTH_DISABLE_MOE_TRITON", "0") == "1": return False
    if is_mlx_available(): return False

    global _GROUPED_GEMM_AVAILABLE
    if _GROUPED_GEMM_AVAILABLE is not None: return _GROUPED_GEMM_AVAILABLE

    try:
        from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm, supports_tma
        _GROUPED_GEMM_AVAILABLE = True
        _init_triton_allocator()
    except (ImportError, ModuleNotFoundError):
        _GROUPED_GEMM_AVAILABLE = False
    return _GROUPED_GEMM_AVAILABLE


from functools import lru_cache, wraps


@lru_cache(maxsize=1)
def select_moe_backend():
    """Select MoE backend from UNSLOTH_MOE_BACKEND + availability.

    Choices: "grouped_mm", "unsloth_triton", "native_torch" (default "grouped_mm").
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    requested = os.environ.get("UNSLOTH_MOE_BACKEND")
    if requested:
        if requested == "grouped_mm" and _check_torch_grouped_mm_supported():
            return "grouped_mm"
        if requested == "unsloth_triton" and _check_grouped_gemm_available():
            return "unsloth_triton"
        if requested == "native_torch":
            return "native_torch"
        _log_info(f"Unsloth: '{requested}' backend requested but is not available. Falling back to next available.")

    if _check_torch_grouped_mm_supported():
        _log_info("Unsloth: Using MoE backend 'grouped_mm'")
        return "grouped_mm"
    if _check_grouped_gemm_available():
        _log_info("Unsloth: Using MoE backend 'unsloth_triton'")
        return "unsloth_triton"
    return "native_torch"


def swap_moe_weights_for_call(experts_module, gate_up_proj, down_proj, forward_fn, *args):
    """Temporarily install dequantized weights for one forward call, then restore.

    Uses object.__setattr__ to bypass nn.Module Parameter (de)registration
    (re-registers hooks, unnecessary for read-only temp tensors). Used by the
    FP8 and bnb4bit MoE dispatchers.
    """
    original_gate_up = experts_module.gate_up_proj
    original_down = experts_module.down_proj
    object.__setattr__(experts_module, "gate_up_proj", gate_up_proj)
    object.__setattr__(experts_module, "down_proj", down_proj)
    try:
        return forward_fn(experts_module, *args)
    finally:
        object.__setattr__(experts_module, "gate_up_proj", original_gate_up)
        object.__setattr__(experts_module, "down_proj", original_down)


def forward_moe_backend(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Dispatch MoE forward to the selected backend (keeps model-specific patches minimal)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    # Absolute imports: this function is also copied into
    # unsloth_compiled_cache/moe_utils.py where relative imports of sibling
    # helpers don't resolve (only the dispatcher is copied).
    # Keep `except ImportError` around ONLY the import; runtime errors in the
    # bnb4bit/fp8 path must propagate, not fall through to a crashing backend.
    _moe_uses_bnb4bit_expert_weights = forward_moe_backend_bnb4bit = None
    try:
        from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import (
            _moe_uses_bnb4bit_expert_weights,
            forward_moe_backend_bnb4bit,
        )
    except ImportError:
        pass
    if _moe_uses_bnb4bit_expert_weights is not None and _moe_uses_bnb4bit_expert_weights(self):
        result = forward_moe_backend_bnb4bit(self, hidden_states, top_k_index, top_k_weights)
        if result is not None:
            return result

    _moe_uses_fp8_expert_weights = forward_moe_backend_fp8 = None
    try:
        from unsloth_zoo.temporary_patches.moe_utils_fp8 import (
            _moe_uses_fp8_expert_weights,
            forward_moe_backend_fp8,
        )
    except ImportError:
        pass
    if _moe_uses_fp8_expert_weights is not None and _moe_uses_fp8_expert_weights(self):
        return forward_moe_backend_fp8(self, hidden_states, top_k_index, top_k_weights)

    backend = select_moe_backend()
    if backend == "grouped_mm":
        return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)
    if backend == "unsloth_triton":
        return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)
    return forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights)


# Test-only call counter, wired at import time and OFF by default: incrementing a
# module global is a Dynamo side effect, and leaving it always-on re-traced the
# compiled MoE block every step (3.6 ms -> 572 ms).
_EXPERT_COUNT_CALLS = [0]
_COUNT_DEBUG = os.environ.get("UNSLOTH_MOE_COUNT_DEBUG", "0") == "1"


def _count_tokens_per_expert(
    flat_experts: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Per-expert token counts, WITHOUT synchronising the device.

    Drop-in replacement for ``torch.bincount(flat_experts, minlength=num_experts)``
    returning shape ``[num_experts]`` and the requested ``dtype``.

    Why not bincount: on CUDA the output size of ``bincount`` depends on the data
    (it must learn ``max(input)`` even when ``minlength`` is given), so the kernel
    D2H-copies and blocks the host. Measured on B200, E=256 / 16384 routed rows:
    2 sync warnings under ``torch.cuda.set_sync_debug_mode("warn")`` and ~58 us of
    host time per call, versus 0 warnings and ~19 us here. With one MoE layer per
    block that drain happens tens of times per step.

    ``scatter_add_`` accumulates with atomics, so the ORDER of accumulation is
    nondeterministic; integer addition is associative and commutative and every
    addend is exactly 1, so the SUM is bit-exact and run-to-run deterministic
    regardless of order. Verified bit-identical to ``bincount`` over random,
    skewed, empty and single-token routings, and it does not trip
    ``torch.use_deterministic_algorithms(True)``.

    Caveat, same precondition as the caller already relies on: ``flat_experts``
    must hold values in ``[0, num_experts)``. That holds for router ``topk``
    output by construction. Out-of-range values make ``bincount`` silently return
    a LONGER tensor (breaking the downstream ``offs=`` shape) whereas this writes
    out of bounds, so neither is safe; the difference is only in how it fails.
    """
    # This Unsloth Zoo code section is licensed under AGPL3
    if flat_experts.dim() != 1:
        flat_experts = flat_experts.reshape(-1)
    # scatter_add_ needs int64; router topk already gives one, so no copy.
    index = flat_experts if flat_experts.dtype == torch.int64 else flat_experts.long()
    counts = torch.zeros(num_experts, dtype=dtype, device=flat_experts.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=dtype))
    return counts


if _COUNT_DEBUG:
    def count_tokens_per_expert(flat_experts, num_experts, dtype=torch.int32):
        _EXPERT_COUNT_CALLS[0] += 1
        return _count_tokens_per_expert(flat_experts, num_experts, dtype)

    count_tokens_per_expert.__doc__ = _count_tokens_per_expert.__doc__
else:
    count_tokens_per_expert = _count_tokens_per_expert


@torch.no_grad()
def _get_routing_indices(selected_experts, num_experts):
    """Compute token->expert mapping for grouped GEMM.

    Returns (token_counts_by_expert (num_experts,), gather_indices (total_tokens,)).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    flat_experts = selected_experts.view(-1)

    # Sync-free; see count_tokens_per_expert.
    token_counts_by_expert = count_tokens_per_expert(flat_experts, num_experts, torch.int32)

    # stable=True preserves order within each expert.
    gather_indices = flat_experts.argsort(stable=True)

    return token_counts_by_expert, gather_indices


def combine_permuted_moe_outputs(
    permuted_output: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    top_k: int,
    out_dtype = None,
) -> torch.Tensor:
    """Sum the top_k expert outputs belonging to each token, in a fixed order.

    The obvious spelling of this reduction is
    ``zeros(num_tokens, hidden).index_add_(0, token_indices, permuted_output)``
    with ``token_indices = sorted_indices // top_k``. Because every token index
    appears ``top_k`` times in one call, that lands in ``index_add_``'s CUDA
    atomicAdd path, and atomicAdd fixes no accumulation order. Float addition is
    not associative, so the result moves from run to run: repeating one identical
    forward 40 times on a single already-loaded Gemma-4 MoE, with no optimizer
    step and no data change, gave 40 distinct losses spread over 0.0284 nats,
    while the same 40 repeats through the reduction below returned one value.

    ``sorted_indices`` is ``argsort`` of the flat expert assignment, so it is a
    permutation of ``range(num_tokens * top_k)`` - every slot exactly once. That
    means the permutation can simply be undone (a gather through the inverse
    permutation: unique indices, no accumulation and so no atomics) and the
    ``top_k`` axis reduced with an ordinary ``sum``, which has a fixed reduction
    order. Same arithmetic, same values up to that ordering, reproducible.

    ``out_dtype`` is applied AFTER the reduction, not before it. When the routed
    slots arrive wider than ``out_dtype`` - the fp32-router case, where the
    routing-weight multiply promotes the expert output to fp32 - casting first
    would round every one of the ``top_k`` summands to bf16 and only then add
    them. Summing first and rounding once is what ``transformers``
    (``integrations/moe.py``) does, and it is measurably closer to an fp64
    reference. With a bf16 router the two orders are bit-identical.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # The gather below is safe only because sorted_indices is an argsort, so the indices
    # are unique and, given the count check here matches, cover every row: nothing is left
    # unread and no slot is counted twice. A bare `assert` would vanish under `python -O`,
    # which is exactly the build where a silently wrong reduction is hardest to notice.
    if sorted_indices.numel() != permuted_output.shape[0]:
        raise ValueError(
            f"Unsloth: expected a full permutation of {permuted_output.shape[0]} routed slots, "
            f"got {sorted_indices.numel()} indices"
        )
    # inverse_indices[sorted_indices[i]] = i, i.e. "which routed slot holds original row j".
    # Gathering with it is one [num_tokens * top_k, hidden] buffer, in the incoming dtype;
    # the narrowing cast then only ever touches the small [num_tokens, hidden] result.
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices.scatter_(
        0,
        sorted_indices,
        torch.arange(
            sorted_indices.numel(),
            device = sorted_indices.device,
            dtype = sorted_indices.dtype,
        ),
    )
    combined = (
        permuted_output
        .index_select(0, inverse_indices)
        .view(num_tokens, top_k, permuted_output.shape[-1])
        .sum(dim = 1)
    )
    if out_dtype is not None:
        combined = combined.to(out_dtype)
    return combined


def _silu_and_mul(x):
    """Fused SiLU + element-wise multiply for gate/up projections."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


# Separated LoRA helpers.


def _has_lora_adapters(param) -> bool:
    """Check for active LoRA adapters (PEFT ParamWrapper)."""
    if not hasattr(param, "lora_A") or not hasattr(param, "lora_B"):
        return False
    if hasattr(param, "disable_adapters") and param.disable_adapters:
        return False
    if hasattr(param, "merged") and param.merged:
        return False
    return len(param.lora_A) > 0


# How the flat `num_experts * rank` axis of a fused expert `lora_B` is laid out.
#
# PEFT stores a fused MoE expert LoRA as two ordinary Linear weights, lora_A of shape
# (num_experts * rank, in) and lora_B of shape (out, num_experts * rank), and the two are
# NOT flattened the same way. lora_A is expert-slowest, `reshape(E, r, in)`. lora_B is
# expert-FASTEST: `reshape(out, r, E)`, so column j belongs to expert `j % E`. That
# asymmetry is easy to miss, and it is the whole of this constant's reason to exist.
#
# Expert-fastest is what PEFT's own forward (`ParamWrapper.get_delta_factors`) and its
# merge (`ParamWrapper.get_delta_weight`) both do, unchanged across PEFT 0.18 to 0.21;
# what vLLM's `_stack_moe_lora_weights` does when it serves the adapter; and what prime-rl
# writes. It is the only reading any consumer of a saved adapter implements, so it is what
# the separated forward has to train.
LORA_B_LAYOUT_RANK_MAJOR = "rank_major"

# What Unsloth's separated forward read before this fix: expert-SLOWEST, a rank-wide
# contiguous block of columns per expert. Self-consistent, and understood by nothing else.
# Kept only so an adapter trained that way can still be read back, see
# `moe_lora_b_layout()`.
LORA_B_LAYOUT_GROUPED_BY_EXPERT = "grouped_by_expert"

_LORA_B_LAYOUTS = (LORA_B_LAYOUT_RANK_MAJOR, LORA_B_LAYOUT_GROUPED_BY_EXPERT)


class MoELoRABLayoutError(ValueError):
    """`UNSLOTH_MOE_LORA_B_LAYOUT` is set to something that is not a layout.

    Its own class because `_extract_lora_from_wrapper` turns every exception into "this
    wrapper has no LoRA", which would silently drop the adapter and train the base model.
    A configuration mistake has to be louder than that, so it is re-raised there by type.
    """


def _read_moe_lora_b_layout_from_env() -> str:
    """`UNSLOTH_MOE_LORA_B_LAYOUT`, validated. Split out so the value can be refreshed into
    a module global on every eager call, which is what makes the switch visible to a
    compiled graph."""
    # This Unsloth Zoo code section is licensed under AGPL3

    layout = os.environ.get("UNSLOTH_MOE_LORA_B_LAYOUT", LORA_B_LAYOUT_RANK_MAJOR)
    if layout not in _LORA_B_LAYOUTS:
        raise MoELoRABLayoutError(
            f"Unsloth: UNSLOTH_MOE_LORA_B_LAYOUT must be one of {_LORA_B_LAYOUTS}, got "
            f"{layout!r}."
        )
    return layout


_MOE_LORA_B_LAYOUT = _read_moe_lora_b_layout_from_env()


def moe_lora_b_layout() -> str:
    """Which packing to read a fused expert `lora_B` with.

    `rank_major` (PEFT's, and therefore everyone's) unless
    `UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert` asks for the pre-fix reading, which is
    what an adapter trained by an older Unsloth needs.

    It is a process-wide mode, not a load-time option. Nothing is recorded on the adapter,
    and every forward and every merge resolves it again, so setting it only around
    `load_adapter` leaves the rest of the run reading a legacy adapter as `rank_major` and
    scrambling it. Set it before the process does any MoE work and leave it set. Reading
    the layout back off the adapter itself needs the `adapter_config.json` marker, which is
    the migration path, not this switch.
    """
    global _MOE_LORA_B_LAYOUT
    # Refreshed on every call, including while Dynamo traces. Reading the environment at
    # trace time bakes the then-current value into the graph, which is what the supported
    # usage needs: `import unsloth` runs this module's import long before the application
    # sets the variable, so an import-time value alone would be stale for exactly the
    # normal case of setting it before training starts. Gating the refresh on
    # `not is_compiling()` got this wrong, because a run whose first MoE call is already
    # compiled never executes the eager branch at all.
    #
    # What this still does not do is notice a change made AFTER the first compiled call:
    # Dynamo installs a guard on an `os.environ.get` only when the key is set at trace
    # time, so there is nothing to invalidate the graph. That is the documented contract,
    # a process-wide mode set before the process does any MoE work.
    _MOE_LORA_B_LAYOUT = _read_moe_lora_b_layout_from_env()
    return _MOE_LORA_B_LAYOUT


def _resolve_moe_lora_b_layout(layout) -> str:
    """The layout to use, whether the caller named one or left it to the environment.

    A caller that names one still gets it validated. Both readers branch on "is this
    rank_major" and fall through to grouped_by_expert otherwise, so an unvalidated typo
    is not a no-op: it permutes the columns of a standard adapter, silently, which is the
    exact damage `moe_lora_b_layout()` validates the environment variable to prevent. A
    converter passing the layout in from a marker or a command line is the likeliest
    source of one."""
    # This Unsloth Zoo code section is licensed under AGPL3

    if layout is None:
        return moe_lora_b_layout()
    if layout not in _LORA_B_LAYOUTS:
        raise MoELoRABLayoutError(
            f"Unsloth: lora_B layout must be one of {_LORA_B_LAYOUTS}, got {layout!r}."
        )
    return layout


def _moe_lora_b_column_permutation(num_experts, rank_per_expert, device, invert = False):
    """The column permutation between PEFT's rank-major packing and expert-major order.

    Forward: `out[:, k] = weight_B[:, perm[k]]`. The inverse is the same construction with
    the two axes swapped, so neither direction needs the other's indices kept around."""
    total = num_experts * rank_per_expert
    rows, columns = (num_experts, rank_per_expert) if invert else (rank_per_expert, num_experts)
    return torch.arange(total, device = device).view(rows, columns).t().reshape(-1)


class _ExpertMajorGroupedOperand(torch.autograd.Function):
    """`(out, E*rank)` -> `(E, rank, out)` for the grouped GEMM, in ONE materialization.

    Gathering into expert-major column order and then permuting to the operand's shape
    copies the whole tensor twice per projection, on every expert layer of every step.
    Transposing to `(E*rank, out)` first makes the gather itself produce the operand: the
    rows land in expert-major order, `index_select` on dim 0 writes a contiguous result,
    and the final reshape is free.

    Still a real gather, which is the property the grouped GEMM needs: under rank-major
    packing the expert axis has stride 1, so a pure view chain lets Inductor elide the
    copy and hand `aten._grouped_mm` an operand that is neither row nor column major.
    Still saves nothing for the backward either, since the permutation is recomputed from
    two integers, which is what non-reentrant checkpointing requires."""
    # This Unsloth Zoo code section is licensed under AGPL3

    @staticmethod
    def forward(ctx, weight_B, num_experts, rank_per_expert, layout):
        ctx.num_experts = num_experts
        ctx.rank_per_expert = rank_per_expert
        ctx.layout = layout
        ctx.dim_B = weight_B.shape[0]
        rows = weight_B.t()
        if layout == LORA_B_LAYOUT_RANK_MAJOR:
            columns = _moe_lora_b_column_permutation(
                num_experts, rank_per_expert, weight_B.device,
            )
            rows = rows.index_select(0, columns)
        else:
            rows = rows.contiguous()
        return rows.reshape(num_experts, rank_per_expert, ctx.dim_B)

    @staticmethod
    def backward(ctx, grad_output):
        rows = grad_output.reshape(ctx.num_experts * ctx.rank_per_expert, ctx.dim_B)
        if ctx.layout == LORA_B_LAYOUT_RANK_MAJOR:
            columns = _moe_lora_b_column_permutation(
                ctx.num_experts, ctx.rank_per_expert, grad_output.device, invert = True,
            )
            rows = rows.index_select(0, columns)
        return rows.t(), None, None, None


class _ExpertMajorColumns(torch.autograd.Function):
    """Reorder `lora_B`'s columns expert-major, saving nothing for the backward.

    `index_select` would do this in one line, but it saves its index tensor, and
    `torch.utils.checkpoint(use_reentrant = False)` counts every saved tensor and refuses
    when the recompute saves a different number than the forward. The measuring call in
    `_patched_param_wrapper_forward` already runs the experts forward a different number
    of times than the recompute does, so one more saved tensor on the default path is
    enough to turn that into `CheckpointError`, which is transformers 5's default
    configuration and Unsloth's own distributed vision path.

    The permutation is fully determined by two integers, so the backward rebuilds the
    inverse from `ctx` rather than from a saved tensor: nothing is saved, the count is
    unchanged from the pure-view spelling, and the gradient still reaches `lora_B`.
    A real gather is also what Inductor cannot fold back into a strided view of the
    source, which is the other half of why this is not a plain `permute`."""
    # This Unsloth Zoo code section is licensed under AGPL3

    @staticmethod
    def forward(ctx, weight_B, num_experts, rank_per_expert):
        ctx.num_experts = num_experts
        ctx.rank_per_expert = rank_per_expert
        columns = _moe_lora_b_column_permutation(
            num_experts, rank_per_expert, weight_B.device,
        )
        return weight_B.index_select(1, columns)

    @staticmethod
    def backward(ctx, grad_output):
        columns = _moe_lora_b_column_permutation(
            ctx.num_experts, ctx.rank_per_expert, grad_output.device, invert = True,
        )
        return grad_output.index_select(1, columns), None, None


def _moe_lora_b_columns_expert_major(weight_B, num_experts, rank_per_expert, layout):
    """`lora_B`'s columns reordered so the expert index is slowest, without copying.

    Shared by `unflatten_moe_lora_b` and the grouped-GEMM path so the two cannot disagree
    about the column order, while each finishes with its OWN single permute into the shape
    it wants. Doing the reorder here and the permute there is what keeps the copy count at
    one: composing the two public shapes instead (unflatten, then transpose + contiguous)
    materialises the tensor twice, which is an extra saved tensor and breaks non-reentrant
    gradient checkpointing, whose recompute must save exactly what the forward did.

    Returns `weight_B` untouched under grouped-by-expert, where the columns already are
    expert-slowest."""
    # This Unsloth Zoo code section is licensed under AGPL3

    layout = _resolve_moe_lora_b_layout(layout)
    if layout != LORA_B_LAYOUT_RANK_MAJOR:
        return weight_B
    # PEFT packs expert index fastest: column j holds expert `j % num_experts`.
    return _ExpertMajorColumns.apply(weight_B, num_experts, rank_per_expert)


def unflatten_moe_lora_b(
    weight_B: torch.Tensor,
    num_experts: int,
    rank_per_expert: int,
    dim_B: int,
    layout: str = None,
) -> torch.Tensor:
    """`(out, num_experts * rank)` -> `(num_experts, out, rank)`.

    The single place the expert axis of a fused `lora_B` is resolved, so the forward, the
    merge and any converter cannot drift apart. Both layouts end in `.contiguous()` on a
    permuted view, so neither is cheaper than the other and the choice is purely one of
    which convention the stored tensor was written in.

    The rank-major branch regroups the columns with `index_select` before it views them,
    rather than permuting the expert axis out of a three-way view. The two spell the same
    tensor, but only the first survives `torch.compile`. Under rank-major packing the
    expert axis has stride 1 in `weight_B`, so every view that puts experts first leaves
    BOTH remaining axes with a stride above 1, and `aten._grouped_mm` rejects a `mat_b`
    that is neither row nor column major. Eager is fine because `.contiguous()` really
    copies; Inductor folds the whole view chain into one strided read of `weight_B` and
    picks the source layout, so the copy disappears and the consumer sees
    `(1, num_experts, num_experts * rank)` and raises `Invalid strides/sizes`. Measured on
    `gemma-4-26B-A4B-it` (E=128, rank=8, out=1408) and reproduced standalone; the
    pre-existing grouped-by-expert spelling escaped only because eliding ITS copy happens
    to leave a legal column-major operand. `index_select` is a real gather that Inductor
    cannot fold into a view, it keeps the gradient flowing to `lora_B`, and its indices are
    a permutation, so its `index_add` backward has no duplicate targets to reduce
    nondeterministically."""
    weight_B = _moe_lora_b_columns_expert_major(
        weight_B, num_experts, rank_per_expert, layout,
    )
    return weight_B.reshape(dim_B, num_experts, rank_per_expert).permute(1, 0, 2).contiguous()


def moe_lora_b_expert_columns(
    expert_idx: int,
    num_experts: int,
    rank_per_expert: int,
    layout: str = None,
) -> slice:
    """Which columns of a flat `(out, num_experts * rank)` `lora_B` belong to one expert.

    A `slice`, so indexing with it stays a view. The companion of
    `unflatten_moe_lora_b` for the merge paths that walk one expert at a time; the two
    must agree, which is why both live here. `lora_A`'s rows for the same expert are
    always `expert_idx * rank : (expert_idx + 1) * rank`, in both layouts, and the
    resulting column order matches that rank order."""
    layout = _resolve_moe_lora_b_layout(layout)
    if layout == LORA_B_LAYOUT_RANK_MAJOR:
        return slice(expert_idx, num_experts * rank_per_expert, num_experts)
    return slice(expert_idx * rank_per_expert, (expert_idx + 1) * rank_per_expert)


def _canonical_lora_weights_for_grouped_mm(
    weight_A: torch.Tensor,
    weight_B: torch.Tensor,
    num_experts: int,
    rank_per_expert: int,
    dim_A: int,
    dim_B: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
    first_weight = first_weight.permute(0, 2, 1).contiguous()
    # (num_experts, rank, out) for X @ first @ second, in ONE copy. Going through
    # unflatten_moe_lora_b and transposing its result materialises the tensor twice, and
    # the extra saved tensor makes non-reentrant gradient checkpointing recompute a
    # different number of tensors than the forward saved.
    second_weight = _ExpertMajorGroupedOperand.apply(
        weight_B, num_experts, rank_per_expert, _resolve_moe_lora_b_layout(None),
    )
    return first_weight, second_weight


def _reversed_lora_weights_for_grouped_mm(
    weight_A: torch.Tensor,
    weight_B: torch.Tensor,
    num_experts: int,
    rank_per_expert: int,
    dim_A: int,
    dim_B: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_weight = unflatten_moe_lora_b(weight_B, num_experts, rank_per_expert, dim_B)
    second_weight = weight_A.view(num_experts, rank_per_expert, dim_A).contiguous()
    return first_weight, second_weight


def _get_param_shape_from_module(module, parameter_name):
    if module is None or parameter_name is None or not hasattr(module, parameter_name):
        return None
    param = getattr(module, parameter_name)
    if hasattr(param, "get_param"):
        param = param.get_param()
    elif hasattr(param, "weight"):
        param = param.weight
    return tuple(param.shape)


def _get_moe_lora_io_dims(wrapper, experts_module=None):
    base = None
    if wrapper is not None and hasattr(wrapper, "get_base_layer"):
        base = wrapper.get_base_layer()
    if experts_module is None:
        experts_module = base
    if experts_module is None:
        experts_module = getattr(wrapper, "base_layer", None)

    parameter_name = getattr(wrapper, "parameter_name", None)
    source = experts_module if experts_module is not None else base
    if source is None:
        return None, None
    _set_gpt_oss_grouped_mm_format_on_experts(source)

    shape = _get_param_shape_from_module(source, parameter_name)
    if shape is not None and len(shape) >= 3:
        grouped_mm_format = bool(getattr(source, "_unsloth_grouped_mm_format", False))
        if grouped_mm_format:
            return shape[-2], shape[-1]
        return shape[-1], shape[-2]

    hidden_dim = getattr(source, "hidden_dim", None)
    intermediate_dim = getattr(source, "intermediate_dim", None)
    if hidden_dim is None or intermediate_dim is None:
        return None, None
    if parameter_name == "gate_up_proj":
        return hidden_dim, 2 * intermediate_dim
    if parameter_name == "down_proj":
        return intermediate_dim, hidden_dim
    return None, None


def extract_moe_lora_weights_for_grouped_mm(
    wrapper,
    weight_A: torch.Tensor,
    weight_B: torch.Tensor,
    scaling,
    num_experts: int,
    *,
    experts_module=None,
    input_dim=None,
    output_dim=None,
    model_name: str = "MoE",
    enable_logging: bool = None,
    logger_obj=None,
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    total_rank = weight_A.shape[0]
    rank_per_expert = total_rank // num_experts
    dim_A = weight_A.shape[1]
    dim_B = weight_B.shape[0]

    if num_experts <= 1:
        return weight_A.T, weight_B.T, scaling, num_experts

    if input_dim is None or output_dim is None:
        inferred_input_dim, inferred_output_dim = _get_moe_lora_io_dims(
            wrapper, experts_module=experts_module,
        )
        if input_dim is None:
            input_dim = inferred_input_dim
        if output_dim is None:
            output_dim = inferred_output_dim

    canonical_match = (
        input_dim is not None
        and output_dim is not None
        and dim_A == input_dim
        and dim_B == output_dim
    )
    reversed_match = (
        input_dim is not None
        and output_dim is not None
        and dim_A == output_dim
        and dim_B == input_dim
    )

    if canonical_match and reversed_match:
        if bool(getattr(wrapper, "_did_swap_in_out_features", False)):
            first_weight, second_weight = _reversed_lora_weights_for_grouped_mm(
                weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
            )
        else:
            first_weight, second_weight = _canonical_lora_weights_for_grouped_mm(
                weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
            )
        return first_weight, second_weight, scaling, num_experts

    if canonical_match:
        first_weight, second_weight = _canonical_lora_weights_for_grouped_mm(
            weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
        )
        return first_weight, second_weight, scaling, num_experts

    if reversed_match:
        first_weight, second_weight = _reversed_lora_weights_for_grouped_mm(
            weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
        )
        return first_weight, second_weight, scaling, num_experts

    if logger_obj is not None:
        if enable_logging is None:
            enable_logging = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
        if enable_logging and (input_dim is not None or output_dim is not None):
            logger_obj.warning(
                f"Unsloth: {model_name} LoRA extractor could not match either layout "
                f"(weight_A={tuple(weight_A.shape)}, weight_B={tuple(weight_B.shape)}, "
                f"expected input_dim={input_dim}, output_dim={output_dim}, "
                f"num_experts={num_experts}). Falling back to canonical layout. "
                "If this is a new PEFT version, the LoRA delta may be wrong."
        )

    first_weight, second_weight = _canonical_lora_weights_for_grouped_mm(
        weight_A, weight_B, num_experts, rank_per_expert, dim_A, dim_B,
    )
    return first_weight, second_weight, scaling, num_experts


def _extract_lora_from_wrapper(
    wrapper, adapter_name: str = "default", experts_module=None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, int]]:
    """Extract LoRA weights from a PEFT ParamWrapper for MoE separated grouped_mm.

    PEFT 3D ParamWrapper gives lora_A: (E*R, in_dim), lora_B: (out_dim, E*R);
    reshaped to first_weight (E, in_dim, R), second_weight (E, R, out_dim) so
    delta = X @ first @ second. Handles both standard (E, out, in) Qwen3-MoE and
    transposed (E, in, out) Qwen3-VL-MoE base weight layouts.

    Returns (first_weight, second_weight, scaling, num_experts) or None.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    try:
        if not hasattr(wrapper, "lora_A") or not hasattr(wrapper, "lora_B"):
            return None

        if hasattr(wrapper, "disable_adapters") and wrapper.disable_adapters:
            return None
        if hasattr(wrapper, "merged") and wrapper.merged:
            return None

        if not wrapper.lora_A:
            return None

        if adapter_name not in wrapper.lora_A:
            adapter_name = list(wrapper.lora_A.keys())[0]

        lora_A_module = wrapper.lora_A[adapter_name]
        lora_B_module = wrapper.lora_B[adapter_name]

        weight_A = lora_A_module.weight  # (E*R, dim1)
        weight_B = lora_B_module.weight  # (dim2, E*R)
        scaling = wrapper.scaling[adapter_name]
        num_experts = getattr(wrapper, "num_experts", 1)

        if experts_module is None:
            experts_module = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None

        # Model-specific LoRA extractor attached to the experts module, if any.
        extractor_fn = getattr(experts_module, "_unsloth_lora_extractor_fn", None)

        if extractor_fn is not None:
            return extractor_fn(wrapper, weight_A, weight_B, scaling, num_experts)

        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            experts_module=experts_module,
            model_name="MoE",
        )
    except MoELoRABLayoutError:
        # A misspelled UNSLOTH_MOE_LORA_B_LAYOUT must not read as "no adapter here".
        raise
    except Exception:
        return None


def _extract_lora_weights(
    param, adapter_name: str = "default", num_experts: int = None, experts_module=None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Compat wrapper around _extract_lora_from_wrapper; returns (first, second, scaling)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    # Pass num_experts through so _extract_lora_from_wrapper can use it.
    if num_experts is not None and not hasattr(param, "num_experts"):
        param.num_experts = num_experts

    result = _extract_lora_from_wrapper(param, adapter_name, experts_module=experts_module)
    if result is None:
        return None
    return result[0], result[1], result[2]


_DEQUANTIZE_4BIT_IN_SLICES = None          # unresolved
_DEQUANTIZE_4BIT_IN_SLICES_MISSING = False  # resolved, and there is none


def _get_dequantize_4bit_in_slices():
    """The sliced 4-bit read, resolved once.

    Absolute, like the dispatcher above and for the same reason: this file is
    also copied to unsloth_compiled_cache/moe_utils.py and imported as a
    top-level module, where a relative import of a sibling raises and the
    caller would quietly put an oversized stack back on the call that aborts.
    Only the import is guarded; a real failure inside the helper must propagate.
    """
    global _DEQUANTIZE_4BIT_IN_SLICES, _DEQUANTIZE_4BIT_IN_SLICES_MISSING
    if _DEQUANTIZE_4BIT_IN_SLICES is None and not _DEQUANTIZE_4BIT_IN_SLICES_MISSING:
        try:
            from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import (
                _dequantize_4bit_in_slices,
            )
            _DEQUANTIZE_4BIT_IN_SLICES = _dequantize_4bit_in_slices
        except ImportError:
            _DEQUANTIZE_4BIT_IN_SLICES_MISSING = True
    return _DEQUANTIZE_4BIT_IN_SLICES


def _get_base_weight(param, target_dtype=None):
    """Get base weight from a potentially wrapped parameter or module. target_dtype (recompute
    providers) restores the packed Params4bit to its logical shape and casts."""
    # This Unsloth Zoo code section is licensed under AGPL3

    while hasattr(param, "base_layer"):
        param = param.base_layer

    if HAS_BNB and isinstance(param, Params4bit):
        if getattr(param, "quant_state", None) is None:
            raise RuntimeError(
                "unsloth: _get_base_weight saw a Params4bit with quant_state=None. "
                "This usually means the model was used in forward before loading "
                "completed quantization (meta placeholder still in place), or the "
                "MoE quantizer patch did not fire for this expert. "
                f"data.shape={tuple(param.data.shape)}, device={param.device}."
            )
        # An expert stack of 2**31 elements or more aborts inside the
        # bitsandbytes dequantize kernel (csrc/ops.cu line 93), and this is the
        # read the recompute and grouped-mm providers take on every forward and
        # again on every backward recomputation. Slice it the same way the load
        # does; the helper returns None for everything smaller, which leaves the
        # single call below untouched.
        # Resolved once and memoized rather than imported per call, since this
        # is a read on every forward and every backward recomputation.
        slicer = _get_dequantize_4bit_in_slices()
        weight = slicer(param) if slicer is not None else None
        if weight is None:
            weight = bnb.functional.dequantize_4bit(param.data, param.quant_state)
        original_shape = getattr(param, "_original_shape", None)
        if original_shape is not None and weight.shape != original_shape:
            weight = weight.reshape(original_shape)
        if target_dtype is not None:
            weight = weight.to(target_dtype)
        return weight

    if hasattr(param, "get_param"):
        return param.get_param()

    if hasattr(param, "weight"):
        return param.weight

    return param


def _get_lora_wrapper_for_param(experts_module, param_name):
    """Get the PEFT ParamWrapper for gate_up_proj or down_proj; does not lazily set up wrappers.

    A forward that asks for the wrapper applies that parameter's LoRA itself rather than
    reading the stash, so finding one counts as a read (see `mark_moe_lora_stash_read`).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    wrapper = None
    if hasattr(experts_module, f"{param_name}_lora_wrapper"):
        wrapper = getattr(experts_module, f"{param_name}_lora_wrapper")
    elif hasattr(experts_module, param_name):
        attr = getattr(experts_module, param_name)
        if hasattr(attr, "lora_A"):  # ParamWrapper
            wrapper = attr

    if wrapper is not None:
        mark_moe_lora_stash_read(experts_module, param_name)
    return wrapper


def native_moe_grouped_mm(
    inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Grouped_mm with backward fix for PyTorch's grouped_mm backward stride bug."""
    return _grouped_mm_with_backward_fix(inputs, weight, offsets)


def _apply_lora_grouped_mm(
    inputs: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    offsets: torch.Tensor,
    scaling: float,
    grouped_mm_func=native_moe_grouped_mm,
) -> torch.Tensor:
    """Apply LoRA via grouped GEMM: result = ((X @ B) @ A) * scaling.

    inputs (total_tokens, in_dim); lora_B (E, in_dim, R); lora_A (E, R, out_dim).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # X @ B then result @ A; both already in native (E, ...) layout, no transpose.
    lora_intermediate = grouped_mm_func(inputs, lora_B.contiguous(), offsets)
    lora_delta = grouped_mm_func(lora_intermediate, lora_A.contiguous(), offsets)

    return lora_delta * scaling


def _should_use_separated_lora() -> bool:
    """Use separated LoRA (default True); UNSLOTH_MOE_LORA_MERGED=1 forces the merged path."""
    return os.environ.get("UNSLOTH_MOE_LORA_MERGED", "0") != "1"


# Model-specific weight preprocessing hooks: each model registers a transposition
# function so the generic backend works across weight layouts.
#
# unsloth_compiled_cache holds copies of this file, loaded as their own module objects
# (unsloth_cached_moe_utils, whose forward get_forward_moe_backend() prefers, and the
# bare `moe_utils` compiled modules import), each with its own empty dict. Resolving
# through the package keeps registration and lookup on one dict whichever copy runs,
# so a registration is not dropped and left to layout inference (#849).

_WEIGHT_PREPROCESSORS = {}


def _weight_preprocessor_registry():
    """The registry shared by every loaded copy of this module (see above)."""
    package_module = sys.modules.get("unsloth_zoo.temporary_patches.moe_utils")
    if package_module is None:
        return _WEIGHT_PREPROCESSORS  # package copy not loaded, so nothing registered
    return getattr(package_module, "_WEIGHT_PREPROCESSORS", _WEIGHT_PREPROCESSORS)


def register_weight_preprocessor(model_type: str, preprocessor_fn):
    """Register a weight preprocessor (weight, proj_type, hidden_dim) -> weight for a model type."""
    _weight_preprocessor_registry()[model_type] = preprocessor_fn


def get_weight_preprocessor(model_type: str):
    """Get registered weight preprocessor for model type."""
    return _weight_preprocessor_registry().get(model_type)


def _logical_expert_shape(param):
    """Return the logical expert shape without dequantizing or materializing.

    Resolve PEFT parameters before module wrappers. A recorded logical shape (bnb
    Params4bit _original_shape, ParameterModule shape_3d) is read directly so a
    get_param provider is never materialized just to read its shape.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    seen = set()
    while param is not None and id(param) not in seen:
        seen.add(id(param))

        # Recorded logical shape avoids materializing a get_param provider.
        recorded = getattr(param, "_original_shape", None)
        if recorded is None:
            recorded = getattr(param, "shape_3d", None)
        if recorded is not None:
            return tuple(int(x) for x in recorded)

        get_param = getattr(param, "get_param", None)
        if callable(get_param):
            try:
                inner = get_param()
            except Exception:
                inner = None
            if inner is not None and inner is not param:
                param = inner
                continue

        base_layer = getattr(param, "base_layer", None)
        if base_layer is not None and base_layer is not param:
            param = base_layer
            continue

        # Linear modules store parameters on .weight.
        weight = getattr(param, "weight", None)
        if weight is not None and weight is not param:
            param = weight
            continue

        break

    shape = getattr(param, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except TypeError:
        return None


def _orientation_needs_transpose(shape, proj_type, hidden_dim):
    """Return whether a weight needs transposition, or None when ambiguous."""
    # This Unsloth Zoo code section is licensed under AGPL3

    if shape is None or len(shape) < 3:
        return None
    d1, d2 = int(shape[1]), int(shape[2])
    if d1 == d2:
        return None
    if proj_type == "gate_up":
        # grouped_mm uses (E, hidden, 2I).
        return d1 != hidden_dim
    # grouped_mm down uses (E, I, hidden).
    return d2 != hidden_dim


_WARNED_AMBIGUOUS_LAYOUTS = set()


def _warn_ambiguous_layout_once(proj_type, shape, hidden_dim):
    # This Unsloth Zoo code section is licensed under AGPL3

    key = (proj_type, tuple(shape), hidden_dim)
    if key in _WARNED_AMBIGUOUS_LAYOUTS:
        return
    _WARNED_AMBIGUOUS_LAYOUTS.add(key)
    # Always warn because a wrong guess corrupts training (#849).
    print(
        f"Unsloth: MoE '{proj_type}' expert weight of shape {tuple(shape)} has equal matmul "
        f"dims (hidden_dim={hidden_dim}), so its layout is ambiguous, and no unambiguous "
        f"sibling projection was reachable to disambiguate it. Assuming the transformers "
        f"F.linear layout and transposing to grouped_mm layout. If these experts were "
        f"already in grouped_mm layout this transpose is WRONG and trains on transposed "
        f"weights (see unslothai/unsloth-zoo#849). Register an explicit preprocessor via "
        f"register_weight_preprocessor('<model_type>', ...) to remove the guess.",
        file=sys.stderr,
    )


def preprocess_weight(
    weight: torch.Tensor, proj_type: str, hidden_dim: int, model_type=None,
    experts_module=None,
):
    """Convert an expert weight to grouped_mm layout.

    Registered preprocessors take precedence. Square weights use the non-square sibling.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # Guarded: this runs per projection per MoE forward and again in recompute
    # backward, and the shipped models pass model_type=None.
    if model_type:
        registry = _weight_preprocessor_registry()
        if model_type in registry:
            return registry[model_type](weight, proj_type, hidden_dim)

    # Non-square shapes reveal layout directly.
    needs_transpose = _orientation_needs_transpose(tuple(weight.shape), proj_type, hidden_dim)
    if needs_transpose is not None:
        return weight.transpose(-2, -1) if needs_transpose else weight

    # One sibling is always non-square and reveals the shared layout.
    if experts_module is not None:
        sibling_name = "down_proj" if proj_type == "gate_up" else "gate_up_proj"
        sibling_type = "down" if proj_type == "gate_up" else "gate_up"
        sibling = getattr(experts_module, sibling_name, None)
        if sibling is not None:
            sibling_transpose = _orientation_needs_transpose(
                _logical_expert_shape(sibling), sibling_type, hidden_dim,
            )
            if sibling_transpose is not None:
                return weight.transpose(-2, -1) if sibling_transpose else weight

    # Default to F.linear when no sibling is readable.
    _warn_ambiguous_layout_once(proj_type, weight.shape, hidden_dim)
    return weight.transpose(-2, -1)


# Generic MoE detection and ParamWrapper patching.


def _normalize_model_type(value) -> str:
    if value is None:
        return ""
    return str(value).lower().replace("-", "_")


def _iter_model_configs(model):
    seen = set()
    queue = [model]
    while queue and len(seen) < 8:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        config = getattr(current, "config", None)
        if config is not None:
            yield config

        for attr in ("base_model", "model"):
            nested = getattr(current, attr, None)
            if nested is not None and nested is not current:
                queue.append(nested)


def _is_gpt_oss_model(model) -> bool:
    for config in _iter_model_configs(model):
        model_type = _normalize_model_type(getattr(config, "model_type", None))
        if model_type == "gpt_oss":
            return True

        for attr in ("_name_or_path", "name_or_path"):
            name = getattr(config, attr, None)
            if name is None:
                continue
            # Match only the final path component so parent directories like
            # /data/gpt-oss-tests/qwen3-7b do not count as gpt-oss.
            base = str(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if "gpt_oss" in _normalize_model_type(base):
                return True

    return False


def _set_gpt_oss_grouped_mm_format_on_experts(module) -> bool:
    if module is None:
        return False
    if module.__class__.__name__ != "GptOssExperts":
        return False
    if bool(getattr(module, "_unsloth_grouped_mm_format", False)):
        return False
    # Require the gpt-oss (E, in, out) weight signature: gate_up's out dim is
    # twice down's in dim. Same-named classes with other layouts stay unflagged.
    gate_shape = _get_param_shape_from_module(module, "gate_up_proj")
    down_shape = _get_param_shape_from_module(module, "down_proj")
    if gate_shape is None or down_shape is None:
        return False
    if len(gate_shape) < 3 or len(down_shape) < 3:
        return False
    if gate_shape[0] != down_shape[0]:
        return False
    if gate_shape[-2] != down_shape[-1] or gate_shape[-1] != 2 * down_shape[-2]:
        return False
    module._unsloth_grouped_mm_format = True
    return True


def patch_gpt_oss_grouped_mm_format(model) -> int:
    """
    Mark GPT-OSS experts as storing weights in grouped_mm format.

    Stock transformers GPT-OSS experts use (E, in_dim, out_dim) tensors but do
    not carry Unsloth's `_unsloth_grouped_mm_format` instance flag. Set it on
    live expert modules so the shared MoE LoRA extractor chooses GPT-OSS
    ordering instead of the Qwen-style fallback.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    if model is None or not _is_gpt_oss_model(model):
        return 0

    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0

    updated = 0
    for module in modules():
        if _set_gpt_oss_grouped_mm_format_on_experts(module):
            updated += 1
    return updated


def _patch_peft_get_peft_model_for_moe():
    # This Unsloth Zoo code section is licensed under AGPL3

    global _original_peft_get_peft_model
    if _original_peft_get_peft_model is not None:
        return

    try:
        import peft
    except Exception:
        return

    original_get_peft_model = getattr(peft, "get_peft_model", None)
    if original_get_peft_model is None:
        return
    if getattr(original_get_peft_model, "_unsloth_moe_patched", False):
        return

    _original_peft_get_peft_model = original_get_peft_model

    @wraps(original_get_peft_model)
    def patched_get_peft_model(model, *args, **kwargs):
        peft_model = original_get_peft_model(model, *args, **kwargs)
        try:
            patch_gpt_oss_grouped_mm_format(model)
            if peft_model is not model:
                patch_gpt_oss_grouped_mm_format(peft_model)
        except Exception:
            pass
        return peft_model

    patched_get_peft_model._unsloth_moe_patched = True
    peft.get_peft_model = patched_get_peft_model

    for module_name in ("peft.mapping_func", "peft.mapping"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if getattr(module, "get_peft_model", None) is original_get_peft_model:
            module.get_peft_model = patched_get_peft_model


def _is_moe_experts_module(module) -> bool:
    """Generic check for an MoE experts layer with stacked 3D expert weights.

    Matches gate_up_proj/down_proj (Qwen3-MoE etc.) or w1/w2/w3 (older models).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    import torch.nn as nn

    # After PEFT's parametrize wrapping, gate_up_proj is a Tensor (not Parameter),
    # so accept both.
    if hasattr(module, "gate_up_proj"):
        param = module.gate_up_proj
        # 4-bit params are packed into 2D tensors.
        if HAS_BNB and isinstance(param, Params4bit) and param.ndim == 2:
            return True
        # Standard MoE weights are 3D (num_experts, in, out).
        if isinstance(param, (nn.Parameter, torch.Tensor)) and param.ndim in (2, 3):
            return True

    # w1/w2 pattern (separate gate/up projections).
    if hasattr(module, "w1") and hasattr(module, "w2"):
        w1 = module.w1
        if isinstance(w1, (nn.Parameter, torch.Tensor)) and w1.ndim in (2, 3):
            return True

    return False


# Aliases for compatibility with gpt_oss.py
_get_moe_lora_weights = _extract_lora_from_wrapper


# ---------------------------------------------------------------------------------------
# Which convention packs lora_B's (num_experts * rank) axis.
#
# PEFT's ParamWrapper builds lora_A as (E*R, in) and lora_B as (out, E*R) and pairs them
# in get_delta_weight by reading A as (E, R, in) and B as (out, R, E): lora_A is grouped
# by expert, lora_B is NOT, its expert index is the fastest axis. That reading is
# unchanged in PEFT 0.18.0, 0.19.0, 0.19.1, 0.20.0 and 0.21.0, and it is what vLLM's
# `_stack_moe_lora_weights` and every other consumer of a saved adapter implements, so it
# is the only reading a stored fused expert lora_B can be expected to have.
#
# Unsloth's separated forward read the same lora_B as (out, E, R) instead
# (`_canonical_lora_weights_for_grouped_mm`), so a rank-contiguous block of columns
# belonged to one expert. Both reshapes always fit, since E*R == R*E, so nothing ever
# raises and the disagreement is always silent. They disagree whenever num_experts > 1
# and rank > 1, which is every real MoE checkpoint: equal num_experts and rank does not
# make them agree, because the two readings send flat column r*E + e and flat column
# e*R + r to the same place and those coincide for all (e, r) only when E == 1 or R == 1.
#
# The separated forward is being moved onto PEFT's packing (the `lora_B` packing fix,
# unsloth_zoo#1269), with `UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert` left as the
# opt-in for reading an adapter an older Unsloth trained. What stays here is the
# *recording* side: which packing the adapter on this model was trained with, written
# into adapter_config.json on save (unsloth#6930), because a downstream converter has no
# wrapper to ask and Unsloth stamps no version into the file either, so a marker-less
# adapter cannot be dated from its bytes.
# ---------------------------------------------------------------------------------------

# Also defined next to the packing helpers by unsloth_zoo#1269; same strings, so the two
# definitions are interchangeable and either branch can land first.
LORA_B_LAYOUT_GROUPED_BY_EXPERT = "grouped_by_expert"
LORA_B_LAYOUT_RANK_MAJOR = "rank_major"


def _process_lora_b_layout() -> str:
    """Which packing the separated MoE forward uses in THIS process.

    `moe_lora_b_layout()` (unsloth_zoo#1269) is the authority once that fix is in: it
    reads `UNSLOTH_MOE_LORA_B_LAYOUT` and defaults to PEFT's `rank_major`. It is looked
    up rather than imported because it lives in this same module, so the two can land in
    either order. Without it the separated forward is unconditionally grouped by expert,
    which is what a build predating the packing fix does, so that is the honest answer
    there."""
    resolver = globals().get("moe_lora_b_layout")
    if callable(resolver):
        return resolver()
    return LORA_B_LAYOUT_GROUPED_BY_EXPERT


def _legacy_lora_b_layout_requested() -> bool:
    """Whether the caller has explicitly declared that the adapters in this process were
    packed the pre-fix way, by setting `UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert`.

    Deliberately the raw environment variable and not `_process_lora_b_layout()`: this
    gates a patch over PEFT's own `get_delta_weight`, and the packing of a *stored*
    adapter is a property of the adapter, not of how this process happens to be routing
    its forwards. Only an explicit statement about provenance may redirect PEFT's merge;
    an unset variable leaves PEFT's reconstruction, which is the correct one for every
    adapter trained with the packing fix in place, completely untouched."""
    return os.environ.get("UNSLOTH_MOE_LORA_B_LAYOUT") == LORA_B_LAYOUT_GROUPED_BY_EXPERT

# The only parameter names the separated forward claims. Anything else on an experts
# module (the unfused experts.gate_proj / experts.up_proj pair that NemotronH uses, for
# instance) stays on PEFT's own forward and keeps PEFT's own packing.
_SEPARATED_MOE_LORA_PARAMETER_NAMES = ("gate_up_proj", "down_proj")


def _wrapper_uses_separated_moe_lora(wrapper, experts_module = None) -> bool:
    """True when `_patched_param_wrapper_forward` routes this wrapper to the separated
    MoE forward instead of PEFT's `_activate_lora`. Kept in one place so the forward, the
    delta reconstruction and the saved layout marker cannot drift apart."""
    if not _should_use_separated_lora():
        return False
    if getattr(wrapper, "parameter_name", None) not in _SEPARATED_MOE_LORA_PARAMETER_NAMES:
        return False
    if experts_module is None:
        get_base_layer = getattr(wrapper, "get_base_layer", None)
        if get_base_layer is None:
            return False
        try:
            experts_module = get_base_layer()
        except Exception:
            return False
    return _is_moe_experts_module(experts_module)


def _wrapper_forward_applies_stash(wrapper):
    """The measured verdict for this wrapper's experts forward, or None if unmeasured.

    `_wrapper_uses_separated_moe_lora` answers a structural question, "does the separated
    forward claim this parameter". `_patched_param_wrapper_forward` then asks a stronger
    one at run time, "did the forward that actually ran read the stash", and falls back to
    PEFT when the answer is False. The layout follows the forward that ran, so it has to
    read the same verdict."""
    # This Unsloth Zoo code section is licensed under AGPL3

    parameter_name = getattr(wrapper, "parameter_name", None)
    if parameter_name is None:
        return None
    get_base_layer = getattr(wrapper, "get_base_layer", None)
    if get_base_layer is None:
        return None
    try:
        experts_module = get_base_layer()
    except Exception:
        return None
    if experts_module is None:
        return None
    try:
        return moe_lora_forward_applies_stash(experts_module, parameter_name)
    except Exception:
        return None


def _wrapper_has_adapter(wrapper, adapter_name) -> bool:
    """Whether `adapter_name` has a LoRA on this wrapper at all. A PeftModel can carry a
    fused expert adapter and a dense adapter side by side, and the fused expert wrappers
    then exist for the whole model while only one adapter has weights in them, so every
    layout answer has to be scoped to one adapter or it describes the wrong one.

    `adapter_name = None` asks the weaker question "does this wrapper hold any adapter at
    all", which is what the unnamed form of `moe_lora_b_layout_for_wrapper` needs. Still
    a question:
    `delete_adapter` empties lora_A and leaves the wrapper in place, and claiming a
    packing for weights that are gone is the same mistake in a smaller form."""
    lora_A = getattr(wrapper, "lora_A", None)
    if lora_A is None:
        return False
    try:
        if adapter_name is None:
            return len(lora_A) != 0
        return adapter_name in lora_A
    except Exception:
        return False


def moe_lora_b_layout_for_wrapper(wrapper, adapter_name = None) -> str:
    """Which convention packs `wrapper`'s lora_B columns for `adapter_name`.

    A wrapper the separated MoE forward owns is packed the way that forward packs, which
    is `_process_lora_b_layout()`: PEFT's `rank_major` with the packing fix in place,
    `grouped_by_expert` under `UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert` or on a build
    predating the fix. Every other wrapper is on PEFT's own forward and is therefore
    `rank_major`. A single expert (or a wrapper PEFT collapses to a plain Linear) has no
    packing to get wrong and is reported as rank_major, which is what PEFT's own
    reconstruction does for it.

    Named for the wrapper on purpose: `moe_lora_b_layout()` is the process-wide setting
    and takes no arguments, this one answers for one wrapper and one adapter.

    `adapter_name` defaults to "whichever adapter this wrapper holds", which is the old
    behaviour and is right for a model with one adapter. Name an adapter that has no LoRA
    on this wrapper and the answer is rank_major, so the caller falls back to PEFT rather
    than claiming a packing for weights that are not there.

    `wrapper` must be the PEFT ParamWrapper. Hand it anything else, the experts module
    itself for instance, and the answer is rank_major, because nothing on it says the
    separated forward claimed a parameter. That reads as a confident "PEFT packed this"
    when the truth is "cannot tell", so callers that may hold something else have to
    check for `parameter_name` and `lora_A` first."""
    if int(getattr(wrapper, "num_experts", 1) or 1) <= 1:
        return LORA_B_LAYOUT_RANK_MAJOR
    if not _wrapper_has_adapter(wrapper, adapter_name):
        return LORA_B_LAYOUT_RANK_MAJOR
    if _wrapper_uses_separated_moe_lora(wrapper):
        if _wrapper_forward_applies_stash(wrapper) is False:
            # The structural test says the separated forward claims this wrapper, but the
            # forward that actually ran did not read the stash, so `_patched_param_wrapper_forward`
            # handed the wrapper back to PEFT and PEFT trained it rank-major. That happens
            # for a stacked-expert family Unsloth does not patch (Olmoe, for one). Believing
            # the structural answer here would write a grouped_by_expert marker for
            # rank-major weights, and, under the legacy override, merge them with the wrong
            # pairing. Only a measured False overrides it; None means "not measured yet",
            # which is not evidence either way.
            return LORA_B_LAYOUT_RANK_MAJOR
        return _process_lora_b_layout()
    return LORA_B_LAYOUT_RANK_MAJOR


# Did the experts forward actually read the LoRA that ParamWrapper.forward handed it?
#
# `_patched_param_wrapper_forward` deliberately does not let PEFT fold the expert LoRA into
# the stacked expert weight. It stashes the factors on the experts module as
# `_unsloth_lora_<parameter_name>` and lets the experts forward apply them as a separate
# grouped GEMM, which is cheaper and keeps the base weight quantized. That is only correct
# when the forward that runs is one that reads the stash. Unsloth installs such a forward for
# the MoE families it patches. For a stacked-expert family it does not patch, transformers'
# own experts forward runs, the stash is written and then deleted unread, and the expert LoRA
# has no effect on the output at all while requires_grad and the optimizer still report a
# healthy adapter. `_forward_native_fp8_expert_loop` already refuses rather than train an
# adapter that nothing reads; this does the same job without having to enumerate the
# families, by recording the read and asking afterwards.
#
# Every stash read goes through `take_moe_lora_stash`, which records the read on the experts
# module. The wrapper resets the record, calls the base layer, and checks it: an unread stash
# means this forward does not apply the expert LoRA, so the wrapper redoes the call through
# PEFT's own `ParamWrapper.forward`, which folds the delta into the weight. The verdict is
# cached per parameter name against the forward it was measured with, so the double call
# happens at most once per experts module and never for a family whose forward does read it.

_MOE_LORA_STASH_READ_ATTR = "_unsloth_moe_lora_stash_read"
_MOE_LORA_STASH_VERDICT_ATTR = "_unsloth_moe_lora_forward_applies"


def moe_lora_stash_name(parameter_name: str) -> str:
    """Attribute name `_patched_param_wrapper_forward` stashes `parameter_name`'s LoRA under."""
    # This Unsloth Zoo code section is licensed under AGPL3

    return f"_unsloth_lora_{parameter_name}"


def _moe_module_dict(module, attr):
    """Per-module bookkeeping dict for `attr`, created on first use.

    Read through `__dict__` and written with `setattr`: these are plain dicts, so
    `nn.Module.__setattr__` stores them in the instance `__dict__`, and going straight to
    `__dict__` on the read path skips `nn.Module.__getattr__` for a value that is never a
    parameter, buffer or submodule.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    try:
        store = module.__dict__.get(attr)
    except AttributeError:
        return None
    if store is None:
        store = {}
        try:
            setattr(module, attr, store)
        except Exception:
            return None
    return store


def mark_moe_lora_stash_read(experts_module, parameter_name: str) -> None:
    """Record that this experts forward applies `parameter_name`'s expert LoRA itself.

    Called by `take_moe_lora_stash` for every stash read, and by
    `_get_lora_wrapper_for_param` for the forwards that reach past the stash and pull the
    LoRA straight off the PEFT wrapper (the MXFP4 GPT-OSS experts forward does that).
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    store = _moe_module_dict(experts_module, _MOE_LORA_STASH_READ_ATTR)
    if store is not None:
        store[parameter_name] = True


def take_moe_lora_stash(experts_module, parameter_name: str):
    """The stashed LoRA for `parameter_name`, or None, recording that this forward read it.

    Records the read whatever the value is: an attempted read is what proves the forward
    knows about the stash, and the stash is legitimately absent when no adapter is attached.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    mark_moe_lora_stash_read(experts_module, parameter_name)
    return getattr(experts_module, moe_lora_stash_name(parameter_name), None)


def _resolve_experts_forward(experts_module):
    """The function object that `experts_module(...)` will run, or None."""
    # This Unsloth Zoo code section is licensed under AGPL3

    forward = getattr(experts_module, "forward", None)
    return getattr(forward, "__func__", forward)


def moe_lora_forward_applies_stash(experts_module, parameter_name: str):
    """Cached verdict: does this experts forward apply the stashed expert LoRA itself?

    True or False once measured, None when it has not been measured for the forward that is
    currently installed. Re-patching the forward invalidates the verdict.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    try:
        cache = experts_module.__dict__.get(_MOE_LORA_STASH_VERDICT_ATTR)
    except AttributeError:
        return None
    if not cache:
        return None
    entry = cache.get(parameter_name)
    if entry is None:
        return None
    forward, verdict = entry
    if forward is not _resolve_experts_forward(experts_module):
        return None
    return verdict


def _record_moe_lora_forward_verdict(experts_module, parameter_name: str, verdict) -> None:
    """Remember `verdict` for `parameter_name` against the forward currently installed."""
    # This Unsloth Zoo code section is licensed under AGPL3

    store = _moe_module_dict(experts_module, _MOE_LORA_STASH_VERDICT_ATTR)
    if store is not None:
        store[parameter_name] = (_resolve_experts_forward(experts_module), bool(verdict))


def _reset_moe_lora_stash_read(experts_module, parameter_name: str) -> None:
    """Clear the read record for `parameter_name` before calling the experts forward."""
    # This Unsloth Zoo code section is licensed under AGPL3

    store = _moe_module_dict(experts_module, _MOE_LORA_STASH_READ_ATTR)
    if store is not None:
        store[parameter_name] = False


def _moe_lora_stash_was_read(experts_module, parameter_name: str) -> bool:
    """Whether the experts forward read `parameter_name`'s stash since the last reset."""
    # This Unsloth Zoo code section is licensed under AGPL3

    try:
        store = experts_module.__dict__.get(_MOE_LORA_STASH_READ_ATTR)
    except AttributeError:
        return False
    return bool(store and store.get(parameter_name))


_MOE_LORA_STASH_UNREAD_LOGGED = set()


def _log_moe_lora_stash_unread_once(experts_module, parameter_name: str) -> None:
    """One message per experts class and parameter, the first time the stash goes unread."""
    # This Unsloth Zoo code section is licensed under AGPL3

    key = (type(experts_module).__name__, parameter_name)
    if key in _MOE_LORA_STASH_UNREAD_LOGGED:
        return
    _MOE_LORA_STASH_UNREAD_LOGGED.add(key)
    _log_info(
        f"Unsloth: {key[0]}.forward does not apply Unsloth's separated expert LoRA for "
        f"{parameter_name}, so the {parameter_name} adapter is applied through PEFT's "
        "parameter path instead."
    )


_STASH_READ_MARKERS = frozenset(("take_moe_lora_stash", "moe_lora_stash_name"))

# How far the scan follows a call before giving up. The installed forwards delegate at most
# twice (dispatcher -> backend -> helper), and a bound keeps a pathological import graph
# from turning a cold start into a walk of the whole module tree.
_STASH_SCAN_MAX_DEPTH = 4


def _forward_statically_reads_stash(experts_module):
    """Does this experts forward reach the stash API at all, read from its bytecode?

    A static answer, which is the only kind available while Dynamo is tracing: the probe
    cannot run inside a captured graph without being captured with it. Every forward that
    applies Unsloth's separated expert LoRA reaches `take_moe_lora_stash`, so that name
    appearing anywhere the forward can call is what distinguishes an Unsloth-installed
    forward from transformers' own. Cheap and side-effect free: no call is made, only
    names are read.

    The reach matters. The forward installed for the Qwen MoE families IS
    `forward_moe_backend`, a dispatcher that names only `forward_native_grouped_mm`,
    `forward_triton_grouped_gemm` and `forward_native_moe_loop`, each of which reads the
    stash. Scanning the dispatcher's own code object alone calls those supported families
    stash-ignorant, which routes the first compiled call through PEFT's dynamic
    parametrization and brings back the `fullgraph=True` failure this patch exists to
    avoid. So a global name that resolves to a Python function is followed too, bounded by
    `_STASH_SCAN_MAX_DEPTH` and by the visited set.

    Only module globals are resolvable: a name imported inside the function body, or
    reached through an instance attribute, is not bound anywhere this can read. That makes
    the answer one-sided, True is certain and False is only "no route found", which is why
    the caller treats False as a reason to log rather than a guarantee.

    Returns True, False, or None when there is no code object to read.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    forward = getattr(experts_module, "forward", None)
    forward = getattr(forward, "__func__", forward)
    code = getattr(forward, "__code__", None)
    if code is None:
        return None

    # Keyed on the code objects themselves, never on id(). Tracing id(forward) makes
    # Dynamo guard the expression it tracked, `experts_module.forward`, a bound method
    # CPython reallocates on every access, so ___check_obj_id can never match again and
    # Dynamo raises "Guard failed on the same frame it was created" under both fullgraph
    # settings with no eager fallback. Code objects are hashable, stable and 1:1 with the
    # functions here, so they also make the separate function set redundant.
    # Keyed on the code objects themselves, never on id(), and remembering the SHALLOWEST
    # depth each was reached at. A plain visited set makes the answer depend on the hash
    # seed: a helper reachable both directly and through a chain can be popped first at
    # the depth limit, where its own callees are not followed, and the later shallow entry
    # is then dropped as already seen. Measured on a synthetic forward with both routes,
    # 5 of 14 PYTHONHASHSEED values returned False for a forward that does reach the
    # stash, which would send that compiled cold start down the failing PEFT path.
    # A LIST keyed by `is`, not a dict and not id().
    #
    # Equality is wrong: two functions compiled from identical source at the same filename
    # and name have code objects that compare and hash equal while being distinct objects
    # with different __globals__, so a dict collapses them and the scan misses whichever
    # route is second.
    #
    # id() is right but untraceable: Dynamo rejects it on a code object with
    # "Unsupported: id() with unsupported args" on some torch versions, which is a hard
    # compile failure in the branch that exists to keep compilation working. `is` gives
    # the same identity semantics and traces everywhere. The list stays tiny, bounded by
    # the depth limit, so the linear scan costs nothing.
    seen_code = []
    pending = [(code, getattr(forward, "__globals__", {}), 0)]

    def _visited_at_or_above(target, depth):
        for seen, seen_depth in seen_code:
            if seen is target:
                return seen_depth <= depth
        return False

    while pending:
        current, namespace, depth = pending.pop()
        # Keyed by IDENTITY, not equality. Two functions compiled from identical source at
        # the same filename and name have code objects that compare equal and hash equal
        # while being distinct objects with different __globals__, so a dict keyed on the
        # code objects themselves collapses them: if the first resolves its names to an
        # unrelated helper and the second to take_moe_lora_stash, the second is skipped
        # and the scan wrongly answers False. `alive` holds a reference to everything
        # visited, so no id() can be recycled by the collector mid-walk.
        if _visited_at_or_above(current, depth):
            continue
        seen_code.append((current, depth))
        names = set(getattr(current, "co_names", ()))
        if names & _STASH_READ_MARKERS:
            return True
        for constant in getattr(current, "co_consts", ()):
            if hasattr(constant, "co_names"):
                pending.append((constant, namespace, depth))
        if depth >= _STASH_SCAN_MAX_DEPTH:
            continue
        for name in names:
            called = namespace.get(name)
            called = getattr(called, "__func__", called)
            called_code = getattr(called, "__code__", None)
            if called_code is None:
                continue
            if _visited_at_or_above(called_code, depth + 1):
                continue
            pending.append((called_code, getattr(called, "__globals__", namespace), depth + 1))
    return False


@contextlib.contextmanager
def _preserved_rng_for_probe(x):
    """Run the probe forward without advancing any generator the real forward will read.

    `no_grad` only turns off the graph; every random draw inside the throwaway forward
    still advances the generator, so a family with dropout in the experts path would get
    different numbers out of the call that counts than it got before this probe existed.
    Gradient checkpointing makes that a wrong-gradient bug rather than a reproducibility
    one: non-reentrant checkpointing restores the RNG state at the start of the region and
    replays it, so the original pass (probe plus real forward) and the recompute (verdict
    cached, real forward only) would draw different masks for the same region.

    Three things this is careful about, all of them ways a preservation attempt could be
    worse than none:

    * The BACKEND is named, not inferred. `fork_rng`'s `devices` identifies devices within
      `device_type`, which it resolves from `torch.accelerator.current_accelerator()` and
      falls back to "cuda" for, so handing it an XPU device without saying so can ask the
      wrong module for a generator state.
    * A failure here must not become the probe's answer. The caller turns any exception
      into `None`, which caches no verdict and leaves the stash path selected, so a
      `fork_rng` that raised would silently keep the LoRA unapplied on a family that
      ignores the stash. Anything that goes wrong setting the fork up leaves the forward
      to run unforked instead.
    Reached only on the eager path: `_measure_moe_lora_stash_read` returns before it while
    Dynamo is tracing, so a generator-based context manager never enters a captured graph.
    The check is kept here as well because this helper is usable on its own.
    """
    if torch.compiler.is_compiling():
        yield
        return
    device = x.device if isinstance(x, torch.Tensor) else None
    fork = None
    try:
        if device is None or device.type in ("cpu", "meta"):
            # The CPU generator is forked whatever `devices` says; an empty list is what
            # keeps `fork_rng` from initialising every visible accelerator, which it does
            # when `devices` is None, and warns about.
            fork = torch.random.fork_rng(devices = [], device_type = "cpu")
        else:
            fork = torch.random.fork_rng(devices = [device], device_type = device.type)
        fork.__enter__()
    except Exception:
        fork = None
    try:
        yield
    finally:
        if fork is not None:
            try:
                fork.__exit__(None, None, None)
            except Exception:
                pass


def _measure_moe_lora_stash_read(wrapper, experts_module, parameter_name, x, args, kwargs) -> bool:
    """Run one throwaway experts forward under `no_grad` and report whether it read the stash.

    Measured on a separate graph-free call rather than on the call that counts. The call
    that counts then runs exactly one experts forward, on whichever path the verdict names,
    and it runs that same single path again on a recompute. That is the property
    non-reentrant gradient checkpointing requires: it replays the region and refuses, with
    `CheckpointError: a different number of tensors was saved during the original forward
    and recomputation`, if the replay does not save what the forward saved. Measuring in
    band cannot have that property, because the measuring call has to run the forward once
    to find out and then a second time through PEFT when the answer is no, while the
    recompute already knows the answer and runs it once. transformers 5 enables gradient
    checkpointing with `use_reentrant = False` by default
    (`modeling_utils.py`: `gradient_checkpointing_kwargs = {"use_reentrant": False}`), and
    Unsloth's own vision path selects it whenever the run is distributed, so that is the
    common configuration and not a corner.

    `no_grad` and the discarded result are what make this cheap enough to do eagerly: no
    graph is built, nothing is saved for backward, and it happens once per experts module
    and parameter for the lifetime of the process. A stash-reading family pays one extra
    forward on its first step and nothing afterwards; a family that ignores the stash now
    pays two forwards on its first step where the in-band measurement paid three.

    Returns True, False, or None when the probe itself raised. None caches nothing and
    leaves the call on the stash path, so the real call raises the real error with the real
    traceback instead of a second, confusing one from PEFT's fold.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    if torch.compiler.is_compiling():
        # Never inside a captured graph. Dynamo has no notion of a one-time measurement: it
        # traces this whole throwaway forward into the graph alongside the call that counts,
        # so the compiled region ends up running the experts forward three times on EVERY
        # invocation, not once during a single warm-up. Measured on PyTorch 2.12 with
        # fullgraph: six expert einsums in one graph where two are correct.
        #
        # None, which is the "no evidence" answer: it caches no verdict and leaves the call
        # on the stash path, which is exactly what this module did before the probe existed.
        # An eager forward before or after compilation still establishes the verdict, and the
        # cached one is consulted before this function is ever called.
        return None
    lora_data = _extract_lora_from_wrapper(wrapper)
    lora_attr = moe_lora_stash_name(parameter_name)
    if lora_data is not None:
        setattr(experts_module, lora_attr, lora_data)
    _reset_moe_lora_stash_read(experts_module, parameter_name)
    try:
        with _preserved_rng_for_probe(x):
            with torch.no_grad():
                wrapper.base_layer(x, *args, **kwargs)
    except Exception:
        # Not evidence either way, and not ours to report. Do not cache a verdict.
        return None
    finally:
        if hasattr(experts_module, lora_attr):
            delattr(experts_module, lora_attr)
    was_read = _moe_lora_stash_was_read(experts_module, parameter_name)
    _record_moe_lora_forward_verdict(experts_module, parameter_name, was_read)
    return was_read


def _moe_lora_folded_weight(self, param, active):
    """`W` with the active adapters folded in, by the cheaper of PEFT's two routes.

    PEFT's own choice, mirrored: with ONE active adapter over many experts it keeps the
    low-rank factors and folds with a single `baddbmm`, and only otherwise materialises a
    `get_delta_weight` the size of the whole expert stack. The difference is real on the
    families this path exists for, OlmoE having 64 experts: the dense route allocates a
    delta as large as the parameter and then a second tensor for the sum, so mirroring the
    factored route halves the peak of every fold. `get_delta_factors` is newer than the
    oldest PEFT with `target_parameters`, so it is asked for rather than assumed.

    The autocast is disabled around the fold for PEFT's reason: a parametrization may not
    change the dtype of the parameter, so the arithmetic stays in W's dtype. Float8 never
    arrives here, since `_can_fold_moe_lora_through_peft` refuses it, so PEFT's low
    precision add has no counterpart.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    factors = getattr(self, "get_delta_factors", None)
    if factors is not None and len(active) == 1 and getattr(self, "num_experts", 1) > 1:
        try:
            lhs, rhs, scaling = factors(active[0])
            with torch.autocast(device_type = param.device.type, enabled = False):
                return torch.baddbmm(param, lhs, rhs, alpha = scaling)
        except Exception:
            pass
    delta = None
    try:
        for name in active:
            contribution = self.get_delta_weight(name)
            delta = contribution if delta is None else delta + contribution
    except Exception:
        return None
    return None if delta is None else param + delta


def _fold_moe_lora_without_parametrization(
    self, immediate_base_layer, experts_module, parameter_name, x, args, kwargs
):
    """PEFT's fold, arithmetically, with the parametrization left out. None if not doable.

    Same folded weight PEFT would produce, by the same route it would choose (see
    `_moe_lora_folded_weight`). The difference is only how the base forward gets to see
    it: PEFT registers a parametrization on the stored parameter,
    whose `set_` Dynamo rejects as a graph-input mutation, while this swaps the attribute
    for `W + delta` for the duration of the call and puts the parameter back after.
    Autograd is unaffected, since the sum is an ordinary op on both tensors.

    Only for the compiled path. Eagerly PEFT's own forward stays in charge, so adapter
    bookkeeping this does not reproduce (variants, merge state, anything future PEFT does
    inside `_activate_lora`) is only ever bypassed where the alternative is not running at
    all. Returns None whenever anything is missing, and the caller then falls through to
    the stash path rather than losing the call.

    Nested wrappers compose: the wrapper chain is down_proj -> gate_up_proj -> experts and
    each level folds and restores its own parameter around the next, so both are folded
    for the innermost forward.

    The swap is done inside `_parameters` and nowhere else. `nn.Module.__getattr__` reads
    the value straight out of that dict, so a plain tensor placed there is what the forward
    sees, and the module's `__setattr__`, which would reject a plain tensor where a
    Parameter is registered, is never involved. The obvious alternative, shadowing the
    name in the instance `__dict__`, is what this used to do and it does not trace:
    `experts_module.__dict__` is an unknown type to Dynamo, so removing the shadow in the
    `finally` raises `Unsupported: Dynamo does not know how to trace method 'pop' of class
    '<unknown type>'` on torch 2.10, the floor this package supports. That turned the
    fullgraph compile this function exists to make possible back into a hard error.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    try:
        active = [name for name in self.active_adapters if name in self.lora_A]
    except Exception:
        return None
    if not active:
        return None

    parameters = getattr(experts_module, "_parameters", None)
    if not isinstance(parameters, dict) or parameter_name not in parameters:
        return None
    if parameter_name in experts_module.__dict__:
        # Something already shadows the registered parameter, so swapping `_parameters`
        # would not reach the forward. Refuse rather than fold into a value nothing reads.
        return None
    original = parameters[parameter_name]
    folded = _moe_lora_folded_weight(self, original, active)
    if folded is None:
        return None

    parameters[parameter_name] = folded
    try:
        return immediate_base_layer(x, *args, **kwargs)
    finally:
        parameters[parameter_name] = original


# PEFT names the float8 storage dtypes it must upcast in `peft.utils.UPCAST_DTYPES`. Track
# that list rather than restating it, so a dtype PEFT adds later is excluded here too, and
# fall back to resolving the names off torch for PEFT versions that do not export it. Every
# one of these raises on `param + delta`, including the fnuz pair ROCm uses.
def _resolve_float8_storage_dtypes():
    names = None
    try:
        from peft.utils import UPCAST_DTYPES as names
    except Exception:
        names = (
            "float8_e4m3fn", "float8_e4m3fnuz",
            "float8_e5m2",   "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
    resolved = []
    for name in names:
        dtype = getattr(torch, name, None)
        # Older torch does not define every float8 variant.
        if isinstance(dtype, torch.dtype):
            resolved.append(dtype)
    return tuple(resolved)


_FLOAT8_STORAGE_DTYPES = _resolve_float8_storage_dtypes()


def _can_fold_moe_lora_through_peft(experts_module, parameter_name: str) -> bool:
    """Whether handing this parameter back to PEFT would fold a delta into a real weight.

    PEFT folds by registering a parametrization that adds `delta_weight` to the stored
    parameter, which is only defined for a plain floating-point tensor. A stacked expert
    weight held as bitsandbytes `Params4bit`, an MXFP4 blocked tensor or an FP8 tensor is
    none of those: the add is a dtype or shape error at best and silent corruption at
    worst. `_forward_native_fp8_expert_loop` says the same thing from the other side, that
    the reroute must not take charge of a quantized parameter, and enforces it by recording
    its reads; this enforces it for the quantized forwards that have no such hook, the
    MXFP4 GPT-OSS experts forward among them, where `_get_lora_wrapper_for_param` resolves
    to None against PEFT's `target_parameters` layout and so records nothing.

    Refusing to reroute leaves that case exactly as it is on main: the stash is written and
    the forward does what it does with it. That is not a fix for a quantized family, and it
    is not meant to be one; it is a guarantee that this change cannot make one worse.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    if _original_param_wrapper_forward is None:
        return False
    param = getattr(experts_module, parameter_name, None)
    if not isinstance(param, torch.Tensor):
        return False
    if not param.dtype.is_floating_point:
        return False
    # `is_floating_point` is True for every float8 storage dtype, and PEFT's own
    # `get_delta_weight` special-cases them precisely because the add is not defined there:
    # `param + delta` raises "Promotion for Float8 Types is not supported" for all five, and
    # naming only the two CUDA ones let the fnuz pair that ROCm uses through.
    if param.dtype in _FLOAT8_STORAGE_DTYPES:
        return False
    # bitsandbytes stores 4-bit weights in a float or uint8 tensor of the packed shape, and
    # marks them with `quant_state`. Neither the dtype nor the shape gives it away.
    if getattr(param, "quant_state", None) is not None:
        return False
    if HAS_BNB and Params4bit is not None and isinstance(param, Params4bit):
        return False
    return True


# Store original ParamWrapper.forward for fallback
_original_param_wrapper_forward = None


def _patched_param_wrapper_forward(
    self, x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """Patched ParamWrapper.forward for MoE separated LoRA.

    For MoE experts: bypass PEFT's _activate_lora and stash LoRA data by
    parameter_name for forward_native_grouped_mm. For non-MoE: original forward.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # Use self.base_layer (immediate parent), NOT get_base_layer() which recurses
    # to the deepest layer; the wrapper chain down_proj -> gate_up_proj ->
    # Qwen3MoeExperts must be preserved.
    immediate_base_layer = self.base_layer

    # For stashing LoRA data we need the actual experts module (recursive lookup).
    experts_module = self.get_base_layer()

    param_name = getattr(self, "parameter_name", None)

    if _wrapper_uses_separated_moe_lora(self, experts_module):
        # MoE experts: bypass PEFT's _activate_lora, use separated computation.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return immediate_base_layer(x, *args, **kwargs)

        if self.merged:
            return immediate_base_layer(x, *args, **kwargs)

        # Ensure wrapper.num_experts is set for LoRA weight reshaping.
        if not hasattr(self, "num_experts"):
            if hasattr(experts_module, "num_experts"):
                self.num_experts = experts_module.num_experts
            elif hasattr(experts_module, param_name):
                p = getattr(experts_module, param_name)
                if hasattr(p, "shape") and len(p.shape) >= 1:
                    self.num_experts = p.shape[0]

        # An experts forward that does not read the stash never sees this LoRA, so hand the
        # wrapper back to PEFT, which folds the delta into the weight instead. Measured once
        # per experts module and parameter, before the call that counts, so the call that
        # counts always runs exactly one experts forward on the path the verdict names.
        applies_stash = moe_lora_forward_applies_stash(experts_module, param_name)
        if applies_stash is None and torch.compiler.is_compiling():
            # First invocation is a compiled one, so there is no verdict and no way to
            # measure one: the probe cannot run inside a captured graph without being
            # captured with it and re-run on every call. Answer statically instead, from
            # whether the forward references the stash API at all.
            #
            # Both wrong answers are costly, which is why this is not a fixed assumption.
            # Assuming unread sends a supported stash-reading family into PEFT's own
            # ParamWrapper.forward, which registers and removes a parametrization while
            # tracing and hard-fails under fullgraph with "Getting an inplace view on a
            # graph input is not supported". Assuming read leaves the expert LoRA out of
            # every output and gradient on a family that ignores the stash, silently and
            # for the life of the captured graph. The bytecode says which family this is.
            #
            # Nothing is recorded either way, so the first eager call still measures and
            # every later compile follows the real verdict.
            applies_stash = _forward_statically_reads_stash(experts_module)
            if applies_stash is False:
                _log_moe_lora_stash_unread_once(experts_module, param_name)
        elif applies_stash is None:
            applies_stash = _measure_moe_lora_stash_read(
                self, experts_module, param_name, x, args, kwargs
            )
            if applies_stash is False:
                _log_moe_lora_stash_unread_once(experts_module, param_name)
        if applies_stash is False and _can_fold_moe_lora_through_peft(experts_module, param_name):
            if torch.compiler.is_compiling():
                # PEFT's own fold cannot be traced. `_activate_lora` registers a
                # parametrization for the call and removes it after, and the `set_` that
                # registration performs on the stored parameter is a graph-input mutation,
                # so Dynamo refuses it: "Getting an inplace view on a graph input is not
                # supported". Under fullgraph that is a hard error, which would mean these
                # families cannot compile at all. Folding the same delta ourselves keeps
                # the arithmetic and drops the parametrization, and traces.
                folded = _fold_moe_lora_without_parametrization(
                    self, immediate_base_layer, experts_module, param_name, x, args, kwargs
                )
                if folded is not None:
                    return folded
            else:
                return _original_param_wrapper_forward(self, x, *args, **kwargs)

        # Extract LoRA for this parameter and stash on the experts module
        # (not base_layer): _unsloth_lora_gate_up_proj / _unsloth_lora_down_proj.
        lora_data = _extract_lora_from_wrapper(self)

        if lora_data is not None and param_name:
            lora_attr = moe_lora_stash_name(param_name)
            setattr(experts_module, lora_attr, lora_data)

        try:
            # Immediate base_layer preserves the wrapper chain.
            result = immediate_base_layer(x, *args, **kwargs)
        finally:
            if param_name:
                lora_attr = moe_lora_stash_name(param_name)
                if hasattr(experts_module, lora_attr):
                    delattr(experts_module, lora_attr)

        return result

    # Non-MoE: original PEFT forward with _activate_lora.
    return _original_param_wrapper_forward(self, x, *args, **kwargs)


# Store original ParamWrapper.get_delta_weight for fallback
_original_param_wrapper_get_delta_weight = None


def _grouped_by_expert_delta_weight(wrapper, adapter_name):
    """PEFT's ParamWrapper.get_delta_weight arithmetic with lora_B read the way the
    separated MoE forward reads it: (out, num_experts, rank) rather than PEFT's
    (out, rank, num_experts). Everything else, including which einsum the swapped
    in/out orientation needs, is PEFT's own."""
    weight_A = wrapper.lora_A[adapter_name].weight
    weight_B = wrapper.lora_B[adapter_name].weight
    num_experts = int(wrapper.num_experts)

    # experts x rank x in_features, as in PEFT: lora_A is grouped by expert there too.
    weight_A = weight_A.reshape(num_experts, -1, weight_A.shape[-1])
    # out_features x experts x rank. This is the one line that differs from PEFT.
    weight_B = weight_B.reshape(weight_B.shape[0], num_experts, -1)

    scaling = wrapper.scaling[adapter_name]
    if not getattr(wrapper, "_did_swap_in_out_features", False):
        return torch.einsum("o e r, e r i -> e i o", weight_B, weight_A) * scaling
    # for some MoE layers, the order is (experts, out_features, in_features)
    return torch.einsum("o e r, e r i -> e o i", weight_B, weight_A) * scaling


def _cast_delta_weight_like_param(delta_weight, param):
    """PEFT's own tail: move the delta to the parameter, and match its dtype unless that
    dtype is a low precision one PEFT deliberately adds in higher precision (float8 and
    friends). Probed from PEFT when it publishes the set, so this follows PEFT rather
    than hardcoding a dtype list that a new release would make wrong."""
    try:
        from peft.tuners.lora.layer import ALLOWED_COMPUTE_DTYPES
        allowed = param.dtype in ALLOWED_COMPUTE_DTYPES
    except Exception:
        # PEFT 0.18 has no such set and casts unconditionally, 0.19.0 onwards do; every dtype
        # 0.18 accepts for a 3D expert parameter is an ordinary floating point one.
        allowed = param.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if allowed:
        return delta_weight.to(param.device, param.dtype)
    return delta_weight.to(param.device)


def _patched_param_wrapper_get_delta_weight(self, adapter_name, *args, **kwargs):
    """PEFT's `get_delta_weight` for a legacy, grouped-by-expert fused MoE expert LoRA.

    PEFT reconstructs a fused expert delta with lora_B packed rank-major, which is what
    it trained and what every consumer reads, so by default this does nothing at all and
    PEFT's own code runs. It only takes over when the caller has explicitly declared that
    the adapters in this process were packed the pre-fix way, with
    `UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert`; that same variable is what makes the
    separated forward read them that way (unsloth_zoo#1269), and without this patch the
    forward and PEFT's `merge_and_unload` / `merge_adapter` / `unmerge` would disagree for
    exactly the checkpoints that switch exists to support (unsloth#6930).

    Gating on the declaration rather than on "is this wrapper routed to the separated
    forward" is the point: the packing of a stored adapter is a property of the adapter,
    and reading it off this process's routing would scramble a correctly packed adapter
    merged inside an Unsloth process."""
    # This Unsloth Zoo code section is licensed under AGPL3

    if not _legacy_lora_b_layout_requested():
        return _original_param_wrapper_get_delta_weight(self, adapter_name, *args, **kwargs)
    if moe_lora_b_layout_for_wrapper(self, adapter_name) != LORA_B_LAYOUT_GROUPED_BY_EXPERT:
        return _original_param_wrapper_get_delta_weight(self, adapter_name, *args, **kwargs)

    delta_weight = _grouped_by_expert_delta_weight(self, adapter_name)
    return _cast_delta_weight_like_param(delta_weight, self.get_param())


def _moe_utils_logger():
    """The shared patch logger, imported lazily and absolutely: this file is also copied
    into unsloth_compiled_cache and executed there as a top level module, where a
    relative import has no package to resolve against."""
    try:
        from unsloth_zoo.temporary_patches.common import logger
        return logger
    except Exception:
        return None


# The adapter_config.json keys. The flat one is what a converter branches on
# (unsloth#6930 asks for exactly this key); the nested one carries the detail, because a
# checkpoint can in principle hold one fused parameter trained through the separated
# forward and another PEFT kept for itself.
FUSED_EXPERT_LORA_LAYOUT_KEY = "lora_B_layout"
FUSED_EXPERT_LORA_DETAIL_KEY = "unsloth_fused_expert_lora"


def _default_adapter_name(peft_model) -> str:
    """The adapter `fused_expert_lora_layout` describes when the caller names none: the
    model's active one, or PEFT's own default name. Both attributes are properties on
    some PEFT versions and can raise on a partly built model, so neither is trusted."""
    # delete_adapter leaves active_adapter a list, and active_adapters is always one.
    for attribute in ("active_adapter", "active_adapters"):
        try:
            active = getattr(peft_model, attribute, None)
        except Exception:
            continue
        if isinstance(active, str) and active:
            return active
        if isinstance(active, (list, tuple)) and len(active) != 0:
            if isinstance(active[0], str) and active[0]:
                return active[0]
    return "default"


def fused_expert_lora_layout(peft_model, adapter_name = None) -> Optional[dict]:
    """How ONE adapter's fused MoE expert LoRA on `peft_model` packs its lora_A and
    lora_B, or None when that adapter has no fused expert LoRA at all.

    lora_A is grouped by expert in both stacks. lora_B is the one that differs, and it is
    reported per PEFT parameter name, since that is what decides whether Unsloth's
    separated forward or PEFT's own forward owns the wrapper.

    The answer is per adapter because a PeftModel can hold several. A fused expert
    adapter and a dense adapter side by side share the fused expert wrappers, so a scan
    that only asks "does this model have a fused expert LoRA anywhere" answers yes for
    the dense adapter too, and that answer would be written into the dense adapter's own
    adapter_config.json. `adapter_name` defaults to the model's active adapter, so the
    single adapter case is unchanged."""
    if adapter_name is None:
        adapter_name = _default_adapter_name(peft_model)

    parameters = {}
    try:
        modules = list(peft_model.modules())
    except Exception:
        return None

    for module in modules:
        parameter_name = getattr(module, "parameter_name", None)
        if not parameter_name or not hasattr(module, "lora_A"):
            continue
        if not _wrapper_has_adapter(module, adapter_name):
            # A wrapper another adapter owns. Its packing is not this adapter's business.
            continue
        num_experts = int(getattr(module, "num_experts", 1) or 1)
        if num_experts <= 1:
            # Not a fused expert stack, so there is no expert axis to pack.
            continue
        layout = moe_lora_b_layout_for_wrapper(module, adapter_name)
        entry = parameters.get(parameter_name)
        if entry is None:
            parameters[parameter_name] = {
                "lora_B_layout": layout,
                "num_experts": num_experts,
            }
        elif entry["lora_B_layout"] != layout:
            entry["lora_B_layout"] = "mixed"

    if not parameters:
        return None

    detail = {
        "lora_A_layout": LORA_B_LAYOUT_GROUPED_BY_EXPERT,
        "parameters": parameters,
    }
    layouts = {entry["lora_B_layout"] for entry in parameters.values()}
    # The flat key is what a converter branches on, so it is only written when every
    # fused expert parameter agrees on one of the two real names. "mixed" is a report,
    # not a layout, and a converter testing `== "grouped_by_expert"` would read it as
    # rank_major, so it stays in the nested detail where it has to be read deliberately.
    if len(layouts) == 1:
        layout = layouts.pop()
        if layout in (LORA_B_LAYOUT_GROUPED_BY_EXPERT, LORA_B_LAYOUT_RANK_MAJOR):
            detail["lora_B_layout"] = layout
    return detail


def _fused_expert_lora_adapter_config_paths(peft_model, save_directory, selected_adapters):
    """Where PEFT just wrote an adapter_config.json: the root for `default`, a
    subdirectory named after any other adapter. Paired with the adapter each file
    belongs to, because the marker is per adapter."""
    if selected_adapters is None:
        try:
            selected_adapters = list(peft_model.peft_config.keys())
        except Exception:
            selected_adapters = ["default"]
    paths = []
    for adapter_name in selected_adapters:
        directory = save_directory
        if adapter_name != "default":
            directory = os.path.join(save_directory, adapter_name)
        path = os.path.join(directory, "adapter_config.json")
        if os.path.isfile(path):
            paths.append((adapter_name, path))
    return paths


def write_fused_expert_lora_layout(peft_model, save_directory, selected_adapters = None):
    """Record the fused MoE expert lora_B packing in the adapter_config.json PEFT has
    just written, and return the files updated. PEFT ignores keys it does not know, so
    the checkpoint stays loadable by any PEFT version; this only stops a downstream
    converter, or PEFT's own merge in a differently configured process, from having to
    guess (unsloth#6930).

    The packing is recomputed for each adapter being saved, and an adapter with no fused
    expert LoRA is left exactly as PEFT wrote it: not written, and not stripped either,
    since a key this never put there is not this function's to remove. A model holding a
    fused expert adapter and a dense one would otherwise describe the fused adapter's
    experts in the dense adapter's config, which is worse than saying nothing."""
    written = []
    for adapter_name, path in _fused_expert_lora_adapter_config_paths(
        peft_model, save_directory, selected_adapters,
    ):
        detail = fused_expert_lora_layout(peft_model, adapter_name)
        if detail is None:
            continue
        with open(path, "r", encoding = "utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            continue
        config[FUSED_EXPERT_LORA_DETAIL_KEY] = detail
        if "lora_B_layout" in detail:
            config[FUSED_EXPERT_LORA_LAYOUT_KEY] = detail["lora_B_layout"]
        elif config.get(FUSED_EXPERT_LORA_LAYOUT_KEY) in (
            LORA_B_LAYOUT_GROUPED_BY_EXPERT, LORA_B_LAYOUT_RANK_MAJOR,
        ):
            # An earlier save of this adapter wrote a flat layout and the parameters no
            # longer agree on one, so that key is now false. Only a value this could have
            # written is dropped: anything else is somebody else's key.
            config.pop(FUSED_EXPERT_LORA_LAYOUT_KEY, None)
        # Byte for byte how PeftConfigMixin.save_pretrained writes it, so adding the marker
        # does not also reformat the file PEFT produced.
        _atomic_write_text(path, json.dumps(config, indent = 2, sort_keys = True))
        written.append(path)
    return written


def _atomic_write_text(path, text):
    """Replace `path` with `text`, or leave it exactly as it was.

    Opening the real file "w" truncates it before the first byte is written, so a write
    that fails part way (a full disk, an erroring network mount) leaves a truncated
    adapter_config.json behind. `_patched_peft_model_save_pretrained` then swallows the
    exception and reports a successful save, so the checkpoint is unloadable and nothing
    says so. Writing a sibling temporary file first means a failure destroys only that
    file, and `os.replace` is atomic on POSIX and on Windows, so no reader can observe a
    half-written config either.

    The temporary file is a sibling, not a tempdir entry, because `os.replace` across
    filesystems raises. It is removed on failure so a failed save leaves no litter."""
    # This Unsloth Zoo code section is licensed under AGPL3

    directory = os.path.dirname(path) or "."
    handle, temporary = tempfile.mkstemp(
        dir = directory, prefix = os.path.basename(path) + ".", suffix = ".tmp",
    )
    try:
        # mkstemp creates the file 0600 and os.replace keeps the NEW inode's mode, so
        # without this the config silently drops from (say) 0644 to 0600 on every save
        # and a checkpoint shared with other users or a serving process stops being
        # readable, while its weight files stay readable. Carry the existing mode over;
        # for a config PEFT has not written yet there is nothing to carry, so leave
        # mkstemp's private mode rather than inventing a laxer one.
        try:
            os.chmod(temporary, stat.S_IMODE(os.stat(path).st_mode))
        except OSError:
            pass
        with os.fdopen(handle, "w", encoding = "utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


# Store original PeftModel.save_pretrained for fallback
_original_peft_model_save_pretrained = None


def _save_pretrained_argument(name, default, args, kwargs):
    """One of PeftModel.save_pretrained's arguments, whether the caller passed it by
    keyword or positionally. Resolved against the real signature rather than a copy of
    it, so a PEFT release that adds an argument cannot silently shift the index. `args`
    excludes self and save_directory."""
    if name in kwargs:
        return kwargs[name]
    try:
        import inspect
        parameters = list(
            inspect.signature(_original_peft_model_save_pretrained).parameters
        )
        index = parameters.index(name) - 2
        if 0 <= index < len(args):
            return args[index]
    except Exception:
        pass
    return default


def _patched_peft_model_save_pretrained(self, save_directory, *args, **kwargs):
    """Add the fused expert layout marker to what PEFT saved. Never fails a save: the
    adapter itself is already on disk by the time this runs."""
    # This Unsloth Zoo code section is licensed under AGPL3

    result = _original_peft_model_save_pretrained(self, save_directory, *args, **kwargs)
    if _save_pretrained_argument("is_main_process", True, args, kwargs):
        try:
            write_fused_expert_lora_layout(
                self, save_directory,
                selected_adapters = _save_pretrained_argument(
                    "selected_adapters", None, args, kwargs,
                ),
            )
        except Exception as exception:
            logger = _moe_utils_logger()
            if logger is not None:
                logger.warning(
                    f"Unsloth: could not record the fused MoE expert LoRA layout in "
                    f"adapter_config.json ({type(exception).__name__}: {exception}). The "
                    f"adapter itself saved correctly."
                )
    return result


def _patch_peft_save_pretrained_for_moe_layout():
    # This Unsloth Zoo code section is licensed under AGPL3

    global _original_peft_model_save_pretrained

    try:
        from peft import PeftModel
    except Exception:
        return False

    if getattr(PeftModel.save_pretrained, "_unsloth_moe_layout_patched", False):
        return True

    if _original_peft_model_save_pretrained is None:
        _original_peft_model_save_pretrained = PeftModel.save_pretrained

    patched = wraps(_original_peft_model_save_pretrained)(
        _patched_peft_model_save_pretrained,
    )
    patched._unsloth_moe_layout_patched = True
    PeftModel.save_pretrained = patched
    return True


def patch_param_wrapper_for_moe():
    """Patch PEFT's ParamWrapper.forward for MoE separated LoRA (call after PEFT import)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    global _original_param_wrapper_forward
    global _original_param_wrapper_get_delta_weight

    module = _load_cached_moe_utils_module()
    if module is not None and hasattr(module, "patch_param_wrapper_for_moe"):
        try:
            return module.patch_param_wrapper_for_moe()
        except Exception:
            pass

    try:
        from peft.tuners.lora.layer import ParamWrapper

        if _original_param_wrapper_forward is None:
            _original_param_wrapper_forward = ParamWrapper.forward

        ParamWrapper.forward = _patched_param_wrapper_forward

        # The forward is only half of it: the merge path reads lora_B through
        # get_delta_weight, which is PEFT's and packs it the other way.
        if not getattr(ParamWrapper.get_delta_weight, "_unsloth_moe_layout_patched", False):
            if _original_param_wrapper_get_delta_weight is None:
                _original_param_wrapper_get_delta_weight = ParamWrapper.get_delta_weight
            patched = wraps(_original_param_wrapper_get_delta_weight)(
                _patched_param_wrapper_get_delta_weight,
            )
            patched._unsloth_moe_layout_patched = True
            ParamWrapper.get_delta_weight = patched

        _patch_peft_save_pretrained_for_moe_layout()
        _patch_peft_get_peft_model_for_moe()

        return True
    except ImportError:
        return False


# Gate gradient via the inner-product identity dGate = <A, dA> / gate, instead of
# <dOut, Y> which pins the down-projection output Y on the tape solely for that
# gradient. Output unchanged; on by default to save memory. The runtime gate below
# still auto-disables it for fp16, down-bias models and frozen routers. Set
# UNSLOTH_MOE_GATEGRAD=0 to revert to the standard <dOut, Y> path.


@lru_cache(maxsize=1)
def _moe_gategrad_enabled() -> bool:
    """Whether the MoE gate-gradient identity path is active (on by default).

    Cached with maxsize=1 since UNSLOTH_MOE_GATEGRAD is read once per process. Any
    code or test that toggles the env var at runtime must call
    _moe_gategrad_enabled.cache_clear() afterwards for the change to take effect.
    """
    return os.environ.get("UNSLOTH_MOE_GATEGRAD", "1") != "0"


class _MoEGateGradIdentity(torch.autograd.Function):
    """Identity over the pre-down activation ``inter``; backward derives the gate
    gradient as ``dGate = <inter, dA> / gate`` instead of ``<dOut, Y>``.

    The incoming dA equals ``gate * (dOut @ W2_eff.T)`` for the effective down
    weight (base + LoRA), so ``<inter, dA> / gate`` is exactly ``<Y, dOut>``
    without ever materialising Y for the gradient. The caller passes a
    sign-floored gate and multiplies Y by that same value, so the identity holds
    for any routing weight, including exact zeros. Exact for any linear down
    projection; NOT valid with a post-matmul down bias, so callers must
    disable it when one is present.
    """

    @staticmethod
    def forward(ctx, inter, permuted_weights):
        ctx.save_for_backward(inter, permuted_weights)
        return inter

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_inter):
        # once_differentiable: the reconstruction is first-order exact but its
        # graph is not the original one, so create_graph must raise, not
        # silently return wrong second derivatives.
        # grad_inter is None when nothing downstream needs inter's gradient
        # (e.g. a frozen down projection); there is nothing to propagate then.
        if grad_inter is None:
            return None, None
        inter, gate = ctx.saved_tensors
        grad_gate = None
        # Skip the gate gradient when the routing weight is frozen (e.g. a frozen
        # router under LoRA), where its gradient is never consumed.
        if ctx.needs_input_grad[1]:
            dgate = (inter.to(torch.float32) * grad_inter.to(torch.float32)).sum(dim=-1)
            # The call site multiplies Y by this same sign-floored gate, so the
            # floor cancels exactly and the quotient equals <Y, dOut> for any
            # gate, including exact zeros. The re-clamp is a no-op for the safe
            # gate passed by forward_native_grouped_mm; it only protects direct
            # callers from dividing by a raw zero.
            gate_f = gate.to(torch.float32)
            safe_gate = torch.where(
                gate_f >= 0, gate_f.clamp_min(1e-12), gate_f.clamp_max(-1e-12)
            )
            grad_gate = (dgate / safe_gate).to(gate.dtype)
        # inter passes through unchanged, so its gradient is grad_inter.
        return grad_inter, grad_gate


def forward_native_grouped_mm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Native PyTorch grouped-GEMM MoE forward via torch._grouped_mm (no Triton; needs runtime support)."""
    # This Unsloth Zoo code section is licensed under AGPL3

    # Runtime safety check (defense in depth).
    if not _check_torch_grouped_mm_supported():
        # Compute Capability is CUDA-only; on XPU it would mask this message.
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            where = f"this device (Compute Capability {major}.{minor})"
        else:
            where = "this device"
        raise RuntimeError(
            f"torch._grouped_mm is not supported on {where}. "
            f"Set UNSLOTH_MOE_BACKEND='unsloth_triton' or 'native_torch' to use a compatible backend."
        )

    is_2d_input = hidden_states.dim() == 2
    if is_2d_input:
        sequence_length, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

    hidden_states = hidden_states.view(-1, hidden_dim)

    # Routing: count tokens per expert, sort to group by expert, gather inputs.
    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = count_tokens_per_expert(flat_top_k, self.num_experts, torch.int32)
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[-1]
    permuted_input = hidden_states[token_indices]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # Gate + Up projection with optional separated LoRA (default).
    use_separated_lora = _should_use_separated_lora()
    gate_up_lora = None

    # Prefer LoRA injected by the patched ParamWrapper; fall back to the parameter.
    _stashed_gate_up_lora = take_moe_lora_stash(self, "gate_up_proj")
    if _stashed_gate_up_lora is not None:
        gate_up_lora = _stashed_gate_up_lora[:3]  # (first, second, scaling)
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts, experts_module=self
        )

    if hasattr(self, "gate_up_proj"):
        model_type = getattr(self, "_unsloth_model_type", None)
        _gate_up_src = self.gate_up_proj

        # Provider re-derives the base weight on demand so Fix 3 can recompute it in
        # backward instead of pinning it (grouped_mm needs contiguous weights).
        def _gate_up_provider(_src=_gate_up_src, _mt=model_type, _h=hidden_dim, _dt=hidden_states.dtype, _mod=self):
            return preprocess_weight(_get_base_weight(_src, _dt), "gate_up", _h, _mt, experts_module=_mod)
        mm1_out = _base_grouped_mm(
            permuted_input, offsets, _gate_up_provider, _moe_recompute_enabled(_gate_up_src),
        )

        # Separated LoRA: + ((X @ first) @ second) * scaling.
        if gate_up_lora is not None:
            first_weight, second_weight, scaling = gate_up_lora

            # Cast to input dtype (LoRA is float32) and make contiguous for grouped_mm.
            first_weight = first_weight.to(permuted_input.dtype).contiguous()
            second_weight = second_weight.to(permuted_input.dtype).contiguous()

            try:
                lora_out = _grouped_mm_with_backward_fix(permuted_input, first_weight, offsets)
                lora_out = lora_out.contiguous()
            except RuntimeError as e:
                raise e

            # Second matmul; pad an unaligned output dim or fall back on failure.
            try:
                if second_weight.shape[-1] % 8 != 0:
                    pad_size = 8 - (second_weight.shape[-1] % 8)
                    second_weight_padded = F.pad(
                        second_weight, (0, pad_size)
                    ).contiguous()
                    lora_delta = _grouped_mm_with_backward_fix(
                        lora_out, second_weight_padded, offsets
                    )
                    lora_delta = lora_delta[:, :-pad_size]
                else:
                    lora_delta = _grouped_mm_with_backward_fix(
                        lora_out, second_weight, offsets
                    )
            except RuntimeError:
                # Manual loop fallback on grouped_mm failure (e.g. stride alignment).
                lora_delta = torch.empty(
                    (lora_out.shape[0], second_weight.shape[-1]),
                    dtype=lora_out.dtype,
                    device=lora_out.device,
                )
                cpu_offsets = offsets.cpu().tolist()
                prev_offset = 0
                for i, end in enumerate(cpu_offsets):
                    if prev_offset < end:
                        lora_delta[prev_offset:end] = torch.matmul(
                            lora_out[prev_offset:end], second_weight[i]
                        )
                    prev_offset = end

            mm1_out = mm1_out + lora_delta * scaling

        if hasattr(self, "gate_up_proj_bias") and self.gate_up_proj_bias is not None:
            # repeat_interleave without output_size= D2H-syncs. sorted_indices is
            # already in expert order, so gathering by expert id matches it, sync-free
            # (74 us -> 8 us host at E=128 x 16384 rows).
            sorted_expert_ids = flat_top_k[sorted_indices]
            bias_expanded = self.gate_up_proj_bias.index_select(
                0, sorted_expert_ids.to(self.gate_up_proj_bias.device)
            )
            mm1_out = mm1_out + bias_expanded.to(mm1_out.dtype)

        if "GptOssExperts" in self.__class__.__name__:
            gate = mm1_out[..., ::2]
            up = mm1_out[..., 1::2]
        else:
            gate, up = mm1_out.chunk(2, dim=-1)

    elif hasattr(self, "w1") and hasattr(self, "w3"):
        # Separate w1/w3 weights (older models).
        w1_base = _get_base_weight(self.w1)
        w3_base = _get_base_weight(self.w3)

        w1 = w1_base.transpose(-2, -1)
        w3 = w3_base.transpose(-2, -1)

        gate = _grouped_mm_with_backward_fix(permuted_input, w1, offsets)
        up = _grouped_mm_with_backward_fix(permuted_input, w3, offsets)

        # Add LoRA for w1 and w3 separately if present.
        if use_separated_lora:
            if _has_lora_adapters(self.w1):
                w1_lora = _extract_lora_weights(self.w1, experts_module=self)
                if w1_lora is not None:
                    lora_A, lora_B, scaling = w1_lora
                    lora_A_t = lora_A.transpose(-2, -1)
                    lora_A_out = _grouped_mm_with_backward_fix(
                        permuted_input, lora_A_t, offsets
                    )
                    lora_B_t = lora_B.transpose(-2, -1)
                    lora_B_out = _grouped_mm_with_backward_fix(lora_A_out, lora_B_t, offsets)
                    gate = gate + lora_B_out * scaling

            if _has_lora_adapters(self.w3):
                w3_lora = _extract_lora_weights(self.w3, experts_module=self)
                if w3_lora is not None:
                    lora_A, lora_B, scaling = w3_lora
                    lora_A_t = lora_A.transpose(-2, -1)
                    lora_A_out = _grouped_mm_with_backward_fix(
                        permuted_input, lora_A_t, offsets
                    )
                    lora_B_t = lora_B.transpose(-2, -1)
                    lora_B_out = _grouped_mm_with_backward_fix(lora_A_out, lora_B_t, offsets)
                    up = up + lora_B_out * scaling
    else:
        raise AttributeError("MoE layer must have 'gate_up_proj' or 'w1'/'w3'.")

    # Activation
    if "GptOssExperts" in self.__class__.__name__:
        # Custom GptOss activation.
        limit = getattr(self, "limit", 7.0)
        alpha = getattr(self, "alpha", 1.702)

        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        inter = (up + 1.0) * glu
    elif hasattr(self, 'act_fn') and callable(self.act_fn):
        inter = self.act_fn(gate) * up
    else:
        inter = F.silu(gate) * up

    # Env-gated gate-grad identity; disabled when a down bias exists (identity
    # assumes a linear down), the router is frozen (nothing to synthesize, and
    # the Function would needlessly keep inter saved), or inter is float16. For
    # fp16, the gradient into the identity carries the tiny floored gate as a
    # factor (grad_inter = gate * (W2 @ dOut)); with a near-zero gate that product
    # underflows fp16 to 0 before backward runs, dropping the router gradient for
    # near-zero routes. fp16 therefore keeps the standard multiply (correct
    # gradient, saves Y); bf16/fp32 hold the tiny value and are unaffected.
    _gategrad = (
        _moe_gategrad_enabled()
        and getattr(self, "down_proj_bias", None) is None
        and top_k_weights.requires_grad
        and inter.dtype != torch.float16
    )
    permuted_weights = None
    if _gategrad:
        raw_weights = top_k_weights.reshape(-1)[sorted_indices]
        # Sign-floor the gate away from zero (straight-through, so the synthesized
        # gradient still reaches top_k_weights). The multiply below and the
        # identity's divide use the SAME floored value, so they cancel exactly and
        # the gate gradient <Y, dOut> survives even a routing weight of exactly
        # zero. An fp16 gate cannot represent eps=1e-12 (its smallest normal is
        # ~6e-5), so upcast it to float32; inter is non-fp16 here, so the floored
        # value survives in grad_inter. The forward then changes by at most
        # 1e-12 * |Y| for |gate| < 1e-12.
        if raw_weights.dtype == torch.float16:
            raw_weights = raw_weights.to(torch.float32)
        eps = 1e-12
        floored = torch.where(
            raw_weights >= 0,
            raw_weights.clamp(min=eps),
            raw_weights.clamp(max=-eps),
        )
        permuted_weights = raw_weights + (floored - raw_weights).detach()
        inter = _MoEGateGradIdentity.apply(inter, permuted_weights)

    # Down projection with optional separated LoRA (default).
    down_lora = None

    # Prefer LoRA injected by the patched ParamWrapper; fall back to the parameter.
    _stashed_down_lora = take_moe_lora_stash(self, "down_proj")
    if _stashed_down_lora is not None:
        down_lora = _stashed_down_lora[:3]  # (first, second, scaling)
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(self.down_proj, num_experts=self.num_experts, experts_module=self)

    if hasattr(self, "down_proj"):
        model_type = getattr(self, "_unsloth_model_type", None)
        _down_src = self.down_proj
        def _down_provider(_src=_down_src, _mt=model_type, _h=hidden_dim, _dt=hidden_states.dtype, _mod=self):
            return preprocess_weight(_get_base_weight(_src, _dt), "down", _h, _mt, experts_module=_mod)
        mm2_out = _base_grouped_mm(
            inter, offsets, _down_provider, _moe_recompute_enabled(_down_src),
        )

        if down_lora is not None:
            first_weight, second_weight, scaling = down_lora

            # Cast to input dtype (LoRA is float32) and make contiguous for grouped_mm.
            first_weight = first_weight.to(inter.dtype).contiguous()
            second_weight = second_weight.to(inter.dtype).contiguous()

            lora_out = _grouped_mm_with_backward_fix(inter, first_weight, offsets)
            lora_out = lora_out.contiguous()

            try:
                lora_delta = _grouped_mm_with_backward_fix(lora_out, second_weight, offsets)
            except RuntimeError:
                # Manual loop fallback.
                lora_delta = torch.empty(
                    (lora_out.shape[0], second_weight.shape[-1]),
                    dtype=lora_out.dtype,
                    device=lora_out.device,
                )
                cpu_offsets = offsets.cpu().tolist()
                prev_offset = 0
                for i, end in enumerate(cpu_offsets):
                    if prev_offset < end:
                        lora_delta[prev_offset:end] = torch.matmul(
                            lora_out[prev_offset:end], second_weight[i]
                        )
                    prev_offset = end

            mm2_out = mm2_out + lora_delta * scaling

        if hasattr(self, "down_proj_bias") and self.down_proj_bias is not None:
            # Capture-safe gather; see gate_up_proj_bias above.
            sorted_expert_ids = flat_top_k[sorted_indices]
            bias_expanded = self.down_proj_bias.index_select(
                0, sorted_expert_ids.to(self.down_proj_bias.device)
            ).to(mm2_out.device)
            mm2_out = mm2_out + bias_expanded.to(mm2_out.dtype)

    elif hasattr(self, "w2"):
        w2_base = _get_base_weight(self.w2)
        w2 = w2_base.transpose(-2, -1)
        mm2_out = _grouped_mm_with_backward_fix(inter, w2, offsets)

        if use_separated_lora and _has_lora_adapters(self.w2):
            w2_lora = _extract_lora_weights(self.w2, experts_module=self)
            if w2_lora is not None:
                lora_A, lora_B, scaling = w2_lora
                lora_A_t = lora_A.transpose(-2, -1).contiguous()
                lora_A_out = _grouped_mm_with_backward_fix(inter, lora_A_t, offsets)
                lora_B_t = lora_B.transpose(-2, -1).contiguous()
                lora_B_out = _grouped_mm_with_backward_fix(lora_A_out, lora_B_t, offsets)
                mm2_out = mm2_out + lora_B_out * scaling
    else:
        raise AttributeError("MoE layer must have 'down_proj' or 'w2'.")

    # Apply routing weights and scatter-add (reduce).
    if _gategrad:
        # Gate grad comes from the identity; detach so the multiply does not pin Y.
        mm2_out = mm2_out * permuted_weights.detach().unsqueeze(-1)
    else:
        flat_weights = top_k_weights.reshape(-1)
        permuted_weights = flat_weights[sorted_indices]
        mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    final_hidden_states = combine_permuted_moe_outputs(
        mm2_out,
        sorted_indices,
        batch_size * sequence_length,
        top_k_index.shape[-1],
        out_dtype = hidden_states.dtype,
    )

    if is_2d_input:
        return final_hidden_states

    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def forward_triton_grouped_gemm(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Grouped-GEMM MoE forward via Triton kernels (torch.compile-compatible, mode="max-autotune")."""
    # This Unsloth Zoo code section is licensed under AGPL3

    from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm
    from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

    if not hasattr(self, "_unsloth_moe_configs"):
        self._unsloth_moe_configs = None

    use_separated_lora = _should_use_separated_lora()

    # gate_up LoRA from the patched ParamWrapper (mirrors the down block below).
    gate_up_lora = None
    _stashed_gate_up_lora = take_moe_lora_stash(self, "gate_up_proj")
    if _stashed_gate_up_lora is not None:
        gate_up_lora = _stashed_gate_up_lora[:3]
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts
        )

    # Flatten 3D inputs (batch_size, seq_len, hidden_dim).
    is_3d = hidden_states.dim() == 3
    if is_3d:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = batch_size * seq_len
        if top_k_index.dim() == 3:
            top_k_index = top_k_index.view(-1, top_k_index.shape[-1])
        if top_k_weights.dim() == 3:
            top_k_weights = top_k_weights.view(-1, top_k_weights.shape[-1])
    else:
        num_tokens, hidden_dim = hidden_states.shape

    top_k = top_k_index.shape[1]

    # Cache model dims and kernel configs on first call.
    if self._unsloth_moe_configs is None:
        intermediate_dim = self.gate_up_proj.shape[1] // 2

        # Autotune first GEMM.
        gemm1_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim * 2,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )

        # Autotune second GEMM (output dim is hidden_dim).
        gemm2_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=intermediate_dim,
            intermediate_dim=hidden_dim,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )

        self._unsloth_moe_configs = (intermediate_dim, gemm1_configs, gemm2_configs)
        torch.cuda.empty_cache()

    intermediate_dim, gemm1_configs, gemm2_configs = self._unsloth_moe_configs
    fwd_config_1, bwd_dX_config_1, bwd_dW_config_1 = gemm1_configs
    fwd_config_2, bwd_dX_config_2, bwd_dW_config_2 = gemm2_configs

    token_counts_by_expert, gather_indices = _get_routing_indices(
        top_k_index, self.num_experts
    )
    offsets = torch.cumsum(token_counts_by_expert, dim=0, dtype=torch.int32)

    if self.gate_up_proj.shape[-1] == hidden_dim:
        w1 = self.gate_up_proj
    else:
        w1 = self.gate_up_proj.transpose(-2, -1).contiguous()

    # First grouped GEMM: gate_up projection.
    first_gemm_output = grouped_gemm(
        X=hidden_states,
        W=w1,
        m_sizes=token_counts_by_expert,
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=True,
        permute_y=False,
        autotune=False,  # cached configs
        kernel_config_fwd=fwd_config_1,
        kernel_config_bwd_dX=bwd_dX_config_1,
        kernel_config_bwd_dW=bwd_dW_config_1,
        is_first_gemm=True,
    )

    # Separated LoRA for gate_up. grouped_gemm ran permute_x=True so first_gemm_output
    # is expert-sorted; _apply_lora_grouped_mm wants pre-permuted input, so gather via
    # gather_indices // top_k (expert-sorted row -> originating token row).
    if gate_up_lora is not None:
        first_weight, second_weight, scaling = gate_up_lora
        first_weight = first_weight.to(hidden_states.dtype)
        second_weight = second_weight.to(hidden_states.dtype)
        permuted_hidden = hidden_states[gather_indices // top_k]
        gate_up_lora_delta = _apply_lora_grouped_mm(
            permuted_hidden,
            first_weight,
            second_weight,
            offsets,
            scaling,
            grouped_mm_func=native_moe_grouped_mm,
        )
        first_gemm_output = first_gemm_output + gate_up_lora_delta

    # Activation + gate*up.
    if hasattr(self, 'act_fn') and callable(self.act_fn):
        gate, up = first_gemm_output.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
    else:
        intermediate = _silu_and_mul(first_gemm_output)

    # Grouped GEMM 2: down projection.
    down_lora = None
    _stashed_down_lora = take_moe_lora_stash(self, "down_proj")
    if _stashed_down_lora is not None:
        down_lora = _stashed_down_lora[:3]
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(self.down_proj, num_experts=self.num_experts)

    if self.down_proj.shape[-1] == intermediate.shape[-1]:
        w2 = self.down_proj
    else:
        w2 = self.down_proj.transpose(-2, -1).contiguous()

    second_gemm_output = grouped_gemm(
        X=intermediate,
        W=w2,
        m_sizes=token_counts_by_expert,
        topk=top_k,
        gather_indices=gather_indices,
        permute_x=False,
        permute_y=True,
        autotune=False,  # cached configs
        kernel_config_fwd=fwd_config_2,
        kernel_config_bwd_dX=bwd_dX_config_2,
        kernel_config_bwd_dW=bwd_dW_config_2,
        is_first_gemm=False,
    )

    # Separated LoRA for down (intermediate already permuted from step 1, same offsets).
    if down_lora is not None:
        first_weight, second_weight, scaling = down_lora

        first_weight = first_weight.to(intermediate.dtype)
        second_weight = second_weight.to(intermediate.dtype)

        lora_delta = _apply_lora_grouped_mm(
            intermediate,
            first_weight,
            second_weight,
            offsets,
            scaling,
            grouped_mm_func=native_moe_grouped_mm
        )

        # permute_y=True put second_gemm_output in token order; lora_delta is still
        # expert-sorted, so scatter it rather than adding it row-for-row.
        if lora_delta.dtype == second_gemm_output.dtype:
            second_gemm_output.index_add_(0, gather_indices, lora_delta)
        else:
            # index_add_ requires matching dtypes; the plain add this replaced promoted.
            promoted = torch.promote_types(second_gemm_output.dtype, lora_delta.dtype)
            second_gemm_output = second_gemm_output.to(promoted).index_add(
                0, gather_indices, lora_delta.to(promoted)
            )

    # Apply routing weights and sum across top_k: (num_tokens, top_k, hidden) -> (num_tokens, hidden).
    top_k_weights_casted = top_k_weights.to(hidden_states.dtype)
    final_hidden_states = (
        second_gemm_output.view(num_tokens, top_k, hidden_dim)
        * top_k_weights_casted[..., None]
    )
    final_hidden_states = final_hidden_states.sum(dim=1)

    if is_3d:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

    return final_hidden_states


@torch.compiler.disable
def forward_native_moe_loop(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Loop over experts with routed tokens; torch.compile-disabled to avoid graph breaks on dynamic control flow."""
    # This Unsloth Zoo code section is licensed under AGPL3
    final_hidden_states = torch.zeros_like(hidden_states)
    use_separated_lora = _should_use_separated_lora()

    gate_up_lora = take_moe_lora_stash(self, "gate_up_proj")
    if gate_up_lora is not None:
        gate_up_lora = gate_up_lora[:3]
    elif (
        use_separated_lora
        and hasattr(self, "gate_up_proj")
        and _has_lora_adapters(self.gate_up_proj)
    ):
        gate_up_lora = _extract_lora_weights(
            self.gate_up_proj, num_experts=self.num_experts, experts_module=self
        )
    # Pre-cast LoRA factors to the activation dtype once (avoid per-expert .to()).
    # `scaling` is left alone: a Python float is a no-op, a tensor broadcasts.
    if gate_up_lora is not None:
        _gate_up_first, _gate_up_second, _gate_up_scaling = gate_up_lora
        gate_up_lora = (
            _gate_up_first.to(hidden_states.dtype),
            _gate_up_second.to(hidden_states.dtype),
            _gate_up_scaling,
        )

    down_lora = take_moe_lora_stash(self, "down_proj")
    if down_lora is not None:
        down_lora = down_lora[:3]
    elif (
        use_separated_lora
        and hasattr(self, "down_proj")
        and _has_lora_adapters(self.down_proj)
    ):
        down_lora = _extract_lora_weights(
            self.down_proj, num_experts=self.num_experts, experts_module=self
        )
    if down_lora is not None:
        _down_first, _down_second, _down_scaling = down_lora
        down_lora = (
            _down_first.to(hidden_states.dtype),
            _down_second.to(hidden_states.dtype),
            _down_scaling,
        )

    # Expert mask -> which experts have tokens.
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, n_tokens)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    # Some patches (Qwen3-VL-MoE) store experts in grouped_mm layout (E, in, out)
    # rather than F.linear's (E, out, in) and set _unsloth_grouped_mm_format=True.
    # Prefer it over the shape check, which is unsafe when intermediate_dim == hidden_dim.
    grouped_mm_format = bool(getattr(self, "_unsloth_grouped_mm_format", False))

    # GPT-OSS uses interleaved gate/up, clamped swiglu, and per-expert biases.
    is_gpt_oss = "GptOssExperts" in self.__class__.__name__

    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t.item()

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # gate_up projection for this expert ('gate_up_proj' or 'w1'/'w3').
        if hasattr(self, "gate_up_proj"):
            gate_up_weight = self.gate_up_proj[expert_idx]
            if grouped_mm_format or gate_up_weight.shape[-1] != current_state.shape[-1]:
                gate_up_weight = gate_up_weight.T
            gate_up = F.linear(current_state, gate_up_weight)
            if gate_up_lora is not None:
                first_weight, second_weight, scaling = gate_up_lora
                lora_delta = current_state @ first_weight[expert_idx]
                lora_delta = lora_delta @ second_weight[expert_idx]
                gate_up = gate_up + lora_delta * scaling
            if is_gpt_oss:
                gate_up_bias = getattr(self, "gate_up_proj_bias", None)
                if gate_up_bias is not None:
                    gate_up = gate_up + gate_up_bias[expert_idx].to(gate_up.dtype)
                gate = gate_up[..., ::2]
                up = gate_up[..., 1::2]
            else:
                gate, up = gate_up.chunk(2, dim=-1)
        else:
            gate = F.linear(current_state, self.w1[expert_idx])
            up = F.linear(current_state, self.w3[expert_idx])

        if is_gpt_oss:
            limit = getattr(self, "limit", 7.0)
            alpha = getattr(self, "alpha", 1.702)
            gate = gate.clamp(min=None, max=limit)
            up = up.clamp(min=-limit, max=limit)
            current_hidden_states = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
        elif hasattr(self, "act_fn") and callable(self.act_fn):
            current_hidden_states = self.act_fn(gate) * up
        else:
            current_hidden_states = F.silu(gate) * up

        # down projection for this expert.
        if hasattr(self, "down_proj"):
            down_weight = self.down_proj[expert_idx]
            # Mirror gate_up: prefer the flag over the shape heuristic (unsafe at square dims).
            if grouped_mm_format or down_weight.shape[-1] != current_hidden_states.shape[-1]:
                down_weight = down_weight.T
            down = F.linear(current_hidden_states, down_weight)
            if down_lora is not None:
                first_weight, second_weight, scaling = down_lora
                lora_delta = current_hidden_states @ first_weight[expert_idx]
                lora_delta = lora_delta @ second_weight[expert_idx]
                down = down + lora_delta * scaling
            if is_gpt_oss:
                down_bias = getattr(self, "down_proj_bias", None)
                if down_bias is not None:
                    down = down + down_bias[expert_idx].to(down.dtype)
            current_hidden_states = down
        else:
            current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )

        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states
