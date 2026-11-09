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
"""Cached eager Triton launches for small, frequently launched kernels (gpt-oss decode).

JITFunction.run re-binds every argument per call (15-30 us host time); launch() keeps the compiled
kernel from the first call per specialization and calls its launcher directly afterwards. The key is
at least as fine as Triton's specialization. Each kernel's first fast launch is checked bitwise against
the full launch (outputs poisoned first); a mismatch or exception falls back to Triton's own path.
Kill switch: UNSLOTH_TRITON_FAST_LAUNCH=0, read at import.
"""

__all__ = [
    "launch",
    "FAST_LAUNCH_STATS",
]

import os

import torch

try:
    import triton
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_current_device = getattr(torch._C, "_cuda_getDevice", None)


def _launcher_verified():
    try:
        major, minor = (int(part) for part in triton.__version__.split(".")[:2])
    except Exception:
        return False
    return (major, minor) >= (3, 6)


_ENABLED = (
    HAS_TRITON
    and _raw_stream is not None
    and _current_device is not None
    and torch.version.hip is None
    and os.environ.get("TRITON_INTERPRET") != "1"
    and os.environ.get("UNSLOTH_TRITON_FAST_LAUNCH", "1") != "0"
    and _launcher_verified()
)
FAST_LAUNCH_STATS = {"fast": 0, "full": 0}
_PLANS = {}  # keyed by id(kernel): the kernels are module globals, never freed
_PLANS_MAX = 4096
_KERNEL_OK = {}  # id(kernel) -> True after its bitwise self-check, False when it failed
_runtime_hooks = None


def _hook_active(hook):
    # Triton 3.6+ always has a HookChain; empty = no profiler.
    return hook is not None and bool(getattr(hook, "calls", True))


_knobs = None


def _knobs_set():
    # Triton reads these per call; a cached plan would ignore them.
    global _knobs
    if _knobs is None:
        try:
            from triton import knobs
            _knobs = (knobs.compilation, knobs.runtime)
        except Exception:
            _knobs = ()
    if not _knobs:
        return False
    return bool(getattr(_knobs[0], "instrumentation_mode", None)) or bool(getattr(_knobs[1], "debug", None))


def _hooks_set():
    global _runtime_hooks
    if _runtime_hooks is None:
        try:
            from triton import knobs
            _runtime_hooks = knobs.runtime
        except Exception:
            from triton.compiler.compiler import CompiledKernel
            _runtime_hooks = CompiledKernel
    return _hook_active(_runtime_hooks.launch_enter_hook) or _hook_active(_runtime_hooks.launch_exit_hook)


_Tensor = torch.Tensor


def _spec(arg):
    kind = type(arg)
    if kind is int:
        return arg
    if kind is bool:
        return (bool, arg)
    if kind is float:
        return float
    if arg is None:
        return None
    if isinstance(arg, _Tensor):
        return (arg.dtype, not (_Tensor.data_ptr(arg) & 15))
    return (kind, arg)


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _poison(t):
    if t.element_size() in _BITS:
        t.view(_BITS[t.element_size()]).bitwise_not_()
    else:
        t.add_(1)


def _same_bits(a, b):
    # Bitwise, so NaN outputs compare equal to themselves.
    if a.is_floating_point() and a.element_size() in _BITS:
        try:
            return torch.equal(a.view(_BITS[a.element_size()]), b.view(_BITS[b.element_size()]))
        except RuntimeError:
            pass
    return torch.equal(a, b)


def _full(kernel, grid, args, constexprs, options):
    FAST_LAUNCH_STATS["full"] += 1
    return kernel[grid](*args, **constexprs, **options)


def _full_on(kernel, grid, args, constexprs, options, device):
    # Triton launches on the current device, not the tensors' (multi-GPU device_map).
    if device >= 0 and device != torch.cuda.current_device():
        with torch.cuda.device(device):
            return _full(kernel, grid, args, constexprs, options)
    return _full(kernel, grid, args, constexprs, options)


def _run(entry, grid, device, args):
    compiled, tail = entry
    compiled.run(
        grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1, _raw_stream(device),
        compiled.function, compiled.packed_metadata, None, None, None, *args, *tail,
    )


def launch(kernel, grid, args, constexprs, device, outs, **options):
    """kernel[grid](*args, **constexprs, **options). device: index, -1 for CPU. constexprs must be the
    signature arguments after args, in order. outs: tensors written but never read (self-checked)."""
    global _ENABLED
    if (
        not _ENABLED or device < 0 or device != _current_device() or _hooks_set()
        or getattr(kernel, "pre_run_hooks", None) or _knobs_set()
    ):
        return _full_on(kernel, grid, args, constexprs, options, device)
    # id: JITFunction hashes its source cache key (a Python property) on every hash.
    key = (id(kernel), device, *constexprs.values(), *options.items(), *map(_spec, args))
    entry = _PLANS.get(key)
    if entry is None:
        compiled = _full(kernel, grid, args, constexprs, options)
        names = kernel.arg_names[len(args):]
        if hasattr(compiled, "packed_metadata") and tuple(names) == tuple(constexprs):
            compiled.run  # load the module now, not on the fast path
            if len(_PLANS) >= _PLANS_MAX:
                _PLANS.pop(next(iter(_PLANS)))
            _PLANS[key] = (compiled, tuple(constexprs.values()))
        return compiled
    ok = _KERNEL_OK.get(id(kernel))
    if ok is None:
        compiled = _full(kernel, grid, args, constexprs, options)
        if torch.cuda.is_current_stream_capturing():
            return compiled  # check syncs to host: defer to an eager call
        refs = [t.clone() for t in outs]
        for t in outs:  # flip bits so an unwritten element cannot match
            _poison(t)
        try:
            _run(entry, grid, device, args)
            ok = all(map(_same_bits, outs, refs))
        except Exception:
            for t, r in zip(outs, refs):
                t.copy_(r)
            _ENABLED = False
            _PLANS.clear()
            return compiled
        _KERNEL_OK[id(kernel)] = ok
        if not ok:
            for t, r in zip(outs, refs):
                t.copy_(r)
        return compiled
    if not ok:
        return _full(kernel, grid, args, constexprs, options)
    try:
        _run(entry, grid, device, args)
    except Exception:
        # Incompatible launcher signature (raises before launching): disable globally.
        _ENABLED = False
        _PLANS.clear()
        return _full(kernel, grid, args, constexprs, options)
    FAST_LAUNCH_STATS["fast"] += 1
    return entry[0]
