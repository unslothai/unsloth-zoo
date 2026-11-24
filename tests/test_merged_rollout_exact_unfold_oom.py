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

"""An out-of-memory error anywhere in exact_unfold never leaves a weight half restored without the caller knowing.

Each case fails the k-th allocation exact_unfold makes, for every k, then does what Unsloth's restore does: retry
when no target is marked restoring, else give up (poisoned). Every k must end bit exact after at most one retry.
After the first in-place write nothing may allocate on the device, and any failure there must leave the targets
restoring (never unfolded without the checksum compared). Without CUDA the kernels run under TRITON_INTERPRET=1 in a
subprocess; torch.version.hip is mocked only for the simulated ROCm cases, so a native ROCm run keeps its own."""

import contextlib
import os
import subprocess
import sys
import types
from unittest import mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from unsloth_zoo import merged_rollout_exact as ex

_ALLOCS = ("tensor", "empty", "zeros", "zeros_like", "stack", "cat", "arange", "repeat_interleave", "cumsum")


def _has_triton():
    # A launchable kernel, not zoo's stub (UNSLOTH_ZOO_DISABLE_GPU_INIT=1).
    return ex.triton is not None and hasattr(getattr(ex, "_exact_merge_kernel", None), "__getitem__")


def _cuda():
    return torch.cuda.is_available() and _has_triton()


class _FailingTorch:
    """The module's torch, with its k-th allocation (counted from arming) raising an out-of-memory error once."""

    def __init__(self, k):
        self.k, self.seen, self.armed, self.fired = k, 0, False, False

    def __getattr__(self, name):
        real = getattr(torch, name)
        if name not in _ALLOCS:
            return real

        def call(*args, **kwargs):
            if self.armed and not self.fired:
                self.seen += 1
                if self.seen == self.k:
                    self.fired = True
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")
            return real(*args, **kwargs)
        return call


class _AfterWrite(TorchDispatchMode):
    """Every aten op after the first unfold kernel (Triton launches excluded): records the ones that return new
    device storage, and with fail = k raises an out-of-memory error at the k-th op (any op, allocating or not)."""

    def __init__(self, device, fail = 0):
        super().__init__()
        self.device, self.fail, self.seen, self.fired, self.allocs = torch.device(device), fail, 0, False, []

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        self.seen += 1
        if self.seen == self.fail:
            self.fired = True
            raise torch.cuda.OutOfMemoryError(f"CUDA out of memory (injected at {func})")
        out = func(*args, **kwargs)
        inputs = {a.untyped_storage().data_ptr() for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)}
        for o in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(o, torch.Tensor) and o.device.type == self.device.type and o.untyped_storage().nbytes() \
                    and o.untyped_storage().data_ptr() not in inputs:
                self.allocs.append(str(func))
        return out


def _targets(device, count, seed, dtype = torch.float16, strided = False):
    g = torch.Generator().manual_seed(seed)
    ws, ts = [], []
    for i in range(count):
        rows = 32 * (i + 1)
        if strided:
            # A transposed view at a nonzero storage offset, same logical shape.
            W = (torch.randn(2, 128, rows, generator = g) * 0.02).to(dtype).to(device)[1:].transpose(1, 2)
        else:
            W = (torch.randn(1, rows, 128, generator = g) * 0.02).to(dtype).to(device)
        A = torch.randn(1, 4, 128, generator = g).to(dtype).to(device)
        B = torch.randn(1, rows, 4, generator = g).to(dtype).to(device)
        ws.append(W)
        ts.append(ex.ExactFoldTarget(W, A, B, 0.5))
    return ws, ts


def _bits(t):
    return t.view(torch.int16)


def _hip(hip):
    # None keeps the runner's own backend (native ROCm stays ROCm).
    return contextlib.nullcontext() if hip is None else mock.patch.object(torch.version, "hip", hip)


def _sweep(device, count, after_kernels, hip = None, subset = None, strided = False):
    fired_after_kernels = 0
    for k in range(1, 64):
        ws, ts = _targets(device, count, seed = 7 * count + k, strided = strided)
        before = [w.clone() for w in ws]
        ex.exact_fold(ts)
        assert sum(t.bad for t in ts) > 0  # escapes and codes are exercised
        unfold = ts if subset is None else ts[:subset]
        uploads = [t.store for t in unfold if t.store.dev is None]
        fake = _FailingTorch(k)
        real_launch = ex._launch

        def launch(t, mode, **kwargs):
            out = real_launch(t, mode, **kwargs)
            if mode == ex._UNFOLD: fake.armed = True
            return out
        if not after_kernels: fake.armed = True
        with mock.patch.object(ex, "torch", fake), mock.patch.object(ex, "_launch", launch), _hip(hip):
            try:
                ex.exact_unfold(unfold)
            except torch.cuda.OutOfMemoryError:
                if not fake.fired: raise
        if not fake.fired:
            assert after_kernels or k > 1, "no allocation to fail before the first write"
            with _hip(hip): ex.exact_unfold(ts)
            for w, w0 in zip(ws, before): assert torch.equal(_bits(w), _bits(w0))
            return fired_after_kernels
        # Unsloth's restore: a target marked restoring means a retry would subtract the adapter twice.
        assert not any(t.restoring for t in unfold), f"allocation {k} poisoned the weights"
        # The failed attempt keeps no device copy of a store it uploaded.
        assert all(s.dev is None for s in uploads), f"allocation {k} kept an uploaded store"
        fired_after_kernels += after_kernels
        with _hip(hip): ex.exact_unfold(ts)
        for w, w0 in zip(ws, before): assert torch.equal(_bits(w), _bits(w0)), f"allocation {k} restore not exact"
    raise AssertionError("allocation sweep did not end")


def _armed_after_first_write(mode):
    # The dispatch mode from the first unfold kernel on; Triton launches themselves run outside it.
    real_launch, real_escape = ex._launch, ex._escape_kernel

    def launch(t, kind, **kwargs):
        with _disable_current_modes():
            out = real_launch(t, kind, **kwargs)
        if kind == ex._UNFOLD and not stack.enter: stack.enter = mode.__enter__() or True
        return out

    class _Escape:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                with _disable_current_modes():
                    return real_escape[grid](*args, **kwargs)
            return run
    stack = types.SimpleNamespace(enter = None)
    return stack, mock.patch.object(ex, "_launch", launch), mock.patch.object(ex, "_escape_kernel", _Escape())


def _after_write(device, count, hip = None, subset = None, strided = False, fail = 0, corrupt = False):
    ws, ts = _targets(device, count, seed = 31 * count + fail, strided = strided)
    before = [w.clone() for w in ws]
    ex.exact_fold(ts)
    if corrupt:
        # A different adapter at unfold: only the checksum can tell.
        ts[0].b = ts[0].b + 0.5
    unfold = ts if subset is None else ts[:subset]
    mode = _AfterWrite(device, fail)
    stack, p_launch, p_escape = _armed_after_first_write(mode)
    error = None
    with p_launch, p_escape, _hip(hip):
        try:
            ex.exact_unfold(unfold)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error = e
        finally:
            if stack.enter: mode.__exit__(None, None, None)
    return ws, ts, unfold, before, mode, error


_CASES = (
    # (count, hip, subset): whole store (one escape launch), one weight, simulated ROCm per-target, part of a store.
    (2, None, None), (1, None, None), (2, "7.0", None), (3, None, 2),
)


def check_preparation(device):
    # From the start of exact_unfold (after the first write the dispatch check below covers every op); a strided view
    # on the per-target path, which addresses it by storage offsets.
    for count, hip, subset in _CASES:
        _sweep(device, count, False, hip, subset)
    _sweep(device, 1, False, strided = True)


def check_no_device_allocation_after_write(device):
    for count, hip, subset in _CASES:
        for strided in (False, True):
            ws, ts, unfold, before, mode, error = _after_write(device, count, hip, subset, strided)
            assert error is None, error
            assert mode.seen > 0, "the dispatch mode saw nothing"
            assert not mode.allocs, f"{(count, hip, subset, strided)}: allocates after the first write: {mode.allocs}"
            for w, w0 in list(zip(ws, before))[:len(unfold)]: assert torch.equal(_bits(w), _bits(w0))


def check_failure_after_write_is_never_silent(device):
    # Fail the k-th op after the first write with a corrupted adapter: the caller must see restoring targets (and
    # give up) or the mismatch; never a clean state over wrong bits that a retry accepts.
    for count, hip, subset in ((2, None, None), (1, None, None), (2, "7.0", None)):
        for k in range(1, 200):
            ws, ts, unfold, before, mode, error = _after_write(device, count, hip, subset, fail = k, corrupt = True)
            if not mode.fired:
                assert isinstance(error, RuntimeError) and "checksum mismatch" in str(error), error
                break
            assert isinstance(error, torch.cuda.OutOfMemoryError), error
            if any(t.restoring for t in unfold): continue  # poisoned, and the caller knows
            with _hip(hip):
                with pytest.raises(RuntimeError, match = "checksum mismatch"):
                    ex.exact_unfold(unfold)
        else:
            raise AssertionError("op sweep did not end")


def check_grad_enabled_view(device):
    # A grad-enabled non-leaf weight restores directly, as on main (the per-target gather is out=).
    for count, hip in ((1, None), (2, "7.0")):
        ws, ts = _targets(device, count, seed = 53)
        leaves = [w.clone().requires_grad_() for w in ws]
        ts = [ex.ExactFoldTarget(l * 1, t.a, t.b, t.scale) for l, t in zip(leaves, ts)]
        with torch.no_grad(): before = [t.w.clone() for t in ts]
        ex.exact_fold(ts)
        with torch.enable_grad(), _hip(hip):
            ex.exact_unfold(ts)
        for t, w0 in zip(ts, before): assert torch.equal(_bits(t.w.detach()), _bits(w0))


def check_op_status_oom_keeps_keys(device):
    # The unfold op's status tensor fails to allocate: the weight stays folded under its key and a retry restores it.
    ws, ts = _targets(device, 1, seed = 37)
    w, a, b = ws[0][0], ts[0].a[0], ts[0].b[0]
    w0 = w.clone()
    ex.fold_weight_(w, a, b, 0.5, 9801)
    real = ex._status
    calls = {"n": 0}

    def status(rows, dev):
        calls["n"] += 1
        if calls["n"] == 1: raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected status)")
        return real(rows, dev)
    with mock.patch.object(ex, "_status", status):
        with pytest.raises(torch.cuda.OutOfMemoryError):
            ex.unfold_weight_(w, a, b, 0.5, 9801)
    ex.unfold_weight_(w, a, b, 0.5, 9801)
    assert torch.equal(_bits(w), _bits(w0))


def run_all(device, only = None):
    for name, check in (
        ("preparation", check_preparation),
        ("no_alloc", check_no_device_allocation_after_write),
        ("never_silent", check_failure_after_write_is_never_silent),
        ("grad", check_grad_enabled_view),
        ("status", check_op_status_oom_keeps_keys),
    ):
        if only is None or name in only:
            check(device)
            print("ok", name, flush = True)


def test_vllm_exact_restore_retries_one_oom_before_any_write():
    # vLLM's merged rollout: an out-of-memory error with no target restoring is retried once; a restoring one poisons.
    from unsloth_zoo import vllm_utils
    for restoring, expect_ok in ((False, True), (True, False)):
        target = types.SimpleNamespace(folded = True, restoring = False)
        calls = []

        def unfold(targets, stream = None):
            calls.append(1)
            if len(calls) == 1:
                target.restoring = restoring
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")
        state = types.SimpleNamespace(
            entries = [(None, torch.zeros(1), None)], exact_targets = [target], stream = None, poisoned = None,
            record = lambda: None,
        )
        with mock.patch.object(ex, "exact_unfold", unfold), mock.patch.object(torch.cuda, "empty_cache", lambda: None):
            if expect_ok:
                vllm_utils._merged_rollout_restore(state)
                assert state.poisoned is None and len(calls) == 2
            else:
                with pytest.raises(RuntimeError, match = "could not restore"):
                    vllm_utils._merged_rollout_restore(state)
                assert state.poisoned is not None and len(calls) == 1


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_unfold_oom_never_poisons_on_cuda():
    run_all("cuda")


def _interpreter(only):
    root = os.path.dirname(os.path.dirname(os.path.abspath(ex.__file__)))
    path = os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "", PYTHONPATH = path)
    result = subprocess.run([sys.executable, __file__, only], env = env, capture_output = True, text = True, timeout = 600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "UNFOLD OOM OK" in result.stdout


_CPU = pytest.mark.skipif(
    _cuda() or not _has_triton(), reason = "CUDA runs the checks directly; this is the CPU interpreter",
)


@_CPU
def test_cpu_interpreter_unfold_oom_never_poisons():
    _interpreter("preparation")


@_CPU
def test_cpu_interpreter_nothing_allocates_after_the_first_write():
    _interpreter("no_alloc,never_silent")


@_CPU
def test_cpu_interpreter_grad_view_and_op_status():
    _interpreter("grad,status")


if __name__ == "__main__":
    assert os.environ.get("TRITON_INTERPRET") == "1", "run under TRITON_INTERPRET=1"
    run_all("cpu", sys.argv[1].split(",") if len(sys.argv) > 1 else None)
    print("UNFOLD OOM OK")
