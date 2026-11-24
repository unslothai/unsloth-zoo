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
Without CUDA the kernels run under TRITON_INTERPRET=1 in a subprocess."""

import os
import subprocess
import sys
from unittest import mock

import pytest
import torch

from unsloth_zoo import merged_rollout_exact as ex

_ALLOCS = ("tensor", "empty", "zeros", "zeros_like", "stack", "cat", "arange", "repeat_interleave", "cumsum")


def _cuda():
    return torch.cuda.is_available() and ex.triton is not None


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


def _targets(device, count, seed, dtype = torch.float16):
    g = torch.Generator().manual_seed(seed)
    ws, ts = [], []
    for i in range(count):
        W = (torch.randn(1, 32 * (i + 1), 128, generator = g) * 0.02).to(dtype).to(device)
        A = torch.randn(1, 4, 128, generator = g).to(dtype).to(device)
        B = torch.randn(1, 32 * (i + 1), 4, generator = g).to(dtype).to(device)
        ws.append(W)
        ts.append(ex.ExactFoldTarget(W, A, B, 0.5))
    return ws, ts


def _bits(t):
    return t.view(torch.int16)


def _sweep(device, count, after_kernels, hip = None, subset = None):
    fired_after_kernels = 0
    for k in range(1, 64):
        ws, ts = _targets(device, count, seed = 7 * count + k)
        before = [w.clone() for w in ws]
        ex.exact_fold(ts)
        assert sum(t.bad for t in ts) > 0  # escapes and codes are exercised
        unfold = ts if subset is None else ts[:subset]
        fake = _FailingTorch(k)
        real_launch = ex._launch

        def launch(t, mode, **kwargs):
            out = real_launch(t, mode, **kwargs)
            if mode == ex._UNFOLD: fake.armed = True
            return out
        if not after_kernels: fake.armed = True
        with mock.patch.object(ex, "torch", fake), mock.patch.object(ex, "_launch", launch), \
             mock.patch.object(torch.version, "hip", hip):
            try:
                ex.exact_unfold(unfold)
            except torch.cuda.OutOfMemoryError:
                pass
        if not fake.fired:
            assert k > 1, "no allocation to fail"
            ex.exact_unfold(ts)
            for w, w0 in zip(ws, before): assert torch.equal(_bits(w), _bits(w0))
            return fired_after_kernels
        # Unsloth's restore: a target marked restoring means a retry would subtract the adapter twice.
        assert not any(t.restoring for t in unfold), f"allocation {k} poisoned the weights"
        fired_after_kernels += after_kernels
        ex.exact_unfold(ts)
        for w, w0 in zip(ws, before): assert torch.equal(_bits(w), _bits(w0)), f"allocation {k} restore not exact"
    raise AssertionError("allocation sweep did not end")


def run_all(device):
    # Whole store (one escape launch), one weight, ROCm per-target escapes, part of a store; OOMs after the first
    # unfold kernel, then anywhere in the unfold.
    for count, hip, subset in ((2, None, None), (1, None, None), (2, "7.0", None), (3, None, 2)):
        assert _sweep(device, count, True, hip, subset) > 0
    _sweep(device, 2, False)


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_unfold_oom_never_poisons_on_cuda():
    run_all("cuda")


@pytest.mark.skipif(_cuda() or ex.triton is None, reason = "CUDA runs the checks directly; this is the CPU interpreter")
def test_cpu_interpreter_unfold_oom_never_poisons():
    root = os.path.dirname(os.path.dirname(os.path.abspath(ex.__file__)))
    path = os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "", PYTHONPATH = path)
    result = subprocess.run([sys.executable, __file__], env = env, capture_output = True, text = True, timeout = 600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "UNFOLD OOM OK" in result.stdout


if __name__ == "__main__":
    assert os.environ.get("TRITON_INTERPRET") == "1", "run under TRITON_INTERPRET=1"
    run_all("cpu")
    print("UNFOLD OOM OK")
