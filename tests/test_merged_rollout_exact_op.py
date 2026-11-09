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

"""Exact fold as torch.compile-able custom ops, and the reused host store.

fold_weight_ / unfold_weight_ must compile with fullgraph=True (0 graph breaks, at most 2 frames over 3 shapes) under
the counting, aot_eager and inductor backends and restore bit for bit; the host store comes from an arena that grows
in place and allocates nothing once a repeated fold / unfold loop is warm. Without CUDA the kernel runs under
TRITON_INTERPRET=1 in a subprocess (test_cpu_interpreter_ops)."""

import os
import subprocess
import sys

import pytest
import torch

from unsloth_zoo import merged_rollout_exact as ex


def _bits(t):
    return t.view(torch.int16)


def _cuda():
    return torch.cuda.is_available() and ex.triton is not None


def _case(device, n, k, rank = 4, seed = 0, dtype = torch.float16):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(n, k, generator = g) * 0.02).to(dtype).to(device)
    A = torch.randn(rank, k, generator = g).to(dtype).to(device)
    B = torch.randn(n, rank, generator = g).to(dtype).to(device)
    return W, A, B


def test_ops_are_registered():
    assert callable(getattr(ex, "fold_weight_", None)) and callable(getattr(ex, "unfold_weight_", None))
    assert hasattr(torch.ops.unsloth_zoo, "exact_fold_weight_v1")
    assert hasattr(torch.ops.unsloth_zoo, "exact_unfold_weight_v1")


def run_compiled(device):
    import torch._dynamo
    from torch._dynamo.testing import CompileCounter
    from torch._dynamo.utils import counters

    torch._logging.set_logs(graph_breaks = True, recompiles = True)

    def step(w, a, b):
        s1 = ex.fold_weight_(w, a, b, 0.5, 3)
        y = w.float().sum()
        s2 = ex.unfold_weight_(w, a, b, 0.5, 3)
        return y, s1 + s2

    for backend in ("count", "aot_eager", "inductor"):
        torch._dynamo.reset()
        counters.clear()
        cnt = CompileCounter()
        fn = torch.compile(step, fullgraph = True, backend = cnt if backend == "count" else backend)
        for i, (n, k) in enumerate(((64, 256), (96, 384), (128, 512))):
            W, A, B = _case(device, n, k, seed = i)
            W0 = W.clone()
            folded = (W.float() + (B.double() @ A.double() * 0.5).float()).to(W.dtype)
            y, status = fn(W, A, B)
            assert torch.equal(_bits(W), _bits(W0)), (backend, n, k)
            # The traced region saw the folded weight.
            assert abs(float(y) - float(folded.float().sum())) <= 1e-2 * max(1.0, abs(float(y))), (backend, n, k)
            assert int(status[0]) > 0
        assert not counters["graph_break"], (backend, dict(counters["graph_break"]))
        if backend == "count":
            assert cnt.frame_count <= 2, cnt.frame_count


def run_arena_growth(device):
    for arena in ex._ARENAS.values():
        arena.clear()
    old_min = ex._ARENA_MIN_BYTES
    ex._ARENA_MIN_BYTES = 64
    try:
        # Growing stores with several live at once: every fold gets room, live stores never move, all restore.
        cases = [_case(device, 32 * (i + 1), 256, seed = 10 + i) for i in range(4)]
        before = [W.clone() for W, _, _ in cases]
        targets = []
        for W, A, B in cases:
            t = ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), 3.0)
            ex.exact_fold([t])
            targets.append(t)
        stats = ex.host_arena_stats()["pageable" if device == "cpu" else "pinned"]
        assert stats["live"] == 4 and stats["chunks"] >= 2, stats
        ex.exact_unfold(targets)
        for (W, _, _), W0 in zip(cases, before):
            assert torch.equal(_bits(W), _bits(W0))
        stats = ex.host_arena_stats()["pageable" if device == "cpu" else "pinned"]
        # Nothing live: the chunks are merged into one of the combined size.
        assert stats["live"] == 0 and stats["chunks"] == 1, stats
    finally:
        ex._ARENA_MIN_BYTES = old_min


def run_arena_reuse(device):
    for arena in ex._ARENAS.values():
        arena.clear()
    kind = "pageable" if device == "cpu" else "pinned"
    cases = [_case(device, 64, 256, seed = 20 + i) for i in range(3)]
    before = [W.clone() for W, _, _ in cases]
    allocations = []
    for cycle in range(5):
        targets = [ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), 1.0 + cycle) for W, A, B in cases]
        ex.exact_fold(targets)
        ex.exact_unfold(targets)
        allocations.append(ex.host_arena_stats()[kind]["allocations"])
    for (W, _, _), W0 in zip(cases, before):
        assert torch.equal(_bits(W), _bits(W0))
    # Warm after the first cycle: no host allocation in cycles 2 to 5.
    assert allocations[1:] == [allocations[0]] * 4, allocations


def run_opcheck(device):
    # opcheck may run the op on copies of the arguments, so it checks the schema and the fake only; the restore is
    # checked above. The slot and the weight are reset after every check.
    W, A, B = _case(device, 64, 256, seed = 30)
    W0 = W.clone()

    def reset():
        ex._OP_LIVE.clear()
        W.copy_(W0)

    for test in ("test_schema", "test_faketensor"):
        torch.library.opcheck(torch.ops.unsloth_zoo.exact_fold_weight_v1.default, (W, A, B, 0.5, 11), test_utils = test)
        reset()
        ex.fold_weight_(W, A, B, 0.5, 11)
        torch.library.opcheck(torch.ops.unsloth_zoo.exact_unfold_weight_v1.default, (W, A, B, 0.5, 11), test_utils = test)
        reset()


def run_all(device):
    run_compiled(device)
    run_arena_growth(device)
    run_arena_reuse(device)
    run_opcheck(device)


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
def test_ops_on_cuda():
    run_all("cuda")


@pytest.mark.skipif(_cuda() or ex.triton is None, reason = "CUDA runs the checks directly; this is the CPU interpreter")
def test_cpu_interpreter_ops():
    root = os.path.dirname(os.path.dirname(os.path.abspath(ex.__file__)))
    path = os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "", PYTHONPATH = path)
    result = subprocess.run([sys.executable, __file__], env = env, capture_output = True, text = True, timeout = 1800)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "OPS OK" in result.stdout


if __name__ == "__main__":
    assert os.environ.get("TRITON_INTERPRET") == "1", "run under TRITON_INTERPRET=1"
    run_all("cpu")
    print("OPS OK")
