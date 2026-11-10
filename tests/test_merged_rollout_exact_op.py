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
from unittest import mock

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


def _lists(device, count, seed, scale = 1e-3, rows = 32):
    """ws, As, Bs * scale for `count` weights of rows, 2 * rows, ... rows, plus copies of the weights."""
    cases = [_case(device, rows * (i + 1), 256, seed = seed + i) for i in range(count)]
    ws, As, Bs = [c[0] for c in cases], [c[1] for c in cases], [c[2] * scale for c in cases]
    return ws, As, Bs, [w.clone() for w in ws]


def _restored(ws, before):
    for w, w0 in zip(ws, before):
        assert torch.equal(_bits(w), _bits(w0))


def _fresh_arenas(device):
    for arena in ex._ARENAS.values():
        arena.clear()
    return "pageable" if device == "cpu" else "pinned"


def test_ops_are_registered():
    assert callable(getattr(ex, "fold_weight_", None)) and callable(getattr(ex, "unfold_weight_", None))
    for name in ("exact_fold_weight_v1", "exact_unfold_weight_v1", "exact_fold_weights_v1", "exact_unfold_weights_v1"):
        assert hasattr(torch.ops.unsloth_zoo, name), name


def test_list_ops_stay_eager_where_tensor_lists_cannot_be_functionalized(monkeypatch):
    # torch 2.4 cannot functionalize a mutated Tensor[]: compiling the list ops failed there.
    import torch._higher_order_ops.auto_functionalize as af

    monkeypatch.setattr(af, "can_auto_functionalize", lambda op: "weights" not in str(op))
    fold, unfold, folds, unfolds = ex._register_weight_ops()
    assert fold is not None and unfold is not None and folds is None and unfolds is None


def run_compiled(device):
    # Single and list ops under torch.compile fullgraph on every backend, restoring every weight bit for bit.
    import torch._dynamo
    from torch._dynamo.testing import CompileCounter
    from torch._dynamo.utils import counters

    torch._logging.set_logs(graph_breaks = True, recompiles = True)

    def step(w, a, b):
        s1 = ex.fold_weight_(w, a, b, 0.5, 3)
        y = w.float().sum()
        s2 = ex.unfold_weight_(w, a, b, 0.5, 3)
        return y, s1 + s2

    def step_list(ws, As, Bs):
        s1 = ex.fold_weights_(ws, As, Bs, [0.5] * len(ws), list(range(len(ws))))
        y = sum(w.float().sum() for w in ws)
        s2 = ex.unfold_weights_(ws, As, Bs, [0.5] * len(ws), list(range(len(ws))))
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
        if backend == "count":
            assert cnt.frame_count <= 2, cnt.frame_count
        fn = torch.compile(step_list, fullgraph = True, backend = "eager" if backend == "count" else backend)
        ws, As, Bs, before = _lists(device, 3, seed = 60)
        folded = sum((w.float() + (b.double() @ a.double() * 0.5).float()).to(w.dtype).float().sum() for w, a, b in zip(ws, As, Bs))
        y, status = fn(ws, As, Bs)
        _restored(ws, before)
        assert status.shape == (3, 2)
        assert abs(float(y) - float(folded)) <= 1e-2 * max(1.0, abs(float(y))), backend
        assert not counters["graph_break"], (backend, dict(counters["graph_break"]))


def _targets(ws, As, Bs, scale):
    return [ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), scale) for W, A, B in zip(ws, As, Bs)]


def run_arena_growth(device):
    kind = _fresh_arenas(device)
    with mock.patch.object(ex, "_ARENA_MIN_BYTES", 64):
        # Growing stores with several live at once: every fold gets room, live stores never move, all restore.
        ws, As, Bs, before = _lists(device, 4, seed = 10, scale = 1)
        targets = _targets(ws, As, Bs, 3.0)
        for t in targets:
            ex.exact_fold([t])
        stats = ex.host_arena_stats()[kind]
        assert stats["live"] == 4 and stats["chunks"] >= 2, stats
        ex.exact_unfold(targets)
        _restored(ws, before)
        stats = ex.host_arena_stats()[kind]
        assert stats["live"] == 0 and stats["chunks"] == 1, stats


def run_arena_reuse(device):
    kind = _fresh_arenas(device)
    ws, As, Bs = zip(*(_case(device, 64, 256, seed = 20 + i) for i in range(3)))
    before = [W.clone() for W in ws]
    allocations = []
    for cycle in range(5):
        targets = _targets(ws, As, Bs, 1.0 + cycle)
        ex.exact_fold(targets)
        ex.exact_unfold(targets)
        allocations.append(ex.host_arena_stats()[kind]["allocations"])
    _restored(ws, before)
    # Warm after the first cycle: no host allocation in cycles 2 to 5.
    assert allocations[1:] == [allocations[0]] * 4, allocations


def run_arena_outlives_inference_mode(device):
    # A first cycle under inference_mode must not leave inference tensors in the arena for later cycles.
    _fresh_arenas(device)
    ws, As, Bs = zip(*(_case(device, 64, 256, seed = 25 + i) for i in range(2)))
    before = [W.clone() for W in ws]
    for mode in (True, False):
        with torch.inference_mode(mode):
            targets = _targets(ws, As, Bs, 2.0)
            ex.exact_fold(targets)
            ex.exact_unfold(targets)
    _restored(ws, before)


def run_opcheck(device):
    # opcheck may run the op on argument copies: it checks schema and fake only, restores are checked elsewhere.
    W, A, B = _case(device, 64, 256, seed = 30)
    ws, As, Bs, before = _lists(device, 3, seed = 60, scale = 1)
    W0 = W.clone()

    def reset():
        ex._OP_LIVE.clear()
        for w, w0 in zip([W, *ws], [W0, *before]): w.copy_(w0)

    ops = torch.ops.unsloth_zoo
    for test in ("test_schema", "test_faketensor"):
        torch.library.opcheck(ops.exact_fold_weight_v1.default, (W, A, B, 0.5, 11), test_utils = test)
        reset()
        ex.fold_weight_(W, A, B, 0.5, 11)
        torch.library.opcheck(ops.exact_unfold_weight_v1.default, (W, A, B, 0.5, 11), test_utils = test)
        reset()
        torch.library.opcheck(ops.exact_fold_weights_v1.default, (ws, As, Bs, [0.5] * 3, [21, 22, 23]), test_utils = test)
        reset()


def _host_ops(fn):
    # aten calls made by fn outside the kernel launches (the interpreter's own tensor traffic is not counted).
    from torch.utils._python_dispatch import TorchDispatchMode
    state = {"ops": 0, "in_kernel": 0}

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            if not state["in_kernel"]: state["ops"] += 1
            return func(*args, **(kwargs or {}))

    kernel = ex._exact_merge_kernel

    class Launcher:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                state["in_kernel"] += 1
                try:
                    return kernel[grid](*args, **kwargs)
                finally:
                    state["in_kernel"] -= 1
            return run

    ex._exact_merge_kernel = Launcher()
    try:
        with Count():
            fn()
    finally:
        ex._exact_merge_kernel = kernel
    return state["ops"]


def run_batched_store(device):
    # One exact_fold call shares one store: the host work of a fold must not grow with the number of weights.
    kind = _fresh_arenas(device)
    # Leases other checks left live (opcheck folds on copies it never unfolds) are not this check's.
    live = ex.host_arena_stats()[kind]["live"]
    cases = []
    for i in range(6):
        dtype = torch.float16 if i % 2 else torch.bfloat16
        W, A, B = _case(device, 32 * (1 + i % 3), 128 * (1 + i % 2), seed = 40 + i, dtype = dtype)
        W[0, :8] = 2.0**-20   # tiny weights under a large delta: escapes in some targets, at varying offsets
        cases.append((W, A, B * (1.0, 1e-4, 1e-5)[i % 3]))
    before = [W.clone() for W, _, _ in cases]

    def targets_of(cs):
        return _targets(*zip(*cs), 0.5)

    ops = {}
    for n in (2, 6):
        ts = targets_of(cases[:n])
        ex.exact_fold(ts)   # warm-up
        ex.exact_unfold(ts)
        ts = targets_of(cases[:n])
        ops[n] = _host_ops(lambda ts = ts: ex.exact_fold(ts))
        if n == 6:
            # Both store forms in one call: the small adapters indexed, the large ones on the stream.
            assert set(ts[0].store.indexed) == {True, False}, ts[0].store.indexed
            assert ex.host_arena_stats()[kind]["live"] == live + 1, ex.host_arena_stats()
            assert sum(t.host_idx.numel() for t in ts) > 0 and sum(t.host_words.numel() for t in ts) > 0
            # Unfold in two parts: the store is shared, so it must stay readable (and leased) until the last target.
            ex.exact_unfold(ts[2:5])
            assert ex.host_arena_stats()[kind]["live"] == live + 1
            _restored([c[0] for c in cases[2:5]], before[2:5])
        ex.exact_unfold(ts)
        if n == 6:
            assert ex.host_arena_stats()[kind]["live"] == live
        _restored([c[0] for c in cases[:n]], before[:n])
    assert ops[6] == ops[2], ops
    # Folded targets dropped without an unfold (an aborted rollout) give their shared lease back at once.
    ts = targets_of([(W.clone(), A, B) for W, A, B in cases[:3]])
    ex.exact_fold(ts)
    assert ex.host_arena_stats()[kind]["live"] == live + 1
    if device == "cuda":
        # Offloaded on a side stream: the dropped lease waits for that copy before its slice can be handed out again.
        stream = torch.cuda.Stream()
        ts = targets_of([(W.clone(), A, B) for W, A, B in cases[:3]])
        ex.exact_fold(ts, stream = stream)
        store = ts[0].store
        assert store.lease.pending is store.ready is not None
        del store
    del ts
    assert ex.host_arena_stats()[kind]["live"] == live


@pytest.mark.skipif(not _cuda(), reason = "needs CUDA + Triton")
@pytest.mark.parametrize("how", ["refused", "disabled"])
def test_ops_fold_exactly_without_pinned_memory(monkeypatch, how):
    # Pinning refused (WSL2) or disabled: stores and status tensors fall back to pageable memory, still exact.
    if how == "disabled":
        monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    else:
        def refuse(*args, **kwargs):
            raise RuntimeError("pinned memory refused")

        monkeypatch.setattr(torch.Tensor, "pin_memory", refuse)
        monkeypatch.setattr(ex._ARENAS[True], "_new", lambda nbytes: refuse())
    ws, As, Bs, before = _lists("cuda", 2, seed = 90, rows = 64)
    ex.fold_weights_(ws, As, Bs, [0.5, 0.5], [901, 902])
    ex.unfold_weights_(ws, As, Bs, [0.5, 0.5], [901, 902])
    torch.cuda.synchronize()
    _restored(ws, before)
    w, A, B = _case("cuda", 32, 256, seed = 99)
    w0 = w.clone()
    ex.fold_weight_(w, A, B * 1e-3, 0.5, 903)
    ex.unfold_weight_(w, A, B * 1e-3, 0.5, 903)
    assert torch.equal(_bits(w), _bits(w0))


def test_dropped_lease_waits_for_its_pending_copy():
    # A lease dropped before its unfold waits for its pending copy, and frees the slice even if that wait raises.
    arena = ex._HostArena(False)
    order = []

    class Event:
        def __init__(self, fail = False):
            self.fail = fail

        def synchronize(self):
            order.append(("sync", arena.live))
            if self.fail:
                raise RuntimeError("copy failed")

    lease = arena.take(100)
    lease.pending = Event()
    del lease
    assert order == [("sync", 1)] and arena.live == 0
    lease = arena.take(100)
    lease.pending = Event(fail = True)
    with pytest.raises(RuntimeError):
        lease.release()
    assert arena.live == 0 and lease.pending is None


def run_unknown_keys(device):
    # Unknown keys raise and leave the folded weights unfoldable.
    ws, As, Bs, before = _lists(device, 3, seed = 60, scale = 1)
    ex.fold_weights_(ws, As, Bs, [0.5] * 3, [31, 32, 33])
    with pytest.raises(RuntimeError, match = "99"):
        ex.unfold_weights_(ws, As, Bs, [0.5] * 3, [31, 32, 99])
    ex.unfold_weights_(ws, As, Bs, [0.5] * 3, [31, 32, 33])
    _restored(ws, before)


def run_empty_tiles(device):
    # Rows 96.. have B = 0 (a fresh LoRA): those tiles hold no code words and take the fast path; the others keep a
    # stream, including small-adapter rows 32..95 where a few elements still cross an exponent boundary.
    def variants():
        caches = getattr(ex._exact_merge_kernel, "device_caches", None) or {}
        return {key: set(kernels[0]) for key, kernels in caches.items()}

    # Other tests in the process may have compiled variants for other shapes already: count only this run's.
    seen = variants()
    g = torch.Generator().manual_seed(7)
    for dtype in (torch.float16, torch.bfloat16):
        W = (torch.randn(160, 512, generator = g) * 0.02).to(dtype).to(device)
        W[5, :7] = torch.tensor([0.0, -0.0, 6e-8, -6e-8, 1e-3, 60000.0, -1e-6]).to(dtype)
        A = (torch.randn(4, 512, generator = g) * 0.05).to(device)
        B = (torch.randn(160, 4, generator = g) * 0.0002).to(device)
        B[:32] *= 300
        B[96:] = 0
        W0 = W.clone()
        t = ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), 0.5)
        ex.exact_fold([t])
        wcnt = t.host_wcnt.to(torch.int64)
        assert (wcnt == 0).any() and (wcnt > 0).any(), wcnt
        assert not torch.equal(_bits(W), _bits(W0))
        ex.exact_unfold([t])
        assert torch.equal(_bits(W), _bits(W0)), dtype
    if device == "cuda":
        for key, kernels in variants().items():
            # One compiled variant per pass (count / fold / unfold) and weight dtype: fp16 and bf16 here.
            new = kernels - seen.get(key, set())
            assert len(new) <= 6, len(new)


def run_fold_failure_rolls_back(device):
    # A fold that raises after changing the weights (offload fails) rolls back bit for bit and frees its keys.
    ws, As, Bs, before = _lists(device, 2, seed = 70)
    with mock.patch.object(ex, "_offload", side_effect = RuntimeError("offload failed")):
        with pytest.raises(RuntimeError, match = "offload failed"):
            ex.fold_weights_(ws, As, Bs, [0.5, 0.5], [801, 802])
    _restored(ws, before)
    assert not any(k[0] in (801, 802) for k in ex._OP_LIVE)
    ex.fold_weights_(ws, As, Bs, [0.5, 0.5], [801, 802])
    ex.unfold_weights_(ws, As, Bs, [0.5, 0.5], [801, 802])
    _restored(ws, before)


def run_status_failure_rolls_back(device):
    # The status allocation after the fold is inside the rollback too: an OOM there must not strand folded weights.
    ws, As, Bs, before = _lists(device, 1, seed = 72)
    with mock.patch.object(ex, "_status", side_effect = torch.OutOfMemoryError("out of memory")):
        with pytest.raises(torch.OutOfMemoryError):
            ex.fold_weights_(ws, As, Bs, [0.5], [805])
    _restored(ws, before)
    assert not any(k[0] == 805 for k in ex._OP_LIVE)


def run_unfold_failure_keeps_keys(device):
    # An unfold that raises before restoring (launch OOM) keeps its keys; a retry restores bit for bit.
    ws, As, Bs, before = _lists(device, 2, seed = 75)
    ex.fold_weights_(ws, As, Bs, [0.5, 0.5], [811, 812])
    with mock.patch.object(ex, "_launch", side_effect = RuntimeError("out of memory")):
        with pytest.raises(RuntimeError, match = "out of memory"):
            ex.unfold_weights_(ws, As, Bs, [0.5, 0.5], [811, 812])
    assert sum(k[0] in (811, 812) for k in ex._OP_LIVE) == 2
    ex.unfold_weights_(ws, As, Bs, [0.5, 0.5], [811, 812])
    _restored(ws, before)


def run_partial_unfold_is_not_retried(device):
    # An unfold that raises after its inverse kernels ran (escape restore OOM) must drop its keys: a retry would
    # subtract the adapter a second time.
    ws, As, Bs, before = _lists(device, 2, seed = 80)
    ex.fold_weights_(ws, As, Bs, [0.5, 0.5], [821, 822])
    with mock.patch.object(ex, "_restore_escapes", side_effect = torch.OutOfMemoryError("out of memory")):
        with pytest.raises(torch.OutOfMemoryError):
            ex.unfold_weights_(ws, As, Bs, [0.5, 0.5], [821, 822])
    after = [_bits(w).clone() for w in ws]
    assert not any(k[0] in (821, 822) for k in ex._OP_LIVE)
    with pytest.raises(RuntimeError, match = "were not folded"):
        ex.unfold_weights_(ws, As, Bs, [0.5, 0.5], [821, 822])
    assert all(torch.equal(_bits(w), a) for w, a in zip(ws, after))


def run_fold_keeps_no_weight_reference(device):
    # The live store must not pin the folded tensors (a functionalization clone under compile would stay alive).
    import weakref

    ws, As, Bs, before = _lists(device, 1, seed = 85)
    clone = ws[0].clone()
    alive = weakref.ref(clone)
    ex.fold_weights_([clone], As, Bs, [0.5], [831])
    ws[0].copy_(clone)
    del clone
    assert alive() is None
    ex.unfold_weights_(ws, As, Bs, [0.5], [831])
    _restored(ws, before)


def run_all(device):
    run_empty_tiles(device)
    run_fold_keeps_no_weight_reference(device)
    run_partial_unfold_is_not_retried(device)
    run_fold_failure_rolls_back(device)
    run_status_failure_rolls_back(device)
    run_unfold_failure_keeps_keys(device)
    run_compiled(device)
    run_arena_growth(device)
    run_arena_reuse(device)
    run_arena_outlives_inference_mode(device)
    run_opcheck(device)
    run_batched_store(device)
    run_unknown_keys(device)


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
