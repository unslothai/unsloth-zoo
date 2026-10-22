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

"""moe_routing_triton vs the torch routing path, bit for bit: sizes, sync-free, graphs, compile, fallbacks. GPU only."""
import importlib
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
pytest.importorskip("triton")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_FILE = os.path.join(_HERE, "..", "unsloth_zoo", "temporary_patches", "moe_routing_triton.py")


def _load_module():
    name = "unsloth_zoo.temporary_patches.moe_routing_triton"
    if name in sys.modules:
        return sys.modules[name]
    # Standalone: load the file itself so the package's heavy imports are not needed.
    name = "_unsloth_moe_routing_triton_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # Dynamo resolves the traced function's globals by module name
    spec.loader.exec_module(module)
    return module


mrt = _load_module()
DEV = torch.device("cuda", torch.cuda.current_device())
if not mrt.triton_routing_available(DEV):
    pytest.skip(f"Triton routing off: {mrt._DISABLED_REASON or 'kill switch / ROCm / no Triton'}", allow_module_level = True)

EXPERTS = (8, 32, 64, 128, 256, 384)
TOP_K = (1, 2, 4, 6, 8)


def _ids(T, k, E, dist, gen):
    if dist == "random":
        ids = torch.randint(0, E, (T, k), generator = gen)
    elif dist == "skewed":
        w = 1.0 / torch.arange(1, E + 1, dtype = torch.float64) ** 1.5
        ids = torch.multinomial(w, T * k, replacement = True, generator = gen).view(T, k)
    elif dist == "one_expert":
        ids = torch.full((T, k), int(torch.randint(0, E, (1,), generator = gen)))
    elif dist == "empty_experts":
        live = torch.randperm(E, generator = gen)[: max(1, E // 5)]
        ids = live[torch.randint(0, live.numel(), (T, k), generator = gen)]
    else:
        raise ValueError(dist)
    return ids.to(DEV)


def _check(ids, E, **launch_kw):
    got = mrt._launch(ids, E, **launch_kw)
    ref = mrt.torch_moe_routing_sort(ids, E)
    for name, a, b in zip(("counts", "sorted_indices", "offsets"), got, ref):
        assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
        assert torch.equal(a, b), f"{name} differs: T*k={ids.numel()} E={E} {ids.dtype} {launch_kw}"


@pytest.mark.parametrize("E", EXPERTS)
@pytest.mark.parametrize("k", TOP_K)
@pytest.mark.parametrize("dist", ["random", "skewed", "one_expert", "empty_experts"])
def test_matches_stable_argsort(E, k, dist):
    gen = torch.Generator().manual_seed(E * 131 + k)
    for T in (1, 2, 7, 100, 511, 683, 1024, 4097, 20000):
        ids = _ids(T, k, E, dist, gen)
        for t in (ids, ids.to(torch.int32)):
            _check(t, E)
            # Force the three-kernel path with tiny blocks: every block boundary case.
            _check(t, E, single_block_max = 0, block = 16)


@pytest.mark.parametrize("E", EXPERTS)
@pytest.mark.parametrize("k", (1, 8))
@pytest.mark.parametrize("T", (32768, 65536))
def test_large_T(E, k, T):
    gen = torch.Generator().manual_seed(T + E + k)
    for dist in ("random", "skewed"):
        ids = _ids(T, k, E, dist, gen)
        _check(ids, E)
        _check(ids.to(torch.int32), E)


@pytest.mark.parametrize("T", (1, 5, 3000, 9000))
def test_non_contiguous(T):
    gen = torch.Generator().manual_seed(T)
    E, k = 64, 6
    wide = _ids(T, 2 * k, E, "random", gen)
    for t in (
        wide[:, ::2],                               # strided columns
        wide[:, :k],                                # sliced, row stride 2k
        _ids(k, T, E, "random", gen).t(),           # transposed storage
        wide.to(torch.int32)[:, 1::2],
        _ids(T * k, 1, E, "random", gen).view(-1),  # 1-D input
    ):
        assert T == 1 or not t.is_contiguous() or t.dim() == 1
        _check(t, E)
        _check(t, E, single_block_max = 0, block = 32)
        got = mrt.moe_routing_sort(t, E)
        ref = mrt.torch_moe_routing_sort(t, E)
        assert all(torch.equal(a, b) for a, b in zip(got, ref))


def test_out_of_range_ids_stay_a_permutation():
    E = 16
    ids = torch.randint(0, E, (3000, 4), device = DEV)
    ids[5, 1] = E
    ids[2000, 0] = -1
    ids[2999, 3] = 1 << 40
    for kw in ({}, {"single_block_max": 0, "block": 64}):
        counts, order, offsets = mrt._launch(ids, E, **kw)
        assert int(counts.sum()) == ids.numel() - 3
        assert torch.equal(order.sort().values, torch.arange(ids.numel(), device = DEV))
        valid = (ids.reshape(-1) >= 0) & (ids.reshape(-1) < E)
        assert torch.equal(order[: int(valid.sum())], torch.nonzero(valid).view(-1)[ids.reshape(-1)[valid].argsort(stable = True)])


def test_empty_and_single():
    for E in (8, 384):
        _check(torch.empty((0, 4), dtype = torch.int64, device = DEV), E)
        _check(torch.zeros((1, 1), dtype = torch.int64, device = DEV), E)
        _check(torch.full((1, 1), E - 1, dtype = torch.int32, device = DEV), E)


def test_no_host_sync():
    for T in (64, 50000):
        ids = torch.randint(0, 128, (T, 8), device = DEV)
        mrt.moe_routing_sort(ids, 128)   # warm (compile + self-check) outside the check
        torch.cuda.synchronize()
        before = mrt.ROUTING_CALLS["triton"]
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = mrt.moe_routing_sort(ids, 128)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        assert mrt.ROUTING_CALLS["triton"] == before + 1
        ref = mrt.torch_moe_routing_sort(ids, 128)
        assert all(torch.equal(a, b) for a, b in zip(out, ref))


def test_cuda_graph_capture_replay():
    for T in (100, 30000):
        E, k = 64, 4
        static = torch.randint(0, E, (T, k), device = DEV)
        mrt.moe_routing_sort(static, E)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mrt.moe_routing_sort(static, E)   # warm on the side stream
            with torch.cuda.graph(graph, stream = stream):
                outs = mrt.moe_routing_sort(static, E)
        torch.cuda.current_stream().wait_stream(stream)
        for seed in range(3):
            static.copy_(torch.randint(0, E, (T, k), device = DEV, generator = torch.Generator(DEV).manual_seed(seed)))
            graph.replay()
            ref = mrt.torch_moe_routing_sort(static, E)
            assert all(torch.equal(a, b) for a, b in zip(outs, ref))


def _routed(ids, E):
    counts, order, offsets = mrt.moe_routing_sort(ids, E)
    return counts, order // ids.shape[-1], offsets


@pytest.mark.parametrize("dynamic", [False, True])
def test_compile_fullgraph_no_breaks(dynamic):
    import torch._dynamo
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    fn = torch.compile(_routed, fullgraph = True, dynamic = dynamic)
    E = 32
    before = mrt.ROUTING_CALLS["triton"]
    for T in (64, 100, 3000, 7000):
        ids = torch.randint(0, E, (T, 4), device = DEV)
        got = fn(ids, E)
        c, o, f = mrt.torch_moe_routing_sort(ids, E)
        assert torch.equal(got[0], c) and torch.equal(got[1], o // 4) and torch.equal(got[2], f)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    assert mrt.ROUTING_CALLS["triton"] >= before + 4
    if dynamic:
        assert counters["stats"]["unique_graphs"] <= 2, dict(counters["stats"])


def test_compile_reduce_overhead_cudagraphs():
    import torch._dynamo
    torch._dynamo.reset()
    fn = torch.compile(_routed, mode = "reduce-overhead", fullgraph = True)
    E = 128
    for seed in range(4):
        ids = torch.randint(0, E, (8192, 8), device = DEV, generator = torch.Generator(DEV).manual_seed(seed))
        got = [t.clone() for t in fn(ids, E)]
        c, o, f = mrt.torch_moe_routing_sort(ids, E)
        assert torch.equal(got[0], c) and torch.equal(got[1], o // 8) and torch.equal(got[2], f)


def test_launch_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated Triton launch failure")
    monkeypatch.setattr(mrt, "_launch", boom)
    monkeypatch.setattr(mrt, "_DISABLED_REASON", None)
    ids = torch.randint(0, 8, (300, 2), device = DEV)
    before = dict(mrt.ROUTING_CALLS)
    got = mrt._routing_impl(ids, 8)
    ref = mrt.torch_moe_routing_sort(ids, 8)
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
    assert mrt.ROUTING_CALLS["fallback"] == before["fallback"] + 1
    assert mrt._DISABLED_REASON is not None
    monkeypatch.setattr(mrt, "_DISABLED_REASON", None)


def test_first_call_verification_disables_on_mismatch(monkeypatch):
    real = mrt._launch

    def wrong(ids, E, **kw):
        counts, order, offsets = real(ids, E, **kw)
        return counts, order.flip(0), offsets   # a plausible but wrong permutation
    monkeypatch.setattr(mrt, "_launch", wrong)
    monkeypatch.setattr(mrt, "_VERIFIED", {})
    monkeypatch.setattr(mrt, "_DISABLED_REASON", None)
    ids = torch.randint(0, 8, (300, 2), device = DEV)
    got = mrt._routing_impl(ids, 8)
    ref = mrt.torch_moe_routing_sort(ids, 8)
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
    assert mrt._DISABLED_REASON is not None and "differs" in mrt._DISABLED_REASON
    monkeypatch.setattr(mrt, "_DISABLED_REASON", None)


def test_negative_control_unstable_order_is_caught():
    # The equality check must be able to fail: a swapped pair inside one expert is detected.
    ids = torch.zeros((8, 1), dtype = torch.int64, device = DEV)
    counts, order, offsets = mrt._launch(ids, 4)
    bad = order.clone()
    bad[[0, 1]] = bad[[1, 0]]
    assert not torch.equal(bad, mrt.torch_moe_routing_sort(ids, 4)[1])


@pytest.mark.parametrize("value", ["0", "1"])
def test_kill_switch(value):
    code = (
        "import importlib.util, torch\n"
        f"spec = importlib.util.spec_from_file_location('m', {_MODULE_FILE!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "ids = torch.randint(0, 8, (100, 2), device = 'cuda')\n"
        "got = m.moe_routing_sort(ids, 8); ref = m.torch_moe_routing_sort(ids, 8)\n"
        "assert all(torch.equal(a, b) for a, b in zip(got, ref))\n"
        "print(m.ROUTING_CALLS['triton'], m.ROUTING_CALLS['torch'])\n"
    )
    env = dict(os.environ, UNSLOTH_MOE_TRITON_ROUTING = value)
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 1800)
    assert out.returncode == 0, out.stderr[-2000:]
    triton_calls, torch_calls = map(int, out.stdout.split()[-2:])
    if value == "0":
        assert triton_calls == 0
    else:
        assert triton_calls == 1 and torch_calls == 0


def _compiled_variants(kernel):
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(v[0]) for v in caches.values())
    return sum(len(v) for v in kernel.cache.values())


def _clear_compiled(kernel):
    # Earlier tests in this file may already have compiled every tile size: start empty.
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        for v in caches.values():
            v[0].clear()
    else:
        for v in kernel.cache.values():
            v.clear()


def test_scan_kernel_compiles_once_across_token_counts():
    # The scan tile must not follow the block count: training steps with different token counts
    # (here 9 NB values from 16 to 512 blocks) reuse one compiled scan kernel.
    E, k = 64, 8
    _clear_compiled(mrt._route_scan_kernel)
    before = _compiled_variants(mrt._route_scan_kernel)
    seen = set()
    for T in (520, 1100, 2100, 3300, 4300, 6000, 8200, 12000, 16384):
        ids = torch.randint(0, E, (T, k), device = "cuda")
        got = mrt._launch(ids, E)
        ref = mrt.torch_moe_routing_sort(ids, E)
        assert all(torch.equal(a, b) for a, b in zip(got, ref))
        seen.add(_compiled_variants(mrt._route_scan_kernel) - before)
    # <= 2: Triton's own 16-byte pointer-alignment specialization of the scratch slices.
    assert _compiled_variants(mrt._route_scan_kernel) - before <= 2, seen
