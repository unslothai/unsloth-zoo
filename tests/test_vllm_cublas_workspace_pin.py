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

"""vLLM's captured CUDA graphs bake in the cuBLAS workspace address; torch._C._cuda_clearCublasWorkspaces (Inductor
CUDA graph trees, e.g. Unsloth's block CUDA graphs in training) frees it, and the next decode replay writes split-K
partials into memory the trainer reuses or empty_cache unmapped (cudaErrorIllegalAddress). The capture-time
workspace must live in a private pool that outlives any clear, on V1 and V2 model runners alike."""

import contextlib
import importlib.util
import sys
import types

import pytest
import torch

from unsloth_zoo import vllm_utils


def _fake_vllm(monkeypatch):
    calls = []

    class V1Runner:
        def capture_model(self, *args, **kwargs):
            calls.append("capture_v1")
            return "v1"

    class V2Runner:
        def capture_model(self, *args, **kwargs):
            calls.append("capture_v2")
            return "v2"

    logger = types.SimpleNamespace(info = lambda *args, **kwargs: None)
    modules = {
        "vllm": types.SimpleNamespace(__version__ = "0.29.0"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_model_runner": types.SimpleNamespace(GPUModelRunner = V1Runner, logger = logger),
        "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
        "vllm.v1.worker.gpu.model_runner": types.SimpleNamespace(GPUModelRunner = V2Runner),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *args: object() if name == "vllm.v1.worker.gpu.model_runner" else real_find_spec(name, *args),
    )
    monkeypatch.setattr(vllm_utils, "collect_garbage", lambda *args, **kwargs: None)
    @contextlib.contextmanager
    def held(*args, **kwargs):
        calls.append("hold")
        yield True
        calls.append("release_cache")

    monkeypatch.setattr(vllm_utils, "_vllm_cublas_workspace_held", held)
    return V1Runner, V2Runner, calls


def test_both_model_runners_capture_inside_the_workspace_hold(monkeypatch):
    v1, v2, calls = _fake_vllm(monkeypatch)
    for _ in range(3):
        vllm_utils.patch_vllm_graph_capture()
    assert v1().capture_model() == "v1"
    assert calls == ["hold", "capture_v1", "release_cache"]
    calls.clear()
    # Model Runner V2 (vLLM's CUDA default) captured with the bare cuBLAS workspace before.
    assert v2().capture_model() == "v2"
    assert calls == ["hold", "capture_v2", "release_cache"]


class _Pool:
    id = (7, 1)


class _Stream:
    def __init__(self, ptr):
        self.cuda_stream = ptr
        self.device = torch.device("cuda", 0)


def _fake_cuda(monkeypatch):
    events = []
    state = {"pool": None, "stream": _Stream(100)}

    @contextlib.contextmanager
    def use_mem_pool(pool, device = None):
        state["pool"] = pool
        try:
            yield
        finally:
            state["pool"] = None

    @contextlib.contextmanager
    def on_stream(stream):
        previous, state["stream"] = state["stream"], stream
        try:
            yield
        finally:
            state["stream"] = previous

    def gemm(name):
        def run(*args, **kwargs):
            # The workspace is fetched (and allocated when missing) by the first cuBLAS / cuBLASLt call.
            events.append((name, state["pool"], state["stream"].cuda_stream))
        return run

    class CUDAGraph:
        def capture_begin(self, *args, **kwargs):
            events.append(("capture_begin", state["stream"].cuda_stream))

    # Free blocks of the pool: (address, bytes, stream); the allocator hands a block back from its start.
    blocks = [[1000, 4096, 200], [5096, 512, 200], [20000, 3 << 20, 200]]

    def empty(size, **kwargs):
        block = next(b for b in blocks if b[1] >= size and b[2] == state["stream"].cuda_stream)
        address = block[0]
        block[0] += size
        block[1] -= size
        events.append(("hold", size, state["pool"], state["stream"].cuda_stream))
        return types.SimpleNamespace(data_ptr = lambda: address, size = size)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: state["stream"])
    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda ptr, device = None: _Stream(ptr), raising = False)
    monkeypatch.setattr(torch.cuda, "MemPool", _Pool, raising = False)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", use_mem_pool, raising = False)
    monkeypatch.setattr(torch.cuda, "graphs", types.SimpleNamespace(CUDAGraph = CUDAGraph), raising = False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: events.append("device_sync"))
    monkeypatch.setattr(torch.cuda, "device", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: events.append("clear"), raising = False)
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda: {"segments": [
        {"device": 0, "address": 1000, "stream": 200, "segment_pool_id": (7, 1), "blocks": [
            {"size": 4096, "state": "inactive"}, {"size": 512, "state": "inactive"}, {"size": 64, "state": "active_allocated"}]},
        {"device": 0, "address": 9000, "stream": 200, "segment_pool_id": (0, 0), "blocks": [{"size": 8192, "state": "inactive"}]},
        # A small segment (the 1 MiB cuBLASLt workspace): only requests up to 1 MiB are served from it.
        {"device": 0, "address": 20000, "stream": 200, "segment_pool_id": (7, 1), "segment_type": "small",
         "blocks": [{"size": 3 << 20, "state": "inactive"}]},
    ]})
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "mm", gemm("mm"))
    monkeypatch.setattr(torch, "addmm", gemm("addmm"))
    monkeypatch.setattr(vllm_utils, "_VLLM_CUBLAS_WORKSPACES", {})
    return CUDAGraph, on_stream, events


def test_capture_stream_workspace_comes_from_a_pool_and_is_held(monkeypatch):
    CUDAGraph, on_stream, events = _fake_cuda(monkeypatch)
    original = CUDAGraph.capture_begin
    with vllm_utils._vllm_cublas_workspace_held() as ok:
        assert ok is True
        pool = vllm_utils._VLLM_CUBLAS_WORKSPACES[0][0]
        # vLLM's graph_capture() captures on a fresh stream (200), not the stream it replays on (100).
        with on_stream(_Stream(200)):
            CUDAGraph().capture_begin()
            CUDAGraph().capture_begin()
    # Only the capture stream's first graph drops its eager workspace and fetches both workspaces from the pool.
    assert events[:4] == ["clear", ("mm", pool, 200), ("addmm", pool, 200), ("capture_begin", 200)]
    assert events[4] == ("capture_begin", 200)
    # After the capture: out of cuBLAS's cache, and every free block of the pool held on the stream it belongs to.
    mib = 1 << 20
    assert events[5:] == ["device_sync", "clear", ("hold", mib, pool, 200), ("hold", mib, pool, 200), ("hold", mib, pool, 200),
                          ("hold", 4096, pool, 200), ("hold", 512, pool, 200)]
    held = vllm_utils._VLLM_CUBLAS_WORKSPACES[0][1]
    assert [t.size for t in held] == [mib, mib, mib, 4096, 512]
    # Every piece sits where the freed block was.
    assert [t.data_ptr() for t in held] == [20000, 20000 + mib, 20000 + 2 * mib, 1000, 5096]
    assert CUDAGraph.capture_begin is original

    # A later capture (vLLM profiles, then captures) reuses the pool: the old hold goes back first.
    events.clear()
    with vllm_utils._vllm_cublas_workspace_held():
        assert vllm_utils._VLLM_CUBLAS_WORKSPACES[0] == [pool, []]
        with on_stream(_Stream(200)):
            CUDAGraph().capture_begin()
    assert events[:3] == ["clear", ("mm", pool, 200), ("addmm", pool, 200)]


def test_no_capture_no_clear(monkeypatch):
    CUDAGraph, on_stream, events = _fake_cuda(monkeypatch)
    with vllm_utils._vllm_cublas_workspace_held():
        pass
    assert events == []


def test_capture_error_still_holds_and_restores(monkeypatch):
    CUDAGraph, on_stream, events = _fake_cuda(monkeypatch)
    original = CUDAGraph.capture_begin
    with pytest.raises(RuntimeError):
        with vllm_utils._vllm_cublas_workspace_held():
            with on_stream(_Stream(200)):
                CUDAGraph().capture_begin()
            raise RuntimeError("capture failed")
    assert CUDAGraph.capture_begin is original
    assert len(vllm_utils._VLLM_CUBLAS_WORKSPACES[0][1]) == 5


def test_no_mem_pool_api_is_a_no_op(monkeypatch):
    CUDAGraph, on_stream, events = _fake_cuda(monkeypatch)
    monkeypatch.delattr(torch.cuda, "MemPool")
    original = CUDAGraph.capture_begin
    with vllm_utils._vllm_cublas_workspace_held() as ok:
        assert ok is False
        assert CUDAGraph.capture_begin is original
    assert events == []


def _graph_writes_freed_workspace(pin):
    """Like vLLM: capture a decode-shaped GEMM on a fresh stream, replay on another; clear the cuBLAS workspaces, refill
    the freed memory with a sentinel, replay: True when the replay overwrote the sentinel."""
    device = torch.device("cuda", torch.cuda.current_device())
    replay_stream = torch.cuda.Stream(device = device)
    size = max(int(torch._C._cuda_getCublasWorkspaceSize()), 1 << 20)
    x = torch.randn(16, 3072, device = device, dtype = torch.bfloat16)
    w = torch.randn(1024, 3072, device = device, dtype = torch.bfloat16)
    torch.cuda.synchronize()
    hold = vllm_utils._vllm_cublas_workspace_held() if pin else contextlib.nullcontext(True)
    with hold as ok:
        assert ok
        capture_stream = torch.cuda.Stream(device = device)
        with torch.cuda.stream(capture_stream):
            # Decode-shaped (M = 16) GEMMs: cuBLAS splits K over its workspace for these on A100-class GPUs.
            y = torch.nn.functional.linear(x, w)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream = capture_stream):
                y = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
    with torch.cuda.stream(capture_stream):
        sentinels = []
    with torch.cuda.stream(replay_stream):
        graph.replay()
        replay_stream.synchronize()
        reference = y.clone()
    torch._C._cuda_clearCublasWorkspaces()
    with torch.cuda.stream(capture_stream):
        # The freed workspace goes back to the capture stream's free list: the next same-size block there reuses it.
        sentinels = [torch.full((size // 2,), 0x2B2B, dtype = torch.int16, device = device) for _ in range(4)]
    torch.cuda.synchronize()
    with torch.cuda.stream(replay_stream):
        graph.replay()
        replay_stream.synchronize()
        overwritten = any(bool((s != 0x2B2B).any()) for s in sentinels)
        assert torch.equal(y, reference)
    del graph
    return overwritten


@pytest.mark.skipif(not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None,
                    reason = "needs CUDA cuBLAS and CUDA graphs")
def test_pinned_workspace_survives_a_clear():
    if getattr(torch.cuda, "MemPool", None) is None or getattr(torch.cuda, "use_mem_pool", None) is None:
        pytest.skip("this torch has no torch.cuda.MemPool")
    hazard = _graph_writes_freed_workspace(pin = False)
    assert not _graph_writes_freed_workspace(pin = True)
    if not hazard:
        pytest.skip("this GPU's decode GEMM used no cuBLAS workspace, so the unpinned hazard could not be shown")
