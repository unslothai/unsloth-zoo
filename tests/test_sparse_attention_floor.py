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

"""Sparse attention dispatch across Triton / torch versions and under torch.compile: gather is only selected where
its Triton kernels import, and traced dispatch does not recompile on lazily-filled state."""

import os
import sys

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo import sparse_attention as sa  # noqa: E402

_VENDORED_TRITON = ("unsloth_zoo._vendored.attn_gym._backends.triton",
                    "unsloth_zoo._vendored.attn_gym.sparse.gather_attn.impl.triton")


def test_gather_needs_its_triton_kernels_not_only_the_api(monkeypatch):
    # torch 2.4 - 2.8 bundle Triton 3.0 - 3.4, which lack triton.constexpr_function: the api and op registrations
    # import there, the kernels (loaded on the first call) do not, so gather must not be selected.
    triton = pytest.importorskip("triton")
    monkeypatch.setattr(sa, "_GATHER", None)
    if hasattr(triton, "constexpr_function"):
        assert sa._gather_importable()   # api + ops cached; their op definitions never run twice below
        monkeypatch.delattr(triton, "constexpr_function")
        for name in [m for m in sys.modules if m.startswith(_VENDORED_TRITON)]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sa, "_GATHER", None)
    assert sa._gather_importable() is False


_COLD = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
case = sys.argv[2]
torch._dynamo.config.error_on_recompile = True
if hasattr(torch.fx.experimental._config, "use_duck_shape"):   # torch >= 2.5
    torch.fx.experimental._config.use_duck_shape = False
if case != "import_in_trace":
    from unsloth_zoo import sparse_attention as sa
dev, dt, h, d = "cuda", torch.bfloat16, 16, 512

def index_default(q, kv, sink, idx, ckv, cidx):
    return sa.sparse_attn(q, kv, sink, idx, d ** -0.5)

def window_triton(q, kv, sink, idx, ckv, cidx):
    return sa.sparse_attn_window(q, kv, ckv, cidx, sink, d ** -0.5, 64, backend = "triton")

def import_in_trace(q, kv, sink, idx, ckv, cidx):
    from unsloth_zoo import sparse_attention as sa
    return sa.sparse_attn_window(q, kv, ckv, cidx[..., :0], sink, d ** -0.5, 64)

fn = torch.compile({"index_default": index_default, "window_triton": window_triton,
                    "import_in_trace": import_in_trace}[case], backend = "eager", fullgraph = True, dynamic = True)
# The guards under test are Dynamo's own: the eager backend sees the same recompile without Inductor compiles.
for s in (64, 80):
    g = torch.Generator(device = dev).manual_seed(s)
    r = lambda *shape: (torch.randn(*shape, device = dev, generator = g) * 0.3).to(dt).requires_grad_()
    q, kv, ckv = r(1, s, h, d), r(1, s, d), r(1, s // 4 + 3, d)
    sink = torch.randn(h, device = dev, generator = g).requires_grad_()
    idx = torch.randint(-1, s, (1, s, 64), device = dev, dtype = torch.int32, generator = g)
    cidx = torch.randint(-1, s // 4 + 3, (1, s, 32), device = dev, dtype = torch.int32, generator = g)
    fn(q, kv, sink, idx, ckv, cidx).float().square().mean().backward()
from unsloth_zoo import sparse_mqa_kernel as sk
print("COLD_OK", sk.SPARSE_MQA_STATS["fwd"] + sk.SPARSE_MQA_STATS["window_fwd"])
"""


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8, reason = "bf16 CUDA kernels")
@pytest.mark.parametrize("case", ["index_default", "window_triton", "import_in_trace"])
def test_cold_compiled_dispatch_does_not_recompile_across_lengths(case):
    # A fresh process whose first sparse call is compiled (dynamic shapes, two lengths): dispatch state filled during
    # the first trace (shared-memory caches, the gather import) must not become a guard that recompiles the second.
    import subprocess
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-c", _COLD, root, case], capture_output = True, text = True, timeout = 900,
                          env = dict(os.environ, PYTHONPATH = root + os.pathsep + os.environ.get("PYTHONPATH", ""),
                                     OMP_NUM_THREADS = "4"))   # tiny host tensors: a full-width pool only spins
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("COLD_OK")), None)
    assert line is not None, proc.stderr[-3000:]
    if case == "index_default" and torch.cuda.get_device_capability()[0] in (10, 11):
        # Blackwell picks gather first; a cold trace used to skip it (the import had not run yet) and take Triton.
        assert int(line.split()[1]) == 0, line


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8, reason = "bf16 CUDA kernels")
def test_triton_kernels_compile_once_across_lengths():
    # 8 heads (QSA's folded groups): the lse batch stride s * 8 is 16-aligned for even s only, and Triton specializes
    # integer arguments on % 16; every kernel must keep one compiled variant across even / odd lengths and batch.
    import triton
    from unsloth_zoo import sparse_mqa_kernel as sk
    names = ("_sparse_mqa_fwd", "_sparse_mqa_bwd", "_window_fwd", "_window_bwd_dq", "_window_bwd_dkv")

    def jit(k):
        while not isinstance(k, triton.runtime.jit.JITFunction):
            k = k.fn
        return k

    def caches(n):   # Triton >= 3.3: device_caches {device: (kernels, ...)}; older: cache {device: kernels}
        k = jit(getattr(sk, n))
        return k.device_caches if hasattr(k, "device_caches") else k.cache

    def variants():
        return {n: sum(len(c[0] if isinstance(c, tuple) else c) for c in caches(n).values()) for n in names}

    for n in names:   # earlier tests may have compiled the same signatures
        caches(n).clear()
    before = variants()
    h, d, dt = 8, 64, torch.bfloat16
    for s, b in ((256, 1), (301, 1), (256, 2), (77, 2)):
        g = torch.Generator(device = "cuda").manual_seed(s)
        r = lambda *shape: (torch.randn(*shape, device = "cuda", generator = g) * 0.3).to(dt).requires_grad_()
        q, kv, wkv, ckv = r(b, s, h, d), r(b, s, d), r(b, s, d), r(b, s // 4 + 3, d)
        sink = torch.randn(h, device = "cuda", generator = g).requires_grad_()
        idx = torch.randint(-1, s, (b, s, 40), device = "cuda", dtype = torch.int32, generator = g)
        cidx = torch.randint(-1, s // 4 + 3, (b, s, 24), device = "cuda", dtype = torch.int32, generator = g)
        sk.sparse_mqa_attention(q, kv, sink, idx, d ** -0.5).float().square().mean().backward()
        sk.sparse_mqa_window_attention(q, wkv, ckv, cidx, sink, d ** -0.5, 32).float().square().mean().backward()
    grown = {n: variants()[n] - before[n] for n in names}
    # index fwd: with the sink, and without it for the window's compressed part; window fwd / dq: one with prior.
    assert grown == {"_sparse_mqa_fwd": 2, "_sparse_mqa_bwd": 1, "_window_fwd": 1, "_window_bwd_dq": 1,
                     "_window_bwd_dkv": 1}, grown


@pytest.mark.parametrize("form", ["index", "window"])
def test_dispatch_traces_fullgraph_with_env_reads(monkeypatch, form):
    # The backend override is an environment variable read during dispatch; torch 2.4 cannot inline os.environ.get
    # under fullgraph. The CPU path (torch reference) traces the same dispatch code.
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    torch._dynamo.reset()
    q, kv = torch.randn(1, 9, 2, 16), torch.randn(1, 11, 16)
    sink, idx = torch.randn(2), torch.randint(-1, 11, (1, 9, 4)).int()
    wkv = torch.randn(1, 9, 16)
    if form == "index":
        fn = lambda q, kv, sink, idx: sa.sparse_attn(q, kv, sink, idx, 0.25)
    else:
        fn = lambda q, kv, sink, idx: sa.sparse_attn_window(q, wkv, kv, idx, sink, 0.25, 4)
    got = torch.compile(fn, backend = "eager", fullgraph = True)(q, kv, sink, idx)
    torch.testing.assert_close(got, fn(q, kv, sink, idx))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8, reason = "bf16 CUDA kernels")
def test_window_support_check_traces_with_symbolic_head_dim():
    # dynamic=True makes the head dim a SymInt; torch 2.4 SymInts have no bitwise and, so a power-of-two test
    # written d & (d - 1) failed every compiled window call there (TorchRuntimeError, Colab L4 torch 2.4.0).
    from unsloth_zoo import sparse_mqa_kernel as sk
    torch._dynamo.reset()
    fn = torch.compile(lambda q, kv: q.sum() if sk.sparse_mqa_window_supported(q, kv) else q.sum() * 0,
                       backend = "eager", fullgraph = True, dynamic = True)
    for d in (64, 128):
        q = torch.randn(1, 8, 16, d, device = "cuda", dtype = torch.bfloat16)
        kv = torch.randn(1, 8, d, device = "cuda", dtype = torch.bfloat16)
        assert fn(q, kv).item() == pytest.approx(q.sum().item(), rel = 1e-2)
    q = torch.randn(1, 8, 16, 48, device = "cuda", dtype = torch.bfloat16)   # not a power of two
    assert fn(q, torch.randn(1, 8, 48, device = "cuda", dtype = torch.bfloat16)).item() == 0


def test_flex_is_never_eligible_without_flex_attention(monkeypatch):
    # torch 2.4 has no torch.nn.attention.flex_attention: QSA tried flex first, failed, warned and latched it per
    # shape class instead of going straight to the kernel fold.
    import importlib.util
    assert sa._HAS_FLEX == (importlib.util.find_spec("torch.nn.attention.flex_attention") is not None)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("flex eligibility needs sm80+")
    q = torch.empty(1, 256, 8, 128, device = "cuda", dtype = torch.bfloat16)
    if sa._HAS_FLEX:
        assert sa._flex_eligible(q)
        monkeypatch.setattr(sa, "_HAS_FLEX", False)
    assert not sa._flex_eligible(q)


def _backend_compiler_failed(inner):
    from torch._dynamo.exc import BackendCompilerFailed
    try:
        return BackendCompilerFailed(lambda *args: None, inner, None)   # torch >= 2.7: + first_useful_frame
    except TypeError:
        return BackendCompilerFailed(lambda *args: None, inner)


def test_a_backend_compile_failure_is_latched_not_raised(monkeypatch):
    # Eager QSA tries flex first and compiles it; on torch 2.6 (Colab L4) Inductor could not lower flex at the QSA
    # shape and raised BackendCompilerFailed, which _latch re-raised as a trace-time error: the eager step crashed
    # instead of falling back to the kernel fold. Real trace-time and memory failures still propagate.
    import torch._dynamo.exc
    monkeypatch.setattr(sa, "_FAILED", {})
    q = torch.randn(1, 4, 2, 16)
    sa._latch("flex", q, 4, _backend_compiler_failed(RuntimeError("No valid triton configs")))
    assert sa._failure_key("flex", q, 4) in sa._FAILED
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed):
        sa._latch("flex", q, 5, _backend_compiler_failed(torch.OutOfMemoryError("out of memory")))
    with pytest.raises(torch._dynamo.exc.Unsupported):
        sa._latch("flex", q, 6, torch._dynamo.exc.Unsupported("trace-time"))
    assert list(sa._FAILED) == [sa._failure_key("flex", q, 4)]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
                    reason = "gather runs on datacenter Blackwell only")
@pytest.mark.parametrize("form", ["index", "window"])
def test_gather_compiles_one_variant_set_per_width_bucket(form):
    # DeepSeek's selection width is min(index_topk, s // ratio): it changes with the length below the cap. The vendored
    # kernels take TOPK as a constexpr (and a tuner key), so every width compiled and tuned a new variant set.
    import triton
    from unsloth_zoo._vendored.attn_gym.sparse.gather_attn.impl.triton import forward

    def variants():
        k = forward._gather_attn_fwd_shared
        while not isinstance(k, triton.runtime.jit.JITFunction):
            k = k.fn
        caches = k.device_caches if hasattr(k, "device_caches") else k.cache
        return sum(len(c[0] if isinstance(c, tuple) else c) for c in caches.values())

    h, d, s = 16, 64, 96
    grown = []
    for width in (40, 50, 61):   # one power-of-two bucket (64)
        g = torch.Generator(device = "cuda").manual_seed(width)
        r = lambda *shape: (torch.randn(*shape, device = "cuda", generator = g) * 0.3).to(torch.bfloat16).requires_grad_()
        q, kv, ckv = r(1, s, h, d), r(1, s, d), r(1, 70, d)
        sink = torch.randn(h, device = "cuda", generator = g).requires_grad_()
        before = variants()
        if form == "index":
            idx = torch.randint(-1, s, (1, s, width), device = "cuda", dtype = torch.int32, generator = g)
            fn = lambda q, kv, ckv, sink: sa.sparse_attn(q, kv, sink, idx, d ** -0.5, backend = "gather")
            ref = lambda q, kv, ckv, sink: sa.sparse_attn(q, kv, sink, idx, d ** -0.5, backend = "torch")
        else:
            idx = torch.randint(-1, 70, (1, s, width), device = "cuda", dtype = torch.int32, generator = g)
            fn = lambda q, kv, ckv, sink: sa.sparse_attn_window(q, kv, ckv, idx, sink, d ** -0.5, 32, backend = "gather")
            ref = lambda q, kv, ckv, sink: sa.sparse_attn_window(q, kv, ckv, idx, sink, d ** -0.5, 32, backend = "torch")
        go = torch.randn(1, s, h, d, device = "cuda", dtype = torch.bfloat16)

        def run(f, *xs):
            xs = [x.detach().clone().requires_grad_() for x in xs]
            out = f(*xs)
            return [out, *torch.autograd.grad(out, (xs[0], xs[1], xs[3]), go.to(out.dtype))]

        got = run(fn, q, kv, ckv, sink)
        exact = run(ref, q.double(), kv.double(), ckv.double(), sink.double())
        low = run(ref, q, kv, ckv, sink)
        # Within twice the same-dtype torch path's error vs fp64, plus 4 bf16 ulp (as test_cuda_backend_matches_fp64).
        for x, r, lo in zip(got, exact, low):
            err, base = (x.double() - r).abs().max().item(), (lo.double() - r).abs().max().item()
            assert err <= 2 * base + 4 * torch.finfo(torch.bfloat16).eps * max(1.0, r.abs().max().item()), (width, err, base)
        grown.append(variants() - before)
    assert grown[1:] == [0, 0], grown
