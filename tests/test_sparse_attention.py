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

"""sparse_attn must match deepseek-ai's reference sparse_attn (K = V = kv, sink only in the denominator,
-1 = unused slot, empty row -> 0) forward and backward on every backend."""

import math
import os

import pytest
import torch

from unsloth_zoo import sparse_attention as sa


def _naive(q, kv, sink, idxs, scale):
    """Row-by-row fp64 transcription of the reference TileLang kernel's math."""
    b, s, h, d = q.shape
    out = torch.zeros(b, s, h, d, dtype = torch.float64)
    for bi in range(b):
        for t in range(s):
            rows = [int(i) for i in idxs[bi, t] if 0 <= int(i) < kv.shape[1]]
            for hi in range(h):
                if not rows:
                    continue
                logits = [float(q[bi, t, hi].double() @ kv[bi, r].double()) * scale for r in rows]
                m = max(logits)
                weights = [math.exp(x - m) for x in logits]
                denom = sum(weights) + math.exp(float(sink[hi]) - m)
                acc = sum(w * kv[bi, r].double() for w, r in zip(weights, rows))
                out[bi, t, hi] = acc / denom
    return out


def _unique_idxs(b, s, k, n, pad_frac, generator, empty_rows = ()):
    idxs = torch.stack([torch.stack([torch.randperm(n, generator = generator)[:k] for _ in range(s)]) for _ in range(b)])
    pad = torch.rand(b, s, k, generator = generator) < pad_frac
    idxs = torch.where(pad, -1, idxs)
    for bi, t in empty_rows:
        idxs[bi, t] = -1
    return idxs.to(torch.int32)


def _inputs(b, s, h, d, n, k, dtype, device, seed = 0, empty_rows = ()):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(b, s, h, d, generator = g) * 0.5).to(device, dtype)
    kv = (torch.randn(b, n, d, generator = g) * 0.5).to(device, dtype)
    sink = (torch.randn(h, generator = g)).to(device, torch.float32)
    idxs = _unique_idxs(b, s, k, n, 0.2, g, empty_rows).to(device)
    return q, kv, sink, idxs


def test_reference_matches_naive_row_math():
    q, kv, sink, idxs = _inputs(2, 5, 3, 8, 7, 4, torch.float64, "cpu", empty_rows = [(1, 2)])
    scale = 8 ** -0.5
    out = sa.sparse_attn_reference(q, kv, sink, idxs, scale)
    torch.testing.assert_close(out, _naive(q, kv, sink, idxs, scale), rtol = 1e-12, atol = 1e-12)
    assert out[1, 2].abs().max() == 0


def test_reference_matches_vendored_attn_gym_reference():
    from unsloth_zoo._vendored.attn_gym.sparse.gather_attn import gather_attn
    q, kv, sink, idxs = _inputs(2, 9, 4, 16, 11, 5, torch.float64, "cpu")
    ours = sa.sparse_attn_reference(q, kv, sink, idxs, 0.3)
    theirs = gather_attn(q.transpose(1, 2), None, kv.unsqueeze(1), idxs, sink.double(), sliding_window_size = 0,
                         impl = "reference", scale = 0.3).transpose(1, 2)
    torch.testing.assert_close(ours, theirs, rtol = 1e-12, atol = 1e-12)


def test_window_form_equals_concatenated_indices():
    g = torch.Generator().manual_seed(1)
    b, s, h, d, nc, k, w = 2, 13, 2, 8, 6, 3, 4
    q = torch.randn(b, s, h, d, generator = g, dtype = torch.float64)
    win = torch.randn(b, s, d, generator = g, dtype = torch.float64)
    comp = torch.randn(b, nc, d, generator = g, dtype = torch.float64)
    sink = torch.randn(h, generator = g, dtype = torch.float64)
    cidx = _unique_idxs(b, s, k, nc, 0.3, g)
    out = sa.sparse_attn_window(q, win, comp, cidx, sink, 0.4, w)
    # deepseek-ai get_window_topk_idxs + offset by seq_len, then the plain kernel.
    end = torch.arange(s).unsqueeze(1)
    widx = (end - w + 1).clamp(0) + torch.arange(min(s, w))
    widx = torch.where(widx > end, -1, widx).int().unsqueeze(0).expand(b, -1, -1)
    full = torch.cat([widx, torch.where(cidx >= 0, cidx + s, -1).int()], -1)
    ref = _naive(q, torch.cat([win, comp], 1), sink, full, 0.4)
    torch.testing.assert_close(out, ref, rtol = 1e-12, atol = 1e-12)


def test_reference_grads_match_fp64_autograd_of_naive_formula():
    q, kv, sink, idxs = _inputs(1, 6, 2, 8, 9, 4, torch.float64, "cpu", empty_rows = [(0, 3)])
    q.requires_grad_(); kv.requires_grad_(); sink = sink.double().requires_grad_()
    scale = 0.35

    def dense(q, kv, sink):
        logits = torch.einsum("bshd,bnd->bshn", q, kv) * scale
        member = torch.zeros(1, 6, 9, dtype = torch.bool)
        for t in range(6):
            for i in idxs[0, t]:
                if i >= 0:
                    member[0, t, int(i)] = True
        logits = logits.masked_fill(~member.unsqueeze(2), float("-inf"))
        full = torch.cat([logits, sink.view(1, 1, -1, 1).expand(1, 6, -1, 1)], -1)
        p = torch.softmax(full, -1)[..., :-1]
        p = torch.where(member.any(-1).view(1, 6, 1, 1), p, 0)
        return torch.einsum("bshn,bnd->bshd", p, kv)

    go = torch.randn(1, 6, 2, 8, dtype = torch.float64)
    a = torch.autograd.grad(sa.sparse_attn_reference(q, kv, sink, idxs, scale), (q, kv, sink), go)
    r = torch.autograd.grad(dense(q, kv, sink), (q, kv, sink), go)
    for x, y in zip(a, r):
        torch.testing.assert_close(x, y, rtol = 1e-10, atol = 1e-10)


def test_forced_torch_backend_and_stats(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    q, kv, sink, idxs = _inputs(1, 4, 2, 8, 5, 3, torch.float32, "cpu")
    before = sa.SPARSE_ATTN_STATS["torch"]
    sa.sparse_attn(q, kv, sink, idxs, 0.5)
    assert sa.SPARSE_ATTN_STATS["torch"] == before + 1


_CUDA = torch.cuda.is_available()


def _cuda_dtypes():
    if not _CUDA:
        return []
    dtypes = [torch.float16]
    if torch.cuda.get_device_capability()[0] >= 8:
        dtypes.append(torch.bfloat16)
    return dtypes


def _grads(fn, q, kv, sink, go):
    q, kv, sink = (t.detach().clone().requires_grad_() for t in (q, kv, sink))
    out = fn(q, kv, sink)
    return (out, *torch.autograd.grad(out, (q, kv, sink), go))


_GATHER_SHAPES = [(2, 64, 16, 512, 96, 24), (1, 40, 4, 128, 300, 64), (1, 130, 8, 128, 140, 33)]
_FLEX_SHAPES = _GATHER_SHAPES + [(1, 130, 8, 256, 140, 33)]


def _gather_runs():
    # Only the shared-KV schedule is used in production: capability 10.x / 11.x and bf16.
    return _CUDA and torch.cuda.get_device_capability()[0] in (10, 11)


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("backend,dtype,shape", [(be, dt, sh) for be in ("flex", "triton") for dt in _cuda_dtypes()
                                                  for sh in _FLEX_SHAPES]
                         + ([("gather", torch.bfloat16, sh) for sh in _GATHER_SHAPES] if _gather_runs() else []))
def test_cuda_backend_matches_fp64(monkeypatch, backend, dtype, shape):
    b, s, h, d, n, k = shape
    if backend == "flex" and not sa._flex_eligible(torch.empty(1, 1, h, d, device = "cuda", dtype = dtype)):
        pytest.skip(f"flex at head_dim {d} is gated off here: {sa._shared_memory(torch.device('cuda'))} B shared memory "
                    f"per block < {sa._FLEX_D512_MIN_SHARED_MEMORY} B (test_flex_gate_follows_shared_memory checks the gate)")
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", backend)
    q, kv, sink, idxs = _inputs(b, s, h, d, n, k, dtype, "cuda", empty_rows = [(0, 0), (b - 1, s - 1)])
    scale = d ** -0.5
    go = torch.randn(b, s, h, d, device = "cuda", dtype = dtype)
    before = dict(sa.SPARSE_ATTN_STATS)
    got = _grads(lambda q, kv, sink: sa.sparse_attn(q, kv, sink, idxs, scale), q, kv, sink, go)
    assert sa.SPARSE_ATTN_STATS[backend] == before[backend] + 1 and sa.SPARSE_ATTN_STATS["fallback"] == before["fallback"]
    ref = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, scale),
                 q.double(), kv.double(), sink.double(), go.double())
    low = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, scale), q, kv, sink, go)
    for name, x, r, lo in zip(("out", "dq", "dkv", "dsink"), got, ref, low):
        err = (x.double() - r).abs().max().item()
        base = (lo.double() - r).abs().max().item()
        # Same-dtype torch reference error, with 4 ulp headroom of the output dtype for the fused order.
        bound = 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item())
        assert err <= bound, f"{backend} {name}: err {err:.3e} > bound {bound:.3e} (torch-same-dtype {base:.3e})"
    assert got[0][0, 0].abs().max() == 0 and got[0][b - 1, s - 1].abs().max() == 0


def _window_cases():
    cases = [("triton", dt) for dt in _cuda_dtypes()]
    return cases + ([("gather", torch.bfloat16)] if _gather_runs() else [])


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("backend,dtype", _window_cases())
@pytest.mark.parametrize("s", [40, 300])
@pytest.mark.parametrize("k", [32, 0])
def test_cuda_window_form_matches_concatenated(monkeypatch, backend, dtype, s, k):
    # Forward and backward of the window form (DeepSeek window-only and compressed layers, head_dim 512, 16 heads)
    # on the backend this device dispatches to: gather on sm100+, the Triton kernel elsewhere (T4 / L4 / A100).
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", backend)
    g = torch.Generator().manual_seed(2)
    from unsloth_zoo import sparse_mqa_kernel as mk
    b, h, d, nc, w = 1, 16, 512, 150, 128
    q = (torch.randn(b, s, h, d, generator = g) * 0.5).cuda().to(dtype)
    win = (torch.randn(b, s, d, generator = g) * 0.5).cuda().to(dtype)
    comp = (torch.randn(b, nc, d, generator = g) * 0.5).cuda().to(dtype)
    sink = torch.randn(h, generator = g).cuda()
    cidx = _unique_idxs(b, s, k, nc, 0.2, g).cuda() if k else torch.empty(b, s, 0, dtype = torch.int32, device = "cuda")
    go = torch.randn(b, s, h, d, generator = g).cuda().to(dtype)
    banded = backend == "triton" and mk.sparse_mqa_window_supported(q, win)
    window_calls = dict(mk.SPARSE_MQA_STATS)
    end = torch.arange(s, device = "cuda").unsqueeze(1)
    widx = (end - w + 1).clamp(0) + torch.arange(min(s, w), device = "cuda")
    widx = torch.where(widx > end, -1, widx).int().unsqueeze(0)
    full = torch.cat([widx, torch.where(cidx >= 0, cidx + s, -1).int()], -1)

    def run(fn, *xs):
        xs = [x.detach().clone().requires_grad_() for x in xs]
        out = fn(*xs)
        return [out, *torch.autograd.grad(out, xs, go.to(out.dtype))]

    before = dict(sa.SPARSE_ATTN_STATS)
    got = run(lambda q, win, comp, sink: sa.sparse_attn_window(q, win, comp, cidx, sink, d ** -0.5, w), q, win, comp, sink)
    assert sa.SPARSE_ATTN_STATS[backend] == before[backend] + 1 and sa.SPARSE_ATTN_STATS["fallback"] == before["fallback"]
    if banded:   # the banded window kernel ran (forward and backward), not the index path over a concatenation
        assert mk.SPARSE_MQA_STATS["window_fwd"] == window_calls["window_fwd"] + 1
        assert mk.SPARSE_MQA_STATS["window_bwd"] == window_calls["window_bwd"] + 1
    def ref_fn(q, win, comp, sink):
        return sa.sparse_attn_reference(q, torch.cat([win, comp], 1), sink, full, d ** -0.5)
    ref = run(ref_fn, q.double(), win.double(), comp.double(), sink.double())
    low = run(ref_fn, q, win, comp, sink)
    for name, x, r, lo in zip(("out", "dq", "dwin", "dcomp", "dsink"), got, ref, low):
        err, base = (x.double() - r).abs().max().item(), (lo.double() - r).abs().max().item()
        bound = 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item())
        assert err <= bound, f"{backend} {name}: err {err:.3e} > bound {bound:.3e} (torch-same-dtype {base:.3e})"


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("dtype", _cuda_dtypes())
@pytest.mark.parametrize("d", [512, 256])
def test_cuda_triton_kernel_on_a_64kb_shared_memory_tier(monkeypatch, dtype, d):
    # The configs a T4 (64 KB per block) gets, run here: head_dim 512 backward splits D in two programs.
    from unsloth_zoo import sparse_mqa_kernel as mk
    monkeypatch.setattr(mk, "_shared_memory", lambda device: 64 * 1024)
    monkeypatch.setattr(mk, "_CHOSEN", {})
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "triton")
    b, s, h, n, k = 1, 70, 16, 150, 160
    q, kv, sink, idxs = _inputs(b, s, h, d, n, min(k, n), dtype, "cuda", empty_rows = [(0, 3)])
    go = torch.randn(b, s, h, d, device = "cuda", dtype = dtype)
    got = _grads(lambda q, kv, sink: sa.sparse_attn(q, kv, sink, idxs, d ** -0.5), q, kv, sink, go)
    chosen = {key[3]: mk._configs(None, d, key[3])[i] for key, i in mk._CHOSEN.items() if key[0] != "window"}
    if d == 512:
        assert chosen[True][3] == d // 2, chosen   # backward split
    ref = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, d ** -0.5),
                 q.double(), kv.double(), sink.double(), go.double())
    low = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, d ** -0.5), q, kv, sink, go)
    for name, x, r, lo in zip(("out", "dq", "dkv", "dsink"), got, ref, low):
        err, base = (x.double() - r).abs().max().item(), (lo.double() - r).abs().max().item()
        bound = 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item())
        assert err <= bound, f"{name}: err {err:.3e} > bound {bound:.3e}"


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
def test_cuda_refused_kernel_config_steps_down(monkeypatch):
    # A part whose limit is below what the first config needs: the refusal moves to the next (split) config, the
    # result is unchanged, and the choice is remembered (one refusal, not one per call).
    import triton
    from unsloth_zoo import sparse_mqa_kernel as mk
    monkeypatch.setattr(mk, "_CHOSEN", {})
    monkeypatch.setattr(mk, "_shared_memory", lambda device: 228 * 1024)
    real, refused = mk._sparse_mqa_bwd, []

    class Refusing:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                if kw["SPLIT_D"] == kw["D"]:
                    refused.append(grid)
                    raise triton.runtime.errors.OutOfResources(98304, 65536, "shared memory")
                return real[grid](*args, **kw)
            return launch

    q, kv, sink, idxs = _inputs(1, 33, 16, 512, 80, 40, torch.float16, "cuda")
    go = torch.randn(1, 33, 16, 512, device = "cuda", dtype = torch.float16)
    want = _grads(lambda q, kv, sink: mk.sparse_mqa_attention(q, kv, sink, idxs, 512 ** -0.5), q, kv, sink, go)
    mk._CHOSEN.clear()   # a real 64 KB part already stepped down while computing `want`
    monkeypatch.setattr(mk, "_sparse_mqa_bwd", Refusing())
    for _ in range(2):
        got = _grads(lambda q, kv, sink: mk.sparse_mqa_attention(q, kv, sink, idxs, 512 ** -0.5), q, kv, sink, go)
        for x, y in zip(got, want):
            torch.testing.assert_close(x.float(), y.float(), atol = 2e-3, rtol = 2e-2)
    assert len(refused) == 1


def test_gather_eligibility_is_the_shared_schedule_only():
    def q(d, h, dt = torch.bfloat16):
        return torch.empty(1, 1, h, d, dtype = dt, device = "meta")
    # meta tensors are not CUDA: never eligible; the shape rules are checked through a CUDA stand-in below.
    assert not sa._gather_eligible(q(512, 16))
    if _gather_runs():
        def cuda(d, h, dt = torch.bfloat16):
            return torch.empty(1, 1, h, d, dtype = dt, device = "cuda")
        assert sa._gather_eligible(cuda(512, 16)) and sa._gather_eligible(cuda(128, 4))
        assert not sa._gather_eligible(cuda(256, 16)) and not sa._gather_eligible(cuda(512, 16, torch.float16))
        assert not sa._gather_eligible(cuda(512, 129)) and not sa._gather_eligible(cuda(512, 16), window = 4096)


@pytest.mark.skipif(not _gather_runs(), reason = "gather_attn shared-KV schedule needs capability >= 10")
def test_cuda_gather_compiles_without_graph_breaks(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "gather")
    import torch._dynamo
    from torch._dynamo.utils import counters
    dtype = torch.bfloat16
    q, kv, sink, idxs = _inputs(1, 32, 8, 128, 48, 16, dtype, "cuda")
    q.requires_grad_(); kv.requires_grad_(); sink.requires_grad_()
    torch._dynamo.reset(); counters.clear()
    fn = torch.compile(lambda q, kv, sink: sa.sparse_attn(q, kv, sink, idxs, 0.1).float().square().sum())
    fn(q, kv, sink).backward()
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    eager_q = q.grad.clone(); q.grad = None
    sa.sparse_attn(q, kv, sink, idxs, 0.1).float().square().sum().backward()
    torch.testing.assert_close(q.grad, eager_q, rtol = 0, atol = 0)


def test_chunked_reference_recomputes_in_backward_with_identical_grads(monkeypatch):
    # The torch path must not keep every gathered [chunk, k, d] slab for backward: chunks are recomputed.
    q, kv, sink, idxs = _inputs(1, 33, 2, 8, 20, 6, torch.float64, "cpu", empty_rows = [(0, 5)])
    sink = sink.double()
    go = torch.randn(1, 33, 2, 8, dtype = torch.float64)

    def grads():
        leaves = [t.clone().requires_grad_() for t in (q, kv, sink)]
        return torch.autograd.grad(sa.sparse_attn_reference(*leaves[:2], leaves[2], idxs, 0.3), leaves, go)

    whole = grads()
    import torch.utils.checkpoint as ckpt
    calls = []
    original = ckpt.checkpoint
    monkeypatch.setattr(ckpt, "checkpoint", lambda *a, **k: calls.append(1) or original(*a, **k))
    monkeypatch.setattr(sa, "_TORCH_CHUNK_ELEMS", 6 * 8 * 4)  # 4 query rows per chunk
    chunked = grads()
    assert len(calls) == 9
    for x, y in zip(chunked, whole):
        torch.testing.assert_close(x, y, rtol = 1e-12, atol = 1e-12)


def _all_pad_inputs(device, dtype, h = 16, d = 128):
    g = torch.Generator().manual_seed(5)
    q = (torch.randn(1, 12, h, d, generator = g) * 0.5).to(device, dtype)
    kv = (torch.randn(1, 20, d, generator = g) * 0.5).to(device, dtype)
    idxs = _unique_idxs(1, 12, 6, 20, 0.2, g, empty_rows = [(0, 0), (0, 5), (0, 11)]).to(device)
    return q, kv, idxs


@pytest.mark.parametrize("with_sink", [True, False])
def test_all_pad_rows_are_zero_and_finite_on_the_torch_path(with_sink):
    q, kv, idxs = _all_pad_inputs("cpu", torch.float32)
    sink = torch.randn(16) if with_sink else torch.full((16,), float("-inf"))
    q, kv, sink = q.requires_grad_(), kv.requires_grad_(), sink.requires_grad_(with_sink)
    out = sa.sparse_attn_reference(q, kv, sink, idxs, 0.2)
    leaves = [q, kv] + ([sink] if with_sink else [])
    grads = torch.autograd.grad(out.square().sum(), leaves)
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
    assert out[0, [0, 5, 11]].abs().max() == 0 and grads[0][0, [0, 5, 11]].abs().max() == 0


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("with_sink", [True, False])
@pytest.mark.parametrize("backend", ["gather", "triton", "flex"])
def test_cuda_all_pad_rows_are_zero_and_finite(monkeypatch, backend, with_sink):
    if backend == "gather" and not _gather_runs():
        pytest.skip("gather_attn shared-KV schedule needs capability >= 10")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", backend)
    q, kv, idxs = _all_pad_inputs("cuda", dtype)
    if backend == "flex" and not sa._flex_eligible(q, kv.shape[1]):
        # Dispatch never picks flex here (sm75 shared memory).
        pytest.skip("flex is not eligible on this device")
    sink = (torch.randn(16) if with_sink else torch.full((16,), float("-inf"))).cuda()
    q, kv = q.requires_grad_(), kv.requires_grad_()
    sink.requires_grad_(with_sink)
    out = sa.sparse_attn(q, kv, sink, idxs, 0.2)
    leaves = [q, kv] + ([sink] if with_sink else [])
    grads = torch.autograd.grad(out.float().square().sum(), leaves)
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
    assert out[0, [0, 5, 11]].abs().max() == 0 and grads[0][0, [0, 5, 11]].abs().max() == 0


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
def test_cuda_triton_kernel_compiles_without_graph_breaks(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "triton")
    import torch._dynamo
    from torch._dynamo.utils import counters
    dtype = _cuda_dtypes()[-1]
    q, kv, sink, idxs = _inputs(1, 40, 16, 512, 60, 24, dtype, "cuda")
    q.requires_grad_(); kv.requires_grad_(); sink.requires_grad_()
    from unsloth_zoo.sparse_mqa_kernel import SPARSE_MQA_STATS
    torch._dynamo.reset(); counters.clear()
    before = dict(SPARSE_MQA_STATS)
    fn = torch.compile(lambda q, kv, sink: sa.sparse_attn(q, kv, sink, idxs, 0.05).float().square().sum())
    fn(q, kv, sink).backward()
    assert counters["stats"]["unique_graphs"] >= 1
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    # The kernels ran inside the compiled graph (opaque custom ops, not an eager fallback).
    assert SPARSE_MQA_STATS["fwd"] > before["fwd"] and SPARSE_MQA_STATS["bwd"] > before["bwd"]
    compiled = [t.grad.clone() for t in (q, kv, sink)]
    for t in (q, kv, sink):
        t.grad = None
    sa.sparse_attn(q, kv, sink, idxs, 0.05).float().square().sum().backward()
    # Inductor fuses the loss backward and the dO . O delta around the opaque ops in another order than eager
    # (1 ulp on a few elements); dkv also uses fp32 atomics. Compare to bf16 rounding.
    for got, want in zip((q.grad, kv.grad, sink.grad), compiled):
        torch.testing.assert_close(got.float(), want.float(), rtol = 2e-2, atol = 1e-4)


_INTERP = """
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
from unsloth_zoo import sparse_attention as sa
from unsloth_zoo.sparse_mqa_kernel import sparse_mqa_attention
torch.manual_seed(0)
worst = 0.0
for (b, s, h, d, n, k, sink_on) in ((1, 7, 3, 16, 9, 5, True), (2, 5, 17, 32, 11, 21, False)):
    q = torch.randn(b, s, h, d, requires_grad = True)
    kv = torch.randn(b, n, d, requires_grad = True)
    sink = torch.randn(h, requires_grad = True) if sink_on else None
    idx = torch.stack([torch.stack([torch.randperm(n)[:min(k, n)] for _ in range(s)]) for _ in range(b)])
    if k > n:
        idx = torch.cat([idx, torch.full((b, s, k - n), -1)], -1)
    idx = torch.where(torch.rand(idx.shape) < 0.25, -1, idx).int()
    idx[0, 1] = -1
    go = torch.randn(b, s, h, d)
    out = sparse_mqa_attention(q, kv, sink, idx, d ** -0.5)
    leaves = [q, kv] + ([sink] if sink_on else [])
    got = [out] + list(torch.autograd.grad(out, leaves, go))
    ref_leaves = [t.detach().double().requires_grad_() for t in leaves]
    ref_sink = ref_leaves[2] if sink_on else torch.full((h,), float("-inf"), dtype = torch.float64)
    ref = sa.sparse_attn_reference(ref_leaves[0], ref_leaves[1], ref_sink, idx, d ** -0.5)
    want = [ref] + list(torch.autograd.grad(ref, ref_leaves, go.double()))
    worst = max(worst, max(float((x.double() - y).abs().max()) for x, y in zip(got, want)))
    assert out[0, 1].abs().max() == 0
# The D-split backward (64 KB parts at head_dim 512), forced at a small head_dim.
import unsloth_zoo.sparse_mqa_kernel as mk
mk._configs = lambda device, d, backward: ((16, 4, 1, d // 4),) if backward else ((16, 4, 1, d),)
mk._CHOSEN.clear()
b, s, h, d, n, k = 1, 6, 5, 64, 13, 18
q = torch.randn(b, s, h, d, requires_grad = True)
kv = torch.randn(b, n, d, requires_grad = True)
sink = torch.randn(h, requires_grad = True)
idx = torch.stack([torch.randperm(n)[:k - 5] for _ in range(s)]).unsqueeze(0)
idx = torch.cat([idx, torch.full((b, s, 5), -1)], -1).int()
go = torch.randn(b, s, h, d)
out = sparse_mqa_attention(q, kv, sink, idx, d ** -0.5)
got = [out] + list(torch.autograd.grad(out, [q, kv, sink], go))
ref_leaves = [t.detach().double().requires_grad_() for t in (q, kv, sink)]
ref = sa.sparse_attn_reference(*ref_leaves, idx, d ** -0.5)
want = [ref] + list(torch.autograd.grad(ref, ref_leaves, go.double()))
worst = max(worst, max(float((x.double() - y).abs().max()) for x, y in zip(got, want)))
print("INTERP_WORST", worst)
"""


@pytest.fixture(scope = "module")
def _interpreter_run():
    # One subprocess for both scripts (window first: the index script patches _configs at its end).
    import subprocess, sys
    from pathlib import Path
    env = dict(os.environ, PYTHONPATH = str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1", OMP_NUM_THREADS = "4",
               UNSLOTH_ZOO_DISABLE_GPU_INIT = "0")  # "1" swaps in the triton stub: no interpreter
    return subprocess.run([sys.executable, "-c", _WINDOW_INTERP + _INTERP], env = env, capture_output = True, text = True,
                          timeout = 900)


def _interpreter_worst(proc, tag):
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(tag + " ")), None)
    if line is None and "No module named 'triton'" in proc.stderr:
        pytest.skip("Triton is not installed (Windows CPU runners)")
    assert line is not None, proc.stderr[-3000:]
    assert float(line.split()[1]) < 1e-5, line


def test_triton_kernel_math_in_the_interpreter(_interpreter_run):
    _interpreter_worst(_interpreter_run, "INTERP_WORST")


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
def test_flex_gate_follows_shared_memory():
    # head_dim 512 flex needs ~160 KB per block: on smaller parts (T4, L4, sm86 / 89, sm120) it must never be picked;
    # sm75 never gets flex at any head_dim (it failed to compile there).
    q = torch.empty(1, 4, 16, 512, device = "cuda", dtype = _cuda_dtypes()[-1])
    sm80 = torch.cuda.get_device_capability()[0] >= 8 and sa._HAS_FLEX   # torch < 2.5: no flex_attention
    big_enough = sm80 and sa._shared_memory(q.device) >= sa._FLEX_D512_MIN_SHARED_MEMORY
    assert sa._flex_eligible(q) == big_enough
    assert ("flex" in sa._order(q, 8)) == big_enough
    assert sa._flex_eligible(torch.empty(1, 4, 16, 128, device = "cuda", dtype = _cuda_dtypes()[-1])) == sm80


def test_refused_launch_latches_and_falls_through(monkeypatch):
    # A backend whose launch is refused (Triton OutOfResources: shared memory over the device limit) is latched off
    # for that shape class and the next backend runs; later calls skip it; the result is the next backend's.
    from triton.runtime.errors import OutOfResources
    q, kv, sink, idxs = _inputs(1, 6, 2, 8, 9, 4, torch.float32, "cpu")
    monkeypatch.setattr(sa, "_order", lambda q, k, backend = None, kv_len = None: tuple(
        b for b in ("triton", "torch") if b == "torch" or sa._failure_key(b, q, k) not in sa._FAILED))
    calls = []
    original = sa._run

    def refusing(backend, *args):
        calls.append(backend)
        if backend == "triton":
            raise OutOfResources(98304, 65536, "shared memory")
        return original(backend, *args)

    monkeypatch.setattr(sa, "_run", refusing)
    monkeypatch.setattr(sa, "_FAILED", {})
    before = dict(sa.SPARSE_ATTN_STATS)
    out = sa.sparse_attn(q, kv, sink, idxs, 0.3)
    torch.testing.assert_close(out, sa.sparse_attn_reference(q, kv, sink, idxs, 0.3), rtol = 0, atol = 0)
    assert calls == ["triton", "torch"]
    assert sa.SPARSE_ATTN_STATS["fallback"] == before["fallback"] + 1 and len(sa._FAILED) == 1
    sa.sparse_attn(q, kv, sink, idxs, 0.3)
    assert calls == ["triton", "torch", "torch"]  # latched: the refused backend is not retried


def test_out_of_memory_is_not_latched(monkeypatch):
    # A CUDA OOM is not a property of the shape class: it propagates instead of silently switching backends.
    q, kv, sink, idxs = _inputs(1, 6, 2, 8, 9, 4, torch.float32, "cpu")
    monkeypatch.setattr(sa, "_order", lambda q, k, backend = None, kv_len = None: ("triton", "torch"))
    monkeypatch.setattr(sa, "_run", lambda backend, *a: (_ for _ in ()).throw(torch.OutOfMemoryError("oom")))
    monkeypatch.setattr(sa, "_FAILED", {})
    with pytest.raises(torch.OutOfMemoryError):
        sa.sparse_attn(q, kv, sink, idxs, 0.3)
    assert not sa._FAILED


# (name, capability, shared memory per block, dtype): the parts below sm100 plus sm120, as their dispatch sees them.
_TIERS = [("T4", (7, 5), 64 * 1024, torch.float16), ("A100", (8, 0), 163 * 1024, torch.bfloat16),
          ("L4", (8, 9), 99 * 1024, torch.bfloat16), ("G4", (12, 0), 99 * 1024, torch.bfloat16)]


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("tier", _TIERS, ids = [t[0] for t in _TIERS])
def test_deepseek_head_dim_512_dispatch_below_sm100(monkeypatch, tier):
    # DeepSeek-V4.1 attention (head_dim 512, 16 heads, window 128 + compressed top-k) with no backend forced, on
    # each tier's capability and shared memory: sm80+ tiers must pick the Triton kernel (not gather, not torch) for
    # the plain and the window form; T4 (sm75, where Triton has no tensor-core dot) must pick torch, and the forced
    # Triton kernel still runs there. All match the fp64 reference. Configs run on this device's codegen (independent
    # of the real part); their per-target shared memory is checked by smem_targets.py.
    from unsloth_zoo import sparse_mqa_kernel as mk
    name, cap, smem, dtype = tier
    real_smem = mk._shared_memory(torch.device("cuda"))
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 tier needs a real sm80+ device")
    want = "torch" if cap[0] < 8 else "triton"
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: cap)
    monkeypatch.setattr(sa, "_shared_memory", lambda device: smem)
    monkeypatch.setattr(mk, "_shared_memory", lambda device: smem)
    monkeypatch.setattr(mk, "_CHOSEN", {})
    monkeypatch.setattr(sa, "_FAILED", {})
    b, s, h, d, n, k = 1, 300, 16, 512, 75, 64
    q, kv, sink, idxs = _inputs(b, s, h, d, n, k, dtype, "cuda", empty_rows = [(0, 1)])
    assert sa._order(q, k, None, n)[0] == want, sa._order(q, k, None, n)
    go = torch.randn(b, s, h, d, device = "cuda", dtype = dtype)
    before = dict(sa.SPARSE_ATTN_STATS)
    got = _grads(lambda q, kv, sink: sa.sparse_attn(q, kv, sink, idxs, d ** -0.5), q, kv, sink, go)
    win = (torch.randn(b, s, d, device = "cuda") * 0.5).to(dtype)
    out_w = sa.sparse_attn_window(q, win, kv, idxs, sink, d ** -0.5, 128)
    delta = {key: sa.SPARSE_ATTN_STATS[key] - before[key] for key in before}
    assert delta == {**{key: 0 for key in before}, want: 2}, delta
    if want == "torch":   # opt-in still reaches the kernel
        forced = _grads(lambda q, kv, sink: sa.sparse_attn(q, kv, sink, idxs, d ** -0.5, backend = "triton"), q, kv, sink, go)
        assert sa.SPARSE_ATTN_STATS["triton"] == before["triton"] + 1
        for x, y in zip(forced, got):
            torch.testing.assert_close(x.float(), y.float(), atol = 2e-2, rtol = 2e-2)
    ref = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, d ** -0.5),
                 q.double(), kv.double(), sink.double(), go.double())
    low = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, d ** -0.5), q, kv, sink, go)
    for label, x, r, lo in zip(("out", "dq", "dkv", "dsink"), got, ref, low):
        err, base = (x.double() - r).abs().max().item(), (lo.double() - r).abs().max().item()
        assert err <= 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item()), (name, label, err, base)
    full = torch.cat([sa.window_topk_idxs(s, 128, b, "cuda"), torch.where(idxs >= 0, idxs + s, -1).int()], -1)
    ref_w = sa.sparse_attn_reference(q.double(), torch.cat([win, kv], 1).double(), sink.double(), full, d ** -0.5)
    low_w = sa.sparse_attn_reference(q, torch.cat([win, kv], 1), sink, full, d ** -0.5)
    err, base = (out_w.double() - ref_w).abs().max().item(), (low_w.double() - ref_w).abs().max().item()
    assert err <= 2 * base + 4 * torch.finfo(dtype).eps, (name, "window", err, base)
    chosen = {key[3]: mk._configs(None, d, key[3])[i] for key, i in mk._CHOSEN.items() if key[0] != "window"}
    expect_split = smem < 96 * 1024
    if expect_split or real_smem >= 96 * 1024:   # a real 64 KB part refuses the unsplit config and steps down
        assert chosen[True][3] == (d // 2 if expect_split else d), chosen
    else:
        assert chosen[True] in mk._configs(None, d, True), chosen


_WINDOW_INTERP = """
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
from unsloth_zoo import sparse_attention as sa
from unsloth_zoo.sparse_mqa_kernel import sparse_mqa_window_attention, sparse_mqa_attention
torch.manual_seed(0)
worst = 0.0
# (b, s, h, d, nc, k, window, sink): heads not a multiple of 16, no compressed slots, window 1, window > s.
for (b, s, h, d, nc, k, w, sink_on) in ((1, 37, 3, 16, 9, 5, 8, True), (2, 21, 17, 32, 11, 0, 5, False),
                                        (1, 20, 16, 16, 7, 4, 64, True), (1, 9, 2, 16, 6, 3, 1, True)):
    q = torch.randn(b, s, h, d, requires_grad = True)
    win = torch.randn(b, s, d, requires_grad = True)
    comp = torch.randn(b, nc, d, requires_grad = True)
    sink = torch.randn(h, requires_grad = True) if sink_on else None
    if k:
        cidx = torch.stack([torch.stack([torch.randperm(nc)[:k] for _ in range(s)]) for _ in range(b)])
        cidx = torch.where(torch.rand(cidx.shape) < 0.3, -1, cidx).int()
        cidx[0, 2] = -1
    else:
        cidx = torch.empty(b, s, 0, dtype = torch.int32)
    go = torch.randn(b, s, h, d)
    out = sparse_mqa_window_attention(q, win, comp, cidx, sink, d ** -0.5, w)
    leaves = [q, win, comp] + ([sink] if sink_on else [])
    got = [out] + list(torch.autograd.grad(out, leaves, go))
    ref_leaves = [t.detach().double().requires_grad_() for t in leaves]
    ref_sink = ref_leaves[3] if sink_on else torch.full((h,), float("-inf"), dtype = torch.float64)
    full = torch.cat([sa.window_topk_idxs(s, w, b, "cpu"), torch.where(cidx >= 0, cidx + s, -1).int()], -1)
    ref = sa.sparse_attn_reference(ref_leaves[0], torch.cat(ref_leaves[1:3], 1), ref_sink, full, d ** -0.5)
    want = [ref] + list(torch.autograd.grad(ref, ref_leaves, go.double()))
    worst = max(worst, max(float((x.double() - y).abs().max()) for x, y in zip(got, want)))
# One index row shared by g consecutive batches (the GQA fold) == the repeated rows.
b, g, s, h, d, n, k = 2, 2, 5, 4, 16, 10, 6
q = torch.randn(b * g, s, h, d, requires_grad = True)
kv = torch.randn(b * g, n, d, requires_grad = True)
idx = torch.stack([torch.stack([torch.randperm(n)[:k] for _ in range(s)]) for _ in range(b)]).int()
go = torch.randn(b * g, s, h, d)
shared = sparse_mqa_attention(q, kv, None, idx, d ** -0.5)
grads = torch.autograd.grad(shared, [q, kv], go)
rep = sparse_mqa_attention(q, kv, None, idx.repeat_interleave(g, 0), d ** -0.5)
grads_rep = torch.autograd.grad(rep, [q, kv], go)
worst = max(worst, float((shared - rep).abs().max()), *(float((x - y).abs().max()) for x, y in zip(grads, grads_rep)))
print("WINDOW_INTERP_WORST", worst)
"""


def test_window_kernel_and_shared_index_rows_in_the_interpreter(_interpreter_run):
    _interpreter_worst(_interpreter_run, "WINDOW_INTERP_WORST")


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("cap,smem,want", [((10, 0), 228 * 1024, "flex"), ((8, 9), 99 * 1024, "flex"),
                                           ((8, 0), 64 * 1024, "triton"), ((7, 5), 64 * 1024, "torch")],
                         ids = ["sm100", "sm89", "sm80_64KB", "sm75"])
def test_qsa_gqa_dispatch_prefers_flex_where_it_compiles(monkeypatch, cap, smem, want):
    # QSA (8 q / 2 kv heads x 256): flex where head_dim 256 flex compiles (sm80+, > 64 KB), else the [K | V] fold
    # through the Triton kernel with one shared index row per KV head (no repeat), and torch on sm75 (where the
    # Qwen patch keeps transformers' path: sparse_gqa_fast_path is False); all match the fp64 reference.
    from unsloth_zoo import sparse_mqa_kernel as mk
    real_cap = torch.cuda.get_device_capability()
    if want == "flex" and real_cap[0] < 8:
        pytest.skip("flex needs a real sm80+ device")
    if want == "flex" and not sa._HAS_FLEX:   # torch < 2.5: no flex_attention, the fold runs
        want = "triton"
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: cap)
    monkeypatch.setattr(sa, "_shared_memory", lambda device: smem)
    monkeypatch.setattr(mk, "_shared_memory", lambda device: smem)
    monkeypatch.setattr(mk, "_CHOSEN", {})
    monkeypatch.setattr(sa, "_FAILED", {})
    g = torch.Generator().manual_seed(5)
    b, hq, hk, s, d, k = 1, 8, 2, 200, 256, 48
    dtype = torch.bfloat16 if real_cap[0] >= 8 else torch.float16
    query = (torch.randn(b, hq, s, d, generator = g) * 0.5).cuda().to(dtype)
    key = (torch.randn(b, hk, s, d, generator = g) * 0.5).cuda().to(dtype)
    value = (torch.randn(b, hk, s, d, generator = g) * 0.5).cuda().to(dtype)
    idxs = _unique_idxs(b, s, k, s, 0.2, g).cuda()
    go = torch.randn(b, s, hq, d, generator = g).cuda().to(dtype)
    repeats = []
    real_repeat = torch.Tensor.repeat_interleave
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", lambda self, *a, **kw: repeats.append(1) or real_repeat(self, *a, **kw))

    def run(fn, *xs):
        xs = [x.detach().clone().requires_grad_() for x in xs]
        out = fn(*xs)
        return [out, *torch.autograd.grad(out, xs, go.to(out.dtype))]

    before = dict(sa.SPARSE_ATTN_STATS)
    got = run(lambda q, k_, v: sa.sparse_gqa_attn(q, k_, v, idxs, d ** -0.5), query, key, value)
    delta = {n: sa.SPARSE_ATTN_STATS[n] - before[n] for n in before}
    assert delta == {**{n: 0 for n in before}, want: 1}, delta
    assert sa.sparse_gqa_fast_path(query, key, value) == (want != "torch")
    assert not repeats or want == "torch"

    def dense(q, k_, v):
        # -1 slots land in a spare column (a clamp to 0 would race valid writes to key 0).
        sel = torch.zeros(b, s, s + 1, dtype = torch.bool, device = "cuda")
        sel.scatter_(2, torch.where(idxs >= 0, idxs, s).long(), True)
        sel = sel[..., :s]
        att = torch.einsum("bhsd,bhnd->bhsn", q, k_.repeat_interleave(hq // hk, 1)) * d ** -0.5
        att = att.masked_fill(~sel.unsqueeze(1), float("-inf")).softmax(-1).nan_to_num()
        return torch.einsum("bhsn,bhnd->bshd", att, v.repeat_interleave(hq // hk, 1))

    ref = run(dense, query.double(), key.double(), value.double())
    low = run(dense, query, key, value)
    for label, x, r, lo in zip(("out", "dq", "dk", "dv"), got, ref, low):
        err, base = (x.double() - r).abs().max().item(), (lo.double() - r).abs().max().item()
        assert err <= 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item()), (want, label, err, base)


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("backend", [None, "hd512", "triton", "torch"] + (["gather"] if _gather_runs() else []))
@pytest.mark.parametrize("with_sink", [True, False])
def test_cuda_window_only_layer_with_no_compressed_rows(monkeypatch, backend, with_sink):
    # DeepSeek window-only layers (compress_ratio 0) call sparse_attn_window with an empty compressed set:
    # compress_kv [b, 0, d], compress_idxs [b, s, 0]. Every backend accepts it and matches the reference.
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    g = torch.Generator().manual_seed(9)
    b, s, h, d, w = 1, 300, 16, 512, 128
    q = (torch.randn(b, s, h, d, generator = g) * 0.5).cuda().to(dtype)
    win = (torch.randn(b, s, d, generator = g) * 0.5).cuda().to(dtype)
    sink = torch.randn(h, generator = g).cuda()
    if not with_sink:
        sink = torch.full((h,), float("-inf"), device = "cuda")
    go = torch.randn(b, s, h, d, generator = g).cuda().to(dtype)
    from unsloth_zoo import sparse_mqa_kernel as mk
    hd512 = mk.window_hd512_supported(q, win)
    if backend == "hd512" and not hd512:
        pytest.skip("Unsloth's head_dim 512 kernel is not importable or does not fit this device's shared memory")
    before = dict(sa.SPARSE_ATTN_STATS)
    empty_kv, empty_idx = win.new_zeros(b, 0, d), torch.empty(b, s, 0, dtype = torch.int32, device = "cuda")
    widx = sa.window_topk_idxs(s, w, b, "cuda")
    got = _grads(lambda q, win, sink: sa.sparse_attn_window(q, win, empty_kv, empty_idx, sink, d ** -0.5, w, backend = backend),
                 q, win, sink, go)
    ran = {n: sa.SPARSE_ATTN_STATS[n] - before[n] for n in before if sa.SPARSE_ATTN_STATS[n] != before[n]}
    want = backend or ("hd512" if hd512 else ("gather" if sa._gather_eligible(q, w) else
                                              ("triton" if sa._triton_eligible(q) else "torch")))
    assert ran == {want: 1}, ran
    ref = _grads(lambda q, win, sink: sa.sparse_attn_reference(q, win, sink, widx, d ** -0.5),
                 q.double(), win.double(), sink.double(), go.double())
    low = _grads(lambda q, win, sink: sa.sparse_attn_reference(q, win, sink, widx, d ** -0.5), q, win, sink, go)
    for label, x, r, lo in list(zip(("out", "dq", "dwin", "dsink"), got, ref, low))[:4 if with_sink else 3]:
        err, base = (x.double() - r).abs().max().item(), (lo.double() - r).abs().max().item()
        assert err <= 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item()), (backend, label, err, base)


def _opcheck_cases():
    if not _CUDA:
        return []
    from unsloth_zoo import sparse_mqa_kernel as mk
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    g = torch.Generator().manual_seed(11)
    def rnd(*shape):
        return (torch.randn(*shape, generator = g) * 0.5).cuda().to(dtype)
    b, s, h, d, n, k = 1, 40, 4, 64, 30, 12
    # q as a transposed view ([b, h, s, d] storage): outputs must still match the fake kernels' contiguous layout.
    q = rnd(b, h, s, d).transpose(1, 2)
    kv, wkv, ckv = rnd(b, n, d), rnd(b, s, d), rnd(b, n, d)
    idx = _unique_idxs(b, s, k, n, 0.2, g).cuda()
    sink = torch.randn(h, generator = g).cuda()
    out, lse = mk._fwd_op(q, kv, idx, sink, d ** -0.5, True)
    go, delta = rnd(b, s, h, d), torch.randn(b, s, h, generator = g).cuda()
    cases = [
        ("index_fwd", mk._fwd_op, (q, kv, idx, sink, d ** -0.5, True)),
        ("index_bwd", mk._bwd_op, (q, kv, idx, go, lse, delta, d ** -0.5)),
        ("window_fwd", mk._wfwd_op, (q, wkv, ckv, idx, sink, d ** -0.5, 8, True)),
        ("window_bwd", mk._wbwd_op, (q, wkv, ckv, idx, go, lse, delta, d ** -0.5, 8)),
    ]
    q5 = rnd(b, h, s, 512).transpose(1, 2)
    w5 = rnd(b, s, 512)
    if mk.window_hd512_supported(q5, w5):
        out5, total5, lse25 = mk._hfwd_op(q5, w5, sink, 512 ** -0.5, 8, True)
        cases += [("hd512_fwd", mk._hfwd_op, (q5, w5, sink, 512 ** -0.5, 8, True)),
                  ("hd512_bwd", mk._hbwd_op, (rnd(b, s, h, 512), q5, w5, out5, lse25, 512 ** -0.5, 8))]
    return cases


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("name,op,args", _opcheck_cases(), ids = [c[0] for c in _opcheck_cases()])
def test_custom_op_opcheck(name, op, args):
    # Schema, fake-tensor (shapes, dtypes and strides of the real outputs), autograd registration, AOT dispatch.
    args = tuple(a.detach().requires_grad_() if isinstance(a, torch.Tensor) and a.is_floating_point() and name.endswith("fwd") else a
                 for a in args)
    torch.library.opcheck(op, args)


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("with_sink", [True, False])
def test_window_only_hd512_compiles_fullgraph(with_sink):
    # DeepSeek window-only layer (head_dim 512, no compressed set) under torch.compile(fullgraph = True): the hd512
    # path engages and the compiled forward + backward match eager.
    from unsloth_zoo import sparse_mqa_kernel as mk
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    g = torch.Generator().manual_seed(12)
    b, s, h, d, w = 1, 300, 16, 512, 128
    q = (torch.randn(b, s, h, d, generator = g) * 0.5).cuda().to(dtype)
    win = (torch.randn(b, s, d, generator = g) * 0.5).cuda().to(dtype)
    sink = torch.randn(h, generator = g).cuda() if with_sink else torch.full((h,), float("-inf"), device = "cuda")
    if not mk.window_hd512_supported(q, win):
        pytest.skip("Unsloth's head_dim 512 kernel is not importable or does not fit here")
    empty_kv, empty_idx = win.new_zeros(b, 0, d), torch.empty(b, s, 0, dtype = torch.int32, device = "cuda")
    go = torch.randn(b, s, h, d, generator = g).cuda().to(dtype)

    def layer(q, win, sink):
        return sa.sparse_attn_window(q, win, empty_kv, empty_idx, sink, d ** -0.5, w)

    torch._dynamo.reset()
    calls = mk.SPARSE_MQA_STATS["hd512"]
    eager = _grads(layer, q, win, sink, go)
    assert mk.SPARSE_MQA_STATS["hd512"] == calls + 1
    compiled = _grads(torch.compile(layer, fullgraph = True), q, win, sink, go)
    for x, y in zip(compiled, eager):
        torch.testing.assert_close(x.float(), y.float(), atol = 1e-2, rtol = 1e-2)


def _bool_membership(topk_idxs, kv_len, block = 128):
    """The [b, s, n] bool table the flex path used to keep for backward: the allowed-key oracle."""
    b, s, k = topk_idxs.shape
    nq, nk = -(-s // block), -(-kv_len // block)
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    cols = torch.where(valid, topk_idxs, nk * block).long()
    table = torch.zeros((b, nq * block, nk * block + 1), dtype = torch.bool)
    table[:, :s].scatter_(-1, cols, True)
    table = table[..., : nk * block]
    tiles = table.view(b, nq, block, nk, block)
    return table, tiles.any(dim = 4).any(dim = 2), tiles.all(dim = 4).all(dim = 2)


@pytest.mark.parametrize("s,n,k", [(300, 300, 40), (130, 517, 200), (256, 256, 256), (77, 1, 1)])
def test_bit_packed_membership_mask_allows_exactly_the_listed_keys(s, n, k):
    # One bit per key: 8 MB per layer at 8k instead of 64 MB.
    g = torch.Generator().manual_seed(s + n)
    b = 2
    idxs = _unique_idxs(b, s, min(k, n), n, 0.2, g, empty_rows = [(0, 3), (1, s - 1)])
    if k >= n:   # rows that list every key: fully visible tiles
        idxs[1, : min(s, 128)] = torch.arange(n, dtype = torch.int32)
    idxs[0, 5, 0] = n + 7   # out of range: ignored, as before
    table, any_ref, all_ref = _bool_membership(idxs, n)
    mask = sa._membership_block_mask(idxs, n)
    kept = [t for t in mask.mask_mod.__closure__ or () if isinstance(t.cell_contents, torch.Tensor)]
    held = sum(t.cell_contents.numel() * t.cell_contents.element_size() for t in kept)
    assert held * 8 <= table.numel() + 32 * 8 * table.shape[0] * table.shape[1], (held, table.numel())
    bi, qi, ki = (x.reshape(-1) for x in torch.meshgrid(
        torch.arange(b), torch.arange(table.shape[1]), torch.arange(table.shape[2]), indexing = "ij"))
    got = mask.mask_mod(bi, torch.zeros_like(bi), qi, ki).view(table.shape)
    assert torch.equal(got, table)
    # Tiles: visible = partial + full, as the bool table's any(); full as its all().
    assert torch.equal(mask.to_dense()[:, 0].bool(), any_ref)
    full = torch.zeros_like(all_ref)
    for bb in range(b):
        for qq in range(any_ref.shape[1]):
            c = int(mask.full_kv_num_blocks[bb, 0, qq])
            full[bb, qq, mask.full_kv_indices[bb, 0, qq, :c].long()] = True
    assert torch.equal(full, all_ref)


def test_flex_budget_admits_32k_tokens():
    # At 1 bit per entry the same 256 MB budget holds QSA at 32k tokens (b = 1); 64k still falls back.
    assert 32768 * 32768 <= sa._FLEX_MAX_TABLE < 65536 * 65536


def _compiled_graphs(fn, *args):
    """Dynamo FX graphs of fn(*args) (backward through aot_eager)."""
    from torch._dynamo.backends.common import aot_autograd
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return aot_autograd(fw_compiler = lambda g, _: g.forward, bw_compiler = lambda g, _: g.forward)(gm, example_inputs)

    torch._dynamo.reset()
    out = torch.compile(fn, backend = backend, dynamic = True, fullgraph = True)(*args)
    return out, graphs


@pytest.mark.parametrize("with_sink", [True, False])
def test_compiled_chunked_reference_is_one_op_and_matches_fp64(monkeypatch, with_sink):
    # The torch path (T4 / sm75, CPU) under torch.compile: the chunk loop unrolled into the graph (one checkpoint HOP
    # and one inductor index simplification per chunk) took minutes per layer to compile; it is one opaque op whose
    # backward recomputes chunk by chunk.
    q, kv, sink, idxs = _inputs(1, 24, 2, 8, 20, 6, torch.float64, "cpu", empty_rows = [(0, 5)])
    sink = sink.double() if with_sink else None
    go = torch.randn(1, 24, 2, 8, dtype = torch.float64)

    def leaves():
        return [t.clone().requires_grad_() for t in (q, kv)] + ([sink.clone().requires_grad_()] if with_sink else [])

    def fn(*ts):
        return sa.sparse_attn_reference(ts[0], ts[1], ts[2] if with_sink else None, idxs, 0.3)

    want = leaves()
    want_out = fn(*want)
    want_grads = torch.autograd.grad(want_out, want, go)
    naive_sink = sink if with_sink else torch.full((2,), float("-inf"), dtype = torch.float64)
    torch.testing.assert_close(want_out.detach(), _naive(q, kv, naive_sink, idxs, 0.3), rtol = 1e-12, atol = 1e-12)

    for rows in (4, 2):  # 6 and 12 chunks
        monkeypatch.setattr(sa, "_TORCH_CHUNK_ELEMS", rows * 6 * 8)
        got = leaves()
        out, graphs = _compiled_graphs(fn, *got)
        grads = torch.autograd.grad(out, got, go)
        torch.testing.assert_close(out, want_out.detach(), rtol = 1e-12, atol = 1e-12)
        for a, b in zip(grads, want_grads):
            torch.testing.assert_close(a, b, rtol = 1e-12, atol = 1e-12)
        assert len(graphs) == 1
        targets = [str(n.target) for n in graphs[0].graph.nodes if n.op == "call_function"]
        assert not any("checkpoint" in t for t in targets), targets
        assert sum("sparse_attn_chunked_fwd" in t for t in targets) == 1, targets
        assert not any("einsum" in t for t in targets), targets  # no chunk is traced into the graph


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_chunked_reference_op_grads_match_the_checkpointed_chunks(monkeypatch, dtype):
    q, kv, sink, idxs = _inputs(2, 13, 3, 16, 17, 5, dtype, "cpu", empty_rows = [(1, 0)])
    monkeypatch.setattr(sa, "_TORCH_CHUNK_ELEMS", 4 * 5 * 16 * 2)  # 4 rows per chunk, a short last chunk
    sink = sink.float().view(1, 1, 3)
    go = torch.randn(2, 13, 3, 16).to(dtype)
    ref = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink.view(3), idxs, 0.25),
                 q.double(), kv.double(), sink.double(), go.double())
    eager = _grads(lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink.view(3), idxs, 0.25), q, kv, sink, go)
    op = _grads(lambda q, kv, sink: sa._chunked_fwd_op(q, kv, sink, idxs, 0.25), q, kv, sink, go)
    for name, x, e, r in zip(("out", "dq", "dkv", "dsink"), op, eager, ref):
        assert x.dtype == e.dtype and x.shape == e.shape, name
        err = (x.double() - r).abs().max().item()
        base = (e.double() - r).abs().max().item()
        bound = 2 * base + 4 * torch.finfo(dtype).eps * max(1.0, r.abs().max().item())
        assert err <= bound, f"{name}: err {err:.3e} > bound {bound:.3e} (checkpointed chunks {base:.3e})"
    torch.library.opcheck(sa._chunked_fwd_op, tuple(t.detach().requires_grad_() for t in (q, kv, sink)) + (idxs, 0.25))
