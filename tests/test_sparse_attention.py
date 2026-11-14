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
    # Only the shared-KV schedule is used in production: capability >= 10 and bf16.
    return _CUDA and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.skipif(not _CUDA, reason = "CUDA backends")
@pytest.mark.parametrize("backend,dtype,shape", [("flex", dt, sh) for dt in _cuda_dtypes() for sh in _FLEX_SHAPES]
                         + ([("gather", torch.bfloat16, sh) for sh in _GATHER_SHAPES] if _gather_runs() else []))
def test_cuda_backend_matches_fp64(monkeypatch, backend, dtype, shape):
    b, s, h, d, n, k = shape
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


@pytest.mark.skipif(not _gather_runs(), reason = "gather_attn shared-KV schedule needs capability >= 10")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("s", [40, 300])
def test_cuda_window_form_matches_concatenated(monkeypatch, dtype, s):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "gather")
    g = torch.Generator().manual_seed(2)
    b, h, d, nc, k, w = 1, 16, 512, 150, 32, 128
    q = (torch.randn(b, s, h, d, generator = g) * 0.5).cuda().to(dtype)
    win = (torch.randn(b, s, d, generator = g) * 0.5).cuda().to(dtype)
    comp = (torch.randn(b, nc, d, generator = g) * 0.5).cuda().to(dtype)
    sink = torch.randn(h, generator = g).cuda()
    cidx = _unique_idxs(b, s, k, nc, 0.2, g).cuda()
    out = sa.sparse_attn_window(q, win, comp, cidx, sink, d ** -0.5, w)
    end = torch.arange(s, device = "cuda").unsqueeze(1)
    widx = (end - w + 1).clamp(0) + torch.arange(min(s, w), device = "cuda")
    widx = torch.where(widx > end, -1, widx).int().unsqueeze(0)
    full = torch.cat([widx, torch.where(cidx >= 0, cidx + s, -1).int()], -1)
    ref = sa.sparse_attn_reference(q.double(), torch.cat([win, comp], 1).double(), sink.double(), full, d ** -0.5)
    low = sa.sparse_attn_reference(q, torch.cat([win, comp], 1), sink, full, d ** -0.5)
    err, base = (out.double() - ref).abs().max().item(), (low.double() - ref).abs().max().item()
    assert err <= 2 * base + 4 * torch.finfo(dtype).eps


def test_gather_eligibility_is_the_shared_schedule_only():
    q = lambda d, h, dt = torch.bfloat16: torch.empty(1, 1, h, d, dtype = dt, device = "meta")
    # meta tensors are not CUDA: never eligible; the shape rules are checked through a CUDA stand-in below.
    assert not sa._gather_eligible(q(512, 16))
    if _gather_runs():
        cuda = lambda d, h, dt = torch.bfloat16: torch.empty(1, 1, h, d, dtype = dt, device = "cuda")
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
