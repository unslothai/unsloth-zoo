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

"""Fixed rows for the fused CE and GKD JSD ops: every row computed, dropped rows masked, so a CUDA graph can capture
forward + backward (compaction's nonzero and kept-row count are a host sync). Must equal compaction and an fp64
reference."""

import math

import pytest
import torch
import torch.nn.functional as F

import unsloth_zoo.fused_losses.fused_linear as fl

N, H, V = 37, 24, 101


def _ce_case(seed, dtype, device = "cpu", ignore = -100, all_dropped = False):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, H, generator = g)
    w = torch.randn(V, H, generator = g) / 4
    b = torch.randn(V, generator = g) / 4
    t = torch.randint(0, V, (N,), generator = g)
    drop = torch.rand(N, generator = g) < 0.3
    if all_dropped:
        drop[:] = True
    t[drop] = ignore
    h[drop] = 3.0e4 if dtype != torch.float16 else 6.0e4  # overflows the logits if a dropped row were used
    to = lambda x: x.to(device)  # noqa: E731
    return to(h).to(dtype), to(w).to(dtype), to(b).to(dtype), to(t)


def _ce(fixed, monkeypatch, h, w, b, t, head, ignore = -100, **kw):
    monkeypatch.setattr(fl, "_FIXED_ROWS", fixed)
    h = h.clone().requires_grad_()
    w = w.clone().requires_grad_(head)
    b = b.clone().requires_grad_(head)
    loss = fl.fused_linear_cross_entropy(h, w, b, t, divisor = 7.0, ignore_index = ignore, **kw)
    (loss * 3.0).backward()
    return (loss, h.grad) + ((w.grad, b.grad) if head else ())


def _ce_ref(h, w, b, t, head, ignore = -100, label_smoothing = 0.0, logit_softcapping = 0.0):
    h = h.double().requires_grad_()
    w = w.double().requires_grad_(head)
    b = b.double().requires_grad_(head)
    z = h @ w.t() + b
    if logit_softcapping:
        z = logit_softcapping * torch.tanh(z / logit_softcapping)
    loss = F.cross_entropy(z, t, ignore_index = ignore, reduction = "sum", label_smoothing = label_smoothing) / 7.0
    (loss * 3.0).backward()
    return (loss, h.grad) + ((w.grad, b.grad) if head else ())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head", [False, True])
@pytest.mark.parametrize("ignore,smooth,cap", [(-100, 0.0, 0.0), (5, 0.1, 0.0), (-100, 0.0, 3.0)])
def test_ce_fixed_rows_equal_compaction(dtype, head, ignore, smooth, cap, monkeypatch):
    case = _ce_case(0, dtype, ignore = ignore)
    kw = dict(label_smoothing = smooth, logit_softcapping = cap)
    ref = _ce(False, monkeypatch, *case, head, ignore, **kw)
    got = _ce(True, monkeypatch, *case, head, ignore, **kw)
    oracle = _ce_ref(*case, head, ignore, smooth, cap)
    tol = dict(rtol = 1e-5, atol = 1e-6) if dtype == torch.float32 else dict(rtol = 0, atol = 0)
    dropped = case[3] == ignore
    for r, x, o in zip(ref, got, oracle):
        assert torch.isfinite(x.float()).all()
        torch.testing.assert_close(x, r, **tol)
        # Same error vs fp64 as compaction (fp32 to summation order).
        err_x, err_r = (x.double() - o).abs().max().item(), (r.double() - o).abs().max().item()
        assert err_x <= err_r * 1.01 + 1e-6
    assert (got[1][dropped] == 0).all()


@pytest.mark.parametrize("head", [False, True])
def test_ce_fixed_rows_all_dropped(head, monkeypatch):
    case = _ce_case(1, torch.float32, all_dropped = True)
    for x in _ce(True, monkeypatch, *case, head):
        assert torch.isfinite(x).all() and (x == 0).all()


def _jsd_case(seed, dtype, device = "cpu", all_dropped = False):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, H, generator = g)
    w = torch.randn(V, H, generator = g) / 4
    b = torch.randn(V, generator = g) / 4
    th = torch.randn(N, H + 8, generator = g)
    tw = torch.randn(V, H + 8, generator = g) / 4
    tb = torch.randn(V, generator = g) / 4
    mask = torch.rand(N, generator = g) > 0.3
    if all_dropped:
        mask[:] = False
    h[~mask] = 3.0e4 if dtype != torch.float16 else 6.0e4
    th[~mask] = -3.0e4 if dtype != torch.float16 else -6.0e4
    to = lambda x: x.to(device)  # noqa: E731
    return tuple(to(x).to(dtype) for x in (h, w, b, th, tw, tb)) + (to(mask),)


def _jsd(fixed, monkeypatch, h, w, b, th, tw, tb, mask, head, beta, **kw):
    monkeypatch.setattr(fl, "_FIXED_ROWS", fixed)
    h = h.clone().requires_grad_()
    w = w.clone().requires_grad_(head)
    b = b.clone().requires_grad_(head)
    loss = fl.fused_linear_jsd(h, w, th, tw, mask = mask, beta = beta, student_bias = b, teacher_bias = tb, **kw)
    (loss * 3.0).backward()
    return (loss, h.grad) + ((w.grad, b.grad) if head else ())


def _jsd_ref(h, w, b, th, tw, tb, mask, head, beta, temperature = 1.0, student_logit_softcapping = 0.0):
    h = h.double().requires_grad_()
    w = w.double().requires_grad_(head)
    b = b.double().requires_grad_(head)
    s = h @ w.t() + b
    if student_logit_softcapping:
        s = student_logit_softcapping * torch.tanh(s / student_logit_softcapping)
    t = th.double() @ tw.double().t() + tb.double()
    ls, lt = torch.log_softmax(s / temperature, -1), torch.log_softmax(t / temperature, -1)
    if beta == 0.0:
        row = (lt.exp() * (lt - ls)).sum(-1)
    elif beta == 1.0:
        row = (ls.exp() * (ls - lt)).sum(-1)
    else:
        m = torch.logaddexp(ls + math.log1p(-beta), lt + math.log(beta))
        row = beta * (lt.exp() * (lt - m)).sum(-1) + (1 - beta) * (ls.exp() * (ls - m)).sum(-1)
    loss = torch.where(mask, row, 0).sum() / max(int(mask.sum()), 1)
    (loss * 3.0).backward()
    return (loss, h.grad) + ((w.grad, b.grad) if head else ())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head", [False, True])
@pytest.mark.parametrize("beta,kw", [(0.0, {}), (0.5, {"temperature": 0.7}), (1.0, {"student_logit_softcapping": 3.0})])
def test_jsd_fixed_rows_equal_compaction(dtype, head, beta, kw, monkeypatch):
    case = _jsd_case(0, dtype)
    ref = _jsd(False, monkeypatch, *case, head, beta, **kw)
    got = _jsd(True, monkeypatch, *case, head, beta, **kw)
    oracle = _jsd_ref(*case, head, beta, **kw)
    tol = dict(rtol = 1e-5, atol = 1e-6) if dtype == torch.float32 else dict(rtol = 0, atol = 0)
    for r, x, o in zip(ref, got, oracle):
        assert torch.isfinite(x.float()).all()
        torch.testing.assert_close(x, r, **tol)
        err_x, err_r = (x.double() - o).abs().max().item(), (r.double() - o).abs().max().item()
        assert err_x <= err_r * 1.01 + 1e-6
    assert (got[1][~case[-1]] == 0).all()


@pytest.mark.parametrize("head", [False, True])
def test_jsd_fixed_rows_all_dropped(head, monkeypatch):
    case = _jsd_case(1, torch.float32, all_dropped = True)
    for x in _jsd(True, monkeypatch, *case, head, 0.5):
        assert torch.isfinite(x).all() and (x == 0).all()


def _cuda_dtype():
    # is_bf16_supported() is True on a T4 (emulated): fp16 below sm80, as Unsloth picks.
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16


def _ce_step(h, w, b, t):
    loss = fl.fused_linear_cross_entropy(h, w, b, t, divisor = 7.0)
    (loss * 3.0).backward()
    return loss


def _jsd_step(h, w, b, th, tw, tb, mask):
    loss = fl.fused_linear_jsd(h, w, th, tw, mask = mask, beta = 0.5, student_bias = b, teacher_bias = tb)
    (loss * 3.0).backward()
    return loss


def _capture(step, leaves, inputs):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for x in leaves:
                x.grad = None
            step(*inputs)
    torch.cuda.current_stream().wait_stream(s)
    for x in leaves:
        x.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(*inputs)
    return graph, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("op", ["ce", "jsd"])
@pytest.mark.parametrize("head", [False, True])
@pytest.mark.parametrize("dtype", ["auto", torch.float16])
def test_capture_in_a_cuda_graph(op, head, dtype):
    """Forward + backward captured in a CUDA graph, replayed on new data (new dropped rows), equal to eager."""
    dtype = _cuda_dtype() if dtype == "auto" else dtype
    make = _ce_case if op == "ce" else _jsd_case
    step = _ce_step if op == "ce" else _jsd_step
    case = [x.clone() for x in make(1, dtype, "cuda")]
    case[0].requires_grad_()
    case[1].requires_grad_(head)
    case[2].requires_grad_(head)
    leaves = case[:3]
    before = fl.CALLS.get("fixed_rows", 0)
    graph, loss = _capture(step, leaves, case)
    assert fl.CALLS.get("fixed_rows", 0) > before  # the capture took the fixed-row path
    for seed in (2, 3, 4):
        new = make(seed, dtype, "cuda", **({"all_dropped": True} if seed == 4 else {}))
        with torch.no_grad():
            for x, y in zip(case, new):
                x.copy_(y)
        graph.replay()
        torch.cuda.synchronize()
        ref = [x.clone() for x in new]
        ref[0].requires_grad_()
        ref[1].requires_grad_(head)
        ref[2].requires_grad_(head)
        lr = step(*ref)
        torch.testing.assert_close(loss, lr, rtol = 1e-4, atol = 1e-5)
        for x, r in zip(leaves[: 3 if head else 1], ref[: 3 if head else 1]):
            torch.testing.assert_close(x.grad.float(), r.grad.float(), rtol = 2e-2, atol = 2e-3)
        assert torch.isfinite(loss) and torch.isfinite(leaves[0].grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("op", ["ce", "jsd"])
def test_compaction_cannot_be_captured(op, monkeypatch):
    """Negative control: with the fixed-row path off, capture fails (nonzero / the kept count sync the host)."""
    monkeypatch.setattr(fl, "_fixed_rows", lambda device: False)
    make = _ce_case if op == "ce" else _jsd_case
    step = _ce_step if op == "ce" else _jsd_step
    case = [x.clone() for x in make(1, _cuda_dtype(), "cuda")]
    case[0].requires_grad_()
    with pytest.raises(Exception):
        _capture(step, case[:1], case)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("op", ["ce", "jsd"])
def test_fixed_rows_never_sync_the_host(op, monkeypatch):
    monkeypatch.setattr(fl, "_FIXED_ROWS", True)
    make = _ce_case if op == "ce" else _jsd_case
    step = _ce_step if op == "ce" else _jsd_step
    case = [x.clone() for x in make(4, _cuda_dtype(), "cuda")]
    case[0].requires_grad_()
    case[1].requires_grad_()
    for _ in range(2):
        step(*case)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step(*case)
    finally:
        torch.cuda.set_sync_debug_mode(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("op", ["ce", "jsd"])
def test_reduce_overhead_compile(op):
    """torch.compile(mode = "reduce-overhead", fullgraph = True) over hidden -> op -> loss -> backward: no graph
    break, CUDA graphs recorded through the fixed-row path, equal to eager on new data."""
    import torch._dynamo.testing

    torch._dynamo.reset()
    dtype = _cuda_dtype()
    make = _ce_case if op == "ce" else _jsd_case
    case = [x.clone() for x in make(1, dtype, "cuda")]
    proj = torch.nn.Linear(H, H, device = "cuda", dtype = dtype)

    def fwd(x, *rest):
        hidden = proj(x)
        if op == "ce":
            w, b, t = rest
            return fl.fused_linear_cross_entropy(hidden, w, b, t, divisor = 7.0)
        w, b, th, tw, tb, mask = rest
        return fl.fused_linear_jsd(hidden, w, th, tw, mask = mask, beta = 0.5, student_bias = b, teacher_bias = tb)

    counter = torch._dynamo.testing.CompileCounter()
    compiled_eager = torch.compile(fwd, backend = counter, fullgraph = True)
    compiled_eager(*case).backward()
    assert counter.frame_count == 1
    compiled = torch.compile(fwd, mode = "reduce-overhead", fullgraph = True)
    before = fl.CALLS.get("fixed_rows", 0)
    for seed in (2, 3, 4, 5):
        new = [x.clone() for x in make(seed, dtype, "cuda")]
        proj.zero_grad(set_to_none = True)
        loss = compiled(*new)
        loss.backward()
        got = (loss.detach().clone(), proj.weight.grad.clone())
        proj.zero_grad(set_to_none = True)
        ref = fwd(*new)
        ref.backward()
        torch.testing.assert_close(got[0], ref.detach(), rtol = 1e-4, atol = 1e-5)
        torch.testing.assert_close(got[1].float(), proj.weight.grad.float(), rtol = 2e-2, atol = 2e-3)
    assert fl.CALLS.get("fixed_rows", 0) > before  # recorded under capture
