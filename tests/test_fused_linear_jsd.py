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

"""fused_linear_jsd (two-head generalized JSD for GKD) against TRL's GKDTrainer.generalized_jsd_loss in float64."""

import pytest
import torch
import torch.nn.functional as F

import unsloth_zoo.fused_losses.fused_linear as fl

B, S, H, HT, V = 2, 11, 16, 24, 50


def trl_generalized_jsd_loss(student_logits, teacher_logits, labels = None, beta = 0.5, temperature = 1.0,
                             num_items_in_batch = None):
    """TRL 1.15.0 trl/experimental/gkd/gkd_trainer.py:238 (reduction = "batchmean"), verbatim math."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim = -1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim = -1)
    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction = "none", log_target = True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction = "none", log_target = True)
    else:
        beta = torch.tensor(beta, dtype = student_log_probs.dtype, device = student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim = 0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction = "none", log_target = True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction = "none", log_target = True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]
    if num_items_in_batch is not None:
        return jsd.sum() / num_items_in_batch
    denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
    return jsd.sum() / denom


def _real_trl_jsd():
    try:
        from trl.experimental.gkd.gkd_trainer import GKDTrainer
    except Exception:
        try:
            from trl.trainer.gkd_trainer import GKDTrainer
        except Exception:
            return None
    return GKDTrainer.generalized_jsd_loss


def _transform(z, scale, cap):
    z = z * scale
    if cap:
        z = cap * torch.tanh(z / cap)
    return z


def _case(seed, device = "cpu", dtype = torch.float32, bias = False):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(B, S, H, generator = g)
    ws = torch.randn(V, H, generator = g) / 3
    ht = torch.randn(B, S, HT, generator = g)
    wt = torch.randn(V, HT, generator = g) / 3
    bs = torch.randn(V, generator = g) / 4 if bias else None
    bt = torch.randn(V, generator = g) / 4 if bias else None
    labels = torch.randint(0, V, (B, S), generator = g)
    labels[0, :4] = -100
    labels[1, -3:] = -100
    to = lambda x: None if x is None else x.to(device = device, dtype = dtype)  # noqa: E731
    return to(hs), to(ws), to(bs), to(ht), to(wt), to(bt), labels.to(device)


def _oracle(hs, ws, bs, ht, wt, bt, labels, beta, temperature, s_tf = (1.0, 0.0), t_tf = (1.0, 0.0), n_items = None,
            fn = trl_generalized_jsd_loss, head_grad = False):
    hd = hs.detach().double().requires_grad_(True)
    wd = ws.detach().double().requires_grad_(head_grad)
    bd = bs.detach().double().requires_grad_(head_grad) if bs is not None else None
    zs = hd @ wd.t() + (bd if bd is not None else 0)
    zt = ht.double() @ wt.double().t() + (bt.double() if bt is not None else 0)
    zs, zt = _transform(zs, *s_tf), _transform(zt, *t_tf)
    kw = {} if n_items is None else {"num_items_in_batch": n_items}
    loss = fn(zs, zt, labels = labels, beta = beta, temperature = temperature, **kw)
    loss.backward()
    return loss.detach(), hd.grad, (wd.grad if head_grad else None), (bd.grad if head_grad and bd is not None else None)


@pytest.mark.parametrize("temperature", [1.0, 0.9])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 0.1])
@pytest.mark.parametrize("variant", ["plain", "bias_softcap", "head_grad", "num_items"])
def test_jsd_matches_trl_fp64(beta, temperature, variant):
    hs, ws, bs, ht, wt, bt, labels = _case(1, dtype = torch.float64, bias = variant != "plain")
    s_tf = (1.3, 4.0) if variant == "bias_softcap" else (1.0, 0.0)
    t_tf = (0.8, 3.0) if variant == "bias_softcap" else (1.0, 0.0)
    head_grad = variant == "head_grad"
    n_items = 37 if variant == "num_items" else None
    ref, gh, gw, gb = _oracle(hs, ws, bs, ht, wt, bt, labels, beta, temperature, s_tf, t_tf, n_items,
                              head_grad = head_grad)
    h = hs.clone().requires_grad_(True)
    w = ws.clone().requires_grad_(head_grad)
    b = bs.clone().requires_grad_(head_grad) if bs is not None else None
    before = fl.CALLS["jsd"]
    loss = fl.fused_linear_jsd(
        h, w, ht, wt, labels = labels, beta = beta, temperature = temperature, student_bias = b, teacher_bias = bt,
        student_logit_scale = s_tf[0], student_logit_softcapping = s_tf[1],
        teacher_logit_scale = t_tf[0], teacher_logit_softcapping = t_tf[1],
        num_items_in_batch = n_items, rows_per_chunk = 5,
    )
    assert fl.CALLS["jsd"] == before + 1
    loss.backward()
    torch.testing.assert_close(loss.double(), ref, rtol = 1e-6, atol = 1e-7)
    torch.testing.assert_close(h.grad, gh, rtol = 1e-6, atol = 1e-9)
    if head_grad:
        torch.testing.assert_close(w.grad, gw, rtol = 1e-6, atol = 1e-9)
        torch.testing.assert_close(b.grad, gb, rtol = 1e-6, atol = 1e-9)
    real = _real_trl_jsd()
    import inspect
    if real is not None and (n_items is None or "num_items_in_batch" in inspect.signature(real).parameters):
        ref2 = _oracle(hs, ws, bs, ht, wt, bt, labels, beta, temperature, s_tf, t_tf, n_items, fn = real)[0]
        torch.testing.assert_close(ref2, ref, rtol = 1e-12, atol = 1e-12)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_jsd_fp32_and_chunking(beta):
    hs, ws, bs, ht, wt, bt, labels = _case(2)
    ref, gh, _, _ = _oracle(hs, ws, None, ht, wt, None, labels, beta, 0.9)
    outs = []
    for rows in (1, 7, 0):
        h = hs.clone().requires_grad_(True)
        loss = fl.fused_linear_jsd(h, ws, ht, wt, labels = labels, beta = beta, temperature = 0.9,
                                   rows_per_chunk = rows)
        loss.backward()
        torch.testing.assert_close(loss.double(), ref, rtol = 2e-5, atol = 1e-6)
        torch.testing.assert_close(h.grad.double(), gh, rtol = 2e-4, atol = 1e-6)
        outs.append(h.grad)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_peaked_rows_match_fp64(beta):
    """Top logit leading the rest by 20..80 (Liger #1555: fp32 softmax rounds to 1 past ~16.6): still exact in fp32."""
    g = torch.Generator().manual_seed(11)
    n = 12
    ws = torch.eye(V)  # logits == hidden
    hs = torch.randn(n, V, generator = g)
    ht = torch.randn(n, V, generator = g)
    lead = torch.tensor([20.0, 40.0, 80.0]).repeat(n // 3)
    peak = torch.randint(0, V, (n,), generator = g)
    hs[torch.arange(n), peak] += lead
    # Teacher: mildly peaked on the student's token on odd rows, sharply peaked on another token on even rows,
    # so the true gradient is O(1) (both collapsed onto one token leaves only rounding noise to compare).
    ht[torch.arange(1, n, 2), peak[1::2]] += 3.0
    ht[torch.arange(0, n, 2), (peak[::2] + 1) % V] += lead[::2]
    labels = torch.randint(0, V, (n,), generator = g)
    ref, gh, _, _ = _oracle(hs, ws, None, ht, ws, None, labels, beta, 1.0)
    h = hs.clone().requires_grad_(True)
    loss = fl.fused_linear_jsd(h, ws, ht, ws, labels = labels, beta = beta)
    loss.backward()
    torch.testing.assert_close(loss.double(), ref, rtol = 1e-5, atol = 1e-6)
    # Peaked rows have exponentially small true gradients for beta > 0, so compare at the fp32 noise floor of O(1)
    # logits; the beta = 0 rows keep O(0.1) gradients and must match tightly.
    err = (h.grad.double() - gh).norm(dim = -1)
    assert torch.isfinite(h.grad).all()
    assert (err <= 1e-6 + 1e-5 * gh.norm(dim = -1)).all(), err
    # The collapse symptom: with W = I the hidden gradient is d loss / d logits, which must sum to zero per row.
    assert h.grad.double().sum(-1).abs().max().item() < 1e-6


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_unlabelled_3d_divides_by_batch_as_trl(beta):
    """No labels, mask or count: divide by batch (TRL >= 0.24 batchmean); a mask keeps the kept-row count."""
    g = torch.Generator().manual_seed(21)
    w = torch.randn(V, H, generator = g) / 3
    wt = torch.randn(V, HT, generator = g) / 3
    hs = torch.randn(2, 3, H, generator = g)
    ht = torch.randn(2, 3, HT, generator = g)
    fns = [trl_generalized_jsd_loss] + ([_real_trl_jsd()] if _real_trl_jsd() is not None else [])
    for fn in fns:
        zs, zt = hs.double() @ w.double().t(), ht.double() @ wt.double().t()
        ref3 = fn(zs, zt, beta = beta)
        ref2 = fn(zs.reshape(6, V), zt.reshape(6, V), beta = beta)
        torch.testing.assert_close(fl.fused_linear_jsd(hs, w, ht, wt, beta = beta).double(), ref3, rtol = 1e-5, atol = 1e-7)
        torch.testing.assert_close(fl.fused_linear_jsd(hs.reshape(6, H), w, ht.reshape(6, HT), wt, beta = beta).double(),
                                   ref2, rtol = 1e-5, atol = 1e-7)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    masked = fl.fused_linear_jsd(hs, w, ht, wt, beta = beta, mask = mask)
    summed = fl.fused_linear_jsd(hs, w, ht, wt, beta = beta, mask = mask, num_items_in_batch = 1)
    torch.testing.assert_close(masked, summed / 3)


def test_fully_masked_is_finite_zero():
    hs, ws, _, ht, wt, _, labels = _case(3)
    h = hs.clone().requires_grad_(True)
    loss = fl.fused_linear_jsd(h, ws, ht, wt, labels = torch.full_like(labels, -100), beta = 0.5)
    loss.backward()
    assert loss.item() == 0.0 and torch.count_nonzero(h.grad) == 0


def test_scaling_and_grad_output():
    hs, ws, _, ht, wt, _, labels = _case(4)
    ref, gh, _, _ = _oracle(hs, ws, None, ht, wt, None, labels, 0.5, 1.0)
    h = hs.clone().requires_grad_(True)
    loss = fl.fused_linear_jsd(h, ws, ht, wt, labels = labels, beta = 0.5, scaling = 1024.0)
    (loss * 3).backward()
    torch.testing.assert_close(loss.double(), ref, rtol = 2e-5, atol = 1e-6)
    torch.testing.assert_close(h.grad.double(), 3 * gh, rtol = 2e-4, atol = 1e-6)


def test_compiles_without_graph_breaks():
    import torch._dynamo as dynamo

    hs, ws, _, ht, wt, _, labels = _case(5)

    def f(h, w, t, wt, labels):
        return fl.fused_linear_jsd(h, w, t, wt, labels = labels, beta = 0.5, temperature = 0.9) * 2

    explain = dynamo.explain(f)(hs.requires_grad_(True), ws, ht, wt, labels)
    assert explain.graph_break_count == 0, explain.break_reasons
    dynamo.reset()
    h = hs.detach().clone().requires_grad_(True)
    out = torch.compile(f, fullgraph = True, backend = "aot_eager")(h, ws, ht, wt, labels)
    out.backward()
    ref, gh, _, _ = _oracle(hs, ws, None, ht, wt, None, labels, 0.5, 0.9)
    torch.testing.assert_close(out.double(), 2 * ref, rtol = 2e-5, atol = 1e-6)
    torch.testing.assert_close(h.grad.double(), 2 * gh, rtol = 2e-4, atol = 1e-6)


@pytest.mark.parametrize("temperature", [None, 1.0, 0.9])
def test_compiles_dynamic_fullgraph(temperature):
    """dynamic = True makes temperature a SymFloat; validating it must not break the graph."""
    kw = {} if temperature is None else {"temperature": temperature}

    def f(h, w, t, wt, labels):
        return fl.fused_linear_jsd(h, w, t, wt, labels = labels, beta = 0.5, **kw)

    torch._dynamo.reset()
    compiled = torch.compile(f, fullgraph = True, dynamic = True, backend = "aot_eager")
    hs, ws, _, ht, wt, _, labels = _case(6)
    for n in (5, 9, 13):
        h = hs[:, :n].detach().clone().requires_grad_(True)
        out = compiled(h, ws, ht[:, :n], wt, labels[:, :n])
        out.backward()
        ref, gh, _, _ = _oracle(h, ws, None, ht[:, :n], wt, None, labels[:, :n], 0.5, temperature or 1.0)
        torch.testing.assert_close(out.double(), ref, rtol = 2e-5, atol = 1e-6)
        torch.testing.assert_close(h.grad.double(), gh, rtol = 2e-4, atol = 1e-6)
    torch._dynamo.reset()


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_rejects_bad_temperature(temperature):
    hs, ws, _, ht, wt, _, labels = _case(7)
    with pytest.raises(ValueError, match = "temperature must be finite"):
        fl.fused_linear_jsd(hs, ws, ht, wt, labels = labels, temperature = temperature)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("tf", [(1.0, 0.0, 1.0, 0.0), (1.3, 4.0, 0.8, 3.0)])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_triton_body_matches_compiled_math(beta, tf):
    """_jsd_triton against _jsd_body (fp64) on peaked and plain rows, odd vocab."""
    if not fl._use_triton(torch.device("cuda")):
        pytest.skip("Triton unavailable")
    g = torch.Generator().manual_seed(4)
    Vb = 10007
    s = torch.randn(7, Vb, generator = g) * 3
    t = torch.randn(7, Vb, generator = g) * 3
    s[0, 5] += 60.0
    t[1, 9] += 40.0
    f = torch.tensor(0.37)
    row, dz = fl._jsd_triton(s.cuda(), t.cuda(), f.cuda(), beta, 1 / 0.9, *tf, True, torch.float32)
    row2, dz2 = fl._jsd_body(s.double(), t.double(), f.double(), beta, 1 / 0.9, *tf, True, torch.float64)
    torch.testing.assert_close(row.cpu().double(), row2, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(dz.cpu().double(), dz2, rtol = 1e-4, atol = 1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
def test_triton_chunk_respects_row_cap(monkeypatch):
    """A negative rows_per_chunk caps the Triton chunk too, not only the eager / compiled one."""
    if not fl._use_triton(torch.device("cuda")):
        pytest.skip("Triton unavailable")
    seen = []
    real = fl._jsd_triton

    def spy(s, *args):
        seen.append(s.shape[0])
        return real(s, *args)

    monkeypatch.setattr(fl, "_jsd_triton", spy)
    g = torch.Generator().manual_seed(5)
    h = torch.randn(300, 16, generator = g).cuda().requires_grad_()
    ht = torch.randn(300, 24, generator = g).cuda()
    w = (torch.randn(151936, 16, generator = g) * 0.1).cuda()
    wt = (torch.randn(151936, 24, generator = g) * 0.1).cuda()
    fl.fused_linear_jsd(h, w, ht, wt, beta = 0.5, rows_per_chunk = -32).backward()
    assert seen and max(seen) <= 32, seen


def test_fused_body_matches_eager_chunk():
    """_jsd_body (the compiled CUDA form) and _jsd_chunk (eager, in place) are the same math."""
    g = torch.Generator().manual_seed(9)
    for beta in (0.0, 0.3, 1.0):
        s = torch.randn(6, V, generator = g, dtype = torch.float64) * 3
        t = torch.randn(6, V, generator = g, dtype = torch.float64) * 3
        row, dz = fl._jsd_body(s, t, torch.tensor(1.0, dtype = torch.float64), beta, 1 / 0.9, 1.0, 0.0, 1.0, 0.0,
                               True, torch.float64)
        ls = torch.log_softmax(s / 0.9, -1)
        lt = torch.log_softmax(t / 0.9, -1)
        row2, dz2 = fl._jsd_chunk(ls, lt, beta, True)
        torch.testing.assert_close(row, row2)
        torch.testing.assert_close(dz, dz2 / 0.9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("compiled", ["triton", "inductor", "eager"])
@pytest.mark.parametrize("temperature", [1.0, 0.9])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_bf16_error_not_worse_than_trl_bf16(beta, temperature, compiled, monkeypatch):
    calls = []
    if compiled != "eager" and not fl._HAS_TRITON:
        pytest.skip("no Triton installed (AMD Windows wheels): Triton and Inductor bodies cannot run")
    if compiled == "triton":
        assert fl._use_triton(torch.device("cuda")), "Triton JSD path unavailable"
        real = fl._jsd_triton
        monkeypatch.setattr(fl, "_jsd_triton", lambda *a: calls.append(1) or real(*a))
    else:
        monkeypatch.setattr(fl, "_use_triton", lambda device: False)
    if compiled == "inductor":
        assert fl._jsd_compiled(torch.device("cuda")) is not None, "compiled JSD body unavailable"
    elif compiled == "eager":
        monkeypatch.setattr(fl, "_jsd_compiled", lambda device: None)
    """TRL's dense path on bf16 logits (the model's own lm_head output) vs this op, both against the fp64 oracle."""
    g = torch.Generator().manual_seed(7)
    N, Hs, Ht, Vb = 384, 256, 320, 32000
    hs = torch.randn(N, Hs, generator = g).cuda().bfloat16()
    ws = (torch.randn(Vb, Hs, generator = g) / 8).cuda().bfloat16()
    ht = torch.randn(N, Ht, generator = g).cuda().bfloat16()
    wt = (torch.randn(Vb, Ht, generator = g) / 8).cuda().bfloat16()
    labels = torch.randint(0, Vb, (N,), generator = g).cuda()
    labels[::5] = -100
    ref, gh, _, _ = _oracle(hs, ws, None, ht, wt, None, labels, beta, temperature)
    h = hs.clone().requires_grad_(True)
    trl_loss = trl_generalized_jsd_loss(h @ ws.t(), ht @ wt.t(), labels = labels, beta = beta,
                                        temperature = temperature)
    trl_loss.backward()
    trl_gh = h.grad
    h2 = hs.clone().requires_grad_(True)
    loss = fl.fused_linear_jsd(h2, ws, ht, wt, labels = labels, beta = beta, temperature = temperature)
    loss.backward()
    rel = lambda a, b: ((a.double() - b).norm() / b.norm()).item()  # noqa: E731
    ours_l, trl_l = abs(loss.double() - ref).item(), abs(trl_loss.double() - ref).item()
    ours_g, trl_g = rel(h2.grad, gh), rel(trl_gh, gh)
    print(f"compiled={compiled} beta={beta} T={temperature} loss err ours {ours_l:.3e} trl {trl_l:.3e}; grad rel ours {ours_g:.3e} "
          f"trl {trl_g:.3e}")
    assert ours_l <= max(trl_l, 1e-6 * abs(ref.item()))
    assert ours_g <= trl_g
    assert compiled != "triton" or calls, "Triton JSD path did not run"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "two CUDA devices")
@pytest.mark.parametrize("teacher_device", ["cuda:1", "cuda:0"])
def test_triton_runs_on_the_operand_device_not_the_current_one(teacher_device):
    """Student head on cuda:1 while cuda:0 is current: Triton must launch there, not fall back to eager."""
    import warnings
    dev = torch.device("cuda:1")
    if not fl._use_triton(dev):
        pytest.skip("Triton JSD path unavailable on cuda:1")
    g = torch.Generator().manual_seed(0)
    h = torch.randn(8, 16, generator = g)
    w = torch.randn(65, 16, generator = g)
    t = torch.randn(8, 12, generator = g)
    wt = torch.randn(65, 12, generator = g)
    with torch.cuda.device(1):
        hr = h.to(dev).requires_grad_()
        ref = fl.fused_linear_jsd(hr, w.to(dev), t.to(teacher_device), wt.to(teacher_device), beta = 0.5)
        ref.backward()
    torch.cuda.set_device(0)
    fl._TRITON_BROKEN.discard(dev)
    hx = h.to(dev).requires_grad_()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the eager fallback warns
        loss = fl.fused_linear_jsd(hx, w.to(dev), t.to(teacher_device), wt.to(teacher_device), beta = 0.5)
        loss.backward()
    torch.cuda.synchronize(dev)
    assert dev not in fl._TRITON_BROKEN
    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(loss.detach(), ref.detach())
    torch.testing.assert_close(hx.grad, hr.grad)
