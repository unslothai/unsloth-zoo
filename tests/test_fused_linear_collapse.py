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

"""Softmax collapse (arXiv 2610.31642): rows whose target leads the rest by > ~16.6 round p_t to 1 in fp32, so
1 - p_t and lse - z_t read 0 while the other classes keep their gradient. fp64 oracle written as
softplus(lse_others - z_t) (plain fp64 log_softmax collapses too past ~36.7), every route."""

import json
import os

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy, fused_linear_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

GAPS = (17.0, 20.0, 30.0, 50.0)
H = 128


def _transform(x, cap):
    return cap * torch.tanh(x / cap) if cap else x


def _inputs(V, dtype, T, cap, seed = 0, controls = 4):
    """Rows whose final logit (after softcap and temperature) leads lse of the rest by GAPS (as far as the softcap
    allows), plus random control rows. Peaked rows own their target's head row: w_t = beta h / |h|^2."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, H, generator = g, dtype = torch.float64) / H**0.5
    n = len(GAPS) * 2 + controls
    t = torch.randperm(V, generator = g)[:n]
    h = torch.randn(n, H, generator = g, dtype = torch.float64)
    h[: 2 * len(GAPS)] /= h[: 2 * len(GAPS)].norm(dim = 1, keepdim = True)
    h[2 * len(GAPS):] *= 2.0
    for i in range(2 * len(GAPS)):
        zo = _transform(h[i] @ w.t(), cap) / T
        zo[t[i]] = -float("inf")
        need = GAPS[i % len(GAPS)] + float(torch.logsumexp(zo, 0))
        # Invert z_t = cap tanh(x / cap) / T; past what the cap allows, take x = 2.5 cap.
        x = (cap * float(torch.atanh(torch.tensor(need * T / cap))) if need * T < 0.98 * cap else 2.5 * cap) \
            if cap else need * T
        w[t[i]] = x * h[i] / h[i].norm() ** 2
    return h.cuda().to(dtype), w.cuda().to(dtype), t.cuda()


def _oracle(h, w, t, T, cap):
    """fp64 logp, d logp / dh, entropy, d entropy / dh, without collapse."""
    hd, wd = h.double(), w.double()
    x = hd @ wd.t()
    z = _transform(x, cap) / T
    d = (1 - torch.tanh(x / cap) ** 2 if cap else torch.ones_like(x)) / T
    zo = z.scatter(1, t[:, None], -float("inf"))
    lse_o = torch.logsumexp(zo, 1)
    zt = z.gather(1, t[:, None])[:, 0]
    lp = -torch.nn.functional.softplus(lse_o - zt)
    lse = zt - lp
    logp_all = (z - lse[:, None]).scatter(1, t[:, None], lp[:, None])
    p = logp_all.exp()
    # d logp / dz = onehot - p, target entry 1 - p_t = sum of the others.
    dz = (-p).scatter(1, t[:, None], torch.exp(lse_o - lse)[:, None])
    ent = -(p * logp_all).sum(1)
    dent = -p * (logp_all + ent[:, None])
    gap = lse_o.neg().add(zt)
    return dict(lp = lp, gh = (dz * d) @ wd, ent = ent, gent = (dent * d) @ wd, gap = gap)


def _row_rel(x, y):
    return ((x.double() - y).norm(dim = 1) / y.norm(dim = 1).clamp_min(1e-300))


def _routes(V, dtype):
    half = dtype in (torch.float16, torch.bfloat16)
    tiled = fl._tiled_vocab(V)
    r = [("lp_chunked", dict(rows = 1000, frozen = True)),
         ("lp_trainable_head", dict(rows = 0, frozen = False)),
         ("lp_trainable_head_chunked", dict(rows = 1000, frozen = False)),
         ("lp_recompute", dict(rows = 0, frozen = True, fwd_grad = False)),
         ("lp_entropy_grad", dict(rows = 0, frozen = True, ent = True)),
         ("lp_entropy_lazy", dict(rows = 0, frozen = True, ent = "lazy")),
         ("ce_chunked", dict(rows = 1000, frozen = True, ce = True)),
         ("ce_trainable_head", dict(rows = 0, frozen = False, ce = True)),
         ("ce_gradscaler", dict(rows = 0, frozen = True, ce = True, scaling = 2.0**10))]
    if half and tiled:
        r += [("lp_flash", dict(rows = 0, frozen = True)), ("ce_flash", dict(rows = 0, frozen = True, ce = True))]
    return r


def _run(route, h, w, t, T, cap):
    hh = h.clone().requires_grad_(True)
    ww = w.clone().requires_grad_(not route.get("frozen", True))
    before = dict(fl.CALLS)
    if route.get("ce"):
        out = {}
        # Per-row losses: one call per row would hide the chunk routes, so take the gradient of the sum and the
        # loss from the logprob of each row through the CE (sum over rows, divisor 1).
        S = route.get("scaling")
        loss = fused_linear_cross_entropy(hh, ww, None, t, divisor = 1.0, logit_softcapping = cap,
                                          rows_per_chunk = route["rows"] or None, scaling = S)
        # GradScaler: backward of loss * S with the stored gradients scaled by S, divided out here in fp64.
        (gh,) = torch.autograd.grad(loss * S if S else loss, (hh,))
        gh = gh.double() / S if S else gh
        out["gh"] = -gh
        n_peak = 2 * len(GAPS)
        tp = torch.where(torch.arange(t.shape[0], device = t.device) < n_peak, t, torch.full_like(t, -100))
        with torch.no_grad():
            out["loss"] = fused_linear_cross_entropy(h, w, None, tp, divisor = 1.0, logit_softcapping = cap,
                                                     rows_per_chunk = route["rows"] or None).double()
    else:
        ent = route.get("ent")
        kw = dict(temperature = T, logit_softcapping = cap, rows_per_chunk = route["rows"] or None)
        if ent:
            kw.update(return_entropy = True, entropy_requires_grad = None if ent == "lazy" else True)
        res = fused_linear_logprobs(hh, ww, t, **kw)
        lp, e = res if ent else (res, None)
        (gh,) = torch.autograd.grad(lp.sum(), (hh,), retain_graph = e is not None)
        out = dict(lp = lp.detach().double(), gh = gh)
        if e is not None:
            out["ent"] = e.detach().double()
            (out["gent"],) = torch.autograd.grad(e.sum(), (hh,))
    out["calls"] = {k: fl.CALLS[k] - before.get(k, 0) for k in fl.CALLS if fl.CALLS[k] != before.get(k, 0)}
    return out


def measure(V, dtype, T, cap, routes = None):
    fl._FLASH_MIN_ROWS = 0
    h, w, t = _inputs(V, dtype, T, cap)
    ref = _oracle(h, w, t, T, cap)
    peaked = torch.arange(h.shape[0], device = "cuda") < 2 * len(GAPS)
    rows = []
    for name, route in _routes(V, dtype):
        if (routes and name not in routes) or (route.get("ce") and T != 1.0):
            continue
        fwd = fl._FORWARD_GRAD
        try:
            if route.get("fwd_grad") is False:
                fl._FORWARD_GRAD = False
            out = _run(route, h, w, t, T, cap)
        finally:
            fl._FORWARD_GRAD = fwd
        rel = _row_rel(out["gh"], ref["gh"])
        row = dict(route = name, V = V, dtype = str(dtype).split(".")[-1], T = T, cap = cap, calls = out["calls"],
                   gap = [round(float(x), 2) for x in ref["gap"][peaked]],
                   gh_rel_peaked = [float(x) for x in rel[peaked]], gh_rel_control = float(rel[~peaked].max()))
        # The same gradient's own rounding to the output dtype: what a collapse must beat to be visible.
        S = route.get("scaling") or 1.0
        rnd = _row_rel((ref["gh"] * S).to(dtype).double() / S, ref["gh"])[peaked]
        row["gh_rel_round"] = float(rnd.max())
        # Rows whose true gradient survives the rounding to the output dtype (fp16 flushes e^-gap without a scale).
        row["representable"] = [bool(x < 1e-2) for x in rnd]
        if "lp" in out:
            err = (out["lp"] - ref["lp"]).abs()
            row["lp_abs_peaked"] = [float(x) for x in err[peaked]]
            row["lp_rel_peaked"] = [float(x) for x in (err / ref["lp"].abs())[peaked]]
            row["lp_abs_control"] = float(err[~peaked].max())
        if "loss" in out:
            row["loss_abs"] = float((out["loss"] + ref["lp"][peaked].sum()).abs())
            row["loss_true_peaked"] = float(-ref["lp"][peaked].sum())
        if "ent" in out:
            row["ent_rel_peaked"] = [float(x) for x in ((out["ent"] - ref["ent"]).abs() / ref["ent"].abs())[peaked]]
            row["gent_rel_peaked"] = [float(x) for x in _row_rel(out["gent"], ref["gent"])[peaked]]
        rows.append(row)
    assert not fl._TRITON_BROKEN, f"Triton fell back to torch on {fl._TRITON_BROKEN}"
    return rows


def _dtypes():
    return [torch.float32, torch.float16] + ([torch.bfloat16] if torch.cuda.get_device_capability()[0] >= 8 else [])


# Per-row hidden-grad tolerance on rows whose true gradient survives the output dtype: the base reads 1 - p_t as 0
# (relative error ~1, or ~1e16 where an FMA makes log p_t positive); bf16 / fp16 limits are the half dz buffer.
_GH_TOL = {torch.float32: 1e-3, torch.float16: 0.15, torch.bfloat16: 2e-2}
# Half logits round a peaked row's z_t (30 to 60) by up to 0.125 in bf16 / 0.016 in fp16, which scales e^-gap alike.
_HALF_TOL = {torch.float16: 0.15, torch.bfloat16: 0.3}


@pytest.fixture
def accuracy(request, monkeypatch):
    # 1: fp32 logits (CUDA default), 2: half logits + exact fp32 target logit, 0: half logits (ROCm default).
    monkeypatch.setattr(fl, "_ACCURACY", request.param)
    monkeypatch.setattr(fl, "_FLASH_MIN_ROWS", 0)
    return request.param


def _check(rows, dtype, fp32_logits = True):
    for r in rows:
        tag = (r["route"], r["dtype"], r["V"], r["T"], r["cap"], r["gap"])
        gh = [g for g, ok in zip(r["gh_rel_peaked"], r["representable"]) if ok]
        tol = _GH_TOL[dtype] if fp32_logits else _HALF_TOL[dtype]
        assert max(gh, default = 0.0) < max(tol, 2.0 * r["gh_rel_control"]), (tag, r["gh_rel_peaked"])
        if "lp_rel_peaked" in r:
            assert max(r["lp_rel_peaked"]) < (1e-3 if fp32_logits else tol), (tag, r["lp_rel_peaked"])
        if "loss_abs" in r:
            assert r["loss_abs"] < (1e-3 if fp32_logits else tol) * r["loss_true_peaked"], (tag, r["loss_abs"])
        if "ent_rel_peaked" in r and fp32_logits:
            # Half logits round z by up to 2e-2 here, which moves an entropy of e^-gap * gap by a percent.
            assert max(r["ent_rel_peaked"]) < 1e-3, (tag, r["ent_rel_peaked"])
            if dtype == torch.float32:
                assert max(r["gent_rel_peaked"]) < 1e-3, (tag, r["gent_rel_peaked"])


@pytest.mark.parametrize("accuracy", ["1", "2", "0"], indirect = True)
@pytest.mark.parametrize("V", [1003, 40001, 151936])
@pytest.mark.parametrize("T,cap", [(1.0, 0.0), (0.7, 30.0)])
@pytest.mark.parametrize("triton_on", [True, False])
def test_peaked_rows_match_the_oracle(monkeypatch, accuracy, V, T, cap, triton_on):
    if not triton_on and (accuracy != "1" or V == 151936):
        pytest.skip("torch bodies: one logits mode and the two smaller vocabularies cover them")
    monkeypatch.setattr(fl, "_TRITON", triton_on)
    for dtype in _dtypes():
        if accuracy != "1" and dtype == torch.float32:
            continue
        _check(measure(V, dtype, T, cap), dtype, accuracy == "1" or dtype == torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_collapsed_logprob_is_negative_not_zero(dtype):
    h, w, t = _inputs(40001, dtype, 1.0, 0.0)
    with torch.no_grad():
        lp = fused_linear_logprobs(h, w, t).double()
    ref = _oracle(h, w, t, 1.0, 0.0)["lp"]
    peaked = slice(0, 2 * len(GAPS))
    assert (lp[peaked] < 0).all(), lp[peaked]
    assert float(((lp - ref).abs() / ref.abs())[peaked].max()) < 1e-3


def test_ordinary_rows_unchanged_vs_logsumexp():
    # Negative control: rows far from the cliff keep log_softmax to fp32 rounding (the rewrite changes nothing there).
    g = torch.Generator().manual_seed(3)
    h = torch.randn(64, H, generator = g).cuda()
    w = torch.randn(40001, H, generator = g).cuda() / H**0.5
    t = torch.randint(0, 40001, (64,), generator = g).cuda()
    with torch.no_grad():
        lp = fused_linear_logprobs(h, w, t).double()
    ref = (h.double() @ w.double().t()).log_softmax(-1).gather(1, t[:, None])[:, 0]
    assert float((lp - ref).abs().max()) < 2e-5


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--V", default = "1003,40001,151936")
    args = ap.parse_args()
    report = []
    for torch_path, acc in ((False, "1"), (True, "1"), (False, "2"), (False, "0")):
        fl._TRITON = not torch_path
        fl._ACCURACY = acc
        for V in [int(v) for v in args.V.split(",")]:
            for dtype in _dtypes():
                if acc != "1" and dtype == torch.float32:
                    continue
                for T, cap in ((1.0, 0.0), (0.7, 0.0), (1.0, 30.0), (0.7, 30.0)):
                    for r in measure(V, dtype, T, cap):
                        r["torch_path"] = torch_path
                        r["accuracy"] = acc
                        report.append(r)
                        print(json.dumps({k: (([round(x, 4) if abs(x) > 1e-3 else float(f"{x:.2e}") for x in v])
                                              if isinstance(v, list) else v) for k, v in r.items()}), flush = True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok = True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent = 1)
