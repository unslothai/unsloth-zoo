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

# GRPO-family loss with the gradient taken in the forward, against an fp64 transcription of TRL 1.15
# GRPOTrainer._compute_loss (token-level importance sampling) differentiated by autograd.
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
import unsloth_zoo.fused_losses.fused_grpo as fg  # noqa: E402
import unsloth_zoo.fused_losses.fused_linear as fl  # noqa: E402

B, T, H, V = 3, 11, 24, 37


def _case(seed, temperature, with_bias):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, H, generator = g)
    W = torch.randn(V, H, generator = g) * 0.6
    bias = torch.randn(V, generator = g) * 0.3 if with_bias else None
    tgt = torch.randint(0, V, (B, T), generator = g)
    mask = torch.rand(B, T, generator = g) > 0.25
    mask[:, 0] = True
    mask[1, 5:] = False
    adv = torch.randn(B, generator = g)
    adv[2] = 0.0
    z = (h.double() @ W.double().t() + (bias.double() if bias is not None else 0)) / temperature
    lp = torch.log_softmax(z, -1).gather(-1, tgt[..., None])[..., 0]
    # Old / ref log-probs near the current ones so ratios straddle the clip range; a few rows exactly at 1.
    old = (lp + 0.35 * torch.randn(B, T, generator = g, dtype = torch.float64)).float()
    old[0, :3] = lp[0, :3].float()
    ref = (lp + 0.3 * torch.randn(B, T, generator = g, dtype = torch.float64)).float()
    isr = torch.rand(B, T, generator = g) * 0.5 + 0.75
    return h, W, bias, tgt, mask, adv, old, ref, isr


def _trl_loss(h, W, bias, tgt, mask, adv, old, ref, isr, cfg, temperature):
    z = (h @ W.t() + (bias if bias is not None else 0)) / temperature
    logp_all = torch.log_softmax(z, -1)
    per_token_logps = logp_all.gather(-1, tgt[..., None])[..., 0]
    entropies = -(logp_all.exp() * logp_all).sum(-1)
    m = mask.to(per_token_logps.dtype)
    advantages = adv.to(per_token_logps.dtype).unsqueeze(1)
    old = old.to(per_token_logps.dtype)
    coef_1 = torch.exp(per_token_logps - old)
    lt = cfg["loss_type"]
    if cfg["beta"] != 0.0:
        r = ref.to(per_token_logps.dtype)
        per_token_kl = torch.exp(r - per_token_logps) - (r - per_token_logps) - 1
        if cfg["bias_kl"]:
            per_token_kl = per_token_kl * coef_1
    if lt == "cispo":
        per_token_loss = -torch.clamp(coef_1, max = cfg["eps_high"]).detach() * advantages * per_token_logps
    elif lt == "sapo":
        temps = torch.where(advantages > 0, 1.0, 1.05)
        per_token_loss = -(torch.sigmoid(temps * (coef_1 - 1)) * 4 / temps) * advantages
    else:
        coef_2 = torch.clamp(coef_1, 1 - cfg["eps_low"], 1 + cfg["eps_high"])
        if cfg["delta"]:
            coef_1 = torch.clamp(coef_1, max = cfg["delta"])
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
    if cfg["isr"]:
        per_token_loss = per_token_loss * isr.to(per_token_logps.dtype)
    if cfg["beta"] != 0.0:
        per_token_loss = per_token_loss + cfg["beta"] * per_token_kl
    ga = 2.0
    if lt in ("grpo", "sapo"):
        loss = ((per_token_loss * m).sum(-1) / m.sum(-1).clamp(min = 1.0)).mean() / ga
    elif lt == "bnpo":
        loss = (per_token_loss * m).sum() / m.sum().clamp(min = 1.0) / ga
    elif lt == "dr_grpo":
        loss = (per_token_loss * m).sum() / (per_token_loss.size(0) * 16) / ga
    elif lt in ("cispo", "dapo"):
        loss = (per_token_loss * m).sum() / (torch.tensor(20.0, dtype = m.dtype) / 1 * ga / 2)
    else:  # luspo
        loss = (per_token_loss * m).sum(-1).mean() / ga
    if cfg["ent"]:
        loss = loss - cfg["ent"] * ((entropies * m).sum() / m.sum().clamp(min = 1.0) / ga)
    return loss


CFGS = [
    dict(loss_type = lt, beta = beta, bias_kl = bkl, eps_low = 0.2, eps_high = 0.28 if lt != "cispo" else 4.0,
         delta = delta, isr = isr, ent = ent)
    for lt, beta, bkl, delta, isr, ent in [
        ("grpo", 0.0, False, None, False, 0.0),
        ("grpo", 0.04, False, None, False, 0.01),
        ("grpo", 0.04, True, 1.5, True, 0.0),
        ("bnpo", 0.0, False, None, True, 0.02),
        ("dr_grpo", 0.1, False, None, False, 0.0),
        ("dapo", 0.0, False, 1.2, False, 0.01),
        ("luspo", 0.0, False, None, False, 0.0),
        ("cispo", 0.0, False, None, False, 0.0),
        ("cispo", 0.04, False, None, True, 0.01),
        ("sapo", 0.0, False, None, False, 0.0),
        ("sapo", 0.02, True, None, False, 0.01),
    ]
]


def _run_op(h, W, bias, tgt, mask, adv, old, ref, isr, cfg, temperature, rows):
    return fg.fused_linear_grpo_loss(
        h, W, tgt, mask = mask, advantages = adv, old_logps = old, loss_type = cfg["loss_type"], bias = bias,
        ref_logps = ref, beta = cfg["beta"], use_bias_correction_kl = cfg["bias_kl"], epsilon_low = cfg["eps_low"],
        epsilon_high = cfg["eps_high"], delta = cfg["delta"], importance_sampling_ratio = isr if cfg["isr"] else None,
        entropy_coef = cfg["ent"], num_items_in_batch = 20.0, max_completion_length = 16,
        gradient_accumulation_steps = 2, steps_per_generation = 2, temperature = temperature, rows_per_chunk = rows,
    )


@pytest.mark.parametrize("cfg", CFGS, ids = lambda c: f"{c['loss_type']}-b{c['beta']}-e{c['ent']}-d{c['delta']}")
@pytest.mark.parametrize("temperature,with_bias,head", [(1.0, False, False), (0.7, True, True), (1.3, False, True)])
def test_matches_trl_fp64(cfg, temperature, with_bias, head):
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(0, temperature, with_bias)
    hr = h.double().requires_grad_()
    Wr = W.double().requires_grad_(head)
    br = bias.double().requires_grad_(head) if bias is not None else None
    ref_loss = _trl_loss(hr, Wr, br, tgt, mask, adv, old, ref, isr, cfg, temperature)
    ins = [hr] + ([Wr] if head else []) + ([br] if head and br is not None else [])
    ref_grads = torch.autograd.grad(ref_loss, ins)
    for rows in (0, 7):  # one chunk / several chunks
        hh = h.clone().requires_grad_()
        WW = W.clone().requires_grad_(head)
        bb = bias.clone().requires_grad_(head) if bias is not None else None
        before = fl.CALLS.get("grpo_fwd_grad", 0)
        loss, lp, en = _run_op(hh, WW, bb, tgt, mask, adv, old, ref, isr, cfg, temperature, rows)
        assert fl.CALLS.get("grpo_fwd_grad", 0) == before + 1
        outs = [hh] + ([WW] if head else []) + ([bb] if head and bb is not None else [])
        grads = torch.autograd.grad(loss * 3.0, outs)
        torch.testing.assert_close(loss.double(), ref_loss.detach(), rtol = 2e-5, atol = 1e-6)
        for g, r in zip(grads, ref_grads):
            torch.testing.assert_close(g.double(), 3.0 * r, rtol = 2e-4, atol = 2e-6)


@pytest.mark.parametrize("cfg", CFGS, ids = lambda c: f"{c['loss_type']}-b{c['beta']}-e{c['ent']}-d{c['delta']}")
def test_kill_switch_path_matches(monkeypatch, cfg):
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(1, 1.0, False)
    outs = []
    for on in (True, False):
        monkeypatch.setattr(fg, "GRPO_FWD_GRAD", on)
        hh = h.clone().requires_grad_()
        loss, lp, en = _run_op(hh, W, None, tgt, mask, adv, old, ref, isr, cfg, 1.0, 0)
        (g,) = torch.autograd.grad(loss, [hh])
        outs.append((loss.detach(), lp * mask, en * mask, g))  # the log-prob op also fills masked-out rows
    for a, b in zip(*outs):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


def test_no_grad_returns_loss_and_metrics_only():
    cfg = CFGS[1]
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(2, 1.0, False)
    with torch.no_grad():
        loss, lp, en = _run_op(h.requires_grad_(), W, None, tgt, mask, adv, old, ref, isr, cfg, 1.0, 0)
    assert loss.requires_grad is False and torch.isfinite(loss)
    assert (lp[~mask] == 0).all() and (en[~mask] == 0).all()


def test_compiled_fullgraph_one_frame():
    cfg = CFGS[1]
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(3, 1.0, False)
    torch._dynamo.reset()

    def f(hh):
        return _run_op(hh, W, None, tgt, mask, adv, old, ref, isr, cfg, 1.0, 0)[0]
    hh = h.clone().requires_grad_()
    eager = torch.autograd.grad(f(hh), [hh])[0]
    from torch._dynamo.utils import counters
    counters.clear()
    cf = torch.compile(f, fullgraph = True, backend = "aot_eager")
    hc = h.clone().requires_grad_()
    got = torch.autograd.grad(cf(hc), [hc])[0]
    torch.testing.assert_close(got, eager, rtol = 1e-6, atol = 1e-7)
    assert sum(counters["graph_break"].values()) == 0


def test_rejects_unknown_loss_type():
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(4, 1.0, False)
    with pytest.raises(ValueError):
        fg.fused_linear_grpo_loss(h, W, tgt, mask = mask, advantages = adv, old_logps = old, loss_type = "vespo")


def test_bf16_head_grad_accumulates_in_fp32():
    cfg = CFGS[1]
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(5, 1.0, False)
    Wr = W.bfloat16().double().requires_grad_()
    ref_loss = _trl_loss(h.bfloat16().double(), Wr, None, tgt, mask, adv, old, ref, isr, cfg, 1.0)
    (gw_ref,) = torch.autograd.grad(ref_loss, [Wr])
    errs = {}
    for fp32 in (True, False):
        WW = W.bfloat16().requires_grad_()
        loss = fg.fused_linear_grpo_loss(
            h.bfloat16(), WW, tgt, mask = mask, advantages = adv, old_logps = old, ref_logps = ref, beta = cfg["beta"],
            epsilon_high = cfg["eps_high"], entropy_coef = cfg["ent"], gradient_accumulation_steps = 2,
            rows_per_chunk = 2, fp32_head_grad = fp32,
        )[0]
        (gw,) = torch.autograd.grad(loss, [WW])
        assert gw.dtype == torch.bfloat16
        errs[fp32] = ((gw.double() - gw_ref).norm() / gw_ref.norm()).item()
    assert errs[True] <= errs[False] and errs[True] < 1e-2


@pytest.mark.parametrize("kind", ["grpo", "dapo", "cispo", "sapo"])
def test_opcheck(kind, monkeypatch):
    # Eager backward scales the saved gradients (op outputs) in place; opcheck compares outputs after backward.
    monkeypatch.setattr(fl, "_KEEP_GRAPH", None)
    h, W, bias, tgt, mask, adv, old, ref, isr = _case(5, 0.9, True)
    w_tok, e_tok = fg.grpo_token_weights(mask, kind, num_items_in_batch = mask.sum(), max_completion_length = T,
                                         entropy_coef = 0.01)
    flat = lambda x: x.reshape(-1).contiguous()  # noqa: E731
    W = W.requires_grad_(True)
    b = fl._head_bias(bias.requires_grad_(True), W)
    torch.library.opcheck(
        getattr(torch.ops.unsloth_zoo, fg._OP).default,
        (h.reshape(-1, H).requires_grad_(True), W, b, flat(tgt), flat(mask), flat(adv[:, None].expand(B, T)),
         flat(old), flat(ref), flat(isr), flat(w_tok), flat(e_tok), torch.ones(()), fg._KIND[kind], 0.2, 0.28, 0.0,
         0.04, 1.0, 1.05, False, 0.9, 0.0, 0.0, 0.0, torch.float32, True, True, True, 8, True),
    )
