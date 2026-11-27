"""GRPO loss_type="reinforce" + score centering (arXiv 2609.20807): fp64 oracle, verl parity, invariants."""
import math
import os
import types
from typing import Callable

import pytest
import torch

from unsloth_zoo import score_centering as sc  # noqa: E402
from unsloth_zoo.rl_replacements import UnslothEfficientGRPO, grpo_compute_loss  # noqa: E402

D64 = torch.float64


def _rand_logp(shape, V, gen, scale = 3.0):
    return torch.log_softmax(torch.randn(*shape, V, generator = gen, dtype = D64) * scale, dim = -1)


def _topk_head(logq, k):
    vals, ids = logq.topk(k, dim = -1)
    return ids, vals


TIS = ("token_truncate", None, 2.0)
ICEPOP = ("token_mask", 0.5, 2.0)


@pytest.mark.parametrize("rule", [None, TIS, ICEPOP])
@pytest.mark.parametrize("k", [8, 32, 1000])
def test_zero_gradient_when_q_equals_p(rule, k):
    gen = torch.Generator().manual_seed(0)
    V = 1000
    logits = torch.randn(64, V, generator = gen, dtype = D64, requires_grad = True)
    logp = torch.log_softmax(logits, -1)
    ids, logq = _topk_head(logp.detach(), k)
    wfn = sc.score_centering_weight_fn(*rule) if rule else sc.score_centering_weight_fn(None)
    corr, qm, pm = sc.score_centering_correction(logp.gather(-1, ids), logq, wfn)
    (g,) = torch.autograd.grad(corr.sum(), logits)
    assert g.abs().max().item() < 1e-12, g.abs().max().item()


def _exact_centering_grad(logits, logq_full, wfn):
    # Exact: sum_V sg[q w(p / q)] log p over the whole vocabulary.
    logp = torch.log_softmax(logits, -1)
    with torch.no_grad():
        q = logq_full.exp()
        coef = q * wfn((logp - logq_full).exp())
    (g,) = torch.autograd.grad((coef * logp).sum(), logits)
    return g


@pytest.mark.parametrize("rule", [None, TIS, ICEPOP])
def test_full_vocab_head_equals_exact_centering(rule):
    gen = torch.Generator().manual_seed(1)
    V = 1000
    logits = torch.randn(32, V, generator = gen, dtype = D64, requires_grad = True)
    logq = torch.log_softmax(logits.detach() + 0.3 * torch.randn(32, V, generator = gen, dtype = D64), -1)
    wfn = sc.score_centering_weight_fn(*rule) if rule else sc.score_centering_weight_fn(None)
    ids = torch.arange(V).expand(32, V)
    corr, _, _ = sc.score_centering_correction(torch.log_softmax(logits, -1).gather(-1, ids), logq.gather(-1, ids), wfn)
    (g,) = torch.autograd.grad(corr.sum(), logits)
    g_exact = _exact_centering_grad(logits, logq, wfn)
    err = (g - g_exact).abs().max().item()
    assert err < 1e-10, err


@pytest.mark.parametrize("rule", [None, TIS, ICEPOP])
@pytest.mark.parametrize("k", [32, 128])
def test_head_equals_exact_when_tail_is_rescaled_trainer_tail(rule, k):
    # The estimator is exact whenever the sampler tail is the trainer tail rescaled (its model).
    gen = torch.Generator().manual_seed(2)
    V, N = 1000, 16
    logits = torch.randn(N, V, generator = gen, dtype = D64, requires_grad = True)
    p = torch.softmax(logits.detach(), -1)
    head_ids = p.topk(k, -1).indices
    in_head = torch.zeros(N, V, dtype = torch.bool).scatter_(1, head_ids, True)
    q_head = torch.softmax(torch.randn(N, k, generator = gen, dtype = D64), -1) * 0.8
    tail_p = torch.where(in_head, 0.0, p)
    q = (tail_p / tail_p.sum(-1, keepdim = True)) * 0.2
    q = q.scatter(1, head_ids, q_head)
    logq = q.log()
    wfn = sc.score_centering_weight_fn(*rule) if rule else sc.score_centering_weight_fn(None)
    corr, qm, pm = sc.score_centering_correction(
        torch.log_softmax(logits, -1).gather(-1, head_ids), logq.gather(-1, head_ids), wfn, eps = 1e-30,
    )
    (g,) = torch.autograd.grad(corr.sum(), logits)
    g_exact = _exact_centering_grad(logits, logq, wfn)
    err = (g - g_exact).abs().max().item()
    assert err < 1e-10, err


# verl/trainer/ppo/score_centering.py score_centering_correction (Apache-2.0, volcengine/verl), verbatim
# minus the docstring: the reference the parity test below holds the Unsloth estimator to.
def _verl_score_centering_correction(
    train_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_head_log_probs = train_head_log_probs.float()
    with torch.no_grad():
        p = train_head_log_probs.exp()
        q = sampler_head_log_probs.float().exp()
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * weight_fn(1.0 / rho)
        head_weights = weight_fn((train_head_log_probs - sampler_head_log_probs.float()).exp())
        residual = q * head_weights - alpha.unsqueeze(-1) * p
    correction = (residual * train_head_log_probs).sum(-1)
    return correction, q_mass, p_mass


@pytest.mark.parametrize("rule", ["none", "tis", "icepop"])
def test_parity_with_verl(rule):
    verl_fn = _verl_score_centering_correction
    # verl score_centering_weight_fn: None -> ones, "2.0" -> clamp(max=2), "0.5_2.0" -> band.
    verl_w = {
        "none": torch.ones_like,
        "tis": lambda r: r.clamp(max = 2.0),
        "icepop": lambda r: torch.where((r >= 0.5) & (r <= 2.0), r, torch.zeros_like(r)),
    }[rule]
    ours_w = {"none": sc.score_centering_weight_fn(None), "tis": sc.score_centering_weight_fn(*TIS),
              "icepop": sc.score_centering_weight_fn(*ICEPOP)}[rule]
    gen = torch.Generator().manual_seed(3)
    for dtype, tol in ((torch.float32, 1e-6), (D64, 1e-6)):
        logits = torch.randn(48, 5000, generator = gen, dtype = dtype) * 2
        logq_full = torch.log_softmax(logits + 0.2 * torch.randn(48, 5000, generator = gen, dtype = dtype), -1)
        ids = logq_full.topk(32, -1).indices
        x1 = logits.clone().requires_grad_(True)
        x2 = logits.clone().requires_grad_(True)
        c1, qm1, pm1 = verl_fn(torch.log_softmax(x1, -1).gather(-1, ids), logq_full.gather(-1, ids), verl_w)
        c2, qm2, pm2 = sc.score_centering_correction(torch.log_softmax(x2, -1).gather(-1, ids), logq_full.gather(-1, ids), ours_w)
        g1, = torch.autograd.grad(c1.sum(), x1)
        g2, = torch.autograd.grad(c2.sum(), x2)
        assert (c1 - c2).abs().max().item() < tol
        assert (qm1 - qm2).abs().max().item() < tol and (pm1 - pm2).abs().max().item() < tol
        assert (g1 - g2).abs().max().item() < tol


def test_head_mask_and_neg_inf_sampler():
    gen = torch.Generator().manual_seed(4)
    logits = torch.randn(4, 100, generator = gen, dtype = D64, requires_grad = True)
    logp = torch.log_softmax(logits, -1)
    ids = logp.detach().topk(8, -1).indices
    logq = logp.detach().gather(-1, ids).clone()
    logq[:, 5:] = -math.inf  # outside a top_k = 5 sampler support
    mask = torch.ones_like(ids, dtype = torch.bool)
    mask[0] = False  # an unmatched row
    corr, qm, pm = sc.score_centering_correction(logp.gather(-1, ids), logq, sc.score_centering_weight_fn("token_truncate", None, None), mask)
    assert torch.isfinite(corr).all() and corr[0].item() == 0.0
    g, = torch.autograd.grad(corr.sum(), logits)
    assert torch.isfinite(g).all() and g[0].abs().max().item() == 0.0


def _oracle_reinforce(new, old, sampling, adv, mask, sc_corr, mode, cmin, cmax, bypass, ga = 1):
    """Independent fp64 reimplementation of verl compute_policy_loss_reinforce + seq-mean-token-sum."""
    with torch.no_grad():
        if bypass:
            lr = ((new - old) * mask).clamp(-20, 20)
        elif sampling is not None:
            lr = (old - sampling) * mask
        else:
            lr = None
        if lr is None:
            w = torch.ones_like(new)
        else:
            if mode.startswith("sequence"):
                lr = lr.sum(-1, keepdim = True)
                if bypass:
                    lr = lr.clamp(-20, 20)
            w = lr.exp()
            if mode.endswith("truncate"):
                w = w.clamp(min = cmin, max = cmax)
            else:
                lo = -math.inf if cmin is None else cmin
                hi = math.inf if cmax is None else cmax
                w = torch.where((w >= lo) & (w <= hi), w, 0.0)
    a = adv.unsqueeze(1)
    li = -w * a * new
    if sc_corr is not None:
        li = li + a * sc_corr
    return (li * mask).sum() / new.shape[0] / ga


def _kw(mode, cmin, cmax, bypass, use_is, ga = 1):
    return dict(
        loss_type = "reinforce", use_vllm = True, vllm_importance_sampling_correction = use_is,
        vllm_importance_sampling_mode = mode, vllm_importance_sampling_clip_min = cmin,
        vllm_importance_sampling_clip_max = cmax, rollout_logps_as_old = bypass,
        current_gradient_accumulation_steps = ga, num_items_in_batch = 1.0,
    )


MODES = [("token_truncate", None, 2.0), ("token_mask", 0.5, 2.0), ("sequence_truncate", 0.0, 3.0), ("sequence_mask", 0.5, 5.0)]


@pytest.mark.parametrize("mode,cmin,cmax", MODES)
@pytest.mark.parametrize("bypass", [True, False])
@pytest.mark.parametrize("with_sc", [True, False])
def test_reinforce_loss_matches_oracle(mode, cmin, cmax, bypass, with_sc):
    gen = torch.Generator().manual_seed(5)
    B, T = 6, 9
    mask = (torch.rand(B, T, generator = gen) > 0.2).to(D64)
    mask[0] = 0  # fully masked row
    mask[1, 1:] = 0  # one-token row
    new = (-torch.rand(B, T, generator = gen, dtype = D64) * 3).requires_grad_(True)
    rollout = new.detach() + torch.randn(B, T, generator = gen, dtype = D64) * 0.3
    old, sampling = (rollout, None) if bypass else (new.detach().clone(), rollout)
    adv = torch.randn(B, generator = gen, dtype = D64)
    corr = torch.randn(B, T, generator = gen, dtype = D64, requires_grad = True) if with_sc else None
    kwargs = _kw(mode, cmin, cmax, bypass, True)
    if with_sc:
        kwargs["score_centering_correction"] = corr
    loss = grpo_compute_loss(None, new, old, sampling, torch.zeros(B, T, dtype = torch.long), mask, 0.0, adv, **kwargs)[0]
    ref = _oracle_reinforce(new, old, sampling, adv, mask, corr, mode, cmin, cmax, bypass)
    assert abs(loss.item() - ref.item()) < 1e-12
    inputs = [new] + ([corr] if with_sc else [])
    g = torch.autograd.grad(loss, inputs)
    gr = torch.autograd.grad(ref, inputs)
    for a, b in zip(g, gr):
        assert (a - b).abs().max().item() < 1e-12
    assert g[0][0].abs().max().item() == 0.0  # masked row: zero grad


@pytest.mark.parametrize("d", [0.0, 1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0])
def test_bypass_log_ratio_extremes_finite(d):
    B, T = 2, 4
    mask = torch.ones(B, T, dtype = torch.float32)
    new = torch.full((B, T), -1.0, requires_grad = True)
    old = new.detach() - d
    adv = torch.tensor([1.0, -1.0])
    for mode, cmin, cmax in MODES:
        loss = grpo_compute_loss(None, new, old, None, torch.zeros(B, T, dtype = torch.long), mask, 0.0, adv,
                                 **_kw(mode, cmin, cmax, True, True))[0]
        g, = torch.autograd.grad(loss, new)
        assert torch.isfinite(loss) and torch.isfinite(g).all(), (mode, d)


def test_ga_split_equals_one_batch():
    gen = torch.Generator().manual_seed(6)
    B, T = 8, 7
    mask = (torch.rand(B, T, generator = gen) > 0.3).to(D64)
    new = (-torch.rand(B, T, generator = gen, dtype = D64) * 2).requires_grad_(True)
    old = new.detach() + 0.2 * torch.randn(B, T, generator = gen, dtype = D64)
    adv = torch.randn(B, generator = gen, dtype = D64)
    corr = torch.randn(B, T, generator = gen, dtype = D64, requires_grad = True)
    ids = torch.zeros(B, T, dtype = torch.long)
    kw1 = _kw("token_truncate", None, 2.0, True, True, ga = 1)
    full = grpo_compute_loss(None, new, old, None, ids, mask, 0.0, adv, score_centering_correction = corr, **kw1)[0]
    g_full = torch.autograd.grad(full, [new, corr])
    kw2 = _kw("token_truncate", None, 2.0, True, True, ga = 2)
    parts = sum(
        grpo_compute_loss(None, new[s], old[s], None, ids[s], mask[s], 0.0, adv[s], score_centering_correction = corr[s], **kw2)[0]
        for s in (slice(0, 4), slice(4, 8))
    )
    g_parts = torch.autograd.grad(parts, [new, corr])
    assert abs(full.item() - parts.item()) < 1e-12
    for a, b in zip(g_full, g_parts):
        assert (a - b).abs().max().item() < 1e-12


def test_efficient_grpo_chains_correction_gradient():
    # UnslothEfficientGRPO (compiled, grad wrt new_logps and the correction) == plain autograd.
    gen = torch.Generator().manual_seed(7)
    B, T, V, k = 4, 6, 50, 8
    mask = (torch.rand(B, T, generator = gen) > 0.2).float()
    logits = torch.randn(B, T, V, generator = gen, requires_grad = True)
    ids = torch.randint(0, V, (B, T), generator = gen)
    logp = torch.log_softmax(logits, -1)
    new = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    rollout = new.detach() + 0.1 * torch.randn(B, T, generator = gen)
    head_ids = logp.detach().topk(k, -1).indices
    logq = (logp.detach() + 0.05 * torch.randn(B, T, V, generator = gen)).log_softmax(-1).gather(-1, head_ids)
    wfn = sc.score_centering_weight_fn("token_truncate", None, 2.0)
    corr, _, _ = sc.score_centering_correction(logp.gather(-1, head_ids), logq, wfn)
    adv = torch.randn(B, generator = gen)
    kw = _kw("token_truncate", None, 2.0, True, True)
    out = UnslothEfficientGRPO.apply(new, rollout, None, None, None, ids, mask, adv, 0.0, None, 1, kw, None, corr)
    g_eff, = torch.autograd.grad(out[0], logits, retain_graph = True)
    ref = grpo_compute_loss(None, new, rollout, None, ids, mask, 0.0, adv, score_centering_correction = corr, **kw)[0]
    g_ref, = torch.autograd.grad(ref, logits)
    assert abs(out[0].item() - ref.item()) < 1e-5
    assert (g_eff - g_ref).abs().max().item() < 1e-6
    # Without a correction the class is unchanged (13-argument call).
    out2 = UnslothEfficientGRPO.apply(new.detach().requires_grad_(True), rollout, None, None, None, ids, mask, adv, 0.0, None, 1, kw, None)
    assert torch.isfinite(out2[0])


@pytest.mark.parametrize("softcap,mult,div,temp", [(0.0, 0.0, 0.0, 1.0), (30.0, 0.5, 0.0, 0.7), (0.0, 0.0, 4.0, 1.3)])
def test_topk_head_logprobs_matches_full_softmax(softcap, mult, div, temp):
    gen = torch.Generator().manual_seed(8)
    R, T, H, V, k = 3, 5, 16, 97, 7
    h = torch.randn(R, T, H, generator = gen, dtype = D64, requires_grad = True)
    W = torch.randn(V, H, generator = gen, dtype = D64, requires_grad = True)
    bias = torch.randn(V, generator = gen, dtype = D64, requires_grad = True)
    ids = torch.randint(0, V, (R, T, k), generator = gen)
    out = sc.grpo_topk_head_logprobs(h, W, ids, 4, mult, div, softcap, temp, lm_head_bias = bias)
    z = torch.nn.functional.linear(h, W, bias)
    if mult:
        z = z * mult
    if div:
        z = z / div
    if softcap:
        z = softcap * torch.tanh(z / softcap)
    ref = torch.log_softmax(z / temp, -1).gather(-1, ids)
    assert (out - ref).abs().max().item() < 1e-12
    gw = torch.randn(R, T, k, generator = gen, dtype = D64)
    g1 = torch.autograd.grad((out * gw).sum(), [h, W, bias])
    g2 = torch.autograd.grad((ref * gw).sum(), [h, W, bias])
    for a, b in zip(g1, g2):
        assert (a - b).abs().max().item() < 1e-12
    # Raw logits input (the forward already scaled them): only temperature.
    lz = torch.randn(R, T, V, generator = gen, dtype = D64)
    out = sc.grpo_topk_head_logprobs(lz, W, ids, 2, temperature = temp)
    assert (out - torch.log_softmax(lz / temp, -1).gather(-1, ids)).abs().max().item() < 1e-12


@pytest.mark.parametrize("chunks", [1, 4, 9])
def test_topk_head_logprobs_skips_positions_without_a_head(chunks, monkeypatch):
    gen = torch.Generator().manual_seed(11)
    R, T, H, V, k = 4, 9, 16, 97, 5
    h = torch.randn(R, T, H, generator = gen, dtype = D64, requires_grad = True)
    W = torch.randn(V, H, generator = gen, dtype = D64, requires_grad = True)
    ids = torch.randint(0, V, (R, T, k), generator = gen)
    lengths = [9, 3, 0, 6]  # row 2 unmatched
    for r, n in enumerate(lengths):
        ids[r, n:] = -1
    ids[3, 2, 3:] = -1  # a short head keeps its position
    seen = []
    real = sc._head_chunk_logps

    def counting(x, *a, **kw):
        seen.append(x.shape[0])
        return real(x, *a, **kw)

    monkeypatch.setattr(sc, "_head_chunk_logps", counting)
    out = sc.grpo_topk_head_logprobs(h, W, ids, chunks, temperature = 0.8)
    kept = (ids >= 0).any(-1)
    assert sum(seen) == int(kept.sum()) == sum(lengths)
    assert max(seen) <= -(-R * T // min(chunks, R * T))  # rows per chunk do not grow
    ref = torch.log_softmax(torch.nn.functional.linear(h, W) / 0.8, -1).gather(-1, ids.clamp_min(0))
    assert (out[kept] - ref[kept]).abs().max().item() < 1e-12
    assert (out[~kept] == 0).all()
    mask = ids >= 0
    gw = torch.randn(R, T, k, generator = gen, dtype = D64)
    g1 = torch.autograd.grad(torch.where(mask, out * gw, 0.0).sum(), [h, W])
    g2 = torch.autograd.grad(torch.where(mask, ref * gw, 0.0).sum(), [h, W])
    for a, b in zip(g1, g2):
        assert (a - b).abs().max().item() < 1e-12
    none = sc.grpo_topk_head_logprobs(h, W, torch.full_like(ids, -1), chunks)
    assert none.shape == (R, T, k) and (none == 0).all() and none.requires_grad
    # Raw logits are not gathered (that would copy the [rows, V] input outside the checkpoints).
    seen.clear()
    logits = torch.nn.functional.linear(h, W).detach().requires_grad_()
    raw = sc.grpo_topk_head_logprobs(logits, W, ids, chunks)
    assert sum(seen) == R * T
    ref_raw = torch.log_softmax(logits, -1).gather(-1, ids.clamp_min(0))
    assert (raw[kept] - ref_raw[kept]).abs().max().item() < 1e-12


class _LP:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


def _fake_output(prompt, completion, V, k, gen):
    positions = []
    for tok in completion:
        logq = torch.log_softmax(torch.randn(V, generator = gen), -1)
        order = logq.argsort(descending = True).tolist()
        rank = {t: r + 1 for r, t in enumerate(order)}
        d = {tok: _LP(float(logq[tok]), rank[tok])}  # sampled first, true rank
        for t in order[:k]:
            d[t] = _LP(float(logq[t]), rank[t])
        positions.append((d, order[:k], logq))
    comp = types.SimpleNamespace(token_ids = list(completion), logprobs = [p[0] for p in positions])
    return types.SimpleNamespace(prompt_token_ids = list(prompt), outputs = [comp]), positions


def test_record_lookup_align_roundtrip():
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(9)
    V, k, pad = 40, 4, 0
    rows = [([5, 6, 7], [8, 9, 0]), ([5, 6], [11, 12, 13, 14]), ([3, 6, 7], [8, 9])]  # pad == eos ends row 0
    outs, truth = [], []
    for p, c in rows:
        o, pos = _fake_output(p, c, V, k, gen)
        outs.append(o)
        truth.append(pos)
    assert sc.record_rollout_topk(outs, k) == 3
    L = 4
    P = 3
    input_ids = torch.full((4, P + L), pad, dtype = torch.long)
    for b, (p, c) in enumerate(rows):
        input_ids[b, P - len(p):P] = torch.tensor(p)
        input_ids[b, P:P + len(c)] = torch.tensor(c)
    input_ids[3, :P] = torch.tensor([1, 2, 3])
    input_ids[3, P:P + 2] = torch.tensor([4, 4])  # never generated
    ids, logps, matched = sc.lookup_rollout_topk(input_ids, L, k, pad)
    assert matched.tolist() == [True, True, True, False]
    for b, (p, c) in enumerate(rows):
        for j in range(len(c)):
            _, top, logq = truth[b][j]
            assert ids[b, j].tolist() == top
            assert torch.allclose(logps[b, j], logq[top])
    assert (ids[3] == -1).all()
    # align: same rule as align_logprobs_with_mask, on each head slot
    from unsloth_zoo.rl_replacements import align_logprobs_with_mask
    cmask = torch.zeros(4, L + 2)
    cmask[0, 2:5] = 1
    cmask[1, 1:5] = 1
    cmask[2, 3:5] = 1
    a = sc.align_topk_with_mask(logps, cmask, float("-inf"))
    for slot in range(k):
        assert torch.equal(a[..., slot], align_logprobs_with_mask(logps[..., slot], cmask, float("-inf")))


def _trl1_sampled_logprobs(outputs):
    """Mirrors TRL >= 1.x extract_logprobs: rank-sort, keep entry 0."""
    rows = []
    for o in outputs:
        for c in o.outputs:
            rows.append([sorted(lp.items(), key = lambda x: x[1].rank)[0][1].logprob for lp in c.logprobs])
    return rows


@pytest.mark.parametrize("asked", [0, None, 2, 8])
def test_wrapped_generate_returns_the_requested_logprobs(asked):
    # Centering raises logprobs to k for its head; TRL must still read the sampled token's logprob.
    sc.reset_score_centering()
    gen = torch.Generator().manual_seed(3)
    V, k = 50, 4
    completion = [7, 21, 33, 2, 40]
    outs, truth = zip(*[_fake_output([1, 2, 3], completion, V, k, gen), _fake_output([4, 5], completion[:3], V, k, gen)])
    seen = {}

    class _LLM:
        def generate(self, prompts, sampling_params = None):
            seen["logprobs"] = sampling_params.logprobs
            return list(outs)

    llm = sc.wrap_vllm_generate_topk(_LLM(), k)
    params = types.SimpleNamespace(logprobs = asked)
    got = llm.generate(["a", "b"], sampling_params = params)
    assert seen["logprobs"] == (k if asked is None or asked < k else asked)
    assert params.logprobs == asked
    assert sc.score_centering_stats()["recorded"] == 2
    head_ids, _ = sc._lookup(sc._make_key([1, 2, 3] + completion, None), None)
    assert head_ids[0].tolist() == truth[0][0][1]
    if asked is None:
        assert all(c.logprobs is None for o in got for c in o.outputs)
        return
    if asked == 0:
        sampled = [[float(pos[0][tok].logprob) for pos, tok in zip(t, o.outputs[0].token_ids)]
                   for t, o in zip(truth, got)]
        assert _trl1_sampled_logprobs(got) == sampled
    for o in got:
        for lp, tok in zip(o.outputs[0].logprobs, o.outputs[0].token_ids):
            assert tok in lp
            if asked < k:
                assert all(t == tok or v.rank <= asked for t, v in lp.items())
            else:
                assert len(lp) >= k


def test_raise_logprobs_and_env():
    p = types.SimpleNamespace(logprobs = 0)
    sc._raise_logprobs(p, 32)
    assert p.logprobs == 32
    p = types.SimpleNamespace(logprobs = 64)
    sc._raise_logprobs([p], 32)
    assert p.logprobs == 64
    for v, k in (("0", 0), ("", 0), ("1", 32), ("on", 32), ("128", 128)):
        os.environ[sc.SCORE_CENTERING_ENV] = v
        assert sc.score_centering_top_k() == k
    os.environ[sc.SCORE_CENTERING_ENV] = "abc"
    with pytest.raises(ValueError):
        sc.score_centering_top_k()
    os.environ.pop(sc.SCORE_CENTERING_ENV)


def test_existing_loss_types_unchanged_by_reinforce_edit():
    # The vLLM IS multiply still applies to the other loss types (only vespo / reinforce skip it).
    gen = torch.Generator().manual_seed(10)
    B, T = 3, 5
    mask = torch.ones(B, T)
    new = (-torch.rand(B, T, generator = gen)).requires_grad_(True)
    old = new.detach().clone()
    sampling = old + 0.3 * torch.randn(B, T, generator = gen)
    adv = torch.randn(B, generator = gen)
    kw = _kw("token_truncate", 0.0, 2.0, False, True)
    kw["loss_type"] = "cispo"
    kw["epsilon_high"] = 5.0
    kw["num_items_in_batch"] = float(mask.sum())
    loss = grpo_compute_loss(None, new, old, sampling, torch.zeros(B, T, dtype = torch.long), mask, 0.0, adv, **kw)[0]
    w = (old - sampling).exp().clamp(0.0, 2.0)
    ref = (-(torch.ones_like(new)) * adv.unsqueeze(1) * new * w * mask).sum() / mask.sum()
    assert abs(loss.item() - ref.item()) < 1e-6



def _cuda_head_precision_path(torch_mod = torch):
    """The fp32 head default the next test checks: an NVIDIA CUDA build. ROCm keeps the head in its own dtype
    (score_centering._head_projection, is_rocm_torch), and torch.cuda.is_available() is True there too."""
    from unsloth_zoo.rocm_build import is_rocm_torch
    return torch_mod.cuda.is_available() and not is_rocm_torch(torch_mod)


@pytest.mark.parametrize("hip, version, cuda, expected", [
    (None, "2.11.0+cu130", True, True),
    ("7.13.99004", "2.11.0+rocm7.13.0", True, False),  # Linux ROCm wheel
    (None, "2.11.0+rocm7.14.0", True, False),          # AMD SDK wheel: version.hip unset
    (None, "2.11.0+cpu", False, False),
])
def test_cuda_head_precision_skip_excludes_rocm(hip, version, cuda, expected):
    fake = types.SimpleNamespace(version = types.SimpleNamespace(hip = hip), __version__ = version,
                                 cuda = types.SimpleNamespace(is_available = lambda: cuda))
    assert _cuda_head_precision_path(fake) is expected


@pytest.mark.skipif(not _cuda_head_precision_path(), reason = "NVIDIA CUDA head precision path (ROCm keeps the head dtype)")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_head_logprobs_match_policy_scorer_cuda(dtype):
    # Same projection precision as the sampled-token scorer, so p and the loss describe one policy.
    from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as scorer
    gen = torch.Generator(device = "cuda").manual_seed(11)
    R, T, H, V, k = 4, 33, 256, 32000, 8
    h = torch.randn(R, T, H, device = "cuda", dtype = dtype, generator = gen)
    W = (torch.randn(V, H, device = "cuda", dtype = dtype, generator = gen) * 0.05)
    ids = torch.randint(0, V, (R, T, k), device = "cuda", generator = gen)
    with torch.autocast("cuda", dtype = dtype):
        head = sc.grpo_topk_head_logprobs(h, W, ids, 4, temperature = 0.9)
        ref = torch.stack([scorer(h, W, ids[..., j], 4, temperature = 0.9) for j in range(k)], dim = -1)
    exact = torch.log_softmax((h.float() @ W.float().t()) / 0.9, -1).gather(-1, ids)
    assert (head - ref).abs().max().item() < 5e-5
    assert (head - exact).abs().max().item() < 5e-4


def test_score_centering_gate(monkeypatch):
    from unsloth_zoo.rl_replacements import grpo_score_centering_k
    trainer = types.SimpleNamespace(use_vllm = True)
    monkeypatch.setenv(sc.SCORE_CENTERING_ENV, "32")
    is_on = {"vllm_importance_sampling_correction": True}
    # Bypass: the head and the sampled token share pi_theta / pi_rollout.
    assert grpo_score_centering_k(trainer, is_on, rollout_logps_as_old = True) == 32
    # No IS weight on either side.
    assert grpo_score_centering_k(trainer, {"vllm_importance_sampling_correction": False}) == 32
    # vLLM IS ratio pi_old / pi_rollout outside bypass, and kept-set scoring: off.
    assert grpo_score_centering_k(types.SimpleNamespace(use_vllm = True), is_on) == 0
    assert grpo_score_centering_k(types.SimpleNamespace(use_vllm = True), is_on, True, kept_set_scoring = True) == 0
    monkeypatch.setenv(sc.SCORE_CENTERING_ENV, "0")
    assert grpo_score_centering_k(trainer, is_on, rollout_logps_as_old = True) == 0


def test_score_centering_refuses_untruncated_filtered_logprobs(monkeypatch):
    # Colocated vLLM with the IS correction returns logprobs from before top_k / top_p / min_p; under a filter
    # those are not the sampler's q, so score centering is off (as unsloth_distillation_coef refuses it).
    import unsloth_zoo.rl_replacements as rl
    vllm_utils = pytest.importorskip("unsloth_zoo.vllm_utils")
    state = vllm_utils.UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS
    monkeypatch.setitem(state, "on", True)
    monkeypatch.setitem(state, "installed", True)
    monkeypatch.setenv(sc.SCORE_CENTERING_ENV, "32")
    is_on = {"vllm_importance_sampling_correction": True}
    args = types.SimpleNamespace(top_k = None, top_p = 1.0, min_p = 0.1, temperature = 1.0)
    trainer = types.SimpleNamespace(use_vllm = True, vllm_importance_sampling_correction = True, args = args,
                                    _unsloth_vllm_untruncated_logprobs = True)
    assert rl.grpo_sampling_support(trainer) == (0, 1.0, 0.1)
    assert rl.grpo_truncated_support_kwargs(trainer) == {}
    assert rl.grpo_score_centering_k(trainer, is_on, rollout_logps_as_old = True) == 0
    assert trainer._unsloth_score_centering_off_warned is True
    # Unfiltered sampling: untruncated logprobs are the sampler's.
    args.min_p = 0.0
    assert rl.grpo_score_centering_k(trainer, is_on, rollout_logps_as_old = True) == 32
    # Processed (kept-set) logprobs from the engine: the sampler's again.
    args.min_p = 0.1
    trainer._unsloth_vllm_untruncated_logprobs = False
    assert rl.grpo_score_centering_k(trainer, is_on, rollout_logps_as_old = True) == 32
    monkeypatch.setitem(state, "on", False)
    trainer._unsloth_vllm_untruncated_logprobs = True
    assert rl.grpo_score_centering_k(trainer, is_on, rollout_logps_as_old = True) == 32


@pytest.mark.parametrize("rule", [TIS, ICEPOP])
def test_centering_needs_the_filtered_sampler_q(rule):
    # fp64 oracle, on-policy (p = pre-filter q), min_p = 0.1 sampler q'. With constant A the centered estimator's
    # expected gradient under y ~ q' is zero with q' as the head q, and the full sum_y q'(y) grad log p(y) with the
    # pre-filter q (whose correction is zero when q = p): untruncated logprobs bias it.
    gen = torch.Generator().manual_seed(1)
    V, k = 200, 32
    logits = torch.randn(16, V, generator = gen, dtype = D64) * 3.0
    logits.requires_grad_(True)
    logp = torch.log_softmax(logits, -1)
    probs = logp.detach().exp()
    keep = probs >= 0.1 * probs.max(-1, keepdim = True).values
    assert int(keep.sum(-1).max()) <= k and bool((~keep).any(-1).all())
    q_filtered = torch.where(keep, probs, 0.0)
    q_filtered = q_filtered / q_filtered.sum(-1, keepdim = True)
    wfn = sc.score_centering_weight_fn(*rule)

    def expected_grad(log_q_used):
        ids = probs.topk(k, dim = -1).indices
        corr, _, _ = sc.score_centering_correction(logp.gather(-1, ids), log_q_used.gather(-1, ids), wfn)
        with torch.no_grad():
            w = wfn(torch.where(q_filtered > 0, (logp - log_q_used).exp(), 0.0))
        sampled = (q_filtered * w * logp).sum(-1)
        (g,) = torch.autograd.grad((sampled - corr).sum(), logits, retain_graph = True)
        return g.abs().max().item()

    # Up to the eps clamp of the tail mass (q'(H) = 1).
    assert expected_grad(q_filtered.log()) < 1e-5
    assert expected_grad(logp.detach()) > 1e-2
