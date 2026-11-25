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

"""Kept-set backward on W[kept ids] (CSR saved ids) vs the dense backward and a float64 oracle."""

import os
import sys

import pytest
import torch

import unsloth_zoo.rl_replacements as rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _vllm_topk_topp_reference import apply_top_k_top_p_pytorch  # noqa: E402

V, H = 3001, 40

CASES = [
    # temperature, top_k, top_p, min_p, softcap
    (1.0, 1, 1.0, 0.0, 0.0),     # singleton kept sets (target forced in)
    (1.0, 8, 1.0, 0.0, 0.0),
    (0.7, 64, 0.95, 0.0, 0.0),   # the GRPO setting
    (1.0, 0, 0.5, 0.0, 0.0),
    (0.9, 0, 0.95, 0.02, 30.0),
    (1.0, 0, 0.999, 0.0, 0.0),   # dense kept sets
]


def _inputs(rows = (3, 37), seed = 0, bias = False):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(*rows, H, generator = g) * 2.0
    head = torch.randn(V, H, generator = g) * 0.5
    index = torch.randint(0, V, rows, generator = g)
    b = torch.randn(V, generator = g) * 0.1 if bias else None
    return hidden, head, index, b


def _oracle(hidden, head, index, temperature, top_k, top_p, min_p, softcap, bias = None, excluded = None):
    h = hidden.detach().double().requires_grad_(True)
    w = head.detach().double()
    z = torch.nn.functional.linear(h.reshape(-1, H), w, None if bias is None else bias.double())
    if softcap:
        z = softcap * torch.tanh(z / softcap)
    z = z / temperature
    idx = index.reshape(-1)
    with torch.no_grad():
        m = z.detach().clone()
        if excluded is not None:
            m[:, excluded] = float("-inf")
        if min_p > 0.0:
            probs = torch.softmax(m, dim = -1)
            m = m.masked_fill(probs < min_p * probs.amax(dim = -1, keepdim = True), float("-inf"))
        rows = m.shape[0]
        k = torch.full((rows,), top_k, dtype = torch.int32) if top_k > 0 else None
        p = torch.full((rows,), top_p, dtype = torch.float32) if top_p < 1.0 else None
        m = apply_top_k_top_p_pytorch(m, k, p)
        kept = torch.isfinite(m)
        kept[torch.arange(rows), idx] = True
    lse = torch.logsumexp(z.masked_fill(~kept, float("-inf")), dim = -1)
    logp = z.gather(-1, idx[:, None]).squeeze(-1) - lse
    return logp.reshape(tuple(index.shape)), kept, h


def _run(monkeypatch, mode, hidden, head, index, temperature, top_k, top_p, min_p, softcap, bias = None,
         chunks = 3, excluded = None, head_grad = False):
    if mode == "dense":
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", "0")
    else:
        monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", raising = False)
    if mode == "sparse_full":
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", "1.0")
    else:
        monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", raising = False)
    h = hidden.clone().requires_grad_(True)
    w = head.clone().requires_grad_(head_grad)
    before = dict(rr._GRPO_TRUNCATED_SUPPORT_STATS)
    out = rr.grpo_chunked_hidden_states_logprobs(
        h, w, index, chunks, 0.0, 0.0, softcap, temperature, lm_head_bias = bias,
        support_top_k = top_k, support_top_p = top_p, support_min_p = min_p, support_excluded_ids = excluded,
    )
    weights = torch.randn(tuple(out.shape), generator = torch.Generator().manual_seed(1))
    (out * weights).sum().backward()
    after = rr._GRPO_TRUNCATED_SUPPORT_STATS
    counts = {k: after.get(k, 0) - before.get(k, 0) for k in ("kept_set_sparse_bwd", "kept_set_dense_bwd")}
    return out.detach(), h.grad, (w.grad if head_grad else None), counts, weights


@pytest.mark.parametrize("temperature, top_k, top_p, min_p, softcap", CASES)
@pytest.mark.parametrize("bias", [False, True])
def test_sparse_backward_matches_dense_and_float64(monkeypatch, temperature, top_k, top_p, min_p, softcap, bias):
    hidden, head, index, b = _inputs(bias = bias)
    args = (hidden, head, index, temperature, top_k, top_p, min_p, softcap, b)
    out_d, gd, _, cd, weights = _run(monkeypatch, "dense", *args)
    out_s, gs, _, cs, _ = _run(monkeypatch, "sparse_full", *args)
    assert cd["kept_set_sparse_bwd"] == 0 and cd["kept_set_dense_bwd"] > 0
    if top_k > 0 or top_p < 0.99:
        assert cs["kept_set_sparse_bwd"] > 0 and cs["kept_set_dense_bwd"] == 0
    else:
        # Kept sets denser than ~1/64 (CSR ids + float32 z) are saved as the bit mask: dense backward, bit-equal.
        assert cs["kept_set_sparse_bwd"] == 0 and torch.equal(gs, gd)
    assert torch.equal(out_d, out_s)
    ref, kept, h64 = _oracle(*args)
    (ref * weights.double()).sum().backward()
    # Each path against the float64 oracle on its own, then against each other: the relative bound allows the dh
    # GEMM's blocking to round either way (torch 2.15 CPU: 1.85e-5 vs 1.80e-5) but not a wrong p or kept set.
    torch.testing.assert_close(gd.double(), h64.grad, atol = 2e-5, rtol = 1e-4)
    torch.testing.assert_close(gs.double(), h64.grad, atol = 2e-5, rtol = 1e-4)
    err_dense = (gd.double() - h64.grad).abs().max().item()
    err_sparse = (gs.double() - h64.grad).abs().max().item()
    assert err_sparse <= 1.25 * err_dense + 1e-7, (err_sparse, err_dense)
    # Same z and p as the dense backward (the forward's z is saved): only the dh GEMM blocking differs.
    torch.testing.assert_close(gs, gd, atol = 1e-6, rtol = 1e-6)


def test_float64_inputs_agree_to_float32_rounding(monkeypatch):
    # The kept-set math runs in float32 whatever the inputs; sparse and dense differ only by GEMM order.
    hidden, head, index, _ = _inputs(seed = 3)
    args = (hidden.double(), head.double(), index, 0.8, 64, 0.95, 0.0, 0.0)
    _, gd, _, _, _ = _run(monkeypatch, "dense", *args)
    _, gs, _, cs, _ = _run(monkeypatch, "sparse_full", *args)
    assert cs["kept_set_sparse_bwd"] > 0
    torch.testing.assert_close(gs, gd, atol = 1e-6, rtol = 1e-6)


def test_full_rows_and_excluded_ids(monkeypatch):
    # top_k = 0, top_p = 1 with excluded ids: nearly every id kept. Default crossover runs dense; forced sparse agrees.
    hidden, head, index, _ = _inputs(seed = 5)
    excluded = torch.arange(V - 30, V)
    index[0, :4] = V - 1  # an excluded target is still scored
    args = (hidden, head, index, 1.0, 0, 1.0, 0.0, 0.0)
    _, gd, _, _, _ = _run(monkeypatch, "dense", *args, excluded = excluded)
    _, g_default, _, c_default, _ = _run(monkeypatch, "sparse", *args, excluded = excluded)
    _, g_full, _, c_full, _ = _run(monkeypatch, "sparse_full", *args, excluded = excluded)
    assert c_default["kept_set_sparse_bwd"] == 0  # packed mask (density > ~1/64): dense backward
    assert torch.equal(g_default, gd)
    # density > ~1/64 is saved packed, so even a forced crossover stays dense here
    assert c_full["kept_set_sparse_bwd"] == 0
    torch.testing.assert_close(g_full, gd, atol = 0, rtol = 0)


def test_singleton_kept_sets_have_zero_gradient(monkeypatch):
    hidden, head, index, _ = _inputs(seed = 7)
    top1 = (hidden.reshape(-1, H) @ head.t()).argmax(-1).reshape(index.shape)
    out, g, _, c, _ = _run(monkeypatch, "sparse", hidden, head, top1, 1.0, 1, 1.0, 0.0, 0.0)
    assert c["kept_set_sparse_bwd"] > 0
    assert out.abs().max() == 0 and g.abs().max() == 0


def test_trainable_head_stays_dense_and_bit_equal(monkeypatch):
    hidden, head, index, _ = _inputs(seed = 9)
    args = (hidden, head, index, 0.7, 64, 0.95, 0.0, 0.0)
    _, gd, wd, _, _ = _run(monkeypatch, "dense", *args, head_grad = True)
    _, gs, ws, cs, _ = _run(monkeypatch, "sparse_full", *args, head_grad = True)
    assert cs["kept_set_sparse_bwd"] == 0 and cs["kept_set_dense_bwd"] > 0
    assert torch.equal(gs, gd) and torch.equal(ws, wd)


def test_tied_head_stays_dense(monkeypatch):
    # A tied embedding / head that trains is one leaf: needs_input_grad on the head forces the dense branch.
    hidden, head, index, _ = _inputs(seed = 10)
    emb = torch.nn.Embedding(V, H)
    with torch.no_grad():
        emb.weight.copy_(head)
    monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", raising = False)
    before = rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_sparse_bwd", 0)
    h = hidden.clone().requires_grad_(True)
    out = rr.grpo_chunked_hidden_states_logprobs(h, emb.weight, index, 3, 0.0, 0.0, 0.0, 1.0, support_top_k = 8)
    out.sum().backward()
    assert rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_sparse_bwd", 0) == before
    assert emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0


def test_csr_saves_less_than_the_bit_mask():
    g = torch.Generator().manual_seed(0)
    sparse_mask = torch.zeros(16, V, dtype = torch.bool)
    sparse_mask[torch.arange(16)[:, None], torch.randint(0, V, (16, 16), generator = g)] = True
    z = torch.randn(16, V, generator = g)
    kind, counts, cols, z_kept, raw, _ = rr._kept_set_save(sparse_mask, True, z)
    assert kind == "csr" and counts.dtype == torch.int32 and cols.dtype == torch.int32 and raw is None
    assert counts.numel() * 4 + cols.numel() * 8 < rr._pack_bool_rows(sparse_mask).numel()
    assert torch.equal(z_kept, z[sparse_mask])
    rows = torch.repeat_interleave(torch.arange(16), counts)
    rebuilt = torch.zeros_like(sparse_mask)
    rebuilt[rows, cols.long()] = True
    assert torch.equal(rebuilt, sparse_mask)
    dense_mask = torch.rand(16, V, generator = g) < 0.1
    # Declined: None, so the forward frees z before packing the mask.
    assert rr._kept_set_save(dense_mask, True, z) is None
    assert rr._kept_set_save(sparse_mask, False, z) is None
    # A softcap also keeps the raw logits there, for the float32 tanh derivative.
    raw_logits = z.to(torch.bfloat16)
    saved = rr._kept_set_save(sparse_mask, True, z, raw_logits, raw_logits.dtype)
    assert saved[0] == "csr" and torch.equal(saved[4], raw_logits[sparse_mask]) and saved[5] == torch.bfloat16


@pytest.mark.parametrize("chunks", [1, 2, 5, 111])
def test_ragged_chunks(monkeypatch, chunks):
    hidden, head, index, _ = _inputs(rows = (3, 37), seed = 13)
    args = (hidden, head, index, 1.0, 16, 0.9, 0.0, 0.0)
    _, gd, _, _, _ = _run(monkeypatch, "dense", *args, chunks = chunks)
    _, gs, _, cs, _ = _run(monkeypatch, "sparse", *args, chunks = chunks)
    assert cs["kept_set_sparse_bwd"] > 0
    torch.testing.assert_close(gs, gd, atol = 1e-6, rtol = 1e-6)


@pytest.mark.parametrize("limit, sparse_calls", [("0.001", 0), ("1.0", 1)])
@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_crossover_routes_by_union_density(monkeypatch, limit, sparse_calls, softcap):
    # Union ~25% of V: cutoff 0.001 gives full-width CSR, 1.0 gives W[U]; both match dense.
    hidden, head, index, _ = _inputs(seed = 15)
    _, gd, _, _, _ = _run(monkeypatch, "dense", hidden, head, index, 1.0, 8, 1.0, 0.0, softcap, chunks = 1)
    monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", raising = False)
    monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", limit)
    h = hidden.clone().requires_grad_(True)
    keys = ("kept_set_sparse_bwd", "kept_set_dense_bwd")
    before = dict(rr._GRPO_TRUNCATED_SUPPORT_STATS)
    out = rr.grpo_chunked_hidden_states_logprobs(h, head, index, 1, 0.0, 0.0, softcap, 1.0, support_top_k = 8)
    weights = torch.randn(tuple(out.shape), generator = torch.Generator().manual_seed(1))
    (out * weights).sum().backward()
    after = rr._GRPO_TRUNCATED_SUPPORT_STATS
    assert tuple(after.get(k, 0) - before.get(k, 0) for k in keys) == (sparse_calls, 1 - sparse_calls)
    torch.testing.assert_close(h.grad, gd, atol = 1e-6, rtol = 1e-6)


def test_wide_head_short_chunk_stays_dense(monkeypatch):
    # 4 rows, 4096-wide head: gathering W[U] costs more than 4 x V; byte guard keeps it dense.
    g = torch.Generator().manual_seed(21)
    hidden = torch.randn(1, 4, 4096, generator = g)
    head = torch.randn(V, 4096, generator = g) * 0.05
    index = torch.randint(0, V, (1, 4), generator = g)
    _, gd, _, _, _ = _run(monkeypatch, "dense", hidden, head, index, 1.0, 8, 1.0, 0.0, 0.0, chunks = 1)
    _, gs, _, cs, _ = _run(monkeypatch, "sparse_full", hidden, head, index, 1.0, 8, 1.0, 0.0, 0.0, chunks = 1)
    assert cs == {"kept_set_sparse_bwd": 0, "kept_set_dense_bwd": 1}
    torch.testing.assert_close(gs, gd, atol = 1e-6, rtol = 1e-6)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "junk"])
def test_non_finite_crossover_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", raising = False)
    monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", value)
    assert rr._kept_set_sparse_max_density() == rr._KEPT_SET_SPARSE_MAX_DENSITY


def test_trainable_bias_frozen_head_stays_dense(monkeypatch):
    hidden, head, index, b = _inputs(seed = 23, bias = True)
    grads = {}
    for mode in ("dense", "sparse_full"):
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", "0" if mode == "dense" else "1")
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", "1.0")
        h = hidden.clone().requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        before = rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_sparse_bwd", 0)
        out = rr.grpo_chunked_hidden_states_logprobs(h, head, index, 3, 0.0, 0.0, 0.0, 1.0, lm_head_bias = bb,
                                                     support_top_k = 8)
        out.sum().backward()
        assert rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_sparse_bwd", 0) == before
        grads[mode] = (h.grad, bb.grad)
    assert torch.equal(grads["dense"][0], grads["sparse_full"][0]) and torch.equal(grads["dense"][1], grads["sparse_full"][1])


def test_bf16_autocast_matches_dense(monkeypatch):
    hidden, head, index, _ = _inputs(seed = 25)
    outs = {}
    for mode in ("dense", "sparse_full"):
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", "0" if mode == "dense" else "1")
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", "1.0")
        h = hidden.to(torch.bfloat16).requires_grad_(True)
        before = rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_sparse_bwd", 0)
        with torch.autocast("cpu", dtype = torch.bfloat16):
            out = rr.grpo_chunked_hidden_states_logprobs(h, head.to(torch.bfloat16), index, 3, 2.0, 4.0, 0.0, 0.8,
                                                         support_top_k = 16, support_top_p = 0.9)
        out.float().sum().backward()
        sparse_calls = rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_sparse_bwd", 0) - before
        assert sparse_calls == (0 if mode == "dense" else 3)
        outs[mode] = (out.detach(), h.grad.float())
    assert torch.equal(outs["dense"][0], outs["sparse_full"][0])
    # bf16 hidden grad: dh GEMM blocking can move the last bf16 bit.
    torch.testing.assert_close(outs["sparse_full"][1], outs["dense"][1], atol = 1e-2, rtol = 1e-2)


@pytest.mark.parametrize("mode", ["sparse_full", "dense"])
def test_kept_sets_go_through_saved_tensor_hooks(monkeypatch, mode):
    # save_on_cpu / offload hooks must see the kept-set tensors in both layouts (CSR int32 ids, packed uint8 bits).
    monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", "0" if mode == "dense" else "1")
    monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", "1.0")
    hidden, head, index, _ = _inputs(seed = 27)
    seen = []
    def pack(t):
        seen.append(t.dtype)
        return t
    h = hidden.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = rr.grpo_chunked_hidden_states_logprobs(h, head, index, 3, 0.0, 0.0, 0.0, 1.0, support_top_k = 8)
    out.sum().backward()
    assert (torch.int32 in seen) if mode == "sparse_full" else (torch.uint8 in seen), seen


def test_outer_compile_traces_one_graph(monkeypatch):
    # Outer compile: no CSR size check or counter write while tracing; else fullgraph breaks.
    monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", raising = False)
    torch._dynamo.reset()
    hidden, head, index, _ = _inputs(rows = (2, 16), seed = 29)

    def score(h):
        return rr._KeptSetLogprobs.apply(
            h.reshape(-1, H), head, None, index.reshape(-1), (0, 16, 32), (0.0, 0.0, 0.0, 1.0),
            (8, 1.0, 0.0, False, None),
        )

    h_eager = hidden.clone().requires_grad_(True)
    ref = score(h_eager)
    ref.sum().backward()
    h = hidden.clone().requires_grad_(True)
    out = torch.compile(score, fullgraph = True, backend = "aot_eager")(h)
    out.sum().backward()
    torch.testing.assert_close(out, ref, atol = 1e-5, rtol = 0)
    torch.testing.assert_close(h.grad, h_eager.grad, atol = 1e-5, rtol = 1e-4)
    torch._dynamo.reset()


@pytest.mark.parametrize("mode", ["sparse_full", "dense"])
def test_retain_graph_backward_runs_twice(monkeypatch, mode):
    # The kept sets live in save_for_backward, so a retained graph can be differentiated again with equal grads.
    if mode == "dense":
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", "0")
    else:
        monkeypatch.delenv("UNSLOTH_GRPO_KEPT_SET_SPARSE", raising = False)
        monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_SPARSE_MAX_DENSITY", "1.0")
    hidden, head, index, _ = _inputs()
    h = hidden.clone().requires_grad_(True)
    out = rr.grpo_chunked_hidden_states_logprobs(
        h, head, index, 3, 0.0, 0.0, 30.0, 0.7, support_top_k = 64, support_top_p = 0.95,
    )
    loss = out.sum()
    (g1,) = torch.autograd.grad(loss, h, retain_graph = True)
    (g2,) = torch.autograd.grad(loss, h)
    assert torch.equal(g1, g2)
