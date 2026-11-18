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

"""Kept-set GRPO scorer (top_k / top_p / min_p / excluded ids): values and gradients against a float64 oracle
log p(t) - log p(kept set), equality with the plain chunked helper's support branch, and the memory / compile
contract: no float32 [rows, vocab] tensor is held for the backward and the fullgraph helper is not entered."""

import os
import sys

import pytest
import torch

import unsloth_zoo.rl_replacements as rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _vllm_topk_topp_reference import apply_top_k_top_p_pytorch  # noqa: E402

V, H = 5003, 48  # vocab not a multiple of 8: exercises the bit packing tail

CASES = [
    # temperature, top_k, top_p, min_p
    (1.0, 0, 0.9, 0.0),
    (0.7, 0, 0.95, 0.0),
    (1.0, 20, 1.0, 0.0),
    (0.8, 0, 1.0, 0.05),
    (0.9, 50, 0.9, 0.02),
]


def _inputs(rows = (3, 29), dtype = torch.float32, device = "cpu", seed = 0, bias = False):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(*rows, H, generator = g) * 2.0).to(device = device, dtype = dtype)
    head = (torch.randn(V, H, generator = g) * 0.5).to(device = device, dtype = dtype)
    index = torch.randint(0, V, rows, generator = g).to(device)
    b = (torch.randn(V, generator = g) * 0.1).to(device = device, dtype = dtype) if bias else None
    return hidden, head, index, b


def _oracle(hidden, head, index, temperature, top_k, top_p, min_p, bias = None, softcap = 0.0, excluded = None):
    """float64: kept set as vLLM v1 (min_p on probs, then apply_top_k_top_p), target forced in, then
    log p(t) - log p(K) = z_t - logsumexp_K z. Returns (logp, kept) with logp differentiable in float64."""
    h = hidden.detach().cpu().double().requires_grad_(True)
    w = head.detach().cpu().double().requires_grad_(True)
    b = bias.detach().cpu().double().requires_grad_(True) if bias is not None else None
    z = torch.nn.functional.linear(h.reshape(-1, H), w, b)
    if softcap:
        z = softcap * torch.tanh(z / softcap)
    z = z / temperature
    idx = index.reshape(-1).cpu()
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
    return logp.reshape(tuple(index.shape)), kept, (h, w, b)


def _new(hidden, head, index, temperature, top_k, top_p, min_p, bias = None, chunks = 4, **kw):
    return rr.grpo_chunked_hidden_states_logprobs(
        hidden, head, index, chunks, 0.0, 0.0, kw.pop("softcap", 0.0), temperature, lm_head_bias = bias,
        support_top_k = top_k, support_top_p = top_p, support_min_p = min_p, **kw,
    )


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("temperature, top_k, top_p, min_p", CASES)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_values_and_grads_match_float64_oracle(temperature, top_k, top_p, min_p, bias, device):
    hidden, head, index, b = _inputs(bias = bias, device = device)
    hidden.requires_grad_(True)
    head.requires_grad_(True)
    if b is not None:
        b.requires_grad_(True)
    got = _new(hidden, head, index, temperature, top_k, top_p, min_p, bias = b)
    ref, kept, leaves = _oracle(hidden, head, index, temperature, top_k, top_p, min_p, bias = b)
    torch.testing.assert_close(got.double().cpu(), ref.detach(), atol = 2e-5, rtol = 0)
    # The kept set really is a strict subset (else this tests the full vocabulary).
    assert (kept.sum(-1) < V).all()
    w = torch.randn(tuple(got.shape), generator = torch.Generator().manual_seed(1)).abs()
    (got * w.to(device)).sum().backward()
    (ref * w.double()).sum().backward()
    torch.testing.assert_close(hidden.grad.double().cpu(), leaves[0].grad, atol = 2e-5, rtol = 1e-4)
    torch.testing.assert_close(head.grad.double().cpu(), leaves[1].grad, atol = 2e-5, rtol = 1e-4)
    if b is not None:
        torch.testing.assert_close(b.grad.double().cpu(), leaves[2].grad, atol = 2e-5, rtol = 1e-4)


@pytest.mark.parametrize("temperature, top_k, top_p, min_p", CASES)
def test_softcap_and_excluded_ids_match_float64_oracle(temperature, top_k, top_p, min_p):
    hidden, head, index, _ = _inputs(seed = 3)
    excluded = torch.arange(V - 40, V)
    index[0, :5] = V - 1  # targets inside the excluded rows stay scored (forced into the kept set)
    hidden.requires_grad_(True)
    head.requires_grad_(True)
    got = _new(hidden, head, index, temperature, top_k, top_p, min_p, softcap = 30.0, support_excluded_ids = excluded)
    ref, _, leaves = _oracle(hidden, head, index, temperature, top_k, top_p, min_p, softcap = 30.0, excluded = excluded)
    torch.testing.assert_close(got.double(), ref.detach(), atol = 2e-5, rtol = 0)
    got.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(hidden.grad.double(), leaves[0].grad, atol = 2e-5, rtol = 1e-4)
    torch.testing.assert_close(head.grad.double(), leaves[1].grad, atol = 2e-5, rtol = 1e-4)


@pytest.mark.parametrize("temperature, top_k, top_p, min_p", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_the_plain_helper_support_branch(temperature, top_k, top_p, min_p, dtype):
    # Same kept set and tie rules as chunked_hidden_states_selective_log_softmax's support branch (main's scorer),
    # at the half dtypes training runs in: forward equal, gradients equal to rounding.
    hidden, head, index, _ = _inputs(dtype = dtype, seed = 5)
    h1 = hidden.clone().requires_grad_(True)
    w1 = head.clone().requires_grad_(True)
    h2 = hidden.clone().requires_grad_(True)
    w2 = head.clone().requires_grad_(True)
    old_fn = getattr(rr.chunked_hidden_states_selective_log_softmax, "__wrapped__",
                     rr.chunked_hidden_states_selective_log_softmax)
    old = old_fn(h1, w1, index, 4, 0.0, 0.0, 0.0, temperature, support_top_k = top_k, support_top_p = top_p,
                 support_min_p = min_p, fused_linear = False)
    new = _new(h2, w2, index, temperature, top_k, top_p, min_p)
    torch.testing.assert_close(new, old, atol = 0, rtol = 0)
    old.sum().backward()
    new.sum().backward()
    tol = dict(atol = 1e-2, rtol = 1e-2)
    torch.testing.assert_close(h2.grad.float(), h1.grad.float(), **tol)
    torch.testing.assert_close(w2.grad.float(), w1.grad.float(), **tol)


def test_target_outside_the_kept_set_and_singleton_sets():
    # A target the kept set drops is swapped in (finite value, gradient flows); a singleton kept set scores 0.
    hidden, head, index, _ = _inputs(seed = 7)
    with torch.no_grad():
        z = (hidden.reshape(-1, H) @ head.t())
        index = z.argmin(-1).reshape(index.shape)  # least likely token: never in a top-p 0.5 set
    hidden.requires_grad_(True)
    got = _new(hidden, head, index, 1.0, 0, 0.5, 0.0)
    ref, _, _ = _oracle(hidden, head, index, 1.0, 0, 0.5, 0.0)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.double(), ref.detach(), atol = 2e-5, rtol = 0)
    got.sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    hidden2 = hidden.detach().clone().requires_grad_(True)
    top1 = (hidden2.detach().reshape(-1, H) @ head.t()).argmax(-1).reshape(index.shape)
    one = _new(hidden2, head, top1, 1.0, 1, 1.0, 0.0)
    assert one.abs().max() == 0
    one.sum().backward()
    assert hidden2.grad.abs().max() == 0


def test_no_grad_scoring_equals_the_grad_forward():
    hidden, head, index, _ = _inputs(seed = 9)
    with torch.no_grad():
        a = _new(hidden, head, index, 0.7, 0, 0.9, 0.0)
    b = _new(hidden.clone().requires_grad_(True), head, index, 0.7, 0, 0.9, 0.0)
    torch.testing.assert_close(a, b.detach(), atol = 0, rtol = 0)


def test_chunk_count_does_not_change_values():
    hidden, head, index, _ = _inputs(seed = 11)
    outs = [_new(hidden, head, index, 1.0, 0, 0.9, 0.0, chunks = c) for c in (1, 3, 87)]
    # Only the matmul row blocking changes with the chunking (GEMM rounding), never the kept set.
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol = 2e-5, rtol = 0)


def test_bit_packing_round_trips():
    g = torch.Generator().manual_seed(0)
    for width in (1, 7, 8, 9, 5003):
        mask = torch.rand(13, width, generator = g) < 0.3
        assert torch.equal(rr._unpack_bool_rows(rr._pack_bool_rows(mask), width), mask)


def _saved_bytes(fn):
    total = [0]
    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, total[0]


@pytest.mark.parametrize("temperature, top_k, top_p, min_p", CASES)
def test_policy_backward_holds_no_float32_logits(temperature, top_k, top_p, min_p):
    # Flip test: main's scorer keeps float32 [rows, vocab] tensors per chunk for the backward (4.6-6.5 GB on a T4
    # in zoo #2084, the L4 OOM at 32 x 256); the recompute scorer keeps hidden, head and a 1-bit kept set.
    hidden, head, index, _ = _inputs(rows = (4, 64), seed = 13)
    hidden.requires_grad_(True)
    rows = index.numel()
    out, saved = _saved_bytes(lambda: _new(hidden, head, index, temperature, top_k, top_p, min_p))
    inputs = hidden.numel() * hidden.element_size() + head.numel() * head.element_size()
    # Main: >= rows * V * 4 bytes (one float32 logits copy). Head: inputs + rows * V / 8 bytes + per-row stats.
    assert saved - inputs < rows * V // 2, (saved, inputs, rows * V)
    out.sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_kept_set_does_not_enter_the_fullgraph_helper(monkeypatch):
    # Flip test: main routes the kept set through the fullgraph-compiled chunk loop (sort / cumsum / scatter per
    # chunk), which took about 7 minutes to compile on a Colab L4 and fell back to eager (InductorError).
    def refuse(*args, **kwargs):
        raise AssertionError("kept-set scoring entered chunked_hidden_states_selective_log_softmax")
    monkeypatch.setattr(rr, "chunked_hidden_states_selective_log_softmax", refuse)
    hidden, head, index, _ = _inputs(seed = 15)
    before = rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_recompute", 0)
    _new(hidden.requires_grad_(True), head, index, 1.0, 0, 0.9, 0.0).sum().backward()
    assert rr._GRPO_TRUNCATED_SUPPORT_STATS.get("kept_set_recompute", 0) == before + 1


def test_kill_switch_restores_the_plain_helper(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_KEPT_SET_RECOMPUTE", "0")
    calls = []
    real = rr.chunked_hidden_states_selective_log_softmax
    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(rr, "chunked_hidden_states_selective_log_softmax", spy)
    hidden, head, index, _ = _inputs(seed = 17)
    _new(hidden, head, index, 1.0, 0, 0.9, 0.0)
    assert calls == [1]


def test_full_vocabulary_calls_are_untouched(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("full-vocabulary scoring took the kept-set scorer")
    monkeypatch.setattr(rr, "grpo_kept_set_logprobs", refuse, raising = False)
    hidden, head, index, _ = _inputs(seed = 19)
    rr.grpo_chunked_hidden_states_logprobs(hidden, head, index, 4, 0.0, 0.0, 0.0, 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA memory check")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_memory_is_bounded_and_values_match(dtype):
    # Llama-3.2 vocabulary, 8 x 256 completion rows: retained memory after the forward stays near the 1-bit kept
    # set, the forward peak near one capped chunk, and values / gradients match the plain helper.
    torch.manual_seed(0)
    Vb, Hb = 128256, 256
    head = (torch.randn(Vb, Hb, device = "cuda") * 0.05).to(dtype)
    hidden = torch.randn(8, 256, Hb, device = "cuda", dtype = dtype)
    index = torch.randint(0, Vb, (8, 256), device = "cuda")
    rows = index.numel()
    results = {}
    for name in ("old", "new"):
        h = hidden.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype = dtype):
            if name == "old":
                fn = getattr(rr.chunked_hidden_states_selective_log_softmax, "__wrapped__",
                             rr.chunked_hidden_states_selective_log_softmax)
                out = fn(h, head, index, 4, 0.0, 0.0, 0.0, 0.8, support_top_p = 0.95, fused_linear = False)
            else:
                out = rr.grpo_chunked_hidden_states_logprobs(h, head, index, 4, 0.0, 0.0, 0.0, 0.8, support_top_p = 0.95)
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - base
        out.float().sum().backward()
        torch.cuda.synchronize()
        results[name] = dict(out = out.detach().float(), grad = h.grad.float(), retained = retained,
                             peak = torch.cuda.max_memory_allocated() - base)
        del out, h
    print({k: (v["retained"], v["peak"]) for k, v in results.items()})
    # Flip: main's scorer holds float32 [rows, vocab] logits per chunk until the backward.
    assert results["old"]["retained"] > rows * Vb * 4
    assert results["new"]["retained"] < rows * Vb // 4, results["new"]["retained"]
    assert results["new"]["peak"] < 0.75 * results["old"]["peak"], (results["new"]["peak"], results["old"]["peak"])
    torch.testing.assert_close(results["new"]["out"], results["old"]["out"], atol = 1e-5, rtol = 0)
    torch.testing.assert_close(results["new"]["grad"], results["old"]["grad"], atol = 2e-2, rtol = 2e-2)
