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

"""unsloth_chunked_logprobs recomputes logits in backward; values and grads must match an fp64 oracle, and its GRPO path must match the old chunked function."""
import inspect

import pytest
import torch

from unsloth_zoo.fused_losses import chunked_logprob as cl
from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_logprobs, unsloth_masked_logprobs
from unsloth_zoo import rl_replacements as rr

VOCAB, HIDDEN = 97, 24


def _oracle(states, weight, index, bias = None, temperature = 1.0, mul = 0.0, div = 0.0, cap = 0.0):
    logits = torch.nn.functional.linear(states.double(), weight.double(), None if bias is None else bias.double())
    if mul:
        logits = logits * mul
    if div:
        logits = logits / div
    if cap:
        logits = cap * torch.tanh(logits / cap)
    logits = logits / temperature
    return logits.gather(-1, index.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def _inputs(shape = (3, 7), dtype = torch.float32, bias = False, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(*shape, HIDDEN, generator = gen, dtype = dtype).requires_grad_(True)
    weight = (0.5 * torch.randn(VOCAB, HIDDEN, generator = gen, dtype = dtype)).requires_grad_(True)
    b = (torch.randn(VOCAB, generator = gen, dtype = dtype)).requires_grad_(True) if bias else None
    index = torch.randint(0, VOCAB, shape, generator = gen)
    upstream = torch.randn(*shape, generator = gen, dtype = torch.float64)
    return states, weight, b, index, upstream


def _grads(fn, states, weight, bias, index, upstream):
    leaves = [t for t in (states, weight, bias) if t is not None]
    for t in leaves:
        t.grad = None
    out = fn()
    (out.double() * upstream).sum().backward()
    got = [out.detach().double()] + [t.grad.double().clone() for t in leaves]
    for t in leaves:
        t.grad = None
    return got


def _assert_close(got, ref, atol):
    for g, r in zip(got, ref):
        assert g.shape == r.shape
        torch.testing.assert_close(g, r, atol = atol, rtol = 0)


TRANSFORMS = [
    dict(),
    dict(temperature = 0.7),
    dict(cap = 30.0),
    dict(cap = 2.0, temperature = 1.3),
    dict(mul = 0.25),
    dict(div = 3.0),
    dict(mul = 1.5, div = 2.0, cap = 5.0, temperature = 0.9),
]


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("transform", TRANSFORMS, ids = str)
@pytest.mark.parametrize("rows_per_chunk", [1, 5, 2048])
def test_matches_fp64_oracle(bias, transform, rows_per_chunk):
    states, weight, b, index, upstream = _inputs(dtype = torch.float64, bias = bias)
    kw = dict(
        temperature = transform.get("temperature", 1.0),
        logit_scale_multiply = transform.get("mul", 0.0),
        logit_scale_divide = transform.get("div", 0.0),
        logit_softcapping = transform.get("cap", 0.0),
    )
    got = _grads(
        lambda: unsloth_chunked_logprobs(states, weight, index, lm_head_bias = b, rows_per_chunk = rows_per_chunk, **kw),
        states, weight, b, index, upstream,
    )
    ref = _grads(lambda: _oracle(states, weight, index, b, **transform), states, weight, b, index, upstream)
    _assert_close(got, ref, atol = 2e-5)


def test_output_contract():
    states, weight, _, index, _ = _inputs(shape = (2, 5))
    out = unsloth_chunked_logprobs(states, weight, index)
    assert out.shape == index.shape and out.dtype == torch.float32
    out, mean = unsloth_chunked_logprobs(states, weight, index, return_mean_logits = True)
    torch.testing.assert_close(mean.double(), (states.double() @ weight.double().t()).mean(-1), atol = 1e-5, rtol = 0)
    assert not mean.requires_grad


def test_frozen_head_gets_no_gradient_and_hidden_still_does():
    states, weight, _, index, upstream = _inputs()
    weight.requires_grad_(False)
    (unsloth_chunked_logprobs(states, weight, index, rows_per_chunk = 4).double() * upstream).sum().backward()
    assert weight.grad is None and states.grad is not None


def test_masked_helper_matches_oracle_and_zeroes_masked_rows():
    states, weight, _, index, upstream = _inputs(dtype = torch.float64)
    labels = index.clone()
    labels[0, :3] = -100
    labels[2, -1] = -100
    keep = labels != -100
    got = _grads(lambda: unsloth_masked_logprobs(states, weight, labels, rows_per_chunk = 3),
                 states, weight, None, index, upstream)
    ref = _grads(lambda: _oracle(states, weight, index) * keep, states, weight, None, index, upstream)
    _assert_close(got, ref, atol = 2e-5)
    assert (got[0][~keep] == 0).all()


@pytest.mark.parametrize("fused_linear", [False, True])
def test_integer_mask_selects_rows(monkeypatch, fused_linear):
    monkeypatch.setattr(cl, "_FUSED_LINEAR", fused_linear)
    states, weight, _, index, _ = _inputs()
    keep = torch.ones_like(index, dtype = torch.bool)
    keep[0, :3] = False
    want = unsloth_masked_logprobs(states, weight, index, mask = keep)
    got = unsloth_masked_logprobs(states, weight, index, mask = keep.long())
    assert torch.equal(got, want)


def test_all_masked_batch_is_a_connected_zero():
    states, weight, _, index, _ = _inputs()
    labels = torch.full_like(index, -100)
    out = unsloth_masked_logprobs(states, weight, labels)
    assert (out == 0).all() and out.requires_grad
    out.sum().backward()
    assert states.grad is None or (states.grad == 0).all()


def test_chunk_rows_floor():
    assert cl.chunked_logprob_rows(151936) >= 2
    assert cl.chunked_logprob_rows(10**12) == 2



def test_grpo_accumulated_loss_routes_through_the_recompute_kernel():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    assert "grpo_chunked_hidden_states_logprobs as chunked_hidden_states_selective_log_softmax" in src
    assert "_keeps_logits()" in src


def _old(*args, **kwargs):
    return rr.chunked_hidden_states_selective_log_softmax(*args, fused_linear = False, **kwargs)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("transform", [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 30.0, 0.7), (1.5, 2.0, 0.0, 1.2)], ids = str)
def test_grpo_kill_switch_gradient_is_the_old_function(monkeypatch, bias, transform):
    monkeypatch.setattr(cl, "_FUSED_LINEAR", False)
    states, weight, b, index, upstream = _inputs(shape = (4, 9), bias = bias, seed = 1)
    calls = []
    real = cl.UnslothChunkedLogProbs.apply
    cl.UnslothChunkedLogProbs.apply = lambda *a: calls.append(1) or real(*a)
    try:
        got = _grads(lambda: rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 8, *transform, lm_head_bias = b),
                     states, weight, b, index, upstream)
    finally:
        cl.UnslothChunkedLogProbs.apply = real
    assert not calls
    ref = _grads(lambda: _old(states, weight, index, 8, *transform, lm_head_bias = b), states, weight, b, index, upstream)
    _assert_close(got, ref, atol = 0)


def test_grpo_route_no_grad_is_the_old_function_bitwise(monkeypatch):
    monkeypatch.setattr(cl, "_FUSED_LINEAR", False)
    states, weight, _, index, _ = _inputs()
    calls = []
    real = cl.UnslothChunkedLogProbs.apply
    cl.UnslothChunkedLogProbs.apply = lambda *a: calls.append(1) or real(*a)
    try:
        with torch.no_grad():
            got = rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4, 0.0, 0.0, 10.0, 0.8)
            ref = _old(states, weight, index, 4, 0.0, 0.0, 10.0, 0.8)
    finally:
        cl.UnslothChunkedLogProbs.apply = real
    assert not calls
    assert torch.equal(got, ref)


def test_grpo_route_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_CHUNKED_LOGPROBS", "0")
    states, weight, _, index, upstream = _inputs()
    calls = []
    real = cl.UnslothChunkedLogProbs.apply
    cl.UnslothChunkedLogProbs.apply = lambda *a: calls.append(1) or real(*a)
    try:
        got = _grads(lambda: rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4),
                     states, weight, None, index, upstream)
    finally:
        cl.UnslothChunkedLogProbs.apply = real
    assert not calls
    ref = _grads(lambda: _old(states, weight, index, 4), states, weight, None, index, upstream)
    _assert_close(got, ref, atol = 0)


def test_grpo_kill_switch_passes_max_rows_per_chunk(monkeypatch):
    monkeypatch.setattr(cl, "_FUSED_LINEAR", False)
    seen = []
    real = rr.chunked_hidden_states_selective_log_softmax
    monkeypatch.setattr(rr, "chunked_hidden_states_selective_log_softmax",
                        lambda *a, **k: seen.append(k.get("max_rows_per_chunk")) or real(*a, **k))
    states, weight, _, index, _ = _inputs()
    rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4, max_rows_per_chunk = 3)
    rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4)
    assert seen == [3, None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the fp32-output GEMM path only runs on a GPU")
@pytest.mark.parametrize("fp32_out_gemm", [True, False])
def test_fp32_output_gemm_only_where_fast(monkeypatch, fp32_out_gemm):
    monkeypatch.setattr(cl, "_FP32_OUT_GEMM", fp32_out_gemm)
    a = torch.randn(8, 16, device = "cuda", dtype = torch.bfloat16)
    w = torch.randn(32, 16, device = "cuda", dtype = torch.bfloat16)
    has_out_dtype = tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 8)
    want = torch.float32 if fp32_out_gemm and has_out_dtype and torch.cuda.get_device_capability() >= (8, 0) \
        else torch.bfloat16
    assert cl._mm_fp32(a, w.t()).dtype == want
    assert cl._project(a, w, None, torch.bfloat16, fp32_out = True).dtype == want
    hidden = a.float().requires_grad_(True)
    logps = unsloth_chunked_logprobs(hidden.bfloat16(), w, torch.randint(0, 32, (8,), device = "cuda"))
    logps.sum().backward()
    assert torch.isfinite(logps).all() and torch.isfinite(hidden.grad).all()


def test_default_chunk_budget_shrinks_with_free_memory(monkeypatch):
    import unsloth_zoo.tiled_mlp as tm

    for free, want in ((0.25, 0.25), (64.0, cl._CHUNK_GB)):
        monkeypatch.setattr(tm, "_default_target_gb", lambda device = None, free = free: free)
        assert cl._default_chunk_gb() == want
        assert cl.chunked_logprob_rows(1000) == cl.chunked_logprob_rows(1000, target_gb = want)


def test_default_chunk_budget_is_read_per_call(monkeypatch):
    # A first call before the optimizer states exist must not size later chunks for memory that is gone.
    import unsloth_zoo.tiled_mlp as tm

    free = [64.0]
    monkeypatch.setattr(tm, "_default_target_gb", lambda device = None: free[0])
    assert cl._default_chunk_gb() == cl._CHUNK_GB
    free[0] = 0.25
    assert cl._default_chunk_gb() == 0.25


def test_fp32_projection_reraises_oom(monkeypatch):
    # An OOM in the fp32-output GEMM must surface, not fall back to half logits mixed with an fp32 lse.
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(cl, "_FP32_OUT_GEMM", True)
    monkeypatch.setattr(torch, "mm", oom)
    a = type("CudaLike", (), {"dtype": torch.bfloat16, "is_cuda": True})()
    with pytest.raises(torch.OutOfMemoryError):
        cl._mm_fp32(a, None)


def test_chunk_budget_reads_the_head_device(monkeypatch):
    import unsloth_zoo.tiled_mlp as tm

    seen = []
    monkeypatch.setattr(tm, "_default_target_gb", lambda device = None: seen.append(device) or 0.25)
    dev = torch.device("cuda", 1)
    assert cl._default_chunk_gb(dev) == 0.25 and seen == [dev]


def test_forward_casts_the_head_once(monkeypatch):
    monkeypatch.setattr(cl, "_FUSED_LINEAR", False)
    weights = []
    project = cl._project

    def spy(hidden, weight, *args, **kwargs):
        weights.append(weight)
        return project(hidden, weight, *args, **kwargs)

    monkeypatch.setattr(cl, "_project", spy)
    head = torch.randn(32, 16)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        unsloth_chunked_logprobs(torch.randn(10, 16), head, torch.randint(0, 32, (10,)), rows_per_chunk = 3)
    assert len(weights) == 4 and all(w.dtype == torch.bfloat16 for w in weights)
    assert len({w.data_ptr() for w in weights}) == 1


def test_half_trainable_head_gradient_keeps_its_dtype():
    torch.manual_seed(0)
    hidden = torch.randn(12, 16, requires_grad = True)
    head = torch.randn(32, 16).bfloat16().requires_grad_(True)
    targets = torch.randint(0, 32, (12,))
    unsloth_chunked_logprobs(hidden.bfloat16(), head, targets, rows_per_chunk = 5).sum().backward()
    hd, wd = hidden.detach().double(), head.detach().double().requires_grad_(True)
    torch.log_softmax(hd.bfloat16().double() @ wd.t(), -1).gather(-1, targets[:, None]).sum().backward()
    assert head.grad.dtype == torch.bfloat16
    torch.testing.assert_close(head.grad.double(), wd.grad, rtol = 3e-2, atol = 3e-2)


def test_rows_honour_the_opt_in_128_row_alignment(monkeypatch):
    monkeypatch.setattr(cl, "CHUNK_ALIGN_ROWS", 128)
    assert cl.chunked_logprob_rows(1000, target_gb = 1.0, n_rows = 8 * 700) == 640
    assert cl.chunked_logprob_rows(1000, target_gb = 1.0, n_rows = 8 * 150) == 512
    monkeypatch.setattr(cl, "CHUNK_ALIGN_ROWS", 0)
    assert cl.chunked_logprob_rows(1000, target_gb = 1.0, n_rows = 8 * 700) == 700


@pytest.mark.parametrize("extra", [{"support_top_k": 4}, {"support_top_p": 0.9}, {"fp32_lm_head": True}])
def test_grpo_route_hands_support_and_fp32_head_calls_to_the_old_function(monkeypatch, extra):
    seen = []
    monkeypatch.setattr(rr, "chunked_hidden_states_selective_log_softmax", lambda *a, **k: seen.append(k) or "old")
    hidden = torch.randn(1, 4, 8, requires_grad = True)
    out = rr.grpo_chunked_hidden_states_logprobs(hidden, torch.randn(16, 8), torch.randint(0, 16, (1, 4)), **extra)
    assert out == "old" and all(seen[0][k] == v for k, v in extra.items())


def test_masked_logprobs_forwards_max_rows_per_chunk(monkeypatch):
    from unsloth_zoo.fused_losses import fused_linear as fl

    monkeypatch.setattr(cl, "_FUSED_LINEAR", True)
    seen = {}
    real = fl.fused_linear_logprobs
    monkeypatch.setattr(fl, "fused_linear_logprobs", lambda *a, **k: seen.update(k) or real(*a, **k))
    states = torch.randn(1, 6, 8)
    weight = torch.randn(16, 8)
    labels = torch.randint(0, 16, (1, 6))
    labels[0, 0] = -100
    out = unsloth_masked_logprobs(states, weight, labels, max_rows_per_chunk = 2)
    assert seen["max_rows_per_chunk"] == 2
    torch.testing.assert_close(out, unsloth_masked_logprobs(states, weight, labels))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_public_scorer_matches_the_grpo_policy_scorer():
    # function, while grpo_accumulated_loss scores the policy with the fused kernel: identical frozen weights must
    # give a ratio of exactly 1.
    if not cl._FUSED_LINEAR:
        pytest.skip("UNSLOTH_FUSED_LINEAR=0")
    gen = torch.Generator().manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        h = torch.randn(2, 33, 64, generator = gen).cuda().to(dtype)
        w = (torch.randn(40000, 64, generator = gen) / 8).cuda().to(dtype)
        idx = torch.randint(0, 40000, (2, 33), generator = gen).cuda()
        with torch.no_grad():
            ref = rr.chunked_hidden_states_selective_log_softmax(h, w, idx)
        pol = rr.grpo_chunked_hidden_states_logprobs(h.clone().requires_grad_(True), w, idx)
        assert ref.shape == pol.shape == idx.shape
        assert torch.equal(pol.detach(), ref), dtype
        old = rr.chunked_hidden_states_selective_log_softmax(h, w, idx, fused_linear = False)
        assert old.shape == idx.shape


def test_grpo_scorer_keeps_logits_truth_table(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_CHUNKED_LOGPROBS", raising = False)
    monkeypatch.delenv("UNSLOTH_VLLM_FP32_LM_HEAD", raising = False)
    monkeypatch.setattr(cl, "_FUSED_LINEAR", True)
    assert not rr.grpo_scorer_keeps_logits()
    assert rr.grpo_scorer_keeps_logits({"support_top_k": 20})
    assert rr.grpo_scorer_keeps_logits({"support_top_p": 0.9})
    assert rr.grpo_scorer_keeps_logits(fp32_lm_head = True)
    monkeypatch.setenv("UNSLOTH_GRPO_CHUNKED_LOGPROBS", "0")
    assert rr.grpo_scorer_keeps_logits()
    monkeypatch.delenv("UNSLOTH_GRPO_CHUNKED_LOGPROBS")
    monkeypatch.setattr(cl, "_FUSED_LINEAR", False)
    assert rr.grpo_scorer_keeps_logits()


@pytest.mark.parametrize("fused, support, length, batch, offloaded", [
    (True, {}, 2000, 16, False),            # fused scorer: recomputes in backward, never offloads
    (False, {}, 2000, 16, True),            # kill switch: the old scorer keeps logits, long completions offload
    (True, {"support_top_k": 20}, 2000, 16, True),
    (False, {}, 100, 4, False),             # short completions: the plain path, as before
])
def test_efficient_log_softmax_offloads_only_for_the_old_scorer(monkeypatch, fused, support, length, batch, offloaded):
    import os, textwrap
    monkeypatch.delenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_CHUNKED_LOGPROBS", raising = False)
    monkeypatch.delenv("UNSLOTH_VLLM_FP32_LM_HEAD", raising = False)
    monkeypatch.setattr(cl, "_FUSED_LINEAR", fused)
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.rindex("\n", 0, src.index("def to_device")) + 1
    end = src.rindex("\n", 0, src.index("def compute_logprobs_chunk")) + 1
    calls = []
    ns = {"torch": torch, "os": os, "_support_kwargs": support,
          "chunked_hidden_states_selective_log_softmax": lambda *a, **k: calls.append("plain")}
    exec(textwrap.dedent(src[start:end]), ns)
    ns["Unsloth_Offloaded_Log_Softmax"].apply = staticmethod(lambda *a: calls.append("offloaded"))
    hidden = torch.zeros(batch, length, 4)
    ns["efficient_log_softmax"](hidden, torch.zeros(8, 4), torch.zeros(batch, length, dtype = torch.long), batch_size = batch)
    assert calls == ["offloaded" if offloaded else "plain"]


@pytest.mark.parametrize("flag", ["UNSLOTH_GRPO_CHUNKED_LOGPROBS", "UNSLOTH_FUSED_LINEAR"])
def test_grpo_switches_keep_entropy_policy_and_reference_on_one_scorer(flag):
    import json, os, subprocess, sys
    from unsloth_zoo.fused_losses import fused_linear as fl
    if torch.cuda.is_available() and not fl._HAS_TRITON:
        pytest.skip("the old scorers compile through Inductor, which needs Triton for GPU code")
    child = (
        "import json, torch\n"
        "from unsloth_zoo import rl_replacements as rr\n"
        "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "g = torch.Generator().manual_seed(0)\n"
        "h = torch.randn(2, 17, 32, generator = g).to(dev, torch.bfloat16 if dev == 'cuda' else torch.float32)\n"
        "w = (torch.randn(5000, 32, generator = g) / 6).to(dev, h.dtype)\n"
        "idx = torch.randint(0, 5000, (2, 17), generator = g).to(dev)\n"
        "with torch.no_grad():\n"
        "    ref = rr.chunked_hidden_states_selective_log_softmax(h, w, idx, 4)\n"
        "pol, ent = rr.chunked_hidden_states_selective_log_softmax_recompute(h.clone().requires_grad_(True), w, idx, 4, return_entropy = True)\n"
        "print('RESULT ' + json.dumps({'equal': bool(torch.equal(pol.detach().reshape(ref.shape), ref)),\n"
        "    'maxdiff': float((pol.detach().reshape(ref.shape) - ref).abs().max())}))\n"
    )
    env = dict(os.environ, **{flag: "0", "UNSLOTH_IS_PRESENT": "1"})
    p = subprocess.run([sys.executable, "-c", child], env = env, capture_output = True, text = True, timeout = 900)
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, p.stdout[-2000:] + p.stderr[-2000:]
    r = json.loads(line[-1][7:])
    assert r["equal"], r
