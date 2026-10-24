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

"""chunked_hidden_states_selective_log_softmax_recompute against a float64 full-logits oracle.

Like the plain chunked function, it takes the softmax in float32 whatever the input dtype, so the
oracle tolerances are float32 ones, and the plain function's own error is the bar to stay under.

The function's source is executed on its own in a namespace holding only torch, os, typing and
math, the way unsloth copies these helpers into generated trainers, so a hidden dependency on
rl_replacements module globals fails here.
"""
import inspect
import math
import os
import typing

import pytest
import torch

import unsloth_zoo.rl_replacements as rr


def _standalone(compiled = False):
    namespace = {"torch": torch, "os": os, "math": math, "Optional": typing.Optional}
    source = inspect.getsource(rr.chunked_hidden_states_selective_log_softmax_recompute)
    exec(compile(source, "<recompute>", "exec"), namespace)
    function = namespace["chunked_hidden_states_selective_log_softmax_recompute"]
    if not compiled:
        function._use_compiled = False
    return function


def _oracle(hidden, head, bias, index, multiply, divide, softcap, temperature):
    logits = hidden @ head.t()
    if bias is not None:
        logits = logits + bias
    if multiply != 0.0:
        logits = logits * multiply
    if divide != 0.0:
        logits = logits / divide
    if softcap != 0.0:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits / temperature
    log_probs = torch.log_softmax(logits, dim = -1)
    logps = log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return logps, entropy


def _inputs(rows, vocab = 37, hidden = 6, bias = False, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    head = (torch.randn(vocab, hidden, generator = gen, dtype = torch.float64) * 0.7).requires_grad_(True)
    states = torch.randn(2, rows, hidden, generator = gen, dtype = torch.float64).requires_grad_(True)
    head_bias = (torch.randn(vocab, generator = gen, dtype = torch.float64) * 0.3).requires_grad_(True) if bias else None
    index = torch.randint(0, vocab, (2, rows), generator = gen)
    weights = torch.randn(2, rows, generator = gen, dtype = torch.float64)
    entropy_weights = torch.randn(2, rows, generator = gen, dtype = torch.float64)
    return head, states, head_bias, index, weights, entropy_weights


def _close(got, want):
    torch.testing.assert_close(got.double(), want.double(), rtol = 2e-5, atol = 2e-6)


def _error(got, want):
    return float((got.double() - want.double()).abs().max())


def _grads(outputs, weights, entropy_weights, leaves):
    logps, entropy = outputs
    loss = (logps * weights).sum() + (entropy * entropy_weights).sum()
    return torch.autograd.grad(loss, [x for x in leaves if x is not None])


TRANSFORMS = [
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.7),
    (1.7, 0.0, 0.0, 1.0),
    (0.0, 2.5, 0.0, 1.3),
    (0.0, 0.0, 3.0, 0.9),
    (0.6, 1.5, 2.0, 1.1),
]


@pytest.mark.parametrize("transforms", TRANSFORMS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("chunks, max_rows", [(1, 0), (3, 0), (7, 0), (2, 3)])
def test_matches_float64_oracle(transforms, bias, chunks, max_rows):
    function = _standalone()
    head, states, head_bias, index, weights, entropy_weights = _inputs(rows = 11, bias = bias)
    leaves = [states, head, head_bias]
    got = function(
        states, head, index, chunks, *transforms,
        max_rows_per_chunk = max_rows, lm_head_bias = head_bias, return_entropy = True,
    )
    want = _oracle(states.reshape(-1, states.shape[-1]), head, head_bias, index.reshape(-1), *transforms)
    want = tuple(x.reshape(2, -1) for x in want)
    for a, b in zip(got, want):
        _close(a, b)
    for a, b in zip(_grads(got, weights, entropy_weights, leaves), _grads(want, weights, entropy_weights, leaves)):
        _close(a, b)


@pytest.mark.parametrize("transforms", TRANSFORMS)
def test_oracle_error_no_worse_than_plain_function(transforms):
    function = _standalone()
    head, states, head_bias, index, weights, _ = _inputs(rows = 9, bias = True)
    leaves = [states, head, head_bias]
    oracle = _oracle(states.reshape(-1, states.shape[-1]), head, head_bias, index.reshape(-1), *transforms)[0].reshape(2, -1)
    got = function(states, head, index, 4, *transforms, lm_head_bias = head_bias)
    plain = rr.chunked_hidden_states_selective_log_softmax(states, head, index, 4, *transforms, lm_head_bias = head_bias)
    assert got.dtype == plain.dtype == torch.float32
    torch.testing.assert_close(got, plain, rtol = 1e-6, atol = 1e-6)
    want = torch.autograd.grad((oracle * weights).sum(), leaves)
    for x, y, z in zip(torch.autograd.grad((got * weights).sum(), leaves), torch.autograd.grad((plain * weights).sum(), leaves), want):
        # float32 rounding alone moves either error by a few ulps of the gradient scale
        slack = 16 * torch.finfo(torch.float32).eps * max(1.0, float(z.abs().max()))
        assert _error(x, z) <= 2 * _error(y, z) + slack, (_error(x, z), _error(y, z), slack)


def test_chunk_count_invariance():
    function = _standalone()
    head, states, head_bias, index, weights, entropy_weights = _inputs(rows = 13, bias = True, seed = 3)
    leaves = [states, head, head_bias]
    reference = None
    for chunks in (1, 2, 5, 13, 26):
        out = function(states, head, index, chunks, 0.0, 0.0, 0.0, 0.9, lm_head_bias = head_bias, return_entropy = True)
        result = list(out) + list(_grads(out, weights, entropy_weights, leaves))
        if reference is None:
            reference = result
            continue
        for a, b in zip(result, reference):
            torch.testing.assert_close(a, b, rtol = 1e-6, atol = 1e-7)


@pytest.mark.parametrize("rows, chunks, max_rows", [(1, 4, 0), (3, 16, 0), (49, 16, 2), (5, 2, 1)])
def test_one_row_and_tiny_chunks(rows, chunks, max_rows):
    function = _standalone()
    head, states, head_bias, index, weights, entropy_weights = _inputs(rows = rows, seed = rows)
    leaves = [states, head]
    got = function(states, head, index, chunks, max_rows_per_chunk = max_rows, return_entropy = True)
    want = _oracle(states.reshape(-1, states.shape[-1]), head, None, index.reshape(-1), 0.0, 0.0, 0.0, 1.0)
    want = tuple(x.reshape(2, -1) for x in want)
    for a, b in zip(_grads(got, weights, entropy_weights, leaves), _grads(want, weights, entropy_weights, leaves)):
        _close(a, b)


def test_gradcheck():
    function = _standalone()
    head, states, head_bias, index, _, _ = _inputs(rows = 4, vocab = 7, hidden = 3, bias = True, seed = 7)

    def f(s, h, b):
        logps, entropy = function(s, h, index, 3, 0.0, 0.0, 2.0, 0.8, lm_head_bias = b, return_entropy = True)
        return logps, entropy

    # The softmax runs in float32, so a coarse finite-difference step.
    assert torch.autograd.gradcheck(f, (states, head, head_bias), eps = 1e-3, atol = 2e-3, rtol = 1e-2)


def test_frozen_head_and_no_grad():
    function = _standalone()
    head, states, _, index, weights, _ = _inputs(rows = 6)
    head = head.detach()
    got = function(states, head, index, 3)
    (grad,) = torch.autograd.grad((got * weights).sum(), [states])
    want = _oracle(states.reshape(-1, states.shape[-1]), head, None, index.reshape(-1), 0.0, 0.0, 0.0, 1.0)[0].reshape(2, -1)
    (want_grad,) = torch.autograd.grad((want * weights).sum(), [states])
    _close(grad, want_grad)
    with torch.no_grad():
        _close(function(states, head, index, 3), want)


def test_saves_no_logits_for_backward():
    # The plain chunked function keeps rows x vocab logits per chunk alive for backward; this one
    # must keep only its inputs plus per-row statistics, whatever the vocabulary.
    function = _standalone()
    vocab, hidden, rows = 4096, 8, 64
    head = torch.randn(vocab, hidden, dtype = torch.float64, requires_grad = True)
    states = torch.randn(1, rows, hidden, dtype = torch.float64, requires_grad = True)
    index = torch.randint(0, vocab, (1, rows))
    saved = []

    def pack(tensor):
        saved.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = function(states, head, index, 4, return_entropy = True)
    assert max(saved) <= vocab * hidden, saved
    assert sum(saved) <= vocab * hidden + rows * hidden + 4 * rows, saved
    sum(x.sum() for x in out).backward()

    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        rr.chunked_hidden_states_selective_log_softmax(states, head, index, 4, fused_linear = False)
    assert sum(saved) >= rows * vocab


def test_registered_and_grpo_routing_is_opt_in():
    assert rr.RL_REPLACEMENTS["grpo_selective_log_softmax_recompute"] is rr.chunked_hidden_states_selective_log_softmax_recompute
    source = inspect.getsource(rr.grpo_accumulated_loss)
    assert 'os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1"' in source


def test_compiled_kernels_match_eager():
    eager = _standalone()
    compiled = _standalone(compiled = True)
    head, states, head_bias, index, weights, entropy_weights = _inputs(rows = 10, bias = True, seed = 5)
    leaves = [states, head, head_bias]
    results = []
    for function in (eager, compiled):
        out = function(
            states, head, index, 3, 0.5, 0.0, 4.0, 0.9, lm_head_bias = head_bias, return_entropy = True,
            fused_linear = False,
        )
        results.append(list(out) + list(_grads(out, weights, entropy_weights, leaves)))
    # Engaged, not silently fallen back to eager.
    assert compiled._use_compiled
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "half-precision autocast on CUDA")
@pytest.mark.parametrize("rows, head_dtype", [(49, torch.float32), (300, torch.bfloat16), (17, torch.float16)])
@pytest.mark.parametrize("fused_linear", [True, False])
def test_cuda_autocast_matches_plain_function(rows, head_dtype, fused_linear):
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation = False) else torch.float16
    if head_dtype == torch.bfloat16 and amp != torch.bfloat16:
        pytest.skip("no bfloat16")
    gen = torch.Generator().manual_seed(rows)
    vocab, hidden = 3000, 64
    head = (torch.randn(vocab, hidden, generator = gen) * 0.05).to(head_dtype).cuda().requires_grad_(True)
    states = torch.randn(1, rows, hidden, generator = gen).to(amp).cuda().requires_grad_(True)
    index = torch.randint(0, vocab, (1, rows), generator = gen).cuda()
    weights = torch.randn(1, rows, generator = gen).cuda()
    results = []
    for function in (rr.chunked_hidden_states_selective_log_softmax, rr.chunked_hidden_states_selective_log_softmax_recompute):
        with torch.autocast("cuda", dtype = amp):
            if function is rr.chunked_hidden_states_selective_log_softmax_recompute:
                out = function(states, head, index, 16, 0.0, 0.0, 30.0, 0.8, fused_linear = fused_linear)
            else:
                out = function(states, head, index, 16, 0.0, 0.0, 30.0, 0.8)
        grads = torch.autograd.grad((out * weights).sum(), [states, head])
        results.append([out.float()] + [g.float() for g in grads])
    if not fused_linear:
        assert rr.chunked_hidden_states_selective_log_softmax_recompute._use_compiled
    for a, b in zip(*results):
        assert a.shape == b.shape
        assert float((a - b).abs().max() / b.abs().max()) < 1e-2


def test_cached_function_survives_a_missing_compiled_flag():
    # A monkeypatch of _use_compiled on the module function, undone after the autograd function
    # was cached, deletes the flag; later calls must fall back to the default instead of raising.
    function = _standalone()
    head, states, _, index, weights, _ = _inputs(rows = 6, bias = False, seed = 3)
    function(states, head, index, 2, 0.0, 0.0, 0.0, 1.0, fused_linear = False)
    assert getattr(function, "_autograd_function", None) is not None
    del function._use_compiled
    out = function(states, head, index, 2, 0.0, 0.0, 0.0, 1.0, fused_linear = False)
    (out * weights).sum().backward()
    assert out.isfinite().all()


def _fp16_head_case(function, head_dtype = torch.float16):
    # Three chunks of two rows; each chunk's head gradient (2 x 1.5 x 32768 = 98304) and the running
    # sum leave the fp16 range before the last chunk's opposite sign cancels half of it.
    hidden = torch.full((1, 6, 1), 1.5, dtype = torch.float16)
    head = torch.zeros((2, 1), dtype = head_dtype, requires_grad = True)
    index = torch.zeros((1, 6), dtype = torch.long)
    upstream = torch.tensor([[32768.0, 32768.0, 32768.0, 32768.0, -32768.0, -32768.0]])
    out = function(hidden, head, index, 3)
    (grad,) = torch.autograd.grad((out.float() * upstream).sum(), [head])
    head64 = head.detach().double().requires_grad_(True)
    logps = torch.log_softmax(hidden.double() @ head64.t(), -1)[..., 0]
    (want,) = torch.autograd.grad((logps * upstream.double()).sum(), [head64])
    return grad, want


@pytest.mark.parametrize("compiled", [False, True])
def test_fp16_head_grad_accumulates_in_float32(compiled):
    for function in (_standalone(compiled = compiled), rr.chunked_hidden_states_selective_log_softmax_recompute):
        grad, want = _fp16_head_case(function)
        assert grad.dtype == torch.float16
        assert grad.isfinite().all(), grad
        assert grad.double().flatten().tolist() == want.flatten().tolist() == [49152.0, -49152.0]


def test_bf16_hidden_grad_with_fp16_head_does_not_overflow():
    # Row hidden gradient sum_v grad_z[v] * W[v] is about -72000: fine in bf16, inf as an fp16 product.
    function = _standalone()
    vocab, hidden_size, rows = 9, 64, 4
    head = torch.full((vocab, hidden_size), 8000.0)
    head[0] = 0.0
    head = head.half().requires_grad_(True)
    gen = torch.Generator().manual_seed(0)
    states = (torch.randn(1, rows, hidden_size, generator = gen) * 1e-6).to(torch.bfloat16).requires_grad_(True)
    index = torch.zeros((1, rows), dtype = torch.long)
    out = function(states, head, index, 2)
    grad_states, grad_head = torch.autograd.grad((out * 10.0).sum(), [states, head])
    states64 = states.detach().double().requires_grad_(True)
    logps = torch.log_softmax(states64 @ head.detach().double().t(), -1)[..., 0]
    (want,) = torch.autograd.grad((logps * 10.0).sum(), [states64])
    assert grad_head.isfinite().all()
    assert grad_states.isfinite().all(), grad_states
    # Within bf16 / fp16 rounding of the float64 oracle.
    assert float((grad_states.double() - want).abs().max()) <= 2 ** -7 * float(want.abs().max())


def test_cold_outer_fullgraph_compile():
    # A fresh process, so nothing has called the function yet: it must build its autograd function
    # at import, leaving a cold outer torch.compile(fullgraph = True) only the cached function to trace.
    import subprocess, sys, textwrap
    script = textwrap.dedent("""
        import torch
        import unsloth_zoo.rl_replacements as rr
        function = rr.chunked_hidden_states_selective_log_softmax_recompute
        gen = torch.Generator().manual_seed(9)
        states = torch.randn(2, 8, 6, generator = gen, requires_grad = True)
        head = torch.randn(37, 6, generator = gen, requires_grad = True)
        index = torch.randint(0, 37, (2, 8), generator = gen)
        def outer(states, head, index):
            return function(states, head, index, 2, return_entropy = True)
        logps, entropy = torch.compile(outer, fullgraph = True, dynamic = True)(states, head, index)
        (logps.sum() + entropy.sum()).backward()
        assert function._use_compiled
        want = outer(states, head, index)
        torch.testing.assert_close(logps, want[0])
        torch.testing.assert_close(entropy, want[1])
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(outer)(states, head, index)
        assert explanation.graph_break_count == 0, explanation.break_reasons
        print("COLD_COMPILE_OK", explanation.graph_count)
    """)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH = root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if not torch.cuda.is_available():
        env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    result = subprocess.run([sys.executable, "-c", script], capture_output = True, text = True, timeout = 900, env = env)
    assert "COLD_COMPILE_OK" in result.stdout, (result.stdout[-2000:], result.stderr[-4000:])


@pytest.mark.parametrize("compiled", [False, True])
def test_fp32_head_under_fp16_autocast_does_not_nan(compiled):
    # fp32 trainable head, fp16 autocast, loss scale 65536, hidden 4, two 512 row chunks with opposite
    # upstream signs: each fp16 chunk product overflowed to +-inf before the float32 sum, leaving NaN.
    function = _standalone(compiled = compiled)
    rows, hidden_size, vocab = 1024, 8, 64
    sign = torch.ones(rows)
    sign[rows // 2:] = -1
    states = torch.full((1, rows, hidden_size), 4.0, dtype = torch.float16)
    head = torch.zeros((vocab, hidden_size), requires_grad = True)
    index = torch.zeros((1, rows), dtype = torch.long)
    with torch.autocast("cpu", dtype = torch.float16):
        out = function(states, head, index, 2)
    (grad,) = torch.autograd.grad((out.float() * 65536.0 * sign).sum(), [head])
    assert grad.dtype == torch.float32
    assert torch.equal(grad, torch.zeros_like(grad)), grad
