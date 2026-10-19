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
        rr.chunked_hidden_states_selective_log_softmax(states, head, index, 4)
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
        # The torch chunk loop behind UNSLOTH_FUSED_LINEAR=0, whose compiled bodies this checks.
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
                # The fused linear op by default; fused_linear=False is the compiled torch chunk loop.
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
