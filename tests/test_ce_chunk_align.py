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

"""128-row chunk alignment: same chunk count, losses and gradients up to sum order."""
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")

ROOT = Path(__file__).resolve().parents[1]
ALIGN = 128


def _ce_module():
    try:
        import unsloth_zoo.fused_losses.cross_entropy_loss as ce
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    return ce


def _run_fused(ce, bsz, qlen, n_chunks, dtype, hd = 8, vocab = 48, bias = False):
    """Fused CE loss and grads, plus the row count of every chunk it ran."""
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(bsz, qlen, hd, generator = g, dtype = dtype).requires_grad_(True)
    weight = (torch.randn(vocab, hd, generator = g, dtype = dtype) * 0.5).requires_grad_(True)
    b = (torch.randn(vocab, generator = g, dtype = dtype) * 0.1).requires_grad_(True) if bias else None
    labels = torch.randint(0, vocab, (bsz, qlen), generator = g)
    labels[0, :3] = -100
    rows = []
    def loss_function(hidden_states, *args, **kwargs):
        rows.append(int(hidden_states.shape[0]))
        return ce.compute_fused_ce_loss(hidden_states, *args, **kwargs)
    # Positional: loss_function, hidden_states, lm_head_weight, lm_head_bias, labels, mask,
    # n_items, scaling, shift_labels, target_gb, torch_compile, overwrite, extra_kwargs.
    loss = ce.UnslothFusedLoss.apply(
        loss_function, hidden, weight, b, labels, None, None, None, True, None, False, False,
        {"n_chunks": n_chunks},
    )
    loss.backward()
    grads = [hidden.grad, weight.grad] + ([b.grad] if bias else [])
    return loss.detach(), grads, rows


# bsz * qlen = 8194 in 12 chunks: torch.chunk gives 683 rows per chunk.
ODD_SHAPES = [(2, 4097, 12), (3, 1001, 4), (1, 5000, 8), (2, 777, 3)]


@pytest.mark.parametrize("bsz, qlen, n_chunks", ODD_SHAPES)
def test_fused_ce_chunks_are_aligned(monkeypatch, bsz, qlen, n_chunks):
    ce = _ce_module()
    monkeypatch.setattr(ce, "CHUNK_ALIGN_ROWS", ALIGN, raising = False)
    n = bsz * qlen
    _, _, rows = _run_fused(ce, bsz, qlen, n_chunks, torch.float32)
    old_count = len(torch.chunk(torch.empty(n), n_chunks))
    assert len(rows) == old_count == n_chunks
    assert sum(rows) == n
    assert all(r % ALIGN == 0 for r in rows[:-1]), rows
    assert 0 < rows[-1]
    # Memory: no chunk more than ALIGN - 1 rows over torch.chunk's largest.
    assert max(rows) <= math.ceil(n / n_chunks) + ALIGN - 1, rows


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-12), (torch.float32, 2e-6)])
@pytest.mark.parametrize("bsz, qlen, n_chunks", ODD_SHAPES)
@pytest.mark.parametrize("bias", [False, True])
def test_fused_ce_matches_previous_chunking(monkeypatch, dtype, tol, bsz, qlen, n_chunks, bias):
    ce = _ce_module()
    monkeypatch.setattr(ce, "CHUNK_ALIGN_ROWS", ALIGN, raising = False)
    loss, grads, _ = _run_fused(ce, bsz, qlen, n_chunks, dtype, bias = bias)
    monkeypatch.setattr(ce, "CHUNK_ALIGN_ROWS", 0, raising = False)
    old_loss, old_grads, old_rows = _run_fused(ce, bsz, qlen, n_chunks, dtype, bias = bias)
    assert old_rows == [t.shape[0] for t in torch.chunk(torch.empty(bsz * qlen), n_chunks)]
    torch.testing.assert_close(loss, old_loss, rtol = 2e-6, atol = 2e-6)
    # The input gradient of a row depends only on that row.
    torch.testing.assert_close(grads[0], old_grads[0], rtol = tol, atol = tol)
    for a, b in zip(grads[1:], old_grads[1:]):
        torch.testing.assert_close(a, b, rtol = tol, atol = tol)


@pytest.mark.parametrize("bsz, qlen, n_chunks", [(1, 50, 4), (2, 9, 4), (1, 1000, 12), (1, 7, 3)])
def test_fused_ce_small_inputs_keep_torch_chunk(monkeypatch, bsz, qlen, n_chunks):
    """Under 128 rows per chunk the boundaries are torch.chunk's, unchanged."""
    ce = _ce_module()
    monkeypatch.setattr(ce, "CHUNK_ALIGN_ROWS", ALIGN, raising = False)
    _, _, rows = _run_fused(ce, bsz, qlen, n_chunks, torch.float32)
    assert rows == [t.shape[0] for t in torch.chunk(torch.empty(bsz * qlen), n_chunks)]


def test_chunk_row_bounds_invariants():
    from unsloth_zoo.temporary_patches.common import chunk_row_bounds
    for chunks in (1, 2, 3, 4, 7, 12, 16, 32):
        for n in list(range(chunks, 600)) + [8194, 4097, 12345, 131071]:
            bounds = chunk_row_bounds(n, chunks, align = ALIGN)
            sizes = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
            assert len(sizes) == chunks and bounds[0] == 0 and bounds[-1] == n
            assert max(sizes) <= math.ceil(n / chunks) + ALIGN - 1, (n, chunks, sizes)
            if n >= ALIGN * chunks:
                assert min(sizes) >= ALIGN, (n, chunks, sizes)
                assert all(s % ALIGN == 0 for s in sizes[:-1]), (n, chunks, sizes)
            else:
                assert bounds == [(i * n) // chunks for i in range(chunks + 1)]
            assert chunk_row_bounds(n, chunks, align = 0) == [(i * n) // chunks for i in range(chunks + 1)]


CHILD = textwrap.dedent(
    """
    import json, sys
    import torch, torch._dynamo
    from torch._dynamo.utils import counters
    _compile = torch.compile
    def _dynamo_only(fn = None, **kw):
        kw.pop("options", None)
        kw.pop("mode", None)
        kw["backend"] = "eager"
        return _compile(**kw) if fn is None else _compile(fn, **kw)
    torch.compile = _dynamo_only
    import unsloth_zoo.rl_replacements as R

    torch.manual_seed(0)
    hidden, vocab, chunks = 8, 40, 4
    lm_head = (torch.randn(vocab, hidden, dtype = torch.float64) * 0.3).requires_grad_(True)
    worst = 0.0
    counters.clear()
    for T in (777, 1001, 1301):
        states = torch.randn(1, T, hidden, dtype = torch.float64, requires_grad = True)
        index = torch.randint(0, vocab, (1, T))
        weights = torch.randn(1, T, dtype = torch.float64)
        out = R.chunked_hidden_states_selective_log_softmax(states, lm_head, index, chunks)
        (out * weights).sum().backward()
        got = [out.detach(), states.grad.clone(), lm_head.grad.clone()]
        states.grad = None
        lm_head.grad = None
        logits = states.reshape(-1, hidden) @ lm_head.t()
        ref = (logits.gather(-1, index.reshape(-1, 1)).squeeze(-1) - logits.logsumexp(-1)).reshape(1, T)
        (ref * weights).sum().backward()
        want = [ref.detach(), states.grad, lm_head.grad]
        lm_head.grad = None
        worst = max(worst, max(float((a - b).abs().max()) for a, b in zip(got, want)))
    print(json.dumps({"graphs": counters["stats"]["unique_graphs"], "worst": worst}))
    """
)


def test_grpo_logprobs_aligned_without_recompiles():
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", UNSLOTH_CE_CHUNK_ALIGN = "1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    run = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output = True, text = True, timeout = 600, env = env,
    )
    if run.returncode != 0 and "unsloth_zoo import unavailable" in run.stderr:
        pytest.skip(run.stderr[-500:])
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["graphs"] == 1, out
    # The helper upcasts logits to float32, so float32 rounding even from float64 inputs.
    assert out["worst"] < 1e-5, out


@pytest.mark.parametrize("bsz, qlen, n_chunks", ODD_SHAPES[:2])
def test_default_is_off_and_keeps_torch_chunk(bsz, qlen, n_chunks):
    """Opt-in: without UNSLOTH_CE_CHUNK_ALIGN=1 the boundaries are torch.chunk's."""
    ce = _ce_module()
    if os.environ.get("UNSLOTH_CE_CHUNK_ALIGN", "0").strip() == "1":
        pytest.skip("UNSLOTH_CE_CHUNK_ALIGN=1 set in the environment")
    _, _, rows = _run_fused(ce, bsz, qlen, n_chunks, torch.float32)
    assert rows == [t.shape[0] for t in torch.chunk(torch.empty(bsz * qlen), n_chunks)]


def test_grpo_helper_source_is_self_contained():
    """unsloth pastes inspect.getsource of this helper into the generated GRPO trainer file,
    whose header imports only torch, os, typing and friends: it must not call zoo helpers."""
    import inspect
    from typing import Optional
    try:
        import unsloth_zoo.rl_replacements as R
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    fn = R.RL_REPLACEMENTS["grpo_selective_log_softmax"]
    src = textwrap.dedent(inspect.getsource(getattr(fn, "__wrapped__", fn)))
    src = "\n".join(l for l in src.splitlines() if not l.startswith("@"))
    old = os.environ.get("UNSLOTH_CE_CHUNK_ALIGN")
    os.environ["UNSLOTH_CE_CHUNK_ALIGN"] = "1"
    try:
        namespace = {"torch": torch, "os": os, "Optional": Optional, "math": math}
        exec(src, namespace)
    finally:
        if old is None: os.environ.pop("UNSLOTH_CE_CHUNK_ALIGN", None)
        else: os.environ["UNSLOTH_CE_CHUNK_ALIGN"] = old
    helper = namespace[fn.__name__ if hasattr(fn, "__name__") else "chunked_hidden_states_selective_log_softmax"]
    torch.manual_seed(0)
    states = torch.randn(1, 1001, 8, dtype = torch.float64)
    lm_head = torch.randn(40, 8, dtype = torch.float64)
    index = torch.randint(0, 40, (1, 1001))
    out = helper(states, lm_head, index, 4)
    logits = states.reshape(-1, 8) @ lm_head.t()
    ref = (logits.gather(-1, index.reshape(-1, 1)).squeeze(-1) - logits.logsumexp(-1)).reshape(1, 1001)
    torch.testing.assert_close(out, ref.to(out.dtype), rtol = 1e-5, atol = 1e-5)
