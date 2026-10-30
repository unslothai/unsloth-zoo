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

"""128-row chunk alignment: no chunk above torch.chunk's largest, losses and gradients up to sum order."""
import math
import os

import pytest
import torch

ALIGN = 128


def _chunk_row_bounds():
    try:
        from unsloth_zoo.temporary_patches.common import chunk_row_bounds
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    return chunk_row_bounds


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
    # Positional order of UnslothFusedLoss.apply: loss_function, hidden_states, lm_head_weight, lm_head_bias, labels,
    # mask, n_items, scaling, shift_labels, target_gb, torch_compile, overwrite, extra_kwargs.
    loss = ce.UnslothFusedLoss.apply(
        loss_function, hidden, weight, b, labels, None, None, None, True, None, False, False,
        {"n_chunks": n_chunks},
    )
    loss.backward()
    grads = [hidden.grad, weight.grad] + ([b.grad] if bias else [])
    return loss.detach(), grads, rows


ODD_SHAPES = [(2, 4097, 12), (3, 1001, 4), (1, 5000, 8), (2, 777, 3)]


@pytest.mark.parametrize("bsz, qlen, n_chunks", ODD_SHAPES)
def test_fused_ce_chunks_are_aligned(monkeypatch, bsz, qlen, n_chunks):
    ce = _ce_module()
    monkeypatch.setattr(ce, "CHUNK_ALIGN_ROWS", ALIGN, raising = False)
    n = bsz * qlen
    _, _, rows = _run_fused(ce, bsz, qlen, n_chunks, torch.float32)
    assert sum(rows) == n
    assert all(r % ALIGN == 0 for r in rows[:-1]), rows
    assert 0 < rows[-1]
    assert max(rows) <= math.ceil(n / n_chunks), rows


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
    chunk_row_bounds = _chunk_row_bounds()
    for chunks in (1, 2, 3, 4, 7, 12, 16, 32, 128):
        for n in list(range(chunks, 600)) + [8194, 4097, 12345, 16512, 131071]:
            assert chunk_row_bounds(n, chunks, align = 0) is None
            bounds = chunk_row_bounds(n, chunks, align = ALIGN)
            if bounds is None:
                continue
            sizes = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
            assert bounds[0] == 0 and bounds[-1] == n and min(sizes) > 0
            assert max(sizes) <= math.ceil(n / chunks), (n, chunks, sizes)
            assert all(s % ALIGN == 0 for s in sizes[:-1]), (n, chunks, sizes)
            assert sizes[-1] != 1, (n, chunks, sizes)


def test_chunk_row_bounds_keeps_budget_for_small_chunks():
    chunk_row_bounds = _chunk_row_bounds()
    # 129-row torch.chunk ranges: aligned ranges stay at 128 rows (one more range), never 256.
    bounds = chunk_row_bounds(16512, 128, align = ALIGN)
    sizes = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    assert max(sizes) == 128 and len(sizes) == 129
    # Ranges would shrink by over a quarter (255 -> 128): keep torch.chunk.
    assert chunk_row_bounds(255 * 8, 8, align = ALIGN) is None
    # A one-row tail (8193 = 8 x 1024 + 1) keeps torch.chunk.
    assert chunk_row_bounds(8193, 8, align = ALIGN) is None


@pytest.mark.parametrize("bsz, qlen, n_chunks", ODD_SHAPES[:2])
def test_default_is_off_and_keeps_torch_chunk(bsz, qlen, n_chunks):
    """Opt-in: without UNSLOTH_CE_CHUNK_ALIGN=1 the boundaries are torch.chunk's."""
    ce = _ce_module()
    if os.environ.get("UNSLOTH_CE_CHUNK_ALIGN", "0").strip() == "1":
        pytest.skip("UNSLOTH_CE_CHUNK_ALIGN=1 set in the environment")
    _, _, rows = _run_fused(ce, bsz, qlen, n_chunks, torch.float32)
    assert rows == [t.shape[0] for t in torch.chunk(torch.empty(bsz * qlen), n_chunks)]
