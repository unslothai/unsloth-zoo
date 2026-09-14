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

"""Chunks whose labels are entirely ignored in the fused cross entropy.

`UnslothFusedLoss.forward` splits the sequence into chunks and accumulates each
one. A chunk can end up with every label set to `ignore_index`, which happens
constantly with a last-assistant-turn-only mask, and always for a sample that
was truncated before its assistant turn. Those chunks contribute nothing, so
skipping them is free speed. The cases that must hold:

  1. skipping does not move the answer - loss and every gradient stay equal to
     the unchunked reference, for a mask sparse enough that most chunks are
     empty
  2. `grad_inputs` is allocated with `torch.empty_like`, so every skipped chunk
     must still be written. If a skip forgets to zero its slice the caller gets
     whatever was in that memory
  3. a batch where EVERY label is ignored and `n_items` is not supplied used to
     divide by a zero divisor and hand back NaN, which poisons the optimizer
     state for the rest of the run. It must be a finite zero with zero grads
  4. the skip predicate has to use the configured `ignore_index`, not a
     hardcoded -100, or it silently stops skipping anything
  5. skipping must not disturb the compile probe: a fully ignored batch has no
     chunk to probe with, and the next real batch still has to work

CPU only, no model downloads. `target_gb` is passed explicitly so chunk sizing
never calls `torch.cuda.mem_get_info`.
"""
import inspect

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch")

try:
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss
    import unsloth_zoo.fused_losses.cross_entropy_loss as ce
except ImportError as e:  # zoo-only checkout without `unsloth` installed
    pytest.skip(f"unsloth_zoo import unavailable: {e}", allow_module_level=True)


BSZ, QLEN, HD, VOCAB = 2, 96, 16, 128
TINY_GB = 1e-5  # forces many chunks for this shape


def _inputs(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BSZ, QLEN, HD, generator=g, dtype=dtype)
    w = torch.randn(VOCAB, HD, generator=g, dtype=dtype) / 4
    return x, w


def _shift(labels, ignore=-100):
    out = torch.full_like(labels, ignore)
    out[..., :-1] = labels[..., 1:]
    return out


def _reference(x, w, labels, n_items, ignore=-100):
    """Unchunked: materialize every logit, one cross entropy, one backward."""
    xr = x.detach().clone().requires_grad_(True)
    shifted = _shift(labels, ignore).reshape(-1)
    logits = F.linear(xr, w).float().reshape(-1, VOCAB)
    if n_items is None:
        loss = F.cross_entropy(logits, shifted, ignore_index=ignore, reduction="mean")
    else:
        loss = F.cross_entropy(logits, shifted, ignore_index=ignore, reduction="sum")
        loss = loss / n_items.float()
    if torch.isfinite(loss):
        loss.backward()
    return loss.detach(), xr.grad


def _fused(x, w, labels, n_items, ignore=-100, torch_compile=False):
    xf = x.detach().clone().requires_grad_(True)
    kwargs = {} if ignore == -100 else {"ignore_index": ignore}
    loss = unsloth_fused_ce_loss(
        None, hidden_states=xf, lm_head_weight=w, lm_head_bias=None,
        labels=labels, n_items=n_items, scaling=None, target_gb=TINY_GB,
        torch_compile=torch_compile, overwrite=False, **kwargs,
    )
    if torch.isfinite(loss):
        loss.backward()
    return loss.detach(), xf.grad


def _suffix_labels(keep=8, ignore=-100, seed=1):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, VOCAB, (BSZ, QLEN), generator=g)
    labels[:, : QLEN - keep] = ignore
    return labels


def _n_empty_chunks(labels, ignore=-100):
    shifted = _shift(labels, ignore).reshape(-1)
    n = ce.get_chunk_size(BSZ, QLEN, VOCAB, target_gb=TINY_GB)
    return sum(1 for c in torch.chunk(shifted, n) if not bool((c != ignore).any()))


def test_sparse_mask_matches_unchunked_reference():
    """Most chunks fully ignored: the answer must not move."""
    x, w = _inputs()
    labels = _suffix_labels()
    assert _n_empty_chunks(labels) > 0, "test shape stopped producing empty chunks"

    n_items = torch.tensor(int((_shift(labels) != -100).sum()))
    ref_loss, ref_grad = _reference(x, w, labels, n_items)
    got_loss, got_grad = _fused(x, w, labels, n_items)

    torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_grad, ref_grad, rtol=1e-4, atol=1e-6)
    # A mask this sparse must still move the tokens it does keep, otherwise
    # "matches the reference" would pass on two all-zero tensors.
    assert torch.count_nonzero(got_grad) > 0


def test_skipped_chunks_are_zeroed_not_left_uninitialised():
    """grad_inputs comes from torch.empty_like; an unwritten slice leaks memory."""
    # Poison the allocator so a forgotten slice reads back as a huge value
    # rather than as an accidental zero.
    poison = [torch.full((BSZ * QLEN, HD), 1e30) for _ in range(8)]
    del poison

    x, w = _inputs(seed=2)
    labels = _suffix_labels(seed=3)
    n_items = torch.tensor(int((_shift(labels) != -100).sum()))
    _, grad = _fused(x, w, labels, n_items)

    assert torch.isfinite(grad).all()
    ignored = (_shift(labels) == -100)
    assert torch.count_nonzero(grad[ignored]) == 0, "ignored positions must have zero grad"


@pytest.mark.parametrize("n_items_given", [False, True])
def test_fully_ignored_batch_is_finite_zero(n_items_given):
    """Every label ignored: finite zero loss and zero grads, never NaN.

    With no `n_items` the divisor is the count of kept labels, which is zero
    here. `_unsloth_get_batch_samples` legitimately leaves `n_items` unset (see
    test_loss_normalization_contract), and a sample truncated before its
    assistant turn is fully ignored, so this pair really does co-occur.
    """
    x, w = _inputs(seed=4)
    labels = torch.full((BSZ, QLEN), -100, dtype=torch.long)
    n_items = torch.tensor(1) if n_items_given else None

    loss, grad = _fused(x, w, labels, n_items)

    assert torch.isfinite(loss), f"fully ignored batch produced {loss}"
    assert float(loss) == 0.0
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) == 0


def test_skip_uses_configured_ignore_index():
    """A non-default ignore_index must be honoured by the skip predicate."""
    ignore = -1
    x, w = _inputs(seed=5)
    labels = _suffix_labels(ignore=ignore, seed=6)
    assert _n_empty_chunks(labels, ignore) > 0

    n_items = torch.tensor(int((_shift(labels, ignore) != ignore).sum()))
    ref_loss, ref_grad = _reference(x, w, labels, n_items, ignore=ignore)
    got_loss, got_grad = _fused(x, w, labels, n_items, ignore=ignore)

    torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_grad, ref_grad, rtol=1e-4, atol=1e-6)

    # There ARE chunks to skip here, but a predicate written against a
    # hardcoded -100 finds none of them, so the skip silently stops firing for
    # any caller that configures a different sentinel. Answers stay correct
    # either way, which is exactly why this needs pinning: nothing else in the
    # suite would notice the optimisation had turned itself off.
    shifted = _shift(labels, ignore).reshape(-1)
    n = ce.get_chunk_size(BSZ, QLEN, VOCAB, target_gb=TINY_GB)
    by_hardcoded = sum(1 for c in torch.chunk(shifted, n) if not bool((c != -100).any()))
    assert by_hardcoded == 0, "sanity: -100 is not the sentinel in this batch"
    assert _n_empty_chunks(labels, ignore) > by_hardcoded

    src = inspect.getsource(ce.UnslothFusedLoss.forward)
    assert "!= -100" not in src, (
        "the chunk-skip predicate must use the configured ignore_index, "
        "not a hardcoded -100"
    )


def test_fully_ignored_batch_does_not_strand_the_compile_probe():
    """An all-ignored batch has no chunk to probe with; the next one must work."""
    previous = ce._FUSED_CE_COMPILE_SUPPORTED
    try:
        ce._FUSED_CE_COMPILE_SUPPORTED = None
        x, w = _inputs(seed=7)

        empty = torch.full((BSZ, QLEN), -100, dtype=torch.long)
        loss, _ = _fused(x, w, empty, torch.tensor(1))
        assert float(loss) == 0.0

        labels = _suffix_labels(seed=8)
        n_items = torch.tensor(int((_shift(labels) != -100).sum()))
        ref_loss, ref_grad = _reference(x, w, labels, n_items)
        got_loss, got_grad = _fused(x, w, labels, n_items)
        torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got_grad, ref_grad, rtol=1e-4, atol=1e-6)
    finally:
        ce._FUSED_CE_COMPILE_SUPPORTED = previous


def test_dense_mask_is_unchanged_by_the_skip():
    """No chunk is empty: the skip must be a no-op on the common path."""
    x, w = _inputs(seed=9)
    g = torch.Generator().manual_seed(10)
    labels = torch.randint(0, VOCAB, (BSZ, QLEN), generator=g)
    assert _n_empty_chunks(labels) == 0

    n_items = torch.tensor(int((_shift(labels) != -100).sum()))
    ref_loss, ref_grad = _reference(x, w, labels, n_items)
    got_loss, got_grad = _fused(x, w, labels, n_items)
    torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_grad, ref_grad, rtol=1e-4, atol=1e-6)
