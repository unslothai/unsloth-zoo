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

"""Packed Gemma-4 rows: band probe rejects mask-free packed rows; float32 path runs per-document SDPA."""

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import gemma4_banded_attention as gba
from unsloth_zoo.temporary_patches.gemma4_banded_attention import (
    _mask_is_block_causal,
    _mask_is_plain_band,
    _packed_document_sdpa,
)


def _block_causal(lengths, w=None, as_float=False):
    T = sum(lengths)
    seg = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    i = torch.arange(T)
    m = (seg[:, None] == seg[None, :]) & (i[None, :] <= i[:, None])
    if w:
        m &= i[None, :] > i[:, None] - w
    m = m[None, None]
    if as_float:
        m = torch.zeros(m.shape).masked_fill(~m, torch.finfo(torch.float32).min)
    return m


def test_mask_free_packed_row_is_not_a_plain_band():
    assert _mask_is_plain_band(None, 8, 4)
    assert _mask_is_plain_band(None, 8, 4, packed_seq_lengths=torch.tensor([8]))
    assert not _mask_is_plain_band(None, 8, 4, packed_seq_lengths=torch.tensor([3, 5]))
    assert not _mask_is_plain_band(None, 8, 4, packed_seq_lengths=[3, 5])


class _Sliding:
    is_sliding = True
    is_causal = True
    training = False

    def __init__(self, w, ng):
        self.sliding_window = w
        self.num_key_value_groups = ng


# The router gates on a class name starting with "Gemma4".
Gemma4SlidingFake = type("Gemma4SlidingFake", (_Sliding,), {})


@pytest.mark.parametrize("lengths", [[300], [100, 200]])
def test_router_defers_a_mask_free_packed_row(monkeypatch, lengths):
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    monkeypatch.setattr(gf, "_HAS_FA2", False)
    monkeypatch.setattr(gf, "_flash_attn_func", None)
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1")
    gf._enabled.cache_clear()
    monkeypatch.delenv("UNSLOTH_BANDED_SDPA", raising=False)
    gf._force_banded.cache_clear()
    S, w, H, Hkv, d = 300, 64, 4, 2, 32
    q, k, v = torch.randn(1, H, S, d), torch.randn(1, Hkv, S, d), torch.randn(1, Hkv, S, d)
    deferred = []
    orig = gf._ORIG_SDPA[0]
    gf._ORIG_SDPA[0] = lambda *a, **kw: deferred.append(kw.get("packed_seq_lengths")) or ("orig", None)
    before = gf._BANDED_ENGAGED[0]
    try:
        out = gf._sdpa_maybe_flash_sliding(
            Gemma4SlidingFake(w, H // Hkv), q, k, v, None, dropout=0.0, scaling=d**-0.5,
            packed_seq_lengths=torch.tensor(lengths),
        )
    finally:
        gf._ORIG_SDPA[0] = orig
    if len(lengths) == 1:
        assert gf._BANDED_ENGAGED[0] == before + 1 and not deferred
    else:
        assert out == ("orig", None) and len(deferred) == 1
        assert gf._BANDED_ENGAGED[0] == before


@pytest.mark.parametrize("as_float", [False, True])
@pytest.mark.parametrize("w", [None, 4])
def test_block_causal_mask_detection(as_float, w):
    lengths = (3, 9, 5)
    assert _mask_is_block_causal(_block_causal(lengths, w, as_float), lengths, w)
    assert not _mask_is_block_causal(_block_causal(lengths, w, as_float), lengths, 3 if w else 4)
    assert not _mask_is_block_causal(_block_causal(lengths, w, as_float), (3, 14), w)
    leaky = _block_causal(lengths, w)
    leaky[0, 0, 5, 1] = True
    assert not _mask_is_block_causal(leaky, lengths, w)
    assert not _mask_is_block_causal(None, lengths, w)


def test_an_edited_mask_is_checked_again():
    lengths = (3, 5)
    m = _block_causal(lengths)
    assert _mask_is_block_causal(m, lengths, None)
    m[0, 0, 4, 0] = True
    assert not _mask_is_block_causal(m, lengths, None)


@pytest.mark.parametrize("w", [None, 4])
@pytest.mark.parametrize("Hkv", [4, 2, 1])
def test_per_document_sdpa_matches_the_dense_mask(w, Hkv):
    torch.manual_seed(0)
    lengths = [3, 17, 1, 9]
    T, H, d = sum(lengths), 4, 32
    q = torch.randn(1, H, T, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, Hkv, T, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, Hkv, T, d, dtype=torch.float64, requires_grad=True)
    mask = _block_causal(lengths, w)
    before = gba._PACKED_DOC_ENGAGED[0]
    out = _packed_document_sdpa(q, k, v, mask, torch.tensor(lengths), w, 1.0, H // Hkv)
    assert gba._PACKED_DOC_ENGAGED[0] == before + 1
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0, enable_gqa=Hkv != H)
    torch.testing.assert_close(out, ref)
    g = torch.randn_like(ref)
    for a, b in zip(torch.autograd.grad(out, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a, b)


def test_per_document_sdpa_declines_other_masks():
    lengths = [5, 7]
    q = torch.randn(1, 2, 12, 8)
    psl = torch.tensor(lengths)
    leaky = _block_causal(lengths)
    leaky[0, 0, 8, 2] = True
    assert _packed_document_sdpa(q, q, q, leaky, psl, None, 1.0, 1) is None
    assert _packed_document_sdpa(q, q, q, _block_causal(lengths), torch.tensor([5, 6]), None, 1.0, 1) is None
    assert _packed_document_sdpa(q.expand(2, -1, -1, -1), q.expand(2, -1, -1, -1), q.expand(2, -1, -1, -1),
                                 _block_causal(lengths), psl, None, 1.0, 1) is None
    assert _packed_document_sdpa(q, q, q, _block_causal(lengths), psl, 3, 1.0, 1) is None


def test_inference_tensors_are_checked_without_a_version():
    lengths = (4, 6)
    with torch.inference_mode():
        m = _block_causal(lengths)
        q = torch.randn(1, 2, 10, 8)
        assert _mask_is_block_causal(m, lengths, None)
        assert _packed_document_sdpa(q, q, q, m, torch.tensor(lengths), None, 1.0, 1) is not None


def _counting_sdpa(monkeypatch):
    calls = []
    sdpa = F.scaled_dot_product_attention

    def counted(q, k, v, **kwargs):
        calls.append(q.shape[0])
        return sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", counted)
    return calls


def _runs(lengths):
    return sum(1 for i, L in enumerate(lengths) if i == 0 or lengths[i - 1] != L)


_LENGTHS = [
    [5, 5, 5, 7, 7, 1, 9],  # runs of equal length, a 1-token document, a lone document
    [3, 3, 8, 8, 8, 1, 1],
    [1, 1, 1, 1],           # only 1-token documents
    [12],                   # a single document
    [4, 6, 4, 6],           # equal lengths that are not consecutive stay separate
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("w", [None, 4])
@pytest.mark.parametrize("Hkv", [4, 2])
@pytest.mark.parametrize("lengths", _LENGTHS)
def test_equal_length_runs_match_the_loop_bit_for_bit(monkeypatch, lengths, Hkv, w, dtype):
    torch.manual_seed(0)
    T, H, d = sum(lengths), 4, 16
    mask = _block_causal(lengths, w)
    psl = torch.tensor(lengths)
    leaves = (torch.randn(1, H, T, d, dtype=dtype), torch.randn(1, Hkv, T, d, dtype=dtype),
              torch.randn(1, Hkv, T, d, dtype=dtype))
    g = torch.randn(1, H, T, d, dtype=dtype)
    calls = _counting_sdpa(monkeypatch)
    results = {}
    for batched in (False, True):
        q, k, v = (x.clone().requires_grad_() for x in leaves)
        calls.clear()
        out = _packed_document_sdpa(q, k, v, mask, psl, w, 1.0, H // Hkv, batch_equal_lengths=batched)
        results[batched] = (out.detach(), *torch.autograd.grad(out, (q, k, v), g), len(calls))
    loop, batched = results[False], results[True]
    for a, b in zip(loop[:4], batched[:4]):
        assert torch.equal(a, b)
    assert loop[4] == len(lengths) and batched[4] == _runs(lengths)
    ref = F.scaled_dot_product_attention(*(x.repeat_interleave(H // x.shape[1], 1) for x in leaves),
                                         attn_mask=mask, scale=1.0)
    torch.testing.assert_close(batched[0], ref)


def test_equal_length_documents_share_one_call_and_the_kill_switch_restores_the_loop(monkeypatch):
    lengths = [6, 6, 6, 6, 2, 9, 9, 1]  # 8 documents, 4 runs, 2 of them batched
    q = torch.randn(1, 2, sum(lengths), 8)
    calls = _counting_sdpa(monkeypatch)
    mask, psl = _block_causal(lengths), torch.tensor(lengths)
    monkeypatch.delenv("UNSLOTH_GEMMA4_PACKED_BATCH", raising=False)
    batched = _packed_document_sdpa(q, q, q, mask, psl, None, 1.0, 1)
    assert sorted(calls) == [1, 1, 2, 4]
    calls.clear()
    monkeypatch.setenv("UNSLOTH_GEMMA4_PACKED_BATCH", "0")
    loop = _packed_document_sdpa(q, q, q, mask, psl, None, 1.0, 1)
    assert calls == [1] * len(lengths)
    assert torch.equal(batched, loop)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SDPA kernels")
@pytest.mark.parametrize("backend", ["EFFICIENT_ATTENTION", "FLASH_ATTENTION"])
def test_equal_length_runs_match_the_loop_on_cuda_float16(backend):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(0)
    lengths = [64, 64, 64, 32, 32, 1, 128]
    T, H, d = sum(lengths), 4, 64
    mask, psl = _block_causal(lengths).cuda(), torch.tensor(lengths)
    leaves = [torch.randn(1, H, T, d, device="cuda", dtype=torch.float16) for _ in range(3)]
    g = torch.randn_like(leaves[0])
    results = []
    with sdpa_kernel(getattr(SDPBackend, backend)):
        for batched in (False, True):
            q, k, v = (x.clone().requires_grad_() for x in leaves)
            out = _packed_document_sdpa(q, k, v, mask, psl, None, 1.0, 1, batch_equal_lengths=batched)
            results.append((out.detach(), *torch.autograd.grad(out, (q, k, v), g)))
    for a, b in zip(*results):
        assert torch.equal(a, b)
