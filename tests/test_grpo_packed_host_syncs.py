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

"""GRPO sequence-packed logp path: the host-sized index build must select exactly what the boolean masks did,
with one device-to-host read per batch instead of one per mask."""

import inspect

import pytest
import torch

import unsloth_zoo.rl_replacements as rr
from unsloth_zoo.rl_replacements import (
    calculate_pad_tokens_in_prompt,
    grpo_host_lengths_to_device,
    grpo_nonzero_static,
    left_pack_padding,
)

PAD = 0


def _batch(g, rows, prompt, completion):
    ids = torch.randint(1, 50, (rows, prompt + completion), generator = g)
    for r in range(rows):
        lp = int(torch.randint(0, prompt, (1,), generator = g))
        ids[r, :lp] = PAD  # left pad in the prompt
        cl = int(torch.randint(0, completion + 1, (1,), generator = g))
        ids[r, prompt + cl :] = PAD  # right pad in the completion
        if r % 5 == 3:
            ids[r, prompt + cl // 2] = PAD  # a pad token inside the completion (pad == eos models)
    return ids


def _old(input_ids, logits_to_keep, left_pad, hidden):
    keep = input_ids != PAD
    lengths_cpu = keep.sum(dim = 1).tolist()
    flat_ids = input_ids[keep].unsqueeze(0)
    pos = (keep.cumsum(dim = 1) - 1)[keep].unsqueeze(0)
    nz = keep.nonzero(as_tuple = False)
    within = nz[1:, 0] == nz[:-1, 0]
    L = input_ids.shape[1]
    cstart = (L - logits_to_keep) - left_pad
    ctgt = (nz[1:, 1] >= cstart[nz[1:, 0]]) & within
    h = hidden[0, :-1, :][ctgt]
    tid = flat_ids[0, 1:][ctgt]
    tgt = (nz[1:, 0] * L + nz[1:, 1])[ctgt]
    return lengths_cpu, flat_ids, pos, nz, h, tid, tgt


def _new(input_ids, logits_to_keep, left_pad, hidden, completion_mask):
    # Mirrors the packed block of grpo_accumulated_loss.
    keep = input_ids != PAD
    lengths = keep.sum(dim = 1)
    L = input_ids.shape[1]
    csum = keep.cumsum(dim = 1)
    cstart = (L - logits_to_keep) - left_pad
    ctgt_2d = (keep & (csum > 1)) & (torch.arange(L) >= cstart.unsqueeze(1))
    meta = torch.cat([
        lengths.to(torch.long),
        (completion_mask.sum(dim = 1) > 0).sum().to(torch.long).reshape(1),
        ctgt_2d.sum().to(torch.long).reshape(1),
    ]).tolist()
    lengths_cpu, active, ntgt = meta[:-2], meta[-2], meta[-1]
    T = sum(n for n in lengths_cpu if n > 0)
    nz = grpo_nonzero_static(keep, T)
    flat_ids = input_ids[nz[:, 0], nz[:, 1]].unsqueeze(0)
    pos = (csum - 1)[nz[:, 0], nz[:, 1]].unsqueeze(0)
    within = nz[1:, 0] == nz[:-1, 0]
    ctgt = (nz[1:, 1] >= cstart[nz[1:, 0]]) & within
    idx = grpo_nonzero_static(ctgt, ntgt).squeeze(1)
    h = hidden[0, :-1, :][idx]
    tid = flat_ids[0, 1:][idx]
    tgt = (nz[1:, 0] * L + nz[1:, 1])[idx]
    return lengths_cpu, active, flat_ids, pos, nz, h, tid, tgt


def test_packed_index_build_matches_boolean_masks():
    g = torch.Generator().manual_seed(0)
    for trial in range(300):
        rows = int(torch.randint(1, 9, (1,), generator = g))
        prompt = int(torch.randint(2, 12, (1,), generator = g))
        completion = int(torch.randint(1, 12, (1,), generator = g))
        raw = _batch(g, rows, prompt, completion)
        left_pad = calculate_pad_tokens_in_prompt(raw, completion, PAD)
        input_ids = left_pack_padding(raw, PAD)
        completion_mask = (input_ids[:, -completion:] != PAD).long()
        if trial % 7 == 0:
            completion_mask.zero_()
        T = int((input_ids != PAD).sum())
        hidden = torch.randn(1, max(T, 1), 3, generator = g, requires_grad = True)
        o = _old(input_ids, completion, left_pad, hidden)
        n = _new(input_ids, completion, left_pad, hidden, completion_mask)
        assert o[0] == n[0]
        assert n[1] == int((completion_mask.sum(dim = 1) > 0).sum())
        for a, b in zip(o[1:], n[2:]):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if o[4].numel():
            # same gradient into the hidden states (index backward, unique rows)
            w = torch.randn(o[4].shape, generator = g)
            go, = torch.autograd.grad((o[4] * w).sum(), hidden)
            gn, = torch.autograd.grad((n[5] * w).sum(), hidden)
            assert torch.equal(go, gn)


@pytest.mark.parametrize("shape", [(0,), (7,), (5, 9), (3, 4, 2)])
def test_nonzero_static_matches_nonzero(shape):
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        m = torch.rand(shape, generator = g) > 0.5
        ref = m.nonzero(as_tuple = False)
        got = grpo_nonzero_static(m, int(m.sum()))
        assert got.dtype == ref.dtype and torch.equal(got, ref)


def test_nonzero_static_falls_back(monkeypatch):
    monkeypatch.delattr(torch, "nonzero_static", raising = False)
    m = torch.tensor([[True, False], [True, True]])
    assert torch.equal(grpo_nonzero_static(m, 3), m.nonzero())


def test_host_lengths_to_device_cpu():
    out = grpo_host_lengths_to_device([3, 1, 7], "cpu")
    assert out.dtype == torch.int32 and out.tolist() == [3, 1, 7]
    assert out._unsloth_lengths_tuple == (3, 1, 7)
    assert all(type(x) is int for x in out._unsloth_lengths_tuple)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
def test_packed_index_build_no_sync_cuda():
    dev = "cuda"
    g = torch.Generator().manual_seed(2)
    raw = _batch(g, 6, 8, 10).to(dev)
    left_pad = calculate_pad_tokens_in_prompt(raw, 10, PAD)
    input_ids = left_pack_padding(raw, PAD)
    keep = input_ids != PAD
    T = int(keep.sum())
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        nz = grpo_nonzero_static(keep, T)
        psl = grpo_host_lengths_to_device([5, 6], dev)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(nz, keep.nonzero())
    assert psl.tolist() == [5, 6]


def test_packed_block_has_no_boolean_mask_reads():
    # The packed block of grpo_accumulated_loss: one .tolist(), no mask-indexed reads or bare nonzero().
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.index("_pack_keep = input_ids != _pack_pad_id")
    end = src.index("_pack_in_packed = True")
    block = src[start:end]
    assert block.count(".tolist()") == 1
    for bad in ("[_pack_keep]", "[_pack_ctgt]", ".nonzero(", ".item()", "int((", "torch.tensor(_pack_nz_cpu"):
        assert bad not in block, bad
