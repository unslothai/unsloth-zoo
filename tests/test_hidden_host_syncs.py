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

"""Per-step paths that used to sync the host with the device: the device-side packed boundary mask must drop
exactly the slots the host filter drops, and an integer n_items divisor must be bitwise torch.tensor's."""

import warnings

import pytest
import torch

from unsloth_zoo import loss_utils


def _host_mask(lengths, token_count):
    return loss_utils._packed_boundary_targets(lengths, token_count)


def _device_style_mask(lengths, token_count, monkeypatch):
    # The CUDA branch is the compiled branch (clamp, no filter); run it on CPU through is_compiling.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    try:
        return loss_utils._packed_boundary_targets(lengths, token_count)
    finally:
        monkeypatch.undo()


def _batches(n = 400, seed = 0):
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        rows = int(torch.randint(1, 4, (1,), generator = g))
        width = int(torch.randint(2, 70, (1,), generator = g))
        k = int(torch.randint(1, 12, (1,), generator = g))
        lengths = torch.randint(-3 if i % 5 == 0 else 0, width + 3, (k,), generator = g)
        if i % 3 == 0:  # exact cover of the flattened row
            lengths = torch.full((k,), rows * width // k)
        token_count = torch.rand(rows, width - 1, generator = g) > 0.2
        if rows == 1 and i % 2:
            token_count = token_count[0]
        yield lengths.to([torch.int32, torch.int64][i % 2]), token_count


def test_device_boundary_mask_drops_the_same_slots_as_the_host_filter(monkeypatch):
    for lengths, token_count in _batches():
        host = _host_mask(lengths, token_count)
        dev = _device_style_mask(lengths, token_count, monkeypatch)
        host_count = (token_count & ~host).sum() if host is not None else token_count.sum()
        dev_count = (token_count & ~dev).sum() if dev is not None else token_count.sum()
        assert torch.equal(host_count, dev_count), (lengths.tolist(), token_count.shape)
        if host is not None and dev is not None:
            assert torch.equal(host, dev)
        elif dev is not None:
            assert not dev.any()


def test_count_batch_items_unchanged_with_packed_lengths(monkeypatch):
    for lengths, token_count in _batches(100, seed = 1):
        if token_count.ndim != 2:
            continue
        labels = torch.where(
            torch.cat([torch.ones(token_count.shape[0], 1, dtype = torch.bool), token_count], 1), 5, -100
        )
        host = loss_utils.count_batch_items(labels, packed_seq_lengths = lengths)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        dev = loss_utils.count_batch_items(labels, packed_seq_lengths = lengths)
        monkeypatch.undo()
        assert torch.equal(host[0], dev[0]) and host[1:] == dev[1:]


def test_int_divisor_fill_is_bitwise_torch_tensor():
    for v in (0, 1, 7, 2**24 + 1, 2**31 + 12345, 2**40 + 999, 3.5, True):
        a = torch.tensor(v, dtype = torch.float32)
        b = torch.full((), v, dtype = torch.float32)
        assert a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_cuda_packed_count_and_divisor_do_not_sync():
    lengths = torch.tensor([5, 0, 70, 7], device = "cuda")
    labels = torch.randint(0, 9, (1, 96), device = "cuda")
    torch.cuda.synchronize()
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            count, _, _ = loss_utils.count_batch_items(labels, packed_seq_lengths = lengths)
            torch.full((), 1234, dtype = torch.float32, device = "cuda")
        finally:
            torch.cuda.set_sync_debug_mode("default")
    n = sum("synchronizing CUDA operation" in str(x.message) for x in w)
    assert n == 0, f"{n} syncs"
    host = loss_utils.count_batch_items(labels.cpu(), packed_seq_lengths = lengths.cpu())[0]
    assert int(count) == int(host)
