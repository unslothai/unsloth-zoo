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


"""CPU logits must never be written by fp16 torch.mm(out=) into a strided buffer slice: avx512_fp16 (torch 2.10) reads
stale data; ISA-dependent, so tests pin the call shape."""

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl


def _strided_out_calls(monkeypatch):
    calls = []
    real = torch.mm

    def mm(*args, **kwargs):
        out = kwargs.get("out")
        if out is not None and not out.is_cuda and not out.is_contiguous():
            calls.append(tuple(out.shape))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", mm)
    return calls


@pytest.mark.parametrize("vocab", [128, 129, 135])
def test_cpu_logits_are_not_written_through_a_strided_mm_out(monkeypatch, vocab):
    calls = _strided_out_calls(monkeypatch)
    gen = torch.Generator().manual_seed(0)
    h = (torch.randn(4, 64, generator = gen) * 1e-1).half()
    w = torch.randn(vocab, 64, generator = gen).half()
    buf = fl._Buffers(4, vocab, h.device, False, torch.float16)
    buf.buf.fill_(float("nan"))
    out = fl._logits_into(h, w, buf.logits(4), False)
    assert not calls, f"torch.mm(out=) into a strided CPU slice: {calls}"
    want = (h.float() @ w.float().t()).half()
    assert out.isfinite().all()
    torch.testing.assert_close(out, want, rtol = 2e-3, atol = 2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_logits_write_a_dense_destination_through_mm_out(monkeypatch, dtype):
    dense, fresh = [], []
    real = torch.mm

    def mm(*args, **kwargs):
        out = kwargs.get("out")
        (fresh if out is None else dense).append(None if out is None else out.data_ptr())
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", mm)
    gen = torch.Generator().manual_seed(0)
    h = (torch.randn(4, 64, generator = gen) * 1e-1).to(dtype)
    w = torch.randn(128, 64, generator = gen).to(dtype)
    buf = fl._Buffers(4, 128, h.device, False, dtype)
    buf.buf.fill_(float("nan"))
    out = fl._logits_into(h, w, buf.logits(4), False)
    assert dense == [buf.buf.data_ptr()] and not fresh
    want = (h.float() @ w.float().t()).to(dtype)
    torch.testing.assert_close(out, want, rtol = 2e-2, atol = 2e-2)

    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU], profile_memory = True) as prof:
        fl._logits_into(h, w, buf.logits(4), False)
    assert all(e.self_cpu_memory_usage == 0 for e in prof.key_averages() if e.key == "aten::mm")
