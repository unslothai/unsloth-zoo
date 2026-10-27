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

"""Overlapping threads must keep the process-wide reduced-precision switch off and restore it after."""

import threading

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl

FLAGS = torch.backends.cuda.matmul


def _overlapping(enter, observe, dtype):
    """A enters, B enters, A leaves, B reads, B leaves."""
    name = "allow_bf16_reduced_precision_reduction" if dtype == torch.bfloat16 else "allow_fp16_reduced_precision_reduction"
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen, errors = {}, []

    def run(tag):
        try:
            if tag == "A":
                with enter(dtype):
                    a_in.set()
                    assert b_in.wait(15)
                    seen["A"] = observe(name)
                a_out.set()
            else:
                assert a_in.wait(15)
                with enter(dtype):
                    b_in.set()
                    assert a_out.wait(15)
                    seen["B"] = observe(name)
        except Exception as e:
            errors.append(repr(e))

    old = getattr(FLAGS, name)
    try:
        setattr(FLAGS, name, True)
        threads = [threading.Thread(target = run, args = (t,)) for t in ("A", "B")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        return seen, getattr(FLAGS, name), errors
    finally:
        setattr(FLAGS, name, old)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_overlapping_callers_keep_the_switch_off_and_restore_it(dtype):
    seen, after, errors = _overlapping(fl._full_precision_reduction, lambda name: getattr(FLAGS, name), dtype)
    assert not errors, errors
    assert seen == {"A": False, "B": False}
    assert after is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_into_overlapping_threads_cuda(dtype, monkeypatch):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16 GEMM on this card")
    name = "allow_bf16_reduced_precision_reduction" if dtype == torch.bfloat16 else "allow_fp16_reduced_precision_reduction"
    a = torch.randn(64, 32, device = "cuda", dtype = dtype)
    b = torch.randn(32, 64, device = "cuda", dtype = dtype)
    outs = [torch.empty(64, 64, device = "cuda", dtype = dtype) for _ in range(2)]
    torch.mm(a, b)
    torch.cuda.synchronize()
    real_mm = torch.mm
    a_in, b_in, a_done = threading.Event(), threading.Event(), threading.Event()
    seen, errors = {}, []

    def held_mm(*args, **kwargs):
        tag = threading.current_thread().name
        if tag == "A":
            a_in.set()
            assert b_in.wait(15)
        else:
            b_in.set()
            assert a_done.wait(15)
        seen[tag] = getattr(FLAGS, name)
        return real_mm(*args, **kwargs)

    def work(i):
        try:
            fl._mm_into(outs[i], a, b)
        except Exception as e:
            errors.append(repr(e))
        finally:
            if i == 0:
                a_done.set()

    old = getattr(FLAGS, name)
    try:
        setattr(FLAGS, name, True)
        monkeypatch.setattr(torch, "mm", held_mm)
        A = threading.Thread(target = work, args = (0,), name = "A")
        B = threading.Thread(target = work, args = (1,), name = "B")
        A.start()
        assert a_in.wait(15)
        B.start()
        A.join(30)
        B.join(30)
        monkeypatch.setattr(torch, "mm", real_mm)
        torch.cuda.synchronize()
        assert not errors, errors
        assert seen == {"A": False, "B": False}
        assert getattr(FLAGS, name) is True
        ref = (a.double() @ b.double()).to(dtype)
        for out in outs:
            torch.testing.assert_close(out.float(), ref.float(), rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3,
                                       atol = 1e-2 if dtype == torch.bfloat16 else 1e-3)
    finally:
        setattr(FLAGS, name, old)
