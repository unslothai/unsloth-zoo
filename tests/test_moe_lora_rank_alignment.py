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

"""A LoRA rank that is not a multiple of 8 (4, 6, ...) must still reach torch._grouped_mm.

torch._grouped_mm needs 16-byte aligned strides. A bf16 rank of 4 gives the (E, in, R)
factor and the (T, R) intermediate an 8-byte row, so eager fell back to the per-expert
host-synced loop on every LoRA matmul and torch.compile aborted the trace outright.
"""
import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


def _loop_reference(inputs, weight, offsets):
    out, start = [], 0
    for idx, end in enumerate(offsets.tolist()):
        out.append(inputs[start:end] @ weight[idx])
        start = end
    return torch.cat(out, dim=0)


def _strict_grouped_mm(inputs, weight, offs=None):
    """CPU stand-in that enforces the kernel's 16-byte stride rule on both operands."""
    for tensor in (inputs, weight):
        for stride in tensor.stride():
            if stride != 1 and stride * tensor.element_size() % 16 != 0:
                raise RuntimeError("strides should be multiple of 16 bytes")
    return _loop_reference(inputs, weight, offs)


@pytest.mark.parametrize("rank", [1, 4, 6, 8, 12, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pad_lora_rank_is_exact_and_aligned(rank, dtype):
    E, H, O = 3, 16, 24
    first = torch.randn(E, H, rank, dtype=dtype, requires_grad=True)
    second = torch.randn(E, rank, O, dtype=dtype, requires_grad=True)
    pf, ps = M._pad_lora_rank_for_grouped_mm(first, second)
    assert pf.shape[-1] == ps.shape[1] and pf.shape[-1] % 8 == 0
    assert pf.shape[-1] - rank < 8
    if rank % 8 == 0:
        assert pf is first and ps is second
    x = torch.randn(7, H, dtype=dtype)
    torch.testing.assert_close(
        torch.einsum("th,ehr,ero->eto", x, pf, ps),
        torch.einsum("th,ehr,ero->eto", x, first, second),
    )
    (torch.einsum("th,ehr,ero->eto", x, pf, ps).float().sum()).backward()
    assert first.grad.shape == first.shape and second.grad.shape == second.shape


@pytest.mark.parametrize("rank", [4, 6])
def test_apply_lora_grouped_mm_never_hits_the_alignment_fallback(monkeypatch, rank):
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda: True)
    monkeypatch.setattr(torch, "_grouped_mm", _strict_grouped_mm, raising=False)
    calls = []
    monkeypatch.setattr(M, "_manual_grouped_mm", lambda *a: calls.append(a) or _loop_reference(*a))

    E, H, O = 4, 32, 40
    x = torch.randn(10, H, dtype=torch.bfloat16)
    B = torch.randn(E, H, rank, dtype=torch.bfloat16, requires_grad=True)
    A = torch.randn(E, rank, O, dtype=torch.bfloat16, requires_grad=True)
    offsets = torch.tensor([2, 2, 7, 10], dtype=torch.int32)

    out = M._apply_lora_grouped_mm(x, B, A, offsets, 0.5)
    assert not calls, "a misaligned rank fell back to the per-expert loop"
    expected = _loop_reference(_loop_reference(x, B, offsets), A, offsets) * 0.5
    torch.testing.assert_close(out, expected)
    out.float().sum().backward()
    assert B.grad.shape == B.shape and A.grad.shape == A.shape


@pytest.mark.skipif(
    not torch.cuda.is_available() or not M._check_torch_grouped_mm_supported(),
    reason="needs a GPU with torch._grouped_mm",
)
@pytest.mark.parametrize("rank", [4, 6])
def test_rank4_lora_grouped_mm_compiles_fullgraph(rank):
    torch._dynamo.reset()
    E, T, H = 8, 64, 128
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(E, H, rank, device="cuda", dtype=torch.bfloat16) * 0.1
    A = torch.randn(E, rank, H, device="cuda", dtype=torch.bfloat16) * 0.1
    offsets = torch.arange(T // E, T + 1, T // E, device="cuda", dtype=torch.int32)
    fn = lambda *a: M._apply_lora_grouped_mm(*a, 1.0)
    eager = fn(x, B, A, offsets)
    compiled = torch.compile(fn, fullgraph=True)(x, B, A, offsets)
    torch.testing.assert_close(compiled, eager)
    reference = _loop_reference(_loop_reference(x, B, offsets), A, offsets)
    torch.testing.assert_close(eager, reference, atol=2e-2, rtol=2e-2)
