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


@pytest.mark.parametrize("rank", [1, 2, 4, 6])
def test_pad_skipped_only_when_both_lora_gemms_take_triton(monkeypatch, rank):
    """The Triton grouped GEMM masks any K / N, so the pad there is two copies for nothing."""
    first = torch.randn(3, 16, rank, dtype=torch.bfloat16)
    second = torch.randn(3, rank, 24, dtype=torch.bfloat16)
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg
    monkeypatch.delenv("UNSLOTH_MOE_LORA_PAD_ALWAYS", raising=False)
    monkeypatch.setattr(M, "_triton_grouped_mm_wanted", lambda inputs, weight: True)
    monkeypatch.setattr(mg, "triton_grouped_available", lambda device: True)
    pf, ps = M._pad_lora_rank_for_grouped_mm(first, second, x)
    assert pf is first and ps is second
    # a Triton kernel that failed its self-check falls back to torch._grouped_mm: keep the pad
    monkeypatch.setattr(mg, "triton_grouped_available", lambda device: False)
    assert M._pad_lora_rank_for_grouped_mm(first, second, x)[0].shape[-1] == 8
    monkeypatch.setattr(mg, "triton_grouped_available", lambda device: True)
    # no inputs = the caller cannot tell the route: keep the pad
    assert M._pad_lora_rank_for_grouped_mm(first, second)[0].shape[-1] == 8
    monkeypatch.setenv("UNSLOTH_MOE_LORA_PAD_ALWAYS", "1")
    assert M._pad_lora_rank_for_grouped_mm(first, second, x)[0].shape[-1] == 8
    monkeypatch.delenv("UNSLOTH_MOE_LORA_PAD_ALWAYS")
    monkeypatch.setattr(M, "_triton_grouped_mm_wanted", lambda inputs, weight: False)
    assert M._pad_lora_rank_for_grouped_mm(first, second, x)[0].shape[-1] == 8


def test_pad_kept_above_the_lora_row_class(monkeypatch):
    """rank > 32 moves the GEMMs to a row-limited class that may take torch._grouped_mm."""
    monkeypatch.setattr(M, "_triton_grouped_mm_wanted", lambda inputs, weight: True)
    first = torch.randn(3, 64, 33, dtype=torch.bfloat16)
    second = torch.randn(3, 33, 64, dtype=torch.bfloat16)
    pf, _ = M._pad_lora_rank_for_grouped_mm(first, second, torch.randn(5, 64, dtype=torch.bfloat16))
    assert pf.shape[-1] == 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 6, 7, 12, 31])
def test_unpadded_rank_on_triton_matches_fp32_reference(monkeypatch, dtype, rank):
    from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")   # Triton on any sm80+ card
    if M._triton_grouped_mm_max_rows(torch.cuda.current_device(), "lora") == 0 \
            or not mg.triton_grouped_available(torch.device("cuda")):
        pytest.skip("Triton grouped GEMM unavailable")
    E, T, H, O = 8, 200, 96, 80
    g = torch.Generator().manual_seed(rank)
    counts = torch.tensor([0, 31, 7, 50, 1, 0, 71, 40])
    offsets = torch.cumsum(counts, 0).to(torch.int32).cuda()
    x = torch.randn(T, H, generator=g).cuda().to(dtype).requires_grad_(True)
    B = (torch.randn(E, H, rank, generator=g) * 0.1).cuda().requires_grad_(True)
    A = (torch.randn(E, rank, O, generator=g) * 0.1).cuda().requires_grad_(True)
    gy = torch.randn(T, O, generator=g).cuda().to(dtype)
    before = dict(mg.GENERIC_CALLS)
    out = M._apply_lora_grouped_mm(x, B.to(dtype), A.to(dtype), offsets, 0.5)
    out.backward(gy)
    assert mg.GENERIC_CALLS["gemm"] > before.get("gemm", 0), "Triton path did not run"
    xr = x.detach().float().requires_grad_(True)
    Br = B.detach().clone().requires_grad_(True)
    Ar = A.detach().clone().requires_grad_(True)
    ref = _loop_reference(_loop_reference(xr, Br.to(dtype).float(), offsets.cpu()), Ar.to(dtype).float(), offsets.cpu()) * 0.5
    ref.backward(gy.float())
    tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    for got, want in ((out.float(), ref), (x.grad.float(), xr.grad), (B.grad, Br.grad), (A.grad, Ar.grad)):
        assert ((got - want).abs().max() / want.abs().max()).item() < tol


def test_every_pad_call_site_passes_the_gemm_input():
    """Without the input the pad cannot see the Triton route and always copies."""
    import ast
    import inspect
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

    calls = 0
    for module in (M, ML):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "_pad_lora_rank_for_grouped_mm":
                calls += 1
                assert len(node.args) + len(node.keywords) == 3, ast.unparse(node)
    assert calls >= 6
