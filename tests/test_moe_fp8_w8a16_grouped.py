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

"""Block-FP8 routed experts (transformers FP8Experts) take the W8A16 grouped GEMM on CUDA sm89+: weights decode in
the GEMM main loop to the values the whole-stack dequant writes, so no bf16 expert stack is built per call.
UNSLOTH_MOE_FP8_FUSED=0, ROCm and older GPUs keep the dequant path."""

import copy
import pickle
import sys, os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from unsloth_zoo.packed_experts import Fp8BlockExpertParam
from unsloth_zoo.temporary_patches import moe_fp8_grouped as F

_FP8 = getattr(torch, "float8_e4m3fn", None)
pytestmark = pytest.mark.skipif(_FP8 is None, reason="no float8_e4m3fn")


def _quantize(E, R, C, block = (128, 128), device = "cpu", seed = 0, std = 0.02):
    g = torch.Generator(device = device).manual_seed(seed)
    w = torch.randn(E, R, C, device = device, generator = g) * std
    br, bc = block
    pr, pc = -(-R // br), -(-C // bc)
    wp = torch.nn.functional.pad(w, (0, pc * bc - C, 0, pr * br - R)).view(E, pr, br, pc, bc)
    s = wp.abs().amax(dim = (2, 4)).clamp(min = 1e-12) / 448.0
    q = (wp / s[:, :, None, :, None]).to(_FP8).view(E, pr * br, pc * bc)[:, :R, :C].contiguous()
    return q, s.float().contiguous()


def _fused_ok():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability() < (8, 9):
        return False
    return F.fp8_fused_enabled(torch.device("cuda", torch.cuda.current_device()))


needs_fused = pytest.mark.skipif(not _fused_ok(), reason = "needs the W8A16 grouped GEMM (CUDA sm89+, not ROCm)")


def test_packed_view_decodes_like_the_dequant_and_survives_copies():
    q, s = _quantize(3, 200, 328)
    view = Fp8BlockExpertParam(q.view(torch.uint8), packed_scales = s, packed_block = (128, 128))
    assert view.packed_block == (128, 128) and tuple(view._original_shape) == (3, 200, 328)
    dense = view.dequantize(torch.bfloat16)
    want = (q.float() * s.repeat_interleave(128, 1)[:, :200].repeat_interleave(128, 2)[:, :, :328]).to(torch.bfloat16)
    assert torch.equal(dense, want)
    assert torch.equal(view.dequantize(torch.bfloat16, transpose = True), want.transpose(-2, -1))
    for other in (copy.deepcopy(view), pickle.loads(pickle.dumps(view)), view.to("cpu", copy = True), view.detach()):
        assert type(other) is Fp8BlockExpertParam and other.packed_block == (128, 128)
        assert torch.equal(other.dequantize(torch.bfloat16), want)


def test_packed_view_rejects_a_scale_grid_that_does_not_match_the_block():
    q, s = _quantize(2, 256, 256)
    with pytest.raises(ValueError):
        Fp8BlockExpertParam(q.view(torch.uint8), packed_scales = s, packed_block = (64, 64))


def test_kill_switch_and_rocm_keep_the_dequant_path(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_FP8_FUSED", "0")
    assert F.fp8_fused_mode() == "0"
    assert not F.fp8_fused_enabled(torch.device("cuda", 0))
    monkeypatch.setenv("UNSLOTH_MOE_FP8_FUSED", "auto")
    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(F, "_AVAILABLE", {})
    assert not F.fp8_fused_enabled(torch.device("cuda", 0))
    assert not F.fp8_fused_enabled(torch.device("cpu"))


def test_cpu_experts_are_not_viewed():
    q, s = _quantize(2, 256, 128)
    module = types.SimpleNamespace(
        gate_up_proj = q, gate_up_proj_scale_inv = s, down_proj = q.transpose(1, 2).contiguous(),
        down_proj_scale_inv = s.transpose(1, 2).contiguous(), block_size = (128, 128),
    )
    module.__dict__["_unsloth_fp8_block_views"] = {}
    x = torch.randn(4, 128, dtype = torch.bfloat16)
    assert F.fp8_experts_as_packed(module, x) is None


@needs_fused
@pytest.mark.parametrize("shape", [(128, 1536, 2048), (128, 2048, 768), (16, 1408, 704), (4, 200, 328)])
@pytest.mark.parametrize("rows", [1, 37, 4096])
def test_fused_matches_dequant_plus_grouped_mm(shape, rows):
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    E, R, C = shape
    q, s = _quantize(E, R, C, device = "cuda", seed = rows)
    g = torch.Generator(device = "cuda").manual_seed(1)
    ids = torch.randint(0, max(1, E // 2), (rows,), device = "cuda", generator = g)  # half the experts stay empty
    counts = torch.bincount(ids, minlength = E).to(torch.int32)
    offsets = torch.cumsum(counts, 0, dtype = torch.int32)
    dense = F._dequant_stack(q, s, (128, 128), torch.bfloat16)
    for transpose_b, K in ((True, C), (False, R)):
        x = torch.randn(rows, K, device = "cuda", generator = g).to(torch.bfloat16)
        got = F.fp8_block_grouped_mm(x, q, s, counts, transpose_b, (128, 128)).float()
        want = _grouped_mm_with_backward_fix(x, dense.transpose(-2, -1) if transpose_b else dense, offsets).float()
        torch.testing.assert_close(got, want, rtol = 2e-2, atol = 2e-2 * want.abs().max().item())
    assert not F.FAILED_SHAPES


@needs_fused
def test_fused_backward_is_dx_and_compiles_without_graph_breaks():
    from torch._dynamo.utils import counters
    q, s = _quantize(8, 256, 384, device = "cuda")
    counts = torch.tensor([5, 0, 19, 1, 0, 0, 40, 7], dtype = torch.int32, device = "cuda")
    g = torch.Generator(device = "cuda").manual_seed(3)
    x = torch.randn(72, 384, device = "cuda", generator = g).to(torch.bfloat16)
    gy = torch.randn(72, 256, device = "cuda", generator = g).to(torch.bfloat16)

    def fn(t):
        return F.fp8_block_grouped_mm(t, q, s, counts, True, (128, 128)) * 2.0

    eager = x.clone().requires_grad_(True)
    out = fn(eager)
    out.backward(gy)
    want_dx = F.fp8_block_grouped_mm(gy * 2.0, q, s, counts, False, (128, 128))
    assert torch.equal(eager.grad, want_dx)
    torch._dynamo.reset()
    counters.clear()
    compiled = x.clone().requires_grad_(True)
    out_c = torch.compile(fn, fullgraph = True)(compiled)
    out_c.backward(gy)
    assert sum(counters["graph_break"].values()) == 0
    assert torch.equal(out_c, out) and torch.equal(compiled.grad, eager.grad)


def _fp8_experts(E = 8, hidden = 256, inter = 256):
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    experts = getattr(fp8, "FP8Experts", None)
    if experts is None:
        pytest.skip("this transformers has no FP8Experts")
    config = types.SimpleNamespace(hidden_size = hidden, num_experts = E, moe_intermediate_size = inter, hidden_act = "silu")
    try:
        module = experts(config, block_size = (128, 128)).cuda()
    except TypeError:
        pytest.skip("this FP8Experts takes other arguments")
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(module, name)
            q, s = _quantize(*w.shape, device = "cuda", seed = w.shape[1])
            w.copy_(q)
            getattr(module, f"{name}_scale_inv").copy_(s)
            w.requires_grad_(False)
    return module


@needs_fused
@pytest.mark.parametrize("tokens", [3, 512])
def test_fp8_experts_take_the_fused_gemm_and_match_the_dequant_path(monkeypatch, tokens):
    from unsloth_zoo.temporary_patches import moe_utils_fp8
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    module = _fp8_experts()
    g = torch.Generator(device = "cuda").manual_seed(2)
    x = torch.randn(tokens, 256, device = "cuda", dtype = torch.bfloat16, generator = g)
    index = torch.randint(0, 8, (tokens, 2), device = "cuda", generator = g)
    weights = torch.rand(tokens, 2, device = "cuda", generator = g)

    def run(mode):
        monkeypatch.setenv("UNSLOTH_MOE_FP8_FUSED", mode)
        before = dict(F.FP8_FUSED_STATS)
        inputs = x.clone().requires_grad_(True)
        out = moe_utils_fp8.forward_moe_backend_fp8(module, inputs, index, weights)
        out.float().pow(2).sum().backward()
        used = {k: F.FP8_FUSED_STATS[k] - before[k] for k in before}
        return out.detach().float(), inputs.grad.float(), used

    want_out, want_dx, used_off = run("0")
    assert all(v == 0 for v in used_off.values()), used_off
    got_out, got_dx, used_on = run("auto")
    assert used_on["fwd"] == 2 and used_on["dx"] == 2 and used_on["fallback"] == 0
    torch.testing.assert_close(got_out, want_out, rtol = 2e-2, atol = 2e-2 * want_out.abs().max().item())
    torch.testing.assert_close(got_dx, want_dx, rtol = 2e-2, atol = 2e-2 * want_dx.abs().max().item())


@needs_fused
def test_views_cached_under_inference_mode_still_train(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_utils_fp8
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    monkeypatch.setenv("UNSLOTH_MOE_FP8_FUSED", "auto")
    module = _fp8_experts()
    g = torch.Generator(device = "cuda").manual_seed(4)
    x = torch.randn(16, 256, device = "cuda", dtype = torch.bfloat16, generator = g)
    index = torch.randint(0, 8, (16, 2), device = "cuda", generator = g)
    weights = torch.rand(16, 2, device = "cuda", generator = g)
    with torch.inference_mode():
        moe_utils_fp8.forward_moe_backend_fp8(module, x, index, weights)
    inputs = x.clone().requires_grad_(True)
    moe_utils_fp8.forward_moe_backend_fp8(module, inputs, index, weights).float().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@needs_fused
def test_stack_past_int32_indexing():
    # E * R * C > 2**31: the expert offset is int64; one row on the last expert reads its codes correctly.
    E, R, C = 160, 2048, 7168
    free, _ = torch.cuda.mem_get_info()
    if free < 8 * 2**30:
        pytest.skip("needs 8 GB free")
    q = torch.empty(E, R, C, dtype = _FP8, device = "cuda")
    s = torch.ones(E, R // 128, C // 128, device = "cuda")
    last, ls = _quantize(1, R, C, device = "cuda", seed = 9)
    q[-1].copy_(last[0]); s[-1].copy_(ls[0])
    counts = torch.zeros(E, dtype = torch.int32, device = "cuda"); counts[-1] = 3
    x = torch.randn(3, C, device = "cuda").to(torch.bfloat16)
    got = F.fp8_block_grouped_mm(x, q, s, counts, True, (128, 128)).float()
    dense = F._dequant_stack(last, ls, (128, 128), torch.bfloat16)[0].float()
    want = x.float() @ dense.T
    torch.testing.assert_close(got, want, rtol = 2e-2, atol = 2e-2 * want.abs().max().item())


@needs_fused
@pytest.mark.parametrize("transpose_b", [True, False])
def test_lora_folded_into_the_fused_gemm_matches_the_separated_path(transpose_b):
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    E, R, C, r, rows = 8, 384, 256, 16, 300
    q, s = _quantize(E, R, C, device = "cuda", seed = 5)
    view = Fp8BlockExpertParam(q.view(torch.uint8), packed_scales = s, packed_block = (128, 128))
    K, N = (C, R) if transpose_b else (R, C)
    g = torch.Generator(device = "cuda").manual_seed(6)
    counts = torch.bincount(torch.randint(0, E - 2, (rows,), device = "cuda", generator = g), minlength = E).to(torch.int32)
    offsets = torch.cumsum(counts, 0, dtype = torch.int32)
    x0 = torch.randn(rows, K, device = "cuda", generator = g).to(torch.bfloat16)
    a0 = torch.randn(E, K, r, device = "cuda", generator = g) * 0.05
    b0 = torch.randn(E, r, N, device = "cuda", generator = g) * 0.05
    gy = torch.randn(rows, N, device = "cuda", generator = g).to(torch.bfloat16)

    def leaves():
        return (x0.clone().requires_grad_(True), a0.clone().requires_grad_(True), b0.clone().requires_grad_(True))

    before = F.FP8_FUSED_STATS["lora_fold"]
    x, a, b = leaves()
    out, left = F.fp8_block_base_and_lora(x, view, counts, offsets, transpose_b, (a, b, 2.0))
    assert left is None and F.FP8_FUSED_STATS["lora_fold"] == before + 1
    out.backward(gy)
    assert F.FP8_FUSED_STATS["lora_fold"] == before + 2  # dX also folded
    xr, ar, br = leaves()
    base = F.fp8_block_grouped_mm(xr, q, s, counts, transpose_b, (128, 128))
    t = _grouped_mm_with_backward_fix(xr, ar.to(torch.bfloat16), offsets)
    ref = base + _grouped_mm_with_backward_fix(t, br.to(torch.bfloat16), offsets) * 2.0
    ref.backward(gy)
    for got, want in ((out, ref), (x.grad, xr.grad), (a.grad, ar.grad), (b.grad, br.grad)):
        got, want = got.float(), want.float()
        torch.testing.assert_close(got, want, rtol = 2e-2, atol = 2e-2 * want.abs().max().item())
