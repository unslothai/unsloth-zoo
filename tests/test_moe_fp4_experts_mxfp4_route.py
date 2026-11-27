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

"""expert_dtype="fp4" (transformers 5.x): grouped_mm decodes MXFP4 in-GEMM, no bf16 rebuilds."""

import os
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fp4_experts(E=4, hidden=128, inter=64, seed=0):
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    experts = getattr(fp8, "FP8Experts", None)
    if (
        experts is None
        or "expert_dtype" not in experts.__init__.__code__.co_varnames
        or not hasattr(torch, "float8_e8m0fnu")
    ):
        pytest.skip("this transformers' FP8Experts has no FP4 experts")
    config = types.SimpleNamespace(
        hidden_size=hidden,
        num_experts=E,
        moe_intermediate_size=inter,
        hidden_act="silu",
        swiglu_limit=10.0,
    )
    module = fp8.FP8Experts(config, scale_fmt="ue8m0", expert_dtype="fp4").cuda()
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            w, s = getattr(module, name), getattr(module, f"{name}_scale_inv")
            w.copy_(
                torch.randint(
                    -128, 128, w.shape, dtype=torch.int8, device="cuda", generator=g
                )
            )
            codes = torch.randint(
                120, 128, s.shape, dtype=torch.uint8, device="cuda", generator=g
            )
            s.data = codes.view(torch.float8_e8m0fnu)
    return module


@pytest.mark.parametrize("tokens", [16, 400])
def test_fp4_experts_take_the_mxfp4_grouped_gemm_with_the_dequantized_result(
    monkeypatch, tokens
):
    import unsloth_zoo.mxfp4_gemm as mxfp4_gemm
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm_available
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    if not mxfp4_grouped_mm_available(torch.device("cuda")):
        pytest.skip("no fused MXFP4 grouped GEMM on this GPU")
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", "1")
    calls = []
    fused = mxfp4_gemm.mxfp4_expert_grouped_mm
    monkeypatch.setattr(
        mxfp4_gemm,
        "mxfp4_expert_grouped_mm",
        lambda *a, **k: calls.append(1) or fused(*a, **k),
    )
    module = _fp4_experts()
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    index = torch.randint(0, 4, (tokens, 2), device="cuda", generator=g)
    weights = torch.rand(tokens, 2, device="cuda", generator=g)

    def run(route):
        monkeypatch.setenv("UNSLOTH_FP8_EXPERTS_MXFP4", route)
        inputs = x.clone().requires_grad_(True)
        out = moe_utils_fp8.forward_moe_backend_fp8(module, inputs, index, weights)
        out.float().pow(2).sum().backward()
        return out.detach().float(), inputs.grad.float()

    want_out, want_dx = run("0")  # every pass dequantizes both stacks
    assert calls == []
    got_out, got_dx = run("1")
    assert (
        len(calls) >= 2
    )  # gate_up and down, forward (backward reuses the kernel for dX)
    torch.testing.assert_close(
        got_out, want_out, rtol=2e-2, atol=2e-2 * want_out.abs().max().item()
    )
    torch.testing.assert_close(
        got_dx, want_dx, rtol=2e-2, atol=2e-2 * want_dx.abs().max().item()
    )


def test_fp4_views_follow_an_in_place_scale_update():
    # Float power-of-two scales are converted to codes (a copy), so the cache must notice an in-place update.
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    module = _fp4_experts()
    for name in ("gate_up_proj", "down_proj"):
        s = getattr(module, f"{name}_scale_inv")
        floats = s.detach().view(torch.uint8).float().sub(127).exp2()
        setattr(module, f"{name}_scale_inv", torch.nn.Parameter(floats, requires_grad = False))
    before = moe_utils_fp8._fp4_experts_as_mxfp4(module)
    if before is None:
        pytest.skip("these FP4 experts do not map to MXFP4 views here")
    with torch.no_grad():
        module.down_proj_scale_inv.mul_(2)
    after = moe_utils_fp8._fp4_experts_as_mxfp4(module)
    old_codes, new_codes = (v.mxfp4_scales for v in (before[1], after[1]))
    assert torch.equal(new_codes.to(torch.int16), old_codes.to(torch.int16) + 1)


def test_fp4_views_cached_under_inference_mode_still_train(monkeypatch):
    # generate() runs under inference_mode; training after it reuses the cached views and saves them for backward.
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm_available
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    if not mxfp4_grouped_mm_available(torch.device("cuda")):
        pytest.skip("no fused MXFP4 grouped GEMM on this GPU")
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", "1")
    monkeypatch.setenv("UNSLOTH_FP8_EXPERTS_MXFP4", "1")
    module = _fp4_experts()
    for name in ("gate_up_proj", "down_proj"):
        s = getattr(module, f"{name}_scale_inv")
        floats = s.detach().view(torch.uint8).float().sub(127).exp2()
        setattr(module, f"{name}_scale_inv", torch.nn.Parameter(floats, requires_grad = False))
    g = torch.Generator(device = "cuda").manual_seed(2)
    x = torch.randn(16, 128, device = "cuda", dtype = torch.bfloat16, generator = g)
    index = torch.randint(0, 4, (16, 2), device = "cuda", generator = g)
    weights = torch.rand(16, 2, device = "cuda", generator = g)
    with torch.inference_mode():
        moe_utils_fp8.forward_moe_backend_fp8(module, x, index, weights)
    inputs = x.clone().requires_grad_(True)
    moe_utils_fp8.forward_moe_backend_fp8(module, inputs, index, weights).float().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
