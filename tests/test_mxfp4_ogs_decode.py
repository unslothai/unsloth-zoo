# SPDX-License-Identifier: AGPL-3.0-only
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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "matmul_ogs runs on CUDA only")

from unsloth_zoo.temporary_patches import gpt_oss  # noqa: E402
from test_mxfp4_packed_experts import _tiny_gpt_oss  # noqa: E402


@pytest.fixture
def ogs(monkeypatch):
    if gpt_oss._ogs_modules() is None:
        pytest.skip("triton_kernels is not importable here")
    monkeypatch.setenv("UNSLOTH_MXFP4_OGS", "1")
    monkeypatch.setattr(gpt_oss, "_OGS_FAILED", set())
    calls = {"op": 0}
    original = gpt_oss._ogs_weight

    def counted(*args, **kwargs):
        calls["op"] += 1
        return original(*args, **kwargs)

    # _ogs_weight runs only inside the custom op body, so this counts real matmul_ogs launches (not replays).
    monkeypatch.setattr(gpt_oss, "_ogs_weight", counted)
    yield calls
    # Compiled decode variants outlive the test; later exact-path tests must not inherit them.
    torch._dynamo.reset()


def _inputs():
    g = torch.Generator(device = "cuda").manual_seed(13)
    return [torch.randn(b, 1, 128, device = "cuda", dtype = torch.bfloat16, generator = g) for b in (1, 1, 4)]


def _dense_reference(mlp, experts, inputs):
    """The same experts dequantized once into plain bf16 stacks (no shared decode slots involved)."""
    packed = experts.gate_up_proj, experts.down_proj
    experts.gate_up_proj = torch.nn.Parameter(packed[0].dequantize(), requires_grad = False)
    experts.down_proj = torch.nn.Parameter(packed[1].dequantize(), requires_grad = False)
    try:
        with torch.no_grad():
            return [gpt_oss.moe_forward_inference_bf16(mlp, h).float() for h in inputs]
    finally:
        experts.gate_up_proj, experts.down_proj = packed


def test_no_grad_decode_runs_matmul_ogs_and_matches_the_dequantized_experts(ogs):
    model, experts = _tiny_gpt_oss("cuda")
    mlp = model.model.layers[0].mlp
    inputs = _inputs()
    want = _dense_reference(mlp, experts, inputs)
    with torch.no_grad():
        got = [gpt_oss.moe_forward_inference_bf16(mlp, h).float() for h in inputs]
    assert ogs["op"] > 0
    for a, b in zip(got, want):
        # bf16 activations x MXFP4 in one fused GEMM vs dequantize-then-bf16 GEMM: rounding only.
        assert torch.isfinite(a).all() and (a - b).norm() / b.norm() < 2e-2


def test_grad_enabled_forward_keeps_the_exact_path(ogs):
    model, _ = _tiny_gpt_oss("cuda")
    mlp = model.model.layers[0].mlp
    with torch.enable_grad():
        assert gpt_oss._mxfp4_ogs_decode(mlp, _inputs()[0]) is None
        out = gpt_oss.moe_forward_inference_bf16(mlp, _inputs()[0])
    assert ogs["op"] == 0 and torch.isfinite(out).all()


def test_the_switch_turns_it_off(ogs, monkeypatch):
    model, _ = _tiny_gpt_oss("cuda")
    monkeypatch.setenv("UNSLOTH_MXFP4_OGS", "0")
    with torch.no_grad():
        assert gpt_oss._mxfp4_ogs_decode(model.model.layers[0].mlp, _inputs()[0]) is None
    assert ogs["op"] == 0


def test_a_failing_kernel_falls_back_to_the_exact_path(ogs, monkeypatch):
    model, experts = _tiny_gpt_oss("cuda")
    mlp = model.model.layers[0].mlp

    def broken(*args, **kwargs):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(gpt_oss, "_moe_forward_inference_ogs_kernel", broken)
    with torch.no_grad():
        out = gpt_oss.moe_forward_inference_bf16(mlp, _inputs()[0])
        assert gpt_oss._mxfp4_ogs_decode(mlp, _inputs()[0]) is None
    assert experts.gate_up_proj.device in gpt_oss._OGS_FAILED and torch.isfinite(out).all()
