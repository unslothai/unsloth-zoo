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

"""Native DeepSeek-V4.1 experts on Unsloth's MoE backend match the native sorted loop, MXFP4 stacks included."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsv41_impl import native  # noqa: E402

IMPL = native()
pytestmark = pytest.mark.skipif(IMPL is None, reason = "needs Unsloth's native DeepSeek-V4.1 module")

E, H, I, TOP_K, T = 4, 64, 32, 2, 24


def _config(mxfp4 = False):
    config = IMPL.TextConfig(
        vocab_size = 256, hidden_size = H, num_hidden_layers = 1, num_attention_heads = 2, head_dim = 64,
        qk_rope_head_dim = 16, q_lora_rank = 32, o_groups = 2, o_lora_rank = 16, moe_intermediate_size = I,
        n_routed_experts = E, num_experts_per_tok = TOP_K, engram_layer_ids = [], num_nextn_predict_layers = 0,
    )
    if mxfp4:
        config.unsloth_experts_format = "mxfp4"
    return config


def _routing(seed = 0, dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(T, H, generator = g).to(dtype)
    scores = torch.rand(T, E, generator = g)
    top_k_weights, top_k_index = scores.topk(TOP_K, dim = -1)
    return hidden, top_k_index, top_k_weights.float()


def _dense_experts(dtype = torch.bfloat16, seed = 0):
    experts = IMPL.modeling.DeepseekV41Experts(_config(), E)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn(E, 2 * I, H, generator = g) * 0.3)
        experts.down_proj.copy_(torch.randn(E, H, I, generator = g) * 0.3)
    return experts.to(dtype)


def _packed_experts(finalize = True, seed = 0):
    experts = IMPL.modeling.DeepseekV41Experts(_config(mxfp4 = True), E)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in ("gate_up", "down"):
            blocks = experts._parameters[f"{name}_blocks"]
            blocks.copy_(torch.randint(0, 256, blocks.shape, generator = g, dtype = torch.uint8))
            scales = experts._parameters[f"{name}_scales"]
            scales.copy_(torch.randint(124, 130, scales.shape, generator = g, dtype = torch.uint8))
    return experts.finalize_packed() if finalize else experts


@pytest.fixture(scope = "module")
def reference_forward():
    """The native sorted loop, captured before the MoE patch replaces the class forward."""
    import inspect

    from unsloth_zoo.temporary_patches.utils import _get_unique_storage_name

    cls = IMPL.modeling._DeepseekV41Experts
    source_forward = None
    for fn in (cls.__dict__.get("forward"), getattr(cls, _get_unique_storage_name(cls, "forward"), None)):
        while fn is not None and source_forward is None:
            try:
                if "experts_sorted_loop" in inspect.getsource(fn):
                    source_forward = fn
            except (OSError, TypeError):
                pass
            fn = getattr(fn, "__wrapped__", None)
    if source_forward is None:
        pytest.skip("cannot reach the native experts forward")
    return source_forward


@pytest.fixture(scope = "module")
def patched():
    IMPL.register()
    from unsloth_zoo.temporary_patches.deepseek_v41 import patch_deepseek_v41_moe

    saved = os.environ.pop("UNSLOTH_DSV41_MOE_BACKEND", None)  # the kill switch must not turn these into skips
    try:
        patch_deepseek_v41_moe()
    finally:
        if saved is not None:
            os.environ["UNSLOTH_DSV41_MOE_BACKEND"] = saved
    cls = IMPL.modeling.DeepseekV41Experts
    assert getattr(cls, "_unsloth_already_patched", False), "the DeepSeek-V4.1 MoE backend patch did not install"
    return cls


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_loop_backend_matches_the_native_forward(reference_forward, patched, dtype):
    from unsloth_zoo.temporary_patches.moe_utils import forward_native_moe_loop

    experts = _dense_experts(dtype)
    hidden, index, weights = _routing(dtype = dtype)
    want = reference_forward(experts, hidden, index, weights)
    got = forward_native_moe_loop(experts, hidden, index, weights)
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want), (got.float() - want.float()).abs().max()


@pytest.mark.parametrize(
    "weight_dtype,dtype",
    [(torch.float32, torch.bfloat16), (torch.float32, torch.float16), (torch.bfloat16, torch.float32)],
)
def test_native_loop_backend_casts_dense_experts_like_the_native_forward(reference_forward, patched, weight_dtype, dtype):
    from unsloth_zoo.temporary_patches.moe_utils import forward_native_moe_loop

    experts = _dense_experts(weight_dtype)
    hidden, index, weights = _routing(dtype = dtype)
    want = reference_forward(experts, hidden, index, weights)
    got = forward_native_moe_loop(experts, hidden, index, weights)
    assert got.dtype == want.dtype == dtype
    if dtype == torch.float32:
        # An fp32 result keeps the last-bit differences of the CPU's matmul kernel, which varies by runner
        # (seen: 5.7e-6 abs on some CI hosts); fp32 tolerances still catch a forward that skipped the cast.
        torch.testing.assert_close(got, want)
    else:
        torch.testing.assert_close(got, want, rtol = 0, atol = 0)


def test_native_loop_backend_keeps_a_cancelling_residual(patched):
    from unsloth_zoo.temporary_patches.moe_utils import forward_native_moe_loop

    experts = _dense_experts(torch.bfloat16)
    hidden, index, _ = _routing()
    weights = torch.tensor([[1.0, -1.0 + 2 ** -12]] * T)
    index = torch.tensor([[0, 1]] * T)
    with torch.no_grad():
        experts.gate_up_proj[1].copy_(experts.gate_up_proj[0])
        experts.down_proj[1].copy_(experts.down_proj[0])
    got = forward_native_moe_loop(experts, hidden, index, weights).float()
    y0 = experts._apply_gate(torch.nn.functional.linear(hidden, experts.gate_up_proj[0]).float()).to(torch.bfloat16)
    y0 = torch.nn.functional.linear(y0, experts.down_proj[0]).float()
    want = (y0 + y0 * (-1.0 + 2 ** -12)).to(torch.bfloat16).float()
    assert want.abs().max() > 0
    assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "torch._grouped_mm needs CUDA here")
@pytest.mark.parametrize("packed", [False, True], ids = ["dense", "mxfp4"])
def test_grouped_mm_backend_matches_the_native_forward(reference_forward, patched, packed):
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_supported_on, forward_native_grouped_mm

    if not _grouped_mm_supported_on(torch.device("cuda")):
        pytest.skip("no torch._grouped_mm on this GPU")
    experts = (_packed_experts() if packed else _dense_experts(torch.bfloat16)).cuda()
    hidden, index, weights = (t.cuda() for t in _routing())
    want = reference_forward(experts, hidden, index, weights).float()
    got = forward_native_grouped_mm(experts, hidden, index, weights).float()
    torch.testing.assert_close(got, want, rtol = 1e-2, atol = 1e-2)


def test_packed_mxfp4_experts_on_the_loop_backend(reference_forward, patched):
    from unsloth_zoo.mxfp4_dequant import is_mxfp4_expert_param
    from unsloth_zoo.temporary_patches.moe_utils import forward_native_moe_loop

    experts = _packed_experts()
    assert is_mxfp4_expert_param(experts.gate_up_proj) and is_mxfp4_expert_param(experts.down_proj)
    hidden, index, weights = _routing()
    want = reference_forward(experts, hidden, index, weights)
    got = forward_native_moe_loop(experts, hidden, index, weights)
    assert torch.isfinite(want).all()
    assert torch.equal(got, want)


def test_dispatch_finalizes_raw_packed_blocks(reference_forward, patched):
    experts = _packed_experts(finalize = False)
    assert "gate_up_blocks" in experts._parameters
    hidden, index, weights = _routing()
    got = experts(hidden, index, weights)  # the patched class forward: backend dispatch
    assert "gate_up_blocks" not in experts._parameters
    want = reference_forward(_packed_experts(), hidden, index, weights)
    torch.testing.assert_close(got.float(), want.float(), rtol = 1e-2, atol = 1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton grouped GEMM needs CUDA")
@pytest.mark.parametrize("packed", [False, True], ids = ["dense", "mxfp4"])
def test_triton_backend_matches_the_native_forward(reference_forward, patched, packed):
    pytest.importorskip("unsloth.kernels.moe.grouped_gemm.interface")
    from unsloth_zoo.temporary_patches.moe_utils import forward_triton_grouped_gemm

    experts = (_packed_experts() if packed else _dense_experts(torch.bfloat16)).cuda()
    hidden, index, weights = (t.cuda() for t in _routing())
    want = reference_forward(experts, hidden, index, weights).float()
    got = forward_triton_grouped_gemm(experts, hidden, index, weights).float()
    torch.testing.assert_close(got, want, rtol = 2e-2, atol = 2e-2)


def test_cpu_standalone_load_keeps_the_native_loop(monkeypatch):
    import importlib.util

    import unsloth_zoo.temporary_patches.deepseek_v41 as patches

    spec = importlib.util.spec_from_file_location("unsloth_zoo.cpu._guard_test_dsv41", patches.__file__)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "unsloth_zoo.cpu"
    common = type(sys)("unsloth_zoo.cpu.common")
    common.TEMPORARY_PATCHES = []
    monkeypatch.setitem(sys.modules, "unsloth_zoo.cpu.common", common)
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module, "_native_modeling", lambda: called.append(1) or IMPL.modeling)
    monkeypatch.delenv("UNSLOTH_DSV41_MOE_BACKEND", raising = False)
    forward = IMPL.modeling.DeepseekV41Experts.__dict__.get("forward")
    module.patch_deepseek_v41_moe()  # must not raise, must not touch the class
    assert IMPL.modeling.DeepseekV41Experts.__dict__.get("forward") is forward
