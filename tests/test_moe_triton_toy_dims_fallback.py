"""forward_triton_grouped_gemm with a hidden / MoE intermediate size the Triton tiles cannot cover.

tl.dot needs tile dims >= 16 and the kernels static_assert K % BLOCK_SIZE_K == 0, so a toy config
(tiny-Qwen3MoeForCausalLM: hidden_size 8) failed to compile on every GPU. It now runs the same routing
on grouped_mm (or the eager loop); sizes that are multiples of 16 still take the Triton kernels.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_utils as MU

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason = "needs a CUDA device"
)


def _dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16


def _experts(num_experts, hidden, intermediate, dtype):
    torch.manual_seed(0)
    experts = nn.Module()
    experts.num_experts = num_experts
    experts.gate_up_proj = nn.Parameter(
        (torch.randn(num_experts, 2 * intermediate, hidden) / hidden**0.5).to("cuda", dtype)
    )
    experts.down_proj = nn.Parameter(
        (torch.randn(num_experts, hidden, intermediate) / intermediate**0.5).to("cuda", dtype)
    )
    experts.act_fn = F.silu
    return experts


def _routing(tokens, hidden, num_experts, top_k, dtype):
    g = torch.Generator(device = "cuda").manual_seed(1)
    x = torch.randn(tokens, hidden, device = "cuda", generator = g).to(dtype)
    logits = torch.randn(tokens, num_experts, device = "cuda", generator = g)
    w, idx = torch.topk(logits.softmax(-1), top_k, dim = -1)
    return x, idx, (w / w.sum(-1, keepdim = True)).to(dtype)


def _reference(experts, x, idx, w):
    out = torch.zeros(x.shape, dtype = torch.float32, device = x.device)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            gate, up = (experts.gate_up_proj[e].float() @ x[t].float()).chunk(2)
            h = F.silu(gate) * up
            out[t] += w[t, k].float() * (experts.down_proj[e].float() @ h)
    return out


@pytest.mark.parametrize("hidden, intermediate", [(8, 768), (64, 24)])
def test_toy_dims_fall_back_and_match_reference(hidden, intermediate):
    dtype = _dtype()
    experts = _experts(4, hidden, intermediate, dtype)
    x, idx, w = _routing(32, hidden, 4, 2, dtype)
    with torch.no_grad():
        got = MU.forward_triton_grouped_gemm(experts, x, idx, w)
    want = _reference(experts, x, idx, w)
    assert got.shape == x.shape and torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), want, atol = 2e-2, rtol = 2e-2)


def test_multiple_of_16_dims_still_take_the_triton_kernels(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", "1")

    def _no_fallback(*args, **kwargs):
        raise AssertionError("fell back off the Triton kernels")

    monkeypatch.setattr(MU, "forward_native_grouped_mm", _no_fallback)
    monkeypatch.setattr(MU, "forward_native_moe_loop", _no_fallback)
    dtype = _dtype()
    hidden, intermediate = 64, 48
    experts = _experts(4, hidden, intermediate, dtype)
    x, idx, w = _routing(32, hidden, 4, 2, dtype)
    with torch.no_grad():
        got = MU.forward_triton_grouped_gemm(experts, x, idx, w)
    torch.testing.assert_close(got.float(), _reference(experts, x, idx, w), atol = 2e-2, rtol = 2e-2)
