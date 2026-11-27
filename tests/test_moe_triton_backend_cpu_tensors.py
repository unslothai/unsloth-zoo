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

"""The unsloth_triton MoE backend is picked once per process (a host whose grouped_mm probe fails, or
UNSLOTH_MOE_BACKEND=unsloth_triton), but a call's tensors can still be on the CPU: those must run the eager loop,
never the CUDA-only Triton kernel ("X and W must be on CUDA"). CPU only."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts  # noqa: F401
except Exception:
    pytest.skip("needs transformers >= 5 fused expert modules", allow_module_level = True)

from unsloth_zoo.temporary_patches import moe_utils as mu

TOP_K, NUM_EXPERTS = 2, 8


def _experts_and_routing(seed = 0, tokens = 13):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16, num_hidden_layers = 1,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, num_experts = NUM_EXPERTS,
        num_experts_per_tok = TOP_K, norm_topk_prob = True,
    )
    experts = Qwen3MoeForCausalLM(config).float().model.layers[0].mlp.experts
    g = torch.Generator().manual_seed(seed + 1)
    hidden = torch.randn(tokens, config.hidden_size, generator = g)
    weights, index = torch.randn(tokens, NUM_EXPERTS, generator = g).softmax(-1).topk(TOP_K, dim = -1)
    return experts, hidden, index, weights / weights.sum(-1, keepdim = True)


def _run(fn, experts, hidden, index, weights):
    experts.zero_grad(set_to_none = True)
    h = hidden.clone().requires_grad_(True)
    out = fn(experts, h, index, weights)
    out.pow(2).sum().backward()
    return out.detach(), h.grad, [p.grad.clone() for p in experts.parameters()]


def _assert_same(got, want):
    torch.testing.assert_close(got[0], want[0], rtol = 0, atol = 0)
    torch.testing.assert_close(got[1], want[1], rtol = 0, atol = 0)
    for a, b in zip(got[2], want[2]):
        torch.testing.assert_close(a, b, rtol = 0, atol = 0)


def test_triton_forward_on_cpu_tensors_runs_the_eager_loop():
    experts, hidden, index, weights = _experts_and_routing()
    want = _run(mu.forward_native_moe_loop, experts, hidden, index, weights)
    _assert_same(_run(mu.forward_triton_grouped_gemm, experts, hidden, index, weights), want)


def test_unsloth_triton_backend_dispatch_with_cpu_tensors(monkeypatch):
    # Forced as on a host whose grouped_mm probe failed: the process-wide choice must not reach the CUDA kernel.
    monkeypatch.setattr(mu, "select_moe_backend", lambda *a, **k: "unsloth_triton")
    experts, hidden, index, weights = _experts_and_routing(seed = 3)
    want = _run(mu.forward_native_moe_loop, experts, hidden, index, weights)
    _assert_same(_run(mu.forward_moe_backend, experts, hidden, index, weights), want)
