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

import pytest
import torch
import torch.nn.functional as F

E, K, FB = 4, 2, 128


def _quantize_blocks(w):
    fmax = torch.finfo(torch.float8_e4m3fn).max
    e, n, k = w.shape
    blk = w.float().reshape(e, n // FB, FB, k // FB, FB)
    scale = blk.abs().amax(dim = (2, 4)).clamp(min = 1e-12) / fmax
    q = (blk / scale[:, :, None, :, None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn).reshape(e, n, k)
    deq = (q.float().reshape(e, n // FB, FB, k // FB, FB) * scale[:, :, None, :, None]).reshape(e, n, k)
    return q, scale, deq


def test_only_classes_with_their_own_gate_are_marked():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_experts_own_gate

    class Own(torch.nn.Module):
        def _apply_gate(self, x):
            return x

    def _default_apply_gate(self, x):
        return x

    class Default(torch.nn.Module):
        _apply_gate = _default_apply_gate

    class GptOssExperts(Own):
        pass

    assert _fp8_experts_own_gate(Own())
    assert not _fp8_experts_own_gate(Default())
    assert not _fp8_experts_own_gate(GptOssExperts())
    ungated = Own()
    ungated.has_gate = False
    assert not _fp8_experts_own_gate(ungated)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the FP8 MoE backend needs CUDA")
def test_stock_fp8_experts_keep_the_configured_swiglu_through_the_unsloth_backend():
    finegrained_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    if not hasattr(finegrained_fp8, "FP8Experts"):
        pytest.skip("FP8Experts is transformers 5")
    from transformers import PretrainedConfig
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import forward_moe_backend_fp8

    config = PretrainedConfig()
    config.hidden_size, config.moe_intermediate_size, config.num_experts = FB, FB, E
    config.intermediate_size = FB
    config.hidden_act = "silu"
    config.swiglu_alpha, config.swiglu_limit = 1.702, 1.0
    experts = finegrained_fp8.FP8Experts(config, block_size = (FB, FB)).cuda()
    g = torch.Generator().manual_seed(0)
    dense = {}
    with torch.no_grad():
        for name, shape, scale in (("gate_up_proj", (E, 2 * FB, FB), 0.3), ("down_proj", (E, FB, FB), 0.05)):
            q, s, deq = _quantize_blocks(torch.randn(*shape, generator = g) * scale)
            getattr(experts, name).copy_(q.cuda())
            getattr(experts, name + "_scale_inv").copy_(s.cuda())
            dense[name] = deq.cuda()
    g = torch.Generator().manual_seed(1)
    hidden = (torch.randn(64, FB, generator = g) * 2).to("cuda", torch.bfloat16)
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(64)]).cuda()
    top_k_weights = torch.rand(64, K, generator = g).to("cuda", torch.bfloat16)

    def reference(gate):
        final = torch.zeros(64, FB, device = "cuda")
        for t in range(64):
            for j in range(K):
                e = int(top_k_index[t, j])
                h = gate(F.linear(hidden[t].float(), dense["gate_up_proj"][e]))
                final[t] += F.linear(h, dense["down_proj"][e]) * top_k_weights[t, j].float()
        return final

    expected = reference(experts._apply_gate)
    plain = reference(lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
    out = forward_moe_backend_fp8(experts, hidden, top_k_index, top_k_weights).float()
    err = (out - expected).abs().max().item()
    gap = (plain - expected).abs().max().item()
    if gap <= 0.1:
        assert err < 1e-2 * expected.abs().max().item(), (err, gap)
        return
    assert err < gap / 5, (err, gap)
