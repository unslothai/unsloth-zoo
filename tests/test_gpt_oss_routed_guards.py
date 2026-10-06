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

"""Value guards for the routed NF4 gpt-oss declines: a decline test only proves the routed
path returned None, so these check the decode entry (torch_native_forward, which tries the
routed kernels first) against an fp64 reference that applies everything PEFT applies."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

from unsloth_zoo.temporary_patches.gpt_oss import torch_native_forward

from test_gpt_oss_routed_nf4 import H, TOP_K, _Experts, _lora_wrap, _routing


def _reference_with_lora_bias(ex, x, idx, w):
    def proj(m, v):
        base = m.base_layer
        W = bnb.functional.dequantize_4bit(base.weight.data, base.weight.quant_state).double()
        a = m.active_adapters[0]
        y = W @ v + base.bias.double()
        y = y + (m.lora_B[a].weight.double() @ (m.lora_A[a].weight.double() @ v) + m.lora_B[a].bias.double()) * m.scaling[a]
        return y
    x2 = x.reshape(-1, H).double()
    out = torch.zeros_like(x2)
    for t in range(x2.shape[0]):
        for e in idx[t].tolist():
            gu = proj(ex.gate_up_projs[e], x2[t])
            gate, up = gu[::2].clamp(max = 7.0), gu[1::2].clamp(-7.0, 7.0)
            out[t] += w[t, e].double() * proj(ex.down_projs[e], (up + 1) * gate * torch.sigmoid(1.702 * gate))
    return out.view(x.shape)


def test_lora_bias_decode_matches_peft_math():
    # lora_bias=True: PEFT adds scaling * lora_B.bias; the routed kernels would drop it.
    ex = _lora_wrap(_Experts(True), lora_bias = True).eval()
    for name, p in ex.named_parameters():
        if "lora_B" in name and name.endswith("bias"):
            p.data.fill_(0.5)
    T = 2
    x = torch.randn(1, T, H, device = "cuda", dtype = torch.float32)
    idx, w = _routing(T, seed = 4)
    assert idx.numel() == T * TOP_K
    with torch.no_grad():
        got = torch_native_forward(ex, x, idx, w)
        ref = _reference_with_lora_bias(ex, x, idx, w)
    err = (got.double() - ref).abs().max().item()
    assert err <= 2 ** -6 * ref.abs().max().item() + 1e-3, (err, ref.abs().max().item())


class _NF4MLP(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
        cfg = GptOssConfig(hidden_size = H, num_local_experts = len(experts.gate_up_projs), num_experts_per_tok = TOP_K)
        self.router = GptOssTopKRouter(cfg).cuda()
        with torch.no_grad():
            self.router.weight.normal_(0, 0.1)
            self.router.bias.normal_(0, 0.1)
        self.experts = experts


def test_routed_mlp_forward_nf4_compiles_fullgraph():
    # The pointer tables are read, not built, while compiling (data_ptr is not traceable).
    from unsloth_zoo.temporary_patches.gpt_oss_routed import routed_mlp_forward
    mlp = _NF4MLP(_Experts(True).eval()).eval()
    x = torch.randn(3, 1, H, device = "cuda", dtype = torch.float32)
    with torch.no_grad():
        eager = routed_mlp_forward(mlp, x)
        assert eager is not None and mlp.experts._unsloth_routed_nf4
        torch._dynamo.reset()
        compiled = torch.compile(lambda h: routed_mlp_forward(mlp, h), fullgraph = True)
        torch.testing.assert_close(compiled(x), eager, rtol = 1e-5, atol = 1e-5)
