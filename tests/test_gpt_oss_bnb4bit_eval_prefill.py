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

"""Eval-mode GptOssExpertsBnb4bit (torch_native_forward) must not run every expert on every token for long inputs.

The dense eval branch holds experts x tokens fp32 swiglu temporaries: a GRPO prefill of
4 x 4608 tokens on gpt-oss-120b asked for 27.5 GiB in one layer (unsloth#3411). Large eval
calls now take the routed per-expert loop; small ones (decode) keep the dense branch.
"""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_GPTOSS_GROUPED", "0")

try:
    from transformers import GptOssConfig
    import unsloth_zoo.temporary_patches.gpt_oss as gpt_oss
    from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit
except Exception as e:  # pragma: no cover - unsloth_zoo import needs an accelerator
    pytest.skip(f"cannot import unsloth_zoo gpt_oss patches: {e}", allow_module_level=True)


E, TOP_K, H, INTER = 8, 2, 16, 12


def _experts():
    config = GptOssConfig(
        num_local_experts = E, num_experts_per_tok = TOP_K, hidden_size = H,
        intermediate_size = INTER, num_hidden_layers = 1, torch_dtype = torch.float32,
    )
    torch.manual_seed(0)
    experts = GptOssExpertsBnb4bit(config).float().eval()
    for lin in list(experts.gate_up_projs) + list(experts.down_projs):
        torch.nn.init.normal_(lin.weight, std = 0.2)
        torch.nn.init.normal_(lin.bias, std = 0.2)
    rows = []
    for lin in experts.gate_up_projs:
        lin.register_forward_hook(lambda m, args, out: rows.append(args[0].shape[0]))
    return experts, rows


def _routing(num_tokens):
    logits = torch.randn(num_tokens, E)
    top, idx = logits.topk(TOP_K, dim = -1)
    weights = torch.zeros(num_tokens, E).scatter_(1, idx, top.softmax(-1))
    return idx, weights


def _reference(experts, x, idx, weights):
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for e in idx[t].tolist():
            gu = experts.gate_up_projs[e](x[t])
            gate = gu[::2].clamp(max = experts.limit)
            up = gu[1::2].clamp(-experts.limit, experts.limit)
            out[t] += weights[t, e] * experts.down_projs[e]((up + 1) * gate * torch.sigmoid(experts.alpha * gate))
    return out


@pytest.mark.parametrize("num_tokens, dense", [(4, True), (64, False)])
def test_eval_branch_by_size(monkeypatch, num_tokens, dense):
    experts, rows = _experts()
    monkeypatch.setattr(gpt_oss, "DENSE_EVAL_MAX_ROWS", 16 * E, raising = False)
    torch.manual_seed(1)
    x = torch.randn(1, num_tokens, H)
    idx, weights = _routing(num_tokens)
    with torch.no_grad():
        out = experts(x, router_indices = idx, routing_weights = weights)
        fed = sum(rows)
        ref = _reference(experts, x[0], idx, weights)
    # Dense feeds every token to every expert; routed only the top_k picks.
    assert fed == (E * num_tokens if dense else TOP_K * num_tokens)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out[0], ref, rtol = 1e-4, atol = 1e-4)


def test_default_cap_routes_long_prefill():
    experts, rows = _experts()
    num_tokens = getattr(gpt_oss, "DENSE_EVAL_MAX_ROWS", 8192) // E + 1
    x = torch.randn(1, num_tokens, H)
    idx, weights = _routing(num_tokens)
    with torch.no_grad():
        experts(x, router_indices = idx, routing_weights = weights)
    assert sum(rows) == TOP_K * num_tokens
