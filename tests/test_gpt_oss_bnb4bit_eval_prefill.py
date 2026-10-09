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

"""Long eval calls of GptOssExpertsBnb4bit skip the all-experts dense branch (unsloth#3411),
through both the bound torch_native_forward and the class body the compiled cache emits."""
import ast
import inspect
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

try:
    from transformers import GptOssConfig
    from unsloth_zoo.temporary_patches import gpt_oss
    from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit
except Exception as e:  # pragma: no cover - unsloth_zoo import needs an accelerator
    pytest.skip(f"cannot import unsloth_zoo gpt_oss patches: {e}", allow_module_level=True)


E, TOP_K, H, INTER = 8, 2, 16, 12
FORWARDS = ["module", "class"]


def _class_body_forward():
    src = inspect.getsource(gpt_oss)
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "GptOssExpertsBnb4bit")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    ns = dict(vars(gpt_oss))
    exec(compile(ast.Module([fn], []), gpt_oss.__file__, "exec"), ns)
    return ns["forward"]


def _bind(experts, which):
    if which == "class":
        experts.forward = _class_body_forward().__get__(experts)
    return experts


def _experts(which):
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
    return _bind(experts, which), rows


def _routing(num_tokens, device = "cpu"):
    logits = torch.randn(num_tokens, E)
    top, idx = logits.topk(TOP_K, dim = -1)
    weights = torch.zeros(num_tokens, E).scatter_(1, idx, top.softmax(-1))
    return idx.to(device), weights.to(device)


def _reference(experts, x, idx, weights):
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for e in idx[t].tolist():
            gu = experts.gate_up_projs[e](x[t])
            gate = gu[::2].clamp(max = experts.limit)
            up = gu[1::2].clamp(-experts.limit, experts.limit)
            out[t] += weights[t, e] * experts.down_projs[e]((up + 1) * gate * torch.sigmoid(experts.alpha * gate))
    return out


@pytest.mark.parametrize("which", FORWARDS)
@pytest.mark.parametrize("num_tokens, dense", [(4, True), (64, False)])
def test_eval_branch_by_size(monkeypatch, which, num_tokens, dense):
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "0")
    experts, rows = _experts(which)
    monkeypatch.setattr(GptOssExpertsBnb4bit, "_dense_eval_max_rows", 16 * E, raising = False)
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


@pytest.mark.parametrize("which", FORWARDS)
def test_default_cap_routes_long_prefill(monkeypatch, which):
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "0")
    experts, rows = _experts(which)
    num_tokens = getattr(GptOssExpertsBnb4bit, "_dense_eval_max_rows", 8192) // E + 1
    x = torch.randn(1, num_tokens, H)
    idx, weights = _routing(num_tokens)
    with torch.no_grad():
        experts(x, router_indices = idx, routing_weights = weights)
    assert sum(rows) == TOP_K * num_tokens


def _grouped_ready():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return False
    try:
        import bitsandbytes  # noqa: F401
        from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported
        return _check_torch_grouped_mm_supported()
    except Exception:
        return False


@pytest.mark.skipif(not _grouped_ready(), reason = "needs CUDA bf16, bitsandbytes and torch._grouped_mm")
@pytest.mark.parametrize("which", FORWARDS)
def test_eval_prefill_takes_grouped_path(monkeypatch, which):
    import bitsandbytes as bnb
    from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising = False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    H4, I4 = 256, 192
    config = GptOssConfig(
        num_local_experts = E, num_experts_per_tok = TOP_K, hidden_size = H4,
        intermediate_size = I4, num_hidden_layers = 1, torch_dtype = torch.bfloat16,
    )
    experts = GptOssExpertsBnb4bit(config)
    g = torch.Generator().manual_seed(0)
    for name, (i, o) in (("gate_up_projs", (H4, 2 * I4)), ("down_projs", (I4, H4))):
        lins = []
        for _ in range(E):
            lin = bnb.nn.Linear4bit(i, o, bias = True, compute_dtype = torch.bfloat16, quant_type = "nf4")
            lin.weight = bnb.nn.Params4bit((torch.randn(o, i, generator = g) * 0.05).to(torch.bfloat16),
                                           requires_grad = False, quant_type = "nf4")
            lin.bias = torch.nn.Parameter((torch.randn(o, generator = g) * 0.1).to(torch.bfloat16), requires_grad = False)
            lins.append(lin)
        setattr(experts, name, torch.nn.ModuleList(lins))
    experts = _bind(experts.cuda().eval(), which)
    T = getattr(experts, "_dense_eval_max_rows", 8192) // E + 1
    x = torch.randn(1, T, H4, device = "cuda", dtype = torch.bfloat16)
    idx, weights = _routing(T, "cuda")
    weights = weights.to(torch.bfloat16)
    with torch.no_grad():
        before = gq.CALLS["forward"]
        out = experts(x, router_indices = idx, routing_weights = weights)
        assert gq.CALLS["forward"] == before + 1
        monkeypatch.setattr(experts, "_dense_eval_max_rows", 10**12, raising = False)
        dense = experts(x, router_indices = idx, routing_weights = weights)
        assert gq.CALLS["forward"] == before + 1
    assert out.dtype == x.dtype and out.shape == dense.shape
    rel = float((out.float() - dense.float()).norm() / dense.float().norm())
    assert rel < 2e-2, rel
