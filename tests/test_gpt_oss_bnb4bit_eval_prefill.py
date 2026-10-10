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

from transformers import GptOssConfig
from unsloth_zoo.temporary_patches import gpt_oss
from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit


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


def _bnb_cuda():
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _grouped_mm_bf16():
    from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported
    return torch.cuda.is_bf16_supported() and _check_torch_grouped_mm_supported()


@pytest.mark.skipif(not _bnb_cuda(), reason = "needs CUDA and bitsandbytes")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("which", FORWARDS)
def test_eval_prefill_on_nf4_experts(monkeypatch, which, dtype):
    # bf16 with torch._grouped_mm must take the grouped path; elsewhere (fp16, T4) grouped or the
    # loop, but never the dense branch, and the output must match it.
    import bitsandbytes as bnb
    from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising = False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    H4, I4 = 256, 192
    config = GptOssConfig(
        num_local_experts = E, num_experts_per_tok = TOP_K, hidden_size = H4,
        intermediate_size = I4, num_hidden_layers = 1, torch_dtype = dtype,
    )
    experts = GptOssExpertsBnb4bit(config)
    g = torch.Generator().manual_seed(0)
    rows = []
    for name, (i, o) in (("gate_up_projs", (H4, 2 * I4)), ("down_projs", (I4, H4))):
        lins = []
        for _ in range(E):
            lin = bnb.nn.Linear4bit(i, o, bias = True, compute_dtype = dtype, quant_type = "nf4")
            lin.weight = bnb.nn.Params4bit((torch.randn(o, i, generator = g) * 0.05).to(dtype),
                                           requires_grad = False, quant_type = "nf4")
            lin.bias = torch.nn.Parameter((torch.randn(o, generator = g) * 0.1).to(dtype), requires_grad = False)
            if name == "gate_up_projs":
                lin.register_forward_hook(lambda m, args, out: rows.append(args[0].shape[0]))
            lins.append(lin)
        setattr(experts, name, torch.nn.ModuleList(lins))
    experts = _bind(experts.cuda().eval(), which)
    T = getattr(experts, "_dense_eval_max_rows", 8192) // E + 1
    x = torch.randn(1, T, H4, device = "cuda", dtype = dtype)
    idx, weights = _routing(T, "cuda")
    weights = weights.to(dtype)
    with torch.no_grad():
        before = gq.CALLS["forward"] + gq.CALLS.get("forward_fp16", 0)
        out = experts(x, router_indices = idx, routing_weights = weights)
        grouped = gq.CALLS["forward"] + gq.CALLS.get("forward_fp16", 0) - before
        fed = sum(rows)
        monkeypatch.setattr(experts, "_dense_eval_max_rows", 10**12, raising = False)
        rows.clear()
        dense = experts(x, router_indices = idx, routing_weights = weights)
        assert sum(rows) == E * T
    if dtype is torch.bfloat16 and _grouped_mm_bf16():
        assert grouped == 1 and fed == 0
    else:
        assert (grouped, fed) in ((1, 0), (0, TOP_K * T)), (grouped, fed)
    assert out.dtype == x.dtype and out.shape == dense.shape
    rel = float((out.float() - dense.float()).norm() / dense.float().norm())
    assert rel < 2e-2, rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("which", FORWARDS)
def test_eval_prefill_unquantized_16bit_experts(monkeypatch, which, dtype):
    # Experts left in 16-bit (llm_int8_skip_modules) must keep their dtype on the routed path.
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "0")
    config = GptOssConfig(
        num_local_experts = E, num_experts_per_tok = TOP_K, hidden_size = H,
        intermediate_size = INTER, num_hidden_layers = 1, torch_dtype = dtype,
    )
    experts = _bind(GptOssExpertsBnb4bit(config).to("cuda", dtype).eval(), which)
    T = getattr(experts, "_dense_eval_max_rows", 8192) // E + 1
    x = torch.randn(1, T, H, device = "cuda", dtype = dtype)
    idx, weights = _routing(T, "cuda")
    with torch.no_grad():
        out = experts(x, router_indices = idx, routing_weights = weights.to(dtype))
    assert out.dtype == dtype and bool(torch.isfinite(out).all())
