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

# Guards: routed gpt-oss BF16 expert inference (only the top-k experts, no host sync)
# against an fp32 reference built from the effective weights (base + PEFT delta), under
# eager, torch.compile(fullgraph) and CUDA graph replay with changed routes; and the
# decode path that dropped expert LoRA (moe_forward_inference_bf16 read the base weights).
import os

import pytest
import torch

# T4 (sm75) has no bf16: run the same checks in fp16 there. UNSLOTH_TEST_DTYPE=float16 simulates it.
DT = getattr(torch, os.environ.get("UNSLOTH_TEST_DTYPE", "")) if os.environ.get("UNSLOTH_TEST_DTYPE") else (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton kernels need CUDA")

peft = pytest.importorskip("peft")
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
import transformers.models.gpt_oss.modeling_gpt_oss as M

# Non power-of-two sizes; E > top_k so most experts are unused per token.
H, I, E, TOP_K = 96, 80, 8, 2


def _config():
    return GptOssConfig(hidden_size = H, intermediate_size = I, num_local_experts = E,
                        num_experts_per_tok = TOP_K, num_hidden_layers = 1, num_attention_heads = 2,
                        num_key_value_heads = 1, head_dim = 16, vocab_size = 64)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = M.GptOssMLP(_config())

    def forward(self, x):
        return self.mlp(x)


def _toy(lora, seed = 0):
    torch.manual_seed(seed)
    t = _Toy()
    with torch.no_grad():
        for p in t.parameters():
            p.normal_(0, 0.05)
    t = t.cuda().to(DT)
    if lora:
        t = peft.get_peft_model(t, peft.LoraConfig(
            r = 4, lora_alpha = 8, target_modules = [],
            target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        ))
        with torch.no_grad():
            for n, p in t.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.2)
        mlp = t.base_model.model.mlp
    else:
        mlp = t.mlp
    return t, mlp.eval()


def _base_and_deltas(experts):
    m, deltas = experts, {}
    while hasattr(m, "base_layer"):
        if hasattr(m, "lora_A") and not m.disable_adapters:
            deltas[m.parameter_name] = m.get_delta_weight("default").float()
        m = m.base_layer
    return m, deltas


def _reference(experts, x, idx, weights):
    """fp32 per-token loop over the picked experts with effective weights."""
    base, deltas = _base_and_deltas(experts)
    w_gu = base.gate_up_proj.float() + deltas.get("gate_up_proj", 0)
    w_dn = base.down_proj.float() + deltas.get("down_proj", 0)
    xs = x.reshape(-1, H).float()
    out = torch.zeros_like(xs)
    for t in range(xs.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            gu = xs[t] @ w_gu[e] + base.gate_up_proj_bias[e].float()
            gate, up = gu[::2].clamp(max = base.limit), gu[1::2].clamp(-base.limit, base.limit)
            h = (up + 1) * gate * torch.sigmoid(gate * base.alpha)
            out[t] += float(weights[t, k]) * (h @ w_dn[e] + base.down_proj_bias[e].float())
    return out


def _routes(T, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator = g)
    val, idx = logits.topk(TOP_K, -1)
    return idx.cuda(), val.softmax(-1).cuda()


from unsloth_zoo.temporary_patches.gpt_oss_routed import (
    routed_bf16_forward, routed_bf16_eligible, routed_bf16_gemm,
)


@pytest.mark.parametrize("T", [1, 3, 8, 20])
def test_routed_gemm_matches_reference(T):
    torch.manual_seed(T)
    w = (torch.randn(E, H, 2 * I, device = "cuda") * 0.05).to(DT)
    b = (torch.randn(E, 2 * I, device = "cuda") * 0.05).to(DT)
    x = torch.randn(T, H, device = "cuda").to(DT)
    idx, _ = _routes(T, T)
    flat = idx.reshape(-1)
    ref = torch.einsum("pk,pkn->pn", x.float().repeat_interleave(TOP_K, 0), w.float()[flat]) + b.float()[flat]
    got = routed_bf16_gemm(x, flat, w, b, row_div = TOP_K)
    torch.testing.assert_close(got, ref, atol = 2e-3, rtol = 2e-3)
    # Transposed storage (zoo's ParameterModule view) gives the same result.
    wt = w.transpose(1, 2).contiguous().transpose(1, 2)
    torch.testing.assert_close(routed_bf16_gemm(x, flat, wt, b, row_div = TOP_K), got, atol = 1e-4, rtol = 1e-4)


@pytest.mark.parametrize("lora", [False, True], ids = ["base", "lora"])
@pytest.mark.parametrize("T", [1, 5])
def test_routed_forward_matches_effective_weights(lora, T):
    model, mlp = _toy(lora)
    x = torch.randn(1, T, H, device = "cuda").to(DT)
    idx, wts = _routes(T, 7)
    with torch.no_grad():
        assert routed_bf16_eligible(mlp.experts, x)
        got = routed_bf16_forward(mlp.experts, x, idx, wts).float().reshape(T, H)
        ref = _reference(mlp.experts, x, idx, wts)
        torch.testing.assert_close(got, ref, atol = 2e-2, rtol = 2e-2)
        if lora:
            # Dense [T, E] routing weights give the same answer as compact [T, top_k].
            dense = torch.zeros(T, E, device = "cuda").scatter_(1, idx, wts)
            torch.testing.assert_close(routed_bf16_forward(mlp.experts, x, idx, dense).float().reshape(T, H), got)
            with model.disable_adapter():
                off = routed_bf16_forward(mlp.experts, x, idx, wts).float().reshape(T, H)
                torch.testing.assert_close(off, _reference(mlp.experts, x, idx, wts), atol = 2e-2, rtol = 2e-2)
            assert (off - got).abs().max() > 1e-2  # the adapter is really applied


def test_grad_enabled_fp32_and_kill_switch_are_not_eligible(monkeypatch):
    _, mlp = _toy(False)
    x = torch.randn(1, 1, H, device = "cuda").to(DT)
    assert not routed_bf16_eligible(mlp.experts, x)  # grad enabled
    with torch.no_grad():
        assert not routed_bf16_eligible(mlp.float().experts, x.float())  # tl.dot would use TF32
        mlp.to(DT)
        assert routed_bf16_eligible(mlp.experts, x)
        monkeypatch.setenv("UNSLOTH_GPTOSS_ROUTED_KERNEL", "0")
        assert not routed_bf16_eligible(mlp.experts, x)


@pytest.mark.parametrize("lora", [False, True], ids = ["base", "lora"])
def test_fullgraph_compile_and_cuda_graph_replay_with_changed_routes(lora):
    torch._dynamo.reset()
    _, mlp = _toy(lora)
    x = torch.randn(1, 2, H, device = "cuda").to(DT)
    idx, wts = _routes(2, 1)
    with torch.no_grad():
        eager = routed_bf16_forward(mlp.experts, x, idx, wts)
        compiled = torch.compile(routed_bf16_forward, fullgraph = True)
        torch.testing.assert_close(compiled(mlp.experts, x, idx, wts), eager, atol = 1e-2, rtol = 1e-2)

        s_idx, s_w = idx.clone(), wts.clone()
        routed_bf16_forward(mlp.experts, x, s_idx, s_w)  # warm the Triton JIT outside capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = routed_bf16_forward(mlp.experts, x, s_idx, s_w)
        new_idx, new_w = _routes(2, 99)
        assert not torch.equal(new_idx, idx)
        s_idx.copy_(new_idx); s_w.copy_(new_w)
        g.replay()
        torch.testing.assert_close(out, routed_bf16_forward(mlp.experts, x, new_idx, new_w))


def test_decode_path_applies_expert_lora():
    """moe_forward_inference_bf16 (the compiled qlen == 1 decode branch) used to unwrap PEFT and
    read the base expert weights, so generation ignored expert LoRA that training applied."""
    from unsloth_zoo.temporary_patches.gpt_oss import moe_forward_inference_bf16
    model, mlp = _toy(True)
    x = torch.randn(1, 1, H, device = "cuda").to(DT)
    with torch.no_grad():
        ref = mlp(x)
        ref = (ref[0] if isinstance(ref, tuple) else ref).float()
        with model.disable_adapter():
            off = mlp(x)
            off = (off[0] if isinstance(off, tuple) else off).float()
        got = moe_forward_inference_bf16(mlp, x).float()
    assert (ref - off).abs().max() > 1e-2
    torch.testing.assert_close(got.reshape(ref.shape), ref, atol = 2e-2, rtol = 2e-2)


def test_param_wrapper_hook_routes_only_at_the_outermost_wrapper():
    from unsloth_zoo.temporary_patches.moe_utils import _gpt_oss_routed_wrapper_forward
    _, mlp = _toy(lora = True)
    outer = mlp.experts
    assert hasattr(outer, "base_layer") and hasattr(outer.base_layer, "base_layer")
    base = outer.get_base_layer()
    x = torch.randn(3, H, device = "cuda", dtype = DT)
    idx, weights = _routes(3, seed = 5)
    with torch.no_grad():
        got = _gpt_oss_routed_wrapper_forward(outer, base, x, (), {"router_indices": idx, "routing_weights": weights})
        # An inner wrapper sees only part of the adapter chain, so it must decline.
        assert _gpt_oss_routed_wrapper_forward(outer.base_layer, base, x, (), {"router_indices": idx, "routing_weights": weights}) is None
    assert got is not None
    ref = _reference(outer, x, idx, weights)
    assert (got.float().reshape_as(ref) - ref).abs().max().item() < 2e-2 * max(1.0, ref.abs().max().item())
    # Grad-enabled calls keep the module forward.
    assert _gpt_oss_routed_wrapper_forward(outer, base, x, (), {"router_indices": idx, "routing_weights": weights}) is None


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("B", [1, 3])
def test_routed_mlp_forward_matches_module_on_batched_decode(monkeypatch, lora, B):
    """routed_mlp_forward gets [B, 1, H] decode states. The transformers 5 router normalizes its
    top-k scores with softmax(dim=1), so it has to see [tokens, H], else every pick weighs 1."""
    from unsloth_zoo.temporary_patches.gpt_oss_routed import routed_mlp_forward
    model, mlp = _toy(lora)
    x = torch.randn(B, 1, H, device = "cuda").to(DT)
    with torch.no_grad():
        got = routed_mlp_forward(mlp, x)
        monkeypatch.setenv("UNSLOTH_GPTOSS_ROUTED_KERNEL", "0")
        ref = mlp(x)
        ref = (ref[0] if isinstance(ref, tuple) else ref).float()
    assert got is not None
    torch.testing.assert_close(got.float().reshape(ref.shape), ref, atol = 2e-2, rtol = 2e-2)


@pytest.mark.parametrize("env", ["UNSLOTH_GPTOSS_ROUTED_KERNEL", "UNSLOTH_GPTOSS_ROUTED_INFERENCE"])
def test_both_kill_switch_names_turn_routing_off(monkeypatch, env):
    _, mlp = _toy(False)
    x = torch.randn(1, 1, H, device = "cuda").to(DT)
    monkeypatch.setenv(env, "0")
    assert not routed_bf16_eligible(mlp.experts, x)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "needs two GPUs")
def test_routed_kernels_launch_on_the_tensors_device():
    # A multi-GPU device_map puts layers on a non-current GPU; the launch must follow the tensors.
    x = torch.randn(3, H, device = "cuda:1").to(DT)
    w = torch.randn(E, H, I, device = "cuda:1").to(DT)
    idx = torch.tensor([0, 3, 5], device = "cuda:1")
    with torch.cuda.device(0):
        out = routed_bf16_gemm(x, idx, w)
    ref = torch.stack([x[p].float() @ w[idx[p]].float() for p in range(3)])
    torch.testing.assert_close(out, ref, atol = 2e-2, rtol = 2e-2)


def test_mixed_adapter_batch_is_left_to_peft():
    from functools import partial
    from peft.tuners.lora.model import _adapter_names_pre_forward_hook
    _, mlp = _toy(True)
    x = torch.randn(1, 1, H, device = "cuda").to(DT)
    handles = []
    try:
        with torch.no_grad():
            assert routed_bf16_eligible(mlp.experts, x)
            handles = [m.register_forward_pre_hook(partial(_adapter_names_pre_forward_hook, adapter_names = ["default"]), with_kwargs = True)
                       for m in mlp.modules() if hasattr(m, "lora_A")]
            assert not routed_bf16_eligible(mlp.experts, x)
    finally:
        for h in handles:
            h.remove()
