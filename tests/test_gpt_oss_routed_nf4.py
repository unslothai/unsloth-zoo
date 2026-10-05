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

"""Routed NF4 gpt-oss experts: the routed GEMV against bitsandbytes' dequantized weights
per expert (nested and plain, distinct per-expert offsets), the routed forward against
the dense eval branch (with and without LoRA), torch.compile fullgraph and CUDA graph
replay with changed routes."""
import os

import pytest
import torch

# T4 (sm75) has no bf16: run the same checks in fp16 there. UNSLOTH_TEST_DTYPE=float16 simulates it.
DT = getattr(torch, os.environ.get("UNSLOTH_TEST_DTYPE", "")) if os.environ.get("UNSLOTH_TEST_DTYPE") else (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
bnb = pytest.importorskip("bitsandbytes")

from unsloth_zoo.temporary_patches.gpt_oss import torch_native_forward
from unsloth_zoo.temporary_patches.gpt_oss_routed import (
    prepare_routed_experts,
    routed_experts_forward,
    routed_gate_up,
)

E, TOP_K, H, I = 8, 4, 256, 192


def _linear4bit(i, o, nested, seed):
    g = torch.Generator().manual_seed(seed)
    lin = bnb.nn.Linear4bit(i, o, bias = True, compute_dtype = DT, quant_type = "nf4",
                            compress_statistics = nested)
    scale = 0.02 * (1 + seed % 5)  # distinct absmax / offsets per expert
    lin.weight = bnb.nn.Params4bit(torch.randn(o, i, generator = g) * scale, requires_grad = False,
                                   quant_type = "nf4", compress_statistics = nested)
    lin.bias = torch.nn.Parameter(torch.randn(o, generator = g) * 0.1, requires_grad = False)
    return lin.cuda()


class _Experts(torch.nn.Module):
    def __init__(self, nested):
        super().__init__()
        self.gate_up_projs = torch.nn.ModuleList([_linear4bit(H, 2 * I, nested, e) for e in range(E)])
        self.down_projs = torch.nn.ModuleList([_linear4bit(I, H, nested, 100 + e) for e in range(E)])
        self.hidden_size, self.alpha, self.limit = H, 1.702, 7.0

    def dense(self, x, idx, w):
        return torch_native_forward(self, x, idx, w)


def _routing(T, seed = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator = g)
    vals, idx = logits.topk(TOP_K, dim = -1)
    dense = torch.zeros(T, E).scatter_(1, idx, vals.softmax(-1))
    return idx.cuda(), dense.cuda().to(DT)


@pytest.mark.parametrize("nested", [True, False])
def test_kernel_matches_bnb_dequant(nested):
    ex = _Experts(nested).eval()
    state = prepare_routed_experts(ex)
    assert state
    x = torch.randn(5, H, device = "cuda", dtype = DT)
    idx, _ = _routing(5)
    inter = routed_gate_up(x, idx.reshape(-1), state["gate_up"], TOP_K, 1.702, 7.0)
    for p in range(idx.numel()):
        lin = ex.gate_up_projs[idx.view(-1)[p].item()]
        W = bnb.functional.dequantize_4bit(lin.weight.data, lin.weight.quant_state).float()
        gu = W @ x[p // TOP_K].float() + lin.bias.float()
        gate, up = gu[::2].clamp(max = 7.0), gu[1::2].clamp(-7.0, 7.0)
        torch.testing.assert_close(inter[p], (up + 1) * gate * torch.sigmoid(1.702 * gate), rtol = 1e-5, atol = 1e-5)


def _lora_wrap(ex, **kwargs):
    peft = pytest.importorskip("peft")
    cfg = peft.LoraConfig(r = 4, lora_alpha = 8, lora_dropout = 0.0,
                          target_modules = r".*(gate_up_projs|down_projs)\.\d+", **kwargs)
    model = peft.inject_adapter_in_model(cfg, ex)
    g = torch.Generator().manual_seed(7)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = (torch.randn(p.shape, generator = g) * 0.05).to(p.device, p.dtype)
    return model


def _reference(ex, x, idx, w):
    """fp64 routed reference from bitsandbytes' dequantized weights (+ LoRA if present)."""
    def proj(m, v):
        base = getattr(m, "base_layer", m)
        W = bnb.functional.dequantize_4bit(base.weight.data, base.weight.quant_state).double()
        y = W @ v + base.bias.double()
        if hasattr(m, "lora_A") and m.active_adapters and not m.disable_adapters:
            a = m.active_adapters[0]
            y = y + m.lora_B[a].weight.double() @ (m.lora_A[a].weight.double() @ v) * m.scaling[a]
        return y
    x2 = x.reshape(-1, H).double()
    out = torch.zeros_like(x2)
    for t in range(x2.shape[0]):
        for e in idx[t].tolist():
            gu = proj(ex.gate_up_projs[e], x2[t])
            gate, up = gu[::2].clamp(max = 7.0), gu[1::2].clamp(-7.0, 7.0)
            out[t] += w[t, e].double() * proj(ex.down_projs[e], (up + 1) * gate * torch.sigmoid(1.702 * gate))
    return out.view(x.shape)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("T", [1, 4, 9])
def test_routed_forward_matches_reference(lora, T):
    ex = _Experts(True).eval()
    if lora:
        ex = _lora_wrap(ex).eval()
    g = torch.Generator(device = "cuda").manual_seed(T)
    x = torch.randn(1, T, H, device = "cuda", generator = g).to(DT)
    idx, w = _routing(T, seed = T)
    with torch.no_grad():
        routed = routed_experts_forward(ex, x, idx, w)
        dense = torch_native_forward(ex, x, idx, w)
        ref = _reference(ex, x, idx, w)
    assert routed is not None and routed.shape == dense.shape and routed.dtype == dense.dtype
    err_routed = (routed.double() - ref).abs().max().item()
    err_dense = (dense.double() - ref).abs().max().item()
    # Routed accumulates in fp32 and rounds once; dense rounds every expert to bf16 first.
    assert err_routed <= err_dense + 1e-6, (err_routed, err_dense)
    assert err_routed <= 2 ** -8 * ref.abs().max().item() + 1e-3, (err_routed, ref.abs().max().item())
    if lora:
        for m in ex.modules():
            if hasattr(m, "lora_A"):
                m.enable_adapters(False)
        with torch.no_grad():
            base = routed_experts_forward(ex, x, idx, w)
        assert not torch.allclose(base.float(), routed.float())


def test_ineligible_returns_none():
    ex = _Experts(True).eval()
    x = torch.randn(1, 1, H, device = "cuda", dtype = DT)
    idx, w = _routing(1)
    assert routed_experts_forward(ex, x, idx, w) is None  # grad enabled
    os.environ["UNSLOTH_GPTOSS_ROUTED_KERNEL"] = "0"
    try:
        with torch.no_grad():
            assert routed_experts_forward(ex, x, idx, w) is None
    finally:
        del os.environ["UNSLOTH_GPTOSS_ROUTED_KERNEL"]


@pytest.mark.parametrize("lora", [False, True])
def test_compile_fullgraph_and_cuda_graph_replay(lora):
    ex = _Experts(True).eval()
    if lora:
        ex = _lora_wrap(ex).eval()
    assert prepare_routed_experts(ex)
    T = 4
    # fp32 activations: compiled and eager then agree to rounding (bf16 output casts can flip an ulp).
    x = torch.randn(1, T, H, device = "cuda", dtype = torch.float32)
    idx, w = _routing(T, seed = 1)
    idx2, w2 = _routing(T, seed = 2)
    assert not torch.equal(idx, idx2)
    with torch.no_grad():
        ref1 = routed_experts_forward(ex, x, idx, w)
        ref2 = routed_experts_forward(ex, x, idx2, w2)
        torch._dynamo.reset()
        compiled = torch.compile(lambda a, b, c: routed_experts_forward(ex, a, b, c), fullgraph = True)
        torch.testing.assert_close(compiled(x, idx, w), ref1, rtol = 1e-5, atol = 1e-5)

        static = [x.clone(), idx.clone(), w.clone()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            routed_experts_forward(ex, *static)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = routed_experts_forward(ex, *static)
        static[1].copy_(idx2); static[2].copy_(w2)
        graph.replay()
        torch.testing.assert_close(out, ref2)
        static[1].copy_(idx); static[2].copy_(w)
        graph.replay()
        torch.testing.assert_close(out, ref1)


def test_lora_update_is_not_stale():
    # An optimizer step edits the adapters in place; the next routed call must see it.
    ex = _lora_wrap(_Experts(True)).eval()
    x = torch.randn(1, 2, H, device = "cuda", dtype = torch.float32)
    idx, w = _routing(2, seed = 3)
    with torch.no_grad():
        before = routed_experts_forward(ex, x, idx, w)
        for name, p in ex.named_parameters():
            if "lora_B" in name:
                p.add_(0.05)
        after = routed_experts_forward(ex, x, idx, w)
        ref = _reference(ex, x, idx, w)
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after.double(), ref, rtol = 1e-4, atol = 1e-4)


def test_lora_storage_swap_is_not_stale():
    # Module.to() swaps .data without bumping _version; the cached stacks must follow it.
    ex = _lora_wrap(_Experts(True)).eval()
    x = torch.randn(1, 2, H, device = "cuda", dtype = torch.float32)
    idx, w = _routing(2, seed = 4)
    with torch.no_grad():
        before = routed_experts_forward(ex, x, idx, w)
        for name, p in ex.named_parameters():
            if "lora_B" in name:
                p.data = p.data + 0.05
        after = routed_experts_forward(ex, x, idx, w)
        ref = _reference(ex, x, idx, w)
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after.double(), ref, rtol = 1e-4, atol = 1e-4)


def test_mixed_adapter_batch_is_left_to_peft():
    # PEFT's adapter_names pre-hook marks a mixed-adapter batch the routed kernels cannot honour.
    from functools import partial
    from peft.tuners.lora.model import _adapter_names_pre_forward_hook
    ex = _lora_wrap(_Experts(True)).eval()
    x = torch.randn(1, 1, H, device = "cuda", dtype = torch.float32)
    idx, w = _routing(1)
    with torch.no_grad():
        assert routed_experts_forward(ex, x, idx, w) is not None
        handles = [m.register_forward_pre_hook(partial(_adapter_names_pre_forward_hook, adapter_names = ["default"]), with_kwargs = True)
                   for m in ex.modules() if hasattr(m, "lora_A")]
        try:
            assert routed_experts_forward(ex, x, idx, w) is None
        finally:
            for h in handles:
                h.remove()


def test_lora_bias_is_left_to_peft():
    # lora_bias=True adds lora_B's bias, which the routed kernels do not apply.
    ex = _lora_wrap(_Experts(True), lora_bias = True).eval()
    for name, p in ex.named_parameters():
        if "lora_B" in name and name.endswith("bias"):
            p.data.fill_(0.5)
    x = torch.randn(1, 1, H, device = "cuda", dtype = torch.float32)
    idx, w = _routing(1)
    with torch.no_grad():
        assert routed_experts_forward(ex, x, idx, w) is None
