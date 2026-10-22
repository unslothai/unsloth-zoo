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

"""Routed gpt-oss experts read only the routed experts: poisoning every other expert's
weights, NF4 bytes / absmax and biases (NaN) leaves the output bit-identical, eager and
under torch.compile fullgraph, while changing a routed expert changes it."""
import os

import pytest
import torch

# T4 (sm75) has no bf16: run the same checks in fp16 there. UNSLOTH_TEST_DTYPE=float16 simulates it.
DT = getattr(torch, os.environ.get("UNSLOTH_TEST_DTYPE", "")) if os.environ.get("UNSLOTH_TEST_DTYPE") else (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)

from unsloth_zoo.temporary_patches import gpt_oss_routed as R

E, TOP_K, H, I = 16, 4, 256, 192


def _routes(T, seed):
    g = torch.Generator().manual_seed(seed)
    vals, idx = torch.randn(T, E, generator = g).topk(TOP_K, -1)
    return idx.cuda(), vals.softmax(-1).cuda()


def _unrouted(idx):
    used = set(idx.flatten().tolist())
    assert len(used) < E
    return [e for e in range(E) if e not in used]


def _maybe_compile(fn, compiled):
    if not compiled:
        return fn
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph = True)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("T", [1, 3])
def test_bf16_reads_only_routed_experts(compiled, T):
    g = torch.Generator(device = "cuda").manual_seed(T)
    ex = torch.nn.Module()
    ex.gate_up_proj = torch.randn(E, H, 2 * I, device = "cuda", generator = g).to(DT) / H ** 0.5
    ex.down_proj = torch.randn(E, I, H, device = "cuda", generator = g).to(DT) / I ** 0.5
    ex.gate_up_proj_bias = torch.randn(E, 2 * I, device = "cuda", generator = g).to(DT) * 0.1
    ex.down_proj_bias = torch.randn(E, H, device = "cuda", generator = g).to(DT) * 0.1
    ex.hidden_size, ex.alpha, ex.limit = H, 1.702, 7.0
    x = torch.randn(1, T, H, device = "cuda").to(DT)
    idx, w = _routes(T, T)
    fn = _maybe_compile(R.routed_bf16_forward, compiled)
    with torch.no_grad():
        before = fn(ex, x, idx, w).clone()
        un = _unrouted(idx)
        for t in (ex.gate_up_proj, ex.down_proj, ex.gate_up_proj_bias, ex.down_proj_bias):
            t[un] = float("nan")
        after = fn(ex, x, idx, w)
        assert torch.isfinite(before).all()
        assert torch.equal(before, after)
        ex.gate_up_proj[idx[0, 0]] += 1.0
        assert not torch.equal(after, fn(ex, x, idx, w))


def _nf4_experts(nested, lora):
    bnb = pytest.importorskip("bitsandbytes")

    def lin(i, o, seed):
        g = torch.Generator().manual_seed(seed)
        m = bnb.nn.Linear4bit(i, o, bias = True, compute_dtype = DT, quant_type = "nf4",
                              compress_statistics = nested)
        m.weight = bnb.nn.Params4bit(torch.randn(o, i, generator = g) * 0.02 * (1 + seed % 5), requires_grad = False,
                                     quant_type = "nf4", compress_statistics = nested)
        m.bias = torch.nn.Parameter(torch.randn(o, generator = g) * 0.1, requires_grad = False)
        return m.cuda()

    ex = torch.nn.Module()
    ex.gate_up_projs = torch.nn.ModuleList([lin(H, 2 * I, e) for e in range(E)])
    ex.down_projs = torch.nn.ModuleList([lin(I, H, 100 + e) for e in range(E)])
    ex.hidden_size, ex.alpha, ex.limit = H, 1.702, 7.0
    if lora:
        peft = pytest.importorskip("peft")
        ex = peft.inject_adapter_in_model(peft.LoraConfig(r = 8, lora_alpha = 16, target_modules = r".*(gate_up_projs|down_projs)\.\d+"), ex)
        g = torch.Generator().manual_seed(3)
        for n, p in ex.named_parameters():
            if "lora_" in n:
                p.data = (torch.randn(p.shape, generator = g) * 0.05).to(p.device, p.dtype)
    return ex.eval()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("lora", [False, True])
def test_nf4_reads_only_routed_experts(compiled, nested, lora):
    ex = _nf4_experts(nested, lora)
    state = R.prepare_routed_experts(ex)
    assert state
    T = 3
    x = torch.randn(1, T, H, device = "cuda").to(DT)
    idx, w = _routes(T, 7)
    dense = torch.zeros(T, E, device = "cuda").scatter_(1, idx, w).to(DT)
    fn = _maybe_compile(lambda a, b, c: R.routed_experts_forward(ex, a, b, c), compiled)
    with torch.no_grad():
        before = fn(x, idx, dense).clone()
        un = _unrouted(idx)
        for e in un:
            for proj in (ex.gate_up_projs[e], ex.down_projs[e]):
                base = getattr(proj, "base_layer", proj)
                base.weight.data.copy_(torch.randint(0, 256, base.weight.data.shape, device = "cuda", dtype = torch.uint8))
                qs = base.weight.quant_state
                if nested:
                    qs.absmax.copy_(torch.randint(0, 256, qs.absmax.shape, device = "cuda", dtype = torch.uint8))
                    qs.state2.absmax.fill_(float("nan"))
                else:
                    qs.absmax.fill_(float("nan"))
                base.bias.fill_(float("nan"))
                if lora:
                    proj.lora_A["default"].weight.fill_(float("nan"))
                    proj.lora_B["default"].weight.fill_(float("nan"))
        assert R.prepare_routed_experts(ex) is state  # same buffers: the tables stay valid
        after = fn(x, idx, dense)
        assert torch.isfinite(before).all()
        assert torch.equal(before, after)
        base = getattr(ex.gate_up_projs[idx[0, 0]], "base_layer", ex.gate_up_projs[idx[0, 0]])
        base.weight.data.bitwise_xor_(0x11)
        assert not torch.equal(after, fn(x, idx, dense))
