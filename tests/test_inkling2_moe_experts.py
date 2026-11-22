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

"""Inkling 2 experts (transformers >= 5.24) take `act_fn(gate, up)`, a softcapped SwiGLU, and accumulate in fp32: the
MoE backends must run the experts' own `_apply_gate`, not `act_fn(gate) * up`."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

from unsloth_zoo.temporary_patches.inkling2 import INKLING2_EXPERT_CLASSES, patch_inkling2_moe_experts  # noqa: E402
from unsloth_zoo.temporary_patches.moe_utils import forward_native_moe_loop  # noqa: E402


def _experts(loop, dtype):
    if loop:
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig as Cfg
        from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopExperts as Experts
    else:
        from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig as Cfg
        from transformers.models.inkling2.modeling_inkling2 import Inkling2Experts as Experts
    cfg = Cfg(hidden_size = 32, moe_intermediate_size = 24, n_routed_experts = 6, num_experts_per_tok = 2,
              num_hidden_layers = 6, mlp_gate_softcap = 1.5, mlp_up_softcap = 2.0)
    torch.manual_seed(0)
    experts = Experts(cfg)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.8)
        experts.down_proj.normal_(0, 0.3)
    return experts.to(dtype), cfg


def _reference(experts, cfg, x, idx, w):
    """Dense per-token sum of the routed experts, softcapped SwiGLU in fp32, fp32 accumulation."""
    out = torch.zeros(x.shape, dtype = torch.float32)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            gate, up = F.linear(x[t], experts.gate_up_proj[e]).chunk(2, dim = -1)
            g, u = gate.float(), up.float()
            h = (torch.tanh(g / cfg.mlp_gate_softcap) * cfg.mlp_gate_softcap * torch.sigmoid(g)
                 * torch.tanh(u / cfg.mlp_up_softcap) * cfg.mlp_up_softcap).to(x.dtype)
            out[t] += (F.linear(h, experts.down_proj[e]) * w[t, k]).float()
    return out.to(x.dtype)


def test_flags_declared_on_both_classes():
    import importlib

    patch_inkling2_moe_experts()
    for module_name, class_name in INKLING2_EXPERT_CLASSES:
        klass = getattr(importlib.import_module(module_name), class_name)
        assert klass._unsloth_own_apply_gate is True
        assert klass._unsloth_moe_fp32_reference is True
        # Training reads the frozen 4-bit stacks in bounded chunks (a whole Mini gate_up stack is 1.27 GiB).
        assert 0 < klass._unsloth_dequant_scratch_fraction < 0.1


@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_loop_matches_softcapped_reference(loop, dtype):
    patch_inkling2_moe_experts()
    experts, cfg = _experts(loop, dtype)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(10, 32, generator = g).to(dtype)
    idx = torch.stack([torch.randperm(6, generator = g)[:2] for _ in range(10)])
    w = torch.rand(10, 2, generator = g)  # the router returns fp32 weights
    got = forward_native_moe_loop(experts, x, idx, w)
    ref = _reference(experts, cfg, x, idx, w)
    assert got.dtype == dtype
    tol = dict(atol = 1e-5, rtol = 1e-5) if dtype == torch.float32 else dict(atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(got, ref, **tol)


def _nf4_experts(loop, dtype, h = 256, i = 192, e = 8):
    """Inkling 2 experts with stacked NF4 [E, N, K] weights, as Unsloth's 4-bit load leaves them."""
    bnb = pytest.importorskip("bitsandbytes")
    if loop:
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig as Cfg
        from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopExperts as Experts
    else:
        from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig as Cfg
        from transformers.models.inkling2.modeling_inkling2 import Inkling2Experts as Experts
    cfg = Cfg(hidden_size = h, moe_intermediate_size = i, n_routed_experts = e, num_experts_per_tok = 2,
              num_hidden_layers = 6, mlp_gate_softcap = 1.5, mlp_up_softcap = 2.0)
    experts = Experts(cfg).eval()
    g = torch.Generator().manual_seed(0)
    for name, shape in (("gate_up_proj", (e, 2 * i, h)), ("down_proj", (e, h, i))):
        w = torch.randn(shape, generator = g) * 0.08
        p = bnb.nn.Params4bit(w.to(dtype), requires_grad = False, compress_statistics = True, quant_type = "nf4",
                              blocksize = 64).to("cuda")
        p._original_shape = shape
        setattr(experts, name, p)
    return experts.to("cuda"), cfg


def _dense_nf4(p):
    import bitsandbytes as bnb

    return bnb.functional.dequantize_4bit(p.data, p.quant_state).view(p._original_shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("use_lora", [False, True])
def test_nf4_decode_routes_only_the_routed_experts(loop, T, use_lora, monkeypatch):
    # Decode used to decline own-gate experts and dequantize all experts of every layer per token (real Mini 4-bit on an
    # A100: 2.28 tok/s vs 3.09 for transformers bf16). The routed path must engage and match the dense softcapped
    # reference on the same dequantized weights, as the current path does.
    pytest.importorskip("triton")
    from unsloth_zoo.temporary_patches import moe_routed as MR
    from unsloth_zoo.temporary_patches import moe_utils as MU
    from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit

    patch_inkling2_moe_experts()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    experts, cfg = _nf4_experts(loop, dtype)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(T, 256, generator = g).to(dtype).cuda()
    idx = torch.stack([torch.randperm(8, generator = g)[:2] for _ in range(T)]).cuda()
    w = torch.rand(T, 2, generator = g).cuda()
    lora = None
    if use_lora:
        e, r = 8, 4
        lora = ((torch.randn(e, 256, r, generator = g) * 0.1).cuda(), (torch.randn(e, r, 384, generator = g) * 0.1).cuda(), 0.5, e)
        lora = (lora, ((torch.randn(e, 192, r, generator = g) * 0.1).cuda(), (torch.randn(e, r, 256, generator = g) * 0.1).cuda(), 2.0, e))

    def stash():
        for n, t in zip(("gate_up_proj", "down_proj"), lora or (None, None)):
            if t is None:
                experts.__dict__.pop(MU.moe_lora_stash_name(n), None)
            else:
                setattr(experts, MU.moe_lora_stash_name(n), t)

    calls = []
    real = MR._nf4_own_gate
    monkeypatch.setattr(MR, "_nf4_own_gate", lambda *a, **k: calls.append(1) or real(*a, **k))
    with torch.no_grad():
        stash()
        got = forward_moe_backend_bnb4bit(experts, x, idx, w)
        assert calls, "routed own-gate path did not engage"
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "0")
        stash()
        cur = forward_moe_backend_bnb4bit(experts, x, idx, w)
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1")
    assert len(calls) == 1 and got.dtype == dtype and got.shape == x.shape
    # fp64 reference on the dequantized weights, the experts' softcapped SwiGLU and fp32 accumulation
    Wg, Wd = _dense_nf4(experts.gate_up_proj).double(), _dense_nf4(experts.down_proj).double()
    ref = torch.zeros(T, 256, dtype = torch.float64, device = "cuda")
    for t in range(T):
        for k in range(2):
            e = int(idx[t, k])
            gu = Wg[e] @ x[t].double()
            if lora is not None:
                f, s, sc, _ = lora[0]
                gu = gu + (x[t].double() @ f[e].double() @ s[e].double()) * sc
            gate, up = gu.chunk(2)
            inter = (torch.tanh(gate / 1.5) * 1.5 * torch.sigmoid(gate) * torch.tanh(up / 2.0) * 2.0)
            d = Wd[e] @ inter
            if lora is not None:
                f, s, sc, _ = lora[1]
                d = d + (inter @ f[e].double() @ s[e].double()) * sc
            ref[t] += d * float(w[t, k])
    err_got = (got.double() - ref).abs().mean().item()
    err_cur = (cur.double() - ref).abs().mean().item()
    assert err_got <= 1.25 * err_cur + 1e-6, (err_got, err_cur)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_nf4_own_gate_declines_large_calls_and_training(monkeypatch):
    pytest.importorskip("triton")
    from unsloth_zoo.temporary_patches import moe_routed as MR

    patch_inkling2_moe_experts()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    experts, _ = _nf4_experts(True, dtype)
    monkeypatch.setattr(MR, "NF4_MAX_SLOTS", None)
    monkeypatch.setattr(MR, "NF4_SLOTS_PER_EXPERT", 64.0)
    T = MR.OWN_GATE_MAX_SLOTS // 2 + 1
    x = torch.randn(T, 256, device = "cuda").to(dtype)
    idx = torch.randint(0, 8, (T, 2), device = "cuda")
    w = torch.rand(T, 2, device = "cuda")
    with torch.no_grad():
        assert MR.routed_moe_forward(experts, x, idx, w) is None
        assert MR.routed_moe_forward(experts, x[:1], idx[:1], w[:1]) is not None
        experts.train()
        assert MR.routed_moe_forward(experts, x[:1], idx[:1], w[:1]) is None
