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

"""Per-arch decode routing: which kernel a T4 (sm75) and sm80+ cards take, with the capability faked.

Measured on Colab T4 fp16 (torch 2.10 and 2.14): from 4 decode tokens up the 16-bit routed MoE kernels are 4-9x
slower than the grouped_mm path they replace, so sm75 routes one-token decode only (there grouped_mm's host syncs
cost more end to end); stacked NF4 routing wins or ties at every routed size and keeps its sm80+ limits.
"""

import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "routed decode kernels are CUDA only")
pytest.importorskip("triton")

from unsloth_zoo.temporary_patches import moe_routed as MR  # noqa: E402
from unsloth_zoo.temporary_patches import gpt_oss_routed as GR  # noqa: E402

E, H, I, TOP_K = 8, 256, 192, 2
# T4, A100, A10 / RTX 30, L4 / RTX 40, H100, B200, RTX PRO 6000 (G4)
CAPS = [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
REAL_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@pytest.fixture
def fake_cap(monkeypatch):
    def set_cap(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)
        MR._bf16_supported_index.cache_clear()
        monkeypatch.setattr(MR, "_SUB_SM80_DEVICES", None, raising = False)
    yield set_cap
    MR._bf16_supported_index.cache_clear()


def _dtypes():
    return [torch.float16, torch.bfloat16] if REAL_BF16 else [torch.float16]


class Qwen3MoeExperts(nn.Module):
    # HF layout: gate_up [E, 2I, H], down [E, H, I], SiLU.
    def __init__(self, dtype):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.num_experts, self.hidden_dim, self.intermediate_dim = E, H, I
        self.act_fn = nn.SiLU()
        self.is_concatenated = True
        self.gate_up_proj = nn.Parameter((torch.randn(E, 2 * I, H, generator = g) * 0.05).to(dtype).cuda(), requires_grad = False)
        self.down_proj = nn.Parameter((torch.randn(E, H, I, generator = g) * 0.05).to(dtype).cuda(), requires_grad = False)


class GptOssExperts(nn.Module):
    # gpt-oss layout: gate_up [E, H, 2I] (interleaved), down [E, I, H], biases.
    def __init__(self, dtype):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.num_experts, self.hidden_size, self.expert_dim = E, H, I
        self.alpha, self.limit = 1.702, 7.0
        self.gate_up_proj = nn.Parameter((torch.randn(E, H, 2 * I, generator = g) * 0.05).to(dtype).cuda(), requires_grad = False)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, 2 * I, dtype = dtype, device = "cuda"), requires_grad = False)
        self.down_proj = nn.Parameter((torch.randn(E, I, H, generator = g) * 0.05).to(dtype).cuda(), requires_grad = False)
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H, dtype = dtype, device = "cuda"), requires_grad = False)


def _route(T, dtype):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(T, H, generator = g).to(dtype).cuda()
    idx = torch.stack([torch.randperm(E, generator = g)[:TOP_K] for _ in range(T)]).cuda()
    w = torch.softmax(torch.randn(T, TOP_K, generator = g), -1).to(dtype).cuda()
    return x, idx, w


@pytest.mark.parametrize("cap", CAPS, ids = lambda c: "sm%d%d" % c)
@pytest.mark.parametrize("dtype", _dtypes(), ids = str)
@pytest.mark.parametrize("T", [1, 4])
def test_16bit_routed_moe_t4_routes_one_token_only(cap, dtype, T, fake_cap, monkeypatch):
    fake_cap(cap)
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1")
    calls = []
    real = MR.routed_bf16_moe
    monkeypatch.setattr(MR, "routed_bf16_moe", lambda *a, **k: calls.append(1) or real(*a, **k))
    ex = Qwen3MoeExperts(dtype).eval()
    x, idx, w = _route(T, dtype)
    with torch.no_grad():
        out = MR.routed_moe_forward(ex, x, idx, w)
    # bf16 never routes below sm80 (no bf16 tensor cores); fp16 routes one-token decode there.
    routed = cap[0] >= 8 or (T == 1 and dtype == torch.float16)
    assert (out is not None) == routed, f"sm{cap[0]}{cap[1]} {dtype}: routed={out is not None}, want {routed}"
    assert bool(calls) == routed


@pytest.mark.parametrize("cap", CAPS, ids = lambda c: "sm%d%d" % c)
@pytest.mark.parametrize("dtype", _dtypes(), ids = str)
@pytest.mark.parametrize("T", [1, 4])
def test_gpt_oss_16bit_routed_eligible_t4_one_token_only(cap, dtype, T, fake_cap):
    fake_cap(cap)
    ex = GptOssExperts(dtype).eval()
    x = torch.randn(1, T, H, dtype = dtype, device = "cuda")
    with torch.no_grad():
        assert GR.routed_bf16_eligible(ex, x) == (cap[0] >= 8 or (T == 1 and dtype == torch.float16))


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (10, 0)], ids = lambda c: "sm%d%d" % c)
def test_stacked_nf4_routing_is_kept_on_t4(cap, fake_cap, monkeypatch):
    # NF4 routed decode wins or ties the dequant + grouped_mm fallback on T4 too: no arch gate.
    bnb = pytest.importorskip("bitsandbytes")
    fake_cap(cap)
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1")
    monkeypatch.setattr(MR, "NF4_MAX_SLOTS", None)
    dtype = torch.float16
    ex = Qwen3MoeExperts(dtype)
    for name in ("gate_up_proj", "down_proj"):
        w = getattr(ex, name).detach()
        p = bnb.nn.Params4bit(w.cpu(), requires_grad = False, compress_statistics = True, quant_type = "nf4",
                              blocksize = 64).to("cuda")
        p._original_shape = tuple(w.shape)
        setattr(ex, name, p)
    ex.eval()
    x, idx, w = _route(1, dtype)
    with torch.no_grad():
        assert MR.routed_moe_forward(ex, x, idx, w) is not None


# ---------------------------------------------------------------- GDN ReplaySSM decode launch config
from unsloth_zoo._vendored.vllm import gdn_replayssm_decode as GDN  # noqa: E402

# Upstream (vllm-project/vllm#48792) table, unchanged off T4; T4 measured (Colab T4, torch 2.10 / 2.14, B=1/8/64):
# (64, 1, 3, 2) 0.24 / 1.46 / 12.3 ms vs (32, 1, 1, 4) 0.11 / 0.11 / 0.52 ms (fla fused_recurrent 0.18 / 0.28 / 1.23).
GDN_EXPECTED = {
    (7, 5): (32, 1, 1, 4),
    (8, 0): (64, 1, 3, 2), (8, 6): (64, 1, 3, 2), (8, 9): (64, 1, 3, 2), (9, 0): (64, 1, 3, 2),
    (10, 0): (128, 1, 3, 4), (12, 0): (128, 1, 3, 4),
}


@pytest.fixture
def fake_gdn_cap(monkeypatch):
    def set_cap(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)
        getattr(GDN, "_REPLAYSSM_CONFIG_BY_DEVICE", {}).clear()
    yield set_cap
    getattr(GDN, "_REPLAYSSM_CONFIG_BY_DEVICE", {}).clear()


@pytest.mark.parametrize("cap", CAPS, ids = lambda c: "sm%d%d" % c)
def test_gdn_replayssm_config_per_arch(cap, fake_gdn_cap):
    fake_gdn_cap(cap)
    assert tuple(GDN.get_replayssm_config("gdn_decode", L = 16)) == GDN_EXPECTED[cap]


def _gdn_inputs(B, dtype, L = 16, H_ = 2, HV = 4, K = 128, V = 128, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    S = B + 1
    r = lambda *s: torch.randn(*s, device = "cuda", generator = g)
    return dict(
        mixed_qkv = r(B, 2 * H_ * K + HV * V).to(dtype), a = r(B, HV).to(dtype), b = r(B, HV).to(dtype),
        A_log = torch.log(torch.linspace(1, 16, HV, device = "cuda")), dt_bias = torch.ones(HV, device = "cuda"),
        scale = K ** -0.5, initial_state = (r(S, HV, V, K) * 0.1).to(dtype),
        d_cache = (r(S, HV, L, V) * 0.1).to(dtype), k_cache = (r(S, H_, L, K) * 0.1).to(dtype),
        g_cache = -torch.rand(S, HV, L, device = "cuda", generator = g) * 0.1,
        out = torch.empty(B, HV, V, device = "cuda", dtype = dtype),
        ssm_state_indices = torch.arange(1, B + 1, device = "cuda", dtype = torch.int32),
        write_pos = torch.arange(B, device = "cuda", dtype = torch.int32) % L,
        is_flush = (torch.arange(B, device = "cuda") % L == L - 1).to(torch.int8),
        use_qk_l2norm_in_kernel = True,
    )


@pytest.mark.parametrize("B", [1, 8, 33])
def test_gdn_replayssm_t4_config_matches_upstream_config(B, fake_gdn_cap, monkeypatch):
    # The T4 entry is only a launch shape: same outputs and checkpoint as the upstream config.
    calls = []
    real = GDN.get_replayssm_config
    monkeypatch.setattr(GDN, "get_replayssm_config", lambda *a, **k: calls.append(real(*a, **k)) or calls[-1])
    got = {}
    for cap in [(7, 5), (8, 0)]:
        fake_gdn_cap(cap)
        kw = _gdn_inputs(B, torch.float16)
        GDN.fused_recurrent_gated_delta_rule_replayssm(**kw)
        torch.cuda.synchronize()
        got[cap] = (kw["out"].float(), kw["initial_state"].float())
    assert calls == [GDN_EXPECTED[(7, 5)], GDN_EXPECTED[(8, 0)]]
    for a, b in zip(got[(7, 5)], got[(8, 0)]):
        torch.testing.assert_close(a, b, atol = 2e-3, rtol = 2e-3)
