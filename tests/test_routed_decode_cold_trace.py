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

"""Decode arch gates without a GPU (capability faked): gpt-oss 16-bit routed eligibility must trace under fullgraph in a
cold process, as torch.cuda.device_count() is not traceable; GDN ReplaySSM shrinks nk for small key dims instead of
rejecting BKT < 16."""

import os
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
pytest.importorskip("triton")

from unsloth_zoo.temporary_patches import moe_routed as MR  # noqa: E402
from unsloth_zoo.temporary_patches import gpt_oss_routed as GR  # noqa: E402
from unsloth_zoo._vendored.vllm import gdn_replayssm_decode as GDN  # noqa: E402


@pytest.fixture
def fake_cap(monkeypatch):
    def set_cap(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)
        MR._bf16_supported_index.cache_clear()
        GDN._REPLAYSSM_CONFIG_BY_DEVICE.clear()
        monkeypatch.setattr(MR, "_SUB_SM80_DEVICES", None)
    yield set_cap
    MR._bf16_supported_index.cache_clear()
    GDN._REPLAYSSM_CONFIG_BY_DEVICE.clear()


@pytest.mark.parametrize("dynamic", [False, True], ids = ["static", "dynamic"])
@pytest.mark.parametrize("cap,T,want", [((7, 5), 1, True), ((7, 5), 4, False), ((9, 0), 4, True)],
                         ids = ["sm75-1tok", "sm75-4tok", "sm90-4tok"])
def test_gpt_oss_fp16_eligibility_cold_fullgraph(cap, T, want, dynamic, fake_cap, monkeypatch):
    fake_cap(cap)
    monkeypatch.delenv("UNSLOTH_GPTOSS_ROUTED_KERNEL", raising = False)
    monkeypatch.delenv("UNSLOTH_GPTOSS_ROUTED_INFERENCE", raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_ROUTED_KERNEL", raising = False)
    torch._dynamo.reset()
    with FakeTensorMode(), torch.no_grad():
        dev = torch.device("cuda", 0)
        experts = types.SimpleNamespace(
            gate_up_proj = torch.empty(8, 128, 256, device = dev, dtype = torch.float16),
            down_proj = torch.empty(8, 128, 128, device = dev, dtype = torch.float16),
        )
        x = torch.empty(1, T, 128, device = dev, dtype = torch.float16)
        fn = torch.compile(lambda h: h + float(GR.routed_bf16_eligible(experts, h)),
                           backend = "eager", fullgraph = True, dynamic = dynamic)
        y = fn(x)
    assert MR._SUB_SM80_DEVICES is None  # traced without listing the devices
    assert y.shape == x.shape
    torch._dynamo.reset()
    # The eager verdict is the T4 one-token rule.
    with FakeTensorMode(), torch.no_grad():
        dev = torch.device("cuda", 0)
        experts = types.SimpleNamespace(
            gate_up_proj = torch.empty(8, 128, 256, device = dev, dtype = torch.float16),
            down_proj = torch.empty(8, 128, 128, device = dev, dtype = torch.float16),
        )
        assert GR.routed_bf16_eligible(experts, torch.empty(1, T, 128, device = dev, dtype = torch.float16)) is want


class _Recorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda **kw: self.launches.append((kw["NK"], kw["BKT"]))


def _gdn_cpu_inputs(K, B = 1, H_ = 2, HV = 4, V = 128, L = 16):
    S = B + 1
    z = lambda *s, dt = torch.float16: torch.zeros(*s, dtype = dt)
    return dict(
        mixed_qkv = z(B, 2 * H_ * K + HV * V), a = z(B, HV), b = z(B, HV), A_log = z(HV, dt = torch.float32),
        dt_bias = z(HV, dt = torch.float32), scale = K ** -0.5, initial_state = z(S, HV, V, K),
        d_cache = z(S, HV, L, V), k_cache = z(S, H_, L, K), g_cache = z(S, HV, L, dt = torch.float32),
        out = z(B, HV, V), ssm_state_indices = torch.ones(B, dtype = torch.int32),
        write_pos = torch.zeros(B, dtype = torch.int32), is_flush = torch.zeros(B, dtype = torch.int8),
        use_qk_l2norm_in_kernel = True,
    )


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (10, 0)], ids = lambda c: "sm%d%d" % c)
@pytest.mark.parametrize("K", [16, 32, 64, 128])
def test_gdn_replayssm_small_key_dim_launches(cap, K, fake_cap, monkeypatch):
    fake_cap(cap)
    rec = _Recorder()
    monkeypatch.setattr(GDN, "fused_recurrent_gated_delta_rule_replayssm_kernel", rec)
    GDN.fused_recurrent_gated_delta_rule_replayssm(**_gdn_cpu_inputs(K))
    table_nk = GDN._gdn_decode_config(cap)[3]
    nk = min(table_nk, K // 16)
    assert rec.launches == [(nk, K // nk)]
    # K = 128 (Qwen3-Next and the usual GDN configs) keeps the table launch unchanged.
    if K == 128:
        assert nk == table_nk


def test_gdn_replayssm_explicit_nk_still_validated(fake_cap, monkeypatch):
    fake_cap((7, 5))
    monkeypatch.setattr(GDN, "fused_recurrent_gated_delta_rule_replayssm_kernel", _Recorder())
    with pytest.raises(ValueError, match = "BKT=8"):
        GDN.fused_recurrent_gated_delta_rule_replayssm(**_gdn_cpu_inputs(32), nk = 4)
