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

"""Native MXFP4 expert LoRA runs its grouped GEMMs in the adapter's dtype. matmul_ogs rows are bf16, but T4 trains in
fp16, so fp16 adapters met bf16 inputs and grouped_mm raised "expected mat1 and mat2 to have the same dtype". CPU
only: the triton_kernels matmuls are replaced by bf16 stand-ins."""

import types

import pytest
import torch

from unsloth_zoo.temporary_patches import gpt_oss as go


@pytest.mark.parametrize("adapter_dtype", [torch.float16, torch.bfloat16])
def test_lora_grouped_mm_gets_the_adapter_dtype(monkeypatch, adapter_dtype):
    E, H, I, T, k, r = 2, 8, 8, 3, 2, 4
    seen = []

    def grouped_mm(x, w, offsets):
        seen.append((x.dtype, w.dtype))
        assert x.dtype == w.dtype
        return x @ w[0]

    monkeypatch.setattr(go, "native_moe_grouped_mm", grouped_mm)
    monkeypatch.setattr(go, "_ogs_lora_offsets", lambda routing_data: None)
    monkeypatch.setattr(go._OgsGateUp, "apply", staticmethod(lambda x, *a: torch.zeros(T * k, 2 * I, dtype = torch.bfloat16)))
    monkeypatch.setattr(go._OgsSwiglu, "apply", staticmethod(lambda pre, *a: pre[..., ::2].clone()))
    monkeypatch.setattr(go._OgsDown, "apply", staticmethod(lambda h, *a: torch.zeros(T, H, dtype = torch.bfloat16)))
    routing = types.SimpleNamespace(n_expts_act = k, gate_scal = torch.ones(T * k))
    gather = types.SimpleNamespace(src_indx = torch.arange(T * k))
    scatter = types.SimpleNamespace(dst_indx = torch.arange(T * k))
    lora = lambda i, o: (torch.randn(E, i, r, dtype = adapter_dtype), torch.randn(E, r, o, dtype = adapter_dtype), 2.0, E)
    module = types.SimpleNamespace(alpha = 1.702, limit = 7.0, gate_up_proj_bias = None, down_proj_bias = None)
    out = go.mxfp4_ogs_experts_forward(
        module, torch.randn(T, H, dtype = adapter_dtype), routing, gather, scatter,
        gate_up_lora = lora(H, 2 * I), down_lora = lora(I, H),
    )
    assert out.shape == (T, H) and len(seen) == 4
    assert all(pair == (adapter_dtype, adapter_dtype) for pair in seen)
