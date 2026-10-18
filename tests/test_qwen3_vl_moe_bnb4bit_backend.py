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

"""Qwen3-VL-MoE bnb 4-bit experts on the Triton / loop backends dequantize first: both index a packed Params4bit."""
import types

import pytest
import torch

Q = pytest.importorskip("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
if not hasattr(Q, "Qwen3VLMoeTextExperts"):
    pytest.skip("Qwen3VLMoeTextExperts needs transformers 5", allow_module_level = True)

import unsloth_zoo.temporary_patches.qwen3_vl_moe as V


def _experts():
    config = types.SimpleNamespace(num_experts = 4, hidden_size = 8, moe_intermediate_size = 4, hidden_act = "silu")
    return Q.Qwen3VLMoeTextExperts(config)


@pytest.mark.parametrize("backend", ["unsloth_triton", "native_torch"])
@pytest.mark.parametrize("bnb4bit", [True, False])
def test_bnb4bit_experts_dequantize_before_triton_or_loop(monkeypatch, backend, bnb4bit):
    original_forward = Q.Qwen3VLMoeTextExperts.forward
    original_init = Q.Qwen3VLMoeTextExperts.__init__
    original_block_forward = Q.Qwen3VLMoeTextSparseMoeBlock.forward
    calls = []

    def stack_reader(name):
        def fn(self, *args):
            calls.append(name)
            return name
        return fn

    monkeypatch.setattr(V, "select_moe_backend", lambda: backend)
    monkeypatch.setattr(V, "forward_triton_grouped_gemm", stack_reader("unsloth_triton"))
    monkeypatch.setattr(V, "forward_native_moe_loop", stack_reader("native_torch"))
    monkeypatch.setattr(V, "_moe_uses_bnb4bit_expert_weights", lambda self: bnb4bit)
    monkeypatch.setattr(V, "forward_moe_backend_bnb4bit", stack_reader("bnb4bit"))
    try:
        V.patch_qwen3_vl_moe()
        x = torch.zeros(2, 8)
        out = _experts().forward(x, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1))
    finally:
        Q.Qwen3VLMoeTextExperts.forward = original_forward
        Q.Qwen3VLMoeTextExperts.__init__ = original_init
        Q.Qwen3VLMoeTextSparseMoeBlock.forward = original_block_forward
    assert out == ("bnb4bit" if bnb4bit else backend)
    assert calls == [out]
