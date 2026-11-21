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
