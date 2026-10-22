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

"""The second grouped GEMM runs permute_y=True, so its output is in token order
while the down-LoRA delta stays expert-sorted: without scattering the delta
through gather_indices the LoRA rows land on the wrong tokens.
"""

import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_utils
from unsloth_zoo.temporary_patches.moe_utils import forward_triton_grouped_gemm


def _reference_grouped_gemm(
    X=None,
    W=None,
    m_sizes=None,
    topk=None,
    gather_indices=None,
    permute_x=False,
    permute_y=False,
    **kwargs,
):
    """Pure-torch grouped GEMM implementing the documented permute semantics:
    ``permute_x`` gathers token-order rows into expert-sorted order, ``permute_y``
    scatters the expert-sorted output back to token order.
    """
    if permute_x:
        X = X[gather_indices // topk]
    outputs, start = [], 0
    for expert_idx, size in enumerate(m_sizes.tolist()):
        end = start + int(size)
        outputs.append(X[start:end] @ W[expert_idx].T)
        start = end
    Y = torch.cat(outputs, dim=0)
    if permute_y:
        Y = Y[torch.argsort(gather_indices)]
    return Y


def _install_fake_unsloth_kernels(monkeypatch):
    parent = None
    for name in (
        "unsloth",
        "unsloth.kernels",
        "unsloth.kernels.moe",
        "unsloth.kernels.moe.grouped_gemm",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            monkeypatch.setattr(parent, name.rsplit(".", 1)[1], module, raising=False)
        parent = module

    interface = types.ModuleType("unsloth.kernels.moe.grouped_gemm.interface")
    interface.grouped_gemm = _reference_grouped_gemm
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setattr(parent, "interface", interface, raising=False)

    autotune_cache = types.ModuleType("unsloth.kernels.moe.autotune_cache")
    autotune_cache.get_or_autotune_moe_kernels = lambda **kw: (None, None, None)
    monkeypatch.setitem(sys.modules, autotune_cache.__name__, autotune_cache)
    monkeypatch.setattr(
        sys.modules["unsloth.kernels.moe"], "autotune_cache", autotune_cache,
        raising=False,
    )


def _build_experts(num_experts, hidden, intermediate):
    experts = nn.Module()
    experts.num_experts = num_experts
    experts.gate_up_proj = nn.Parameter(
        torch.randn(num_experts, 2 * intermediate, hidden)
    )
    experts.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
    experts.act_fn = F.silu
    # Pre-set so the forward skips the CUDA autotune / empty_cache branch.
    experts._unsloth_moe_configs = (intermediate, (None, None, None), (None, None, None))
    return experts


def _reference_forward(experts, X, top_k_index, top_k_weights, first, second, scaling):
    num_tokens, hidden = X.shape
    top_k = top_k_index.shape[1]
    out = torch.zeros(num_tokens, hidden)
    for t in range(num_tokens):
        for k in range(top_k):
            e = int(top_k_index[t, k])
            gu = X[t] @ experts.gate_up_proj[e].T
            gate, up = gu.chunk(2, dim=-1)
            h = F.silu(gate) * up
            d = h @ experts.down_proj[e].T + ((h @ first[e]) @ second[e]) * scaling
            out = out + torch.zeros(num_tokens, hidden).index_put_(
                (torch.tensor(t),), top_k_weights[t, k] * d
            )
    return out


def _setup(monkeypatch):
    _install_fake_unsloth_kernels(monkeypatch)
    monkeypatch.setattr(moe_utils, "native_moe_grouped_mm", moe_utils._manual_grouped_mm)

    num_experts, hidden, intermediate, rank = 4, 16, 12, 4
    num_tokens, top_k = 8, 2

    torch.manual_seed(0)
    experts = _build_experts(num_experts, hidden, intermediate)
    first = torch.randn(num_experts, intermediate, rank)
    second = torch.randn(num_experts, rank, hidden)
    scaling = 0.5

    X = torch.randn(num_tokens, hidden)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    # Guard: the routing must actually permute rows, else the bug can't bite.
    _, gather_indices = moe_utils._get_routing_indices(top_k_index, num_experts)
    assert not torch.equal(gather_indices, torch.arange(num_tokens * top_k))

    return experts, first, second, scaling, X, top_k_index, top_k_weights


def test_down_lora_forward_matches_reference(monkeypatch):
    experts, first, second, scaling, X, top_k_index, top_k_weights = _setup(monkeypatch)
    experts._unsloth_lora_down_proj = (first, second, scaling)

    out = forward_triton_grouped_gemm(experts, X, top_k_index, top_k_weights)
    ref = _reference_forward(
        experts, X, top_k_index, top_k_weights, first, second, scaling
    )

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_down_lora_grads_match_reference(monkeypatch):
    experts, first, second, scaling, X, top_k_index, top_k_weights = _setup(monkeypatch)
    first = first.requires_grad_(True)
    second = second.requires_grad_(True)
    experts._unsloth_lora_down_proj = (first, second, scaling)

    first_ref = first.detach().clone().requires_grad_(True)
    second_ref = second.detach().clone().requires_grad_(True)

    out = forward_triton_grouped_gemm(experts, X, top_k_index, top_k_weights)
    ref = _reference_forward(
        experts, X, top_k_index, top_k_weights, first_ref, second_ref, scaling
    )

    out.sum().backward()
    ref.sum().backward()

    torch.testing.assert_close(first.grad, first_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(second.grad, second_ref.grad, rtol=1e-4, atol=1e-4)


def test_down_lora_promotes_a_wider_scaling_tensor(monkeypatch):
    """A non-0-dim `scaling` promotes the delta past the output dtype, which index_add_ rejects."""
    experts, first, second, _, X, top_k_index, top_k_weights = _setup(monkeypatch)

    experts._unsloth_lora_down_proj = (first, second, 0.5)
    baseline = forward_triton_grouped_gemm(experts, X, top_k_index, top_k_weights)

    experts._unsloth_lora_down_proj = (first, second, torch.tensor([0.5], dtype=torch.float64))
    promoted = forward_triton_grouped_gemm(experts, X, top_k_index, top_k_weights)

    assert promoted.dtype == torch.float64
    torch.testing.assert_close(promoted.float(), baseline, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_down_lora_matches_reference_on_real_kernels():
    """Same check against the real kernel, whose permute semantics the tests above assume."""
    pytest.importorskip("unsloth.kernels.moe.grouped_gemm.interface")
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )

    num_experts, hidden, intermediate, rank = 8, 256, 512, 16
    num_tokens, top_k = 512, 2
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)

    experts = nn.Module()
    experts.num_experts = num_experts
    experts.act_fn = F.silu
    experts.gate_up_proj = nn.Parameter(
        (torch.randn(num_experts, 2 * intermediate, hidden, device=device) * 0.2).to(dtype)
    )
    experts.down_proj = nn.Parameter(
        (torch.randn(num_experts, hidden, intermediate, device=device) * 0.2).to(dtype)
    )
    # Reduction dims must stay multiples of the 32-wide K block the kernel asserts on.
    configs = lambda: (KernelConfigForward(), KernelConfigBackward_dX(), KernelConfigBackward_dW())
    experts._unsloth_moe_configs = (intermediate, configs(), configs())

    first = (torch.randn(num_experts, intermediate, rank, device=device) * 0.2).to(dtype)
    second = (torch.randn(num_experts, rank, hidden, device=device) * 0.2).to(dtype)
    first.requires_grad_(True)
    second.requires_grad_(True)
    scaling = 0.5
    experts._unsloth_lora_down_proj = (first, second, scaling)

    X = torch.randn(num_tokens, hidden, device=device).to(dtype)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

    out = forward_triton_grouped_gemm(experts, X, top_k_index, top_k_weights)

    first_ref = first.detach().float().requires_grad_(True)
    second_ref = second.detach().float().requires_grad_(True)
    ref = torch.zeros(num_tokens, hidden, device=device)
    for e in range(num_experts):
        t, k = (top_k_index == e).nonzero(as_tuple=True)
        gate, up = (X[t].float() @ experts.gate_up_proj[e].float().T).chunk(2, dim=-1)
        h = F.silu(gate) * up
        d = h @ experts.down_proj[e].float().T + (h @ first_ref[e] @ second_ref[e]) * scaling
        ref.index_add_(0, t, top_k_weights[t, k].unsqueeze(1) * d)

    out.float().sum().backward()
    ref.sum().backward()

    # bf16 rounds at every stage, so allow 2% of the reference range.
    def check(got, want):
        assert (got.float() - want).abs().max() <= 0.02 * want.abs().max()

    check(out, ref)
    check(first.grad, first_ref.grad)
    check(second.grad, second_ref.grad)
