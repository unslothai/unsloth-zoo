"""The down-projection LoRA delta must be scattered back to token order.

The second grouped GEMM runs with permute_y=True, so its output is in token
order, but the down-LoRA delta stays expert-sorted and must be scattered through
gather_indices before the add. Without it the LoRA rows land on the wrong tokens
and the gradients are silently wrong.

Runs on CPU against fake unsloth.kernels modules with the documented
grouped_gemm permute semantics.
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
    """Pure-torch grouped GEMM with the documented permute semantics.

    W is (E, N, K) and each expert computes ``Y = X @ W[e].T``. ``permute_x``
    gathers token-order rows into expert-sorted order (``X[gather_indices // topk]``);
    ``permute_y`` scatters the expert-sorted output back to token order
    (``Y[argsort(gather_indices)]``). Differentiable.
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
    """Install fake unsloth.kernels modules so the Triton forward runs on CPU.

    ``setitem`` overrides any real installed unsloth for the test duration and
    restores it afterwards.
    """
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
    """Minimal `experts` module for the Triton grouped-GEMM forward."""
    experts = nn.Module()
    experts.num_experts = num_experts
    # Canonical (E, out, in) storage: gate_up (E, 2*I, H), down (E, H, I).
    experts.gate_up_proj = nn.Parameter(
        torch.randn(num_experts, 2 * intermediate, hidden)
    )
    experts.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
    experts.act_fn = F.silu
    # Pre-set so the forward skips the CUDA autotune / empty_cache branch;
    # the fake grouped_gemm ignores the (None, None, None) kernel configs.
    experts._unsloth_moe_configs = (intermediate, (None, None, None), (None, None, None))
    return experts


def _reference_forward(experts, X, top_k_index, top_k_weights, first, second, scaling):
    """Naive per-token / per-k reference with down-LoRA only."""
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
    """Common fixture: fake kernels + CPU LoRA grouped-mm + seeded inputs."""
    _install_fake_unsloth_kernels(monkeypatch)
    # The LoRA delta path calls the module global at call time; route it to the
    # CPU per-group matmul so no CUDA _grouped_mm is needed.
    monkeypatch.setattr(moe_utils, "native_moe_grouped_mm", moe_utils._manual_grouped_mm)

    num_experts, hidden, intermediate, rank = 4, 16, 12, 4
    num_tokens, top_k = 8, 2

    torch.manual_seed(0)
    experts = _build_experts(num_experts, hidden, intermediate)
    first = torch.randn(num_experts, intermediate, rank)   # (E, in, R)
    second = torch.randn(num_experts, rank, hidden)        # (E, R, out)
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
