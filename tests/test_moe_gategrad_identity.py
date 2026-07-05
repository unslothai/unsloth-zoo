"""The gate-grad identity must reproduce autograd's routing-weight gradient.

`_MoEGateGradIdentity` derives the routing-weight (gate) gradient from
``dGate = <A, dA> / gate`` over the pre-down-projection activation instead of
differentiating the post-down multiply, so the down-projection output never has
to stay on the autograd tape. The identity is exact for any linear down
projection; these tests check it in fp32 against plain autograd, with and
without an additive LoRA term on the down weight.
"""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_MOE_GATEGRAD", "0")

from unsloth_zoo.temporary_patches.moe_utils import (
    _MoEGateGradIdentity,
    _moe_gategrad_enabled,
)


def _reference(inter, gate, W2, dOut):
    inter = inter.detach().requires_grad_(True)
    gate = gate.detach().requires_grad_(True)
    out = (inter @ W2) * gate.unsqueeze(-1)
    out.backward(dOut)
    return inter.grad, gate.grad


def _identity_path(inter, gate, W2, dOut):
    inter = inter.detach().requires_grad_(True)
    gate = gate.detach().requires_grad_(True)
    inter_id = _MoEGateGradIdentity.apply(inter, gate)
    # The caller multiplies by the detached gate; the gate gradient comes only
    # from the identity's backward.
    out = (inter_id @ W2) * gate.detach().unsqueeze(-1)
    out.backward(dOut)
    return inter.grad, gate.grad


@pytest.mark.parametrize("n_tokens,d_inter,d_out", [(7, 16, 12), (64, 128, 96)])
def test_matches_autograd_fp32(n_tokens, d_inter, d_out):
    torch.manual_seed(0)
    inter = torch.randn(n_tokens, d_inter)
    gate = torch.rand(n_tokens) * 0.9 + 0.05  # routing weights, strictly positive
    W2 = torch.randn(d_inter, d_out)
    dOut = torch.randn(n_tokens, d_out)

    g_inter_ref, g_gate_ref = _reference(inter, gate, W2, dOut)
    g_inter, g_gate = _identity_path(inter, gate, W2, dOut)

    torch.testing.assert_close(g_inter, g_inter_ref, rtol=1e-6, atol=1e-6)
    # dGate reduces over d_inter in a different order than autograd's <Y, dOut>,
    # so allow fp32 accumulation-order noise.
    torch.testing.assert_close(g_gate, g_gate_ref, rtol=5e-4, atol=5e-5)


def test_exact_with_lora_down_projection():
    torch.manual_seed(1)
    n, d, r, h = 32, 64, 8, 48
    inter = torch.randn(n, d)
    gate = torch.rand(n) * 0.9 + 0.05
    W2 = torch.randn(d, h)
    A, B = torch.randn(d, r), torch.randn(r, h)
    W2_eff = W2 + (A @ B) * 0.5
    dOut = torch.randn(n, h)

    _, g_gate_ref = _reference(inter, gate, W2_eff, dOut)
    _, g_gate = _identity_path(inter, gate, W2_eff, dOut)
    torch.testing.assert_close(g_gate, g_gate_ref, rtol=5e-4, atol=5e-5)


def test_matches_autograd_with_negative_gate():
    # Gemma4 folds an unconstrained per_expert_scale into top_k_weights, so the
    # routing weight can be negative; the identity must divide by the signed
    # weight rather than clamp it toward a positive floor.
    torch.manual_seed(2)
    inter = torch.randn(16, 24)
    gate = torch.randn(16) * 0.5
    gate = torch.where(gate.abs() < 0.05, torch.full_like(gate, 0.1), gate)
    W2 = torch.randn(24, 20)
    dOut = torch.randn(16, 20)

    _, g_gate_ref = _reference(inter, gate, W2, dOut)
    _, g_gate = _identity_path(inter, gate, W2, dOut)
    torch.testing.assert_close(g_gate, g_gate_ref, rtol=5e-4, atol=5e-5)


def test_gate_gradient_skipped_when_router_frozen():
    # A frozen router (routing weights with requires_grad=False) must not crash
    # the backward; inter keeps its passthrough gradient, the gate gets none.
    inter = torch.randn(6, 8, requires_grad=True)
    gate = torch.rand(6)
    W2 = torch.randn(8, 5)

    inter_id = _MoEGateGradIdentity.apply(inter, gate)
    out = (inter_id @ W2) * gate.detach().unsqueeze(-1)
    out.backward(torch.randn(6, 5))

    assert inter.grad is not None
    assert gate.grad is None


def test_double_backward_raises():
    # The identity is only first-order exact; create_graph must fail loudly
    # instead of returning wrong second derivatives.
    inter = torch.randn(4, 6, requires_grad=True)
    gate = torch.rand(4, requires_grad=True) + 0.1
    W2 = torch.randn(6, 3)

    inter_id = _MoEGateGradIdentity.apply(inter, gate)
    out = (inter_id @ W2) * gate.detach().unsqueeze(-1)
    (g_gate,) = torch.autograd.grad(out.sum(), gate, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(g_gate.sum(), gate)


def test_forward_is_identity_and_env_gated(monkeypatch):
    x = torch.randn(5, 3, requires_grad=True)
    g = torch.rand(5)
    assert torch.equal(_MoEGateGradIdentity.apply(x, g), x)

    monkeypatch.delenv("UNSLOTH_MOE_GATEGRAD", raising=False)
    assert not _moe_gategrad_enabled()
    monkeypatch.setenv("UNSLOTH_MOE_GATEGRAD", "1")
    assert _moe_gategrad_enabled()
