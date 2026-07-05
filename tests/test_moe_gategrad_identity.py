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


def test_matches_autograd_with_zero_gate():
    # A routing weight of exactly zero (or below the floor) must still get its
    # true gradient <Y, dOut>. The call site sign-floors the gate away from zero
    # (straight-through) and both the multiply and the identity's divide use the
    # same floored value, so the floor cancels exactly.
    torch.manual_seed(3)
    inter0 = torch.randn(12, 16)
    gate0 = torch.randn(12) * 0.5
    gate0[0] = 0.0
    gate0[1] = 1e-30
    gate0[2] = -1e-30
    W2 = torch.randn(16, 10)
    dOut = torch.randn(12, 10)

    _, g_gate_ref = _reference(inter0, gate0, W2, dOut)

    inter = inter0.detach().requires_grad_(True)
    gate = gate0.detach().requires_grad_(True)
    eps = max(1e-12, float(torch.finfo(gate.dtype).tiny))
    floored = torch.where(gate >= 0, gate.clamp(min=eps), gate.clamp(max=-eps))
    safe = gate + (floored - gate).detach()
    inter_id = _MoEGateGradIdentity.apply(inter, safe)
    out = (inter_id @ W2) * safe.detach().unsqueeze(-1)
    out.backward(dOut)

    torch.testing.assert_close(gate.grad, g_gate_ref, rtol=5e-4, atol=5e-5)


def test_fp16_gate_floors_in_fp32_without_output_leak():
    # fp16 routing weights must be floored in fp32 (eps 1e-12), not at fp16's
    # smallest normal (~6e-5): flooring a masked (zero) route at 6e-5 would leak
    # eps * |Y| into the output at a magnitude fp16 can represent. Mirrors the
    # call-site upcast in forward_native_grouped_mm.
    torch.manual_seed(5)
    gate16 = torch.rand(8, dtype=torch.float16) * 0.5
    gate16[0] = 0.0
    inter = torch.randn(8, 16)
    W2 = torch.randn(16, 10)

    gate = gate16.detach().requires_grad_(True)
    raw = gate.to(torch.float32)
    eps = 1e-12
    floored = torch.where(raw >= 0, raw.clamp(min=eps), raw.clamp(max=-eps))
    safe = raw + (floored - raw).detach()
    inter_id = _MoEGateGradIdentity.apply(inter, safe)
    out = (inter_id @ W2) * safe.detach().unsqueeze(-1)

    # The masked route's output leak is bounded by eps * |Y|, invisible in fp16.
    leak = out[0].abs().max().item()
    assert leak < 1e-10, leak

    # The zero route still gets its true gradient <Y, dOut>.
    dOut = torch.randn(8, 10)
    out.backward(dOut)
    ref = ((inter[0] @ W2) * dOut[0]).sum()
    torch.testing.assert_close(gate.grad[0], ref.to(gate.dtype), rtol=5e-3, atol=5e-3)


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
