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


def test_forward_is_identity_and_env_gated(monkeypatch):
    x = torch.randn(5, 3, requires_grad=True)
    g = torch.rand(5)
    assert torch.equal(_MoEGateGradIdentity.apply(x, g), x)

    monkeypatch.delenv("UNSLOTH_MOE_GATEGRAD", raising=False)
    assert not _moe_gategrad_enabled()
    monkeypatch.setenv("UNSLOTH_MOE_GATEGRAD", "1")
    assert _moe_gategrad_enabled()
