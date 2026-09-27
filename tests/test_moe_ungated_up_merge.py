"""Ungated MoE experts (NemotronH: experts.up_proj of I rows, no gate_proj) merge their up LoRA.

_merge_moe_gate_or_up_expert classified every up LoRA against a fused gate_up_proj
(out_dim = 2 * I), so NemotronH's up LoRA was "layout not detected" and
save_pretrained_merged raised. Shapes below are the tiny NemotronH ones from the
regression suite failure: A=(32, 16), B=(32, 32), per-expert W=(32, 16).
"""
import torch

from unsloth_zoo import saving_utils as SU
from unsloth_zoo.saving_utils import LoraStats, _merge_moe_gate_expert, _merge_moe_up_expert


def _peft_expert_lora_b(lora_B, expert_idx, num_experts):
    return lora_B.reshape(lora_B.shape[0], -1, num_experts)[:, :, expert_idx]


def _reset():
    for k in list(SU._MOE_MERGE_STATE):
        v = SU._MOE_MERGE_STATE[k]
        SU._MOE_MERGE_STATE[k] = type(v)() if not isinstance(v, int) else 0


def test_ungated_up_standard_layout_nemotronh_shapes():
    torch.manual_seed(0)
    E, r, I, H, alpha = 4, 8, 32, 16, 2.0
    W = torch.randn(I, H)
    A = torch.randn(E * r, H)          # (E*r, in=H)
    B = torch.randn(I, E * r)          # (out=I, E*r)
    _reset()
    for e in range(E):
        out = _merge_moe_up_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=alpha),
                                   e, E, torch.float32)
        exp = W + alpha * (_peft_expert_lora_b(B, e, E) @ A[e * r:(e + 1) * r])
        torch.testing.assert_close(out.cpu(), exp, atol=1e-4, rtol=1e-4)
    assert SU._MOE_MERGE_STATE["applied"] == E
    assert SU._MOE_MERGE_STATE["attempted"] == E


def test_ungated_up_swapped_layout():
    torch.manual_seed(1)
    E, r, I, H, alpha = 4, 3, 8, 12, 4.0
    W = torch.randn(I, H)
    A = torch.randn(E * r, I)          # swapped: (E*r, out=I)
    B = torch.randn(H, E * r)          # (in=H, E*r)
    for e in range(E):
        out = _merge_moe_up_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=alpha),
                                   e, E, torch.float32)
        exp = W + alpha * (_peft_expert_lora_b(B, e, E) @ A[e * r:(e + 1) * r]).T
        torch.testing.assert_close(out.cpu(), exp, atol=1e-4, rtol=1e-4)


def test_gate_role_never_takes_the_ungated_branch():
    # An I-row LoRA reaching the gate role is not a fused gate_up: still refused.
    E, r, I, H = 4, 2, 8, 12
    W = torch.randn(I, H)
    A, B = torch.randn(E * r, H), torch.randn(I, E * r)
    out = _merge_moe_gate_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=1.0),
                                 0, E, torch.float32)
    assert torch.equal(out, W)


def test_fused_gate_up_unchanged():
    torch.manual_seed(2)
    E, r, I, H, alpha = 4, 3, 8, 12, 8.0
    W = torch.randn(I, H)
    A = torch.randn(E * r, H)
    B = torch.randn(2 * I, E * r)      # fused gate_up, standard
    for e in range(E):
        out = _merge_moe_up_expert(W.clone(), LoraStats(module=None, lora_A=A, lora_B=B, alpha=alpha),
                                   e, E, torch.float32)
        exp = W + alpha * (_peft_expert_lora_b(B, e, E)[I:] @ A[e * r:(e + 1) * r])
        torch.testing.assert_close(out.cpu(), exp, atol=1e-4, rtol=1e-4)
