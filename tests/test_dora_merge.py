"""DoRA (use_dora=True) merge support in the safetensors merge path.

The dense merge must (1) no longer raise a key-mismatch on a DoRA adapter (the magnitude vector
is now captured), and (2) produce the same merged weight as PEFT's own DoRA merge. MoE-expert
DoRA is explicitly refused (fail loud) rather than silently dropping the magnitude.
"""
import copy

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.saving_utils import create_lora_statistics, _merge_lora, LoraStats


class _Tiny(nn.Module):
    def __init__(self, d_in=32, d_out=24):
        super().__init__()
        self.q_proj = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def _find_q_stats(lora_weights):
    for v in lora_weights.values():
        if v.lora_A is not None and v.lora_B is not None:
            return v
    return None


def test_dora_merge_matches_peft():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = _Tiny().to(torch.float32)
    W0 = base.q_proj.weight.detach().clone()

    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"], use_dora=True)
    pm = get_peft_model(copy.deepcopy(base), cfg)

    # Give the adapter a non-trivial delta and magnitude so DoRA actually rescales.
    for n, p in pm.named_parameters():
        if n.endswith("lora_B.default.weight"):
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.1)
        if n.endswith("lora_magnitude_vector.default.weight"):
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.1)

    # Ground truth: PEFT's own DoRA merge.
    merged_peft = copy.deepcopy(pm).merge_and_unload()
    W_peft = None
    for n, p in merged_peft.named_parameters():
        if n.endswith("q_proj.weight"):
            W_peft = p.detach().float().clone()
    assert W_peft is not None

    # Unsloth merge path: capture stats (must NOT raise on DoRA) then fold via _merge_lora.
    result = create_lora_statistics(pm, merge_into_original=True, return_state_dict=True)
    lora_weights = result[0] if isinstance(result, tuple) else result
    stats = _find_q_stats(lora_weights)
    assert stats is not None
    assert stats.magnitude is not None, "DoRA magnitude was not captured"

    W_uns = _merge_lora(W0.clone(), stats, "q_proj").cpu().float()

    max_abs = (W_uns - W_peft).abs().max().item()
    assert torch.allclose(W_uns, W_peft, atol=1e-4, rtol=1e-4), f"max abs diff {max_abs}"
    # Sanity: DoRA actually changed the weight vs the plain base.
    assert (W_uns - W0.float()).abs().max().item() > 1e-3


def test_plain_lora_unaffected():
    """A non-DoRA adapter has magnitude None and merges as W0 + alpha*BA."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    base = _Tiny().to(torch.float32)
    W0 = base.q_proj.weight.detach().clone()
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"], use_dora=False)
    pm = get_peft_model(copy.deepcopy(base), cfg)
    for n, p in pm.named_parameters():
        if n.endswith("lora_B.default.weight"):
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.1)

    result = create_lora_statistics(pm, merge_into_original=True, return_state_dict=True)
    lora_weights = result[0] if isinstance(result, tuple) else result
    stats = _find_q_stats(lora_weights)
    assert stats is not None and stats.magnitude is None

    W_uns = _merge_lora(W0.clone(), stats, "q_proj").cpu().float()
    expected = W0.float() + stats.alpha * (stats.lora_B.float() @ stats.lora_A.float())
    assert torch.allclose(W_uns, expected, atol=1e-5)


def test_dora_on_moe_expert_is_refused():
    from unsloth_zoo.saving_utils import _merge_moe_fused_gate_up_expert

    E, rank, H, I = 4, 4, 8, 6
    gate_up_W = torch.randn(E, 2 * I, H)
    A = torch.randn(E * rank, H)
    B = torch.randn(2 * I, E * rank)
    stats = LoraStats(None, A, B, 1.0, magnitude=torch.randn(2 * I))
    with pytest.raises(RuntimeError, match="DoRA"):
        _merge_moe_fused_gate_up_expert(gate_up_W, stats, torch.float32)
