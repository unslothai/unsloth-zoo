# Separated expert LoRA on a SQUARE stack (2 * intermediate == hidden, as on
# Inkling-Small's (256, 4096, 4096) gate_up_proj) must equal PEFT's merged
# per-expert delta. Both layout readings match a square stack by shape, and the
# tie used to be broken the wrong way, applying the LoRA through swapped dims.
import sys

import pytest
import torch

pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from unsloth_zoo.temporary_patches import moe_utils as MU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _peft_original_wrapper_forward():
    for module in list(sys.modules.values()):
        original = getattr(module, "_original_param_wrapper_forward", None)
        if isinstance(module, type(sys)) and callable(original):
            return original
    return None


def _experts(hidden, intermediate, num_experts = 4):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    config = Qwen3MoeConfig(hidden_size = hidden, moe_intermediate_size = intermediate,
                            num_experts = num_experts, num_experts_per_tok = 2)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Qwen3MoeExperts(config)

        def forward(self, x, index, weights):
            return self.experts(x, index, weights)

    torch.manual_seed(0)
    module = Wrap().to(DEVICE).float()
    for p in module.parameters():
        torch.nn.init.normal_(p, std = 0.2)
    return module


@pytest.mark.parametrize("hidden,intermediate", [(32, 16), (32, 24), (64, 32)])
def test_separated_lora_matches_merged_delta(hidden, intermediate):
    from peft.tuners.lora.layer import ParamWrapper
    from unsloth_zoo.temporary_patches.qwen3_moe import patch_qwen3_moe
    from unsloth_zoo.temporary_patches.moe_experts_interface import expert_forward_is_handled
    patch_qwen3_moe()
    assert MU.patch_param_wrapper_for_moe()
    original_forward = _peft_original_wrapper_forward()
    assert original_forward is not None

    base = _experts(hidden, intermediate)
    assert expert_forward_is_handled(base.experts), "the experts forward must be Unsloth's for this to test anything"
    E = base.experts.gate_up_proj.shape[0]
    x = torch.randn(8, hidden, device = DEVICE)
    index = torch.stack([torch.randperm(E, device = DEVICE)[:2] for _ in range(8)])
    weights = torch.rand(8, 2, device = DEVICE)

    config = peft.LoraConfig(r = 3, lora_alpha = 6, target_modules = [],
                             target_parameters = ["experts.gate_up_proj", "experts.down_proj"])
    model = peft.get_peft_model(base, config)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = generator).to(DEVICE) * 0.3)

    # PEFT's own merged forward (per-expert delta from get_delta_weight) is the reference.
    installed = ParamWrapper.forward
    try:
        ParamWrapper.forward = original_forward
        with torch.no_grad():
            expected = model(x, index, weights)
    finally:
        ParamWrapper.forward = installed
    with torch.no_grad():
        got = model(x, index, weights)
    assert torch.allclose(got, expected, atol = 1e-4, rtol = 1e-4), \
        (hidden, intermediate, (got - expected).abs().max().item(), expected.abs().max().item())
