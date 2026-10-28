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

import sys

import pytest
import torch

pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from unsloth_zoo.temporary_patches import moe_utils as MU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _peft_original_wrapper_forward():
    # Read module dicts: getattr on a transformers lazy module imports optional vision deps.
    for module in [MU, *list(sys.modules.values())]:
        if not isinstance(module, type(sys)):
            continue
        original = vars(module).get("_original_param_wrapper_forward", None)
        if callable(original):
            return original
    return None


def _experts(hidden, intermediate, num_experts = 4):
    Qwen3MoeExperts = getattr(pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe"), "Qwen3MoeExperts", None)
    if Qwen3MoeExperts is None:
        pytest.skip("stacked Qwen3MoeExperts is transformers 5")
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


def _stacked_experts(hidden, intermediate, stored_in_out, flag):
    E = 4

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = E
            if stored_in_out:
                self.gate_up_proj = torch.nn.Parameter(torch.randn(E, hidden, 2 * intermediate) * 0.1)
                self.down_proj = torch.nn.Parameter(torch.randn(E, intermediate, hidden) * 0.1)
            else:
                self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * intermediate, hidden) * 0.1)
                self.down_proj = torch.nn.Parameter(torch.randn(E, hidden, intermediate) * 0.1)

        def forward(self, x):
            return x

    if flag == "is_transposed":
        Experts.is_transposed = True
    elif flag == "grouped_mm_format":
        Experts._unsloth_grouped_mm_format = True

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

        def forward(self, x):
            return self.experts(x)

    return Parent().to(DEVICE)


@pytest.mark.parametrize("flag", [None, "is_transposed", "grouped_mm_format"])
@pytest.mark.parametrize("hidden,intermediate", [(64, 32), (64, 48)])
@pytest.mark.parametrize("parameter_name", ["gate_up_proj", "down_proj"])
def test_extractor_matches_peft_delta_for_every_stored_layout(flag, hidden, intermediate, parameter_name):
    from peft.tuners.lora.layer import ParamWrapper
    stored_in_out = flag is not None
    torch.manual_seed(0)
    model = peft.get_peft_model(
        _stacked_experts(hidden, intermediate, stored_in_out, flag),
        peft.LoraConfig(r = 3, lora_alpha = 6, target_modules = [], target_parameters = [f"experts.{parameter_name}"]),
    )
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator = generator).to(DEVICE) * 0.3)
    wrapper = next(m for m in model.modules() if isinstance(m, ParamWrapper))
    E = wrapper.get_base_layer().num_experts
    with torch.no_grad():
        delta = wrapper.get_delta_weight("default")
        first, second, scaling, _ = MU.extract_moe_lora_weights_for_grouped_mm(
            wrapper, wrapper.lora_A["default"].weight, wrapper.lora_B["default"].weight,
            wrapper.scaling["default"], E,
        )
        in_dim = hidden if parameter_name == "gate_up_proj" else intermediate
        x = torch.randn(5, in_dim, device = DEVICE)
        for e in range(E):
            expected = x @ delta[e] if stored_in_out else x @ delta[e].T
            got = (x @ first[e].float() @ second[e].float()) * scaling
            assert torch.allclose(got, expected, atol = 1e-4, rtol = 1e-4), \
                (flag, hidden, intermediate, parameter_name, e, (got - expected).abs().max().item())
