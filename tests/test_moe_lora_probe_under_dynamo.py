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

"""The stash probe must still give a verdict when Dynamo compiles its frame alone (fullgraph=False)."""

import importlib.util
import pathlib

import pytest
import torch

# By path, not `tests.`: that name binds to unsloth's own tests package under its CI.
_spec = importlib.util.spec_from_file_location(
    "_moe_stacked_expert_lora_sibling",
    pathlib.Path(__file__).with_name("test_moe_stacked_expert_lora_reaches_forward.py"),
)
_sibling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sibling)
MU = _sibling.MU
_StashIgnoringExperts = _sibling._StashIgnoringExperts
_StashReadingExperts = _sibling._StashReadingExperts
_build = _sibling._build
_inputs = _sibling._inputs
restore_param_wrapper = _sibling.restore_param_wrapper


def _wrapper_and_experts(model):
    for module in model.modules():
        if type(module).__name__ == "ParamWrapper" and getattr(module, "parameter_name", None) == "gate_up_proj":
            return module, module.get_base_layer()
    raise AssertionError("no gate_up_proj ParamWrapper")


@pytest.mark.parametrize("experts_cls, expected", [(_StashIgnoringExperts, False), (_StashReadingExperts, True)])
def test_probe_gives_a_verdict_when_dynamo_compiles_its_frame(restore_param_wrapper, experts_cls, expected):
    assert MU.patch_param_wrapper_for_moe()
    model = _build(experts_cls)
    wrapper, experts = _wrapper_and_experts(model)
    x = _inputs()
    torch._dynamo.reset()
    probe = torch.compile(
        lambda: MU._measure_moe_lora_stash_read(wrapper, experts, "gate_up_proj", x, (), {}),
        fullgraph = False, backend = "eager",
    )
    assert probe() is expected
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is expected

