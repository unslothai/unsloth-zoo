# SPDX-License-Identifier: AGPL-3.0-only
"""Stash scan traces under fullgraph for a real experts class (torch 2.14 AttributeError('__globals__'))."""
import pytest
import torch

MU = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils")


def _experts():
    modeling = pytest.importorskip("transformers.models.mimo_v2_flash.modeling_mimo_v2_flash")
    config_mod = pytest.importorskip("transformers.models.mimo_v2_flash.configuration_mimo_v2_flash")
    config = config_mod.MiMoV2FlashConfig(num_local_experts = 4, hidden_size = 8, moe_intermediate_size = 4)
    return modeling.MiMoV2FlashExperts(config)


@pytest.mark.parametrize("fullgraph", [True, False])
def test_scan_of_real_experts_compiles(fullgraph):
    experts = _experts()
    eager = MU._forward_statically_reads_stash(experts)
    assert eager is False

    def scan(x):
        return x + (1.0 if MU._forward_statically_reads_stash(experts) else 0.0)

    torch._dynamo.reset()
    compiled = torch.compile(scan, fullgraph = fullgraph)
    assert torch.equal(compiled(torch.zeros(3)), torch.zeros(3))
    assert torch.equal(compiled(torch.zeros(3)), torch.zeros(3))


def test_instance_forward_still_wins():
    import types

    experts = _experts()

    def forward(self, *args, **kwargs):
        return take_moe_lora_stash(self)  # noqa: F821

    experts.forward = types.MethodType(forward, experts)
    assert MU._forward_statically_reads_stash(experts) is True
