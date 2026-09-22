# The "unsloth" experts implementation and the quantizer's handled check.
import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from unsloth_zoo.temporary_patches.moe_experts_interface import (
    UNSLOTH_EXPERTS_IMPLEMENTATION,
    expert_forward_is_handled,
    patch_experts_interface,
    unsloth_experts_forward,
)
from unsloth_zoo.temporary_patches.moe_utils import forward_moe_backend


@pytest.fixture(autouse = True)
def _restore_transformers_experts_state():
    """These patches are process-global; put transformers back so later tests
    in the same session see the stock dispatch."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel
    getter = PreTrainedModel.__dict__.get("get_correct_experts_implementation")
    local = dict(getattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping", {}))
    yield
    if getter is not None:
        PreTrainedModel.get_correct_experts_implementation = getter
    if hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping"):
        ALL_EXPERTS_FUNCTIONS._local_mapping.clear()
        ALL_EXPERTS_FUNCTIONS._local_mapping.update(local)


def _experts(cls_forward, module_name, has_gate = True, implementation = None, wrapped = False):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states
    forward.__module__ = module_name
    if wrapped:
        forward.__wrapped__ = cls_forward
    cls = type("SomeExperts", (nn.Module,), {"forward": forward})
    m = cls()
    m.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
    m.down_proj = nn.Parameter(torch.zeros(2, 4, 4))
    m.has_gate = has_gate
    m.config = types.SimpleNamespace(_experts_implementation = implementation)
    return m


def test_registered_and_default():
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel
    patch_experts_interface()
    assert ALL_EXPERTS_FUNCTIONS[UNSLOTH_EXPERTS_IMPLEMENTATION] is unsloth_experts_forward
    dummy = types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True)
    getter = PreTrainedModel.get_correct_experts_implementation
    assert getter(dummy, None) == UNSLOTH_EXPERTS_IMPLEMENTATION
    # An explicit request is honoured, so a user can still pick transformers' own path.
    assert getter(dummy, "grouped_mm") == "grouped_mm"


def test_handled_when_forward_is_unsloths():
    cls = type("PatchedExperts", (nn.Module,), {"forward": forward_moe_backend})
    m = cls()
    assert expert_forward_is_handled(m)


def test_handled_via_decorated_class_and_config():
    m = _experts(None, "transformers.models.x.modeling_x", wrapped = True, implementation = "unsloth")
    assert expert_forward_is_handled(m)
    m = _experts(None, "transformers.models.x.modeling_x", wrapped = True, implementation = "grouped_mm")
    assert not expert_forward_is_handled(m)
    m = _experts(None, "transformers.models.x.modeling_x", wrapped = True, implementation = "unsloth", has_gate = False)
    assert not expert_forward_is_handled(m)


def test_not_handled_when_forward_is_the_models_own():
    # Llama-4 before its patch: plain forward, no decorator.
    m = _experts(None, "transformers.models.llama4.modeling_llama4")
    assert not expert_forward_is_handled(m)
    # A compiled-cache copy of a model class is still the model's own code.
    m = _experts(None, "unsloth_compiled_module_llama4")
    assert not expert_forward_is_handled(m)


def test_ungated_falls_back_to_transformers():
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    calls = []
    def fake(self, h, i, w):
        calls.append("grouped_mm"); return h
    original = ALL_EXPERTS_FUNCTIONS._local_mapping.get("grouped_mm") if hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping") else None
    ALL_EXPERTS_FUNCTIONS["grouped_mm"] = fake
    try:
        m = _experts(None, "transformers.models.x.modeling_x", has_gate = False)
        h = torch.zeros(2, 4)
        assert unsloth_experts_forward(m, h, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)) is h
        assert calls == ["grouped_mm"]
    finally:
        if original is not None:
            ALL_EXPERTS_FUNCTIONS["grouped_mm"] = original
        elif hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping"):
            ALL_EXPERTS_FUNCTIONS._local_mapping.pop("grouped_mm", None)
