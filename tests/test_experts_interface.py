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


def test_custom_gate_falls_back_and_is_not_packed():
    """DeepSeek-V4, MiniMax-M3 and others override _apply_gate with clamps or an
    offset; the Unsloth backends would compute plain SiLU gating for them."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    calls = []
    def fake(self, h, i, w):
        calls.append("grouped_mm"); return h
    original = ALL_EXPERTS_FUNCTIONS._local_mapping.get("grouped_mm") if hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping") else None
    ALL_EXPERTS_FUNCTIONS["grouped_mm"] = fake
    try:
        m = _experts(None, "transformers.models.x.modeling_x", wrapped = True, implementation = "unsloth")
        assert expert_forward_is_handled(m)
        type(m)._apply_gate = lambda self, x: x.clamp(max = 7.0)
        assert not expert_forward_is_handled(m)
        h = torch.zeros(2, 4)
        assert unsloth_experts_forward(m, h, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)) is h
        assert calls == ["grouped_mm"]
        # transformers' own default gate is not a custom gate.
        from transformers.integrations.moe import _default_apply_gate
        type(m)._apply_gate = _default_apply_gate
        assert expert_forward_is_handled(m)
    finally:
        if original is not None:
            ALL_EXPERTS_FUNCTIONS["grouped_mm"] = original
        elif hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping"):
            ALL_EXPERTS_FUNCTIONS._local_mapping.pop("grouped_mm", None)


def test_expert_parallel_keeps_transformers_default():
    """RouterParallel sends non-local slots to a num_experts sentinel that only
    transformers' implementations mask."""
    from transformers.modeling_utils import PreTrainedModel
    patch_experts_interface()
    getter = PreTrainedModel.get_correct_experts_implementation
    ep = types.SimpleNamespace(
        _grouped_mm_can_dispatch = lambda: True,
        config = types.SimpleNamespace(distributed_config = types.SimpleNamespace(enable_expert_parallel = True)),
    )
    assert getter(ep, None) == "grouped_mm"
    no_ep = types.SimpleNamespace(
        _grouped_mm_can_dispatch = lambda: True,
        config = types.SimpleNamespace(distributed_config = types.SimpleNamespace(enable_expert_parallel = False)),
    )
    assert getter(no_ep, None) == UNSLOTH_EXPERTS_IMPLEMENTATION


def test_every_transformers_custom_gate_class_is_detected():
    import importlib
    from unsloth_zoo.temporary_patches.moe_experts_interface import _has_custom_gate
    found = []
    for module_name, class_name in (
        ("transformers.models.deepseek_v4.modeling_deepseek_v4", "DeepseekV4Experts"),
        ("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl", "MiniMaxM3VLExperts"),
        ("transformers.models.openai_privacy_filter.modeling_openai_privacy_filter", "OpenAIPrivacyFilterExperts"),
    ):
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            continue
        found.append(class_name)
        assert _has_custom_gate(cls.__new__(cls)), class_name
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
        assert not _has_custom_gate(Qwen3MoeExperts.__new__(Qwen3MoeExperts))
    except ImportError:
        pass
    if not found:
        pytest.skip("this transformers has none of the custom-gate expert classes")


def test_fp8_experts_registry_resolves_the_unsloth_default(monkeypatch):
    # The FP8Experts swap keeps config._experts_implementation, which Unsloth defaults to
    # "unsloth"; transformers' FP8 registry must resolve it instead of raising KeyError.
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    moe_utils_fp8.patch_fp8_experts_interface()
    forward = fp8.ALL_FP8_EXPERTS_FUNCTIONS.get_interface("unsloth", None)
    calls = []
    monkeypatch.setattr(moe_utils_fp8, "forward_moe_backend_fp8", lambda *a: calls.append(a) or "ok")
    assert forward(object(), "h", "i", "w") == "ok"
    assert len(calls) == 1


def test_a_class_marked_for_its_own_gate_counts_as_handled():
    # The bnb 4-bit route marks classes whose _apply_gate every backend applies; those are
    # not custom-gate fallbacks.
    pytest.importorskip("transformers.integrations.moe")
    from unsloth_zoo.temporary_patches.moe_experts_interface import _has_custom_gate

    class Clamped(nn.Module):
        def _apply_gate(self, gate_up_out):
            return gate_up_out

    assert _has_custom_gate(Clamped())
    Clamped._unsloth_own_apply_gate = True
    assert not _has_custom_gate(Clamped())


def test_bnb_handled_check_defers_to_the_generic_route(monkeypatch):
    # With the generic bnb 4-bit route present, a class it takes over counts as handled;
    # without one, a user-forced implementation stays unpacked as before.
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb

    m = _experts(None, "transformers.models.x.modeling_x", wrapped = True, implementation = "grouped_mm")
    monkeypatch.delattr(mb, "_route_generic_bnb4bit_experts_class", raising = False)
    assert not mb._expert_forward_is_handled(m)

    def route(module):
        type(module).forward = forward_moe_backend
        return True

    monkeypatch.setattr(mb, "_route_generic_bnb4bit_experts_class", route, raising = False)
    assert mb._expert_forward_is_handled(m)

    declined = _experts(None, "transformers.models.x.modeling_x", wrapped = True, implementation = "grouped_mm")
    monkeypatch.setattr(mb, "_route_generic_bnb4bit_experts_class", lambda module: False, raising = False)
    assert not mb._expert_forward_is_handled(declined)
