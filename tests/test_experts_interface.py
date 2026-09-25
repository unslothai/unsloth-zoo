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

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.integrations.moe")

from unsloth_zoo.temporary_patches.moe_experts_interface import (
    UNSLOTH_EXPERTS_IMPLEMENTATION,
    expert_forward_is_handled,
    patch_experts_interface,
    unsloth_experts_forward,
    _unsloth_experts_dispatch,
)
from unsloth_zoo.temporary_patches.moe_utils import forward_moe_backend


@pytest.fixture(autouse = True)
def _restore_transformers_experts_state():
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
    m = _experts(None, "transformers.models.llama4.modeling_llama4")
    assert not expert_forward_is_handled(m)
    m = _experts(None, "unsloth_compiled_module_llama4")
    assert not expert_forward_is_handled(m)


def test_ungated_falls_back_to_transformers(monkeypatch):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    import unsloth_zoo.temporary_patches.moe_utils as moe_utils
    monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: True)
    calls = []
    def fake(self, h, i, w):
        calls.append("grouped_mm"); return h
    original = ALL_EXPERTS_FUNCTIONS._local_mapping.get("grouped_mm") if hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping") else None
    ALL_EXPERTS_FUNCTIONS["grouped_mm"] = fake
    try:
        m = _experts(None, "transformers.models.x.modeling_x", has_gate = False)
        h = torch.zeros(2, 4)
        assert _unsloth_experts_dispatch(m, h, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)) is h
        assert calls == ["grouped_mm"]
    finally:
        if original is not None:
            ALL_EXPERTS_FUNCTIONS["grouped_mm"] = original
        elif hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping"):
            ALL_EXPERTS_FUNCTIONS._local_mapping.pop("grouped_mm", None)


def test_custom_gate_falls_back_and_is_not_packed(monkeypatch):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    import unsloth_zoo.temporary_patches.moe_utils as moe_utils
    monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: True)
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
        assert _unsloth_experts_dispatch(m, h, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)) is h
        assert calls == ["grouped_mm"]
        from transformers.integrations.moe import _default_apply_gate
        type(m)._apply_gate = _default_apply_gate
        assert expert_forward_is_handled(m)
    finally:
        if original is not None:
            ALL_EXPERTS_FUNCTIONS["grouped_mm"] = original
        elif hasattr(ALL_EXPERTS_FUNCTIONS, "_local_mapping"):
            ALL_EXPERTS_FUNCTIONS._local_mapping.pop("grouped_mm", None)


def test_expert_parallel_keeps_transformers_default():
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
    assert getter(ep, UNSLOTH_EXPERTS_IMPLEMENTATION) == "grouped_mm"


def test_fallback_uses_the_eager_forward_without_grouped_mm(monkeypatch):
    import unsloth_zoo.temporary_patches.moe_utils as moe_utils
    monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: False)
    calls = []
    def eager(self, h, i, w):
        calls.append("eager"); return h
    m = _experts(eager, "transformers.models.x.modeling_x", has_gate = False, wrapped = True)
    h = torch.zeros(2, 4)
    assert _unsloth_experts_dispatch(m, h, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)) is h
    assert calls == ["eager"]


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
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    moe_utils_fp8.patch_fp8_experts_interface()
    forward = fp8.ALL_FP8_EXPERTS_FUNCTIONS.get_interface("unsloth", None)
    calls = []
    monkeypatch.setattr(moe_utils_fp8, "forward_moe_backend_fp8", lambda *a: calls.append(a) or "ok")
    assert forward(object(), "h", "i", "w") == "ok"
    assert len(calls) == 1


def test_a_class_marked_for_its_own_gate_counts_as_handled():
    pytest.importorskip("transformers.integrations.moe")
    from unsloth_zoo.temporary_patches.moe_experts_interface import _has_custom_gate

    class Clamped(nn.Module):
        def _apply_gate(self, gate_up_out):
            return gate_up_out

    assert _has_custom_gate(Clamped())
    Clamped._unsloth_own_apply_gate = True
    assert not _has_custom_gate(Clamped())


def test_bnb_handled_check_defers_to_the_generic_route(monkeypatch):
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


def test_only_the_cached_dispatcher_module_counts_as_unsloth():
    from unsloth_zoo.temporary_patches.moe_experts_interface import _forward_is_unsloth
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states
    for ours in ("unsloth_cached_moe_utils", "moe_utils"):
        forward.__module__ = ours
        assert _forward_is_unsloth(forward), ours
    for other in ("my_project.moe_utils", "transformers_modules.x.custom_moe_utils"):
        forward.__module__ = other
        assert not _forward_is_unsloth(forward), other


def test_fp4_experts_keep_transformers_dispatchers(monkeypatch):
    from transformers.modeling_utils import PreTrainedModel
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    patch_experts_interface()
    getter = PreTrainedModel.get_correct_experts_implementation
    fp4 = types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True, config = types.SimpleNamespace(expert_dtype = "fp4"))
    fp8_model = types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True, config = types.SimpleNamespace(expert_dtype = "fp8"))
    assert getter(fp4, None) == "grouped_mm"
    assert getter(fp8_model, None) == UNSLOTH_EXPERTS_IMPLEMENTATION

    moe_utils_fp8.patch_fp8_experts_interface()
    calls = []
    monkeypatch.setattr(moe_utils_fp8, "forward_moe_backend_fp8", lambda *a: calls.append("unsloth") or "unsloth")

    class Experts:
        def forward(self, *a):
            raise AssertionError("dispatched forward")
        forward.__wrapped__ = lambda self, *a: calls.append("eager") or "eager"

    experts = Experts()
    forward = fp8.ALL_FP8_EXPERTS_FUNCTIONS.get_interface("unsloth", None)
    fp4_route = "unsloth" if hasattr(moe_utils_fp8, "_dequantize_full_expert_weights_fp4") else "eager"
    experts.config = types.SimpleNamespace(expert_dtype = "fp4")
    assert forward(experts, "h", "i", "w") == fp4_route
    experts.config = types.SimpleNamespace(expert_dtype = "fp8")
    assert forward(experts, "h", "i", "w") == "unsloth"
    experts.has_gate = False
    assert forward(experts, "h", "i", "w") == "eager"
    assert calls == [fp4_route, "unsloth", "eager"]


def test_expert_parallel_reaches_the_nested_text_model():
    from transformers import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel
    patch_experts_interface()
    getter = PreTrainedModel.get_correct_experts_implementation

    class Outer(PretrainedConfig):
        sub_configs = {"text_config": PretrainedConfig}

    text = PretrainedConfig()
    outer = Outer()
    outer.text_config = text
    outer.distributed_config = types.SimpleNamespace(enable_expert_parallel = True)
    nested = types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True, config = text)
    assert getter(nested, None) == UNSLOTH_EXPERTS_IMPLEMENTATION
    assert getter(types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True, config = outer), None) == "grouped_mm"
    assert getter(nested, None) == "grouped_mm"
    assert "unsloth" not in str(text.to_dict())
    other = types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True, config = PretrainedConfig())
    assert getter(other, None) == UNSLOTH_EXPERTS_IMPLEMENTATION


def test_the_fp8_eager_fallback_survives_repeated_decoration():
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from transformers.integrations.moe import use_experts_implementation
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    moe_utils_fp8.patch_fp8_experts_interface()

    class Ungated(torch.nn.Module):
        def forward(self, hidden_states, top_k_index, top_k_weights):
            return "eager"

    for _ in range(3):
        Ungated = use_experts_implementation(
            experts_class = Ungated, experts_interface = fp8.ALL_FP8_EXPERTS_FUNCTIONS,
            has_bias = False, has_gate = False,
        )
    experts = Ungated.__new__(Ungated)
    torch.nn.Module.__init__(experts)
    experts.config = types.SimpleNamespace(_experts_implementation = UNSLOTH_EXPERTS_IMPLEMENTATION)
    experts.has_gate = False
    assert experts.forward("h", "i", "w") == "eager"


def test_a_runtime_switch_is_refused_once_experts_are_packed_4bit():
    from transformers.modeling_utils import PreTrainedModel

    class Params4bit(nn.Parameter):
        pass

    def model(packed):
        root = nn.Module()
        root._grouped_mm_can_dispatch = lambda: True
        root.experts = _experts(None, "transformers.integrations.moe")
        if packed:
            root.experts.gate_up_proj = Params4bit(torch.zeros(8, 1, dtype = torch.uint8), requires_grad = False)
        return root

    patch_experts_interface()
    getter = PreTrainedModel.get_correct_experts_implementation
    with pytest.raises(RuntimeError, match = "4-bit"):
        getter(model(True), "grouped_mm")
    assert getter(model(True), None) == UNSLOTH_EXPERTS_IMPLEMENTATION
    assert getter(model(False), "grouped_mm") == "grouped_mm"


def test_expert_parallel_fp8_experts_keep_transformers_path(monkeypatch):
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from unsloth_zoo.temporary_patches import moe_utils_fp8

    moe_utils_fp8.patch_fp8_experts_interface()
    calls = []
    monkeypatch.setattr(moe_utils_fp8, "forward_moe_backend_fp8", lambda *a: calls.append("unsloth") or "unsloth")

    class Experts:
        def forward(self, *a):
            raise AssertionError("dispatched forward")
        forward.__wrapped__ = lambda self, *a: calls.append("eager") or "eager"

    forward = fp8.ALL_FP8_EXPERTS_FUNCTIONS.get_interface("unsloth", None)
    experts = Experts()
    experts.config = types.SimpleNamespace(
        expert_dtype = "fp8",
        distributed_config = types.SimpleNamespace(enable_expert_parallel = True),
    )
    assert forward(experts, "h", "i", "w") == "eager"
    experts.config = types.SimpleNamespace(
        expert_dtype = "fp8",
        distributed_config = types.SimpleNamespace(enable_expert_parallel = False),
    )
    assert forward(experts, "h", "i", "w") == "unsloth"
    assert calls == ["eager", "unsloth"]


def test_dense_stacks_without_expert_lora_take_transformers_grouped_mm(monkeypatch):
    import unsloth_zoo.temporary_patches.moe_experts_interface as MEI
    import unsloth_zoo.temporary_patches.moe_utils as moe_utils
    calls = []
    monkeypatch.setattr(MEI, "_TRANSFORMERS_GROUPED_MM", lambda self, h, i, w: calls.append("grouped_mm") or h)
    monkeypatch.setattr(MEI, "_unsloth_experts_dispatch", lambda self, h, i, w: calls.append("unsloth") or h)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", True)
    m = _experts(None, "transformers.models.x.modeling_x", implementation = "unsloth")
    h, i, w = torch.zeros(2, 4), torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)
    assert unsloth_experts_forward(m, h, i, w) is h
    assert calls == ["grouped_mm"]

    calls.clear()
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", False)
    monkeypatch.setattr(MEI, "_TRANSFORMERS_GROUPED_MM_HAS_FALLBACK", True)
    unsloth_experts_forward(m, h, i, w)
    assert calls == ["grouped_mm"]
    calls.clear()
    monkeypatch.setattr(MEI, "_TRANSFORMERS_GROUPED_MM_HAS_FALLBACK", False)
    unsloth_experts_forward(m, h, i, w)
    assert calls == ["unsloth"]
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", True)

    calls.clear()
    m._unsloth_lora_gate_up_proj = ("A", "B", 1.0)
    unsloth_experts_forward(m, h, i, w)
    assert calls == ["unsloth"]
    del m._unsloth_lora_gate_up_proj

    class Packed(nn.Parameter):
        pass
    calls.clear()
    m.down_proj = Packed(torch.zeros(2, 4, 4))
    unsloth_experts_forward(m, h, i, w)
    assert calls == ["unsloth"]
    calls.clear()
    m.down_proj = nn.Parameter(torch.zeros(2, 4, 4, dtype = torch.uint8), requires_grad = False)
    unsloth_experts_forward(m, h, i, w)
    assert calls == ["unsloth"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "grouped_mm experts forward on CUDA")
def test_dense_experts_stay_inside_a_fullgraph_compiled_block():
    olmoe = pytest.importorskip("transformers.models.olmoe.modeling_olmoe")
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    patch_experts_interface()
    config = OlmoeConfig(hidden_size = 64, intermediate_size = 32, num_experts = 4, num_experts_per_tok = 2)
    config._experts_implementation = UNSLOTH_EXPERTS_IMPLEMENTATION
    experts = olmoe.OlmoeExperts(config).cuda().to(torch.bfloat16)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std = 0.02)
    h = torch.randn(8, 64, device = "cuda", dtype = torch.bfloat16)
    index = torch.randint(0, 4, (8, 2), device = "cuda")
    weights = torch.rand(8, 2, device = "cuda", dtype = torch.bfloat16)
    torch._dynamo.reset()
    compiled = torch.compile(lambda h, i, w: experts(h, i, w), fullgraph = True)
    with torch.no_grad():
        got = compiled(h, index, weights)
        config._experts_implementation = "grouped_mm"
        expected = experts(h, index, weights)
    torch.testing.assert_close(got, expected, atol = 2e-2, rtol = 2e-2)


def test_nested_recheck_of_unsloth_does_not_reach_a_fixed_list_validator(monkeypatch):
    from transformers.modeling_utils import PreTrainedModel

    def fixed_list_original(self, requested_experts):
        applicable = "grouped_mm" if requested_experts is None else requested_experts
        if applicable not in ["eager", "grouped_mm", "batched_mm", "deepgemm"]:
            raise ValueError(f'Specified `experts_implementation="{applicable}"` is not supported.')
        return applicable

    monkeypatch.setattr(PreTrainedModel, "get_correct_experts_implementation", fixed_list_original)
    patch_experts_interface()
    getter = PreTrainedModel.get_correct_experts_implementation
    model = types.SimpleNamespace(_grouped_mm_can_dispatch = lambda: True, config = types.SimpleNamespace())
    assert getter(model, None) == UNSLOTH_EXPERTS_IMPLEMENTATION
    assert getter(model, UNSLOTH_EXPERTS_IMPLEMENTATION) == UNSLOTH_EXPERTS_IMPLEMENTATION
    assert getter(model, "eager") == "eager"


def test_dense_stacks_follow_generates_decode_switch_to_batched_mm(monkeypatch):
    import unsloth_zoo.temporary_patches.moe_experts_interface as MEI
    calls = []
    monkeypatch.setattr(MEI, "_TRANSFORMERS_GROUPED_MM", lambda self, h, i, w: calls.append("grouped_mm") or h)
    monkeypatch.setattr(MEI, "_TRANSFORMERS_BATCHED_MM", lambda self, h, i, w: calls.append("batched_mm") or h)
    monkeypatch.setattr(MEI, "_unsloth_experts_dispatch", lambda self, h, i, w: calls.append("unsloth") or h)
    monkeypatch.setattr(MEI, "_TRANSFORMERS_GROUPED_MM_HAS_FALLBACK", True)
    m = _experts(None, "transformers.models.x.modeling_x", implementation = "unsloth")
    h, i, w = torch.zeros(2, 4), torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)

    monkeypatch.setattr(MEI, "_DECODING_DEPTH", 1)
    with torch.no_grad():
        unsloth_experts_forward(m, h, i, w)
    unsloth_experts_forward(m, h, i, w)
    m._unsloth_lora_gate_up_proj = ("A", "B", 1.0)
    with torch.no_grad():
        unsloth_experts_forward(m, h, i, w)
    assert calls == ["batched_mm", "grouped_mm", "unsloth"]


def test_decode_switch_wrapper_counts_only_unsloth_models_off_cpu(monkeypatch):
    import contextlib
    import unsloth_zoo.temporary_patches.moe_experts_interface as MEI
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:
        pytest.skip("no GenerationMixin")
    seen = []

    @contextlib.contextmanager
    def original(self, *args, **kwargs):
        seen.append(self.device.type)
        yield

    monkeypatch.setattr(GenerationMixin, "_optimize_model_for_decode", original, raising = False)
    MEI._patch_decode_switch()
    wrapped = GenerationMixin.__dict__["_optimize_model_for_decode"]
    assert getattr(wrapped, "_unsloth_patched", False)
    MEI._patch_decode_switch()
    assert GenerationMixin.__dict__["_optimize_model_for_decode"] is wrapped

    def model(device, impl):
        return types.SimpleNamespace(
            device = torch.device(device), get_experts_implementation = lambda: impl,
        )
    depths = []
    for device, impl in [("cuda", "unsloth"), ("cuda", {"text": "unsloth"}), ("cpu", "unsloth"), ("cuda", "grouped_mm")]:
        with wrapped(model(device, impl)):
            depths.append(MEI._DECODING_DEPTH)
        assert MEI._DECODING_DEPTH == 0
    assert depths == [1, 1, 0, 0]
    assert seen == ["cuda", "cuda", "cpu", "cuda"]
    with pytest.raises(RuntimeError):
        with wrapped(model("cuda", "unsloth")):
            raise RuntimeError
    assert MEI._DECODING_DEPTH == 0
