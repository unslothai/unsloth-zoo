import torch
import torch.nn as nn
import pytest

from unsloth_zoo.temporary_patches.moe_utils import (
    _get_moe_lora_io_dims,
    patch_param_wrapper_for_moe,
    patch_gpt_oss_grouped_mm_format,
)


class GptOssExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.empty(4, 16, 24))
        self.down_proj = nn.Parameter(torch.empty(4, 12, 16))


class Qwen3MoeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.empty(4, 24, 16))
        self.down_proj = nn.Parameter(torch.empty(4, 16, 12))


class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.empty(4, 16, 24))
        self.down_proj = nn.Parameter(torch.empty(4, 12, 16))


class Mxfp4GptOssExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.empty(4, 24, 16))
        self.down_proj = nn.Parameter(torch.empty(4, 16, 12))


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        return {"model_type": self.model_type}


class _Model(nn.Module):
    def __init__(self, model_type, experts=None):
        super().__init__()
        self.config = _Config(model_type)
        if experts is not None:
            self.experts = experts


class _Wrapper:
    def __init__(self, parameter_name, base_layer):
        self.parameter_name = parameter_name
        self.base_layer = base_layer

    def get_base_layer(self):
        return self.base_layer


class _PeftLikeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = model
        self.config = _Config("lora")


def _unwrap_base_layer(module):
    while hasattr(module, "base_layer"):
        module = module.base_layer
    return module


def test_stock_gpt_oss_experts_get_grouped_mm_flag_and_dims():
    experts = GptOssExperts()
    model = _Model("gpt_oss", experts)

    assert not hasattr(experts, "_unsloth_grouped_mm_format")
    assert patch_gpt_oss_grouped_mm_format(model) == 1
    assert experts._unsloth_grouped_mm_format is True

    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)


def test_peft_get_peft_model_marks_nested_gpt_oss_experts_and_dims():
    peft = pytest.importorskip("peft")
    LoraConfig = peft.LoraConfig

    try:
        config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            bias="none",
        )
    except TypeError:
        pytest.skip("Installed PEFT does not support target_parameters")

    model = _Model("gpt_oss", GptOssExperts())
    patch_param_wrapper_for_moe()

    peft_model = peft.get_peft_model(model, config)
    experts = _unwrap_base_layer(model.experts)

    assert experts._unsloth_grouped_mm_format is True

    wrappers = {
        module.parameter_name: module
        for module in peft_model.modules()
        if module.__class__.__name__ == "ParamWrapper"
        and getattr(module, "parameter_name", None) in ("gate_up_proj", "down_proj")
    }
    assert set(wrappers) == {"gate_up_proj", "down_proj"}
    assert _get_moe_lora_io_dims(wrappers["gate_up_proj"]) == (16, 24)
    assert _get_moe_lora_io_dims(wrappers["down_proj"]) == (12, 16)


def test_qwen3_moe_experts_do_not_get_gpt_oss_grouped_mm_flag():
    experts = Qwen3MoeExperts()
    model = _Model("qwen3_moe", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(experts, "_unsloth_grouped_mm_format")

    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)
    assert not hasattr(experts, "_unsloth_grouped_mm_format")


def test_qwen3_vl_grouped_mm_layout_still_requires_explicit_flag():
    experts = Qwen3VLMoeTextExperts()
    model = _Model("qwen3_vl_moe", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(experts, "_unsloth_grouped_mm_format")

    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (24, 16)

    experts._unsloth_grouped_mm_format = True
    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)


def test_mxfp4_gpt_oss_experts_name_is_not_treated_as_bf16_gpt_oss():
    experts = Mxfp4GptOssExperts()
    model = _Model("gpt_oss", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(experts, "_unsloth_grouped_mm_format")

    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)


def test_gpt_oss_experts_name_is_gated_by_model_type():
    experts = GptOssExperts()
    model = _Model("qwen3_moe", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(experts, "_unsloth_grouped_mm_format")


def test_gpt_oss_detection_unwraps_peft_like_model():
    experts = GptOssExperts()
    model = _Model("gpt_oss", experts)
    wrapped_model = _PeftLikeWrapper(model)

    assert patch_gpt_oss_grouped_mm_format(wrapped_model) == 1
    assert experts._unsloth_grouped_mm_format is True


def test_existing_gpt_oss_grouped_mm_flag_is_preserved():
    experts = GptOssExperts()
    experts._unsloth_grouped_mm_format = True
    model = _Model("gpt-oss", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert experts._unsloth_grouped_mm_format is True
    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)


def test_model_without_experts_does_not_raise_or_set_flags():
    model = _Model("gpt_oss")
    model.dense = nn.Linear(2, 2)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(model, "_unsloth_grouped_mm_format")
    assert not hasattr(model.dense, "_unsloth_grouped_mm_format")


def test_lazy_path_does_not_flag_qwen_layout_named_gpt_oss():
    class GptOssExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 4
            self.gate_up_proj = nn.Parameter(torch.empty(4, 24, 16))
            self.down_proj = nn.Parameter(torch.empty(4, 16, 12))

    experts = GptOssExperts()
    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)
    assert not hasattr(experts, "_unsloth_grouped_mm_format")


def test_name_or_path_matches_only_final_path_component():
    from unsloth_zoo.temporary_patches.moe_utils import _is_gpt_oss_model

    class _NamedConfig:
        def __init__(self, name):
            self.model_type = "unknown"
            self._name_or_path = name

    class _NamedModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.config = _NamedConfig(name)

    assert _is_gpt_oss_model(_NamedModel("unsloth/gpt-oss-20b-BF16"))
    assert _is_gpt_oss_model(_NamedModel("/data/models/gpt-oss-20b/"))
    assert _is_gpt_oss_model(_NamedModel("C:\\models\\gpt-oss-20b"))
    assert not _is_gpt_oss_model(_NamedModel("/data/gpt-oss-tests/qwen3-7b"))
    assert not _is_gpt_oss_model(_NamedModel("/home/u/my-gpt-oss-runs/Qwen3-30B-A3B"))
