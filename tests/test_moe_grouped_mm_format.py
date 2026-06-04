import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches.moe_utils import (
    _get_moe_lora_io_dims,
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


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type


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


def test_stock_gpt_oss_experts_get_grouped_mm_flag_and_dims():
    experts = GptOssExperts()
    model = _Model("gpt_oss", experts)

    assert not hasattr(experts, "_unsloth_grouped_mm_format")
    assert patch_gpt_oss_grouped_mm_format(model) == 1
    assert experts._unsloth_grouped_mm_format is True

    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)


def test_qwen3_moe_experts_do_not_get_gpt_oss_grouped_mm_flag():
    experts = Qwen3MoeExperts()
    model = _Model("qwen3_moe", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(experts, "_unsloth_grouped_mm_format")

    assert _get_moe_lora_io_dims(_Wrapper("gate_up_proj", experts)) == (16, 24)
    assert _get_moe_lora_io_dims(_Wrapper("down_proj", experts)) == (12, 16)
    assert not hasattr(experts, "_unsloth_grouped_mm_format")


def test_gpt_oss_experts_name_is_gated_by_model_type():
    experts = GptOssExperts()
    model = _Model("qwen3_moe", experts)

    assert patch_gpt_oss_grouped_mm_format(model) == 0
    assert not hasattr(experts, "_unsloth_grouped_mm_format")


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
