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
import types

import pytest

from unsloth_zoo import vllm_utils
from unsloth_zoo.vllm_utils import _moe_expert_lora_refusal_reason, patch_vllm_stacked_moe_lora


class _FakeQwen3Moe:
    is_3d_moe_weight = False


def _install(monkeypatch, *, has_3d = True, hf_stacked = True):
    vllm_cls = type("Qwen3MoeForCausalLM", (_FakeQwen3Moe,), {})
    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.lora": types.ModuleType("vllm.lora"),
        "vllm.lora.layers": types.ModuleType("vllm.lora.layers"),
        "vllm.lora.layers.fused_moe": types.ModuleType("vllm.lora.layers.fused_moe"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": types.ModuleType("vllm.model_executor.models"),
        "vllm.model_executor.models.qwen3_moe": types.ModuleType("vllm.model_executor.models.qwen3_moe"),
        "transformers.models.qwen3_moe.modeling_qwen3_moe": types.ModuleType("modeling_qwen3_moe"),
    }
    if has_3d:
        mods["vllm.lora.layers.fused_moe"].FusedMoE3DWithLoRA = type("FusedMoE3DWithLoRA", (), {})
    mods["vllm.model_executor.models.qwen3_moe"].Qwen3MoeForCausalLM = vllm_cls
    if hf_stacked:
        mods["transformers.models.qwen3_moe.modeling_qwen3_moe"].Qwen3MoeExperts = type("Qwen3MoeExperts", (), {})
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return vllm_cls


def _config(model_type):
    return types.SimpleNamespace(model_type = model_type)


def test_qwen3_moe_is_marked_3d(monkeypatch):
    vllm_cls = _install(monkeypatch)
    assert patch_vllm_stacked_moe_lora(_config("qwen3_moe")) is True
    assert vllm_cls.is_3d_moe_weight is True
    assert _FakeQwen3Moe.is_3d_moe_weight is False
    assert patch_vllm_stacked_moe_lora(_config("qwen3_moe")) is True
    assert vllm_cls.is_3d_moe_weight is True


def test_marked_class_clears_the_expert_lora_refusal(monkeypatch):
    vllm_cls = _install(monkeypatch)
    vllm_model = vllm_cls()
    monkeypatch.setattr(vllm_utils, "_get_vllm_model_runner", lambda model: None)
    monkeypatch.setattr(vllm_utils, "_get_vllm_lora_model", lambda model, runner: vllm_model)
    monkeypatch.setattr(vllm_utils, "_vllm_mixed_moe_lora_enabled", lambda *a: False)
    monkeypatch.setattr(vllm_utils, "_is_bitsandbytes_quantized", lambda model: False)
    peft_config = types.SimpleNamespace(r = 16, rank_pattern = {})
    assert "is_3d_moe_weight" in _moe_expert_lora_refusal_reason(object(), peft_config)
    patch_vllm_stacked_moe_lora(_config("qwen3_moe"))
    assert _moe_expert_lora_refusal_reason(object(), peft_config) is None


@pytest.mark.parametrize("kwargs", [dict(has_3d = False), dict(hf_stacked = False)])
def test_not_marked_without_3d_wrapper_or_stacked_hf_experts(monkeypatch, kwargs):
    vllm_cls = _install(monkeypatch, **kwargs)
    assert patch_vllm_stacked_moe_lora(_config("qwen3_moe")) is False
    assert vllm_cls.is_3d_moe_weight is False


@pytest.mark.parametrize("model_type", ["qwen3", "qwen2_moe", "gpt_oss", None])
def test_other_model_types_untouched(monkeypatch, model_type):
    vllm_cls = _install(monkeypatch)
    assert patch_vllm_stacked_moe_lora(_config(model_type)) is False
    assert vllm_cls.is_3d_moe_weight is False


@pytest.mark.parametrize("model_type,vllm_class,hf_module,hf_experts", [
    ("deepseek_v2", "DeepseekV2ForCausalLM", "deepseek_v2.modeling_deepseek_v2", "DeepseekV2Experts"),
    ("deepseek_v3", "DeepseekV3ForCausalLM", "deepseek_v3.modeling_deepseek_v3", "DeepseekV3Experts"),
])
@pytest.mark.parametrize("hf_stacked", [True, False])
def test_deepseek_marked_3d_only_with_stacked_hf_experts(monkeypatch, model_type, vllm_class, hf_module, hf_experts, hf_stacked):
    # vLLM's deepseek_v2.py holds both classes; transformers >= 5 stacks DeepSeek experts like Qwen3-MoE.
    vllm_cls = type(vllm_class, (_FakeQwen3Moe,), {})
    vllm_mod = types.ModuleType("vllm.model_executor.models.deepseek_v2")
    setattr(vllm_mod, vllm_class, vllm_cls)
    hf_mod = types.ModuleType(hf_module)
    if hf_stacked:
        setattr(hf_mod, hf_experts, type(hf_experts, (), {}))
    fused_moe = types.ModuleType("vllm.lora.layers.fused_moe")
    fused_moe.FusedMoE3DWithLoRA = type("FusedMoE3DWithLoRA", (), {})
    for name, mod in {
        "vllm.lora.layers.fused_moe": fused_moe,
        "vllm.model_executor.models.deepseek_v2": vllm_mod,
        f"transformers.models.{hf_module}": hf_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    assert patch_vllm_stacked_moe_lora(_config(model_type)) is hf_stacked
    assert vllm_cls.is_3d_moe_weight is hf_stacked
    assert _FakeQwen3Moe.is_3d_moe_weight is False
