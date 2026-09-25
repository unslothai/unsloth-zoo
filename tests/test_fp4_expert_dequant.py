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

"""FP4-packed experts (DeepSeek-V4) dequantize to the same values transformers produces."""
import pytest
import torch

pytest.importorskip("transformers")


def _transformers_fp4_reference():
    try:
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    except Exception:
        return None
    return Fp8Dequantize if hasattr(Fp8Dequantize, "_dequantize_one") else None


needs_fp4_reference = pytest.mark.skipif(
    _transformers_fp4_reference() is None, reason = "this transformers has no FP4 expert dequant"
)


def _reference_dequant(packed, scale, dtype):
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    op = Fp8Dequantize.__new__(Fp8Dequantize)
    return torch.stack([op._dequantize_one(packed[e], scale[e], dtype) for e in range(packed.shape[0])])


def _fixture(E = 3, M = 8, K = 64, group = 32):
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (E, M, K // 2), dtype = torch.uint8).view(torch.int8)
    exponents = torch.randint(120, 134, (E, M, K // group), dtype = torch.uint8)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    scale = (exponents.to(torch.float32) - 127.0).exp2().to(e8m0) if e8m0 is not None else exponents
    return packed, scale


@needs_fp4_reference
def test_fp4_unpack_matches_transformers_reference():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _dequantize_full_expert_weights_fp4
    packed, scale = _fixture()
    out = _dequantize_full_expert_weights_fp4(packed, scale, torch.float32)
    assert out.shape == (3, 8, 64)
    torch.testing.assert_close(out, _reference_dequant(packed, scale, torch.float32))


@needs_fp4_reference
def test_the_generic_entry_point_now_handles_fp4():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _dequantize_full_expert_weights
    packed, scale = _fixture()
    out = _dequantize_full_expert_weights(packed, scale, torch.bfloat16)
    assert out is not None and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), _reference_dequant(packed, scale, torch.bfloat16).float())


@needs_fp4_reference
def test_float_scales_round_once_like_transformers():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _dequantize_full_expert_weights_fp4
    packed, _ = _fixture(E = 3, M = 16, K = 128)
    scale = torch.rand(3, 16, 4, generator = torch.Generator().manual_seed(1)) * 0.3 + 0.01
    out = _dequantize_full_expert_weights_fp4(packed, scale, torch.bfloat16)
    torch.testing.assert_close(out, _reference_dequant(packed, scale, torch.bfloat16), rtol = 0, atol = 0)

def test_chunking_over_experts_is_seamless():
    from unsloth_zoo.temporary_patches import moe_utils_fp8
    packed, scale = _fixture(E = 5)
    whole = moe_utils_fp8._dequantize_full_expert_weights_fp4(packed, scale, torch.float32)
    old = moe_utils_fp8._FP4_EXPERT_CHUNK
    moe_utils_fp8._FP4_EXPERT_CHUNK = 2
    try:
        chunked = moe_utils_fp8._dequantize_full_expert_weights_fp4(packed, scale, torch.float32)
    finally:
        moe_utils_fp8._FP4_EXPERT_CHUNK = old
    torch.testing.assert_close(whole, chunked)


def test_fp8_and_mismatched_scales_are_declined():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _dequantize_full_expert_weights_fp4
    packed, scale = _fixture()
    assert _dequantize_full_expert_weights_fp4(packed.view(torch.uint8).to(torch.float8_e4m3fn), scale, torch.float32) is None
    assert _dequantize_full_expert_weights_fp4(packed, scale[:, :4], torch.float32) is None


class _PackedExperts(torch.nn.Module):
    def __init__(self, E = 2, M = 8, K = 64):
        super().__init__()
        packed, scale = _fixture(E, M, K)
        self.gate_up_proj = torch.nn.Parameter(packed, requires_grad = False)
        self.gate_up_proj_scale_inv = torch.nn.Parameter(scale, requires_grad = False)
        self.num_experts = E

    def forward(self, x):
        return x


def test_peft_sizes_the_lora_on_the_logical_shape():
    pytest.importorskip("peft")
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import patch_peft_param_wrapper_fp4_expert_shape
    patch_peft_param_wrapper_fp4_expert_shape()
    from peft.tuners.lora.layer import ParamWrapper
    from peft import LoraConfig
    module = _PackedExperts()
    wrapper = ParamWrapper(module, "default", parameter_name = "gate_up_proj", config = LoraConfig(r = 4, target_parameters = ["gate_up_proj"]), r = 4)
    assert wrapper.num_experts == 2
    assert wrapper.get_param().shape == (2, 8, 64)
    assert module.gate_up_proj._original_shape == (2, 8, 64)
    # PEFT 0.21 swaps the 3-D in/out dims vs 0.18, so check the set, never the packed 32.
    dims = {wrapper.lora_A["default"].weight.shape[-1], wrapper.lora_B["default"].weight.shape[0]}
    assert dims == {8, 64}


def test_megamoe_fp4_experts_refuse_a_lora_its_kernel_would_skip():
    pytest.importorskip("peft")
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import patch_peft_param_wrapper_fp4_expert_shape
    patch_peft_param_wrapper_fp4_expert_shape()
    from peft.tuners.lora.layer import ParamWrapper
    from peft import LoraConfig
    module = _PackedExperts()
    module.config = SimpleNamespace(_experts_implementation = "deepgemm_megamoe")
    with pytest.raises(NotImplementedError, match = "deepgemm_megamoe"):
        ParamWrapper(module, "default", parameter_name = "gate_up_proj", config = LoraConfig(r = 4, target_parameters = ["gate_up_proj"]), r = 4)
    module.config = SimpleNamespace(_experts_implementation = "eager")
    with pytest.raises(NotImplementedError, match = "'eager'"):
        ParamWrapper(module, "default", parameter_name = "gate_up_proj", config = LoraConfig(r = 4, target_parameters = ["gate_up_proj"]), r = 4)
    for impl in ("grouped_mm", "batched_mm"):
        module.config = SimpleNamespace(_experts_implementation = impl)
        ParamWrapper(module, "default", parameter_name = "gate_up_proj", config = LoraConfig(r = 4, target_parameters = ["gate_up_proj"]), r = 4)


def test_fp4_experts_with_their_own_gate_are_refused_until_a_backend_applies_it():
    from unsloth_zoo.temporary_patches import moe_utils_fp8 as fp8

    packed = torch.zeros(2, 4, 8, dtype = torch.int8)

    class Clamped(torch.nn.Module):
        def _apply_gate(self, gate_up):
            return gate_up

    class Plain(torch.nn.Module):
        pass

    if "_fp8_experts_own_gate" in vars(fp8):
        fp8._refuse_fp4_with_an_unapplied_gate(Clamped(), packed)
        return
    with pytest.raises(NotImplementedError, match = "own gate"):
        fp8._refuse_fp4_with_an_unapplied_gate(Clamped(), packed)
    flagged = Clamped()
    flagged._unsloth_own_apply_gate = True
    fp8._refuse_fp4_with_an_unapplied_gate(flagged, packed)
    fp8._refuse_fp4_with_an_unapplied_gate(Plain(), packed)
    fp8._refuse_fp4_with_an_unapplied_gate(Clamped(), torch.zeros(2, 4, 8, dtype = torch.bfloat16))


def test_fp8_moe_forward_stays_compiler_disabled():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import forward_moe_backend_fp8
    assert getattr(forward_moe_backend_fp8, "_torchdynamo_disable", False)
