"""FP4-packed experts (DeepSeek-V4) dequantize to the same values transformers produces.

DeepSeek-V4-Flash ships its experts as `config.expert_dtype = "fp4"`: int8 bytes
holding two e2m1 nibbles with per-row UE8M0 scales every 32 values. The zoo's
FP8 MoE backend only knew float8_e4m3fn weights, so every dequant helper
returned None and the dispatcher fell through to the per-expert fp8_linear loop,
which refuses to run with LoRA attached.
"""
import pytest
import torch

pytest.importorskip("transformers")


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


def test_fp4_unpack_matches_transformers_reference():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _dequantize_full_expert_weights_fp4
    packed, scale = _fixture()
    out = _dequantize_full_expert_weights_fp4(packed, scale, torch.float32)
    assert out.shape == (3, 8, 64)
    torch.testing.assert_close(out, _reference_dequant(packed, scale, torch.float32))


def test_the_generic_entry_point_now_handles_fp4():
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import _dequantize_full_expert_weights
    packed, scale = _fixture()
    out = _dequantize_full_expert_weights(packed, scale, torch.bfloat16)
    assert out is not None and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), _reference_dequant(packed, scale, torch.bfloat16).float())


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
