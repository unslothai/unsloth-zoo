# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A fused MoE expert stack quantized in place (bare Params4bit) dequantizes to quant_state.dtype, like a Linear4bit
weight. A pre-quantized checkpoint stores the dtype it was made in (unsloth/Inkling-2-Mini-unsloth-bnb-4bit:
bfloat16), so on a float16 GPU its experts dequantized in bfloat16: on Kaggle 2x T4, 36.5 s per step against 25.4 s
for the quantize-on-load path, 1.2 GB more memory per card and different losses. patch_model_and_tokenizer now
gives bare stacks the compute dtype, as it does for Linear4bit; forced float32 keeps them as loaded."""

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")


def _params4bit(value, module):
    data, state = bnb.functional.quantize_4bit(
        value.contiguous(), blocksize = 64, compress_statistics = True, quant_type = "nf4"
    )
    new = torch.Tensor._make_subclass(bnb.nn.Params4bit, data, False)
    new.quant_state = state
    new.blocksize = 64
    new.compress_statistics = True
    new.quant_type = "nf4"
    new.quant_storage = torch.uint8
    new.bnb_quantized = True
    new.module = module
    new._original_shape = value.shape
    return new


def _model(dtype = torch.bfloat16):
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.register_parameter(
        "gate_up_proj", _params4bit(torch.randn(4, 64, 32, dtype = dtype), model.experts)
    )
    model.experts.register_parameter(
        "down_proj", _params4bit(torch.randn(4, 32, 64, dtype = dtype), model.experts)
    )
    model.pinned = torch.nn.Module()
    model.pinned._pre_set_compute_dtype = torch.float32
    model.pinned.register_parameter("stack", _params4bit(torch.randn(2, 64, 64, dtype = dtype), model.pinned))
    return model


def _patch(model, dtype, forced = False):
    from unsloth_zoo.patching_utils import patch_model_and_tokenizer

    patch_model_and_tokenizer(
        model,
        None,
        downcast_rope = False,
        fix_embeddings = False,
        correct_dtype = dtype,
        do_forced_float32 = forced,
    )


def test_float16_load_dequantizes_bare_stacks_in_float16():
    model = _model(torch.bfloat16)
    want = bnb.functional.dequantize_4bit(model.experts.gate_up_proj.data, model.experts.gate_up_proj.quant_state)
    assert model.experts.gate_up_proj.quant_state.dtype == torch.bfloat16
    _patch(model, torch.float16)
    for p in (model.experts.gate_up_proj, model.experts.down_proj):
        assert p.quant_state.dtype == torch.float16
    assert model.pinned.stack.quant_state.dtype == torch.float32
    got = bnb.functional.dequantize_4bit(model.experts.gate_up_proj.data, model.experts.gate_up_proj.quant_state)
    assert got.dtype == torch.float16
    # Same codes and scales: only the output rounding changes.
    torch.testing.assert_close(got.float(), want.float(), rtol = 1e-2, atol = 1e-2)


def test_matching_dtype_and_forced_float32_are_untouched():
    model = _model(torch.bfloat16)
    _patch(model, torch.bfloat16)
    assert model.experts.gate_up_proj.quant_state.dtype == torch.bfloat16
    model = _model(torch.bfloat16)
    _patch(model, torch.bfloat16, forced = True)
    assert model.experts.gate_up_proj.quant_state.dtype == torch.bfloat16
    assert model.experts.down_proj.quant_state.dtype == torch.bfloat16


def test_feature_flag_for_unsloth():
    # unsloth reads this to decide whether a 4-bit load may use a pre-quantized repo with bare 4-bit stacks.
    from unsloth_zoo import patching_utils

    assert patching_utils.BARE_PARAMS4BIT_COMPUTE_DTYPE is True
