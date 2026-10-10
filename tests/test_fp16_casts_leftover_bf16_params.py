# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A float16 load leaves no bfloat16 parameters behind (unsloth#3459).

A pre-quantized checkpoint keeps its skipped layers and biases in the bfloat16 it was saved in.
On a T4 Inductor skips every graph reading one, so a non-reentrant checkpoint recompute stopped
matching its forward and multi-GPU training died at step 0.
"""

import pytest
import torch

pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")


def _model():
    model = torch.nn.Module()
    model.skipped = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.kept = torch.nn.Linear(4, 4).to(torch.float16)
    model.norm = torch.nn.LayerNorm(4).to(torch.float32)
    model.pinned = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.pinned.weight._pre_set_compute_dtype = torch.bfloat16
    # A 4-bit weight packed into bfloat16 storage: its bits are not bfloat16 values.
    model.packed = torch.nn.Linear(4, 4, bias = False).to(torch.bfloat16)
    model.packed.weight.quant_state = object()
    return model


def _patch(model, dtype):
    from unsloth_zoo.patching_utils import patch_model_and_tokenizer

    patch_model_and_tokenizer(
        model, None, downcast_rope=False, fix_embeddings=False, correct_dtype=dtype
    )


def test_float16_load_casts_leftover_bfloat16_params():
    model = _model()
    before = model.skipped.weight.detach().float().clone()
    _patch(model, torch.float16)
    assert model.skipped.weight.dtype == torch.float16
    assert model.skipped.bias.dtype == torch.float16
    assert torch.equal(model.skipped.weight.float(), before)
    assert model.norm.weight.dtype == torch.float32
    assert model.pinned.weight.dtype == torch.bfloat16
    assert model.packed.weight.dtype == torch.bfloat16


def test_bfloat16_load_is_untouched():
    model = _model()
    _patch(model, torch.bfloat16)
    assert model.skipped.weight.dtype == torch.bfloat16
    assert model.kept.weight.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_large_leftover_param_takes_the_bounded_cast(monkeypatch):
    import unsloth_zoo.patching_utils as pu

    calls = []
    real = pu._cast_large_param
    monkeypatch.setattr(pu, "_FORCED_FLOAT32_STAGE_BYTES", 0)
    monkeypatch.setattr(pu, "_cast_large_param", lambda p, d: (calls.append(d), real(p, d)))
    model = _model().cuda()
    before = model.skipped.weight.detach().float().clone()
    _patch(model, torch.float16)
    assert calls and set(calls) == {torch.float16}
    assert model.skipped.weight.dtype == torch.float16
    assert torch.equal(model.skipped.weight.float(), before)
    assert model.pinned.weight.dtype == torch.bfloat16
    assert model.packed.weight.dtype == torch.bfloat16
