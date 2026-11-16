# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The dtype pass keeps storage-free stand-ins storage-free: a pipeline rank holds the other stages' weights as one
element expanded to the full shape, and a plain cast materializes the full tensor (OOM on a T4 rank for a 27B model)."""

import pytest
import torch

pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")


def _standin(shape, dtype):
    return torch.nn.Parameter(torch.zeros((), dtype = dtype).expand(shape), requires_grad = False)


def _model():
    model = torch.nn.Module()
    model.real = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.down_proj = torch.nn.Linear(64, 64, bias = False)
    model.down_proj.weight = _standin((64, 64), torch.bfloat16)
    model.norm = torch.nn.LayerNorm(64)
    model.norm.weight = _standin((64,), torch.bfloat16)
    model.pinned = torch.nn.Sequential(torch.nn.Linear(64, 64, bias = False))
    model.pinned[0].weight = _standin((64, 64), torch.bfloat16)
    model.pinned._pre_set_compute_dtype = torch.float32
    return model


def _storage_free(t):
    return t.numel() > 1 and not any(t.stride())


@pytest.mark.parametrize("forced", [False, True])
def test_dtype_pass_keeps_standins_storage_free(forced):
    from unsloth_zoo.patching_utils import patch_model_and_tokenizer

    model = _model()
    patch_model_and_tokenizer(
        model, None, downcast_rope = False, fix_embeddings = False,
        do_forced_float32 = forced, correct_dtype = torch.float16,
    )
    assert model.real.weight.dtype == torch.float16 and not _storage_free(model.real.weight)
    for t, dtype in (
        (model.down_proj.weight, torch.float16),
        (model.norm.weight, torch.float16),
        (model.pinned[0].weight, torch.float32),
    ):
        assert t.dtype == dtype and _storage_free(t) and t.shape[-1] == 64
