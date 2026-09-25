# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""MXFP4 LoRA forward must not shadow the training autograd path when no LoRA is attached."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def test_mxfp4_lora_forward_without_adapters_calls_original_forward(monkeypatch):
    """No LoRA: delegate to the saved class forward (training vs inference dispatch)."""
    pytest.importorskip("transformers.integrations.mxfp4")

    gpt_oss_mod = importlib.import_module("unsloth_zoo.temporary_patches.gpt_oss")
    mxfp4_mod = importlib.import_module("unsloth_zoo.temporary_patches.mxfp4")

    monkeypatch.setattr(mxfp4_mod, "is_triton_kernels_available", lambda: True)

    import transformers.integrations.mxfp4 as mxfp4_mod_tf

    experts_cls = getattr(mxfp4_mod_tf, "Mxfp4GptOssExperts", None)
    if experts_cls is None:
        pytest.skip("Mxfp4GptOssExperts not in this transformers build")

    calls = []

    def _stub_original(self, hidden_states, routing_data, gather_idx, scatter_idx):
        calls.append(hidden_states.requires_grad)
        return hidden_states

    experts_cls._original_forward = _stub_original

    mod = experts_cls.__new__(experts_cls)
    mod.num_experts = 1
    mod.alpha = 1.0
    mod.limit = 7.0
    mod.gate_up_proj_precision_config = None
    mod.down_proj_precision_config = None

    hs = torch.zeros(4, 8, requires_grad=True)
    out = gpt_oss_mod.forward_mxfp4_gpt_oss_with_lora(mod, hs, None, None, None)
    assert calls == [True]
    assert out is hs


def test_mxfp4_lora_patch_stores_original_forward_on_class(monkeypatch):
    """patch_mxfp4_gpt_oss_for_lora must save the pre-LoRA forward on the experts class."""
    pytest.importorskip("transformers.integrations.mxfp4")

    gpt_oss_mod = importlib.import_module("unsloth_zoo.temporary_patches.gpt_oss")
    mxfp4_mod = importlib.import_module("unsloth_zoo.temporary_patches.mxfp4")
    monkeypatch.setattr(mxfp4_mod, "is_triton_kernels_available", lambda: True)

    import transformers.integrations.mxfp4 as mxfp4_mod_tf

    experts_cls = getattr(mxfp4_mod_tf, "Mxfp4GptOssExperts", None)
    if experts_cls is None:
        pytest.skip("Mxfp4GptOssExperts not in this transformers build")

    sentinel = object()
    experts_cls.forward = sentinel
    experts_cls._unsloth_mxfp4_lora_patched = False
    experts_cls._original_forward = None

    gpt_oss_mod.patch_mxfp4_gpt_oss_for_lora()

    assert getattr(experts_cls, "_original_forward", None) is sentinel
