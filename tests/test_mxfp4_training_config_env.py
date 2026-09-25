# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""UNSLOTH_MXFP4_NO_DEQUANTIZE via Unsloth load entry points (unsloth-zoo #1251 §2)."""

from __future__ import annotations

import importlib

import pytest

from unsloth_zoo.temporary_patches.mxfp4 import (
    apply_mxfp4_quantization_config_to_from_pretrained_kwargs,
    get_mxfp4_config_for_training,
)


def test_get_mxfp4_config_for_training_respects_env(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mxfp4Config"):
        pytest.skip("Mxfp4Config not in this transformers build")

    mxfp4_mod = importlib.import_module("unsloth_zoo.temporary_patches.mxfp4")

    monkeypatch.setattr(mxfp4_mod, "UNSLOTH_MXFP4_NO_DEQUANTIZE", False)
    monkeypatch.setattr(mxfp4_mod, "is_triton_kernels_available", lambda: True)
    assert get_mxfp4_config_for_training().dequantize is True

    monkeypatch.setattr(mxfp4_mod, "UNSLOTH_MXFP4_NO_DEQUANTIZE", True)
    assert get_mxfp4_config_for_training().dequantize is False


def test_mxfp4_from_dict_not_globally_overridden(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mxfp4Config"):
        pytest.skip("Mxfp4Config not in this transformers build")

    cfg = transformers.Mxfp4Config.from_dict(
        {"quant_method": "mxfp4", "dequantize": False}
    )
    assert cfg.dequantize is False


def test_from_pretrained_helper_respects_explicit_quantization_config(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mxfp4Config"):
        pytest.skip("Mxfp4Config not in this transformers build")

    mxfp4_mod = importlib.import_module("unsloth_zoo.temporary_patches.mxfp4")
    monkeypatch.setattr(mxfp4_mod, "UNSLOTH_MXFP4_NO_DEQUANTIZE", True)
    explicit = transformers.Mxfp4Config(dequantize=False)
    kwargs = {"quantization_config": explicit}
    apply_mxfp4_quantization_config_to_from_pretrained_kwargs(
        "openai/gpt-oss-20b", kwargs
    )
    assert kwargs["quantization_config"] is explicit
    assert kwargs["quantization_config"].dequantize is False


def test_from_pretrained_helper_injects_only_when_env_set(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Mxfp4Config"):
        pytest.skip("Mxfp4Config not in this transformers build")

    mxfp4_mod = importlib.import_module("unsloth_zoo.temporary_patches.mxfp4")
    monkeypatch.setattr(mxfp4_mod, "is_triton_kernels_available", lambda: True)

    kwargs = {}
    monkeypatch.setattr(mxfp4_mod, "UNSLOTH_MXFP4_NO_DEQUANTIZE", False)
    apply_mxfp4_quantization_config_to_from_pretrained_kwargs(
        "openai/gpt-oss-20b", kwargs
    )
    assert "quantization_config" not in kwargs

    monkeypatch.setattr(mxfp4_mod, "UNSLOTH_MXFP4_NO_DEQUANTIZE", True)
    apply_mxfp4_quantization_config_to_from_pretrained_kwargs(
        "openai/gpt-oss-20b", kwargs
    )
    assert kwargs["quantization_config"].dequantize is False

    kwargs = {}
    apply_mxfp4_quantization_config_to_from_pretrained_kwargs(
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit", kwargs
    )
    assert "quantization_config" not in kwargs
