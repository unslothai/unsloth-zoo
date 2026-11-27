# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""CPU checks of the fused NVFP4 / MXFP8 grouped GEMM auto cutoff: the per-architecture default rows per expert, the
env override, and that CPU inputs never take the fused path."""

import pytest
import torch

from unsloth_zoo import mxfp4_gemm as mg
from unsloth_zoo import packed_experts as pe


@pytest.mark.parametrize("capability, rows", [
    ((8, 0), 32), ((8, 6), 64), ((8, 7), 64), ((8, 9), 64), ((9, 0), 32), ((10, 0), 32), ((12, 0), 192), ((12, 1), 192),
])
def test_default_cutoff_per_architecture(capability, rows):
    assert mg._packed_default_max_rows(capability) == rows


def test_env_overrides_cutoff(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED_MAX_ROWS", "7")
    assert mg.packed_grouped_mm_max_rows("cpu") == 7
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED_MAX_ROWS", "-3")
    assert mg.packed_grouped_mm_max_rows("cpu") == 0


@pytest.mark.parametrize("mode", ["auto", "1"])
def test_cpu_inputs_never_fuse(monkeypatch, mode):
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", mode)
    inputs = torch.zeros((4, 32), dtype = torch.bfloat16)
    codes = torch.zeros((2, 8, 16), dtype = torch.uint8)
    assert pe._fused_grouped_ok(inputs, codes, "nvfp4") is False
