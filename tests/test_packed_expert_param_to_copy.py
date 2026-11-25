# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Packed experts take Tensor.to(copy=...) from compressed-tensors offload; torch refuses it."""
import pytest
import torch

pe = pytest.importorskip("unsloth_zoo.packed_experts")


@pytest.mark.parametrize("quantize", ["quantize_experts_nvfp4", "quantize_experts_mxfp8"])
def test_to_accepts_copy(quantize):
    w = torch.randn(2, 32, 64, generator = torch.Generator().manual_seed(0))
    p = getattr(pe, quantize)(w)
    assert p.to("cpu", copy = False) is p and p.to(torch.float32, copy = False) is p
    q = p.to("cpu", copy = True)
    assert type(q) is type(p) and q.data_ptr() != p.data_ptr() and torch.equal(q.data, p.data)
    assert torch.equal(q.packed_scales, p.packed_scales) and q.packed_scales.data_ptr() != p.packed_scales.data_ptr()
    torch.testing.assert_close(q.dequantize(), p.dequantize())
