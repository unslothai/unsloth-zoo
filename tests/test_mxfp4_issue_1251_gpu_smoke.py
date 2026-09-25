# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Optional GPU / triton_kernels smoke for unsloth-zoo #1251.

Skipped on CPU-only CI and developer laptops without OpenAI triton_kernels.
On a Blackwell/B200 box, run:

  UNSLOTH_IS_PRESENT=1 pytest tests/test_mxfp4_issue_1251_gpu_smoke.py -v
"""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.integrations.mxfp4")


def _has_triton_kernels() -> bool:
    return importlib.util.find_spec("triton_kernels") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_triton_kernels(), reason="triton_kernels required")
def test_mxfp4_dequantize_helper_matches_convertops_on_device():
    """Parity check on GPU for the §1 dequant helper (random micro tensors)."""
    import inspect

    import transformers.integrations.mxfp4 as mxfp4_mod

    from unsloth_zoo.temporary_patches.mxfp4 import (
        dequantize_mxfp4_moe_blocks_scales,
        patch_convert_moe_packed_tensors,
    )

    if not hasattr(mxfp4_mod, "dequantize_convertops"):
        pytest.skip("dequantize_convertops absent on this transformers version")

    patch_convert_moe_packed_tensors()

    convertops = mxfp4_mod.dequantize_convertops
    n_pos = len(
        [
            p
            for p in inspect.signature(convertops).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    )

    E, D, G, B = 2, 4, 3, 16
    device = torch.device("cuda")
    blocks = torch.randint(0, 255, (E, D, G, B), dtype=torch.uint8, device=device)
    scales = torch.full((E, D, G), 127, dtype=torch.uint8, device=device)

    if n_pos >= 3:
        want = convertops(blocks, scales, device)
    else:
        want = convertops(blocks, scales)
    want = want.data if isinstance(want, torch.nn.Parameter) else want

    got = dequantize_mxfp4_moe_blocks_scales(blocks, scales)
    assert tuple(got.shape) == tuple(want.shape)
    assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_triton_kernels(), reason="triton_kernels required")
def test_mxfp4_lora_forward_without_adapters_calls_original_forward():
    """§3: patched class forward must delegate when no LoRA wrappers exist."""
    import transformers.integrations.mxfp4 as mxfp4_mod

    from unsloth_zoo.temporary_patches.gpt_oss import forward_mxfp4_gpt_oss_with_lora

    experts_cls = getattr(mxfp4_mod, "Mxfp4GptOssExperts", None)
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

    hs = torch.zeros(4, 8, device="cuda", requires_grad=True)
    out = forward_mxfp4_gpt_oss_with_lora(mod, hs, None, None, None)
    assert calls == [True]
    assert out is hs
