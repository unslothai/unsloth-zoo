# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The native Pixtral tower must keep packed images independent on the FA2 path."""

import pytest
import torch
import torch.nn.functional as F


def test_pixtral_flash_attention_keeps_images_isolated(monkeypatch):
    pixtral = pytest.importorskip("transformers.models.pixtral.modeling_pixtral")
    if not any(getattr(pixtral.PixtralVisionModel, name, False)
               for name in ("_supports_flash_attn", "_supports_flash_attn_2")):
        pytest.skip("this Transformers version does not offer Pixtral FlashAttention")
    from transformers.integrations import flash_attention
    from unsloth_zoo.temporary_patches import pixtral as patch

    # Restore the class after applying the real patch; use a CPU reference only at
    # the CUDA-kernel boundary, keeping HF's attention dispatch and the vision tower.
    monkeypatch.setattr(patch, "patch_function", lambda cls, name, fn, **kw: monkeypatch.setattr(cls, name, fn))
    patch.patch_PixtralAttention()

    def flash_reference(q, k, v, attention_mask, *, position_ids, is_causal, softmax_scale, **kwargs):
        assert attention_mask is None and not is_causal
        assert position_ids.shape == (1, q.shape[1])
        image_ids = (position_ids == 0).cumsum(-1)
        mask = (image_ids[:, :, None] == image_ids[:, None, :]).unsqueeze(1)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask = mask, scale = softmax_scale,
        ).transpose(1, 2)

    monkeypatch.setattr(flash_attention, "_flash_attention_forward", flash_reference)
    torch.manual_seed(31)
    config = pixtral.PixtralVisionConfig(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
        num_attention_heads = 4, head_dim = 8, image_size = 8, patch_size = 2,
        attention_dropout = 0.0, attn_implementation = "sdpa",
    )
    model = pixtral.PixtralVisionModel(config).float().eval()
    pixels = torch.randn(2, 3, 8, 8, requires_grad = True)
    sizes = [(8, 8), (4, 8)]
    expected = model(pixels, image_sizes = sizes).last_hidden_state
    # Select FA2 after construction so CPU CI needs no installed CUDA kernel.
    config._attn_implementation = "flash_attention_2"
    actual = model(pixels, image_sizes = sizes).last_hidden_state
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual[:, 16:].square().sum(), pixels)[0]
    assert torch.count_nonzero(grad[0]) == 0
    assert torch.count_nonzero(grad[1]) > 0
    changed = pixels.detach().clone()
    changed[0] += 1
    with torch.no_grad():
        after = model(changed, image_sizes = sizes).last_hidden_state
    torch.testing.assert_close(after[:, 16:], actual[:, 16:], rtol = 0, atol = 0)
