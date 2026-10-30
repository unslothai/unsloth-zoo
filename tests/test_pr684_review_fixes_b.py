"""Regression pins for review-fix cluster B: VLM sliding masks, tall resize, wide labels, NF4 scales."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_torch_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# --- Thread 1: gemma sliding-window settings read from config ---------------


def test_run_hidden_stack_reads_sliding_window_from_config():
    """Sliding settings live only on .config, so windowed layers still need the sliding token-type mask."""
    import mlx.core as mx
    import unsloth_zoo.mlx.utils as mutils

    class RecordingLayer:
        def __init__(self):
            self.seen_mask = None

        def __call__(self, h, mask, _cache):
            self.seen_mask = mask
            return h

    class IdentityNorm:
        def __call__(self, h):
            return h

    layer_global = RecordingLayer()
    layer_sliding = RecordingLayer()
    # sliding_window_pattern=2 -> layer index 0 is sliding, index 1 is global.
    stack = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma3_text",
            hidden_size=4,
            sliding_window_pattern=2,
            sliding_window=1,
        ),
        embed_tokens=object(),
        layers=[layer_sliding, layer_global],
        norm=IdentityNorm(),
    )
    assert not hasattr(stack, "sliding_window_pattern")
    assert not hasattr(stack, "window_size")

    embeds = mx.ones((1, 4, 4), dtype=mx.float32)
    token_type_ids = mx.array([[0, 1, 1, 0]], dtype=mx.int32)

    mutils._run_hidden_stack(
        stack,
        mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        inputs_embeds=embeds,
        token_type_ids=token_type_ids,
        attention_mask=mx.array([[1, 1, 1, 1]], dtype=mx.int32),
    )

    # Both layers get a 4x4 token-type mask, not a 1D causal mask, which proves the config fallback ran.
    assert layer_sliding.seen_mask is not None
    assert layer_global.seen_mask is not None
    sliding = layer_sliding.seen_mask[0, 0]
    global_mask = layer_global.seen_mask[0, 0]
    # The sliding mask is strictly tighter than the global one: with window=1 query 3 cannot see token 0, but the global
    # layer can; this differs only if config fallback set the window.
    assert bool(global_mask[3][0].item()) is True
    assert bool(sliding[3][0].item()) is False
    # The image block stays bidirectional in both masks.
    assert bool(sliding[1][2].item()) is True
    assert bool(global_mask[1][2].item()) is True


def test_run_hidden_stack_skips_sliding_when_config_lacks_window():
    """No sliding settings anywhere -> every layer uses the global mask."""
    import mlx.core as mx
    import unsloth_zoo.mlx.utils as mutils

    class RecordingLayer:
        def __init__(self):
            self.seen_mask = None

        def __call__(self, h, mask, _cache):
            self.seen_mask = mask
            return h

    class IdentityNorm:
        def __call__(self, h):
            return h

    layer = RecordingLayer()
    stack = SimpleNamespace(
        config=SimpleNamespace(model_type="gemma3_text", hidden_size=4),
        embed_tokens=object(),
        layers=[layer],
        norm=IdentityNorm(),
    )
    embeds = mx.ones((1, 4, 4), dtype=mx.float32)
    token_type_ids = mx.array([[0, 1, 1, 0]], dtype=mx.int32)

    mutils._run_hidden_stack(
        stack,
        mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        inputs_embeds=embeds,
        token_type_ids=token_type_ids,
        attention_mask=mx.array([[1, 1, 1, 1]], dtype=mx.int32),
    )
    # Image block stays bidirectional (global token-type mask).
    assert bool(layer.seen_mask[0, 0][2][1].item()) is True


# --- Thread 2: downscale tall VLM images ------------------------------------


def test_resize_downscales_tall_portrait_images():
    from PIL import Image

    from unsloth_zoo.mlx.utils import _resize_vlm_images

    # 512x2048 portrait with a 512 cap: width fits but the longer side (2048) must still shrink.
    image = Image.new("RGB", (512, 2048))
    resized = _resize_vlm_images([image], 512)

    assert resized[0].size == (128, 512)


# --- Thread 3: preserve wide labels until CCE validation --------------------


def test_apply_vlm_label_masks_preserves_wide_unsigned_ids():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import (
        _RAW_INPUT_IDS_FOR_LABELS,
        _apply_vlm_label_masks,
    )

    # A uint32 id near the top wraps negative as int32, so the label must stay wide (positive sentinel).
    wide_id = (2 ** 32) - 100
    raw = mx.array([[5, wide_id, 7]], dtype=mx.uint32)
    batch = {
        "input_ids": raw.astype(mx.int32),
        "attention_mask": mx.array([[1, 1, 1]], dtype=mx.int32),
        _RAW_INPUT_IDS_FOR_LABELS: raw.astype(mx.int64),
    }

    labels = _apply_vlm_label_masks(batch, ignore_token_ids=None)

    values = labels[0].tolist()
    assert values[1] == wide_id
    assert values[0] == 5
    assert values[2] == 7


def test_apply_vlm_label_masks_does_not_narrow_explicit_labels():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _apply_vlm_label_masks

    wide_id = (2 ** 32) - 50
    explicit = mx.array([[3, wide_id, 9]], dtype=mx.int64)
    batch = {
        "input_ids": mx.array([[3, 1, 9]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1]], dtype=mx.int32),
    }

    labels = _apply_vlm_label_masks(batch, labels=explicit, ignore_token_ids=None)

    assert labels[0].tolist()[1] == wide_id


def test_apply_vlm_label_masks_still_masks_padding_and_ignore_tokens():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _apply_vlm_label_masks

    batch = {
        "input_ids": mx.array([[10, 200, 11, 0]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 0]], dtype=mx.int32),
    }
    labels = _apply_vlm_label_masks(batch, ignore_token_ids=[200])

    # image token (200) and padding position both become -100.
    assert labels[0].tolist() == [10, -100, 11, -100]


# --- Thread 4: strip collator-only markers before VLM forward ---------------


def test_baseline_loss_fn_strips_collator_position_marker():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import make_vlm_baseline_loss_fn

    seen_kwargs = {}

    class FakeModel:
        config = SimpleNamespace(model_type="qwen2_vl")

        def __call__(self, inputs, pixel_values=None, mask=None, **kwargs):
            # Reject every unknown kwarg the way a strict mlx-vlm model would.
            allowed = {"position_ids"}
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    f"unexpected keyword arguments: {sorted(unexpected)}"
                )
            seen_kwargs.update(kwargs)
            vocab = 16
            seq = inputs.shape[1]
            logits = mx.zeros((inputs.shape[0], seq, vocab), dtype=mx.float32)
            return SimpleNamespace(logits=logits)

    loss_fn = make_vlm_baseline_loss_fn(model=None, ignore_token_ids=[])
    model = FakeModel()
    batch = {
        "input_ids": mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 1]], dtype=mx.int32),
        "labels": mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        "position_ids": mx.array([[0, 1, 2, 3]], dtype=mx.int32),
        "_unsloth_collated_position_ids": True,
    }

    # Must not raise: the private marker is stripped, position_ids is kept.
    loss, _ntoks = loss_fn(model, batch)
    assert "_unsloth_collated_position_ids" not in seen_kwargs
    assert "position_ids" in seen_kwargs
    assert float(loss.item()) >= 0.0


# --- Thread 5: keep NF4 scales un-nested unless requested -------------------


def test_nf4_dense_default_keeps_plain_absmax():
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import _nf4_dense_dequantize_weight

    torch.manual_seed(0)
    weight = (torch.randn(8, 64, dtype=torch.float32) * 0.3)

    default = _nf4_dense_dequantize_weight(weight, group_size=64)
    plain = _nf4_dense_dequantize_weight(weight, group_size=64, use_double_quant=False)
    nested = _nf4_dense_dequantize_weight(weight, group_size=64, use_double_quant=True)

    # Default equals plain un-nested absmax; nested (double-quant) absmax re-quantizes the scales.
    assert torch.allclose(mx.array(default), mx.array(plain))
    assert not torch.allclose(mx.array(default), mx.array(nested))


def test_apply_dense_nf4_quantization_uses_plain_scales():
    import mlx.core as mx
    import unsloth_zoo.mlx.loader as mloader
    from unsloth_zoo.mlx.loader import (
        _MLXQuantizationSpec,
        _apply_dense_nf4_quantization,
        _nf4_dense_dequantize_weight,
    )

    torch.manual_seed(1)
    w = torch.randn(8, 64, dtype=torch.float32) * 0.2

    class LinearLike:
        def __init__(self, weight):
            self.weight = weight

    module = LinearLike(w.clone())

    class FakeModel:
        def __init__(self, module):
            self._module = module

        def named_modules(self):
            return [("layers.0.mlp", self._module)]

    spec = _MLXQuantizationSpec(
        enabled=True, bits=4, group_size=64, mode="nf4_dense",
        source="mlx_quantization_config",
    )
    model = FakeModel(module)
    _apply_dense_nf4_quantization(
        model, {}, spec, predicate=lambda path, mod: True,
    )

    expected = _nf4_dense_dequantize_weight(w.clone(), 64)
    assert torch.allclose(mx.array(module.weight), mx.array(expected))
