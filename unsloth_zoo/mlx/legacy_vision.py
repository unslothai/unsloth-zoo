# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the GNU Affero General Public License, version 3 or later.

import inspect
import math
import types

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import BaseImageProcessor

_SPEC = "_unsloth_legacy_image_spec"


def bind_legacy_image_processor(model, processor):
    image_processor = getattr(processor, "image_processor", None)
    if hasattr(processor, "tokenizer") or not isinstance(image_processor, BaseImageProcessor):
        return
    from PIL import Image
    from .utils import _config_get, _preserved_preprocessing_rng

    merge_name = "_prepare_inputs_for_multimodal"
    merge = getattr(model, merge_name, None)
    token_id = _config_get(getattr(model, "config", None), "image_token_index")
    if (not callable(merge) or token_id is None
            or tuple(inspect.signature(merge).parameters) !=
            ("image_features", "inputs_embeds", "input_ids")):
        return
    shapes = []

    def capture(self, image_features, inputs_embeds, input_ids):
        shapes.append(image_features.shape)
        return inputs_embeds

    namespace = vars(model)
    existed, previous = merge_name in namespace, namespace.get(merge_name)
    try:
        object.__setattr__(model, merge_name, types.MethodType(capture, model))
        with _preserved_preprocessing_rng():
            pixels = np.stack(image_processor.preprocess([Image.new("RGB", (32, 32))]))
            model.get_input_embeddings(mx.array([[0, token_id, 0]]), mx.array(pixels))
    finally:
        if existed:
            object.__setattr__(model, merge_name, previous)
        else:
            object.__delattr__(model, merge_name)
    if len(shapes) != 1 or len(shapes[0]) not in (2, 3) or math.prod(shapes[0][:-1]) < 1:
        raise ValueError("Unsloth MLX: legacy image preprocessing requires one nonempty projected feature sequence.")
    count = math.prod(shapes[0][:-1])
    setattr(processor, _SPEC, (int(token_id), count, tuple(pixels.shape[1:])))
    model._unsloth_legacy_image_token_count = count


def legacy_image_inputs(processor, texts, all_images, max_seq_length, truncation=True):
    if hasattr(processor, "tokenizer") or not isinstance(getattr(processor, "image_processor", None), BaseImageProcessor) or not any(all_images):
        return None
    from mlx_vlm.utils import prepare_inputs
    from .utils import _expand_image_token_sequences

    spec = getattr(processor, _SPEC, None)
    if spec is None:
        raise ValueError("Unsloth MLX: legacy image training requires an expanded-image merge; load with patch_mode='patched'.")
    token_id, count, pixel_shape = spec
    if (not all(len(images) == 1 for images in all_images)
            or not all(text.count("<image>") == 1 for text in texts)):
        raise ValueError("Unsloth MLX: legacy image rows require exactly one image and <image> placeholder.")
    inputs = prepare_inputs(
        processor, images=[images[0] for images in all_images],
        prompts=texts, image_token_index=token_id,
    )
    if tuple(inputs["pixel_values"].shape[1:]) != pixel_shape:
        raise ValueError("Unsloth MLX: variable image shapes require a processor that expands its own image tokens.")
    ids, mask = _expand_image_token_sequences(
        inputs["input_ids"], inputs["attention_mask"], token_id, count,
    )
    if truncation and max_seq_length and ids.shape[1] > max_seq_length:
        side = getattr(processor, "truncation_side", "right")
        columns = slice(-max_seq_length, None) if side == "left" else slice(0, max_seq_length)
        ids, mask = ids[:, columns], mask[:, columns]
    inputs["input_ids"], inputs["attention_mask"] = ids, mask
    inputs[_SPEC] = (token_id, count)
    validate_legacy_image_batch(inputs)
    return inputs


def validate_legacy_image_batch(batch):
    spec = batch.get(_SPEC)
    if spec is None:
        return
    token_id, count = spec
    ids = np.asarray(batch["input_ids"])
    retained = (ids == token_id).sum(axis=1)
    if np.any(retained != count):
        raise ValueError(
            f"Unsloth MLX: truncation split or removed an image span "
            f"({count} visual tokens required per row, retained {retained.tolist()}). "
            "Increase max_seq_length."
        )
