# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MLX utilities for Apple Silicon training.

Provides loss functions (CCE and baseline CE), data batching,
weight extraction helpers, and model save/load/export for LoRA adapters
and merged models (safetensors, GGUF, HuggingFace Hub).
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import copy
import inspect
import json
import numpy as np
import os
import sys
import shutil
import tempfile
from pathlib import Path


from .mlx_cce import _get_runtime_cce


def _safe_token_denominator(ntoks):
    return mx.maximum(ntoks.astype(mx.float32), mx.array(1.0, dtype=mx.float32))


def _get_transformer_layers(model):
    """Find transformer layers, unwrapping VLM wrappers if needed.

    VLMs: model.language_model.model.layers
    Text: model.layers or model.model.layers
    """
    m = getattr(model, 'language_model', model)
    m = getattr(m, 'model', m)
    return getattr(m, 'layers', None)


def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to all transformer layers.

    Patches the layer class's __call__ to use mx.checkpoint, which
    recomputes activations during backward instead of storing them.
    Trades ~30% more compute for significant memory savings.

    Follows the same pattern as mlx_lm.tuner.trainer.grad_checkpoint.
    """
    layers = _get_transformer_layers(model)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    if getattr(layer_cls, '_orig_call', None) is not None:
        return  # already applied
    layer_cls._orig_call = layer_cls.__call__
    fn = layer_cls.__call__

    def checkpointed_fn(self, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            self.update(params)
            return fn(self, *args, **kwargs)
        return mx.checkpoint(inner_fn)(self.trainable_parameters(), *args, **kwargs)

    layer_cls.__call__ = checkpointed_fn


def remove_gradient_checkpointing(model):
    """Remove gradient checkpointing, restoring original layer __call__."""
    layers = _get_transformer_layers(model)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    orig = getattr(layer_cls, '_orig_call', None)
    if orig is not None:
        layer_cls.__call__ = orig
        del layer_cls._orig_call


def _get_text_model(model):
    """Get the inner text model, unwrapping multimodal wrappers if present.

    Standard models (Llama, Qwen): model itself is the text model.
    Multimodal wrappers (Gemma 4): model.language_model is the text model.

    Returns the text model that has .model (backbone) and optionally .lm_head.
    """
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def _get_text_backbone(model):
    """Get a separable hidden-state backbone when the model exposes one."""
    tm = _get_text_model(model)
    return getattr(tm, "model", None)


def _has_hidden_stack(obj):
    """Whether an object exposes embed_tokens/layers/norm for manual hidden-state forward."""
    if obj is None:
        return False
    return (
        hasattr(obj, "embed_tokens")
        and hasattr(obj, "layers")
        and hasattr(obj, "norm")
    )


def _run_hidden_stack(stack, inputs, inputs_embeds=None, **kwargs):
    """Execute a language stack up to pre-lm_head hidden states."""
    from mlx_vlm.models.base import create_attention_mask

    norm_weight = getattr(getattr(stack, "norm", None), "weight", None)
    if inputs_embeds is None:
        h = stack.embed_tokens(inputs)
    elif norm_weight is not None:
        h = inputs_embeds.astype(norm_weight.dtype)
    else:
        h = inputs_embeds

    cache = kwargs.get("cache")
    if cache is None:
        cache = [None] * len(stack.layers)
    mask = kwargs.get("mask")
    if mask is None:
        mask = kwargs.get("attention_mask_4d")
    if mask is None:
        mask = kwargs.get("attention_mask")
    if mask is None:
        mask = create_attention_mask(h, cache)

    for layer, c in zip(stack.layers, cache):
        h = layer(h, mask, c)
    return stack.norm(h)


def _has_direct_hidden_stack(model):
    """Whether the text model exposes layers/norm directly instead of under .model."""
    tm = _get_text_model(model)
    return not hasattr(tm, "model") and _has_hidden_stack(tm)


def _forward_text_hidden_states(model, inputs, inputs_embeds=None, **kwargs):
    """Run a text stack up to pre-lm_head hidden states for CCE."""
    tm = _get_text_model(model)
    backbone = getattr(tm, "model", None)
    if backbone is not None:
        if getattr(backbone, "lm_head", None) is not None and _has_hidden_stack(backbone):
            return _run_hidden_stack(backbone, inputs, inputs_embeds=inputs_embeds, **kwargs)
        embed_kwarg = _get_backbone_embed_kwarg(backbone)
        backbone_kwargs = _filter_backbone_kwargs(backbone, kwargs)
        if inputs_embeds is not None:
            backbone_kwargs[embed_kwarg] = inputs_embeds
        return backbone(inputs, **backbone_kwargs)

    if not _has_direct_hidden_stack(model):
        raise ValueError("Text model does not expose a separable hidden-state backbone")
    return _run_hidden_stack(tm, inputs, inputs_embeds=inputs_embeds, **kwargs)


def _get_lm_head_layer(model):
    """Get the raw LM head layer (QuantizedLinear or Linear/Embedding).

    Checks for a separate lm_head first (untied models like Qwen), then
    falls back to embed_tokens (tied models like Gemma/Llama).
    Handles multimodal wrappers (e.g. Gemma 4) via _get_text_model.

    Returns the layer object (not its weight), so callers can access
    .weight, .scales, .biases, .group_size, .bits for quantized layers.
    """
    tm = _get_text_model(model)
    if hasattr(tm, "lm_head") and tm.lm_head is not None:
        return tm.lm_head
    backbone = getattr(tm, "model", tm)
    if hasattr(backbone, "lm_head") and backbone.lm_head is not None:
        return backbone.lm_head
    return backbone.embed_tokens


def _is_quantized_layer(layer):
    """Check if a layer has quantized weights (has .scales attribute)."""
    return hasattr(layer, "scales")


def _get_logit_softcap(model):
    """Get logit softcapping value if model uses it (e.g. Gemma-2/4), else 0.0."""
    tm = _get_text_model(model)
    softcap = getattr(tm, "final_logit_softcapping", None)
    if softcap is None and hasattr(tm, "args"):
        softcap = getattr(tm.args, "final_logit_softcapping", None)
    return float(softcap) if softcap is not None and softcap > 0 else 0.0


def _is_lm_head_trainable(model):
    """Check if the LM head weight is trainable (not frozen by LoRA).

    For LoRA training, the LM head weight is frozen — computing its gradient
    in CCE is a wasted V x chunk_size x H matmul per chunk. Returns False
    when the weight should be wrapped with mx.stop_gradient.
    """
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    for key in trainable:
        if 'lora' not in key:
            if 'lm_head' in key or 'embed_tokens.weight' in key:
                return True
    return len(trainable) == 0  # no LoRA = full fine-tuning = trainable


def make_cce_loss_fn(model):
    """Create a CCE loss function using the bundled chunked cross-entropy engine.

    CCE computes cross-entropy directly from hidden states and the LM head weight,
    avoiding full logit materialization. This saves significant memory for large
    vocabularies.

    Args:
        model: MLX model.

    Returns:
        A function (model, batch, lengths, labels=None) -> (loss, ntoks).
        When labels is provided, uses labels[:,1:] for targets with
        (targets != -100) as the loss mask.
        The function has a ``_unsloth_cce_backend`` attribute for logging.
    """
    softcap = _get_logit_softcap(model)
    if softcap > 0:
        print(f"Unsloth: CCE using logit_softcap={softcap} for this model.")

    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)

    _has_wrapper = hasattr(model, "language_model")
    tm = _get_text_model(model)
    backbone = getattr(tm, "model", None)
    _has_lm_head = (
        (hasattr(tm, "lm_head") and tm.lm_head is not None)
        or (backbone is not None and hasattr(backbone, "lm_head") and backbone.lm_head is not None)
    )

    def _get_backbone(model):
        """Get backbone (for hidden states) from the live model tree."""
        return _forward_text_hidden_states

    def _get_lm_weight_layer(model):
        """Get LM head or embed_tokens layer from the live model tree."""
        if _has_wrapper:
            tm = model.language_model
        else:
            tm = model
        if _has_lm_head:
            if hasattr(tm, "lm_head") and tm.lm_head is not None:
                return tm.lm_head
            backbone = getattr(tm, "model", tm)
            if hasattr(backbone, "lm_head") and backbone.lm_head is not None:
                return backbone.lm_head
        backbone = getattr(tm, "model", tm)
        return backbone.embed_tokens

    if use_quantized:
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        print(f"Unsloth: CCE using quantized matmul (group_size={group_size}, bits={bits})")
        _has_biases = hasattr(lm_layer, "biases")

        rt_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
            quantized=True,
            group_size=group_size,
            bits=bits,
        )

        def loss_fn(model, batch, lengths, labels=None):
            if labels is None:
                inputs, targets = batch[:, :-1], batch[:, 1:]
            else:
                inputs = batch[:, :-1]
                targets = labels[:, 1:]
            hidden = _get_backbone(model)(model, inputs)
            layer = _get_lm_weight_layer(model)
            w = layer.weight
            sc = layer.scales
            bi = layer.biases if _has_biases else mx.zeros_like(layer.scales)
            steps = mx.arange(1, targets.shape[1] + 1)
            length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            if labels is None:
                mask = length_mask
            else:
                mask = mx.logical_and(targets != -100, length_mask)
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,)).astype(mx.int32)
            loss = rt_cce(hidden_flat, w, sc, bi, targets_flat)
            loss = loss.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
            return loss, ntoks
    else:
        _skip_weight_grad = not _is_lm_head_trainable(model)
        if _skip_weight_grad:
            print("Unsloth: CCE skipping weight gradient (LM head is frozen).")

        rt_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
        )

        def loss_fn(model, batch, lengths, labels=None):
            if labels is None:
                inputs, targets = batch[:, :-1], batch[:, 1:]
            else:
                inputs = batch[:, :-1]
                targets = labels[:, 1:]
            hidden = _get_backbone(model)(model, inputs)
            w = _get_lm_weight_layer(model).weight
            if _skip_weight_grad:
                w = mx.stop_gradient(w)
            steps = mx.arange(1, targets.shape[1] + 1)
            length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            if labels is None:
                mask = length_mask
            else:
                mask = mx.logical_and(targets != -100, length_mask)
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,)).astype(mx.int32)
            loss = rt_cce(hidden_flat, w, targets_flat)
            loss = loss.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
            return loss, ntoks

    loss_fn._unsloth_cce_backend = "runtime-cce"
    return loss_fn


def make_baseline_loss_fn():
    """Create a standard cross-entropy loss function.

    Uses the full logit computation through the LM head, then applies
    nn.losses.cross_entropy. Used when use_cce=False.

    Returns:
        A function (model, batch, lengths, labels=None) -> (loss, ntoks).
        When labels is provided, uses labels[:,1:] for targets with
        (targets != -100) as the loss mask.
    """
    def loss_fn(model, batch, lengths, labels=None):
        if labels is None:
            inputs, targets = batch[:, :-1], batch[:, 1:]
        else:
            inputs = batch[:, :-1]
            targets = labels[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        if labels is None:
            mask = length_mask.astype(mx.float32)
        else:
            mask = mx.logical_and(targets != -100, length_mask).astype(mx.float32)
        # Replace -100 with 0 before CE — MLX has no ignore_index;
        # the mask already zeros out these positions in the loss.
        safe_targets = mx.where(targets == -100, 0, targets)
        ce = nn.losses.cross_entropy(logits, safe_targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
        return loss, ntoks

    return loss_fn


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------

# Image/vision special tokens that should never contribute to loss.
# Mirrors unsloth's IMAGE_TOKENS list from vision_utils.py.
_IMAGE_TOKEN_STRINGS = (
    "<|image|>",           # Llama 3.2 Vision, Phi 3.5, Gemma4
    "<|vision_start|>",    # Qwen
    "<|vision_end|>",      # Qwen
    "<|vision_pad|>",      # Qwen
    "<|image_pad|>",       # Qwen
    "<image>",             # PaliGemma, Llava, InternVL
    "</image>",            # InternVL
    "<image_soft_token>",  # Gemma 3
    "<start_of_image>",    # Gemma 3
    "<end_of_image>",      # Gemma 3
)


def _get_image_token_ids(model):
    """Resolve image token IDs from model's processor/tokenizer.

    Returns an mx.array of token IDs to mask from loss, or None if
    no image tokens are found (non-VLM or tokenizer doesn't have them).
    """
    processor = getattr(model, "_processor", None)
    tokenizer = getattr(processor, "tokenizer", processor) if processor else None
    if tokenizer is None:
        return None

    ids = []
    for tok_str in _IMAGE_TOKEN_STRINGS:
        try:
            tok_ids = tokenizer.convert_tokens_to_ids([tok_str])
            if tok_ids and tok_ids[0] is not None:
                # Some tokenizers return the unk_token_id for unknown tokens
                unk_id = getattr(tokenizer, "unk_token_id", None)
                if tok_ids[0] != unk_id:
                    ids.append(tok_ids[0])
        except Exception:
            continue

    # Also check config for image_token_index / image_token_id
    config = getattr(model, "_config", {})
    for key in ("image_token_index", "image_token_id"):
        val = config.get(key)
        if val is not None and val not in ids:
            ids.append(val)

    if not ids:
        return None
    return ids  # plain Python list; avoids mx.eval in the hot path


def _mask_image_tokens(targets, image_token_ids):
    """Set image token positions in targets to -100 (ignore_index).

    Prevents the model from training to predict image placeholder tokens,
    which are fixed special tokens that provide no useful gradient signal.

    Args:
        targets: mx.array of token IDs.
        image_token_ids: plain Python list of int token IDs, or None.
    """
    if not image_token_ids:
        return targets
    # Build a mask: True where target is any image token
    is_image = targets == image_token_ids[0]
    for tok_id in image_token_ids[1:]:
        is_image = is_image | (targets == tok_id)
    return mx.where(is_image, -100, targets)


def _mask_prompt_tokens(targets, assistant_token_id):
    """Mask all tokens before the first assistant response in each sequence.

    Scans each row of targets for assistant_token_id. All positions before
    the first occurrence are set to -100 (ignore_index). If the token is
    not found, the entire sequence is left unmasked (assumes all completion).

    This implements train_on_completions_only for VLM training.
    """
    if assistant_token_id <= 0:
        return targets
    # Find the first occurrence of assistant_token_id in each row
    is_assistant = (targets == assistant_token_id)
    # cumsum along seq dim: positions after first assistant token have cumsum > 0
    cumulative = mx.cumsum(is_assistant.astype(mx.int32), axis=1)
    # Mask everything before first assistant token (cumsum == 0)
    prompt_mask = (cumulative == 0)
    return mx.where(prompt_mask, -100, targets)


def _is_vlm_model(model) -> bool:
    """Check if model is a VLM (has language_model + vision component)."""
    explicit_flag = getattr(model, "_is_vlm_model", None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    if not hasattr(model, "language_model"):
        return False
    return any(hasattr(model, attr) for attr in
               ("vision_tower", "vision_model", "vision_encoder", "visual",
                "multi_modal_projector", "audio_tower"))


def _align_logits_with_labels(logits, labels):
    """Truncate logits or labels so their sequence dimensions match.

    VLMs can inject vision tokens that change the sequence length of logits
    relative to the label sequence.
    """
    l_seq = logits.shape[1]
    t_seq = labels.shape[1]
    if l_seq > t_seq:
        logits = logits[:, :t_seq, :]
    elif t_seq > l_seq:
        labels = labels[:, :l_seq]
    return logits, labels


def make_vlm_baseline_loss_fn(model=None, assistant_token_id=0):
    """Create a standard cross-entropy loss function for VLMs.

    Takes a batch dict with input_ids, pixel_values, attention_mask.

    Returns:
        A function (model, batch_dict) -> (loss, ntoks).
    """
    _image_token_ids = _get_image_token_ids(model) if model is not None else None
    _assistant_token_id = assistant_token_id

    def loss_fn(model, batch_dict):
        input_ids = batch_dict["input_ids"]
        pixel_values = batch_dict.get("pixel_values")
        attention_mask = batch_dict.get("attention_mask")
        labels = batch_dict.get("labels")

        # Standard causal LM shift
        inputs = input_ids[:, :-1]

        # Forward pass — let the model create its own causal mask.
        # Pass extra keys (e.g. image_grid_thw for Qwen) that some models need.
        fwd_kwargs = {
            k: v for k, v in batch_dict.items()
            if k not in ("input_ids", "pixel_values", "attention_mask", "labels")
            and v is not None
        }
        output = model(inputs, pixel_values=pixel_values, mask=attention_mask, **fwd_kwargs)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits.astype(mx.float32)

        if labels is not None:
            # train_on_responses_only: labels encode instruction/padding masking.
            # Still mask image placeholder tokens.
            targets = labels[:, 1:]
            targets = _mask_image_tokens(targets, _image_token_ids)
            logits, targets = _align_logits_with_labels(logits, targets)
            mask = (targets != -100).astype(mx.float32)
        else:
            targets = input_ids[:, 1:]

            # Handle sequence length mismatch from vision token injection
            logits, targets = _align_logits_with_labels(logits, targets)

            # Build mask from attention_mask (shifted to match targets)
            if attention_mask is not None:
                length_mask = attention_mask[:, 1:]
                length_mask = length_mask[:, :targets.shape[1]]
            else:
                length_mask = mx.ones_like(targets, dtype=mx.float32)

            # Mask image placeholder tokens and prompt tokens
            targets = _mask_image_tokens(targets, _image_token_ids)
            targets = _mask_prompt_tokens(targets, _assistant_token_id)
            # Update length_mask to exclude masked positions
            mask = mx.where(targets == -100, 0, length_mask)

        # Replace -100 with 0 before CE — MLX has no ignore_index;
        # the mask already zeros out these positions in the loss.
        safe_targets = mx.where(targets == -100, 0, targets)
        ce = nn.losses.cross_entropy(logits, safe_targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
        return loss, ntoks

    loss_fn._unsloth_cce_backend = "baseline-ce"
    return loss_fn


def _unpack_embed_result(embed_result, model):
    """Unpack get_input_embeddings result into embeds + backbone kwargs.

    Handles both plain mx.array returns and InputEmbeddingsFeatures dataclass
    (gemma4 per_layer_inputs, qwen3-vl position_ids/deepstack, etc.).
    """
    backbone_kwargs = {}
    if hasattr(embed_result, "inputs_embeds"):
        merged_embeds = embed_result.inputs_embeds
        if getattr(embed_result, "attention_mask_4d", None) is not None:
            backbone_kwargs["attention_mask_4d"] = embed_result.attention_mask_4d
        if getattr(embed_result, "position_ids", None) is not None:
            backbone_kwargs["position_ids"] = embed_result.position_ids
        # Gemma4: per-layer inputs for vision token injection
        if getattr(embed_result, "per_layer_inputs", None) is not None:
            backbone_kwargs["per_layer_inputs"] = embed_result.per_layer_inputs
        # Qwen3-VL deepstack: visual position masks + visual embeds
        if getattr(embed_result, "visual_pos_masks", None) is not None:
            backbone_kwargs["visual_pos_masks"] = embed_result.visual_pos_masks
        if getattr(embed_result, "deepstack_visual_embeds", None) is not None:
            backbone_kwargs["deepstack_visual_embeds"] = embed_result.deepstack_visual_embeds
        if getattr(embed_result, "cross_attention_states", None) is not None:
            backbone_kwargs["cross_attention_states"] = embed_result.cross_attention_states
        if getattr(embed_result, "cross_attention_mask", None) is not None:
            backbone_kwargs["cross_attention_mask"] = embed_result.cross_attention_mask
        if getattr(embed_result, "full_text_row_masked_out_mask", None) is not None:
            backbone_kwargs["full_text_row_masked_out_mask"] = (
                embed_result.full_text_row_masked_out_mask
            )
        if getattr(embed_result, "decoder_inputs_embeds", None) is not None:
            backbone_kwargs["decoder_inputs_embeds"] = embed_result.decoder_inputs_embeds
        if getattr(embed_result, "attention_mask", None) is not None:
            backbone_kwargs["attention_mask"] = embed_result.attention_mask
    else:
        merged_embeds = embed_result

    # Qwen-VL family: get_input_embeddings stashes position_ids on the
    # language model wrapper; the inner backbone needs them explicitly.
    # When no position_ids were stashed (e.g. text-only samples or simple
    # images without grid_thw), generate sequential ones so the backbone
    # doesn't crash accessing cache.offset with cache=None.
    lm = getattr(model, "language_model", None)
    if lm is not None:
        _MISSING = object()
        pos_ids = getattr(lm, "_position_ids", _MISSING)
        if pos_ids is not _MISSING and pos_ids is not None:
            backbone_kwargs["position_ids"] = pos_ids
        elif pos_ids is None:
            # Fallback: sequential position_ids. Correct for text-only and
            # single-image samples. For multi-image with spatial m-RoPE
            # (Qwen VL), the per-axis positions should differ for image
            # regions — but grid_thw metadata is unavailable here so we
            # use sequential as an approximation.
            seq_len = merged_embeds.shape[1]
            pos_ids = mx.arange(seq_len).reshape(1, -1)
            pos_ids = mx.broadcast_to(pos_ids, (merged_embeds.shape[0], seq_len))
            # Qwen VL m-RoPE uses 3 axes: temporal, height, width
            MROPE_AXES = 3
            pos_ids = mx.expand_dims(pos_ids, axis=0)
            pos_ids = mx.tile(pos_ids, (MROPE_AXES, 1, 1))
            backbone_kwargs["position_ids"] = pos_ids

    return merged_embeds, backbone_kwargs


def _get_backbone_embed_kwarg(backbone):
    try:
        params = inspect.signature(backbone.__call__).parameters
    except (TypeError, ValueError):
        return "inputs_embeds"
    if "inputs_embeds" in params:
        return "inputs_embeds"
    if "input_embeddings" in params:
        return "input_embeddings"
    if "input_embeds" in params:
        return "input_embeds"
    return "inputs_embeds"


def _filter_backbone_kwargs(backbone, kwargs):
    try:
        params = inspect.signature(backbone.__call__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    allowed = set(params)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _vlm_cce_forward(model, batch_dict, image_token_ids=None,
                     assistant_token_id=0):
    """Shared VLM CCE forward: embed -> backbone -> hidden + masked_targets + ntoks."""
    input_ids = batch_dict["input_ids"]
    pixel_values = batch_dict.get("pixel_values")
    attention_mask = batch_dict.get("attention_mask")
    labels = batch_dict.get("labels")

    inputs = input_ids[:, :-1]

    # Collect extra keys (e.g. image_grid_thw for Qwen) that some models need.
    extra_kwargs = {
        k: v for k, v in batch_dict.items()
        if k not in ("input_ids", "pixel_values", "attention_mask", "labels")
        and v is not None
    }
    position_ids = extra_kwargs.get("position_ids")
    if position_ids is not None and hasattr(position_ids, "shape"):
        seq_len = inputs.shape[1]
        if position_ids.shape[-1] != seq_len:
            extra_kwargs["position_ids"] = position_ids[..., :seq_len]

    embed_result = model.get_input_embeddings(
        inputs,
        pixel_values,
        mask=attention_mask,
        **extra_kwargs,
    )
    merged_embeds, backbone_kwargs = _unpack_embed_result(embed_result, model)
    if "position_ids" in extra_kwargs and "position_ids" not in backbone_kwargs:
        backbone_kwargs["position_ids"] = extra_kwargs["position_ids"]

    hidden = _forward_text_hidden_states(
        model,
        inputs,
        inputs_embeds=merged_embeds,
        **backbone_kwargs,
    )

    if labels is not None:
        # train_on_responses_only: labels already encode instruction and
        # padding masking. Still need to mask image placeholder tokens
        # since they provide no useful gradient signal.
        targets = labels[:, 1:]
        masked_targets = _mask_image_tokens(targets, image_token_ids)
        ntoks = (masked_targets != -100).sum()
    else:
        targets = input_ids[:, 1:]

        if attention_mask is not None:
            length_mask = attention_mask[:, 1:][:, :targets.shape[1]]
        else:
            length_mask = mx.ones_like(targets, dtype=mx.float32)

        masked_targets = mx.where(length_mask, targets, -100)

        # Mask image placeholder tokens — they're fixed special tokens that
        # provide no useful gradient signal.
        masked_targets = _mask_image_tokens(masked_targets, image_token_ids)

        # Completion-only masking: mask prompt tokens before first assistant response
        masked_targets = _mask_prompt_tokens(masked_targets, assistant_token_id)

        ntoks = (masked_targets != -100).sum()

    # Align sequence lengths (vision token injection changes seq len)
    h_seq, t_seq = hidden.shape[1], masked_targets.shape[1]
    if h_seq > t_seq:
        hidden = hidden[:, :t_seq, :]
    elif t_seq > h_seq:
        masked_targets = masked_targets[:, :h_seq]
        ntoks = (masked_targets != -100).sum()

    return hidden, masked_targets, ntoks


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_to_mapping(config):
    if isinstance(config, dict):
        return config
    if config is None:
        return {}
    return {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("_") and not callable(getattr(config, key))
    }


def _normalize_grid_thw(grid_thw):
    if grid_thw is None:
        return None
    if isinstance(grid_thw, mx.array):
        grid_thw = grid_thw.tolist()
    elif hasattr(grid_thw, "tolist"):
        grid_thw = grid_thw.tolist()

    normalized = []
    for item in grid_thw:
        if hasattr(item, "tolist"):
            item = item.tolist()
        normalized.append(tuple(int(x) for x in item))
    return tuple(normalized)


def _normalize_size_tuples(values):
    if values is None:
        return None
    if isinstance(values, mx.array):
        values = values.tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()

    normalized = []
    for item in values:
        if hasattr(item, "tolist"):
            item = item.tolist()
        normalized.append(tuple(int(x) for x in item))
    return tuple(normalized)


def _normalize_int_tuple(values):
    if values is None:
        return None
    if isinstance(values, mx.array):
        values = values.tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(x) for x in values)


def _expand_image_token_sequences(
    input_ids,
    attention_mask,
    image_token_id,
    repeat_count,
):
    input_ids_np = np.asarray(input_ids)
    attention_mask_np = (
        np.asarray(attention_mask)
        if attention_mask is not None
        else np.ones_like(input_ids_np, dtype=np.int32)
    )

    expanded_ids = []
    expanded_masks = []
    max_len = 0
    for row_ids, row_mask in zip(input_ids_np, attention_mask_np):
        new_ids = []
        new_mask = []
        for token_id, mask_value in zip(row_ids.tolist(), row_mask.tolist()):
            if int(token_id) == int(image_token_id):
                new_ids.extend([int(image_token_id)] * int(repeat_count))
                new_mask.extend([int(mask_value)] * int(repeat_count))
            else:
                new_ids.append(int(token_id))
                new_mask.append(int(mask_value))
        expanded_ids.append(new_ids)
        expanded_masks.append(new_mask)
        max_len = max(max_len, len(new_ids))

    padded_ids = np.zeros((len(expanded_ids), max_len), dtype=np.int32)
    padded_masks = np.zeros((len(expanded_masks), max_len), dtype=np.int32)
    for row_idx, (row_ids, row_mask) in enumerate(zip(expanded_ids, expanded_masks)):
        row_len = len(row_ids)
        padded_ids[row_idx, :row_len] = row_ids
        padded_masks[row_idx, :row_len] = row_mask

    return mx.array(padded_ids), mx.array(padded_masks)


def _expand_token_runs(
    input_ids,
    attention_mask,
    replacements_by_batch,
):
    input_ids_np = np.asarray(input_ids)
    attention_mask_np = (
        np.asarray(attention_mask)
        if attention_mask is not None
        else np.ones_like(input_ids_np, dtype=np.int32)
    )

    expanded_ids = []
    expanded_masks = []
    max_len = 0
    for row_ids, row_mask, replacements in zip(
        input_ids_np,
        attention_mask_np,
        replacements_by_batch,
    ):
        replacements = sorted(replacements, key=lambda item: item[0])
        new_ids = []
        new_mask = []
        prev = 0
        row_ids_list = row_ids.tolist()
        row_mask_list = row_mask.tolist()
        for start, end, token_id, repeat in replacements:
            if start > prev:
                new_ids.extend(row_ids_list[prev:start])
                new_mask.extend(row_mask_list[prev:start])
            new_ids.extend([int(token_id)] * int(repeat))
            fill_mask = int(row_mask_list[start]) if start < len(row_mask_list) else 1
            new_mask.extend([fill_mask] * int(repeat))
            prev = int(end)
        if prev < len(row_ids_list):
            new_ids.extend(row_ids_list[prev:])
            new_mask.extend(row_mask_list[prev:])
        expanded_ids.append(new_ids)
        expanded_masks.append(new_mask)
        max_len = max(max_len, len(new_ids))

    padded_ids = np.zeros((len(expanded_ids), max_len), dtype=np.int32)
    padded_masks = np.zeros((len(expanded_masks), max_len), dtype=np.int32)
    for row_idx, (row_ids, row_mask) in enumerate(zip(expanded_ids, expanded_masks)):
        row_len = len(row_ids)
        padded_ids[row_idx, :row_len] = row_ids
        padded_masks[row_idx, :row_len] = row_mask

    return mx.array(padded_ids), mx.array(padded_masks)


def _build_qwen_position_ids(
    input_ids,
    attention_mask,
    image_grid_thw,
    video_grid_thw,
    image_token_id,
    video_token_id,
    spatial_merge_size,
):
    batch_size, seq_length = input_ids.shape
    position_ids = np.ones((3, batch_size, seq_length), dtype=np.int32)

    image_grids = list(image_grid_thw or ())
    video_grids = list(video_grid_thw or ())
    image_index = 0
    video_index = 0

    for batch_idx in range(batch_size):
        valid_len = (
            int(attention_mask[batch_idx].sum())
            if attention_mask is not None
            else seq_length
        )
        tokens = input_ids[batch_idx, :valid_len].tolist()
        if valid_len == 0:
            continue

        llm_pos_ids_list = []
        st = 0

        while st < valid_len:
            try:
                ed_image = tokens.index(image_token_id, st)
            except ValueError:
                ed_image = valid_len + 1

            try:
                ed_video = tokens.index(video_token_id, st)
            except ValueError:
                ed_video = valid_len + 1

            ed = min(ed_image, ed_video)
            if ed > valid_len:
                text_len = valid_len - st
                if text_len > 0:
                    start_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
                    text_ids = np.broadcast_to(
                        np.arange(text_len, dtype=np.int32)[None, :],
                        (3, text_len),
                    )
                    llm_pos_ids_list.append(text_ids + start_idx)
                break

            text_len = ed - st
            start_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            if text_len > 0:
                text_ids = np.broadcast_to(
                    np.arange(text_len, dtype=np.int32)[None, :],
                    (3, text_len),
                )
                llm_pos_ids_list.append(text_ids + start_idx)
                start_idx += text_len

            if ed_image < ed_video:
                t, h, w = image_grids[image_index]
                image_index += 1
            else:
                t, h, w = video_grids[video_index]
                video_index += 1

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size

            t_index = np.broadcast_to(
                np.arange(llm_grid_t, dtype=np.int32)[:, None],
                (llm_grid_t, llm_grid_h * llm_grid_w),
            ).reshape(-1)
            h_index = np.broadcast_to(
                np.arange(llm_grid_h, dtype=np.int32)[None, :, None],
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            w_index = np.broadcast_to(
                np.arange(llm_grid_w, dtype=np.int32)[None, None, :],
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            llm_pos_ids_list.append(
                np.stack([t_index, h_index, w_index], axis=0) + start_idx
            )
            st = ed + (llm_grid_t * llm_grid_h * llm_grid_w)

        if llm_pos_ids_list:
            batch_positions = np.concatenate(llm_pos_ids_list, axis=1)
            batch_positions = batch_positions[:, :valid_len]
        else:
            batch_positions = np.broadcast_to(
                np.arange(valid_len, dtype=np.int32)[None, :],
                (3, valid_len),
            )

        position_ids[:, batch_idx, :valid_len] = batch_positions

    return mx.array(position_ids)


def _build_glm_ocr_position_ids(
    input_ids,
    attention_mask,
    image_grid_thw,
    video_grid_thw,
    image_start_token_id,
    image_token_id,
    video_token_id,
    spatial_merge_size,
):
    batch_size, seq_length = input_ids.shape
    position_ids = np.ones((3, batch_size, seq_length), dtype=np.int32)

    image_grids = list(image_grid_thw or ())
    video_grids = list(video_grid_thw or ())
    image_index = 0
    video_index = 0

    for batch_idx in range(batch_size):
        valid_len = (
            int(attention_mask[batch_idx].sum())
            if attention_mask is not None
            else seq_length
        )
        tokens = input_ids[batch_idx, :valid_len].tolist()
        if valid_len == 0:
            continue

        masked_tokens = tokens[:]
        llm_pos_ids_list = []
        st = 0

        try:
            vision_start_index = masked_tokens.index(image_start_token_id)
            vision_token = (
                masked_tokens[vision_start_index + 1]
                if vision_start_index + 1 < len(masked_tokens)
                else None
            )
            image_nums = int(vision_token == image_token_id)
            video_nums = int(vision_token == video_token_id)
        except ValueError:
            image_nums = 0
            video_nums = 0

        remain_images = image_nums
        remain_videos = video_nums

        for _ in range(image_nums + video_nums):
            try:
                ed_image = masked_tokens.index(image_token_id, st) if remain_images > 0 else valid_len + 1
            except ValueError:
                ed_image = valid_len + 1
            try:
                ed_video = masked_tokens.index(video_token_id, st) if remain_videos > 0 else valid_len + 1
            except ValueError:
                ed_video = valid_len + 1

            if ed_image < ed_video:
                t, h, w = image_grids[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grids[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size

            text_len = ed - st
            start_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            if text_len > 0:
                text_ids = np.broadcast_to(
                    np.arange(text_len, dtype=np.int32)[None, :],
                    (3, text_len),
                )
                llm_pos_ids_list.append(text_ids + start_idx)
                start_idx += text_len

            t_index = np.broadcast_to(
                np.arange(llm_grid_t, dtype=np.int32)[:, None],
                (llm_grid_t, llm_grid_h * llm_grid_w),
            ).reshape(-1)
            h_index = np.broadcast_to(
                np.arange(llm_grid_h, dtype=np.int32)[None, :, None],
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            w_index = np.broadcast_to(
                np.arange(llm_grid_w, dtype=np.int32)[None, None, :],
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)

            llm_pos_ids_list.append(
                np.stack([t_index, h_index, w_index], axis=0) + start_idx
            )
            st = ed + (llm_grid_t * llm_grid_h * llm_grid_w)

        if st < valid_len:
            start_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = valid_len - st
            text_ids = np.broadcast_to(
                np.arange(text_len, dtype=np.int32)[None, :],
                (3, text_len),
            )
            llm_pos_ids_list.append(text_ids + start_idx)

        if llm_pos_ids_list:
            batch_positions = np.concatenate(llm_pos_ids_list, axis=1)
            batch_positions = batch_positions[:, :valid_len]
        else:
            batch_positions = np.broadcast_to(
                np.arange(valid_len, dtype=np.int32)[None, :],
                (3, valid_len),
            )

        position_ids[:, batch_idx, :valid_len] = batch_positions

    return mx.array(position_ids)


def _prepare_vlm_batch_for_compile(batch_dict, config):
    model_type = _config_get(config, "model_type")
    vision_config = _config_to_mapping(_config_get(config, "vision_config", {}))

    image_grid_thw = _normalize_grid_thw(batch_dict.get("image_grid_thw"))
    video_grid_thw = _normalize_grid_thw(batch_dict.get("video_grid_thw"))
    image_sizes = _normalize_size_tuples(batch_dict.get("image_sizes"))
    spatial_shapes = _normalize_size_tuples(batch_dict.get("spatial_shapes"))
    images_spatial_crop = _normalize_size_tuples(batch_dict.get("images_spatial_crop"))
    audio_embed_sizes = _normalize_int_tuple(batch_dict.get("audio_embed_sizes"))
    if image_grid_thw is not None:
        batch_dict["image_grid_thw"] = image_grid_thw
    if video_grid_thw is not None:
        batch_dict["video_grid_thw"] = video_grid_thw
    if image_sizes is not None:
        batch_dict["image_sizes"] = image_sizes
    if spatial_shapes is not None:
        batch_dict["spatial_shapes"] = spatial_shapes
    if images_spatial_crop is not None:
        batch_dict["images_spatial_crop"] = images_spatial_crop
    if audio_embed_sizes is not None:
        batch_dict["audio_embed_sizes"] = audio_embed_sizes

    if model_type in {
        "qwen2_vl",
        "qwen2_5_vl",
        "paddleocr_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
    }:
        input_ids = batch_dict.get("input_ids")
        if input_ids is not None:
            input_ids_np = np.asarray(input_ids)
            attention_mask = batch_dict.get("attention_mask")
            attention_mask_np = (
                np.asarray(attention_mask)
                if attention_mask is not None
                else None
            )
            batch_dict["position_ids"] = _build_qwen_position_ids(
                input_ids=input_ids_np,
                attention_mask=attention_mask_np,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_token_id=int(_config_get(config, "image_token_id", _config_get(config, "image_token_index"))),
                video_token_id=int(_config_get(config, "video_token_id", _config_get(config, "video_token_index"))),
                spatial_merge_size=int(vision_config.get("spatial_merge_size", 2)),
            )

    if model_type == "glm_ocr":
        input_ids = batch_dict.get("input_ids")
        if input_ids is not None:
            input_ids_np = np.asarray(input_ids)
            attention_mask = batch_dict.get("attention_mask")
            attention_mask_np = (
                np.asarray(attention_mask)
                if attention_mask is not None
                else None
            )
            batch_dict["position_ids"] = _build_glm_ocr_position_ids(
                input_ids=input_ids_np,
                attention_mask=attention_mask_np,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_start_token_id=int(_config_get(config, "image_start_token_id")),
                image_token_id=int(_config_get(config, "image_token_id")),
                video_token_id=int(_config_get(config, "video_token_id")),
                spatial_merge_size=int(vision_config.get("spatial_merge_size", 2)),
            )

    if model_type == "phi3_v":
        input_ids = batch_dict.get("input_ids")
        if input_ids is not None:
            input_ids_np = np.asarray(input_ids)
            batch_dict["image_positions"] = tuple(
                tuple(int(x) for x in pos)
                for pos in np.argwhere(input_ids_np < 0).tolist()
            )

    if model_type == "multi_modality":
        input_ids = batch_dict.get("input_ids")
        if input_ids is not None:
            expanded_ids, expanded_mask = _expand_image_token_sequences(
                input_ids=input_ids,
                attention_mask=batch_dict.get("attention_mask"),
                image_token_id=int(_config_get(config, "image_token_index")),
                repeat_count=int(_config_get(config, "num_image_tokens")),
            )
            batch_dict["input_ids"] = expanded_ids
            batch_dict["attention_mask"] = expanded_mask

    if model_type in {"phi4-siglip", "phi4_siglip"}:
        input_ids = batch_dict.get("input_ids")
        pixel_attention_mask = batch_dict.get("pixel_attention_mask")
        if input_ids is not None:
            input_ids_np = np.asarray(input_ids)
            batch_dict["image_token_positions"] = tuple(
                tuple(
                    int(pos)
                    for pos in np.where(row == -200)[0].tolist()
                )
                for row in input_ids_np
            )
        if pixel_attention_mask is not None:
            pam_np = np.asarray(pixel_attention_mask)
            pixel_valid_lengths = tuple(
                int(row.astype(np.int32).sum()) for row in pam_np
            )
            batch_dict["pixel_valid_lengths"] = pixel_valid_lengths
            if input_ids is not None:
                replacements = []
                image_idx = 0
                for batch_positions in batch_dict["image_token_positions"]:
                    batch_replacements = []
                    for pos in batch_positions:
                        repeat = pixel_valid_lengths[image_idx]
                        batch_replacements.append((pos, pos + 1, -200, repeat))
                        image_idx += 1
                    replacements.append(tuple(batch_replacements))
                expanded_ids, expanded_mask = _expand_token_runs(
                    input_ids=input_ids,
                    attention_mask=batch_dict.get("attention_mask"),
                    replacements_by_batch=tuple(replacements),
                )
                batch_dict["input_ids"] = expanded_ids
                batch_dict["attention_mask"] = expanded_mask

    if model_type == "phi4mm":
        input_ids = batch_dict.get("input_ids")
        pixel_attention_mask = batch_dict.get("pixel_attention_mask")
        if input_ids is not None:
            input_ids_np = np.asarray(input_ids)
            image_token_id = int(_config_get(config, "image_token_index"))
            audio_token_id = int(_config_get(config, "audio_token_index"))
            batch_dict["image_token_positions"] = tuple(
                tuple(
                    int(pos)
                    for pos in np.where(row == image_token_id)[0].tolist()
                )
                for row in input_ids_np
            )
            audio_spans = []
            for row in input_ids_np:
                spans = []
                idx = 0
                row_list = row.tolist()
                while idx < len(row_list):
                    if int(row_list[idx]) != audio_token_id:
                        idx += 1
                        continue
                    start = idx
                    while idx < len(row_list) and int(row_list[idx]) == audio_token_id:
                        idx += 1
                    spans.append((start, idx))
                audio_spans.append(tuple(spans))
            batch_dict["audio_token_spans"] = tuple(audio_spans)
            input_ids_np = np.asarray(input_ids)
        if pixel_attention_mask is not None:
            pam_np = np.asarray(pixel_attention_mask)
            batch_dict["pixel_valid_lengths"] = tuple(
                int(row.astype(np.int32).sum()) for row in pam_np
            )
        if input_ids is not None:
            image_positions = batch_dict.get("image_token_positions", ())
            audio_spans = batch_dict.get("audio_token_spans", ())
            pixel_valid_lengths = batch_dict.get("pixel_valid_lengths")
            replacements = []
            image_idx = 0
            audio_idx = 0
            image_token_id = int(_config_get(config, "image_token_index"))
            audio_token_id = int(_config_get(config, "audio_token_index"))
            for batch_idx, row in enumerate(input_ids_np):
                batch_replacements = []
                for pos in image_positions[batch_idx]:
                    repeat = (
                        int(pixel_valid_lengths[image_idx])
                        if pixel_valid_lengths is not None
                        else 1
                    )
                    batch_replacements.append((pos, pos + 1, image_token_id, repeat))
                    image_idx += 1
                for start, end in audio_spans[batch_idx]:
                    repeat = int(audio_embed_sizes[audio_idx]) if audio_embed_sizes is not None else int(end - start)
                    batch_replacements.append((start, end, audio_token_id, repeat))
                    audio_idx += 1
                replacements.append(tuple(batch_replacements))
            if replacements:
                expanded_ids, expanded_mask = _expand_token_runs(
                    input_ids=input_ids,
                    attention_mask=batch_dict.get("attention_mask"),
                    replacements_by_batch=tuple(replacements),
                )
                batch_dict["input_ids"] = expanded_ids
                batch_dict["attention_mask"] = expanded_mask

    return batch_dict


def make_vlm_cce_loss_fn(model, assistant_token_id=0):
    """Create a CCE loss function for VLMs.

    Uses model.get_input_embeddings() to get merged vision+text embeddings,
    then runs through the language model backbone to get hidden states before
    the LM head, and applies CCE.

    Falls back to baseline loss if get_input_embeddings is not available.

    Args:
        model: VLM model.
        assistant_token_id: Token ID marking start of assistant responses.
            When > 0, all tokens before the first occurrence are masked
            from the loss (completion-only training).

    Returns:
        A function (model, batch_dict) -> (loss, ntoks).
    """
    # Check if the model supports get_input_embeddings
    if not hasattr(model, "get_input_embeddings"):
        import warnings
        warnings.warn(
            "VLM model does not have get_input_embeddings — "
            "falling back to baseline CE loss.",
            stacklevel=2,
        )
        return make_vlm_baseline_loss_fn(model, assistant_token_id=assistant_token_id)

    tm = _get_text_model(model)
    if getattr(tm, "model", None) is None and not _has_direct_hidden_stack(model):
        import warnings
        warnings.warn(
            "VLM text stack does not expose a separable hidden-state backbone for CCE; "
            "falling back to baseline CE loss.",
            stacklevel=2,
        )
        return make_vlm_baseline_loss_fn(model, assistant_token_id=assistant_token_id)

    softcap = _get_logit_softcap(model)
    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)
    # Evaluate once — trainability doesn't change during training.
    # Must be called after LoRA setup.
    _skip_weight_grad = not _is_lm_head_trainable(model)

    _image_token_ids = _get_image_token_ids(model)
    if _image_token_ids is not None:
        print(f"Unsloth: Masking {len(_image_token_ids)} image token IDs from VLM loss.")
    _assistant_token_id = assistant_token_id
    if _assistant_token_id > 0:
        print(f"Unsloth: Completion-only training (assistant_token_id={_assistant_token_id}).")

    if use_quantized:
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)

        rt_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
            quantized=True,
            group_size=group_size,
            bits=bits,
        )

        def loss_fn(model, batch_dict):
            hidden, masked_targets, ntoks = _vlm_cce_forward(
                model, batch_dict, image_token_ids=_image_token_ids,
                assistant_token_id=_assistant_token_id)
            lm_head = _get_lm_head_layer(model)
            w = lm_head.weight
            sc = lm_head.scales
            bi = getattr(lm_head, "biases", None)
            if bi is None:
                bi = mx.zeros_like(sc)
            # Quantized backward already returns zero weight/scales/biases
            # gradients (see runtime_cce.py VJP), so stop_gradient is
            # redundant here even when the LM head is frozen.
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,)).astype(mx.int32)
            loss = rt_cce(hidden_flat, w, sc, bi, targets_flat)
            loss = loss.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
            return loss, ntoks
    else:
        if _skip_weight_grad:
            print("Unsloth: VLM CCE skipping weight gradient (LM head is frozen).")

        rt_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
        )

        def loss_fn(model, batch_dict):
            hidden, masked_targets, ntoks = _vlm_cce_forward(
                model, batch_dict, image_token_ids=_image_token_ids,
                assistant_token_id=_assistant_token_id)
            w = _get_lm_head_layer(model).weight
            if _skip_weight_grad:
                w = mx.stop_gradient(w)
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,)).astype(mx.int32)
            loss = rt_cce(hidden_flat, w, targets_flat)
            loss = loss.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
            return loss, ntoks

    loss_fn._unsloth_cce_backend = "runtime-cce"
    return loss_fn


def _get_vlm_image_size(config, processor):
    """Get target image size for uniform resizing, matching GPU collator.

    Tries vision_config.image_size, then processor.image_processor.size,
    falls back to 512.
    """
    vc = config.get("vision_config", {})
    if isinstance(vc, dict):
        sz = vc.get("image_size")
        if isinstance(sz, int) and sz > 0:
            return sz
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        sz = getattr(ip, "size", None)
        if isinstance(sz, dict):
            h = sz.get("height", sz.get("shortest_edge", 0))
            if isinstance(h, int) and h > 0:
                return h
        elif isinstance(sz, int) and sz > 0:
            return sz
    return 512


def _has_chat_template(obj):
    template = getattr(obj, "chat_template", None)
    return isinstance(template, str) and len(template.strip()) > 0


def _get_processor_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


_MLX_BUILTIN_CHAT_TEMPLATES = {
    "alpaca": """{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:
{{ message['content'] }}

{% elif message['role'] == 'assistant' %}### Response:
{{ message['content'] }}
{% endif %}{% endfor %}""",
}


def _mlx_builtin_chat_template(chat_template):
    if not isinstance(chat_template, str):
        return None
    return _MLX_BUILTIN_CHAT_TEMPLATES.get(chat_template.strip().lower())


def _apply_mlx_chat_template_override(target, chat_template, *, is_vlm=False):
    """Apply an explicit template override to a tokenizer or VLM processor."""
    if chat_template is None:
        return target

    original_tokenizer = _get_processor_tokenizer(target)
    tokenizer = original_tokenizer
    builtin_template = _mlx_builtin_chat_template(chat_template)
    if builtin_template is not None:
        tokenizer.chat_template = builtin_template
    elif isinstance(chat_template, str) and ("{{" in chat_template or "{%" in chat_template):
        tokenizer.chat_template = chat_template
    else:
        try:
            from unsloth.chat_templates import get_chat_template

            if isinstance(chat_template, (tuple, list)):
                tokenizer = get_chat_template(tokenizer, *chat_template)
            else:
                tokenizer = get_chat_template(tokenizer, chat_template)
        except Exception as exc:
            kind = "VLM" if is_vlm else "text"
            raise ValueError(
                f"Unsloth MLX {kind}: could not apply the requested chat template "
                f"{chat_template!r}. Pass a valid Unsloth template name or a raw "
                "Jinja chat_template string."
            ) from exc

    if is_vlm:
        if tokenizer is not original_tokenizer and hasattr(target, "tokenizer"):
            target.tokenizer = tokenizer
        if _has_chat_template(tokenizer):
            target.chat_template = tokenizer.chat_template
        return target
    return tokenizer


def normalize_mlx_chat_template(
    target,
    *,
    chat_template=None,
    model_name=None,
    model_type=None,
    is_vlm=False,
    strict=False,
):
    """Normalize chat template state for MLX text tokenizers and VLM processors."""
    if target is None:
        return target

    if chat_template is not None:
        target = _apply_mlx_chat_template_override(target, chat_template, is_vlm=is_vlm)

    if model_name is not None:
        setattr(target, "_unsloth_model_name", model_name)
    if model_type is not None:
        setattr(target, "_unsloth_model_type", model_type)

    tokenizer = _get_processor_tokenizer(target)
    if is_vlm and not _has_chat_template(target) and _has_chat_template(tokenizer):
        target.chat_template = tokenizer.chat_template

    template_target = target if is_vlm else tokenizer
    if strict and not _has_chat_template(template_target):
        _raise_mlx_chat_template_error(target, is_vlm=is_vlm)
    return target


def normalize_vlm_processor_chat_template(
    processor,
    *,
    chat_template=None,
    model_name=None,
    model_type=None,
    strict=False,
):
    """Make mlx-vlm processors behave like regular Unsloth vision processors.

    mlx-vlm often returns processors where the inner tokenizer has a template
    but the processor itself does not. Transformers' VLM processors expect the
    template on the processor for apply_chat_template(), so copy it once here.
    """
    return normalize_mlx_chat_template(
        processor,
        chat_template=chat_template,
        model_name=model_name,
        model_type=model_type,
        is_vlm=True,
        strict=strict,
    )


def _raise_mlx_chat_template_error(target, *, is_vlm=False):
    if is_vlm:
        _raise_vlm_chat_template_error(target)
    model_name = getattr(target, "_unsloth_model_name", None) or "this text model"
    model_type = getattr(target, "_unsloth_model_type", None)
    detail = f" ({model_type})" if model_type else ""
    raise ValueError(
        f"Unsloth MLX text: {model_name}{detail} has no tokenizer chat_template. "
        "Use an instruction/chat checkpoint, pass chat_template= to "
        "FastLanguageModel.from_pretrained(), set MLXTrainingConfig(chat_template=...), "
        "or provide formatting_func / a pre-rendered text column."
    )


def _raise_vlm_chat_template_error(processor):
    model_name = getattr(processor, "_unsloth_model_name", None) or "this VLM"
    model_type = getattr(processor, "_unsloth_model_type", None)
    detail = f" ({model_type})" if model_type else ""
    raise ValueError(
        f"Unsloth MLX VLM: {model_name}{detail} has no processor chat_template. "
        "For VLM training, Unsloth will not silently install ChatML because base "
        "vision-language models often require model-specific image tokens. Use an "
        "instruction/chat checkpoint, pass chat_template= to "
        "FastLanguageModel.from_pretrained(), set MLXTrainingConfig(vlm_chat_template=...), "
        "or provide formatting_func that returns already-templated text."
    )


def _select_vlm_messages_or_raw(item):
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return item
    for key in ("messages", "conversations"):
        value = item.get(key)
        if value is not None:
            return value
    return item


def _select_mlx_messages_or_raw(item):
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return item
    for key in ("messages", "conversations"):
        value = item.get(key)
        if value is not None:
            return value
    return item


def _clean_vlm_none_keys(obj):
    """Remove Arrow-created None keys without mutating the source dataset row."""
    if isinstance(obj, list):
        return [_clean_vlm_none_keys(x) for x in obj]
    if isinstance(obj, dict):
        return {
            key: _clean_vlm_none_keys(value)
            for key, value in obj.items()
            if value is not None
        }
    return obj


def _normalize_mlx_messages(messages, *, is_vlm=False):
    if isinstance(messages, str):
        return messages
    messages = _clean_vlm_none_keys(messages)
    if isinstance(messages, dict) and "role" in messages and "content" in messages:
        messages = [messages]
    if not isinstance(messages, list):
        kind = "VLM" if is_vlm else "text"
        raise TypeError(
            f"Unsloth MLX {kind}: expected a dataset row with messages/conversations, "
            "a list of chat messages, or a formatted text string."
        )

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            kind = "VLM" if is_vlm else "text"
            raise TypeError(
                f"Unsloth MLX {kind}: every chat message must be a dict with role/content."
            )
        msg = dict(message)
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = content
        elif isinstance(content, list):
            if is_vlm:
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        clean = _clean_vlm_none_keys(part)
                        if "type" not in clean:
                            if "text" in clean:
                                clean["type"] = "text"
                            elif "image" in clean:
                                clean["type"] = "image"
                        parts.append(clean)
                msg["content"] = parts
            else:
                texts = []
                for part in content:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict):
                        clean = _clean_vlm_none_keys(part)
                        if clean.get("type") == "text" or "text" in clean:
                            texts.append(str(clean.get("text", "")))
                msg["content"] = "".join(texts)
        else:
            msg["content"] = str(content)
        normalized.append(msg)
    return normalized


def _normalize_vlm_messages(messages):
    return _normalize_mlx_messages(messages, is_vlm=True)


def _collapse_vlm_assistant_content(messages):
    collapsed = copy.deepcopy(messages)
    for message in collapsed:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
            continue
        texts = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                texts.append(part)
        message["content"] = "".join(texts)
    return collapsed


def _flatten_vlm_content_for_text_template(messages):
    flattened = copy.deepcopy(messages)
    for message in flattened:
        content = message.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    texts.append(part)
            message["content"] = "".join(texts)
    return flattened


def _flatten_vlm_messages_to_content_parts(messages):
    """Flatten role messages for processors whose template expects content parts.

    Some mlx-vlm base processors ship a VLM token template, not a chat template.
    For example Qwen2-VL base templates iterate directly over content parts like
    {"type": "image"} / {"type": "text"}, and render an empty string when given
    role-wrapped messages.
    """
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    parts.append(_clean_vlm_none_keys(part))
        elif content:
            parts.append({"type": "text", "text": str(content)})
    return parts


def _processor_accepts_assistant_list_content(processor):
    cached = getattr(processor, "_unsloth_assistant_single_content", None)
    if cached is not None:
        return not cached

    probe = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]
    try:
        processor.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
        processor._unsloth_assistant_single_content = False
        return True
    except Exception:
        try:
            processor.apply_chat_template(
                _collapse_vlm_assistant_content(probe),
                tokenize=False,
                add_generation_prompt=False,
            )
            processor._unsloth_assistant_single_content = True
            return False
        except Exception:
            processor._unsloth_assistant_single_content = False
            return True


def _render_vlm_messages(processor, messages):
    normalize_vlm_processor_chat_template(processor, strict=True)
    if isinstance(messages, str):
        return messages

    render_messages = messages
    if not _processor_accepts_assistant_list_content(processor):
        render_messages = _collapse_vlm_assistant_content(render_messages)

    try:
        text = processor.apply_chat_template(
            render_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(text, str) and text.strip():
            return text
    except Exception as first_exc:
        first_error = first_exc
    else:
        first_error = None

    try:
        text = processor.apply_chat_template(
            _flatten_vlm_messages_to_content_parts(messages),
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(text, str) and text.strip():
            return text
    except Exception as second_exc:
        second_error = second_exc
    else:
        second_error = None

    try:
        text = processor.apply_chat_template(
            _flatten_vlm_content_for_text_template(render_messages),
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(text, str) and text.strip():
            return text
    except Exception as third_exc:
        third_error = third_exc
    else:
        third_error = None

    if first_error is not None:
        raise RuntimeError(
            "Unsloth MLX VLM: failed to render chat messages with this "
            "processor chat_template. Check that the dataset roles/content "
            "schema matches the model family, or pass a formatting_func that "
            "returns pre-rendered text."
        ) from (third_error or second_error or first_error)

    return ""


def _looks_like_mlx_chat_messages(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, dict) and ("role" in x or "content" in x) for x in value)
    )


def _render_mlx_messages(target, messages, *, is_vlm=False):
    if is_vlm:
        return _render_vlm_messages(target, _normalize_mlx_messages(messages, is_vlm=True))

    tokenizer = _get_processor_tokenizer(target)
    normalize_mlx_chat_template(tokenizer, is_vlm=False, strict=True)
    messages = _normalize_mlx_messages(messages, is_vlm=False)
    if isinstance(messages, str):
        return messages

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unsloth MLX text: failed to render chat messages with this "
            "tokenizer chat_template. Check that dataset roles/content match the "
            "model template, or pass formatting_func that returns pre-rendered text."
        ) from exc


def render_mlx_chat_example(
    target,
    item,
    *,
    dataset_text_field="text",
    is_vlm=False,
    allow_raw_text=True,
):
    """Render one text/chat training example for MLX text or VLM pipelines."""
    if item is None:
        return None
    if isinstance(item, str):
        return item if allow_raw_text else None
    if _looks_like_mlx_chat_messages(item):
        return _render_mlx_messages(target, item, is_vlm=is_vlm)
    if not isinstance(item, dict):
        return None

    if dataset_text_field in item and item[dataset_text_field]:
        value = item[dataset_text_field]
        if isinstance(value, str):
            return value
        if _looks_like_mlx_chat_messages(value):
            return _render_mlx_messages(target, value, is_vlm=is_vlm)

    for key in ("text", "content", "instruction"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value

    messages = _select_mlx_messages_or_raw(item)
    if messages is not item:
        return _render_mlx_messages(target, messages, is_vlm=is_vlm)

    if "prompt" in item and "completion" in item:
        prompt = render_mlx_chat_example(
            target, item.get("prompt"), dataset_text_field=dataset_text_field,
            is_vlm=is_vlm, allow_raw_text=allow_raw_text,
        )
        completion = render_mlx_chat_example(
            target, item.get("completion"), dataset_text_field=dataset_text_field,
            is_vlm=is_vlm, allow_raw_text=allow_raw_text,
        )
        if prompt is not None or completion is not None:
            return (prompt or "") + (completion or "")

    return None


def collect_mlx_texts(target, item, *, dataset_text_field="text", is_vlm=False):
    """Return one or more rendered text strings from a dataset row or formatter output."""
    if item is None:
        return []
    if isinstance(item, list) and not _looks_like_mlx_chat_messages(item):
        texts = []
        for value in item:
            texts.extend(
                collect_mlx_texts(
                    target, value,
                    dataset_text_field=dataset_text_field,
                    is_vlm=is_vlm,
                )
            )
        return texts
    text = render_mlx_chat_example(
        target,
        item,
        dataset_text_field=dataset_text_field,
        is_vlm=is_vlm,
    )
    return [text] if text else []


def _resize_vlm_images(images, image_size):
    from PIL import Image

    target = (image_size, image_size) if isinstance(image_size, int) else image_size
    resized = []
    for image in images:
        if isinstance(image, Image.Image):
            resized.append(image.convert("RGB").resize(target, Image.Resampling.LANCZOS))
        else:
            resized.append(image)
    return resized


def _extract_vlm_images(item, messages, image_size):
    images = []
    if isinstance(item, dict):
        image = item.get("image", item.get("images"))
        if image is not None:
            images = image if isinstance(image, list) else [image]

    if not images and isinstance(messages, list):
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    image = part.get("image")
                    if image is not None:
                        images.append(image)

    if not images and isinstance(messages, list):
        try:
            from .vision_utils import process_vision_info

            extracted = process_vision_info(messages, return_video_kwargs=True)
            if isinstance(extracted, tuple) and extracted:
                maybe_images = extracted[0]
                if maybe_images is not None:
                    images = maybe_images if isinstance(maybe_images, list) else [maybe_images]
        except Exception:
            pass

    return _resize_vlm_images(images, image_size)


def _format_vlm_images_for_processor(all_images):
    if not any(all_images):
        return None
    if all(len(images) == 1 for images in all_images):
        return [images[0] for images in all_images]
    return all_images


def _to_mx_vlm_batch(inputs):
    batch = {}
    for key, value in inputs.items():
        if isinstance(value, mx.array):
            batch[key] = value
        elif hasattr(value, "shape"):
            batch[key] = mx.array(value)
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], "shape"):
            try:
                batch[key] = mx.stack([
                    mx.array(x) if not isinstance(x, mx.array) else x
                    for x in value
                ])
            except Exception:
                batch[key] = mx.array(value[0]) if not isinstance(value[0], mx.array) else value[0]
        else:
            batch[key] = value

    if "input_ids" in batch:
        batch["input_ids"] = batch["input_ids"].astype(mx.int32)
    if "attention_mask" in batch:
        batch["attention_mask"] = batch["attention_mask"].astype(mx.int32)
    if "labels" in batch:
        batch["labels"] = batch["labels"].astype(mx.int32)
    if "pixel_values" in batch and batch["pixel_values"] is not None:
        batch["pixel_values"] = batch["pixel_values"].astype(mx.float32)

    return batch


def _processor_vlm_inputs(processor, texts, all_images, max_seq_length, suffixes=None):
    proc_kwargs = dict(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="np",
        add_special_tokens=False,
    )
    images = _format_vlm_images_for_processor(all_images)
    if images is not None:
        proc_kwargs["images"] = images
    if suffixes is not None and any(suffix is not None for suffix in suffixes):
        proc_kwargs["suffix"] = [suffix or "" for suffix in suffixes]
    return processor(**proc_kwargs)


def _collate_vlm_prompt_completion_batch(items, processor, max_seq_length, image_size):
    prompt_texts = []
    combined_texts = []
    all_images = []

    for item in items:
        prompt = _normalize_vlm_messages(item.get("prompt", ""))
        completion = _normalize_vlm_messages(item.get("completion", ""))
        prompt_messages = prompt if isinstance(prompt, list) else None
        completion_messages = completion if isinstance(completion, list) else None

        if prompt_messages is not None and completion_messages is not None:
            combined = prompt_messages + completion_messages
            images = _extract_vlm_images(item, combined, image_size)
            prompt_text = _render_vlm_messages(processor, prompt_messages)
            combined_text = _render_vlm_messages(processor, combined)
        else:
            images = _extract_vlm_images(item, prompt_messages or [], image_size)
            prompt_text = _render_vlm_messages(processor, prompt)
            completion_text = _render_vlm_messages(processor, completion)
            combined_text = prompt_text + completion_text

        prompt_texts.append(prompt_text)
        combined_texts.append(combined_text)
        all_images.append(images)

    combined_inputs = _processor_vlm_inputs(
        processor, combined_texts, all_images, max_seq_length
    )
    prompt_inputs = _processor_vlm_inputs(
        processor, prompt_texts, all_images, max_seq_length
    )
    batch = _to_mx_vlm_batch(combined_inputs)

    labels_np = np.array(batch["input_ids"].tolist(), dtype=np.int32)
    prompt_batch = _to_mx_vlm_batch(prompt_inputs)
    prompt_mask = prompt_batch.get("attention_mask")
    prompt_ids = prompt_batch["input_ids"]
    for row in range(labels_np.shape[0]):
        if prompt_mask is not None:
            prompt_len = int(mx.sum(prompt_mask[row]).item())
        else:
            prompt_len = int(mx.sum(prompt_ids[row] != 0).item())
        labels_np[row, :prompt_len] = -100
    labels = mx.array(labels_np)
    if "attention_mask" in batch:
        labels = mx.where(batch["attention_mask"] == 0, mx.array(-100), labels)
    batch["labels"] = labels.astype(mx.int32)
    return batch


def _collate_vlm_batch(items, processor, max_seq_length, image_size, formatting_func=None):
    """Collate a batch of VLM samples using the processor directly.

    Mirrors Unsloth's GPU UnslothVisionDataCollator:
    1. Extract images from messages or top-level keys
    2. Resize to uniform size
    3. apply_chat_template for text
    4. Single processor() call handles tokenization + image processing + padding
    """
    normalize_vlm_processor_chat_template(processor, strict=False)
    formatted_items = []
    for item in items:
        if formatting_func is not None:
            item = formatting_func(item)
        formatted_items.append(item)

    if (
        formatted_items
        and isinstance(formatted_items[0], dict)
        and "prompt" in formatted_items[0]
        and "completion" in formatted_items[0]
    ):
        return _collate_vlm_prompt_completion_batch(
            formatted_items, processor, max_seq_length, image_size
        )

    all_texts = []
    all_images = []
    all_suffixes = []

    for item in formatted_items:
        raw = _select_vlm_messages_or_raw(item)
        if raw is item:
            text = render_mlx_chat_example(processor, item, is_vlm=True)
            messages = []
        else:
            messages = _normalize_vlm_messages(raw)
            text = _render_vlm_messages(processor, messages)
        if not text:
            raise ValueError(
                "Unsloth MLX VLM: no text could be rendered for this dataset row. "
                "Provide messages/conversations, a text field, or a formatting_func."
            )
        images = _extract_vlm_images(item, messages, image_size)
        all_texts.append(text)
        all_images.append(images)
        all_suffixes.append(item.get("suffix") if isinstance(item, dict) else None)

    inputs = _processor_vlm_inputs(
        processor, all_texts, all_images, max_seq_length,
        suffixes=all_suffixes,
    )
    return _to_mx_vlm_batch(inputs)


def _apply_response_mask_to_vlm_batch(batch_dict, mask_fn):
    """Apply response masking to a VLM batch dict, adding 'labels' key.

    Converts input_ids to plain lists, runs the masking closure from
    dataset_utils.train_on_responses_only, and stores the result as
    an mx.array in batch_dict["labels"].
    """
    input_ids = batch_dict["input_ids"]
    ids_list = input_ids.tolist()
    result = mask_fn({"input_ids": ids_list})
    labels_list = result["labels"]
    if hasattr(labels_list, "tolist"):
        labels_list = labels_list.tolist()
    attention_mask = batch_dict.get("attention_mask")
    if attention_mask is not None:
        labels_array = mx.where(attention_mask == 0, mx.array(-100), mx.array(labels_list))
    else:
        labels_array = mx.array(labels_list)
    batch_dict["labels"] = labels_array
    return batch_dict


def create_vlm_batches(dataset, processor, config, batch_size, max_seq_length,
                       num_batches=None, seed=42, response_mask_fn=None,
                       formatting_func=None):
    """Pre-materialize VLM training batches using the processor directly.

    Mirrors Unsloth's GPU UnslothVisionDataCollator:
    resize images → processor(text, images, padding=True) → uniform batches.
    """
    import numpy as np

    image_size = _get_vlm_image_size(config, processor)

    indices = list(range(len(dataset)))
    np.random.seed(seed)
    if num_batches is not None:
        np.random.shuffle(indices)

    batch_indices = [
        indices[i : i + batch_size]
        for i in range(0, len(indices) - batch_size + 1, batch_size)
    ]

    batch_list = []
    for bi in batch_indices:
        items = [dataset[idx] for idx in bi]
        batch_dict = _collate_vlm_batch(
            items, processor, max_seq_length, image_size,
            formatting_func=formatting_func,
        )
        batch_dict = _prepare_vlm_batch_for_compile(batch_dict, config)
        if response_mask_fn is not None:
            batch_dict = _apply_response_mask_to_vlm_batch(batch_dict, response_mask_fn)
        batch_list.append(batch_dict)
        if num_batches is not None and len(batch_list) >= num_batches:
            break

    # Evaluate all tensors
    all_tensors = []
    for bd in batch_list:
        for v in bd.values():
            if isinstance(v, mx.array):
                all_tensors.append(v)
    if all_tensors:
        mx.eval(all_tensors)

    return batch_list


def iterate_vlm_training_batches(dataset, processor, config, batch_size,
                                  max_seq_length, seed=42,
                                  response_mask_fn=None,
                                  formatting_func=None):
    """Streaming VLM batch generator using processor directly.

    Yields batch dicts with input_ids, pixel_values, attention_mask,
    and optionally labels.
    """
    import numpy as np

    image_size = _get_vlm_image_size(config, processor)

    indices = list(range(len(dataset)))
    batch_indices = [
        indices[i : i + batch_size]
        for i in range(0, len(indices) - batch_size + 1, batch_size)
    ]

    while True:
        order = np.random.permutation(len(batch_indices))
        for b in order:
            items = [dataset[idx] for idx in batch_indices[b]]
            batch_dict = _collate_vlm_batch(
                items, processor, max_seq_length, image_size,
                formatting_func=formatting_func,
            )
            batch_dict = _prepare_vlm_batch_for_compile(batch_dict, config)
            if response_mask_fn is not None:
                batch_dict = _apply_response_mask_to_vlm_batch(batch_dict, response_mask_fn)
            yield batch_dict


def _prepare_dataset(dataset, tokenizer, dataset_text_field="text",
                     formatting_func=None, chat_template=None,
                     model_name=None, model_type=None):
    """Wrap a HuggingFace dataset into mlx-lm's dataset classes.

    Uses TextDataset + CacheDataset from mlx_lm so that tokenization
    (including EOS appending) matches mlx-lm's own training pipeline exactly.

    If a formatting_func is provided, each item is pre-formatted into a
    ``{"text": ...}`` dict before wrapping.

    Returns:
        A CacheDataset ready for ``iterate_batches``.
    """
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    normalize_mlx_chat_template(
        tokenizer,
        chat_template=chat_template,
        model_name=model_name,
        model_type=model_type,
        is_vlm=False,
        strict=False,
    )

    # Pre-format items into [{"text": str}, ...] so TextDataset can consume them.
    formatted = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            texts = collect_mlx_texts(
                tokenizer, result, dataset_text_field=dataset_text_field,
                is_vlm=False,
            )
        else:
            texts = collect_mlx_texts(
                tokenizer, item, dataset_text_field=dataset_text_field,
                is_vlm=False,
            )

        for text in texts:
            if text:
                formatted.append({"text": text})

    if not formatted:
        raise ValueError(
            f"No text data found. Provide a dataset with a '{dataset_text_field}' "
            "column, messages/conversations with a tokenizer chat_template, or "
            "a formatting_func that returns text."
        )

    return CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))


def create_batches(dataset, tokenizer, batch_size, max_seq_length,
                   num_batches=None, seed=42, dataset_text_field="text",
                   formatting_func=None, chat_template=None,
                   model_name=None, model_type=None):
    """Pre-tokenize and batch a HuggingFace dataset for MLX training.

    Uses iterate_batches from mlx_lm for efficient dynamic-padding batching:
    samples are sorted by length, grouped into batches, and padded to the
    max length within each batch (rounded up to the nearest multiple of 32),
    capped at max_seq_length.

    Tokenization is delegated to mlx_lm's TextDataset (appends EOS, etc.)
    so behaviour matches ``mlx_lm.lora`` exactly.

    Returns:
        List of (batch, lengths) tuples, where batch has shape
        (batch_size, padded_length) and lengths has shape (batch_size, 2)
        with [offset, length] per sequence (from iterate_batches).
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(
        dataset, tokenizer, dataset_text_field, formatting_func,
        chat_template=chat_template,
        model_name=model_name,
        model_type=model_type,
    )

    batch_pairs = []
    for batch, lengths_info in iterate_batches(
        ds, batch_size, max_seq_length,
        loop=(num_batches is not None),
        seed=seed,
    ):
        batch_pairs.append((batch, lengths_info, None))
        if num_batches is not None and len(batch_pairs) >= num_batches:
            break

    mx.eval([b for b, l, _ in batch_pairs] + [l for _, l, _ in batch_pairs])
    return batch_pairs


def iterate_training_batches(dataset, tokenizer, batch_size, max_seq_length,
                             seed=42, dataset_text_field="text",
                             formatting_func=None, chat_template=None,
                             model_name=None, model_type=None):
    """Streaming batch generator for MLX training.

    Wraps mlx-lm's iterate_batches(loop=True) as a generator, avoiding
    materializing all batches in memory at once. Useful for large datasets.

    Yields:
        (batch, lengths) tuples — same format as create_batches.
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(
        dataset, tokenizer, dataset_text_field, formatting_func,
        chat_template=chat_template,
        model_name=model_name,
        model_type=model_type,
    )

    for batch, lengths_info in iterate_batches(
        ds, batch_size, max_seq_length,
        loop=True,
        seed=seed,
    ):
        yield batch, lengths_info, None


def save_lora_adapters(model, path, adapter_config=None):
    """Save LoRA adapter weights to disk.

    Args:
        model: MLX model with LoRA layers.
        path: Directory to save adapters.
        adapter_config: Optional dict with LoRA config metadata.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Collect only trainable (LoRA) parameters — flatten nested dict for safetensors
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))

    if trainable:
        mx.save_safetensors(str(path / "adapters.safetensors"), trainable)

    if adapter_config:
        with open(path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)


def _get_model_config(model):
    """Extract config dict from an MLX model.

    mlx-lm stores the raw config dict at model._config when loaded.
    Falls back to reconstructing from model.args dataclass.
    """
    # Prefer the raw config dict stashed by our loader
    if hasattr(model, "_config") and isinstance(model._config, dict):
        return dict(model._config)

    # Reconstruct from the ModelArgs dataclass
    if hasattr(model, "args"):
        import dataclasses
        if dataclasses.is_dataclass(model.args):
            return dataclasses.asdict(model.args)

    return {}


def _get_src_path(model):
    """Get the original model source path/repo for copying auxiliary files."""
    return getattr(model, "_src_path", None)


def save_merged_model(model, tokenizer, path):
    """Fuse LoRA weights and save the full merged model.

    Produces an HF-compatible directory with sharded safetensors,
    config.json, tokenizer files, and a model card. The output can
    be reloaded with ``mlx_lm.load()`` or uploaded to HuggingFace Hub.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        path: Directory to save merged model.
    """
    from mlx_lm.utils import save_model, save_config, create_model_card
    from mlx.utils import tree_unflatten

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Fuse LoRA weights into base model using mlx-lm's pattern
    model.eval()
    fused_linears = [
        (n, m.fuse())
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
    de_lora_model = model

    # Save sharded safetensors + index.json
    save_model(path, de_lora_model, donate_model=False)

    # Save config.json
    config = _get_model_config(model)
    if config:
        save_config(config, config_path=path / "config.json")

    # Save tokenizer
    tokenizer.save_pretrained(str(path))

    # Copy auxiliary files (generation_config.json, *.py) from source
    src_path = _get_src_path(model)
    if src_path is not None:
        src_path = Path(src_path)
        if src_path.exists():
            import glob as globmod
            for pattern in ["generation_config.json", "*.py"]:
                for f in globmod.glob(str(src_path / pattern)):
                    shutil.copy(f, path)

    # Model card
    hf_repo = getattr(model, "_hf_repo", None)
    try:
        create_model_card(path, hf_repo)
    except Exception:
        # Fails if hf_repo doesn't exist on Hub — create a minimal card
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text("---\nlibrary_name: mlx\ntags:\n- mlx\n- unsloth\n---\n")

    print(f"Unsloth: Merged model saved to {path}")


def save_pretrained_merged(
    model,
    tokenizer,
    save_directory,
    push_to_hub=False,
    token=None,
    private=None,
    tags=None,
):
    """Save LoRA-fused model in HF-compatible format.

    This is the user-facing API matching the CUDA path's
    ``model.save_pretrained_merged()``.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        save_directory: Output directory path.
        push_to_hub: If True, upload to HuggingFace Hub after saving.
        token: HuggingFace token for pushing.
        private: Whether the HF repo should be private.
        tags: Additional tags for the model card.
    """
    save_merged_model(model, tokenizer, save_directory)

    if push_to_hub:
        push_to_hub_merged(
            model, tokenizer, save_directory,
            token=token, private=private, tags=tags,
        )


def _install_llama_cpp_macos(llama_cpp_folder="llama.cpp"):
    """Install llama.cpp on macOS by cloning and building with cmake."""
    import subprocess

    if not os.path.exists(llama_cpp_folder):
        print("Unsloth: Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp", llama_cpp_folder],
            check=True,
        )

    # Install Python dependencies — use gguf from the cloned repo to stay in sync
    gguf_py_dir = os.path.join(llama_cpp_folder, "gguf-py")
    if os.path.exists(gguf_py_dir):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", gguf_py_dir,
             "protobuf", "sentencepiece"],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gguf",
             "protobuf", "sentencepiece"],
            check=True, capture_output=True,
        )

    # Build with cmake (Metal support on macOS)
    build_dir = os.path.join(llama_cpp_folder, "build")
    print("Unsloth: Building llama.cpp with cmake...")
    subprocess.run(
        ["cmake", llama_cpp_folder, "-B", build_dir,
         "-DBUILD_SHARED_LIBS=OFF", "-DGGML_METAL=ON"],
        check=True, capture_output=True,
    )

    import psutil
    n_jobs = psutil.cpu_count() or 4
    targets = ["llama-quantize", "llama-cli", "llama-gguf-split"]
    target_args = []
    for t in targets:
        target_args += ["--target", t]

    subprocess.run(
        ["cmake", "--build", build_dir, "--config", "Release",
         f"-j{n_jobs}", "--clean-first"] + target_args,
        check=True, capture_output=True,
    )

    # Copy binaries to llama.cpp root
    bin_dir = os.path.join(build_dir, "bin")
    if os.path.exists(bin_dir):
        import glob as globmod
        for binary in globmod.glob(os.path.join(bin_dir, "llama-*")):
            shutil.copy(binary, llama_cpp_folder)

    print("Unsloth: llama.cpp installed successfully.")


def save_pretrained_gguf(
    model,
    tokenizer,
    save_directory,
    quantization_method="fast_quantized",
):
    """Save LoRA-fused model in GGUF format for llama.cpp inference.

    Follows the same pipeline as unsloth's CUDA path:
    1. Merge LoRA and save as HF-compatible safetensors
    2. Install/check llama.cpp
    3. Download and patch convert_hf_to_gguf.py
    4. Convert safetensors -> GGUF (bf16/f16 intermediate)
    5. Quantize to target format if needed

    Args:
        model: MLX model (with or without LoRA).
        tokenizer: Tokenizer to save alongside.
        save_directory: Output directory for GGUF file(s).
        quantization_method: Quantization to apply. Options:
            "not_quantized" - bf16, no quantization
            "fast_quantized" - q8_0 (fast, good quality)
            "quantized" - q4_k_m (small, fast inference)
            Or any llama.cpp quant type: q2_k, q3_k_m, q4_k_m, q5_k_m,
            q6_k, q8_0, f16, bf16, f32, etc.
    """
    from .llama_cpp import (
        convert_to_gguf,
        quantize_gguf,
        install_llama_cpp,
        check_llama_cpp,
        _download_convert_hf_to_gguf,
    )

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # Map friendly names to llama.cpp quant types
    quant_map = {
        "not_quantized": "bf16",
        "fast_quantized": "q8_0",
        "quantized": "q4_k_m",
        None: "q8_0",
    }
    quant_type = quant_map.get(quantization_method, quantization_method)

    # Apple Silicon always supports bf16
    model_dtype = "bf16"

    # Determine first_conversion (intermediate GGUF format before quantizing)
    # Same logic as unsloth CUDA path's save_to_gguf()
    if quant_type in ("bf16", "f16", "f32"):
        first_conversion = quant_type
    elif quant_type == "q8_0":
        # q8_0 can be done directly by convert_hf_to_gguf.py
        first_conversion = "None"
    else:
        # For all other quant types, first convert to bf16 then quantize
        first_conversion = "bf16"

    # GGUF conversion requires torch (used by llama.cpp's convert_hf_to_gguf.py)
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "Unsloth: GGUF export requires PyTorch.\n"
            "Install via: pip install torch\n"
            "torch is only needed for GGUF export, not for training."
        )

    # Step 1: Save merged model to a temp HF-format directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "merged"
        print("Unsloth: Merging LoRA weights and saving to 16-bit...")
        save_merged_model(model, tokenizer, tmp_path)

        # Step 2: Ensure llama.cpp is installed
        llama_cpp_folder = "llama.cpp"
        try:
            check_llama_cpp(llama_cpp_folder)
        except Exception:
            print("Unsloth: Installing llama.cpp (this only happens once)...")
            _install_llama_cpp_macos(llama_cpp_folder)

        # Step 3: Download and patch convert_hf_to_gguf.py
        converter = os.path.join(llama_cpp_folder, "unsloth_convert_hf_to_gguf.py")
        supported_text_archs = None
        supported_vision_archs = None
        if not os.path.exists(converter):
            result = _download_convert_hf_to_gguf()  # no args — uses defaults
            if isinstance(result, tuple) and len(result) >= 3:
                converter, supported_text_archs, supported_vision_archs = result[:3]

        # Step 4: Get model name for output filename
        hf_repo = getattr(model, "_hf_repo", None)
        if hf_repo:
            model_name = hf_repo.split("/")[-1]
        else:
            model_name = "model"

        output_base = str(save_directory / model_name)

        # Step 5: Convert HF -> GGUF
        print(f"Unsloth: Converting to GGUF format...")
        kwargs = dict(
            model_name=output_base,
            input_folder=str(tmp_path),
            model_dtype=model_dtype,
            quantization_type=first_conversion,
            converter_location=converter,
            is_vlm=False,
            is_gpt_oss=False,
            print_output=True,
        )
        if supported_text_archs is not None:
            kwargs["supported_text_archs"] = supported_text_archs
            kwargs["supported_vision_archs"] = supported_vision_archs
        convert_to_gguf(**kwargs)

        # Step 6: Quantize if the target quant differs from first_conversion
        if quant_type not in ("bf16", "f16", "f32") and first_conversion != "None":
            quantizer = os.path.join(llama_cpp_folder, "llama-quantize")
            base_gguf = f"{output_base}.{first_conversion.upper()}.gguf"
            final_gguf = f"{output_base}.{quant_type.upper()}.gguf"

            print(f"Unsloth: Quantizing to {quant_type}...")
            quantize_gguf(
                input_gguf=base_gguf,
                output_gguf=final_gguf,
                quant_type=quant_type,
                quantizer_location=quantizer,
                print_output=True,
            )
            # Remove intermediate bf16 gguf to save space
            if os.path.exists(base_gguf) and base_gguf != final_gguf:
                os.remove(base_gguf)
                print(f"Unsloth: Removed intermediate {Path(base_gguf).name}")

    # List produced files
    gguf_files = sorted(save_directory.glob("*.gguf"))
    for f in gguf_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"Unsloth: Saved {f.name} ({size_gb:.2f} GB)")
    print(f"Unsloth: GGUF export complete -> {save_directory}")


def push_to_hub_merged(
    model,
    tokenizer,
    save_directory,
    repo_id=None,
    token=None,
    private=None,
    tags=None,
):
    """Push merged model to HuggingFace Hub.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        save_directory: Local path with saved model (or where to save).
        repo_id: HuggingFace repo ID (e.g. "username/model-name").
            If None, uses save_directory as repo_id.
        token: HuggingFace token.
        private: Whether repo should be private.
        tags: Additional tags.
    """
    from mlx_lm.utils import upload_to_hub

    save_directory = Path(save_directory)

    # Save first if not already saved
    if not (save_directory / "model.safetensors.index.json").exists():
        save_merged_model(model, tokenizer, save_directory)

    if repo_id is None:
        repo_id = save_directory.name

    upload_to_hub(str(save_directory), repo_id)
    print(f"Unsloth: Pushed to https://huggingface.co/{repo_id}")


def push_to_hub_gguf(
    model,
    tokenizer,
    save_directory,
    repo_id,
    quantization_method="fast_quantized",
    token=None,
    private=None,
):
    """Export to GGUF and push to HuggingFace Hub.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        save_directory: Local path for GGUF output.
        repo_id: HuggingFace repo ID.
        quantization_method: GGUF quantization type.
        token: HuggingFace token.
        private: Whether repo should be private.
    """
    from huggingface_hub import HfApi

    save_directory = Path(save_directory)

    # Export to GGUF
    save_pretrained_gguf(model, tokenizer, save_directory, quantization_method)

    # Upload GGUF files
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    gguf_files = list(save_directory.glob("*.gguf"))
    for gguf_file in gguf_files:
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_id,
        )

    print(f"Unsloth: GGUF pushed to https://huggingface.co/{repo_id}")
