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
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path


from .mlx_cce import _get_runtime_cce


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
    return tm.model.embed_tokens


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
    _has_lm_head = hasattr(tm, "lm_head") and tm.lm_head is not None

    def _get_backbone(model):
        """Get backbone (for hidden states) from the live model tree."""
        if _has_wrapper:
            return model.language_model.model
        return model.model

    def _get_lm_weight_layer(model):
        """Get LM head or embed_tokens layer from the live model tree."""
        if _has_wrapper:
            tm = model.language_model
        else:
            tm = model
        if _has_lm_head:
            return tm.lm_head
        return tm.model.embed_tokens

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
            hidden = _get_backbone(model)(inputs)
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
            loss = loss.astype(mx.float32).sum() / ntoks
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
            hidden = _get_backbone(model)(inputs)
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
            loss = loss.astype(mx.float32).sum() / ntoks
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
        loss = ce.astype(mx.float32).sum() / ntoks
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
    if not hasattr(model, "language_model"):
        return False
    return any(hasattr(model, attr) for attr in
               ("vision_tower", "vision_model", "vision_encoder",
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
        # attention_mask is a padding indicator, not a causal attention mask;
        # we use it only for loss masking below.
        output = model(inputs, pixel_values=pixel_values)
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
        loss = ce.astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn


def _unpack_embed_result(embed_result, model):
    """Unpack get_input_embeddings result into embeds + backbone kwargs.

    Handles both plain mx.array returns and InputEmbeddingsFeatures dataclass
    (gemma4 per_layer_inputs, qwen3-vl position_ids/deepstack, etc.).
    """
    backbone_kwargs = {}
    if hasattr(embed_result, "inputs_embeds"):
        merged_embeds = embed_result.inputs_embeds
        # Gemma4: per-layer inputs for vision token injection
        if getattr(embed_result, "per_layer_inputs", None) is not None:
            backbone_kwargs["per_layer_inputs"] = embed_result.per_layer_inputs
        # Qwen3-VL deepstack: visual position masks + visual embeds
        if getattr(embed_result, "visual_pos_masks", None) is not None:
            backbone_kwargs["visual_pos_masks"] = embed_result.visual_pos_masks
        if getattr(embed_result, "deepstack_visual_embeds", None) is not None:
            backbone_kwargs["deepstack_visual_embeds"] = embed_result.deepstack_visual_embeds
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


def _vlm_cce_forward(model, batch_dict, image_token_ids=None,
                     assistant_token_id=0):
    """Shared VLM CCE forward: embed -> backbone -> hidden + masked_targets + ntoks."""
    input_ids = batch_dict["input_ids"]
    pixel_values = batch_dict.get("pixel_values")
    attention_mask = batch_dict.get("attention_mask")
    labels = batch_dict.get("labels")

    inputs = input_ids[:, :-1]

    embed_result = model.get_input_embeddings(inputs, pixel_values)
    merged_embeds, backbone_kwargs = _unpack_embed_result(embed_result, model)

    lm_model = model.language_model.model
    # Pass inputs alongside inputs_embeds — some backbones access inputs.shape
    # before checking inputs_embeds. The backbone ignores inputs when
    # inputs_embeds is provided.
    hidden = lm_model(inputs, inputs_embeds=merged_embeds, **backbone_kwargs)

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
            loss = loss.astype(mx.float32).sum() / ntoks
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
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks

    loss_fn._unsloth_cce_backend = "runtime-cce"
    return loss_fn


_VLM_BATCH_API = None  # cached: {"vds_config_second": bool, "use_train_kwarg": bool}


def _detect_vlm_batch_api():
    """Detect mlx-vlm API variant (>= 0.22 vs older) and cache the result."""
    global _VLM_BATCH_API
    if _VLM_BATCH_API is not None:
        return _VLM_BATCH_API
    import inspect
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import iterate_batches
    vds_params = list(inspect.signature(VisionDataset.__init__).parameters.keys())
    _VLM_BATCH_API = {
        "vds_config_second": len(vds_params) >= 4 and vds_params[2] == "config",
        "use_train_kwarg": "train" in inspect.signature(iterate_batches).parameters,
    }
    return _VLM_BATCH_API


def _make_vision_dataset(dataset, config, processor):
    """Create a VisionDataset, adapting to the installed mlx-vlm API."""
    api = _detect_vlm_batch_api()
    from mlx_vlm.trainer.datasets import VisionDataset
    if api["vds_config_second"]:
        return VisionDataset(dataset, config, processor)
    return VisionDataset(dataset, processor, config)


def _vlm_iter_kwargs(*, train, seed=42):
    """Build iterate_batches kwargs, adapting to the installed mlx-vlm API."""
    api = _detect_vlm_batch_api()
    if api["use_train_kwarg"]:
        return {"train": train}
    return {"loop": train, "seed": seed}


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
    # mask_fn returns plain lists when given plain list input
    if hasattr(labels_list, "tolist"):
        labels_list = labels_list.tolist()
    # Re-apply attention_mask: mlx_vlm pads input_ids with zeros, and the
    # HF mask_fn's "last turn" branch copies those zeros into labels as real
    # token IDs (not -100). These zeros pass (targets != -100), causing
    # spurious gradients and overcounted ntoks.
    attention_mask = batch_dict.get("attention_mask")
    if attention_mask is not None:
        labels_array = mx.where(attention_mask == 0, mx.array(-100), mx.array(labels_list))
    else:
        labels_array = mx.array(labels_list)
    batch_dict["labels"] = labels_array
    return batch_dict


def create_vlm_batches(dataset, processor, config, batch_size, max_seq_length,
                       num_batches=None, seed=42, response_mask_fn=None):
    """Pre-materialize VLM training batches using mlx_vlm's data pipeline.

    Args:
        dataset: HuggingFace dataset with image/text conversations.
        processor: VLM processor (from mlx_vlm.load).
        config: Model config dict.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        num_batches: Number of batches to materialize (None = all).
        seed: Random seed.
        response_mask_fn: Optional masking closure from train_on_responses_only.
            When provided, each batch dict gets a "labels" key with -100 for
            instruction tokens.

    Returns:
        List of batch dicts, each with input_ids, pixel_values, attention_mask,
        and optionally labels.
    """
    try:
        from mlx_vlm.trainer.sft_trainer import iterate_batches
    except ImportError:
        raise ImportError(
            "Unsloth: mlx-vlm trainer is required for VLM training. "
            "Install via: pip install mlx-vlm"
        )

    vision_ds = _make_vision_dataset(dataset, config, processor)
    iter_kwargs = _vlm_iter_kwargs(train=(num_batches is not None), seed=seed)

    batch_list = []
    for batch_dict in iterate_batches(
        vision_ds, batch_size, max_seq_length, **iter_kwargs,
    ):
        if response_mask_fn is not None:
            batch_dict = _apply_response_mask_to_vlm_batch(batch_dict, response_mask_fn)
        batch_list.append(batch_dict)
        if num_batches is not None and len(batch_list) >= num_batches:
            break

    # Evaluate all tensors in the batch dicts
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
                                  response_mask_fn=None):
    """Streaming VLM batch generator.

    Wraps mlx_vlm's iterate_batches(train=True) as a generator.

    Args:
        response_mask_fn: Optional masking closure from train_on_responses_only.

    Yields:
        Batch dicts with input_ids, pixel_values, attention_mask,
        and optionally labels.
    """
    try:
        from mlx_vlm.trainer.sft_trainer import iterate_batches
    except ImportError:
        raise ImportError(
            "Unsloth: mlx-vlm trainer is required for VLM training. "
            "Install via: pip install mlx-vlm"
        )

    vision_ds = _make_vision_dataset(dataset, config, processor)
    iter_kwargs = _vlm_iter_kwargs(train=True, seed=seed)

    for batch_dict in iterate_batches(
        vision_ds, batch_size, max_seq_length, **iter_kwargs,
    ):
        if response_mask_fn is not None:
            batch_dict = _apply_response_mask_to_vlm_batch(batch_dict, response_mask_fn)
        yield batch_dict


def _prepare_dataset(dataset, tokenizer, dataset_text_field="text",
                     formatting_func=None):
    """Wrap a HuggingFace dataset into mlx-lm's dataset classes.

    Uses TextDataset + CacheDataset from mlx_lm so that tokenization
    (including EOS appending) matches mlx-lm's own training pipeline exactly.

    If a formatting_func is provided, each item is pre-formatted into a
    ``{"text": ...}`` dict before wrapping.

    Returns:
        A CacheDataset ready for ``iterate_batches``.
    """
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    # Pre-format items into [{"text": str}, ...] so TextDataset can consume them.
    formatted = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            texts = result if isinstance(result, list) else [result]
        elif isinstance(item, dict):
            texts = []
            if dataset_text_field in item:
                texts = [item[dataset_text_field]]
            else:
                for key in ("text", "content", "instruction"):
                    if key in item:
                        texts = [item[key]]
                        break
        elif isinstance(item, str):
            texts = [item]
        else:
            continue

        for text in texts:
            if text:
                formatted.append({"text": text})

    if not formatted:
        raise ValueError(
            f"No text data found. Provide a dataset with a "
            f"'{dataset_text_field}' column."
        )

    return CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))


def create_batches(dataset, tokenizer, batch_size, max_seq_length,
                   num_batches=None, seed=42, dataset_text_field="text",
                   formatting_func=None):
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
        dataset, tokenizer, dataset_text_field, formatting_func
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
                             formatting_func=None):
    """Streaming batch generator for MLX training.

    Wraps mlx-lm's iterate_batches(loop=True) as a generator, avoiding
    materializing all batches in memory at once. Useful for large datasets.

    Yields:
        (batch, lengths) tuples — same format as create_batches.
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(
        dataset, tokenizer, dataset_text_field, formatting_func
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
