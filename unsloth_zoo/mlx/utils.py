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
import importlib
import json
import numpy as np
import os
import sys
import shutil
import tempfile
import threading
from pathlib import Path


from .cce import _get_runtime_cce


_LLAMA_CPP_PATCHER_ENV_LOCK = threading.Lock()


def _safe_token_denominator(ntoks):
    return mx.maximum(ntoks.astype(mx.float32), mx.array(1.0, dtype=mx.float32))


def _normalize_seed(seed, default=3407):
    return default if seed is None else int(seed)


def _get_transformer_layers(model):
    """Find transformer layers, unwrapping VLM wrappers if needed.

    VLMs: model.language_model.model.layers; text: model.(model.)layers.
    """
    m = getattr(model, 'language_model', model)
    m = getattr(m, 'model', m)
    return getattr(m, 'layers', None)


def _get_vision_encoder_layers(model):
    """Find vision tower encoder layers (e.g. SigLIP for Gemma3).

    Walks ``model.vision_tower.vision_model.encoder.layers`` and a few
    fallback paths used by other VLM families.
    """
    vt = getattr(model, "vision_tower", None) or getattr(model, "vision_model", None)
    if vt is None:
        return None
    for path in (
        ("vision_model", "encoder", "layers"),  # Gemma3 SigLIP
        ("blocks",),                             # Qwen2.5-VL ViT
        ("encoder", "layers"),
        ("vision_model", "layers"),
        ("layers",),
    ):
        cur = vt
        for name in path:
            cur = getattr(cur, name, None)
            if cur is None:
                break
        if cur is not None and hasattr(cur, "__len__") and len(cur) > 0:
            return cur
    return None


def _patch_layer_class_for_gc(layer_cls):
    if getattr(layer_cls, '_orig_call', None) is not None:
        return  # already patched
    layer_cls._orig_call = layer_cls.__call__
    fn = layer_cls.__call__

    def checkpointed_fn(self, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            self.update(params)
            return fn(self, *args, **kwargs)
        return mx.checkpoint(inner_fn)(self.trainable_parameters(), *args, **kwargs)

    layer_cls.__call__ = checkpointed_fn


def _unpatch_layer_class_gc(layer_cls):
    orig = getattr(layer_cls, '_orig_call', None)
    if orig is not None:
        layer_cls.__call__ = orig
        del layer_cls._orig_call


def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to language and vision tower layers.

    Patches each layer class's ``__call__`` with ``mx.checkpoint`` to recompute
    the forward during backward instead of storing activations. Trades ~30%
    extra compute for large memory savings — critical for VLM vision towers at
    native resolution, which can otherwise materialize tens of GB.
    """
    lm_layers = _get_transformer_layers(model)
    if lm_layers is not None and len(lm_layers) > 0:
        _patch_layer_class_for_gc(type(lm_layers[0]))

    vt_layers = _get_vision_encoder_layers(model)
    if vt_layers is not None and len(vt_layers) > 0:
        _patch_layer_class_for_gc(type(vt_layers[0]))


def remove_gradient_checkpointing(model):
    """Remove gradient checkpointing, restoring original layer __call__."""
    lm_layers = _get_transformer_layers(model)
    if lm_layers is not None and len(lm_layers) > 0:
        _unpatch_layer_class_gc(type(lm_layers[0]))
    vt_layers = _get_vision_encoder_layers(model)
    if vt_layers is not None and len(vt_layers) > 0:
        _unpatch_layer_class_gc(type(vt_layers[0]))


def _get_text_model(model):
    """Get the inner text model, unwrapping multimodal wrappers if present.

    Standard models (Llama, Qwen) are themselves the text model; multimodal
    wrappers (Gemma 4) expose it at model.language_model.
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


def _build_gemma_image_attention_mask(token_type_ids, attention_mask=None,
                                      window_size=None):
    seq_len = token_type_ids.shape[1]
    q_idx = mx.arange(seq_len)[:, None]
    kv_idx = mx.arange(seq_len)[None, :]
    causal = q_idx >= kv_idx
    if window_size is not None:
        causal = mx.logical_and(causal, q_idx < kv_idx + int(window_size))

    is_image = token_type_ids == 1
    previous_image = mx.concatenate(
        [mx.zeros_like(is_image[:, :1]), is_image[:, :-1]],
        axis=1,
    )
    new_image_start = mx.logical_and(is_image, mx.logical_not(previous_image))
    group_ids = mx.cumsum(new_image_start.astype(mx.int32), axis=1) - 1
    group_ids = mx.where(is_image, group_ids, -1)
    same_image_group = mx.logical_and(
        group_ids[:, :, None] == group_ids[:, None, :],
        group_ids[:, :, None] >= 0,
    )

    mask = mx.logical_or(causal[None, :, :], same_image_group)
    if attention_mask is not None:
        valid = attention_mask.astype(mx.bool_)
        mask = mx.logical_and(mask, valid[:, :, None])
        mask = mx.logical_and(mask, valid[:, None, :])
    return mx.expand_dims(mask, axis=1)


def _run_hidden_stack(stack, inputs, inputs_embeds=None, **kwargs):
    """Execute a language stack up to pre-lm_head hidden states."""
    from mlx_vlm.models.base import create_attention_mask

    config = getattr(stack, "config", None)
    if inputs_embeds is None:
        h = stack.embed_tokens(inputs)
    else:
        h = inputs_embeds
    if inputs_embeds is not None and _config_get(config, "model_type") == "gemma3_text":
        h *= mx.array(_config_get(config, "hidden_size")**0.5, mx.bfloat16).astype(h.dtype)

    cache = kwargs.get("cache")
    if cache is None:
        cache = [None] * len(stack.layers)
    mask = kwargs.get("mask")
    if mask is None:
        mask = kwargs.get("attention_mask_4d")
    if mask is None:
        mask = kwargs.get("attention_mask")
    token_type_ids = kwargs.get("token_type_ids")
    token_type_mask = None
    if token_type_ids is not None:
        attention_mask = kwargs.get("attention_mask")
        token_type_mask = _build_gemma_image_attention_mask(
            token_type_ids,
            attention_mask=attention_mask,
        )
    if mask is None and token_type_mask is None:
        mask = create_attention_mask(h, cache)

    # mlx-vlm gemma3/gemma4 stacks copy these onto the module, but fall back to
    # config (sliding_window) so a stack that only stores them there still
    # builds windowed masks instead of treating every layer as global.
    sliding_window_pattern = getattr(stack, "sliding_window_pattern", None)
    if sliding_window_pattern is None:
        sliding_window_pattern = _config_get(config, "sliding_window_pattern")
    window_size = getattr(stack, "window_size", None)
    if window_size is None:
        window_size = _config_get(config, "sliding_window")
    sliding_token_type_mask = None
    if token_type_ids is not None and sliding_window_pattern and window_size:
        sliding_token_type_mask = _build_gemma_image_attention_mask(
            token_type_ids,
            attention_mask=kwargs.get("attention_mask"),
            window_size=window_size,
        )

    for i, (layer, c) in enumerate(zip(stack.layers, cache)):
        local_mask = mask
        if token_type_mask is not None:
            is_global = not sliding_window_pattern or (
                i % sliding_window_pattern == sliding_window_pattern - 1
            )
            local_mask = token_type_mask if is_global else sliding_token_type_mask
        h = layer(h, local_mask, c)
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
        if (
            inputs_embeds is not None
            and "token_type_ids" in kwargs
            and _config_get(getattr(backbone, "config", None), "model_type") == "gemma3_text"
            and _has_hidden_stack(backbone)
        ):
            return _run_hidden_stack(backbone, inputs, inputs_embeds=inputs_embeds, **kwargs)
        if getattr(backbone, "lm_head", None) is not None and _has_hidden_stack(backbone):
            return _run_hidden_stack(backbone, inputs, inputs_embeds=inputs_embeds, **kwargs)
        embed_kwarg = _get_backbone_embed_kwarg(backbone)

        if "attention_mask_4d" in kwargs and "mask" not in kwargs:
            kwargs["mask"] = kwargs.pop("attention_mask_4d")
        backbone_kwargs = _filter_backbone_kwargs(backbone, kwargs)
        if inputs_embeds is not None:
            backbone_kwargs[embed_kwarg] = inputs_embeds
        return backbone(inputs, **backbone_kwargs)

    if not _has_direct_hidden_stack(model):
        raise ValueError("Text model does not expose a separable hidden-state backbone")
    return _run_hidden_stack(tm, inputs, inputs_embeds=inputs_embeds, **kwargs)


def _get_lm_head_layer(model):
    """Get the raw LM head layer (QuantizedLinear or Linear/Embedding).

    Prefers a separate lm_head (untied models like Qwen), else falls back to
    embed_tokens (tied models like Gemma/Llama). Returns the layer object (not
    its weight) so callers can read .weight/.scales/.biases/.group_size/.bits.
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
    """Whether the LM head weight is trainable (not frozen by LoRA).

    When frozen, its CCE gradient is a wasted V x chunk_size x H matmul per
    chunk, so callers wrap the weight with mx.stop_gradient (returns False).
    """
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    # Module-anchored so unrelated trainables containing the substring
    # "lora" are not treated as adapter state.
    adapter_keys = set(collect_mlx_lora_adapter_tensors(model))
    # Drop reload-leaked base tensors INSIDE a LoRA-wrapped lm_head (would
    # defeat the CCE memory guard) while keeping intentional trainables
    # like `lm_head.bias`. Shares the filter with save_trainable_adapters.
    _lora_module_names = [name for name, _ in iter_mlx_lora_modules(model)]
    lora_module_prefixes = tuple(f"{name}." for name in _lora_module_names if name)
    has_root_lora_module = any(name == "" for name in _lora_module_names)
    for key in trainable:
        if key in adapter_keys:
            continue
        if _is_base_tensor_inside_lora_module(
            key, lora_module_prefixes, has_root_lora_module,
        ):
            continue
        # Segment-match (not substring) so e.g.
        # `decoder.not_lm_head_router.weight` is not classified as lm_head.
        segments = key.split(".")
        is_lm_head_param = "lm_head" in segments
        is_embed_tokens_weight = (
            len(segments) >= 2
            and segments[-2] == "embed_tokens"
            and segments[-1] == "weight"
        )
        if is_lm_head_param or is_embed_tokens_weight:
            return True
    return len(trainable) == 0  # no LoRA = full fine-tuning


def make_cce_loss_fn(model):
    """Create a chunked cross-entropy (CCE) loss function.

    CCE computes loss directly from hidden states and the LM head weight,
    avoiding full logit materialization (big memory savings for large vocabs).

    Returns a function (model, batch, lengths, labels=None) -> (loss, ntoks).
    With labels, uses labels[:,1:] as targets and (targets != -100) as mask.
    The returned function has a ``_unsloth_cce_backend`` attribute for logging.
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
        # Backstop: quantized CCE backward zeros the weight gradient
        # (dequant->grad->requant unimplemented). Fine when the LM head is
        # frozen (LoRA on a quantized base flows grad through grad_hidden), but
        # full fine-tuning would silently skip the LM head update. The loader
        # rejects this earlier; this is a safety net.
        if getattr(model, "_unsloth_full_finetuning", False):
            raise ValueError(
                "Unsloth: full_finetuning=True with a quantized LM head is "
                "not supported. The CCE backward zeros the quantized weight "
                "gradient, so the LM head would never update. Load the "
                "unquantized base model for full fine-tuning, or use LoRA "
                "(full_finetuning=False) on this quantized base."
            )
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        quant_mode = getattr(lm_layer, "mode", "affine")
        print(
            "Unsloth: CCE using quantized matmul "
            f"(group_size={group_size}, bits={bits}, mode={quant_mode})"
        )
        _has_biases = hasattr(lm_layer, "biases")

        rt_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
            quantized=True,
            group_size=group_size,
            bits=bits,
            mode=quant_mode,
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
            bi = layer.biases if _has_biases else None
            if bi is None and quant_mode == "affine":
                bi = mx.zeros_like(sc)
            steps = mx.arange(1, targets.shape[1] + 1)
            length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
            # Widen unsigned dtypes so mx.where can inject the signed -100
            # (mx.where on uint crashes the torch-backed shim).
            targets = _normalize_cce_label_dtype(targets)
            if labels is None:
                mask = length_mask
            else:
                mask = mx.logical_and(targets != -100, length_mask)
            ignore = mx.array(-100, dtype=targets.dtype)
            masked_targets = mx.where(mask, targets, ignore)
            ntoks = mask.sum()
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,))  # runtime CCE validates dtype before narrowing
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
            length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
            # Same widen-to-int64 rationale as the quantized branch above.
            targets = _normalize_cce_label_dtype(targets)
            if labels is None:
                mask = length_mask
            else:
                mask = mx.logical_and(targets != -100, length_mask)
            ignore = mx.array(-100, dtype=targets.dtype)
            masked_targets = mx.where(mask, targets, ignore)
            ntoks = mask.sum()
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,))  # runtime CCE validates dtype before narrowing
            loss = rt_cce(hidden_flat, w, targets_flat)
            loss = loss.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
            return loss, ntoks

    loss_fn._unsloth_cce_backend = "runtime-cce"
    return loss_fn


def make_baseline_loss_fn():
    """Create a standard cross-entropy loss function (full logits via LM head).

    Used when use_cce=False. Returns a function
    (model, batch, lengths, labels=None) -> (loss, ntoks). With labels, uses
    labels[:,1:] and (targets != -100) as mask. The labels=None branch is
    byte-identical to ``mlx_lm.tuner.trainer.default_loss``.
    """
    def loss_fn(model, batch, lengths, labels=None):
        if labels is None:
            # Half-open [start, end) end-exclusive mask; matches CCE/labels paths
            # (:360, :393, :439) and mlx_lm's lengths convention.
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
            ce = nn.losses.cross_entropy(logits, targets) * mask
            ntoks = mask.sum()
            ce = ce.astype(mx.float32).sum() / ntoks
            return ce, ntoks
        # labels-aware path: train_on_responses_only style masking.
        inputs = batch[:, :-1]
        # Widen unsigned dtypes so mx.where(..., -100, ...) and the
        # `targets != -100` compare both see signed int64.
        targets = _normalize_cce_label_dtype(labels[:, 1:])
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
        if labels is None:
            mask = length_mask.astype(mx.float32)
        else:
            mask = mx.logical_and(targets != -100, length_mask).astype(mx.float32)
        # Replace -100 with 0 before CE, MLX has no ignore_index;
        # the mask already zeros out these positions in the loss.
        safe_targets = mx.where(
            targets == -100, mx.array(0, dtype=targets.dtype), targets,
        )
        ce = nn.losses.cross_entropy(logits, safe_targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
        return loss, ntoks

    return loss_fn


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------

# Image/vision special tokens that should never contribute to loss.
# Mirrors unsloth's IMAGE_TOKENS list in vision_utils.py.
_IMAGE_TOKEN_STRINGS = (
    "<|image|>",           # Llama 3.2 Vision, Phi 3.5, Gemma4
    "<|vision_start|>",    # Qwen
    "<|vision_end|>",      # Qwen
    "<|vision_pad|>",      # Qwen
    "<|image_pad|>",       # Qwen
    "<|video_pad|>",       # Qwen
    "<image>",             # PaliGemma, Llava, InternVL
    "</image>",            # InternVL
    "[IMG]",               # Mistral
    "[IMG_BREAK]",         # Mistral
    "[IMG_END]",           # Mistral
    "<image_soft_token>",  # Gemma 3
    "<start_of_image>",    # Gemma 3
    "<end_of_image>",      # Gemma 3
    "<|START_OF_IMG|>",    # Cohere
    "<|END_OF_IMG|>",      # Cohere
    "<|IMG_LINE_BREAK|>",  # Cohere
    "<|IMG_PATCH|>",       # Cohere
)


def _append_unique_int(ids, value):
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_unique_int(ids, item)
        return
    try:
        value = int(value)
    except (TypeError, ValueError):
        return
    if value not in ids:
        ids.append(value)


def _convert_token_to_id(tokenizer, token):
    try:
        token_ids = tokenizer.convert_tokens_to_ids([token])
    except Exception:
        try:
            token_ids = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return None
    if isinstance(token_ids, (list, tuple)):
        token_id = token_ids[0] if token_ids else None
    else:
        token_id = token_ids
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        return None
    return token_id


def _get_vlm_ignore_token_ids(processor=None, config=None, model=None):
    """Resolve VLM token IDs that should be ignored by SFT loss labels.

    Mirrors the CUDA vision collator's best-effort token masking without making
    the loss depend on processor state attached to the model.
    """
    if processor is None and model is not None:
        processor = getattr(model, "_processor", None)
    if config is None and model is not None:
        config = getattr(model, "_config", None)

    ids = []
    tokenizer = _get_processor_tokenizer(processor)
    if tokenizer is not None:
        for tok_str in _IMAGE_TOKEN_STRINGS:
            _append_unique_int(ids, _convert_token_to_id(tokenizer, tok_str))

        for attr in (
            "image_token",
            "video_token",
            "audio_token",
            "boi_token",
            "eoi_token",
        ):
            token = getattr(tokenizer, attr, None)
            if token is not None:
                _append_unique_int(ids, _convert_token_to_id(tokenizer, token))

        for attr in (
            "image_token_id",
            "video_token_id",
            "audio_token_id",
        ):
            _append_unique_int(ids, getattr(tokenizer, attr, None))

    for key in (
        "image_token_index",
        "image_token_id",
        "video_token_index",
        "video_token_id",
        "audio_token_index",
        "audio_token_id",
        "boi_token_index",
        "boi_token_id",
        "eoi_token_index",
        "eoi_token_id",
    ):
        _append_unique_int(ids, _config_get(config, key, None))

    if not ids:
        return None
    return ids  # plain Python list; avoids mx.eval in the hot path


def _get_image_token_ids(model):
    """Backward-compatible wrapper for legacy callers."""
    return _get_vlm_ignore_token_ids(model=model)


def _normalize_cce_label_dtype(labels):
    """Widen unsigned label dtypes to int64 so masking can inject -100.

    uint8/16/32 fit losslessly in int64. uint64 values in (2**63, 2**64)
    wrap negative on cast and are routed to a positive out-of-vocab
    sentinel (1<<62) so runtime CCE's validity check NaN-poisons those
    rows instead of letting an overflowed uint64 like 2**64-100 collide
    with ignore_index=-100. Signed/float dtypes pass through unchanged.
    """
    if labels is None:
        return labels
    dtype = getattr(labels, "dtype", None)
    if dtype is None:
        return labels
    uint64_dtype = getattr(mx, "uint64", None)
    if uint64_dtype is not None and dtype == uint64_dtype:
        # Cast to signed int64 first; uint64 compares crash the torch shim.
        # `labels_i64 < 0` catches every uint64 >= 2**63.
        labels_i64 = labels.astype(mx.int64)
        invalid_sentinel = mx.array((1 << 62), dtype=mx.int64)
        return mx.where(labels_i64 < 0, invalid_sentinel, labels_i64)
    unsigned_dtypes = tuple(
        d for d in (
            getattr(mx, "uint8", None),
            getattr(mx, "uint16", None),
            getattr(mx, "uint32", None),
        )
        if d is not None
    )
    if dtype in unsigned_dtypes:
        return labels.astype(mx.int64)
    return labels


def _mask_label_token_ids(targets, ignore_token_ids, ignore_index=-100):
    if not ignore_token_ids:
        return targets
    # Widen unsigned dtypes so the signed ignore_index can be injected.
    targets = _normalize_cce_label_dtype(targets)
    should_ignore = targets == ignore_token_ids[0]
    for tok_id in ignore_token_ids[1:]:
        should_ignore = should_ignore | (targets == tok_id)
    ignore = mx.array(ignore_index, dtype=targets.dtype)
    return mx.where(should_ignore, ignore, targets)


def _mask_image_tokens(targets, image_token_ids):
    """Set image/vision token positions in targets to -100."""
    return _mask_label_token_ids(targets, image_token_ids)


def _apply_vlm_label_masks(batch_dict, labels=None, ignore_token_ids=None,
                           ignore_index=-100):
    # Do NOT narrow to int32: runtime CCE validates the original dtype before
    # its own narrow, so wide/unsigned invalid ids (e.g. uint32(2**32-100))
    # must survive as out-of-vocab sentinels instead of wrapping to -100.
    # Prefer the pre-narrow raw carrier when deriving labels from input_ids.
    if labels is None:
        labels = batch_dict.get(_RAW_INPUT_IDS_FOR_LABELS, batch_dict["input_ids"])
    labels = _normalize_cce_label_dtype(labels)
    labels = _mask_label_token_ids(labels, ignore_token_ids, ignore_index)
    attention_mask = batch_dict.get("attention_mask")
    if attention_mask is not None:
        ignore = mx.array(ignore_index, dtype=labels.dtype)
        labels = mx.where(attention_mask == 0, ignore, labels)
    return labels


def _mask_prompt_tokens(targets, assistant_token_id):
    """Mask tokens before the first assistant response (train_on_completions_only).

    Per row, positions before the first assistant_token_id become -100. If the
    token is absent, the row is left unmasked (assumed all completion).
    """
    if assistant_token_id <= 0:
        return targets
    targets = _normalize_cce_label_dtype(targets)
    is_assistant = (targets == assistant_token_id)
    cumulative = mx.cumsum(is_assistant.astype(mx.int32), axis=1)
    has_assistant = mx.any(is_assistant, axis=1, keepdims=True)
    prompt_mask = mx.logical_and(has_assistant, cumulative == 0)
    ignore = mx.array(-100, dtype=targets.dtype)
    return mx.where(prompt_mask, ignore, targets)


def _is_vlm_model(model) -> bool:
    """Check if model is a VLM (has language_model + vision component)."""
    if getattr(model, "_unsloth_text_only_vlm", False):
        return False
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


def _trim_sequence_aligned_vlm_kwargs(extra_kwargs, seq_len):
    """Trim auxiliary VLM tensors that track token sequence length.

    Some VLMs, notably Qwen3-VL, pass position_ids through the batch. The loss
    function shifts input_ids for causal LM training, so these auxiliary tensors
    must be shifted to the same token length before the model builds rotary
    embeddings.
    """
    position_ids = extra_kwargs.get("position_ids")
    if position_ids is not None and hasattr(position_ids, "shape"):
        if position_ids.shape[-1] != seq_len:
            extra_kwargs["position_ids"] = position_ids[..., :seq_len]
    return extra_kwargs


def make_vlm_baseline_loss_fn(model=None, assistant_token_id=0,
                              ignore_token_ids=None):
    """Create a standard cross-entropy loss function for VLMs.

    Takes a batch dict with input_ids, pixel_values, attention_mask.

    Returns:
        A function (model, batch_dict) -> (loss, ntoks).
    """
    _image_token_ids = (
        ignore_token_ids
        if ignore_token_ids is not None
        else (_get_image_token_ids(model) if model is not None else None)
    )
    _assistant_token_id = assistant_token_id

    def loss_fn(model, batch_dict):
        input_ids = batch_dict["input_ids"]
        pixel_values = batch_dict.get("pixel_values")
        attention_mask = batch_dict.get("attention_mask")
        labels = batch_dict.get("labels")

        # Forward full sequence then shift (Qwen3-VL mRoPE/deepstack need it);
        # mirrors `_vlm_cce_forward` so use_cce={True,False} stay in parity.
        inputs = input_ids
        fwd_mask = attention_mask

        # Model owns the causal mask. Pass through extras (e.g. image_grid_thw
        # for Qwen). Strip every private `_unsloth_*` carrier (raw-ids plus the
        # _unsloth_collated_position_ids marker) so model(...) never sees a kwarg
        # it would reject, matching _vlm_cce_forward's filter.
        fwd_kwargs = {
            k: v for k, v in batch_dict.items()
            if k not in ("input_ids", "pixel_values", "attention_mask", "labels")
            and not k.startswith("_unsloth_")
            and v is not None
        }
        fwd_kwargs = _trim_sequence_aligned_vlm_kwargs(fwd_kwargs, inputs.shape[1])
        output = model(inputs, pixel_values=pixel_values, mask=fwd_mask, **fwd_kwargs)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits.astype(mx.float32)
        # Drop the final position so logits predict the next token.
        logits = logits[:, :-1, :]

        if labels is not None:
            # Extra masks keep externally supplied labels compatible.
            targets = _normalize_cce_label_dtype(labels[:, 1:])
            targets = _mask_image_tokens(targets, _image_token_ids)
            # _apply_vlm_label_masks ignores assistant boundaries; mask here.
            targets = _mask_prompt_tokens(targets, _assistant_token_id)
            logits, targets = _align_logits_with_labels(logits, targets)
            if attention_mask is not None:
                length_mask = attention_mask[:, 1:][:, :targets.shape[1]]
            else:
                length_mask = mx.ones_like(targets, dtype=mx.float32)
            mask = mx.logical_and(targets != -100, length_mask).astype(mx.float32)
        else:
            # Prefer the raw (pre-narrow) input_ids so wide invalid ids
            # (e.g. np.uint32(2**32-100)) reach runtime CCE as out-of-vocab
            # sentinels instead of wrapping to -100 via the int32 narrow.
            target_source = batch_dict.get(_RAW_INPUT_IDS_FOR_LABELS, input_ids)
            targets = _normalize_cce_label_dtype(target_source[:, 1:])

            # Vision token injection can change seq length
            logits, targets = _align_logits_with_labels(logits, targets)

            # Build mask from attention_mask (shifted to match targets)
            if attention_mask is not None:
                length_mask = attention_mask[:, 1:]
                length_mask = length_mask[:, :targets.shape[1]]
            else:
                length_mask = mx.ones_like(targets, dtype=mx.float32)

            targets = _mask_image_tokens(targets, _image_token_ids)
            targets = _mask_prompt_tokens(targets, _assistant_token_id)
            # Exclude masked positions from length_mask
            mask = mx.where(
                targets == -100,
                mx.array(0, dtype=length_mask.dtype),
                length_mask,
            )

        # Replace -100 with 0 before CE, MLX has no ignore_index;
        # the mask already zeros out these positions in the loss.
        safe_targets = mx.where(
            targets == -100, mx.array(0, dtype=targets.dtype), targets,
        )
        ce = nn.losses.cross_entropy(logits, safe_targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
        return loss, ntoks

    loss_fn._unsloth_cce_backend = "baseline-ce"
    return loss_fn


def _unpack_embed_result(embed_result, model):
    """Unpack get_input_embeddings result into embeds + backbone kwargs.

    Handles plain mx.array returns and the InputEmbeddingsFeatures dataclass
    (gemma4 per_layer_inputs, qwen3-vl position_ids/deepstack, etc.).
    """
    backbone_kwargs = {}
    if hasattr(embed_result, "inputs_embeds"):
        merged_embeds = embed_result.inputs_embeds
        if getattr(embed_result, "attention_mask_4d", None) is not None:
            model_type = _config_get(getattr(model, "config", None), "model_type")
            if model_type != "gemma3":
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

    # Qwen-VL family: some get_input_embeddings paths stash position_ids on the
    # language model wrapper; the inner backbone needs them explicitly.
    # Do not override position_ids explicitly returned by InputEmbeddingsFeatures
    # (for example when the collator passed CUDA-parity mRoPE IDs through the
    # embedder).
    # When no position_ids were stashed (e.g. text-only samples or simple
    # images without grid_thw), generate sequential ones so the backbone
    # doesn't crash accessing cache.offset with cache=None.
    lm = getattr(model, "language_model", None)
    if lm is not None and "position_ids" not in backbone_kwargs:
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

    # Forward full sequence then shift hidden[:, :-1] (Qwen3-VL mRoPE/deepstack).
    inputs = input_ids
    fwd_attn_mask = attention_mask

    # Collect extra keys (e.g. image_grid_thw for Qwen). Read the private
    # collated-position flag first, then strip all `_unsloth_*` carriers
    # so the embedder/backbone never sees them.
    use_collated_position_ids = bool(
        batch_dict.get("_unsloth_collated_position_ids")
    )
    extra_kwargs = {
        k: v for k, v in batch_dict.items()
        if k not in ("input_ids", "pixel_values", "attention_mask", "labels")
        and not k.startswith("_unsloth_")
        and v is not None
    }
    extra_kwargs = _trim_sequence_aligned_vlm_kwargs(extra_kwargs, inputs.shape[1])

    embed_result = model.get_input_embeddings(
        inputs,
        pixel_values,
        mask=fwd_attn_mask,
        **extra_kwargs,
    )
    merged_embeds, backbone_kwargs = _unpack_embed_result(embed_result, model)
    # Prefer collator-built mRoPE IDs when present. Qwen/GLM collators build
    # CUDA-parity full-sequence positions; recomputing inside the embedder moved
    # Qwen3-VL first-step loss from ~6.45 to ~6.90 on the real-cat fixture.
    if use_collated_position_ids and "position_ids" in extra_kwargs:
        backbone_kwargs["position_ids"] = extra_kwargs["position_ids"]
    if "token_type_ids" in extra_kwargs:
        backbone_kwargs["token_type_ids"] = extra_kwargs["token_type_ids"]
        if attention_mask is not None:
            backbone_kwargs["attention_mask"] = attention_mask

    hidden = _forward_text_hidden_states(
        model,
        inputs,
        inputs_embeds=merged_embeds,
        **backbone_kwargs,
    )
    hidden = hidden[:, :-1, :]

    if labels is not None:
        # Extra mask keeps externally supplied labels compatible.
        targets = labels[:, 1:]
        masked_targets = _mask_image_tokens(targets, image_token_ids)
        if attention_mask is not None:
            length_mask = attention_mask[:, 1:][:, :masked_targets.shape[1]]
        else:
            length_mask = mx.ones_like(masked_targets, dtype=mx.bool_)
        masked_targets = mx.where(
            mx.logical_and(masked_targets != -100, length_mask),
            masked_targets,
            -100,
        )
        # Completion-only masking; _apply_vlm_label_masks doesn't do this.
        masked_targets = _mask_prompt_tokens(masked_targets, assistant_token_id)
        ntoks = (masked_targets != -100).sum()
    else:
        # Prefer the raw (pre-narrow) input_ids so wide invalid ids
        # (e.g. np.uint32(2**32-100)) reach runtime CCE as out-of-vocab
        # sentinels instead of wrapping to -100 via the int32 narrow.
        target_source = batch_dict.get(_RAW_INPUT_IDS_FOR_LABELS, input_ids)
        targets = _normalize_cce_label_dtype(target_source[:, 1:])

        if attention_mask is not None:
            length_mask = attention_mask[:, 1:][:, :targets.shape[1]]
        else:
            length_mask = mx.ones_like(targets, dtype=mx.float32)

        ignore = mx.array(-100, dtype=targets.dtype)
        masked_targets = mx.where(length_mask, targets, ignore)

        masked_targets = _mask_image_tokens(masked_targets, image_token_ids)
        # Completion-only: mask prompt before first assistant response
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


def _grid_thw_to_mx_array(grid_thw):
    if grid_thw is None:
        return None
    return mx.array(grid_thw, dtype=mx.int32)


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


def _normalize_numpy_cce_labels(labels):
    """numpy-side analogue of `_normalize_cce_label_dtype` for VLM expand paths.

    np.uint64 values above 2**63-1 would OverflowError when packed into the
    int64 buffer used by `_expand_image_token_sequences` / `_expand_token_runs`;
    route them to a positive 1<<62 sentinel (so an overflowed 2**64-100 cannot
    silently masquerade as ignore_index=-100) and downcast other unsigned
    dtypes to int64.
    """
    labels_np = np.asarray(labels)
    if labels_np.dtype == np.uint64:
        overflow_mask = labels_np > np.uint64((1 << 63) - 1)
        return np.where(
            overflow_mask,
            np.int64(1 << 62),
            labels_np.astype(np.int64),
        )
    if np.issubdtype(labels_np.dtype, np.unsignedinteger):
        return labels_np.astype(np.int64)
    return labels_np


def _expand_image_token_sequences(
    input_ids,
    attention_mask,
    image_token_id,
    repeat_count,
    labels=None,
):
    input_ids_np = np.asarray(input_ids)
    attention_mask_np = (
        np.asarray(attention_mask)
        if attention_mask is not None
        else np.ones_like(input_ids_np, dtype=np.int32)
    )
    labels_np = _normalize_numpy_cce_labels(labels) if labels is not None else None

    expanded_ids = []
    expanded_masks = []
    expanded_labels = [] if labels_np is not None else None
    max_len = 0
    for row_idx, (row_ids, row_mask) in enumerate(zip(input_ids_np, attention_mask_np)):
        new_ids = []
        new_mask = []
        new_labels = [] if labels_np is not None else None
        row_labels_list = labels_np[row_idx].tolist() if labels_np is not None else None
        for pos, (token_id, mask_value) in enumerate(zip(row_ids.tolist(), row_mask.tolist())):
            if int(token_id) == int(image_token_id):
                new_ids.extend([int(image_token_id)] * int(repeat_count))
                new_mask.extend([int(mask_value)] * int(repeat_count))
                if new_labels is not None:
                    new_labels.extend([-100] * int(repeat_count))
            else:
                new_ids.append(int(token_id))
                new_mask.append(int(mask_value))
                if new_labels is not None:
                    new_labels.append(int(row_labels_list[pos]))
        expanded_ids.append(new_ids)
        expanded_masks.append(new_mask)
        if expanded_labels is not None:
            expanded_labels.append(new_labels)
        max_len = max(max_len, len(new_ids))

    padded_ids = np.zeros((len(expanded_ids), max_len), dtype=np.int32)
    padded_masks = np.zeros((len(expanded_masks), max_len), dtype=np.int32)
    # int64 so wide invalid labels survive without OverflowError before the
    # runtime CCE validity check; CCE owns dtype narrowing post-validate.
    padded_labels = (
        np.full((len(expanded_labels), max_len), -100, dtype=np.int64)
        if expanded_labels is not None else None
    )
    for row_idx, (row_ids, row_mask) in enumerate(zip(expanded_ids, expanded_masks)):
        row_len = len(row_ids)
        padded_ids[row_idx, :row_len] = row_ids
        padded_masks[row_idx, :row_len] = row_mask
        if padded_labels is not None:
            padded_labels[row_idx, :row_len] = expanded_labels[row_idx]

    if padded_labels is not None:
        return mx.array(padded_ids), mx.array(padded_masks), mx.array(padded_labels)
    return mx.array(padded_ids), mx.array(padded_masks)


def _expand_token_runs(
    input_ids,
    attention_mask,
    replacements_by_batch,
    labels=None,
):
    input_ids_np = np.asarray(input_ids)
    attention_mask_np = (
        np.asarray(attention_mask)
        if attention_mask is not None
        else np.ones_like(input_ids_np, dtype=np.int32)
    )
    labels_np = _normalize_numpy_cce_labels(labels) if labels is not None else None

    expanded_ids = []
    expanded_masks = []
    expanded_labels = [] if labels_np is not None else None
    max_len = 0
    for row_idx, (row_ids, row_mask, replacements) in enumerate(zip(
        input_ids_np,
        attention_mask_np,
        replacements_by_batch,
    )):
        replacements = sorted(replacements, key=lambda item: item[0])
        new_ids = []
        new_mask = []
        new_labels = [] if labels_np is not None else None
        row_labels_list = labels_np[row_idx].tolist() if labels_np is not None else None
        prev = 0
        row_ids_list = row_ids.tolist()
        row_mask_list = row_mask.tolist()
        for start, end, token_id, repeat in replacements:
            start = int(start)
            end = int(end)
            repeat = int(repeat)
            if start > prev:
                new_ids.extend(row_ids_list[prev:start])
                new_mask.extend(row_mask_list[prev:start])
                if new_labels is not None:
                    new_labels.extend(row_labels_list[prev:start])
            new_ids.extend([int(token_id)] * repeat)
            fill_mask = int(row_mask_list[start]) if start < len(row_mask_list) else 1
            new_mask.extend([fill_mask] * repeat)
            if new_labels is not None:
                new_labels.extend([-100] * repeat)
            prev = end
        if prev < len(row_ids_list):
            new_ids.extend(row_ids_list[prev:])
            new_mask.extend(row_mask_list[prev:])
            if new_labels is not None:
                new_labels.extend(row_labels_list[prev:])
        expanded_ids.append(new_ids)
        expanded_masks.append(new_mask)
        if expanded_labels is not None:
            expanded_labels.append(new_labels)
        max_len = max(max_len, len(new_ids))

    padded_ids = np.zeros((len(expanded_ids), max_len), dtype=np.int32)
    padded_masks = np.zeros((len(expanded_masks), max_len), dtype=np.int32)
    # int64 so wide invalid labels survive expansion without OverflowError.
    padded_labels = (
        np.full((len(expanded_labels), max_len), -100, dtype=np.int64)
        if expanded_labels is not None else None
    )
    for row_idx, (row_ids, row_mask) in enumerate(zip(expanded_ids, expanded_masks)):
        row_len = len(row_ids)
        padded_ids[row_idx, :row_len] = row_ids
        padded_masks[row_idx, :row_len] = row_mask
        if padded_labels is not None:
            padded_labels[row_idx, :row_len] = expanded_labels[row_idx]

    if padded_labels is not None:
        return mx.array(padded_ids), mx.array(padded_masks), mx.array(padded_labels)
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
    grid_as_array = model_type in {"glm4v", "glm_ocr"}
    if image_grid_thw is not None:
        # GLM native mlx-vlm paths call .tolist(), .prod(), and slicing on
        # grids; Qwen/Paddle compile patches expect Python tuples.
        batch_dict["image_grid_thw"] = (
            _grid_thw_to_mx_array(image_grid_thw) if grid_as_array else image_grid_thw
        )
    if video_grid_thw is not None:
        batch_dict["video_grid_thw"] = (
            _grid_thw_to_mx_array(video_grid_thw) if grid_as_array else video_grid_thw
        )
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
            batch_dict["_unsloth_collated_position_ids"] = True

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
            batch_dict["_unsloth_collated_position_ids"] = True

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
            _labels = batch_dict.get("labels")
            _raw_labels = batch_dict.get(_RAW_INPUT_IDS_FOR_LABELS)
            # Labels absent: expand the raw pre-narrow carrier so labels-free
            # paths still see wide invalid ids for NaN-poisoning. Labels
            # present: pop the now-stale carrier so a pre-expansion copy
            # cannot break _apply_response_mask_to_vlm_batch with a shape mismatch.
            _label_source = _labels if _labels is not None else _raw_labels
            _expanded = _expand_image_token_sequences(
                input_ids=input_ids,
                attention_mask=batch_dict.get("attention_mask"),
                image_token_id=int(_config_get(config, "image_token_index")),
                repeat_count=int(_config_get(config, "num_image_tokens")),
                labels=_label_source,
            )
            if _labels is not None:
                batch_dict["input_ids"], batch_dict["attention_mask"], batch_dict["labels"] = _expanded
                batch_dict.pop(_RAW_INPUT_IDS_FOR_LABELS, None)
            elif _raw_labels is not None:
                batch_dict["input_ids"], batch_dict["attention_mask"], batch_dict[_RAW_INPUT_IDS_FOR_LABELS] = _expanded
            else:
                batch_dict["input_ids"], batch_dict["attention_mask"] = _expanded

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
                _labels = batch_dict.get("labels")
                _raw_labels = batch_dict.get(_RAW_INPUT_IDS_FOR_LABELS)
                # See multi_modality branch above for why we expand the raw
                # carrier when labels are absent and pop it when labels exist.
                _label_source = _labels if _labels is not None else _raw_labels
                _expanded = _expand_token_runs(
                    input_ids=input_ids,
                    attention_mask=batch_dict.get("attention_mask"),
                    replacements_by_batch=tuple(replacements),
                    labels=_label_source,
                )
                if _labels is not None:
                    batch_dict["input_ids"], batch_dict["attention_mask"], batch_dict["labels"] = _expanded
                    batch_dict.pop(_RAW_INPUT_IDS_FOR_LABELS, None)
                elif _raw_labels is not None:
                    batch_dict["input_ids"], batch_dict["attention_mask"], batch_dict[_RAW_INPUT_IDS_FOR_LABELS] = _expanded
                else:
                    batch_dict["input_ids"], batch_dict["attention_mask"] = _expanded

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
                _labels = batch_dict.get("labels")
                _raw_labels = batch_dict.get(_RAW_INPUT_IDS_FOR_LABELS)
                # See multi_modality branch above for why we expand the raw
                # carrier when labels are absent and pop it when labels exist.
                _label_source = _labels if _labels is not None else _raw_labels
                _expanded = _expand_token_runs(
                    input_ids=input_ids,
                    attention_mask=batch_dict.get("attention_mask"),
                    replacements_by_batch=tuple(replacements),
                    labels=_label_source,
                )
                if _labels is not None:
                    batch_dict["input_ids"], batch_dict["attention_mask"], batch_dict["labels"] = _expanded
                    batch_dict.pop(_RAW_INPUT_IDS_FOR_LABELS, None)
                elif _raw_labels is not None:
                    batch_dict["input_ids"], batch_dict["attention_mask"], batch_dict[_RAW_INPUT_IDS_FOR_LABELS] = _expanded
                else:
                    batch_dict["input_ids"], batch_dict["attention_mask"] = _expanded

    return batch_dict


def make_vlm_cce_loss_fn(model, assistant_token_id=0, ignore_token_ids=None):
    """Create a CCE loss function for VLMs.

    Uses model.get_input_embeddings() for merged vision+text embeddings, runs
    the backbone to pre-lm_head hidden states, and applies CCE. Falls back to
    baseline loss when get_input_embeddings is unavailable.

    assistant_token_id > 0 enables completion-only training (mask tokens before
    the first occurrence). Returns a function (model, batch_dict) -> (loss, ntoks).
    """
    if not hasattr(model, "get_input_embeddings"):
        import warnings
        warnings.warn(
            "VLM model does not have get_input_embeddings — "
            "falling back to baseline CE loss.",
            stacklevel=2,
        )
        return make_vlm_baseline_loss_fn(
            model,
            assistant_token_id=assistant_token_id,
            ignore_token_ids=ignore_token_ids,
        )

    tm = _get_text_model(model)
    if getattr(tm, "model", None) is None and not _has_direct_hidden_stack(model):
        import warnings
        warnings.warn(
            "VLM text stack does not expose a separable hidden-state backbone for CCE; "
            "falling back to baseline CE loss.",
            stacklevel=2,
        )
        return make_vlm_baseline_loss_fn(
            model,
            assistant_token_id=assistant_token_id,
            ignore_token_ids=ignore_token_ids,
        )

    softcap = _get_logit_softcap(model)
    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)
    # Evaluate once (after LoRA setup); trainability doesn't change mid-training.
    _skip_weight_grad = not _is_lm_head_trainable(model)

    _image_token_ids = (
        ignore_token_ids
        if ignore_token_ids is not None
        else _get_image_token_ids(model)
    )
    if _image_token_ids is not None:
        print(f"Unsloth: Masking {len(_image_token_ids)} image token IDs from VLM loss.")
    _assistant_token_id = assistant_token_id
    if _assistant_token_id > 0:
        print(f"Unsloth: Completion-only training (assistant_token_id={_assistant_token_id}).")

    if use_quantized:
        # Backstop (as in the text CCE path): full FT against a quantized LM
        # head silently skips the LM head update. Reject loudly here in case
        # the loader-level check is bypassed.
        if getattr(model, "_unsloth_full_finetuning", False):
            raise ValueError(
                "Unsloth: full_finetuning=True with a quantized VLM LM head "
                "is not supported. The CCE backward zeros the quantized "
                "weight gradient, so the LM head would never update. Load "
                "the unquantized base model for full fine-tuning, or use "
                "LoRA on this quantized base."
            )
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        quant_mode = getattr(lm_layer, "mode", "affine")

        rt_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
            quantized=True,
            group_size=group_size,
            bits=bits,
            mode=quant_mode,
        )

        def loss_fn(model, batch_dict):
            hidden, masked_targets, ntoks = _vlm_cce_forward(
                model, batch_dict, image_token_ids=_image_token_ids,
                assistant_token_id=_assistant_token_id)
            lm_head = _get_lm_head_layer(model)
            w = lm_head.weight
            sc = lm_head.scales
            bi = getattr(lm_head, "biases", None)
            if bi is None and quant_mode == "affine":
                bi = mx.zeros_like(sc)
            # Quantized backward already returns zero weight/scales/biases
            # gradients (see runtime_cce.py VJP), so stop_gradient is
            # redundant here even when the LM head is frozen.
            hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
            targets_flat = masked_targets.reshape((-1,))  # runtime CCE validates dtype before narrowing
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
            targets_flat = masked_targets.reshape((-1,))  # runtime CCE validates dtype before narrowing
            loss = rt_cce(hidden_flat, w, targets_flat)
            loss = loss.astype(mx.float32).sum() / _safe_token_denominator(ntoks)
            return loss, ntoks

    loss_fn._unsloth_cce_backend = "runtime-cce"
    return loss_fn


def _get_vlm_image_size(config, processor):
    """Get target image size for uniform resizing, matching the GPU collator.

    Tries vision_config.image_size (dict or dataclass), then
    processor.image_processor.size, falls back to 512.

    Note: Qwen2.5-VL's image_processor.size holds *area pixel counts*
    (shortest_edge/longest_edge = min/max pixels), not dimensions. Reading
    them as height would 3136x3136-upsample -> 50k patches -> OOM, so skip
    area-style ``size`` dicts.
    """
    vc = config.get("vision_config") if isinstance(config, dict) else getattr(config, "vision_config", None)
    if vc is not None:
        sz = vc.get("image_size") if isinstance(vc, dict) else getattr(vc, "image_size", None)
        if isinstance(sz, int) and sz > 0:
            return sz
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        sz = getattr(ip, "size", None)
        if isinstance(sz, dict):
            # Skip area-pixel-count dicts (Qwen2.5-VL style) — those use
            # {shortest_edge, longest_edge} as area bounds, not dimensions.
            if "longest_edge" in sz and sz.get("longest_edge", 0) > 1_000_000:
                pass  # area-count form, fall through to default
            else:
                h = sz.get("height", sz.get("shortest_edge", 0))
                if isinstance(h, int) and 0 < h < 4096:
                    return h
        elif isinstance(sz, int) and sz > 0:
            return sz
    return 512


def _has_chat_template(obj):
    template = getattr(obj, "chat_template", None)
    return isinstance(template, str) and len(template.strip()) > 0


def _get_processor_tokenizer(processor):
    if processor is None:
        return None
    # HF fast tokenizers expose the public API directly while their _tokenizer
    # is the low-level Rust backend (no convert_tokens_to_ids / chat_template /
    # apply_chat_template). Only unwrap _tokenizer for wrappers that do not
    # already speak the HF tokenizer API (mlx-lm's TokenizerWrapper proxies it).
    if hasattr(processor, "convert_tokens_to_ids"):
        return processor
    if hasattr(processor, "_tokenizer"):
        return processor._tokenizer
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


def encode_mlx_text(tokenizer, text):
    """Tokenize text while mirroring Unsloth's double-BOS guard."""
    add_special_tokens = True
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token is not None and text.startswith(bos_token):
        add_special_tokens = False

    try:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        return tokenizer.encode(text)


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
    """Render list-style VLM content as text for text-only chat templates."""
    flattened = copy.deepcopy(messages)
    for message in flattened:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                texts.append(part)
        message["content"] = "".join(texts)
    return flattened


def _flatten_vlm_messages_to_content_parts(messages):
    """Flatten role messages for mlx-vlm processors with content-part templates."""
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


def _count_vlm_image_parts(messages):
    if isinstance(messages, str):
        return 0
    count = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                count += 1
    return count


def _repair_deepseek_rendered_image_tokens(processor, text, messages):
    if not isinstance(text, str) or not text.strip():
        return text
    marker = (
        f"{processor.__class__.__module__}.{processor.__class__.__name__}"
    ).lower()
    if "deepseek" not in marker:
        return text
    image_count = _count_vlm_image_parts(messages)
    if image_count <= 0:
        return text
    image_token = getattr(processor, "image_token", None)
    if not image_token:
        return text
    missing = image_count - text.count(image_token)
    if missing <= 0:
        return text
    return (image_token * missing) + text


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


def _render_vlm_messages(
    processor,
    messages,
    *,
    add_generation_prompt=False,
):
    normalize_vlm_processor_chat_template(processor, strict=True)
    if isinstance(messages, str):
        return messages

    render_messages = messages
    if not _processor_accepts_assistant_list_content(processor):
        render_messages = _collapse_vlm_assistant_content(render_messages)

    first_error = None
    second_error = None
    third_error = None

    try:
        text = processor.apply_chat_template(
            render_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        text = _repair_deepseek_rendered_image_tokens(processor, text, messages)
        if isinstance(text, str) and text.strip():
            return text
    except Exception as exc:
        first_error = exc

    try:
        text = processor.apply_chat_template(
            _flatten_vlm_messages_to_content_parts(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        text = _repair_deepseek_rendered_image_tokens(processor, text, messages)
        if isinstance(text, str) and text.strip():
            return text
    except Exception as exc:
        second_error = exc

    try:
        text = processor.apply_chat_template(
            _flatten_vlm_content_for_text_template(render_messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        text = _repair_deepseek_rendered_image_tokens(processor, text, messages)
        if isinstance(text, str) and text.strip():
            return text
    except Exception as exc:
        third_error = exc

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
            image = image.convert("RGB")
            if isinstance(image_size, int):
                # Match UnslothVisionDataCollator resize="min": shrink large
                # images to the model limit, but let processors handle upscaling.
                # Scale on the larger side so tall portrait images (e.g.
                # 512x2048 with a 512 cap) also shrink instead of slipping
                # through on a width-only check.
                width, height = image.size
                side = max(width, height)
                if side > image_size:
                    new_width = max(1, (width * image_size + side // 2) // side)
                    new_height = max(1, (height * image_size + side // 2) // side)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                image = image.resize(target, Image.Resampling.LANCZOS)
            resized.append(image)
        else:
            resized.append(image)
    return resized


def _extract_vlm_images(
    item,
    messages,
    image_size,
    *,
    suppress_process_errors=False,
):
    images = []
    if isinstance(item, dict):
        image = item.get("images")
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
            from ..vision_utils import process_vision_info

            extracted = process_vision_info(messages, return_video_kwargs=True)
            if isinstance(extracted, tuple) and extracted:
                maybe_images = extracted[0]
                if maybe_images is not None:
                    images = maybe_images if isinstance(maybe_images, list) else [maybe_images]
        except Exception:
            if not suppress_process_errors:
                raise

    return _resize_vlm_images(images, image_size)


def _extract_vlm_pc_images(item, prompt_messages, completion_messages, image_size):
    """Extract VLM PC images with CUDA's embedded-then-top-level preference."""
    messages = (prompt_messages or []) + (completion_messages or [])
    if messages:
        images = _extract_vlm_images(
            {},
            messages,
            image_size,
            suppress_process_errors=True,
        )
        if images:
            return images
        return []

    if isinstance(item, dict) and "images" in item:
        try:
            from ..vision_utils import process_vision_info

            raw_images = item["images"]
            vision_infos = [{"image": raw_images[i]} for i in range(len(raw_images))]
            extracted = process_vision_info(vision_infos, return_video_kwargs=True)
            images = extracted[0] if isinstance(extracted, tuple) and extracted else None
            if images is None:
                images = []
            elif not isinstance(images, list):
                images = [images]
        except Exception:
            images = []
        return _resize_vlm_images(images, image_size)

    return []


def _flatten_vlm_images(all_images):
    flattened = []
    for images in all_images:
        if isinstance(images, (list, tuple)):
            flattened.extend(images)
        else:
            flattened.append(images)
    return flattened


def _nest_vlm_images_by_sample(all_images):
    nested = []
    for images in all_images:
        if images is None:
            nested.append([])
        elif isinstance(images, (list, tuple)):
            nested.append(list(images))
        else:
            nested.append([images])
    return nested


def _vlm_processor_prefers_nested_images(processor):
    cls = processor.__class__
    marker = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', '')}".lower()
    # These processors need images grouped per-prompt; others take a flat list.
    return any(
        name in marker
        for name in (
            "deepseek_vl",
            "falcon",
            "gemma3",
            "gemma3n",
            "idefics",
            "lfm2_vl",
            "minicpmo",
            "mistral",
            "mllama",
            "paligemma",
            "pixtral",
            "smolvlm",
        )
    )


def _format_vlm_images_for_processor(all_images, processor=None, image_layout=None):
    if not any(all_images):
        return None
    if image_layout == "nested":
        return _nest_vlm_images_by_sample(all_images)
    if image_layout == "flat":
        return _flatten_vlm_images(all_images)
    if processor is not None and _vlm_processor_prefers_nested_images(processor):
        return _nest_vlm_images_by_sample(all_images)
    return _flatten_vlm_images(all_images)


# Private key used to pass raw (pre-int32-narrowing) input_ids through
# the VLM batch dict to labels-free / response-mask paths. Stripped from
# model forward kwargs so the backbone never sees it.
_RAW_INPUT_IDS_FOR_LABELS = "_unsloth_raw_input_ids_for_labels"


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
        # Preserve raw input_ids under a private key BEFORE the int32 narrow
        # so labels-free / response-mask paths can derive labels from the
        # original processor output (wide invalid ids like uint32(2**32-100)
        # would otherwise wrap to -100 and be treated as ignore_index).
        if "labels" not in batch:
            batch[_RAW_INPUT_IDS_FOR_LABELS] = _normalize_cce_label_dtype(
                batch["input_ids"]
            )
        batch["input_ids"] = batch["input_ids"].astype(mx.int32)
    if "attention_mask" in batch:
        batch["attention_mask"] = batch["attention_mask"].astype(mx.int32)
    # Do NOT narrow labels to int32: runtime CCE validates the original
    # dtype before its own narrow. Unsigned dtypes still need widening to
    # int64 so the masking helpers can mx.where the signed -100 sentinel.
    if "labels" in batch:
        batch["labels"] = _normalize_cce_label_dtype(batch["labels"])
    if "token_type_ids" in batch:
        batch["token_type_ids"] = batch["token_type_ids"].astype(mx.int32)
    if "mm_token_type_ids" in batch:
        batch["mm_token_type_ids"] = batch["mm_token_type_ids"].astype(mx.int32)

    return batch


def _processor_vlm_inputs(
    processor,
    texts,
    all_images,
    max_seq_length,
    suffixes=None,
    truncation=True,
    padding_side=None,
):
    base_kwargs = dict(
        text=texts,
        padding=True,
        return_tensors="np",
        add_special_tokens=False,
    )
    if truncation:
        base_kwargs["truncation"] = True
        if max_seq_length is not None:
            base_kwargs["max_length"] = max_seq_length
    if padding_side is not None:
        base_kwargs["padding_side"] = padding_side
    images = _format_vlm_images_for_processor(all_images, processor=processor)
    if images is not None:
        image_layouts = (
            ("nested", "flat")
            if _vlm_processor_prefers_nested_images(processor)
            else ("flat", "nested")
        )
    else:
        image_layouts = (None,)
    if suffixes is not None and any(suffix is not None for suffix in suffixes):
        base_kwargs["suffix"] = [suffix or "" for suffix in suffixes]
    marker = f"{processor.__class__.__module__}.{processor.__class__.__name__}".lower()
    if (
        "gemma3" in marker
        or "gemma4" in marker
        or "qwen3_vl" in marker
        or "qwen3_5" in marker
    ):
        base_kwargs["return_mm_token_type_ids"] = True

    first_error = None
    for image_layout in image_layouts:
        proc_kwargs = dict(base_kwargs)
        if image_layout is not None:
            proc_kwargs["images"] = _format_vlm_images_for_processor(
                all_images,
                processor=processor,
                image_layout=image_layout,
            )
        try:
            return processor(**proc_kwargs)
        except TypeError as exc:
            if (
                "add_special_tokens" in str(exc)
                and "multiple values" in str(exc)
                and "add_special_tokens" in proc_kwargs
            ):
                proc_kwargs.pop("add_special_tokens", None)
                try:
                    return processor(**proc_kwargs)
                except Exception as retry_exc:
                    if first_error is None:
                        first_error = retry_exc
                    if len(image_layouts) == 1:
                        raise
                    continue
            if "padding_side" in str(exc) and "padding_side" in proc_kwargs:
                proc_kwargs.pop("padding_side", None)
                try:
                    return processor(**proc_kwargs)
                except Exception as retry_exc:
                    if first_error is None:
                        first_error = retry_exc
                    if len(image_layouts) == 1:
                        raise
                    continue
            if first_error is None:
                first_error = exc
            if len(image_layouts) == 1:
                raise
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if len(image_layouts) == 1:
                raise
    raise first_error


def _as_numpy_vlm_field(inputs, key):
    """Return a processor output field as a 2-D numpy array."""
    value = inputs[key]
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _common_text_prefix(left, right):
    """Return CUDA's shared rendered prompt prefix for VLM PC rows."""
    end = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        end += 1
    return left[:end]


def _vlm_tokenizer_padding_side(processor):
    """Resolve the tokenizer padding side used by CUDA's VLM collator."""
    tokenizer = _get_processor_tokenizer(processor)
    side = getattr(tokenizer, "padding_side", "right")
    return "left" if side == "left" else "right"


def _vlm_pad_token_id(processor):
    """Return the processor tokenizer pad id required for PC collation."""
    tokenizer = _get_processor_tokenizer(processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise ValueError(
            "Tokenizer must define `pad_token_id` for prompt-completion collation."
        )
    return int(pad_id)


def _concat_vlm_token_type_ids(prompt_inputs, completion_inputs, p_ids, c_ids):
    """Concatenate token-type metadata the same way CUDA's collator does."""
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key not in prompt_inputs and key not in completion_inputs:
            continue
        p_tt = (
            _as_numpy_vlm_field(prompt_inputs, key)
            if key in prompt_inputs else np.zeros_like(p_ids)
        )
        c_tt = (
            _as_numpy_vlm_field(completion_inputs, key)
            if key in completion_inputs else np.zeros_like(c_ids)
        )
        return key, np.concatenate((p_tt, c_tt), axis=1)
    return None, None


def _flush_vlm_arrays_to_side(input_ids, attention_mask, side, pad_id, extras):
    """Move non-pad VLM PC tokens to the tokenizer padding side."""
    keep = attention_mask.astype(bool)
    if keep.all():
        return input_ids, attention_mask, extras

    batch, seq_len = input_ids.shape
    counts = keep.sum(axis=1)
    ranks = np.cumsum(keep, axis=1) - 1
    if side == "left":
        dst = (seq_len - counts)[:, None] + ranks
    else:
        dst = ranks

    row_idx, col_src = np.nonzero(keep)
    col_dst = dst[row_idx, col_src].astype(np.int64)

    new_ids = np.full(input_ids.shape, pad_id, dtype=input_ids.dtype)
    new_mask = np.zeros(attention_mask.shape, dtype=attention_mask.dtype)
    new_ids[row_idx, col_dst] = input_ids[row_idx, col_src]
    new_mask[row_idx, col_dst] = 1

    new_extras = {}
    for key, values in extras.items():
        out = np.zeros(values.shape, dtype=values.dtype)
        out[row_idx, col_dst] = values[row_idx, col_src]
        new_extras[key] = out

    max_count = int(counts.max()) if counts.size else 0
    if 0 < max_count < seq_len:
        sl = slice(seq_len - max_count, seq_len) if side == "left" else slice(0, max_count)
        new_ids = new_ids[:, sl]
        new_mask = new_mask[:, sl]
        new_extras = {key: value[:, sl] for key, value in new_extras.items()}

    return new_ids, new_mask, new_extras


def _truncate_vlm_arrays_by_side(input_ids, attention_mask, side, max_seq_length, extras):
    """Truncate VLM PC arrays from the same side as CUDA."""
    if max_seq_length is None or max_seq_length <= 0 or input_ids.shape[1] <= max_seq_length:
        return input_ids, attention_mask, extras
    sl = slice(-max_seq_length, None) if side == "left" else slice(0, max_seq_length)
    return (
        input_ids[:, sl],
        attention_mask[:, sl],
        {key: value[:, sl] for key, value in extras.items()},
    )


def _collate_vlm_prompt_completion_batch(
    items,
    processor,
    max_seq_length,
    image_size,
    ignore_token_ids=None,
    completion_only_loss=None,
):
    prompt_texts = []
    completion_texts = []
    all_images = []

    for item in items:
        prompt_raw = item.get("prompt", "")
        completion_raw = item.get("completion", "")
        prompt = _normalize_vlm_messages(prompt_raw)
        completion = _normalize_vlm_messages(completion_raw)
        # Coerce a raw prompt to a chat message when the completion is chat-style so
        # both render through one template and split cleanly (avoids None + list).
        if isinstance(completion, list) and isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        prompt_messages = prompt if isinstance(prompt, list) else None
        completion_messages = completion if isinstance(completion, list) else None

        if prompt_messages is not None:
            prompt_text = _render_vlm_messages(
                processor,
                prompt_messages,
                add_generation_prompt=True,
            )
        else:
            prompt_text = str(prompt)

        if completion_messages is not None:
            combined = prompt_messages + completion_messages
            prompt_completion_text = _render_vlm_messages(processor, combined)
            images = _extract_vlm_pc_images(
                item, prompt_messages, completion_messages, image_size,
            )
            prompt_text = _common_text_prefix(
                prompt_text, prompt_completion_text,
            )
            completion_text = prompt_completion_text[len(prompt_text):]
        else:
            images = _extract_vlm_pc_images(
                item, prompt_messages, completion_messages, image_size,
            )
            completion_text = str(completion)

        prompt_texts.append(prompt_text)
        completion_texts.append(completion_text)
        all_images.append(images)

    prompt_inputs = _processor_vlm_inputs(
        processor,
        prompt_texts,
        all_images,
        max_seq_length,
        truncation=False,
        padding_side="left",
    )
    completion_inputs = _processor_vlm_inputs(
        processor,
        completion_texts,
        [[] for _ in completion_texts],
        max_seq_length,
        truncation=False,
        padding_side="right",
    )

    p_ids = _as_numpy_vlm_field(prompt_inputs, "input_ids")
    c_ids = _as_numpy_vlm_field(completion_inputs, "input_ids")
    p_mask = _as_numpy_vlm_field(prompt_inputs, "attention_mask")
    c_mask = _as_numpy_vlm_field(completion_inputs, "attention_mask")
    input_ids = np.concatenate((p_ids, c_ids), axis=1)
    attention_mask = np.concatenate((p_mask, c_mask), axis=1)
    completion_mask = np.concatenate((np.zeros_like(p_mask), c_mask), axis=1)
    token_type_key, token_type_ids = _concat_vlm_token_type_ids(
        prompt_inputs, completion_inputs, p_ids, c_ids,
    )

    extras = {"completion_mask": completion_mask}
    if token_type_key is not None:
        extras[token_type_key] = token_type_ids

    flush_side = _vlm_tokenizer_padding_side(processor)
    pad_id = _vlm_pad_token_id(processor)
    input_ids, attention_mask, extras = _flush_vlm_arrays_to_side(
        input_ids, attention_mask, flush_side, pad_id, extras,
    )
    input_ids, attention_mask, extras = _truncate_vlm_arrays_by_side(
        input_ids, attention_mask, flush_side, max_seq_length, extras,
    )

    combined_inputs = dict(prompt_inputs)
    combined_inputs["input_ids"] = input_ids
    combined_inputs["attention_mask"] = attention_mask
    if token_type_key is not None:
        combined_inputs[token_type_key] = extras[token_type_key]
    batch = _to_mx_vlm_batch(combined_inputs)
    batch["labels"] = _apply_vlm_label_masks(
        batch,
        ignore_token_ids=ignore_token_ids,
    )

    completion_only_loss_enabled = True if completion_only_loss is None else bool(completion_only_loss)
    if completion_only_loss_enabled:
        labels_np = np.asarray(batch["labels"].tolist(), dtype=np.int64)
        labels_np[np.asarray(extras["completion_mask"]) == 0] = -100
        batch["labels"] = mx.array(labels_np)
    return batch


def _collate_vlm_batch(items, processor, max_seq_length, image_size,
                       formatting_func=None, ignore_token_ids=None,
                       completion_only_loss=None,
                       return_prompt_completion=False):
    """Collate a batch of VLM samples using the processor directly.

    Mirrors Unsloth's GPU UnslothVisionDataCollator: extract images, resize
    to uniform size, apply_chat_template, then one processor() call for
    tokenization + image processing + padding.
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
        batch = _collate_vlm_prompt_completion_batch(
            formatted_items, processor, max_seq_length, image_size,
            ignore_token_ids=ignore_token_ids,
            completion_only_loss=completion_only_loss,
        )
        return (batch, True) if return_prompt_completion else batch

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
    batch = _to_mx_vlm_batch(inputs)
    batch["labels"] = _apply_vlm_label_masks(
        batch,
        ignore_token_ids=ignore_token_ids,
    )
    return (batch, False) if return_prompt_completion else batch


def _apply_response_mask_to_vlm_batch(batch_dict, mask_fn, ignore_token_ids=None):
    """Apply response masking to a VLM batch dict, adding 'labels' key.

    Converts input_ids to plain lists, runs the masking closure from
    dataset_utils.train_on_responses_only with the current VLM labels, and
    stores the result as an mx.array in batch_dict["labels"].
    """
    # Prefer the raw (pre-int32-narrowing) input_ids so the masking closure
    # sees original processor ids; wide invalid ids like uint32(2**32-100)
    # would otherwise have already wrapped to -100 in batch_dict["input_ids"].
    raw_input_ids = batch_dict.pop(_RAW_INPUT_IDS_FOR_LABELS, None)
    input_ids = raw_input_ids if raw_input_ids is not None else batch_dict["input_ids"]
    ids_list = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    mask_batch = {"input_ids": ids_list}
    existing_labels = batch_dict.get("labels")
    if existing_labels is not None:
        labels_list = (
            existing_labels.tolist()
            if hasattr(existing_labels, "tolist") else existing_labels
        )
        mask_batch["labels"] = _normalize_numpy_cce_labels(labels_list)
    result = mask_fn(mask_batch)
    labels_list = result["labels"]
    if hasattr(labels_list, "tolist"):
        labels_list = labels_list.tolist()
    # Widen unsigned label dtypes (numpy + mx normalizers) so mx.where with
    # -100 does not crash on uint* and wide invalid ids reach runtime CCE
    # as sentinels instead of wrapping. Then apply image-token + attention
    # masking via the shared helper.
    labels_np = _normalize_numpy_cce_labels(labels_list)
    labels_array = _normalize_cce_label_dtype(mx.array(labels_np))
    batch_dict["labels"] = _apply_vlm_label_masks(
        batch_dict,
        labels=labels_array,
        ignore_token_ids=ignore_token_ids,
    )
    return batch_dict


def _vlm_trainable_label_rows(batch_dict):
    """Return per-row trainability from VLM labels after response masking."""
    labels = batch_dict.get("labels")
    if labels is None:
        return None
    labels_np = np.asarray(labels.tolist() if hasattr(labels, "tolist") else labels)
    if labels_np.ndim == 1:
        labels_np = labels_np.reshape(1, -1)
    # Loss supervises labels[:, 1:] (causal shift), so the first column never trains.
    return [bool(np.any(row[1:] != -100)) for row in labels_np]


def _build_response_masked_vlm_batch(
    items,
    processor,
    config,
    max_seq_length,
    image_size,
    response_mask_fn=None,
    formatting_func=None,
    ignore_token_ids=None,
    completion_only_loss=None,
    return_prompt_completion=False,
):
    """Collate VLM rows and apply the CUDA response-mask closure."""
    batch_dict, is_prompt_completion = _collate_vlm_batch(
        items, processor, max_seq_length, image_size,
        formatting_func=formatting_func,
        ignore_token_ids=ignore_token_ids,
        completion_only_loss=completion_only_loss,
        return_prompt_completion=True,
    )
    batch_dict = _prepare_vlm_batch_for_compile(batch_dict, config)
    if response_mask_fn is not None and not is_prompt_completion:
        batch_dict = _apply_response_mask_to_vlm_batch(
            batch_dict,
            response_mask_fn,
            ignore_token_ids=ignore_token_ids,
        )
    if return_prompt_completion:
        return batch_dict, is_prompt_completion
    return batch_dict


def _filter_trainable_vlm_indices(
    dataset,
    indices,
    processor,
    config,
    max_seq_length,
    image_size,
    response_mask_fn,
    formatting_func=None,
    ignore_token_ids=None,
    completion_only_loss=None,
):
    """Filter VLM rows before batching, matching CUDA dataset.filter order."""
    kept_indices = []
    formatted_items = {} if formatting_func is not None else None
    removed = 0
    for idx in indices:
        item = dataset[idx]
        if formatting_func is not None:
            item = formatting_func(item)
        batch_dict, is_prompt_completion = _build_response_masked_vlm_batch(
            [item],
            processor,
            config,
            max_seq_length,
            image_size,
            response_mask_fn=response_mask_fn,
            formatting_func=None,
            ignore_token_ids=ignore_token_ids,
            completion_only_loss=completion_only_loss,
            return_prompt_completion=True,
        )
        if is_prompt_completion:
            kept_indices.append(idx)
            if formatted_items is not None:
                formatted_items[idx] = item
            continue
        valid_rows = _vlm_trainable_label_rows(batch_dict)
        if valid_rows is not None and len(valid_rows) == 1 and not valid_rows[0]:
            removed += 1
            continue
        kept_indices.append(idx)
        if formatted_items is not None:
            formatted_items[idx] = item
    return kept_indices, removed, formatted_items


def create_vlm_batches(dataset, processor, config, batch_size, max_seq_length,
                       num_batches=None, seed=42, response_mask_fn=None,
                       formatting_func=None, dataset_order="default",
                       num_epochs=None, completion_only_loss=None):
    """Pre-materialize VLM training batches using the processor directly.

    Mirrors Unsloth's GPU UnslothVisionDataCollator:
    resize images → processor(text, images, padding=True) → uniform batches.
    """
    import numpy as np

    image_size = _get_vlm_image_size(config, processor)
    ignore_token_ids = _get_vlm_ignore_token_ids(processor=processor, config=config)

    batch_list = []
    seen = 0
    epoch = 0
    base_seed = _normalize_seed(seed)
    target_epochs = 1 if num_batches is None and num_epochs is None else num_epochs
    base_indices = list(range(len(dataset)))
    total_removed = 0
    formatted_items = None
    if response_mask_fn is not None:
        base_indices, total_removed, formatted_items = _filter_trainable_vlm_indices(
            dataset,
            base_indices,
            processor,
            config,
            max_seq_length,
            image_size,
            response_mask_fn,
            formatting_func=formatting_func,
            ignore_token_ids=ignore_token_ids,
            completion_only_loss=completion_only_loss,
        )
        if not base_indices and total_removed > 0:
            raise ValueError(
                "Unsloth MLX VLM: no trainable rows remain after "
                "train_on_responses_only masking. Check instruction_part / "
                "response_part and max_seq_length."
            )
    batch_formatting_func = None if formatted_items is not None else formatting_func

    def _item(idx):
        return formatted_items[idx] if formatted_items is not None else dataset[idx]

    if dataset_order not in (None, "default", "sequential", "torch_randperm"):
        raise ValueError(f"Unsupported MLX VLM dataset_order: {dataset_order!r}")
    if not base_indices:
        return []

    def _epoch_indices(epoch_idx):
        """Return CUDA-style sampler order over the filtered VLM dataset."""
        if dataset_order == "torch_randperm":
            order = _torch_randperm_order(len(base_indices), base_seed + epoch_idx)
            return [base_indices[i] for i in order]
        indices = list(base_indices)
        if dataset_order in (None, "default") and (
            num_batches is not None or epoch_idx > 0
        ):
            np.random.seed(base_seed + epoch_idx)
            np.random.shuffle(indices)
        return indices

    indices = _epoch_indices(epoch)

    while num_batches is None or len(batch_list) < num_batches:
        if seen >= len(indices):
            if num_batches is None and target_epochs is not None and epoch + 1 >= target_epochs:
                break
            epoch += 1
            seen = 0
            indices = _epoch_indices(epoch)

        bi = indices[seen : seen + batch_size]
        seen += len(bi)
        if not bi:
            break
        batch_dict = _build_response_masked_vlm_batch(
            [_item(idx) for idx in bi],
            processor,
            config,
            max_seq_length,
            image_size,
            response_mask_fn=response_mask_fn,
            formatting_func=batch_formatting_func,
            ignore_token_ids=ignore_token_ids,
            completion_only_loss=completion_only_loss,
        )
        batch_list.append(batch_dict)

    if total_removed > 0:
        print(
            f"Unsloth: Removed {total_removed} VLM samples where all labels "
            f"were -100 after train_on_responses_only masking."
        )

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
                                  formatting_func=None,
                                  dataset_order="default",
                                  completion_only_loss=None):
    """Streaming VLM batch generator using processor directly.

    Yields batch dicts with input_ids, pixel_values, attention_mask,
    and optionally labels.
    """
    import numpy as np

    image_size = _get_vlm_image_size(config, processor)
    ignore_token_ids = _get_vlm_ignore_token_ids(processor=processor, config=config)
    base_seed = _normalize_seed(seed)

    def _build_batch(items, batch_formatting_func):
        """Build one VLM batch with the selected formatting function."""
        return _build_response_masked_vlm_batch(
            items,
            processor,
            config,
            max_seq_length,
            image_size,
            response_mask_fn=response_mask_fn,
            formatting_func=batch_formatting_func,
            ignore_token_ids=ignore_token_ids,
            completion_only_loss=completion_only_loss,
        )

    if hasattr(dataset, "__len__"):
        if len(dataset) <= 0:
            raise ValueError("Unsloth MLX VLM: streaming dataset produced no rows.")
        base_indices = list(range(len(dataset)))
        total_removed = 0
        formatted_items = None
        if response_mask_fn is not None:
            base_indices, total_removed, formatted_items = _filter_trainable_vlm_indices(
                dataset,
                base_indices,
                processor,
                config,
                max_seq_length,
                image_size,
                response_mask_fn,
                formatting_func=formatting_func,
                ignore_token_ids=ignore_token_ids,
                completion_only_loss=completion_only_loss,
            )
            if not base_indices and total_removed > 0:
                raise ValueError(
                    "Unsloth MLX VLM: no trainable rows remain after "
                    "train_on_responses_only masking. Check instruction_part / "
                    "response_part and max_seq_length."
                )
            if total_removed > 0:
                print(
                    f"Unsloth: Removed {total_removed} VLM samples where all "
                    f"labels were -100 after train_on_responses_only masking."
                )
        if not base_indices:
            raise ValueError("Unsloth MLX VLM: streaming dataset produced no rows.")

        def _item(idx):
            return formatted_items[idx] if formatted_items is not None else dataset[idx]

        batch_formatting_func = None if formatted_items is not None else formatting_func
        epoch = 0
        while True:
            if dataset_order == "torch_randperm":
                order = _torch_randperm_order(len(base_indices), base_seed + epoch)
                indices = [base_indices[i] for i in order]
            elif dataset_order == "sequential":
                indices = list(base_indices)
            elif dataset_order in (None, "default"):
                indices = list(base_indices)
                batch_indices = [
                    indices[i : i + batch_size]
                    for i in range(0, len(indices), batch_size)
                ]
                # Local RNG keeps order reproducible under `seed`; reseed per epoch.
                rng = np.random.default_rng(base_seed + epoch)
                order = rng.permutation(len(batch_indices))
                for b in order:
                    yield _build_batch(
                        [_item(idx) for idx in batch_indices[b]],
                        batch_formatting_func,
                    )
                epoch += 1
                continue
            else:
                raise ValueError(f"Unsupported MLX VLM dataset_order: {dataset_order!r}")
            for start in range(0, len(indices), batch_size):
                yield _build_batch(
                    [_item(idx) for idx in indices[start : start + batch_size]],
                    batch_formatting_func,
                )
            epoch += 1
    else:
        # Streaming has no index space; refuse rather than silently misorder.
        if dataset_order not in (None, "default"):
            raise ValueError(
                "Unsloth MLX VLM: preserve_dataset_order / "
                f"dataset_order={dataset_order!r} requires a sized "
                "(`__len__`) dataset. Materialize the dataset (e.g. "
                "via `dataset.to_iterable_dataset()` -> list) or drop "
                "the order request."
            )
        def _filter_stream_item(item):
            """Return a formatted trainable streaming row, or None to skip it."""
            if response_mask_fn is None:
                return item
            if formatting_func is not None:
                item = formatting_func(item)
            batch_dict, is_prompt_completion = _build_response_masked_vlm_batch(
                [item],
                processor,
                config,
                max_seq_length,
                image_size,
                response_mask_fn=response_mask_fn,
                formatting_func=None,
                ignore_token_ids=ignore_token_ids,
                completion_only_loss=completion_only_loss,
                return_prompt_completion=True,
            )
            if is_prompt_completion:
                return item
            valid_rows = _vlm_trainable_label_rows(batch_dict)
            if valid_rows is not None and len(valid_rows) == 1 and not valid_rows[0]:
                return None
            return item

        batch_formatting_func = None if response_mask_fn is not None else formatting_func
        while True:
            pending = []
            yielded = False
            for item in dataset:
                item = _filter_stream_item(item)
                if item is None:
                    continue
                pending.append(item)
                if len(pending) >= batch_size:
                    yielded = True
                    yield _build_batch(pending, batch_formatting_func)
                    pending = []
            if pending:
                yielded = True
                yield _build_batch(pending, batch_formatting_func)
            if not yielded:
                raise ValueError("Unsloth MLX VLM: streaming dataset produced no rows.")


def _prepare_dataset(dataset, tokenizer, dataset_text_field="text",
                     formatting_func=None, chat_template=None,
                     model_name=None, model_type=None,
                     append_eos=True):
    """Wrap a HuggingFace dataset into mlx-lm's dataset classes.

    Uses CacheDataset from mlx_lm while leaving rendered text token-exact.

    If a formatting_func is provided, each item is pre-formatted into a
    ``{"text": ...}`` dict before wrapping.

    ``append_eos`` controls whether the tokenizer's EOS id is appended to
    each encoded row. Default True preserves the pre-PR behavior that
    delegated EOS appending to ``mlx_lm.tuner.datasets.TextDataset`` for
    direct MLX text fine-tuning callers (raw ``{"text": str}`` rows
    without already-rendered EOS). Studio passes False because its
    chat-template rendering already includes EOS.

    Returns:
        A CacheDataset ready for ``iterate_batches``.
    """
    from mlx_lm.tuner.datasets import CacheDataset

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

    _eos_id = getattr(tokenizer, "eos_token_id", None) if append_eos else None

    class _StudioTextDataset:
        """TextDataset variant. Optionally appends EOS (mlx-lm parity);
        Studio passes append_eos=False because chat templates already render it."""

        def __init__(self, data, tokenizer, text_key="text", eos_id=None):
            self._data = data
            self.tokenizer = tokenizer
            self.text_key = text_key
            self._eos_id = eos_id

        def process(self, item):
            encoded = encode_mlx_text(self.tokenizer, item[self.text_key])
            if (
                self._eos_id is not None
                and (not encoded or encoded[-1] != self._eos_id)
            ):
                encoded = list(encoded) + [int(self._eos_id)]
            return (encoded, 0)

        def __getitem__(self, idx):
            return self._data[idx]

        def __len__(self):
            return len(self._data)

    return CacheDataset(
        _StudioTextDataset(formatted, tokenizer, text_key="text", eos_id=_eos_id)
    )


def create_batches(dataset, tokenizer, batch_size, max_seq_length,
                   num_batches=None, seed=42, dataset_text_field="text",
                   formatting_func=None, chat_template=None,
                   model_name=None, model_type=None, append_eos=True):
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
        append_eos=append_eos,
    )

    batch_pairs = []
    for batch, lengths_info in iterate_batches(
        ds, batch_size, max_seq_length,
        loop=(num_batches is not None),
        seed=seed,
    ):
        max_length = int(mx.max(lengths_info[:, 1]).item())
        batch = batch[:, :max_length]
        batch_pairs.append((batch, lengths_info, None))
        if num_batches is not None and len(batch_pairs) >= num_batches:
            break

    mx.eval(
        [b for b, lengths, _ in batch_pairs]
        + [lengths for _, lengths, _ in batch_pairs]
    )
    return batch_pairs


def _torch_randperm_order(length, seed):
    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "Unsloth MLX: dataset_order='torch_randperm' requires torch so MLX "
            "Studio can mirror CUDA Studio batch order."
        ) from exc
    generator = torch.Generator()
    generator.manual_seed(3407 if seed is None else int(seed))
    return torch.randperm(length, generator=generator).tolist()


def create_ordered_batches(dataset, tokenizer, batch_size, max_seq_length,
                           num_batches=None, seed=None, dataset_order="sequential",
                           dataset_text_field="text",
                           formatting_func=None, chat_template=None,
                           model_name=None, model_type=None,
                           num_epochs=None, append_eos=True):
    """Create text batches with an explicit dataset order.

    Studio uses this to mirror CUDA's effective sampler stream without
    changing generic mlx-lm batching behavior.
    """

    ds = _prepare_dataset(
        dataset, tokenizer, dataset_text_field, formatting_func,
        chat_template=chat_template,
        model_name=model_name,
        model_type=model_type,
        append_eos=append_eos,
    )

    tokenized = []
    for row in ds:
        ids = row[0] if isinstance(row, (tuple, list)) else row
        ids = list(ids)[:max_seq_length]
        if len(ids) >= 2:
            tokenized.append(ids)

    if not tokenized:
        raise ValueError(
            "Unsloth MLX: ordered dataset produced no trainable token sequences "
            "(need at least two tokens after formatting/truncation)."
        )

    def make_order(epoch):
        base_seed = _normalize_seed(seed)
        if dataset_order == "torch_randperm":
            return _torch_randperm_order(len(tokenized), base_seed + epoch)
        if dataset_order not in (None, "sequential"):
            raise ValueError(f"Unsupported MLX dataset_order: {dataset_order!r}")
        return list(range(len(tokenized)))

    batch_pairs = []
    epoch = 0
    order = make_order(epoch)
    order_pos = 0
    seen = 0
    target_items = (
        len(tokenized) * (1 if num_epochs is None else int(num_epochs))
        if num_batches is None else None
    )
    while num_batches is None or len(batch_pairs) < num_batches:
        # Don't mix epochs in one batch; emit a partial then restart at epoch+1.
        # Matches CUDA SequentialSampler `drop_last=False` and VLM path at :2539.
        if order_pos >= len(order):
            # Stop when num_batches or num_epochs*len(dataset) reached.
            if (
                num_batches is None
                and (target_items is None or seen >= target_items)
            ):
                break
            epoch += 1
            order = make_order(epoch)
            order_pos = 0

        chunk = order[order_pos : order_pos + batch_size]
        if not chunk:
            break
        order_pos += len(chunk)
        seen += len(chunk)
        if num_batches is None and target_items is not None and seen > target_items:
            chunk = chunk[: len(chunk) - (seen - target_items)]
            seen = target_items
        batch_items = [tokenized[i] for i in chunk]

        max_length = max(len(ids) for ids in batch_items)
        # mlx-lm iterate_batches pad convention; raw 0 only if no pad_token_id.
        _pad_id = getattr(tokenizer, "pad_token_id", None)
        if _pad_id is None:
            _pad_id = 0
        _pad_id = int(_pad_id)
        batch_ids = []
        lengths = []
        for ids in batch_items:
            length = len(ids)
            batch_ids.append(ids + [_pad_id] * (max_length - length))
            lengths.append([0, length])
        batch_pairs.append((mx.array(batch_ids), mx.array(lengths), None))

        if num_batches is None and target_items is not None and seen >= target_items:
            break

    mx.eval(
        [b for b, lengths, _ in batch_pairs]
        + [lengths for _, lengths, _ in batch_pairs]
    )
    return batch_pairs


def iterate_training_batches(dataset, tokenizer, batch_size, max_seq_length,
                             seed=42, dataset_text_field="text",
                             formatting_func=None, chat_template=None,
                             model_name=None, model_type=None,
                             append_eos=True):
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
        append_eos=append_eos,
    )

    for batch, lengths_info in iterate_batches(
        ds, batch_size, max_seq_length,
        loop=True,
        seed=seed,
    ):
        max_length = int(mx.max(lengths_info[:, 1]).item())
        batch = batch[:, :max_length]
        yield batch, lengths_info, None


def _save_adapter_artifacts(model, path, tensors, adapter_config=None):
    # Refuse to write adapter_config.json without adapters.safetensors next
    # to it; mlx-lm reload chokes on the missing weights file.
    if not tensors:
        raise ValueError(
            "Unsloth: _save_adapter_artifacts() requires non-empty "
            "tensors; use save_lora_adapters() or "
            "save_trainable_adapters() at the public entry point."
        )
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    mx.save_safetensors(str(path / "adapters.safetensors"), tensors)

    adapter_config = _enrich_mlx_adapter_config(model, adapter_config or {})
    if adapter_config:
        with open(path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)


def _extract_mlx_lora_parameters(model):
    """Extract global rank, scale, and dropout from the model's first LoRA module."""
    rank, scale, dropout = 8, 1.0, 0.0
    for _, m in iter_mlx_lora_modules(model):
        a_tensor = m.lora_a
        a_shape = tuple(getattr(a_tensor, "shape", ()))
        # Switch LoRA layouts vary across mlx-lm versions; lora_b's last axis is
        # always `rank`, so prefer it when num_experts is declared. Plain
        # LoRALinear uses (in_dims, rank) where rank is shape[-1].
        b_tensor = getattr(m, "lora_b", None)
        b_shape = tuple(getattr(b_tensor, "shape", ())) if b_tensor is not None else ()
        if hasattr(m, "num_experts") and len(b_shape) >= 3:
            rank = int(b_shape[-1])
        elif len(a_shape) >= 3:
            rank = int(a_shape[-2])
        elif len(a_shape) >= 2:
            rank = int(a_shape[-1])
        scale = getattr(m, "scale", 1.0)

        drop = getattr(m, "dropout", None)
        # mlx.nn.Dropout stores keep-prob in _p_1; fall back to .p for shims
        if drop is None:
            dropout = 0.0
        else:
            keep = getattr(drop, "_p_1", None)
            if keep is not None:
                try:
                    dropout = float(1.0 - float(keep))
                except (TypeError, ValueError):
                    dropout = 0.0
            else:
                p = getattr(drop, "p", None)
                try:
                    dropout = float(p) if p is not None else 0.0
                except (TypeError, ValueError):
                    dropout = 0.0
        break
    return rank, scale, dropout


def iter_mlx_lora_modules(model):
    """Yield (module_name, module) for each module that owns a complete LoRA pair.

    mlx-lm reload only recreates lowercase lora_a / lora_b wrappers, so we
    skip uppercase or half-built modules that cannot be reloaded.
    """
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            yield module_name, module


def collect_mlx_lora_adapter_tensors(model):
    """Collect tensors for every module exposing a complete LoRA attr pair.

    Anchors on the modules themselves so substring-`lora_` paths (e.g.
    ``router.lora_gate.weight``) are not exported, and so callers can
    still detect LoRA after reload/freeze when trainable_parameters()
    no longer lists adapter tensors.
    """
    parameters = dict(mlx.utils.tree_flatten(model.parameters()))
    adapter_keys = set()
    for module_name, module in iter_mlx_lora_modules(model):
        prefix = f"{module_name}." if module_name else ""
        adapter_keys.add(f"{prefix}lora_a")
        adapter_keys.add(f"{prefix}lora_b")
        # Include DoRA magnitude `m`, gated on the DoRA class name so a
        # future LoRA wrapper with an unrelated `m` attribute isn't exported.
        if hasattr(module, "m") and type(module).__name__.startswith("DoRA"):
            adapter_keys.add(f"{prefix}m")
    return {name: value for name, value in parameters.items() if name in adapter_keys}


# Wrapped base / quantization tensor suffixes (LoRALinear, DoRALinear,
# LoRAEmbedding) that are reload-leaked state we must drop from adapter
# saves. `.linear.bias` and `.bias` are intentionally NOT here: those are
# legitimate trainable user state.
_LORA_WRAPPED_BASE_SUFFIXES = (
    ".weight",
    ".scales",
    ".biases",
    ".linear.weight",
    ".linear.scales",
    ".linear.biases",
    ".embedding.weight",
    ".embedding.scales",
    ".embedding.biases",
)

# Same wrapped-base set for the rare root-level LoRA wrapper case where
# the empty module prefix means we cannot match by suffix alone.
_ROOT_LORA_WRAPPED_BASE_KEYS = frozenset({
    "weight", "scales", "biases",
    "linear.weight", "linear.scales", "linear.biases",
    "embedding.weight", "embedding.scales", "embedding.biases",
})


def _is_base_tensor_inside_lora_module(
    key, lora_module_prefixes, has_root_lora_module=False,
):
    """True when `key` looks like the wrapped base tensor of a LoRA module.

    Prefix-match against LoRA module names + suffix whitelist of base /
    quantization tensor names. `has_root_lora_module` separately covers
    a root-level wrapper where the empty-name prefix is intentionally
    omitted from `lora_module_prefixes` to avoid matching every key.
    """
    matches_prefix = lora_module_prefixes and any(
        key.startswith(p) for p in lora_module_prefixes
    )
    if matches_prefix:
        return key.endswith(_LORA_WRAPPED_BASE_SUFFIXES)
    if has_root_lora_module:
        return key in _ROOT_LORA_WRAPPED_BASE_KEYS
    return False


def save_trainable_adapters(model, path, adapter_config=None):
    """Save the current trainable parameter tree for training checkpoints.

    Includes all LoRA adapter tensors (frozen or not). Excludes wrapped
    base weights INSIDE a LoRA module (reload-leaked state that would
    reintroduce the original Studio adapter-export bloat).
    """
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    adapter_tensors = collect_mlx_lora_adapter_tensors(model)
    _lora_module_names = [name for name, _ in iter_mlx_lora_modules(model)]
    lora_module_prefixes = tuple(f"{name}." for name in _lora_module_names if name)
    has_root_lora_module = any(name == "" for name in _lora_module_names)

    tensors = dict(adapter_tensors)
    for key, value in trainable.items():
        if key in adapter_tensors:
            continue
        if _is_base_tensor_inside_lora_module(
            key, lora_module_prefixes, has_root_lora_module,
        ):
            continue
        tensors[key] = value

    if not tensors:
        raise ValueError(
            "Unsloth: save_trainable_adapters() found no trainable or LoRA "
            "parameters to save. The model may be fully frozen without LoRA."
        )
    _save_adapter_artifacts(model, path, tensors, adapter_config=adapter_config)


def save_optimizer_state(optimizer, path):
    """Save MLX optimizer state (Adam/AdamW m,v moments, step, learning_rate)
    to ``<path>/optimizer_state.safetensors`` so training can resume from a
    checkpoint with identical optimizer dynamics.

    The optimizer's ``.state`` is a nested dict whose leaves are ``mx.array``.
    ``tree_flatten`` produces dotted-name string keys (e.g.
    ``"layers.0.lora_a.weight.m"``), all values are arrays, so the whole tree
    serializes cleanly with ``mx.save_safetensors``. Round-trip preserves
    bytes exactly for the optimizer's ``.state`` dict.
    """
    import os
    os.makedirs(path, exist_ok=True)
    flat = dict(mlx.utils.tree_flatten(optimizer.state))
    mx.save_safetensors(f"{path}/optimizer_state.safetensors", flat)


def load_optimizer_state(optimizer, path):
    """Inverse of save_optimizer_state. Loads
    ``<path>/optimizer_state.safetensors`` and replaces ``optimizer.state``
    with the unflattened tree.

    Raises FileNotFoundError if the file is missing -- a resume request with
    no optimizer state is a hard error, not silent fall-back, because resuming
    with a fresh optimizer would silently restart Adam's moment estimates.
    """
    state_path = f"{path}/optimizer_state.safetensors"
    flat = mx.load(state_path)
    optimizer.state = mlx.utils.tree_unflatten(list(flat.items()))


def save_trainer_state(trainer_state, path):
    """Save trainer scalar state to ``<path>/trainer_state.json``.

    ``trainer_state`` is a plain dict (JSON-serializable). Currently:
      - ``global_step``: int, the step the checkpoint represents
      - ``train_loss_history``: list[float], for UI continuity
    Kept separate from the safetensors blob because these are scalars/lists,
    not tensors, and JSON is easier to inspect.
    """
    import json
    import os
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/trainer_state.json", "w") as f:
        json.dump(trainer_state, f, indent=2)


def load_trainer_state(path):
    """Inverse of save_trainer_state. Returns the dict, or raises
    FileNotFoundError if not present (resume requires it)."""
    import json
    with open(f"{path}/trainer_state.json", "r") as f:
        return json.load(f)


def save_lora_adapters(model, path, adapter_config=None):
    """Save LoRA adapter weights (lora_a / lora_b only) to disk.

    Args:
        model: MLX model with LoRA-wrapped modules.
        path: Directory to save adapters.
        adapter_config: Optional dict with LoRA config metadata.
    """
    adapter_tensors = collect_mlx_lora_adapter_tensors(model)
    if not adapter_tensors:
        raise ValueError(
            "Unsloth: no MLX LoRA adapter tensors were found to save. "
            "The model may have no LoRA layers, or adapters may have been "
            "merged. Use save_trainable_adapters() to checkpoint non-LoRA "
            "trainable state instead."
        )
    _save_adapter_artifacts(
        model, path, adapter_tensors, adapter_config=adapter_config
    )


def _infer_snapshot_commit(path):
    if not path:
        return None
    parts = os.path.normpath(str(path)).split(os.sep)
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1] or None


def _effective_mlx_quantization_map(model):
    quantized = {}
    config = getattr(model, "_config", None)
    if isinstance(config, dict):
        quantized.update(_quantization_config_to_path_map(
            config.get("quantization") or config.get("quantization_config")
        ))
    quantized.update(_quantization_config_to_path_map(
        getattr(model, "_unsloth_quantization_config", None)
    ))
    for name, module in model.named_modules():
        if not name:
            continue
        if type(module).__name__ not in {"QuantizedLinear", "QuantizedEmbedding"}:
            continue
        name = _canonical_mlx_quantization_path(name)
        entry = {}
        for key in ("bits", "group_size", "mode"):
            if hasattr(module, key):
                value = getattr(module, key)
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
                entry[key] = value
        quantized[name] = entry
    return quantized


def _quantization_config_to_path_map(config):
    if not isinstance(config, dict):
        return {}
    defaults = {
        key: config.get(key)
        for key in ("bits", "group_size", "mode")
        if config.get(key) is not None
    }
    if defaults and "mode" not in defaults:
        defaults["mode"] = "affine"
    reserved = {
        "bits", "group_size", "mode", "quant_method", "skip_vision",
        "skip_projector", "skip_lm_head",
    }
    quantized = {}
    for name, value in config.items():
        if name in reserved or not isinstance(value, dict):
            continue
        entry = dict(defaults)
        entry.update(value)
        if "bits" not in entry or "group_size" not in entry:
            continue
        if "mode" not in entry:
            entry["mode"] = "affine"
        quantized[_canonical_mlx_quantization_path(str(name))] = {
            key: int(item) if key in ("bits", "group_size") else item
            for key, item in entry.items()
            if key in ("bits", "group_size", "mode")
        }
    return quantized


def _canonical_mlx_quantization_path(path):
    if path.endswith(".linear"):
        return path[:-len(".linear")]
    return path


def _strip_mlx_quantization_metadata(config):
    if not isinstance(config, dict):
        return config
    stripped = {}
    for key, value in config.items():
        if key in {"quantization", "quantization_config"}:
            continue
        if isinstance(value, dict):
            value = _strip_mlx_quantization_metadata(value)
        stripped[key] = value
    return stripped


def _get_mlx_config_quantization(model):
    config = getattr(model, "_config", None)
    if not isinstance(config, dict):
        return None
    return config.get("quantization") or config.get("quantization_config")


def _get_mlx_dropout_probability(drop):
    if drop is None:
        return 0.0
    # MLX nn.Dropout stores keep-prob as _p_1; shims may also set a stale .p,
    # so _p_1 wins when numeric and we fall back to .p otherwise.
    p1 = getattr(drop, "_p_1", None)
    if p1 is not None:
        try:
            return float(1.0 - float(p1))
        except (TypeError, ValueError):
            pass
    p = getattr(drop, "p", None)
    if p is not None:
        try:
            return float(p)
        except (TypeError, ValueError):
            pass
    return 0.0


def _coerce_mlx_lora_scale(scale, default=1.0):
    """Return a Python float from an mlx-lm LoRA wrapper's `.scale` attribute.

    LoRASwitchLinear stores per-expert mx.array scales; raw float()/.item()
    raise. Read the first broadcast value (= alpha/r for every expert) so
    we don't silently lose the trained alpha/r by defaulting to 1.0.
    """
    if scale is None:
        return float(default)

    try:
        return float(scale)
    except Exception:
        pass

    if hasattr(scale, "item"):
        try:
            return float(scale.item())
        except Exception:
            pass

    # LoRASwitchLinear per-expert mx.array: first broadcast value = alpha/r.
    try:
        flat = scale.reshape((-1,))
        first = flat[0]
        if hasattr(first, "item"):
            return float(first.item())
        return float(first)
    except Exception:
        return float(default)


def _infer_mlx_lora_rank(module):
    lora_a = getattr(module, "lora_a", None)
    lora_b = getattr(module, "lora_b", None)

    # mlx-lm sometimes wraps tensors in nn.Linear layers; unwrap to .weight.
    is_layer = False
    if lora_a is not None and not hasattr(lora_a, "shape") and hasattr(lora_a, "weight"):
        lora_a = lora_a.weight
        is_layer = True
    if lora_b is not None and not hasattr(lora_b, "shape") and hasattr(lora_b, "weight"):
        lora_b = lora_b.weight
        is_layer = True

    lora_a_shape = tuple(lora_a.shape) if lora_a is not None and hasattr(lora_a, "shape") else ()
    lora_b_shape = tuple(lora_b.shape) if lora_b is not None and hasattr(lora_b, "shape") else ()
    # Both halves required; a half-built module is not a reliable rank source.
    if not lora_a_shape or not lora_b_shape:
        return None

    # MoE/switch: lora_a (..., rank, in_dims); lora_b (..., out_dims, rank).
    if len(lora_a_shape) >= 3:
        if len(lora_b_shape) != len(lora_a_shape):
            return None
        rank = lora_a_shape[-2]
        if lora_b_shape[-1] != rank:
            return None
        if lora_a_shape[:-2] != lora_b_shape[:-2]:
            return None
        return int(rank)

    if len(lora_a_shape) < 2 or len(lora_b_shape) < 2:
        return None

    # Standard 2D LoRA, two conventions:
    # 1. mlx-lm layer: lora_a (rank, in), lora_b (out, rank)
    if is_layer:
        if lora_a_shape[0] == lora_b_shape[-1]:
            return int(lora_a_shape[0])
        return None

    # 2. Raw array: lora_a (in, rank), lora_b (rank, out)
    if lora_a_shape[-1] == lora_b_shape[0]:
        return int(lora_a_shape[-1])

    return None


def _sync_mlx_lora_keys(adapter_config, lora_parameters):
    """Mirror `unsloth_mlx_lora_module_paths` into `lora_parameters["keys"]`.

    mlx-lm.load_adapters() reads `lora_parameters.keys` to pick which
    submodules to wrap on reload. Mirror the authoritative path list when
    present (an empty list is its own valid pin); otherwise drop a stale
    caller-supplied `keys` so mlx-lm falls back to its scan default.
    """
    if "unsloth_mlx_lora_module_paths" in adapter_config:
        lora_parameters["keys"] = list(
            adapter_config.get("unsloth_mlx_lora_module_paths") or []
        )
    else:
        lora_parameters.pop("keys", None)
    return lora_parameters


def _enrich_mlx_adapter_config(model, adapter_config):
    adapter_config = dict(adapter_config or {})
    hf_repo = getattr(model, "_hf_repo", None) or adapter_config.get("base_model_name_or_path")
    if hf_repo:
        adapter_config["base_model_name_or_path"] = hf_repo

    base_revision = getattr(model, "_unsloth_base_revision", None)
    if base_revision is not None:
        adapter_config["base_model_revision"] = base_revision
    else:
        adapter_config.pop("base_model_revision", None)

    base_commit = (
        getattr(model, "_unsloth_base_commit_hash", None)
        or _infer_snapshot_commit(getattr(model, "_src_path", None))
    )
    if base_commit is not None:
        adapter_config["base_model_commit_hash"] = base_commit
    else:
        adapter_config.pop("base_model_commit_hash", None)

    quant_config = getattr(model, "_unsloth_quantization_config", None)
    quant_policy = getattr(model, "_unsloth_quantization_policy", None)
    quant_source = getattr(model, "_unsloth_quantized_source", None)
    config_quantization = _get_mlx_config_quantization(model)
    if quant_config is None and config_quantization is not None:
        quant_config = config_quantization
    if quant_source is None and config_quantization is not None:
        quant_source = "mlx_config"
    if quant_config is not None:
        adapter_config["base_quantization_config"] = quant_config
    else:
        adapter_config.pop("base_quantization_config", None)
    if quant_policy is not None:
        adapter_config["base_quantization_policy"] = quant_policy
    else:
        adapter_config.pop("base_quantization_policy", None)
    if quant_source is not None:
        adapter_config["base_quantized_source"] = quant_source
    else:
        adapter_config.pop("base_quantized_source", None)

    resolved_map = _effective_mlx_quantization_map(model)
    if resolved_map:
        adapter_config["base_resolved_quantization_map"] = resolved_map
        adapter_config.pop("base_quantization_map", None)
    else:
        adapter_config.pop("base_resolved_quantization_map", None)
        adapter_config.pop("base_quantization_map", None)

    requires_runtime = False
    if quant_source == "runtime":
        requires_runtime = True
    if isinstance(quant_policy, dict) and quant_policy.get("has_callable_predicate"):
        requires_runtime = True
    if resolved_map and quant_source != "mlx_config":
        requires_runtime = True
    adapter_config["requires_unsloth_mlx_runtime_quantization"] = bool(requires_runtime)

    # Only stamp LoRA fields when the live model has LoRA modules (or the
    # caller declared a lora/dora artifact); otherwise mlx-lm.load_adapters()
    # would inject LoRA wrappers before binding full-precision weights and
    # break reload.
    has_lora_modules = any(True for _ in iter_mlx_lora_modules(model))
    declared_lora_artifact = (
        has_lora_modules
        and adapter_config.get("fine_tune_type") in {"lora", "dora"}
    )
    # Override a stale caller fine_tune_type="full" when LoRA modules are
    # live, else mlx-lm reload would skip LoRA wrapping and drop tensors.
    if has_lora_modules and adapter_config.get("fine_tune_type") == "full":
        adapter_config["fine_tune_type"] = "lora"
        declared_lora_artifact = True
    # Detect DoRA so we stamp 'dora' instead of 'lora'; otherwise mlx-lm's
    # load_adapters() rebuilds plain LoRA and the saved q_proj.m magnitude
    # tensor silently drops via strict=False.
    has_dora_modules = any(
        type(module).__name__.startswith("DoRA")
        for _, module in iter_mlx_lora_modules(model)
    )
    if has_lora_modules or declared_lora_artifact:
        # Saved tensor shapes are authoritative: always re-derive
        # rank/scale/dropout from live modules so a stale caller
        # `lora_parameters` (e.g. rank=99) cannot survive. The path-
        # filtered walker below owns the explicit-paths case so we do
        # not borrow rank from an unselected module.
        _has_explicit_paths_hint = "unsloth_mlx_lora_module_paths" in adapter_config
        if has_lora_modules and not _has_explicit_paths_hint:
            rank, scale, dropout = _extract_mlx_lora_parameters(model)
            adapter_config["lora_parameters"] = {
                "rank": rank,
                "scale": scale,
                "dropout": dropout,
            }
        elif "lora_parameters" not in adapter_config and not _has_explicit_paths_hint:
            rank, scale, dropout = _extract_mlx_lora_parameters(model)
            adapter_config["lora_parameters"] = {
                "rank": rank,
                "scale": scale,
                "dropout": dropout,
            }
        # Mirror lora_parameters to top-level for mlx-vlm; overwrite
        # caller-supplied top-level rank/scale/dropout so stale values
        # cannot shadow the canonical ones.
        if "lora_parameters" in adapter_config:
            lora_parameters = adapter_config["lora_parameters"]
            for key in ("rank", "scale", "dropout"):
                if key in lora_parameters:
                    adapter_config[key] = lora_parameters[key]
        # mlx-lm.load_adapters() reads num_layers off the config.
        if "num_layers" not in adapter_config:
            try:
                layers = _get_transformer_layers(model)
                if layers and len(layers) > 0:
                    adapter_config["num_layers"] = len(layers)
            except Exception:
                pass
        # Derive fine_tune_type from the live model so a stale caller value
        # (e.g. 'dora' over plain LoRA) cannot survive; otherwise mlx-lm
        # would expect a `m` tensor and drop every adapter via strict=False.
        adapter_config["fine_tune_type"] = "dora" if has_dora_modules else "lora"
        adapter_config["peft_type"] = "LORA"
    else:
        # Full fine-tune: stamp 'full' explicitly so reload routes to the
        # no-LoRA path (mlx-lm defaults missing fine_tune_type to 'lora').
        adapter_config["fine_tune_type"] = "full"
        for _stale in (
            "peft_type", "lora_parameters", "rank", "scale", "dropout",
            "num_layers", "unsloth_mlx_lora_module_paths",
        ):
            adapter_config.pop(_stale, None)

    # Persist module paths + rank/scale/dropout so reload reproduces logits.
    # The walker below backfills lora_parameters via _infer_mlx_lora_rank
    # and respects explicit caller path filters.
    try:
        # Reuse the loader-side normalizer so save/load accept the same
        # shapes (str / list / tuple / set / dict / pathlib.Path).
        from .loader import _normalize_mlx_lora_module_paths
        # distinguish "caller passed nothing" from "caller passed [] / None".
        has_explicit_paths = "unsloth_mlx_lora_module_paths" in adapter_config
        raw_explicit_paths = (
            adapter_config.get("unsloth_mlx_lora_module_paths")
            if has_explicit_paths else None
        )
        if has_explicit_paths:
            explicit_paths = _normalize_mlx_lora_module_paths(raw_explicit_paths)
            adapter_config["unsloth_mlx_lora_module_paths"] = explicit_paths
        else:
            explicit_paths = None
        # Empty explicit list pins caller topology but should NOT suppress
        # global parameter inference; treat empty as "no filter".
        explicit_path_set = set(explicit_paths) if explicit_paths else None

        lora_paths = []
        lora_rank = None
        lora_scale = None
        lora_dropout = None
        # Track whether an explicit filter selected any real live module:
        # if it did but rank inference failed, refusing the caller fallback
        # below is what prevents persisting stale rank=8 placeholders.
        selected_lora_seen = False
        for name, module in iter_mlx_lora_modules(model):
            lora_paths.append(name)
            inferred_rank = _infer_mlx_lora_rank(module)
            # Only infer from caller-selected modules so an unrelated LoRA
            # cannot write the wrong language-tower params.
            if explicit_path_set is not None and name not in explicit_path_set:
                continue
            selected_lora_seen = True
            if inferred_rank is None:
                continue
            if lora_rank is None:
                lora_rank = inferred_rank
                lora_scale = _coerce_mlx_lora_scale(
                    getattr(module, "scale", 1.0),
                )
                lora_dropout = _get_mlx_dropout_probability(
                    getattr(module, "dropout", None)
                )

        # Auto-fill only when the caller did not supply the key.
        if lora_paths and not has_explicit_paths:
            adapter_config["unsloth_mlx_lora_module_paths"] = lora_paths

        # Live LoRA module state describes the tensors being saved, so an
        # inferable live rank ALWAYS overrides caller scalar metadata.
        existing_lora_parameters = dict(adapter_config.get("lora_parameters") or {})
        has_caller_lora_metadata = any(
            key in existing_lora_parameters or key in adapter_config
            for key in ("rank", "scale", "dropout")
        )
        # When the caller pinned explicit paths that selected real modules
        # but rank inference failed everywhere, do NOT fall back to caller
        # metadata; it is stale by construction.
        allow_caller_metadata_fallback = not (
            explicit_path_set is not None
            and selected_lora_seen
            and lora_rank is None
        )

        if lora_rank is not None:
            lora_parameters = existing_lora_parameters
            lora_parameters.update({
                "rank": lora_rank,
                "scale": lora_scale,
                "dropout": lora_dropout,
            })
            # Mirror the authoritative path list under `keys` so mlx-lm.
            # load_adapters() does not interpret a missing `keys` as "scan
            # every Linear/Embedding/Switch layer".
            lora_parameters = _sync_mlx_lora_keys(adapter_config, lora_parameters)
            adapter_config["lora_parameters"] = lora_parameters
            adapter_config["rank"] = lora_rank
            adapter_config["scale"] = lora_scale
            adapter_config["dropout"] = lora_dropout
            adapter_config.setdefault("peft_type", "LORA")
            adapter_config.setdefault("fine_tune_type", "lora")
            # mlx-lm.load_adapters() attr-accesses `config.num_layers` on
            # a SimpleNamespace, so the key MUST be present. -1 is the
            # legacy "all layers" sentinel for the no-detect case.
            if "num_layers" not in adapter_config:
                layers = _get_transformer_layers(model)
                try:
                    n_layers = len(layers) if layers is not None else -1
                except TypeError:
                    n_layers = -1
                if n_layers <= 0:
                    n_layers = -1
                adapter_config["num_layers"] = n_layers
        elif has_caller_lora_metadata and allow_caller_metadata_fallback:
            # Caller-supplied metadata path: copy top-level rank/scale/dropout
            # into lora_parameters (or vice versa) and backfill missing keys
            # from inferred values so mlx-lm.load_adapters sees a complete dict.
            lora_parameters = existing_lora_parameters
            for key in ("rank", "scale", "dropout"):
                if key not in lora_parameters and key in adapter_config:
                    lora_parameters[key] = adapter_config[key]
            # Rank first because it gates the final write below.
            inferred_fallbacks = (
                ("rank", lora_rank, None),
                ("scale", lora_scale, 1.0),
                ("dropout", lora_dropout, 0.0),
            )
            for key, inferred_value, default_value in inferred_fallbacks:
                if key in lora_parameters:
                    continue
                if inferred_value is None and default_value is None:
                    # No inferred value AND no safe default (rank); leave absent.
                    continue
                lora_parameters[key] = (
                    inferred_value if inferred_value is not None else default_value
                )
            if "rank" in lora_parameters:
                lora_parameters = _sync_mlx_lora_keys(adapter_config, lora_parameters)
                adapter_config["lora_parameters"] = lora_parameters
                for key in ("rank", "scale", "dropout"):
                    if key in lora_parameters:
                        adapter_config[key] = lora_parameters[key]
                adapter_config.setdefault("peft_type", "LORA")
                adapter_config.setdefault("fine_tune_type", "lora")
                # Same -1 sentinel as the main branch.
                if "num_layers" not in adapter_config:
                    layers = _get_transformer_layers(model)
                    try:
                        n_layers = len(layers) if layers is not None else -1
                    except TypeError:
                        n_layers = -1
                    if n_layers <= 0:
                        n_layers = -1
                    adapter_config["num_layers"] = n_layers
    except (TypeError, ValueError, AttributeError) as _enrich_exc:
        # Surface enrichment failures so the user knows adapter_config
        # metadata may be incomplete on reload.
        import warnings as _warnings
        _warnings.warn(
            f"Unsloth MLX: skipped LoRA metadata enrichment "
            f"({_enrich_exc!r}); reloaded adapters may use placeholder "
            f"rank/scale until adapter_config is rewritten.",
            stacklevel=2,
        )
    return adapter_config


def _config_to_plain_python(value):
    """Recursively convert config dataclasses and containers to plain Python."""
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif isinstance(value, dict):
        value = copy.deepcopy(value)
    elif isinstance(value, (list, tuple)):
        return [_config_to_plain_python(item) for item in value]
    else:
        return value

    if isinstance(value, dict):
        return {
            key: _config_to_plain_python(item)
            for key, item in value.items()
        }
    return value


def _get_model_config(model):
    """Extract config dict from an MLX model.

    mlx-lm stores the raw config dict at model._config when loaded.
    mlx-vlm exposes config dataclasses at model.config.
    Falls back to reconstructing from model.args dataclass.
    """
    import dataclasses

    # Prefer the raw config dict stashed by our loader
    if hasattr(model, "_config") and isinstance(model._config, dict):
        return _config_to_plain_python(model._config)

    if hasattr(model, "config"):
        config = model.config
        if isinstance(config, dict) or (
            dataclasses.is_dataclass(config) and not isinstance(config, type)
        ):
            return _config_to_plain_python(config)
        if hasattr(config, "to_dict"):
            config = config.to_dict()
            if isinstance(config, dict):
                return _config_to_plain_python(config)

    # Reconstruct from the ModelArgs dataclass
    if hasattr(model, "args"):
        if dataclasses.is_dataclass(model.args) and not isinstance(model.args, type):
            return _config_to_plain_python(model.args)

    return {}


def _get_src_path(model):
    """Get the original model source path/repo for copying auxiliary files."""
    return getattr(model, "_src_path", None)


def _save_mlx_config(config, config_path, *, is_vlm=False):
    """Save MLX config using the backend-aware upstream helper."""
    config = copy.deepcopy(config)
    if is_vlm:
        if "quantization" in config:
            config["quantization_config"] = config["quantization"]
        from mlx_vlm.utils import save_config as save_vlm_config
        save_vlm_config(config, config_path)
    else:
        from mlx_lm.utils import save_config as save_lm_config
        save_lm_config(config, config_path)


def _has_vision_config(config):
    """Return whether a raw or thinker-wrapped VLM config has vision settings."""
    if not isinstance(config, dict):
        return False
    thinker_config = config.get("thinker_config")
    return (
        "vision_config" in config
        or (
            isinstance(thinker_config, dict)
            and "vision_config" in thinker_config
        )
    )


class _MlxVlmSanitizeProxy:
    """Minimal instance shim for mlx-vlm class sanitize methods."""
    def __init__(self, config):
        self.config = config
        self.args = config


def _copy_mlx_vlm_sanitize_weights(weights):
    """Copy MLX arrays before replaying sanitizer transforms."""
    return {
        key: mx.array(value) if isinstance(value, mx.array) else value
        for key, value in weights.items()
    }


def _call_mlx_vlm_sanitize(cls, config, weights):
    """Call an mlx-vlm sanitize method with its expected signature."""
    sanitize = getattr(cls, "sanitize", None)
    if sanitize is None:
        return weights

    weights = _copy_mlx_vlm_sanitize_weights(weights)
    params = inspect.signature(sanitize).parameters
    if len(params) == 1:
        return sanitize(weights)
    return sanitize(_MlxVlmSanitizeProxy(config), weights)


def _add_mlx_vlm_sanitize_step(steps, module):
    """Append a real mlx-vlm module sanitizer once, preserving order."""
    if module is None or getattr(module, "sanitize", None) is None:
        return
    if all(existing is not module for existing, _ in steps):
        steps.append((module, None))


def _get_mlx_vlm_model_sanitize_pipelines(model):
    """Build sanitizer pipelines from a loaded mlx-vlm model and submodules."""
    if model is None:
        return []

    # Submodule-only sanitizers (wrapper without its own sanitize) still
    # need replay pipelines, so the wrapper step is optional.
    model_step = [(model, None)] if getattr(model, "sanitize", None) is not None else []
    pipelines = [model_step] if model_step else []

    extra_steps = []
    for attr in ("thinker", "vision_tower", "vision_model", "vision_encoder", "visual"):
        _add_mlx_vlm_sanitize_step(extra_steps, getattr(model, attr, None))

    thinker = getattr(model, "thinker", None)
    for attr in ("vision_tower", "vision_model", "vision_encoder", "visual"):
        _add_mlx_vlm_sanitize_step(extra_steps, getattr(thinker, attr, None))

    for idx in range(len(extra_steps)):
        pipelines.append(model_step + extra_steps[: idx + 1])

    return pipelines


def _get_nested_config(config, *names):
    """Walk nested config attributes, returning None for missing segments."""
    cur = config
    for name in names:
        cur = getattr(cur, name, None)
        if cur is None:
            return None
    return cur


def _build_mlx_vlm_sanitize_steps(config):
    """Build class-based mlx-vlm sanitizer steps from a saved VLM config."""
    if not _has_vision_config(config):
        return []

    try:
        from mlx_vlm.utils import get_model_and_args, update_module_configs

        config_copy = copy.deepcopy(config)
        model_module, model_type = get_model_and_args(config_copy)
        config_copy.setdefault("text_config", config_copy.pop("llm_config", {}))
        config_copy.setdefault("vision_config", {})
        config_copy.setdefault("audio_config", {})

        model_config = model_module.ModelConfig.from_dict(config_copy)
        try:
            model_config = update_module_configs(
                model_config,
                model_module,
                config_copy,
                ["text", "vision", "perceiver", "projector", "audio"],
            )
        except Exception:
            pass
    except Exception:
        return []

    steps = []
    if hasattr(model_module, "Model"):
        steps.append((model_module.Model, model_config))

    thinker_config = _get_nested_config(model_config, "thinker_config")
    if thinker_config is not None:
        thinker_cls = getattr(model_module, "Thinker", None)
        if thinker_cls is None:
            try:
                thinker_mod = importlib.import_module(
                    f"mlx_vlm.models.{model_type}.thinker"
                )
                thinker_cls = getattr(thinker_mod, "Thinker", None)
            except Exception:
                thinker_cls = None
        if thinker_cls is not None:
            steps.append((thinker_cls, thinker_config))

    vision_config = (
        _get_nested_config(model_config, "vision_config")
        or _get_nested_config(model_config, "thinker_config", "vision_config")
    )
    if vision_config is not None and hasattr(model_module, "VisionModel"):
        steps.append((model_module.VisionModel, vision_config))

    return [
        (cls, step_config)
        for cls, step_config in steps
        if getattr(cls, "sanitize", None) is not None
    ]


def _build_mlx_vlm_sanitize_pipelines(config, model=None):
    """Combine real-model and config-derived sanitizer replay pipelines."""
    pipelines = _get_mlx_vlm_model_sanitize_pipelines(model)
    class_steps = _build_mlx_vlm_sanitize_steps(config)
    if class_steps:
        pipelines.append(class_steps)
    return pipelines


def _apply_mlx_vlm_sanitizers(steps, weights):
    """Replay a sanitizer pipeline and return None if any step rejects it."""
    sanitized = dict(weights)
    for cls, config in steps:
        try:
            sanitized = _call_mlx_vlm_sanitize(cls, config, sanitized)
        except Exception:
            return None
    return sanitized


def _vlm_gguf_name_candidates(name):
    """Yield HF/llama.cpp tensor-name candidates for an MLX VLM tensor."""
    candidates = []

    def add(value):
        if value not in candidates:
            candidates.append(value)

    if name.startswith("thinker.vision_tower."):
        suffix = name[len("thinker.vision_tower."):]
        add(f"thinker.visual.{suffix}")
    if name.startswith("model.vision_tower."):
        suffix = name[len("model.vision_tower."):]
        add(f"model.visual.{suffix}")
    if name.startswith("vision_tower."):
        suffix = name[len("vision_tower."):]
        add(f"visual.{suffix}")
        add(f"model.visual.{suffix}")
        add(f"model.language_model.visual.{suffix}")
        add(f"vit.{suffix}")

    add(name)
    return candidates


def _vlm_gguf_tensor_candidates(tensor):
    """Yield HF-layout tensor candidates for an MLX VLM tensor."""
    candidates = []
    shape = getattr(tensor, "shape", ())

    if len(shape) == 5:
        candidates.append(mx.transpose(tensor, (0, 4, 1, 2, 3)))
    elif len(shape) == 4:
        candidates.append(mx.transpose(tensor, (0, 3, 1, 2)))

    if len(shape) == 1 and mx.issubdtype(tensor.dtype, mx.floating):
        candidates.append(tensor - 1)

    candidates.append(tensor)
    return candidates


def _has_vlm_gguf_tensor_candidate(tensor):
    """Return whether a tensor shape can require HF-layout recovery."""
    shape = getattr(tensor, "shape", ())
    if len(shape) in (4, 5):
        return True
    if len(shape) == 1:
        dtype = getattr(tensor, "dtype", None)
        return dtype is not None and mx.issubdtype(dtype, mx.floating)
    return False


def _has_vlm_gguf_rewrite_candidate(name, tensor):
    """Return whether a tensor can differ between mlx-vlm and GGUF layouts."""
    if any(candidate_name != name for candidate_name in _vlm_gguf_name_candidates(name)):
        return True
    return _has_vlm_gguf_tensor_candidate(tensor)


def _mlx_arrays_match(actual, expected):
    """Compare MLX-like arrays without assuming a concrete backend type."""
    shape = getattr(actual, "shape", None)
    if shape != getattr(expected, "shape", None):
        return False
    if actual is expected:
        return True
    if shape is None:
        return actual == expected
    try:
        result = mx.all(actual == expected)
        item = getattr(result, "item", None)
        return bool(item() if callable(item) else result)
    except Exception:
        return False


def _is_mlx_vlm_sanitize_step(value):
    """Return whether a value is one sanitizer step tuple."""
    return isinstance(value, tuple) and len(value) == 2


def _normalize_mlx_vlm_sanitize_pipelines(sanitize_steps):
    """Normalize legacy step lists and multi-pipeline sanitizer inputs."""
    if not sanitize_steps:
        return []
    if all(_is_mlx_vlm_sanitize_step(step) for step in sanitize_steps):
        return [sanitize_steps]
    return sanitize_steps


def _rewrite_mlx_vlm_tensor_for_gguf(name, tensor, sanitize_steps):
    """Invert mlx-vlm sanitizers to recover HF tensor names/layouts for GGUF."""
    if not _has_vlm_gguf_rewrite_candidate(name, tensor):
        return name, tensor, False

    for candidate_name in _vlm_gguf_name_candidates(name):
        for candidate_tensor in _vlm_gguf_tensor_candidates(tensor):
            for pipeline in _normalize_mlx_vlm_sanitize_pipelines(sanitize_steps):
                sanitized = _apply_mlx_vlm_sanitizers(
                    pipeline,
                    {candidate_name: candidate_tensor},
                )
                if not sanitized or len(sanitized) != 1:
                    continue
                sanitized_name, sanitized_tensor = next(iter(sanitized.items()))
                if sanitized_name != name:
                    continue
                if not _mlx_arrays_match(sanitized_tensor, tensor):
                    continue
                changed = (
                    candidate_name != name
                    or not _mlx_arrays_match(candidate_tensor, tensor)
                )
                if not changed:
                    continue
                return candidate_name, candidate_tensor, True

    return name, tensor, False


def _sync_gguf_nextn_layer_config(config, model):
    """Align speculative-layer config metadata with exported MLX layers."""
    if model is None or not isinstance(config, dict):
        return False

    layers = _get_transformer_layers(model)
    if layers is None:
        return False
    try:
        actual_layers = len(layers)
    except Exception:
        return False

    thinker_config = config.get("thinker_config")
    text_configs = [
        config.get("text_config"),
        config.get("language_config"),
        (
            thinker_config.get("text_config")
            if isinstance(thinker_config, dict)
            else None
        ),
    ]
    changed = False
    for text_config in text_configs:
        if not isinstance(text_config, dict):
            continue
        num_hidden_layers = text_config.get("num_hidden_layers")
        if not isinstance(num_hidden_layers, int):
            continue

        actual_nextn = actual_layers - num_hidden_layers
        for key in (
            "num_nextn_predict_layers",
            "mtp_num_hidden_layers",
            "nextn_predict_layers",
        ):
            num_nextn = text_config.get(key)
            if not isinstance(num_nextn, int) or num_nextn <= 0:
                continue
            if actual_layers < num_hidden_layers:
                continue
            if actual_nextn >= num_nextn:
                continue
            if actual_nextn > 0:
                text_config[key] = actual_nextn
            else:
                text_config.pop(key, None)
            changed = True

    return changed


def _prepare_vlm_gguf_export_directory(path, model=None):
    """Rewrite MLX-native VLM tensor names in the temporary GGUF export dir."""
    path = Path(path)
    config_path = path / "config.json"
    if not config_path.exists():
        return 0
    with open(config_path, "r") as f:
        config = json.load(f)
    config_changed = _sync_gguf_nextn_layer_config(config, model)
    sanitize_steps = _build_mlx_vlm_sanitize_pipelines(config, model=model)
    if not sanitize_steps:
        if config_changed:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        return 0

    rewritten = 0
    name_map = {}
    for file in sorted(path.glob("*.safetensors")):
        tensors = mx.load(str(file))
        updated = {}
        file_rewritten = 0
        for name, tensor in tensors.items():
            new_name, tensor, changed = _rewrite_mlx_vlm_tensor_for_gguf(
                name, tensor, sanitize_steps
            )
            if new_name in updated:
                raise RuntimeError(
                    f"Unsloth: duplicate tensor name after GGUF VLM rewrite: {new_name}"
                )
            updated[new_name] = tensor
            name_map[name] = new_name
            file_rewritten += int(changed)
        if file_rewritten:
            # mx.load() arrays may be file-backed; saving over the source can
            # truncate them before they materialize, so write beside and replace.
            mx.eval(*updated.values())
            tmp_file = file.with_name(f"{file.stem}.tmp{file.suffix}")
            mx.save_safetensors(str(tmp_file), updated, metadata={"format": "mlx"})
            os.replace(tmp_file, file)
            rewritten += file_rewritten

    index_path = path / "model.safetensors.index.json"
    if rewritten and index_path.exists():
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = {}
        for name, shard in index_data.get("weight_map", {}).items():
            new_name = name_map.get(name, name)
            if new_name in weight_map:
                raise RuntimeError(
                    f"Unsloth: duplicate index tensor name after GGUF VLM rewrite: {new_name}"
                )
            weight_map[new_name] = shard
        index_data["weight_map"] = dict(sorted(weight_map.items()))
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=4)

    if config_changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    return rewritten


_CORE_SAVE_FILENAMES = {
    "config.json",
    "model.safetensors.index.json",
    "README.md",
    ".gitattributes",
}
_MODEL_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".gguf",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
)
_MODEL_SIDECAR_SUFFIXES = (".json", ".jinja", ".model", ".txt", ".py")


def _copy_source_sidecars(src_path, path):
    """Copy non-weight source sidecars that tokenizer/model saves may omit."""
    copied = 0
    src_path = Path(src_path)
    path = Path(path)
    if not src_path.is_dir():
        return copied
    for source in src_path.iterdir():
        if not source.is_file():
            continue
        name = source.name
        if name in _CORE_SAVE_FILENAMES:
            continue
        if name.startswith("model-") or name.startswith("pytorch_model"):
            continue
        suffix = source.suffix
        if suffix in _MODEL_WEIGHT_SUFFIXES:
            continue
        if suffix not in _MODEL_SIDECAR_SUFFIXES:
            continue
        target = path / name
        if target.exists():
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied

def save_merged_model(model, tokenizer, path, dequantize=False):
    """Fuse LoRA weights and save the full merged model.

    Produces an HF-compatible directory with sharded safetensors,
    config.json, tokenizer files, and a model card. The output can
    be reloaded with ``mlx_lm.load()`` or uploaded to HuggingFace Hub.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        path: Directory to save merged model.
        dequantize: If True, dequantize quantized layers when fusing
            (saves as fp16/bf16 — needed for GGUF). If False, keep the
            base quantization (smaller checkpoint, only meaningful when
            the base was quantized).
    """
    from mlx_lm.utils import save_model, create_model_card, dequantize_model
    from mlx.utils import tree_unflatten

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Fuse LoRA weights into base model (mlx-lm pattern)
    model.eval()
    fused_linears = [
        (n, m.fuse(dequantize=dequantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if dequantize:
        model = dequantize_model(model)
        cfg = getattr(model, "_config", None)
        if isinstance(cfg, dict):
            model._config = _strip_mlx_quantization_metadata(cfg)

    de_lora_model = model

    # Save sharded safetensors + index.json
    save_model(path, de_lora_model, donate_model=False)

    # Save config.json
    config = _get_model_config(model)
    if config:
        is_vlm = _is_vlm_model(model) or _has_vision_config(config)
        _save_mlx_config(
            config,
            path / "config.json",
            is_vlm=is_vlm,
        )

    # Save tokenizer
    tokenizer.save_pretrained(str(path))

    src_path = _get_src_path(model)
    if src_path is not None:
        _copy_source_sidecars(src_path, path)

    # Model card
    hf_repo = getattr(model, "_hf_repo", None)
    try:
        create_model_card(path, hf_repo)
    except Exception:
        # hf_repo missing on Hub — write a minimal card
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text("---\nlibrary_name: mlx\ntags:\n- mlx\n- unsloth\n---\n")

    print(f"Unsloth: Merged model saved to {path}")


def _ensure_hub_repo_visibility(api, repo_id, private):
    """create_repo + update_repo_settings + verify on failure.

    private=None  : skip the visibility update entirely (Hub policy).
    private=False : soft-fail on update error (stuck-private is not a leak).
    private=True  : verify via repo_info on update failure; only raise when
                    the repo is confirmed non-private (covers tokens lacking
                    write:repo_settings on a just-created private repo).
    """
    _create_repo_kwargs = {"repo_id": repo_id, "exist_ok": True}
    if private is not None:
        _create_repo_kwargs["private"] = bool(private)
    api.create_repo(**_create_repo_kwargs)

    if private is None:
        return

    try:
        api.update_repo_settings(
            repo_id=repo_id,
            private=bool(private),
            repo_type="model",
        )
        return
    except Exception as exc:
        if not bool(private):
            print(f"Unsloth: Could not update repo visibility ({exc}); continuing.")
            return

        # private=True: verify before blocking the upload.
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="model")
            if bool(getattr(info, "private", False)):
                return
        except Exception:
            pass

        raise RuntimeError(
            "Unsloth: private=True was requested but the Hub "
            f"repo {repo_id!r} visibility could not be confirmed "
            "private (likely token lacks `write:repo_settings` or "
            "the repo is owned by another user). Refusing to upload "
            "to avoid publishing artifacts to an existing public "
            "repository."
        ) from exc


# Module-level constants so _caller_wants_commit_metadata can distinguish
# "caller explicitly wants this Unsloth default" from "caller wants their
# own commit string"; same pattern as push_to_hub_merged.
_LORA_DEFAULT_COMMIT_MESSAGE = "Trained with Unsloth"
_LORA_DEFAULT_COMMIT_DESCRIPTION = "Upload LoRA adapters trained with Unsloth 2x faster"


def _push_lora_adapters_to_hub(
    save_directory,
    repo_id=None,
    token=None,
    private=None,
    tags=None,
    commit_message=None,
    commit_description=None,
    create_pr=False,
    revision=None,
):
    """Upload an already-written LoRA adapter directory to the Hub.

    Mirrors push_to_hub_merged's repo / commit / privacy / tag / upload
    semantics, but skips the merged save_model fallback so it does not
    overwrite adapters.safetensors with a full merged model.
    """
    from huggingface_hub import HfApi

    save_directory = Path(save_directory)
    if repo_id is None:
        repo_id = save_directory.name

    # Capture caller intent BEFORE backfilling defaults. Treat the Unsloth
    # default strings the same as None so an outer wrapper that forwards
    # the default verbatim is not mistaken for a custom request (the
    # upload_large_folder fallback cannot preserve commit metadata).
    _caller_wants_commit_metadata = bool(
        create_pr
        or commit_message not in (None, _LORA_DEFAULT_COMMIT_MESSAGE)
        or commit_description not in (None, _LORA_DEFAULT_COMMIT_DESCRIPTION)
    )

    # Match push_to_hub_merged's commit conventions so the history is
    # recognizable across CUDA and MLX backends.
    if commit_message is None:
        commit_message = _LORA_DEFAULT_COMMIT_MESSAGE
    if "Unsloth" not in commit_message:
        commit_message = (commit_message + " (Trained with Unsloth)").lstrip()
    if commit_description is None:
        commit_description = _LORA_DEFAULT_COMMIT_DESCRIPTION
    elif "Unsloth 2x faster" not in commit_description:
        commit_description += " (Trained with Unsloth 2x faster)"

    api = HfApi(token=token)
    _ensure_hub_repo_visibility(api, repo_id, private)

    if tags:
        try:
            from huggingface_hub import ModelCard
            card_path = save_directory / "README.md"
            # Seed a minimal card: fresh adapter dirs have no README, so
            # ModelCard.load() would otherwise raise and tags would be lost.
            if not card_path.exists():
                card_path.write_text(
                    "---\n"
                    "library_name: mlx\n"
                    "tags:\n"
                    "- mlx\n"
                    "- unsloth\n"
                    "---\n",
                    encoding="utf-8",
                )
            card = ModelCard.load(card_path)
            existing = list(getattr(card.data, "tags", None) or [])
            merged = list(dict.fromkeys(existing + list(tags) + ["mlx", "unsloth"]))
            card.data.tags = merged
            card.save(card_path)
        except Exception as exc:
            print(f"Unsloth: Could not set tags in model card ({exc}); continuing.")

    # Allow-list adapter-relevant files only; prevents accidentally pushing
    # stale merged-model artifacts or external GGUF exports under the LoRA
    # adapter repo (data-leak guard for the public-by-default case).
    _lora_allow_patterns = [
        "adapters.safetensors",
        "adapter_config.json",
        # `adapter_model.safetensors` is intentionally excluded: MLX writes
        # `adapters.safetensors`, so a matching PEFT file is stale by
        # definition and would corrupt the remote artifact.
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_template.jinja",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "config.json",
        "generation_config.json",
        "README.md",
        ".gitattributes",
    ]

    # Prefer upload_folder for LoRA dirs (small, single commit honours
    # commit_message / create_pr / revision); upload_large_folder is the
    # last-resort fallback when upload_folder is unavailable.
    try:
        api.upload_folder(
            folder_path=str(save_directory),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            revision=revision,
            allow_patterns=_lora_allow_patterns,
        )
    except (AttributeError, TypeError) as exc:
        # Refuse to fall back to upload_large_folder when the caller asked
        # for commit metadata it cannot preserve.
        if _caller_wants_commit_metadata:
            raise RuntimeError(
                "Unsloth: upload_folder() failed but commit metadata or "
                "create_pr=True was requested; upload_large_folder() "
                "cannot preserve commit_message, commit_description, or "
                "create a PR. Upgrade huggingface_hub or retry without "
                "those kwargs."
            ) from exc
        api.upload_large_folder(
            folder_path=str(save_directory),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            allow_patterns=_lora_allow_patterns,
        )
    print(f"Unsloth: Pushed LoRA adapters to https://huggingface.co/{repo_id}")


def save_pretrained_merged(
    model,
    tokenizer,
    save_directory,
    save_method="lora",
    push_to_hub=False,
    token=None,
    private=None,
    tags=None,
    repo_id=None,
    commit_message=None,
    commit_description=None,
    create_pr=False,
    revision=None,
):
    """Save the model in HF-compatible format using the requested method.

    This mirrors the CUDA path's ``model.save_pretrained_merged()``.

    Args:
        model: MLX model (with optional LoRA layers).
        tokenizer: Tokenizer to save alongside.
        save_directory: Output directory path.
        save_method: One of:
            - ``"lora"``: save adapter weights only (smallest, default).
            - ``"merged_16bit"``: fuse LoRA into base, dequantize, save full
              fp16/bf16 model. Needed for GGUF / llama.cpp downstream.
            - ``"merged_4bit"``: fuse LoRA into base while keeping the
              base's 4-bit quantization. Only meaningful for QLoRA.
        push_to_hub: If True, upload to HuggingFace Hub after saving.
        token: HuggingFace token for pushing.
        private: Whether the HF repo should be private.
        tags: Additional tags for the model card.
        repo_id: HuggingFace repo ID. If None, uses save_directory name.
        commit_message: Commit title for the upload.
        commit_description: Optional longer commit body.
        create_pr: If True, push to a PR branch instead of main.
        revision: Target branch (defaults to main).
    """
    method = (save_method or "lora").lower().replace(" ", "_")
    if method not in ("lora", "merged_16bit", "merged_4bit"):
        raise ValueError(
            f"Unsloth: Unknown save_method {save_method!r}. "
            f"Use 'lora', 'merged_16bit', or 'merged_4bit'."
        )

    if method == "lora":
        adapter_tensors = collect_mlx_lora_adapter_tensors(model)
        if not adapter_tensors:
            raise ValueError(
                "Unsloth: save_method='lora' but the model has no LoRA "
                "layers — there's nothing to save. Use 'merged_16bit' instead."
            )
        # Preserve intentionally trainable non-LoRA tensors OUTSIDE any
        # LoRA module; exclude wrapped base weights INSIDE a LoRA module
        # so reload-trainable q_proj.weight cannot leak through the save.
        trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
        adapter_keys = set(adapter_tensors)
        lora_module_prefixes = tuple(
            f"{name}." for name, _ in iter_mlx_lora_modules(model) if name
        )
        # Route to save_trainable_adapters when an external param or an
        # intentionally trainable non-base param under a LoRA module (e.g.
        # q_proj.bias) is present. _is_base_tensor_inside_lora_module
        # treats wrapped `.weight`/`.scales`/`.biases` (+ `.linear.*`) as
        # reload-leaks; everything else under a LoRA prefix is user state.
        has_root_lora_module = any(
            name == "" for name, _ in iter_mlx_lora_modules(model)
        )
        has_non_lora_trainable = any(
            key not in adapter_keys
            and not _is_base_tensor_inside_lora_module(
                key, lora_module_prefixes, has_root_lora_module,
            )
            for key in trainable
        )
        if has_non_lora_trainable:
            save_trainable_adapters(model, save_directory)
        else:
            save_lora_adapters(model, save_directory)
        try:
            tokenizer.save_pretrained(str(save_directory))
        except Exception:
            pass
    else:
        # merged_16bit → dequantize fused weights to fp16/bf16
        # merged_4bit  → keep base quantization (LoRA absorbed)
        save_merged_model(
            model, tokenizer, save_directory,
            dequantize=(method == "merged_16bit"),
        )

    if push_to_hub:
        if method == "lora":
            # LoRA artifacts must NOT route through push_to_hub_merged: it
            # would fall back to save_merged_model() (no index.json) and
            # overwrite adapters.safetensors with a full merged model.
            _push_lora_adapters_to_hub(
                save_directory,
                repo_id=repo_id,
                token=token,
                private=private,
                tags=tags,
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
                revision=revision,
            )
        else:
            push_to_hub_merged(
                model, tokenizer, save_directory,
                repo_id=repo_id,
                token=token, private=private, tags=tags,
                commit_message=commit_message or "Trained with Unsloth",
                commit_description=commit_description or (
                    "Upload model trained with Unsloth 2x faster"
                ),
                create_pr=create_pr,
                revision=revision,
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

    # Install deps; prefer gguf from the cloned repo to stay in sync
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
    first_conversion=None,
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
        first_conversion: Optional override for the intermediate GGUF
            dtype produced by convert_hf_to_gguf before llama-quantize
            compresses it to ``quantization_method``. Pass ``"f32"`` /
            ``"f16"`` / ``"bf16"`` to force a specific intermediate
    """
    from ..llama_cpp import (
        convert_to_gguf,
        quantize_gguf,
        check_llama_cpp,
        LLAMA_CPP_DEFAULT_DIR,
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
    if first_conversion is None:
        if quant_type in ("bf16", "f16", "f32"):
            first_conversion = quant_type
        else:
            # k-quants and q8_0 go through a bf16 intermediate, then llama-quantize
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
        is_vlm_model = _is_vlm_model(model)
        print("Unsloth: Merging LoRA weights and saving to 16-bit...")
        save_merged_model(model, tokenizer, tmp_path, dequantize=True)
        if is_vlm_model:
            rewritten = _prepare_vlm_gguf_export_directory(tmp_path, model=model)
            if rewritten:
                print(
                    "Unsloth: Rewrote "
                    f"{rewritten} MLX VLM tensors for llama.cpp GGUF export."
                )

        # Step 2: Ensure llama.cpp is installed and gguf package is available
        llama_cpp_folder = LLAMA_CPP_DEFAULT_DIR
        try:
            quantizer_location, converter_location = check_llama_cpp(llama_cpp_folder)
        except Exception:
            print("Unsloth: Installing llama.cpp (this only happens once)...")
            quantizer_location, converter_location = install_llama_cpp(llama_cpp_folder)
        llama_cpp_folder = os.path.dirname(converter_location)

        # Ensure gguf is installed (may be missing if llama.cpp was built
        # in a different venv)
        try:
            import gguf  # noqa: F401
        except ImportError:
            import subprocess
            gguf_py_dir = os.path.join(llama_cpp_folder, "gguf-py")
            if os.path.exists(gguf_py_dir):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", gguf_py_dir],
                    check=True, capture_output=True,
                )
            else:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "gguf"],
                    check=True, capture_output=True,
                )
            print("Unsloth: Installed gguf Python package.")

        # Step 3: Download and patch convert_hf_to_gguf.py.
        # why: always go through the wrapper so UNSLOTH_LLAMA_CPP_SCRIPTS_DIR
        # is honored even when a cached converter file exists.
        converter = os.path.join(llama_cpp_folder, "unsloth_convert_hf_to_gguf.py")
        supported_text_archs = None
        supported_vision_archs = None
        with _LLAMA_CPP_PATCHER_ENV_LOCK:
            old_scripts_dir = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
            if old_scripts_dir is None:
                os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = llama_cpp_folder
            try:
                result = _download_convert_hf_to_gguf()
            finally:
                if old_scripts_dir is None:
                    os.environ.pop("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", None)
                else:
                    os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = old_scripts_dir
        if isinstance(result, tuple) and len(result) >= 3:
            converter, supported_text_archs, supported_vision_archs = result[:3]
        elif isinstance(result, str):
            converter = result

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
            is_vlm=is_vlm_model,
            is_gpt_oss=False,
            print_output=True,
        )
        if supported_text_archs is not None:
            kwargs["supported_text_archs"] = supported_text_archs
            kwargs["supported_vision_archs"] = supported_vision_archs
        convert_to_gguf(**kwargs)

        # Step 6: Quantize if the target quant differs from first_conversion
        if quant_type not in ("bf16", "f16", "f32") and first_conversion != quant_type:
            quantizer = quantizer_location
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


_PUSH_MERGED_DEFAULT_COMMIT_MESSAGE = "Trained with Unsloth"
_PUSH_MERGED_DEFAULT_COMMIT_DESCRIPTION = "Upload model trained with Unsloth 2x faster"


def push_to_hub_merged(
    model,
    tokenizer,
    save_directory,
    repo_id=None,
    token=None,
    private=None,
    tags=None,
    commit_message=_PUSH_MERGED_DEFAULT_COMMIT_MESSAGE,
    commit_description=_PUSH_MERGED_DEFAULT_COMMIT_DESCRIPTION,
    create_pr=False,
    revision=None,
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
        tags: Additional tags appended to the model card.
        commit_message: Commit title for the upload.
        commit_description: Optional longer commit body.
        create_pr: If True, push to a PR branch instead of main.
        revision: Target branch (defaults to main).
    """
    from huggingface_hub import HfApi

    save_directory = Path(save_directory)

    # Save first if not already saved
    if not (save_directory / "model.safetensors.index.json").exists():
        save_merged_model(model, tokenizer, save_directory)

    if repo_id is None:
        repo_id = save_directory.name

    # Capture caller intent BEFORE backfilling defaults; upload_large_folder
    # cannot preserve commit_message / commit_description / create_pr (only
    # those three force the upload_folder route). Treat None and the
    # Unsloth default strings the same so a wrapper forwarding the default
    # is not mistaken for a custom request.
    _caller_wants_commit_metadata = bool(
        create_pr
        or commit_message not in (None, _PUSH_MERGED_DEFAULT_COMMIT_MESSAGE)
        or commit_description not in (None, _PUSH_MERGED_DEFAULT_COMMIT_DESCRIPTION)
    )

    # Match the GPU path's "(Trained with Unsloth)" suffix convention so
    # the commit history is recognizable across both backends.
    if commit_message is None:
        commit_message = ""
    if "Unsloth" not in commit_message:
        commit_message = (commit_message + " (Trained with Unsloth)").lstrip()
    if commit_description is None:
        commit_description = "Upload model trained with Unsloth 2x faster"
    elif "Unsloth 2x faster" not in commit_description:
        commit_description += " (Trained with Unsloth 2x faster)"

    api = HfApi(token=token)
    _ensure_hub_repo_visibility(api, repo_id, private)

    if tags:
        try:
            from huggingface_hub import ModelCard
            card_path = save_directory / "README.md"
            # Seed a minimal card: fresh dirs without one would crash
            # ModelCard.load() and silently lose the requested tags.
            if not card_path.exists():
                card_path.write_text(
                    "---\n"
                    "library_name: mlx\n"
                    "tags:\n"
                    "- mlx\n"
                    "- unsloth\n"
                    "---\n",
                    encoding="utf-8",
                )
            card = ModelCard.load(card_path)
            existing = list(getattr(card.data, "tags", None) or [])
            merged = list(dict.fromkeys(existing + list(tags) + ["mlx", "unsloth"]))
            card.data.tags = merged
            card.save(card_path)
        except Exception as exc:
            print(f"Unsloth: Could not set tags in model card ({exc}); continuing.")

    # upload_large_folder resumes / chunks (good for multi-GB merges) but
    # drops commit_message / commit_description / create_pr; route through
    # upload_folder when the caller asked for any of those.
    if _caller_wants_commit_metadata:
        try:
            api.upload_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
                revision=revision,
            )
        except (AttributeError, TypeError) as exc:
            # Refuse to fall back to upload_large_folder; it cannot preserve
            # commit_message / commit_description / create_pr.
            if _caller_wants_commit_metadata:
                raise RuntimeError(
                    "Unsloth: upload_folder() failed but commit metadata "
                    "or create_pr=True was requested; upload_large_folder() "
                    "cannot preserve commit_message, commit_description, or "
                    "create a PR. Upgrade huggingface_hub or retry without "
                    "those kwargs."
                ) from exc
            api.upload_large_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
            )
    else:
        try:
            api.upload_large_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
            )
        except (AttributeError, TypeError):
            api.upload_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
                revision=revision,
            )
    print(f"Unsloth: Pushed to https://huggingface.co/{repo_id}")


def push_to_hub_gguf(
    model,
    tokenizer,
    save_directory,
    repo_id,
    quantization_method="fast_quantized",
    token=None,
    private=None,
    first_conversion=None,
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
        first_conversion: Optional intermediate GGUF dtype passed through to
            save_pretrained_gguf. Placed after the pre-existing arguments so
            positional callers keep their meaning.
    """
    from huggingface_hub import HfApi

    save_directory = Path(save_directory)

    # Export to GGUF
    save_pretrained_gguf(
        model,
        tokenizer,
        save_directory,
        quantization_method=quantization_method,
        first_conversion=first_conversion,
    )

    # Upload GGUF files
    api = HfApi(token=token)
    # Same fail-loud private=True rule as the LoRA / merged paths so a
    # private=True request never silently leaks GGUF shards public.
    _ensure_hub_repo_visibility(api, repo_id, private)

    gguf_files = list(save_directory.glob("*.gguf"))
    for gguf_file in gguf_files:
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_id,
        )

    print(f"Unsloth: GGUF pushed to https://huggingface.co/{repo_id}")
