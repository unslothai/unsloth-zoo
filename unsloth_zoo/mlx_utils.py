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

Provides loss functions (CCE via mlx-cce, baseline CE), data batching,
weight extraction helpers, and model save/load for LoRA adapters.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import json
from pathlib import Path


def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to all transformer layers.

    Patches the layer class's __call__ to use mx.checkpoint, which
    recomputes activations during backward instead of storing them.
    Trades ~30% more compute for significant memory savings.

    Follows the same pattern as mlx_lm.tuner.trainer.grad_checkpoint.
    """
    layers = getattr(model, 'layers', None)
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
    layers = getattr(model, 'layers', None)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    orig = getattr(layer_cls, '_orig_call', None)
    if orig is not None:
        layer_cls.__call__ = orig
        del layer_cls._orig_call


def _get_lm_head_layer(model):
    """Get the raw LM head layer (QuantizedLinear or Linear/Embedding).

    Checks for a separate lm_head first (untied models like Qwen), then
    falls back to embed_tokens (tied models like Gemma/Llama).

    Returns the layer object (not its weight), so callers can access
    .weight, .scales, .biases, .group_size, .bits for quantized layers.
    """
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    return model.model.embed_tokens


def _is_quantized_layer(layer):
    """Check if a layer has quantized weights (has .scales attribute)."""
    return hasattr(layer, "scales")


def has_cce_kernel():
    """Check if mx.fast.cce_loss is available (requires mlx-cce)."""
    return hasattr(mx.fast, "cce_loss")


def _get_logit_softcap(model):
    """Get logit softcapping value if model uses it (e.g. Gemma-2), else 0.0."""
    softcap = getattr(model, "final_logit_softcapping", None)
    if softcap is None and hasattr(model, "args"):
        softcap = getattr(model.args, "final_logit_softcapping", None)
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
    """Create a CCE loss function using mx.fast.cce_loss from mlx-cce.

    CCE computes cross-entropy directly from hidden states and the LM head weight,
    avoiding full logit materialization. This saves significant memory for large
    vocabularies. For V=256K+ models, CCE is both faster and more memory-efficient.

    If the LM head is quantized, passes raw uint32 weight + scales + biases
    directly to cce_loss, using fused quantized matmul kernels.

    Returns:
        A function (model, batch, lengths) -> (loss, ntoks).
    """
    if not has_cce_kernel():
        raise RuntimeError(
            "mx.fast.cce_loss not available. Install mlx-cce: pip install mlx-cce"
        )

    softcap = _get_logit_softcap(model)
    if softcap > 0:
        print(f"Unsloth: CCE using logit_softcap={softcap} for this model.")

    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)

    if use_quantized:
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        print(f"Unsloth: CCE using quantized matmul (group_size={group_size}, bits={bits})")
        _has_lm_head_q = (hasattr(model, "lm_head")
                          and model.lm_head is not None
                          and hasattr(model.lm_head, "scales"))
        _has_biases = hasattr(lm_layer, "biases")

        def loss_fn(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(inputs)
            layer = model.lm_head if _has_lm_head_q else model.model.embed_tokens
            w = layer.weight
            sc = layer.scales
            bi = layer.biases if _has_biases else None
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            loss = mx.fast.cce_loss(
                hidden, w, masked_targets,
                scales=sc, biases=bi,
                group_size=group_size, bits=bits,
                ignore_index=-100,
                logit_softcap=softcap,
            )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks
    else:
        _has_lm_head = (hasattr(model, "lm_head")
                        and model.lm_head is not None
                        and hasattr(model.lm_head, "weight"))
        _skip_weight_grad = not _is_lm_head_trainable(model)
        if _skip_weight_grad:
            print("Unsloth: CCE skipping weight gradient (LM head is frozen).")

        def loss_fn(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(inputs)
            w = model.lm_head.weight if _has_lm_head else model.model.embed_tokens.weight
            if _skip_weight_grad:
                w = mx.stop_gradient(w)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            loss = mx.fast.cce_loss(
                hidden, w, masked_targets,
                ignore_index=-100,
                logit_softcap=softcap,
            )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks

    return loss_fn


def make_baseline_loss_fn():
    """Create a standard cross-entropy loss function.

    Uses the full logit computation through the LM head, then applies
    nn.losses.cross_entropy. This is the fallback when CCE is not available.

    Returns:
        A function (model, batch, lengths) -> (loss, ntoks).
    """
    def loss_fn(model, batch, lengths):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn


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
        batch_pairs.append((batch, lengths_info))
        if num_batches is not None and len(batch_pairs) >= num_batches:
            break

    mx.eval([b for b, _ in batch_pairs] + [l for _, l in batch_pairs])
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
        yield batch, lengths_info


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


def save_merged_model(model, tokenizer, path):
    """Fuse LoRA weights and save the full merged model.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        path: Directory to save merged model.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Fuse LoRA weights
    model.eval()
    de_lora_model = nn.utils.fuse_lora(model)

    # Save all weights — flatten nested dict for safetensors
    weights = dict(mlx.utils.tree_flatten(de_lora_model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)

    # Save tokenizer
    tokenizer.save_pretrained(str(path))
