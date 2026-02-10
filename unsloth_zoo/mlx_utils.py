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
import os
from pathlib import Path


def _dequantize_weight(layer):
    """Dequantize a QuantizedLinear or QuantizedEmbedding layer's weight."""
    return mx.dequantize(
        layer.weight, layer.scales,
        getattr(layer, "biases", None),
        group_size=getattr(layer, "group_size", 64),
        bits=getattr(layer, "bits", 4),
    )


def get_lm_weight(model):
    """Extract the language model head weight matrix.

    Checks for a separate lm_head first (untied models like Qwen), then
    falls back to embed_tokens (tied models like Gemma/Llama).
    Handles both quantized and unquantized layers.
    """
    # Check for separate lm_head (untied embeddings — e.g. Qwen2.5-7B+)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm_head = model.lm_head
        if hasattr(lm_head, "scales"):
            return _dequantize_weight(lm_head)
        if hasattr(lm_head, "weight"):
            return lm_head.weight

    # Fall back to embed_tokens (tied embeddings — e.g. Gemma, Llama)
    embed = model.model.embed_tokens
    if hasattr(embed, "scales"):
        return _dequantize_weight(embed)
    return embed.weight


def has_cce_kernel():
    """Check if mx.fast.cce_loss is available (requires mlx-cce)."""
    return hasattr(mx.fast, "cce_loss")


def _get_logit_softcap(model):
    """Get logit softcapping value if model uses it (e.g. Gemma-2), else 0.0."""
    softcap = getattr(model, "final_logit_softcapping", None)
    if softcap is None and hasattr(model, "args"):
        softcap = getattr(model.args, "final_logit_softcapping", None)
    return float(softcap) if softcap is not None and softcap > 0 else 0.0


def make_cce_loss_fn(model):
    """Create a CCE loss function using mx.fast.cce_loss from mlx-cce.

    CCE computes cross-entropy directly from hidden states and the LM head weight,
    avoiding full logit materialization. This saves significant memory.

    Weight is read from the model inside the loss function so that autograd
    traces it — the CCE kernel's VJP computes gradients for both hidden and
    weight automatically. No separate gradient computation needed.

    Supports logit softcapping (Gemma-2) and untied lm_head (Qwen-7B+).

    Args:
        model: MLX language model (used to detect softcap at setup time).

    Returns:
        A function (model, batch) -> scalar loss.
    """
    if not has_cce_kernel():
        raise RuntimeError(
            "mx.fast.cce_loss not available. Install mlx-cce: pip install mlx-cce"
        )

    softcap = _get_logit_softcap(model)
    if softcap > 0:
        print(f"Unsloth: CCE using logit_softcap={softcap} for this model.")

    def loss_fn(model, batch):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        hidden = model.model(inputs)
        # Read weight from model so autograd traces it —
        # CCE backward computes both d(loss)/d(hidden) and d(loss)/d(weight).
        weight = get_lm_weight(model)
        loss = mx.fast.cce_loss(
            hidden,
            weight,
            targets,
            logit_softcap=softcap,
        )
        return loss.astype(mx.float32).mean()

    return loss_fn


def make_baseline_loss_fn():
    """Create a standard cross-entropy loss function.

    Uses the full logit computation through the LM head, then applies
    nn.losses.cross_entropy. This is the fallback when CCE is not available.

    Returns:
        A function (model, batch) -> scalar loss.
    """
    def loss_fn(model, batch):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = nn.losses.cross_entropy(logits, targets)
        return loss.astype(mx.float32).mean()

    return loss_fn


def create_batches(dataset, tokenizer, batch_size, max_seq_length,
                   num_batches=None, seed=42, dataset_text_field="text",
                   formatting_func=None):
    """Pre-tokenize and batch a HuggingFace dataset for MLX training.

    Each batch is a single mx.array of shape (batch_size, max_seq_length).
    Sequences shorter than max_seq_length are padded with eos_token_id.

    Args:
        dataset: HuggingFace dataset or list of strings.
        tokenizer: Tokenizer compatible with the model.
        batch_size: Number of sequences per batch.
        max_seq_length: Sequence length (pad/truncate to this).
        num_batches: If set, only create this many batches.
        seed: Random seed for shuffling.
        dataset_text_field: Column name for text data.
        formatting_func: Optional function to format dataset items to text.

    Returns:
        List of mx.array, each of shape (batch_size, max_seq_length).
    """
    # Collect texts
    texts = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            if isinstance(result, list):
                texts.extend(result)
            else:
                texts.append(result)
        elif isinstance(item, dict):
            if dataset_text_field in item:
                texts.append(item[dataset_text_field])
            else:
                # Try common column names
                for key in ("text", "content", "instruction"):
                    if key in item:
                        texts.append(item[key])
                        break
        elif isinstance(item, str):
            texts.append(item)

    if not texts:
        raise ValueError(
            f"No text data found. Provide a dataset with a '{dataset_text_field}' column."
        )

    # Shuffle deterministically
    mx.random.seed(seed)
    indices = mx.random.permutation(len(texts)).tolist()

    pad_id = tokenizer.eos_token_id or 0

    batches = []
    for b_start in range(0, len(indices), batch_size):
        if num_batches is not None and len(batches) >= num_batches:
            break

        batch_indices = indices[b_start:b_start + batch_size]
        if len(batch_indices) < batch_size:
            # Wrap around to fill incomplete batch
            batch_indices += indices[:batch_size - len(batch_indices)]

        batch_tokens = []
        for idx in batch_indices:
            tokens = tokenizer.encode(texts[idx])
            if len(tokens) >= max_seq_length:
                tokens = tokens[:max_seq_length]
            else:
                tokens = tokens + [pad_id] * (max_seq_length - len(tokens))
            batch_tokens.append(tokens)

        batches.append(mx.array(batch_tokens, dtype=mx.int32))

    mx.eval(batches)
    return batches


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
