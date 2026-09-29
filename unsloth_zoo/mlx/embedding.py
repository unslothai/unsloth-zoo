# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Pooling, contrastive losses and a sentence-transformers layout shim for MLX embedding training."""

# Bind MLX at import, not inside functions, so tests/mlx_simulation stubs cannot win.
import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "POOLING_MODES",
    "SENTENCE_TRANSFORMERS_POOLING_MAP",
    "pool",
    "l2_normalize",
    "multiple_negatives_ranking_loss",
    "cosent_loss",
    "triplet_loss",
    "read_pooling_mode",
    "remap_sentence_transformer_weights",
    "recommend_batch_size",
    "PEAK_GB_RESIDENT_DEFAULT",
    "PEAK_GB_PER_SAMPLE_AT_REFERENCE",
    "REFERENCE_SEQ_LEN",
    "REFERENCE_HIDDEN",
    "MEMORY_SAFETY_FRACTION",
]

POOLING_MODES = ("cls", "mean", "max", "mean_sqrt_len", "weightedmean", "lasttoken")

# Verbatim from sentence_transformer.py:587-599.
SENTENCE_TRANSFORMERS_POOLING_MAP = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}


def l2_normalize(x, eps = 1e-12):
    """Unit-norm along the last axis."""
    return x / mx.maximum(mx.linalg.norm(x, axis = -1, keepdims = True), eps)


def pool(hidden_states, attention_mask, mode = "mean"):
    """Reduce ``[batch, seq, hidden]`` to ``[batch, hidden]``; must ignore padding."""
    if mode not in POOLING_MODES:
        raise ValueError(
            f"Unsloth: unknown pooling mode {mode!r}. Supported: {', '.join(POOLING_MODES)}."
        )
    weights = attention_mask.astype(hidden_states.dtype)[..., None]

    if mode == "cls":
        return hidden_states[:, 0, :]

    if mode == "mean":
        return (hidden_states * weights).sum(1) / mx.maximum(weights.sum(1), 1e-9)

    if mode == "max":
        masked = mx.where(
            weights > 0, hidden_states,
            mx.full(hidden_states.shape, -mx.inf).astype(hidden_states.dtype),
        )
        return masked.max(axis = 1)

    if mode == "mean_sqrt_len":
        lengths = mx.maximum(weights.sum(1), 1e-9)
        return (hidden_states * weights).sum(1) / mx.sqrt(lengths)

    if mode == "weightedmean":
        positions = mx.cumsum(weights.squeeze(-1), axis = 1)[..., None] * weights
        return (hidden_states * positions).sum(1) / mx.maximum(positions.sum(1), 1e-9)

    last_index = mx.maximum(attention_mask.astype(mx.int32).sum(1) - 1, 0)
    gathered = mx.take_along_axis(
        hidden_states, last_index[:, None, None].astype(mx.int32), axis = 1,
    )
    return gathered.squeeze(1)


def multiple_negatives_ranking_loss(anchors, positives, scale = 20.0):
    """In-batch negatives: cross-entropy over the scaled cosine-similarity matrix."""
    anchors = l2_normalize(anchors)
    positives = l2_normalize(positives)
    scores = (anchors @ positives.T) * scale
    labels = mx.arange(anchors.shape[0])
    return nn.losses.cross_entropy(scores, labels, reduction = "mean")


def cosent_loss(anchors, positives, labels, scale = 20.0):
    """CoSENT pairwise ranking loss; higher ``labels`` means more similar."""
    cosine = (l2_normalize(anchors) * l2_normalize(positives)).sum(-1) * scale
    differences = cosine[None, :] - cosine[:, None]
    should_rank = (labels[:, None] > labels[None, :])
    differences = mx.where(
        should_rank, differences, mx.full(differences.shape, -mx.inf),
    )
    return mx.logaddexp(mx.zeros(()), mx.logsumexp(differences.reshape(-1)))


def triplet_loss(anchors, positives, negatives, margin = 0.5):
    anchors = l2_normalize(anchors)
    positives = l2_normalize(positives)
    negatives = l2_normalize(negatives)
    positive_distance = mx.sum((anchors - positives) ** 2, axis = -1)
    negative_distance = mx.sum((anchors - negatives) ** 2, axis = -1)
    return mx.maximum(positive_distance - negative_distance + margin, 0.0).mean()


def read_pooling_mode(pooling_config, default = "mean"):
    """Resolve ``1_Pooling/config.json`` to a mode (sentence_transformer.py:536-599)."""
    if not pooling_config:
        return default
    for config_key, mode in SENTENCE_TRANSFORMERS_POOLING_MAP.items():
        if pooling_config.get(config_key):
            return mode
    return default


def is_sentence_transformers_layout(weight_keys):
    keys = list(weight_keys)
    if not keys:
        return False
    return not any(key.startswith("model.") for key in keys)


def remap_sentence_transformer_weights(weights, prefix = "model."):
    """Restore the ``model.`` prefix mlx_lm expects; no-op on plain HF layout."""
    if not is_sentence_transformers_layout(weights.keys()):
        return dict(weights)
    return {
        key if key.startswith((prefix, "lm_head")) else f"{prefix}{key}": value
        for key, value in weights.items()
    }


# Linear peak-GB fit on Qwen3-0.6B + LoRA r8 at seq_len 512, hidden 1024.
PEAK_GB_RESIDENT_DEFAULT = 1.12
PEAK_GB_PER_SAMPLE_AT_REFERENCE = 0.79
REFERENCE_SEQ_LEN = 512
REFERENCE_HIDDEN = 1024
MEMORY_SAFETY_FRACTION = 0.85


def recommend_batch_size(ram_gb, seq_len = REFERENCE_SEQ_LEN, hidden = REFERENCE_HIDDEN,
                         resident_gb = PEAK_GB_RESIDENT_DEFAULT):
    """Advisory ``(batch_size, warning)``; never raises (oversubscription swaps, not fails)."""
    per_sample = (
        PEAK_GB_PER_SAMPLE_AT_REFERENCE
        * (seq_len / REFERENCE_SEQ_LEN)
        * (hidden / REFERENCE_HIDDEN)
    )
    budget = ram_gb * MEMORY_SAFETY_FRACTION
    usable = budget - resident_gb
    if usable <= 0 or per_sample <= 0:
        return 1, (
            f"Unsloth: {ram_gb:.0f} GB leaves no headroom after a {resident_gb:.2f} GB "
            "resident model; batch_size=1 is the most that can be recommended."
        )
    batch_size = max(1, int(usable / per_sample))
    warning = ""
    if batch_size < 8:
        warning = (
            f"Unsloth: only batch_size={batch_size} fits in {ram_gb:.0f} GB at "
            f"seq_len={seq_len}. In-batch-negative losses get their difficulty from "
            "batch size, so small batches train weaker embeddings; consider a shorter "
            "seq_len or a smaller backbone."
        )
    return batch_size, warning


def estimate_peak_gb(batch_size, seq_len = REFERENCE_SEQ_LEN, hidden = REFERENCE_HIDDEN,
                     resident_gb = PEAK_GB_RESIDENT_DEFAULT):
    per_sample = (
        PEAK_GB_PER_SAMPLE_AT_REFERENCE
        * (seq_len / REFERENCE_SEQ_LEN)
        * (hidden / REFERENCE_HIDDEN)
    )
    return resident_gb + per_sample * batch_size
