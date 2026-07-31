# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Pooling and contrastive-loss primitives for embedding training on MLX.

LIBRARY only: pooling, losses, and a sentence-transformers layout shim. No
MLXTrainer integration -- the trainer's loss contract is
``loss_fn(model, batch, lengths, labels=None)``, one sequence per row, and
contrastive training needs two or three aligned sequences per row.

Only decoder-only backbones that mlx_lm implements are reachable. BERT, RoBERTa,
MiniLM and mpnet fail with ``Model type bert not supported`` -- the families most
sentence-transformers checkpoints use.

Plain HF layout (gte-Qwen2-1.5B-instruct) loads unmodified. sentence-transformers
layout (Qwen3-Embedding-0.6B) stores the transformer at the repo root, so keys are
``layers.0...`` while mlx_lm expects ``model.layers.0...`` and loading raw fails
with ``Received 310 parameters not in model``;
``remap_sentence_transformer_weights`` restores the prefix and the mode then comes
from ``1_Pooling/config.json``, mirroring sentence_transformer.py:587-599.

Mask handling is the correctness risk: a mean-pool over padding is silently wrong,
differing by ~26 on the test fixture with no error. Every mode takes the mask.
"""

# Bind MLX at import, NOT inside functions: a deferred `import mlx.core` resolves
# against sys.modules at call time, so tests/mlx_simulation's stub would win.
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

# Verbatim from sentence_transformer.py:587-599, so a checkpoint resolves to the
# same mode on MLX as on CUDA.
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
    """Reduce ``[batch, seq, hidden]`` to ``[batch, hidden]``, honouring the mask.

    Every mode must be invariant to trailing padding -- getting this wrong
    produces no error, only worse embeddings."""
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
        # -inf in pad slots so they cannot win the max
        masked = mx.where(
            weights > 0, hidden_states,
            mx.full(hidden_states.shape, -mx.inf).astype(hidden_states.dtype),
        )
        return masked.max(axis = 1)

    if mode == "mean_sqrt_len":
        lengths = mx.maximum(weights.sum(1), 1e-9)
        return (hidden_states * weights).sum(1) / mx.sqrt(lengths)

    if mode == "weightedmean":
        # weights 1..n over REAL tokens only
        positions = mx.cumsum(weights.squeeze(-1), axis = 1)[..., None] * weights
        return (hidden_states * positions).sum(1) / mx.maximum(positions.sum(1), 1e-9)

    # final REAL token, not the final slot
    last_index = mx.maximum(attention_mask.astype(mx.int32).sum(1) - 1, 0)
    gathered = mx.take_along_axis(
        hidden_states, last_index[:, None, None].astype(mx.int32), axis = 1,
    )
    return gathered.squeeze(1)


def multiple_negatives_ranking_loss(anchors, positives, scale = 20.0):
    """In-batch negatives: cross-entropy over the scaled cosine-similarity matrix.
    Difficulty grows with batch size; see ``recommend_batch_size``."""
    anchors = l2_normalize(anchors)
    positives = l2_normalize(positives)
    scores = (anchors @ positives.T) * scale
    labels = mx.arange(anchors.shape[0])
    return nn.losses.cross_entropy(scores, labels, reduction = "mean")


def cosent_loss(anchors, positives, labels, scale = 20.0):
    """CoSENT: pairwise ranking over cosine scores. Higher ``labels`` means a more
    similar pair; pairs that should not rank are masked to -inf."""
    cosine = (l2_normalize(anchors) * l2_normalize(positives)).sum(-1) * scale
    differences = cosine[None, :] - cosine[:, None]
    should_rank = (labels[:, None] > labels[None, :])
    differences = mx.where(
        should_rank, differences, mx.full(differences.shape, -mx.inf),
    )
    return mx.logaddexp(mx.zeros(()), mx.logsumexp(differences.reshape(-1)))


def triplet_loss(anchors, positives, negatives, margin = 0.5):
    """Explicit negatives with a margin, on squared L2 over normalized vectors."""
    anchors = l2_normalize(anchors)
    positives = l2_normalize(positives)
    negatives = l2_normalize(negatives)
    positive_distance = mx.sum((anchors - positives) ** 2, axis = -1)
    negative_distance = mx.sum((anchors - negatives) ** 2, axis = -1)
    return mx.maximum(positive_distance - negative_distance + margin, 0.0).mean()


def read_pooling_mode(pooling_config, default = "mean"):
    """Resolve ``1_Pooling/config.json`` to a mode, mirroring
    sentence_transformer.py:536-599. Qwen3-Embedding sets lasttoken, not mean."""
    if not pooling_config:
        return default
    for config_key, mode in SENTENCE_TRANSFORMERS_POOLING_MAP.items():
        if pooling_config.get(config_key):
            return mode
    return default


def is_sentence_transformers_layout(weight_keys):
    """True when the checkpoint stores the transformer at the repo root, without
    the ``model.`` prefix mlx_lm expects."""
    keys = list(weight_keys)
    if not keys:
        return False
    return not any(key.startswith("model.") for key in keys)


def remap_sentence_transformer_weights(weights, prefix = "model."):
    """Restore the ``model.`` prefix. Without it Qwen3-Embedding-0.6B fails with
    ``Received 310 parameters not in model``. No-op on plain HF layout."""
    if not is_sentence_transformers_layout(weights.keys()):
        return dict(weights)
    return {
        key if key.startswith((prefix, "lm_head")) else f"{prefix}{key}": value
        for key, value in weights.items()
    }


# Fitted on a Qwen3-0.6B + LoRA r8 backbone, seq_len 512, hidden 1024:
#     batch 4 -> 4.76 GB, 8 -> 7.92 GB, 16 -> 14.24 GB, i.e. 1.12 + 0.79 * batch
# (predicted 26.4 GB at batch 32 against 26.46 measured).
PEAK_GB_RESIDENT_DEFAULT = 1.12
PEAK_GB_PER_SAMPLE_AT_REFERENCE = 0.79
REFERENCE_SEQ_LEN = 512
REFERENCE_HIDDEN = 1024
MEMORY_SAFETY_FRACTION = 0.85


def recommend_batch_size(ram_gb, seq_len = REFERENCE_SEQ_LEN, hidden = REFERENCE_HIDDEN,
                         resident_gb = PEAK_GB_RESIDENT_DEFAULT):
    """Advisory ``(batch_size, warning)``; never raises.

    Advisory because oversubscription degrades into swap, not failure: batch 32 on
    a 16 GB machine completed at 26.5 GB peak but took 5x as long."""
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
    """Estimated peak GB for one contrastive step."""
    per_sample = (
        PEAK_GB_PER_SAMPLE_AT_REFERENCE
        * (seq_len / REFERENCE_SEQ_LEN)
        * (hidden / REFERENCE_HIDDEN)
    )
    return resident_gb + per_sample * batch_size

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
