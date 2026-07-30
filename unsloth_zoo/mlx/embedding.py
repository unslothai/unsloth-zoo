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

This is the LIBRARY half only: pooling, losses, and a sentence-transformers
layout shim. It deliberately does not integrate with MLXTrainer. The trainer's
loss contract is ``loss_fn(model, batch, lengths, labels=None)`` -- one token
sequence per row -- and contrastive training needs two or three aligned sequences
per row (anchor / positive / negative). Carrying a pair through that signature is
not possible without a separate entry point and data-path work, so integration is
a separate change.

Scope, stated plainly because the feature name over-promises: only decoder-only
backbones that ``mlx_lm`` already implements are reachable. BERT, RoBERTa, MiniLM
and mpnet fail with ``ValueError: Model type bert not supported`` -- there is no
BERT in mlx_lm at all -- and those are the families most sentence-transformers
checkpoints actually use.

Two layouts are handled:

* Plain HF layout (e.g. Alibaba-NLP/gte-Qwen2-1.5B-instruct) loads unmodified.
* sentence-transformers layout (e.g. Qwen/Qwen3-Embedding-0.6B) stores the inner
  transformer at the repo root, so its keys are ``layers.0.self_attn.q_proj.weight``
  while mlx_lm expects ``model.layers.0...``. Loading it raw fails with
  ``ValueError: Received 310 parameters not in model``. ``remap_sentence_transformer_weights``
  restores the prefix; the pooling mode then comes from ``1_Pooling/config.json``,
  mirroring the map in unsloth/models/sentence_transformer.py:587-599.

Mask handling is the correctness risk here. A mean-pool that averages over padding
is silently wrong rather than loud: on a padded batch it differs from the correct
answer by ~26 in the test fixture, with no error anywhere. Every mode below takes
the attention mask.
"""

# Bind MLX at import, NOT inside functions. A deferred `import mlx.core` resolves
# against sys.modules at call time, so anything that swaps in a stub afterwards
# (tests/mlx_simulation does exactly that, session-wide) would silently hand this
# module the stub. Binding here captures whichever MLX was present at first import.
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

# Verbatim from unsloth/models/sentence_transformer.py:587-599, which reads these
# keys out of 1_Pooling/config.json. Kept identical so a checkpoint resolves to the
# same mode on MLX as it does on CUDA.
SENTENCE_TRANSFORMERS_POOLING_MAP = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}


def l2_normalize(x, eps = 1e-12):
    """Unit-norm along the last axis, guarded against a zero vector."""
    return x / mx.maximum(mx.linalg.norm(x, axis = -1, keepdims = True), eps)


def pool(hidden_states, attention_mask, mode = "mean"):
    """Reduce ``[batch, seq, hidden]`` to ``[batch, hidden]``, honouring the mask.

    ``attention_mask`` is ``[batch, seq]`` with 1 for real tokens and 0 for
    padding. Every mode must be invariant to trailing padding: pooling a padded
    batch has to equal pooling the unpadded one, whatever junk sits in the pad
    slots. Getting this wrong produces no error, only worse embeddings.
    """
    if mode not in POOLING_MODES:
        raise ValueError(
            f"Unsloth: unknown pooling mode {mode!r}. Supported: {', '.join(POOLING_MODES)}."
        )
    weights = attention_mask.astype(hidden_states.dtype)[..., None]

    if mode == "cls":
        # Position 0 is the CLS token and is never padding.
        return hidden_states[:, 0, :]

    if mode == "mean":
        return (hidden_states * weights).sum(1) / mx.maximum(weights.sum(1), 1e-9)

    if mode == "max":
        # -inf in the pad slots so they can never win the max.
        masked = mx.where(
            weights > 0, hidden_states,
            mx.full(hidden_states.shape, -mx.inf).astype(hidden_states.dtype),
        )
        return masked.max(axis = 1)

    if mode == "mean_sqrt_len":
        lengths = mx.maximum(weights.sum(1), 1e-9)
        return (hidden_states * weights).sum(1) / mx.sqrt(lengths)

    if mode == "weightedmean":
        # Position weights 1..n over REAL tokens only, so padding cannot shift the
        # weighting of the tokens that precede it.
        positions = mx.cumsum(weights.squeeze(-1), axis = 1)[..., None] * weights
        return (hidden_states * positions).sum(1) / mx.maximum(positions.sum(1), 1e-9)

    # lasttoken: the final REAL token, not the final slot.
    last_index = mx.maximum(attention_mask.astype(mx.int32).sum(1) - 1, 0)
    gathered = mx.take_along_axis(
        hidden_states, last_index[:, None, None].astype(mx.int32), axis = 1,
    )
    return gathered.squeeze(1)


def multiple_negatives_ranking_loss(anchors, positives, scale = 20.0):
    """In-batch negatives: row i's positive is column i, every other column is a
    negative. Cross-entropy over the scaled cosine-similarity matrix.

    Effective difficulty grows with batch size, since the number of negatives is
    ``batch - 1``. See ``recommend_batch_size`` before pushing it up.
    """
    anchors = l2_normalize(anchors)
    positives = l2_normalize(positives)
    scores = (anchors @ positives.T) * scale
    labels = mx.arange(anchors.shape[0])
    return nn.losses.cross_entropy(scores, labels, reduction = "mean")


def cosent_loss(anchors, positives, labels, scale = 20.0):
    """CoSENT: pairwise ranking over cosine scores.

    ``labels`` is ``[batch]`` with a higher value meaning a more similar pair
    (1/0 for the binary case). Every ordered pair where i should outrank j
    contributes; pairs that should not are masked to -inf so they drop out of the
    logsumexp.
    """
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
    """Resolve a sentence-transformers ``1_Pooling/config.json`` dict to a mode.

    Mirrors unsloth/models/sentence_transformer.py:536-599 including its map and
    its fall-back to "mean" when nothing is set. Qwen3-Embedding-0.6B sets
    ``pooling_mode_lasttoken``, not mean, so defaulting blindly would pool the
    wrong thing.
    """
    if not pooling_config:
        return default
    for config_key, mode in SENTENCE_TRANSFORMERS_POOLING_MAP.items():
        if pooling_config.get(config_key):
            return mode
    return default


def is_sentence_transformers_layout(weight_keys):
    """True when the checkpoint stores the transformer at the repo root.

    sentence-transformers writes the inner Transformer module with ``path: ""``,
    stripping the ``model.`` prefix mlx_lm's decoder classes expect.
    """
    keys = list(weight_keys)
    if not keys:
        return False
    return not any(key.startswith("model.") for key in keys)


def remap_sentence_transformer_weights(weights, prefix = "model."):
    """Restore the ``model.`` prefix on a sentence-transformers-layout checkpoint.

    Loading Qwen/Qwen3-Embedding-0.6B without this fails with
    ``ValueError: Received 310 parameters not in model`` -- 310 checkpoint keys,
    310 expected keys, none of them matching because every one is missing the
    prefix. Keys already prefixed, and top-level heads like ``lm_head``, are left
    alone. A no-op on plain HF layout.
    """
    if not is_sentence_transformers_layout(weights.keys()):
        return dict(weights)
    return {
        key if key.startswith((prefix, "lm_head")) else f"{prefix}{key}": value
        for key, value in weights.items()
    }


# Peak memory for a contrastive step, fitted on measured runs of a Qwen3-0.6B
# backbone with LoRA r8 on 8 layers, seq_len 512, hidden 1024:
#     batch  4 -> 4.76 GB, batch 8 -> 7.92 GB, batch 16 -> 14.24 GB
# giving peak_GB ~= 1.12 + 0.79 * batch. Reproduces those three to ~0.15 GB and
# predicted 26.4 GB at batch 32 against 26.46 GB measured.
PEAK_GB_RESIDENT_DEFAULT = 1.12
PEAK_GB_PER_SAMPLE_AT_REFERENCE = 0.79
REFERENCE_SEQ_LEN = 512
REFERENCE_HIDDEN = 1024
MEMORY_SAFETY_FRACTION = 0.85


def recommend_batch_size(ram_gb, seq_len = REFERENCE_SEQ_LEN, hidden = REFERENCE_HIDDEN,
                         resident_gb = PEAK_GB_RESIDENT_DEFAULT):
    """Advisory batch size for in-batch-negative training. Never raises.

    Returns ``(batch_size, warning)`` where ``warning`` is "" when the estimate is
    comfortable. Advisory rather than enforcing because oversubscription here
    degrades into swap rather than failing: a measured batch of 32 on a 16 GB
    machine completed at 26.5 GB peak but took 5x as long. (Only much further out
    does the process get killed outright.) A hard refusal would be wrong for a
    cost that is slowness, not breakage.
    """
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
    """Estimated peak GB for one contrastive step. Inverse of recommend_batch_size."""
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
