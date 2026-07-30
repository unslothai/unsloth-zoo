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

"""Pooling correctness and contrastive-loss behaviour on MLX.

Two things are asserted that a weaker suite would miss.

Padding invariance: a pool that averages over padding produces no error, only
worse embeddings. The fixture pads with junk at 100x scale so a mask-ignoring
mean differs by ~26 -- that difference is asserted directly, so the test cannot
pass against a broken implementation.

Separation, not loss: a falling loss does not show the objective is doing what it
claims. Each loss also asserts that positive/negative similarity SEPARATION rises.
Note the synthetic encoder starts collapsed (all pairs ~0.995 cosine), so
"positive similarity rises" is unreachable from that ceiling and would be the
wrong assertion; separation is the metric that discriminates.

Real MLX required, so this skips on Linux; the shim-runnable half is in
test_mlx_embedding_guards.py. Synthetic tensors only -- no weights, no network.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is only available on Apple Silicon")
nn = pytest.importorskip("mlx.nn", reason="MLX is only available on Apple Silicon")
optim = pytest.importorskip("mlx.optimizers", reason="MLX is only available on Apple Silicon")

# Hoisted to module level on purpose. Inside a test body these resolve against
# sys.modules at CALL time, and tests/mlx_simulation replaces mlx.* session-wide,
# so a deferred import silently yields the stub instead of real MLX.
from mlx.utils import tree_flatten
_mlx_lm = pytest.importorskip("mlx_lm", reason="MLX is only available on Apple Silicon")
from mlx_lm.models import qwen3 as _qwen3

from unsloth_zoo.mlx.embedding import (
    POOLING_MODES,
    cosent_loss,
    l2_normalize,
    multiple_negatives_ranking_loss,
    pool,
    triplet_loss,
)


BATCH, SEQ, HIDDEN, VOCAB = 8, 12, 32, 256
STEPS = 8


# --- pooling ---------------------------------------------------------------
def _padded_fixture():
    """Same content twice: once unpadded, once padded with extreme junk."""
    mx.random.seed(0)
    rows, length, width, pad = 3, 8, 5, 4
    hidden = mx.random.normal((rows, length, width))
    mask = mx.ones((rows, length))
    junk = mx.random.normal((rows, pad, width)) * 100.0
    hidden_padded = mx.concatenate([hidden, junk], axis=1)
    mask_padded = mx.concatenate([mx.ones((rows, length)), mx.zeros((rows, pad))], axis=1)
    return hidden, mask, hidden_padded, mask_padded


@pytest.mark.parametrize("mode", POOLING_MODES)
def test_pooling_is_invariant_to_padding(mode):
    hidden, mask, hidden_padded, mask_padded = _padded_fixture()

    unpadded = pool(hidden, mask, mode)
    padded = pool(hidden_padded, mask_padded, mode)
    mx.eval(unpadded, padded)

    assert unpadded.shape == padded.shape == (3, 5)
    worst = float(mx.abs(unpadded - padded).max())
    assert worst < 1e-4, (
        f"{mode} pooling changed by {worst:.3e} when trailing padding was added; "
        "the attention mask is being ignored"
    )


def test_mask_ignoring_mean_is_measurably_wrong():
    """Discriminating counterexample: the fixture must be able to fail."""
    _, _, hidden_padded, mask_padded = _padded_fixture()

    naive = hidden_padded.mean(axis=1)                       # ignores the mask
    correct = pool(hidden_padded, mask_padded, "mean")
    difference = float(mx.abs(naive - correct).max())

    assert difference > 1.0, (
        f"fixture no longer discriminates: a mask-ignoring mean differs by only "
        f"{difference:.3e}, so the invariance tests above would pass while broken"
    )


def test_ragged_batch_mean_matches_hand_computation():
    mx.random.seed(0)
    hidden = mx.random.normal((3, 8, 5))
    lengths = mx.array([8, 5, 2])
    mask = (mx.arange(8)[None, :] < lengths[:, None]).astype(mx.float32)

    pooled = pool(hidden, mask, "mean")
    manual = hidden[1, :5, :].mean(axis=0)                   # row 1 has 5 real tokens

    assert float(mx.abs(pooled[1] - manual).max()) < 1e-5


def test_ragged_batch_lasttoken_uses_last_real_token():
    """Row 1 has 5 real tokens, so lasttoken must return position 4, not 7."""
    mx.random.seed(0)
    hidden = mx.random.normal((3, 8, 5))
    lengths = mx.array([8, 5, 2])
    mask = (mx.arange(8)[None, :] < lengths[:, None]).astype(mx.float32)

    pooled = pool(hidden, mask, "lasttoken")

    assert float(mx.abs(pooled[1] - hidden[1, 4, :]).max()) < 1e-6
    assert float(mx.abs(pooled[1] - hidden[1, 7, :]).max()) > 1e-3, (
        "lasttoken returned the final SLOT rather than the final real token"
    )
    assert float(mx.abs(pooled[2] - hidden[2, 1, :]).max()) < 1e-6


def test_unknown_pooling_mode_is_rejected():
    hidden, mask, _, _ = _padded_fixture()
    with pytest.raises(ValueError, match="unknown pooling mode"):
        pool(hidden, mask, "definitely_not_a_mode")


# --- losses ----------------------------------------------------------------
class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.first = nn.Linear(HIDDEN, HIDDEN)
        self.second = nn.Linear(HIDDEN, HIDDEN)

    def __call__(self, ids, mask):
        hidden = self.second(nn.gelu(self.first(self.emb(ids))))
        return pool(hidden, mask, "mean")


def _loss_fixture():
    mx.random.seed(0)
    anchors = mx.random.randint(0, VOCAB, (BATCH, SEQ))
    positives = mx.random.randint(0, VOCAB, (BATCH, SEQ))
    negatives = mx.random.randint(0, VOCAB, (BATCH, SEQ))
    return anchors, positives, negatives, mx.ones((BATCH, SEQ))


def _new_encoder():
    mx.random.seed(0)
    encoder = _Encoder()
    mx.eval(encoder.parameters())
    return encoder


def _train(encoder, loss_fn, steps=STEPS, lr=1e-3):
    anchors, positives, negatives, mask = _loss_fixture()
    grad_fn = nn.value_and_grad(encoder, loss_fn)
    opt = optim.Adam(learning_rate=lr)
    losses = []
    for _ in range(steps):
        loss, grads = grad_fn(encoder, anchors, positives, negatives, mask)
        opt.update(encoder, grads)
        mx.eval(encoder.parameters(), opt.state)
        losses.append(float(loss))
    return losses


def _inbatch_separation(encoder):
    anchors, positives, _, mask = _loss_fixture()
    a, p = l2_normalize(encoder(anchors, mask)), l2_normalize(encoder(positives, mask))
    scores = a @ p.T
    positive = float(mx.diag(scores).mean())
    negative = float((scores.sum() - mx.diag(scores).sum()) / (BATCH * BATCH - BATCH))
    return positive - negative


def test_multiple_negatives_ranking_separates_pairs():
    encoder = _new_encoder()
    before = _inbatch_separation(encoder)

    losses = _train(encoder, lambda m, a, p, n, k: multiple_negatives_ranking_loss(m(a, k), m(p, k)))

    after = _inbatch_separation(encoder)
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), losses
    assert after > before, f"separation did not rise: {before:+.4f} -> {after:+.4f}"
    assert after > 0.15, f"separation only reached {after:+.4f} after {STEPS} steps"


COSENT_LABELS = mx.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def _cosent_groups(encoder):
    anchors, positives, _, mask = _loss_fixture()
    cosine = (l2_normalize(encoder(anchors, mask)) * l2_normalize(encoder(positives, mask))).sum(-1)
    similar = float((cosine * COSENT_LABELS).sum() / COSENT_LABELS.sum())
    dissimilar = float((cosine * (1 - COSENT_LABELS)).sum() / (1 - COSENT_LABELS).sum())
    return similar, dissimilar


def test_cosent_raises_similar_and_lowers_dissimilar():
    """The cleanest evidence available: the two labelled groups move apart."""
    encoder = _new_encoder()
    similar_before, dissimilar_before = _cosent_groups(encoder)

    losses = _train(encoder, lambda m, a, p, n, k: cosent_loss(m(a, k), m(p, k), COSENT_LABELS))

    similar_after, dissimilar_after = _cosent_groups(encoder)
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), losses
    assert similar_after > similar_before, (
        f"labelled-similar cosine fell: {similar_before:+.4f} -> {similar_after:+.4f}"
    )
    assert dissimilar_after < dissimilar_before, (
        f"labelled-dissimilar cosine rose: {dissimilar_before:+.4f} -> {dissimilar_after:+.4f}"
    )
    separation = (similar_after - dissimilar_after) - (similar_before - dissimilar_before)
    assert separation > 0.3, f"separation only improved by {separation:+.4f}"


def _triplet_separation(encoder):
    anchors, positives, negatives, mask = _loss_fixture()
    a = l2_normalize(encoder(anchors, mask))
    p = l2_normalize(encoder(positives, mask))
    n = l2_normalize(encoder(negatives, mask))
    return float((a * p).sum(-1).mean()) - float((a * n).sum(-1).mean())


def test_triplet_separates_positive_from_explicit_negative():
    encoder = _new_encoder()
    before = _triplet_separation(encoder)

    losses = _train(encoder, lambda m, a, p, n, k: triplet_loss(m(a, k), m(p, k), m(n, k)))

    after = _triplet_separation(encoder)
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), losses
    assert after > before, f"separation did not rise: {before:+.4f} -> {after:+.4f}"
    assert after > 0.2, f"separation only reached {after:+.4f}"


def test_identical_inputs_give_no_separation():
    """Sanity floor: anchors distilled against themselves cannot separate."""
    encoder = _new_encoder()
    anchors, _, _, mask = _loss_fixture()
    embedded = l2_normalize(encoder(anchors, mask))
    scores = embedded @ embedded.T
    assert abs(float(mx.diag(scores).mean()) - 1.0) < 1e-4


# --- sentence-transformers layout, against a real checkpoint ----------------
ST_REPO = "Qwen/Qwen3-Embedding-0.6B"


def _cached_st_checkpoint():
    """Resolve the cached repo WITHOUT downloading. Skips if it is not present."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        path = huggingface_hub.snapshot_download(ST_REPO, local_files_only=True)
    except Exception as exception:                # not cached on this machine
        pytest.skip(f"{ST_REPO} is not in the local HF cache: {type(exception).__name__}")
    import os
    weights_path = os.path.join(path, "model.safetensors")
    if not os.path.exists(weights_path):
        pytest.skip(f"{ST_REPO} cache has no model.safetensors")
    return path, weights_path


def test_real_st_checkpoint_needs_the_remap_and_loads_after_it():
    """End to end on Qwen3-Embedding-0.6B: without the shim mlx_lm reports
    'Received 310 parameters not in model'; with it the weights load and match."""
    import json
    import os

    from unsloth_zoo.mlx.embedding import (
        is_sentence_transformers_layout,
        remap_sentence_transformer_weights,
    )
    qwen3 = _qwen3

    path, weights_path = _cached_st_checkpoint()
    weights = mx.load(weights_path)

    # The checkpoint really is in sentence-transformers layout.
    assert is_sentence_transformers_layout(weights.keys())
    assert not any(k.startswith("model.") for k in weights)

    config = json.load(open(os.path.join(path, "config.json")))
    model = qwen3.Model(qwen3.ModelArgs.from_dict(config))
    expected = {name for name, _ in tree_flatten(model.parameters())}
    # Same count, zero overlap: exactly the prefix problem, not a real mismatch.
    assert len(weights) == len(expected)
    assert not (set(weights) & expected)

    remapped = remap_sentence_transformer_weights(weights)
    model.load_weights(list(remapped.items()), strict=False)
    mx.eval(model.parameters())

    loaded = dict(tree_flatten(model.parameters()))
    probe = "model.layers.0.self_attn.q_proj.weight"
    assert bool(mx.array_equal(loaded[probe], weights["layers.0.self_attn.q_proj.weight"]).item()), (
        "remapped weights loaded but do not match the checkpoint tensor"
    )

    hidden = model.model(mx.random.randint(0, 500, (1, 8)))
    mx.eval(hidden)
    assert hidden.shape == (1, 8, config["hidden_size"])


def test_real_st_checkpoint_declares_lasttoken_pooling():
    """Reading 1_Pooling is not optional: this model is not mean-pooled."""
    import json
    import os

    from unsloth_zoo.mlx.embedding import read_pooling_mode

    path, _ = _cached_st_checkpoint()
    pooling_config_path = os.path.join(path, "1_Pooling", "config.json")
    if not os.path.exists(pooling_config_path):
        pytest.skip("cached snapshot has no 1_Pooling/config.json")

    assert read_pooling_mode(json.load(open(pooling_config_path))) == "lasttoken"
