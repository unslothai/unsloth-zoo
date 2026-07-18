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

"""Pure-logic regression for the MLX KTO loss primitives.

Covers the unpaired KTO loss (`_kto_loss`), the batch KL baseline clamp
(`_kto_kl_baseline`), the summed-logp extractor (`_kto_sum_logp`), and the
batch-size guard in `_build_kto_batches`. Runs under the torch shim so Linux
CI collection covers it without MLX/Metal (all four primitives are backend-
agnostic MLX-array math).

Reference values are the ones proven in the KTO Step 1 investigation, where
`_kto_loss` matched a torch reimplementation of TRL's kto_loss and a pure-numpy
hand computation to ~1e-8 across desirable/undesirable weight settings.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# --------------------------------------------------------------------------
# Fixed inputs + independent numpy reference (the KTO Step 1 fixtures).
# --------------------------------------------------------------------------
def _fixed_logps():
    rng = np.random.default_rng(0)
    return dict(
        pol_ch=rng.normal(-8, 2, size=3).astype(np.float32),
        pol_rej=rng.normal(-9, 2, size=2).astype(np.float32),
        pol_kl=rng.normal(-8.5, 2, size=5).astype(np.float32),
        ref_ch=rng.normal(-8, 2, size=3).astype(np.float32),
        ref_rej=rng.normal(-9, 2, size=2).astype(np.float32),
        ref_kl=rng.normal(-8.5, 2, size=5).astype(np.float32),
    )


def _numpy_kto_loss(pol_ch, pol_rej, ref_ch, ref_rej, kl, beta, wd, wu):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    ch = wd * (1 - sig(beta * ((pol_ch - ref_ch) - kl)))
    rej = wu * (1 - sig(beta * (kl - (pol_rej - ref_rej))))
    return float(np.concatenate([ch, rej]).mean())


# Pinned from Step 1 (MLX == numpy == TRL-torch to ~1e-8).
_EXPECTED = {(1.0, 1.0): 0.4772096276283264,
             (1.33, 1.0): 0.566311240196228,
             (1.0, 1.5): 0.5808119177818298}
_EXPECTED_KL = 0.31288090348243713


def test_kto_loss_matches_reference_across_weights():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_loss, _kto_kl_baseline
    f = _fixed_logps()
    kl = float(_kto_kl_baseline(mx.array(f["pol_kl"]), mx.array(f["ref_kl"])))
    assert kl == pytest.approx(_EXPECTED_KL, abs=1e-6)  # matches TRL 0.31288
    for (wd, wu), expected in _EXPECTED.items():
        loss = float(_kto_loss(
            mx.array(f["pol_ch"]), mx.array(f["pol_rej"]),
            mx.array(f["ref_ch"]), mx.array(f["ref_rej"]), mx.array(kl),
            0.1, wd, wu,
        ))
        npy = _numpy_kto_loss(f["pol_ch"], f["pol_rej"], f["ref_ch"], f["ref_rej"], kl, 0.1, wd, wu)
        assert loss == pytest.approx(npy, abs=1e-6), f"w=({wd},{wu}) MLX vs numpy"
        assert loss == pytest.approx(expected, abs=1e-6), f"w=({wd},{wu}) vs pinned"


def test_kl_baseline_clamps_negative_to_zero():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_kl_baseline
    # policy far below reference on the mismatched completions -> raw mean < 0.
    pol_kl = mx.array([-30.0, -28.0, -35.0])
    ref_kl = mx.array([-20.0, -22.0, -19.0])
    assert float(_kto_kl_baseline(pol_kl, ref_kl)) == 0.0
    # positive estimate passes through unclamped.
    assert float(_kto_kl_baseline(ref_kl, pol_kl)) > 0.0


@pytest.mark.parametrize("labels_present", ["desirable_only", "undesirable_only"])
def test_kto_loss_single_label_batches_are_finite(labels_present):
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _kto_loss
    empty = mx.array(np.array([], dtype=np.float32))
    if labels_present == "desirable_only":
        pol_ch, ref_ch = mx.array([-8.0, -9.0]), mx.array([-8.3, -9.4])
        pol_rej = ref_rej = empty
    else:
        pol_rej, ref_rej = mx.array([-8.0, -9.0]), mx.array([-8.3, -9.4])
        pol_ch = ref_ch = empty
    loss = float(_kto_loss(pol_ch, pol_rej, ref_ch, ref_rej, mx.array(0.5), 0.1, 1.0, 1.0))
    assert np.isfinite(loss) and 0.0 <= loss <= 1.0


# NOTE: _kto_sum_logp's numpy-reference check lives in the Metal-gated
# test_mlx_kto_train_metal.py. Its .astype(mx.float32) / cross_entropy path
# resolves a real mlx dtype under the torch shim in the pytest import order, so
# it is exercised against real MLX instead of the shim.


def test_build_kto_batches_requires_batch_size_two():
    from unsloth_zoo.mlx.trainer import _build_kto_batches, MLXKTOConfig

    class _DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    dataset = [{"prompt": "a", "completion": " b", "label": True},
               {"prompt": "c", "completion": " d", "label": False}]
    with pytest.raises(ValueError) as exc:
        _build_kto_batches(dataset, _DummyTokenizer(), MLXKTOConfig(per_device_train_batch_size=1))
    msg = str(exc.value)
    assert "per_device_train_batch_size" in msg and ">= 2" in msg

    # batch_size >= 2 builds batches with the mismatched-pair KL variant.
    batches = _build_kto_batches(dataset, _DummyTokenizer(), MLXKTOConfig(per_device_train_batch_size=2))
    assert len(batches) == 1
    for key in ("comp_ids", "comp_labels", "kl_ids", "kl_labels", "label"):
        assert key in batches[0]
