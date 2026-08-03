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

"""QAT efficacy on real Metal: does it actually improve the saved artifact?

Everything else about QAT can pass while the feature does nothing useful. This
trains the same model twice from one seed -- with and without QAT -- then runs
the real `fuse(dequantize=False)` that `save_method='merged_4bit'` uses, and
compares the losses of the two *saved* models.

Measured on Qwen2.5-0.5B-Instruct-4bit / wikitext-2 (300 steps): the non-QAT
run retained only 36% of its training gains through the save, while QAT
retained ~100% and shipped a 0.13 nats better model. This test reproduces the
mechanism on a synthetic stack small enough for CI.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("mlx.core")


def _real_mlx_runtime():
    try:
        lora = importlib.import_module("mlx_lm.tuner.lora")
    except Exception:
        return False
    if not isinstance(getattr(lora, "LoRALinear", None), type):
        return False
    origin = getattr(sys.modules.get("mlx.core"), "__file__", "") or ""
    return "mlx_simulation" not in origin


if not _real_mlx_runtime():
    pytest.skip("needs the real mlx runtime", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tuner.lora import LoRALinear

from unsloth_zoo.mlx.qat import apply_mlx_qat

DIMS = 256
GROUP_SIZE = 64
BITS = 4
N_LAYERS = 3
# QAT's advantage widens with training (it trades a worse free-floating loss
# for a lossless save), so a short run is genuinely marginal: at 120 steps the
# saved-model margin was -0.00003 on one of four seeds. Measured minimum margin
# across seeds 0-3: 120 steps -0.000030, 400 steps +0.001303, 900 +0.002600.
STEPS = 400


@pytest.fixture(autouse=True)
def _require_real_metal():
    import mlx.core as _mx
    if not (getattr(_mx, "metal", None) and _mx.metal.is_available()
            and _mx.default_device() == _mx.gpu):
        pytest.skip("real Metal required; shim active or no GPU")


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            LoRALinear.from_base(
                nn.QuantizedLinear.from_linear(
                    nn.Linear(DIMS, DIMS, bias=False),
                    group_size=GROUP_SIZE, bits=BITS, mode="affine",
                ),
                r=16, scale=2.0,
            )
            for _ in range(N_LAYERS)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = mx.tanh(layer(x))
        return x


def _fuse_in_place(model):
    from mlx.utils import tree_unflatten
    fused = [
        (name, module.fuse(dequantize=False))
        for name, module in model.named_modules()
        if hasattr(module, "fuse")
    ]
    model.update_modules(tree_unflatten(fused))
    return model


def _run(use_qat, seed=0):
    mx.random.seed(seed)
    model = _Stack()
    mx.random.seed(1234)
    xs = mx.random.normal((32, DIMS))
    target = mx.tanh(mx.random.normal((32, DIMS)) * 0.5)
    mx.eval(xs, target)

    if use_qat:
        apply_mlx_qat(model, "auto")

    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    def loss_fn(m):
        return ((m(xs) - target) ** 2).mean()

    opt = optim.Adam(learning_rate=3e-3)
    step = nn.value_and_grad(model, loss_fn)
    for _ in range(STEPS):
        _, grads = step(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    pre = float(loss_fn(model))
    _fuse_in_place(model)
    post = float(((model(xs) - target) ** 2).mean())
    return pre, post


@pytest.mark.parametrize("seed", [0, 1])
def test_qat_removes_post_fuse_degradation_and_improves_the_saved_model(seed):
    base_pre, base_post = _run(use_qat=False, seed=seed)
    qat_pre, qat_post = _run(use_qat=True, seed=seed)

    base_degradation = base_post - base_pre
    qat_degradation = qat_post - qat_pre

    # 1. There is a problem to solve: fusing hurts the non-QAT run.
    assert base_degradation > 0, (
        "expected merged_4bit fusing to degrade a non-QAT run; got "
        f"{base_degradation:+.6f}"
    )

    # 2. QAT closes the train/deploy gap -- its own fuse is near lossless.
    assert abs(qat_degradation) < base_degradation / 10, (
        f"QAT degradation {qat_degradation:+.6f} should be far below the "
        f"baseline's {base_degradation:+.6f}"
    )

    # 3. The only comparison that ships: QAT's saved model is better.
    assert qat_post < base_post, (
        f"QAT post-fuse loss {qat_post:.6f} should beat baseline "
        f"{base_post:.6f}"
    )
