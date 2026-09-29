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

"""QAT efficacy on real Metal: does it actually improve the saved artifact?"""

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
# QAT's saved-model margin grows with steps; short runs are marginal.
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

    assert base_degradation > 0, (
        "expected merged_4bit fusing to degrade a non-QAT run; got "
        f"{base_degradation:+.6f}"
    )

    assert abs(qat_degradation) < base_degradation / 10, (
        f"QAT degradation {qat_degradation:+.6f} should be far below the "
        f"baseline's {base_degradation:+.6f}"
    )

    assert qat_post < base_post, (
        f"QAT post-fuse loss {qat_post:.6f} should beat baseline "
        f"{base_post:.6f}"
    )
