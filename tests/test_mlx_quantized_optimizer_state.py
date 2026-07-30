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

"""8-bit first-moment optimizer state for MLX.

Behaviour over steps, not construction. Real MLX required, so this skips on Linux;
the routing half runs under the torch shim. No ``Dataset.map`` here, so the
spawn-vs-fork matrix does not apply.
"""

import tempfile

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is only available on Apple Silicon")
nn = pytest.importorskip("mlx.nn", reason="MLX is only available on Apple Silicon")
optim = pytest.importorskip("mlx.optimizers", reason="MLX is only available on Apple Silicon")

from mlx.utils import tree_flatten, tree_unflatten

from unsloth_zoo.mlx.optimizers_quantized import (
    DEFAULT_BITS,
    DEFAULT_GROUP_SIZE,
    SUPPORTED_GROUP_SIZES,
    QuantizedMomentAdam,
    QuantizedMomentAdamW,
)
from unsloth_zoo.mlx.utils import load_optimizer_state, save_optimizer_state


STEPS = 8


def _build_model(width=256):
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
    mx.eval(model.parameters())
    return model


def _build_ineligible_model():
    """Nothing here is quantization-eligible: 100 is not a multiple of 64."""
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(96, 100), nn.ReLU(), nn.Linear(100, 100))
    mx.eval(model.parameters())
    return model


def _batch(width=256, rows=32):
    mx.random.seed(1)
    return mx.random.normal((rows, width)), mx.random.normal((rows, width))


def _mse(model, x, y):
    return ((model(x) - y) ** 2).mean()


def _train(optimizer, model, x, y, steps=STEPS, compiled=False):
    grad_fn = nn.value_and_grad(model, _mse)
    # Mirrors MLXTrainer: state = [model.state, optimizer.state, mx.random.state]
    state = [model.state, optimizer.state, mx.random.state]

    def step_fn(x, y):
        loss, grads = grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    if compiled:
        step_fn = mx.compile(step_fn, inputs=state, outputs=state)
    losses = []
    for _ in range(steps):
        loss = step_fn(x, y)
        mx.eval(state)
        losses.append(float(loss))
    return losses


def _state_bytes(state):
    """Sum every array in the state tree. Default traversal only: marking
    tuple/list as a leaf would stop at the top-level ``layers`` list."""
    return sum(v.nbytes for _, v in tree_flatten(state) if isinstance(v, mx.array))


def _moment(optimizer, key="m"):
    return optimizer.state["layers"][0]["weight"][key]


# 1 ------------------------------------------------------------------------
def test_loss_decreases_with_quantized_first_moment():
    x, y = _batch()
    losses = _train(QuantizedMomentAdam(1e-3, bias_correction=True), _build_model(), x, y)

    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), (
        f"loss did not decrease monotonically over {STEPS} steps: {losses}"
    )
    assert losses[-1] < losses[0]


# 2 ------------------------------------------------------------------------
def test_final_loss_matches_fp32_baseline():
    x, y = _batch()
    fp32 = _train(optim.Adam(learning_rate=1e-3, bias_correction=True), _build_model(), x, y)
    int8 = _train(QuantizedMomentAdam(1e-3, bias_correction=True), _build_model(), x, y)

    delta = abs(fp32[-1] - int8[-1])
    assert delta < 1e-4, (
        f"quantized first moment changed the final loss by {delta:.3e} "
        f"(fp32 {fp32[-1]:.6f} vs int8 {int8[-1]:.6f})"
    )


# 3 ------------------------------------------------------------------------
def test_optimizer_state_is_measurably_smaller():
    x, y = _batch()
    fp32_opt = optim.Adam(learning_rate=1e-3, bias_correction=True)
    int8_opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(fp32_opt, _build_model(), x, y, steps=1)
    _train(int8_opt, _build_model(), x, y, steps=1)

    fp32_bytes = _state_bytes(fp32_opt.state)
    int8_bytes = _state_bytes(int8_opt.state)
    reduction = 1.0 - (int8_bytes / fp32_bytes)

    assert fp32_bytes > 0 and int8_bytes > 0
    assert reduction >= 0.30, (
        f"expected >=30% optimizer-state reduction, got {reduction:.1%} "
        f"({fp32_bytes:,} -> {int8_bytes:,} bytes)"
    )


# 4 ------------------------------------------------------------------------
def test_quantized_state_saves_reloads_and_resumes_identically():
    x, y = _batch()
    model = _build_model()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(opt, model, x, y, steps=4)

    checkpoint = tempfile.mkdtemp()
    save_optimizer_state(opt, checkpoint)
    weights = dict(tree_flatten(model.parameters()))

    continuous = _train(opt, model, x, y, steps=1)[0]

    resumed_model = _build_model()
    resumed_model.update(tree_unflatten(list(weights.items())))
    resumed_opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    # Materialise the state tree, then overwrite both weights and state so only
    # the checkpoint contributes to the next step.
    _train(resumed_opt, resumed_model, x, y, steps=1)
    resumed_model.update(tree_unflatten(list(weights.items())))
    load_optimizer_state(resumed_opt, checkpoint)
    mx.eval(resumed_model.parameters(), resumed_opt.state)

    resumed = _train(resumed_opt, resumed_model, x, y, steps=1)[0]

    assert resumed == continuous, (
        f"resume diverged: continuous {continuous:.8f} vs resumed {resumed:.8f}"
    )


# 5 ------------------------------------------------------------------------
def test_second_moment_is_never_quantized():
    """Quantizing v diverges to NaN; it must not be silently enabled."""
    x, y = _batch()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(opt, _build_model(), x, y, steps=2)

    second = _moment(opt, "v")
    assert isinstance(second, mx.array), (
        f"second moment must stay a single fp32 array, got {type(second).__name__} "
        "(a tuple/list means it was quantized)"
    )
    assert second.dtype == mx.float32, f"second moment dtype is {second.dtype}, expected float32"

    first = _moment(opt, "m")
    assert isinstance(first, (tuple, list)), (
        "first moment should be a packed triple for an eligible parameter"
    )


# 6 ------------------------------------------------------------------------
def test_compiled_and_uncompiled_agree():
    x, y = _batch()
    plain = _train(QuantizedMomentAdam(1e-3, bias_correction=True), _build_model(), x, y)
    compiled = _train(
        QuantizedMomentAdam(1e-3, bias_correction=True), _build_model(), x, y, compiled=True,
    )

    worst = max(abs(a - b) for a, b in zip(plain, compiled))
    assert worst == 0.0, f"mx.compile changed the loss trajectory by {worst:.3e}: {plain} vs {compiled}"


# 7 ------------------------------------------------------------------------
def test_model_with_no_quantizable_parameters_still_trains():
    """Nothing eligible -> every moment fp32, must still train."""
    model = _build_ineligible_model()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True)

    for _, param in tree_flatten(model.parameters()):
        assert not opt.is_quantizable(param), f"fixture is wrong: {param.shape} is eligible"

    mx.random.seed(1)
    x = mx.random.normal((32, 96))
    y = mx.random.normal((32, 100))
    losses = _train(opt, model, x, y)

    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), (
        f"ineligible-parameter model failed to train: {losses}"
    )
    assert isinstance(_moment(opt, "m"), mx.array), (
        "an ineligible parameter must keep a plain fp32 first moment"
    )


# 8 ------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [2, 4, 16, 32])
def test_bit_widths_other_than_8_are_rejected(bits):
    """Narrower widths are refused, not offered untested."""
    with pytest.raises(ValueError, match=r"supports bits=8 only"):
        QuantizedMomentAdam(1e-3, bits=bits)
    with pytest.raises(ValueError, match=r"supports bits=8 only"):
        QuantizedMomentAdamW(1e-3, bits=bits)


# 9 ------------------------------------------------------------------------
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
def test_every_supported_group_size_trains(group_size):

    x, y = _batch()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True, group_size=group_size)
    losses = _train(opt, _build_model(), x, y)

    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), (
        f"group_size={group_size} failed to train: {losses}"
    )
    assert isinstance(_moment(opt, "m"), (tuple, list))


@pytest.mark.parametrize("group_size", [1, 48, 100, 256])
def test_unsupported_group_size_is_rejected(group_size):
    with pytest.raises(ValueError, match=r"supports group_size in"):
        QuantizedMomentAdam(1e-3, group_size=group_size)


# 10 -----------------------------------------------------------------------
def test_adamw_variant_also_quantizes():

    x, y = _batch()
    opt = QuantizedMomentAdamW(1e-3, weight_decay=0.0, bias_correction=True)
    losses = _train(opt, _build_model(), x, y)

    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), losses
    assert isinstance(_moment(opt, "m"), (tuple, list))
    assert _moment(opt, "v").dtype == mx.float32
    assert (DEFAULT_GROUP_SIZE, DEFAULT_BITS) == (opt.group_size, opt.bits)
