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

"""8-bit first-moment optimizer state for MLX.

Behaviour over steps, not construction. Needs a real mlx runtime: Apple Silicon
Metal, or Linux via ``mlx-cpu`` (both are wired up in CI). The routing half runs
under the torch shim, which cannot implement ``mx.quantize``. No ``Dataset.map``
here, so the spawn-vs-fork matrix does not apply.
"""

import tempfile

import pytest

_SKIP = "Requires a real mlx runtime (Apple Silicon Metal, or Linux mlx-cpu)"

mx = pytest.importorskip("mlx.core", reason=_SKIP)
nn = pytest.importorskip("mlx.nn", reason=_SKIP)
optim = pytest.importorskip("mlx.optimizers", reason=_SKIP)
# importorskip alone is not enough: another test module may have installed the
# mlx-on-torch shim into sys.modules first, and the shim raises on mx.quantize.
if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip(_SKIP, allow_module_level=True)

from mlx.utils import tree_flatten, tree_unflatten

from unsloth_zoo.mlx.optimizers_quantized import (
    DEFAULT_BITS,
    DEFAULT_GROUP_SIZE,
    SUPPORTED_GROUP_SIZES,
    QuantizedMomentAdam,
    QuantizedMomentAdamW,
    unpack_quantized_moments,
)
from unsloth_zoo.mlx.utils import load_optimizer_state, save_optimizer_state


@pytest.fixture(autouse=True)
def _reject_shimmed_session():
    """The shim is installed by other modules at test time, after this one was
    imported, and it then owns mx.quantize for whichever unsloth_zoo module
    imported mlx later. CI runs this file in its own pytest invocation (and fails
    the job if it skips), so skipping here only affects mixed local runs."""
    import sys
    if "mlx_simulation" in sys.modules:
        pytest.skip("the mlx-on-torch shim is active in this process")


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


def _build_bottleneck_model(width=256, hidden=64):
    """Both weights stay quantization-eligible, but the model has far fewer
    parameters than _underdetermined_batch has targets, so the loss floors above
    zero instead of collapsing to ~1e-9."""
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, width))
    mx.eval(model.parameters())
    return model


def _underdetermined_batch(width=256, rows=512):
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
def test_loss_trajectory_tracks_fp32_baseline():
    """Compare the whole trajectory over a target the model CANNOT fit. On the
    overparameterised fixture the loss reaches ~1e-9, where a relative comparison
    measures nothing but float noise."""
    x, y = _underdetermined_batch()
    steps = 200
    fp32 = _train(optim.Adam(learning_rate=1e-3, bias_correction=True),
                  _build_bottleneck_model(), x, y, steps=steps)
    int8 = _train(QuantizedMomentAdam(1e-3, bias_correction=True),
                  _build_bottleneck_model(), x, y, steps=steps)

    tail = slice(steps // 2, None)
    mape = sum(abs(a - b) / abs(b) for a, b in zip(int8[tail], fp32[tail])) / (steps - steps // 2)
    assert mape < 0.05, (
        f"quantized first moment moved the loss trajectory by {mape:.2%} MAPE over the "
        f"last {steps - steps // 2} steps (fp32 {fp32[-1]:.6e} vs int8 {int8[-1]:.6e})"
    )
    assert all(l == l for l in int8), "quantized run produced a NaN"


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
    """Compares POST-update parameters and every state leaf. ``_train`` returns the
    loss computed before the update, so comparing that would pass even with
    load_optimizer_state removed entirely."""
    x, y = _batch()
    model = _build_model()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(opt, model, x, y, steps=4)

    checkpoint = tempfile.mkdtemp()
    save_optimizer_state(opt, checkpoint)
    weights = dict(tree_flatten(model.parameters()))

    _train(opt, model, x, y, steps=1)
    continuous = dict(tree_flatten(model.parameters()))
    continuous_state = dict(tree_flatten(opt.state))

    resumed_model = _build_model()
    resumed_model.update(tree_unflatten(list(weights.items())))
    resumed_opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    # Materialise the state tree, then overwrite both weights and state so only
    # the checkpoint contributes to the next step.
    _train(resumed_opt, resumed_model, x, y, steps=1)
    resumed_model.update(tree_unflatten(list(weights.items())))
    load_optimizer_state(resumed_opt, checkpoint)
    mx.eval(resumed_model.parameters(), resumed_opt.state)

    _train(resumed_opt, resumed_model, x, y, steps=1)
    resumed = dict(tree_flatten(resumed_model.parameters()))
    resumed_state = dict(tree_flatten(resumed_opt.state))

    for name, expected in continuous.items():
        delta = float(mx.abs(resumed[name] - expected).max())
        assert delta == 0.0, f"resumed parameter {name} differs by {delta:.3e}"
    assert resumed_state.keys() == continuous_state.keys(), "state tree shape changed"
    for name, expected in continuous_state.items():
        delta = float(mx.abs(resumed_state[name].astype(mx.float32)
                             - expected.astype(mx.float32)).max())
        assert delta == 0.0, f"resumed state leaf {name} differs by {delta:.3e}"


# 5 ------------------------------------------------------------------------
def test_second_moment_is_never_quantized():
    """Quantizing v diverges to NaN; it must not be silently enabled."""
    x, y = _batch()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(opt, _build_model(), x, y, steps=2)

    second = _moment(opt, "v")
    assert isinstance(second, mx.array), (
        f"second moment must stay a single unquantized array, got {type(second).__name__} "
        "(a tuple/list means it was quantized)"
    )
    # This fixture is fp32; dtype tracking in general is test_moments_follow_the_parameter_dtype.
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
# 11 -----------------------------------------------------------------------
def test_zero_gradient_coordinates_never_move():
    """A coordinate whose gradient is identically zero keeps v == 0, so the Adam
    denominator is eps alone and any dequantization residue is amplified ~1e8x.
    Plain Adam moves it by exactly nothing and so must this."""
    steps, lr = 500, 1e-3
    grad = mx.array([[1.0] + [0.0] * (DEFAULT_GROUP_SIZE - 1)])

    def drive(opt):
        param = mx.zeros((1, DEFAULT_GROUP_SIZE))
        opt.init({"w": param})
        for _ in range(steps):
            param = opt.apply_gradients({"w": grad}, {"w": param})["w"]
            mx.eval(param, opt.state)
        return param

    got = drive(QuantizedMomentAdam(lr, bias_correction=True))
    moved = float(mx.abs(got[0, 1:]).max())
    assert moved == 0.0, (
        f"a zero-gradient coordinate moved {moved:.6f} ({moved / (lr * steps):.1f}x the "
        "lr*steps budget); plain Adam leaves it at exactly 0"
    )
    assert float(got[0, 0]) != 0.0, "the live coordinate should still train"


# 12 -----------------------------------------------------------------------
def test_plain_checkpoint_migrates_to_packed_on_resume():
    """Before this optimizer existed, adamw_8bit produced plain AdamW state. Such a
    checkpoint must not leave the moment full width for the rest of the run."""
    x, y = _batch()
    model = _build_model()
    plain = optim.Adam(learning_rate=1e-3, bias_correction=True)
    _train(plain, model, x, y, steps=4)
    checkpoint = tempfile.mkdtemp()
    save_optimizer_state(plain, checkpoint)

    resumed = QuantizedMomentAdam(1e-3, bias_correction=True)
    resumed_model = _build_model()
    _train(resumed, resumed_model, x, y, steps=1)
    load_optimizer_state(resumed, checkpoint)
    assert isinstance(_moment(resumed, "m"), mx.array), "fixture is wrong: not a plain array"

    _train(resumed, resumed_model, x, y, steps=1)
    assert isinstance(_moment(resumed, "m"), (tuple, list)), (
        "an eligible moment loaded from a pre-quantization checkpoint stayed full width"
    )


# 13 -----------------------------------------------------------------------
def test_packed_checkpoint_loads_into_plain_optimizer():
    """Switching back to optim="adamw" must not hand a list to mlx's b1 * m."""
    x, y = _batch()
    model = _build_model()
    packed = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(packed, model, x, y, steps=4)
    checkpoint = tempfile.mkdtemp()
    save_optimizer_state(packed, checkpoint)

    plain = optim.Adam(learning_rate=1e-3, bias_correction=True)
    plain_model = _build_model()
    _train(plain, plain_model, x, y, steps=1)
    load_optimizer_state(plain, checkpoint)
    assert isinstance(_moment(plain, "m"), mx.array), "packed moment was not unpacked"
    _train(plain, plain_model, x, y, steps=1)


# 13b ----------------------------------------------------------------------
def test_unpacking_does_not_carry_the_zero_residue_out():
    """Dequantization leaves a residue where the true moment is zero. Handing that
    to a plain optimizer would divide it by eps while v == 0, reintroducing the
    blow-up on the far side of the switch."""
    steps, lr = 200, 1e-3
    grad = mx.array([[1.0] + [0.0] * (DEFAULT_GROUP_SIZE - 1)])

    packed = QuantizedMomentAdam(lr, bias_correction=True)
    param = mx.zeros((1, DEFAULT_GROUP_SIZE))
    packed.init({"w": param})
    for _ in range(steps):
        param = packed.apply_gradients({"w": grad}, {"w": param})["w"]
        mx.eval(param, packed.state)

    unpacked = unpack_quantized_moments(packed.state, packed.group_size, packed.bits)
    m, v = unpacked["w"]["m"], unpacked["w"]["v"]
    residue = float(mx.max(mx.abs(mx.where(v == 0, m, mx.zeros_like(m)))))
    assert residue == 0.0, f"unpacked moment kept a {residue:.3e} residue where v == 0"

    plain = optim.Adam(learning_rate=lr, bias_correction=True)
    resumed = mx.zeros((1, DEFAULT_GROUP_SIZE))
    plain.init({"w": resumed})
    plain.state = unpacked
    for _ in range(steps):
        resumed = plain.apply_gradients({"w": grad}, {"w": resumed})["w"]
        mx.eval(resumed, plain.state)
    drift = max(abs(float(resumed[0, i])) for i in range(1, DEFAULT_GROUP_SIZE))
    assert drift == 0.0, f"zero-gradient coordinates drifted {drift:.3e} after unpacking"


# 14 -----------------------------------------------------------------------
@pytest.mark.parametrize("saved,resumed", [(32, 128), (128, 32)])
def test_group_size_is_read_back_from_the_checkpoint(saved, resumed):
    """group_size is an instance attribute, so it rides in safetensors metadata."""
    x, y = _batch()
    opt = QuantizedMomentAdam(1e-3, bias_correction=True, group_size=saved)
    _train(opt, _build_model(), x, y, steps=2)
    checkpoint = tempfile.mkdtemp()
    save_optimizer_state(opt, checkpoint)

    other = QuantizedMomentAdam(1e-3, bias_correction=True, group_size=resumed)
    other_model = _build_model()
    _train(other, other_model, x, y, steps=1)
    load_optimizer_state(other, checkpoint)
    assert other.group_size == saved
    _train(other, other_model, x, y, steps=1)


# 15 -----------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{"bits": 8.9}, {"group_size": 64.9}, {"bits": True}])
def test_non_integral_settings_are_rejected(kwargs):
    """int() truncates, so bits=8.9 would silently pass the bits == 8 check."""
    with pytest.raises(ValueError, match="must be an integer"):
        QuantizedMomentAdam(1e-3, **kwargs)


# 16 -----------------------------------------------------------------------
def test_moments_follow_the_parameter_dtype():
    """The saving is 8-bit against the parameter dtype, not against float32."""
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    mx.random.seed(1)
    x = mx.random.normal((32, 256)).astype(mx.bfloat16)
    y = mx.random.normal((32, 256)).astype(mx.bfloat16)

    opt = QuantizedMomentAdam(1e-3, bias_correction=True)
    _train(opt, model, x, y, steps=2)
    assert _moment(opt, "v").dtype == mx.bfloat16, (
        "v follows the parameter dtype; it is unquantized, not float32"
    )
    assert isinstance(_moment(opt, "m"), (tuple, list))


def test_adamw_variant_also_quantizes():

    x, y = _batch()
    opt = QuantizedMomentAdamW(1e-3, weight_decay=0.0, bias_correction=True)
    losses = _train(opt, _build_model(), x, y)

    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), losses
    assert isinstance(_moment(opt, "m"), (tuple, list))
    assert _moment(opt, "v").dtype == mx.float32
    assert (DEFAULT_GROUP_SIZE, DEFAULT_BITS) == (opt.group_size, opt.bits)
