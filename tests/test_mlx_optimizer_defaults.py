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

"""Real MLX optimizer defaults, checked against the installed torch.

The companion file ``test_mlx_optimizer_torch_parity.py`` runs under the
mlx-on-torch shim and can only see what the trainer *passes*. The shim's
optimizer adapters take ``**kw`` and supply their own fallbacks, so they cannot
answer what real MLX would have defaulted to -- and their fallbacks are torch's
values, which is exactly how an unpinned default hides: shim tests agree with
torch while the Apple run does not.

That question is what this module exists for, so it needs the genuine mlx
package and skips when mlx is absent or shimmed.
"""

import inspect

import numpy as np
import pytest

_SKIP = "Requires a real mlx runtime (Apple Silicon Metal, or Linux mlx-cpu)"

mx = pytest.importorskip("mlx.core", reason=_SKIP)
optim = pytest.importorskip("mlx.optimizers", reason=_SKIP)
# importorskip alone is not enough: another test module may have installed the
# mlx-on-torch shim into sys.modules first, and the shim's optimizers take
# **kw, so signature introspection would silently measure the wrong thing.
if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip(_SKIP, allow_module_level=True)


def _torch_default(torch_class_name, parameter):
    import torch

    cls = getattr(torch.optim, torch_class_name)
    default = inspect.signature(cls).parameters[parameter].default
    assert default is not inspect.Parameter.empty
    return default


def _mlx_default(mlx_class_name, parameter):
    cls = getattr(optim, mlx_class_name)
    default = inspect.signature(cls).parameters[parameter].default
    assert default is not inspect.Parameter.empty
    return default


def _build(optim_name, **config_kwargs):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(optim=optim_name, **config_kwargs)
    return trainer._build_optimizer(total_steps=4)


def test_real_mlx_adagrad_is_built_with_the_torch_epsilon():
    """The value that actually reaches Apple's Adagrad, not just the kwarg."""
    expected = _torch_default("Adagrad", "eps")
    mlx_own_default = _mlx_default("Adagrad", "eps")
    assert mlx_own_default != expected, (
        "MLX now defaults Adagrad eps to torch's value, so this test no longer "
        "distinguishes a pinned epsilon from an unpinned one -- re-derive it"
    )

    optimizer = _build("adagrad")

    assert optimizer.eps == pytest.approx(expected, rel=1e-12), (
        f"real MLX Adagrad was left on {optimizer.eps!r}; an `optim='adagrad'` "
        f"recipe gets torch's {expected!r} on the HF backend"
    )


@pytest.mark.parametrize("optim_name,mlx_class,torch_class,parameter", [
    ("rmsprop",  "RMSprop",  "RMSprop",  "eps"),
    ("rmsprop",  "RMSprop",  "RMSprop",  "alpha"),
    ("adadelta", "AdaDelta", "Adadelta", "eps"),
    ("adadelta", "AdaDelta", "Adadelta", "rho"),
    ("adamax",   "Adamax",   "Adamax",   "eps"),
])
def test_remaining_defaults_still_agree_with_torch(
    optim_name, mlx_class, torch_class, parameter,
):
    """These MLX defaults happen to equal torch's, so the trainer leaves them
    alone. That is only safe while it stays true: if MLX ever retunes one, the
    trainer would silently drift from the recipe whose name it advertises, and
    this is what notices."""
    mlx_value = _mlx_default(mlx_class, parameter)
    torch_value = _torch_default(torch_class, parameter)

    assert mlx_value == pytest.approx(torch_value, rel=1e-12), (
        f"MLX {mlx_class} default {parameter}={mlx_value!r} no longer matches "
        f"torch's {torch_value!r}; _build_optimizer must now pin {parameter} "
        f"explicitly for {optim_name}, the way it already pins Adagrad's eps"
    )


_ADAMAX_LR = 1e-2
_ADAMAX_BETAS = (0.9, 0.999)
_ADAMAX_EPS = 1e-8


def _adamax_case():
    initial = np.arange(4, dtype=np.float32).reshape(2, 2) / 4.0
    base = np.array([[0.3, -0.7], [0.05, 0.9]], dtype=np.float32)
    return initial, [base * (1 + i) for i in range(5)]


def _mlx_adamax_trajectory(cls, initial_weight, gradients):
    import mlx.nn as nn

    model = nn.Linear(2, 2, bias=False)
    model.update({"weight": mx.array(initial_weight)})
    optimizer = cls(
        learning_rate=_ADAMAX_LR, betas=_ADAMAX_BETAS, eps=_ADAMAX_EPS,
    )
    trajectory = []
    for gradient in gradients:
        optimizer.update(model, {"weight": mx.array(gradient)})
        mx.eval(model.parameters(), optimizer.state)
        trajectory.append(np.array(model.parameters()["weight"], copy=True))
    return trajectory


def _torch_adamax_trajectory(initial_weight, gradients):
    import torch

    parameter = torch.nn.Parameter(torch.tensor(initial_weight))
    optimizer = torch.optim.Adamax(
        [parameter], lr=_ADAMAX_LR, betas=_ADAMAX_BETAS, eps=_ADAMAX_EPS,
    )
    trajectory = []
    for gradient in gradients:
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
        optimizer.zero_grad()
        trajectory.append(parameter.detach().numpy().copy())
    return trajectory


def test_adamax_tracks_torch_adamax_step_for_step():
    """MLX Adamax omits torch's ``lr / (1 - beta1**t)`` first-moment correction.

    The trainer builds ``_BiasCorrectedAdamax`` instead, so an
    ``optim='adamax'`` run has to follow ``torch.optim.Adamax`` from step 1, not
    merely converge to it. Tolerance is fp32 rounding plus the residual from
    keeping eps on the denominator rather than inside torch's ``max``, which is
    bounded by eps.
    """
    from unsloth_zoo.mlx.trainer import _BiasCorrectedAdamax

    initial, gradients = _adamax_case()
    expected = _torch_adamax_trajectory(initial, gradients)
    got = _mlx_adamax_trajectory(_BiasCorrectedAdamax, initial, gradients)

    for step, (mlx_weight, torch_weight) in enumerate(zip(got, expected), start=1):
        assert np.abs(mlx_weight - torch_weight).max() < 1e-6, (
            f"step {step} diverged from torch.optim.Adamax: "
            f"{mlx_weight} vs {torch_weight}"
        )


def test_plain_mlx_adamax_still_needs_the_correction():
    """Canary for the test above: it only proves something while stock MLX
    Adamax actually disagrees with torch. If MLX adds the correction upstream,
    ``_BiasCorrectedAdamax`` is dead weight and should be dropped."""
    initial, gradients = _adamax_case()
    expected = _torch_adamax_trajectory(initial, gradients)
    got = _mlx_adamax_trajectory(optim.Adamax, initial, gradients)

    ratio = np.abs(got[0] - initial).max() / np.abs(expected[0] - initial).max()
    assert ratio == pytest.approx(1.0 - _ADAMAX_BETAS[0], rel=1e-4), (
        "stock MLX Adamax no longer scales its first step by (1 - beta1); "
        f"ratio to torch is {ratio}, so re-derive _BiasCorrectedAdamax"
    )


def test_adagrad_step_stays_bounded_by_the_learning_rate():
    """Why the smaller epsilon is safe rather than merely faithful.

    Adagrad divides by ``sqrt(sum g^2) + eps`` and the running sum includes the
    current gradient, so the denominator is never below ``|g|`` and the update
    magnitude is at most ``lr`` for any epsilon. Shrinking eps cannot produce
    the blow-up a smaller stabiliser normally risks; it only stops small
    gradients from being damped. Exercised at gradient scales that straddle both
    epsilons, including exact zero.
    """
    import mlx.nn as nn

    lr = 1e-3
    for scale in (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1.0):
        model = nn.Linear(2, 2, bias=False)
        model.update({"weight": mx.zeros((2, 2))})
        optimizer = optim.Adagrad(learning_rate=lr, eps=1e-10)
        gradient = {"weight": mx.full((2, 2), scale)}
        for _ in range(3):
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state)
        moved = mx.abs(model.parameters()["weight"]).max().item()
        assert moved == moved, f"NaN at gradient scale {scale}"  # NaN != NaN
        # 3 steps, each bounded by lr.
        assert moved <= 3 * lr + 1e-12, (
            f"gradient scale {scale} moved the weight {moved} past the "
            f"{3 * lr} bound Adagrad's own denominator guarantees"
        )
