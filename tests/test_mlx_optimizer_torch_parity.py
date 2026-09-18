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

"""Hyperparameter defaults of the torch-style MLX optimizers, pinned to torch.

MLX and PyTorch pick different defaults for the same optimizer, and only some of
them agree. Exposing an optimizer name that HF Trainer also accepts is a promise
that the same recipe trains the same way, so every default is asserted against
the *installed* upstream rather than a literal copied out of the trainer:

  * the reference value is read from ``inspect.signature(torch.optim.X)``,
  * and, where HF Trainer has a branch for the name, from
    ``Trainer.get_optimizer_cls_and_kwargs`` -- which is what decides whether a
    recipe gets torch's default at all or an explicit override.

A literal here would just re-encode whatever the trainer already does and would
pass no matter which value was wrong.

Name/plumbing logic only, so this runs on Linux CI too. Note that requesting the
shim does not guarantee getting it: ``conftest`` imports ``unsloth_zoo`` during
collection, so on a host that has real mlx the trainer's ``optim`` global is
already bound to the genuine package and the module-scope fixture cannot rebind
it. The assertions below are written to hold either way -- they read the pinned
hyperparameter from the shim's recorded kwargs or from a real optimizer's
attribute, and an unpinned value reads as absent under both. Defaults that only
real mlx can answer live in ``test_mlx_optimizer_defaults.py``.
"""

import inspect

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# The four optimizer names this module covers, and the torch class each one is
# the MLX counterpart of.
_TORCH_COUNTERPART = {
    "rmsprop": "RMSprop",
    "adamax": "Adamax",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
}


def _torch_default(torch_class_name, parameter):
    """The installed torch optimizer's own default for `parameter`."""
    import torch

    cls = getattr(torch.optim, torch_class_name)
    default = inspect.signature(cls).parameters[parameter].default
    assert default is not inspect.Parameter.empty, (
        f"torch.optim.{torch_class_name} has no default for {parameter!r}; "
        "the parity reference cannot be read from the signature"
    )
    return default


def _build(optim_name, **config_kwargs):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(optim=optim_name, **config_kwargs)
    return trainer, trainer._build_optimizer(total_steps=4)


def _hyperparameter(optimizer, name):
    """Read a hyperparameter off either a real MLX optimizer or the torch shim.

    Real MLX stores it as an attribute; the shim keeps constructor kwargs in
    ``_kw`` and only applies its own fallback when the trainer passed nothing --
    so a missing key in ``_kw`` genuinely means "trainer did not pin this".
    """
    kw = getattr(optimizer, "_kw", None)
    if kw is not None and name in kw:
        return kw[name]
    if kw is not None and not hasattr(optimizer, name):
        return None
    return getattr(optimizer, name, None)


def test_adagrad_epsilon_matches_the_torch_default_a_recipe_would_get():
    """MLX Adagrad defaults eps=1e-8, torch defaults 1e-10.

    Adagrad's denominator is ``sqrt(sum g^2) + eps``, so for gradient elements
    near or below the epsilon the larger MLX value damps the step by up to two
    orders of magnitude relative to the same recipe on the torch backend.
    """
    expected = _torch_default("Adagrad", "eps")

    _, optimizer = _build("adagrad")

    assert _hyperparameter(optimizer, "eps") == pytest.approx(expected, rel=1e-12), (
        "MLX Adagrad was built on MLX's default epsilon instead of the "
        f"torch default {expected!r} that an `optim='adagrad'` recipe gets on "
        "the HF backend"
    )


def test_hf_trainer_leaves_adagrad_epsilon_at_the_torch_default():
    """The reason the Adagrad epsilon above is torch's default and not
    ``args.adam_epsilon``: HF Trainer's ADAGRAD branch sets only the class, so
    the returned kwargs carry `lr` and nothing else."""
    transformers = pytest.importorskip("transformers")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        args = transformers.TrainingArguments(
            output_dir=directory, optim="adagrad", report_to=[],
        )
        cls, kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(args)

    assert cls is __import__("torch").optim.Adagrad
    assert "eps" not in kwargs, (
        "HF Trainer now overrides Adagrad's epsilon; the MLX branch must "
        f"forward that value instead of torch's signature default ({kwargs})"
    )


@pytest.mark.parametrize("optim_name", sorted(_TORCH_COUNTERPART))
def test_every_new_name_is_advertised_and_buildable(optim_name):
    """Guards the whole class of surface forms a config can carry the name in,
    not just the lowercase literal: enum-with-.value, dotted enum repr, upper
    case and hyphenation all funnel through _normalize_mlx_optimizer_name."""
    from unsloth_zoo.mlx.trainer import (
        SUPPORTED_MLX_OPTIMIZERS,
        _normalize_mlx_optimizer_name,
    )

    assert optim_name in SUPPORTED_MLX_OPTIMIZERS

    class _EnumLike:
        value = optim_name

    surface_forms = [
        optim_name,
        optim_name.upper(),
        f"  {optim_name}  ",
        f"OptimizerNames.{optim_name.upper()}",
        optim_name.replace("_", "-"),
        _EnumLike(),
    ]
    for form in surface_forms:
        assert _normalize_mlx_optimizer_name(form) == optim_name, (
            f"{form!r} did not normalize to {optim_name!r}"
        )

    _, optimizer = _build(optim_name)
    assert optimizer is not None


@pytest.mark.parametrize("optim_name", sorted(_TORCH_COUNTERPART))
def test_new_optimizers_use_coupled_decay_and_leave_adamw_path_alone(optim_name):
    """torch folds weight decay into the gradient for all four of these, unlike
    AdamW's decoupled shrink. Also pins that turning these on did not disturb
    the pre-existing decoupled optimizers a saved config may still request."""
    trainer, _ = _build(optim_name, weight_decay=0.05)
    assert trainer._coupled_weight_decay == pytest.approx(0.05)
    assert trainer._manual_weight_decay == pytest.approx(0.0)

    adamw_trainer, _ = _build("adamw", weight_decay=0.05)
    assert adamw_trainer._manual_weight_decay == pytest.approx(0.05)
    assert adamw_trainer._coupled_weight_decay == pytest.approx(0.0)


def test_adamax_is_built_with_the_torch_first_moment_bias_correction():
    """``mlx.optimizers.Adamax`` takes no ``bias_correction`` flag and overrides
    ``Adam.apply_single`` with the uncorrected update, so pinning it is not a
    kwarg -- the trainer has to substitute a subclass. Numerics are checked
    against torch in ``test_mlx_optimizer_defaults.py``; this pins the wiring,
    which is the part the shim can see."""
    from unsloth_zoo.mlx.trainer import _BiasCorrectedAdamax

    _, optimizer = _build("adamax")

    assert isinstance(optimizer, _BiasCorrectedAdamax), (
        f"optim='adamax' built {type(optimizer).__name__}; stock MLX Adamax "
        "scales the first update by (1 - beta1), ~10x below torch's"
    )
    assert "apply_single" in vars(_BiasCorrectedAdamax), (
        "_BiasCorrectedAdamax no longer overrides the update, so it is stock "
        "MLX Adamax under a different name"
    )


def test_hf_trainer_drops_optim_args_for_rmsprop_too():
    """Why the RMSprop branch takes no ``optim_args``: HF Trainer's RMSPROP
    branch sets only ``optimizer_cls``, and ``optim_args`` is merged in by the
    GaLore/Apollo/GrokAdamW branches alone. ``momentum``/``alpha``/``centered``
    are dropped on the torch backend as well, and MLX RMSprop's own defaults
    already equal torch's, so honouring them here would *introduce* a
    divergence from the recipe rather than remove one."""
    transformers = pytest.importorskip("transformers")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        args = transformers.TrainingArguments(
            output_dir=directory,
            optim="rmsprop",
            optim_args="momentum=0.9,alpha=0.95,centered=True",
            report_to=[],
        )
        cls, kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(args)

    assert cls is __import__("torch").optim.RMSprop
    assert set(kwargs) == {"lr"}, (
        "HF Trainer now forwards optim_args to RMSprop; the MLX branch must "
        f"parse and apply them instead of relying on defaults ({kwargs})"
    )
