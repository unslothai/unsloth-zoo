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

"""torch-style MLX optimizer defaults, pinned to the installed torch / HF Trainer.

Holds under the shim or real mlx (conftest may bind real mlx first); real-mlx-only checks: test_mlx_optimizer_defaults.py.
"""

import inspect

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


_TORCH_COUNTERPART = {
    "rmsprop": "RMSprop",
    "adamax": "Adamax",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
}


def _torch_default(torch_class_name, parameter):
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
    # Shim keeps passed kwargs in _kw: a missing key means the trainer did not pin it.
    kw = getattr(optimizer, "_kw", None)
    if kw is not None and name in kw:
        return kw[name]
    if kw is not None and not hasattr(optimizer, name):
        return None
    return getattr(optimizer, name, None)


def test_adagrad_epsilon_matches_the_torch_default_a_recipe_would_get():
    expected = _torch_default("Adagrad", "eps")

    _, optimizer = _build("adagrad")

    assert _hyperparameter(optimizer, "eps") == pytest.approx(expected, rel=1e-12), (
        "MLX Adagrad was built on MLX's default epsilon instead of the "
        f"torch default {expected!r} that an `optim='adagrad'` recipe gets on "
        "the HF backend"
    )


def test_hf_trainer_leaves_adagrad_epsilon_at_the_torch_default():
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
    trainer, _ = _build(optim_name, weight_decay=0.05)
    assert trainer._coupled_weight_decay == pytest.approx(0.05)
    assert trainer._manual_weight_decay == pytest.approx(0.0)

    adamw_trainer, _ = _build("adamw", weight_decay=0.05)
    assert adamw_trainer._manual_weight_decay == pytest.approx(0.05)
    assert adamw_trainer._coupled_weight_decay == pytest.approx(0.0)


def test_adamax_is_built_with_the_torch_first_moment_bias_correction():
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


def _build_with_optim_args(optim_name, optim_args):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(optim=optim_name)
    trainer.args.optim_args = optim_args
    return trainer._build_optimizer(total_steps=4)


def _hf_kwargs(optim_name, optim_args):
    transformers = pytest.importorskip("transformers")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        args = transformers.TrainingArguments(
            output_dir=directory, optim=optim_name, optim_args=optim_args,
            report_to=[],
        )
        return transformers.Trainer.get_optimizer_cls_and_kwargs(args)[1]


@pytest.mark.parametrize(
    "optim_name, optim_args, key",
    [("rmsprop", "alpha=0.95", "alpha"), ("rmsprop", "eps=1e-6", "eps"),
     ("adagrad", "eps=1e-7", "eps")],
)
def test_optim_args_hf_forwards_reach_the_mlx_optimizer(optim_name, optim_args, key):
    hf = _hf_kwargs(optim_name, optim_args)
    optimizer = _build_with_optim_args(optim_name, optim_args)
    expected = hf.get(key, float(optim_args.split("=")[1]))
    assert _hyperparameter(optimizer, key) == pytest.approx(expected)


@pytest.mark.parametrize(
    "optim_name, optim_args",
    [("rmsprop", "momentum=0.9"), ("rmsprop", "centered=True"),
     ("adagrad", "lr_decay=0.1")],
)
def test_optim_args_mlx_cannot_honour_raise(optim_name, optim_args):
    with pytest.raises(ValueError, match="does not support optim_args"):
        _build_with_optim_args(optim_name, optim_args)


def test_optim_args_at_noop_values_are_accepted():
    _build_with_optim_args("rmsprop", "momentum=0,centered=False")
    _build_with_optim_args("adagrad", "lr_decay=0.0")


@pytest.mark.parametrize(
    "alias", ["rmsprop_bnb", "rmsprop_bnb_8bit", "rmsprop_bnb_32bit"]
)
def test_hf_rmsprop_bnb_aliases_route_to_rmsprop(alias):
    from unsloth_zoo.mlx.trainer import _normalize_mlx_optimizer_name

    assert _normalize_mlx_optimizer_name(alias) == "rmsprop"
