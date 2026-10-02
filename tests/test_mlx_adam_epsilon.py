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

"""adam_epsilon on the MLX trainer (HF TrainingArguments parity).

Asserts on the kwargs _build_optimizer passes, so it holds under real MLX
(bound at conftest preload) and under the torch simulation shim.
"""

from __future__ import annotations

import pytest
import torch  # noqa: F401  (the simulation shim runs MLX ops on torch)


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    """No teardown: later files hold references to the mlx modules."""
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _trainer(model=None, **config_kwargs):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel() if model is None else model
    trainer.args = MLXTrainingConfig(**config_kwargs)
    trainer._distributed_is_main_process = True
    return trainer


# Every constructor _build_optimizer can reach; an unlisted new branch fails _forwarded.
_OPTIMIZER_CONSTRUCTORS = {
    "unsloth_zoo.mlx.trainer": (
        "optim",
        ("Adafactor", "AdamW", "Adam", "SGD", "Muon", "Lion"),
    ),
    "unsloth_zoo.mlx.optimizers_quantized": (
        None,
        ("QuantizedMomentAdam", "QuantizedMomentAdamW"),
    ),
}


def _forwarded(monkeypatch, optim_name, model=None, **config_kwargs):
    """Return ``(constructor_name, kwargs)`` of the optimizer the trainer builds."""
    import importlib

    recorded = []

    def _recorder(name):
        def _record(*args, **kwargs):
            recorded.append((name, args, kwargs))
            return object()
        return _record

    # Import before patching: optimizers_quantized subclasses optim.Adam at import.
    targets = []
    for module_path, (attribute, constructors) in _OPTIMIZER_CONSTRUCTORS.items():
        module = importlib.import_module(module_path)
        targets.append((getattr(module, attribute) if attribute else module, constructors))
    for target, constructors in targets:
        for constructor in constructors:
            monkeypatch.setattr(target, constructor, _recorder(constructor))

    _trainer(
        model=model, optim=optim_name, **config_kwargs,
    )._build_optimizer(total_steps=4)
    assert len(recorded) == 1, f"expected one optimizer, recorded {recorded}"
    name, args, kwargs = recorded[0]
    assert not args, f"{name} was given positional arguments: {args}"
    return name, kwargs


_EPSILON_TAKING_OPTIMIZERS = ["adamw", "adam", "adamw_8bit", "adam_8bit"]
# HF spellings _normalize_mlx_optimizer_name collapses onto "adamw".
_ADAMW_ALIASES = [
    "adamw_torch",
    "adamw_torch_fused",
    "paged_adamw_32bit",
    "adamw_hf",
    "adamw_bnb_8bit",
    "paged_adamw_8bit",
]


def test_config_accepts_adam_epsilon():
    """Used to raise TypeError: unexpected arguments: adam_epsilon."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    assert MLXTrainingConfig(adam_epsilon=1e-6).adam_epsilon == pytest.approx(1e-6)
    assert MLXTrainingConfig().adam_epsilon is None


@pytest.mark.parametrize("optim_name", _EPSILON_TAKING_OPTIMIZERS)
def test_adam_epsilon_reaches_the_optimizer(monkeypatch, optim_name):
    _name, kwargs = _forwarded(monkeypatch, optim_name, adam_epsilon=1e-6)
    assert kwargs["eps"] == pytest.approx(1e-6)


@pytest.mark.parametrize("optim_name", _ADAMW_ALIASES)
def test_hf_optimizer_aliases_still_receive_the_epsilon(monkeypatch, optim_name):
    name, kwargs = _forwarded(monkeypatch, optim_name, adam_epsilon=1e-6)
    assert kwargs["eps"] == pytest.approx(1e-6), name


@pytest.mark.parametrize("optim_name", _EPSILON_TAKING_OPTIMIZERS)
def test_the_optimizer_accepts_the_epsilon_it_is_given(optim_name):
    """Build the real optimizer so a kwarg MLX rejects fails here."""
    optimizer = _trainer(
        optim=optim_name, adam_epsilon=1e-6,
    )._build_optimizer(total_steps=4)
    # Real MLX exposes .eps; the shim keeps kwargs in _kw.
    stored = getattr(optimizer, "eps", None)
    if stored is None:
        stored = optimizer._kw["eps"]
    assert stored == pytest.approx(1e-6)


@pytest.mark.parametrize("optim_name", _EPSILON_TAKING_OPTIMIZERS)
def test_unset_epsilon_forwards_no_eps_kwarg(monkeypatch, optim_name):
    _name, kwargs = _forwarded(monkeypatch, optim_name)
    assert "eps" not in kwargs


def test_the_mlx_default_epsilon_is_the_hf_default():
    import inspect

    from transformers import TrainingArguments

    # The module the trainer bound, not sys.modules["mlx.optimizers"].
    from unsloth_zoo.mlx.trainer import optim

    hf_default = TrainingArguments.__dataclass_fields__["adam_epsilon"].default
    checked = 0
    for constructor in (optim.Adam, optim.AdamW):
        parameter = inspect.signature(constructor).parameters.get("eps")
        if parameter is None:
            continue
        assert parameter.default == pytest.approx(hf_default), constructor
        checked += 1
    if not checked:
        pytest.skip(reason="simulation shim takes **kw, so it has no default eps to compare")


def test_epsilon_composes_with_betas(monkeypatch):
    _name, kwargs = _forwarded(
        monkeypatch, "adamw", adam_beta1=0.85, adam_beta2=0.95, adam_epsilon=1e-7,
    )
    assert kwargs["eps"] == pytest.approx(1e-7)
    assert kwargs["betas"] == (pytest.approx(0.85), pytest.approx(0.95))


def test_lion_receives_the_betas(monkeypatch):
    """HF builds Lion with betas=(adam_beta1, adam_beta2), no eps."""
    name, kwargs = _forwarded(
        monkeypatch, "lion", adam_beta1=0.95, adam_beta2=0.98, adam_epsilon=1e-6,
    )
    assert name == "Lion"
    assert kwargs["betas"] == (pytest.approx(0.95), pytest.approx(0.98))
    assert "eps" not in kwargs


def test_lion_fills_an_unset_beta_with_its_own_default(monkeypatch):
    _name, kwargs = _forwarded(monkeypatch, "lion", adam_beta2=0.98)
    assert kwargs["betas"] == (pytest.approx(0.9), pytest.approx(0.98))
    _name, kwargs = _forwarded(monkeypatch, "lion", adam_beta1=0.95)
    assert kwargs["betas"] == (pytest.approx(0.95), pytest.approx(0.99))


def test_unset_betas_keep_the_lion_default(monkeypatch):
    _name, kwargs = _forwarded(monkeypatch, "lion")
    assert "betas" not in kwargs


def test_the_lion_optimizer_keeps_the_betas_it_is_given():
    """Build the real optimizer so a kwarg MLX rejects fails here."""
    optimizer = _trainer(
        optim="lion", adam_beta1=0.95, adam_beta2=0.98,
    )._build_optimizer(total_steps=4)
    stored = getattr(optimizer, "betas", None)
    if stored is None:
        stored = optimizer._kw["betas"]
    assert tuple(stored) == (pytest.approx(0.95), pytest.approx(0.98))


@pytest.mark.parametrize("optim_name", ["sgd", "muon", "lion"])
def test_epsilon_is_not_forwarded_to_epsilon_free_optimizers(monkeypatch, optim_name):
    _name, kwargs = _forwarded(monkeypatch, optim_name, adam_epsilon=1e-6)
    assert "eps" not in kwargs


def test_adafactor_does_not_receive_the_scalar_epsilon(monkeypatch):
    """MLX Adafactor's eps is a 2-tuple, not HF's scalar."""
    _name, kwargs = _forwarded(monkeypatch, "adafactor", adam_epsilon=1e-6)
    assert "eps" not in kwargs


def _rank3_model():
    """rank>2 trainable parameter: Adafactor falls back to AdamW."""
    import types

    parameter = types.SimpleNamespace(ndim=3, shape=(2, 2, 2))
    return types.SimpleNamespace(
        trainable_parameters=lambda: {"vision.patch_embed.weight": parameter}
    )


def test_the_adafactor_fallback_carries_the_epsilon(monkeypatch):
    name, kwargs = _forwarded(
        monkeypatch, "adafactor", model=_rank3_model(), adam_epsilon=1e-6,
    )
    assert name == "AdamW"
    assert kwargs["eps"] == pytest.approx(1e-6)


def test_the_adafactor_fallback_rejects_a_bad_epsilon():
    with pytest.raises(ValueError, match="adam_epsilon"):
        _trainer(
            model=_rank3_model(), optim="adafactor", adam_epsilon=-1.0,
        )._build_optimizer(total_steps=4)


@pytest.mark.parametrize("optim_name", ["sgd", "muon", "lion", "adafactor"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), "not-a-number"])
def test_epsilon_free_optimizers_ignore_an_invalid_epsilon(
    monkeypatch, optim_name, bad,
):
    """HF never passes adam_epsilon to these, so it must not be validated."""
    _name, kwargs = _forwarded(monkeypatch, optim_name, adam_epsilon=bad)
    assert "eps" not in kwargs


@pytest.mark.parametrize("bad", [-1e-8, -1.0, float("nan"), "not-a-number"])
def test_epsilon_values_pytorch_rejects_are_rejected_here(bad):
    """torch.optim.Adam rejects `not 0.0 <= eps`; MLX would train on garbage."""
    with pytest.raises(ValueError, match="adam_epsilon"):
        _trainer(optim="adamw", adam_epsilon=bad)._build_optimizer(total_steps=4)


@pytest.mark.parametrize("good", [0.0, 1e-8, 1e-30, float("inf"), "1e-6"])
def test_epsilon_values_pytorch_accepts_are_accepted_here(monkeypatch, good):
    """Numeric strings come from JSON round-tripped configs."""
    _name, kwargs = _forwarded(monkeypatch, "adamw", adam_epsilon=good)
    assert kwargs["eps"] == pytest.approx(float(good))


def test_the_epsilon_guard_covers_non_mlx_config_objects():
    """HF TrainingArguments or a bare namespace reach the same check."""
    import types

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = types.SimpleNamespace(trainable_parameters=lambda: {})
    trainer._distributed_is_main_process = True
    defaults = MLXTrainingConfig()
    trainer.args = types.SimpleNamespace(
        **{**vars(defaults), "optim": "adamw", "adam_epsilon": -1.0}
    )
    with pytest.raises(ValueError, match="adam_epsilon"):
        trainer._build_optimizer(total_steps=4)


def test_appended_field_stays_an_exact_suffix():
    """Positional binding needs the copy fields to be a suffix of the positional
    (non kw_only) fields, preference config subclasses included."""
    from dataclasses import fields as dataclass_fields
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig,
        MLXORPOConfig,
        MLXTrainingConfig,
        _MLX_CONFIG_OPTIONAL_COPY_FIELDS,
    )

    assert "adam_epsilon" in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    for config_class in (MLXTrainingConfig, MLXORPOConfig, MLXDPOConfig):
        names = [
            f.name
            for f in dataclass_fields(config_class)
            if f.init and not f.kw_only
        ]
        tail = tuple(names[-len(_MLX_CONFIG_OPTIONAL_COPY_FIELDS):])
        assert tail == _MLX_CONFIG_OPTIONAL_COPY_FIELDS, config_class.__name__


def test_the_preference_trainers_share_the_one_entry_point():
    """ORPO / DPO inherit _build_optimizer, so one guard covers every objective."""
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig,
        MLXDPOTrainer,
        MLXORPOConfig,
        MLXORPOTrainer,
        MLXTrainer,
    )

    for trainer_class in (MLXORPOTrainer, MLXDPOTrainer):
        assert (
            trainer_class._build_optimizer is MLXTrainer._build_optimizer
        ), trainer_class.__name__
    for config_class in (MLXORPOConfig, MLXDPOConfig):
        assert config_class(adam_epsilon=1e-6).adam_epsilon == pytest.approx(1e-6)
        assert config_class().adam_epsilon is None


def test_a_pre_pr_positional_config_copy_still_maps(monkeypatch):
    """An old positional config dump (no adam_epsilon) keeps every slot."""
    from dataclasses import fields as dataclass_fields
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    names = [
        f.name
        for f in dataclass_fields(MLXTrainingConfig)
        if f.init and not f.kw_only
    ]
    original = MLXTrainingConfig(optim="adam", learning_rate=1.5e-4, run_name="old")
    pre_pr = [getattr(original, name) for name in names if name != "adam_epsilon"]
    copied = MLXTrainingConfig(*pre_pr)

    assert copied.optim == "adam"
    assert copied.learning_rate == pytest.approx(1.5e-4)
    assert copied.run_name == "old"
    assert copied.adam_epsilon is None
    _name, kwargs = _forwarded(monkeypatch, "adam")
    assert "eps" not in kwargs
