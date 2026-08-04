# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""adam_epsilon on the MLX trainer (HF TrainingArguments parity).

HF exposes ``adam_epsilon`` (transformers/training_args.py:919, default 1e-8)
and forwards it to the optimizer as ``eps`` alongside the betas
(transformers/trainer.py:1397-1400). MLX carried ``adam_beta1`` /
``adam_beta2`` but not the epsilon, and because ``MLXTrainingConfig`` rejects
unknown kwargs, passing it raised ``TypeError`` before training began rather
than being quietly ignored.

CPU-pure: no weights, no Metal, no optimizer step. The assertions are made
against the kwargs ``_build_optimizer`` hands the optimizer constructor, which
is the contract this feature owns, rather than against any private attribute of
whichever MLX implementation happens to be importable. That matters: a host with
real MLX installed (every Apple Silicon dev machine, and the Linux mlx-cpu CI
jobs) imports ``unsloth_zoo.mlx`` -- and therefore binds the real
``mlx.optimizers`` into ``unsloth_zoo.mlx.trainer`` -- during conftest's package
preload, long before a module fixture could install the simulation shim.
"""

from __future__ import annotations

import pytest
import torch  # noqa: F401  (the simulation shim runs MLX ops on torch)


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    """Install-only, matching tests/test_mlx_double_quant_reject.py. Deliberately
    no teardown: popping mlx / unsloth_zoo.mlx modules on the way out breaks
    later files that hold references to them (this file sorts before
    test_mlx_generate.py, which does). A no-op where real MLX already won."""
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _trainer(**config_kwargs):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(**config_kwargs)
    # is_main_process is a property over this attribute; the 8-bit branch reads
    # it to decide whether to print its one-line banner.
    trainer._distributed_is_main_process = True
    return trainer


# Every constructor _build_optimizer can reach, and the module each is looked up
# on at call time. Keeping the list explicit is the point: a new optimizer branch
# that this file does not name shows up as a KeyError in _forwarded rather than
# silently escaping the "reaches nothing else" assertions.
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


def _forwarded(monkeypatch, optim_name, **config_kwargs):
    """Return ``(constructor_name, kwargs)`` for the optimizer the trainer builds.

    Records instead of constructing, so the assertions hold identically under
    real MLX and under the torch simulation shim.
    """
    import importlib

    recorded = []

    def _recorder(name):
        def _record(*args, **kwargs):
            recorded.append((name, args, kwargs))
            return object()
        return _record

    # Import every target BEFORE patching any of them: optimizers_quantized
    # subclasses optim.Adam at module scope, so importing it after optim.Adam has
    # been replaced by a recorder function is a metaclass conflict, not a test.
    targets = []
    for module_path, (attribute, constructors) in _OPTIMIZER_CONSTRUCTORS.items():
        module = importlib.import_module(module_path)
        targets.append((getattr(module, attribute) if attribute else module, constructors))
    for target, constructors in targets:
        for constructor in constructors:
            monkeypatch.setattr(target, constructor, _recorder(constructor))

    _trainer(optim=optim_name, **config_kwargs)._build_optimizer(total_steps=4)
    assert len(recorded) == 1, f"expected one optimizer, recorded {recorded}"
    name, args, kwargs = recorded[0]
    assert not args, f"{name} was given positional arguments: {args}"
    return name, kwargs


# adamw_8bit / adam_8bit route to the quantized Adam family, which takes the same
# scalar eps and hands it straight to the stock MLX parent
# (unsloth_zoo/mlx/optimizers_quantized.py:194-210, :215-232).
_EPSILON_TAKING_OPTIMIZERS = ["adamw", "adam", "adamw_8bit", "adam_8bit"]
# _normalize_mlx_optimizer_name collapses these HF spellings onto "adamw", so a
# user who copies a CUDA config verbatim must still get the epsilon they asked
# for. Six entry points, one guard: this is the class, not one instance of it.
_ADAMW_ALIASES = [
    "adamw_torch",
    "adamw_torch_fused",
    "paged_adamw_32bit",
    "adamw_hf",
    "adamw_bnb_8bit",
    "paged_adamw_8bit",
]


def test_config_accepts_adam_epsilon():
    """The regression itself: MLXTrainingConfig validates its kwargs, so
    adam_epsilon=... used to raise TypeError before a step ever ran."""
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
    """The recorder above proves what is passed, not that MLX takes it. Build the
    real thing once per branch so a kwarg MLX does not accept is a failure here
    and not a TypeError in front of a user."""
    optimizer = _trainer(
        optim=optim_name, adam_epsilon=1e-6,
    )._build_optimizer(total_steps=4)
    # Real MLX publishes it (mlx/optimizers/optimizers.py:504); the simulation
    # shim keeps constructor kwargs in _kw.
    stored = getattr(optimizer, "eps", None)
    if stored is None:
        stored = optimizer._kw["eps"]
    assert stored == pytest.approx(1e-6)


@pytest.mark.parametrize("optim_name", _EPSILON_TAKING_OPTIMIZERS)
def test_unset_epsilon_forwards_no_eps_kwarg(monkeypatch, optim_name):
    """None means "not requested": nothing is forwarded, so MLX's own default
    stands and the unset path is bitwise unchanged."""
    _name, kwargs = _forwarded(monkeypatch, optim_name)
    assert "eps" not in kwargs


def test_the_mlx_default_epsilon_is_the_hf_default():
    """What "leave the default alone" is worth depends on the two defaults
    agreeing. Read both, do not assume either."""
    import inspect

    from transformers import TrainingArguments

    # The module the trainer actually calls, not sys.modules["mlx.optimizers"]:
    # the shim can be installed in sys.modules while the trainer still holds the
    # real module it bound at import time.
    from unsloth_zoo.mlx.trainer import optim

    hf_default = TrainingArguments.__dataclass_fields__["adam_epsilon"].default
    checked = 0
    for constructor in (optim.Adam, optim.AdamW):
        parameter = inspect.signature(constructor).parameters.get("eps")
        if parameter is None:
            # The simulation shim takes **kw and models no defaults.
            continue
        assert parameter.default == pytest.approx(hf_default), constructor
        checked += 1
    if not checked:
        pytest.skip("the simulation shim does not model MLX's default eps")


def test_epsilon_composes_with_betas(monkeypatch):
    _name, kwargs = _forwarded(
        monkeypatch, "adamw", adam_beta1=0.85, adam_beta2=0.95, adam_epsilon=1e-7,
    )
    assert kwargs["eps"] == pytest.approx(1e-7)
    assert kwargs["betas"] == (pytest.approx(0.85), pytest.approx(0.95))


@pytest.mark.parametrize("optim_name", ["sgd", "muon", "lion"])
def test_epsilon_is_not_forwarded_to_epsilon_free_optimizers(monkeypatch, optim_name):
    """SGD/Muon/Lion accept no epsilon, so forwarding one would be a TypeError.
    HF ignores adam_epsilon for these too, so setting it stays harmless."""
    _name, kwargs = _forwarded(monkeypatch, optim_name, adam_epsilon=1e-6)
    assert "eps" not in kwargs


def test_adafactor_does_not_receive_the_scalar_epsilon(monkeypatch):
    """MLX's Adafactor eps is a 2-tuple (mlx/optimizers/optimizers.py:744,
    default (1e-30, 1e-3)) with different meaning from HF's scalar, so
    adam_epsilon must not leak into it. The CUDA path scopes it the same way."""
    _name, kwargs = _forwarded(monkeypatch, "adafactor", adam_epsilon=1e-6)
    assert "eps" not in kwargs


def test_appended_field_stays_an_exact_suffix():
    """MLXTrainingConfig binds positional args by field order, so a new field
    must be appended last and keep _MLX_CONFIG_OPTIONAL_COPY_FIELDS a suffix of
    the POSITIONAL field list; inserting mid-list would shift every later field's
    slot. The preference configs add kw_only fields after adam_epsilon, and
    __init__ filters those out before zipping, so the invariant is over
    `not field.kw_only` and has to hold for the subclasses too."""
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
        # Deliberately not `names[-1] == "adam_epsilon"`: the convention is that
        # each new field is appended and registered here, so pinning this field
        # as the permanent last one would fire on the next unrelated addition.
        # The suffix property below is the invariant that actually has to hold.
        tail = tuple(names[-len(_MLX_CONFIG_OPTIONAL_COPY_FIELDS):])
        assert tail == _MLX_CONFIG_OPTIONAL_COPY_FIELDS, config_class.__name__


def test_a_pre_pr_positional_config_copy_still_maps(monkeypatch):
    """An old config dump has no adam_epsilon. Copying it positionally must land
    every value in the same slot it came from and leave the epsilon unset."""
    from dataclasses import fields as dataclass_fields
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    names = [
        f.name
        for f in dataclass_fields(MLXTrainingConfig)
        if f.init and not f.kw_only
    ]
    original = MLXTrainingConfig(optim="adam", learning_rate=1.5e-4, run_name="old")
    # Everything the pre-PR surface had: the full positional list minus the
    # field this PR appended.
    pre_pr = [getattr(original, name) for name in names if name != "adam_epsilon"]
    copied = MLXTrainingConfig(*pre_pr)

    assert copied.optim == "adam"
    assert copied.learning_rate == pytest.approx(1.5e-4)
    assert copied.run_name == "old"
    assert copied.adam_epsilon is None
    _name, kwargs = _forwarded(monkeypatch, "adam")
    assert "eps" not in kwargs
