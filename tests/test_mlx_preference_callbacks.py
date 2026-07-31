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

"""Regression test: the preference trainers must accept ``callbacks=``.

MLXTrainer takes ``callbacks`` and has no ``**kwargs``, so the ORPO/DPO
subclasses -- which redeclare the full signature -- dropped it and raised
TypeError on a call that works for the base trainer and for TRL.
"""

from __future__ import annotations

import inspect
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


def _trainer_classes():
    from unsloth_zoo.mlx.trainer import MLXDPOTrainer, MLXORPOTrainer
    return {"orpo": MLXORPOTrainer, "dpo": MLXDPOTrainer}


ROWS = [{"prompt": "q ", "chosen": "good", "rejected": "bad"}]


class _Tok:
    eos_token_id = 7
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [(ord(ch) % 90) + 8 for ch in text]


@pytest.mark.parametrize("name", ["orpo", "dpo"])
def test_signature_accepts_callbacks(name):
    cls = _trainer_classes()[name]
    assert "callbacks" in inspect.signature(cls.__init__).parameters


@pytest.mark.parametrize("name", ["orpo", "dpo"])
def test_signature_matches_base_trainer(name):
    from unsloth_zoo.mlx.trainer import MLXTrainer

    base = list(inspect.signature(MLXTrainer.__init__).parameters)
    sub = list(inspect.signature(_trainer_classes()[name].__init__).parameters)
    assert sub == base, f"{name} signature drifted from MLXTrainer"


@pytest.mark.parametrize("name", ["orpo", "dpo"])
def test_constructing_with_callbacks_does_not_raise_typeerror(name):
    """The reported failure: passing callbacks= raised TypeError."""
    cls = _trainer_classes()[name]

    class _Cb:
        pass

    try:
        cls(model=None, tokenizer=_Tok(), train_dataset=ROWS, callbacks=[_Cb()])
    except TypeError as exc:
        if "callbacks" in str(exc):
            pytest.fail(f"{name} still rejects callbacks=: {exc}")
    except Exception:
        pass  # later construction failures are not this test's concern


class _Model:
    """Minimal stand-in: enough surface for MLXTrainer.__init__."""

    def trainable_parameters(self):
        return {}

    def parameters(self):
        return {}

    def named_modules(self):
        return []

    def train(self, *args, **kwargs):
        return self

    def eval(self, *args, **kwargs):
        return self


@pytest.mark.parametrize("name", ["orpo", "dpo"])
def test_callback_is_registered_on_the_handler(name):
    """The callback must reach the handler, not merely be accepted."""
    cls = _trainer_classes()[name]

    class _Cb:
        pass

    cb = _Cb()
    trainer = cls(model=_Model(), tokenizer=_Tok(), train_dataset=ROWS,
                  callbacks=[cb])
    registered = getattr(trainer.callback_handler, "callbacks", ())
    assert any(c is cb for c in registered), "callback did not reach the handler"
