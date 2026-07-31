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

"""Regression test: the preference builder reads the same dataset every other
builder does.

``_prepare_data`` binds ``train_dataset = self._train_dataset_for_batches()``,
but the preference branch passed ``self.train_dataset`` directly. Those two can
differ: the ``elif`` in ``MLXTrainer.__init__`` rebinds ``self.train_dataset``
to a ``_MLXTokenizedDatasetView`` without updating
``_mlx_train_dataset_for_batches``.

This is consistency, not a correctness fix. The view copies the row and only
ADDS ``input_ids``/``attention_mask``, so ``prompt``/``chosen``/``rejected``
survive it and both objects yield identical preference values -- verified in
``test_view_does_not_shadow_preference_columns`` below. What it buys is one
source of truth and no tokenization of a text/messages column the preference
path discards.
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


class _Tok:
    eos_token_id = 7
    bos_token = None
    # Needed only so the SFT view can render the messages column it tokenizes;
    # the preference path never uses it.
    chat_template = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}{% endfor %}"

    def encode(self, text, add_special_tokens=True):
        return [(ord(ch) % 90) + 8 for ch in text]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return "".join(f"{m['role']}:{m['content']}" for m in messages)


class _Model:
    _config = {"model_type": "llama"}

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


def _rows(n=4):
    """Preference rows that ALSO carry an SFT-shaped messages column."""
    return [
        {"prompt": f"q{i} ", "chosen": "good answer", "rejected": "bad answer",
         "messages": [{"role": "user", "content": f"hi {i}"}]}
        for i in range(n)
    ]


def _trainer():
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    return MLXORPOTrainer(
        model=_Model(), tokenizer=_Tok(), train_dataset=_rows(),
        args=MLXORPOConfig(per_device_train_batch_size=2, max_seq_length=64),
    )


def test_the_two_dataset_handles_really_can_differ():
    """Pin the divergence, so this test stays meaningful if it is ever fixed."""
    trainer = _trainer()
    assert trainer.train_dataset is not trainer._train_dataset_for_batches()
    assert type(trainer.train_dataset).__name__ == "_MLXTokenizedDatasetView"


def test_view_does_not_shadow_preference_columns():
    """The claimed shadowing does NOT occur; record why the change is safe."""
    trainer = _trainer()
    viewed = trainer.train_dataset[0]
    raw = trainer._train_dataset_for_batches()[0]

    for key in ("prompt", "chosen", "rejected"):
        assert viewed[key] == raw[key], f"{key} differs between the two handles"
    # The view only ADDS tokenization of the discarded messages column.
    assert "input_ids" in viewed and "input_ids" not in raw


def test_preference_builder_uses_the_batch_dataset_handle():
    """The call site must pass the _train_dataset_for_batches() binding."""
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._prepare_data)
    assert "train_dataset = self._train_dataset_for_batches()" in src
    assert "dataset=train_dataset," in src, (
        "preference builder must read the local binding"
    )
    assert "dataset=self.train_dataset," not in src, (
        "preference builder must not bypass _train_dataset_for_batches()"
    )
