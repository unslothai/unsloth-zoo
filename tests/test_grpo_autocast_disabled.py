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

"""grpo_accumulated_loss must not autocast when Unsloth turned autocast off (float32 training on a T4 / V100)."""
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as _rl

HIDDEN, VOCAB = 4, 8
INPUT_IDS = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 3]])


class _Model(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")
        self.seen = seen

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        self.seen.append(torch.is_autocast_enabled("cpu"))
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _autocast_seen(monkeypatch, **trainer_attrs):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    seen = []

    class _Loss:
        @staticmethod
        def apply(*args):
            seen.append(torch.is_autocast_enabled("cpu"))
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = _Model(seen),
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False,
        beta = 0.0,
        **trainer_attrs,
    )
    _rl.grpo_accumulated_loss(
        trainer, INPUT_IDS, torch.ones_like(INPUT_IDS), 3, torch.ones(2, 3, dtype = torch.long),
        torch.zeros(2), None, None, loss_type = "dapo", num_items_in_batch = 6,
    )
    assert seen, "nothing ran under the autocaster"
    return any(seen)


def test_autocast_off_is_honoured_even_with_a_dtype_set(monkeypatch):
    # rl_replacements.py _unsloth_grpo_autocast: precision "no" -> _autocast_enabled False, _autocast_dtype bfloat16.
    assert not _autocast_seen(monkeypatch, _autocast_dtype = torch.bfloat16, _autocast_enabled = False)


@pytest.mark.parametrize("trainer_attrs", [
    {"_autocast_dtype": torch.bfloat16, "_autocast_enabled": True},
    {"_autocast_dtype": torch.bfloat16},  # an older unsloth that never sets _autocast_enabled
])
def test_autocast_on_still_autocasts(monkeypatch, trainer_attrs):
    assert _autocast_seen(monkeypatch, **trainer_attrs)


def test_no_dtype_means_no_autocast(monkeypatch):
    assert not _autocast_seen(monkeypatch, _autocast_dtype = None)


@pytest.mark.parametrize("precision, expected", [("no", False), ("bf16", True), ("fp16", True), ("fp8", True)])
def test_fallback_follows_accelerate_mixed_precision(monkeypatch, precision, expected):
    # Unsloth's compute_loss reaches grpo_accumulated_loss before it latches _autocast_dtype: the env decides.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", precision)
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    assert _autocast_seen(monkeypatch) is expected
