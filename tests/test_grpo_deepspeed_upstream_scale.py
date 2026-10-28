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

"""Under DeepSpeed, upstream_scale must undo the 1/GAS division only when it happened: transformers >= 5.19 skips it if
loss_is_scaled_for_ga is True (TRL >= 1.15), otherwise multiplying by GAS scales grads up GAS times."""
import inspect
import types

import pytest
import torch
import transformers

import unsloth_zoo.rl_replacements as _rl

HIDDEN, VOCAB, GAS = 4, 8, 4
INPUT_IDS = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 3]])
MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])
_REAL_FORWARD = _rl.UnslothEfficientGRPO.forward


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _upstream_scale(monkeypatch, deepspeed = True, **trainer_attrs):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    seen = {}

    class _Loss:
        @staticmethod
        def apply(*args):
            seen["upstream_scale"] = inspect.signature(_REAL_FORWARD).bind(None, *args).arguments.get("upstream_scale")
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    model = _Model()
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None,
        ),
        is_deepspeed_enabled = deepspeed,
        use_vllm = False,
        _autocast_dtype = None,
        beta = 0.0,
        **trainer_attrs,
    )
    _rl.grpo_accumulated_loss(
        trainer, INPUT_IDS, (INPUT_IDS != 0).long(), 3, MASK,
        torch.zeros(INPUT_IDS.shape[0]), None, None,
        loss_type = "grpo", current_gradient_accumulation_steps = GAS,
    )
    return seen["upstream_scale"]


@pytest.fixture
def trainer_reads_the_flag(monkeypatch):
    monkeypatch.setattr(transformers.Trainer, "loss_is_scaled_for_ga", None, raising = False)


@pytest.fixture
def trainer_ignores_the_flag(monkeypatch):
    if hasattr(transformers.Trainer, "loss_is_scaled_for_ga"):
        monkeypatch.delattr(transformers.Trainer, "loss_is_scaled_for_ga")


def test_trl_1_15_grpo_on_transformers_5_19_is_not_scaled_up(monkeypatch, trainer_reads_the_flag):
    # TRL >= 1.15 GRPO: loss_is_scaled_for_ga = True, compute_loss_func left None; training_step does not divide.
    assert _upstream_scale(monkeypatch, loss_is_scaled_for_ga = True, compute_loss_func = None) == 1.0


def test_trainer_declaring_unscaled_loss_gets_gas_back(monkeypatch, trainer_reads_the_flag):
    assert _upstream_scale(monkeypatch, loss_is_scaled_for_ga = False, compute_loss_func = None) == float(GAS)


@pytest.mark.parametrize("flag", ["reads", "ignores"])
def test_trl_0_22_to_1_14_unchanged(monkeypatch, request, flag):
    # TRL 0.22 .. 1.14 GRPO passes a non-None compute_loss_func: training_step never divides.
    request.getfixturevalue(f"trainer_{flag}_the_flag")
    assert _upstream_scale(monkeypatch, compute_loss_func = "non-None value to disable scaling") == 1.0


@pytest.mark.parametrize("flag", ["reads", "ignores"])
def test_trl_0_21_unchanged(monkeypatch, request, flag):
    # TRL <= 0.21: compute_loss_func None and no flag, so training_step divides by GAS and it is undone.
    request.getfixturevalue(f"trainer_{flag}_the_flag")
    assert _upstream_scale(monkeypatch, compute_loss_func = None) == float(GAS)


def test_flag_ignored_when_this_transformers_does_not_read_it(monkeypatch, trainer_ignores_the_flag):
    # Older transformers: the class attribute alone changes nothing in training_step, so follow compute_loss_func.
    assert _upstream_scale(monkeypatch, loss_is_scaled_for_ga = True, compute_loss_func = None) == float(GAS)


def test_without_deepspeed_grad_output_is_ignored(monkeypatch, trainer_reads_the_flag):
    assert _upstream_scale(monkeypatch, deepspeed = False, loss_is_scaled_for_ga = True) is None
