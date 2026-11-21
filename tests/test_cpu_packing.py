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

"""CPU packing / padding_free where transformers keeps packed examples apart."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _tiny_model():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32, attn_implementation = "sdpa")
    except Exception as error:  # offline without the tiny model cached
        pytest.skip(f"{_TINY} unavailable: {error}")


def test_cpu_packing_supported_and_opt_out(monkeypatch):
    from unsloth_zoo.cpu import trainer
    trainer.cpu_packing_supported.cache_clear()
    from transformers import masking_utils
    expected = hasattr(masking_utils, "find_packed_sequence_indices") and \
        getattr(masking_utils, "_is_torch_greater_or_equal_than_2_6", False)
    assert trainer.cpu_packing_supported() == expected
    monkeypatch.setenv("UNSLOTH_CPU_PACKING", "0")
    trainer.cpu_packing_supported.cache_clear()
    try:
        assert trainer.cpu_packing_supported() is False

        class Args:
            packing = True
            padding_free = True
            eval_packing = True
        args = trainer.normalize_cpu_training_arguments(Args())
        assert (args.packing, args.padding_free, args.eval_packing) == (False, False, False)
    finally:
        monkeypatch.delenv("UNSLOTH_CPU_PACKING")
        trainer.cpu_packing_supported.cache_clear()


def test_wrapped_packing_still_turned_off():
    from unsloth_zoo.cpu import trainer
    trainer.cpu_packing_supported.cache_clear()

    class Args:
        packing = True
        packing_strategy = "wrapped"
        padding_free = False
    args = trainer.normalize_cpu_training_arguments(Args())
    assert args.packing is False


def test_packed_probe_flips_with_mask_support(monkeypatch):
    from unsloth_zoo.cpu import trainer
    if not trainer.cpu_packing_supported():
        pytest.skip("transformers / torch too old for packed SDPA masks")
    model = _tiny_model()
    model.train()
    assert trainer._packed_examples_isolated(model) is True
    assert model.training  # restored
    # A model whose mask ignores the packed position_ids attends across examples: the probe must say so.
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    monkeypatch.setattr(qwen3, "create_causal_mask", lambda *a, **k: None)
    assert trainer._packed_examples_isolated(model) is False

    class Args:
        packing = False
        padding_free = True
        eval_packing = False
    args = Args()
    assert trainer._check_cpu_packing(model, args) is False
    assert args.padding_free is False


def test_padding_free_token_loss_matches_padded():
    """Same examples, padded vs padding-free: the summed token loss and its gradient agree (the mean's divisor is
    TRL's count, which in padded mode also counts each row's unshifted first label)."""
    from unsloth_zoo.cpu import trainer
    if not trainer.cpu_packing_supported():
        pytest.skip("transformers / torch too old for packed SDPA masks")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    model = _tiny_model()
    model.config.use_cache = False
    generator = torch.Generator().manual_seed(0)
    examples = [{"input_ids": torch.randint(3, 1000, (n,), generator = generator).tolist()} for n in (9, 14, 5)]
    sums, grads = [], []
    for padding_free in (False, True):
        batch = DataCollatorForLanguageModeling(pad_token_id = 0, padding_free = padding_free)(examples)
        labels = batch.pop("labels")
        model.zero_grad()
        logits = model(**batch, use_cache = False).logits.float()
        shift = labels[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), shift,
                                                 ignore_index = -100, reduction = "sum")
        loss.backward()
        sums.append(loss.detach())
        grads.append(model.model.layers[0].self_attn.q_proj.weight.grad.clone())
    torch.testing.assert_close(sums[0], sums[1], rtol = 1e-5, atol = 1e-4)
    torch.testing.assert_close(grads[0], grads[1], rtol = 1e-4, atol = 1e-6)


@pytest.mark.parametrize("padding_free, packing, strategy, kept", [
    (False, False, "bfd", False),  # eval-only packing: TRL's collator is not padding-free, no position_ids resets
    (True, False, "bfd", True),
    (False, True, "bfd", True),
    (False, True, "bfd_split", True),
])
def test_eval_packing_needs_a_padding_free_collator(monkeypatch, padding_free, packing, strategy, kept):
    from unsloth_zoo.cpu import trainer
    monkeypatch.setattr(trainer, "cpu_packing_supported", lambda: True)

    class Args:
        pass
    args = Args()
    args.padding_free, args.packing, args.packing_strategy, args.eval_packing = padding_free, packing, strategy, True
    args = trainer.normalize_cpu_training_arguments(args)
    assert args.eval_packing is kept
    assert (args.padding_free, args.packing) == (padding_free, packing)


def test_eval_packing_collator_really_has_no_resets():
    """The reason for the rule above: TRL's collator without padding_free emits an attention mask, no position_ids."""
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    batch = DataCollatorForLanguageModeling(pad_token_id = 0, padding_free = False)(
        [{"input_ids": [5, 6, 7, 8, 9], "seq_lengths": [3, 2]}]
    )
    assert "position_ids" not in batch and "attention_mask" in batch


@pytest.mark.parametrize("model", ["facebook/opt-125m", Path("/some/checkpoint")])
def test_string_model_cannot_skip_the_packing_probe(monkeypatch, model):
    from unsloth_zoo.cpu import trainer
    called = []
    monkeypatch.setattr(trainer, "_packed_examples_isolated", lambda m: called.append(m) or True)

    class Args:
        packing = True
        padding_free = True
        eval_packing = True
    args = Args()
    assert trainer._check_cpu_packing(model, args) is False
    assert (args.packing, args.padding_free, args.eval_packing) == (False, False, False)
    assert called == []
