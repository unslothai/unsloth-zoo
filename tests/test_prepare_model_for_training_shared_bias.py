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

"""CPU regression: every owner of a tied bias must cast the same float32 master.

named_parameters() deduplicates tied Parameters, so only the first owner of a tied
float16 bias used to get the forward cast; the second (BERT `cls.predictions.bias` is
`decoder.bias`, RoBERTa `lm_head.bias` is `decoder.bias`) then ran a Float bias against a
Half weight. Generic master behaviour (generate, compile, pickling, offload) is covered in
test_prepare_model_for_training_bias.py; this file only pins the tied-owner case."""
import pytest
import torch


@pytest.mark.parametrize("kind", ["bert", "roberta"])
def test_float16_shared_bias_owners(kind):
    from peft import LoraConfig, get_peft_model
    from transformers import BertConfig, BertForMaskedLM, RobertaConfig, RobertaForMaskedLM
    from unsloth_zoo.training_utils import prepare_model_for_training

    config_cls, model_cls = {
        "bert": (BertConfig, BertForMaskedLM),
        "roberta": (RobertaConfig, RobertaForMaskedLM),
    }[kind]
    torch.manual_seed(42)
    model = model_cls(config_cls(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )).to(torch.float16)
    model.config.torch_dtype = torch.float16
    model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=2, bias="all", target_modules=["query", "value"],
    ))
    model.eval()
    base = model.get_base_model()
    head = base.cls.predictions if kind == "bert" else base.lm_head
    master = head._parameters["bias"]
    assert master is head.decoder._parameters["bias"]
    keys = set(model.state_dict())
    input_ids = torch.tensor([[1, 7, 9, 11]])
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    # Before the fix this forward raised: decoder.bias is Float, decoder.weight Half.
    logits = model(input_ids=input_ids).logits
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    # Still one tied float32 Parameter under the same state_dict keys.
    assert master is head._parameters["bias"] is head.decoder._parameters["bias"]
    assert master.dtype == torch.float32
    assert sum(p is master for p in model.parameters()) == 1
    assert set(model.state_dict()) == keys

    torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1), input_ids.flatten(),
    ).backward()
    assert master.grad.dtype == torch.float32
    assert torch.isfinite(master.grad).all() and master.grad.abs().sum() > 0
