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

"""CPU regression: every owner of a tied bias must cast the same float32 master."""
import io

import pytest
import torch


@pytest.mark.parametrize("kind", ["bert", "roberta_masked", "roberta_causal"])
def test_float16_shared_bias_owners(kind):
    from peft import LoraConfig, get_peft_model
    from transformers import (
        BertConfig, BertForMaskedLM, RobertaConfig,
        RobertaForMaskedLM, RobertaForCausalLM,
    )
    from unsloth_zoo.training_utils import prepare_model_for_training

    def make_model():
        config_cls = BertConfig if kind == "bert" else RobertaConfig
        config = config_cls(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=32, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            is_decoder=kind == "roberta_causal",
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        model_cls = {
            "bert": BertForMaskedLM,
            "roberta_masked": RobertaForMaskedLM,
            "roberta_causal": RobertaForCausalLM,
        }[kind]
        model = model_cls(config).to(torch.float16)
        model.config.torch_dtype = torch.float16
        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=2, bias="all", target_modules=["query", "value"],
        ))
        model.eval()
        base = model.get_base_model()
        head = base.cls.predictions if kind == "bert" else base.lm_head
        return model, head

    def prepare(model):
        prepare_model_for_training(
            model, use_gradient_checkpointing=False, use_reentrant=False,
            full_finetuning=False,
        )

    torch.manual_seed(42)
    model, head = make_model()
    input_ids = torch.tensor([[1, 7, 9, 11]])
    master = head._parameters["bias"]
    assert master is head.decoder._parameters["bias"]
    keys = set(model.state_dict())
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
        if kind == "roberta_causal":
            expected_tokens = model.generate(
                input_ids=input_ids, max_new_tokens=2, do_sample=False,
            )

    prepare(model)
    # This plain forward fails before the fix: decoder.bias is Float, weight Half.
    logits = model(input_ids=input_ids).logits
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert master is head._parameters["bias"]
    assert master is head.decoder._parameters["bias"]
    assert master.dtype == torch.float32
    assert sum(p is master for p in model.parameters()) == 1
    assert set(model.state_dict()) == keys

    loss = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1), input_ids.flatten(),
    )
    loss.backward()
    grad = master.grad.clone()
    assert grad.dtype == torch.float32
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    torch.nn.functional.cross_entropy(
        model(input_ids=input_ids).logits.float().flatten(0, 1), input_ids.flatten(),
    ).backward()
    torch.testing.assert_close(master.grad, 2 * grad, rtol=0, atol=0)
    assert head.decoder.bias.grad is master.grad

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored, restored_head = make_model()
    prepare(restored)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored_head.bias is restored_head.decoder.bias
    assert restored_head.bias.dtype == torch.float32
    with torch.no_grad():
        torch.testing.assert_close(
            restored(input_ids=input_ids).logits, expected, rtol=0, atol=0,
        )
    if kind == "roberta_causal":
        for prepared in (model, restored):
            tokens = prepared.generate(
                input_ids=input_ids, max_new_tokens=2, do_sample=False,
            )
            assert torch.equal(tokens, expected_tokens)
