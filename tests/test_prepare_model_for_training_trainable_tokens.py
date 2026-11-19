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

"""prepare_model_for_training must keep PEFT's trainable_token_indices rows (trainable_tokens_delta) trainable."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

pytest.importorskip("peft")

TOKENS = [5, 17, 60]


def _tiny_llama(tie):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        tie_word_embeddings = tie,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    model.config.torch_dtype = torch.bfloat16
    return model


def _lora_config(tokens = TOKENS):
    from peft import LoraConfig

    return LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"],
                      trainable_token_indices = tokens)


def _prepare(model):
    from unsloth_zoo.training_utils import prepare_model_for_training

    return prepare_model_for_training(
        model, use_gradient_checkpointing = False, use_reentrant = False,
        full_finetuning = False, patch_modules_to_save = True,
    )


def _deltas(model):
    params = {n: p for n, p in model.named_parameters() if ".trainable_tokens_delta." in n}
    assert params, "PEFT should add a trainable_tokens_delta"
    return params


def _rows(model):
    return model.get_input_embeddings().token_adapter.get_merged_weights(["default"])[TOKENS].float().clone()


@pytest.mark.parametrize("tie", [False, True], ids = ["untied", "tied"])
def test_trainable_tokens_stay_trainable_and_train(tie):
    from peft import get_peft_model

    model = get_peft_model(_tiny_llama(tie), _lora_config())
    assert all(p.requires_grad for p in _deltas(model).values())
    _prepare(model)
    for n, p in _deltas(model).items():
        assert p.requires_grad, f"{n} frozen by prepare_model_for_training"
        assert p.dtype == torch.float32, (n, p.dtype)
    base = model.get_input_embeddings().token_adapter.base_layer.weight
    assert not base.requires_grad

    before = _rows(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr = 1e-2, weight_decay = 0.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        input_ids = torch.randint(0, 64, (2, 12), generator = g)
        input_ids[:, ::4] = torch.tensor(TOKENS)
        model(input_ids = input_ids, labels = input_ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    moved = (_rows(model) - before).abs().max().item()
    assert moved > 1e-3, f"trainable_token_indices rows did not train: max |change| = {moved}"


def test_inactive_trainable_tokens_adapter_stays_frozen():
    from peft import get_peft_model

    model = get_peft_model(_tiny_llama(tie = False), _lora_config())
    model.add_adapter("other", _lora_config([30, 31]))
    _prepare(model)
    flags = {n.rsplit(".", 1)[-1]: p.requires_grad for n, p in _deltas(model).items()}
    assert flags == {"default": True, "other": False}, flags
