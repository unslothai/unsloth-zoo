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

"""prepare_model_for_training must keep LoRA on nn.Embedding trainable.

PEFT stores LoRA for nn.Embedding as ParameterDicts named lora_embedding_A /
lora_embedding_B (e.g. "embed_tokens.lora_embedding_A.default"), not lora_A /
lora_B submodules. The LoRA filter only matched ".lora_A." / ".lora_B.", so any
embedding LoRA (target_modules including embed_tokens, or an nn.Embedding audio
tower as in Inkling) was frozen with no warning. Pure CPU, tiny Llama.
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

pytest.importorskip("peft")


def _tiny_lora_llama(dtype = torch.float32):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        tie_word_embeddings = False,
    )
    model = LlamaForCausalLM(cfg)
    if dtype != torch.float32:
        model = model.to(dtype)
    model.config.torch_dtype = dtype
    return get_peft_model(
        model,
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["embed_tokens", "q_proj", "v_proj"]),
    )


def _prepare(model):
    from unsloth_zoo.training_utils import prepare_model_for_training

    return prepare_model_for_training(
        model, use_gradient_checkpointing = False, use_reentrant = False,
        full_finetuning = False,
    )


def _embedding_lora(model):
    params = {n: p for n, p in model.named_parameters() if ".lora_embedding_" in n}
    assert len(params) == 2, f"expected lora_embedding_A/B, got {sorted(params)}"
    return params


def test_embedding_lora_stays_trainable():
    model = _tiny_lora_llama()
    assert all(p.requires_grad for p in _embedding_lora(model).values())
    _prepare(model)
    frozen = [n for n, p in _embedding_lora(model).items() if not p.requires_grad]
    assert not frozen, f"embedding LoRA frozen by prepare_model_for_training: {frozen}"
    # Linear LoRA is still trainable and the base embedding is still frozen.
    params = dict(model.named_parameters())
    assert any(p.requires_grad for n, p in params.items() if ".lora_A." in n)
    base = next(n for n in params if n.endswith("embed_tokens.base_layer.weight"))
    assert not params[base].requires_grad


def test_embedding_lora_receives_gradient():
    model = _tiny_lora_llama()
    _prepare(model)
    input_ids = torch.randint(0, 64, (2, 8))
    # Loss from logits, not labels=, so the test stays CPU-only.
    logits = model(input_ids = input_ids).logits.float()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), input_ids.view(-1))
    loss.backward()
    emb = _embedding_lora(model)
    for n, p in emb.items():
        assert p.grad is not None, f"{n} got no gradient"
        assert torch.isfinite(p.grad).all(), n
    # PEFT zero-inits lora_embedding_A (not B, unlike Linear LoRA), so on the first
    # step only A has a non-zero gradient.
    a = next(p for n, p in emb.items() if ".lora_embedding_A." in n)
    assert a.grad.abs().sum() > 0


def test_embedding_lora_upcast_like_linear_lora():
    """bf16 model: embedding LoRA is upcast to float32 like lora_A / lora_B and
    the forward still runs."""
    model = _tiny_lora_llama(dtype = torch.bfloat16)
    _prepare(model)
    params = dict(model.named_parameters())
    lora_a = next(p for n, p in params.items() if ".lora_A." in n)
    for n, p in _embedding_lora(model).items():
        assert p.requires_grad, n
        assert p.dtype == lora_a.dtype == torch.float32, (n, p.dtype, lora_a.dtype)
    logits = model(input_ids = torch.randint(0, 64, (1, 8))).logits
    assert torch.isfinite(logits.float()).all()
