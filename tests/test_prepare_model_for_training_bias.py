"""Regression tests for unsloth#2343: bias training was silently disabled.

Pre-fix bug: `prepare_model_for_training`'s LoRA branch froze every parameter
that was not a LoRA adapter weight:

    if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
        requires_grad = True
    else:
        requires_grad = False   # <- also froze biases PEFT had marked trainable

So `LoraConfig(bias="all")` / `bias="lora_only"` had no effect: PEFT enabled the
biases, then this loop re-froze them. The fix keeps the PEFT decision for params
whose name ends in `.bias` (preserve their incoming requires_grad and upcast the
trainable ones to fp32 like LoRA adapters). `bias="none"` (the default) leaves
biases frozen, so the common path is byte-identical.

Pure CPU, tiny random Llama. The PEFT end-to-end test is gated on peft being
importable (peft is excluded on darwin/arm64).
"""
import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


def _tiny_llama(dtype=torch.float32, with_bias=True):
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_bias=with_bias,
        mlp_bias=with_bias,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg)
    if dtype != torch.float32:
        model = model.to(dtype)
    model.config.torch_dtype = dtype
    return model


def _bias_names(model):
    return [n for n, _ in model.named_parameters() if n.endswith(".bias")]


# ---------------- core fix (no peft): PEFT's bias decision is respected -------

def test_marked_biases_stay_trainable_after_prepare():
    """Simulate PEFT bias='all' (every bias requires_grad=True). After prepare
    the biases must remain trainable and the non-bias base weights frozen."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama()
    biases = _bias_names(model)
    assert biases, "fixture must have bias params"
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith(".bias"))  # PEFT bias='all' shape

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )

    params = dict(model.named_parameters())
    for n in biases:
        assert params[n].requires_grad, f"{n} should stay trainable (bias='all')"
    # A representative non-bias base weight must be frozen.
    assert not params["model.layers.0.self_attn.q_proj.weight"].requires_grad


def test_default_biases_stay_frozen():
    """bias='none' shape: nothing marked trainable -> biases stay frozen, i.e.
    byte-identical to pre-fix behaviour on the common path."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama()
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )

    params = dict(model.named_parameters())
    for n in _bias_names(model):
        assert not params[n].requires_grad, f"{n} should stay frozen (bias='none')"


def test_trainable_bias_is_upcast_to_fp32():
    """A trainable bias is upcast to fp32 (like LoRA adapters); a frozen one
    keeps its loaded (bf16) dtype."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama(dtype=torch.bfloat16)
    names = _bias_names(model)
    trainable_bias = names[0]
    # Mark exactly one bias trainable (mimics bias='lora_only' picking a subset).
    for n, p in model.named_parameters():
        p.requires_grad_(n == trainable_bias)

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False, float32_mixed_precision=True,
    )

    params = dict(model.named_parameters())
    assert params[trainable_bias].dtype == torch.float32
    assert params[trainable_bias].requires_grad
    # An untouched, frozen bias is not recast.
    frozen_bias = names[1]
    assert params[frozen_bias].dtype == torch.bfloat16
    assert not params[frozen_bias].requires_grad


def test_bias_gradient_flows_after_backward():
    """End-to-end on CPU: with biases trainable, a backward pass populates
    their .grad; a frozen base weight gets no grad."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama()
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith(".bias"))

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )

    input_ids = torch.randint(0, 64, (1, 8))
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()

    params = dict(model.named_parameters())
    a_bias = _bias_names(model)[0]
    assert params[a_bias].grad is not None, "trainable bias must receive a grad"
    assert params[a_bias].grad.abs().sum() >= 0  # finite, allocated
    # A frozen base weight must not accumulate a grad.
    assert params["model.layers.0.self_attn.q_proj.weight"].grad is None


# ---------------- PEFT end-to-end (gated on peft) ----------------------------

def test_peft_bias_all_end_to_end():
    """Through real PEFT: LoraConfig(bias='all') biases stay trainable after
    prepare_model_for_training (the exact path from the issue repro)."""
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=8, bias="all",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    trainable_before = sum(
        p.requires_grad for n, p in model.named_parameters() if n.endswith(".bias"))
    assert trainable_before > 0  # PEFT enabled the biases

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )

    trainable_after = sum(
        p.requires_grad for n, p in model.named_parameters() if n.endswith(".bias"))
    assert trainable_after == trainable_before, (
        f"bias='all' lost trainable biases: {trainable_before} -> {trainable_after}")


def test_peft_bias_none_end_to_end():
    """bias='none' (default): no biases trainable before or after -> unchanged."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=8, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    trainable_after = sum(
        p.requires_grad for n, p in model.named_parameters() if n.endswith(".bias"))
    assert trainable_after == 0
