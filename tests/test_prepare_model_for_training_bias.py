"""Regression tests for unsloth#2343: bias training was silently disabled.

The LoRA branch froze every non-adapter param, so LoraConfig(bias="all"/"lora_only")
had no effect. The fix preserves PEFT's bias decision (trainable biases keep their
loaded dtype, frozen on bias="none", gated to PEFT models). Pure CPU, tiny Llama;
PEFT tests import-or-skip.
"""
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


def _lora(model, bias):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=8, bias=bias,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )


def _trainable_bias_count(model):
    return sum(
        p.requires_grad for n, p in model.named_parameters() if n.endswith(".bias"))


# ---------------- PEFT bias decision is preserved ----------------------------

def test_peft_bias_all_stays_trainable():
    """bias='all': every bias PEFT enabled stays trainable after prepare."""
    pytest.importorskip("peft")
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _lora(_tiny_llama(), bias="all")
    before = _trainable_bias_count(model)
    assert before > 0, "PEFT should enable biases for bias='all'"

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    after = _trainable_bias_count(model)
    assert after == before, f"bias='all' lost trainable biases: {before} -> {after}"
    # Base weights stay frozen.
    params = dict(model.named_parameters())
    base_w = next(n for n in params
                  if n.endswith("q_proj.base_layer.weight"))
    assert not params[base_w].requires_grad


def test_peft_bias_none_stays_frozen():
    """bias='none' (default): no biases trainable before or after -> unchanged."""
    pytest.importorskip("peft")
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _lora(_tiny_llama(), bias="none")
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    assert _trainable_bias_count(model) == 0


def test_peft_trainable_bias_keeps_module_dtype():
    """A trainable bias keeps its Linear's bf16 dtype, not fp32 (else the matmul
    dtype mismatches)."""
    pytest.importorskip("peft")
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _lora(_tiny_llama(dtype=torch.bfloat16), bias="all")
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False, float32_mixed_precision=True,
    )
    params = dict(model.named_parameters())
    bias_name = next(n for n in params
                     if n.endswith(".bias") and params[n].requires_grad)
    weight_name = bias_name[:-len("bias")] + "weight"
    assert params[bias_name].dtype == torch.bfloat16, params[bias_name].dtype
    assert params[bias_name].dtype == params[weight_name].dtype


def test_peft_bias_gradient_flows_after_backward():
    """End-to-end on CPU: with bias='all', a backward pass populates a finite
    gradient on a trainable bias; a frozen base weight gets none."""
    pytest.importorskip("peft")
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _lora(_tiny_llama(), bias="all")
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )

    input_ids = torch.randint(0, 64, (1, 8))
    # Loss from logits, not labels=: the labelled forward uses fused CUDA CE that
    # aborts on a CPU-only build; logits stay CPU-only and still drive a backward.
    logits = model(input_ids=input_ids).logits.float()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), input_ids.view(-1))
    loss.backward()

    params = dict(model.named_parameters())
    a_bias = next(n for n in params
                  if n.endswith(".bias") and params[n].requires_grad)
    a_grad = params[a_bias].grad
    assert a_grad is not None, "trainable bias must receive a grad"
    assert torch.isfinite(a_grad).all(), "bias grad must be finite"
    # > 0 proves a gradient actually flowed, not just that grad exists.
    assert a_grad.abs().sum() > 0, "bias grad must be non-zero (gradient flowed)"
    base_w = next(n for n in params if n.endswith("q_proj.base_layer.weight"))
    assert params[base_w].grad is None, "frozen base weight must not accumulate grad"


# ---------------- non-PEFT models are left frozen ----------------------------

def test_non_peft_model_biases_stay_frozen():
    """A non-PEFT model's biases default to requires_grad=True but must still be
    frozen on the LoRA path; bias preservation is a PEFT-only decision."""
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _tiny_llama()  # plain LlamaForCausalLM, no PEFT
    assert any(p.requires_grad for n, p in model.named_parameters()
               if n.endswith(".bias")), "fresh nn.Linear biases default to trainable"

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    assert _trainable_bias_count(model) == 0, \
        "non-PEFT biases must be frozen on the LoRA path"


# ---------------- modules_to_save biases are not a bias decision -------------

def test_modules_to_save_bias_not_preserved_on_bias_none():
    """bias='none' + modules_to_save: a saved module is trainable because the whole
    module is saved, not as a bias decision; with patch_modules_to_save=False bias and
    weight stay frozen together (#2343 review)."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.training_utils import prepare_model_for_training

    # gate_proj has a bias and isn't a LoRA target, so PEFT wraps it as a saved module.
    model = get_peft_model(
        _tiny_llama(),
        LoraConfig(
            r=8, lora_alpha=8, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["gate_proj"],
        ),
    )
    saved = [n for n, _ in model.named_parameters() if ".modules_to_save." in n]
    saved_bias = next(n for n in saved if n.endswith(".bias"))
    saved_weight = next(n for n in saved if n.endswith(".weight"))
    params = dict(model.named_parameters())
    assert params[saved_bias].requires_grad, "PEFT should mark the saved module trainable"

    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False, patch_modules_to_save=False,
    )
    params = dict(model.named_parameters())
    assert _trainable_bias_count(model) == 0, "bias='none' must leave all biases frozen"
    # Saved bias matches its weight (both frozen, not partially trained).
    assert params[saved_bias].requires_grad == params[saved_weight].requires_grad
