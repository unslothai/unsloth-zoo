"""Regression tests for unsloth#2343: bias training was silently disabled.

Pre-fix bug: `prepare_model_for_training`'s LoRA branch froze every parameter
that was not a LoRA adapter weight, so `LoraConfig(bias="all")` / `bias="lora_only"`
had no effect (PEFT enabled the biases, then this loop re-froze them).

The fix preserves PEFT's bias decision. Important properties (each guarded below):
  * bias="all"/"lora_only": the biases PEFT marked trainable stay trainable.
  * bias="none" (default): biases stay frozen (byte-identical common path).
  * a trainable bias keeps its loaded dtype, NOT fp32: upcasting a bias while its
    Linear's weight stays bf16/fp16 breaks the matmul ("self and mat2 must have
    the same dtype") -- the dtype-mismatch the fix must avoid.
  * the preservation is gated to PEFT models: a freshly-loaded non-PEFT model
    (whose nn.Linear biases default to requires_grad=True) is left frozen on the
    LoRA (not full_finetuning) path.

Pure CPU, tiny random Llama. PEFT is required for the bias-preservation tests
(peft is excluded on darwin/arm64), so those import-or-skip.
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
    # The non-bias base weights must remain frozen.
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
    """A trainable bias must keep its Linear's (bf16) dtype, NOT be upcast to
    fp32 -- otherwise the bf16 weight @ fp32 bias matmul raises
    'self and mat2 must have the same dtype'. Regression guard for the review fix.
    """
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
    model(input_ids=input_ids, labels=input_ids).loss.backward()

    params = dict(model.named_parameters())
    a_bias = next(n for n in params
                  if n.endswith(".bias") and params[n].requires_grad)
    assert params[a_bias].grad is not None, "trainable bias must receive a grad"
    assert torch.isfinite(params[a_bias].grad).all(), "bias grad must be finite"
    base_w = next(n for n in params if n.endswith("q_proj.base_layer.weight"))
    assert params[base_w].grad is None, "frozen base weight must not accumulate grad"


# ---------------- non-PEFT models are left frozen ----------------------------

def test_non_peft_model_biases_stay_frozen():
    """A non-PEFT model's nn.Linear biases default to requires_grad=True. On the
    LoRA (not full_finetuning) path they must still be frozen -- the bias
    preservation is only a PEFT decision, so a raw model is unaffected."""
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
