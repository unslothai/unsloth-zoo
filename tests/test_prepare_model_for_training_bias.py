"""Regression tests: PEFT's bias choice must survive prepare_model_for_training; CPU-only tiny Llama."""
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
    """A trainable bias keeps its Linear's bf16 dtype; fp32 would mismatch the matmul dtype."""
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
    """With bias='all' on CPU, trainable biases get a finite grad after backward; frozen weights get none."""
    pytest.importorskip("peft")
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _lora(_tiny_llama(), bias="all")
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )

    input_ids = torch.randint(0, 64, (1, 8))
    # Compute loss from logits: labels= hits fused CUDA CE, which aborts on a CPU-only build.
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
    assert a_grad.abs().sum() > 0, "bias grad must be non-zero (gradient flowed)"
    base_w = next(n for n in params if n.endswith("q_proj.base_layer.weight"))
    assert params[base_w].grad is None, "frozen base weight must not accumulate grad"


def test_non_peft_model_biases_stay_frozen():
    """Non-PEFT biases start requires_grad=True but stay frozen on the LoRA path; only PEFT keeps bias."""
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


def test_modules_to_save_bias_not_preserved_on_bias_none():
    """With bias=none and modules_to_save the module is trainable as a whole, not as a bias decision; with
    patch_modules_to_save=False bias and weight stay frozen (unsloth issue 2343 review)."""
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


@pytest.mark.parametrize("bias", ["all", "lora_only"])
def test_peft_trainable_bias_float16_gets_float32_master(bias):
    """float16: a float16 trainable bias makes the GradScaler raise "Attempting to unscale FP16 gradients"
    at the first step, so the bias gets a float32 master as LoRA A / B do; the float16 autocast the Trainer
    runs under casts it back for the matmul."""
    pytest.importorskip("peft")
    from unsloth_zoo.training_utils import prepare_model_for_training

    model = _lora(_tiny_llama(dtype=torch.float16), bias=bias)
    trainable_before = _trainable_bias_count(model)
    assert trainable_before > 0
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    assert _trainable_bias_count(model) == trainable_before
    params = dict(model.named_parameters())
    trainable = {n: p.dtype for n, p in params.items() if p.requires_grad}
    assert set(trainable.values()) == {torch.float32}, trainable
    base_w = next(n for n in params if n.endswith("q_proj.base_layer.weight"))
    assert params[base_w].dtype == torch.float16  # frozen base stays float16

    # One float16 AMP step as the Trainer takes it (accelerate's GradScaler + float16 autocast).
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cpu")
    input_ids = torch.randint(0, 64, (2, 8))
    with torch.autocast("cpu", dtype=torch.float16):
        logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)), input_ids.view(-1))
    scaler.scale(loss).backward()
    biases = {n: p.detach().clone() for n, p in params.items()
              if n.endswith(".bias") and p.requires_grad}
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    moved = [n for n, b in biases.items() if not torch.equal(b, params[n].detach())]
    assert moved, "no trainable bias was updated"


def _float16_bias_model(bias, save_base_to=None):
    from unsloth_zoo.training_utils import prepare_model_for_training

    torch.manual_seed(0)
    model = _tiny_llama(dtype=torch.float16)
    if save_base_to is not None:
        model.save_pretrained(save_base_to)
    model = _lora(model, bias=bias)
    prepare_model_for_training(
        model, use_gradient_checkpointing=False, use_reentrant=False,
        full_finetuning=False,
    )
    return model


def _trainable_biases(model):
    return {n: p for n, p in model.named_parameters() if n.endswith(".bias") and p.requires_grad}


@pytest.mark.parametrize("bias", ["all", "lora_only"])
def test_float16_bias_master_runs_without_autocast(bias):
    """eval / generate run outside autocast: the float32 bias master must not reach a float16 matmul
    ("self and mat2 must have the same dtype"), and must stay the float32 Parameter afterwards."""
    pytest.importorskip("peft")
    model = _float16_bias_model(bias)
    masters = _trainable_biases(model)
    assert masters and {p.dtype for p in masters.values()} == {torch.float32}
    model.eval()
    input_ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        assert model(input_ids=input_ids).logits.dtype == torch.float16
    out = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False)
    assert out.shape == (1, 8)
    after = _trainable_biases(model)
    assert after.keys() == masters.keys()
    assert all(after[n] is masters[n] for n in masters)
    assert all(isinstance(p, torch.nn.Parameter) and p.dtype == torch.float32 for p in after.values())


def test_float16_bias_master_trains_without_autocast():
    """A no-autocast float16 step: the bias gradient lands on the float32 master, so the GradScaler unscales it."""
    pytest.importorskip("peft")
    model = _float16_bias_model("all")
    params = _trainable_biases(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cpu")
    input_ids = torch.randint(0, 64, (2, 8))
    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), input_ids.view(-1))
    scaler.scale(loss).backward()
    before = {n: p.detach().clone() for n, p in params.items()}
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in params.values())
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    assert any(not torch.equal(before[n], p.detach()) for n, p in params.items())


def test_float16_bias_master_compiles():
    """The cast hook traces in one graph and matches eager."""
    pytest.importorskip("peft")
    from torch._dynamo.testing import CompileCounter

    model = _float16_bias_model("all")
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj
    x = torch.randn(2, 3, 16, dtype=torch.float16)
    torch._dynamo.reset()
    counter = CompileCounter()
    try:
        out = torch.compile(q_proj, backend=counter, fullgraph=True)(x)
    finally:
        torch._dynamo.reset()
    assert counter.frame_count == 1
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, q_proj(x))
    out.float().sum().backward()
    assert q_proj.base_layer.bias.grad.dtype == torch.float32


def test_float16_bias_master_merges_in_the_model_dtype(tmp_path):
    """Merged exports write the trained bias in the float16 save dtype, and PEFT's merge still runs."""
    pytest.importorskip("peft")
    from safetensors import safe_open
    import _merge_e2e_helpers as H

    H.set_offline_cpu_env()
    base = str(tmp_path / "base")
    model = _float16_bias_model("all", save_base_to=base)
    params = _trainable_biases(model)
    with torch.no_grad():
        for p in params.values():
            p.add_(0.25)
    out = str(tmp_path / "out")
    H.run_merge(model, base, out, save_dtype=torch.float16)
    name = next(n for n in params if n.endswith("q_proj.base_layer.bias"))
    key = name.replace("base_model.model.", "", 1).replace(".base_layer", "")
    files = [f for f in os.listdir(out) if f.endswith(".safetensors")]
    found = None
    for f in files:
        with safe_open(os.path.join(out, f), "pt") as handle:
            if key in handle.keys():
                found = handle.get_tensor(key)
    assert found is not None, key
    assert found.dtype == torch.float16
    torch.testing.assert_close(found, params[name].detach().to(torch.float16))

    merged = model.merge_and_unload()
    with torch.no_grad():
        assert merged(input_ids=torch.randint(0, 64, (1, 4))).logits.dtype == torch.float16


@pytest.mark.parametrize("bias", ["all", "lora_only"])
@pytest.mark.parametrize("target", [torch.float32, torch.bfloat16])
def test_float16_bias_master_follows_a_later_model_cast(bias, target):
    """model.float() / model.bfloat16() after prepare: the bias hook must follow the module's current dtype,
    not the float16 captured at install, or a plain forward fails with "self and mat2 must have the same dtype"."""
    pytest.importorskip("peft")
    model = _float16_bias_model(bias).to(target)
    with torch.no_grad():
        assert model(input_ids=torch.randint(0, 64, (1, 5))).logits.dtype == target
    assert all(p.dtype == target for p in _trainable_biases(model).values())


@pytest.mark.parametrize("target", [torch.float32, torch.bfloat16])
def test_float16_bias_master_merged_then_cast(target):
    """The hooks survive merge_and_unload: a merged model cast to float32 / bfloat16 still runs a plain forward."""
    pytest.importorskip("peft")
    merged = _float16_bias_model("all").merge_and_unload().to(target)
    with torch.no_grad():
        assert merged(input_ids=torch.randint(0, 64, (1, 4))).logits.dtype == target
