"""An integer input never reaches a LoRA matmul on a bitsandbytes 4-bit layer.

PEFT's `Linear4bit.forward` casts the input to the adapter dtype only when autocast is off.
Under autocast it trusts autocast, which leaves integer tensors alone, so the LoRA branch runs
`F.linear(uint8, bf16)` and stops with "expected mat1 and mat2 to have the same dtype". The
Nemotron-H hub checkpoints feed every idle expert `zeros(...).to(expert.down_proj.weight.dtype)`,
which on a 4-bit expert is uint8, so the first step of a 512-expert model on a 512-token batch
hit this on every layer.
"""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "bitsandbytes 4-bit needs CUDA")


def _patch():
    from unsloth_zoo.temporary_patches.misc import patch_peft_lora_integer_input
    patch_peft_lora_integer_input()
    import peft.tuners.lora.bnb as peft_bnb
    return peft_bnb.Linear4bit


def _lora_4bit_linear():
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    base = torch.nn.Sequential(bnb.nn.Linear4bit(32, 64, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4"))
    base[0].weight = bnb.nn.Params4bit(torch.randn(64, 32, dtype = torch.bfloat16), requires_grad = False, quant_type = "nf4")
    base = base.cuda()
    base.is_loaded_in_4bit = True  # what PEFT's dispatcher keys the bitsandbytes Linear4bit wrapper on
    model = get_peft_model(base, LoraConfig(r = 4, target_modules = ["0"], init_lora_weights = False))
    import peft.tuners.lora.bnb as peft_bnb
    assert isinstance(model.base_model.model[0], peft_bnb.Linear4bit)
    return model


def test_patch_is_idempotent():
    cls = _patch()
    forward = cls.forward
    assert getattr(forward, "_unsloth_integer_input", False)
    from unsloth_zoo.temporary_patches.misc import patch_peft_lora_integer_input
    patch_peft_lora_integer_input()
    assert cls.forward is forward


@cuda
def test_uint8_input_under_autocast_runs_and_matches_the_float_input():
    _patch()
    model = _lora_4bit_linear()
    x = torch.randint(0, 3, (8, 32), device = "cuda", dtype = torch.uint8)
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = model(x)
        reference = model(x.to(torch.bfloat16))
    assert out.dtype == reference.dtype
    assert torch.equal(out, reference)


@cuda
def test_float_inputs_take_the_original_path():
    cls = _patch()
    model = _lora_4bit_linear()
    x = torch.randn(8, 32, device = "cuda", dtype = torch.bfloat16)
    with torch.autocast("cuda", dtype = torch.bfloat16):
        patched = model(x)
        original = cls.forward.__wrapped__(model.base_model.model[0], x)
    assert torch.equal(patched, original)


@cuda
def test_original_forward_fails_on_the_integer_input():
    cls = _patch()
    model = _lora_4bit_linear()
    x = torch.randint(0, 3, (8, 32), device = "cuda", dtype = torch.uint8)
    with torch.autocast("cuda", dtype = torch.bfloat16):
        with pytest.raises(RuntimeError, match = "same dtype"):
            cls.forward.__wrapped__(model.base_model.model[0], x)
