"""Integer inputs to a bitsandbytes 4-bit LoRA layer under autocast (Nemotron-H idle experts)."""
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
    base.is_loaded_in_4bit = True  # PEFT's dispatcher keys Linear4bit on this
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


@cuda
def test_integer_input_under_autocast_takes_the_autocast_dtype_not_compute_dtype():
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    import peft.tuners.lora.bnb as peft_bnb

    _patch()
    torch.manual_seed(0)
    base = torch.nn.Sequential(bnb.nn.Linear4bit(32, 64, bias = False, compute_dtype = torch.float32, quant_type = "nf4"))
    base[0].weight = bnb.nn.Params4bit(torch.randn(64, 32, dtype = torch.bfloat16), requires_grad = False, quant_type = "nf4")
    base = base.cuda()
    base.is_loaded_in_4bit = True
    model = get_peft_model(base, LoraConfig(r = 4, target_modules = ["0"], init_lora_weights = False))
    assert isinstance(model.base_model.model[0], peft_bnb.Linear4bit)
    assert model.base_model.model[0].base_layer.compute_dtype == torch.float32
    x = torch.randint(0, 3, (8, 32), device = "cuda", dtype = torch.uint8)
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = model(x)
        reference = model(x.to(torch.bfloat16))
    assert out.dtype == reference.dtype == torch.bfloat16
    assert torch.equal(out, reference)


def test_the_cast_survives_a_regenerated_forward():
    """getsource follows __wrapped__, so patch_lora_forwards must re-apply the wrapper."""
    import inspect
    from unsloth_zoo import compiler
    from unsloth_zoo.temporary_patches.misc import patch_peft_lora_integer_input

    assert "patch_peft_lora_integer_input()" in inspect.getsource(compiler.patch_lora_forwards)
    cls = _patch()
    original_wrapped = cls.forward

    def regenerated(self, x, *args, **kwargs):
        return original_wrapped.__wrapped__(self, x, *args, **kwargs)

    cls.forward = regenerated
    try:
        patch_peft_lora_integer_input()
        assert getattr(cls.forward, "_unsloth_integer_input", False)
        assert cls.forward.__wrapped__ is regenerated
    finally:
        cls.forward = original_wrapped
