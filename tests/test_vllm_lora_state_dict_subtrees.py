"""_lora_state_dict must equal the LoRA slice of model.state_dict() without running the base layers'
_save_to_state_dict (a bitsandbytes Linear4bit packs its QuantState there, with a GPU sync)."""
import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

from unsloth_zoo import vllm_utils as vu  # noqa: E402


def _reference(model):
    return {k: v for k, v in model.state_dict().items() if ".lora_A." in k or ".lora_B." in k}


def _assert_same(got, ref):
    assert list(got) == list(ref)
    for k in ref:
        assert got[k].dtype == ref[k].dtype and got[k].shape == ref[k].shape
        assert torch.equal(got[k], ref[k]), k
        assert not got[k].requires_grad


class _Block(nn.Module):
    def __init__(self, n = 4):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias = False)
        self.v_proj = nn.Linear(16, 16)
        self.experts = nn.ModuleList([nn.Linear(16, 8, bias = False) for _ in range(n)])
        self.norm = nn.LayerNorm(16)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([_Block() for _ in range(2)])
        self.register_buffer("scale", torch.ones(3))


def _peft(adapters = ("default",)):
    torch.manual_seed(0)
    model = get_peft_model(_Model(), LoraConfig(r = 4, target_modules = r".*\.(q_proj|v_proj|experts\.\d+)"),
                           adapter_name = adapters[0])
    for name in adapters[1:]:
        model.add_adapter(name, LoraConfig(r = 2, target_modules = ["q_proj"]))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.normal_()
    return model


def test_matches_state_dict_multi_adapter():
    model = _peft(("default", "second"))
    _assert_same(vu._lora_state_dict(model), _reference(model))


def test_base_layers_not_serialised():
    model = _peft()
    seen = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)) and ".lora_" not in name:
            module.register_state_dict_pre_hook(lambda m, prefix, keep_vars: seen.append(prefix))
    _assert_same(vu._lora_state_dict(model), {k: v for k, v in _reference(model).items()})
    seen.clear()
    vu._lora_state_dict(model)
    assert seen == []


def test_stacked_moe_lora_keeps_per_expert_keys():
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML
    model = _peft()
    before = _reference(model)
    for layer in model.base_model.model.layers:
        ML._stack_projs_lora(list(layer.experts), "default")
    assert any(ML._STACK_NAME in k for k, _ in model.named_parameters())
    ref = _reference(model)
    assert list(ref) == list(before)  # the stack key never leaves the model
    _assert_same(vu._lora_state_dict(model), ref)


def test_bnb_linear4bit_quant_state_not_packed():
    bnb = pytest.importorskip("bitsandbytes")
    model = _peft()
    for layer in model.base_model.model.layers:
        base = layer.q_proj.base_layer
        q = bnb.nn.Linear4bit(16, 16, bias = False, quant_type = "nf4")
        q.weight = bnb.nn.Params4bit(base.weight.data.clone(), requires_grad = False, quant_type = "nf4")
        layer.q_proj.base_layer = q
    calls = []
    orig = bnb.nn.Linear4bit._save_to_state_dict
    def counted(self, *args, **kwargs):
        calls.append(1)
        return orig(self, *args, **kwargs)
    bnb.nn.Linear4bit._save_to_state_dict = counted
    try:
        ref = _reference(model)
        assert len(calls) == 2
        calls.clear()
        _assert_same(vu._lora_state_dict(model), ref)
        assert calls == []
    finally:
        bnb.nn.Linear4bit._save_to_state_dict = orig
