"""CPU without native bf16: a bf16 checkpoint's frozen projections stay float32 when the float32 model fits in a
quarter of the free RAM (no per-call upcast), bf16-stored otherwise; UNSLOTH_CPU_FROZEN_BF16 still decides."""
import pytest
import torch


class _StubTokenizer:
    pad_token, pad_token_id, eos_token, padding_side = "<pad>", 0, "</s>", "right"

    def __len__(self):
        return 64

    def save_pretrained(self, *a, **k):
        pass


def _config():
    from transformers import LlamaConfig
    return LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                       num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
                       tie_word_embeddings = False)


@pytest.fixture(scope = "module")
def checkpoint(tmp_path_factory):
    """A tiny bf16 Llama checkpoint on disk, loaded by the real transformers from_pretrained."""
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_bf16_llama")
    LlamaForCausalLM(_config()).to(torch.bfloat16).save_pretrained(path)
    return str(path)


def _load(monkeypatch, path, env, available):
    import transformers
    from unsloth_zoo.cpu import loader
    if env is None:
        monkeypatch.delenv("UNSLOTH_CPU_FROZEN_BF16", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_CPU_FROZEN_BF16", env)
    monkeypatch.setattr(loader, "cpu_bf16_supported", lambda: False)
    monkeypatch.setattr(loader, "available_ram", lambda: available)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer())
    model, _ = loader.FastCPUModel.from_pretrained(path, load_in_4bit = False)
    return model


def _stored(model):
    from unsloth_zoo.cpu.ops import Bf16StoredLinear
    return sum(isinstance(m, Bf16StoredLinear) and m.weight.dtype == torch.bfloat16 for m in model.modules())


def _fp32_bytes():
    from unsloth_zoo.cpu.loader import _dense_parameter_count
    return 4 * _dense_parameter_count(_config())


@pytest.mark.parametrize("env, ram, stored, mode", [
    (None, "fits", 0, None),            # fits: float32, no upcast per call
    (None, "short", 14, "training"),    # does not fit: bf16 storage as before
    (None, None, 14, "training"),       # free RAM unknown: as before
    ("training", "fits", 14, "training"),  # kill switch: the previous default whatever the RAM
    ("1", "fits", 14, "forced"),
    ("0", "short", 0, None),
])
def test_frozen_storage_follows_the_ram_fit(monkeypatch, capsys, checkpoint, env, ram, stored, mode):
    available = None if ram is None else (4 * _fp32_bytes() if ram == "fits" else 4 * _fp32_bytes() - 1)
    model = _load(monkeypatch, checkpoint, env, available)
    assert _stored(model) == stored
    assert getattr(model, "_unsloth_cpu_frozen_bf16", None) == mode
    assert ("frozen weights kept in float32" in capsys.readouterr().out) == (env is None and ram == "fits")
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if "proj" not in n)


def test_quarter_of_free_ram_is_the_bound():
    from unsloth_zoo.cpu import loader
    need = _fp32_bytes()
    original = loader.available_ram
    try:
        loader.available_ram = lambda: 4 * need
        assert loader._fp32_weights_fit("x", None, None, False, config = _config()) == (need, 4 * need)
        loader.available_ram = lambda: 4 * need - 1
        assert loader._fp32_weights_fit("x", None, None, False, config = _config()) is None
    finally:
        loader.available_ram = original


def test_several_training_processes_keep_bf16_storage(monkeypatch):
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "available_ram", lambda: 1 << 50)
    assert loader._fp32_weights_fit("x", None, None, False, config = _config()) is not None
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert loader._fp32_weights_fit("x", None, None, False, config = _config()) is None


def test_float32_and_bf16_stored_give_the_same_loss_and_lora_grads(monkeypatch, checkpoint):
    """The two storages hold the same bf16 values; from the same adapter weights the float32 path trains identically
    (bit for bit). Only the fresh adapter's init differs: PEFT draws it in the base weight's dtype, so bf16 storage
    rounds lora_A's init to bf16 and float32 storage keeps it float32 (as UNSLOTH_CPU_FROZEN_BF16=0 does)."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (2, 12))
    peft_models = []
    for ram in ("fits", "short"):
        model = _load(monkeypatch, checkpoint, None, 4 * _fp32_bytes() if ram == "fits" else 1)
        model.train()
        torch.manual_seed(2)
        config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"],
                            lora_dropout = 0.0, init_lora_weights = False)
        peft_models.append(get_peft_model(model, config))
    fp32, stored = peft_models
    assert _stored(fp32) == 0 and _stored(stored) == 14
    lora_a = [p for n, p in stored.named_parameters() if "lora_A" in n]
    assert lora_a and all(torch.equal(p, p.bfloat16().float()) for p in lora_a)
    assert not all(torch.equal(p, p.bfloat16().float()) for n, p in fp32.named_parameters() if "lora_A" in n)
    stored.load_state_dict(fp32.state_dict(), strict = False)  # same adapter weights in both
    results = []
    for peft in (fp32, stored):
        out = peft(input_ids = ids, labels = ids)
        out.loss.backward()
        results.append((out.loss.detach(), {n: p.grad.clone() for n, p in peft.named_parameters() if p.requires_grad}))
    (loss_a, grads_a), (loss_b, grads_b) = results
    assert torch.equal(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys() and len(grads_a) == 12
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), name
