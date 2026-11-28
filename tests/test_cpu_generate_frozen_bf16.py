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

"""CPU frozen bf16 storage: on Apple Silicon, generate() called in train mode (GRPO rollouts) decodes with fp32 frozen
weights and stores them in bf16 again afterwards, bit for bit. Elsewhere it keeps decoding through the bf16 weights."""

import pytest
import torch


@pytest.fixture(autouse = True)
def _apple_silicon(monkeypatch):
    """Every test runs as on Apple Silicon, where the rollout upcast applies."""
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "_slow_bf16_upcast", lambda: True, raising = False)


def _tiny_bf16_llama(mode):
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.ops import store_frozen_linears_in_bf16
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64)
    model = LlamaForCausalLM(config).bfloat16().eval()
    reference = LlamaForCausalLM(config).eval()
    reference.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    assert store_frozen_linears_in_bf16(model) > 0
    model._unsloth_cpu_frozen_bf16 = mode
    for param in model.parameters():
        param.requires_grad_(False)
    loader.FastCPUModel._attach(model)
    return model, reference


def _stored(model):
    from unsloth_zoo.cpu.ops import Bf16StoredLinear
    return [m for m in model.modules() if isinstance(m, Bf16StoredLinear)]


def _record_weight_dtypes(model):
    seen = []
    handles = [m.register_forward_pre_hook(lambda mod, args: seen.append(mod.weight.dtype)) for m in _stored(model)]
    return seen, handles


_IDS = torch.tensor([[1, 5, 9, 3]])
_GEN = dict(max_new_tokens = 6, do_sample = False, pad_token_id = 0)


def test_train_mode_generate_decodes_in_fp32_and_restores_bf16():
    model, reference = _tiny_bf16_llama("training")
    original = {k: v.clone() for k, v in model.state_dict().items() if v.dtype == torch.bfloat16}
    model.train()
    seen, handles = _record_weight_dtypes(model)
    tokens = model.generate(input_ids = _IDS, **_GEN)
    for h in handles:
        h.remove()
    assert seen and set(seen) == {torch.float32}  # no per-token bf16 upcast while decoding
    assert {m.weight.dtype for m in _stored(model)} == {torch.bfloat16}  # the RAM saving is back for training
    state = model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in original.items())  # lossless round trip
    assert torch.equal(tokens, reference.generate(input_ids = _IDS, **_GEN))
    assert model.training


def test_train_mode_generate_matches_bf16_stored_tokens():
    """Same tokens as decoding through the bf16-stored projections (the upcast is exact)."""
    model, _ = _tiny_bf16_llama("training")
    model.train()
    fast = model.generate(input_ids = _IDS, **_GEN)
    model._unsloth_cpu_frozen_bf16 = "forced"
    slow = model.generate(input_ids = _IDS, **_GEN)
    assert torch.equal(fast, slow)


def test_train_mode_generate_restores_bf16_when_generate_raises():
    model, _ = _tiny_bf16_llama("training")
    model.train()
    seen = []

    def boom(mod, args):
        seen.append(mod.weight.dtype)
        raise RuntimeError("stop")
    handle = _stored(model)[0].register_forward_pre_hook(boom)
    with pytest.raises(RuntimeError, match = "stop"):
        model.generate(input_ids = _IDS, **_GEN)
    handle.remove()
    assert seen == [torch.float32]
    assert {m.weight.dtype for m in _stored(model)} == {torch.bfloat16}


def test_forced_storage_stays_bf16_during_train_mode_generate():
    model, _ = _tiny_bf16_llama("forced")  # UNSLOTH_CPU_FROZEN_BF16=1
    model.train()
    seen, handles = _record_weight_dtypes(model)
    model.generate(input_ids = _IDS, **_GEN)
    for h in handles:
        h.remove()
    assert set(seen) == {torch.bfloat16}


def test_lora_train_mode_generate_decodes_in_fp32_and_keeps_adapters():
    from unsloth_zoo.cpu import loader
    model, _ = _tiny_bf16_llama("training")
    lora = loader.FastCPUModel.get_peft_model(model, r = 4, lora_alpha = 4, use_gradient_checkpointing = False)
    lora.train()
    adapters = {n: p.detach().clone() for n, p in lora.named_parameters() if p.requires_grad}
    assert adapters and all(p.dtype == torch.float32 for p in adapters.values())
    seen, handles = _record_weight_dtypes(lora)
    lora.generate(input_ids = _IDS, **_GEN)
    for h in handles:
        h.remove()
    assert seen and set(seen) == {torch.float32}
    assert {m.weight.dtype for m in _stored(lora)} == {torch.bfloat16}
    after = dict(lora.named_parameters())
    assert all(after[n].requires_grad and after[n].dtype == torch.float32 and torch.equal(after[n], v)
               for n, v in adapters.items())
    # Training right after a rollout still runs through the bf16-stored projections.
    out = lora(input_ids = _IDS, labels = _IDS)
    out.loss.backward()
    assert all(after[n].grad is not None for n in adapters)


def test_train_mode_generate_stays_bf16_without_ram_headroom(monkeypatch):
    """No room for the fp32 copy (or RAM unknown): decode through the bf16-stored projections as before."""
    from unsloth_zoo.cpu import loader
    model, _ = _tiny_bf16_llama("training")
    model.train()
    stored = sum(m.weight.numel() for m in _stored(model))
    for available in (None, 2 * stored):  # 2 * stored bytes < 2 * (2 * stored): over the 50% share
        monkeypatch.setattr(loader, "available_ram", lambda: available)
        seen, handles = _record_weight_dtypes(model)
        model.generate(input_ids = _IDS, **_GEN)
        for h in handles:
            h.remove()
        assert set(seen) == {torch.bfloat16}
    monkeypatch.setattr(loader, "available_ram", lambda: 4 * stored)  # exactly the 50% share: fits
    seen, handles = _record_weight_dtypes(model)
    model.generate(input_ids = _IDS, **_GEN)
    for h in handles:
        h.remove()
    assert set(seen) == {torch.float32}


def test_lora_rollout_under_inference_mode_leaves_trainable_weights():
    """A rollout inside torch.inference_mode() must not leave inference tensors behind: the next LoRA forward saves
    the bf16 frozen weights for backward."""
    from unsloth_zoo.cpu import loader
    model, _ = _tiny_bf16_llama("training")
    lora = loader.FastCPUModel.get_peft_model(model, r = 4, lora_alpha = 4, use_gradient_checkpointing = False)
    lora.train()
    with torch.inference_mode():
        lora.generate(input_ids = _IDS, **_GEN)
    assert not any(m.weight.is_inference() for m in _stored(lora))
    assert {m.weight.dtype for m in _stored(lora)} == {torch.bfloat16}
    lora(input_ids = _IDS, labels = _IDS).loss.backward()


def test_train_mode_generate_stays_bf16_off_apple_silicon(monkeypatch):
    """x86 without native bf16 for a narrow, short rollout (1 row, 6 tokens): the fp32 copy would not pay back."""
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "_slow_bf16_upcast", lambda: False, raising = False)
    model, _ = _tiny_bf16_llama("training")
    model.train()
    seen, handles = _record_weight_dtypes(model)
    model.generate(input_ids = _IDS, **_GEN)
    for h in handles:
        h.remove()
    assert set(seen) == {torch.bfloat16}


def test_slow_bf16_upcast_is_apple_silicon_only(monkeypatch):
    import platform
    import sys
    from unsloth_zoo.cpu import loader
    monkeypatch.undo()  # the real check, not the autouse stand-in
    for plat, machine, expected in (("darwin", "arm64", True), ("darwin", "x86_64", False), ("linux", "aarch64", False),
                                    ("linux", "x86_64", False), ("win32", "AMD64", False)):
        monkeypatch.setattr(sys, "platform", plat)
        monkeypatch.setattr(platform, "machine", lambda m = machine: m)
        assert loader._slow_bf16_upcast() is expected, (plat, machine)
