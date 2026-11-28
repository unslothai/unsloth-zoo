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
"""CPU frozen bf16 storage on AVX2-only x86 without native bf16 (EPYC 7763): a train-mode generate (GRPO rollout) of
at least 8 rows and 32 new tokens decodes through an fp32 copy of the frozen projections, then stores them in bf16
again, bit for bit; narrower or shorter rollouts and every other x86 class keep decoding through the bf16 weights."""

import platform

import pytest
import torch


@pytest.fixture
def avx2_host(monkeypatch):
    """Runs as on an AVX2-only x86 host without native bf16 (not Apple Silicon)."""
    from unsloth_zoo.cpu import loader, runtime
    # The fp32 copy is the path when the fp16 decode packs are off or unavailable (test_cpu_fp16_decode covers those).
    monkeypatch.setenv("UNSLOTH_CPU_FP16_DECODE", "0")
    monkeypatch.setattr(loader, "_slow_bf16_upcast", lambda: False)
    monkeypatch.setattr(loader, "cpu_bf16_supported", lambda: False)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset({"sse4_2", "avx", "avx2", "fma"}))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    return loader


@pytest.mark.parametrize("rows, new_tokens, expected", [
    (8, 32, True), (16, 256, True), (7, 64, False), (8, 31, False), (2, 16, False), (None, 64, False), (8, None, False),
])
def test_avx2_gate_by_rollout_shape(avx2_host, rows, new_tokens, expected):
    assert avx2_host._avx2_fp32_rollout(rows, new_tokens) is expected


@pytest.mark.parametrize("machine, flags, bf16, expected", [
    ("x86_64", {"avx2"}, False, True),
    ("AMD64", {"avx2"}, False, True),
    ("x86_64", {"avx2", "avx512f"}, False, False),  # AVX-512 Linux x86 already stores fp32 (fp32 decode was faster)
    ("x86_64", {"avx2"}, True, False),  # native bf16: no bf16-stored frozen weights by default
    ("x86_64", set(), False, False),  # unknown flags
    ("aarch64", {"avx2"}, False, False),
])
def test_avx2_gate_by_cpu(avx2_host, monkeypatch, machine, flags, bf16, expected):
    from unsloth_zoo.cpu import runtime
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset(flags))
    monkeypatch.setattr(avx2_host, "cpu_bf16_supported", lambda: bf16)
    assert avx2_host._avx2_fp32_rollout(8, 64) is expected


def test_rollout_shape_reads_generate_arguments():
    from transformers import GenerationConfig
    from unsloth_zoo.cpu.loader import _rollout_shape

    class Model:
        generation_config = GenerationConfig(max_length = 20)
    ids = torch.zeros(4, 6, dtype = torch.long)
    assert _rollout_shape(Model(), (), {"input_ids": ids, "max_new_tokens": 64}) == (4, 64)
    assert _rollout_shape(Model(), (ids,), {"generation_config": GenerationConfig(max_new_tokens = 48)}) == (4, 48)
    assert _rollout_shape(Model(), (), {"input_ids": ids, "num_return_sequences": 2, "max_new_tokens": 8}) == (8, 8)
    assert _rollout_shape(Model(), (), {"input_ids": ids}) == (4, 14)  # max_length 20 - 6 prompt tokens
    assert _rollout_shape(Model(), (), {}) == (None, None)


def _tiny_bf16_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.ops import store_frozen_linears_in_bf16
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 128)
    model = LlamaForCausalLM(config).bfloat16()
    assert store_frozen_linears_in_bf16(model) > 0
    model._unsloth_cpu_frozen_bf16 = "training"
    for param in model.parameters():
        param.requires_grad_(False)
    loader.FastCPUModel._attach(model)
    return model.train()


def _decode_dtypes(model, ids, **gen):
    from unsloth_zoo.cpu.ops import Bf16StoredLinear
    stored = [m for m in model.modules() if isinstance(m, Bf16StoredLinear)]
    seen = []
    handles = [m.register_forward_pre_hook(lambda mod, args: seen.append(mod.weight.dtype)) for m in stored]
    try:
        tokens = model.generate(input_ids = ids, do_sample = False, pad_token_id = 0, **gen)
    finally:
        for h in handles:
            h.remove()
    return set(seen), tokens, {m.weight.dtype for m in stored}


def test_wide_rollout_decodes_in_fp32_and_restores_bf16(avx2_host):
    model = _tiny_bf16_llama()
    original = {k: v.clone() for k, v in model.state_dict().items() if v.dtype == torch.bfloat16}
    ids = torch.randint(1, 64, (8, 4), generator = torch.Generator().manual_seed(0))
    seen, tokens, after = _decode_dtypes(model, ids, max_new_tokens = 32)
    assert seen == {torch.float32} and after == {torch.bfloat16}
    state = model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in original.items())  # lossless round trip
    model._unsloth_cpu_frozen_bf16 = "forced"  # same rollout through the bf16-stored projections
    seen_bf16, tokens_bf16, _ = _decode_dtypes(model, ids, max_new_tokens = 32)
    assert seen_bf16 == {torch.bfloat16} and torch.equal(tokens, tokens_bf16)


@pytest.mark.parametrize("rows, new_tokens", [(2, 32), (8, 16)])
def test_narrow_or_short_rollout_stays_bf16(avx2_host, rows, new_tokens):
    model = _tiny_bf16_llama()
    ids = torch.randint(1, 64, (rows, 4), generator = torch.Generator().manual_seed(0))
    seen, _, after = _decode_dtypes(model, ids, max_new_tokens = new_tokens)
    assert seen == {torch.bfloat16} and after == {torch.bfloat16}


def test_wide_rollout_stays_bf16_on_avx512(avx2_host, monkeypatch):
    from unsloth_zoo.cpu import runtime
    monkeypatch.setattr(runtime, "cpu_flags", lambda: frozenset({"avx2", "avx512f"}))
    model = _tiny_bf16_llama()
    ids = torch.randint(1, 64, (8, 4), generator = torch.Generator().manual_seed(0))
    seen, _, _ = _decode_dtypes(model, ids, max_new_tokens = 32)
    assert seen == {torch.bfloat16}


def test_wide_rollout_keeps_a_trainable_stored_projection_and_its_optimizer_state(avx2_host):
    """modules_to_save / an unfrozen projection: its bf16 weight and AdamW state survive the fp32 rollout copy."""
    from unsloth_zoo.cpu.ops import Bf16StoredLinear
    model = _tiny_bf16_llama()
    trainable = model.model.layers[0].mlp.down_proj
    assert isinstance(trainable, Bf16StoredLinear) and trainable.weight.dtype == torch.bfloat16
    trainable.weight.requires_grad_(True)
    optimizer = torch.optim.AdamW([trainable.weight], lr = 1e-3, foreach = False)
    ids = torch.randint(1, 64, (8, 4), generator = torch.Generator().manual_seed(0))

    def step():
        optimizer.zero_grad()
        model(input_ids = ids, labels = ids).loss.backward()
        optimizer.step()

    step()  # bf16 optimizer state exists before the first wide rollout
    frozen = model.model.layers[1].mlp.down_proj
    seen = []
    handle = frozen.register_forward_pre_hook(lambda mod, args: seen.append(mod.weight.dtype))
    try:
        model.generate(input_ids = ids, do_sample = False, pad_token_id = 0, max_new_tokens = 32)
    finally:
        handle.remove()
    assert torch.float32 in seen  # the frozen projections still decode through the fp32 copy
    assert frozen.weight.dtype == torch.bfloat16
    assert trainable.weight.dtype == torch.bfloat16
    step()  # bf16 moments against a bf16 parameter and gradient
    assert trainable.weight.dtype == torch.bfloat16


def test_fp32_rollout_copy_counts_every_local_training_process(avx2_host, monkeypatch):
    """Each rank on the host makes its own copy from the same free RAM."""
    from unsloth_zoo.cpu.ops import Bf16StoredLinear
    model = _tiny_bf16_llama()
    extra = 2 * sum(m.weight.numel() for m in model.modules() if isinstance(m, Bf16StoredLinear))
    monkeypatch.setattr(avx2_host, "available_ram", lambda: int(extra / avx2_host._FP32_ROLLOUT_RAM_SHARE) * 2)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    assert avx2_host._fp32_rollout_fits(model, 8, 64) is True
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # four copies do not fit in twice one copy's budget
    assert avx2_host._fp32_rollout_fits(model, 8, 64) is False
