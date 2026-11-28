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
"""bf16 CPU training on Arm: attention calls whose query needs a gradient run through the fp32 SDPA kernel (exact
upcast in, output rounded to bf16, gradients back in bf16), matching an fp32 SDPA on the same bf16 values; no-grad
calls, fp32 queries, other CPUs and UNSLOTH_CPU_SDPA_FP32=0 keep transformers' own function."""
import platform

import pytest
import torch


@pytest.fixture
def mapping(monkeypatch):
    from transformers.integrations import sdpa_attention
    from transformers.modeling_utils import AttentionInterface
    from unsloth_zoo.cpu import ops
    original = AttentionInterface._global_mapping["sdpa"]
    # Restored after each test: the patch writes the global attention registry.
    monkeypatch.setitem(AttentionInterface._global_mapping, "sdpa", original)
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", sdpa_attention.sdpa_attention_forward)
    monkeypatch.setattr(ops, "SDPA_FP32_ENGAGED", [0])
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.delenv("UNSLOTH_CPU_SDPA_FP32", raising = False)
    return AttentionInterface._global_mapping, original


class _Attn(torch.nn.Module):
    num_key_value_groups = 2
    is_causal = True


def _qkv(dtype = torch.bfloat16, grad = True, seq = 24):
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(2, 4, seq, 32, generator = gen).to(dtype)
    k = torch.randn(2, 2, seq, 32, generator = gen).to(dtype)
    v = torch.randn(2, 2, seq, 32, generator = gen).to(dtype)
    return [t.requires_grad_(grad) for t in (q, k, v)]


def _cpu_autocast_on():
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # torch < 2.4
        return torch.is_autocast_cpu_enabled()


def _reference(original, q, k, v, mask = None):
    """transformers' own function in fp32 on the same bf16 values (same GQA route and kernel as the patched call)."""
    q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    out, _ = original(_Attn(), q32, k32, v32, None if mask is None else mask.float())
    return out, (q32, k32, v32)


def test_gate(monkeypatch):
    from unsloth_zoo.cpu import ops
    monkeypatch.delenv("UNSLOTH_CPU_SDPA_FP32", raising = False)
    for machine, expected in (("aarch64", True), ("arm64", True), ("ARM64", True), ("x86_64", False), ("AMD64", False)):
        monkeypatch.setattr(platform, "machine", lambda m = machine: m)
        assert ops._bf16_sdpa_in_fp32_pays() is expected, machine
    monkeypatch.setenv("UNSLOTH_CPU_SDPA_FP32", "1")
    assert ops._bf16_sdpa_in_fp32_pays() is True
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("UNSLOTH_CPU_SDPA_FP32", "0")
    assert ops._bf16_sdpa_in_fp32_pays() is False


def test_install_is_idempotent_and_registered(mapping):
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu.ops import patch_cpu_bf16_sdpa_fp32
    table, original = mapping
    assert patch_cpu_bf16_sdpa_fp32() is True
    wrapped = table["sdpa"]
    assert wrapped is not original and wrapped._unsloth_cpu_fp32 and wrapped.__wrapped__ is original
    assert sdpa_attention.sdpa_attention_forward is wrapped
    assert patch_cpu_bf16_sdpa_fp32() is True and table["sdpa"] is wrapped


def test_off_on_other_cpus_and_with_the_kill_switch(mapping, monkeypatch):
    from unsloth_zoo.cpu.ops import patch_cpu_bf16_sdpa_fp32
    table, original = mapping
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert patch_cpu_bf16_sdpa_fp32() is False and table["sdpa"] is original
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("UNSLOTH_CPU_SDPA_FP32", "0")
    assert patch_cpu_bf16_sdpa_fp32() is False and table["sdpa"] is original


@pytest.mark.parametrize("masked", [False, True])
def test_training_call_matches_fp32_sdpa(mapping, masked):
    from unsloth_zoo.cpu import ops
    table, original = mapping
    ops.patch_cpu_bf16_sdpa_fp32()
    q, k, v = _qkv()
    mask = None
    if masked:  # additive float mask in the model dtype, as transformers builds it for a bf16 model
        mask = torch.zeros(2, 1, 24, 24, dtype = torch.bfloat16)
        mask.masked_fill_(torch.ones(24, 24, dtype = torch.bool).triu(1), torch.finfo(torch.bfloat16).min)
    out, weights = table["sdpa"](_Attn(), q, k, v, mask)
    assert weights is None and out.dtype == torch.bfloat16 and ops.SDPA_FP32_ENGAGED[0] == 1
    ref, leaves = _reference(original, q, k, v, mask)
    assert torch.equal(out.float(), ref.detach().bfloat16().float())
    grad = torch.randn(out.shape, generator = torch.Generator().manual_seed(1)).bfloat16()
    out.backward(grad)
    ref.backward(grad.float())
    for got, want in zip((q.grad, k.grad, v.grad), leaves):
        assert got.dtype == torch.bfloat16
        assert torch.equal(got.float(), want.grad.bfloat16().float())


def test_inner_call_sees_fp32_even_under_autocast(mapping, monkeypatch):
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu import ops
    table, original = mapping
    seen = []

    def spy(module, query, key, value, attention_mask, *args, **kwargs):
        seen.append((query.dtype, key.dtype, value.dtype, None if attention_mask is None else attention_mask.dtype,
                     _cpu_autocast_on()))
        return original(module, query, key, value, attention_mask, *args, **kwargs)

    monkeypatch.setitem(table, "sdpa", spy)
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", spy)
    ops.patch_cpu_bf16_sdpa_fp32()
    q, k, v = _qkv()
    with torch.autocast("cpu", dtype = torch.bfloat16):
        out, _ = table["sdpa"](_Attn(), q, k, v, torch.ones(2, 1, 24, 24, dtype = torch.bool).tril())
    assert out.dtype == torch.bfloat16
    assert seen[0][:4] == (torch.float32, torch.float32, torch.float32, torch.bool)
    assert seen[0][4] is False


def test_no_grad_and_fp32_calls_keep_the_stock_function(mapping, monkeypatch):
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu import ops
    table, original = mapping
    seen = []

    def spy(module, query, *args, **kwargs):
        seen.append(query.dtype)
        return original(module, query, *args, **kwargs)

    monkeypatch.setitem(table, "sdpa", spy)
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", spy)
    ops.patch_cpu_bf16_sdpa_fp32()
    with torch.no_grad():
        table["sdpa"](_Attn(), *_qkv(), None)               # decode / reference forward
    table["sdpa"](_Attn(), *_qkv(grad = False), None)       # frozen activations, nothing to differentiate
    table["sdpa"](_Attn(), *_qkv(torch.float32), None)      # fp32 training
    assert seen == [torch.bfloat16, torch.bfloat16, torch.float32] and ops.SDPA_FP32_ENGAGED[0] == 0


def test_tiny_model_training_step_engages(mapping):
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu import ops
    ops.patch_cpu_bf16_sdpa_fp32()
    config = LlamaConfig(vocab_size = 64, hidden_size = 64, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).bfloat16()
    ids = torch.randint(0, 64, (2, 16), generator = torch.Generator().manual_seed(0))
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    assert ops.SDPA_FP32_ENGAGED[0] == 2  # one per layer
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        model(input_ids = ids)
    assert ops.SDPA_FP32_ENGAGED[0] == 2
