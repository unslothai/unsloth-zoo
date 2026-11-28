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
"""bf16 CPU training on Arm: attention calls whose query needs a gradient, and full-sequence no-grad calls (reference
forwards), run through the fp32 SDPA kernel (exact upcast in, output rounded to bf16, gradients back in bf16),
matching an fp32 SDPA on the same bf16 values; cached decode steps, fp32 queries, other CPUs and
UNSLOTH_CPU_SDPA_FP32=0 keep transformers' own function."""
import platform

import pytest
import torch


@pytest.fixture
def mapping(monkeypatch):
    """The attention registry as models read it (ALL_ATTENTION_FUNCTIONS: its local entry, then the global one),
    starting from transformers' own function and restored after each test; returns (registry, that function)."""
    from transformers.integrations import sdpa_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    from unsloth_zoo.cpu import ops
    original = AttentionInterface._global_mapping["sdpa"]
    # Another patch (the bool causal mask fix) may own the local entry and the module attribute already.
    monkeypatch.setattr(ALL_ATTENTION_FUNCTIONS, "_local_mapping",
                        {k: v for k, v in ALL_ATTENTION_FUNCTIONS._local_mapping.items() if k != "sdpa"})
    monkeypatch.setitem(AttentionInterface._global_mapping, "sdpa", original)
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", original)
    monkeypatch.setattr(ops, "SDPA_FP32_ENGAGED", [0])
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.delenv("UNSLOTH_CPU_SDPA_FP32", raising = False)
    return ALL_ATTENTION_FUNCTIONS, original


class _Attn(torch.nn.Module):
    num_key_value_groups = 2
    is_causal = True


class _Cross(torch.nn.Module):
    num_key_value_groups = 2
    is_causal = False


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


def test_chains_with_another_sdpa_patch_in_either_order(mapping, monkeypatch):
    """A registry override written before this patch is wrapped by it; one written after (it wraps the module
    attribute, as patch_sdpa_bool_causal_mask does) wraps this one. Either way both run."""
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu import ops
    table, original = mapping
    calls = []

    def other_patch():
        inner = sdpa_attention.sdpa_attention_forward

        def other(module, query, *args, **kwargs):
            calls.append(query.dtype)
            return inner(module, query, *args, **kwargs)
        sdpa_attention.sdpa_attention_forward = other
        table["sdpa"] = other
        return other

    first = other_patch()
    ops.patch_cpu_bf16_sdpa_fp32()
    assert table["sdpa"].__wrapped__ is first
    table["sdpa"](_Attn(), *_qkv(), None)
    assert calls == [torch.float32] and ops.SDPA_FP32_ENGAGED[0] == 1

    calls.clear()
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", original)
    del table._local_mapping["sdpa"]
    ops.patch_cpu_bf16_sdpa_fp32()
    other_patch()
    table["sdpa"](_Attn(), *_qkv(), None)
    assert calls == [torch.bfloat16] and ops.SDPA_FP32_ENGAGED[0] == 2


def test_off_on_other_cpus_and_with_the_kill_switch(mapping, monkeypatch):
    from unsloth_zoo.cpu.ops import patch_cpu_bf16_sdpa_fp32
    table, original = mapping
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert patch_cpu_bf16_sdpa_fp32() is False and table["sdpa"] is original
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("UNSLOTH_CPU_SDPA_FP32", "0")
    assert patch_cpu_bf16_sdpa_fp32() is False and table["sdpa"] is original and "sdpa" not in table._local_mapping


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

    table["sdpa"] = spy
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", spy)
    ops.patch_cpu_bf16_sdpa_fp32()
    q, k, v = _qkv()
    with torch.autocast("cpu", dtype = torch.bfloat16):
        out, _ = table["sdpa"](_Attn(), q, k, v, torch.ones(2, 1, 24, 24, dtype = torch.bool).tril())
    assert out.dtype == torch.bfloat16
    assert seen[0][:4] == (torch.float32, torch.float32, torch.float32, torch.bool)
    assert seen[0][4] is False


def test_extra_bf16_tensors_are_upcast_and_a_paged_cache_keeps_the_stock_path(mapping, monkeypatch):
    """T5-style position_bias (keyword or positional) joins the mask, so it reaches the fp32 kernel as fp32; a
    paged cache argument (written with the keys / values) keeps the stock path."""
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu import ops
    table, _ = mapping
    seen = []

    def spy(module, query, key, value, attention_mask, *args, **kwargs):
        seen.append((query.dtype, [a.dtype for a in args if isinstance(a, torch.Tensor)],
                     {k: v.dtype for k, v in kwargs.items() if isinstance(v, torch.Tensor)}))
        return torch.zeros(query.shape[0], query.shape[2], query.shape[1], query.shape[3], dtype = query.dtype), None

    table["sdpa"] = spy
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", spy)
    ops.patch_cpu_bf16_sdpa_fp32()
    bias = torch.zeros(1, 4, 24, 24, dtype = torch.bfloat16)
    table["sdpa"](_Attn(), *_qkv(), None, position_bias = bias, layer_idx = torch.tensor(3))
    table["sdpa"](_Attn(), *_qkv(), None, 0.0, None, None, bias)
    table["sdpa"](_Attn(), *_qkv(), None, cache = object())
    assert seen[0] == (torch.float32, [], {"position_bias": torch.float32, "layer_idx": torch.int64})
    assert seen[1] == (torch.float32, [torch.float32], {})
    assert seen[2] == (torch.bfloat16, [], {})
    assert ops.SDPA_FP32_ENGAGED[0] == 2


def test_tiny_t5_position_bias_trains(mapping):
    from transformers import T5Config, T5ForConditionalGeneration
    from unsloth_zoo.cpu import ops
    ops.patch_cpu_bf16_sdpa_fp32()
    config = T5Config(vocab_size = 64, d_model = 32, d_kv = 8, d_ff = 64, num_layers = 1, num_decoder_layers = 1,
                      num_heads = 4, decoder_start_token_id = 0, pad_token_id = 0, dropout_rate = 0.0)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    try:
        model = T5ForConditionalGeneration(config).bfloat16()
    except ValueError as error:  # transformers 4.x: T5 has no sdpa attention
        pytest.skip(str(error)[:80])
    ids = torch.randint(1, 64, (2, 12), generator = torch.Generator().manual_seed(0))
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    if ops.SDPA_FP32_ENGAGED[0] == 0:
        pytest.skip("this transformers version's T5 does not call the sdpa attention function")
    # Source and target lengths differ: the no-grad reference forward's cross-attention takes the fp32 kernel too.
    target = ids[:, :7]
    policy = model(input_ids = ids, decoder_input_ids = target).logits
    with torch.no_grad():
        assert torch.equal(model(input_ids = ids, decoder_input_ids = target).logits, policy.detach())


def test_cached_decode_and_fp32_calls_keep_the_stock_function_cross_attention_does_not(mapping, monkeypatch):
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu import ops
    table, original = mapping
    seen = []

    def spy(module, query, *args, **kwargs):
        seen.append(query.dtype)
        return original(module, query, *args, **kwargs)

    table["sdpa"] = spy
    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", spy)
    ops.patch_cpu_bf16_sdpa_fp32()
    q, k, v = _qkv(grad = False)
    with torch.no_grad():
        table["sdpa"](_Attn(), q[:, :, -1:], k, v, None)    # one decoded token against a 24-token cache
        table["sdpa"](_Attn(), q[:, :, -3:], k, v, None)    # a few tokens against the cache
        table["sdpa"](_Cross(), q[:, :, -3:], k, v, None, is_causal = True)  # the call's own flag wins
    table["sdpa"](_Attn(), *_qkv(torch.float32), None)      # fp32 training
    assert seen == [torch.bfloat16] * 3 + [torch.float32] and ops.SDPA_FP32_ENGAGED[0] == 0
    with torch.no_grad():  # cross-attention (non-causal), target shorter than source: a reference forward's call
        table["sdpa"](_Cross(), q[:, :, -7:], k, v, None)
        table["sdpa"](_Attn(), q[:, :, -7:], k, v, None, is_causal = False)
    assert seen[4:] == [torch.float32, torch.float32] and ops.SDPA_FP32_ENGAGED[0] == 2


def test_no_grad_full_sequence_matches_the_training_call(mapping):
    """DPO reference / GRPO old-policy forwards run without grad: they take the same fp32 kernel, so with equal
    weights the policy and reference attention outputs are bitwise equal (DPO's first loss stays log 2)."""
    from unsloth_zoo.cpu import ops
    table, _ = mapping
    ops.patch_cpu_bf16_sdpa_fp32()
    q, k, v = _qkv()
    trained, _ = table["sdpa"](_Attn(), q, k, v, None)
    with torch.no_grad():
        reference, _ = table["sdpa"](_Attn(), q.detach(), k.detach(), v.detach(), None)
    reference_no_grad_input, _ = table["sdpa"](_Attn(), *_qkv(grad = False), None)
    assert torch.equal(trained.detach(), reference) and torch.equal(reference, reference_no_grad_input)
    assert ops.SDPA_FP32_ENGAGED[0] == 3


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
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    assert ops.SDPA_FP32_ENGAGED[0] == 2  # one per layer
    assert torch.isfinite(out.loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in model.parameters() if p.requires_grad)
    policy = model(input_ids = ids).logits
    with torch.no_grad():  # a reference forward: same attention, same logits
        assert torch.equal(model(input_ids = ids).logits, policy.detach())
    assert ops.SDPA_FP32_ENGAGED[0] == 6
    model.generate(input_ids = ids[:, :8], max_new_tokens = 4, do_sample = False, pad_token_id = 0)
    assert ops.SDPA_FP32_ENGAGED[0] == 8  # the prefill only; the cached decode steps keep the bf16 kernel
