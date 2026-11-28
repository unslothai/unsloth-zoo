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
"""CPU train-mode rollouts on x86 without native bf16 decode the frozen projections and the LM head through FBGEMM fp16
packs: same greedy tokens as the per-call path, packs built once and rebuilt when a weight changes, forwards restored
after generate (also on error); eval-mode generate, autograd calls, long prefills, trainable weights, weights that
are not bf16 values, other CPUs and UNSLOTH_CPU_FP16_DECODE=0 keep the stock path."""

import platform

import pytest
import torch
import torch.nn.functional as F


def _fbgemm():
    from unsloth_zoo.cpu.fp16_decode import fp16_decode_available
    return fp16_decode_available()


needs_fbgemm = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64"), reason = "FBGEMM fp16 GEMM ships in x86 torch builds only",
)


@pytest.fixture
def no_bf16(monkeypatch):
    """Runs as on x86 without native bf16 (fp32 training), the CPUs this path is for."""
    from unsloth_zoo.cpu import fp16_decode, runtime
    monkeypatch.delenv("UNSLOTH_CPU_FP16_DECODE", raising = False)
    monkeypatch.setattr(runtime, "cpu_bf16_supported", lambda: False)
    fp16_decode.ENGAGED[0] = 0
    return fp16_decode


def _tiny_llama(bf16_stored = True, tie = True):
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.ops import store_frozen_linears_in_bf16
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 96, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 128,
                         tie_word_embeddings = tie)
    model = LlamaForCausalLM(config).bfloat16()
    if bf16_stored:
        assert store_frozen_linears_in_bf16(model) > 0
        model._unsloth_cpu_frozen_bf16 = "training"
    else:
        model.float()  # fp32 storage of bf16 checkpoint values (Windows / AVX-512 Linux)
    for param in model.parameters():
        param.requires_grad_(False)
    loader.FastCPUModel._attach(model)
    return model.train()


def _generate(model, rows = 2, new = 8):
    ids = torch.randint(1, 96, (rows, 5), generator = torch.Generator().manual_seed(0))
    return model.generate(input_ids = ids, do_sample = False, max_new_tokens = new, pad_token_id = 0)


def _packed(model):
    from unsloth_zoo.cpu.fp16_decode import _PACKS
    return [m for m in model.modules() if _PACKS.get(m, (None, None))[1] is not None]


@needs_fbgemm
def test_fbgemm_fp16_gemm_available_on_x86():
    assert _fbgemm()


@needs_fbgemm
@pytest.mark.parametrize("bf16_stored", [True, False])
def test_rollout_decodes_through_packs_same_tokens(no_bf16, monkeypatch, bf16_stored):
    model = _tiny_llama(bf16_stored)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    tokens = _generate(model)
    engaged = no_bf16.ENGAGED[0]
    # 7 projections x 2 layers + the LM head, on the prompt and on every decoded token
    assert engaged >= 15 * 8
    assert len(_packed(model)) == 15
    assert not any("forward" in m.__dict__ for m in model.modules())  # own forwards back after generate
    state = model.state_dict()
    assert all(torch.equal(state[k], v) and state[k].dtype == v.dtype for k, v in weights.items())
    monkeypatch.setenv("UNSLOTH_CPU_FP16_DECODE", "0")
    no_bf16.ENGAGED[0] = 0
    assert torch.equal(_generate(model), tokens)
    assert no_bf16.ENGAGED[0] == 0


@needs_fbgemm
def test_packs_are_built_once_and_rebuilt_when_the_weight_changes(no_bf16, monkeypatch):
    assert _fbgemm()  # the cached availability check runs before the count starts
    model = _tiny_llama()
    calls = []
    prepack = torch.ops.quantized.linear_prepack_fp16

    class Ops:
        linear_dynamic_fp16 = torch.ops.quantized.linear_dynamic_fp16

        @staticmethod
        def linear_prepack_fp16(w, b):
            calls.append(tuple(w.shape))
            return prepack(w, b)
    monkeypatch.setattr(torch.ops, "quantized", Ops)
    _generate(model)
    first = len(calls)
    assert first == 15
    _generate(model)
    assert len(calls) == first  # cached
    q = model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        q.weight.mul_(2)
    _generate(model)
    assert len(calls) == first + 1
    x = torch.randn(3, 32)
    with torch.no_grad(), no_bf16.fp16_decode(model) as ctx:
        assert ctx.active
        y = q(x)
    torch.testing.assert_close(y, F.linear(x, q.weight.float()), rtol = 1e-5, atol = 1e-5)


@needs_fbgemm
def test_packs_follow_peft_merge_and_unmerge(no_bf16):
    """bf16-stored base: the merged weight is still bf16 numbers, so it is packed again (an fp32 merge is not)."""
    from peft import LoraConfig, get_peft_model
    model = _tiny_llama()
    model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False))
    model.train()
    lora = model.base_model.model.model.layers[0].self_attn.q_proj
    x = torch.randn(2, 32)

    def packed_out():
        with torch.no_grad(), no_bf16.fp16_decode(model) as ctx:
            assert ctx.active
            calls = no_bf16.ENGAGED[0]
            y = lora.base_layer(x)
            assert no_bf16.ENGAGED[0] == calls + 1  # answered from the pack
            return y
    before = packed_out()
    torch.testing.assert_close(before, F.linear(x, lora.base_layer.weight.float()), rtol = 1e-5, atol = 1e-5)
    model.merge_adapter()  # writes the merged weight through .data: same storage and version
    merged = packed_out()
    torch.testing.assert_close(merged, F.linear(x, lora.base_layer.weight.float()), rtol = 1e-5, atol = 1e-5)
    assert (merged - before).abs().max() > 1e-3
    model.unmerge_adapter()
    torch.testing.assert_close(packed_out(), F.linear(x, lora.base_layer.weight.float()), rtol = 1e-5, atol = 1e-5)
    # A full merge + unmerge between two rollouts: merged_adapters is back where it was, but bf16 rounding moved the
    # weight, so the pack must be rebuilt from the current bits.
    with torch.no_grad():
        lora.base_layer.weight.add_(torch.randn_like(lora.base_layer.weight.float()).bfloat16())  # a new version
    packed_out()
    model.merge_adapter()
    model.unmerge_adapter()
    with torch.no_grad():
        lora.base_layer.weight.data.mul_(1.5)  # a .data write: no new storage, no new version
    torch.testing.assert_close(packed_out(), F.linear(x, lora.base_layer.weight.float()), rtol = 1e-5, atol = 1e-5)


@needs_fbgemm
def test_disabled_adapter_unmerged_mid_rollout_uses_the_unmerged_weight(no_bf16):
    """merge_adapter, then a rollout under disable_adapter: PEFT unmerges inside the LoRA forward, after the packs
    were picked, so a pack built from the merged weight must not answer that call."""
    from peft import LoraConfig, get_peft_model
    model = _tiny_llama()
    model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False))
    model.train()
    lora = model.base_model.model.model.layers[0].self_attn.q_proj
    x = torch.randn(2, 32)
    model.merge_adapter()
    with torch.no_grad(), no_bf16.fp16_decode(model) as ctx:
        assert ctx.active
        calls = no_bf16.ENGAGED[0]
        merged = lora.base_layer(x)
        assert no_bf16.ENGAGED[0] == calls + 1  # packed from the merged weight
        with model.disable_adapter():
            y = lora(x)  # PEFT unmerges here, then calls base_layer
    assert not lora.merged
    unmerged = F.linear(x, lora.base_layer.weight.float())
    assert (merged - unmerged).abs().max() > 1e-3
    torch.testing.assert_close(y, unmerged, rtol = 1e-5, atol = 1e-5)


@needs_fbgemm
def test_rollout_without_packs_keeps_the_fp32_copy(no_bf16, monkeypatch):
    """No packs (RAM short for them): a wide rollout on an AVX2 host still decodes through the fp32 copy."""
    from unsloth_zoo.cpu import loader
    model = _tiny_llama()
    monkeypatch.setattr(no_bf16, "_prepare", lambda model: None)
    monkeypatch.setattr(loader, "_fp32_rollout_fits", lambda model, rows = None, new_tokens = None: True)
    seen = []
    stored = [m for m in model.modules() if type(m).__name__ == "Bf16StoredLinear"]
    handles = [m.register_forward_pre_hook(lambda mod, args: seen.append(mod.weight.dtype)) for m in stored]
    try:
        _generate(model)
    finally:
        for h in handles:
            h.remove()
    assert set(seen) == {torch.float32} and {m.weight.dtype for m in stored} == {torch.bfloat16}
    assert no_bf16.ENGAGED[0] == 0


@needs_fbgemm
def test_rollout_wider_than_the_pack_limit_keeps_the_fp32_copy(no_bf16, monkeypatch):
    """More rows per decoded token than the packs serve: no packs, the fp32 copy (AVX2 hosts) as before."""
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(no_bf16, "_MAX_ROWS", 4)
    monkeypatch.setattr(loader, "_fp32_rollout_fits", lambda model, rows = None, new_tokens = None: rows == 8)
    model = _tiny_llama()
    stored = [m for m in model.modules() if type(m).__name__ == "Bf16StoredLinear"]
    seen = []
    handles = [m.register_forward_pre_hook(lambda mod, args: seen.append(mod.weight.dtype)) for m in stored]
    try:
        _generate(model, rows = 8)
    finally:
        for h in handles:
            h.remove()
    assert set(seen) == {torch.float32} and no_bf16.ENGAGED[0] == 0 and not _packed(model)
    assert no_bf16.fp16_decode_wanted(model, rows = 4) and not no_bf16.fp16_decode_wanted(model, rows = 5)


@needs_fbgemm
def test_numerics_match_fp32_linear_and_blocked_head(no_bf16, monkeypatch):
    monkeypatch.setattr(no_bf16, "_HEAD_BLOCK", 40)  # 96-row vocabulary in 3 blocks
    model = _tiny_llama(bf16_stored = False)
    head = model.get_output_embeddings()
    x = torch.randn(2, 1, 32)
    with torch.no_grad():
        ref = F.linear(x, head.weight)
        oracle = F.linear(x.double(), head.weight.double())
        with no_bf16.fp16_decode(model) as ctx:
            assert ctx.active
            y = head(x)
    assert len(no_bf16._PACKS[head][1]) == 3
    assert no_bf16.ENGAGED[0] == 1 and y.shape == ref.shape
    torch.testing.assert_close(y, ref, rtol = 1e-5, atol = 1e-5)
    assert (y.double() - oracle).abs().max() < 1e-5


@needs_fbgemm
def test_stock_path_for_autograd_long_prefill_and_wrong_dtype(no_bf16):
    model = _tiny_llama()
    q = model.model.layers[0].self_attn.q_proj
    with no_bf16.fp16_decode(model) as ctx:
        assert ctx.active
        q(torch.randn(2, 32, requires_grad = True))  # autograd enabled
        with torch.no_grad():
            q(torch.randn(no_bf16._MAX_ROWS + 1, 32))  # prefill longer than the gate
            q(torch.randn(2, 32, dtype = torch.float64))  # not fp32 activations
            assert no_bf16.ENGAGED[0] == 0  # none of the above
            q(torch.randn(no_bf16._MAX_ROWS, 32))
    assert no_bf16.ENGAGED[0] == 1


@needs_fbgemm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_autocast_keeps_the_stock_path(no_bf16, dtype):
    model = _tiny_llama()
    q = model.model.layers[0].self_attn.q_proj
    x = torch.randn(2, 32)
    with torch.no_grad(), no_bf16.fp16_decode(model) as ctx:
        assert ctx.active
        with torch.autocast("cpu", dtype = dtype):
            y = q(x)
    assert y.dtype == dtype and no_bf16.ENGAGED[0] == 0


@needs_fbgemm
def test_forwards_restored_on_error(no_bf16):
    model = _tiny_llama()
    with pytest.raises(RuntimeError, match = "boom"):
        with no_bf16.fp16_decode(model) as ctx:
            assert ctx.active and any("forward" in m.__dict__ for m in model.modules())
            raise RuntimeError("boom")
    assert not any("forward" in m.__dict__ for m in model.modules())


@needs_fbgemm
def test_eval_mode_generate_keeps_stock_path(no_bf16):
    model = _tiny_llama()
    model.eval()
    _generate(model)
    assert no_bf16.ENGAGED[0] == 0 and not _packed(model)


@needs_fbgemm
def test_for_inference_releases_packs_and_deepcopy_carries_none(no_bf16):
    import copy
    from unsloth_zoo.cpu.loader import FastCPUModel
    model = _tiny_llama()
    _generate(model)
    assert len(_packed(model)) == 15
    clone = copy.deepcopy(model)
    assert not _packed(clone)
    FastCPUModel.for_inference(model)
    assert not _packed(model)


@needs_fbgemm
def test_skips_trainable_out_of_range_and_non_bf16_weights(no_bf16):
    model = _tiny_llama(bf16_stored = False)
    layer = model.model.layers[0]
    layer.mlp.up_proj.weight.requires_grad_(True)  # trainable (full finetuning, modules_to_save)
    with torch.no_grad():
        layer.mlp.down_proj.weight[0, 0] = 70000.0  # outside fp16's range
        layer.self_attn.o_proj.weight.add_(1e-9)  # fp32 values that are not bf16 numbers (fp32 checkpoint)
    with no_bf16.fp16_decode(model):
        pass
    packed = set(map(id, _packed(model)))
    assert id(layer.mlp.up_proj) not in packed
    assert id(layer.mlp.down_proj) not in packed
    assert id(layer.self_attn.o_proj) not in packed
    assert id(layer.self_attn.q_proj) in packed


@needs_fbgemm
def test_unpackable_weights_are_checked_once_per_version(no_bf16, monkeypatch):
    model = _tiny_llama(bf16_stored = False)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.add_(1e-9)  # fp32 values that are not bf16 numbers: never packed
    checks = []
    packable = no_bf16._packable
    monkeypatch.setattr(no_bf16, "_packable", lambda weight: checks.append(1) or packable(weight))
    _generate(model)
    first = len(checks)
    assert first == 15 and not _packed(model)
    _generate(model)
    assert len(checks) == first  # remembered while the weights are unchanged
    q = model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        q.weight.copy_(q.weight.bfloat16().float())  # now bf16 numbers, a new version
    _generate(model)
    assert len(checks) == first + 1 and q in _packed(model)


@needs_fbgemm
def test_no_packs_when_they_do_not_fit_in_ram(no_bf16, monkeypatch):
    from unsloth_zoo.cpu import runtime
    monkeypatch.setattr(runtime, "available_ram", lambda: 1024)
    checked = []
    monkeypatch.setattr(no_bf16, "_packable", lambda weight: checked.append(weight) or True)
    model = _tiny_llama()
    _generate(model)
    assert no_bf16.ENGAGED[0] == 0 and not _packed(model)
    assert not checked  # the RAM check comes before any eligibility check allocates


def test_eligibility_check_in_row_blocks(monkeypatch):
    from unsloth_zoo.cpu import fp16_decode
    monkeypatch.setattr(fp16_decode, "_CHECK_ROWS", 3)
    w = torch.randn(10, 4).bfloat16().float()
    assert fp16_decode._packable(w) and fp16_decode._packable(w.bfloat16())
    for row, value in ((9, 1e-9), (8, 70000.0), (7, float("nan")), (0, -70000.0)):
        bad = w.clone()
        bad[row, 1] = value
        assert not fp16_decode._packable(bad), (row, value)


@needs_fbgemm
def test_packs_under_an_fbgemm_engine_and_restores_the_users(no_bf16):
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        pytest.skip("no qnnpack engine in this build")
    model = _tiny_llama()
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = "qnnpack"  # its prepack has no fp16 packs
    try:
        _generate(model)
        assert torch.backends.quantized.engine == "qnnpack"
    finally:
        torch.backends.quantized.engine = previous
    assert len(_packed(model)) == 15 and no_bf16.ENGAGED[0] > 0


def test_no_fbgemm_engine_means_unavailable(monkeypatch):
    from unsloth_zoo.cpu import fp16_decode
    engine = type(torch.backends.quantized)
    monkeypatch.setattr(engine, "supported_engines", property(lambda self: ["onednn"]))  # Windows wheels
    monkeypatch.setattr(engine, "engine", property(lambda self: "onednn", lambda self, value: None))
    assert fp16_decode._fbgemm_engine() is None
    fp16_decode.fp16_decode_available.cache_clear()
    try:
        assert fp16_decode.fp16_decode_available() is False
    finally:
        fp16_decode.fp16_decode_available.cache_clear()


@pytest.mark.parametrize("env, machine, bf16, available, expected", [
    ("", "x86_64", False, True, True),
    ("", "AMD64", False, True, True),
    ("0", "x86_64", False, True, False),
    ("", "x86_64", True, True, False),  # native bf16: training computes in bf16, not this path
    ("1", "x86_64", True, True, True),
    ("", "arm64", False, True, False),
    ("", "x86_64", False, False, False),  # build without FBGEMM
])
def test_gate(monkeypatch, env, machine, bf16, available, expected):
    from unsloth_zoo.cpu import fp16_decode, runtime
    monkeypatch.setenv("UNSLOTH_CPU_FP16_DECODE", env)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(runtime, "cpu_bf16_supported", lambda: bf16)
    monkeypatch.setattr(fp16_decode, "fp16_decode_available", lambda: available)
    model = torch.nn.Linear(4, 4).train()
    assert fp16_decode.fp16_decode_wanted(model) is expected
    assert fp16_decode.fp16_decode_wanted(model.eval()) is False
