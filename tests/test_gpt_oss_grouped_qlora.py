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

"""Grouped QLoRA training for gpt-oss ModuleList NF4 experts (gpt_oss_grouped_qlora).

* The pointer-table stacked dequant is torch.equal to bitsandbytes per expert (nested and
  flat absmax, every quant dtype, blocksize 64 / 128, subnormal scales that bnb flushes).
* The grouped forward with per-expert LoRA matches the per-expert loop
  (UNSLOTH_GPTOSS_GROUPED=0) in output and in every LoRA / input gradient, runs without a
  host sync, and moving one expert's lora_B moves the output.
* Unsupported adapters (dropout, DoRA, lora_bias, several active, disabled), fp16 inputs
  and experts not computing in bf16 keep the per-expert loop. fp32 inputs (the residual
  stream after the first MoE layer) take the grouped path in bf16, as Linear4bit does.
"""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
bnb = pytest.importorskip("bitsandbytes")
peft = pytest.importorskip("peft")

from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq
from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit, torch_native_forward
from unsloth_zoo.temporary_patches.gpt_oss_routed import _build_table
from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported

DT = torch.bfloat16
E, TOP_K, H, I = 8, 4, 256, 192

needs_grouped_mm = pytest.mark.skipif(
    not torch.cuda.is_bf16_supported() or not _check_torch_grouped_mm_supported(),
    reason = "torch._grouped_mm / bf16 unavailable",
)


def _linear4bit(i, o, nested, seed, dtype = DT, blocksize = 64):
    g = torch.Generator().manual_seed(seed)
    lin = bnb.nn.Linear4bit(i, o, bias = True, compute_dtype = dtype, quant_type = "nf4",
                            compress_statistics = nested, quant_storage = torch.uint8)
    scale = 0.02 * (1 + seed % 5)  # distinct absmax / offsets per expert
    lin.weight = bnb.nn.Params4bit((torch.randn(o, i, generator = g) * scale).to(dtype), requires_grad = False,
                                   quant_type = "nf4", compress_statistics = nested, blocksize = blocksize)
    lin.bias = torch.nn.Parameter((torch.randn(o, generator = g) * 0.1).to(dtype), requires_grad = False)
    return lin.cuda()


class _Experts(torch.nn.Module):
    _grouped_bnb4bit_ready = GptOssExpertsBnb4bit._grouped_bnb4bit_ready
    _forward_grouped_bnb4bit = GptOssExpertsBnb4bit._forward_grouped_bnb4bit
    forward = torch_native_forward

    def __init__(self, nested = True, dtype = DT, blocksize = 64):
        super().__init__()
        self.gate_up_projs = torch.nn.ModuleList([_linear4bit(H, 2 * I, nested, e, dtype, blocksize) for e in range(E)])
        self.down_projs = torch.nn.ModuleList([_linear4bit(I, H, nested, 100 + e, dtype, blocksize) for e in range(E)])
        self.hidden_size, self.alpha, self.limit = H, 1.702, 7.0


def _routing(T, seed = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator = g)
    vals, idx = logits.topk(TOP_K, dim = -1)
    dense = torch.zeros(T, E).scatter_(1, idx, vals.softmax(-1))
    return idx.cuda(), dense.cuda().to(DT)


def _lora_wrap(ex, r = 16, seed = 7, **kwargs):
    kwargs.setdefault("lora_dropout", 0.0)
    cfg = peft.LoraConfig(r = r, lora_alpha = 2 * r, target_modules = r".*(gate_up_projs|down_projs)\.\d+", **kwargs)
    model = peft.inject_adapter_in_model(cfg, ex)
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = (torch.randn(p.shape, generator = g) * 0.05).to(p.device, p.dtype)
    return model


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("subnormal", [False, True])
def test_stacked_dequant_bit_exact(nested, dtype, blocksize, subnormal):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    projs = torch.nn.ModuleList([_linear4bit(H, 2 * I, nested, e, dtype, blocksize) for e in range(E)])
    if subnormal:
        # Products land below FLT_MIN; bitsandbytes (fast-math) flushes them to zero.
        for p in projs:
            qs = p.weight.quant_state
            if nested:
                qs.state2.absmax.mul_(1e-36)
                qs.offset.zero_()
            else:
                qs.absmax.fill_(2e-38)
    tb = _build_table(projs, projs[0].weight.device)
    assert tb is not None
    out = gq.nf4_dequant_expert_stack(tb, dtype)
    assert out is not None and out.shape == (E, 2 * I, H) and out.dtype == dtype
    for e, p in enumerate(projs):
        ref = bnb.functional.dequantize_4bit(p.weight.data, p.weight.quant_state)
        assert torch.equal(out[e], ref), f"expert {e}: {(out[e].float() - ref.float()).abs().max()}"


def test_stacked_dequant_negative_control():
    # A scale off by one bf16 ulp must be caught by the torch.equal comparison above.
    projs = torch.nn.ModuleList([_linear4bit(H, 2 * I, True, e) for e in range(E)])
    tb = _build_table(projs, projs[0].weight.device)
    out = gq.nf4_dequant_expert_stack(tb, DT) * (1 + 2 ** -7)
    ref = bnb.functional.dequantize_4bit(projs[0].weight.data, projs[0].weight.quant_state)
    assert not torch.equal(out[0], ref)


def _run(ex, x, idx, w, grouped, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "1" if grouped else "0")
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    for p in ex.parameters():
        p.grad = None
    xx = x.detach().clone().requires_grad_(True)
    before = dict(gq.CALLS)
    out = ex(xx, idx, w)
    out.float().square().sum().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in ex.named_parameters() if p.requires_grad and p.grad is not None}
    grads["<input>"] = xx.grad.detach().float().clone()
    delta = {k: gq.CALLS[k] - before[k] for k in before}
    return out.detach().float(), grads, delta


def _rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


@needs_grouped_mm
@pytest.mark.parametrize("r", [16, 4])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])   # fp32: residual after the first MoE layer
def test_grouped_lora_matches_per_expert_loop(r, nested, x_dtype, monkeypatch):
    ex = _lora_wrap(_Experts(nested), r = r).train()
    T = 96
    x = torch.randn(1, T, H, device = "cuda", dtype = x_dtype)
    idx, w = _routing(T)
    ref_out, ref_g, ref_calls = _run(ex, x, idx, w, False, monkeypatch)
    out, g, calls = _run(ex, x, idx, w, True, monkeypatch)
    assert ref_calls["forward"] == 0
    assert calls["forward"] == 1 and calls["forward_lora"] == 1
    assert calls["stacked_dequant"] >= 4 and calls["bnb_fallback_dequant"] == 0   # fwd 2 + bwd recompute 2
    assert out.dtype == ref_out.dtype
    assert _rel(out, ref_out) < 2e-2
    lora_names = [n for n in ref_g if "lora_" in n]
    assert len(lora_names) == 4 * E
    assert set(ref_g) == set(g)
    for n in ref_g:
        assert _rel(g[n], ref_g[n]) <= 0.05 + 1e-6, (n, _rel(g[n], ref_g[n]))


@needs_grouped_mm
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in (9, 10),
    reason = "torch._grouped_mm is native only on sm90 / sm100; elsewhere ATen's fallback copies the offsets to the host",
)
def test_grouped_lora_no_host_sync(monkeypatch):
    ex = _lora_wrap(_Experts(True)).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT, requires_grad = True)
    idx, w = _routing(T)
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "1")
    _run(ex, x, idx, w, True, monkeypatch)  # warm up: tables, probes, Triton JIT
    before = gq.CALLS["forward_lora"]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = ex(x, idx, w)
        out.float().sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert gq.CALLS["forward_lora"] == before + 1


@needs_grouped_mm
def test_one_expert_lora_b_moves_output(monkeypatch):
    ex = _lora_wrap(_Experts(True)).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    out0, _, _ = _run(ex, x, idx, w, True, monkeypatch)
    e = int(idx[0, 0])
    with torch.no_grad():
        ex.down_projs[e].lora_B["default"].weight.add_(0.5)
    out1, _, _ = _run(ex, x, idx, w, True, monkeypatch)
    touched = (idx == e).any(-1)
    diff = (out1 - out0).abs().view(T, H).amax(-1)
    # Untouched tokens move only by fp32 index_add_ atomic-order noise.
    assert bool((diff[touched] > 1e-3).all())
    assert bool((diff[~touched] < 1e-5).all())


@needs_grouped_mm
@pytest.mark.parametrize("case", ["dropout", "dora", "lora_bias", "two_active", "disabled", "fp16_input",
                                  "fp32_compute", "forced_fp32_rule"])
def test_unsupported_lora_keeps_loop(case, monkeypatch):
    kwargs = {}
    if case == "dropout":
        kwargs["lora_dropout"] = 0.1
    elif case == "dora":
        kwargs["use_dora"] = True
    elif case == "lora_bias":
        kwargs["lora_bias"] = True
    ex = _lora_wrap(_Experts(True), **kwargs).train()
    if case == "two_active":
        cfg = peft.LoraConfig(r = 8, lora_alpha = 16, target_modules = r".*(gate_up_projs|down_projs)\.\d+")
        peft.inject_adapter_in_model(cfg, ex, adapter_name = "other")
        for m in list(ex.gate_up_projs) + list(ex.down_projs):
            m.set_adapter(["default", "other"])
    elif case == "disabled":
        for m in list(ex.gate_up_projs) + list(ex.down_projs):
            m.enable_adapters(False)
    elif case == "fp32_compute":
        for m in list(ex.gate_up_projs) + list(ex.down_projs):
            m.base_layer.compute_dtype = torch.float32
    elif case == "forced_fp32_rule":
        for m in list(ex.gate_up_projs) + list(ex.down_projs):
            m.base_layer._pre_set_compute_dtype = torch.float32
    T = 32
    dtype = torch.float16 if case == "fp16_input" else DT
    x = torch.randn(1, T, H, device = "cuda", dtype = dtype)
    idx, w = _routing(T)
    _, _, calls = _run(ex, x, idx, w.to(dtype), True, monkeypatch)
    assert calls["forward"] == 0, case


@needs_grouped_mm
def test_lora_free_grouped_uses_stacked_dequant(monkeypatch):
    ex = _Experts(True).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    ref_out, _, _ = _run(ex, x, idx, w, False, monkeypatch)
    out, g, calls = _run(ex, x, idx, w, True, monkeypatch)
    assert calls["forward"] == 1 and calls["forward_lora"] == 0
    assert calls["stacked_dequant"] >= 2 and calls["bnb_fallback_dequant"] == 0
    assert _rel(out, ref_out) < 2e-2


@needs_grouped_mm
def test_kill_switch_triton_falls_back_to_bnb(monkeypatch):
    ex = _lora_wrap(_Experts(True)).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only = True)   # index_add_ without atomics
    try:
        out_t, g_t, _ = _run(ex, x, idx, w, True, monkeypatch)
        monkeypatch.setenv("UNSLOTH_MOE_TRITON_KERNELS", "0")
        out_b, g_b, calls = _run(ex, x, idx, w, True, monkeypatch)
    finally:
        torch.use_deterministic_algorithms(prev)
    assert calls["stacked_dequant"] == 0 and calls["bnb_fallback_dequant"] >= 2
    # Same dequantized weights either way, so the whole step is bit-identical.
    assert torch.equal(out_t, out_b)
    for n in g_t:
        assert torch.equal(g_t[n], g_b[n]), n


@needs_grouped_mm
def test_ready_cache_follows_adapter_changes(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising = False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    ex = _lora_wrap(_Experts(True)).train()
    assert ex._grouped_bnb4bit_ready() is True and ex._unsloth_grouped_lora is not None
    cfg = peft.LoraConfig(r = 8, lora_alpha = 16, target_modules = r".*(gate_up_projs|down_projs)\.\d+")
    peft.inject_adapter_in_model(cfg, ex, adapter_name = "other")
    projs = list(ex.gate_up_projs) + list(ex.down_projs)
    for m in projs:
        m.set_adapter(["default", "other"])
    assert ex._grouped_bnb4bit_ready() is False and ex._unsloth_grouped_lora is None
    for m in projs:
        m.set_adapter("other")
    assert ex._grouped_bnb4bit_ready() is True and ex._unsloth_grouped_lora["gate_up"][0] == "other"
    for m in projs:
        m.enable_adapters(False)
    assert ex._grouped_bnb4bit_ready() is False


@needs_grouped_mm
def test_in_place_bias_edit_is_not_stale(monkeypatch):
    ex = _Experts(True).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    out0, _, _ = _run(ex, x, idx, w, True, monkeypatch)
    with torch.no_grad():
        for p in ex.down_projs:
            p.bias.add_(1.0)
    out1, _, _ = _run(ex, x, idx, w, True, monkeypatch)
    ref1, _, _ = _run(ex, x, idx, w, False, monkeypatch)
    assert _rel(out1, ref1) < 2e-2 and _rel(out1, out0) > 0.1


@needs_grouped_mm
@pytest.mark.parametrize("change", ["dropout", "dropout_p", "merge", "disable", "dora_flag", "unwrap"])
def test_ready_cache_sees_a_middle_expert(change, monkeypatch):
    # A direct edit of one middle expert must re-run the check.
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising = False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    ex = _lora_wrap(_Experts(True)).train()
    assert ex._grouped_bnb4bit_ready() is True
    mid = ex.down_projs[E // 2]
    if change == "dropout":
        mid.lora_dropout["default"] = torch.nn.Dropout(0.1).train()
    elif change == "dropout_p":
        ex.gate_up_projs[E // 2].lora_dropout["default"] = torch.nn.Dropout(0.0).train()
        assert ex._grouped_bnb4bit_ready() is True
        ex.gate_up_projs[E // 2].lora_dropout["default"].p = 0.1
    elif change == "merge":
        mid.merged_adapters.append("default")
    elif change == "disable":
        mid.enable_adapters(False)
    elif change == "dora_flag":
        mid.use_dora["default"] = True
    elif change == "unwrap":
        ex.down_projs[E // 2] = mid.base_layer
    assert ex._grouped_bnb4bit_ready() is False, change
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    _, _, calls = _run(ex, x, idx, w, True, monkeypatch)
    assert calls["forward"] == 0, change


@needs_grouped_mm
def test_unrouted_expert_gets_zero_lora_grads(monkeypatch):
    # No host sync, so unrouted experts get exact zeros where the loop leaves None.
    ex = _lora_wrap(_Experts(True)).train()
    T = 64
    g = torch.Generator().manual_seed(1)
    idx = torch.stack([torch.randperm(E - 1, generator = g)[:TOP_K] + 1 for _ in range(T)]).cuda()
    w = torch.zeros(T, E, device = "cuda").scatter_(1, idx, 1.0 / TOP_K).to(DT)
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    _, ref_g, _ = _run(ex, x, idx, w, False, monkeypatch)
    _, grads, calls = _run(ex, x, idx, w, True, monkeypatch)
    assert calls["forward_lora"] == 1
    for n, p in ex.named_parameters():
        if "lora_" in n and (n.startswith("gate_up_projs.0.") or n.startswith("down_projs.0.")):
            assert n not in ref_g and n in grads, n
            assert torch.count_nonzero(grads[n]) == 0, n
    routed = [n for n in ref_g if "lora_" in n]
    assert len(routed) == 4 * (E - 1) and all(_rel(grads[n], ref_g[n]) <= 0.05 for n in routed)
