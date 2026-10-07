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
  to bf16 experts and unsupported compute dtypes keep the per-expert loop. fp32 inputs (the
  residual stream after the first MoE layer) take the grouped path in bf16, as Linear4bit does.
* float16 experts (the loader keeps down in fp32) take moe_grouped_fp16's Triton GEMMs and
  match the per-expert loop with Unsloth's forced-float32 LoRA forward, and an fp64 oracle.
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

# The stacked Triton dequant is CUDA-only; elsewhere the grouped path uses bitsandbytes.
STACKED = gq.stacked_dequant_available(torch.device("cuda", torch.cuda.current_device()))
needs_stacked = pytest.mark.skipif(not STACKED, reason = "stacked NF4 dequant kernel is CUDA-only")


def _assert_dequant_path(calls, n):
    if STACKED:
        assert calls["stacked_dequant"] >= n and calls["bnb_fallback_dequant"] == 0
    else:
        assert calls["stacked_dequant"] == 0 and calls["bnb_fallback_dequant"] >= n


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


@needs_stacked
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


@needs_stacked
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
    _assert_dequant_path(calls, 4)   # fwd 2 + bwd recompute 2
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
    _assert_dequant_path(calls, 2)
    assert _rel(out, ref_out) < 2e-2


@needs_grouped_mm
@needs_stacked   # without the kernel both arms are bnb and the comparison is vacuous
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
def test_fallback_sees_replaced_nested_absmax(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_TRITON_KERNELS", "0")
    ex = _Experts(True).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    _run(ex, x, idx, w, True, monkeypatch)
    qs = ex.down_projs[E // 2].weight.quant_state
    qs.state2.absmax = qs.state2.absmax * 1.5
    out, _, delta = _run(ex, x, idx, w, True, monkeypatch)
    ref, _, _ = _run(ex, x, idx, w, False, monkeypatch)
    assert delta["bnb_fallback_dequant"] > 0 and _rel(out, ref) < 2e-2


@needs_grouped_mm
@pytest.mark.parametrize("kernel", ["1", "0"])
@pytest.mark.parametrize("edit", ["absmax", "state2_absmax", "offset", "code"])
def test_in_place_quant_edit_is_not_stale(kernel, edit, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_TRITON_KERNELS", kernel)
    ex = _Experts(True).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    _run(ex, x, idx, w, True, monkeypatch)
    qs = ex.down_projs[E // 2].weight.quant_state
    with torch.no_grad():
        if edit == "absmax":
            qs.absmax.add_(1)   # nested: the uint8 codes index state2.code
        elif edit == "state2_absmax":
            qs.state2.absmax.mul_(1.5)
        elif edit == "offset":
            qs.offset.add_(0.05)
        else:
            qs.code = qs.code.flip(0)
    out, _, delta = _run(ex, x, idx, w, True, monkeypatch)
    ref, _, _ = _run(ex, x, idx, w, False, monkeypatch)
    # A codebook differing from the other experts' must send the layer back to the loop.
    assert delta["forward"] == (0 if edit == "code" else 1) and _rel(out, ref) < 2e-2


@needs_grouped_mm
@pytest.mark.parametrize("forward", ["class", "module"])
def test_checkpoint_control_flow_is_not_swallowed(forward, monkeypatch):
    # The compiled cache emits the class forward standalone, so both entry points must re-raise.
    from torch.utils import checkpoint as ckpt
    monkeypatch.delenv("UNSLOTH_GPTOSS_GROUPED", raising = False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    ex = _Experts(True).train()
    if forward == "class":
        # The class attribute is later replaced; load the class body's forward from source.
        import ast, inspect
        from unsloth_zoo.temporary_patches import gpt_oss
        src = inspect.getsource(gpt_oss)
        cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "GptOssExpertsBnb4bit")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        ns = dict(vars(gpt_oss))
        exec(compile(ast.Module([fn], []), gpt_oss.__file__, "exec"), ns)
        ex.forward = ns["forward"].__get__(ex)
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)

    def stop(*args, **kwargs):
        raise ckpt._StopRecomputationError()

    monkeypatch.setattr(ex, "_forward_grouped_bnb4bit", stop)
    with pytest.raises(ckpt._StopRecomputationError):
        ex(x, idx, w)

    def broken(*args, **kwargs):
        raise RuntimeError("grouped path failed")

    monkeypatch.setattr(ex, "_forward_grouped_bnb4bit", broken)
    assert ex(x, idx, w).shape[-1] == H   # other errors fall back to the per-expert loop


@needs_grouped_mm
def test_grouped_forward_is_run_to_run_deterministic(monkeypatch):
    # An index_add_ combine over repeated tokens is atomic and not reproducible.
    ex = _Experts(True).train()
    T = 4096
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    outs = [_run(ex, x, idx, w, True, monkeypatch) for _ in range(6)]
    assert all(o[2]["forward"] == 1 for o in outs)
    for out, grads, _ in outs[1:]:
        assert torch.equal(out, outs[0][0])
        assert all(torch.equal(grads[k], outs[0][1][k]) for k in grads)


@needs_grouped_mm
def test_mixed_quant_formats_keep_loop(monkeypatch):
    ex = _Experts(True).train()
    fp4 = bnb.nn.Linear4bit(I, H, bias = True, compute_dtype = DT, quant_type = "fp4", quant_storage = torch.uint8)
    fp4.weight = bnb.nn.Params4bit(torch.randn(H, I).to(DT) * 0.02, requires_grad = False, quant_type = "fp4",
                                   compress_statistics = True, blocksize = 64)
    fp4.bias = torch.nn.Parameter(torch.zeros(H, dtype = DT), requires_grad = False)
    ex.down_projs[E // 2] = fp4.cuda()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    _, _, delta = _run(ex, x, idx, w, True, monkeypatch)
    assert delta["forward"] == 0


@needs_grouped_mm
@pytest.mark.parametrize("field", ["compute_dtype", "_pre_set_compute_dtype"])
def test_fp32_middle_expert_keeps_loop(field, monkeypatch):
    ex = _Experts(True).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    setattr(ex.down_projs[E // 2], field, torch.float32)
    _, _, delta = _run(ex, x, idx, w, True, monkeypatch)
    assert delta["forward"] == 0


@needs_grouped_mm
@pytest.mark.parametrize("proj", ["gate_up_projs", "down_projs"])
def test_replaced_middle_expert_rebuilds_tables(proj, monkeypatch):
    ex = _Experts(True).train()
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = DT)
    idx, w = _routing(T)
    _run(ex, x, idx, w, True, monkeypatch)
    shape = (H, 2 * I) if proj == "gate_up_projs" else (I, H)
    getattr(ex, proj)[E // 2] = _linear4bit(*shape, True, 999)
    torch.cuda.empty_cache()
    out, _, delta = _run(ex, x, idx, w, True, monkeypatch)
    ref, _, _ = _run(ex, x, idx, w, False, monkeypatch)
    assert sum(delta.values()) > 0 and _rel(out, ref) < 2e-2


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


# ---------------------------------------------------------------------------------------------
# float16 (T4): gate_up in fp16, down kept in fp32 by the loader (_pre_set_compute_dtype), the
# adapters through Unsloth's forced-float32 LoRA forward. moe_grouped_fp16's Triton GEMMs.
# ---------------------------------------------------------------------------------------------
import types

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg

F16 = torch.float16
FP16_OK = mg.fp16_grouped_available(torch.device("cuda", torch.cuda.current_device()))
needs_fp16_grouped = pytest.mark.skipif(not FP16_OK, reason = "fp16 grouped GEMM unavailable")


def _fp16_experts(nested = True, rule = True, down_scale = None):
    """What unsloth's loader leaves for gpt-oss NF4 on float16: gate_up fp16; down with
    quant_state.dtype / compute_dtype / _pre_set_compute_dtype / bias in fp32 (rule)."""
    ex = _Experts(nested, dtype = F16)
    for p in list(ex.gate_up_projs) + list(ex.down_projs):
        # bitsandbytes < 0.46 adopts an fp32 input's dtype on the first call unless told the
        # compute dtype is final (>= 0.46 sets this whenever compute_dtype is passed).
        p.compute_type_is_set = True
    for p in ex.down_projs:
        if down_scale is not None:
            if nested:
                p.weight.quant_state.state2.absmax.mul_(down_scale)
            else:
                p.weight.quant_state.absmax.mul_(down_scale)
        if rule:
            p.weight.quant_state.dtype = torch.float32
            p.compute_dtype = torch.float32
            p._pre_set_compute_dtype = torch.float32
            p.bias = torch.nn.Parameter(p.bias.float(), requires_grad = False)
    return ex


def _forced_fp32_lora_forward():
    from unsloth_zoo import compiler
    ns = {"torch": torch}
    exec(compiler.COMPILED_LORA_FORWARD_forced_float32, ns)
    return ns["lora_forward"]


def _use_unsloth_fp16_lora_forward(ex):
    # patch_lora_forwards under UNSLOTH_FORCE_FLOAT32: lora_forward(result, ...).to(result.dtype).
    lora_forward = _forced_fp32_lora_forward()

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        name = self.active_adapters[0]
        return lora_forward(result, self.lora_A[name], self.lora_B[name], self.lora_dropout[name],
                            x, self.scaling[name]).to(result.dtype)

    for m in list(ex.gate_up_projs) + list(ex.down_projs):
        m.forward = types.MethodType(forward, m)
    return ex


def _fp16_lora(ex, r = 16, lora_dtype = torch.float32):
    ex = _lora_wrap(ex, r = r)
    for n, p in ex.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(lora_dtype)
    return _use_unsloth_fp16_lora_forward(ex).train()


def _routing32(T, seed = 0):
    idx, w = _routing(T, seed)
    return idx, w.float()


def _oracle_layer(ex, x, idx, w, upstream):
    """fp64 per-expert loop over the bitsandbytes-dequantized weights: output and LoRA grads."""
    out = torch.zeros(x.shape[1], H, dtype = torch.float64, device = "cuda")
    x0 = x[0].detach().double().requires_grad_(True)
    xx = x0
    params = {}
    for n, p in ex.named_parameters():
        if "lora_" in n:
            params[n] = p.detach().double().requires_grad_(True)

    def proj(kind, e, inp):
        m = getattr(ex, kind)[e]
        base = getattr(m, "base_layer", m)
        W = bnb.functional.dequantize_4bit(base.weight.data, base.weight.quant_state).double()
        y = inp @ W.T + base.bias.double()
        if hasattr(m, "lora_A"):
            A = params[f"{kind}.{e}.lora_A.default.weight"]
            B = params[f"{kind}.{e}.lora_B.default.weight"]
            y = y + (inp @ A.T) @ B.T * m.scaling["default"]
        return y

    for e in range(E):
        rows = (idx == e).any(-1).nonzero().flatten()
        if rows.numel() == 0:
            continue
        gu = proj("gate_up_projs", e, xx[rows])
        g, l = gu[:, ::2].clamp(max = ex.limit), gu[:, 1::2].clamp(-ex.limit, ex.limit)
        gated = g * torch.sigmoid(ex.alpha * g) * (l + 1)
        out.index_add_(0, rows, proj("down_projs", e, gated) * w[rows, e, None].double())
    out.backward(upstream.view(-1, H).double())
    grads = {n: p.grad for n, p in params.items() if p.grad is not None}
    grads["<input>"] = x0.grad
    return out.detach(), grads


def _run_up(ex, x, idx, w, grouped, monkeypatch, upstream):
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "1" if grouped else "0")
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    for p in ex.parameters():
        p.grad = None
    xx = x.detach().clone().requires_grad_(True)
    before = dict(gq.CALLS)
    out = ex(xx, idx, w)
    out.backward(upstream.view_as(out).to(out.dtype))
    grads = {n: p.grad.detach().float().clone() for n, p in ex.named_parameters() if p.requires_grad and p.grad is not None}
    grads["<input>"] = xx.grad.detach().float().clone()
    return out.detach().float(), grads, {k: gq.CALLS[k] - before[k] for k in before}


@needs_fp16_grouped
@pytest.mark.parametrize("r", [16, 7])
@pytest.mark.parametrize("x_dtype", [F16, torch.float32])
@pytest.mark.parametrize("rule", [True, False])
@pytest.mark.parametrize("grad_scale", [1.0, 1e-6])
@pytest.mark.parametrize("gemm", ["triton", "cublas"])
@pytest.mark.parametrize("down_operand", ["fp16", "fp32"])
def test_fp16_grouped_matches_loop_and_oracle(r, x_dtype, rule, grad_scale, gemm, down_operand, monkeypatch):
    if down_operand == "fp32" and not rule:
        pytest.skip("fp32 operands apply to the loader's fp32 down only")
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", gemm)
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_DOWN_OPERAND", down_operand)
    ex = _fp16_lora(_fp16_experts(rule = rule), r = r)
    T = 96
    g = torch.Generator(device = "cuda").manual_seed(3)
    x = torch.randn(1, T, H, device = "cuda", generator = g).to(x_dtype)
    upstream = torch.randn(1, T, H, device = "cuda", generator = g) * grad_scale   # 1e-6: no GradScaler
    idx, w = _routing32(T)
    ref_out, ref_g, ref_calls = _run_up(ex, x, idx, w, False, monkeypatch, upstream)
    aa_out, aa_g, _ = _run_up(ex, x, idx, w, False, monkeypatch, upstream)
    out, gr, calls = _run_up(ex, x, idx, w, True, monkeypatch, upstream)
    assert ref_calls["forward_fp16"] == 0
    assert calls["forward_fp16"] == 1 and calls["forward_fp16_lora"] == 1 and calls["forward"] == 0
    assert bool(torch.isfinite(out).all())
    assert set(ref_g) == set(gr) and len([n for n in gr if "lora_" in n]) == 4 * E
    assert _rel(out, ref_out) < 2e-2, _rel(out, ref_out)
    assert all(_rel(aa_g[n], ref_g[n]) == 0 for n in ref_g)   # A/A: the loop is run-to-run exact here
    o_out, o_g = _oracle_layer(ex, x, idx, w, upstream)
    for n in ref_g:
        close = _rel(gr[n], ref_g[n]) <= 0.05
        if grad_scale == 1.0:
            assert close, (n, _rel(gr[n], ref_g[n]))
        else:
            # Unscaled 1e-6 grads sit in fp16's subnormal range on the loop's own fp16 paths
            # (gate_up and its LoRA, down's xA, a fp16 down's dX), so the loop is no reference
            # there: the fp64 oracle is.
            assert close or _rel(gr[n], o_g[n]) <= 1.25 * _rel(ref_g[n], o_g[n]) + 1e-3, (
                n, _rel(gr[n], ref_g[n]), _rel(gr[n], o_g[n]), _rel(ref_g[n], o_g[n]))
    # fp64 oracle: the grouped path is no less accurate than the loop (aggregate over LoRA grads).
    names = [n for n in o_g if n != "<input>"]
    err = lambda gg: sum(float((gg[n].double() - o_g[n]).norm() ** 2) for n in names) ** 0.5 / sum(
        float(o_g[n].norm() ** 2) for n in names) ** 0.5
    assert _rel(out, o_out) <= 1.5 * _rel(ref_out, o_out) + 1e-4, (_rel(out, o_out), _rel(ref_out, o_out))
    assert err(gr) <= 1.5 * err(ref_g) + 1e-4, (err(gr), err(ref_g))


@needs_fp16_grouped
@pytest.mark.parametrize("gemm", ["triton", "cublas"])
def test_fp16_down_output_above_fp16_max_stays_finite(gemm, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", gemm)
    # Under the loader rule down outputs past 65504 are fine in the loop; grouped must match.
    ex = _fp16_lora(_fp16_experts(rule = True, down_scale = 60000.0))
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = F16) * 4
    idx, w = _routing32(T)
    upstream = torch.randn(1, T, H, device = "cuda") * 1e-6
    ref_out, ref_g, _ = _run_up(ex, x, idx, w, False, monkeypatch, upstream)
    out, gr, calls = _run_up(ex, x, idx, w, True, monkeypatch, upstream)
    assert calls["forward_fp16"] == 1
    assert float(ref_out.abs().max()) > 65504, float(ref_out.abs().max())
    assert bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(v).all()) for v in gr.values())
    assert _rel(out, ref_out) < 2e-2
    o_g = None
    for n in ref_g:
        if _rel(gr[n], ref_g[n]) <= 0.05:
            continue
        # Gate / linear values sit on the swiglu clamp edges here, so one fp16 ulp in gate_up
        # flips a clamp mask in either arm (A100 cuBLAS: one lora_B at 6.6%): judge by fp64.
        if o_g is None:
            _, o_g = _oracle_layer(ex, x, idx, w, upstream)
        assert _rel(gr[n], o_g[n]) <= 1.25 * _rel(ref_g[n], o_g[n]) + 1e-3, (
            n, _rel(gr[n], ref_g[n]), _rel(gr[n], o_g[n]), _rel(ref_g[n], o_g[n]))


@needs_fp16_grouped
@pytest.mark.parametrize("gemm", ["triton", "cublas"])
def test_fp16_lora_scaled_before_rounding(gemm, monkeypatch):
    # Unscaled x @ A.T @ B.T past 65504 but finite once scaled: the forced-float32 LoRA forward
    # adds scaling * product in one addmm, so the grouped path must not round the product first.
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", gemm)
    ex = _fp16_lora(_fp16_experts(rule = True))
    g = torch.Generator(device = "cuda").manual_seed(5)
    for m in ex.gate_up_projs:
        m.lora_A["default"].weight.data.copy_(torch.randn(m.lora_A["default"].weight.shape, device = "cuda", generator = g) * 0.5)
        m.lora_B["default"].weight.data.fill_(30000.0)
        m.scaling["default"] = 1e-3
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = F16, generator = g) * 4
    idx, w = _routing32(T)
    upstream = torch.randn(1, T, H, device = "cuda", generator = g) * 1e-6
    xa = x[0].float() @ ex.gate_up_projs[0].lora_A["default"].weight.float().T
    assert float((xa.abs().sum(-1) * 30000.0).max()) > 65504   # the unscaled product overflows fp16
    ref_out, _, _ = _run_up(ex, x, idx, w, False, monkeypatch, upstream)
    out, gr, calls = _run_up(ex, x, idx, w, True, monkeypatch, upstream)
    assert calls["forward_fp16_lora"] == 1
    assert bool(torch.isfinite(ref_out).all())
    assert bool(torch.isfinite(out).all()), "unscaled LoRA product rounded to fp16 before scaling"
    assert _rel(out, ref_out) < 2e-2, _rel(out, ref_out)


@needs_fp16_grouped
@pytest.mark.parametrize("mm_out_dtype", [True, False])
@pytest.mark.parametrize("down_operand", ["fp16", "fp32"])
def test_fp16_cublas_down_stays_fp32_under_autocast(mm_out_dtype, down_operand, monkeypatch):
    # fp16 autocast must not narrow the cuBLAS down GEMM (its fp32 fallback for torch.mm without
    # out_dtype, or fp32 operands) to fp16: down outputs past 65504 stay finite, as in the loop.
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", "cublas")
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_DOWN_OPERAND", down_operand)
    monkeypatch.setitem(mg._MM_OUT_DTYPE, F16, None if mm_out_dtype else False)
    ex = _fp16_lora(_fp16_experts(rule = True, down_scale = 60000.0))
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = F16) * 4
    idx, w = _routing32(T)
    upstream = torch.randn(1, T, H, device = "cuda") * 1e-6
    ref_out, _, _ = _run_up(ex, x, idx, w, False, monkeypatch, upstream)
    with torch.autocast("cuda", dtype = F16):
        out, gr, calls = _run_up(ex, x, idx, w, True, monkeypatch, upstream)
    assert calls["forward_fp16"] == 1
    assert float(ref_out.abs().max()) > 65504
    assert bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(v).all()) for v in gr.values())
    assert _rel(out, ref_out) < 2e-2, _rel(out, ref_out)


@needs_fp16_grouped
def test_fp16_kill_switch(monkeypatch):
    ex = _fp16_lora(_fp16_experts())
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = F16)
    idx, w = _routing32(T)
    up = torch.randn(1, T, H, device = "cuda")
    on, _, c_on = _run_up(ex, x, idx, w, True, monkeypatch, up)
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED_FP16", "0")
    off, _, c_off = _run_up(ex, x, idx, w, True, monkeypatch, up)
    ref, _, _ = _run_up(ex, x, idx, w, False, monkeypatch, up)
    assert c_on["forward_fp16"] == 1 and c_off["forward_fp16"] == 0 and c_off["declined"] >= 1
    assert "UNSLOTH_GPTOSS_GROUPED_FP16" in gq.LAST_DECLINE["reason"]
    assert torch.equal(off, ref)


@needs_fp16_grouped
@pytest.mark.parametrize("case", ["down_bf16", "mixed_down", "gate_up_fp32", "bf16_input"])
def test_fp16_unsupported_dtypes_keep_loop(case, monkeypatch):
    ex = _fp16_lora(_fp16_experts())
    x_dtype = F16
    if case == "down_bf16":
        for p in ex.down_projs:
            p.base_layer.compute_dtype = torch.bfloat16
            p.base_layer._pre_set_compute_dtype = torch.bfloat16
    elif case == "mixed_down":
        ex.down_projs[E // 2].base_layer.compute_dtype = F16
    elif case == "gate_up_fp32":
        ex.gate_up_projs[0].base_layer.compute_dtype = torch.float32
    else:
        x_dtype = torch.bfloat16
    T = 32
    x = torch.randn(1, T, H, device = "cuda", dtype = x_dtype)
    idx, w = _routing32(T)
    _, _, calls = _run_up(ex, x, idx, w, True, monkeypatch, torch.randn(1, T, H, device = "cuda"))
    assert calls["forward_fp16"] == 0 and calls["forward"] == 0, case


@needs_fp16_grouped
@pytest.mark.parametrize("gemm", ["triton", "cublas"])
def test_fp16_lora_free_and_expert_windows(gemm, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", gemm)
    ex = _fp16_experts().train()
    T = 80
    x = torch.randn(1, T, H, device = "cuda", dtype = F16)
    idx, w = _routing32(T)
    up = torch.randn(1, T, H, device = "cuda")
    ref, ref_g, _ = _run_up(ex, x, idx, w, False, monkeypatch, up)
    full, g_full, c = _run_up(ex, x, idx, w, True, monkeypatch, up)
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_EXPERT_WINDOW", "3")
    win, g_win, _ = _run_up(ex, x, idx, w, True, monkeypatch, up)
    assert c["forward_fp16"] == 1 and c["forward_fp16_lora"] == 0
    _assert_dequant_path(c, 4)
    assert _rel(full, ref) < 2e-2 and _rel(g_full["<input>"], ref_g["<input>"]) < 0.05
    # Windows change only which experts each launch covers: bit-identical.
    assert torch.equal(full, win) and torch.equal(g_full["<input>"], g_win["<input>"])


@needs_fp16_grouped
def test_fp16_stack_is_bnb_rounded_once(monkeypatch):
    # fp32 quant state (loader rule) -> fp16 stack: exactly bitsandbytes' fp32 dequant rounded to fp16.
    ex = _fp16_experts()
    state = gq._tables(ex, F16)
    assert state is not None and state["down"]["dtype"] is torch.float32
    p = gq._Fp16StackProvider(state["down"], ex.down_projs, True)
    for lo, hi in ((0, E), (2, 5)):
        st = p(lo, hi)
        for e in range(lo, hi):
            b = ex.down_projs[e]
            ref = bnb.functional.dequantize_4bit(b.weight.data, b.weight.quant_state)
            assert ref.dtype == torch.float32 and torch.equal(st[e - lo], ref.half())


@needs_fp16_grouped
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in (9, 10),
    reason = "only measured where the bf16 grouped path is also sync-free",
)
def test_fp16_grouped_no_host_sync(monkeypatch):
    ex = _fp16_lora(_fp16_experts())
    T = 64
    x = torch.randn(1, T, H, device = "cuda", dtype = F16, requires_grad = True)
    idx, w = _routing32(T)
    monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "1")
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", "triton")
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_EXPERT_WINDOW", str(E))   # mem_get_info is not a stream sync, but pin it
    _run_up(ex, x, idx, w, True, monkeypatch, torch.randn(1, T, H, device = "cuda"))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = ex(x, idx, w)
        out.float().sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@needs_fp16_grouped
@pytest.mark.parametrize("gemm", ["triton", "cublas"])
def test_fp32_down_operands_match_the_loop_down_exactly_enough(gemm, monkeypatch):
    # fp32 operand mode = the loop's fp32 down math: fp32 dequant stack (bit-exact bnb), fp32 x,
    # fp32 dY, IEEE accumulate. Against per-expert fp32 matmuls only summation order differs.
    from unsloth_zoo.temporary_patches.moe_grouped_fp16 import Groups, grouped_frozen_linear
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", gemm)
    ex = _fp16_experts()
    state = gq._tables(ex, F16)
    p32 = gq._Fp16StackProvider(state["down"], ex.down_projs, True, torch.float32)
    T = 300
    counts = torch.tensor([T // E] * (E - 1) + [T - (T // E) * (E - 1)], dtype = torch.int32, device = "cuda")
    counts[2] = 0
    counts[-1] += T // E
    x = (torch.randn(T, I, device = "cuda") * 10).requires_grad_(True)
    bias = torch.stack([p.bias for p in ex.down_projs]).detach()
    y = grouped_frozen_linear(x, Groups(counts), p32, bias = bias, out_dtype = torch.float32)
    g = torch.randn_like(y) * 1e-7
    y.backward(g)
    ref, dref, s0 = torch.empty_like(y), torch.empty_like(x), 0
    for e, c in enumerate(counts.tolist()):
        W = bnb.functional.dequantize_4bit(ex.down_projs[e].weight.data, ex.down_projs[e].weight.quant_state)
        assert W.dtype == torch.float32
        ref[s0:s0 + c] = x.detach()[s0:s0 + c] @ W.T + bias[e]
        dref[s0:s0 + c] = g[s0:s0 + c] @ W
        s0 += c
    assert _rel(y.detach(), ref) < 1e-6 and _rel(x.grad, dref) < 1e-6, (_rel(y.detach(), ref), _rel(x.grad, dref))
