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

"""MoE backward without full-size copies, checked against an fp32 per-expert reference.

1. _GroupedMMRecompute.backward hands the transposed frozen stack to the grouped GEMM as a view.
2. Trainable expert biases: the gather's backward sums in fp32 (one-hot GEMM) instead of a 16-bit index_add_.
3. A transposed trainable stack gets dW in its own layout, so AccumulateGrad does not copy it.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as MG
from unsloth_zoo.temporary_patches import moe_utils as mu

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA or not mu._check_torch_grouped_mm_supported(), reason = "needs CUDA grouped_mm")
needs_kernel = pytest.mark.skipif(
    not CUDA or MG.triton is None or torch.version.hip is not None or mu._GROUPED_MM_TRITON_OP is None,
    reason = "needs CUDA, Triton and torch.library.custom_op",
)
DT = torch.bfloat16
ALPHA, LIMIT = 1.702, 7.0
_ENV = ("UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS",
        "UNSLOTH_MOE_RECOMPUTE", "UNSLOTH_MOE_SELECTIVE_DEQUANT")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    mu._TRITON_GROUPED_MM_POLICY.clear()
    yield
    mu._TRITON_GROUPED_MM_POLICY.clear()


def _triton(monkeypatch, on):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1" if on else "0")
    mu._TRITON_GROUPED_MM_POLICY.clear()


def _backward_copies(fn, shape):
    """aten::copy_ calls of a full `shape` tensor during fn(): AccumulateGrad's layout copy, eager or compiled."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages(group_by_input_shape = True)
               if e.key == "aten::copy_" and e.input_shapes and list(e.input_shapes[0]) == list(shape))


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()


class GptOssExperts(nn.Module):
    """gpt-oss layout: gate_up_proj [E, H, 2I] interleaved + bias, down_proj [E, I, H] + bias."""

    def __init__(self, E, H, I, gen):
        super().__init__()
        self.num_experts, self.hidden_size, self.expert_dim = E, H, I
        self.alpha, self.limit = ALPHA, LIMIT
        rnd = lambda *s, scale: nn.Parameter((torch.randn(*s, generator = gen) * scale).to("cuda", DT))
        self.gate_up_proj = rnd(E, H, 2 * I, scale = H ** -0.5)
        self.gate_up_proj_bias = rnd(E, 2 * I, scale = 0.02)
        self.down_proj = rnd(E, I, H, scale = I ** -0.5)
        self.down_proj_bias = rnd(E, H, scale = 0.02)
        self._unsloth_grouped_mm_format = True


class Qwen3Experts(nn.Module):
    """Qwen3-MoE layout: gate_up_proj [E, 2I, H], down_proj [E, H, I], no bias."""

    def __init__(self, E, H, I, gen):
        super().__init__()
        self.num_experts = E
        rnd = lambda *s, scale: nn.Parameter((torch.randn(*s, generator = gen) * scale).to("cuda", DT))
        self.gate_up_proj = rnd(E, 2 * I, H, scale = H ** -0.5)
        self.down_proj = rnd(E, H, I, scale = I ** -0.5)
        self.act_fn = nn.SiLU()


def _reference(gptoss, num_experts, p, x, topi, w, lo):
    """fp32 per-expert loop (HF eager semantics) on float32 leaves."""
    out = torch.zeros(x.shape, dtype = torch.float32, device = x.device)
    xf = x.float()
    for e in range(num_experts):
        tok, pos = torch.where(topi == e)
        if tok.numel() == 0:
            continue
        h = xf[tok]
        gu = p["gate_up_proj"][e] if gptoss else p["gate_up_proj"][e].t()
        mm1 = h @ gu
        if "gate_up_proj" in lo:
            a, b, s = lo["gate_up_proj"]
            mm1 = mm1 + s * (h @ a[e]) @ b[e]
        if gptoss:
            mm1 = mm1 + p["gate_up_proj_bias"][e]
            gate, up = mm1[..., ::2].clamp(max = LIMIT), mm1[..., 1::2].clamp(min = -LIMIT, max = LIMIT)
            act = (up + 1) * (gate * torch.sigmoid(gate * ALPHA))
        else:
            gate, up = mm1.chunk(2, dim = -1)
            act = F.silu(gate) * up
        dn = p["down_proj"][e] if gptoss else p["down_proj"][e].t()
        y = act @ dn
        if "down_proj" in lo:
            a, b, s = lo["down_proj"]
            y = y + s * (act @ a[e]) @ b[e]
        if gptoss:
            y = y + p["down_proj_bias"][e]
        out.index_add_(0, tok, y * w.float()[tok, pos, None])
    return out


def _case(kind, T, E, H, I, k, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    m = (GptOssExperts if kind == "gptoss" else Qwen3Experts)(E, H, I, gen)
    x = (torch.randn(T, H, generator = gen) * 0.5).to("cuda", DT)
    logits = torch.randn(T, E, generator = gen).to("cuda")
    topv, topi = torch.topk(logits, k, dim = -1)
    w = torch.softmax(topv, dim = -1).to(DT)
    dy = torch.randn(T, H, generator = gen).to("cuda", DT)
    return m, x, topi, w, dy


def _lora(m, rank = 8, seed = 1):
    gen = torch.Generator().manual_seed(seed)
    E = m.num_experts
    gptoss = isinstance(m, GptOssExperts)
    H = m.gate_up_proj.shape[1] if gptoss else m.gate_up_proj.shape[2]
    I2 = m.gate_up_proj.shape[2] if gptoss else m.gate_up_proj.shape[1]
    I = I2 // 2
    mk = lambda *s: (torch.randn(*s, generator = gen) * 0.05).to("cuda", DT).requires_grad_(True)
    return {"gate_up_proj": (mk(E, H, rank), mk(E, rank, I2), 2.0), "down_proj": (mk(E, I, rank), mk(E, rank, H), 2.0)}


def _run(m, x, topi, w, dy, mode, lora, monkeypatch, forward = None):
    """mode: frozen (dX only) / full (dX + every parameter). Returns out, dX, {name: grad}, {name: incoming grad layout}."""
    for prm in m.parameters():
        prm.requires_grad_(mode == "full")
        prm.grad = None
    layouts = {}
    handles = []
    for n, prm in m.named_parameters():
        if prm.requires_grad:
            handles.append(prm.register_hook(lambda g, n = n, prm = prm: layouts.__setitem__(n, g.stride() == prm.stride())))
    monkeypatch.setattr(mu, "take_moe_lora_stash", lambda module, name: lora.get(name))
    xg = x.detach().clone().requires_grad_(True)
    out = (forward or mu.forward_native_grouped_mm)(m, xg, topi, w)
    out.backward(dy)
    for h in handles:
        h.remove()
    grads = {n: prm.grad for n, prm in m.named_parameters() if prm.grad is not None}
    for n, (a, b, _) in lora.items():
        grads[n + ".A"], grads[n + ".B"] = a.grad, b.grad
    return out.detach(), xg.grad, grads, layouts


def _check_parity(m, x, topi, w, dy, out, dx, grads, lora, tol):
    xr = x.detach().float().requires_grad_(True)
    leaves = [n for n, _ in m.named_parameters() if n in grads]
    ref_params = {n: prm.detach().float().requires_grad_(n in grads) for n, prm in m.named_parameters()}
    ref_lora = {n: (a.detach().float().requires_grad_(True), b.detach().float().requires_grad_(True), s) for n, (a, b, s) in lora.items()}
    ref = _reference(isinstance(m, GptOssExperts), m.num_experts, ref_params, xr, topi, w, ref_lora)
    ref.backward(dy.float())
    report = {"out": _rel(out, ref.detach()), "dx": _rel(dx, xr.grad)}
    for n in leaves:
        report[n] = _rel(grads[n], ref_params[n].grad)
    for n, (a, b, _) in ref_lora.items():
        report[n + ".A"], report[n + ".B"] = _rel(grads[n + ".A"], a.grad), _rel(grads[n + ".B"], b.grad)
    bad = {k: v for k, v in report.items() if not v < tol.get(k, tol["default"])}
    assert not bad, report
    return report


_SHAPES = {"gptoss": dict(T = 1024, E = 4, H = 256, I = 256, k = 2), "qwen3": dict(T = 512, E = 16, H = 256, I = 128, k = 4)}
# bf16 everywhere; bias grads are a sum over ~512 rows per expert, which the old 16-bit index_add_ rounded per add.
_TOL = {"default": 1.2e-2, "gate_up_proj_bias": 6e-3, "down_proj_bias": 6e-3}


@needs_cuda
@pytest.mark.parametrize("kind", ["gptoss", "qwen3"])
@pytest.mark.parametrize("mode", ["frozen", "full"])
@pytest.mark.parametrize("triton", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_grads_match_fp32_reference(kind, mode, triton, lora, monkeypatch):
    if triton and (MG.triton is None or mu._GROUPED_MM_TRITON_OP is None or torch.version.hip is not None):
        pytest.skip("no Triton grouped GEMM")
    if lora and mode == "full":
        pytest.skip("LoRA trains adapters on a frozen base")
    _triton(monkeypatch, triton)
    m, x, topi, w, dy = _case(kind, **_SHAPES[kind])
    lo = _lora(m) if lora else {}
    out, dx, grads, layouts = _run(m, x, topi, w, dy, mode, lo, monkeypatch)
    _check_parity(m, x, topi, w, dy, out, dx, grads, lo, _TOL)
    if mode == "full":
        assert set(layouts) == {n for n, _ in m.named_parameters()}
        if triton:
            assert all(layouts.values()), layouts
        for n in ("gate_up_proj_bias", "down_proj_bias"):
            if n in grads:
                assert layouts[n] and grads[n].is_contiguous(), n


@needs_cuda
@pytest.mark.parametrize("kind", ["gptoss", "qwen3"])
def test_recompute_backward_passes_the_frozen_stack_as_a_view(kind, monkeypatch):
    """P1: no fresh [E, N, K] copy of the frozen stack is handed to the backward grouped GEMM."""
    monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", "1")
    _triton(monkeypatch, False)
    m, x, topi, w, dy = _case(kind, **_SHAPES[kind])
    storages = {prm.untyped_storage().data_ptr() for prm in m.parameters()}
    seen = []
    real = mu._grouped_mm_with_backward_fix

    def spy(inputs, weight, offsets):
        if not torch.is_grad_enabled() and weight.dim() == 3:
            seen.append(weight.untyped_storage().data_ptr() in storages)
        return real(inputs, weight, offsets)
    monkeypatch.setattr(mu, "_grouped_mm_with_backward_fix", spy)
    out, dx, grads, _ = _run(m, x, topi, w, dy, "frozen", {}, monkeypatch)
    monkeypatch.setattr(mu, "_grouped_mm_with_backward_fix", real)
    assert len(seen) == 4, seen
    assert all(seen), seen
    _check_parity(m, x, topi, w, dy, out, dx, grads, {}, _TOL)


@needs_cuda
def test_recompute_backward_view_matches_contiguous_bitwise():
    """The view goes through the same probe-gated grouped GEMM: same dX as the old contiguous copy."""
    gen = torch.Generator().manual_seed(0)
    E, K, N = 4, 128, 192
    counts = torch.tensor([100, 0, 37, 91])
    W = (torch.randn(E, K, N, generator = gen) * 0.1).to("cuda", DT)
    x = torch.randn(int(counts.sum()), K, generator = gen).to("cuda", DT).requires_grad_(True)
    offs = counts.cumsum(0).to("cuda", torch.int32)
    dy = torch.randn(x.shape[0], N, generator = gen).to("cuda", DT)
    mu._GroupedMMRecompute.apply(x, offs, lambda: W).backward(dy)
    with torch.no_grad():
        want = mu._grouped_mm_with_backward_fix(dy, W.transpose(-2, -1).contiguous(), offs)
    assert torch.equal(x.grad, want)


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("compiled", [False, True])
def test_triton_wgrad_in_parameter_layout(dtype, compiled, monkeypatch):
    """P3 at the op level: eager autograd.Function and the custom op under torch.compile(fullgraph = True)."""
    _triton(monkeypatch, True)
    torch._dynamo.reset()
    gen = torch.Generator().manual_seed(0)
    E, K, N = 4, 96, 160
    counts = torch.tensor([60, 0, 45, 23])
    param = (torch.randn(E, N, K, generator = gen) * 0.1).to("cuda", dtype).requires_grad_(True)
    x = torch.randn(int(counts.sum()), K, generator = gen).to("cuda", dtype).requires_grad_(True)
    offs = counts.cumsum(0).to("cuda", torch.int32)
    dy = torch.randn(x.shape[0], N, generator = gen).to("cuda", dtype)

    def f(x, p, offs):
        return mu._grouped_mm_with_backward_fix(x, p.transpose(-2, -1), offs)
    if compiled:
        explain = torch._dynamo.explain(f)(x, param, offs)
        assert explain.graph_break_count == 0, explain.break_reasons
        torch._dynamo.reset()
        f = torch.compile(f, fullgraph = True)
    f(x, param, offs).backward(dy)   # compile / autotune outside the profiled step
    param.grad = x.grad = None
    before = dict(MG.GENERIC_CALLS)
    copies = _backward_copies(lambda: f(x, param, offs).backward(dy), param.shape)
    assert MG.GENERIC_CALLS["gemm"] - before["gemm"] == 2 and MG.GENERIC_CALLS["wgrad"] - before["wgrad"] == 1
    assert copies == 0, copies
    assert param.grad.stride() == param.stride()
    e_of = torch.repeat_interleave(torch.arange(E), counts).cuda()
    w64 = param.detach().double().transpose(-2, -1)
    dw = torch.zeros_like(w64)
    for e in range(E):
        rows = e_of == e
        dw[e] = x.detach().double()[rows].T @ dy.double()[rows]
    assert _rel(param.grad, dw.transpose(-2, -1)) < 1e-2
    assert _rel(x.grad, torch.einsum("mn,mkn->mk", dy.double(), w64[e_of])) < 1e-2


@needs_cuda
def test_fp16_compiled_op_wgrad_in_parameter_layout(monkeypatch):
    """The float16 compiled op (no Triton) returns dW in the parameter's layout too."""
    if mu._GROUPED_MM_FP16_OP is None:
        pytest.skip("no fp16 custom op")
    _triton(monkeypatch, False)
    torch._dynamo.reset()
    gen = torch.Generator().manual_seed(0)
    E, K, N = 4, 64, 96
    counts = torch.tensor([40, 8, 0, 24])
    param = (torch.randn(E, N, K, generator = gen) * 0.1).to("cuda", torch.float16).requires_grad_(True)
    x = torch.randn(int(counts.sum()), K, generator = gen).to("cuda", torch.float16).requires_grad_(True)
    offs = counts.cumsum(0).to("cuda", torch.int32)
    try:
        mu._grouped_mm_eager(x.detach(), param.detach().transpose(-2, -1), offs)
    except RuntimeError as exc:
        pytest.skip(f"no float16 grouped_mm here: {exc}")
    f = torch.compile(lambda x, p, o: mu._grouped_mm_with_backward_fix(x, p.transpose(-2, -1), o), fullgraph = True)
    dy = torch.randn(x.shape[0], N, generator = gen).to("cuda", torch.float16)
    f(x, param, offs).backward(dy)
    param.grad = x.grad = None
    copies = _backward_copies(lambda: f(x, param, offs).backward(dy), param.shape)
    assert copies == 0, copies
    assert param.grad.stride() == param.stride()
    e_of = torch.repeat_interleave(torch.arange(E), counts).cuda()
    dw = torch.zeros(E, K, N, dtype = torch.float64, device = "cuda")
    for e in range(E):
        rows = e_of == e
        dw[e] = x.detach().double()[rows].T @ dy.double()[rows]
    assert _rel(param.grad, dw.transpose(-2, -1)) < 1e-2


@needs_cuda
def test_expert_bias_gather_grad_and_compile():
    """P2: forward identical to index_select, bias grad = fp32 per-expert row sum, no graph break."""
    gen = torch.Generator().manual_seed(0)
    E, D, M = 8, 384, 8192
    ids = torch.randint(0, E, (M,), generator = gen).sort().values.cuda()
    bias = (torch.randn(E, D, generator = gen) * 0.02).to("cuda", DT).requires_grad_(True)
    g = torch.randn(M, D, generator = gen).to("cuda", DT)
    out = mu._gather_expert_bias(bias, ids)
    assert torch.equal(out, bias.detach().index_select(0, ids))
    out.backward(g)
    want = torch.zeros(E, D, dtype = torch.float64, device = "cuda").index_add_(0, ids, g.double())
    assert bias.grad.dtype == DT and bias.grad.is_contiguous()
    # bf16 output rounding only (~2e-3); the 16-bit index_add_ lands near 1.5e-2 here
    assert _rel(bias.grad, want) < 4e-3, _rel(bias.grad, want)
    with torch.no_grad():
        assert torch.equal(mu._gather_expert_bias(bias, ids), out)
    frozen = bias.detach()
    assert mu._gather_expert_bias(frozen, ids).grad_fn is None

    def f(bias, ids, base):
        return base + mu._gather_expert_bias(bias, ids).to(base.dtype)
    base = torch.randn(M, D, generator = gen).to("cuda", DT)
    explain = torch._dynamo.explain(f)(bias, ids, base)
    assert explain.graph_break_count == 0, explain.break_reasons
    torch._dynamo.reset()
    bias.grad = None
    torch.compile(f, fullgraph = True)(bias, ids, base).backward(g)
    assert _rel(bias.grad, want) < 4e-3


@needs_cuda
def test_expert_bias_gather_chunks_bound_one_hot(monkeypatch):
    """Long contexts: the one-hot stays at most _EXPERT_BIAS_ONE_HOT_ELEMENTS, ragged last chunk included."""
    gen = torch.Generator().manual_seed(0)
    E, D, M = 128, 64, 65536 + 300
    monkeypatch.setattr(mu, "_EXPERT_BIAS_ONE_HOT_ELEMENTS", 1 << 20)
    ids = torch.randint(0, E, (M,), generator = gen).sort().values.cuda()
    bias = (torch.randn(E, D, generator = gen) * 0.02).to("cuda", DT).requires_grad_(True)
    g = torch.randn(M, D, generator = gen).to("cuda", DT)
    mu._gather_expert_bias(bias, ids).backward(g)  # cuBLAS workspace
    bias.grad = None
    out = mu._gather_expert_bias(bias, ids)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out.backward(g)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    # unchunked: 25 MB (bool + bf16 one-hot); chunked: 5 MB
    assert extra < 8 * 2**20, extra
    want = torch.zeros(E, D, dtype = torch.float64, device = "cuda").index_add_(0, ids, g.double())
    assert bias.grad.dtype == DT
    assert _rel(bias.grad, want) < 4e-3, _rel(bias.grad, want)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks = pytest.mark.skipif(not CUDA, reason = "needs CUDA"))])
def test_expert_bias_gather_grad_stays_fp32_under_fp16_autocast(device):
    """fp32 biases under fp16 autocast: the per-expert sum (131072 here) must not round to fp16 and overflow."""
    E, D, M = 4, 8, 128
    ids = torch.zeros(M, dtype = torch.long, device = device)
    bias = torch.zeros(E, D, device = device, requires_grad = True)
    g = torch.full((M, D), 1024.0, device = device)
    with torch.autocast(device_type = device, dtype = torch.float16):
        out = mu._gather_expert_bias(bias, ids)
        out.backward(g)
    assert bias.grad.dtype == torch.float32
    assert torch.isfinite(bias.grad).all(), bias.grad
    assert torch.equal(bias.grad[0], torch.full((D,), float(M * 1024), device = device))
    assert torch.equal(bias.grad[1:], torch.zeros(E - 1, D, device = device))

    if device == "cuda":
        # Compiled backward replays the forward's autocast state ("same_as_forward" on torch 2.11+).
        torch._dynamo.reset()
        bias.grad = None
        f = torch.compile(lambda b, i: mu._gather_expert_bias(b, i).float(), fullgraph = True)
        with torch.autocast(device_type = device, dtype = torch.float16):
            out = f(bias, ids)
        out.backward(g)
        assert torch.isfinite(bias.grad).all(), bias.grad
        assert torch.equal(bias.grad[0], torch.full((D,), float(M * 1024), device = device))


_BIAS_DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


@pytest.mark.parametrize("fp32_out", [True, False], ids = ["mm_out_dtype", "upcast"])
@pytest.mark.parametrize("device", _BIAS_DEVICES)
def test_expert_bias_gather_chunks_sum_in_fp32_before_rounding(device, fp32_out, monkeypatch):
    """fp16, two chunks whose 16-bit sums overflow (+131072, -131072) but whose total is 0: no inf - inf = NaN."""
    if not fp32_out:
        monkeypatch.setattr(mu, "_MM_OUT_DTYPE", False, raising = False)
    E, D = 128, 2
    chunk = mu._EXPERT_BIAS_ONE_HOT_ELEMENTS // E
    ids = torch.zeros(2 * chunk, dtype = torch.long, device = device)
    ids[-1] = 1
    g = torch.ones(2 * chunk, D, dtype = torch.float16, device = device)
    g[chunk:] = -1
    g[-1] = 3
    bias = torch.zeros(E, D, dtype = torch.float16, device = device, requires_grad = True)
    mu._gather_expert_bias(bias, ids).backward(g)
    want = torch.zeros(E, D, dtype = torch.float64, device = device).index_add_(0, ids, g.double())
    assert torch.isfinite(bias.grad).all(), bias.grad[:2]
    assert torch.equal(bias.grad.double(), want), (bias.grad[:2], want[:2])


@pytest.mark.parametrize("fp32_out", [True, False], ids = ["mm_out_dtype", "upcast"])
@pytest.mark.parametrize("device", _BIAS_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_expert_bias_gather_chunked_matches_fp64(device, dtype, fp32_out, monkeypatch):
    """Ragged chunks, random data: fp32 sums, one output rounding off the fp64 per-expert sum."""
    if not fp32_out:
        monkeypatch.setattr(mu, "_MM_OUT_DTYPE", False, raising = False)
    gen = torch.Generator().manual_seed(0)
    E, D, M = 16, 48, 4096 + 77
    monkeypatch.setattr(mu, "_EXPERT_BIAS_ONE_HOT_ELEMENTS", 1 << 14)
    ids = torch.randint(0, E, (M,), generator = gen).sort().values.to(device)
    g = torch.randn(M, D, generator = gen).to(device, dtype)
    bias = torch.zeros(E, D, dtype = dtype, device = device, requires_grad = True)
    mu._gather_expert_bias(bias, ids).backward(g)
    want = torch.zeros(E, D, dtype = torch.float64, device = device).index_add_(0, ids, g.double())
    assert bias.grad.dtype == dtype
    assert _rel(bias.grad, want) < (1e-3 if dtype == torch.float16 else 5e-3), _rel(bias.grad, want)


def test_expert_bias_gather_compiles_with_a_dynamic_row_count(monkeypatch):
    """mark_dynamic on the routed rows: the chunked backward must not specialize them (fullgraph, dynamic)."""
    E, D = 8, 4
    monkeypatch.setattr(mu, "_EXPERT_BIAS_ONE_HOT_ELEMENTS", 1 << 10)
    chunk = (1 << 10) // E
    gen = torch.Generator().manual_seed(0)

    def f(bias, ids, base):
        return base + mu._gather_expert_bias(bias, ids)

    torch._dynamo.reset()
    compiled = torch.compile(f, backend = "aot_eager", fullgraph = True, dynamic = True)
    for M in (chunk + 1, 3 * chunk + 5):
        ids = torch.randint(0, E, (M,), generator = gen).sort().values
        torch._dynamo.mark_dynamic(ids, 0)
        base = torch.zeros(M, D)
        torch._dynamo.mark_dynamic(base, 0)
        bias = torch.zeros(E, D, requires_grad = True)
        g = torch.randn(M, D, generator = gen)
        compiled(bias, ids, base).backward(g)
        want = torch.zeros(E, D, dtype = torch.float64).index_add_(0, ids, g.double())
        torch.testing.assert_close(bias.grad.double(), want, rtol = 1e-5, atol = 1e-5)


@pytest.mark.skipif(not CUDA, reason = "CUDA determinism")
@pytest.mark.parametrize("path", ["eager", "eager_upcast", "compiled"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_expert_bias_gather_long_backward_is_reproducible(path, dtype, monkeypatch):
    """Rows past one chunk, 10 backwards: bitwise-identical bias grads (GEMMs only, no index_add_ atomics)."""
    gen = torch.Generator().manual_seed(0)
    E, D, M = 16, 96, 3 * 1024 + 77
    monkeypatch.setattr(mu, "_EXPERT_BIAS_ONE_HOT_ELEMENTS", E * 1024)
    if path == "eager_upcast":
        monkeypatch.setattr(mu, "_MM_OUT_DTYPE", False, raising = False)
    # Long sorted runs per expert: where atomic adds would reorder.
    ids = torch.randint(0, E, (M,), generator = gen).sort().values.cuda()
    g = torch.randn(M, D, generator = gen).to("cuda", dtype)
    base = torch.zeros(M, D, dtype = dtype, device = "cuda")

    def f(bias, ids, base):
        return base + mu._gather_expert_bias(bias, ids)

    torch._dynamo.reset()
    fn = torch.compile(f, fullgraph = True) if path == "compiled" else f
    want = torch.zeros(E, D, dtype = torch.float64, device = "cuda").index_add_(0, ids, g.double())
    grads = []
    try:
        for _ in range(10):
            bias = torch.zeros(E, D, dtype = dtype, device = "cuda", requires_grad = True)
            fn(bias, ids, base).backward(g)
            grads.append(bias.grad)
    finally:
        torch._dynamo.reset()
    assert all(torch.equal(grads[0], x) for x in grads[1:])
    assert _rel(grads[0], want) < (1e-3 if dtype == torch.float16 else 5e-3), _rel(grads[0], want)


@pytest.mark.parametrize("dynamic", [False, True], ids = ["static", "dynamic"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_expert_bias_gather_compiled_upcast_caps_the_fp32_copy(dtype, dynamic, monkeypatch):
    """Compiled backward on the upcast fallback (CPU, ROCm, XPU, CUDA torch < 2.9): each fp32 copy of grad stays
    within one chunk instead of all rows x width, and the bias grad is bitwise the eager chunked one."""
    if not hasattr(torch.library, "custom_op"):
        pytest.skip("torch.library.custom_op unavailable")
    monkeypatch.setattr(mu, "_MM_OUT_DTYPE", False, raising = False)
    E, D = 16, 96
    monkeypatch.setattr(mu, "_EXPERT_BIAS_ONE_HOT_ELEMENTS", 1 << 12)
    cap = (1 << 12) // max(E, D)
    upcast_rows = []
    real_one_hot_mm = mu._one_hot_mm_fp32

    def recording_one_hot_mm(one_hot, grad):
        # Only real (eager) upcasts count: an in-graph upcast is the uncapped rows x width copy.
        if grad.dtype != torch.float32 and not torch.compiler.is_compiling():
            upcast_rows.append(grad.shape[0])
        return real_one_hot_mm(one_hot, grad)

    monkeypatch.setattr(mu, "_one_hot_mm_fp32", recording_one_hot_mm)

    def f(bias, ids, base):
        return base + mu._gather_expert_bias(bias, ids)

    def run(fn, M, seed):
        gen = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, E, (M,), generator = gen).sort().values
        g = (torch.randn(M, D, generator = gen) * 64).to(dtype)
        base = torch.zeros(M, D, dtype = dtype)
        bias = torch.zeros(E, D, dtype = dtype, requires_grad = True)
        fn(bias, ids, base).backward(g)
        return bias.grad

    torch._dynamo.reset()
    compiled = torch.compile(f, fullgraph = True, dynamic = dynamic)
    try:
        for M, seed in ((40 * cap + 7, 0), (23 * cap + 3, 1)):
            upcast_rows.clear()
            got = run(compiled, M, seed)
            assert upcast_rows and max(upcast_rows) <= cap, (cap, upcast_rows[:4])
            assert torch.equal(got, run(f, M, seed))
    finally:
        torch._dynamo.reset()
