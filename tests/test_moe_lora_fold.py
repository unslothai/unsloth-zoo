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

"""Routed-expert LoRA folded into the base grouped GEMM (moe_lora_fold): forward, dX, dA and dB against an fp64
oracle and against the separated path, every backend, padded and unpadded ranks, the kill switch."""

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as MG
from unsloth_zoo.temporary_patches import moe_lora_fold as LF
from unsloth_zoo.temporary_patches import moe_utils as MU

CUDA = torch.cuda.is_available()
needs_kernel = pytest.mark.skipif(
    not CUDA or MG.triton is None or torch.version.hip is not None or MU._GROUPED_MM_TRITON_OP is None,
    reason = "needs CUDA, Triton and torch.library.custom_op",
)
_ENV = ("UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS",
        "UNSLOTH_MOE_LORA_FOLD", "UNSLOTH_MOE_LORA_PAD_ALWAYS")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    MU._TRITON_GROUPED_MM_POLICY.clear()
    for k in LF.LORA_FOLD_STATS:
        LF.LORA_FOLD_STATS[k] = 0
    yield
    MU._TRITON_GROUPED_MM_POLICY.clear()


def _force_triton(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    MU._TRITON_GROUPED_MM_POLICY.clear()
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("base-sized GEMMs never take the Triton kernel below sm80")
    if not MG.triton_grouped_available(torch.device("cuda")):
        pytest.skip("Triton grouped GEMM unavailable")


class Experts(nn.Module):
    """Qwen3-MoE-style fused experts: gate_up_proj [E, 2I, H], down_proj [E, H, I] (out, in), SiLU."""

    def __init__(self, E, H, I, dtype, device, seed = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_experts = E
        self.gate_up_proj = nn.Parameter((torch.randn(E, 2 * I, H, generator = g) * H ** -0.5).to(device, dtype),
                                         requires_grad = False)
        self.down_proj = nn.Parameter((torch.randn(E, H, I, generator = g) * I ** -0.5).to(device, dtype),
                                      requires_grad = False)
        self.act_fn = nn.SiLU()


def _lora(E, din, dout, r, device, seed):
    g = torch.Generator().manual_seed(seed)
    first = (torch.randn(E, din, r, generator = g) * din ** -0.5).to(device).requires_grad_(True)
    second = (torch.randn(E, r, dout, generator = g) * 0.5).to(device).requires_grad_(True)
    return first, second


def _routing(T, E, k, device, seed, skew = True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator = g)
    if skew:
        logits[:, 0] += 2.0          # a heavy expert
        logits[:, E - 1] -= 50.0     # an empty expert
    w, idx = torch.topk(torch.softmax(logits, -1), k, -1)
    w = w / w.sum(-1, keepdim = True)
    return idx.to(device), w.to(device)


def _reference(mod, x, idx, w, gu, dn, s):
    """fp64 per-token reference of the same MoE (SiLU-gated, gate first)."""
    T, H = x.shape
    out = torch.zeros(T, H, dtype = torch.float64, device = x.device)
    W1 = mod.gate_up_proj.double()
    W2 = mod.down_proj.double()
    for slot in range(idx.shape[1]):
        e = idx[:, slot]
        h = torch.einsum("th,toh->to", x, W1[e]) + s * torch.einsum("tr,tro->to", torch.einsum("th,thr->tr", x, gu[0][e]), gu[1][e])
        gate, up = h.chunk(2, -1)
        a = torch.nn.functional.silu(gate) * up
        o = torch.einsum("ti,thi->th", a, W2[e]) + s * torch.einsum("tr,trh->th", torch.einsum("ti,tir->tr", a, dn[0][e]), dn[1][e])
        out = out + o * w[:, slot : slot + 1]
    return out


def _run(forward, mod, x0, idx, w, gu, dn, s, dtype):
    for p in (*gu, *dn):
        p.grad = None
    x = x0.detach().to(dtype).requires_grad_(True)
    setattr(mod, MU.moe_lora_stash_name("gate_up_proj"), (gu[0], gu[1], s))
    setattr(mod, MU.moe_lora_stash_name("down_proj"), (dn[0], dn[1], s))
    y = forward(mod, x, idx, w.to(dtype))
    gy = torch.linspace(-1, 1, y.numel(), device = y.device).reshape(y.shape).to(y.dtype)
    (y.float() * gy.float()).sum().backward()
    return y.detach().double(), x.grad.double(), [p.grad.double() for p in (*gu, *dn)]


def _oracle(mod, x0, idx, w, gu, dn, s):
    x = x0.detach().double().requires_grad_(True)
    g64 = [p.detach().double().requires_grad_(True) for p in (*gu, *dn)]
    y = _reference(mod, x, idx, w.double(), g64[:2], g64[2:], s)
    gy = torch.linspace(-1, 1, y.numel(), device = y.device, dtype = torch.float64).reshape(y.shape)
    (y * gy).sum().backward()
    return y.detach(), x.grad, [p.grad for p in g64]


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _compare(fold, split, ref, slack = 1.5, floor = 2e-3):
    """Fold error vs fp64 within slack x the separated path's (plus a bf16 floor), every tensor."""
    names = ["out", "dX", "dA_gate_up", "dB_gate_up", "dA_down", "dB_down"]
    flat = lambda r: [r[0], r[1], *r[2]]
    rows = []
    for name, f, sp, rf in zip(names, flat(fold), flat(split), flat(ref)):
        ef, es = _rel(f, rf), _rel(sp, rf)
        cos = torch.nn.functional.cosine_similarity(f.flatten(), rf.flatten(), dim = 0).item()
        rows.append((name, ef, es, cos))
        assert ef <= slack * es + floor, (name, ef, es)
        assert cos > 0.9999, (name, cos)
    return rows


# ---------------------------------------------------------------- CPU


def test_kill_switch_and_cpu_decline(monkeypatch):
    first, second = torch.randn(4, 16, 8), torch.randn(4, 8, 24)
    x = torch.randn(10, 16, dtype = torch.bfloat16)
    w = torch.zeros(4, 24, 16, dtype = torch.bfloat16)
    assert LF.lora_fold_decline_reason(x, first, second, w) == "device"
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "0")
    assert not LF.lora_fold_enabled()
    assert LF.lora_fold_decline_reason(x, first, second, w) == "disabled"
    assert LF.lora_fold_base_mm(x, None, None, False, (first, second, 1.0), w) is None
    assert LF.LORA_FOLD_STATS["declined"] == 1 and LF.LAST_DECLINE["reason"] == "disabled"


def test_native_loop_unchanged_by_the_switch(monkeypatch):
    """native_torch keeps the separated arithmetic: bitwise the same output and grads with the fold on or off."""
    E, H, I, T, k, r, s = 4, 32, 24, 40, 2, 8, 2.0
    mod = Experts(E, H, I, torch.bfloat16, "cpu")
    gu, dn = _lora(E, H, 2 * I, r, "cpu", 1), _lora(E, I, H, r, "cpu", 2)
    x0 = torch.randn(T, H, generator = torch.Generator().manual_seed(3))
    idx, w = _routing(T, E, k, "cpu", 4)
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "0")
    off = _run(MU.forward_native_moe_loop, mod, x0, idx, w, gu, dn, s, torch.bfloat16)
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "1")
    on = _run(MU.forward_native_moe_loop, mod, x0, idx, w, gu, dn, s, torch.bfloat16)
    for a, b in zip([on[0], on[1], *on[2]], [off[0], off[1], *off[2]]):
        assert torch.equal(a, b)


# ---------------------------------------------------------------- GPU kernel


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [4, 12, 16, 64, 128])
def test_kernel_fold_matches_fp64(dtype, rank):
    dev = torch.device("cuda")
    if not MG.triton_grouped_available(dev):
        pytest.skip("Triton grouped GEMM unavailable")
    g = torch.Generator().manual_seed(rank)
    E, K, N = 5, 96, 80
    counts = torch.tensor([7, 0, 33, 1, 19])
    offsets = torch.cumsum(counts, 0).to(torch.int32).to(dev)
    M = int(counts.sum())
    e_of = torch.repeat_interleave(torch.arange(E), counts).to(dev)
    x = torch.randn(M, K, generator = g).to(dev, dtype)
    t = torch.randn(M, rank, generator = g).to(dev, dtype)
    w_nk = (torch.randn(E, N, K, generator = g) * 0.1).to(dev, dtype)    # stored [E, N, K]; the GEMM takes its view
    b = (torch.randn(E, rank, N, generator = g) * 0.1).to(dev, dtype)
    s = 2.0
    ref = torch.einsum("mk,mnk->mn", x.double(), w_nk.double()[e_of]) + s * torch.einsum("mr,mrn->mn", t.double(), b.double()[e_of])
    got = MG.generic_grouped_mm_lora(x, w_nk.transpose(-2, -1), offsets, t, b, s)
    split = MU._grouped_mm_eager(x, w_nk.transpose(-2, -1), offsets) + MU._grouped_mm_eager(t, b, offsets) * s
    ef, es = _rel(got.double(), ref), _rel(split.double(), ref)
    assert ef <= 1.25 * es + 1e-4, (ef, es)
    # Accumulate mode: out += s * t @ b in place, rows of the empty expert untouched.
    base = torch.randn(M, N, generator = g).to(dev, dtype)
    before = base.clone()
    MG.generic_lora_accumulate_(base, t, b, offsets, s)
    ref_acc = before.double() + s * torch.einsum("mr,mrn->mn", t.double(), b.double()[e_of])
    assert _rel(base.double(), ref_acc) <= 1.25 * _rel((before + MU._grouped_mm_eager(t, b, offsets) * s).double(), ref_acc) + 1e-4


@needs_kernel
def test_kernel_variants_do_not_grow_with_rows():
    """Row counts change every step; the folded launch must not compile a variant per count (README 13.2)."""
    dev = torch.device("cuda")
    if not MG.triton_grouped_available(dev):
        pytest.skip("Triton grouped GEMM unavailable")
    k = MG._grouped_gemm_kernel
    while not hasattr(k, "device_caches"):
        k = k.fn
    count = lambda: sum(len(c[0]) for c in k.device_caches.values())
    E, K, N, r = 8, 128, 96, 16
    w = torch.randn(E, K, N, device = dev, dtype = torch.bfloat16)
    b = torch.randn(E, r, N, device = dev, dtype = torch.bfloat16)
    seen = []
    for M in (64, 200, 333, 1024, 77, 4096):
        counts = torch.full((E,), M // E, dtype = torch.int32)
        counts[0] += M - int(counts.sum())
        offsets = torch.cumsum(counts, 0).to(torch.int32).to(dev)
        MG.generic_grouped_mm_lora(torch.randn(M, K, device = dev, dtype = torch.bfloat16), w, offsets,
                                   torch.randn(M, r, device = dev, dtype = torch.bfloat16), b, 1.0)
        seen.append(count())
    # Tiles follow rows per expert (a few buckets), never the exact count.
    assert seen[-1] - seen[0] <= 2, seen


# ---------------------------------------------------------------- GPU end to end through forward_native_grouped_mm


@needs_kernel
@pytest.mark.parametrize("rank", [4, 16, 31])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_mm_fold_parity(monkeypatch, rank, dtype):
    _force_triton(monkeypatch)
    dev = "cuda"
    E, H, I, T, k, s = 8, 128, 96, 96, 2, 2.0
    mod = Experts(E, H, I, dtype, dev)
    gu, dn = _lora(E, H, 2 * I, rank, dev, 1), _lora(E, I, H, rank, dev, 2)
    x0 = torch.randn(T, H, generator = torch.Generator().manual_seed(3)).to(dev)
    idx, w = _routing(T, E, k, dev, 4)
    ref = _oracle(mod, x0, idx, w, gu, dn, s)
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "0")
    split = _run(MU.forward_native_grouped_mm, mod, x0, idx, w, gu, dn, s, dtype)
    assert LF.LORA_FOLD_STATS["fold"] == 0
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "1")
    fold = _run(MU.forward_native_grouped_mm, mod, x0, idx, w, gu, dn, s, dtype)
    assert LF.LORA_FOLD_STATS["fold"] == 2 and LF.LORA_FOLD_STATS["backward"] == 2, LF.LORA_FOLD_STATS
    _compare(fold, split, ref)


@needs_kernel
def test_fold_recompute_provider_calls(monkeypatch):
    """Recompute keeps the separated path's dequant count: one provider call in forward, one in backward."""
    _force_triton(monkeypatch)
    dev = torch.device("cuda")
    E, K, N, r, M = 4, 64, 48, 16, 80
    counts = torch.tensor([30, 0, 25, 25])
    offsets = torch.cumsum(counts, 0).to(torch.int32).to(dev)
    stored = (torch.randn(E, N, K, device = dev) * 0.1).to(torch.bfloat16)
    calls = []

    def provider():
        calls.append(1)
        return stored.clone().transpose(-2, -1)   # a fresh tensor, as a dequant would return

    x = torch.randn(M, K, device = dev, dtype = torch.bfloat16, requires_grad = True)
    first = (torch.randn(E, K, r, device = dev) * 0.1).to(torch.bfloat16).requires_grad_(True)
    second = (torch.randn(E, r, N, device = dev) * 0.1).to(torch.bfloat16).requires_grad_(True)
    y = LF._GroupedLoraFold.apply(x, first, second, offsets, 1.0, provider, True, None)
    assert len(calls) == 1
    y.float().sum().backward()
    assert len(calls) == 2 and x.grad is not None and first.grad is not None and second.grad is not None
    # Pinned (recompute False): one call, the weight is saved instead.
    calls.clear()
    y = LF._GroupedLoraFold.apply(x, first, second, offsets, 1.0, provider, False, None)
    y.float().sum().backward()
    assert len(calls) == 1


@needs_kernel
def test_fold_declines_where_base_keeps_cublas(monkeypatch):
    """Auto policy off sm80 / sm89 / sm12x (and forced off): the base GEMM is cuBLAS, so the fold stays out."""
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    MU._TRITON_GROUPED_MM_POLICY.clear()
    x = torch.randn(64, 32, device = "cuda", dtype = torch.bfloat16)
    first = torch.randn(4, 32, 8, device = "cuda", dtype = torch.bfloat16)
    second = torch.randn(4, 8, 48, device = "cuda", dtype = torch.bfloat16)
    w = torch.zeros(4, 48, 32, device = "cuda", dtype = torch.bfloat16)
    assert LF.lora_fold_decline_reason(x, first, second, w) == "policy"
    trainable = torch.zeros(4, 48, 32, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    assert LF.lora_fold_decline_reason(x, first, second, trainable) == "trainable_base"
    _force_triton(monkeypatch)
    assert LF.lora_fold_decline_reason(x, first, second, w) is None
    assert LF.lora_fold_decline_reason(x, first, second, w.cpu()) == "device"   # offloaded stack

@needs_kernel
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_fold_traces_without_extra_graphs(monkeypatch, backend, fresh):
    """Traced callers fold through the opaque op: the same frames and graph breaks as the separated path, the fold op
    runs forward and dX, and the result matches eager. fresh: traced before the kernel self-check ran (a real
    training run's first step), which must not cost the fold a recompile."""
    from torch._dynamo.utils import counters
    _force_triton(monkeypatch)
    if LF._FOLD_OP is None:
        pytest.skip("torch.library.custom_op unavailable")
    dev, dtype = "cuda", torch.bfloat16
    E, H, I, T, k, s, r = 8, 128, 96, 96, 2, 2.0, 16
    mod = Experts(E, H, I, dtype, dev)
    gu, dn = _lora(E, H, 2 * I, r, dev, 1), _lora(E, I, H, r, dev, 2)
    x0 = torch.randn(T, H, generator = torch.Generator().manual_seed(3)).to(dev)
    idx, w = _routing(T, E, k, dev, 4)

    def f(mod, x, idx, w):
        return MU.forward_native_grouped_mm(mod, x, idx, w)

    res = {}
    for fold in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", fold)
        torch._dynamo.reset()
        counters.clear()
        if fresh:
            monkeypatch.setattr(MG, "_GENERIC_SELF_CHECKED", {})
        calls = dict(MG.LORA_CALLS)
        compiled = torch.compile(f, backend = backend)
        out = _run(compiled, mod, x0, idx, w, gu, dn, s, dtype)
        res[fold] = (out, counters["stats"]["unique_graphs"], sum(counters["graph_break"].values()),
                     MG.LORA_CALLS["fold"] - calls["fold"])
    assert res["1"][1] == res["0"][1] and res["1"][2] == res["0"][2], (res["0"][1:], res["1"][1:])
    assert res["0"][3] == 0 and res["1"][3] == 4    # gate_up + down, forward + dX
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "1")
    eager = _run(MU.forward_native_grouped_mm, mod, x0, idx, w, gu, dn, s, dtype)
    for a, b in zip([res["1"][0][0], res["1"][0][1], *res["1"][0][2]], [eager[0], eager[1], *eager[2]]):
        assert _rel(a, b) < 1e-2


def test_fold_op_opcheck():
    if LF._FOLD_OP is None or not CUDA or not MG.triton_grouped_available(torch.device("cuda")):
        pytest.skip("needs the op and the Triton kernel")
    dev = "cuda"
    E, K, N, r = 4, 32, 48, 16
    offsets = torch.tensor([5, 5, 20, 33], dtype = torch.int32, device = dev)
    args = (torch.randn(33, K, device = dev, dtype = torch.bfloat16), torch.randn(E, K, N, device = dev, dtype = torch.bfloat16),
            offsets, torch.randn(33, r, device = dev, dtype = torch.bfloat16), torch.randn(E, r, N, device = dev, dtype = torch.bfloat16), 2.0)
    torch.library.opcheck(torch.ops.unsloth_zoo.grouped_mm_lora_fold_v1.default, args,
                          test_utils = ("test_schema", "test_faketensor"))
