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

"""UNSLOTH_MOE_NF4_FUSED: NF4 decode inside the grouped GEMM. Decode is bitwise the bitsandbytes
dequant, forward and dX sit within the dequant + same-dtype GEMM error of an fp64 oracle, the op
compiles fullgraph without graph breaks or recompiles, and the MoE forward with LoRA matches the
dequant path in output and every gradient."""
import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.nn import Params4bit

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
    pytest.skip("NF4 fused grouped GEMM needs a CUDA GPU", allow_module_level = True)

from unsloth_zoo.temporary_patches import moe_nf4_grouped as nf

DEVICE = "cuda"
MODE = "1" if torch.cuda.get_device_capability()[0] >= 8 else "force"
HAS_BF16 = torch.cuda.get_device_capability()[0] >= 8


@pytest.fixture(autouse = True)
def _bf16_needs_sm80(request):
    params = getattr(request.node, "callspec", None)
    if params is not None and params.params.get("dtype") is torch.bfloat16 and not HAS_BF16:
        pytest.skip("Triton bf16 dot needs sm80+")


def _stack(shape, dtype, nested, blocksize = 64, storage = torch.uint8, scale = 0.05):
    w = torch.randn(*shape, dtype = dtype) * scale
    p = Params4bit(
        w, requires_grad = False, quant_type = "nf4", compress_statistics = nested,
        blocksize = blocksize, quant_storage = storage,
    ).to(DEVICE)
    p._original_shape = torch.Size(shape)
    return p


def _dequant(p):
    data = p.data if p.data.dtype == torch.uint8 else p.data.view(torch.uint8)
    return bnb.functional.dequantize_4bit(data.reshape(-1, 1), p.quant_state).reshape(p._original_shape)


def _routing(E, counts):
    counts = torch.tensor(counts, dtype = torch.int32, device = DEVICE)
    assert counts.numel() == E
    return counts.cumsum(0).to(torch.int32)


def _logical(deq, kn_layout):
    return deq if kn_layout else deq.transpose(-2, -1)


def _spec(p, kn_layout, dtype):
    pre = (lambda w: w) if kn_layout else (lambda w: w.transpose(-2, -1))
    spec = nf.nf4_grouped_spec(p, pre, dtype, mode = MODE)
    assert spec is not None
    return spec


def _oracle(x, w, offsets):
    """fp64 y and the per-expert loop in x.dtype (the dequant + GEMM reference error)."""
    y64 = torch.zeros(x.shape[0], w.shape[-1], dtype = torch.float64, device = DEVICE)
    yd = torch.zeros(x.shape[0], w.shape[-1], dtype = x.dtype, device = DEVICE)
    lo = 0
    for e, hi in enumerate(offsets.tolist()):
        if hi > lo:
            y64[lo:hi] = x[lo:hi].double() @ w[e].double()
            yd[lo:hi] = x[lo:hi] @ w[e].to(x.dtype)
        lo = hi
    return y64, yd


def _err(got, ref):
    d = got.double() - ref
    return d.abs().max().item(), (d.norm() / ref.norm().clamp_min(1e-30)).item()


CASES = [
    # (E, K, N, counts, kn_layout, nested, blocksize)
    (7, 64, 96, [5, 0, 9, 0, 0, 3, 17], True, False, 64),
    (7, 64, 96, [5, 0, 9, 0, 0, 3, 17], False, True, 64),
    (4, 128, 256, [1, 0, 0, 0], True, True, 64),         # one token
    (4, 128, 256, [40, 33, 0, 70], False, False, 128),
    (5, 70, 97, [3, 8, 0, 21, 2], True, True, 64),       # odd K / N, not tile multiples
    (5, 70, 97, [3, 8, 0, 21, 2], False, False, 64),
    (3, 96, 160, [100, 0, 300], True, True, 128),
    (3, 96, 160, [0, 0, 0], True, False, 64),            # every expert empty
    (16, 256, 128, [12] * 16, False, True, 256),
    (2, 128, 192, [300, 260], False, True, 64),          # >= 256 rows per expert: two M sub-tiles
    (2, 128, 192, [300, 260], True, False, 64),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES)
def test_forward_and_dx_vs_fp64_oracle(case, dtype):
    torch.manual_seed(0)
    E, K, N, counts, kn_layout, nested, bs = case
    stored = (E, K, N) if kn_layout else (E, N, K)
    p = _stack(stored, dtype, nested, bs)
    w = _logical(_dequant(p), kn_layout)
    offsets = _routing(E, counts)
    M = int(sum(counts))
    x = (torch.randn(M, K, device = DEVICE, dtype = dtype)).requires_grad_()
    dy = torch.randn(M, N, device = DEVICE, dtype = dtype)
    y = nf.nf4_grouped_mm(x, offsets, _spec(p, kn_layout, dtype))
    assert y.shape == (M, N) and y.dtype == dtype
    (dx,) = torch.autograd.grad(y, x, dy) if M else (torch.zeros_like(x),)
    if M == 0:
        return
    y64, yd = _oracle(x.detach(), w, offsets)
    dx64, dxd = _oracle(dy, w.transpose(-2, -1), offsets)
    for got, ref, base in ((y, y64, yd), (dx, dx64, dxd)):
        assert torch.isfinite(got).all()
        abs_err, rel = _err(got, ref)
        base_abs, base_rel = _err(base, ref)
        # Same operands, fp32 accumulate: within the rounding of the dequant + GEMM reference.
        assert rel <= 2 * base_rel + 1e-6, (rel, base_rel)
        assert abs_err <= 2 * base_abs + 1e-6, (abs_err, base_abs)


@pytest.mark.parametrize("lut", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kn_layout", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_is_bitwise_bitsandbytes(nested, kn_layout, dtype, lut, monkeypatch):
    # Identity rows pick each weight out exactly (one product per output, fp32 accumulate).
    fixed = nf._fixed_config
    monkeypatch.setattr(nf, "_fixed_config", lambda *a: ({**fixed(*a)[0], "LUT": lut},) + fixed(*a)[1:])
    torch.manual_seed(1)
    E, K, N = 3, 64, 130
    p = _stack((E, K, N) if kn_layout else (E, N, K), dtype, nested)
    w = _logical(_dequant(p), kn_layout)
    x = torch.eye(K, device = DEVICE, dtype = dtype).repeat(E, 1)
    offsets = _routing(E, [K] * E)
    y = nf.nf4_grouped_mm(x, offsets, _spec(p, kn_layout, dtype))
    assert torch.equal(y.view(E, K, N), w)
    dy = torch.eye(N, device = DEVICE, dtype = dtype).repeat(E, 1)
    offsets = _routing(E, [N] * E)
    dx = torch.ops.unsloth_zoo.nf4_grouped_mm_dx(dy, *_saved(_spec(p, kn_layout, dtype), offsets))
    assert torch.equal(dx.view(E, N, K), w.transpose(-2, -1))


def test_run_to_run_bitwise(monkeypatch):
    # Fixed tiles per shape (no timing-picked config): the same inputs give the same bits.
    monkeypatch.delenv("UNSLOTH_MOE_NF4_AUTOTUNE", raising = False)
    torch.manual_seed(7)
    E, K, N = 8, 256, 192
    dt = torch.bfloat16 if HAS_BF16 else torch.float16
    p = _stack((E, N, K), dt, True)
    spec = _spec(p, False, dt)
    offsets = _routing(E, [30, 0, 70, 1, 0, 16, 9, 2])
    x = torch.randn(128, K, device = DEVICE, dtype = dt)
    ys = [nf.nf4_grouped_mm(x, offsets, spec) for _ in range(3)]
    assert all(torch.equal(ys[0], y) for y in ys[1:])


@pytest.mark.parametrize("kn_layout", [True, False])
def test_two_m_subtiles_match_one(kn_layout, monkeypatch):
    # MSUB 2 reuses one decoded weight tile for two BM row blocks: same per-row sums, same bits.
    torch.manual_seed(11)
    E, K, N = 6, 192, 256
    dt = torch.bfloat16 if HAS_BF16 else torch.float16
    p = _stack((E, K, N) if kn_layout else (E, N, K), dt, True)
    spec = _spec(p, kn_layout, dt)
    offsets = _routing(E, [70, 0, 129, 1, 64, 300])
    x = torch.randn(564, K, device = DEVICE, dtype = dt, requires_grad = True)
    dy = torch.randn(564, N, device = DEVICE, dtype = dt)
    outs = []
    for msub in (1, 2):
        monkeypatch.setattr(nf, "_fixed_config", lambda m, i, o, pair, msub = msub: (
            {"BM": 64, "BN": 64 if pair == 1 else 128, "BK": 64, "LUT": 1, "MSUB": msub}, 4, 3))
        y = nf.nf4_grouped_mm(x, offsets, spec)
        outs.append((y, torch.autograd.grad(y, x, dy)[0]))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def _saved(spec, offsets):
    return (
        spec["packed"], spec["absmax"], spec["code"], spec["absmax2"], spec["code2"], spec["offset"], offsets,
        spec["K"], spec["N"], spec["kn_layout"], spec["blocksize"], spec["blocksize2"], spec["nested"], spec["w_dtype"],
    )


def test_float_quant_storage_and_fp16_extremes():
    torch.manual_seed(2)
    E, K, N = 4, 64, 64
    p = _stack((E, N, K), torch.float16, True, storage = torch.float16, scale = 2000.0)
    w = _logical(_dequant(p), False)
    offsets = _routing(E, [2, 6, 0, 9])
    x = torch.randn(17, K, device = DEVICE, dtype = torch.float16) * 2
    y = nf.nf4_grouped_mm(x, offsets, _spec(p, False, torch.float16))
    y64, yd = _oracle(x, w, offsets)
    finite = torch.isfinite(yd)
    # Overflow where the dequant GEMM overflows, and nowhere else.
    assert torch.equal(torch.isfinite(y), finite)
    assert _err(y[finite], y64[finite])[1] <= 2 * _err(yd[finite], y64[finite])[1] + 1e-6


def test_unsupported_stacks_fall_back():
    p = _stack((2, 64, 64), torch.bfloat16, False)
    assert nf.nf4_grouped_spec(p, lambda w: w, torch.float32, mode = MODE) is None
    assert nf.nf4_grouped_spec(p, lambda w: w.contiguous() * 1, torch.bfloat16, mode = MODE) is None
    assert nf.nf4_grouped_spec(p, lambda w: w, torch.bfloat16, mode = "0") is None
    fp4 = Params4bit(torch.randn(2, 64, 64, dtype = torch.bfloat16), requires_grad = False, quant_type = "fp4").to(DEVICE)
    fp4._original_shape = torch.Size((2, 64, 64))
    assert nf.nf4_grouped_spec(fp4, lambda w: w, torch.bfloat16, mode = MODE) is None
    plain = torch.randn(2, 64, 64, device = DEVICE, dtype = torch.bfloat16)
    assert nf.nf4_grouped_spec(plain, lambda w: w, torch.bfloat16, mode = MODE) is None


@pytest.mark.parametrize("kn_layout", [True, False])
def test_torch_decode_fallback_matches_kernel(kn_layout, monkeypatch):
    torch.manual_seed(6)
    E, K, N = 4, 128, 96
    dt = torch.bfloat16 if HAS_BF16 else torch.float16
    p = _stack((E, K, N) if kn_layout else (E, N, K), dt, True)
    spec = _spec(p, kn_layout, dt)
    offsets = _routing(E, [5, 0, 11, 3])
    x = torch.randn(19, K, device = DEVICE, dtype = dt, requires_grad = True)
    dy = torch.randn(19, N, device = DEVICE, dtype = dt)
    y = nf.nf4_grouped_mm(x, offsets, spec)
    (dx,) = torch.autograd.grad(y, x, dy)
    monkeypatch.setattr(nf, "_DISABLED_REASON", "test")
    assert nf.nf4_grouped_spec(p, lambda w: w, dt, mode = MODE) is None
    y2 = nf.nf4_grouped_mm(x, offsets, spec)
    (dx2,) = torch.autograd.grad(y2, x, dy)
    assert _err(y2, y.double())[1] < 5e-3 and _err(dx2, dx.double())[1] < 5e-3


def test_launch_failure_falls_back_per_shape(monkeypatch):
    torch.manual_seed(8)
    E, K, N = 4, 64, 128
    dt = torch.bfloat16 if HAS_BF16 else torch.float16
    p = _stack((E, N, K), dt, False)
    spec = _spec(p, False, dt)
    offsets = _routing(E, [3, 0, 5, 2])
    x = torch.randn(10, K, device = DEVICE, dtype = dt)
    want = nf.nf4_grouped_mm(x, offsets, spec)
    monkeypatch.setattr(nf, "FAILED_SHAPES", {})
    def boom(*args, **kwargs):
        raise RuntimeError("simulated launch failure")
    monkeypatch.setattr(nf, "_run_kernel", boom)
    got = nf.nf4_grouped_mm(x, offsets, spec)
    assert _err(got, want.double())[1] < 5e-3
    assert list(nf.FAILED_SHAPES) == [(K, N, 1, K)]
    pre = lambda w: w.transpose(-2, -1)
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", MODE)
    assert nf.maybe_nf4_grouped_mm(x, offsets, p, pre) is None   # that stack now keeps the dequant path


def test_auto_switch_boundary(monkeypatch):
    # auto: fused at or below the rows-per-expert limit (from shapes), the dequant path above it.
    torch.manual_seed(9)
    E, K, N = 8, 128, 192
    dt = torch.bfloat16 if HAS_BF16 else torch.float16
    p = _stack((E, N, K), dt, True)
    w = _logical(_dequant(p), False)
    pre = lambda t: t.transpose(-2, -1)
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto" if HAS_BF16 else "force")
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "4")
    if not HAS_BF16:
        pytest.skip("auto needs sm80+")
    for M, fused in ((31, True), (32, True), (33, False), (64, False)):
        counts = [M // E] * E
        counts[0] += M - sum(counts)
        offsets = _routing(E, counts)
        x = torch.randn(M, K, device = DEVICE, dtype = dt)
        before = nf.NF4_FUSED_STATS["launches"]
        y = nf.maybe_nf4_grouped_mm(x, offsets, p, pre)
        assert (y is not None) == fused and (nf.NF4_FUSED_STATS["launches"] > before) == fused
        if fused:
            y64, yd = _oracle(x, w, offsets)
            assert _err(y, y64)[1] <= 2 * _err(yd, y64)[1] + 1e-6
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "1")
    assert nf.maybe_nf4_grouped_mm(x, offsets, p, pre) is not None   # "1" ignores the limit


def test_auto_branch_fullgraph(monkeypatch):
    import torch._dynamo
    from torch._dynamo.utils import counters

    if not HAS_BF16 or not hasattr(torch, "_grouped_mm"):
        pytest.skip("auto needs sm80+ and torch._grouped_mm")
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "4")
    torch.manual_seed(10)
    E, K, N = 8, 128, 192
    p = _stack((E, N, K), torch.bfloat16, True)
    spec = _spec(p, False, torch.bfloat16)
    w = _logical(_dequant(p), False).contiguous()

    def fn(h, offsets):
        if nf.nf4_auto_fused(h.shape[0], E):
            y = nf.nf4_grouped_mm(h, offsets, spec)
        else:
            y = torch._grouped_mm(h, w, offs = offsets)
        return y.float().square().sum()

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, fullgraph = True)
    for M in (16, 24, 32, 40, 64, 48):
        h = torch.randn(M, K, device = DEVICE, dtype = torch.bfloat16, requires_grad = True)
        torch._dynamo.mark_dynamic(h, 0)
        offsets = _routing(E, [M // E] * E)
        loss = compiled(h, offsets)
        (g,) = torch.autograd.grad(loss, h)
        h2 = h.detach().clone().requires_grad_()
        (g_ref,) = torch.autograd.grad(_oracle_loss(h2, torch.eye(K, device = DEVICE, dtype = torch.bfloat16), w, offsets), h2)
        assert _err(g, g_ref.double())[1] < 2e-2
    assert sum(counters["graph_break"].values()) == 0
    # One graph per side of the row guard: the boundary costs exactly one recompile.
    assert counters["stats"]["unique_graphs"] == 2


def test_opcheck():
    torch.manual_seed(3)
    p = _stack((5, 96, 64), torch.float16, True)
    spec = _spec(p, False, torch.float16)
    x = torch.randn(13, 64, device = DEVICE, dtype = torch.float16, requires_grad = True)
    offsets = _routing(5, [4, 0, 6, 0, 3])
    args = (x,) + _saved(spec, offsets)[:7] + _saved(spec, offsets)[7:]
    torch.library.opcheck(torch.ops.unsloth_zoo.nf4_grouped_mm.default, args)
    dy = torch.randn(13, 96, device = DEVICE, dtype = torch.float16)
    torch.library.opcheck(torch.ops.unsloth_zoo.nf4_grouped_mm_dx.default, (dy,) + _saved(spec, offsets))


def test_fullgraph_no_breaks_no_recompiles():
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch.manual_seed(4)
    E, K, N = 8, 128, 192
    dt = torch.bfloat16 if HAS_BF16 else torch.float16
    p = _stack((E, N, K), dt, True)
    spec = _spec(p, False, dt)
    lin = torch.randn(K, K, device = DEVICE, dtype = dt) * 0.05

    def fn(h, offsets):
        return nf.nf4_grouped_mm(h @ lin, offsets, spec).float().square().sum()

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, fullgraph = True)
    w = _logical(_dequant(p), False)
    for M, counts in ((24, [3] * 8), (40, [0, 10, 0, 10, 0, 10, 0, 10]), (7, [7, 0, 0, 0, 0, 0, 0, 0]), (64, [8] * 8)):
        h = torch.randn(M, K, device = DEVICE, dtype = dt, requires_grad = True)
        torch._dynamo.mark_dynamic(h, 0)
        offsets = _routing(E, counts)
        loss = compiled(h, offsets)
        (g,) = torch.autograd.grad(loss, h)
        h2 = h.detach().clone().requires_grad_()
        ref = _oracle_loss(h2, lin, w, offsets)
        (g_ref,) = torch.autograd.grad(ref, h2)
        assert _err(g, g_ref.double())[1] < 2e-2
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1
    assert sum(counters["recompiles"].values()) == 0 if "recompiles" in counters else True


def _oracle_loss(h, lin, w, offsets):
    x = h @ lin
    outs, lo = [], 0
    for e, hi in enumerate(offsets.tolist()):
        outs.append(x[lo:hi] @ w[e].to(x.dtype))
        lo = hi
    return torch.cat(outs).float().square().sum()


def test_other_cuda_device_is_current():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    torch.manual_seed(5)
    p = Params4bit(torch.randn(3, 64, 64, dtype = torch.float16) * 0.05, requires_grad = False, quant_type = "nf4").to("cuda:1")
    p._original_shape = torch.Size((3, 64, 64))
    spec = nf.nf4_grouped_spec(p, lambda w: w, torch.float16, mode = MODE)
    x = torch.randn(9, 64, device = "cuda:1", dtype = torch.float16)
    offsets = torch.tensor([3, 3, 9], dtype = torch.int32, device = "cuda:1")
    with torch.cuda.device(0):
        y = nf.nf4_grouped_mm(x, offsets, spec)
    w = _dequant(p)
    ref = torch.cat([x[:3] @ w[0], x[3:9] @ w[2]])
    assert _err(y, ref.double())[1] < 1e-3


# MoE forward with separated LoRA: fused vs the dequant path.

import unsloth_zoo.temporary_patches.moe_utils as mu

E, H, I, TOP_K, RANK = 16, 128, 64, 2, 4


class _Experts(nn.Module):
    def __init__(self, nested, dtype):
        super().__init__()
        self.num_experts = E
        self.gate_up_proj = _stack((E, 2 * I, H), dtype, nested)
        self.down_proj = _stack((E, H, I), dtype, nested)
        self.act_fn = nn.SiLU()


def _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch):
    stash = {
        name: (first.detach().clone().requires_grad_(), second.detach().clone().requires_grad_(), 2.0)
        for name, (first, second) in lora.items()
    }
    monkeypatch.setattr(mu, "take_moe_lora_stash", lambda module, name: stash.get(name))
    hidden = hidden.detach().clone().requires_grad_()
    weights = top_k_weights.detach().clone().requires_grad_()
    out = mu.forward_native_grouped_mm(experts, hidden, top_k_index, weights)
    out.float().square().sum().backward()
    grads = [hidden.grad, weights.grad] + [t.grad for name in sorted(stash) for t in stash[name][:2]]
    return out.detach(), grads


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_forward_with_lora_matches_dequant_path(nested, dtype, monkeypatch):
    if not mu._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported here")
    for key in ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "UNSLOTH_MOE_RECOMPUTE"):
        monkeypatch.delenv(key, raising = False)
    torch.manual_seed(0)
    experts = _Experts(nested, dtype)
    T = 37
    hidden = torch.randn(T, H, dtype = dtype, device = DEVICE)
    pool = torch.tensor([0, 3, 4, 9, 10, 14], device = DEVICE)
    top_k_index = pool[torch.stack([torch.randperm(6, device = DEVICE)[:TOP_K] for _ in range(T)])]
    top_k_weights = torch.rand(T, TOP_K, dtype = dtype, device = DEVICE)
    lora = {
        "gate_up_proj": (torch.randn(E, H, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, 2 * I, device = DEVICE) * 0.1),
        "down_proj": (torch.randn(E, I, RANK, device = DEVICE) * 0.1, torch.randn(E, RANK, H, device = DEVICE) * 0.1),
    }
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "0")
    nf.NF4_FUSED_STATS["calls"] = 0
    base_out, base_grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert nf.NF4_FUSED_STATS["calls"] == 0
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", MODE)
    out, grads = _run(experts, hidden, top_k_index, top_k_weights, lora, monkeypatch)
    assert nf.NF4_FUSED_STATS["calls"] == 2   # gate_up + down forward (dX runs inside the op's backward)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert _err(out, base_out.double())[1] < tol
    for got, want in zip(grads, base_grads):
        assert _err(got, want.double())[1] < tol
    unused = torch.ones(E, dtype = torch.bool, device = DEVICE)
    unused[pool] = False
    for g in grads[2:]:
        assert g[unused].abs().sum().item() == 0
