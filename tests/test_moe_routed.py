# SPDX-License-Identifier: AGPL-3.0-only
"""Routed NF4 / BF16 MoE decode kernels (moe_routed.py) for transformers v5 3D experts.

Every arm is checked against an fp64 reference built from bitsandbytes' own dequant, and
against the path the kernels replace (dequantize every expert + grouped_mm), which must be
no more accurate than the routed one."""
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
pytest.importorskip("triton")
bnb = pytest.importorskip("bitsandbytes")

from unsloth_zoo.temporary_patches import moe_routed as MR
from unsloth_zoo.temporary_patches import moe_utils as MU
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit

DEV = "cuda"
# T4 (sm75) has no bf16: run the same checks in fp16 there. UNSLOTH_TEST_DTYPE=float16 simulates it.
DT = getattr(torch, os.environ.get("UNSLOTH_TEST_DTYPE", "")) if os.environ.get("UNSLOTH_TEST_DTYPE") else (
    torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
E, H, I = 8, 256, 192


@pytest.fixture(autouse = True)
def _slot_limit(monkeypatch):
    # The toys route up to 8 slots per expert so small E still covers T = 4 at top-k 4 / 8; the
    # production limit (P <= E) has its own test.
    monkeypatch.setattr(MR, "NF4_MAX_SLOTS", None)
    monkeypatch.setattr(MR, "NF4_SLOTS_PER_EXPERT", 8.0)


def _q4(w, blocksize = 64, nested = True, quant_type = "nf4"):
    p = bnb.nn.Params4bit(
        w.to(DT).cpu(), requires_grad = False, compress_statistics = nested,
        quant_type = quant_type, blocksize = blocksize,
    ).to(DEV)
    p._original_shape = tuple(w.shape)
    return p


def _weights(seed, h = H, i = I, e = E):
    g = torch.Generator().manual_seed(seed)
    # Unit-scale entries times a per-expert scale spanning 2^6: distinct absmax / state2 per expert.
    scale = (2.0 ** torch.linspace(-4, 2, e)).view(e, 1, 1)
    gu = torch.randn(e, 2 * i, h, generator = g) * scale * 0.05
    dn = torch.randn(e, h, i, generator = g) * scale.flip(0) * 0.05
    return gu, dn


class ToyExperts(nn.Module):
    def __init__(self, act_fn, interleaved = False, h = H, i = I, e = E):
        super().__init__()
        self.num_experts = e
        self.hidden_dim = h
        self.intermediate_dim = i
        self.act_fn = act_fn
        self.is_concatenated = not interleaved
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h), requires_grad = False)
        self.down_proj = nn.Parameter(torch.empty(e, h, i), requires_grad = False)


class ToyGptOssExperts(ToyExperts):
    pass


ToyGptOssExperts.__name__ = "GptOssExperts"


ACTS = {
    "silu": (lambda: nn.SiLU(), MR.ACT_SILU, False),
    "gelu_tanh": (lambda: nn.GELU(approximate = "tanh"), MR.ACT_GELU_TANH, False),
    "gelu": (lambda: nn.GELU(), MR.ACT_GELU, False),
    "silu_interleaved": (lambda: nn.SiLU(), MR.ACT_SILU, True),
    "gptoss": (lambda: None, MR.ACT_GPTOSS, True),
}


def _make(act, quant = True, seed = 0, nested = True, blocksize = 64, h = H, i = I, e = E):
    make_fn, _, interleaved = ACTS[act]
    klass = ToyGptOssExperts if act == "gptoss" else ToyExperts
    ex = klass(make_fn(), interleaved, h, i, e)
    if act == "gptoss":
        del ex.act_fn
        ex.act_fn = None
        ex.alpha, ex.limit = 1.702, 7.0
        g = torch.Generator().manual_seed(seed + 7)
        ex.gate_up_proj_bias = nn.Parameter((torch.randn(e, 2 * i, generator = g) * 0.1).to(DT).to(DEV), requires_grad = False)
        ex.down_proj_bias = nn.Parameter((torch.randn(e, h, generator = g) * 0.1).to(DT).to(DEV), requires_grad = False)
    gu, dn = _weights(seed, h, i, e)
    if quant:
        ex.gate_up_proj = _q4(gu, blocksize, nested)
        ex.down_proj = _q4(dn, blocksize, nested)
    else:
        ex.gate_up_proj = nn.Parameter(gu.to(DT).to(DEV), requires_grad = False)
        ex.down_proj = nn.Parameter(dn.to(DT).to(DEV), requires_grad = False)
    return ex.eval()


def _route(T, top_k = 4, seed = 1, e = E, h = H):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, h, generator = g).to(DT).to(DEV)
    idx = torch.stack([torch.randperm(e, generator = g)[:top_k] for _ in range(T)]).to(DEV)
    w = torch.softmax(torch.randn(T, top_k, generator = g), -1).to(DEV)
    return x, idx, w


def _lora(ex, seed = 3, r = 8):
    g = torch.Generator().manual_seed(seed)
    e, n_gu, h = ex.gate_up_proj._original_shape if hasattr(ex.gate_up_proj, "_original_shape") else ex.gate_up_proj.shape
    i = n_gu // 2
    gu = (torch.randn(e, h, r, generator = g) * 0.1, torch.randn(e, r, n_gu, generator = g) * 0.1, 0.5, e)
    dn = (torch.randn(e, i, r, generator = g) * 0.1, torch.randn(e, r, h, generator = g) * 0.1, 2.0, e)
    return tuple(t.to(DEV) if isinstance(t, torch.Tensor) else t for t in gu), tuple(t.to(DEV) if isinstance(t, torch.Tensor) else t for t in dn)


def _stash(ex, lora):
    if lora is None:
        for n in ("gate_up_proj", "down_proj"):
            ex.__dict__.pop(MU.moe_lora_stash_name(n), None)
        return
    setattr(ex, MU.moe_lora_stash_name("gate_up_proj"), lora[0])
    setattr(ex, MU.moe_lora_stash_name("down_proj"), lora[1])


def _dense(p):
    if getattr(p, "quant_state", None) is not None:
        return bnb.functional.dequantize_4bit(p.data, p.quant_state).double().view(p._original_shape)
    return p.double()


def _act64(gu, act, interleaved, alpha = 1.702, limit = 7.0):
    gate, up = (gu[..., ::2], gu[..., 1::2]) if interleaved else gu.chunk(2, -1)
    if act == MR.ACT_GPTOSS:
        gate, up = gate.clamp(max = limit), up.clamp(-limit, limit)
        return (up + 1) * gate * torch.sigmoid(gate * alpha)
    if act == MR.ACT_SILU:
        return F.silu(gate) * up
    if act == MR.ACT_GELU_TANH:
        return F.gelu(gate, approximate = "tanh") * up
    return F.gelu(gate) * up


def _ref(ex, act, x, idx, w, lora = None):
    _, code, interleaved = ACTS[act]
    Wg, Wd = _dense(ex.gate_up_proj), _dense(ex.down_proj)
    x64 = x.double()
    gu = torch.einsum("tknh,th->tkn", Wg[idx], x64)
    if getattr(ex, "gate_up_proj_bias", None) is not None:
        gu = gu + ex.gate_up_proj_bias.double()[idx]
    if lora is not None:
        f, s, sc, _ = lora[0]
        gu = gu + torch.einsum("tkhr,th->tkr", f.double()[idx], x64).unsqueeze(-2).matmul(s.double()[idx]).squeeze(-2) * sc
    inter = _act64(gu, code, interleaved)
    d = torch.einsum("tkhi,tki->tkh", Wd[idx], inter)
    if getattr(ex, "down_proj_bias", None) is not None:
        d = d + ex.down_proj_bias.double()[idx]
    if lora is not None:
        f, s, sc, _ = lora[1]
        d = d + torch.einsum("tkir,tki->tkr", f.double()[idx], inter).unsqueeze(-2).matmul(s.double()[idx]).squeeze(-2) * sc
    return (d * w.double()[..., None]).sum(1)


def _current(ex, x, idx, w, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "0")
    try:
        if getattr(ex.gate_up_proj, "quant_state", None) is not None:
            return forward_moe_backend_bnb4bit(ex, x, idx, w)
        return MU.forward_moe_backend(ex, x, idx, w)
    finally:
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1")


def _err(a, ref):
    d = (a.double() - ref).abs()
    return d.max().item(), d.mean().item()


SEEDS = (1, 2, 3, 4)


def _check_vs_current(got, cur, ref, max_factor = 1.5):
    # Lists are pooled over route seeds: a single T=1 seed has only H outputs, and measured
    # single-seed routed / current mean-error ratios reach 1.24 by luck (fp16 gelu_tanh) while
    # the 12-seed mean is below 1 in every act x T x LoRA x dtype cell.
    if isinstance(got, (list, tuple)):
        got, cur, ref = (torch.cat([t.reshape(-1) for t in v]) for v in (got, cur, ref))
    g_max, g_mean = _err(got, ref)
    c_max, c_mean = _err(cur, ref)
    # The routed path keeps fp32 between the two GEMVs; the current one rounds gate_up, the
    # activation and down to the compute dtype, so it must be no worse on average. The output
    # scaled by (1 + 2**-7) must fail both bounds (test_bounds_reject_a_one_ulp_scale_error).
    assert g_mean <= 1.05 * c_mean, (g_mean, c_mean)
    assert g_max <= max_factor * c_max, (g_max, c_max)


@pytest.mark.parametrize("act", list(ACTS))
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("mode", ["1", "grouped"])
def test_stacked_nf4_matches_reference(act, use_lora, T, mode, monkeypatch):
    if mode == "grouped" and not MU._check_torch_grouped_mm_supported():
        pytest.skip("no torch._grouped_mm on this GPU")
    if mode == "grouped" and DT != torch.bfloat16:
        pytest.skip("torch._grouped_mm takes bf16 only")
    ex = _make(act)
    lora = _lora(ex) if use_lora else None
    _stash(ex, lora)
    gots, curs, refs = [], [], []
    for seed in SEEDS:
        x, idx, w = _route(T, seed = seed)
        refs.append(_ref(ex, act, x, idx, w, lora))
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", mode)
        with torch.no_grad():
            got = MR.routed_moe_forward(ex, x, idx, w)
            assert got is not None
            assert got.shape == x.shape and got.dtype == x.dtype
            # The hook in forward_moe_backend_bnb4bit takes the routed path.
            assert torch.equal(forward_moe_backend_bnb4bit(ex, x, idx, w), got)
            curs.append(_current(ex, x, idx, w, monkeypatch))
        gots.append(got)
    # The grouped comparator rounds gate_up and down to the compute dtype like the current path,
    # in a different order, so its worst element may land a rounding step further.
    _check_vs_current(gots, curs, refs, 2.0 if mode == "grouped" else 1.5)


@pytest.mark.parametrize("act", ["silu", "gptoss"])
def test_bounds_reject_a_one_ulp_scale_error(act, monkeypatch):
    ex = _make(act)
    gots, curs, refs = [], [], []
    for seed in SEEDS:
        x, idx, w = _route(4, seed = seed)
        refs.append(_ref(ex, act, x, idx, w))
        with torch.no_grad():
            gots.append(MR.routed_moe_forward(ex, x, idx, w))
            curs.append(_current(ex, x, idx, w, monkeypatch))
    _check_vs_current(gots, curs, refs)
    with pytest.raises(AssertionError):
        _check_vs_current([(g.double() * (1 + 2 ** -7)).to(g.dtype) for g in gots], curs, refs)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("blocksize", [64, 128])
def test_quant_state_variants(nested, blocksize, monkeypatch):
    ex = _make("silu", nested = nested, blocksize = blocksize, i = 256)
    gots, curs, refs = [], [], []
    for seed in SEEDS:
        x, idx, w = _route(3, seed = seed)
        refs.append(_ref(ex, "silu", x, idx, w))
        with torch.no_grad():
            gots.append(MR.routed_moe_forward(ex, x, idx, w))
            curs.append(_current(ex, x, idx, w, monkeypatch))
    _check_vs_current(gots, curs, refs)


def test_selective_dequant_is_bit_exact_to_bitsandbytes():
    ex = _make("silu")
    state = MR.prepare_stacked_nf4(ex)
    for name, p in (("gate_up", ex.gate_up_proj), ("down", ex.down_proj)):
        full = bnb.functional.dequantize_4bit(p.data, p.quant_state).view(p._original_shape)
        uniq = torch.tensor([5, -1, 0, 7, 2, -1], device = DEV)
        got = MR.nf4_select_dequant(state[name], uniq, DT)
        for g, e in enumerate(uniq.tolist()):
            if e >= 0:
                assert torch.equal(got[g], full[e]), (name, e)


@pytest.mark.parametrize("act", ["silu", "gelu_tanh", "silu_interleaved", "gptoss"])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("T", [1, 4])
def test_bf16_experts_match_reference(act, use_lora, T, monkeypatch):
    ex = _make(act, quant = False)
    lora = _lora(ex) if use_lora else None
    _stash(ex, lora)
    calls = []
    real = MR.routed_bf16_moe
    monkeypatch.setattr(MR, "routed_bf16_moe", lambda *a, **k: calls.append(1) or real(*a, **k))
    gots, curs, refs = [], [], []
    for seed in SEEDS:
        x, idx, w = _route(T, seed = seed)
        refs.append(_ref(ex, act, x, idx, w, lora))
        with torch.no_grad():
            gots.append(MU.forward_moe_backend(ex, x, idx, w))
            assert calls, "the BF16 hook in forward_moe_backend did not run"
            curs.append(_current(ex, x, idx, w, monkeypatch))
    _check_vs_current(gots, curs, refs)


def test_kill_switch_and_grad_fallback(monkeypatch):
    ex = _make("silu")
    x, idx, w = _route(2)
    calls = []
    real = MR._nf4_routed
    monkeypatch.setattr(MR, "_nf4_routed", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "0")
    with torch.no_grad():
        assert MR.routed_moe_forward(ex, x, idx, w) is None
        off = forward_moe_backend_bnb4bit(ex, x, idx, w)
    assert not calls
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1")
    with torch.no_grad():
        on = forward_moe_backend_bnb4bit(ex, x, idx, w)
    assert calls == [1] and not torch.equal(on, off)
    # Grad enabled (training): the routed path never runs.
    with torch.enable_grad():
        assert MR.routed_moe_forward(ex, x, idx, w) is None
        assert torch.equal(forward_moe_backend_bnb4bit(ex, x, idx, w), off)
    assert calls == [1]


def test_ineligible_calls_fall_back(monkeypatch):
    with torch.no_grad():
        # Too many slots for a decode call.
        ex = _make("silu")
        x, idx, w = _route(MR.nf4_slot_limit(E) // 4 + 1)
        assert MR.routed_moe_forward(ex, x, idx, w) is None
        x, idx, w = _route(2)
        assert MR.routed_moe_forward(ex, x.cpu(), idx.cpu(), w.cpu()) is None
        # FP4 (not NF4) storage.
        ex = _make("silu")
        ex.gate_up_proj = _q4(_weights(0)[0], quant_type = "fp4")
        assert MR.routed_moe_forward(ex, x, idx, w) is None
        # Rows not a whole number of absmax blocks.
        ex = _make("silu", h = 96, i = 64)
        x96, idx96, w96 = _route(2, h = 96)
        assert MR.routed_moe_forward(ex, x96, idx96, w96) is None
        # An activation the kernels do not implement, an own _apply_gate, a transposed layout.
        ex = _make("silu")
        ex.act_fn = nn.ReLU()
        assert MR.routed_moe_forward(ex, x, idx, w) is None
        assert any("ReLU" in reason for _, reason in MR._DECLINED)  # census of skipped families
        ex = _make("silu")
        type(ex)._unsloth_own_apply_gate = True
        try:
            assert MR.routed_moe_forward(ex, x, idx, w) is None
        finally:
            del type(ex)._unsloth_own_apply_gate
        ex = _make("silu")
        ex.is_transposed = True
        assert MR.routed_moe_forward(ex, x, idx, w) is None
        # The fallback still computes the experts output.
        assert forward_moe_backend_bnb4bit(_make("silu"), x, idx, w) is not None


def test_moved_weights_rebuild_the_tables():
    ex = _make("silu")
    x, idx, w = _route(2)
    with torch.no_grad():
        a = MR.routed_moe_forward(ex, x, idx, w)
        other = _make("silu", seed = 11)
        ex.gate_up_proj, ex.down_proj = other.gate_up_proj, other.down_proj
        b = MR.routed_moe_forward(ex, x, idx, w)
        c = MR.routed_moe_forward(other, x, idx, w)
    assert not torch.equal(a, b) and torch.equal(b, c)


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("mode", ["1", "grouped", "bf16"])
def test_fullgraph_compile_and_cuda_graph_replay(use_lora, mode, monkeypatch):
    if mode == "grouped" and (not MU._check_torch_grouped_mm_supported() or DT != torch.bfloat16):
        pytest.skip("torch._grouped_mm: bf16 on a supported GPU only")
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1" if mode == "bf16" else mode)
    ex = _make("gelu_tanh", quant = mode != "bf16")
    _stash(ex, _lora(ex) if use_lora else None)
    x, idx, w = _route(4)

    def f(x, idx, w):
        return MR.routed_moe_forward(ex, x, idx, w)

    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    with torch.no_grad():
        eager = f(x, idx, w)  # builds the tables eagerly
        assert eager is not None
        compiled = torch.compile(f, fullgraph = True)(x, idx, w)
    assert sum(counters["graph_break"].values()) == 0
    assert compiled is not None
    if use_lora:
        torch.testing.assert_close(compiled, eager, rtol = 2e-2, atol = 2e-3)
    else:
        assert torch.equal(compiled, eager)

    # CUDA graph: capture once, replay with new tokens and new routes.
    sx, sidx, sw = x.clone(), idx.clone(), w.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(2):
            f(sx, sidx, sw)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        out = f(sx, sidx, sw)
    for seed in (5, 6, 7):
        x2, idx2, w2 = _route(4, seed = seed)
        sx.copy_(x2)
        sidx.copy_(idx2)
        sw.copy_(w2)
        graph.replay()
        with torch.no_grad():
            want = f(x2, idx2, w2)
        assert torch.equal(out, want), seed


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "needs two GPUs")
@pytest.mark.parametrize("quant", [True, False])
@pytest.mark.parametrize("mode", ["1", "grouped"])
def test_launches_on_the_tensors_device(quant, mode, monkeypatch):
    # A multi-GPU device_map puts layers on a non-current GPU; every launch must follow the tensors.
    if mode == "grouped" and (not quant or DT != torch.bfloat16):
        pytest.skip("grouped comparator: NF4 experts and bf16 (torch._grouped_mm) only")
    ex = _make("silu", quant = quant)
    for name in ("gate_up_proj", "down_proj"):
        p = getattr(ex, name)
        moved = p.to("cuda:1")
        if quant:
            moved._original_shape = p._original_shape
        setattr(ex, name, moved if isinstance(moved, nn.Parameter) else nn.Parameter(moved, requires_grad = False))
    assert getattr(ex.gate_up_proj, "quant_state", None) is None or ex.gate_up_proj.quant_state.absmax.device.index == 1
    lora = tuple(tuple(t.to("cuda:1") if isinstance(t, torch.Tensor) else t for t in part) for part in _lora(ex))
    _stash(ex, lora)
    gots, curs, refs = [], [], []
    for seed in SEEDS:
        x, idx, w = (t.to("cuda:1") for t in _route(4, seed = seed))
        refs.append(_ref(ex, "silu", x, idx, w, lora))
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", mode)
        with torch.no_grad(), torch.cuda.device(0):
            got = MR.routed_moe_forward(ex, x, idx, w)
        assert got is not None and got.device.index == 1
        gots.append(got)
        with torch.no_grad(), torch.cuda.device(1):
            curs.append(_current(ex, x, idx, w, monkeypatch))
    _check_vs_current(gots, curs, refs, 2.0 if mode == "grouped" else 1.5)


# Real transformers experts classes, configs shrunk to kernel-friendly sizes.
def _real_experts(kind):
    if kind == "qwen3_moe":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as C
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts as X
        cfg = C(hidden_size = H, moe_intermediate_size = I, num_experts = 16, num_experts_per_tok = 4, hidden_act = "silu")
    elif kind == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as C
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts as X
        cfg = C(hidden_size = H, moe_intermediate_size = I, num_experts = 16, num_experts_per_tok = 8, hidden_act = "silu")
    else:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig as C
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts as X
        cfg = C(hidden_size = H, moe_intermediate_size = I, intermediate_size = 2 * I, num_experts = 16,
                top_k_experts = 8, enable_moe_block = True, hidden_activation = "gelu_pytorch_tanh")
    return X(cfg), cfg


@pytest.mark.parametrize("kind", ["qwen3_moe", "qwen3_5_moe", "gemma4"])
@pytest.mark.parametrize("quant", [True, False])
@pytest.mark.parametrize("B", [1, 4])
def test_real_experts_classes(kind, quant, B, monkeypatch):
    try:
        ex, cfg = _real_experts(kind)
    except Exception as exc:  # pragma: no cover - older transformers
        pytest.skip(f"{kind}: {exc}")
    e = ex.gate_up_proj.shape[0]
    top_k = getattr(cfg, "num_experts_per_tok", None) or cfg.top_k_experts
    gu, dn = _weights(0, e = e)
    if quant:
        ex.gate_up_proj, ex.down_proj = _q4(gu), _q4(dn)
    else:
        ex.gate_up_proj = nn.Parameter(gu.to(DT).to(DEV), requires_grad = False)
        ex.down_proj = nn.Parameter(dn.to(DT).to(DEV), requires_grad = False)
    ex = ex.to(DEV).eval()
    act = "gelu_tanh" if kind == "gemma4" else "silu"
    assert MR._act_code(ex) == ACTS[act][1]
    gots, curs, refs = [], [], []
    for seed in SEEDS:
        x, idx, w = _route(B, top_k = top_k, e = e, seed = seed)
        refs.append(_ref(ex, act, x, idx, w))
        with torch.no_grad():
            gots.append(MU.forward_moe_backend(ex, x, idx, w))
            curs.append(_current(ex, x, idx, w, monkeypatch))
            assert MR.routed_moe_forward(ex, x, idx, w) is not None
    _check_vs_current(gots, curs, refs)


# Whole tiny causal LMs (real transformers classes, kernel-friendly shrunk configs, random init),
# experts stored as the bitsandbytes loader stores them: one stacked NF4 Params4bit per projection.
def _tiny_config(kind):
    common = dict(vocab_size = 512, hidden_size = 256, moe_intermediate_size = 192, num_hidden_layers = 2,
                  num_attention_heads = 4, num_key_value_heads = 2, head_dim = 64, num_experts = 16,
                  max_position_embeddings = 256)
    if kind == "qwen3_moe":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as C
        return C(intermediate_size = 512, num_experts_per_tok = 4, **common)
    if kind == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as C
        return C(shared_expert_intermediate_size = 192, num_experts_per_tok = 4,
                 layer_types = ["linear_attention", "full_attention"], linear_num_key_heads = 2,
                 linear_num_value_heads = 4, linear_key_head_dim = 32, linear_value_head_dim = 32, **common)
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig as C
    common.pop("num_experts")
    return C(intermediate_size = 512, global_head_dim = 64, num_experts = 16, top_k_experts = 4,
             enable_moe_block = True, layer_types = ["sliding_attention", "full_attention"], sliding_window = 64,
             hidden_size_per_layer_input = 0, vocab_size_per_layer_input = 512,
             hidden_activation = "gelu_pytorch_tanh", **common)


def _tiny_model(kind):
    from transformers import AutoModelForCausalLM
    from unsloth_zoo.temporary_patches.moe_experts_interface import patch_experts_interface
    patch_experts_interface()
    cfg = _tiny_config(kind)
    cfg._experts_implementation = "unsloth"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype = DT).to(DEV).eval()
    # Random-init routers are near uniform, so a last-bit difference in one layer's output flips a
    # near-tie top-k pick in the next and the comparison would measure route flips, not kernels.
    # Decisive routers ([num_experts, hidden] weights scaled up) keep both arms on the same routes.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2 and tuple(p.shape) == (16, cfg.hidden_size):
                p.mul_(30)
    # fp32 twin: same weights, experts as dense fp32 stacks of the NF4 weights' bnb dequant.
    import copy
    ref = copy.deepcopy(model).float()
    experts, ref_experts = [], []
    for module, twin in zip(model.modules(), ref.modules()):
        gu = module._parameters.get("gate_up_proj") if hasattr(module, "_parameters") else None
        if gu is None or gu.dim() != 3:
            continue
        g = torch.Generator().manual_seed(len(experts))
        module.gate_up_proj = _q4(torch.randn(gu.shape, generator = g) * 0.05)
        module.down_proj = _q4(torch.randn(module.down_proj.shape, generator = g) * 0.05)
        twin.gate_up_proj = nn.Parameter(_dense(module.gate_up_proj).float(), requires_grad = False)
        twin.down_proj = nn.Parameter(_dense(module.down_proj).float(), requires_grad = False)
        experts.append(module)
        ref_experts.append(twin)
    if not experts:
        pytest.skip(f"{kind}: no 3D experts stacks in this transformers version")
    return model, experts, ref, ref_experts


@pytest.mark.parametrize("kind", ["qwen3_moe", "qwen3_5_moe", "gemma4"])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("B", [1, 4])
def test_tiny_causal_lm_decode_routed_vs_current(kind, use_lora, B, monkeypatch):
    try:
        model, experts, ref_model, ref_experts = _tiny_model(kind)
    except (ImportError, LookupError, TypeError) as exc:  # pragma: no cover - older transformers
        pytest.skip(f"{kind}: {exc}")
    for ex, twin in zip(experts, ref_experts):
        lora = _lora(ex) if use_lora else None
        _stash(ex, lora)
        _stash(twin, lora)
    calls = {"routed": 0, "grouped": 0}
    real_r, real_g = MR._nf4_routed, MR._nf4_grouped

    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapped
    monkeypatch.setattr(MR, "_nf4_routed", count("routed", real_r))
    monkeypatch.setattr(MR, "_nf4_grouped", count("grouped", real_g))

    g = torch.Generator().manual_seed(B)
    prompt = torch.randint(0, 512, (B, 40), generator = g).to(DEV)  # prefill: > nf4_slot_limit(16)
    steps = torch.randint(0, 512, (B, 4), generator = g).to(DEV)    # teacher-forced decode tokens

    def run(mode, model = model):
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", mode)
        calls["routed"] = calls["grouped"] = 0
        with torch.no_grad():
            out = model(prompt, use_cache = True)
            assert calls["routed"] == calls["grouped"] == 0  # prefill keeps the current path
            cache, logits = out.past_key_values, []
            for s in range(steps.shape[1]):
                out = model(steps[:, s:s + 1], past_key_values = cache, use_cache = True)
                cache = out.past_key_values
                logits.append(out.logits[:, -1].float())
        return torch.stack(logits), dict(calls)

    cur, n0 = run("0")
    assert n0 == {"routed": 0, "grouped": 0}
    got, n1 = run("1")
    want = len(experts) * steps.shape[1]
    assert n1 == {"routed": want, "grouped": 0}, n1
    modes = [("1", got)]
    if MU._check_torch_grouped_mm_supported() and DT == torch.bfloat16:
        grp, n2 = run("grouped")
        assert n2 == {"routed": 0, "grouped": want}, n2
        modes.append(("grouped", grp))
    ref, _ = run("0", ref_model)
    # Model level, a gross-error tripwire: switching the expert kernels moves the logits about as
    # much as two bf16 runs as accurate as the current one can differ, |a - c| <= |a - ref| +
    # |c - ref|. Routes are identical across arms here, but the decisive routers' softmax turns a
    # last-bit hidden-state change into a visible routing-weight change in the next layer
    # (measured max 2.2x on one cell), so the max gets headroom; the call-level check below is
    # the strict accuracy gate.
    e_cur = (cur - ref).abs()
    for mode, logits in modes:
        assert (logits - cur).abs().max() <= 3 * e_cur.max(), mode
        assert (logits - cur).abs().mean() <= 2 * e_cur.mean(), mode
        # About as close to the fp32 twin as the current path, row by row (tripwire: rows swing by
        # up to 2.3e-3 either way between equally accurate arms).
        cos = F.cosine_similarity(logits.flatten(1), ref.flatten(1), dim = -1)
        cos_cur = F.cosine_similarity(cur.flatten(1), ref.flatten(1), dim = -1)
        assert (cos >= cos_cur - 5e-3).all(), (mode, cos.tolist(), cos_cur.tolist())

    # Call level: every decode-step experts call the model made, replayed against the fp64
    # reference: the routed kernels are no less accurate than the path they replace.
    seen = []
    real_fwd = MR.routed_moe_forward

    def spy(ex, h, i, w):
        seen.append((ex, h.clone(), i.clone(), w.clone()))
        return real_fwd(ex, h, i, w)
    monkeypatch.setattr(MR, "routed_moe_forward", spy)
    run("1")
    monkeypatch.setattr(MR, "routed_moe_forward", real_fwd)
    seen = [c for c in seen if c[2].numel() <= MR.nf4_slot_limit(16)]
    assert len(seen) == want
    act = "gelu_tanh" if kind == "gemma4" else "silu"
    gots, curs, refs = [], [], []
    for ex, h, i, w in seen:
        lora = None
        if use_lora:
            lora = (getattr(ex, MU.moe_lora_stash_name("gate_up_proj")), getattr(ex, MU.moe_lora_stash_name("down_proj")))
        x = h.reshape(-1, h.shape[-1])
        reference = _ref(ex, act, x, i, w, lora)
        monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", "1")
        with torch.no_grad():
            gots.append(MR.routed_moe_forward(ex, x, i, w))
            curs.append(_current(ex, x, i, w, monkeypatch))
        refs.append(reference)
    _check_vs_current(gots, curs, refs)


class _ScaledSiLU(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.silu(x * self.scale) / self.scale


def test_activation_verdict_is_per_module_not_per_class():
    # One activation class, two parameterisations: only the one equal to SiLU may route.
    with torch.no_grad():
        x, idx, w = _route(2)
        silu = _make("silu")
        silu.act_fn = _ScaledSiLU(1.0)
        other = _make("silu")
        other.act_fn = _ScaledSiLU(2.0)
        assert MR.routed_moe_forward(silu, x, idx, w) is not None
        assert MR.routed_moe_forward(other, x, idx, w) is None
        bf = _make("silu", quant = False)
        bf.act_fn = _ScaledSiLU(2.0)
        assert MR.routed_moe_forward(bf, x, idx, w) is None


def test_nf4_slot_limit_scales_with_the_expert_count(monkeypatch):
    # Production rule: route while slots <= experts (measured crossover 1.2E to 1.5E on B200).
    monkeypatch.setattr(MR, "NF4_SLOTS_PER_EXPERT", 1.0)
    for e in (8, 16):
        ex = _make("silu", e = e)
        with torch.no_grad():
            x, idx, w = _route(e // 4, e = e)         # P = E
            assert MR.routed_moe_forward(ex, x, idx, w) is not None
            x, idx, w = _route(e // 4 + 1, e = e)     # P = E + 4
            assert MR.routed_moe_forward(ex, x, idx, w) is None
    monkeypatch.setattr(MR, "NF4_MAX_SLOTS", 4)       # UNSLOTH_MOE_ROUTED_MAX_SLOTS: absolute
    with torch.no_grad():
        x, idx, w = _route(2, e = 16)
        assert MR.routed_moe_forward(_make("silu", e = 16), x, idx, w) is None


def test_gpt_oss_compiled_decode_reads_persistent_lora_stacks_and_follows_a_train_step():
    pytest.importorskip("peft")
    pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
    from test_gpt_oss_routed_guards import _NF4MLP
    from test_gpt_oss_routed_nf4 import H as GH, _Experts, _lora_wrap, _routing
    from unsloth_zoo.temporary_patches import gpt_oss_routed as GR
    torch.manual_seed(0)
    mlp = _NF4MLP(_lora_wrap(_Experts(True)).eval()).eval()
    decode = torch.randn(2, 1, GH, device = DEV, dtype = torch.float32)
    prefill = torch.randn(1, 64, GH, device = DEV, dtype = torch.float32)  # > ROUTED_MAX_SLOTS
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm.code)
        return gm.forward
    with torch.no_grad():
        before = GR.routed_mlp_forward(mlp, decode)
        cache = mlp.experts.gate_up_projs._unsloth_routed_lora
        ptrs = tuple(t.data_ptr() for t in cache["stacked"])
        torch._dynamo.reset()
        compiled = torch.compile(lambda h: GR.routed_mlp_forward(mlp, h), backend = backend, fullgraph = True)
        torch.testing.assert_close(compiled(decode), before, rtol = 1e-5, atol = 1e-5)
    assert graphs and not any("torch.stack(" in code or "aten.stack" in code for code in graphs), "compiled step restacks the adapters"

    # One training step on the adapters (dense path, grad on: the routed path stays out of it).
    params = [p for n, p in mlp.named_parameters() if "lora_" in n]
    opt = torch.optim.SGD(params, lr = 5.0)
    idx, w = _routing(4, seed = 9)
    x = torch.randn(1, 4, GH, device = DEV, dtype = torch.float32)
    mlp.experts.dense(x, idx, w).float().pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    opt.step()

    with torch.no_grad():
        # The eager prefill (too large to route) refreshes the stacks in place ...
        assert GR.routed_mlp_forward(mlp, prefill) is None
        assert tuple(t.data_ptr() for t in cache["stacked"]) == ptrs
        # ... so the compiled decode step (no recompile) follows the new adapter.
        got = compiled(decode)
        want = GR.routed_mlp_forward(mlp, decode)
    assert len(graphs) == 1
    assert not torch.allclose(want, before, rtol = 1e-3, atol = 1e-3)
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("mode", ["1", "grouped"])
def test_stacked_biases_are_read_live(mode, monkeypatch):
    # The stacked layout reads the live [E, N] bias tensors (no copy): in-place updates and
    # load_state_dict are seen without a rebuild, and a swapped tensor rebuilds the table.
    if mode == "grouped" and (DT != torch.bfloat16 or not MU._check_torch_grouped_mm_supported()):
        pytest.skip("torch._grouped_mm: bf16 on a supported GPU only")
    monkeypatch.setenv("UNSLOTH_MOE_ROUTED_KERNEL", mode)
    ex = _make("gptoss")
    x, idx, w = _route(4)

    def check():
        with torch.no_grad():
            got = MR.routed_moe_forward(ex, x, idx, w)
        ref = _ref(ex, "gptoss", x, idx, w)
        err = (got.double() - ref).abs().max().item()
        assert err <= 2 ** -6 * ref.abs().max().item(), err
        return got

    first = check()
    state = ex.__dict__["_unsloth_routed_moe"]
    with torch.no_grad():
        ex.gate_up_proj_bias.add_(0.25)
        ex.down_proj_bias.mul_(-1)
    second = check()
    assert ex.__dict__["_unsloth_routed_moe"] is state and not torch.allclose(first, second)
    sd = {k: v.clone() for k, v in ex.state_dict().items() if k.endswith("_bias")}
    sd["down_proj_bias"].add_(0.5)
    ex.load_state_dict(sd, strict = False)
    third = check()
    assert ex.__dict__["_unsloth_routed_moe"] is state and not torch.allclose(second, third)
    ex.down_proj_bias = nn.Parameter(ex.down_proj_bias.detach().clone() * 2, requires_grad = False)
    check()
    assert ex.__dict__["_unsloth_routed_moe"] is not state
