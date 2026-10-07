"""Expert stack layout on the transformers<5 ModuleList grouped MoE path.

The stack builders return the transposed VIEW of the contiguous [E, N, K] dequantized stack
(gate_up keeps the per-expert cat(gate, up) interleave), the GEMMs take that view uncopied, and
_GroupedFrozenMM's backward reads its transpose, the contiguous stack, with no copy. Checked
bitwise against the copy-based layout (every GEMM weight made contiguous, as the builders did):
output, dX, router grad and every LoRA grad, on torch._grouped_mm and the Triton generic kernel,
bf16 / fp16, NF4 / bf16 base, uniform and skewed routing with empty experts, every base policy and
non-reentrant checkpointing. Also: fullgraph compile, and the readiness-cache key fields.
"""
import contextlib
import os

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

import test_moe_grouped_modulelist_lora as L  # noqa: E402  (pytest puts tests/ on sys.path)

pytestmark = pytest.mark.skipif(
    not (L.DEV == "cuda" and ML._grouped_mm_supported()),
    reason = "needs CUDA torch._grouped_mm",
)


@contextlib.contextmanager
def _dtype(dt):
    old = L.DT
    L.DT = dt
    try:
        yield
    finally:
        L.DT = old


@contextlib.contextmanager
def _copy_layout(monkeypatch):
    """The previous layout: contiguous [E, K, N] stacks, and every grouped GEMM weight copied."""
    fix, gu, dn = ML._grouped_mm_fix, ML._build_gate_up_stack, ML._build_down_stack
    with monkeypatch.context() as m:
        m.setattr(ML, "_build_gate_up_stack", lambda *a: gu(*a).contiguous())
        m.setattr(ML, "_build_down_stack", lambda *a: dn(*a).contiguous())
        m.setattr(ML, "_grouped_mm_fix", lambda x, w, offs: fix(x, w.contiguous(), offs))
        yield


def _block(base, dt, lora, policy, routing, seed = 0):
    with _dtype(dt):
        model, blk = L.build("qwen3", base = base, r = 8, seed = seed)
    if not lora:
        for ex in blk.experts:
            for n in ("gate_proj", "up_proj", "down_proj"):
                setattr(ex, n, getattr(ex, n).base_layer)
    else:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
    blk.gate.weight.requires_grad_(True)
    if routing == "skewed":
        with torch.no_grad():
            blk.gate.weight[5:] = -1.0     # positive inputs: experts 5.. never routed (empty groups)
            blk.gate.weight[0] += 0.05     # expert 0 takes almost every token
    n = ML.enable_grouped_moe(model, verbose = False, recompute = policy == "recompute", cache = policy == "cache")
    assert n == 1, ML.LAST_DECLINE
    blk.train()
    return model, blk


def _step(blk, x, gout, ckpt):
    blk.zero_grad(set_to_none = True)
    x = x.detach().clone().requires_grad_(True)
    if ckpt:
        from torch.utils.checkpoint import checkpoint
        out = checkpoint(lambda t: blk.forward(t)[0], x, use_reentrant = False)
    else:
        out = blk.forward(x)[0]
    (out.float() * gout).sum().backward()
    grads = {n: p.grad.clone() for n, p in blk.named_parameters() if p.grad is not None}
    return out.detach(), x.grad.clone(), grads


def _assert_bitwise(new, ref):
    (o, dx, g), (ro, rdx, rg) = new, ref
    assert torch.equal(o, ro), "output"
    assert torch.equal(dx, rdx), "dX"
    assert set(g) == set(rg) and "gate.weight" in g
    for k in rg:
        assert torch.equal(g[k], rg[k]), k


def _inputs(dt, routing, seed = 1):
    torch.manual_seed(seed)
    x = torch.randn(1, L.T, L.H, device = "cuda", dtype = dt)
    if routing == "skewed":
        x = x.abs()
    gout = torch.randn(1, L.T, L.H, device = "cuda", dtype = torch.float32)
    return x, gout


def _backend(monkeypatch, backend):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1" if backend == "triton" else "0")
    if backend == "triton":
        from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg
        if not mg.triton_grouped_available(torch.device("cuda", torch.cuda.current_device())) \
                or torch.cuda.get_device_capability() < (8, 0):
            pytest.skip("Triton generic grouped GEMM unavailable")
        return mg.GENERIC_CALLS
    return None


@pytest.mark.parametrize("routing", ["uniform", "skewed"])
@pytest.mark.parametrize("policy", ["recompute", "pinned", "cache"])
@pytest.mark.parametrize("base", ["nf4", "bf16"])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids = ["bf16", "fp16"])
@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("lora", [True, False], ids = ["lora", "frozen"])
def test_view_layout_bitwise_equals_copy_layout(lora, backend, dt, base, policy, routing, monkeypatch):
    calls = _backend(monkeypatch, backend)
    model, blk = _block(base, dt, lora, policy, routing)
    x, gout = _inputs(dt, routing)
    if routing == "skewed":
        with torch.no_grad():
            sel = torch.topk(blk.gate(x.view(-1, L.H)), L.TOPK, dim = -1).indices
        counts = torch.bincount(sel.flatten(), minlength = L.E)
        assert (counts == 0).sum() >= 3 and counts.max() >= L.T * 0.9
    before = (ML.CALLS["grouped"], dict(calls) if calls is not None else None)
    new = _step(blk, x, gout, ckpt = False)
    assert ML.CALLS["grouped"] == before[0] + 1
    if calls is not None:
        assert calls["gemm"] > before[1]["gemm"], "Triton generic GEMM not engaged"
    blk.__dict__.pop("_cached_gate_up", None)
    blk.__dict__.pop("_cached_down", None)
    with _copy_layout(monkeypatch):
        ref = _step(blk, x, gout, ckpt = False)
    _assert_bitwise(new, ref)


@pytest.mark.parametrize("policy", ["recompute", "pinned"])
@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("base", ["nf4", "bf16"])
def test_view_layout_bitwise_under_nonreentrant_checkpoint(base, backend, policy, monkeypatch):
    _backend(monkeypatch, backend)
    model, blk = _block(base, torch.bfloat16, True, policy, "uniform")
    x, gout = _inputs(torch.bfloat16, "uniform")
    new = _step(blk, x, gout, ckpt = True)
    with _copy_layout(monkeypatch):
        ref = _step(blk, x, gout, ckpt = True)
    _assert_bitwise(new, ref)
    _assert_bitwise(new, _step(blk, x, gout, ckpt = False))


@pytest.mark.parametrize("base", ["nf4", "bf16"])
def test_builders_return_transposed_views(base):
    """A revert to .contiguous() in any builder fails here: the stack is the [E, N, K] storage."""
    model, blk = _block(base, torch.bfloat16, False, "recompute", "uniform")
    spec = blk._unsloth_moe_spec
    builders = [ML._build_gate_up_stack, ML._build_down_stack, ML._bnb_build_gate_up_stack, ML._bnb_build_down_stack]
    if base == "nf4":
        builders += [ML._nf4_build_gate_up_stack, ML._nf4_build_down_stack]
    for build in builders:
        w = build(blk.experts, spec, torch.bfloat16)
        assert w is not None, build.__name__
        assert not w.is_contiguous() and w.transpose(1, 2).is_contiguous(), build.__name__
    gu = ML._build_gate_up_stack(blk.experts, spec, torch.bfloat16)
    assert gu.shape == (L.E, L.H, 2 * L.I)
    # per expert cat(gate, up) along the output dim
    for e in (0, L.E - 1):
        g = ML._expert_weight(blk.experts[e].gate_proj, torch.bfloat16)
        u = ML._expert_weight(blk.experts[e].up_proj, torch.bfloat16)
        assert torch.equal(gu[e].t(), torch.cat((g, u), 0))


def test_gemms_take_views_and_backward_reads_the_stack(monkeypatch):
    """torch._grouped_mm gets the uncopied view in forward and the contiguous stack in backward;
    the copy fallback still applies when the view probe says no."""
    from unsloth_zoo.temporary_patches import moe_utils
    if not moe_utils._transposed_view_grouped_mm_is_safe():
        pytest.skip("view probe failed on this device: the copy is kept by design")
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    model, blk = _block("nf4", torch.bfloat16, False, "recompute", "uniform")
    x, gout = _inputs(torch.bfloat16, "uniform")
    seen, real = [], torch._grouped_mm

    def spy(a, w, offs = None, **kw):
        if w.dim() == 3 and w.shape[0] == L.E:
            seen.append((tuple(w.shape), w.is_contiguous()))
        return real(a, w, offs = offs, **kw)

    monkeypatch.setattr(torch, "_grouped_mm", spy)
    _step(blk, x, gout, ckpt = False)
    assert seen == [((L.E, L.H, 2 * L.I), False), ((L.E, L.I, L.H), False),
                    ((L.E, L.H, L.I), True), ((L.E, 2 * L.I, L.H), True)], seen
    seen.clear()
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", False)
    _step(blk, x, gout, ckpt = False)
    assert all(c for _, c in seen) and len(seen) == 4, seen


def test_unaligned_view_keeps_the_copy():
    """A view whose stride is not 16-byte aligned is copied first, not handed to the loop fallback."""
    w = torch.randn(3, 20, 12, device = "cuda", dtype = torch.bfloat16)   # [E, N, K]: K * 2 = 24 bytes
    v = w.transpose(1, 2)
    assert not ML._view_weight_ok(v)
    x = torch.randn(9, 12, device = "cuda", dtype = torch.bfloat16)
    offs = torch.tensor([2, 2, 9], device = "cuda", dtype = torch.int32)
    assert torch.equal(ML._grouped_mm_fix(x, v, offs), ML._grouped_mm_fix(x, v.contiguous(), offs))


@pytest.mark.parametrize("lora", [True, False], ids = ["lora", "frozen"])
@pytest.mark.parametrize("policy", ["recompute", "pinned"])
def test_compile_fullgraph_no_graph_breaks(lora, policy, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    model, blk = _block("bf16", torch.bfloat16, lora, policy, "uniform")
    x, gout = _inputs(torch.bfloat16, "uniform")
    torch._dynamo.reset()
    explain = torch._dynamo.explain(blk.forward)(x)
    assert explain.graph_break_count == 0, explain.break_reasons
    torch._dynamo.reset()
    before = ML.CALLS["grouped"]
    compiled = torch.compile(blk.forward, fullgraph = True)
    xx = x.detach().clone().requires_grad_(True)
    out = compiled(xx)[0]
    (out.float() * gout).sum().backward()
    assert ML.CALLS["grouped"] == before + 1
    ref = _step(blk, x, gout, ckpt = False)
    assert (out.detach().float() - ref[0].float()).abs().max().item() < 5e-2


# ----------------------------------------------------------------------------- readiness cache
def _engages(blk, x):
    before = ML.CALLS["grouped"]
    with torch.no_grad():
        blk.forward(x)
    return ML.CALLS["grouped"] - before == 1


def test_signature_tracks_lora_rank_change_via_data():
    """`.data =` keeps the Parameter, so only the shape shows a new rank on one expert."""
    model, blk = L.build("qwen3")
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert _engages(blk, x)
    p = blk.experts[3].up_proj
    A, B = p.lora_A["default"].weight, p.lora_B["default"].weight
    A.data = torch.zeros(4, A.shape[1], device = A.device, dtype = A.dtype)
    B.data = torch.zeros(B.shape[0], 4, device = B.device, dtype = B.dtype)
    assert not _engages(blk, x)
    assert ML.LAST_DECLINE["reason"] == "up_proj LoRA: scaling / rank differ across experts"


def test_signature_tracks_base_weight_shape():
    model, blk = L.build("qwen3")
    spec = ML._BLOCK_SPECS["Qwen3MoeSparseMoeBlock"]
    key = ML._ready_signature(blk.experts, spec)[0]
    w = blk.experts[2].down_proj.base_layer.weight
    w.data = w.data[:, : L.I // 2].clone()
    assert ML._ready_signature(blk.experts, spec)[0] != key


def test_signature_tracks_lora_variant_entry():
    """Setting an existing variant entry from None to a variant keeps the dict's length."""
    model, blk = L.build("qwen3")
    for ex in blk.experts:
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(ex, n).lora_variant = {"default": None}
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert _engages(blk, x)
    state = lambda: ML._cached_state(blk, blk.experts, blk._unsloth_moe_spec, x.device, x.dtype)
    blk.experts[6].gate_proj.lora_variant["default"] = object()   # the loop would now call it
    assert state() == "gate_proj LoRA: DoRA / LoRA variant"
    blk.experts[6].gate_proj.lora_variant["default"] = None
    assert isinstance(state(), dict) and _engages(blk, x)
