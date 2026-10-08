"""Per-call host trimming of the transformers<5 ModuleList grouped MoE forward.

The per-expert projection lists are built once per readiness verdict (_block_projs) and must follow
a projection swap on any expert. UNSLOTH_MOE_FUSED_GATE_UP_LORA=1 runs the gate and up LoRA as one
A GEMM and one block-diagonal B GEMM: output and router grad stay bitwise, bf16 LoRA grads too; dX only differs by
rounding (one fp32 accumulation of the gate and up terms instead of two GEMMs and an add); on fp16
the torch backend's LoRA weight grads also round differently (within 1e-3 relative). The compiled
block keeps zero graph breaks.
"""
import pytest
import torch

from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

import test_moe_grouped_modulelist_lora as L  # noqa: E402  (pytest puts tests/ on sys.path)
import test_moe_grouped_modulelist_layout as LY  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (L.DEV == "cuda" and ML._grouped_mm_supported()),
    reason = "needs CUDA torch._grouped_mm",
)


def test_projection_lists_cached_and_follow_a_swap():
    model, blk = L.build("qwen3")
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert LY._engages(blk, x)
    first = blk.__dict__["_moe_projs"][3]
    assert LY._engages(blk, x)
    assert blk.__dict__["_moe_projs"][3] is first   # reused, not rebuilt
    assert first.gate[5] is blk.experts[5].gate_proj and first.gate_up[11] is blk.experts[5].up_proj
    # Replace one middle expert's down projection: the readiness entry moves, so the lists follow.
    old = blk.experts[4].down_proj
    new = type(old).__new__(type(old))
    new.__dict__.update(old.__dict__)
    blk.experts[4].down_proj = new
    assert LY._engages(blk, x)
    assert blk.__dict__["_moe_projs"][3].down[4] is new


def test_projection_lists_spot_check_without_readiness_change():
    """A readiness cache that kept its entry across a swap of an end expert still rebuilds."""
    model, blk = L.build("qwen3")
    L.enable(model, blk)
    x = torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT)
    assert LY._engages(blk, x)
    spec = blk._unsloth_moe_spec
    projs = ML._block_projs(blk, blk.experts, spec)
    blk.experts[-1]._modules["up_proj"] = blk.experts[0].up_proj   # entry untouched
    again = ML._block_projs(blk, blk.experts, spec)
    assert again is not projs and again.up[-1] is blk.experts[0].up_proj


@pytest.mark.parametrize("r", [4, 8, 16])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids = ["bf16", "fp16"])
@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("stacked", ["1", "0"])
def test_fused_gate_up_lora_matches_unfused(r, dt, backend, stacked, monkeypatch):
    LY._backend(monkeypatch, backend)
    monkeypatch.setenv("UNSLOTH_MOE_STACKED_LORA", stacked)
    with LY._dtype(dt):
        model, blk = L.build("qwen3", base = "nf4", r = r)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    blk.gate.weight.requires_grad_(True)
    assert ML.enable_grouped_moe(model, verbose = False, stack_lora = stacked == "1") == 1
    blk.train()
    x, gout = LY._inputs(dt, "uniform")
    monkeypatch.setenv("UNSLOTH_MOE_FUSED_GATE_UP_LORA", "0")
    ref = LY._step(blk, x, gout, ckpt = False)
    monkeypatch.setenv("UNSLOTH_MOE_FUSED_GATE_UP_LORA", "1")
    calls = []
    orig = ML._lora_delta_gate_up
    monkeypatch.setattr(ML, "_lora_delta_gate_up", lambda *a: calls.append(1) or orig(*a))
    new = LY._step(blk, x, gout, ckpt = False)
    assert calls, "fused path did not run"
    (o, dx, g), (ro, rdx, rg) = new, ref
    assert torch.equal(o, ro)
    assert set(g) == set(rg)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()
    for k in rg:
        if dt is torch.bfloat16:
            assert torch.equal(g[k], rg[k]), k
        else:   # fp16 grouped_mm's weight grad may pick another reduction for the wider B
            assert rel(g[k], rg[k]) < 1e-3, (k, rel(g[k], rg[k]))
    assert rel(dx, rdx) < 1e-2, rel(dx, rdx)


def test_fused_gate_up_lora_compile_fullgraph(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    monkeypatch.setenv("UNSLOTH_MOE_FUSED_GATE_UP_LORA", "1")
    model, blk = LY._block("bf16", torch.bfloat16, True, "recompute", "uniform")
    x, gout = LY._inputs(torch.bfloat16, "uniform")
    torch._dynamo.reset()
    explain = torch._dynamo.explain(blk.forward)(x)
    assert explain.graph_break_count == 0, explain.break_reasons
    torch._dynamo.reset()
