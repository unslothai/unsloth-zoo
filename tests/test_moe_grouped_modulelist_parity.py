"""Parity tests for the transformers<5 ModuleList grouped-GEMM MoE path
(unsloth_zoo/temporary_patches/moe_grouped_modulelist.py).

transformers>=5 stacks its MoE experts, so no shipped model exercises the ModuleList path
directly on new transformers. These tests instead build synthetic ModuleList SparseMoeBlocks
that mirror the real block structure (Qwen3-MoE / Mixtral / OLMoE) and check that
grouped_moe_forward matches a plain per-expert reference loop, in the default, cache and
recompute modes, plus forward+backward. Runs anywhere torch._grouped_mm is supported
(H100+/B200); otherwise skipped.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from unsloth_zoo.temporary_patches.moe_grouped_modulelist import (
    grouped_moe_forward,
    enable_grouped_moe,
    disable_grouped_moe,
    wrap_loader_for_grouped_moe,
    _block_is_eligible,
    _grouped_mm_supported,
    _route_softmax_topk,
    _BLOCK_SPECS,
)

DEV = "cuda"
DTYPE = torch.bfloat16
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _grouped_mm_supported()),
    reason="torch._grouped_mm unsupported on this device",
)


def _make_expert(hidden, inter, gname, uname, dname):
    ex = nn.Module()
    for name, lin in (
        (gname, nn.Linear(hidden, inter, bias=False)),
        (uname, nn.Linear(hidden, inter, bias=False)),
        (dname, nn.Linear(inter, hidden, bias=False)),
    ):
        lin.weight.requires_grad_(False)  # frozen base experts (the grouped path's precondition)
        setattr(ex, name, lin.to(DEV, DTYPE))
    ex.act_fn = F.silu
    return ex


def _make_block(cls_name, spec, hidden, inter, num_experts, top_k, norm_topk_prob):
    gname, uname, dname, _ = spec
    blk = type(cls_name, (nn.Module,), {})()
    blk.experts = nn.ModuleList(
        [_make_expert(hidden, inter, gname, uname, dname) for _ in range(num_experts)]
    )
    blk.gate = nn.Linear(hidden, num_experts, bias=False).to(DEV, DTYPE)
    blk.num_experts = num_experts
    blk.top_k = top_k
    blk.norm_topk_prob = norm_topk_prob
    blk.eval()
    blk._unsloth_moe_spec = spec
    blk._orig_moe_forward = lambda hs: _reference_forward(blk, hs, spec)
    return blk


def _reference_forward(blk, hidden_states, spec):
    """Plain per-expert loop matching grouped_moe_forward's math (fp32 combine)."""
    gname, uname, dname, router = spec
    is_3d = hidden_states.dim() == 3
    if is_3d:
        b, s, h = hidden_states.shape
        hs = hidden_states.reshape(-1, h)
    else:
        hs = hidden_states
        h = hs.shape[-1]
    logits = blk.gate(hs)
    rw, sel = router(blk, logits, blk.top_k)
    rw = rw.to(hs.dtype)
    T = hs.shape[0]
    final = torch.zeros((T, h), dtype=torch.float32, device=hs.device)
    for e, ex in enumerate(blk.experts):
        pos = (sel == e).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        tok, slot = pos[:, 0], pos[:, 1]
        xe = hs[tok]
        inter = F.silu(getattr(ex, gname)(xe)) * getattr(ex, uname)(xe)
        out = getattr(ex, dname)(inter) * rw[tok, slot].unsqueeze(-1)
        final.index_add_(0, tok, out.float())
    final = final.to(hs.dtype)
    return (final.reshape(b, s, h) if is_3d else final), logits


def _rel_l2(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


@pytest.mark.parametrize("cls_name,spec", list(_BLOCK_SPECS.items()))
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_forward_parity(cls_name, spec, norm_topk_prob):
    torch.manual_seed(0)
    hidden, inter, E, top_k, T = 128, 256, 8, 2, 96
    blk = _make_block(cls_name, spec, hidden, inter, E, top_k, norm_topk_prob)
    x = torch.randn(1, T, hidden, device=DEV, dtype=DTYPE)

    ref, _ = _reference_forward(blk, x, spec)
    for mode in ("default", "cache", "recompute"):
        blk._moe_cache = mode == "cache"
        blk._moe_recompute = mode == "recompute"
        for attr in ("_cached_gate_up", "_cached_down"):
            if hasattr(blk, attr):
                delattr(blk, attr)
        out, _ = grouped_moe_forward(blk, x)
        rel = _rel_l2(out, ref)
        assert out.shape == x.shape, f"{cls_name}/{mode}: shape {out.shape} != {x.shape}"
        assert rel < 1e-2, f"{cls_name}/{mode}: relL2 {rel:.2e} exceeds bf16 noise floor"


@pytest.mark.parametrize("cls_name,spec", list(_BLOCK_SPECS.items()))
def test_backward_parity(cls_name, spec):
    torch.manual_seed(0)
    hidden, inter, E, top_k, T = 128, 256, 8, 2, 64
    blk = _make_block(cls_name, spec, hidden, inter, E, top_k, True)

    x1 = torch.randn(1, T, hidden, device=DEV, dtype=DTYPE, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    (_reference_forward(blk, x1, spec)[0]).float().pow(2).sum().backward()
    blk._moe_recompute = True  # exercise the recompute autograd Function's dX
    (grouped_moe_forward(blk, x2)[0]).float().pow(2).sum().backward()
    rel = _rel_l2(x2.grad, x1.grad)
    assert rel < 1e-2, f"{cls_name}: input-grad relL2 {rel:.2e} exceeds bf16 noise floor"


def test_eligibility_and_shared_expert_bail():
    spec = _BLOCK_SPECS["Qwen3MoeSparseMoeBlock"]
    blk = _make_block("Qwen3MoeSparseMoeBlock", spec, 64, 128, 4, 2, True)
    assert _block_is_eligible(blk) is not None, "eligible ModuleList block not recognised"

    blk.shared_expert = nn.Linear(64, 64, bias=False).to(DEV, DTYPE)  # routed grouping cannot cover it
    assert _block_is_eligible(blk) is None, "block with a shared expert must bail to the original loop"

    unknown = _make_block("SomeUnknownMoeBlock", spec, 64, 128, 4, 2, True)
    assert _block_is_eligible(unknown) is None, "unknown block type must not be patched"


def _eligible_block(cls_name, spec):
    """A block with a real bound forward (the reference loop) and no patch attrs set,
    so enable_grouped_moe/disable_grouped_moe drive the whole lifecycle themselves."""
    hidden, inter, E, top_k = 64, 128, 4, 2
    blk = type(cls_name, (nn.Module,), {})()
    blk.experts = nn.ModuleList(
        [_make_expert(hidden, inter, spec[0], spec[1], spec[2]) for _ in range(E)]
    )
    blk.gate = nn.Linear(hidden, E, bias=False).to(DEV, DTYPE)
    blk.num_experts = E
    blk.top_k = top_k
    blk.norm_topk_prob = True
    blk.eval()
    blk.forward = types.MethodType(
        lambda self, hs, _s=spec: _reference_forward(self, hs, _s)[0], blk
    )
    return blk


def test_enable_disable_roundtrip():
    spec = _BLOCK_SPECS["Qwen3MoeSparseMoeBlock"]
    model = nn.Module()
    model.layers = nn.ModuleList(
        [_eligible_block("Qwen3MoeSparseMoeBlock", spec) for _ in range(2)]
    )
    x = torch.randn(1, 48, 64, device=DEV, dtype=DTYPE)
    blk0 = model.layers[0]
    expected = _reference_forward(blk0, x, spec)[0]

    n = enable_grouped_moe(model, verbose=False)
    assert n == 2, f"expected 2 patched blocks, got {n}"
    assert blk0.forward.__func__ is grouped_moe_forward, "forward not swapped to grouped path"
    out, _ = blk0.forward(x)  # patched path returns (final, router_logits)
    assert _rel_l2(out, expected) < 1e-2

    m = disable_grouped_moe(model)
    assert m == 2, f"expected 2 restored blocks, got {m}"
    assert getattr(blk0, "_unsloth_moe_spec", None) is None, "patch attrs not cleaned up"
    assert blk0.forward.__func__ is not grouped_moe_forward, "original forward not restored"
    assert _rel_l2(blk0.forward(x), expected) < 1e-2  # restored path returns the bare tensor


def test_wrap_loader_idempotent_and_enables():
    spec = _BLOCK_SPECS["Qwen3MoeSparseMoeBlock"]

    def fake_from_pretrained(*args, **kwargs):
        model = nn.Module()
        model.layers = nn.ModuleList([_eligible_block("Qwen3MoeSparseMoeBlock", spec)])
        return model, "tokenizer"

    wrapped = wrap_loader_for_grouped_moe(fake_from_pretrained)
    assert wrapped is not fake_from_pretrained, "loader was not wrapped"
    assert wrap_loader_for_grouped_moe(wrapped) is wrapped, "double-wrapping must be a no-op"

    model, tok = wrapped()
    assert tok == "tokenizer", "wrapper must pass the loader's return value through unchanged"
    assert model.layers[0].forward.__func__ is grouped_moe_forward, "grouped MoE not enabled on return"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
