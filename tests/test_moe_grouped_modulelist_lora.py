"""Expert LoRA / QLoRA on the transformers<5 ModuleList grouped MoE path.

Blocks: the real Qwen3-MoE / Mixtral / OLMoE sparse MoE blocks when transformers still stores
experts as an nn.ModuleList (< 5), else synthetic twins running the same per-expert loop.
Experts carry real PEFT LoRA (bf16 or bitsandbytes NF4 base). The grouped forward is checked
against the original loop with a float64 oracle that shares the loop's routing: the grouped
error must stay within a small multiple of the loop's own bf16 error, and a perturbed adapter
must fail the same bound. Unsupported adapter states fall back to the loop (engagement 0), and
a forced engagement shows each guard is load-bearing.
"""
import contextlib
import copy
import os
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.device_type import DEVICE_TYPE_TORCH
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

DEV = DEVICE_TYPE_TORCH
DT = torch.bfloat16
pytestmark = pytest.mark.skipif(
    not (DEV in ("cuda", "xpu") and ML._grouped_mm_supported()),
    reason = "torch._grouped_mm unsupported on this device",
)

KINDS = {
    # kind: (block class name, gate, up, down, norm_topk_prob)
    "qwen3":   ("Qwen3MoeSparseMoeBlock", "gate_proj", "up_proj", "down_proj", True),
    "mixtral": ("MixtralSparseMoeBlock",  "w1",        "w3",      "w2",        True),
    "olmoe":   ("OlmoeSparseMoeBlock",    "gate_proj", "up_proj", "down_proj", False),
}
H, I, E, TOPK, T = 128, 256, 8, 2, 192


# ----------------------------------------------------------------------------- builders
def _hf_block(kind):
    """The real transformers block when its experts are a ModuleList, else None."""
    try:
        if kind == "qwen3":
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeSparseMoeBlock
            cfg = Qwen3MoeConfig(hidden_size = H, moe_intermediate_size = I, num_experts = E,
                                 num_experts_per_tok = TOPK, norm_topk_prob = True)
            blk = Qwen3MoeSparseMoeBlock(cfg)
        elif kind == "mixtral":
            from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralSparseMoeBlock
            cfg = MixtralConfig(hidden_size = H, intermediate_size = I, num_local_experts = E,
                                num_experts_per_tok = TOPK, router_jitter_noise = 0.0)
            blk = MixtralSparseMoeBlock(cfg)
        else:
            from transformers.models.olmoe.modeling_olmoe import OlmoeConfig, OlmoeSparseMoeBlock
            cfg = OlmoeConfig(hidden_size = H, intermediate_size = I, num_experts = E,
                              num_experts_per_tok = TOPK, norm_topk_prob = False)
            blk = OlmoeSparseMoeBlock(cfg)
    except Exception:
        return None
    if not isinstance(getattr(blk, "experts", None), nn.ModuleList):
        return None
    return blk


class _Expert(nn.Module):
    def __init__(self, g, u, d):
        super().__init__()
        self._names = (g, u, d)
        setattr(self, g, nn.Linear(H, I, bias = False))
        setattr(self, u, nn.Linear(H, I, bias = False))
        setattr(self, d, nn.Linear(I, H, bias = False))
        self.act_fn = F.silu

    def forward(self, x):
        g, u, d = (getattr(self, n) for n in self._names)
        return d(self.act_fn(g(x)) * u(x))


def _loop_forward(self, hidden_states):
    """transformers 4.x Qwen3MoeSparseMoeBlock.forward (Mixtral / OLMoE share it)."""
    b, s, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    router_logits = self.gate(hidden_states)
    rw = F.softmax(router_logits, dim = 1, dtype = torch.float)
    rw, sel = torch.topk(rw, self.top_k, dim = -1)
    if self.norm_topk_prob:
        rw /= rw.sum(dim = -1, keepdim = True)
    rw = rw.to(hidden_states.dtype)
    final = torch.zeros((b * s, h), dtype = hidden_states.dtype, device = hidden_states.device)
    mask = F.one_hot(sel, num_classes = self.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim = (-1, -2)), 0).nonzero():
        idx, top_x = torch.where(mask[e].squeeze(0))
        cur = hidden_states[None, top_x].reshape(-1, h)
        out = self.experts[e](cur) * rw[top_x, idx, None]
        final.index_add_(0, top_x, out.to(hidden_states.dtype))
    return final.reshape(b, s, h), router_logits


def _synthetic_block(kind):
    cls_name, g, u, d, norm = KINDS[kind]
    cls = type(cls_name, (nn.Module,), {"forward": _loop_forward})
    blk = cls()
    blk.gate = nn.Linear(H, E, bias = False)
    blk.experts = nn.ModuleList([_Expert(g, u, d) for _ in range(E)])
    blk.num_experts, blk.top_k, blk.norm_topk_prob = E, TOPK, norm
    return blk


def _to_4bit(blk, names):
    import bitsandbytes as bnb
    for ex in blk.experts:
        for n in names:
            lin = getattr(ex, n)
            q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias = False,
                                  compute_dtype = DT, quant_type = "nf4")
            q.weight = bnb.nn.Params4bit(lin.weight.data.to(DT).cpu(), requires_grad = False, quant_type = "nf4")
            setattr(ex, n, q)
    return blk


def build(kind, base = "bf16", r = 8, targets = "all", lora_dtype = torch.float32, seed = 0,
          prefer_hf = True, **lora_kw):
    """(peft model, block) with randomized adapters on the routed experts."""
    torch.manual_seed(seed)
    _, g, u, d, _ = KINDS[kind]
    blk = (_hf_block(kind) if prefer_hf else None) or _synthetic_block(kind)
    if base == "nf4":
        pytest.importorskip("bitsandbytes")
        if not ML.HAS_BNB:
            pytest.skip("bitsandbytes not usable")
        blk = _to_4bit(blk, (g, u, d))
    blk = blk.to(DEV)
    for p in blk.parameters():
        if p.dtype.is_floating_point:
            p.data = p.data.to(DT)
        p.requires_grad_(False)
    root = nn.Module()
    root.mlp = blk
    # As on a from_pretrained(load_in_4bit=True) model: PEFT then wraps with lora.bnb.Linear4bit.
    root.is_loaded_in_4bit = base == "nf4"
    tm = {"all": [g, u, d], "gate": [g], "gate_down": [g, d], "up": [u], "down": [d]}[targets]
    cfg = LoraConfig(r = r, lora_alpha = 2 * r, target_modules = tm, **lora_kw)
    model = get_peft_model(root, cfg)
    blk = model.base_model.model.mlp
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = (torch.randn_like(p, dtype = torch.float32) * (0.5 / p.shape[-1] ** 0.5)).to(lora_dtype)
    model.eval()
    return model, blk


def enable(model, blk, **kw):
    if not hasattr(blk, "_orig_moe_forward"):
        n = ML.enable_grouped_moe(model, verbose = False, **kw)
        assert n == 1, f"block not patched ({n}); last decline {ML.LAST_DECLINE}"
    return blk


def loop(blk, x):
    f = getattr(blk, "_orig_moe_forward", None)
    return f(x) if f is not None else blk.forward(x)


# ----------------------------------------------------------------------------- oracle
def _lora_params(blk):
    """[(expert, proj name, lora_A weight, lora_B weight)] for every wrapped projection."""
    out = []
    for e, ex in enumerate(blk.experts):
        for n in KINDS_BY_CLS[type(blk).__name__][:3]:
            p = getattr(ex, n)
            if hasattr(p, "lora_A"):
                a = p.active_adapters[0]
                out.append((e, n, p.lora_A[a].weight, p.lora_B[a].weight))
    return out


KINDS_BY_CLS = {v[0]: v[1:] for v in KINDS.values()}


def oracle(blk, x, gout):
    """float64 forward + LoRA grads with the bf16 loop's routing. Returns (out, {param: grad})."""
    g, u, d, norm = KINDS_BY_CLS[type(blk).__name__]
    h = x.reshape(-1, x.shape[-1])
    with torch.no_grad():
        logits = blk.gate(h)
        rw = F.softmax(logits, dim = 1, dtype = torch.float)
        rw, sel = torch.topk(rw, blk.top_k, dim = -1)
        if norm:
            rw = rw / rw.sum(dim = -1, keepdim = True)
        rw = rw.to(h.dtype).double()
    hd = h.double()
    final = torch.zeros(h.shape, dtype = torch.float64, device = h.device)
    leaves = {}
    for e, ex in enumerate(blk.experts):
        tok, slot = torch.where(sel == e)
        if tok.numel() == 0:
            continue
        xe = hd[tok]
        outs = {}
        for n, inp in ((g, None), (u, None), (d, "inter")):
            p = getattr(ex, n)
            W = ML._expert_weight(p, DT).double()
            if inp == "inter":
                xi = outs["inter"]
            else:
                xi = xe
            y = xi @ W.t()
            if hasattr(p, "lora_A") and not p.disable_adapters and not p.merged:
                a = p.active_adapters[0]
                A = p.lora_A[a].weight.detach().double().requires_grad_(True)
                B = p.lora_B[a].weight.detach().double().requires_grad_(True)
                leaves[p.lora_A[a].weight] = A
                leaves[p.lora_B[a].weight] = B
                y = y + (xi @ A.t()) @ B.t() * p.scaling[a]
            outs[n] = y
            if n == u:
                outs["inter"] = F.silu(outs[g]) * outs[u]
        final.index_add_(0, tok, outs[d] * rw[tok, slot, None])
    final = final.reshape(x.shape)
    (final * gout.double()).sum().backward()
    return final.detach(), {k: v.grad for k, v in leaves.items()}


def run(blk, fwd, x, gout, autocast = False):
    for _, _, A, B in _lora_params(blk):
        A.grad = B.grad = None
    ctx = torch.autocast(DEV, dtype = DT) if autocast else contextlib.nullcontext()
    with ctx:
        out = fwd(x)[0]
    (out.float() * gout).sum().backward()
    grads = {}
    for _, _, A, B in _lora_params(blk):
        for w in (A, B):
            grads[w] = torch.zeros_like(w) if w.grad is None else w.grad.clone()
    return out.detach(), grads


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def errors(blk, out, grads, ref_out, ref_grads):
    """(output rel L2, max per-parameter grad rel L2, all-grads rel L2) vs the oracle."""
    keys = list(ref_grads)
    per = [rel(grads[k], ref_grads[k]) for k in keys if ref_grads[k].norm() > 0]
    cat = rel(torch.cat([grads[k].flatten().double() for k in keys]),
              torch.cat([ref_grads[k].flatten() for k in keys]))
    return rel(out, ref_out), max(per), cat


def measure(blk, model, autocast = False, seed = 1):
    """Loop and grouped errors vs the float64 oracle on one batch."""
    torch.manual_seed(seed)
    x = torch.randn(1, T, H, device = DEV, dtype = DT)
    gout = torch.randn(1, T, H, device = DEV, dtype = torch.float32)
    ref_out, ref_grads = oracle(blk, x, gout)
    lo, lg = run(blk, lambda t: loop(blk, t), x, gout, autocast)
    before = ML.CALLS["grouped_lora"]
    go, gg = run(blk, blk.forward, x, gout, autocast)
    engaged = ML.CALLS["grouped_lora"] - before
    return (errors(blk, lo, lg, ref_out, ref_grads), errors(blk, go, gg, ref_out, ref_grads),
            rel(go, lo), engaged)


# Grouped error vs the oracle may exceed the loop's own bf16 error by at most this factor
# (plus a small absolute floor for near-exact cases); the perturbed control must exceed it.
RATIO, FLOOR = 2.0, 2e-3


def within(loop_err, grp_err):
    return all(g <= RATIO * l + FLOOR for l, g in zip(loop_err, grp_err))


# ----------------------------------------------------------------------------- parity
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("base", ["bf16", "nf4"])
@pytest.mark.parametrize("r", [4, 8, 16])
def test_lora_parity_all_projections(kind, base, r):
    model, blk = build(kind, base = base, r = r)
    enable(model, blk)
    le, ge, gl, engaged = measure(blk, model)
    assert engaged == 1, f"grouped LoRA path did not run (decline: {ML.LAST_DECLINE})"
    assert within(le, ge), f"{kind}/{base}/r{r}: loop err {le} vs grouped err {ge}"


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("targets", ["gate", "up", "down", "gate_down"])
def test_lora_parity_subset(kind, targets):
    model, blk = build(kind, base = "bf16", r = 8, targets = targets)
    enable(model, blk)
    le, ge, gl, engaged = measure(blk, model)
    assert engaged == 1
    assert within(le, ge), f"{kind}/{targets}: loop err {le} vs grouped err {ge}"


@pytest.mark.parametrize("base", ["bf16", "nf4"])
def test_lora_parity_autocast_and_bf16_adapters(base):
    """Under autocast (the trainer's bf16 mode) and with bf16 adapter weights."""
    for lora_dtype, ac in ((torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)):
        model, blk = build("qwen3", base = base, r = 8, lora_dtype = lora_dtype)
        enable(model, blk)
        le, ge, gl, engaged = measure(blk, model, autocast = ac)
        assert engaged == 1
        assert within(le, ge), f"{base}/{lora_dtype}/autocast={ac}: loop {le} grouped {ge}"


@pytest.mark.parametrize("mode", ["recompute", "pinned", "cache"])
@pytest.mark.parametrize("base", ["bf16", "nf4"])
def test_lora_parity_base_modes(mode, base):
    """Every base-stack policy (rebuild in backward / pinned / resident cache) with expert LoRA."""
    model, blk = build("mixtral", base = base, r = 8)
    enable(model, blk, recompute = mode == "recompute", cache = mode == "cache")
    le, ge, gl, engaged = measure(blk, model)
    assert engaged == 1
    assert within(le, ge), f"{mode}/{base}: loop err {le} vs grouped err {ge}"


def test_lora_parity_under_nonreentrant_checkpoint():
    """Gradient checkpointing (use_reentrant=False) recomputes the grouped LoRA forward in backward."""
    from torch.utils.checkpoint import checkpoint
    model, blk = build("qwen3", base = "nf4", r = 8)
    enable(model, blk, recompute = False)
    torch.manual_seed(1)
    x = torch.randn(1, T, H, device = DEV, dtype = DT)
    gout = torch.randn(1, T, H, device = DEV, dtype = torch.float32)
    ref_out, ref_grads = oracle(blk, x, gout)
    lo, lg = run(blk, lambda t: loop(blk, t), x, gout)
    go, gg = run(blk, lambda t: (checkpoint(lambda u: blk.forward(u)[0], t, use_reentrant = False),), x, gout)
    assert within(errors(blk, lo, lg, ref_out, ref_grads), errors(blk, go, gg, ref_out, ref_grads))


def test_lora_parity_fp16():
    """float16 base and input (T4-class GPUs), fp32 adapters as PEFT creates them."""
    global DT
    old = DT
    DT = torch.float16
    try:
        model, blk = build("qwen3", base = "bf16", r = 8)
        enable(model, blk)
        le, ge, gl, engaged = measure(blk, model)
        assert engaged == 1
        assert within(le, ge), f"fp16: loop err {le} vs grouped err {ge}"
    finally:
        DT = old


def test_resident_cache_rebuilt_after_merge_unmerge():
    """UNSLOTH_MOE_GROUPED_CACHE stacks are dropped when the adapter state changes (a bf16 merge
    edits the base in place; unmerge leaves bf16 rounding behind), so they track the live base."""
    model, blk = build("qwen3", base = "bf16", r = 8)
    enable(model, blk, cache = True)
    x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    _engages(blk, x)
    stale = blk._cached_gate_up
    model.merge_adapter()
    _engages(blk, x)                             # merged: loop, signature changed
    model.unmerge_adapter()
    ok, out = _engages(blk, x)
    assert ok and blk._cached_gate_up is not stale
    with torch.no_grad():
        want = loop(blk, x)[0]
    assert rel(out, want) < 1e-2


def test_perturbed_adapter_fails_the_bound():
    """Negative control: scale one expert's lora_B by 1.05 in the grouped arm only."""
    model, blk = build("qwen3", base = "bf16", r = 8)
    enable(model, blk)
    torch.manual_seed(1)
    x = torch.randn(1, T, H, device = DEV, dtype = DT)
    gout = torch.randn(1, T, H, device = DEV, dtype = torch.float32)
    ref_out, ref_grads = oracle(blk, x, gout)
    lo, lg = run(blk, lambda t: loop(blk, t), x, gout)
    le = errors(blk, lo, lg, ref_out, ref_grads)
    w = blk.experts[3].down_proj.lora_B["default"].weight
    with torch.no_grad():
        w.mul_(1.05)
    go, gg = run(blk, blk.forward, x, gout)
    with torch.no_grad():
        w.div_(1.05)
    ge = errors(blk, go, gg, ref_out, ref_grads)
    assert not within(le, ge), f"perturbed arm passed the bound: loop {le} grouped {ge}"


def test_compile_fullgraph_with_lora():
    """The grouped LoRA block traces with fullgraph=True (no graph break), as the frozen block does."""
    model, blk = build("qwen3", base = "bf16", r = 8, prefer_hf = False)
    enable(model, blk)
    torch._dynamo.reset()
    torch.manual_seed(2)
    x = torch.randn(1, T, H, device = DEV, dtype = DT)
    gout = torch.randn(1, T, H, device = DEV, dtype = torch.float32)
    ref_out, ref_grads = oracle(blk, x, gout)
    lo, lg = run(blk, lambda t: loop(blk, t), x, gout)
    before = ML.CALLS["grouped_lora"]
    co, cg = run(blk, torch.compile(blk.forward, fullgraph = True), x, gout)
    assert ML.CALLS["grouped_lora"] == before + 1, "engagement counter must count compiled calls"
    assert within(errors(blk, lo, lg, ref_out, ref_grads), errors(blk, co, cg, ref_out, ref_grads))


# ----------------------------------------------------------------------------- declines
def _assert_falls_back(blk, x = None, ctx = contextlib.nullcontext):
    if x is None:
        torch.manual_seed(3)
        x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    before = ML.CALLS["grouped"]
    with ctx():
        with torch.no_grad():
            try:
                want = loop(blk, x)[0]
            except Exception as exc:   # the loop itself rejects this state: so must the fallback
                with pytest.raises(type(exc)):
                    blk.forward(x)
                assert ML.CALLS["grouped"] == before, "grouped path ran on an unsupported adapter state"
                return
            got = blk.forward(x)[0]
    assert ML.CALLS["grouped"] == before, "grouped path ran on an unsupported adapter state"
    assert torch.equal(got, want), "fallback must be the original loop, bit for bit"


def _force_engage(monkeypatch):
    """Guard removed: every block claims the first expert's adapter (name, scaling, rank)."""
    def forced(block, experts, spec, device, dtype):
        out = {}
        for key, n in zip(("gate", "up", "down"), spec[:3]):
            p = getattr(experts[0], n)
            if hasattr(p, "lora_A"):
                a = p.active_adapters[0]
                out[key] = (a, float(p.scaling[a]), p.lora_A[a].weight.shape[0])
            else:
                out[key] = None
        return out
    monkeypatch.setattr(ML, "_grouped_state", forced)


def _differs(blk, x = None, ctx = contextlib.nullcontext, train = False):
    """True when the forced grouped forward does not match the loop (or cannot run)."""
    if x is None:
        torch.manual_seed(3)
        x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    try:
        with ctx():
            torch.manual_seed(0)
            got = blk.forward(x)[0]
            torch.manual_seed(0)
            want = loop(blk, x)[0]
    except Exception:
        return True
    return rel(got, want) > 1e-2


def _decline_case(name):
    """(model, blk, ctx) for one unsupported adapter state."""
    ctx = contextlib.nullcontext
    if name == "dropout_train":
        model, blk = build("qwen3", lora_dropout = 0.5)
        enable(model, blk)
        blk.train()
    elif name == "dora":
        model, blk = build("qwen3", use_dora = True)
    elif name == "lora_bias":
        model, blk = build("qwen3", lora_bias = True)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n and n.endswith("bias"):
                    p.normal_()
    elif name == "two_adapters":
        model, blk = build("qwen3")
        enable(model, blk)
        model.add_adapter("other", LoraConfig(r = 8, lora_alpha = 16, target_modules = ["gate_proj", "up_proj", "down_proj"]))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_" in n and ".other." in n:
                    p.normal_(std = 0.05)
        model.base_model.set_adapter(["default", "other"])
    elif name == "disabled":
        model, blk = build("qwen3")
        enable(model, blk)
        ctx = model.disable_adapter
    elif name == "merged":
        model, blk = build("qwen3")
        enable(model, blk)
        model.merge_adapter()
    elif name == "mixed_batch":
        model, blk = build("qwen3")
        enable(model, blk)
        ctx = lambda: model.base_model._enable_peft_forward_hooks(adapter_names = ["__base__"])
    elif name == "hetero_rank":
        model, blk = build("qwen3", rank_pattern = {"experts.2.up_proj": 4})
    elif name == "hetero_scaling":
        model, blk = build("qwen3", alpha_pattern = {"experts.2.up_proj": 64})
    elif name == "partial_wrap":
        model, blk = build("qwen3", targets = "all")
        # unwrap expert 5's up_proj back to its base layer
        ex = blk.experts[5]
        ex.up_proj = ex.up_proj.base_layer
    elif name == "trainable_base":
        model, blk = build("qwen3")
        enable(model, blk)
        blk.experts[1].gate_proj.base_layer.weight.requires_grad_(True)
    elif name == "lora_dtype_mismatch":
        model, blk = build("qwen3")
        enable(model, blk)
        blk.experts[0].up_proj.lora_B["default"].to(torch.bfloat16)
    elif name == "kill_switch":
        model, blk = build("qwen3")
        enable(model, blk)
        os.environ["UNSLOTH_MOE_GROUPED_LORA"] = "0"
    else:
        raise KeyError(name)
    if not hasattr(blk, "_orig_moe_forward"):
        # Not patched at enable (declined there): patch directly to exercise the per-call check.
        blk._orig_moe_forward = blk.forward
        blk._unsloth_moe_spec = ML._BLOCK_SPECS[type(blk).__name__]
        blk._moe_recompute = False
        blk._moe_cache = False
        blk.forward = types.MethodType(ML.grouped_moe_forward, blk)
    return model, blk, ctx


DECLINES = ["dropout_train", "dora", "lora_bias", "two_adapters", "disabled", "merged", "mixed_batch",
            "hetero_rank", "hetero_scaling", "partial_wrap", "trainable_base", "lora_dtype_mismatch",
            "kill_switch"]


@pytest.fixture(autouse = True)
def _clean_env():
    old = os.environ.pop("UNSLOTH_MOE_GROUPED_LORA", None)
    yield
    os.environ.pop("UNSLOTH_MOE_GROUPED_LORA", None)
    if old is not None:
        os.environ["UNSLOTH_MOE_GROUPED_LORA"] = old


@pytest.mark.parametrize("name", DECLINES)
def test_decline_falls_back_to_loop(name):
    model, blk, ctx = _decline_case(name)
    if name == "dropout_train":
        # dropout draws differ between two loop calls: check engagement only
        before = ML.CALLS["grouped"]
        with torch.no_grad():
            blk.forward(torch.randn(1, 64, H, device = DEV, dtype = DT))
        assert ML.CALLS["grouped"] == before
        return
    _assert_falls_back(blk, ctx = ctx)


@pytest.mark.parametrize("name", [n for n in DECLINES if n not in ("kill_switch", "lora_dtype_mismatch")])
def test_decline_guard_is_load_bearing(name, monkeypatch):
    """With the readiness guard removed, the grouped forward no longer matches the loop."""
    model, blk, ctx = _decline_case(name)
    _force_engage(monkeypatch)
    if name == "trainable_base":
        torch.manual_seed(3)
        x = torch.randn(1, 64, H, device = DEV, dtype = DT)
        w = blk.experts[1].gate_proj.base_layer.weight
        w.grad = None
        blk.forward(x)[0].float().sum().backward()
        assert w.grad is None, "a frozen-base grouped forward cannot produce the base weight grad"
        loop(blk, x)[0].float().sum().backward()
        assert w.grad is not None
        return
    if name == "dropout_train":
        torch.manual_seed(3)
        x = torch.randn(1, 64, H, device = DEV, dtype = DT)
        with torch.no_grad():
            torch.manual_seed(0)
            got = blk.forward(x)[0]
            blk.eval()
            want = loop(blk, x)[0]
        # forced grouped skips dropout: equals the eval (no-dropout) loop, so not the train loop
        assert rel(got, want) < 1e-2
        blk.train()
        with torch.no_grad():
            torch.manual_seed(0)
            want_train = loop(blk, x)[0]
        assert rel(got, want_train) > 1e-2
        return
    assert _differs(blk, ctx = ctx), f"{name}: forced engagement still matched the loop"


def test_lora_dtype_mismatch_reason():
    model, blk, _ = _decline_case("lora_dtype_mismatch")
    assert ML._experts_grouped_state(blk.experts, blk._unsloth_moe_spec, torch.device(DEV, torch.cuda.current_device()) if DEV == "cuda" else torch.device(DEV), DT) == "up_proj LoRA: lora_A / lora_B dtypes differ"


def test_kill_switch_keeps_lora_blocks_on_the_loop():
    model, blk = build("qwen3")
    os.environ["UNSLOTH_MOE_GROUPED_LORA"] = "0"
    assert ML.enable_grouped_moe(model, verbose = False) == 0
    assert not hasattr(blk, "_orig_moe_forward")
    os.environ["UNSLOTH_MOE_GROUPED_LORA"] = "1"
    assert ML.enable_grouped_moe(model, verbose = False) == 1
    assert blk.forward.__func__ is ML.grouped_moe_forward
    os.environ["UNSLOTH_MOE_GROUPED"] = "0"
    try:
        assert ML.enable_grouped_moe(model, verbose = False) == 0
        assert not hasattr(blk, "_orig_moe_forward")
    finally:
        os.environ.pop("UNSLOTH_MOE_GROUPED", None)


@pytest.mark.parametrize("wrapped", [True, False])
def test_expert_bias_declines(wrapped, monkeypatch):
    """A base bias on any expert projection keeps the loop (the grouped GEMMs add none), both at
    patch time and per call; with the guard removed the grouped output drops the bias."""
    model, blk = build("qwen3", targets = "all" if wrapped else "gate", seed = 7)
    if not wrapped:
        for ex in blk.experts:
            ex.gate_proj = ex.gate_proj.base_layer
    enable(model, blk)
    lin = ML._base_lin(blk.experts[6].up_proj)
    lin.bias = nn.Parameter(torch.randn(lin.out_features, device = DEV, dtype = DT), requires_grad = False)
    assert ML._block_is_eligible(blk) is None
    _assert_falls_back(blk)
    assert ML.enable_grouped_moe(model, verbose = False) == 0 and not hasattr(blk, "_orig_moe_forward")
    # patch directly and force engagement past the guard
    enable_direct = _decline_case.__globals__["types"].MethodType
    blk._orig_moe_forward = blk.forward
    blk._unsloth_moe_spec = ML._BLOCK_SPECS[type(blk).__name__]
    blk._moe_recompute = blk._moe_cache = False
    blk.forward = enable_direct(ML.grouped_moe_forward, blk)
    _force_engage(monkeypatch)
    assert _differs(blk), "forced grouped forward should drop the expert bias"


def test_non_lora_peft_wrapper_declines(monkeypatch):
    """A non-LoRA PEFT tuner (IA3) wraps the experts with base_layer and no lora_A: the block keeps
    the loop; with the guard removed the grouped output drops the tuner's scaling."""
    from peft import IA3Config
    torch.manual_seed(7)
    _, g, u, d, _ = KINDS["qwen3"]
    blk = (_hf_block("qwen3") or _synthetic_block("qwen3")).to(DEV)
    for p in blk.parameters():
        p.data = p.data.to(DT)
        p.requires_grad_(False)
    root = nn.Module()
    root.mlp = blk
    model = get_peft_model(root, IA3Config(target_modules = [u], feedforward_modules = [u]))
    blk = model.base_model.model.mlp
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "ia3_l" in n:
                p.data = (1 + torch.randn_like(p)).to(p.dtype)
    model.eval()
    assert ML._projs_lora([getattr(ex, u) for ex in blk.experts]) == "PEFT wrapper is not LoRA"
    assert ML._block_is_eligible(blk) is None
    assert ML.enable_grouped_moe(model, verbose = False) == 0 and not hasattr(blk, "_orig_moe_forward")
    _assert_falls_back(blk)
    # patch directly and force engagement past the guard
    blk._orig_moe_forward = blk.forward
    blk._unsloth_moe_spec = ML._BLOCK_SPECS[type(blk).__name__]
    blk._moe_recompute = blk._moe_cache = False
    blk.forward = types.MethodType(ML.grouped_moe_forward, blk)
    _force_engage(monkeypatch)
    assert _differs(blk), "forced grouped forward should drop the IA3 scaling"


def test_signature_tracks_dtype_of_every_expert():
    """Casting one non-first expert's adapter in place keeps every Parameter identity; the cached
    verdict must still be re-checked (here: lora_A / lora_B dtypes now differ -> loop)."""
    model, blk = build("qwen3")
    enable(model, blk)
    x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    ok, _ = _engages(blk, x)
    assert ok
    key = blk._moe_ready[0]
    w = blk.experts[5].up_proj.lora_B["default"].weight
    w.data = w.data.to(torch.bfloat16)
    _assert_falls_back(blk, x)   # the loop rejects mixed lora_A / lora_B dtypes, so must the block
    assert blk._moe_ready[0] != key
    assert ML.LAST_DECLINE["reason"] == "up_proj LoRA: lora_A / lora_B dtypes differ"


def test_enable_declines_unsupported_lora_at_patch_time():
    model, blk = build("qwen3", use_dora = True)
    assert ML.enable_grouped_moe(model, verbose = False) == 0
    assert not hasattr(blk, "_orig_moe_forward")


# ----------------------------------------------------------------------------- toggles
def _engages(blk, x):
    before = ML.CALLS["grouped_lora"]
    with torch.no_grad():
        out = blk.forward(x)[0]
    return ML.CALLS["grouped_lora"] - before == 1, out


def test_adapter_disable_enable_merge_unmerge_and_train_eval():
    model, blk = build("qwen3", base = "nf4", lora_dropout = 0.1)
    enable(model, blk)
    torch.manual_seed(4)
    x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    with torch.no_grad():
        want = loop(blk, x)[0]

    ok, out = _engages(blk, x)                   # eval: dropout inactive -> grouped
    assert ok and rel(out, want) < 1e-2
    blk.train()                                  # train with dropout 0.1 -> loop
    ok, _ = _engages(blk, x)
    assert not ok
    blk.eval()
    ok, out = _engages(blk, x)
    assert ok and rel(out, want) < 1e-2

    with model.disable_adapter():                # disabled -> loop (base only)
        ok, out = _engages(blk, x)
        assert not ok
    ok, out = _engages(blk, x)                   # re-enabled -> grouped again
    assert ok and rel(out, want) < 1e-2

    model.merge_adapter()                        # merged -> loop
    ok, _ = _engages(blk, x)
    assert not ok
    model.unmerge_adapter()                      # nf4 unmerge requantizes the base: new reference
    with torch.no_grad():
        want = loop(blk, x)[0]
    ok, out = _engages(blk, x)
    assert ok and rel(out, want) < 1e-2


def test_dropout_zero_engages_in_train_mode():
    model, blk = build("mixtral", lora_dropout = 0.0)
    enable(model, blk)
    blk.train()
    ok, _ = _engages(blk, torch.randn(1, 64, H, device = DEV, dtype = DT))
    assert ok


def test_adapter_switch_rechecks():
    model, blk = build("qwen3")
    enable(model, blk)
    model.add_adapter("b", LoraConfig(r = 16, lora_alpha = 8, target_modules = ["gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".b." in n and "lora_" in n:
                p.normal_(std = 0.05)
    x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    ok, a_out = _engages(blk, x)
    assert ok
    model.set_adapter("b")
    ok, b_out = _engages(blk, x)
    assert ok
    with torch.no_grad():
        want = loop(blk, x)[0]
    assert rel(b_out, want) < 1e-2 and rel(a_out, want) > 1e-2


def test_signature_cache_is_reused_and_invalidated():
    model, blk = build("qwen3")
    enable(model, blk)
    x = torch.randn(1, 64, H, device = DEV, dtype = DT)
    _engages(blk, x)
    key = blk._moe_ready[0]
    _engages(blk, x)
    assert blk._moe_ready[0] == key
    blk.experts[7].down_proj.lora_dropout["default"] = nn.Dropout(0.3)
    blk.train()
    ok, _ = _engages(blk, x)
    assert not ok and blk._moe_ready[0] != key


def test_frozen_block_still_grouped_after_peft_on_other_layers():
    """Attention-only LoRA: expert projections stay plain, the block keeps the frozen grouped path."""
    model, blk = build("qwen3", targets = "all")
    # a second, LoRA-free block in the same model
    _, plain = build("qwen3", targets = "gate", seed = 5)
    for ex in plain.experts:
        ex.gate_proj = ex.gate_proj.base_layer
    model.base_model.model.plain = plain
    assert ML.enable_grouped_moe(model, verbose = False) == 2
    x = torch.randn(1, 32, H, device = DEV, dtype = DT)
    b0, l0 = ML.CALLS["grouped"], ML.CALLS["grouped_lora"]
    with torch.no_grad():
        plain.forward(x)
    assert ML.CALLS["grouped"] == b0 + 1 and ML.CALLS["grouped_lora"] == l0


if __name__ == "__main__":
    # Parity table: loop vs grouped error against the float64 oracle.
    print(f"{'case':38s} {'loop out':>9s} {'loop gmax':>9s} {'loop gcat':>9s} | "
          f"{'grp out':>9s} {'grp gmax':>9s} {'grp gcat':>9s} | grp-vs-loop")
    for kind in KINDS:
        for base in ("bf16", "nf4"):
            for r in (4, 8, 16):
                for lora_dtype, ac in ((torch.float32, False), (torch.float32, True), (torch.bfloat16, False)):
                    model, blk = build(kind, base = base, r = r, lora_dtype = lora_dtype)
                    enable(model, blk)
                    le, ge, gl, eng = measure(blk, model, autocast = ac)
                    tag = f"{kind}/{base}/r{r}/{str(lora_dtype)[6:]}/ac={int(ac)}"
                    print(f"{tag:38s} {le[0]:9.2e} {le[1]:9.2e} {le[2]:9.2e} | "
                          f"{ge[0]:9.2e} {ge[1]:9.2e} {ge[2]:9.2e} | {gl:.2e} eng={eng}")
