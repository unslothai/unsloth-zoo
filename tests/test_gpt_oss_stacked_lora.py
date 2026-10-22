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

"""Stacked expert LoRA on gpt-oss ModuleList NF4 experts (gate_up_projs / down_projs).

The loader entry point (auto_enable_grouped_moe -> gpt_oss_grouped_qlora.stack_expert_lora) turns
PEFT's per-expert lora_A / lora_B Parameters into one Parameter per projection list when the
grouped training path applies (bf16 torch._grouped_mm or the fp16 Triton GEMMs). Checked BITWISE
against the same model under UNSLOTH_MOE_STACKED_LORA=0: forward, input / LoRA grads (bf16 and
fp16 grouped paths), torch and bnb 8-bit optimizer steps, merge, adapter switches, routed decode
(eager, fullgraph compile, CUDA graph replay). Saved adapters keep PEFT's per-expert keys and load
in a process without Unsloth.
"""
import contextlib
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_gpt_oss_grouped_qlora as G  # noqa: E402  (skips without CUDA / bitsandbytes / peft)
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from peft.utils import set_peft_model_state_dict  # noqa: E402
from unsloth_zoo.temporary_patches import gpt_oss_grouped_qlora as gq  # noqa: E402
from unsloth_zoo.temporary_patches import gpt_oss_routed as gr  # noqa: E402
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML  # noqa: E402
from unsloth_zoo.temporary_patches import moe_utils  # noqa: E402

bnb = G.bnb
STACK = ML._STACK_NAME
BF16_OK = torch.cuda.is_bf16_supported() and moe_utils._check_torch_grouped_mm_supported()
needs_bf16 = pytest.mark.skipif(not BF16_OK, reason = "torch._grouped_mm / bf16 unavailable")
needs_fp16 = G.needs_fp16_grouped
TARGETS = r".*(gate_up_projs|down_projs)\.\d+"


@pytest.fixture(autouse = True)
def _clean_env():
    keys = ("UNSLOTH_MOE_STACKED_LORA", "UNSLOTH_GPTOSS_GROUPED", "UNSLOTH_COMPILE_DISABLE",
            "UNSLOTH_GPTOSS_ROUTED_KERNEL", "UNSLOTH_GPTOSS_ROUTED_INFERENCE")
    old = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in old.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


@contextlib.contextmanager
def _shapes(**kv):
    old = {k: getattr(G, k) for k in kv}
    for k, v in kv.items():
        setattr(G, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(G, k, v)


class _Model(torch.nn.Module):
    is_loaded_in_4bit = True   # PEFT then wraps the experts with its bnb Linear4bit LoRA layer

    def __init__(self, ex):
        super().__init__()
        self.experts = ex

    def forward(self, x, idx, w):
        return self.experts(x, idx, w)


def _randomize(model, seed, lora_dtype = None, adapter = None):
    g = torch.Generator().manual_seed(seed)
    for n, p in model.named_parameters():
        if "lora_" in n and (adapter is None or f".{adapter}." in n):
            dt = lora_dtype or p.dtype
            p.data = (torch.randn(p.shape, generator = g) * 0.05).to(p.device, dt)


def build(mode = "bf16", r = 16, base = "nf4", seed = 7, **lora_kw):
    """PeftModel over gpt-oss-shaped experts (the grouped QLoRA tests' builders), unstacked."""
    if base == "bf16":   # plain frozen bf16 Linear experts: the per-expert loop trains them
        ex = G._Experts(True)
        for projs in (ex.gate_up_projs, ex.down_projs):
            for i, p in enumerate(projs):
                lin = torch.nn.Linear(p.in_features, p.out_features, bias = True, device = "cuda", dtype = G.DT)
                lin.requires_grad_(False)
                projs[i] = lin
    elif mode == "fp16":
        ex = G._fp16_experts()
    else:
        ex = G._Experts(True)
    lora_kw.setdefault("lora_dropout", 0.0)
    model = get_peft_model(_Model(ex), LoraConfig(r = r, lora_alpha = 2 * r, target_modules = TARGETS, **lora_kw))
    _randomize(model, seed, torch.float32 if mode == "fp16" else None)
    if mode == "fp16":
        G._use_unsloth_fp16_lora_forward(model.base_model.model.experts)
    return model.train()


def experts_of(model):
    return model.base_model.model.experts


def stack(model, on = True):
    with _env(UNSLOTH_MOE_STACKED_LORA = "1" if on else "0"):
        ML.auto_enable_grouped_moe(model)   # the loader entry point
    return model


def pair(mode = "bf16", **kw):
    return stack(build(mode, **kw), True), stack(build(mode, **kw), False)


def n_stacked(model):
    return sum(1 for n, _ in model.named_parameters() if n.endswith("." + STACK))


def n_trainable(model):
    return sum(1 for p in model.parameters() if p.requires_grad)


def expert_lora(model):
    """{(proj list, expert, "A"/"B"): (weight, grad)} as PEFT's per-expert view."""
    ex = experts_of(model)
    out = {}
    for lname in ("gate_up_projs", "down_projs"):
        for e, p in enumerate(getattr(ex, lname)):
            a = p.active_adapters[0]
            for kind in ("A", "B"):
                m = getattr(p, "lora_" + kind)[a]
                if type(m) is ML._StackedLoraLinear:
                    st = getattr(m._unsloth_stack_owner, STACK)
                    g = None if st.grad is None else st.grad[m._unsloth_stack_index]
                else:
                    g = m.weight.grad
                out[(lname, e, kind)] = (m.weight.detach(), g)
    return out


def assert_lora_equal(ms, mu, grads = True):
    s, u = expert_lora(ms), expert_lora(mu)
    assert s.keys() == u.keys() and s
    for k in s:
        assert torch.equal(s[k][0], u[k][0]), f"weight {k}"
        if grads:
            assert (s[k][1] is None) == (u[k][1] is None), f"grad presence {k}"
            if s[k][1] is not None:
                assert torch.equal(s[k][1], u[k][1]), f"grad {k}"


def _inputs(mode, x_dtype, T = 96, seed = 3, unrouted = False):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    x = torch.randn(1, T, G.H, device = "cuda", generator = g).to(x_dtype)
    up = torch.randn(1, T, G.H, device = "cuda", generator = g)
    if unrouted:   # expert 0 gets no token: zero grads on the grouped path, stacked or not
        gc = torch.Generator().manual_seed(seed)
        idx = torch.stack([torch.randperm(G.E - 1, generator = gc)[:G.TOP_K] + 1 for _ in range(T)]).cuda()
        w = torch.zeros(T, G.E, device = "cuda").scatter_(1, idx, 1.0 / G.TOP_K)
    else:
        idx, w = G._routing(T, seed)
    w = w.float() if mode == "fp16" else w.to(G.DT)
    return x, idx, w, up


def fwd_bwd(model, x, idx, w, up):
    xi = x.detach().clone().requires_grad_(True)
    out = model(xi, idx, w)
    out.backward(up.view_as(out).to(out.dtype))
    return out.detach(), xi.grad


@contextlib.contextmanager
def _count_stack_reads(monkeypatch):
    hits = []

    def counting(projs, name, _orig = ML._lora_stacks):
        got = _orig(projs, name)
        hits.append(got is not None)
        return got
    monkeypatch.setattr(ML, "_lora_stacks", counting)
    monkeypatch.setattr(gq, "_lora_stacks", counting)
    yield hits


def _per_call_stack(t, depth = 3):
    """True when autograd shows a per-call torch.stack of per-expert weights behind `t`."""
    todo = [(t.grad_fn, 0)]
    while todo:
        fn, d = todo.pop()
        if fn is None or d > depth:
            continue
        if "StackBackward" in type(fn).__name__:
            return True
        todo += [(f, d + 1) for f, _ in fn.next_functions]
    return False


# ----------------------------------------------------------------------------- bitwise parity
@pytest.mark.parametrize("mode,x_dtype", [
    pytest.param("bf16", torch.bfloat16, marks = needs_bf16),
    pytest.param("bf16", torch.float32, marks = needs_bf16),
    pytest.param("fp16", torch.float16, marks = needs_fp16),
    pytest.param("fp16", torch.float32, marks = needs_fp16),
])
@pytest.mark.parametrize("unrouted", [False, True])
def test_stacked_matches_unstacked_bitwise(mode, x_dtype, unrouted, monkeypatch):
    ms, mu = pair(mode)
    E = G.E
    assert n_stacked(ms) == 4 and n_stacked(mu) == 0
    assert n_trainable(ms) == 4 and n_trainable(mu) == 4 * E
    x, idx, w, up = _inputs(mode, x_dtype, unrouted = unrouted)
    key = "forward_fp16_lora" if mode == "fp16" else "forward_lora"
    before = gq.CALLS[key]
    fp16_operands = []
    orig = gq._stack_lora
    monkeypatch.setattr(gq, "_stack_lora", lambda *a: fp16_operands.append(orig(*a)) or fp16_operands[-1])
    with _count_stack_reads(monkeypatch) as hits:
        os_, dxs = fwd_bwd(ms, x, idx, w, up)
    assert gq.CALLS[key] == before + 1
    assert hits and all(hits)   # both projections read the stacks, never torch.stack per expert
    if mode == "fp16":   # the fp16 path's A / B operands come from the stacks
        assert len(fp16_operands) == 4 and not any(map(_per_call_stack, fp16_operands))
    fp16_operands.clear()
    ou, dxu = fwd_bwd(mu, x, idx, w, up)
    if mode == "fp16":
        assert len(fp16_operands) == 4 and all(map(_per_call_stack, fp16_operands))
    assert gq.CALLS[key] == before + 2
    assert torch.equal(os_, ou) and torch.equal(dxs, dxu)
    assert_lora_equal(ms, mu)
    if unrouted:
        for k, (_, g) in expert_lora(ms).items():
            if k[1] == 0:
                assert g is not None and torch.count_nonzero(g) == 0, k


# ----------------------------------------------------------------------------- optimizers
def _opt_steps(model, mode, make_opt, steps = 3):
    opt = make_opt([p for p in model.parameters() if p.requires_grad])
    for s in range(steps):
        x, idx, w, up = _inputs(mode, torch.float32, seed = 20 + s)
        fwd_bwd(model, x, idx, w, up)
        opt.step()
        opt.zero_grad(set_to_none = True)
    return opt


def _bnb_state_per_expert(opt, model):
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        st = opt.state[p]
        if n.endswith("." + STACK):
            E = p.shape[0]
            for e in range(E):
                key = n.replace("_projs.0.", f"_projs.{e}.").replace(STACK, "weight")
                out[key] = tuple(st[k].view(E, -1)[e] for k in ("state1", "state2", "absmax1", "absmax2"))
        else:
            out[n] = tuple(st[k].reshape(-1) for k in ("state1", "state2", "absmax1", "absmax2"))
    return out


@pytest.mark.parametrize("mode", [pytest.param("bf16", marks = needs_bf16), pytest.param("fp16", marks = needs_fp16)])
@pytest.mark.parametrize("optim", ["adamw_torch", "adamw_8bit"])
def test_optimizer_steps_bitwise(mode, optim):
    """3 steps: weights equal bitwise, bnb's 8-bit states per expert too. Every per-expert tensor is
    a whole number of 256-element blocks and >= min_8bit_size (4096): r=16, hidden / inter 256
    (gpt-oss-20b: 2880 x r is block-aligned for r % 4 == 0)."""
    if optim == "adamw_8bit":
        make = lambda ps: bnb.optim.AdamW8bit(ps, lr = 1e-2, weight_decay = 0.01)
    else:
        make = lambda ps: torch.optim.AdamW(ps, lr = 1e-2, weight_decay = 0.01, foreach = True)
    with _shapes(H = 256, I = 256):
        ms, mu = pair(mode)
        os_ = _opt_steps(ms, mode, make)
        ou = _opt_steps(mu, mode, make)
        assert_lora_equal(ms, mu, grads = False)
        w0 = expert_lora(build(mode))
        assert any(not torch.equal(w0[k][0], v[0]) for k, v in expert_lora(ms).items())   # steps moved them
        if optim == "adamw_8bit":
            s, u = _bnb_state_per_expert(os_, ms), _bnb_state_per_expert(ou, mu)
            assert s.keys() == u.keys() and len(s) == 4 * G.E
            for k in s:
                assert all(torch.equal(a, b) for a, b in zip(s[k], u[k])), k


# ----------------------------------------------------------------------------- PEFT compatibility
def _st_load(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(str(path), "adapter_model.safetensors"))


PLAIN_SCRIPT = textwrap.dedent(r"""
import json, sys
import torch
from peft import PeftModel
base_path, out_path, *dirs = sys.argv[1:]
sd = torch.load(base_path)
E, H, I = sd["E"], sd["H"], sd["I"]

class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_projs = torch.nn.ModuleList([torch.nn.Linear(H, 2 * I) for _ in range(E)])
        self.down_projs = torch.nn.ModuleList([torch.nn.Linear(I, H) for _ in range(E)])

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = Experts()

res = {"unsloth": any(m.split(".")[0] in ("unsloth", "unsloth_zoo") for m in sys.modules)}
outs = []
for d in dirs:
    m = Model()
    m.load_state_dict(sd["weights"])
    m = PeftModel.from_pretrained(m, d).cuda().eval()
    ex = m.base_model.model.experts
    x = sd["x"].cuda()

    def run():
        with torch.no_grad():
            return torch.cat([ex.gate_up_projs[e](x) for e in range(E)] + [ex.down_projs[e](x[:, :I]) for e in range(E)], 1).cpu()
    y = run()
    with m.disable_adapter():
        y_base = run()
    outs.append((y, y_base))
    torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if "lora_" in n}, d + "/plain_loaded.pt")
res["equal"] = torch.equal(outs[0][0], outs[1][0])
res["lora_applied"] = not torch.equal(outs[0][0], outs[0][1])
json.dump(res, open(out_path, "w"))
""")


@needs_bf16
def test_save_pretrained_keys_values_and_plain_peft_reload(tmp_path):
    ms, mu = pair("bf16")
    ds, du = tmp_path / "s", tmp_path / "u"
    ms.save_pretrained(str(ds))
    mu.save_pretrained(str(du))
    a, b = _st_load(ds), _st_load(du)
    assert list(a) == list(b) and len(a) == 4 * G.E and all(torch.equal(a[k], b[k]) for k in a)
    assert not any(STACK in k for k in a)
    # Base for a process without Unsloth: plain Linear experts with the dequantized NF4 weights.
    ex = experts_of(mu)
    weights = {}
    for lname in ("gate_up_projs", "down_projs"):
        for e, p in enumerate(getattr(ex, lname)):
            bl = p.base_layer
            weights[f"experts.{lname}.{e}.weight"] = bnb.functional.dequantize_4bit(bl.weight.data, bl.weight.quant_state).float().cpu()
            weights[f"experts.{lname}.{e}.bias"] = bl.bias.detach().float().cpu()
    torch.manual_seed(0)
    torch.save({"E": G.E, "H": G.H, "I": G.I, "weights": weights, "x": torch.randn(5, G.H)}, tmp_path / "base.pt")
    script = tmp_path / "plain.py"
    script.write_text(PLAIN_SCRIPT)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, str(script), str(tmp_path / "base.pt"), str(tmp_path / "res.json"), str(ds), str(du)],
                       env = env, cwd = str(tmp_path), capture_output = True, text = True)
    assert p.returncode == 0, p.stderr[-4000:]
    res = json.load(open(tmp_path / "res.json"))
    assert res == {"unsloth": False, "equal": True, "lora_applied": True}, res
    ls, lu = torch.load(ds / "plain_loaded.pt"), torch.load(du / "plain_loaded.pt")
    assert ls.keys() == lu.keys() and len(ls) == 4 * G.E and all(torch.equal(ls[k], lu[k]) for k in ls)
    ref = {k: v for k, v in expert_lora(ms).items()}
    assert torch.equal(ls["base_model.model.experts.down_projs.3.lora_B.default.weight"].cuda().to(ref[("down_projs", 3, "B")][0].dtype),
                       ref[("down_projs", 3, "B")][0])

    # And back into a stacked model (load hooks write the stack slices).
    m2 = build("bf16", seed = 99)
    stack(m2)
    assert n_stacked(m2) == 4
    res2 = set_peft_model_state_dict(m2, a)
    assert not [k for k in res2.unexpected_keys if "lora_" in k]
    assert_lora_equal(m2, mu, grads = False)


@needs_bf16
def test_merge_and_unload_equal_and_no_stacks():
    ms, mu = pair("bf16")
    x, idx, w, _ = _inputs("bf16", torch.bfloat16)
    ms, mu = ms.merge_and_unload(), mu.merge_and_unload()
    assert not any(STACK in n or "lora_" in n for n, _ in ms.named_parameters())
    for a, b in zip(ms.experts.gate_up_projs, mu.experts.gate_up_projs):
        assert type(a) is type(b) and torch.equal(a.weight.data, b.weight.data)
    with torch.no_grad():
        assert torch.equal(ms(x, idx, w), mu(x, idx, w))


@needs_bf16
def test_adapter_disable_add_switch_delete():
    ms, mu = pair("bf16")
    x, idx, w, _ = _inputs("bf16", torch.bfloat16)

    def same():
        with torch.no_grad():
            a, b = ms(x, idx, w), mu(x, idx, w)
        assert torch.equal(a, b)
        return a

    y0 = same()
    with ms.disable_adapter(), mu.disable_adapter():
        yb = same()
    assert not torch.equal(y0, yb)
    for m in (ms, mu):
        m.add_adapter("other", LoraConfig(r = 8, lora_alpha = 16, target_modules = TARGETS, lora_dropout = 0.0))
        _randomize(m, 11, adapter = "other")
        m.set_adapter("other")
    before = gq.CALLS["forward_lora"]
    y1 = same()
    assert gq.CALLS["forward_lora"] == before + 2 and not torch.equal(y1, y0)
    for m in (ms, mu):
        m.set_adapter("default")
    assert torch.equal(same(), y0)
    for m in (ms, mu):
        m.delete_adapter("other")
    assert torch.equal(same(), y0) and n_stacked(ms) == 4


# ----------------------------------------------------------------------------- routed decode
def _routed_pair():
    ms, mu = pair("bf16")
    return experts_of(ms.eval()), experts_of(mu.eval())


@pytest.mark.parametrize("T", [1, 4])
def test_routed_decode_reads_the_stacks(T):
    es, eu = _routed_pair()
    g = torch.Generator(device = "cuda").manual_seed(T)
    x = torch.randn(1, T, G.H, device = "cuda", generator = g).to(G.DT)
    idx, w = G._routing(T, seed = T)

    def both():
        with torch.no_grad():
            a, b = gr.routed_experts_forward(es, x, idx, w), gr.routed_experts_forward(eu, x, idx, w)
        assert a is not None and b is not None   # routed, not the dense fallback
        assert torch.equal(a, b)
        return a

    y0 = both()
    for projs in (es.gate_up_projs, es.down_projs):
        assert projs._unsloth_routed_lora.get("stacks") is not None   # the stacked reader ran
    with torch.no_grad():
        dense = G.torch_native_forward(es, x, idx, w)
        dense_u = G.torch_native_forward(eu, x, idx, w)
    assert torch.equal(dense, dense_u)
    # An optimizer-style in-place step reaches the kernels; a storage swap (.data =) too.
    with torch.no_grad():
        for projs_s, projs_u in ((es.gate_up_projs, eu.gate_up_projs), (es.down_projs, eu.down_projs)):
            getattr(projs_s[0].lora_B["default"], STACK).add_(0.05)
            for p in projs_u:
                p.lora_B["default"].weight.add_(0.05)
    y1 = both()
    assert not torch.equal(y0, y1)
    for projs_s, projs_u in ((es.gate_up_projs, eu.gate_up_projs), (es.down_projs, eu.down_projs)):
        st = getattr(projs_s[0].lora_A["default"], STACK)
        st.data = st.data * 1.5
        for p in projs_u:
            p.lora_A["default"].weight.data = p.lora_A["default"].weight.data * 1.5
    assert not torch.equal(both(), y1)


def test_routed_decode_compile_fullgraph_and_cuda_graph_replay():
    es, eu = _routed_pair()
    T = 4
    x = torch.randn(1, T, G.H, device = "cuda", dtype = torch.float32)
    idx, w = G._routing(T, seed = 1)
    idx2, w2 = G._routing(T, seed = 2)
    with torch.no_grad():
        ref1 = gr.routed_experts_forward(eu, x, idx, w)
        ref2 = gr.routed_experts_forward(eu, x, idx2, w2)
        assert torch.equal(gr.routed_experts_forward(es, x, idx, w), ref1)
        torch._dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
        compiled = torch.compile(lambda a, b, c: gr.routed_experts_forward(es, a, b, c), fullgraph = True)
        torch.testing.assert_close(compiled(x, idx, w), ref1, rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(compiled(x, idx2, w2), ref2, rtol = 1e-5, atol = 1e-5)
        assert sum(counters["graph_break"].values()) == 0

        static = [x.clone(), idx.clone(), w.clone()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            gr.routed_experts_forward(es, *static)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = gr.routed_experts_forward(es, *static)
        static[1].copy_(idx2)
        static[2].copy_(w2)
        graph.replay()
        assert torch.equal(out, ref2)
        static[1].copy_(idx)
        static[2].copy_(w)
        graph.replay()
        assert torch.equal(out, ref1)


# ----------------------------------------------------------------------------- readiness / declines
@needs_bf16
def test_ready_signature_tracks_the_stack():
    """The readiness cache keys on the stack: an in-place cast (same Parameter object) re-checks."""
    ms, _ = pair("bf16")
    ex = experts_of(ms)
    assert ex._grouped_bnb4bit_ready()
    sig = gq.ready_signature(ex)
    st = getattr(ex.down_projs[0].lora_A["default"], STACK)
    st.data = st.data.to(torch.float64)
    assert gq.ready_signature(ex) != sig
    assert not ex._grouped_bnb4bit_ready()   # "lora_A / lora_B dtypes differ"


@needs_bf16
def test_enable_is_idempotent():
    ms, _ = pair("bf16")
    before = dict(ms.named_parameters())
    stack(ms)
    after = dict(ms.named_parameters())
    assert before.keys() == after.keys() and all(before[k] is after[k] for k in before)


def _decline_model(case, monkeypatch):
    """(model, expected #stacks) for a setup the stacking must not (fully) convert."""
    if case == "dropout":
        return build(lora_dropout = 0.1), 0
    if case == "two_adapters":
        m = build()
        m.add_adapter("other", LoraConfig(r = 8, lora_alpha = 16, target_modules = TARGETS, lora_dropout = 0.0))
        return m, 0
    if case == "dora":
        return build(use_dora = True), 0
    if case == "lora_bias":
        return build(lora_bias = True), 0
    if case == "existing_grads":
        m = build()
        p = experts_of(m).down_projs[3].lora_A["default"].weight
        p.grad = torch.zeros_like(p)
        return m, 2   # gate_up still stacks
    if case == "frozen_expert":
        m = build()
        experts_of(m).gate_up_projs[5].lora_B["default"].weight.requires_grad_(False)
        return m, 2
    if case == "kill_switch":
        monkeypatch.setenv("UNSLOTH_MOE_STACKED_LORA", "0")
        return build(), 0
    if case == "loop_path":
        monkeypatch.setenv("UNSLOTH_GPTOSS_GROUPED", "0")
        return build(), 0
    if case == "compile_disabled":
        monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
        return build(), 0
    if case == "bf16_base":
        return build(base = "bf16"), 0
    if case == "no_bf16_backend":
        if not G.FP16_OK:
            pytest.skip("needs the fp16 backend so readiness passes without torch._grouped_mm")
        monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: False)
        return build(), 0
    raise KeyError(case)


@needs_bf16
@pytest.mark.parametrize("case", ["dropout", "two_adapters", "dora", "lora_bias", "existing_grads", "frozen_expert",
                                  "kill_switch", "loop_path", "compile_disabled", "bf16_base", "no_bf16_backend"])
def test_declines_keep_per_expert_parameters(case, monkeypatch):
    model, expected = _decline_model(case, monkeypatch)
    ref = dict(model.named_parameters())
    with _env(**({} if case == "kill_switch" else {"UNSLOTH_MOE_STACKED_LORA": "1"})):
        ML.auto_enable_grouped_moe(model)
    assert n_stacked(model) == expected
    if expected == 0:
        assert dict(model.named_parameters()).keys() == ref.keys()
        assert all(ref[n] is p for n, p in model.named_parameters())


@needs_bf16
def test_stacking_is_loader_only():
    """enable_grouped_moe (no loader) and gpt-oss experts without LoRA stack nothing."""
    m = build()
    ML.enable_grouped_moe(m, verbose = False)
    ML.enable_grouped_moe(m, verbose = False, stack_lora = True)   # ModuleList MoE blocks only
    assert n_stacked(m) == 0
    plain = _Model(G._Experts(True))
    assert gq.stack_expert_lora(plain) == 0
