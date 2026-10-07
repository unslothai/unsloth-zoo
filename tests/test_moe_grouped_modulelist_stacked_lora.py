"""Stacked expert LoRA Parameters on the transformers<5 ModuleList grouped MoE path.

enable_grouped_moe turns PEFT's per-expert lora_A / lora_B Parameters of a patched block into one
[E, r, in] / [E, out, r] Parameter per projection (UNSLOTH_MOE_STACKED_LORA=0 keeps them). The
stacked model is checked BITWISE against the same model with the kill switch set (forward, dX,
LoRA grads, torch / bnb optimizer steps, GradScaler + accumulation, compile), and for PEFT /
Trainer / DDP / Unsloth compatibility: saved adapters keep PEFT's per-expert keys and load in a
process without Unsloth, merges drop the stacks, resume continues the same loss curve.
"""
import contextlib
import copy
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moe_grouped_modulelist_lora as L  # noqa: E402
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML  # noqa: E402

peft = pytest.importorskip("peft")
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

DEV = L.DEV
pytestmark = L.pytestmark
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse = True)
def _clean_env():
    keys = ("UNSLOTH_MOE_STACKED_LORA", "UNSLOTH_MOE_GROUPED_LORA")
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
    old = {k: getattr(L, k) for k in kv}
    for k, v in kv.items():
        setattr(L, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(L, k, v)


def _enable(model, stacked):
    with _env(UNSLOTH_MOE_STACKED_LORA = "1" if stacked else "0"):
        return ML.enable_grouped_moe(model, verbose = False, stack_lora = True)


def pair(kind = "qwen3", base = "bf16", r = 8, targets = "all", **kw):
    """(stacked model, block, unstacked model, block): one build, deep-copied before enabling."""
    model, blk = L.build(kind, base = base, r = r, targets = targets, **kw)
    model_u = copy.deepcopy(model)
    assert _enable(model, True) == 1 and _enable(model_u, False) == 1
    return model, blk, model_u, model_u.base_model.model.mlp


def n_stacked(model):
    return sum(1 for n, _ in model.named_parameters() if n.endswith("." + ML._STACK_NAME))


def expert_lora(blk):
    """{(expert, proj, "A"/"B"): (weight, grad)} as PEFT's per-expert view."""
    names = L.KINDS_BY_CLS[type(blk).__name__][:3]
    out = {}
    for e, ex in enumerate(blk.experts):
        for n in names:
            p = getattr(ex, n)
            if not hasattr(p, "lora_A"):
                continue
            a = p.active_adapters[0]
            for kind in ("A", "B"):
                m = getattr(p, "lora_" + kind)[a]
                if type(m) is ML._StackedLoraLinear:
                    st = getattr(m._unsloth_stack_owner, ML._STACK_NAME)
                    g = None if st.grad is None else st.grad[m._unsloth_stack_index]
                else:
                    g = m.weight.grad
                out[(e, n, kind)] = (m.weight.detach(), g)
    return out


def fwd_bwd(blk, x, gout, autocast = False):
    xi = x.clone().requires_grad_(True)
    ctx = torch.autocast(DEV, dtype = L.DT) if autocast else contextlib.nullcontext()
    with ctx:
        out = blk(xi)[0]
    (out.float() * gout).sum().backward()
    return out.detach(), xi.grad


def assert_lora_equal(blk_s, blk_u, grads = True):
    s, u = expert_lora(blk_s), expert_lora(blk_u)
    assert s.keys() == u.keys() and s
    for k in s:
        assert torch.equal(s[k][0], u[k][0]), f"weight {k}"
        if grads:
            assert torch.equal(s[k][1], u[k][1]), f"grad {k}"


def _inputs(seed = 1, t = None):
    torch.manual_seed(seed)
    t = t or L.T
    return (torch.randn(1, t, L.H, device = DEV, dtype = L.DT),
            torch.randn(1, t, L.H, device = DEV, dtype = torch.float32))


# ----------------------------------------------------------------------------- bitwise parity
@pytest.mark.parametrize("kind", list(L.KINDS))
@pytest.mark.parametrize("base", ["bf16", "nf4"])
@pytest.mark.parametrize("r", [4, 8, 16])
def test_stacked_matches_unstacked_bitwise(kind, base, r):
    ms, bs, mu, bu = pair(kind, base, r)
    assert n_stacked(ms) == 6 and n_stacked(mu) == 0
    trainable = lambda m: sum(p.requires_grad for p in m.parameters())
    assert trainable(ms) == 6 and trainable(mu) == 6 * L.E
    ms.train(); mu.train()
    x, g = _inputs()
    before = ML.CALLS["grouped_lora"]
    os_, dxs = fwd_bwd(bs, x, g)
    ou, dxu = fwd_bwd(bu, x, g)
    assert ML.CALLS["grouped_lora"] == before + 2
    assert torch.equal(os_, ou) and torch.equal(dxs, dxu)
    assert_lora_equal(bs, bu)


@pytest.mark.parametrize("kind", ["qwen3", "mixtral"])
@pytest.mark.parametrize("targets", ["gate", "gate_down", "down"])
def test_stacked_subsets_and_autocast(kind, targets):
    ms, bs, mu, bu = pair(kind, "nf4", 8, targets)
    assert n_stacked(ms) == 2 * {"gate": 1, "gate_down": 2, "down": 1}[targets]
    x, g = _inputs()
    for ac in (False, True):
        for m in (ms, mu):
            m.zero_grad(set_to_none = True)
        os_, dxs = fwd_bwd(bs, x, g, autocast = ac)
        ou, dxu = fwd_bwd(bu, x, g, autocast = ac)
        assert torch.equal(os_, ou) and torch.equal(dxs, dxu)
        assert_lora_equal(bs, bu)


def test_grouped_path_reads_the_stacks():
    """_lora_operands takes the stacked Parameters: no torch.stack over the experts' views and one
    AccumulateGrad per stack in the backward graph (the unstacked graph has 6 * E of each)."""
    ms, bs, mu, bu = pair("qwen3", "bf16", 8)
    spec = bs._unsloth_moe_spec
    projs = [getattr(ex, spec[0]) for ex in bs.experts]
    A, B = ML._lora_stacks(projs, "default")
    assert A is getattr(projs[0].lora_A["default"], ML._STACK_NAME)
    assert ML._lora_stacks([getattr(ex, spec[0]) for ex in bu.experts], "default") is None
    x, _ = _inputs()

    def nodes(blk):
        out = blk(x)[0]
        seen, todo, names = set(), [out.grad_fn], []
        while todo:
            fn = todo.pop()
            if fn is None or fn in seen:
                continue
            seen.add(fn)
            names.append(type(fn).__name__)
            todo.extend(f for f, _ in fn.next_functions)
        return names

    s, u = nodes(bs), nodes(bu)
    assert s.count("AccumulateGrad") == 6 and u.count("AccumulateGrad") == 6 * L.E
    assert s.count("StackBackward0") == 0 and u.count("StackBackward0") == 6
    assert s.count("SelectBackward0") == 0


def test_lora_stacks_checks_members_and_order():
    """A projection list that is not the one the stack was built over (reordered, partial) keeps
    torch.stack over the per-expert views: same values as the unstacked model."""
    ms, bs, mu, bu = pair("qwen3", "bf16", 8)
    spec = bs._unsloth_moe_spec
    ps = [getattr(ex, spec[2]) for ex in bs.experts][::-1]
    pu = [getattr(ex, spec[2]) for ex in bu.experts][::-1]
    assert ML._lora_stacks(ps, "default") is None
    assert ML._lora_stacks(ps[:-1], "default") is None
    for a, b in zip(ML._lora_operands(ps, "default", L.DT), ML._lora_operands(pu, "default", L.DT)):
        assert torch.equal(a, b)


def test_enable_is_idempotent():
    ms, bs, mu, bu = pair("qwen3", "bf16", 8)
    before = {n: p for n, p in ms.named_parameters()}
    assert _enable(ms, True) == 1
    after = {n: p for n, p in ms.named_parameters()}
    assert before.keys() == after.keys() and all(before[k] is after[k] for k in before)


def test_kill_switch_keeps_per_expert_parameters():
    model, blk = L.build("qwen3", base = "bf16", r = 8)
    with _env(UNSLOTH_MOE_STACKED_LORA = "0"):
        assert ML.enable_grouped_moe(model, verbose = False, stack_lora = True) == 1
    assert n_stacked(model) == 0
    assert all(type(m) is nn.Linear for n, m in model.named_modules() if n.endswith((".lora_A.default", ".lora_B.default")))


def _decline_model(case):
    if case == "two_adapters":
        model, blk = L.build("qwen3", base = "bf16", r = 8)
        model.add_adapter("other", LoraConfig(r = 8, lora_alpha = 16, target_modules = ["gate_proj", "up_proj", "down_proj"]))
        model.set_adapter("default")
        return model, blk, True          # grouped path runs, the block is not stacked
    if case == "frozen_expert":
        model, blk = L.build("qwen3", base = "bf16", r = 8)
        blk.experts[3].up_proj.lora_A["default"].weight.requires_grad_(False)
        return model, blk, True
    if case == "dora":
        model, blk = L.build("qwen3", base = "bf16", r = 8, use_dora = True)
        return model, blk, False
    if case == "dropout_train":
        model, blk = L.build("qwen3", base = "bf16", r = 8, lora_dropout = 0.1)
        model.train()
        return model, blk, False
    raise KeyError(case)


@pytest.mark.parametrize("case", ["two_adapters", "frozen_expert", "dora", "dropout_train"])
def test_declines_keep_per_expert_parameters(case):
    model, blk, patched = _decline_model(case)
    ref = {n: p for n, p in model.named_parameters()}
    n = ML.enable_grouped_moe(model, verbose = False, stack_lora = True)
    assert n == int(patched)
    assert n_stacked(model) == (2 * 2 if case == "frozen_expert" else 0)   # gate / down still stack
    if case == "frozen_expert":
        assert blk.experts[3].up_proj.lora_A["default"].weight is ref["base_model.model.mlp.experts.3.up_proj.lora_A.default.weight"]
    else:
        assert {n for n, _ in model.named_parameters()} == ref.keys()


def test_signature_tracks_the_stack_dtype():
    """The readiness cache keys on the stack: an in-place cast (same Parameter object) changes it."""
    ms, bs, mu, bu = pair("qwen3", "bf16", 8)
    k0 = ML._ready_signature(bs.experts, bs._unsloth_moe_spec)[0]
    st = getattr(bs.experts[0].gate_proj.lora_A["default"], ML._STACK_NAME)
    st.data = st.data.to(torch.bfloat16)
    assert ML._ready_signature(bs.experts, bs._unsloth_moe_spec)[0] != k0
    assert "dtypes differ" in ML._experts_grouped_state(bs.experts, bs._unsloth_moe_spec, x_dev(), L.DT)


def x_dev():
    return torch.device(DEV, torch.cuda.current_device()) if DEV == "cuda" else torch.device(DEV)


def test_per_expert_weight_is_a_live_view():
    """In-place init through the property writes the stack; assigning a Parameter raises."""
    ms, bs, mu, bu = pair("qwen3", "bf16", 8)
    m = bs.experts[5].down_proj.lora_B["default"]
    with torch.no_grad():
        nn.init.constant_(m.weight, 0.25)
    st = getattr(bs.experts[0].down_proj.lora_B["default"], ML._STACK_NAME)
    assert torch.all(st[5] == 0.25) and not torch.all(st[4] == 0.25)
    with pytest.raises((KeyError, AttributeError)):
        m.weight = nn.Parameter(torch.zeros_like(m.weight))


# ----------------------------------------------------------------------------- optimizers
def _opt_steps(model, blk, make_opt, steps = 3, ga = 1, scaler = None, set_to_none = True):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_opt(params)
    for s in range(steps):
        for a in range(ga):
            x, g = _inputs(seed = 10 + s * ga + a)
            ctx = torch.autocast(DEV, dtype = torch.float16) if scaler is not None else contextlib.nullcontext()
            xi = x.clone().requires_grad_(True)
            with ctx:
                out = blk(xi)[0]
            loss = (out.float() * g).sum() / ga
            (scaler.scale(loss) if scaler is not None else loss).backward()
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none = set_to_none)
    return opt


def _bnb_state_per_expert(opt, model):
    """{(param name, expert)}: per-expert (state1, state2, absmax1, absmax2) of a bnb 8-bit optimizer."""
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        st = opt.state[p]
        if n.endswith("." + ML._STACK_NAME):
            E = p.shape[0]
            for e in range(E):
                key = n.replace(".experts.0.", f".experts.{e}.").replace(ML._STACK_NAME, "weight")
                out[key] = tuple(st[k].view(E, -1)[e] for k in ("state1", "state2", "absmax1", "absmax2"))
        else:
            out[n] = tuple(st[k].reshape(-1) for k in ("state1", "state2", "absmax1", "absmax2"))
    return out


@pytest.mark.parametrize("optim", ["adamw_foreach", "adamw_8bit"])
def test_optimizer_steps_bitwise(optim):
    """3 steps: weights equal bitwise; bnb's 8-bit states too. Every per-expert tensor here is a
    whole number of 256-element blocks and >= min_8bit_size (4096), as on Qwen3-30B-A3B r >= 8."""
    if optim == "adamw_8bit":
        bnb = pytest.importorskip("bitsandbytes")
        if not ML.HAS_BNB:
            pytest.skip("bitsandbytes not usable")
        make = lambda ps: bnb.optim.AdamW8bit(ps, lr = 1e-2, weight_decay = 0.01)
    else:
        make = lambda ps: torch.optim.AdamW(ps, lr = 1e-2, weight_decay = 0.01, foreach = True)
    with _shapes(H = 256, I = 256):
        ms, bs, mu, bu = pair("qwen3", "nf4", 16)
        os_ = _opt_steps(ms, bs, make)
        ou = _opt_steps(mu, bu, make)
        assert_lora_equal(bs, bu, grads = False)
        if optim == "adamw_8bit":
            s, u = _bnb_state_per_expert(os_, ms), _bnb_state_per_expert(ou, mu)
            assert s.keys() == u.keys()
            for k in s:
                assert all(torch.equal(a, b) for a, b in zip(s[k], u[k])), k


@pytest.mark.parametrize("shape", ["unaligned", "small"])
def test_bnb_8bit_block_layout_differs_but_stays_close(shape):
    """Documented, not declined: when a per-expert tensor is not a whole number of 256-element
    blocks ("unaligned", in-features 264) a quantization block of the stack spans two experts, and
    under min_8bit_size ("small", r=4 x 256 = 1024) PEFT's per-expert tensors get 32-bit states while
    the stack gets 8-bit ones. Same optimizer, different block partition: close, not bitwise."""
    bnb = pytest.importorskip("bitsandbytes")
    if not ML.HAS_BNB:
        pytest.skip("bitsandbytes not usable")
    make = lambda ps: bnb.optim.AdamW8bit(ps, lr = 1e-3)
    shapes = dict(H = 264, I = 256) if shape == "unaligned" else dict(H = 256, I = 256)
    with _shapes(**shapes):
        ms, bs, mu, bu = pair("qwen3", "bf16", 16 if shape == "unaligned" else 4)
        w0 = {k: v[0].clone() for k, v in expert_lora(bs).items()}
        _opt_steps(ms, bs, make)
        _opt_steps(mu, bu, make)
        s, u = expert_lora(bs), expert_lora(bu)
        step = torch.cat([(u[k][0] - w0[k]).flatten() for k in s])
        diff = torch.cat([(s[k][0] - u[k][0]).flatten() for k in s])
        assert (diff.norm() / step.norm()).item() < 0.05


@pytest.mark.parametrize("set_to_none", [True, False])
def test_gradscaler_fp16_with_accumulation(set_to_none):
    """fp16 base and input, GradScaler, 2 accumulation micro-steps, both zero_grad modes."""
    old = L.DT
    L.DT = torch.float16
    try:
        ms, bs, mu, bu = pair("qwen3", "bf16", 8)
        make = lambda ps: torch.optim.AdamW(ps, lr = 1e-2, foreach = True)
        _opt_steps(ms, bs, make, steps = 2, ga = 2, scaler = torch.amp.GradScaler(DEV), set_to_none = set_to_none)
        _opt_steps(mu, bu, make, steps = 2, ga = 2, scaler = torch.amp.GradScaler(DEV), set_to_none = set_to_none)
        assert_lora_equal(bs, bu, grads = not set_to_none)
    finally:
        L.DT = old


def test_compile_fullgraph_no_breaks():
    ms, bs, mu, bu = pair("qwen3", "bf16", 8, prefer_hf = False)
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    x, g = _inputs(seed = 2)
    before = ML.CALLS["grouped_lora"]
    cs = torch.compile(bs.forward, fullgraph = True)
    os_, dxs = fwd_bwd(cs, x, g)
    assert ML.CALLS["grouped_lora"] == before + 1
    assert sum(counters["graph_break"].values()) == 0
    ou, dxu = fwd_bwd(bu, x, g)
    assert (os_.float() - ou.float()).abs().max() < 2e-2
    s, u = expert_lora(bs), expert_lora(bu)
    gs = torch.cat([s[k][1].flatten() for k in s]); gu = torch.cat([u[k][1].flatten() for k in s])
    assert ((gs - gu).norm() / gu.norm()).item() < 2e-2


# ----------------------------------------------------------------------------- PEFT / HF compatibility
def _tiny_cfg():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
    return Qwen3MoeConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 128,
                          num_experts = 8, num_experts_per_tok = 2, num_hidden_layers = 2, num_attention_heads = 4,
                          num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 128,
                          norm_topk_prob = True, mlp_only_layers = [], decoder_sparse_step = 1)


TARGETS = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]


def _tiny_base(seed = 0):
    from transformers import Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(_tiny_cfg()).to(DEV, torch.bfloat16)
    if not isinstance(model.model.layers[0].mlp.experts, nn.ModuleList):
        pytest.skip("experts are not an nn.ModuleList on this transformers")
    return model


def _tiny_peft(stacked, seed = 0, base = None):
    model = get_peft_model(base if base is not None else _tiny_base(seed),
                           LoraConfig(r = 8, lora_alpha = 16, target_modules = TARGETS))
    with torch.no_grad():   # non-zero lora_B so every check sees the adapter
        torch.manual_seed(seed + 1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    assert _enable(model, stacked) == 2
    return model


def _ids(seed = 5, b = 2, t = 24):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, t), generator = g).to(DEV)


@pytest.fixture(scope = "module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_base")
    _tiny_base().save_pretrained(d)
    return str(d)


def _st_load(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(path, "adapter_model.safetensors"))


def _plain_logits(tiny_dir, adapter_dir, out_path):
    """Logits from a fresh process with transformers + peft only (no Unsloth, no zoo)."""
    code = textwrap.dedent(f"""
        import sys, torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained({tiny_dir!r}, torch_dtype = torch.bfloat16).to({DEV!r})
        m = PeftModel.from_pretrained(m, {adapter_dir!r})
        g = torch.Generator().manual_seed(5)
        ids = torch.randint(0, 256, (2, 24), generator = g).to({DEV!r})
        with torch.no_grad():
            out = m(input_ids = ids).logits
        torch.save(out.cpu(), {out_path!r})
        bad = [k for k in sys.modules if k.startswith(("unsloth", "unsloth_zoo"))]
        assert not bad, bad
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, "-c", code], env = env, cwd = "/", capture_output = True, text = True)
    assert p.returncode == 0, p.stderr[-3000:]
    return torch.load(out_path)


def test_save_pretrained_keys_values_and_plain_reload(tiny_dir, tmp_path):
    """(a) The stacked model saves PEFT's per-expert keys / shapes / values, bitwise as unstacked;
    a fresh plain transformers + peft process loads it to the logits of this model's loop."""
    from transformers import AutoModelForCausalLM
    base = lambda: AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype = torch.bfloat16).to(DEV)
    torch.manual_seed(0)
    ms = _tiny_peft(True, base = base())
    torch.manual_seed(0)
    mu = _tiny_peft(False, base = base())
    assert n_stacked(ms) == 2 * 3 * 2 and n_stacked(mu) == 0
    ms.save_pretrained(tmp_path / "s")
    mu.save_pretrained(tmp_path / "u")
    s, u = _st_load(tmp_path / "s"), _st_load(tmp_path / "u")
    assert list(s) == list(u) and all(s[k].shape == u[k].shape and torch.equal(s[k], u[k]) for k in s)
    assert not any(ML._STACK_NAME in k for k in s)
    assert "base_model.model.model.layers.0.mlp.experts.7.down_proj.lora_B.weight" in s
    # This process, loop (grouped off), through the per-expert views == per-expert Parameters.
    ML.disable_grouped_moe(ms)
    ML.disable_grouped_moe(mu)
    with torch.no_grad():
        mine = ms(input_ids = _ids()).logits.float().cpu()
        assert torch.equal(mine, mu(input_ids = _ids()).logits.float().cpu())
    plain_s = _plain_logits(tiny_dir, str(tmp_path / "s"), str(tmp_path / "logits_s.pt"))
    plain_u = _plain_logits(tiny_dir, str(tmp_path / "u"), str(tmp_path / "logits_u.pt"))
    assert torch.equal(plain_s, plain_u)
    # The pytest process imported unsloth (conftest), whose kernels round differently from stock.
    assert ((plain_s.float() - mine).norm() / mine.norm()).item() < 2e-2


def test_load_then_stack(tiny_dir, tmp_path):
    """(b) PeftModel.from_pretrained of a saved adapter, then enable: stacked values equal the file,
    and the loaded + stacked model runs bitwise as the loaded + unstacked one."""
    from transformers import AutoModelForCausalLM
    base = lambda: AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype = torch.bfloat16).to(DEV)
    torch.manual_seed(0)
    _tiny_peft(False, base = base()).save_pretrained(tmp_path / "a")
    ref = _st_load(tmp_path / "a")
    ms = PeftModel.from_pretrained(base(), tmp_path / "a", is_trainable = True)
    mu = PeftModel.from_pretrained(base(), tmp_path / "a", is_trainable = True)
    assert _enable(ms, True) == 2 and _enable(mu, False) == 2
    assert n_stacked(ms) == 12
    sd = {k.replace(".default", ""): v for k, v in ms.state_dict().items() if "lora_" in k}
    assert sd.keys() == ref.keys() and all(torch.equal(sd[k].cpu(), ref[k].cpu().to(sd[k].dtype)) for k in ref)
    ids = _ids()
    outs = []
    for m in (ms, mu):
        m.train()
        loss = m(input_ids = ids, labels = ids).loss
        loss.backward()
        outs.append((loss.detach(), {k: v for k, v in _expert_grads(m).items()}))
    assert torch.equal(outs[0][0], outs[1][0])
    assert outs[0][1].keys() == outs[1][1].keys() and all(torch.equal(outs[0][1][k], outs[1][1][k]) for k in outs[0][1])


def _expert_grads(model):
    out = {}
    for name, mod in model.named_modules():
        if name.endswith((".lora_A.default", ".lora_B.default")):
            if type(mod) is ML._StackedLoraLinear:
                st = getattr(mod._unsloth_stack_owner, ML._STACK_NAME)
                out[name] = st.grad[mod._unsloth_stack_index]
            else:
                out[name] = mod.weight.grad
    return out


def test_load_state_dict_strict_round_trip():
    """A stacked model's state_dict (per-expert keys) loads strictly into a stacked and an
    unstacked model; a missing per-expert key is reported under its PEFT name."""
    torch.manual_seed(0)
    ms = _tiny_peft(True, seed = 3)
    mu = _tiny_peft(False, seed = 4)
    sd = ms.state_dict()
    mu.load_state_dict(sd, strict = True)
    ms2 = _tiny_peft(True, seed = 4)
    res = ms2.load_state_dict(sd, strict = True)
    assert not res.missing_keys and not res.unexpected_keys
    su, s2 = mu.state_dict(), ms2.state_dict()
    assert list(su) == list(s2) and all(torch.equal(su[k], s2[k]) for k in su)
    key = "base_model.model.model.layers.1.mlp.experts.6.up_proj.lora_A.default.weight"
    sd.pop(key)
    res = ms2.load_state_dict(sd, strict = False)
    assert res.missing_keys == [key], res.missing_keys


def test_deepcopy_keeps_stacks_private():
    """A deep copy gets its own stacks; the copy's per-expert views read the copy's stack."""
    torch.manual_seed(0)
    ms = _tiny_peft(True)
    mc = copy.deepcopy(ms)
    a = mc.base_model.model.model.layers[0].mlp.experts[3].gate_proj.lora_B["default"]
    assert a._unsloth_stack_owner is mc.base_model.model.model.layers[0].mlp.experts[0].gate_proj.lora_B["default"]
    with torch.no_grad():
        a.weight.add_(1.0)
    s0, s1 = ms.state_dict(), mc.state_dict()
    key = "base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_B.default.weight"
    assert torch.equal(s1[key], s0[key] + 1.0)
    assert all(torch.equal(s0[k], s1[k]) for k in s0 if k != key)


def test_merge_and_unload_drops_stacks():
    """(c) merge_and_unload equals the unstacked merge; no stack is left behind."""
    torch.manual_seed(0)
    ms, mu = _tiny_peft(True), _tiny_peft(False)
    a, b = ms.merge_and_unload(), mu.merge_and_unload()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not any(ML._STACK_NAME in n or "lora_" in n for n, _ in a.named_parameters())
    assert not any(ML._STACK_NAME in k for k in sa)


def test_disable_set_and_second_adapter():
    """(e) disable_adapter(), a second adapter, set_adapter back and forth: stacked == unstacked."""
    torch.manual_seed(0)
    ms, mu = _tiny_peft(True), _tiny_peft(False)
    ids = _ids()

    def logits(m):
        with torch.no_grad():
            return m(input_ids = ids).logits

    with ms.disable_adapter(), mu.disable_adapter():
        assert torch.equal(logits(ms), logits(mu))
    assert torch.equal(logits(ms), logits(mu))
    for m in (ms, mu):
        torch.manual_seed(7)
        m.add_adapter("second", LoraConfig(r = 4, lora_alpha = 8, target_modules = TARGETS))
        with torch.no_grad():
            torch.manual_seed(8)
            for n, p in m.named_parameters():
                if "lora_B.second" in n:
                    p.normal_(0, 0.05)
        _enable(m, m is ms)   # re-entrant enable: "second" stays per-expert (two adapters)
    assert n_stacked(ms) == 12
    for name in ("second", "default", "second"):
        ms.set_adapter(name); mu.set_adapter(name)
        assert torch.equal(logits(ms), logits(mu)), name
    ms.base_model.set_adapter(["default", "second"]); mu.base_model.set_adapter(["default", "second"])
    assert torch.equal(logits(ms), logits(mu))
    ms.set_adapter("default"); mu.set_adapter("default")
    ms.delete_adapter("second"); mu.delete_adapter("second")
    assert torch.equal(logits(ms), logits(mu)) and n_stacked(ms) == 12
    for m in (ms, mu):   # the stacked adapter goes with its modules
        m.add_adapter("third", LoraConfig(r = 4, lora_alpha = 8, target_modules = TARGETS))
        m.delete_adapter("default")
    assert n_stacked(ms) == 0 and not any(ML._STACK_NAME in k for k in ms.state_dict())


def test_hf_trainer_resume(tmp_path):
    """(d) Trainer checkpoint + resume_from_checkpoint continues the uninterrupted loss curve
    (stacked), and with clipping off the stacked and unstacked curves are equal."""
    from transformers import Trainer, TrainingArguments

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            ids = torch.randint(0, 256, (24,), generator = g)
            return {"input_ids": ids, "labels": ids}

    class T(Trainer):
        def training_step(self, *a, **k):
            loss = super().training_step(*a, **k)
            self.exact.append(loss.item())   # Trainer's log rounds to 4 digits
            return loss

    def run(stacked, out, max_grad_norm = 1.0, resume = None, optim = "adamw_torch"):
        model = _tiny_peft(stacked)
        args = TrainingArguments(output_dir = str(out), max_steps = 6, save_steps = 3, logging_steps = 1,
                                 per_device_train_batch_size = 4, gradient_accumulation_steps = 2,
                                 learning_rate = 1e-2, bf16 = True, report_to = [], seed = 3,
                                 max_grad_norm = max_grad_norm, optim = optim, save_safetensors = True,
                                 dataloader_num_workers = 0, disable_tqdm = True)
        _ = args.device
        args._n_gpu = 1   # no DataParallel when several GPUs are visible
        tr = T(model = model, args = args, train_dataset = DS())
        tr.exact = []
        tr.train(resume_from_checkpoint = resume)
        weights = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
        return tr.exact, weights

    for optim in ("adamw_torch", "adamw_8bit"):
        full, wf = run(True, tmp_path / f"full_{optim}", optim = optim)
        resumed, wr = run(True, tmp_path / f"res_{optim}", optim = optim,
                          resume = str(tmp_path / f"full_{optim}" / "checkpoint-3"))
        assert len(full) == 12 and len(resumed) == 6
        assert resumed == full[6:], (optim, full, resumed)
        assert wf.keys() == wr.keys() and all(torch.equal(wf[k], wr[k]) for k in wf)
    a, wa = run(True, tmp_path / "noclip_s", max_grad_norm = 0.0)
    b, wb = run(False, tmp_path / "noclip_u", max_grad_norm = 0.0)
    assert a == b, (a, b)
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)


DDP_SCRIPT = r"""
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.environ["TESTS_DIR"])
import test_moe_grouped_modulelist_stacked_lora as S
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML

def grads(world, rank):
    torch.cuda.set_device(rank)
    S.DEV = "cuda"
    torch.manual_seed(0)
    model = S._tiny_peft(True)
    model.train()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [rank])
    ids = S._ids()
    model(input_ids = ids, labels = ids).loss.backward()
    inner = model.module if world > 1 else model
    return {n: p.grad.detach().cpu() for n, p in inner.named_parameters() if p.requires_grad}

def worker(rank, world, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = os.environ["PORT"])
    dist.init_process_group("nccl", rank = rank, world_size = world)
    g = grads(world, rank)
    if rank == 0:
        torch.save(g, out)
    dist.destroy_process_group()

if __name__ == "__main__":
    out = sys.argv[1]
    torch.multiprocessing.spawn(worker, args = (2, out + ".ddp"), nprocs = 2)
    torch.save(grads(1, 0), out + ".single")
"""


def test_ddp_two_gpus_matches_single_process(tmp_path):
    """(f) DDP (NCCL, 2 GPUs, same batch per rank): the stacked grads all-reduce to the
    single-process grads."""
    if DEV != "cuda" or torch.cuda.device_count() < 2:
        pytest.skip("needs 2 CUDA devices")
    script = tmp_path / "ddp.py"
    script.write_text(DDP_SCRIPT)
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "TESTS_DIR": os.path.dirname(os.path.abspath(__file__)), "PORT": str(port),
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.run([sys.executable, str(script), str(tmp_path / "g")], env = env, capture_output = True, text = True)
    assert p.returncode == 0, p.stderr[-3000:]
    a, b = torch.load(tmp_path / "g.ddp"), torch.load(tmp_path / "g.single")
    assert a.keys() == b.keys() and sum(ML._STACK_NAME in k for k in a) == 12
    assert all(torch.equal(a[k], b[k]) for k in a)


UNSLOTH_SCRIPT = r"""
import os, sys, json, torch
import unsloth
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML
out, repo = sys.argv[1], sys.argv[2]
model, tok = FastLanguageModel.from_pretrained(repo,
    max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 3407,
    target_modules = r".*\.(q_proj|v_proj|gate_proj|up_proj|down_proj)", use_gradient_checkpointing = "unsloth")
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2, foreach = True)
g = torch.Generator().manual_seed(0)
losses = []
model.train()
for s in range(4):
    ids = torch.randint(0, 1000, (2, 32), generator = g).cuda()
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none = True)
    losses.append(loss.item())
model.save_pretrained(out + "/lora")
model.save_pretrained_merged(out + "/merged", tok, save_method = "merged_16bit")
json.dump({"losses": losses, "trainable": len(trainable), "calls": ML.CALLS,
           "stacked": sum(ML._STACK_NAME in n for n in trainable)}, open(out + "/info.json", "w"))
"""


def test_unsloth_fast_language_model_path(tmp_path):
    """(i) FastLanguageModel.from_pretrained + get_peft_model (regex targets incl. the experts):
    stacked at get_peft_model; training losses, the saved adapter and save_pretrained_merged (16-bit)
    equal the UNSLOTH_MOE_STACKED_LORA=0 run."""
    pytest.importorskip("unsloth")
    if L._hf_block("qwen3") is None:
        pytest.skip("Qwen3-MoE experts are not an nn.ModuleList on this transformers")
    try:
        from huggingface_hub import snapshot_download
        repo = snapshot_download("trl-internal-testing/tiny-Qwen3MoeForCausalLM", local_dir = str(tmp_path / "tiny"))
    except Exception as e:
        pytest.skip(f"tiny Qwen3-MoE repo unavailable: {e}")
    script = tmp_path / "u.py"
    script.write_text(UNSLOTH_SCRIPT)
    res = {}
    for arm in ("1", "0"):
        out = tmp_path / f"arm{arm}"
        out.mkdir()
        env = {**os.environ, "UNSLOTH_MOE_STACKED_LORA": arm, "UNSLOTH_COMPILE_LOCATION": str(tmp_path / f"cache{arm}"),
               "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
        p = subprocess.run([sys.executable, str(script), str(out), repo], env = env, cwd = str(tmp_path),
                           capture_output = True, text = True)
        assert p.returncode == 0, p.stderr[-4000:]
        res[arm] = json.load(open(out / "info.json"))
    s, u = res["1"], res["0"]
    assert s["calls"]["grouped_lora"] > 0 and u["calls"]["grouped_lora"] > 0
    assert s["stacked"] == 2 * 3 * 2 and u["stacked"] == 0 and s["trainable"] < u["trainable"]
    assert s["losses"] == u["losses"], (s["losses"], u["losses"])
    a, b = _st_load(tmp_path / "arm1" / "lora"), _st_load(tmp_path / "arm0" / "lora")
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    from safetensors.torch import load_file
    import glob
    fa = sorted(glob.glob(str(tmp_path / "arm1" / "merged" / "*.safetensors")))
    fb = sorted(glob.glob(str(tmp_path / "arm0" / "merged" / "*.safetensors")))
    assert fa and [os.path.basename(f) for f in fa] == [os.path.basename(f) for f in fb]
    for x, y in zip(fa, fb):
        da, db = load_file(x), load_file(y)
        assert list(da) == list(db) and all(torch.equal(da[k], db[k]) for k in da)
        assert not any(ML._STACK_NAME in k or "lora_" in k for k in da)


def test_stacking_is_opt_in():
    """Only the loader entry points (auto_enable_grouped_moe) stack; a plain enable keeps PEFT's
    per-expert Parameters, so a later call cannot orphan an optimizer's references."""
    model, blk = L.build("qwen3", base = "bf16", r = 8)
    ref = {n: p for n, p in model.named_parameters()}
    assert ML.enable_grouped_moe(model, verbose = False) == 1
    assert n_stacked(model) == 0
    assert all(ref[n] is p for n, p in model.named_parameters())
    model2, _ = L.build("qwen3", base = "bf16", r = 8)
    ML.auto_enable_grouped_moe(model2)
    assert n_stacked(model2) == 3 * 2


def test_no_stacking_once_training_started():
    """An optimizer built over the per-expert Parameters keeps training them: with grads present
    (training under way) stack_lora declines, and optimizer.step still moves the weights the
    forward reads."""
    model, blk = L.build("qwen3", base = "bf16", r = 8)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 1.0)
    x = torch.randn(1, 64, L.H, device = L.DEV, dtype = L.DT)
    blk(x)[0].float().square().sum().backward()
    assert ML.enable_grouped_moe(model, verbose = False, stack_lora = True) == 1
    assert n_stacked(model) == 0
    w = blk.experts[5].up_proj.lora_B["default"].weight
    before = w.detach().clone()
    opt.step()
    assert not torch.equal(blk.experts[5].up_proj.lora_B["default"].weight, before)
