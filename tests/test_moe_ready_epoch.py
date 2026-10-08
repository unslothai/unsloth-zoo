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

"""moe_ready_epoch: the grouped MoE readiness verdicts and NF4 pointer tables reused between
expert-state changes, for the ModuleList blocks (transformers < 5 Qwen3-MoE / Mixtral / OLMoE)
and gpt-oss NF4 experts.

* Bitwise: outputs, losses and LoRA / input grads over optimizer steps equal UNSLOTH_MOE_FAST_READY=0.
* Invalidation: after every change below, the fast verdict equals a fresh full check, and the fast
  forward equals the FAST_READY=0 forward bitwise (a stale pointer table would differ).
* Steady state: no full check and no table re-key per optimizer step, one lean re-read per block.
"""
import contextlib
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moe_grouped_modulelist_lora as L  # noqa: E402  (skips without grouped_mm / peft)
from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML  # noqa: E402
from unsloth_zoo.temporary_patches import moe_ready_epoch as RE  # noqa: E402
from peft import LoraConfig  # noqa: E402

FAST = "UNSLOTH_MOE_FAST_READY"
MID = L.E // 2


@pytest.fixture(autouse = True)
def _clean_env():
    keys = (FAST, "UNSLOTH_MOE_STACKED_LORA", "UNSLOTH_MOE_GROUPED_LORA", "UNSLOTH_GPTOSS_GROUPED",
            "UNSLOTH_COMPILE_DISABLE")
    old = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in old.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


@contextlib.contextmanager
def _fast(on):
    old = os.environ.get(FAST)
    os.environ[FAST] = "1" if on else "0"
    try:
        yield
    finally:
        os.environ.pop(FAST, None)
        if old is not None:
            os.environ[FAST] = old


def _delta(before):
    return {k: RE.COUNTS[k] - before[k] for k in before}


# ----------------------------------------------------------------------------- ModuleList
def _ml(base = "nf4", stack = True, **kw):
    os.environ["UNSLOTH_MOE_STACKED_LORA"] = "1" if stack else "0"
    model, blk = L.build("qwen3", base = base, **kw)
    assert ML.enable_grouped_moe(model, verbose = False, stack_lora = stack) == 1, ML.LAST_DECLINE
    return model, blk


def _x(seed = 3):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    return torch.randn(1, 64, L.H, device = "cuda", dtype = L.DT, generator = g)


def _verdict(blk, x):
    return ML._cached_state(blk, blk.experts, blk._unsloth_moe_spec, x.device, x.dtype)


def _truth(blk, x):
    return ML._experts_grouped_state(blk.experts, blk._unsloth_moe_spec, x.device, x.dtype)


def _forward(blk, x):
    torch.manual_seed(0)   # dropout in train mode
    with torch.no_grad():
        return blk(x)[0]


def _warm(blk, x):
    """Fast state built and hit: a cheap verdict and cheap NF4 tables on the second forward."""
    with _fast(True):
        _forward(blk, x)
        before = dict(RE.COUNTS)
        _forward(blk, x)
        d = _delta(before)
    assert d["full"] == 0 and d["cheap"] >= 1, d
    assert d["table_full"] == 0, d


def _agree(blk, x, grouped = None, output = True):
    """The fast verdict is the full check's; the fast forward equals the FAST_READY=0 forward."""
    with _fast(True):
        truth = _truth(blk, x)
        assert _verdict(blk, x) == truth
        fast = _forward(blk, x) if output else None
    if grouped is not None:
        assert isinstance(truth, dict) == grouped, truth
    if output:
        with _fast(False):
            full = _forward(blk, x)
        assert torch.equal(fast, full)
    return truth


def _new_params4bit(like, seed):
    import bitsandbytes as bnb
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(like.quant_state.shape, generator = g) * 0.05).to(L.DT)
    return bnb.nn.Params4bit(w, requires_grad = False, quant_type = "nf4").cuda()


def _opt_step():
    p = nn.Parameter(torch.zeros(1, device = "cuda"))
    p.grad = torch.zeros_like(p)
    torch.optim.SGD([p], lr = 0.1).step()


def _train_steps(model, blk, n, ckpt = True, opt = None):
    from torch.utils.checkpoint import checkpoint
    params = [p for p in model.parameters() if p.requires_grad]
    opt = opt or torch.optim.AdamW(params, lr = 1e-2)
    out = []
    for s in range(n):
        x = _x(10 + s).requires_grad_(True)
        y = checkpoint(lambda t: blk(t)[0], x, use_reentrant = False) if ckpt else blk(x)[0]
        loss = y.float().square().mean()
        loss.backward()
        grads = [p.grad.clone() for p in params] + [x.grad.clone()]
        opt.step()
        opt.zero_grad(set_to_none = True)
        out.append((loss.detach(), y.detach(), grads))
    return out


@pytest.mark.parametrize("base", ["nf4", "bf16"])
@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("ckpt", [True, False])
def test_modulelist_bitwise_vs_switch_off(base, stack, ckpt):
    runs = {}
    for fast in (True, False):
        model, blk = _ml(base, stack)
        blk.train()
        before = dict(RE.COUNTS)
        with _fast(fast):
            runs[fast] = _train_steps(model, blk, 4, ckpt)
        d = _delta(before)
        if fast:
            assert d["cheap"] + d["lean"] > 0 and d["full"] <= 2, d
        else:
            assert d["cheap"] == d["lean"] == 0, d
    for (la, ya, ga), (lb, yb, gb) in zip(runs[True], runs[False]):
        assert torch.equal(la, lb) and torch.equal(ya, yb)
        assert len(ga) == len(gb) and all(torch.equal(a, b) for a, b in zip(ga, gb))


@pytest.mark.parametrize("opt", ["adamw", "adamw8bit"])
def test_modulelist_steady_state_counters(opt):
    """One lean re-read per block per optimizer step, everything else cheap (forward, checkpoint
    replay, backward rebuilds); no full check, no table re-key."""
    model, blk = _ml("nf4", True)
    blk.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if opt == "adamw8bit":
        bnb = pytest.importorskip("bitsandbytes")
        o = bnb.optim.AdamW8bit(params, lr = 1e-3)
    else:
        o = torch.optim.AdamW(params, lr = 1e-3)
    with _fast(True):
        _train_steps(model, blk, 2, opt = o)
        before = dict(RE.COUNTS)
        g0 = ML.CALLS["grouped_lora"]
        _train_steps(model, blk, 3, opt = o)
    d = _delta(before)
    assert ML.CALLS["grouped_lora"] - g0 == 6   # forward + replay per step
    assert d["full"] == 0 and d["table_full"] == 0, d
    assert d["lean"] == 3 and d["cheap"] == 3, d
    assert d["table_cheap"] >= 3 * 4, d


def test_kill_switch_reads_every_expert(monkeypatch):
    model, blk = _ml()
    x = _x()
    _warm(blk, x)
    calls = []
    real = ML._ready_signature
    monkeypatch.setattr(ML, "_ready_signature", lambda experts, spec: calls.append(len(experts)) or real(experts, spec))
    with _fast(False):
        before = dict(RE.COUNTS)
        _forward(blk, x)
        _forward(blk, x)
    d = _delta(before)
    assert calls == [L.E, L.E] and d["cheap"] == d["lean"] == d["full"] == d["table_cheap"] == 0
    calls.clear()
    _warm(blk, x)
    assert calls[0] == L.E and set(calls[1:]) == {2}   # one full read, then only the end experts


# One test per invalidation path: `change(model, blk)` then the fast verdict / forward must equal
# the full check's. `grouped`: expected verdict after the change (None: not asserted).
def _add_and_switch(model, blk):
    model.add_adapter("b", LoraConfig(r = 8, lora_alpha = 16, target_modules = ["gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".b." in n and "lora_" in n:
                p.normal_(std = 0.05)
    model.set_adapter("b")


def _delete(model, blk):
    _add_and_switch(model, blk)
    model.set_adapter("default")
    model.delete_adapter("b")


def _requires_grad_mid(model, blk):
    blk.experts[MID].requires_grad_(True)   # Module.requires_grad_: base weights trainable


def _params4bit_setattr(model, blk):
    base = blk.experts[MID].up_proj.base_layer
    base.weight = _new_params4bit(base.weight, 123)


def _data_swap_then_step(model, blk):
    w = blk.experts[MID].down_proj.base_layer.weight
    new = _new_params4bit(w, 77)
    w.data = new.data
    w.quant_state = new.quant_state
    _opt_step()


def _nested_absmax_then_step(model, blk):
    qs = blk.experts[MID].down_proj.base_layer.weight.quant_state
    assert qs.nested
    qs.state2.absmax = qs.state2.absmax * 2   # new storage and scales, no hook sees it
    _opt_step()


def _nested_offset_then_step(model, blk):
    qs = blk.experts[MID].gate_proj.base_layer.weight.quant_state
    assert qs.nested
    qs.offset = qs.offset + 0.01
    _opt_step()


def _lora_dtype_mid(model, blk):
    blk.experts[MID].up_proj.lora_B.to(torch.float16)   # lora_A / lora_B dtypes now differ


def _dropout_mid_train(model, blk):
    blk.experts[MID].up_proj.lora_dropout["default"].train()


def _hf_hook_all(model, blk):
    from accelerate.hooks import ModelHook, add_hook_to_module
    for ex in blk.experts:
        add_hook_to_module(ex.up_proj.base_layer, ModelHook())


def _env_toggle(model, blk):
    os.environ["UNSLOTH_MOE_GROUPED_LORA"] = "0"


def _first_expert_direct(model, blk):
    blk.experts[0].gate_proj.merged_adapters.append("default")   # hook-less: the spot sees it


CHANGES = {
    "add_set_adapter": (_add_and_switch, True, {}),
    "delete_adapter": (_delete, True, {}),
    "requires_grad_mid": (_requires_grad_mid, False, {"base": "bf16"}),
    "params4bit_setattr": (_params4bit_setattr, True, {}),
    "data_swap_then_step": (_data_swap_then_step, True, {}),
    "nested_absmax_then_step": (_nested_absmax_then_step, True, {}),
    "nested_offset_then_step": (_nested_offset_then_step, True, {}),
    "lora_dtype_mid": (_lora_dtype_mid, False, {}),
    "dropout_mid_train": (_dropout_mid_train, False, {"lora_dropout": 0.1}),
    "hf_hook": (_hf_hook_all, True, {}),
    "grouped_lora_env": (_env_toggle, False, {}),
    "first_expert_direct": (_first_expert_direct, False, {}),
}


@pytest.mark.parametrize("name", list(CHANGES))
@pytest.mark.parametrize("stack", [True, False])
def test_modulelist_invalidation(name, stack):
    change, grouped, kw = CHANGES[name]
    if name in ("lora_dtype_mid", "add_set_adapter", "delete_adapter") and stack:
        stack = False   # per-expert Parameters: a stack moves / is replaced as a whole
    if name == "hf_hook":
        pytest.importorskip("accelerate")
    model, blk = _ml(stack = stack, **kw)
    x = _x()
    _warm(blk, x)
    change(model, blk)
    _agree(blk, x, grouped, output = name != "lora_dtype_mid")   # the loop itself rejects mixed A / B dtypes
    if name == "hf_hook":   # hooked experts never get a record: every call is a full check
        with _fast(True):
            _forward(blk, x)
            before = dict(RE.COUNTS)
            _forward(blk, x)
        assert _delta(before)["cheap"] == 0 and _delta(before)["full"] == 1


def test_disable_merge_unmerge_train_eval_and_grad_toggle():
    model, blk = _ml("nf4", False, lora_dropout = 0.1)
    x = _x()
    _warm(blk, x)
    with model.disable_adapter():
        _agree(blk, x, False)
    _agree(blk, x, True)
    blk.train()   # dropout 0.1 while training: the loop
    _agree(blk, x, False)
    blk.eval()
    _agree(blk, x, True)
    model.merge_adapter()
    _agree(blk, x, False)
    model.unmerge_adapter()   # NF4 unmerge requantizes: new weights / addresses
    _warm(blk, x)
    _agree(blk, x, True)


def test_dtype_and_device_moves():
    model, blk = _ml("bf16", False)
    x = _x()
    _warm(blk, x)
    blk.to(torch.float16)
    _agree(blk, x, False, output = False)   # the loop itself rejects bf16 input on fp16 weights
    blk.to(torch.bfloat16)
    _agree(blk, x, True)
    blk.experts[MID].cpu()
    _agree(blk, x, False, output = False)
    blk.experts[MID].cuda()
    _agree(blk, x, True)


def test_nf4_device_round_trip_rebuilds_the_tables():
    model, blk = _ml("nf4", True)
    x = _x()
    _warm(blk, x)
    try:
        blk.experts[MID].cpu()
    except Exception as e:   # pragma: no cover - bitsandbytes without CPU Params4bit moves
        pytest.skip(f"Params4bit cpu move: {e}")
    _agree(blk, x, False, output = False)
    blk.experts[MID].cuda()
    _agree(blk, x, True)
    tb = blk.experts.__dict__["_unsloth_nf4_stack_tables"]["down"][1]
    ptrs = tb["w"].tolist()
    w = blk.experts[MID].down_proj.base_layer.weight
    assert torch._C.TensorBase.data_ptr(w) in ptrs


def test_move_releases_the_tables():
    """A move drops the tables (and the storages they hold) instead of pinning the old copy."""
    model, blk = _ml("nf4", True)
    x = _x()
    _warm(blk, x)
    assert "_unsloth_nf4_stack_tables" in blk.experts.__dict__
    blk.cpu()
    assert "_unsloth_nf4_stack_tables" not in blk.experts.__dict__
    blk.cuda()
    _agree(blk, x, True)


def test_child_move_releases_the_owner_tables():
    """Moving one expert or one projection drops the tables held on the experts container."""
    model, blk = _ml("nf4", True)
    x = _x()
    for move in (lambda: blk.experts[MID].cpu(), lambda: blk.experts[0].up_proj.base_layer.cpu()):
        _warm(blk, x)
        assert "_unsloth_nf4_stack_tables" in blk.experts.__dict__
        move()
        assert "_unsloth_nf4_stack_tables" not in blk.experts.__dict__
        blk.cuda()
        _agree(blk, x, True)


def test_gpt_oss_child_move_releases_the_owner_tables():
    S = _gpt_oss()
    if not S.BF16_OK:
        pytest.skip("bf16 grouped_mm unavailable")
    os.environ["UNSLOTH_MOE_STACKED_LORA"] = "0"
    model = S.build("bf16")
    ex = S.experts_of(model)
    T = 64
    x = torch.randn(1, T, S.G.H, device = "cuda", dtype = S.G.DT)
    idx, w = S.G._routing(T)
    with _fast(True):
        _go_forward(ex, x, idx, w)
    if not ex.__dict__.get("_unsloth_routed_nf4"):
        pytest.skip("NF4 routed tables not built here")
    ex.down_projs[S.G.E // 2].cpu()
    assert "_unsloth_routed_nf4" not in ex.__dict__


def test_tracked_modules_still_pickle():
    import pickle
    model, blk = _ml("bf16", False)
    _warm(blk, _x())
    pickle.loads(pickle.dumps(blk.experts[MID]))


def test_tables_hold_the_storages():
    """A hook-less swap leaves the old storage alive in the table (stale values, not freed memory)."""
    model, blk = _ml("nf4", True)
    x = _x()
    _warm(blk, x)
    tb = blk.experts.__dict__["_unsloth_nf4_stack_tables"]["down"][1]
    held = {torch._C.TensorBase.data_ptr(t) for t in tb["_hold"] if isinstance(t, torch.Tensor)}
    assert set(tb["w"].tolist()) <= held and set(tb["a"].tolist()) <= held


def test_stacking_invalidates():
    """Stacking registers the stack on a tracked module: the registration hook bumps."""
    os.environ["UNSLOTH_MOE_STACKED_LORA"] = "1"
    model, blk = L.build("qwen3", base = "nf4")
    assert ML.enable_grouped_moe(model, verbose = False) == 1
    x = _x()
    _warm(blk, x)
    e0 = RE.stamp()[0]
    assert ML._stack_block_lora(blk, blk._unsloth_moe_spec) == 3
    assert RE.stamp()[0] > e0
    _agree(blk, x, True)


# ----------------------------------------------------------------------------- gpt-oss
def _gpt_oss():
    return pytest.importorskip("test_gpt_oss_stacked_lora")


def _go_pair_run(S, stacked, fast, steps = 3):
    model = S.build("bf16")
    if stacked:
        S.stack(model, True)
    ex = S.experts_of(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2)
    idx, w = S.G._routing(64)
    out = []
    with _fast(fast):
        for s in range(steps):
            g = torch.Generator(device = "cuda").manual_seed(20 + s)
            x = torch.randn(1, 64, S.G.H, device = "cuda", dtype = S.G.DT, generator = g).requires_grad_(True)
            y = ex(x, idx, w)
            loss = y.float().square().mean()
            loss.backward()
            out.append((loss.detach(), y.detach(), [p.grad.clone() for p in params] + [x.grad.clone()]))
            opt.step()
            opt.zero_grad(set_to_none = True)
    return out, ex


@pytest.mark.parametrize("stacked", [True, False])
def test_gpt_oss_bitwise_vs_switch_off(stacked):
    S = _gpt_oss()
    if not S.BF16_OK:
        pytest.skip("bf16 grouped_mm unavailable")
    os.environ["UNSLOTH_MOE_STACKED_LORA"] = "1" if stacked else "0"
    before = dict(RE.COUNTS)
    calls = dict(S.gq.CALLS)
    a, _ = _go_pair_run(S, stacked, True)
    d = _delta(before)
    assert S.gq.CALLS["forward_lora"] - calls["forward_lora"] == 3
    assert d["cheap"] + d["lean"] >= 2 and d["table_cheap"] >= 1, d
    b, _ = _go_pair_run(S, stacked, False)
    for (la, ya, ga), (lb, yb, gb) in zip(a, b):
        assert torch.equal(la, lb) and torch.equal(ya, yb)
        assert all(torch.equal(p, q) for p, q in zip(ga, gb))


def _go_forward(ex, x, idx, w):
    with torch.no_grad():
        return ex(x, idx, w)


@pytest.mark.parametrize("change", ["data_swap_then_step", "nested_absmax_then_step", "params4bit_setattr", "disable", "bias_grad_then_step",
                                    "set_adapter", "first_direct", "hf_hook"])
def test_gpt_oss_invalidation(change):
    """After each change the fast verdict and forward equal the switch-off ones (bitwise)."""
    S = _gpt_oss()
    if not S.BF16_OK:
        pytest.skip("bf16 grouped_mm unavailable")
    os.environ["UNSLOTH_MOE_STACKED_LORA"] = "0"
    model = S.build("bf16")
    ex = S.experts_of(model)
    T = 64
    x = torch.randn(1, T, S.G.H, device = "cuda", dtype = S.G.DT)
    idx, w = S.G._routing(T)
    with _fast(True):
        assert ex._grouped_bnb4bit_ready() is True
        _go_forward(ex, x, idx, w)
        before = dict(RE.COUNTS)
        _go_forward(ex, x, idx, w)
        assert _delta(before)["cheap"] == 1 and _delta(before)["table_cheap"] == 1
    mid = ex.down_projs[S.G.E // 2]
    base = mid.base_layer
    expect = True
    if change == "data_swap_then_step":
        new = _new_params4bit(base.weight, 5)
        base.weight.data, base.weight.quant_state = new.data, new.quant_state
        _opt_step()
    elif change == "nested_absmax_then_step":
        qs = base.weight.quant_state
        assert qs.nested
        qs.state2.absmax = qs.state2.absmax * 2
        _opt_step()
    elif change == "params4bit_setattr":
        base.weight = _new_params4bit(base.weight, 6)
    elif change == "disable":
        mid.enable_adapters(False)
        expect = False
    elif change == "bias_grad_then_step":
        base.bias.requires_grad_(True)   # tensor-level: no hook, the step's lean re-read sees it
        _opt_step()
        expect = False
    elif change == "set_adapter":
        model.add_adapter("b", LoraConfig(r = 8, lora_alpha = 16, target_modules = S.TARGETS))
        S._randomize(model, 3, adapter = "b")
        model.set_adapter("b")
    elif change == "first_direct":
        ex.gate_up_projs[0].use_dora["default"] = True
        expect = False
    else:
        pytest.importorskip("accelerate")
        from accelerate.hooks import ModelHook, add_hook_to_module
        for p in (*ex.gate_up_projs, *ex.down_projs):
            add_hook_to_module(p.base_layer, ModelHook())
    with _fast(True):
        got = ex._grouped_bnb4bit_ready()
        fast = _go_forward(ex, x, idx, w)
        if change == "hf_hook":
            before = dict(RE.COUNTS)
            ex._grouped_bnb4bit_ready()
            assert _delta(before)["cheap"] == 0 and _delta(before)["full"] == 1
    with _fast(False):
        truth = ex._grouped_bnb4bit_ready()
        full = _go_forward(ex, x, idx, w)
    assert got is truth is expect
    assert torch.equal(fast, full)
