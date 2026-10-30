# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Sequence tiling registry: coverage, refusals, self-check, install / uninstall contract, planning and shims.

The family tests go only through `patch_tiled_mlp(model, "arctic")`, which exists before and after the rewrite, so the
same file run against the old TiledMLP shows which cases it got wrong. CPU only."""

import copy
import os
import re
import sys
from types import MethodType

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _tiling_families as F  # noqa: E402

from unsloth_zoo.tiled_mlp import patch_tiled_mlp  # noqa: E402

try:
    from unsloth_zoo import tiling_registry as R
except ImportError:  # the old implementation has no registry; only the family tests apply
    R = None

needs_registry = pytest.mark.skipif(R is None, reason = "registry not present")

BATCH, SEQ = 2, 40  # 80 rows against hidden 32: arctic tiles every MLP


def _tiled(model):
    return {n: m for n, m in model.named_modules() if "_unsloth_forward" in m.__dict__}


def _count_inner_calls(module, counter, key):
    orig = module.__dict__["_unsloth_forward"]

    def counting(self, *args, **kwargs):
        counter[key] = counter.get(key, 0) + 1
        return orig.__get__(self, type(self))(*args, **kwargs)

    module.__dict__["_unsloth_forward"] = counting


def _mlp_linear_names(model):
    """Linear layers inside each module the registry would tile (LoRA on MLP only)."""
    probe = copy.deepcopy(model)
    patch_tiled_mlp(probe, "arctic")
    roots = list(_tiled(probe))
    names = []
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and any(name.startswith(r + ".") for r in roots):
            names.append(name)
    return names


def _forward(model, ids, router):
    kwargs = dict(input_ids = ids, labels = ids)
    if router:
        kwargs["output_router_logits"] = True
    return model(**kwargs)


@pytest.mark.parametrize("family", [f for f in F.FAMILIES if f != "switch"])
def test_family_tiled_matches_untiled_with_mlp_lora(family):
    peft = pytest.importorskip("peft")
    model = F.build(family)
    if model is None:
        pytest.skip(f"{family} not in this transformers")
    targets = _mlp_linear_names(model)
    if targets:
        config = peft.LoraConfig(
            r = 4, lora_alpha = 8, lora_dropout = 0.0, init_lora_weights = False,
            target_modules = "(" + "|".join(re.escape(t) for t in targets) + ")",
        )
        torch.manual_seed(1)
        model = peft.get_peft_model(model, config)
    model.train()
    reference = copy.deepcopy(model)
    patch_tiled_mlp(model, "arctic")
    tiled = _tiled(model)
    assert tiled, f"{family}: nothing was tiled"

    calls = {}
    for name, module in tiled.items():
        _count_inner_calls(module, calls, name)
    outputs = {"ref": {}, "tiled": {}}

    def hook(store, name):
        def fn(module, args, out):
            store[name] = out
        return fn
    handles = []
    for name in tiled:
        handles.append(reference.get_submodule(name).register_forward_hook(hook(outputs["ref"], name)))
        handles.append(model.get_submodule(name).register_forward_hook(hook(outputs["tiled"], name)))

    router = hasattr(model.config, "output_router_logits")
    torch.manual_seed(2)
    ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ))
    try:
        out_ref = _forward(reference, ids, router)
    except TypeError:
        router = False
        out_ref = _forward(reference, ids, router)
    out_tiled = _forward(model, ids, router)
    for h in handles:
        h.remove()

    for name in tiled:
        assert calls.get(name, 0) > 1, f"{family}: {name} ran {calls.get(name, 0)} inner calls, not tiled"
        torch.testing.assert_close(outputs["tiled"][name], outputs["ref"][name], msg = name)
    torch.testing.assert_close(out_tiled.logits, out_ref.logits)
    if router and getattr(out_ref, "aux_loss", None) is not None:
        torch.testing.assert_close(out_tiled.aux_loss, out_ref.aux_loss)

    # Upstream gradient into every tiled output, router logits included, via the loss.
    loss_ref, loss_tiled = out_ref.loss, out_tiled.loss
    if router and getattr(out_ref, "aux_loss", None) is not None:
        loss_ref = loss_ref + out_ref.aux_loss
        loss_tiled = loss_tiled + out_tiled.aux_loss
    loss_ref.backward()
    loss_tiled.backward()
    torch.testing.assert_close(loss_tiled, loss_ref)
    ref_params = dict(reference.named_parameters())
    checked = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        q = ref_params[name]
        if q.grad is None:
            assert p.grad is None or not p.grad.any(), name
            continue
        assert p.grad is not None, f"{family}: {name} got no gradient when tiled"
        torch.testing.assert_close(p.grad, q.grad, rtol = 1e-4, atol = 1e-6, msg = name)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
def test_eval_and_decode_paths(family):
    model = F.build(family)
    if model is None:
        pytest.skip(f"{family} not in this transformers")
    reference = copy.deepcopy(model)
    patch_tiled_mlp(model, "arctic")
    ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, reference(ids).logits)
        out = model(ids[:, :8], use_cache = True)
        ref = reference(ids[:, :8], use_cache = True)
        step = model(ids[:, 8:9], past_key_values = out.past_key_values).logits
        step_ref = reference(ids[:, 8:9], past_key_values = ref.past_key_values).logits
    torch.testing.assert_close(step, step_ref)


def test_capacity_moe_is_never_tiled():
    model = F.build("switch")
    if model is None:
        pytest.skip("switch transformers missing")
    reference = copy.deepcopy(model)
    patch_tiled_mlp(model, "arctic")
    sparse = [m for m in model.modules() if type(m).__name__ == "SwitchTransformersSparseMLP"]
    assert sparse and not any("_unsloth_forward" in m.__dict__ for m in sparse)
    ids = torch.randint(1, 97, (BATCH, SEQ))
    with torch.no_grad():
        got = model(input_ids = ids, decoder_input_ids = ids).logits
        want = reference(input_ids = ids, decoder_input_ids = ids).logits
    torch.testing.assert_close(got, want)


# ------------------------------------------------------------------------------------------------------------------
# Registry-only behaviour
# ------------------------------------------------------------------------------------------------------------------

class _TokenMLP(torch.nn.Module):
    def __init__(self, h = 8, out = None):
        super().__init__()
        self.up = torch.nn.Linear(h, 2 * h)
        self.down = torch.nn.Linear(2 * h, out or h)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class FooBlock(_TokenMLP):
    pass


class BarBlock(_TokenMLP):
    def forward(self, x):
        y = super().forward(x)
        return y + y.mean(dim = -2, keepdim = True)  # mixes tokens


class _Holder(torch.nn.Module):
    def __init__(self, cls, n = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(n)])
        for layer in self.layers:
            layer.mlp = cls()

    def forward(self, x):
        for layer in self.layers:
            x = x + layer.mlp(x)
        return x


@pytest.fixture(autouse = True)
def _reset_registry(monkeypatch):
    if R is not None:
        R._SELF_CHECK.clear()
        R.set_engine("reference")
    monkeypatch.delenv("UNSLOTH_TILED_MLP_MODULES", raising = False)
    yield
    if R is not None:
        R.set_engine(None)


@needs_registry
def test_unknown_class_needs_opt_in_then_self_check(monkeypatch, capsys):
    model = _Holder(FooBlock)
    report = R.install(model, "rows:2")
    assert not report.tiled and report.skipped and "UNSLOTH_TILED_MLP_MODULES" in report.skipped[0][2]
    assert "found no eligible modules" in capsys.readouterr().out

    monkeypatch.setenv("UNSLOTH_TILED_MLP_MODULES", "FooBlock")
    calls = []
    real = R._self_check
    monkeypatch.setattr(R, "_self_check", lambda m, s: calls.append(type(m)) or real(m, s))
    report = R.install(model, "rows:2")
    assert len(report.tiled) == 2 and all(R.is_tiled(layer.mlp) for layer in model.layers)
    assert "on 2 modules [FooBlock x2]" in capsys.readouterr().out
    # One eager check per class: the second instance reuses the verdict.
    assert len(R._SELF_CHECK) == 1
    x = torch.randn(2, 9, 8)
    ref = _Holder(FooBlock)
    ref.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), ref(x))


@needs_registry
def test_self_check_refuses_a_token_mixing_class(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TILED_MLP_MODULES", "*.mlp")
    model = _Holder(BarBlock)
    report = R.install(model, "rows:2", verbose = False)
    assert not report.tiled
    assert all("self-check failed" in reason for _, _, reason in report.skipped)


@needs_registry
def test_refusals():
    class GateWithSeqAux(torch.nn.Module):
        seq_aux = True
        alpha = 0.001

    class DeepseekMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = GateWithSeqAux()

    class GatedMLP(_TokenMLP):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(8, 8, 3)

    capacity = type("SwitchTransformersSparseMLP", (torch.nn.Module,), {})()
    pooled = type("Gemma3MultiModalProjector", (torch.nn.Module,), {})()
    assert "capacity" in R.lookup(capacity).refuse
    assert "seq_aux" in R.lookup(DeepseekMoE()).refuse
    assert "sequence-mixing" in R.lookup(pooled).refuse
    assert "Conv1d" in R.lookup(GatedMLP()).refuse
    # Opt-in does not override a refusal; "!" forces.
    model = torch.nn.Module()
    model.mlp = DeepseekMoE()
    assert not R.install(model, "rows:2:mlp", verbose = False).tiled
    assert R.install(model, "rows:2:!mlp", verbose = False).tiled


@needs_registry
def test_dense_rule_and_known_entries():
    gpt2_mlp = type("GPT2MLP", (torch.nn.Module,), {})()
    sparse = type("FooSparseMLP", (torch.nn.Module,), {})()
    assert R.lookup(gpt2_mlp).family == "dense-mlp"
    assert R.lookup(sparse) is None
    assert R.lookup(type("Qwen3MoeSparseMoeBlock", (torch.nn.Module,), {})()).output_contract == ("rows", "rows_flat")
    assert R.lookup(type("Llama4TextMoe", (torch.nn.Module,), {})()).output_contract == ("rows_flat", "rows_flat")
    assert R.lookup(type("LlavaMultiModalProjector", (torch.nn.Module,), {})()).family == "vlm-projector"
    assert R.lookup(type("NemotronHMoE", (torch.nn.Module,), {})()).family == "nemotron-h"


@needs_registry
def test_width_changing_projector_and_padding_free_input():
    proj = type("LlavaMultiModalProjector", (_TokenMLP,), {})(8, 12)
    ref = copy.deepcopy(proj)
    R.install_module(proj, "rows:3")
    for shape in ((3, 10, 8), (17, 8), (1, 2, 5, 8)):
        x = torch.randn(*shape, requires_grad = True)
        xr = x.detach().clone().requires_grad_(True)
        out, want = proj(x), ref(xr)
        assert out.shape == want.shape == (*shape[:-1], 12)
        torch.testing.assert_close(out, want)
        out.square().sum().backward()
        want.square().sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)


@needs_registry
def test_install_is_idempotent_and_uninstall_restores():
    model = _Holder(lambda: type("FooMLP", (_TokenMLP,), {})())
    x = torch.randn(2, 9, 8)
    want = model(x)
    R.install(model, "rows:2", verbose = False)
    fwd = model.layers[0].mlp.forward
    R.install(model, "rows:2", verbose = False)
    assert model.layers[0].mlp.forward is fwd
    R.install(model, "rows:4", verbose = False)
    assert model.layers[0].mlp.__dict__["_unsloth_tiling"].tile_rows == 4
    torch.testing.assert_close(model(x), want)
    assert R.uninstall(model) == 2
    for layer in model.layers:
        assert "forward" not in layer.mlp.__dict__ and "_unsloth_forward" not in layer.mlp.__dict__
    torch.testing.assert_close(model(x), want)
    R.install(model, "0")
    assert not R.tiled_modules(model)


@needs_registry
def test_inner_forward_slot_contract():
    # llama.py's LoRA-MLP patch writes into _unsloth_forward when present; it must then run inside each tile.
    mlp = type("FooMLP", (_TokenMLP,), {})()
    R.install_module(mlp, "rows:2")
    seen = []

    def fused(self, x):
        seen.append(x.shape[-2])
        return _TokenMLP.forward(self, x)

    mlp._unsloth_forward = MethodType(fused, mlp)
    x = torch.randn(1, 8, 8)
    torch.testing.assert_close(mlp(x), _TokenMLP.forward(mlp, x))
    assert seen == [2, 2, 2, 2]
    # Unbound class forward (what _decline_fused_lora_for_fsdp writes back) also works.
    mlp._unsloth_forward = type(mlp).forward
    torch.testing.assert_close(mlp(x), _TokenMLP.forward(mlp, x))


@needs_registry
def test_instance_forward_is_adopted_and_restored():
    # mxfp4 binds its forward on the instance; the class forward would skip it.
    mlp = type("FooMLP", (_TokenMLP,), {"forward": lambda self, x: _TokenMLP.forward(self, x) + 1000.0})()

    def real(self, x):
        return _TokenMLP.forward(self, x)

    mlp.forward = MethodType(real, mlp)
    x = torch.randn(2, 6, 8)
    want = real(mlp, x)
    R.install_module(mlp, "rows:2")
    torch.testing.assert_close(mlp(x), want)
    R.install_module(mlp, "rows:2")  # re-install keeps it
    torch.testing.assert_close(mlp(x), want)
    R.uninstall(mlp)
    assert mlp.forward.__func__ is real
    torch.testing.assert_close(mlp(x), want)


@needs_registry
def test_grouped_moe_style_chain_does_not_recurse():
    # enable_grouped_moe after tiling stores the tiled forward as _orig_moe_forward and installs its own; the
    # revalidation must adopt it without the two calling each other forever.
    mlp = type("FooMLP", (_TokenMLP,), {})()
    R.install_module(mlp, "rows:2")
    mlp._orig_moe_forward = mlp.forward

    def grouped(self, x):  # falls back on CPU, like grouped_moe_forward
        return self._orig_moe_forward(x)

    mlp.forward = MethodType(grouped, mlp)
    R.install_module(mlp, "rows:2")
    x = torch.randn(1, 8, 8)
    torch.testing.assert_close(mlp(x), _TokenMLP.forward(mlp, x))
    R.uninstall(mlp)
    torch.testing.assert_close(mlp(x), _TokenMLP.forward(mlp, x))


@needs_registry
def test_revalidate_covers_modules_added_later():
    model = _Holder(lambda: type("FooMLP", (_TokenMLP,), {})())
    assert R.revalidate(model) is None
    R.install(model, "rows:2", verbose = False)
    extra = torch.nn.Module()
    extra.mlp = type("FooMLP", (_TokenMLP,), {})()
    model.layers.append(extra)
    report = R.revalidate(model, verbose = False)
    assert len(report.tiled) == 3 and R.is_tiled(extra.mlp)


@needs_registry
def test_deepcopy_runs_the_copy():
    mlp = type("FooMLP", (_TokenMLP,), {})()
    R.install_module(mlp, "rows:2")
    clone = copy.deepcopy(mlp)
    with torch.no_grad():
        clone.down.weight.zero_()
        clone.down.bias.zero_()
    x = torch.randn(1, 8, 8)
    assert not clone(x).any()
    assert mlp(x).any()


@needs_registry
@pytest.mark.parametrize(("value", "mode", "rows", "gb", "mods"), [
    (True, "arctic", 0, 0.0, ()),
    ("1", "arctic", 0, 0.0, ()),
    ("arctic", "arctic", 0, 0.0, ()),
    ("auto", "auto", 0, 0.0, ()),
    ("target", "auto", 0, 0.0, ()),
    ("target:2.5", "target", 0, 2.5, ()),
    ("target:2:mlp,ffn", "target", 0, 2.0, ("mlp", "ffn")),
    ("arctic::*.mlp", "arctic", 0, 0.0, ("*.mlp",)),
    ("rows:512", "rows", 512, 0.0, ()),
])
def test_parse_mode(value, mode, rows, gb, mods):
    plan = R.parse_mode(value)
    assert (plan.mode, plan.rows, plan.target_gb, plan.modules) == (mode, rows, gb, mods)
    assert R.parse_mode(plan.spelling()) == plan


@needs_registry
@pytest.mark.parametrize("value", [None, False, "", "0", "off"])
def test_parse_mode_off(value):
    assert R.parse_mode(value) is None


@needs_registry
@pytest.mark.parametrize("value", ["rows:0", "rows:x", "target:-1", "bogus"])
def test_parse_mode_rejects(value):
    with pytest.raises(ValueError):
        R.parse_mode(value)


@needs_registry
def test_bucket_is_power_of_two_and_never_exceeds_rows():
    assert R._bucket(1, 4) == 1
    assert R._bucket(4, 4) == 1
    assert R._bucket(5, 4) == 2
    assert R._bucket(100, 4) == 32
    assert R._bucket(3, 1) == 2
    for rows in range(1, 70):
        n = R._bucket(rows, 3)
        assert n <= rows and n & (n - 1) == 0 and (n == 1 or rows > 3 * (n // 2))


@needs_registry
def test_short_inputs_skip_the_engine():
    mlp = type("FooMLP", (_TokenMLP,), {})()
    R.install_module(mlp, "rows:4")
    calls = []
    R.set_engine(lambda *a, **k: calls.append(1) or R.reference_tiled_call(*a, **k))
    mlp(torch.randn(1, 1, 8))
    mlp(torch.randn(1, 4, 8))
    assert calls == []
    mlp(torch.randn(1, 5, 8))
    assert calls == [1]


@needs_registry
def test_frozen_input_still_trains_mlp_params():
    # Old TiledMLP: an input that does not require grad left the MLP's own parameters without gradients.
    mlp = type("FooMLP", (_TokenMLP,), {})()
    ref = copy.deepcopy(mlp)
    R.install_module(mlp, "rows:2")
    x = torch.randn(2, 6, 8)
    mlp(x).sum().backward()
    ref(x).sum().backward()
    for (n, p), q in zip(mlp.named_parameters(), ref.parameters()):
        assert p.grad is not None, n
        torch.testing.assert_close(p.grad, q.grad)


@needs_registry
def test_extra_outputs_receive_gradients():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(8, 8)
            self.r = torch.nn.Linear(8, 3)

        def forward(self, x):
            return self.w(x), self.r(x.reshape(-1, 8))

    m = Router()
    ref = copy.deepcopy(m)
    R.install_module(m, "rows:2", spec = R.TileSpec("Router", output_contract = ("rows", "rows_flat")))
    x = torch.randn(2, 5, 8)
    (a, b), (c, d) = m(x), ref(x)
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(b, d)
    (a.sum() + b.square().sum()).backward()
    (c.sum() + d.square().sum()).backward()
    torch.testing.assert_close(m.r.weight.grad, ref.r.weight.grad)


@needs_registry
def test_engine_selector(monkeypatch):
    R.set_engine(None)
    monkeypatch.setenv("UNSLOTH_TILED_ENGINE", "reference")
    assert R.get_engine() is R.reference_tiled_call
    R.set_engine(None)
    monkeypatch.setenv("UNSLOTH_TILED_ENGINE", "bogus")
    with pytest.raises(ValueError):
        R.get_engine()
    R.set_engine(None)
    monkeypatch.setenv("UNSLOTH_TILED_ENGINE", "op")
    with pytest.raises(ValueError):
        R.get_engine()
    from unsloth_zoo.sequence_tiling import tiled_call
    for name in ("auto", "checkpoint"):
        R.set_engine(None)
        monkeypatch.setenv("UNSLOTH_TILED_ENGINE", name)
        assert R.get_engine() is tiled_call
    R.set_engine(None)
    monkeypatch.delenv("UNSLOTH_TILED_ENGINE")
    assert R.get_engine() is tiled_call
    R.set_engine(None)


@needs_registry
def test_parallel_rule_is_documented():
    assert "per rank" in R.PARALLEL_RULE and "Ulysses" in R.PARALLEL_RULE


# ------------------------------------------------------------------------------------------------------------------
# Shims
# ------------------------------------------------------------------------------------------------------------------

@needs_registry
def test_tiled_mlp_shims():
    from unsloth_zoo import memory_budget, tiled_mlp
    assert tiled_mlp._default_target_gb is memory_budget._default_target_gb
    assert tiled_mlp._host_target_gb is memory_budget._host_target_gb
    assert tiled_mlp.get_max_flat_qlen is memory_budget.get_max_flat_qlen
    assert not hasattr(tiled_mlp, "TiledMLP")
    model = _Holder(lambda: type("FooMLP", (_TokenMLP,), {})())
    tiled_mlp.patch_tiled_mlp(model, "target:1")
    assert all(R.is_tiled(layer.mlp) for layer in model.layers)
    assert model.layers[0].mlp.__dict__["_unsloth_tiling"].plan.mode == "target"
    m = FooBlock()  # unknown class: patch_mlp is an explicit request, so it is tiled
    assert tiled_mlp.patch_mlp(m, target_arctic = True) is m and R.is_tiled(m)


@needs_registry
def test_fused_losses_read_the_budget_from_memory_budget(monkeypatch):
    from unsloth_zoo import memory_budget
    from unsloth_zoo.fused_losses import chunked_logprob as cl
    monkeypatch.setattr(memory_budget, "_default_target_gb", lambda device = None: 0.25)
    assert cl._default_chunk_gb() == 0.25
    for path in ("cross_entropy_loss", "fused_linear", "chunked_logprob"):
        import importlib
        src = open(importlib.import_module(f"unsloth_zoo.fused_losses.{path}").__file__).read()
        assert "unsloth_zoo.tiled_mlp" not in src, path


class _InplaceDX(torch.autograd.Function):
    # Like unsloth's LoRA_MLP with inplace=True: dX is written into X's storage in backward.
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x @ w

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gw = x.reshape(-1, x.shape[-1]).t() @ g.reshape(-1, g.shape[-1])
        x.copy_(g @ w.t())
        return x, gw


@needs_registry
def test_inner_forward_writing_dx_in_place():
    class FooMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, x):
            return _InplaceDX.apply(x, self.w)

    m = FooMLP()
    ref = copy.deepcopy(m)
    R.install_module(m, "rows:2")
    x = torch.randn(2, 6, 8)
    xa = (x * 1).requires_grad_(True)
    xb = (x * 1).requires_grad_(True)
    (m(xa * 1) ** 2).sum().backward()
    (ref(xb * 1) ** 2).sum().backward()
    torch.testing.assert_close(xa.grad, xb.grad)
    torch.testing.assert_close(m.w.grad, ref.w.grad)


@needs_registry
def test_install_on_the_module_itself_claims_its_children():
    mlp = type("FooMLP", (_TokenMLP,), {})()
    mlp.inner = type("BarMLP", (_TokenMLP,), {})()
    report = R.install(mlp, "rows:2", verbose = False)
    assert [r[0] for r in report.tiled] == [""] and not R.is_tiled(mlp.inner)


@needs_registry
def test_module_hooks_run_once_per_call_not_per_tile():
    # The engine gets the inner forward, never the hooked module: FSDP2's unshard hooks would run per tile.
    mlp = type("FooMLP", (_TokenMLP,), {})()
    R.install_module(mlp, "rows:2")
    pre, post = [], []
    mlp.register_forward_pre_hook(lambda m, a: pre.append(1))
    mlp.register_forward_hook(lambda m, a, o: post.append(1))
    x = torch.randn(1, 8, 8, requires_grad = True)
    mlp(x).sum().backward()
    assert pre == [1] and post == [1]


@needs_registry
def test_tensor_parallel_modules_are_skipped(monkeypatch, tmp_path):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("needs gloo")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, distribute_tensor
    dist.init_process_group("gloo", init_method = f"file://{tmp_path / 'store'}", rank = 0, world_size = 1)
    try:
        def holder(mesh_name):
            model = _Holder(lambda: type("FooMLP", (_TokenMLP,), {})(), n = 1)
            mesh = init_device_mesh("cpu", (1,), mesh_dim_names = (mesh_name,))
            mlp = model.layers[0].mlp
            for lin in (mlp.up, mlp.down):
                lin.weight = torch.nn.Parameter(distribute_tensor(lin.weight.data, mesh, [Replicate()]))
            return model
        report = R.install(holder("tp"), "rows:2", verbose = False)
        assert not report.tiled and report.skipped[0][2].startswith("tensor parallel")
        assert "UNSLOTH_TILED_MLP_TP" in report.line()
        assert R.install(holder("dp_shard"), "rows:2", verbose = False).tiled  # FSDP2 mesh: tiled
        monkeypatch.setenv("UNSLOTH_TILED_MLP_TP", "1")
        assert R.install(holder("tp"), "rows:2", verbose = False).tiled
    finally:
        dist.destroy_process_group()
    assert "UNSLOTH_TILED_MLP_TP" in R.PARALLEL_RULE


@needs_registry
def test_two_identical_instances_run_their_own_inner_forward():
    # An engine that deduplicates by signature must still run each instance's own slot (LoRA forward, counters).
    a, b = (type("FooMLP", (_TokenMLP,), {})() for _ in range(2))
    b.load_state_dict(a.state_dict())
    seen = []
    for tag, m in (("a", a), ("b", b)):
        R.install_module(m, "rows:2")
        m._unsloth_forward = MethodType(lambda self, x, tag = tag: seen.append(tag) or _TokenMLP.forward(self, x), m)
    x = torch.randn(1, 4, 8)
    a(x)
    b(x)
    assert seen == ["a", "a", "b", "b"]
