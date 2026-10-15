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

"""DeepSeek-V4 / V4.1 mHC mixer compile modes (`UNSLOTH_DSV4_MHC_FAST`)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from unsloth_zoo.temporary_patches.mhc_sinkhorn import (
    MHC_SINKHORN_SOURCE,
    unsloth_sinkhorn_knopp,
)

EPS = 1e-6
ITERS = 20


def _stock(comb, iters = ITERS, eps = EPS):
    # transformers DeepseekV4HyperConnection.forward, verbatim
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def _logits(kind, n = 512, device = "cpu"):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 4, 4, generator = g)
    if kind == "unit":
        pass
    elif kind == "wide":
        x = x * 40  # real V4-Flash comb base reaches |40|
    elif kind == "one_column":
        x[..., 0] += 80  # every row picks column 0: the eps floor dominates the other columns
    elif kind == "huge":
        x = x * 1e30
    elif kind == "neg_huge":
        x = torch.full((n, 4, 4), -1e30)
        x[..., 1] = 0
    return x.to(device)


KINDS = ["unit", "wide", "one_column", "huge", "neg_huge"]


def _grads(fn, logits, grad_out):
    leaf = logits.detach().clone().requires_grad_()
    out = fn(torch.softmax(leaf, dim = -1) + EPS)
    (grad,) = torch.autograd.grad(out, leaf, grad_out.to(out.dtype))
    return out.detach(), grad


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available() and os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") != "1":
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("kind", KINDS)
def test_forward_is_bitwise_stock_and_backward_tracks_fp64(kind, device):
    logits = _logits(kind, device = device)
    grad_out = torch.randn(logits.shape, generator = torch.Generator().manual_seed(1)).to(device) * 1e3
    y_ref, g_ref = _grads(_stock, logits.double(), grad_out.double())
    y_stock, g_stock = _grads(_stock, logits, grad_out)
    y_new, g_new = _grads(lambda c: unsloth_sinkhorn_knopp(c, ITERS, EPS), logits, grad_out)

    # Same ops in the same order: the eager forward is bitwise the transformers one.
    assert torch.equal(y_new, y_stock)
    assert torch.isfinite(y_new).all() and torch.isfinite(g_new).all()
    scale = g_ref.abs().max().clamp_min(1e-30)
    err_new = ((g_new.double() - g_ref).abs().max() / scale).item()
    err_stock = ((g_stock.double() - g_ref).abs().max() / scale).item()
    # Never worse than the stock autograd chain beyond fp32 rounding noise.
    assert err_new <= max(4 * err_stock, 1e-5), (err_new, err_stock)


def test_unrolled_function_gradcheck_fp64_and_matches_stock():
    # The compiled path's autograd.Function, run eagerly: exact derivative (gradcheck) and the
    # same values as the stock loop.
    from unsloth_zoo.temporary_patches.mhc_sinkhorn import _SinkhornKnopp

    x = (torch.randn(3, 4, 4, dtype = torch.float64) * 3).softmax(-1) + EPS
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: _SinkhornKnopp.apply(t, ITERS, EPS), (x,), eps = 1e-7, atol = 1e-6)
    for kind in KINDS:
        logits = _logits(kind)
        grad_out = torch.randn(logits.shape, generator = torch.Generator().manual_seed(1)) * 1e3
        y_ref, g_ref = _grads(_stock, logits.double(), grad_out.double())
        y_f, g_f = _grads(lambda c: _SinkhornKnopp.apply(c, ITERS, EPS), logits, grad_out)
        _, g_e = _grads(_stock, logits, grad_out)
        assert torch.isfinite(y_f).all() and torch.isfinite(g_f).all(), kind
        assert (y_f.double() - y_ref).abs().max().item() < 1e-5, kind
        scale = g_ref.abs().max().clamp_min(1e-30)
        err_f = ((g_f.double() - g_ref).abs().max() / scale).item()
        err_e = ((g_e.double() - g_ref).abs().max() / scale).item()
        assert err_f <= max(4 * err_e, 1e-5), (kind, err_f, err_e)


@pytest.mark.parametrize("iters", [0, 1, 2, 7])
def test_iteration_count_matches_stock(iters):
    x = torch.randn(8, 4, 4).softmax(-1) + EPS
    assert torch.equal(unsloth_sinkhorn_knopp(x, iters, EPS), _stock(x, iters))
    compiled = torch.compile(lambda c: unsloth_sinkhorn_knopp(c, iters, EPS), dynamic = True, fullgraph = True)
    torch.testing.assert_close(compiled(x), _stock(x, iters), rtol = 1e-5, atol = 1e-6)


def test_compiled_path_is_the_unrolled_function_and_eager_is_stock():
    from torch._dynamo.backends.common import aot_autograd

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(str(gm.graph))
        return gm.forward

    x = (torch.randn(8, 4, 4).softmax(-1) + EPS).requires_grad_()
    torch._dynamo.reset()
    torch.compile(lambda c: unsloth_sinkhorn_knopp(c, ITERS, EPS), backend = backend, fullgraph = True)(x)
    assert graphs and "autograd_function_apply" in graphs[-1], graphs
    # The unrolled form has no reductions at all; the stock loop has one per normalisation.
    assert "sum" not in graphs[-1], graphs[-1]
    # hc > 8 keeps the vectorised loop even when compiling.
    graphs.clear()
    torch._dynamo.reset()
    y = (torch.randn(2, 16, 16).softmax(-1) + EPS)
    torch.compile(lambda c: unsloth_sinkhorn_knopp(c, ITERS, EPS), backend = backend, fullgraph = True)(y)
    assert "autograd_function_apply" not in graphs[-1]


@pytest.mark.parametrize("device", _devices())
def test_compiled_matches_eager_and_stays_finite(device):
    compiled = torch.compile(lambda c: unsloth_sinkhorn_knopp(c, ITERS, EPS), dynamic = True, fullgraph = True)
    for kind in KINDS:
        logits = _logits(kind, device = device)
        grad_out = torch.randn(logits.shape, generator = torch.Generator().manual_seed(1)).to(device) * 1e3
        y_ref, g_ref = _grads(_stock, logits.double(), grad_out.double())
        y_c, g_c = _grads(compiled, logits, grad_out)
        _, g_e = _grads(_stock, logits, grad_out)
        assert torch.isfinite(y_c).all() and torch.isfinite(g_c).all(), kind
        assert (y_c.double() - y_ref).abs().max().item() < 1e-4, kind
        scale = g_ref.abs().max().clamp_min(1e-30)
        err_c = ((g_c.double() - g_ref).abs().max() / scale).item()
        err_e = ((g_e.double() - g_ref).abs().max() / scale).item()
        assert err_c <= max(4 * err_e, 1e-5), (kind, err_c, err_e)


def _module_from_source(tmp_path, name, body):
    import importlib.util

    path = tmp_path / f"{name}.py"
    path.write_text(
        "import torch\n"
        "class DeepseekV4HyperConnection(torch.nn.Module):\n"
        "    def forward(self, comb):\n" + body + "        return comb\n"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rewrite_registered_for_v4_and_v41_and_kill_switch(tmp_path, monkeypatch):
    from unsloth_zoo import compiler

    for model_type, cls in (("deepseek_v4", "DeepseekV4HyperConnection"), ("deepseek_v41", "DeepseekV41HyperConnection")):
        assert cls in compiler.MODULE_FORWARD_SOURCE_REWRITES[model_type]
        # Still listed: without a matching rewrite (or with the kill switch) the mixer stays eager.
        assert cls in compiler.DISABLE_COMPILE_MODULES

    module = _module_from_source(tmp_path, "mhc_stock_src", MHC_SINKHORN_SOURCE)
    monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", "unrolled")
    rewrites = compiler.module_forward_source_rewrites(module, "deepseek_v4")
    assert "comb = unsloth_sinkhorn_knopp(comb, self.hc_sinkhorn_iters, self.hc_eps)" in rewrites["DeepseekV4HyperConnection"]
    assert "for _ in range" not in rewrites["DeepseekV4HyperConnection"]
    monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", "stock")
    stock = compiler.module_forward_source_rewrites(module, "deepseek_v4")["DeepseekV4HyperConnection"]
    assert MHC_SINKHORN_SOURCE in stock and "unsloth_sinkhorn_knopp" not in stock
    monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", "0")
    assert compiler.module_forward_source_rewrites(module, "deepseek_v4") == {}
    # Unset: stock on torch >= 2.13, eager below.
    monkeypatch.delenv("UNSLOTH_DSV4_MHC_FAST", raising = False)
    from unsloth_zoo.temporary_patches.mhc_sinkhorn import _torch_at_least
    default = compiler.module_forward_source_rewrites(module, "deepseek_v4")
    if _torch_at_least(2, 13):
        assert default["DeepseekV4HyperConnection"] == stock
    else:
        assert default == {}


@pytest.mark.parametrize("mode", ["unrolled", "stock"])
def test_changed_upstream_source_is_not_rewritten(tmp_path, monkeypatch, mode):
    from unsloth_zoo import compiler

    monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", mode)
    changed = MHC_SINKHORN_SOURCE.replace("self.hc_eps)\n", "self.hc_eps * 2)\n", 1)
    module = _module_from_source(tmp_path, "mhc_changed_src", changed)
    assert compiler.module_forward_source_rewrites(module, "deepseek_v4") == {}
    # Two copies of the loop: ambiguous, not rewritten either.
    module = _module_from_source(tmp_path, "mhc_twice_src", MHC_SINKHORN_SOURCE * 2)
    assert compiler.module_forward_source_rewrites(module, "deepseek_v4") == {}


_CHILD = r'''
import os, sys, json, io, contextlib, importlib.util
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
import torch
from unsloth_zoo.compiler import unsloth_compile_transformers
import transformers.models.deepseek_v4.modeling_deepseek_v4 as modeling
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

stock_cls = modeling.DeepseekV4HyperConnection
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "deepseek_v4", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "unsloth_compiled_module_deepseek_v4.py")
generated = open(path, encoding = "utf-8").read()
out = {}
for cls in ("DeepseekV4HyperConnection", "DeepseekV4HyperHead"):
    where = generated.find(f"def {cls}_forward(")
    out[cls] = generated[:where].rsplit("\n@", 1)[-1].strip() if where != -1 else None
out["calls_fast"] = "comb = unsloth_sinkhorn_knopp(comb, self.hc_sinkhorn_iters, self.hc_eps)" in generated
out["imports_fast"] = "from unsloth_zoo.temporary_patches.mhc_sinkhorn import unsloth_sinkhorn_knopp" in generated

spec = importlib.util.spec_from_file_location("dsv4_cache", path)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
device = "cuda" if torch.cuda.is_available() and os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") != "1" else "cpu"
cfg = DeepseekV4Config(hidden_size = 64, hc_mult = 4, hc_sinkhorn_iters = 20, hc_eps = 1e-6, rms_norm_eps = 1e-6)
torch.manual_seed(0)
ref = stock_cls(cfg)
with torch.no_grad():
    ref.fn.normal_(0, 0.5); ref.base.normal_(0, 20.0); ref.scale.copy_(torch.tensor([0.2, 0.05, 0.4]))
new = cache.DeepseekV4HyperConnection(cfg)
new.load_state_dict(ref.state_dict())
ref.to(device); new.to(device)
x = torch.randn(2, 64, 4, 64, device = device, dtype = torch.bfloat16) * 4
res = {}
for name, mod in (("ref", ref), ("new", new)):
    mod.zero_grad(set_to_none = True)
    leaf = x.detach().clone().requires_grad_()
    post, comb, col = mod(leaf)
    g = torch.Generator().manual_seed(1)
    loss = (post * torch.randn(post.shape, generator = g).to(device)).sum() \
        + (comb * torch.randn(comb.shape, generator = g).to(device)).sum() \
        + (col.float() * torch.randn(col.shape, generator = g).to(device)).sum()
    loss.backward()
    res[name] = [t.detach().double() for t in (post, comb, col, leaf.grad, mod.fn.grad, mod.base.grad, mod.scale.grad)]
errs = []
finite = True
for a, b in zip(res["ref"], res["new"]):
    finite = finite and bool(torch.isfinite(b).all())
    errs.append(((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item())
out["rel_errs"] = errs
out["finite"] = finite
out["device"] = device
print("@@@" + json.dumps(out))
'''


def _run_child(tmp_path, mode):
    pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
    import importlib.util

    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_DSV4_MHC_FAST"] = mode
    if importlib.util.find_spec("unsloth") is None:
        env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env,
        capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            return json.loads(line[3:])
    raise AssertionError(result.stdout[-4000:])


def test_v4_mixer_compiles_through_the_rewrite(tmp_path):
    child = _run_child(tmp_path, "unrolled")
    assert child["DeepseekV4HyperConnection"].startswith("torch_compile_with_fallback("), child
    assert child["calls_fast"] and child["imports_fast"], child
    # HyperHead has no Sinkhorn and runs once per forward: it stays eager.
    assert child["DeepseekV4HyperHead"] == "torch_compiler_disable_unless_decode", child
    assert child["finite"], child
    # post / comb / collapsed / dx / dfn / dbase / dscale vs the stock eager module.
    assert max(child["rel_errs"]) < 2e-2, child


def test_kill_switch_keeps_the_stock_eager_mixer(tmp_path):
    child = _run_child(tmp_path, "0")
    assert child["DeepseekV4HyperConnection"] == "torch_compiler_disable_unless_decode", child
    assert not child["calls_fast"] and not child["imports_fast"], child
    assert child["finite"], child
    assert max(child["rel_errs"]) == 0.0, child


@pytest.mark.parametrize("mode", ["stock", ""])
def test_stock_mode_compiles_the_unchanged_mixer(tmp_path, mode):
    from unsloth_zoo.temporary_patches.mhc_sinkhorn import _torch_at_least

    if mode == "" and not _torch_at_least(2, 13):
        pytest.skip("default is eager below torch 2.13")
    child = _run_child(tmp_path, mode)
    assert child["DeepseekV4HyperConnection"].startswith("torch_compile_with_fallback("), child
    assert not child["calls_fast"] and not child["imports_fast"], child
    assert child["DeepseekV4HyperHead"] == "torch_compiler_disable_unless_decode", child
    assert child["finite"], child
    assert max(child["rel_errs"]) < 2e-2, child
