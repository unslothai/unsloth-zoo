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


"""Every transformers mHC mixer with the V4 Sinkhorn chain gets the rewrite and the eager fallback.

The registry check scans the installed transformers' modeling files as text, so the next model that
copies the mixer (GLM-5.3-Flash did, verbatim) fails here instead of compiling stock unnoticed.
"""

import json
import os
import re
import subprocess
import sys

import pytest

from unsloth_zoo.compiler import DISABLE_COMPILE_MODULES, MODULE_FORWARD_SOURCE_REWRITES
from unsloth_zoo.temporary_patches.mhc_sinkhorn import MHC_SINKHORN_SOURCE

_CLASS_RE = re.compile(r"^class (\w+)\(", re.M)


def _classes(source):
    starts = [(m.start(), m.group(1)) for m in _CLASS_RE.finditer(source)]
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(source)
        yield name, source[start:end]


def _scan_installed_transformers():
    transformers = pytest.importorskip("transformers")
    root = os.path.join(os.path.dirname(transformers.__file__), "models")
    exact, sinkhorn_like = {}, {}
    for model_type in sorted(os.listdir(root)):
        path = os.path.join(root, model_type, f"modeling_{model_type}.py")
        if not os.path.isfile(path):
            continue
        with open(path, encoding = "utf-8") as f:
            source = f.read()
        if "hc_sinkhorn_iters" not in source:
            continue
        for name, body in _classes(source):
            if "self.hc_sinkhorn_iters" in body and "def forward" in body:
                sinkhorn_like.setdefault(model_type, set()).add(name)
            if body.count(MHC_SINKHORN_SOURCE) == 1:
                exact.setdefault(model_type, set()).add(name)
    return transformers.__version__, exact, sinkhorn_like


def test_every_exact_sinkhorn_mixer_is_registered():
    version, exact, _ = _scan_installed_transformers()
    missing = [
        (model_type, name)
        for model_type, names in exact.items()
        for name in names
        if name not in MODULE_FORWARD_SOURCE_REWRITES.get(model_type, {})
    ]
    assert not missing, f"transformers {version}: mHC Sinkhorn mixers without a rewrite entry: {missing}"


def test_every_sinkhorn_mixer_has_the_eager_fallback():
    # A changed upstream chain no longer matches the rewrite; it must then stay eager (#859).
    version, _, sinkhorn_like = _scan_installed_transformers()
    missing = [
        (model_type, name)
        for model_type, names in sinkhorn_like.items()
        for name in names
        if name not in DISABLE_COMPILE_MODULES
    ]
    assert not missing, f"transformers {version}: mHC Sinkhorn mixers missing from DISABLE_COMPILE_MODULES: {missing}"


def test_registered_rewrites_share_one_source_and_fallback():
    for model_type, entries in MODULE_FORWARD_SOURCE_REWRITES.items():
        for name, (old, _new, _mode) in entries.items():
            if old is MHC_SINKHORN_SOURCE:
                assert name in DISABLE_COMPILE_MODULES, (model_type, name)
    assert "Glm5NextTextHyperConnection" in MODULE_FORWARD_SOURCE_REWRITES["glm5_next"]
    assert (
        MODULE_FORWARD_SOURCE_REWRITES["glm5_next"]["Glm5NextTextHyperConnection"]
        == MODULE_FORWARD_SOURCE_REWRITES["deepseek_v4"]["DeepseekV4HyperConnection"]
    )


_CHILD = r'''
import os, sys, json, io, contextlib, importlib.util
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
import torch
from unsloth_zoo.compiler import unsloth_compile_transformers
import transformers.models.glm5_next.modeling_glm5_next as modeling
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

stock_cls = modeling.Glm5NextTextHyperConnection
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "glm5_next", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "unsloth_compiled_module_glm5_next.py")
generated = open(path, encoding = "utf-8").read()
out = {}
for cls in ("Glm5NextTextHyperConnection", "Glm5NextTextHyperHead"):
    where = generated.find(f"def {cls}_forward(")
    out[cls] = generated[:where].rsplit("\n@", 1)[-1].strip() if where != -1 else None
out["calls_fast"] = "comb = unsloth_sinkhorn_knopp(comb, self.hc_sinkhorn_iters, self.hc_eps)" in generated
out["imports_fast"] = "from unsloth_zoo.temporary_patches.mhc_sinkhorn import unsloth_sinkhorn_knopp" in generated

spec = importlib.util.spec_from_file_location("glm5_next_cache", path)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
device = "cuda" if torch.cuda.is_available() and os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") != "1" else "cpu"
cfg = Glm5NextTextConfig(hidden_size = 64, hc_mult = 4, hc_sinkhorn_iters = 20, hc_eps = 1e-6, rms_norm_eps = 1e-6)
torch.manual_seed(0)
ref = stock_cls(cfg)
with torch.no_grad():
    ref.fn.normal_(0, 0.5); ref.base.normal_(0, 20.0); ref.scale.copy_(torch.tensor([0.2, 0.05, 0.4]))
new = cache.Glm5NextTextHyperConnection(cfg)
new.load_state_dict(ref.state_dict())
ref.to(device); new.to(device)
x = torch.randn(2, 64, 4, 64, device = device, dtype = torch.bfloat16) * 4
from torch._dynamo.utils import counters
counters.clear()
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
out["graph_breaks"] = sum(counters["graph_break"].values())
out["frames"] = counters["stats"]["unique_graphs"]
# Without unsloth the child runs under UNSLOTH_ZOO_DISABLE_GPU_INIT, whose triton stub makes
# CPU Inductor fall back to eager, so there is no compiled graph to count.
out["triton_stub"] = getattr(sys.modules.get("triton"), "__name__", "").startswith("unsloth_zoo.stubs")
print("@@@" + json.dumps(out))
'''


def _run_child(tmp_path, mode):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    import importlib.util

    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for var in ("UNSLOTH_COMPILE_DISABLE", "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
        env.pop(var, None)
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


def test_glm5_next_mixer_compiles_through_the_rewrite(tmp_path):
    child = _run_child(tmp_path, "unrolled")
    assert child["Glm5NextTextHyperConnection"].startswith("torch_compile_with_fallback("), child
    assert child["Glm5NextTextHyperHead"].startswith("torch_compile_with_fallback("), child
    assert child["calls_fast"] and child["imports_fast"], child
    assert child["finite"], child
    assert child["graph_breaks"] == 0, child
    assert child["frames"] >= 1 or child["triton_stub"], child
    assert max(child["rel_errs"]) < 2e-2, child


def test_glm5_next_kill_switch_keeps_the_stock_eager_mixer(tmp_path):
    child = _run_child(tmp_path, "0")
    assert child["Glm5NextTextHyperConnection"] == "torch_compiler_disable_unless_decode", child
    assert not child["calls_fast"] and not child["imports_fast"], child
    assert child["finite"], child
    assert max(child["rel_errs"]) == 0.0, child


def test_glm5_next_stock_mode_compiles_the_unchanged_mixer(tmp_path):
    child = _run_child(tmp_path, "stock")
    assert child["Glm5NextTextHyperConnection"].startswith("torch_compile_with_fallback("), child
    assert child["Glm5NextTextHyperHead"].startswith("torch_compile_with_fallback("), child
    assert not child["calls_fast"] and not child["imports_fast"], child
    assert child["finite"], child
    assert child["graph_breaks"] == 0, child
    assert child["frames"] >= 1 or child["triton_stub"], child
    assert max(child["rel_errs"]) < 2e-2, child


@pytest.mark.parametrize("model_type", sorted(MODULE_FORWARD_SOURCE_REWRITES))
def test_unrolled_rewrite_publishes_its_helper_on_the_modeling_module(model_type, monkeypatch):
    # Block compile rebinds a decoder layer's forward to its modeling module's globals, so the helper
    # the unrolled mixer calls must resolve there, not only in the generated module.
    import importlib

    from unsloth_zoo.compiler import module_forward_source_rewrites

    try:
        modeling = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    except Exception:
        pytest.skip(f"transformers has no {model_type}")
    monkeypatch.delattr(modeling, "unsloth_sinkhorn_knopp", raising = False)
    monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", "stock")
    if not module_forward_source_rewrites(modeling, model_type):
        pytest.skip(f"{model_type}: this transformers' mixer does not carry the Sinkhorn chain")
    assert not hasattr(modeling, "unsloth_sinkhorn_knopp")  # stock compiles the source unchanged
    monkeypatch.setenv("UNSLOTH_DSV4_MHC_FAST", "unrolled")
    rewritten = module_forward_source_rewrites(modeling, model_type)
    assert all("unsloth_sinkhorn_knopp" in source for source in rewritten.values())
    from unsloth_zoo.temporary_patches.mhc_sinkhorn import unsloth_sinkhorn_knopp

    assert modeling.unsloth_sinkhorn_knopp is unsloth_sinkhorn_knopp
