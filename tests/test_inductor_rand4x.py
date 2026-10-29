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

"""pytorch#198333 / #198680 backport: the gate, the emitted expression and the generated
Triton kernel. All CPU: the kernel test generates Triton source for a CPU graph while the
4x check sees a CUDA device, and stops before anything is compiled."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch._inductor.codegen import triton as triton_codegen  # noqa: E402

TritonOverrides = triton_codegen.TritonOverrides
PATCH_FILE = (
    Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "inductor_rand4x.py"
)
HAS_4X_PATH = "_can_use_4x_random" in TritonOverrides.__dict__
UPSTREAM_FIXED = hasattr(torch._inductor.config, "use_rand4x")
AFFECTED = HAS_4X_PATH and not UPSTREAM_FIXED


def _load_patch_module():
    # By path, so the test needs neither a GPU nor the rest of unsloth_zoo's imports.
    spec = importlib.util.spec_from_file_location("_unsloth_inductor_rand4x", PATCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(code, env = None):
    full_env = dict(os.environ, UNSLOTH_NO_LOCAL_GPU = "1", CUDA_VISIBLE_DEVICES = "")
    full_env.pop("TORCHINDUCTOR_USE_RAND4X", None)
    full_env.update(env or {})
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output = True, text = True, env = full_env, timeout = 600,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---- gate on synthetic bodies (each torch line) -------------------------------------------

# Only read through inspect.getsource; the bodies are never called.
V = TritonKernel = config = None


def _v213_v214():
    return (
        isinstance(V.kernel, TritonKernel)
        and V.graph.get_current_device_or_throw().type == "cuda"
        and V.kernel.triton_tensor_ndim() == 1
        and not config.align_random_eager
    )


def _v215():
    return (
        isinstance(V.kernel, TritonKernel)
        and config.use_rand4x
        and V.kernel.triton_tensor_ndim() == 1
    )


def test_gate_matches_only_the_unfixed_body():
    m = _load_patch_module()
    old_config = types.SimpleNamespace(align_random_eager = False)
    new_config = types.SimpleNamespace(align_random_eager = False, use_rand4x = False)
    assert m._needs_patch(_v213_v214, old_config)
    # torch 2.15: the body checks use_rand4x and the config has it.
    assert not m._needs_patch(_v215, new_config)
    assert not m._needs_patch(_v215, old_config)
    # A 2.13 / 2.14 body under a config that already grew use_rand4x (a fixed nightly).
    assert not m._needs_patch(_v213_v214, new_config)
    # Source unavailable (frozen / C): leave it alone.
    assert not m._needs_patch(len, old_config)


def test_gate_matches_the_installed_torch():
    m = _load_patch_module()
    original = TritonOverrides.__dict__.get("_can_use_4x_random")
    if original is None:
        assert not AFFECTED
        return
    original = getattr(original, "__func__", original)
    original = getattr(original, "_unsloth_original", original)
    assert m._needs_patch(original, torch._inductor.config) == AFFECTED


def test_patch_is_noop_without_the_4x_path():
    # torch 2.12 and older: no _can_use_4x_random.
    m = _load_patch_module()
    fake = types.SimpleNamespace()
    assert not m._needs_patch(getattr(fake, "missing", None), types.SimpleNamespace())


# ---- installed torch: patch, idempotence, cache key, emitter ------------------------------

PATCH_AND_REPORT = f"""
import json, importlib.util, torch
from torch._inductor.codegen.triton import TritonOverrides
spec = importlib.util.spec_from_file_location("m", {str(PATCH_FILE)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn = TritonOverrides.__dict__["_can_use_4x_random"] if "_can_use_4x_random" in TritonOverrides.__dict__ else None
print(json.dumps({{
    "first": m.patch_inductor_rand4x(),
    "second": m.patch_inductor_rand4x(),
    "patched": bool(fn is not None and getattr(getattr(fn, "__func__", fn), "_unsloth_rand4x_off", False)),
    "tag": torch.compiler.config.cache_key_tag,
}}))
"""


@pytest.mark.skipif(not AFFECTED, reason = "torch without the unfixed rand4x path")
def test_patch_installs_once_and_tags_the_cache_key():
    out = _run(PATCH_AND_REPORT, env = {"TORCH_COMPILE_CACHE_KEY_TAG": "user-tag"})
    assert out["first"] is True and out["second"] is True and out["patched"]
    tags = out["tag"].split(",")
    assert tags[0] == "user-tag", "existing tag must be kept"
    assert tags.count("unsloth/inductor_rand4x_off:pytorch198680") == 1


@pytest.mark.skipif(AFFECTED, reason = "torch with the unfixed rand4x path")
def test_patch_leaves_unaffected_torch_alone():
    out = _run(PATCH_AND_REPORT)
    assert out["first"] is False and not out["patched"]
    assert "inductor_rand4x_off" not in out["tag"]


@pytest.mark.skipif(not AFFECTED, reason = "torch without the unfixed rand4x path")
def test_env_opt_in_keeps_rand4x_like_torch_215():
    out = _run(PATCH_AND_REPORT, env = {"TORCHINDUCTOR_USE_RAND4X": "1"})
    assert out["first"] is False and not out["patched"]
    assert "inductor_rand4x_off" not in out["tag"]


EMIT = f"""
import json, importlib.util, torch
import torch._inductor.select_algorithm  # the 4x check reads it off torch._inductor
from unittest import mock
from torch._inductor.codegen.triton import TritonOverrides, TritonKernel
from torch._inductor.virtualized import V

class _OneD(TritonKernel):
    def __init__(self):
        pass
    def triton_tensor_ndim(self):
        return 1
    def dense_size_list(self):
        return ["XBLOCK"]

graph = mock.Mock()
graph.get_current_device_or_throw.return_value = torch.device("cuda")
def emit():
    with V.set_kernel_handler(_OneD()), V.set_graph_handler(graph):
        return [TritonOverrides.rand("seed", "x0"), TritonOverrides.randn("seed", "x0")]
before = emit()
spec = importlib.util.spec_from_file_location("m", {str(PATCH_FILE)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(json.dumps({{"before": before, "after": emit()}}))
"""


@pytest.mark.skipif(not AFFECTED, reason = "torch without the unfixed rand4x path")
def test_emitter_switches_from_rand4x_to_tl_rand():
    out = _run(EMIT)
    assert out["before"] == [
        "triton_helpers.rand4x(seed, (x0).to(tl.uint32), XBLOCK)",
        "triton_helpers.randn4x(seed, (x0).to(tl.uint32), XBLOCK)",
    ]
    assert out["after"] == ["tl.rand(seed, (x0).to(tl.uint32))", "tl.randn(seed, (x0).to(tl.uint32))"]


# ---- full Inductor codegen of a dropout graph ---------------------------------------------

CODEGEN = f"""
import json, sys, importlib.util, torch
import torch._inductor.config as ic
import torch.utils._triton as ut
from torch._inductor import async_compile
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.graph import GraphLowering

# Triton codegen for a CPU graph; no Triton driver here, so stub the backend queries and stop
# at the first kernel handed to the Triton compiler.
ic.cpu_backend = "triton"
ic.force_disable_caches = True
ut.triton_hash_with_backend = lambda: "cpu-probe"
ut.triton_backend = lambda: None
in_rand = [0]
get_device = GraphLowering.get_current_device_or_throw
def device(self):
    d = get_device(self)
    return torch.device("cuda") if in_rand[0] and d.type == "cpu" else d
GraphLowering.get_current_device_or_throw = device
for name in ("rand", "randn"):
    fn = getattr(TritonOverrides, name)
    def wrapped(*a, _fn = fn, **k):
        in_rand[0] += 1
        try:
            return _fn(*a, **k)
        finally:
            in_rand[0] -= 1
    setattr(TritonOverrides, name, staticmethod(wrapped))
sources = []
class Stop(Exception):
    pass
def capture(self, kernel_name, source_code, device_str = "cuda"):
    sources.append(source_code)
    raise Stop(kernel_name)
async_compile.AsyncCompile.triton = capture

if sys.argv[-1] == "patched":
    spec = importlib.util.spec_from_file_location("m", {str(PATCH_FILE)!r})
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.patch_inductor_rand4x()

def block(x):
    h = x.transpose(1, 2).contiguous()
    h = torch.nn.functional.dropout(h, 0.5, True)
    return h.transpose(1, 2).contiguous()

out = {{}}
# Inference and training forwards (the two graphs Unsloth's checkpointing compiles).
x = torch.ones(8, 64, 4096)
for label, fn, grad in (("dropout_no_grad", block, False), ("dropout_grad", block, True)):
    sources.clear()
    torch._dynamo.reset()
    try:
        with torch.set_grad_enabled(grad):
            torch.compile(fn)(x.clone().requires_grad_(grad))
    except Exception as e:
        if not sources:
            raise
    code = "\\n".join(sources)
    out[label] = {{"rand4x": "rand4x(" in code, "tl_rand": "tl.rand(" in code or "tl.randn(" in code}}
print(json.dumps(out))
"""


@pytest.mark.skipif(not AFFECTED, reason = "torch without the unfixed rand4x path")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason = "needs the triton package")
def test_generated_kernel_drops_rand4x():
    def run(arm):
        full_env = dict(os.environ, UNSLOTH_NO_LOCAL_GPU = "1", CUDA_VISIBLE_DEVICES = "")
        full_env.pop("TORCHINDUCTOR_USE_RAND4X", None)
        out = subprocess.run(
            [sys.executable, "-c", CODEGEN, arm], capture_output = True, text = True,
            env = full_env, timeout = 600,
        )
        assert out.returncode == 0, out.stderr[-4000:]
        return json.loads(out.stdout.strip().splitlines()[-1])

    base, head = run("base"), run("patched")
    # Base arm must show the defect, else the check proves nothing.
    for label in ("dropout_no_grad", "dropout_grad"):
        assert base[label] == {"rand4x": True, "tl_rand": False}, base
        assert head[label] == {"rand4x": False, "tl_rand": True}, head


# ---- the cache key really changes ----------------------------------------------------------

CACHE = f"""
import json, os, sys, importlib.util, torch
from torch._dynamo.utils import counters
import torch._inductor.config as ic
ic.fx_graph_cache = True
def f(x):
    return torch.nn.functional.dropout(x, 0.25, True) * 2
def compile_once():
    torch._dynamo.reset()
    counters.clear()
    torch.compile(f)(torch.ones(64, 64))
    return dict(counters["inductor"])
warm = compile_once()  # populates the cache
hit = compile_once()
spec = importlib.util.spec_from_file_location("m", {str(PATCH_FILE)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m.patch_inductor_rand4x()
after_patch = compile_once()
after_patch_again = compile_once()
print(json.dumps({{"hit": hit, "after_patch": after_patch, "after_patch_again": after_patch_again}}))
"""


@pytest.mark.skipif(not AFFECTED, reason = "torch without the unfixed rand4x path")
def test_patch_misses_a_warm_fx_graph_cache(tmp_path):
    out = _run(CACHE, env = {"TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor")})
    assert out["hit"].get("fxgraph_cache_hit", 0) >= 1, out
    # Same graph, warm cache: the tag must force a miss so no rand4x kernel is replayed.
    assert out["after_patch"].get("fxgraph_cache_hit", 0) == 0, out
    assert out["after_patch"].get("fxgraph_cache_miss", 0) >= 1, out
    assert out["after_patch_again"].get("fxgraph_cache_hit", 0) >= 1, out
