# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""patch_torch_compile lets compiled graphs holding a custom autograd.Function use the AOTAutograd
cache: a second process hits it, and editing the Function's backward between processes misses and
runs the new backward. A user's TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD=0 keeps it off."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

_functorch_config = pytest.importorskip("torch._functorch.config")

if not hasattr(_functorch_config, "autograd_cache_allow_custom_autograd_functions"):
    pytest.skip("torch has no autograd_cache_allow_custom_autograd_functions", allow_module_level = True)

_CHILD = textwrap.dedent("""
    import importlib.util, json, os, sys
    sys.path.insert(0, sys.argv[1])
    scale, workdir = float(sys.argv[2]), sys.argv[3]
    import torch
    from unsloth_zoo.patching_utils import patch_torch_compile
    patch_torch_compile(debug = False, O3 = False, ignore_errors = True)
    from torch._dynamo.utils import counters
    path = os.path.join(workdir, "custom_fn_mod.py")
    with open(path, "w") as f:
        f.write(
            "import torch\\n"
            "class Square(torch.autograd.Function):\\n"
            "    @staticmethod\\n"
            "    def forward(ctx, x):\\n"
            "        ctx.save_for_backward(x)\\n"
            "        return x * x\\n"
            "    @staticmethod\\n"
            "    def backward(ctx, g):\\n"
            "        (x,) = ctx.saved_tensors\\n"
            f"        return g * x * {scale}\\n"
        )
    spec = importlib.util.spec_from_file_location("custom_fn_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_fn_mod"] = module
    spec.loader.exec_module(module)

    @torch.compile(backend = "inductor")
    def f(x):
        return module.Square.apply(x.sin()).cos().sum()

    x = torch.arange(1, 9, dtype = torch.float32).reshape(2, 4).requires_grad_()
    f(x).backward()
    ref = x.detach().clone().requires_grad_()
    u = ref.sin()
    expected = torch.autograd.grad(u, ref, grad_outputs = -torch.sin(u * u) * u * scale)[0]
    import torch._functorch.config as functorch_config
    print("RESULT " + json.dumps({
        "flag": functorch_config.autograd_cache_allow_custom_autograd_functions,
        "grad_ok": bool(torch.allclose(x.grad, expected, atol = 1e-5)),
        "aot": dict(counters["aot_autograd"]),
    }))
""")


def _run(tmp_path, scale, extra_env = None):
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", CUDA_VISIBLE_DEVICES = "",
               TORCHINDUCTOR_CACHE_DIR = str(tmp_path / "inductor"),
               TORCHINDUCTOR_FX_GRAPH_CACHE = "1", TORCHINDUCTOR_AUTOGRAD_CACHE = "1")
    env.pop("TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD", None)
    env.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    if extra_env:
        env.update(extra_env)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", _CHILD, root, str(scale), str(tmp_path)], env = env,
                         capture_output = True, text = True, timeout = 900)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
    assert out.returncode == 0 and line, out.stdout[-2000:] + out.stderr[-4000:]
    return json.loads(line[-1][len("RESULT "):])


def test_custom_function_graph_is_cached_and_edit_misses(tmp_path):
    cold = _run(tmp_path, 1.0)
    assert cold["flag"] is True and cold["grad_ok"], cold
    assert cold["aot"].get("autograd_cache_bypass", 0) == 0, cold
    warm = _run(tmp_path, 1.0)
    assert warm["grad_ok"] and warm["aot"].get("autograd_cache_hit", 0) == 1, warm
    # The edited backward must not be served from the warm cache.
    edited = _run(tmp_path, 3.0)
    assert edited["grad_ok"], edited
    assert edited["aot"].get("autograd_cache_hit", 0) == 0, edited
    assert edited["aot"].get("autograd_cache_miss", 0) == 1, edited


def test_user_env_zero_keeps_bypass(tmp_path):
    off = _run(tmp_path, 1.0, {"TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD": "0"})
    assert off["flag"] is False and off["grad_ok"], off
    assert off["aot"].get("autograd_cache_bypass", 0) == 1, off
