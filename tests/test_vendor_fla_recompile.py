# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla conv / cumsum kernels must not re-tune or recompile per token bucket or batch size."""

import ast
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"

CASES = {
    "modules/conv/triton/kernels.py": (("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"), "NB"),
    "ops/utils/cumsum.py": (
        (
            "chunk_local_cumsum_scalar_kernel",
            "chunk_local_cumsum_vector_kernel",
            "chunk_global_cumsum_scalar_kernel",
            "chunk_global_cumsum_vector_kernel",
        ),
        "B",
    ),
}


def _kernel_defs(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _autotune_key(fn):
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg == "key":
                    return [e.value for e in kw.value.elts]
    return None


@pytest.mark.parametrize("rel", sorted(CASES))
def test_shape_buckets_are_not_autotune_keys_or_constexprs(rel):
    kernels, arg = CASES[rel]
    defs = _kernel_defs(VENDORED / rel)
    for name in kernels:
        fn = defs[name]
        key = _autotune_key(fn)
        assert key is not None, name
        assert arg not in key, (name, key)
        assert arg not in [a.arg for a in fn.args.args], name
        assert not any(isinstance(n, ast.Name) and n.id == arg for n in ast.walk(fn)), name


_RUNTIME = textwrap.dedent(
    """
    import json, os, sys
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla, torch, triton
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.modules.conv.triton import kernels as ck
    from fla.ops.utils import cumsum as cs

    def parts(k):
        tuner = None
        while not isinstance(k, triton.runtime.jit.JITFunction):
            if isinstance(k, triton.runtime.autotuner.Autotuner):
                tuner = k
            k = k.fn
        return tuner, k

    kernels = {
        "conv_fwd": ck.causal_conv1d_fwd_kernel,
        "conv_bwd": ck.causal_conv1d_bwd_kernel,
        "cumsum_local_scalar": cs.chunk_local_cumsum_scalar_kernel,
        "cumsum_global_vector": cs.chunk_global_cumsum_vector_kernel,
    }
    for k in kernels.values():  # one config: the counts measure buckets, not autotuning width
        parts(k)[0].configs = parts(k)[0].configs[:1]
    D = 128
    w = torch.randn(D, 4, device = "cuda", dtype = torch.bfloat16)
    b = torch.randn(D, device = "cuda", dtype = torch.bfloat16)
    for T in (300, 1500, 2600, 3700, 4900, 5800):  # six NB = cdiv(T, 1024) buckets, packed (B = 1)
        cu = torch.tensor([0, T // 3, T], device = "cuda", dtype = torch.int32)
        x = torch.randn(1, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        y = causal_conv1d(x, w, b, activation = "silu", cu_seqlens = cu)
        y = y[0] if isinstance(y, tuple) else y
        y.backward(torch.randn_like(y))
    for B in (1, 2, 3, 5, 6, 7):  # padded batches of six sizes
        cs.chunk_local_cumsum(torch.randn(B, 200, 4, device = "cuda"), chunk_size = 64)
        cs.chunk_global_cumsum(torch.randn(B, 200, 4, 16, device = "cuda"))
    torch.cuda.synchronize()
    out = {}
    for n, k in kernels.items():
        tuner, jit = parts(k)
        out[n] = [len(tuner.cache), sum(len(v[0]) for v in jit.device_caches.values())]
    print("COUNTS" + json.dumps(out))
    """
)


def _injection_supported():
    try:
        import torch
        from unsloth_zoo.temporary_patches import fla_vendor
        return bool(
            torch.cuda.is_available()
            and torch.version.hip is None
            and fla_vendor._vendored_injection_supported()
            and not fla_vendor._gpu_lacks_dot_instructions()
        )
    except Exception:
        return False


@pytest.mark.skipif(not _injection_supported(), reason = "needs CUDA and vendored fla injection")
def test_conv_and_cumsum_tune_and_compile_once_across_buckets():
    pytest.importorskip("einops")
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(ZOO_ROOT), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, "-c", _RUNTIME], capture_output = True, text = True, env = env, timeout = 900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("COUNTS")][-1]
    counts = json.loads(line[len("COUNTS"):])
    # Triton still specializes pointers on 16-byte alignment, hence up to two compiles.
    assert all(tuned == 0 and compiles <= 2 for tuned, compiles in counts.values()), counts
