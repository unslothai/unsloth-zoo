# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla chunk_h (simple_gla / GLA state kernels) must index sequence starts past 2**31 elements."""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK_H = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla" / "ops" / "common" / "chunk_h.py"


def test_chunk_h_launchers_pass_long_indexing():
    src = CHUNK_H.read_text()
    assert src.count("LONG_INDEXING: tl.constexpr = False,") == 2
    assert src.count("if LONG_INDEXING:\n        bos = bos.to(tl.int64)") == 2
    assert "LONG_INDEXING=max(k.numel(), v.numel()) > LONG_INDEXING_THRESHOLD," in src
    assert "LONG_INDEXING=max(q.numel(), do.numel()) > LONG_INDEXING_THRESHOLD," in src


def test_every_long_indexing_launcher_uses_the_shared_threshold():
    vendored = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"
    config = (vendored / "utils" / "_config.py").read_text()
    assert "\nLONG_INDEXING_THRESHOLD = 2**31\n" in config
    assert "LONG_INDEXING_THRESHOLD," in (vendored / "utils" / "__init__.py").read_text()
    sites = 0
    for path in sorted(vendored.rglob("*.py")):
        for line in path.read_text().splitlines():
            if "LONG_INDEXING=" in line:
                sites += 1
                assert line.rstrip().endswith("> LONG_INDEXING_THRESHOLD,"), f"{path.name}: {line.strip()}"
    assert sites >= 14, sites


def _free_gib():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.mem_get_info()[0] / 2**30
    except Exception:
        return 0.0


# Two dense sequences, H 8, K 128, V 16: the second starts at T * H * K elements.
_SUBPROCESS = textwrap.dedent(
    """
    import os, sys
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    import torch
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h

    T = int(sys.argv[1])
    B, H, K, V = 2, 8, 128, 16
    torch.manual_seed(0)
    dev = "cuda"
    g = torch.full((B, T, H), -0.05, device=dev, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16).mul_(0.05)
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
    _, ht = chunk_fwd_h(k=k, v=v, g=g, output_final_state=True)
    _, ht1 = chunk_fwd_h(k=k[1:].contiguous(), v=v[1:].contiguous(), g=g[1:].contiguous(), output_final_state=True)
    fwd = (ht[1] - ht1[0]).abs().max().item()
    del k, v, ht, ht1
    torch.cuda.empty_cache()

    q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16).mul_(0.05)
    do = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
    dht = torch.randn(B, H, K, V, device=dev, dtype=torch.float32)
    kw = dict(k=q, v=do, h0=None, scale=K ** -0.5)
    dh, _ = chunk_bwd_dh(q=q, do=do, dht=dht, g=g, **kw)
    dh_first = dh[1, 0].float().clone()
    del dh
    torch.cuda.empty_cache()
    q1, do1, g1 = q[1:].contiguous(), do[1:].contiguous(), g[1:].contiguous()
    dh1, _ = chunk_bwd_dh(q=q1, do=do1, dht=dht[1:].contiguous(), g=g1, k=q1, v=do1, h0=None, scale=K ** -0.5)
    bwd = (dh_first - dh1[0, 0].float()).abs().max().item()
    print(f"fwd {fwd} bwd {bwd} peak_gib {torch.cuda.max_memory_allocated() / 2**30:.2f}")
    assert fwd == 0.0 and bwd == 0.0, (fwd, bwd)
    print("CHUNK_H_LONG_OK")
    """
)


@pytest.mark.skipif(
    not fla_vendor._vendored_injection_supported() or fla_vendor._gpu_lacks_dot_instructions(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3 (never on RDNA1)",
)
@pytest.mark.parametrize("T", [2**20, 2**21], ids=["below_2_31", "past_2_31"])
def test_chunk_h_second_sequence_matches_alone(T):
    if _free_gib() < 16:
        pytest.skip("needs about 16 GiB of free GPU memory")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(T)], env=env, capture_output=True, text=True,
    )
    assert "CHUNK_H_LONG_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"
