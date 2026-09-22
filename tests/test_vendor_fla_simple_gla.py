# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""The vendored fla snapshot also serves ``fla.ops.simple_gla``.

Remote-code lightning-attention models (inclusionAI Ling 2.5 / 2.6,
``BailingMoeV2_5``) do ``from fla.ops.simple_gla.chunk import chunk_simple_gla``
and ``from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla``
at import time, so without these files ``trust_remote_code`` loading stops with
"requires the following packages that were not found in your environment: fla"
unless the user installs flash-linear-attention separately.

The CPU part checks the files are vendored and narrowed. The GPU part runs the
injection in a fresh interpreter (as test_vendor_fla.py does) and compares both
kernels, forward and backward, against a float64 recurrence.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    import unsloth_zoo  # noqa: F401
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"


def _injection_supported() -> bool:
    try:
        from unsloth_zoo.temporary_patches.fla_vendor import (
            _vendored_injection_supported,
        )
        return bool(_vendored_injection_supported())
    except Exception:
        return False


def test_simple_gla_closure_is_vendored():
    for rel in (
        ("ops", "simple_gla", "__init__.py"),
        ("ops", "simple_gla", "chunk.py"),
        ("ops", "simple_gla", "fused_recurrent.py"),
        ("ops", "common", "chunk_h.py"),
        ("ops", "common", "fused_recurrent.py"),
    ):
        assert (VENDORED.joinpath(*rel)).is_file(), rel
    # The narrowed __init__ must not pull the unvendored fused_chunk / parallel kernels.
    init = (VENDORED / "ops" / "simple_gla" / "__init__.py").read_text()
    code = "\n".join(ln for ln in init.splitlines() if not ln.lstrip().startswith("#"))
    assert "fused_chunk" not in code and "parallel" not in code
    assert "chunk_simple_gla" in code and "fused_recurrent_simple_gla" in code
    assert not (VENDORED / "ops" / "simple_gla" / "naive.py").exists()


_KERNEL_SUBPROCESS = textwrap.dedent(
    """
    import os
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    import sys, torch
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    # The exact import lines the Ling 2.6 remote code uses.
    from fla.ops.simple_gla.chunk import chunk_simple_gla
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
    assert "fla.ops.simple_gla.fused_chunk" not in sys.modules
    assert "fla.ops.simple_gla.parallel" not in sys.modules

    def reference(q, k, v, g, scale):
        B, T, H, K = q.shape
        V = v.shape[-1]
        q, k, v, g = (x.double() for x in (q, k, v, g))
        S = q.new_zeros(B, H, K, V)
        out = []
        for t in range(T):
            S = S * g[:, t].exp()[..., None, None] + k[:, t, :, :, None] * v[:, t, :, None, :]
            out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
        return torch.stack(out, 1), S

    torch.manual_seed(0)
    dev = "cuda"
    for fn_name, T in (("chunk", 200), ("fused_recurrent", 48)):
        fn = chunk_simple_gla if fn_name == "chunk" else fused_recurrent_simple_gla
        B, H, K, V = 2, 4, 64, 64
        scale = K ** -0.5
        q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
        # Ling passes a per-head constant log decay expanded over batch and time.
        slope = -torch.linspace(0.01, 0.5, H, device=dev)
        g = slope[None, None, :].expand(B, T, H).contiguous()
        do = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
        o, ht = fn(q=q, k=k, v=v, g=g, scale=scale, output_final_state=True)
        (o.float() * do.float()).sum().backward()
        grads = [x.grad.double() for x in (q, k, v)]
        for x in (q, k, v):
            x.grad = None
        ref_o, ref_ht = reference(q, k, v, g, scale)
        (ref_o * do.double()).sum().backward()
        ref_grads = [x.grad.double() for x in (q, k, v)]

        def rel(a, b):
            return ((a.double() - b).norm() / b.norm().clamp_min(1e-12)).item()
        errs = [rel(o, ref_o), rel(ht, ref_ht)] + [rel(a, b) for a, b in zip(grads, ref_grads)]
        assert all(e < 2e-2 for e in errs), (fn_name, errs)
        print(fn_name, [round(e, 5) for e in errs])
    print("SIMPLE_GLA_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
def test_simple_gla_kernels_match_reference_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _KERNEL_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "SIMPLE_GLA_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr[-4000:]}"
