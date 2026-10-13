# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla serves fla.modules.ShortConvolution, which Ling 3.0 (BailingMoeV3) remote code imports."""

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
        from unsloth_zoo.temporary_patches import fla_vendor
        return bool(
            fla_vendor._vendored_injection_supported()
            and not fla_vendor._gpu_lacks_dot_instructions()
        )
    except Exception:
        return False


def _code(path):
    return "\n".join(
        ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")
    )


def test_short_conv_closure_is_vendored():
    conv = VENDORED / "modules" / "conv"
    for rel in (
        "__init__.py",
        "short_conv.py",
        "causal_conv1d.py",
        "triton/__init__.py",
        "triton/kernels.py",
        "triton/ops.py",
        "cp/__init__.py",
        "cp/ops.py",
        "cuda/__init__.py",
        "cuda/ops.py",
    ):
        assert (conv / rel).is_file(), rel
    assert not (conv / "long_conv.py").exists()
    assert "long_conv" not in _code(conv / "__init__.py")
    backends = VENDORED / "modules" / "backends"
    assert (backends / "__init__.py").is_file()
    assert not (backends / "triton_ascend").exists()
    assert "triton_ascend" not in _code(backends / "__init__.py")


_LAZY_SUBPROCESS = textwrap.dedent(
    """
    import os
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    import sys
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    # The gated-deltanet import must not pull the conv closure, which needs einops.
    from fla.modules import FusedRMSNormGated
    assert "fla.modules.conv" not in sys.modules
    from fla.modules import FusedRMSNormGated, ShortConvolution
    assert ShortConvolution.__module__ == "fla.modules.conv.short_conv"
    print("LAZY_OK")
    """
)


_KERNEL_SUBPROCESS = textwrap.dedent(
    """
    import os
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    import torch
    import torch.nn.functional as F
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    # The exact import lines the Ling 3.0 remote code uses.
    from fla.modules import FusedRMSNormGated, ShortConvolution
    if torch.version.hip is None:  # ROCm withholds fla.ops.kda on purpose
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
    from fla.ops.simple_gla.chunk import chunk_simple_gla

    def reference(x, weight, bias):
        # Depthwise causal conv1d + SiLU in fp64, one sequence [T, D].
        D, W = weight.shape
        y = F.conv1d(x.double().t()[None], weight.double()[:, None], bias.double(), padding=W - 1, groups=D)
        return F.silu(y[0, :, : x.shape[0]].t())

    def rel(a, b):
        return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)).item()

    torch.manual_seed(0)
    dev, dtype = "cuda", torch.bfloat16
    D, W = 2048, 4  # Ling-3.0-tiny: 16 heads x 128, short_conv_kernel_size 4
    conv = ShortConvolution(hidden_size=D, kernel_size=W, bias=True, activation="silu").to(dev, dtype)
    assert conv.backend == "triton"
    w = conv.weight[:, 0]

    # Dense batch, forward + backward.
    x = torch.randn(2, 300, D, device=dev, dtype=dtype, requires_grad=True)
    y, _ = conv(x)
    dy = torch.randn_like(y)
    (y.float() * dy.float()).sum().backward()
    gx, gw, gb = x.grad.clone(), conv.weight.grad[:, 0].clone(), conv.bias.grad.clone()
    x.grad = None; conv.weight.grad = None; conv.bias.grad = None
    wr = w.detach().double().requires_grad_(True)
    br = conv.bias.detach().double().requires_grad_(True)
    xr = x.detach().double().requires_grad_(True)
    yr = torch.stack([reference(xr[b], wr, br) for b in range(2)])
    (yr * dy.double()).sum().backward()
    dense = [rel(y, yr), rel(gx, xr.grad), rel(gw, wr.grad), rel(gb, br.grad)]
    assert all(e < 2e-2 for e in dense), ("dense", dense)

    # Packed varlen (cu_seqlens) must equal each sequence run alone.
    lens = [5, 130, 64, 1, 77]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.long)
    xp = torch.randn(1, sum(lens), D, device=dev, dtype=dtype)
    yp, _ = conv(xp, cu_seqlens=cu)
    yp_ref = torch.cat([reference(xp[0, s:e], w, conv.bias) for s, e in zip(cu[:-1].tolist(), cu[1:].tolist())])
    varlen = rel(yp[0], yp_ref)
    assert varlen < 2e-2, ("varlen", varlen)

    # Prefill with a returned cache, then single-token decode steps (the generate path).
    xs = torch.randn(3, 40, D, device=dev, dtype=dtype)
    with torch.no_grad():
        y_full, _ = conv(xs)
        y_pre, cache = conv(xs[:, :32], output_final_state=True)
        steps = [y_pre]
        for t in range(32, 40):
            y_t, cache = conv(xs[:, t : t + 1], cache=cache, output_final_state=True)
            steps.append(y_t)
    decode = rel(torch.cat(steps, 1), y_full)
    assert decode < 1e-2, ("decode", decode)
    print("errs", [round(e, 5) for e in dense + [varlen, decode]])
    print("SHORT_CONV_OK")
    """
)


def _run(code, marker):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert marker in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr[-4000:]}"


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla is injected only with CUDA + torch>=2.7 + triton>=3.3 (never on RDNA1)",
)
def test_short_conv_resolves_lazily_subprocess():
    _run(_LAZY_SUBPROCESS, "LAZY_OK")


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3 (never on RDNA1)",
)
def test_short_conv_matches_reference_subprocess():
    _run(_KERNEL_SUBPROCESS, "SHORT_CONV_OK")
