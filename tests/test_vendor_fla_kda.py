# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla serves fla.ops.kda, which glm5_next (GLM-5.3-Flash) and kimi_linear resolve through
their kernel-hub wrappers, so they train on the Triton kernels without fla installed."""

import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"

_GPU = pytest.mark.skipif(
    not fla_vendor._vendored_injection_supported() or fla_vendor._gpu_lacks_dot_instructions(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3 (never on RDNA1)",
)


def test_kda_closure_is_vendored():
    kda = VENDORED / "ops" / "kda"
    for name in ("__init__", "chunk", "chunk_bwd", "chunk_fwd", "chunk_intra",
                 "chunk_intra_token_parallel", "fused_recurrent", "gate", "wy_fast"):
        assert (kda / f"{name}.py").is_file(), name
    for name in ("__init__", "chunk", "fused_chunk", "fused_recurrent"):
        assert (VENDORED / "ops" / "gla" / f"{name}.py").is_file(), name
    assert not (kda / "naive.py").exists()
    assert not (VENDORED / "ops" / "gla" / "naive.py").exists()
    assert not (kda / "backends" / "tilelang").exists()
    backends = (kda / "backends" / "__init__.py").read_text()
    code = "\n".join(ln for ln in backends.splitlines() if not ln.lstrip().startswith("#"))
    assert "FlashKDABackend" in code and "tilelang" not in code.lower()


def test_sm100_autotune_guard_is_backported():
    """fla #1109: BK=32 with 4 or 8 warps hits an illegal memory access on SM100 + triton 3.3."""
    code = (VENDORED / "ops" / "kda" / "chunk_bwd.py").read_text()
    assert "if not (IS_NVIDIA_SM100 and BK == 32 and num_warps != 2)" in code
    device = (VENDORED / "utils" / "_device.py").read_text()
    assert "IS_NVIDIA_SM100 = (IS_NVIDIA and torch.cuda.get_device_capability()[0] == 10)" in device
    assert "'IS_NVIDIA_SM100'" in (VENDORED / "utils" / "__init__.py").read_text()


def test_kernel_hub_table_covers_kda():
    table = fla_vendor._KERNEL_HUB_DECORATED
    assert table["chunk_kimi_delta_attention"] == ("fla.ops.kda", "chunk_kda")
    assert table["recurrent_kimi_delta_attention"] == ("fla.ops.kda", "fused_recurrent_kda")
    for package in ("glm5_next", "kimi_linear"):
        assert package in fla_vendor._repair_kernel_hub_closures.__defaults__[0]
        assert package in fla_vendor._force_kernel_hub_fallback.__defaults__[0]
        assert package in fla_vendor._block_fla_hub_decorator.__defaults__[0]


def test_rdna1_forces_kda_wrappers_to_torch(monkeypatch):
    if importlib.util.find_spec("transformers.integrations.hub_kernels") is None:
        pytest.skip("transformers without kernel-hub wrappers")

    def torch_chunk(*args, **kwargs):
        return "torch"

    def wrapper(*args, **kwargs):
        return "fla"

    wrapper.__wrapped__ = torch_chunk
    name = "transformers.models.kimi_linear.modeling_kimi_linear"
    module = types.ModuleType(name)
    module.chunk_kimi_delta_attention = wrapper
    monkeypatch.setitem(sys.modules, name, module)
    forced = fla_vendor._force_kernel_hub_fallback()
    assert f"{name}.chunk_kimi_delta_attention" in forced
    assert module.chunk_kimi_delta_attention is torch_chunk


_KERNEL_CHILD = textwrap.dedent(
    """
    import os, sys, torch
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    vendored = os.path.realpath(_vendored_fla_dir())
    leaked = [n for n, m in sys.modules.items() if n.startswith("fla")
              and getattr(m, "__file__", None) and not os.path.realpath(m.__file__).startswith(vendored)]
    assert not leaked, leaked

    def reference(q, k, v, g, beta, scale):
        # fla 0.5.1 naive_recurrent_kda in float64.
        q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
        q = q * scale
        B, T, H, K = q.shape
        S = q.new_zeros(B, H, K, v.shape[-1])
        out = []
        for t in range(T):
            S = S * g[:, t].exp()[..., None]
            u = v[:, t] - (k[:, t][..., None] * S).sum(-2)
            S = S + torch.einsum("bhk,bhv->bhkv", beta[:, t][..., None] * k[:, t], u)
            out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
        return torch.stack(out, 1), S

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True)

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm().clamp_min(1e-12)).item()

    torch.manual_seed(0)
    dev = "cuda"
    B, H, K = 2, 4, 128
    for fn_name, T, varlen in (("chunk", 300, False), ("chunk", 300, True), ("fused_recurrent", 64, False)):
        fn = chunk_kda if fn_name == "chunk" else fused_recurrent_kda
        Bx = 1 if varlen else B
        q, k, v = (torch.randn(Bx, T, H, K, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        g = (-torch.rand(Bx, T, H, K, device=dev) * 0.2).requires_grad_(True)
        beta = torch.rand(Bx, T, H, device=dev, dtype=torch.bfloat16).requires_grad_(True)
        do = torch.randn(Bx, T, H, K, device=dev, dtype=torch.bfloat16)
        kw = dict(use_qk_l2norm_in_kernel=True)
        cu = None
        if varlen:
            cu = torch.tensor([0, 37, 200, T], device=dev, dtype=torch.int32)
            kw["cu_seqlens"] = cu
        o, _ = fn(q, k, v, g, beta, **kw)
        grads = None
        if fn_name == "chunk":
            (o.float() * do.float()).sum().backward()
            grads = [x.grad.double() for x in (q, k, v, g, beta)]
            for x in (q, k, v, g, beta):
                x.grad = None
        spans = [(int(a), int(b)) for a, b in zip(cu[:-1], cu[1:])] if varlen else [(0, T)]
        ref_o = []
        for a, b in spans:
            ro, _ = reference(l2(q[:, a:b].double()), l2(k[:, a:b].double()), v[:, a:b], g[:, a:b], beta[:, a:b], K ** -0.5)
            ref_o.append(ro)
        ref_o = torch.cat(ref_o, 1)
        errs = [rel(o, ref_o)]
        if grads is not None:
            (ref_o * do.double()).sum().backward()
            errs += [rel(a, x.grad.double()) for a, x in zip(grads, (q, k, v, g, beta))]
        assert all(e < 2e-2 for e in errs), (fn_name, varlen, errs)
        print(fn_name, varlen, [round(e, 5) for e in errs])
    from fla.ops.kda import chunk_bwd
    from fla.utils import IS_NVIDIA_SM100
    tuner = chunk_bwd.chunk_kda_bwd_kernel_wy_dqkg_fused
    while not hasattr(tuner, "configs"):
        tuner = tuner.fn
    bad = [c for c in tuner.configs if c.kwargs["BK"] == 32 and c.num_warps != 2]
    assert not (IS_NVIDIA_SM100 and bad), bad
    print("KDA_OK")
    """
)


@_GPU
def test_kda_kernels_match_reference_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _KERNEL_CHILD], env = env,
                          capture_output = True, text = True, timeout = 1200)
    assert "KDA_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"


def _installed_fla():
    import importlib.metadata as md
    for name in ("fla-core", "flash-linear-attention"):
        try:
            md.distribution(name)
            return True
        except md.PackageNotFoundError:
            pass
    return False


_BIND_CHILD = textwrap.dedent(
    """
    import importlib, json, sys
    order, package = sys.argv[1], sys.argv[2]
    modname = f"transformers.models.{package}.modeling_{package}"
    from unsloth_zoo.temporary_patches import fla_vendor
    if order == "before":
        importlib.import_module(modname)
    fla_vendor.patch_vendor_fla()
    module = importlib.import_module(modname)
    out = {}
    for attribute in ("chunk_kimi_delta_attention", "recurrent_kimi_delta_attention"):
        impl = fla_vendor._resolved_implementation(getattr(module, attribute))
        out[attribute] = f"{impl.__module__}.{impl.__name__}"
    print("@@@" + json.dumps(out))
    """
)


@_GPU
@pytest.mark.parametrize("order", ["after", "before"])
@pytest.mark.parametrize("package", ["glm5_next", "kimi_linear"])
def test_kda_models_bind_vendored_kernels(package, order):
    if importlib.util.find_spec(f"transformers.models.{package}") is None:
        pytest.skip(f"transformers without {package}")
    if _installed_fla():
        pytest.skip("an installed fla would decide the before-import binding")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _BIND_CHILD, order, package], env = env,
                          capture_output = True, text = True, timeout = 600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    import json
    line = next(l for l in proc.stdout.splitlines() if l.startswith("@@@"))
    out = json.loads(line[3:])
    assert out["chunk_kimi_delta_attention"] == "fla.ops.kda.chunk.chunk_kda", out
    assert out["recurrent_kimi_delta_attention"] == "fla.ops.kda.fused_recurrent.fused_recurrent_kda", out


_ROCM_CHILD = textwrap.dedent(
    """
    import importlib, json, sys, torch
    torch.version.hip = "7.0.0"
    from unsloth_zoo.temporary_patches import fla_vendor
    fla_vendor.patch_vendor_fla()
    import fla
    out = {"vendored": bool(getattr(fla, "_UNSLOTH_VENDORED_FLA", False))}
    try:
        import fla.ops.kda
        out["kda"] = "imported"
    except ImportError:
        out["kda"] = "withheld"
    import fla.ops.gated_delta_rule
    out["gated_delta"] = "imported"
    if importlib.util.find_spec("transformers.models.kimi_linear") is not None:
        from transformers.models.kimi_linear import modeling_kimi_linear as mk
        impl = fla_vendor._resolved_implementation(mk.chunk_kimi_delta_attention)
        out["kimi_chunk"] = f"{impl.__module__}.{impl.__name__}"
    print("@@@" + json.dumps(out))
    """
)


@_GPU
def test_rocm_withholds_kda_and_keeps_gated_delta():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _ROCM_CHILD], env = env,
                          capture_output = True, text = True, timeout = 600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    import json
    out = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("@@@"))[3:])
    assert out["vendored"] and out["kda"] == "withheld" and out["gated_delta"] == "imported", out
    if "kimi_chunk" in out:
        assert not out["kimi_chunk"].startswith("fla."), out
