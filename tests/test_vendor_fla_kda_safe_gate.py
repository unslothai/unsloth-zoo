# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""GLM-5.3-Flash's bounded KDA gates use the fla safe_gate path (upstream fla 1242 and 1554 backported); Kimi-Linear,
unbounded configs and unsupported hosts keep safe_gate=False."""

import importlib.util
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
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"


def _gpu_ok():
    if not fla_vendor._vendored_injection_supported() or fla_vendor._gpu_lacks_dot_instructions():
        return False
    import torch
    import triton
    from packaging.version import Version
    if getattr(torch.version, "hip", None) is not None:
        return False  # fla_vendor withholds fla.ops.kda on ROCm and never routes safe_gate there
    return torch.cuda.get_device_capability() >= (8, 0) and Version(triton.__version__) >= Version("3.6.0")


_GPU = pytest.mark.skipif(not _gpu_ok(), reason = "safe_gate needs the vendored fla on CUDA (not ROCm) SM80+ with triton >= 3.6")


def test_fla_1242_and_1554_are_backported():
    code = (VENDORED / "ops" / "kda" / "chunk_intra.py").read_text()
    # 1242: the diagonal anchor is the block's first gate minus a fixed shift, never a later token.
    assert code.count("b_gn = b_gn - 0.5 * (BC - 1) * GATE_SHIFT") == 3
    assert "min(BC//2, T - i_ti - 1)" not in code and "min(BC // 2, T - i_ti - 1)" not in code
    assert code.count("GATE_SHIFT=(-lower_bound if lower_bound is not None else 2.) * RCP_LN2") == 2
    # 1554: bf16x3 for the four safe-gate backward dots, ieee before triton 3.6.
    assert "SAFE_GATE_DOT_PRECISION = tl.constexpr('bf16x3' if TRITON_ABOVE_3_6_0 and not IS_NPU else 'ieee')" in code
    assert code.count("input_precision=SAFE_GATE_DOT_PRECISION") == 4


def _fake_glm5(monkeypatch, hardware = True):
    """Fake glm5_next module whose chunk_kimi_delta_attention closes over `implementation` like transformers' kernel-hub fallback wrapper."""
    import types
    seen = []

    def chunk_kda(q, g = None, safe_gate = False, lower_bound = None):
        seen.append((safe_gate, lower_bound))

    def torch_ref(q, g = None, **kwargs):
        seen.append("torch")

    implementation = chunk_kda

    def chunk_kimi_delta_attention(*args, **kwargs):
        return implementation(*args, **kwargs)

    chunk_kimi_delta_attention.__wrapped__ = torch_ref
    module = types.ModuleType(fla_vendor._GLM5_NEXT_MODULE)
    module.chunk_kimi_delta_attention = chunk_kimi_delta_attention
    monkeypatch.setitem(sys.modules, fla_vendor._GLM5_NEXT_MODULE, module)
    monkeypatch.setattr(fla_vendor, "_kda_safe_gate_state", {"bound": None, "bounds": set(), "layers": 0})
    monkeypatch.setattr(fla_vendor, "_kda_safe_gate_hardware_ok", lambda: hardware)
    monkeypatch.delenv("UNSLOTH_KDA_SAFE_GATE", raising = False)
    return module, chunk_kda, seen


@pytest.mark.parametrize("bound, expected", [(-5.0, -5.0), (-5, -5.0), (-1.5, -1.5), (None, None), (-8.0, None),
                                             (0.0, None), (True, None), ("-5", None)])
def test_bound_is_read_from_config(monkeypatch, bound, expected):
    module, chunk_kda, seen = _fake_glm5(monkeypatch)
    fla_vendor._glm5_record_kda_bound(bound)
    module.chunk_kimi_delta_attention(1, g = 2)
    assert seen == [(expected is not None, expected)]
    # The repair / binding checks still see the fla kernel itself.
    assert fla_vendor._resolved_implementation(module.chunk_kimi_delta_attention) is chunk_kda


def test_mixed_bounds_kill_switch_and_hardware_disable(monkeypatch):
    module, _, seen = _fake_glm5(monkeypatch)
    fla_vendor._glm5_record_kda_bound(-5.0)
    fla_vendor._glm5_record_kda_bound(-3.0)  # two glm5_next models with different bounds: off
    module.chunk_kimi_delta_attention(1)
    module, _, seen2 = _fake_glm5(monkeypatch)
    monkeypatch.setenv("UNSLOTH_KDA_SAFE_GATE", "0")
    fla_vendor._glm5_record_kda_bound(-5.0)
    module.chunk_kimi_delta_attention(1)
    module, _, seen3 = _fake_glm5(monkeypatch, hardware = False)
    fla_vendor._glm5_record_kda_bound(-5.0)
    module.chunk_kimi_delta_attention(1)
    assert seen == seen2 == seen3 == [(False, None)]


def test_explicit_safe_gate_and_torch_fallback_untouched(monkeypatch):
    module, _, seen = _fake_glm5(monkeypatch)
    fla_vendor._glm5_record_kda_bound(-5.0)
    module.chunk_kimi_delta_attention(1, safe_gate = False)  # an explicit caller choice wins
    assert seen == [(False, -5.0)]
    module, _, seen = _fake_glm5(monkeypatch)
    cell = fla_vendor._hub_implementation_cell(module.chunk_kimi_delta_attention)
    cell.cell_contents = module.chunk_kimi_delta_attention.__wrapped__  # pure torch fallback bound
    fla_vendor._glm5_record_kda_bound(-5.0)
    module.chunk_kimi_delta_attention(1)
    assert seen == ["torch"]


_ROUTE_CHILD = textwrap.dedent(
    """
    import json, os, sys, torch
    from unsloth_zoo.temporary_patches import fla_vendor
    fla_vendor.patch_vendor_fla()
    from transformers.models.glm5_next import modeling_glm5_next as m
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    impl = fla_vendor._resolved_implementation(m.chunk_kimi_delta_attention)
    assert f"{impl.__module__}.{impl.__name__}" == "fla.ops.kda.chunk.chunk_kda", impl
    import importlib.util
    if importlib.util.find_spec("transformers.models.kimi_linear") is not None:
        from transformers.models.kimi_linear import modeling_kimi_linear as km
        kcell = fla_vendor._hub_implementation_cell(km.chunk_kimi_delta_attention)
        assert kcell is None or not hasattr(kcell.cell_contents, fla_vendor._KDA_SAFE_GATE_MARK)
    import fla.ops.kda.chunk as chunk_mod
    real_apply = chunk_mod.ChunkKDAFunction.apply
    safe_calls = [0]
    def spy(*args):
        safe_calls[0] += bool(args[-6])  # ..., safe_gate, lower_bound, chunk_size, disable_recompute, return_intermediate_states, cp_context
        return real_apply(*args)
    chunk_mod.ChunkKDAFunction.apply = spy
    out = {}
    for name, bound in (("bounded", -5.0), ("unbounded", None), ("out_of_range", -8.0)):
        torch.manual_seed(0)
        cfg = Glm5NextTextConfig(hidden_size = 256, linear_num_heads = 2, linear_head_dim = 128,
                                 linear_lower_bound = bound, num_hidden_layers = 1, layer_types = ["linear_attention"])
        x = torch.randn(1, 200, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        res = {}
        for env in ("1", "0"):
            os.environ["UNSLOTH_KDA_SAFE_GATE"] = env
            fla_vendor._kda_safe_gate_state["bounds"].clear()  # one model per process in practice
            torch.manual_seed(0)
            layer = m.Glm5NextTextLinearAttention(cfg, 0).cuda().to(torch.bfloat16)
            for p in layer.parameters():
                torch.nn.init.normal_(p, std = 0.02)
            before = safe_calls[0]
            x.grad = None
            layer.zero_grad()
            y = layer(x)
            y = y[0] if isinstance(y, tuple) else y
            y.float().square().sum().backward()
            res[env] = (y.detach().float(), x.grad.float(), safe_calls[0] - before)
        rel = lambda a, b: float((a - b).norm() / b.norm())
        out[name] = {"routed": res["1"][2], "routed_off": res["0"][2],
                     "rel_o": rel(res["1"][0], res["0"][0]), "rel_dx": rel(res["1"][1], res["0"][1])}
    print("@@@" + json.dumps(out))
    """
)


@_GPU
def test_glm5_layer_routes_bounded_gates_only():
    if importlib.util.find_spec("transformers.models.glm5_next") is None:
        pytest.skip("transformers without glm5_next")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _ROUTE_CHILD], env = env, capture_output = True, text = True, timeout = 1200)
    assert proc.returncode == 0, proc.stderr[-4000:]
    import json
    out = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("@@@"))[3:])
    assert out["bounded"]["routed"] == 1 and out["bounded"]["routed_off"] == 0, out
    assert out["unbounded"]["routed"] == 0 and out["out_of_range"]["routed"] == 0, out
    assert out["bounded"]["rel_o"] < 2e-2 and out["bounded"]["rel_dx"] < 2e-2, out
    for name in ("unbounded", "out_of_range"):
        assert out[name]["rel_o"] == 0.0 and out[name]["rel_dx"] == 0.0, out


_PARITY_CHILD = textwrap.dedent(
    """
    import torch, torch.nn.functional as F
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()
    from fla.ops.kda import chunk_kda

    def naive(q, k, v, g, beta):
        D = q.shape[-1]
        q = F.normalize(q, dim = -1, eps = 1e-6) * D ** -0.5
        k = F.normalize(k, dim = -1, eps = 1e-6)
        B, T, H, _ = q.shape
        S = q.new_zeros(B, H, D, v.shape[-1]); o = []
        for t in range(T):
            S = S * g[:, t].exp()[..., None]
            u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
            S = S + k[:, t][..., :, None] * u[..., None, :]
            o.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
        return torch.stack(o, 1)

    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
    worst = {}
    for gate in ("lower_bound", "glm5_init"):
        gen = torch.Generator(device = "cuda").manual_seed(0)
        r = lambda *s: torch.randn(*s, device = "cuda", generator = gen)
        T, H, D = 256, 4, 128
        q, k, v, do = (r(1, T, H, D).bfloat16().float() for _ in range(4))
        x = r(1, T, H, D)
        g = torch.full_like(x, -5.0) if gate == "lower_bound" else -5.0 * torch.sigmoid(0.5 * x + 1.0)
        beta = torch.sigmoid(r(1, T, H)).bfloat16().float()
        ref = [t.clone().requires_grad_() for t in (q, k, v, g, beta)]
        naive(*ref).backward(do)
        for safe in (False, True):
            xs = [t.to(torch.bfloat16 if i != 3 else torch.float32).detach().clone().requires_grad_()
                  for i, t in enumerate((q, k, v, g, beta))]
            kw = dict(safe_gate = True, lower_bound = -5.0) if safe else {}
            o, _ = chunk_kda(*xs, use_qk_l2norm_in_kernel = True, **kw)
            o.backward(do.bfloat16())
            errs = [rel(x.grad, y.grad) for x, y in zip(xs, ref)]
            worst[(gate, safe)] = max(errs)
            print(gate, safe, ["%.2e" % e for e in errs])
    for gate in ("lower_bound", "glm5_init"):
        # safe_gate stays within 2x of the token-by-token path and well under 1e-2
        assert worst[(gate, True)] < 1e-2 and worst[(gate, True)] < 2 * worst[(gate, False)] + 1e-3, worst
    print("SAFE_GATE_OK")
    """
)


@_GPU
def test_safe_gate_grads_match_fp32_recurrence_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _PARITY_CHILD], env = env, capture_output = True, text = True, timeout = 1200)
    assert "SAFE_GATE_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"
