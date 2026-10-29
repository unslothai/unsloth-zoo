# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla backports for the KDA decode kernel (fla #1415 / #1416 / #1329 / #1135 /
#1571 / #1414), the KDA output kernel score mask (fla #1582) and the cached backend
package probe (fla #996)."""

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
    return getattr(torch.version, "hip", None) is None  # fla.ops.kda is withheld on ROCm


_GPU = pytest.mark.skipif(not _gpu_ok(), reason = "needs the vendored fla KDA kernels on CUDA")


def test_kda_decode_backports_present():
    code = (VENDORED / "ops" / "kda" / "fused_recurrent.py").read_text()
    # 1415: eight-value tiles, one warp, three stages, except on sm8x / ROCm (slower on A100).
    assert "BV, decode_warps, decode_stages = _kda_decode_config(v.device)" in code
    assert "num_warps=decode_warps,\n        num_stages=decode_stages," in code
    # 1416: dense outputs skip the zero fill, packed outputs keep it.
    assert "out = torch.empty_like(v) if cu_seqlens is None else torch.zeros_like(v)" in code
    # 1329 / 1135 / 1571 / 1414.
    assert "pid = tl.program_id(0).to(tl.int64)" in code
    assert "b_g = tl.load(p_g, mask=mask_k, other=0, eviction_policy='evict_last')" in code
    assert "if T == 0 and IS_CONTINUOUS_BATCHING:" in code
    assert code.count("i_t * stride_indices_tok") == 2
    # The public API is unchanged (no #1107 optional-A_log path, no fla.backends import).
    assert "HAS_A" not in code and "fla.backends" not in code
    assert "@torch.compiler.disable\ndef fused_recurrent_kda_fwd(" in code


def test_gla_output_kernel_masks_scores_in_the_load():
    code = (VENDORED / "ops" / "gla" / "chunk.py").read_text()
    assert "m_A = m_t[:, None] & m_s\n" in code
    assert "b_A = tl.load(p_A, mask=m_A, other=0.0).to(b_v.dtype)\n    b_o += tl.dot(b_A, b_v)" in code
    assert "b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)" not in code


_PROBE_CHILD = textwrap.dedent(
    """
    import importlib.util, sys
    sys.path.insert(0, sys.argv[1])
    calls = []
    real = importlib.util.find_spec
    def counting(name, *a, **k):
        calls.append(name)
        return real(name, *a, **k)
    importlib.util.find_spec = counting
    import fla.ops.backends as be

    class Probe(be.BaseBackend):
        backend_type = "probe"
        package_name = "unsloth_no_such_package_xyz"

    before = calls.count("unsloth_no_such_package_xyz")
    for _ in range(100):
        assert Probe.is_available() is False
    print("PROBES", calls.count("unsloth_no_such_package_xyz") - before)
    """
)


def test_backend_package_probe_is_cached():
    # Every dispatched chunk_kda forward / backward calls is_available() on the FlashKDA
    # backend; the find_spec behind it must run once per package, not once per call.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_CHILD, str(VENDORED.parent)],
        env = env, capture_output = True, text = True, timeout = 600,
    )
    assert "PROBES 1" in proc.stdout.splitlines(), f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"


_DECODE_CHILD = textwrap.dedent(
    """
    import torch
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()
    from fla.ops.kda import fused_recurrent_kda

    def naive(q, k, v, g, beta, h0, scale):
        q, k, v, g, beta, S = (x.double() for x in (q, k, v, g, beta, h0))
        o = []
        for t in range(q.shape[1]):
            S = S * g[:, t].exp()[..., None]
            u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
            S = S + k[:, t][..., :, None] * u[..., None, :]
            o.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
        return torch.stack(o, 1), S

    gen = torch.Generator(device = "cuda").manual_seed(0)
    r = lambda *s: torch.randn(*s, device = "cuda", generator = gen)
    l2 = lambda x: x / x.norm(dim = -1, keepdim = True)
    # GLM-5.3-Flash (64 heads) and Kimi K3 (96 heads), head dim 128, bounded gates.
    for B, T, H in ((1, 1, 64), (8, 1, 96), (2, 4, 64)):
        D = 128
        q, k = l2(r(B, T, H, D)).bfloat16(), l2(r(B, T, H, D)).bfloat16()
        v = r(B, T, H, D).bfloat16()
        g = -5.0 * torch.sigmoid(r(B, T, H, D))
        beta = torch.sigmoid(r(B, T, H)).bfloat16()
        h0 = 0.1 * r(B, H, D, D)
        o, ht = fused_recurrent_kda(q, k, v, g, beta, scale = D ** -0.5, initial_state = h0.clone(),
                                    output_final_state = True)
        ro, rh = naive(q, k, v, g, beta, h0, D ** -0.5)
        eo = float((o.double() - ro).norm() / ro.norm())
        eh = float((ht.double() - rh).norm() / rh.norm())
        print("decode", B, T, H, "%.2e %.2e" % (eo, eh))
        assert eo < 1e-2 and eh < 1e-4, (B, T, H, eo, eh)

    # A packed batch with an empty sequence returns that sequence's initial state (fla #1571).
    cu = torch.tensor([0, 0, 3, 5], device = "cuda")
    H, D = 4, 64
    q, k = l2(r(1, 5, H, D)), l2(r(1, 5, H, D))
    v, g, beta = r(1, 5, H, D), -torch.rand(1, 5, H, D, device = "cuda"), torch.rand(1, 5, H, device = "cuda")
    h0 = r(3, H, D, D)
    o, ht = fused_recurrent_kda(q, k, v, g, beta, scale = 0.125, initial_state = h0.clone(),
                                output_final_state = True, cu_seqlens = cu)
    assert torch.equal(ht[0], h0[0]), "empty packed sequence lost its state"
    for i, (bos, eos) in enumerate(((0, 0), (0, 3), (3, 5))):
        if eos == bos:
            continue
        ro, rh = naive(q[:, bos:eos], k[:, bos:eos], v[:, bos:eos], g[:, bos:eos], beta[:, bos:eos], h0[i:i + 1], 0.125)
        assert torch.allclose(o[:, bos:eos].double(), ro, atol = 1e-4) and torch.allclose(ht[i].double(), rh[0], atol = 1e-4)
    print("DECODE_OK")
    """
)


@_GPU
def test_kda_decode_matches_fp64_recurrence_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _DECODE_CHILD], env = env, capture_output = True, text = True, timeout = 1200)
    assert "DECODE_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"




@pytest.mark.parametrize("capability, hip, expected", [
    ((7, 5), None, (8, 1, 3)),     # T4: 2-14% faster with the new tiles
    ((8, 0), None, (32, 4, 2)),    # A100: 5-10% slower at batch 8-128 with them
    ((8, 9), None, (32, 4, 2)),
    ((9, 0), None, (8, 1, 3)),     # upstream measured H200
    ((10, 0), None, (8, 1, 3)),
    ((9, 4), "7.0", (32, 4, 2)),   # ROCm keeps the old config
])
def test_kda_decode_config_by_arch(monkeypatch, capability, hip, expected):
    # The vendored module needs fla injected (a CUDA host); the helper only needs torch.
    import functools
    import torch
    source = (VENDORED / "ops" / "kda" / "fused_recurrent.py").read_text(encoding = "utf-8")
    start = source.index("@functools.lru_cache(maxsize=None)\ndef _kda_decode_config")
    namespace = {"functools": functools, "torch": torch}
    exec(source[start:source.index("@torch.compiler.disable\ndef fused_recurrent_kda_fwd(")], namespace)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: capability)
    assert namespace["_kda_decode_config"](torch.device("cuda", 0)) == expected
