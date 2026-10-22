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

"""Blackwell eight-warp guard in the vendored fla chunk_o (fla PR #1399, issue #1327).

On Blackwell with Triton < 3.8, ``chunk_bwd_kernel_dv_local`` and
``chunk_fwd_kernel_o`` race at num_warps=8: one warp reads a ``tl.dot``
accumulator from TMEM while another zero-fills it for the next ``tl.dot``.
``chunk_bwd_kernel_dv_local`` is the dv of the gated delta rule backward
(Qwen3.5 / Qwen3-Next).

The CPU tests evaluate the module-level config expressions straight from the
source for every (arch, Triton) combination, so they need no GPU. The GPU test
runs every offered dv_local config on identical inputs and requires bitwise
identical outputs that match an fp64 reference.
"""

import ast
import os
import types
from pathlib import Path

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

_FLA = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "_vendored" / "fla"
_CHUNK_O = _FLA / "ops" / "common" / "chunk_o.py"


def _module_ast():
    return ast.parse(_CHUNK_O.read_text())


def _eval_globals(*, hopper, blackwell, triton_3_8):
    """Evaluate chunk_o's module-level warp constants for one host."""
    ns = {
        "IS_NVIDIA_HOPPER": hopper,
        "IS_NVIDIA_BLACKWELL": blackwell,
        "TRITON_ABOVE_3_8_0": triton_3_8,
    }
    wanted = {"NUM_WARPS", "BLACKWELL_8_WARP_RACE", "CHUNK_BWD_DV_LOCAL_NUM_WARPS"}
    for node in _module_ast().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in wanted:
                ns[name] = eval(compile(ast.Expression(node.value), str(_CHUNK_O), "eval"), {}, ns)
    return ns


def _kernel_configs(kernel_name, **host):
    """(num_warps, num_stages, kwargs) offered by the kernel's autotune decorator."""
    ns = _eval_globals(**host)
    fake_triton = types.SimpleNamespace(
        Config=lambda kwargs, num_warps, num_stages: (num_warps, num_stages, tuple(sorted(kwargs.items())))
    )
    for node in _module_ast().body:
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", None) in ("fla_cache_autotune", "autotune") \
                        or (isinstance(dec, ast.Call) and getattr(dec.func, "attr", None) == "autotune"):
                    configs = next(kw.value for kw in dec.keywords if kw.arg == "configs")
                    code = compile(ast.Expression(configs), str(_CHUNK_O), "eval")
                    return eval(code, {"triton": fake_triton}, dict(ns))
    raise AssertionError(f"{kernel_name} autotune decorator not found")


_HOSTS = {
    # name: (hopper, blackwell)
    "hopper": (True, False),
    "blackwell": (False, True),
    "other": (False, False),
}


@pytest.mark.parametrize("triton_3_8", [False, True], ids=["triton<3.8", "triton>=3.8"])
@pytest.mark.parametrize("host", list(_HOSTS))
def test_dv_local_warps_per_arch_and_triton(host, triton_3_8):
    hopper, blackwell = _HOSTS[host]
    configs = _kernel_configs("chunk_bwd_kernel_dv_local", hopper=hopper, blackwell=blackwell, triton_3_8=triton_3_8)
    warps = sorted({c[0] for c in configs})
    if hopper:
        expected = [2, 4]
    elif blackwell and not triton_3_8:
        expected = [2, 4]
    else:
        expected = [2, 4, 8]
    assert warps == expected
    # num_stages untouched
    assert sorted({c[1] for c in configs}) == [2, 3, 4]
    assert len(configs) == len(expected) * 3


@pytest.mark.parametrize("triton_3_8", [False, True], ids=["triton<3.8", "triton>=3.8"])
@pytest.mark.parametrize("host", list(_HOSTS))
def test_fwd_o_configs_per_arch_and_triton(host, triton_3_8):
    hopper, blackwell = _HOSTS[host]
    configs = _kernel_configs("chunk_fwd_kernel_o", hopper=hopper, blackwell=blackwell, triton_3_8=triton_3_8)
    big = (8, 3, (("BK", 128), ("BV", 128)))
    small = [(4, 3, (("BK", 64), ("BV", 64))), (2, 3, (("BK", 32), ("BV", 32)))]
    # Unsloth keeps the 8-warp config off on every Blackwell Triton (not yet stress-tested on 3.8).
    if blackwell:
        assert configs == small
    else:
        assert configs == [big] + small


@pytest.mark.parametrize("host", list(_HOSTS))
def test_other_chunk_o_kernels_unchanged(host):
    """dqkwg and dv keep the shared NUM_WARPS list; only dv_local is narrowed."""
    hopper, blackwell = _HOSTS[host]
    for name in ("chunk_bwd_kernel_dqkwg", "chunk_bwd_kernel_dv"):
        for triton_3_8 in (False, True):
            configs = _kernel_configs(name, hopper=hopper, blackwell=blackwell, triton_3_8=triton_3_8)
            assert sorted({c[0] for c in configs}) == ([2, 4] if hopper else [2, 4, 8])


def test_triton_3_8_flag_matches_version():
    src = (_FLA / "utils" / "_compat.py").read_text()
    line = next(x for x in src.splitlines() if x.startswith("TRITON_ABOVE_3_8_0 ="))
    from packaging import version as package_version

    for ver, expect in [("3.7.1", False), ("3.7.9", False), ("3.8.0", True), ("3.8.1", True), ("3.9.0", True)]:
        ns = {"package_version": package_version, "triton": types.SimpleNamespace(__version__=ver)}
        exec(line, ns)
        assert ns["TRITON_ABOVE_3_8_0"] is expect, ver
    assert "TRITON_ABOVE_3_8_0" in (_FLA / "utils" / "__init__.py").read_text()


def _cuda_blackwell():
    try:
        import torch

        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return False
        return torch.cuda.get_device_capability()[0] in (10, 12)
    except Exception:
        return False


def _load_chunk_o():
    from unsloth_zoo.temporary_patches.fla_vendor import _inject_vendored_fla, _vendored_injection_supported

    if not _vendored_injection_supported():
        pytest.skip("vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3")
    injected, _ = _inject_vendored_fla()
    if not injected:
        pytest.skip("vendored fla injection failed on this host")
    import fla.ops.common.chunk_o as chunk_o

    return chunk_o


REPS = 50


@pytest.mark.skipif(not _cuda_blackwell(), reason="Blackwell determinism regression (fla #1327)")
def test_dv_local_every_offered_config_is_deterministic_and_correct(monkeypatch):
    import torch
    import triton

    chunk_o = _load_chunk_o()
    from fla.ops.utils import cache, chunk_local_cumsum

    monkeypatch.setattr(cache, "FLA_CACHE_MODE", cache.FlaCacheMode.DISABLED)
    tuner = chunk_o.chunk_bwd_kernel_dv_local
    while not isinstance(tuner, triton.runtime.Autotuner):
        tuner = tuner.fn
    # Checked after the run so that a tree without the guard still exercises (and
    # reports) every config it offers.
    from packaging import version as package_version

    triton_3_8 = package_version.parse(triton.__version__) >= package_version.parse("3.8.0")
    expected_warps = {2, 4, 8} if triton_3_8 else {2, 4}
    offered_warps = {c.num_warps for c in tuner.configs}

    # Qwen3.5-style GQA heads (H=16 q/k heads, HV=32 v heads, K=V=128) on a packed
    # varlen batch. On B200 + Triton 3.7.1 the 8-warp configs gave 2-8 of 100
    # identical calls with differing dv here; 2/4 warps gave 0 of 300.
    torch.manual_seed(0)
    T, H, HV, D = 65536, 16, 32, 128
    dev = "cuda"
    q, k = (torch.nn.functional.normalize(torch.randn(1, T, H, D, device=dev), dim=-1).bfloat16() for _ in range(2))
    do = torch.randn(1, T, HV, D, device=dev).bfloat16()
    cu_seqlens = torch.tensor([0, 5000, T], dtype=torch.int32, device=dev)
    g = chunk_local_cumsum(-torch.rand(1, T, HV, device=dev), chunk_size=64, cu_seqlens=cu_seqlens)
    scale = D ** -0.5

    # fp64 reference on the first sequence and v heads 0..3 (q/k heads 0..1), with A
    # rounded to bf16 before the second dot, as the kernel does.
    n_ref, hv_ref, BT = 5000, 4, 64
    rep = HV // H
    qd, kd = (x[0, :n_ref, : hv_ref // rep].double().repeat_interleave(rep, dim=1) for x in (q, k))
    dod, gd = (x[0, :n_ref, :hv_ref].double() for x in (do, g))
    ref = torch.zeros_like(dod)
    for s in range(0, n_ref, BT):
        e = min(s + BT, n_ref)
        A = torch.einsum("thd,shd->hts", kd[s:e], qd[s:e]) * scale
        A = A * torch.exp2(gd[s:e].T[:, None, :] - gd[s:e].T[:, :, None])
        A = A.masked_fill(~torch.ones(e - s, e - s, dtype=torch.bool, device=dev).triu(), 0)
        ref[s:e] = torch.einsum("hts,shd->thd", A.to(torch.bfloat16).double(), dod[s:e])

    def run():
        return chunk_o.chunk_bwd_dv_local(q=q, k=k, do=do, g=g, scale=scale, cu_seqlens=cu_seqlens)

    failures = []
    try:
        for config in list(tuner.configs):
            monkeypatch.setattr(tuner, "configs", [config])
            tuner.cache.clear()
            first = run()
            differing, worst = 0, 0.0
            for _ in range(REPS - 1):
                other = run()
                if not torch.equal(first, other):
                    differing += 1
                    worst = max(worst, (other.float() - first.float()).abs().max().item())
            err = (first[0, :n_ref, :hv_ref].double() - ref).abs().max().item()
            if differing or err > 0.05:
                failures.append((str(config), f"{differing}/{REPS - 1} runs differ (max {worst:.3g})", err))
    finally:
        tuner.cache.clear()
    assert not failures, failures
    assert offered_warps == expected_warps
