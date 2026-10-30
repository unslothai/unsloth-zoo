# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Blackwell + Triton < 3.8: chunk_bwd_kernel_dv_local races at 8 warps (fla issue #1327, triton #10590)."""

import os

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    import unsloth_zoo  # noqa: F401
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_chunk_o():
    from unsloth_zoo.temporary_patches.fla_vendor import (
        _inject_vendored_fla,
        _vendored_injection_supported,
    )

    if not _vendored_injection_supported():
        pytest.skip("vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3")
    injected, _ = _inject_vendored_fla()
    if not injected:
        pytest.skip("vendored fla injection failed on this host")
    import fla.ops.common.chunk_o as chunk_o
    return chunk_o


def _autotuner(fn):
    while not hasattr(fn, "configs"):
        fn = fn.fn
    return fn


@pytest.mark.skipif(not _cuda_available(), reason = "needs CUDA")
def test_dv_local_warps_follow_the_race_guard():
    chunk_o = _load_chunk_o()
    warps = {c.num_warps for c in _autotuner(chunk_o.chunk_bwd_kernel_dv_local).configs}
    race = (chunk_o.IS_NVIDIA_BLACKWELL or chunk_o._any_visible_blackwell()) and not chunk_o.TRITON_ABOVE_3_8_0
    assert chunk_o.BLACKWELL_8_WARP_RACE == race
    assert warps == set(chunk_o.CHUNK_BWD_DV_LOCAL_NUM_WARPS)
    if race:
        assert 8 not in warps
    else:
        assert warps == set(chunk_o.NUM_WARPS)


@pytest.mark.skipif(not _cuda_available(), reason = "needs CUDA")
@pytest.mark.parametrize("varlen", [False, True], ids = ["dense", "varlen"])
def test_every_dv_local_config_is_deterministic(varlen):
    # fla issue #1327 repro: B200 + Triton 3.7.1 gives 2 of 19 differing calls at 8 warps, dense.
    import torch
    import triton

    chunk_o = _load_chunk_o()
    import fla.ops.utils as fla_utils
    tuner = _autotuner(chunk_o.chunk_bwd_kernel_dv_local)
    T, H, D = 65536, 48, 128
    torch.manual_seed(0)
    q, k = (torch.nn.functional.normalize(torch.randn(1, T, H, D, device = "cuda"), dim = -1).bfloat16() for _ in range(2))
    do = torch.randn(1, T, H, D, device = "cuda").bfloat16()
    cu_seqlens = torch.tensor([0, 5000, T], dtype = torch.int32, device = "cuda") if varlen else None
    g = fla_utils.chunk_local_cumsum(-torch.rand(1, T, H, device = "cuda"), chunk_size = 64, cu_seqlens = cu_seqlens)
    configs = list(tuner.configs)
    bad = []
    try:
        for config in configs:
            tuner.configs = [triton.Config({}, num_warps = config.num_warps, num_stages = config.num_stages)]
            tuner.cache.clear()
            first = chunk_o.chunk_bwd_dv_local(q = q, k = k, do = do, g = g, scale = D ** -0.5, cu_seqlens = cu_seqlens)
            differ = sum(
                not torch.equal(chunk_o.chunk_bwd_dv_local(q = q, k = k, do = do, g = g, scale = D ** -0.5, cu_seqlens = cu_seqlens), first)
                for _ in range(19)
            )
            if differ:
                bad.append((config.num_warps, config.num_stages, differ))
    finally:
        tuner.configs = configs
        tuner.cache.clear()
    assert not bad, f"nondeterministic dv_local configs (warps, stages, differing calls of 19): {bad}"


def _guard_warps(capabilities, import_index, triton_above_3_8_0 = False):
    # Run chunk_o.py's own module-level guard with a fake torch whose devices have ``capabilities``
    # and whose import-time current device is ``import_index``: no CUDA or fla import needed.
    import ast
    import pathlib
    import types

    import unsloth_zoo
    src = pathlib.Path(unsloth_zoo.__file__).parent / "_vendored" / "fla" / "ops" / "common" / "chunk_o.py"
    tree = ast.parse(src.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        body.append(node)
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "CHUNK_BWD_DV_LOCAL_NUM_WARPS" for t in node.targets
        ):
            break
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(capabilities),
        current_device = lambda: import_index,
        get_device_capability = lambda i = None: capabilities[import_index if i is None else i],
    )
    ns = {
        "torch": types.SimpleNamespace(cuda = cuda, version = types.SimpleNamespace(hip = None)),
        "functools": __import__("functools"),
        "check_shared_mem": lambda *a, **k: True,
        "IS_NVIDIA": True,
        "IS_NVIDIA_HOPPER": capabilities[import_index][0] == 9,
        "IS_NVIDIA_BLACKWELL": capabilities[import_index][0] in (10, 12),
        "TRITON_ABOVE_3_8_0": triton_above_3_8_0,
    }
    exec(compile(ast.Module(body = body, type_ignores = []), str(src), "exec"), ns)
    return ns["CHUNK_BWD_DV_LOCAL_NUM_WARPS"]


@pytest.mark.parametrize("import_index", [0, 1], ids = ["imported_on_ampere", "imported_on_blackwell"])
def test_dv_local_guard_covers_a_blackwell_card_that_was_not_current_at_import(import_index):
    # The autotune configs (and the autotune cache) are shared by every device in the process, so a
    # Blackwell card anywhere in it must keep 8 warps out, whichever card was current at import.
    assert 8 not in _guard_warps([(8, 0), (10, 0)], import_index)
    assert 8 in _guard_warps([(8, 0), (10, 0)], import_index, triton_above_3_8_0 = True)
    assert 8 in _guard_warps([(8, 0), (8, 9)], import_index)
