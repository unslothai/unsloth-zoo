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
    race = chunk_o.IS_NVIDIA_BLACKWELL and not chunk_o.TRITON_ABOVE_3_8_0
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
