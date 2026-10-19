# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""UNSLOTH_MXFP4_MATMUL_OGS=0 keeps patch_gpt_oss off triton_kernels."""
import importlib.util, os, subprocess, sys
import pytest

_PROBE = (
    "import transformers.integrations.mxfp4 as M\n"
    "from unsloth_zoo.temporary_patches import gpt_oss as G\n"
    "G.patch_gpt_oss()\n"
    "print(M.Mxfp4GptOssExperts.__module__)\n"
)


def _patched_module(switch):
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1", UNSLOTH_MXFP4_MATMUL_OGS = switch)
    out = subprocess.run([sys.executable, "-c", _PROBE], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_switch_off_leaves_transformers_mxfp4_experts_alone():
    if importlib.util.find_spec("triton_kernels") is None:
        pytest.skip("no triton_kernels")
    if _patched_module("1") != "unsloth_zoo.temporary_patches.gpt_oss":
        pytest.skip("patch_gpt_oss does not install Mxfp4GptOssExperts on this transformers")
    assert _patched_module("0") == "transformers.integrations.mxfp4"
