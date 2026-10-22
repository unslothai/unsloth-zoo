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


# The `kernels` package present: transformers' own route would fetch the hub triton_kernels (matmul_ogs).
_HUB_PROBE = (
    "import sys, torch\n"
    "import transformers.quantizers.quantizer_mxfp4 as Q\n"
    "Q.is_kernels_available = lambda *a, **k: True\n"
    "Q.is_accelerate_available = lambda *a, **k: True\n"
    "sys.modules['triton_kernels'] = None\n"
    "from unsloth_zoo.temporary_patches import gpt_oss as G\n"
    "G.patch_gpt_oss()\n"
    "torch.cuda.is_available = lambda: True\n"
    "torch.cuda.get_device_capability = lambda *a, **k: (8, 0)\n"
    "from transformers import Mxfp4Config\n"
    "q = Q.Mxfp4HfQuantizer(Mxfp4Config(), pre_quantized = True)\n"
    "q.validate_environment(device_map = {'': 0})\n"
    "print('DEQUANTIZE', q.quantization_config.dequantize)\n"
)


def _dequantizes_with_hub_kernels(switch):
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1", UNSLOTH_MXFP4_MATMUL_OGS = switch)
    out = subprocess.run([sys.executable, "-c", _HUB_PROBE], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("switch,expected", [("0", "DEQUANTIZE True"), ("1", "DEQUANTIZE False")])
def test_switch_off_also_dequantizes_when_the_kernels_package_is_installed(switch, expected):
    pytest.importorskip("transformers.quantizers.quantizer_mxfp4")
    assert _dequantizes_with_hub_kernels(switch) == expected
