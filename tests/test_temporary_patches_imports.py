# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Import-smoke suite for unsloth_zoo.temporary_patches: every submodule and the __init__ star-import chain must import;
common.torch_compile_options must stay a dict (rl_replacements imports it at module top); runs GPU-free under
tests/conftest.py."""

from __future__ import annotations

import importlib
import os

import pytest

# pytest.importorskip('torch') because every temporary_patches submodule imports torch; without torch this is a
# precondition failure, not a finding (macOS staging: pyproject has no torch on darwin/arm64).
pytest.importorskip("torch")


# Strip the GPU-init switch _GPU_INIT_GATE (UNSLOTH_ZOO_DISABLE_GPU_INIT) from child envs: when set, __init__.py:179
# skips device detection (DEVICE_TYPE cpu, device_memory 0); hf_xet_fallback sets it on the parent around a download
# spawn.
_GPU_INIT_GATE = "UNSLOTH_ZOO_DISABLE_GPU_INIT"


def _child_env(**overrides: str) -> dict:
    """This process's environment minus the switch that would skip what is under test."""
    env = {k: v for k, v in os.environ.items() if k != _GPU_INIT_GATE}
    env.update(overrides)
    return env


# Per-submodule import smoke list kept explicit, not a glob, so a dropped or renamed file shows as a missing test; new files must be added here.


TEMPORARY_PATCHES_SUBMODULES = [
    "unsloth_zoo.temporary_patches.common",
    "unsloth_zoo.temporary_patches.bitsandbytes",
    "unsloth_zoo.temporary_patches.bitsandbytes_large_tensors",
    "unsloth_zoo.temporary_patches.bitsandbytes_optimizer_sync",
    "unsloth_zoo.temporary_patches.compiled_model_identity",
    "unsloth_zoo.temporary_patches.conversion_mapping_rescope",
    "unsloth_zoo.temporary_patches.datasets_folder_metadata",
    "unsloth_zoo.temporary_patches.deepseek_v3_moe",
    "unsloth_zoo.temporary_patches.deepseek_v4",
    "unsloth_zoo.temporary_patches.deepseek_v4_sparse_flex",
    "unsloth_zoo.temporary_patches.ernie4_5_moe",
    "unsloth_zoo.temporary_patches.fla_vendor",
    "unsloth_zoo.temporary_patches.gdn_causal_conv1d",
    "unsloth_zoo.temporary_patches.flex_attention_bwd",
    "unsloth_zoo.temporary_patches.remote_code_vlm",
    "unsloth_zoo.temporary_patches.gemma",
    "unsloth_zoo.temporary_patches.gemma3n",
    "unsloth_zoo.temporary_patches.gemma4",
    "unsloth_zoo.temporary_patches.gemma4_banded_attention",
    "unsloth_zoo.temporary_patches.gemma4_float32",
    "unsloth_zoo.temporary_patches.gemma4_flash_sliding",
    "unsloth_zoo.temporary_patches.gemma4_moe",
    "unsloth_zoo.temporary_patches.glm4_moe",
    "unsloth_zoo.temporary_patches.gpt_oss",
    "unsloth_zoo.temporary_patches.gpt_oss_routed",
    "unsloth_zoo.temporary_patches.gpt_oss_grouped_qlora",
    "unsloth_zoo.temporary_patches.granitemoehybrid",
    "unsloth_zoo.temporary_patches.hub_kernel_export_check",
    "unsloth_zoo.temporary_patches.inductor_stride_guard",
    "unsloth_zoo.temporary_patches.inkling",
    "unsloth_zoo.temporary_patches.lfm2_moe",
    "unsloth_zoo.temporary_patches.llama4_moe",
    "unsloth_zoo.temporary_patches.ministral",
    "unsloth_zoo.temporary_patches.amd_aiter",
    "unsloth_zoo.temporary_patches.fp8_uncontained_weights",
    "unsloth_zoo.temporary_patches.mhc_sinkhorn",
    "unsloth_zoo.temporary_patches.misc",
    "unsloth_zoo.temporary_patches.muse_glimmer_banded_attention",
    "unsloth_zoo.temporary_patches.mixtral_moe",
    "unsloth_zoo.temporary_patches.moe_experts_interface",
    "unsloth_zoo.temporary_patches.moe_fused_swiglu",
    "unsloth_zoo.temporary_patches.moe_grouped_fp16",
    "unsloth_zoo.temporary_patches.moe_grouped_modulelist",
    "unsloth_zoo.temporary_patches.moe_nf4_grouped",
    "unsloth_zoo.temporary_patches.moe_ready_epoch",
    "unsloth_zoo.temporary_patches.moe_triton_kernels",
    "unsloth_zoo.temporary_patches.moe_routed",
    "unsloth_zoo.temporary_patches.moe_router_triton",
    "unsloth_zoo.temporary_patches.moe_routing_triton",
    "unsloth_zoo.temporary_patches.moe_utils",
    "unsloth_zoo.temporary_patches.moe_utils_bnb4bit",
    "unsloth_zoo.temporary_patches.moe_utils_fp8",
    "unsloth_zoo.temporary_patches.moe_utils_sonic",
    "unsloth_zoo.temporary_patches.mxfp4",
    "unsloth_zoo.temporary_patches.pixtral",
    "unsloth_zoo.temporary_patches.qwen3_5_moe",
    "unsloth_zoo.temporary_patches.qwen3_moe",
    "unsloth_zoo.temporary_patches.qwen3_moe_float32",
    "unsloth_zoo.temporary_patches.qwen3_next_moe",
    "unsloth_zoo.temporary_patches.qwen3_vl_moe",
    "unsloth_zoo.temporary_patches.qwen4_exp",
    "unsloth_zoo.temporary_patches.remote_mamba2",
    "unsloth_zoo.temporary_patches.sdpa_gqa",
    "unsloth_zoo.temporary_patches.sparse_attention",
    "unsloth_zoo.temporary_patches.sparse_attention_flex",
    "unsloth_zoo.temporary_patches.step3p7_moe",
    "unsloth_zoo.temporary_patches.t5",
    "unsloth_zoo.temporary_patches.utils",
]


@pytest.mark.parametrize("module_path", TEMPORARY_PATCHES_SUBMODULES)
def test_temporary_patches_submodule_imports(module_path):
    """Each temporary_patches submodule must import without raising."""
    importlib.import_module(module_path)


def test_temporary_patches_star_import_chain():
    """__init__ star-imports every submodule, so one import-time failure makes 'from unsloth_zoo.temporary_patches import *' fail wholesale."""
    importlib.import_module("unsloth_zoo.temporary_patches")


def test_torch_compile_options_is_dict():
    """common.torch_compile_options must stay a dict: rl_replacements imports it at module top, and a non-dict breaks every @torch.compile decorator there."""
    from unsloth_zoo.temporary_patches import common
    assert hasattr(common, "torch_compile_options"), (
        "common.torch_compile_options removed -- rl_replacements.py "
        "module-top import will fail."
    )
    opts = common.torch_compile_options
    assert isinstance(opts, dict), (
        f"common.torch_compile_options changed type to {type(opts).__name__}; "
        "every @torch.compile decorator that references it (selective_log_softmax, "
        "chunked_selective_log_softmax, chunked_hidden_states_selective_log_softmax) "
        "would break at import."
    )


def test_temporary_patches_submodule_list_is_complete():
    """TEMPORARY_PATCHES_SUBMODULES must match the files on disk; a new submodule missing from it would silently bypass the import smoke test."""
    import unsloth_zoo.temporary_patches as tp
    import pathlib

    pkg_dir = pathlib.Path(tp.__file__).parent
    on_disk = {
        f"unsloth_zoo.temporary_patches.{p.stem}"
        for p in pkg_dir.glob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("_")
    }
    on_test = set(TEMPORARY_PATCHES_SUBMODULES)
    missing = on_disk - on_test
    assert not missing, (
        "New temporary_patches submodule(s) on disk are NOT tested by "
        f"this suite: {sorted(missing)}. Add them to "
        "TEMPORARY_PATCHES_SUBMODULES above."
    )
    extra = on_test - on_disk
    assert not extra, (
        "TEMPORARY_PATCHES_SUBMODULES references modules that don't "
        f"exist on disk: {sorted(extra)}. Remove them."
    )


def test_gpt_oss_imports_without_visible_gpus():
    """gpt_oss.py computes device_memory at import; with UNSLOTH_ALLOW_CPU=1 DEVICE_TYPE stays 'cuda' on GPU-less hosts,
    so the lookup must be guarded; subprocess keeps conftest's mem_get_info stub from masking it."""
    import subprocess
    import sys

    env = _child_env(
        UNSLOTH_ALLOW_CPU = "1",
        CUDA_VISIBLE_DEVICES = "",
        HIP_VISIBLE_DEVICES = "",
    )
    result = subprocess.run(
        [sys.executable, "-c",
         "import unsloth_zoo.temporary_patches.gpt_oss; print('IMPORT_OK')"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout


def test_xet_submodule_import_does_not_query_free_device_memory():
    """Importing the xet download submodule must not call mem_get_info, which creates an accelerator context; device
    properties give the total capacity gpt_oss needs."""
    import subprocess
    import sys

    script = r'''
import torch

class Props:
    major = 8
    minor = 9
    total_memory = 48 * 1024**3
    multi_processor_count = 108
    name = "stub"

def forbidden(*args, **kwargs):
    raise AssertionError("mem_get_info was called during import")

torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda.get_device_properties = lambda *args, **kwargs: Props()
torch.cuda.mem_get_info = forbidden
torch.cuda.memory.mem_get_info = forbidden

import unsloth_zoo.hf_xet_fallback
from unsloth_zoo.temporary_patches import gpt_oss
assert gpt_oss.device_memory == Props.total_memory
assert gpt_oss.use_combo_kernels is True
print("IMPORT_OK")
'''
    env = _child_env(UNSLOTH_IS_PRESENT = "1", UNSLOTH_ALLOW_CPU = "1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout


def test_flex_attention_import_does_not_query_free_device_memory():
    """flex_attention/utils.py must not call mem_get_info() at import: it needs only total capacity like gpt_oss, and
    gpt_oss imports it, so the context would return."""
    import subprocess
    import sys

    script = r'''
import torch

class Props:
    major = 8
    minor = 9
    total_memory = 48 * 1024**3
    multi_processor_count = 108
    name = "stub"

def forbidden(*args, **kwargs):
    raise AssertionError("mem_get_info was called during import")

torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda.get_device_properties = lambda *args, **kwargs: Props()
torch.cuda.mem_get_info = forbidden
torch.cuda.memory.mem_get_info = forbidden

from unsloth_zoo.flex_attention import utils
# 48 GB is above the 16 GB threshold, so the low-memory kernel options stay off.
assert utils.kernel_options is None, utils.kernel_options
print("IMPORT_OK")
'''
    env = _child_env(UNSLOTH_IS_PRESENT = "1", UNSLOTH_ALLOW_CPU = "1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout
