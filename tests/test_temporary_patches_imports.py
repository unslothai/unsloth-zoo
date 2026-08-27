# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Import-smoke regression suite for `unsloth_zoo.temporary_patches`.

The subsystem star-imports every model-family submodule from __init__.py, so
a broken decorator or syntax error in any one cascades into a confusing
package-wide ImportError at training time. This suite pins:
  - Every submodule imports cleanly.
  - The __init__.py star-import chain succeeds.
  - `common.torch_compile_options` is a dict (rl_replacements.py imports it
    at module top, so a break here breaks RL training too).

Runs under the GPU-free harness in tests/conftest.py; no GPU required.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Per-submodule import smoke. Explicit (not a glob) so a silent drop or rename
# surfaces as a missing test; new files must be added to this list.
# ---------------------------------------------------------------------------


TEMPORARY_PATCHES_SUBMODULES = [
    "unsloth_zoo.temporary_patches.common",
    "unsloth_zoo.temporary_patches.bitsandbytes",
    "unsloth_zoo.temporary_patches.deepseek_v3_moe",
    "unsloth_zoo.temporary_patches.ernie4_5_moe",
    "unsloth_zoo.temporary_patches.fla_vendor",
    "unsloth_zoo.temporary_patches.flex_attention_bwd",
    "unsloth_zoo.temporary_patches.gemma",
    "unsloth_zoo.temporary_patches.gemma3n",
    "unsloth_zoo.temporary_patches.gemma4",
    "unsloth_zoo.temporary_patches.gemma4_banded_attention",
    "unsloth_zoo.temporary_patches.gemma4_float32",
    "unsloth_zoo.temporary_patches.gemma4_flash_sliding",
    "unsloth_zoo.temporary_patches.gemma4_moe",
    "unsloth_zoo.temporary_patches.glm4_moe",
    "unsloth_zoo.temporary_patches.gpt_oss",
    "unsloth_zoo.temporary_patches.lfm2_moe",
    "unsloth_zoo.temporary_patches.ministral",
    "unsloth_zoo.temporary_patches.amd_aiter",
    "unsloth_zoo.temporary_patches.misc",
    "unsloth_zoo.temporary_patches.muse_glimmer_banded_attention",
    "unsloth_zoo.temporary_patches.mixtral_moe",
    "unsloth_zoo.temporary_patches.moe_bnb",
    "unsloth_zoo.temporary_patches.moe_grouped_modulelist",
    "unsloth_zoo.temporary_patches.moe_utils",
    "unsloth_zoo.temporary_patches.moe_utils_bnb4bit",
    "unsloth_zoo.temporary_patches.moe_utils_fp8",
    "unsloth_zoo.temporary_patches.mxfp4",
    "unsloth_zoo.temporary_patches.pixtral",
    "unsloth_zoo.temporary_patches.qwen3_5_moe",
    "unsloth_zoo.temporary_patches.qwen3_moe",
    "unsloth_zoo.temporary_patches.qwen3_moe_float32",
    "unsloth_zoo.temporary_patches.qwen3_next_moe",
    "unsloth_zoo.temporary_patches.qwen3_vl_moe",
    "unsloth_zoo.temporary_patches.utils",
]


@pytest.mark.parametrize("module_path", TEMPORARY_PATCHES_SUBMODULES)
def test_temporary_patches_submodule_imports(module_path):
    """Each temporary_patches submodule must import without raising."""
    importlib.import_module(module_path)


def test_temporary_patches_star_import_chain():
    """The __init__ star-imports every submodule; if any blows up at import,
    downstream `from unsloth_zoo.temporary_patches import *` fails wholesale."""
    importlib.import_module("unsloth_zoo.temporary_patches")


def test_torch_compile_options_is_dict():
    """`common.torch_compile_options` is imported by rl_replacements at module
    top; a non-dict contract breaks every @torch.compile decorator there."""
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
    """TEMPORARY_PATCHES_SUBMODULES must stay in sync with files on disk; a new
    submodule not added here would silently bypass the import smoke above."""
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
    """gpt_oss.py computes device_memory at import; with UNSLOTH_ALLOW_CPU=1
    DEVICE_TYPE stays "cuda" on GPU-less hosts, so the capacity lookup must be
    guarded. Subprocess so conftest's mem_get_info stub cannot mask it."""
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "UNSLOTH_ALLOW_CPU": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [sys.executable, "-c",
         "import unsloth_zoo.temporary_patches.gpt_oss; print('IMPORT_OK')"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout


def test_xet_submodule_import_does_not_query_free_device_memory():
    """Importing a download helper must not create an accelerator context.

    gpt_oss only needs total capacity to select combo-kernel options. Device
    properties provide that without the context-creating mem_get_info call.
    """
    import os
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
    env = {
        **os.environ,
        "UNSLOTH_IS_PRESENT": "1",
        "UNSLOTH_ALLOW_CPU": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout


def test_flex_attention_import_does_not_query_free_device_memory():
    """The same probe pattern in flex_attention/utils.py.

    It only needs total capacity, like gpt_oss, and gpt_oss imports it, so leaving
    mem_get_info() here would put back the context the gpt_oss fix removes for anyone
    who reaches the flex-attention path.
    """
    import os
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
    env = {
        **os.environ,
        "UNSLOTH_IS_PRESENT": "1",
        "UNSLOTH_ALLOW_CPU": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout
