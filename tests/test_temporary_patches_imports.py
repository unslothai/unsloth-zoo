# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Import-smoke regression suite for `unsloth_zoo.temporary_patches`.

The temporary-patches subsystem is the model-specific monkey-patch
layer that lands ahead of upstream HF/TRL changes. It has 22
submodules (one per model family) and an `__init__.py` that
star-imports every one of them. A broken decorator or top-level
syntax error in ANY submodule cascades into the whole package
failing to import, which is exactly what zoo's downstream users
hit at training time -- a confusing `ImportError: cannot import
name 'PatchUnsloth_GPT_OSS_Triton'` rather than the actual file
that broke.

This suite pins that contract:

  - Every submodule imports cleanly.
  - The `__init__.py` star-import chain succeeds (so
    `from unsloth_zoo.temporary_patches import *` doesn't blow up).
  - `temporary_patches.common.torch_compile_options` is a dict
    (rl_replacements.py imports it at module top, so a contract
    break here breaks RL training too).

Runs under the GPU-free harness in `tests/conftest.py` which
pre-loads `unsloth_zoo.device_type` under a mocked
`torch.cuda.is_available()`. No GPU required; no actual model
forward pass.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Per-submodule import smoke. One parametrize per file under
# unsloth_zoo/temporary_patches/. New files added there should land on
# this list -- the suite is intentionally explicit (not a glob) so a
# silent drop or rename surfaces as a missing test, not a green CI.
# ---------------------------------------------------------------------------


TEMPORARY_PATCHES_SUBMODULES = [
    "unsloth_zoo.temporary_patches.common",
    "unsloth_zoo.temporary_patches.bitsandbytes",
    "unsloth_zoo.temporary_patches.deepseek_v3_moe",
    "unsloth_zoo.temporary_patches.flex_attention_bwd",
    "unsloth_zoo.temporary_patches.gemma",
    "unsloth_zoo.temporary_patches.gemma3n",
    "unsloth_zoo.temporary_patches.gemma4",
    "unsloth_zoo.temporary_patches.gemma4_moe",
    "unsloth_zoo.temporary_patches.glm4_moe",
    "unsloth_zoo.temporary_patches.gpt_oss",
    "unsloth_zoo.temporary_patches.ministral",
    "unsloth_zoo.temporary_patches.misc",
    "unsloth_zoo.temporary_patches.moe_bnb",
    "unsloth_zoo.temporary_patches.moe_utils",
    "unsloth_zoo.temporary_patches.mxfp4",
    "unsloth_zoo.temporary_patches.pixtral",
    "unsloth_zoo.temporary_patches.qwen3_5_moe",
    "unsloth_zoo.temporary_patches.qwen3_moe",
    "unsloth_zoo.temporary_patches.qwen3_next_moe",
    "unsloth_zoo.temporary_patches.qwen3_vl_moe",
    "unsloth_zoo.temporary_patches.utils",
]


@pytest.mark.parametrize("module_path", TEMPORARY_PATCHES_SUBMODULES)
def test_temporary_patches_submodule_imports(module_path):
    """Each temporary_patches submodule must import without raising."""
    importlib.import_module(module_path)


def test_temporary_patches_star_import_chain():
    """`unsloth_zoo.temporary_patches.__init__` star-imports every
    submodule above. If ANY submodule blows up at import time, the
    star-import chain fails wholesale and downstream `from
    unsloth_zoo.temporary_patches import *` users get a wall of red.
    """
    importlib.import_module("unsloth_zoo.temporary_patches")


def test_torch_compile_options_is_dict():
    """`temporary_patches.common.torch_compile_options` is imported
    by `unsloth_zoo.rl_replacements` at module top level. If the
    contract changes from dict to None / callable / removed, every
    @torch.compile decorator in rl_replacements.py breaks at import.
    """
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
    """The hand-maintained TEMPORARY_PATCHES_SUBMODULES list above
    must stay in sync with the actual files on disk. A new
    submodule added to the directory without being added here would
    silently bypass the per-submodule import smoke above.
    """
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
