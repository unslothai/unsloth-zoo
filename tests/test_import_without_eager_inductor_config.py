# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""torch 2.4 does not load torch._inductor.config on `import torch`, so module-scope reads of it must import it first.
Each case runs in a fresh interpreter that unloads the config before importing unsloth_zoo, mimicking torch 2.4 on
any torch version."""

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("torch")

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_UNLOAD_CONFIG = """
import sys, torch, torch._inductor
sys.modules.pop("torch._inductor.config", None)
if "config" in vars(torch._inductor):
    delattr(torch._inductor, "config")
assert not hasattr(torch._inductor, "config")
"""

# Modules that read torch._inductor.config at import time, directly or through temporary_patches.
_MODULES = (
    "unsloth_zoo.temporary_patches.common",
    "unsloth_zoo.temporary_patches",
    "unsloth_zoo.compiler",
    "unsloth_zoo.loss_utils",
    "unsloth_zoo.patching_utils",
)


def _run(code):
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ZOO_DISABLE_GPU_INIT"}
    env.update(PYTHONPATH = os.pathsep.join(path), CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1")
    return subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 900,
    )


@pytest.mark.parametrize("module", _MODULES)
def test_imports_when_inductor_config_is_not_preloaded(module):
    code = _UNLOAD_CONFIG + f"import {module}, unsloth_zoo\nprint('ZOO', unsloth_zoo.__file__)\n"
    out = _run(code)
    assert out.returncode == 0, out.stderr[-3000:]
    assert f"ZOO {_ROOT / 'unsloth_zoo' / '__init__.py'}" in out.stdout, out.stdout
