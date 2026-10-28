# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One pytest session must match one session per file where real mlx is installed (unslothai/unsloth#8138); tests/conftest.py gives each module its own mlx world."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent


def test_installing_a_world_undoes_the_shim_and_puts_it_back():
    script = textwrap.dedent("""
        import platform, sys
        import conftest
        real = (platform.system, platform.machine)
        had_mlx = sys.modules.get("mlx")
        worlds = conftest._MlxModuleWorlds()
        from mlx_simulation import simulate_mlx_on_torch, mlx_is_simulated
        simulate_mlx_on_torch()
        assert mlx_is_simulated()
        shim = worlds._capture()
        worlds._install(worlds.base)
        assert sys.modules.get("mlx") is had_mlx and "mlx_simulation" not in sys.modules
        assert (platform.system, platform.machine) == real
        assert not any(conftest._is_shim(f) for f in sys.meta_path)
        worlds._install(shim)
        import mlx.core as mx
        assert "mlx_simulation" in mx.__name__ and platform.system() == real[0]()
        worlds._install(worlds.base)
        print("OK")
    """)
    out = subprocess.run(
        [sys.executable, "-c", script], cwd = _TESTS, capture_output = True, text = True, timeout = 600,
    )
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]


def test_shim_and_real_mlx_modules_pass_in_one_session(request):
    if not request.config.pluginmanager.has_plugin("unsloth-mlx-module-worlds"):
        pytest.skip("needs real mlx: without it the shim is the only mlx and nothing collides")
    # The issue's example: the first file installs the shim in a module fixture, the second imports real mlx and unsloth_zoo.gated_delta_vjp at collection.
    files = ["test_mlx_gated_delta.py", "test_mlx_gated_delta_batch_grad.py"]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
        cwd = _TESTS, capture_output = True, text = True, timeout = 1800,
    )
    assert out.returncode == 0, out.stdout[-4000:]
