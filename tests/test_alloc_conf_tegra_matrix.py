# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What ``import unsloth_zoo`` leaves in the allocator variables on a Tegra board
(unslothai/unsloth#2401: the CUDA VMM calls behind ``expandable_segments:True`` fail on a
Jetson AGX Orin, so a 1MiB buffer died with ``CUDA driver error: out of memory`` on a board
with ~50GB free).

Each case imports in a FRESH SUBPROCESS: the allocator block is process-global and cannot be
re-run in-process. The child loads the real detector standalone and points it at a temporary
Jetson device tree, so nothing in the product is stubbed and no Tegra hardware is needed.

Imports nothing from the module under test, so it still collects against a checkout that
predates it and fails on the behaviour rather than on an ImportError. Linux only: on Windows
the same import takes the WSL branch, and unsloth-zoo declares no torch on Apple Silicon.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason = "each case fakes a Linux CUDA host in a subprocess: Windows takes the WSL/Windows "
             "branch instead, and unsloth-zoo declares no torch on Apple Silicon, so the "
             "allocator block does not run there at all",
)

_ALLOC_KEYS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
_WIPE = _ALLOC_KEYS + (
    "WSL_DISTRO_NAME", "WSL_INTEROP", "UNSLOTH_VLLM_STANDBY",
    "UNSLOTH_DISABLE_ALLOC_FALLBACK", "UNSLOTH_FORCE_EXPANDABLE_SEGMENTS",
    # Inheriting this makes the allocator block a no-op and every case a vacuous pass.
    "UNSLOTH_ZOO_DISABLE_GPU_INIT",
)

# So the child's `import unsloth_zoo` resolves to this checkout, not a namespace-package shadow.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A Jetson AGX Orin as L4T publishes it, NUL terminators and all.
JETSON_MODEL = "NVIDIA Jetson AGX Orin Developer Kit\x00"
JETSON_COMPATIBLE = "nvidia,p3701-0000\x00nvidia,p3737-0000\x00nvidia,tegra234\x00"
JETSON_RELEASE = "# R36 (release), REVISION: 3.0, GCID: 1, BOARD: generic\n"
_BOARD_FILES = ("nv_tegra_release", "model", "compatible")


def _write_fake_jetson(directory):
    for name, text in zip(_BOARD_FILES, (JETSON_RELEASE, JETSON_MODEL, JETSON_COMPATIBLE)):
        (directory / name).write_text(text)


_CHILD = textwrap.dedent(
    """
    import importlib.machinery, importlib.metadata as _m, importlib.util
    import json, os, platform, sys, types

    ZOO = os.environ.pop("_ZOO_ROOT")
    SPOOF_DIR = os.environ.pop("_SPOOF_DIR", "") or None

    try:
        _has_unsloth = importlib.util.find_spec("unsloth") is not None
    except Exception:
        _has_unsloth = False
    if not _has_unsloth:
        _u = types.ModuleType("unsloth")
        _u.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader = None)
        _u.__path__ = []
        sys.modules["unsloth"] = _u

    _real = _m.version
    _fake_torch = os.environ.pop("_FAKE_TORCH_VERSION", "") or None
    def _version(name, *a, **k):
        if name == "torch" and _fake_torch is not None:
            return _fake_torch
        return _real(name, *a, **k)
    _m.version = _version

    spoofed = False
    _cost = {}
    detector = os.path.join(ZOO, "unsloth_zoo", "integrated_device.py")
    # os.path.exists, not an unconditional load: against a checkout that predates
    # the detector the spoof is simply a no-op, so the case measures the product's
    # behaviour instead of dying on a missing file.
    if SPOOF_DIR is not None and os.path.exists(detector):
        # Load the REAL detector standalone (it imports nothing from the package),
        # point its inputs at the fake board, and register it under its package name
        # so `from .integrated_device import ...` picks up this instance.
        platform.machine = lambda: "aarch64"
        spec = importlib.util.spec_from_file_location(
            "unsloth_zoo.integrated_device", detector,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["unsloth_zoo.integrated_device"] = module
        spec.loader.exec_module(module)
        module.BOARD_IDENTITY_FILES = tuple(
            os.path.join(SPOOF_DIR, name)
            for name in ("nv_tegra_release", "model", "compatible")
        )
        spoofed = bool(module.is_tegra_board())

        # Measure the detector's OWN cost at the moment __init__.py asks it. That is the
        # only way to separate it from `import unsloth_zoo` as a whole: importing the zoo
        # on a CUDA host already initializes CUDA further down, on main exactly as much as
        # on this branch, so looking after the fact proves nothing either way.
        _real_answer = module.expandable_segments_unsupported
        def _measured():
            def _state():
                torch_module = sys.modules.get("torch")
                if torch_module is None:
                    return False, False
                try: return True, bool(torch_module.cuda.is_initialized())
                except Exception: return True, False
            before = _state()
            answer = _real_answer()
            after = _state()
            _cost.update(
                torch_before = before[0], cuda_before = before[1],
                torch_after = after[0], cuda_after = after[1],
                answer = bool(answer),
            )
            return answer
        module.expandable_segments_unsupported = _measured

    import unsloth_zoo

    _torch = sys.modules.get("torch")
    _initialized = None
    if _torch is not None:
        try: _initialized = bool(_torch.cuda.is_initialized())
        except Exception: _initialized = None

    print("RESULT:" + json.dumps({
        "spoofed": spoofed,
        "detector_cost": _cost or None,
        "detector_present": os.path.exists(detector),
        "cuda_initialized": _initialized,
        "env": {k: os.environ.get(k) for k in
                ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")},
    }))
    """
)


def _conf(*, torch_version, jetson, tmp_path = None, preset = None, force = None):
    env = {k: v for k, v in os.environ.items() if k not in _WIPE}
    env["_ZOO_ROOT"] = _REPO_ROOT
    env["_FAKE_TORCH_VERSION"] = torch_version
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if jetson:
        assert tmp_path is not None
        _write_fake_jetson(tmp_path)
        env["_SPOOF_DIR"] = str(tmp_path)
    if force is not None:
        env["UNSLOTH_FORCE_EXPANDABLE_SEGMENTS"] = force
    for key, value in (preset or {}).items():
        env[key] = value

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env = env, cwd = _REPO_ROOT, capture_output = True, text = True, timeout = 600,
    )
    lines = [line for line in (proc.stdout + proc.stderr).splitlines() if line.startswith("RESULT:")]
    if not lines:
        raise AssertionError(
            "child produced no RESULT.\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                proc.stdout[-3000:], proc.stderr[-3000:]
            )
        )
    return json.loads(lines[-1][len("RESULT:"):])


class TestTegraGetsNothing:
    @pytest.mark.parametrize("ver", ["2.6.0", "2.9.1"])
    def test_legacy_torch_no_expandable_no_roundup(self, ver, tmp_path):
        result = _conf(torch_version = ver, jetson = True, tmp_path = tmp_path)
        for key, value in result["env"].items():
            assert "expandable_segments" not in (value or ""), (key, value)
            assert "roundup" not in (value or ""), (key, value)

    @pytest.mark.parametrize("ver", ["2.10.0", "2.14.0"])
    def test_unified_torch_no_expandable_no_roundup(self, ver, tmp_path):
        result = _conf(torch_version = ver, jetson = True, tmp_path = tmp_path)
        for key, value in result["env"].items():
            assert "expandable_segments" not in (value or ""), (key, value)
            assert "roundup" not in (value or ""), (key, value)

    def test_detection_neither_imports_torch_nor_initializes_cuda(self, tmp_path):
        result = _conf(torch_version = "2.14.0", jetson = True, tmp_path = tmp_path)
        if not result["detector_present"]:
            pytest.skip("this checkout has no unsloth_zoo/integrated_device.py")
        cost = result["detector_cost"]
        assert cost, f"the detector was never asked during import: {result}"
        assert cost["torch_before"] is False, cost
        assert cost["torch_after"] is False, cost
        assert cost["cuda_after"] is False, cost
        assert cost["answer"] is True, cost


class TestDiscreteHostUnchanged:
    def test_unified_var_still_gets_expandable(self):
        result = _conf(torch_version = "2.14.0", jetson = False)
        assert result["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True", result

    def test_legacy_var_still_gets_expandable_and_roundup(self):
        result = _conf(torch_version = "2.9.1", jetson = False)
        legacy = result["env"]["PYTORCH_CUDA_ALLOC_CONF"]
        assert "expandable_segments:True" in (legacy or ""), result
        assert "roundup_power2_divisions" in (legacy or ""), result


class TestUserPrecedenceOnTegra:
    def test_other_options_survive_on_the_unified_var(self, tmp_path):
        result = _conf(
            torch_version = "2.14.0", jetson = True, tmp_path = tmp_path,
            preset = {"PYTORCH_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128"},
        )
        assert result["env"]["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:128", result

    def test_legacy_promotion_does_not_re_add_expandable(self, tmp_path):
        # The >= 2.10 legacy-to-unified promotion must not put back what Tegra removed.
        result = _conf(
            torch_version = "2.14.0", jetson = True, tmp_path = tmp_path,
            preset = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128"},
        )
        assert result["env"]["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:128", result

    def test_user_only_expandable_leaves_nothing_behind(self, tmp_path):
        result = _conf(
            torch_version = "2.14.0", jetson = True, tmp_path = tmp_path,
            preset = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )
        assert result["env"]["PYTORCH_ALLOC_CONF"] is None, result

    def test_explicit_empty_is_preserved(self, tmp_path):
        # An explicit empty value is a user opt-out gradient_checkpointing.py advises by name.
        result = _conf(
            torch_version = "2.14.0", jetson = True, tmp_path = tmp_path,
            preset = {"PYTORCH_ALLOC_CONF": ""},
        )
        assert result["env"]["PYTORCH_ALLOC_CONF"] == "", result

    def test_legacy_backend_choice_survives(self, tmp_path):
        result = _conf(
            torch_version = "2.9.1", jetson = True, tmp_path = tmp_path,
            preset = {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"},
        )
        assert result["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync", result


class TestEscapeHatch:
    def test_force_on_restores_the_old_behaviour(self, tmp_path):
        result = _conf(
            torch_version = "2.14.0", jetson = True, tmp_path = tmp_path, force = "1",
        )
        assert result["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True", result

    def test_force_off_applies_the_exclusion_on_a_discrete_host(self):
        result = _conf(torch_version = "2.14.0", jetson = False, force = "0")
        for key, value in result["env"].items():
            assert "expandable_segments" not in (value or ""), (key, value)


class TestStandbyStillWins:
    def test_standby_on_a_tegra_board(self, tmp_path):
        # Standby already refuses expandable segments; Tegra must not resurrect them.
        result = _conf(
            torch_version = "2.14.0", jetson = True, tmp_path = tmp_path,
            preset = {"UNSLOTH_VLLM_STANDBY": "1"},
        )
        for key, value in result["env"].items():
            assert "expandable_segments:True" not in (value or ""), (key, value)
            assert "roundup" not in (value or ""), (key, value)
