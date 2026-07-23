# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the import-time PyTorch allocator config.

``unsloth_zoo/__init__.py`` configures the CUDA/HIP caching allocator env vars
(``PYTORCH_ALLOC_CONF`` / ``PYTORCH_CUDA_ALLOC_CONF`` / ``PYTORCH_HIP_ALLOC_CONF``)
at import time. Two behaviours are covered here:

* The unified ``PYTORCH_ALLOC_CONF`` is only *read* by the allocator from torch
  2.10; torch <= 2.9.x reads only the legacy per-backend vars. The block must
  therefore write the legacy var on torch <= 2.9.x and the unified var on 2.10+.
* Windows / WSL cannot use ``expandable_segments``, so a
  ``roundup_power2_divisions`` fallback is applied there instead of leaving the
  allocator with no fragmentation mitigation (unslothai/unsloth#7203).

Each case runs ``import unsloth_zoo`` in a **fresh subprocess** so the
import-time, process-global allocator config is exercised in isolation (the
block cannot be re-run cleanly in-process, and doing so would leak module state
into sibling tests). The torch version and device backend are faked in the
child so the whole matrix runs deterministically under whatever torch is
actually installed; only the selected env var / value (branch logic) is
asserted, never torch's own reading.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

ROUNDUP = "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
_ALLOC_KEYS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
_WIPE = _ALLOC_KEYS + (
    "WSL_DISTRO_NAME", "WSL_INTEROP", "UNSLOTH_VLLM_STANDBY", "UNSLOTH_DISABLE_ALLOC_FALLBACK",
)

# Repo root (.../unsloth_zoo), so the child's `import unsloth_zoo` resolves to this
# checkout rather than a namespace-package shadow on a crowded sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD = textwrap.dedent(
    """
    import os, json, importlib.metadata as _m
    _real = _m.version
    _fake_torch = os.environ.pop("_FAKE_TORCH_VERSION", "") or None
    def _version(name, *a, **k):
        if name == "torch" and _fake_torch is not None:
            return _fake_torch
        return _real(name, *a, **k)
    _m.version = _version
    # Fake the device backend by injecting a stub unsloth_zoo.device_type BEFORE
    # unsloth_zoo import (faking torch.version.hip instead would make torch route
    # device_count() through amdsmi and crash on a CUDA box). DEVICE_TYPE_TORCH is
    # mapped to "cuda" so the rest of the import still runs on this box; only
    # DEVICE_TYPE / is_hip (which gate the fallback) are what the block reads.
    _fake_dev = os.environ.pop("_FAKE_DEVICE", "") or None
    if _fake_dev:
        import sys as _sys, types as _types
        _dt = _types.ModuleType("unsloth_zoo.device_type")
        _dt.is_hip = lambda: _fake_dev == "hip"
        _dt.get_device_type = lambda: _fake_dev
        _dt.DEVICE_TYPE = _fake_dev
        _dt.DEVICE_TYPE_TORCH = "cuda"
        _dt.DEVICE_COUNT = 1
        _dt.ALLOW_PREQUANTIZED_MODELS = True
        _sys.modules["unsloth_zoo.device_type"] = _dt
    import unsloth_zoo  # runs the import-time allocator block
    print("RESULT:" + json.dumps(
        {k: os.environ.get(k) for k in
         ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}
    ))
    """
)


def _conf(*, torch_version, wsl=False, wsl_interop=False, windows=False,
          standby=False, opt_out=False, device=None, preset=None):
    """Import unsloth_zoo in a fresh child with the given fake env and return the
    resulting allocator env vars as a dict (values may be ``None``)."""
    env = {k: v for k, v in os.environ.items() if k not in _WIPE and k != "_FAKE_TORCH_VERSION"}
    env["_FAKE_TORCH_VERSION"] = torch_version
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if wsl:
        env["WSL_DISTRO_NAME"] = "Ubuntu"
    if wsl_interop:
        env["WSL_INTEROP"] = "/run/WSL/1_interop"
    if standby:
        env["UNSLOTH_VLLM_STANDBY"] = "1"
    if opt_out:
        env["UNSLOTH_DISABLE_ALLOC_FALLBACK"] = "1"
    if device:
        env["_FAKE_DEVICE"] = device
    for key, val in (preset or {}).items():
        env[key] = val
    # `windows` (os.name=='nt') cannot be faked on Linux (pathlib raises), so it
    # is covered by the drift guard test instead, not here.
    assert not windows

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    lines = [l for l in (proc.stdout + proc.stderr).splitlines() if l.startswith("RESULT:")]
    if not lines:
        raise AssertionError(
            "child produced no RESULT.\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                proc.stdout[-2000:], proc.stderr[-2000:]
            )
        )
    return json.loads(lines[-1][len("RESULT:"):])


# --- torch version boundary (Linux CUDA, no user config) -------------------

class TestBoundary:
    @pytest.mark.parametrize("ver", ["2.6.0", "2.8.1", "2.9.1"])
    def test_le_2_9_uses_legacy_cuda_var(self, ver):
        # torch <= 2.9.x only READS PYTORCH_CUDA_ALLOC_CONF, so the mitigation
        # must land there (expandable + roundup); the unified var stays unset.
        conf = _conf(torch_version=ver)
        assert conf["PYTORCH_CUDA_ALLOC_CONF"] is not None, conf
        assert "expandable_segments:True" in conf["PYTORCH_CUDA_ALLOC_CONF"], conf
        assert "roundup_power2_divisions" in conf["PYTORCH_CUDA_ALLOC_CONF"], conf
        assert conf["PYTORCH_ALLOC_CONF"] is None, conf
        assert conf["PYTORCH_HIP_ALLOC_CONF"] is None, conf

    @pytest.mark.parametrize("ver", ["2.10.0", "2.13.0"])
    def test_ge_2_10_uses_unified_var(self, ver):
        # torch >= 2.10 reads the unified var; behaviour matches today (expandable
        # only, no roundup) -> guards against a Linux regression.
        conf = _conf(torch_version=ver)
        assert conf["PYTORCH_ALLOC_CONF"] == "expandable_segments:True", conf
        assert conf["PYTORCH_CUDA_ALLOC_CONF"] is None, conf
        assert conf["PYTORCH_HIP_ALLOC_CONF"] is None, conf


# --- Windows / WSL fragmentation fallback (issue #7203) --------------------

class TestWindowsWslFallback:
    def test_wsl_torch_2_9_roundup_on_legacy(self):
        conf = _conf(torch_version="2.9.1", wsl=True)
        assert conf["PYTORCH_CUDA_ALLOC_CONF"] == ROUNDUP, conf
        assert conf["PYTORCH_ALLOC_CONF"] is None, conf

    def test_wsl_torch_2_10_roundup_on_unified(self):
        conf = _conf(torch_version="2.10.0", wsl=True)
        assert conf["PYTORCH_ALLOC_CONF"] == ROUNDUP, conf
        assert conf["PYTORCH_CUDA_ALLOC_CONF"] is None, conf

    def test_wsl_interop_also_triggers(self):
        conf = _conf(torch_version="2.10.0", wsl_interop=True)
        assert conf["PYTORCH_ALLOC_CONF"] == ROUNDUP, conf

    def test_linux_2_10_no_fallback(self):
        # Regression guard: native Linux keeps expandable, never gets roundup.
        conf = _conf(torch_version="2.10.0")
        assert conf["PYTORCH_ALLOC_CONF"] == "expandable_segments:True", conf
        assert "roundup" not in (conf["PYTORCH_ALLOC_CONF"] or ""), conf

    def test_windows_shares_wsl_branch_source_guard(self):
        # Native Windows (os.name == 'nt') takes the exact same
        # `elif IS_WSL_OR_WINDOWS` branch + fallback as WSL, so the WSL cases
        # above cover the runtime behaviour. A reload under a spoofed os.name
        # cannot run on Linux (pathlib flips to WindowsPath), so guard the shared
        # detection + fallback wiring against drift (same source-inspection style
        # as tests/test_upstream_import_fixes_drift.py).
        src = os.path.join(_REPO_ROOT, "unsloth_zoo", "__init__.py")
        text = open(src).read()
        assert 'os.name == "nt"' in text
        assert "IS_WSL_OR_WINDOWS" in text
        assert "roundup_power2_divisions" in text


# --- user precedence + promotion gap ---------------------------------------

class TestUserPrecedence:
    def test_user_unified_backend_preserved_2_10(self):
        # A custom backend on torch 2.10 must survive untouched (roundup is a
        # no-op under cudaMallocAsync anyway, so none is added).
        conf = _conf(torch_version="2.10.0", wsl=True,
                     preset={"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"})
        assert conf["PYTORCH_ALLOC_CONF"] == "backend:cudaMallocAsync", conf

    def test_user_legacy_backend_preserved_2_9(self):
        conf = _conf(torch_version="2.9.1", wsl=True,
                     preset={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"})
        assert conf["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync", conf

    def test_promotion_gap_closed_2_10(self):
        # A legacy var pairing expandable with another option must end up with
        # expandable stripped (futile on WSL) but the other option preserved, and
        # must NOT re-surface expandable via the >=2.10 promotion step.
        conf = _conf(torch_version="2.10.0", wsl=True,
                     preset={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128"})
        assert conf["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:128", conf
        assert "expandable_segments" not in (conf["PYTORCH_ALLOC_CONF"] or ""), conf

    def test_user_only_expandable_on_wsl_replaced_by_roundup_2_10(self):
        # If the only thing the user set is the futile expandable, strip it and
        # fall back to roundup so the box is not left unmitigated.
        conf = _conf(torch_version="2.10.0", wsl=True,
                     preset={"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
        assert conf["PYTORCH_ALLOC_CONF"] == ROUNDUP, conf


# --- opt-out and vLLM standby ----------------------------------------------

class TestControls:
    def test_opt_out_disables_fallback_2_10(self):
        conf = _conf(torch_version="2.10.0", wsl=True, opt_out=True)
        assert conf["PYTORCH_ALLOC_CONF"] is None, conf
        assert conf["PYTORCH_CUDA_ALLOC_CONF"] is None, conf

    def test_standby_no_expandable_no_roundup(self):
        conf = _conf(torch_version="2.10.0", wsl=True, standby=True)
        for key, val in conf.items():
            assert "expandable_segments:True" not in (val or ""), (key, val)
            assert "roundup" not in (val or ""), (key, val)


# --- backend isolation: AMD (hip) / Intel (xpu) never get the CUDA fallback -

class TestBackendIsolation:
    def test_hip_wsl_no_roundup(self):
        # AMD/ROCm: IS_HIP_RUNTIME short-circuits the CUDA-only fallback.
        conf = _conf(torch_version="2.10.0", wsl=True, device="hip")
        for key, val in conf.items():
            assert "roundup" not in (val or ""), (key, val)
            assert "expandable_segments:True" not in (val or ""), (key, val)

    def test_xpu_wsl_no_roundup(self):
        # Intel XPU: the DEVICE_TYPE == "cuda" clause gates the fallback out.
        conf = _conf(torch_version="2.10.0", wsl=True, device="xpu")
        for key, val in conf.items():
            assert "roundup" not in (val or ""), (key, val)

    def test_fake_cuda_control_still_fires(self):
        # Sanity: same fake-device harness with device="cuda" DOES fire, proving
        # the hip/xpu isolation above is real and not a harness artefact.
        conf = _conf(torch_version="2.10.0", wsl=True, device="cuda")
        assert conf["PYTORCH_ALLOC_CONF"] == ROUNDUP, conf
