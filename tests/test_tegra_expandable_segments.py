# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""expandable_segments must not be set on an NVIDIA Tegra board (Jetson).

``expandable_segments:True`` is not a ``cudaMalloc`` tuning knob. The caching
allocator backs those segments with the CUDA virtual memory management driver
calls (``cuMemAddressReserve``, ``cuMemCreate``, ``cuMemMap``,
``cuMemSetAccess``), each wrapped in ``C10_CUDA_DRIVER_CHECK``, and on a Jetson
AGX Orin that path fails: a 1MiB gradient-checkpointing buffer died with
``RuntimeError: CUDA driver error: out of memory`` on a board with ~50GB free
(unslothai/unsloth#2401). Torch's own "expandable_segments not supported on this
platform" guard is a compile-time ``#if PYTORCH_C10_DRIVER_API_SUPPORTED``, not a
runtime device query, so an aarch64 CUDA wheel enables the mode and only finds
out at allocation time.

Three things are asserted here, and the third is why the detection is shaped the
way it is:

1. A host that reads as a Tegra board gets no ``expandable_segments`` in any of
   the three allocator variables, and gets no ``roundup_power2_divisions``
   either: unified memory has no separate VRAM pool to defragment and the
   rounding would waste the RAM the model needs.
2. A discrete CUDA host is untouched, an explicit user value keeps everything
   except ``expandable_segments``, and ``UNSLOTH_FORCE_EXPANDABLE_SEGMENTS``
   forces either answer.
3. The detection reads files, never the driver. ``torch.cuda.is_available()``
   and ``torch.cuda.device_count()`` leave ``torch.cuda.is_initialized()``
   False, but ``torch.cuda.get_device_properties()`` flips it True, so asking
   for ``cudaDeviceProp::integrated`` at import would create a CUDA context
   before the caller has picked a device.

This file covers the detector itself. What ``import unsloth_zoo`` then leaves in
the three allocator variables is covered by
``tests/test_alloc_conf_tegra_matrix.py``, which needs a fresh subprocess per
case and deliberately does not import this module.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

from unsloth_zoo import integrated_device

_ALLOC_KEYS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
_WIPE = _ALLOC_KEYS + (
    "WSL_DISTRO_NAME", "WSL_INTEROP", "UNSLOTH_VLLM_STANDBY",
    "UNSLOTH_DISABLE_ALLOC_FALLBACK", "UNSLOTH_FORCE_EXPANDABLE_SEGMENTS",
    # See tests/security/test_no_module_scope_env_leaks.py: inheriting this makes the
    # whole allocator block a no-op and every case below report a vacuous pass.
    "UNSLOTH_ZOO_DISABLE_GPU_INIT",
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A Jetson AGX Orin as L4T publishes it: /proc/device-tree/model is NUL terminated
# and /proc/device-tree/compatible is a NUL separated list.
JETSON_MODEL = "NVIDIA Jetson AGX Orin Developer Kit\x00"
JETSON_COMPATIBLE = "nvidia,p3701-0000\x00nvidia,p3737-0000\x00nvidia,tegra234\x00"
JETSON_RELEASE = "# R36 (release), REVISION: 3.0, GCID: 1, BOARD: generic\n"


def _write_fake_jetson(directory):
    """A fake board-identity directory, and the file list that reads it."""
    names = ("nv_tegra_release", "model", "compatible")
    for name, text in zip(names, (JETSON_RELEASE, JETSON_MODEL, JETSON_COMPATIBLE)):
        (directory / name).write_text(text)
    return tuple(str(directory / name) for name in names)


# ---------------------------------------------------------------------------
# Unit: the detector itself
# ---------------------------------------------------------------------------

class TestBoardIdentity:
    def test_reads_nul_separated_device_tree(self, tmp_path):
        paths = _write_fake_jetson(tmp_path)
        identity = integrated_device.read_board_identity(paths)
        assert "nvidia,tegra234" in identity, identity
        assert "jetson agx orin" in identity, identity
        assert "\x00" not in identity, identity
        # The file NAME is part of the text, which is what makes the existence of
        # /etc/nv_tegra_release a signal on its own.
        assert "nv_tegra_release=" in identity, identity

    def test_missing_files_are_not_an_error(self, tmp_path):
        assert integrated_device.read_board_identity(
            (str(tmp_path / "nope"), str(tmp_path / "also-nope"))
        ) == ""

    def test_directory_is_skipped_not_raised(self, tmp_path):
        (tmp_path / "model").mkdir()
        assert integrated_device.read_board_identity((str(tmp_path / "model"),)) == ""

    def test_this_host_is_not_a_tegra_board(self):
        # The regression guard that matters most: the CI hosts and every developer
        # box must read as "not a Tegra", so nothing about their allocator changes.
        assert integrated_device.is_tegra_board() is False
        assert integrated_device.expandable_segments_unsupported() is False


class TestIsTegraBoard:
    JETSON = "model=nvidia jetson agx orin developer kit compatible=nvidia,tegra234"
    SPARK = "model=nvidia dgx spark compatible=nvidia,tegra264"
    ARM_SERVER = "product_name=ampere altra board_name=mt.jade"
    X86 = "product_name=p6-b200.48xlarge"

    @pytest.fixture
    def arm(self, monkeypatch):
        monkeypatch.setattr(integrated_device.platform, "machine", lambda: "aarch64")

    def test_jetson_on_arm(self, arm):
        assert integrated_device.is_tegra_board(self.JETSON) is True

    def test_same_identity_on_x86_is_not_a_tegra(self, monkeypatch):
        # Tegra is ARM only, so the machine gate is what makes a stray file on a
        # normal host harmless.
        monkeypatch.setattr(integrated_device.platform, "machine", lambda: "x86_64")
        assert integrated_device.is_tegra_board(self.JETSON) is False

    def test_dgx_spark_is_excluded_by_name(self, arm):
        # GB10 is Tegra lineage and its device tree can say so, but expandable
        # segments is the reported mitigation there (vllm-project/vllm#55569), so
        # taking it away would be the regression.
        assert integrated_device.is_tegra_board(self.SPARK) is False

    def test_gb10_alone_is_excluded(self, arm):
        assert integrated_device.is_tegra_board("product_name=nvidia gb10") is False

    def test_arm_server_with_a_discrete_card(self, arm):
        assert integrated_device.is_tegra_board(self.ARM_SERVER) is False

    def test_unreadable_identity(self, arm):
        assert integrated_device.is_tegra_board("") is False


class TestForceEnv:
    def test_force_on_keeps_expandable_segments(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UNSLOTH_FORCE_EXPANDABLE_SEGMENTS", "1")
        monkeypatch.setattr(integrated_device.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(
            integrated_device, "BOARD_IDENTITY_FILES", _write_fake_jetson(tmp_path)
        )
        assert integrated_device.is_tegra_board() is True
        assert integrated_device.expandable_segments_unsupported() is False

    def test_force_off_applies_the_exclusion_anywhere(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_FORCE_EXPANDABLE_SEGMENTS", "0")
        assert integrated_device.is_tegra_board() is False
        assert integrated_device.expandable_segments_unsupported() is True

    @pytest.mark.parametrize("word,expected", [
        ("true", False), ("YES", False), ("on", False),
        ("false", True), ("no", True), ("OFF", True),
    ])
    def test_words_not_only_digits(self, monkeypatch, word, expected):
        monkeypatch.setenv("UNSLOTH_FORCE_EXPANDABLE_SEGMENTS", word)
        assert integrated_device.expandable_segments_unsupported() is expected

    def test_unknown_word_falls_through_to_detection(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_FORCE_EXPANDABLE_SEGMENTS", "maybe")
        assert integrated_device.expandable_segments_unsupported() is False

    def test_fake_jetson_without_the_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(integrated_device.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(
            integrated_device, "BOARD_IDENTITY_FILES", _write_fake_jetson(tmp_path)
        )
        assert integrated_device.expandable_segments_unsupported() is True


class TestNoCudaInitialization:
    """The detection must not be what creates a CUDA context."""

    def test_no_device_properties_call_when_no_context_exists(self, monkeypatch):
        torch = pytest.importorskip("torch")
        calls = []

        def _boom(*args, **kwargs):
            calls.append(args)
            raise AssertionError("get_device_properties initializes CUDA")

        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)
        assert integrated_device.expandable_segments_unsupported() is False
        assert calls == []
        assert torch.cuda.is_initialized() is False

    def test_free_second_opinion_fires_for_a_tegra_part(self, monkeypatch):
        # A container can hide the board files, so the driver's integrated flag is
        # used as a second opinion -- but only when a context already exists, which
        # is when reading it is free.
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(integrated_device, "_any_device_integrated", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Orin (nvgpu)")
        assert integrated_device.expandable_segments_unsupported() is True

    def test_free_second_opinion_spares_gb10(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(integrated_device, "_any_device_integrated", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA GB10")
        assert integrated_device.expandable_segments_unsupported() is False

    def test_free_second_opinion_spares_a_discrete_card(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(integrated_device, "_any_device_integrated", lambda: False)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA B200")
        assert integrated_device.expandable_segments_unsupported() is False


class TestHelperRelocation:
    def test_gradient_checkpointing_still_exports_the_old_name(self):
        # Relocating _any_device_integrated must not break its existing reader.
        gc_module = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
        assert (
            gc_module._any_device_integrated
            is integrated_device._any_device_integrated
        )

    def test_detector_module_imports_no_torch(self):
        # The point of the module: it is importable, and answerable, before torch
        # is, because the allocator variables are read during torch initialization.
        # Module scope only -- the driver probe does import torch, inside a function.
        path = os.path.join(_REPO_ROOT, "unsloth_zoo", "integrated_device.py")
        tree = ast.parse(open(path).read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
                assert "torch" not in names, ast.unparse(node)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch", ast.unparse(node)

    def test_answering_the_question_does_not_import_torch(self):
        # A fresh interpreter: load the module standalone, ask it, and show torch
        # never entered sys.modules.
        path = os.path.join(_REPO_ROOT, "unsloth_zoo", "integrated_device.py")
        program = textwrap.dedent(
            f"""
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("_probe", {path!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            answer = module.expandable_segments_unsupported()
            print("TORCH_IMPORTED:", "torch" in sys.modules, "ANSWER:", answer)
            """
        )
        env = {k: v for k, v in os.environ.items() if k not in _WIPE}
        proc = subprocess.run(
            [sys.executable, "-c", program], env = env, capture_output = True,
            text = True, timeout = 300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "TORCH_IMPORTED: False" in proc.stdout, proc.stdout
