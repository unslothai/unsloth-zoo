"""unsloth#12391: on ROCm a grouped_mm probe that segfaults must not take the importing process down."""
import subprocess

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils


@pytest.fixture
def fake_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.1.0", raising = False)
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", None)
    return torch.device("cuda", 0)


@pytest.mark.parametrize("child, survives", [
    ("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)", False),
    ("pass", True),
])
def test_crash_probe_runs_out_of_process(monkeypatch, fake_rocm, child, survives):
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", child)
    assert moe_utils._grouped_mm_survives_out_of_process(fake_rocm) is survives
    # Cached: the child is not launched again.
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", "raise SystemExit(1)")
    assert moe_utils._grouped_mm_survives_out_of_process(fake_rocm) is survives


def test_crash_probe_timeout_is_unsupported(monkeypatch, fake_rocm):
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: real_run(*a, **{**k, "timeout": 1}))
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", "import time; time.sleep(60)")
    assert moe_utils._grouped_mm_survives_out_of_process(fake_rocm) is False


def test_crash_probe_skipped_off_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", None)
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", "raise SystemExit(1)")
    assert moe_utils._grouped_mm_survives_out_of_process(torch.device("cuda", 0)) is True


def test_crashing_probe_disables_both_grouped_mm_probes(monkeypatch, fake_rocm):
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", False)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", None)
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_AVAILABLE", True)
    monkeypatch.setattr(moe_utils, "_grouped_mm_probe_device", lambda: fake_rocm)
    called = []
    monkeypatch.setattr(torch, "_grouped_mm", lambda *a, **k: called.append(1), raising = False)
    assert moe_utils._probe_torch_grouped_mm_supported() is False
    assert moe_utils._probe_transposed_view_grouped_mm_is_safe() is False
    assert called == []
