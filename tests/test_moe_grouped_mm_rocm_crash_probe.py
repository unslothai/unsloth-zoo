# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""unsloth#12391: on ROCm a grouped_mm probe that segfaults must not take the importing process down."""
import os
import subprocess
import sys

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
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", "raise SystemExit(1)")
    assert moe_utils._grouped_mm_survives_out_of_process(fake_rocm) is survives


def test_crash_probe_timeout_is_unsupported(monkeypatch, fake_rocm):
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: real_run(*a, **{**k, "timeout": 1}))
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", "import time; time.sleep(60)")
    assert moe_utils._grouped_mm_survives_out_of_process(fake_rocm) is False


@pytest.mark.parametrize("device", [torch.device("cuda", 0), torch.device("xpu", 0)])
def test_crash_probe_never_runs_on_nvidia_or_intel(monkeypatch, device):
    def boom(*a, **k): raise AssertionError("child probe launched off AMD")
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", None)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(torch._C, "_dispatch_dump", boom)
    assert moe_utils._grouped_mm_survives_out_of_process(device) is True


def test_python_kernel_override_skips_child(monkeypatch, fake_rocm):
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_grouped_mm", lambda *a, **k: None, "CUDA")
    try:
        monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE", "raise SystemExit(1)")
        assert moe_utils._grouped_mm_survives_out_of_process(fake_rocm) is True
    finally:
        lib._destroy()
    assert "CUDA (inactive):" not in torch._C._dispatch_dump("aten::_grouped_mm")

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU so the probe reaches the kernel call")
def test_import_survives_segfaulting_grouped_mm(tmp_path):
    # Every interpreter (this child and the probe's own child) sees ROCm and a grouped_mm that segfaults.
    (tmp_path / "sitecustomize.py").write_text(
        "import os, signal, torch\n"
        "torch.version.hip = '7.1.0'\n"
        "torch._grouped_mm = lambda *a, **k: os.kill(os.getpid(), signal.SIGSEGV)\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(tmp_path), root]), UNSLOTH_IS_PRESENT = "1")
    code = (
        "from unsloth_zoo.temporary_patches import moe_utils as m; "
        "print(m._check_torch_grouped_mm_supported(), m._transposed_view_grouped_mm_is_safe())"
    )
    r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[-2:] == ["False", "False"]
