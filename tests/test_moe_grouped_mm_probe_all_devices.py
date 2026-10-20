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

"""grouped_mm and ROCm crash probes must cover every visible GPU model (device_map spans cards)."""
import subprocess
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils


@pytest.fixture
def spoof(monkeypatch):
    def make(count, failing = (), hip = None, current = 0, models = None):
        models = models or [f"GPU{i}" for i in range(count)]
        def props(i): return types.SimpleNamespace(name = models[i], major = 9, minor = 0, gcnArchName = models[i])
        monkeypatch.setattr(torch.cuda, "get_device_properties", props)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_AVAILABLE", True)
        monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", None)
        monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", None)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        probed, last = [], {}
        real_ones, real_tensor = torch.ones, torch.tensor

        def ones(*a, device = None, **k):
            last["d"] = torch.device(device)
            return real_ones(*a, **k)

        def tensor(*a, device = None, **k):
            return real_tensor(*a, **k)

        def grouped_mm(x, w, offs = None):
            probed.append(last["d"].index)
            if last["d"].index in failing:
                raise RuntimeError("grouped_mm unsupported on this card")

        monkeypatch.setattr(torch, "ones", ones)
        monkeypatch.setattr(torch, "tensor", tensor)
        monkeypatch.setattr(torch, "_grouped_mm", grouped_mm, raising = False)
        return probed
    return make


@pytest.mark.parametrize("count, failing, expected, probed", [
    (1, (), True, [0]),
    (1, (0,), False, [0]),
    (2, (), True, [0, 1]),
    (3, (), True, [0, 1, 2]),
    (2, (1,), False, [0, 1]),     # experts on card 1 must not run grouped_mm unprobed
    (3, (2,), False, [0, 1, 2]),
    (3, (0,), False, [0]),        # current card fails: same answer as today, no extra probes
])
def test_support_probe_covers_every_visible_device(spoof, count, failing, expected, probed):
    seen = spoof(count, failing)
    assert moe_utils._probe_torch_grouped_mm_supported() is expected
    assert seen == probed
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is expected
    assert moe_utils._probe_torch_grouped_mm_supported() is expected
    assert seen == probed


@pytest.mark.parametrize("count", [2, 3, 8])
def test_homogeneous_host_keeps_one_probe(spoof, count):
    seen = spoof(count, models = ["MI300X"] * count)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert seen == [0]


def test_one_probe_per_distinct_model(spoof):
    seen = spoof(4, failing = (3,), models = ["H100", "H100", "A100", "A100"])
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert seen == [0, 2]


def test_unreadable_properties_probe_the_card(spoof, monkeypatch):
    seen = spoof(2, failing = (1,), models = ["MI300X"] * 2)
    def props(i):
        if i == 1:
            raise RuntimeError("no properties")
        return types.SimpleNamespace(name = "MI300X", major = 9, minor = 4)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    assert moe_utils._probe_torch_grouped_mm_supported() is False
    assert seen == [0, 1]


def test_current_device_probed_first(spoof):
    seen = spoof(3, current = 1)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert seen == [1, 0, 2]


def _record_children(monkeypatch, crash_masks = ()):
    children = []

    def run(cmd, env = None, **k):
        children.append((cmd[-1], None if env is None else (env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"])))
        crashed = env is not None and env["HIP_VISIBLE_DEVICES"] in crash_masks
        return subprocess.CompletedProcess(cmd, -11 if crashed else 0)

    monkeypatch.setattr(subprocess, "run", run)
    return children


def test_rocm_single_gpu_child_unchanged(spoof, monkeypatch):
    spoof(1, hip = "7.1.0")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert children == [("0", None)]   # inherited env, device index argv, exactly one child


def test_rocm_homogeneous_host_one_child(spoof, monkeypatch):
    spoof(4, hip = "7.1.0", models = ["MI300X"] * 4)
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert len(children) == 1


@pytest.mark.parametrize("count", [2, 3])
def test_rocm_crash_probe_one_child_per_device(spoof, monkeypatch, count):
    spoof(count, hip = "7.1.0")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert children == [("0", (str(i), str(i))) for i in range(count)]
    assert moe_utils._grouped_mm_survives_out_of_process(torch.device("cuda", 0)) is True
    assert len(children) == count


def test_rocm_crash_on_second_card_disables_grouped_mm(spoof, monkeypatch):
    seen = spoof(2, hip = "7.1.0")
    children = _record_children(monkeypatch, crash_masks = ("1",))
    assert moe_utils._probe_torch_grouped_mm_supported() is False
    assert [c[1] for c in children] == [("0", "0"), ("1", "1")]
    assert seen == []   # the kernel never ran in-process


@pytest.mark.parametrize("var", ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_rocm_crash_probe_indexes_into_inherited_mask(spoof, monkeypatch, var):
    spoof(3, hip = "7.1.0")
    monkeypatch.setenv(var, "5, 2,7")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert [c[1] for c in children] == [("5", "5"), ("2", "2"), ("7", "7")]


def test_crash_probe_never_spawns_on_multi_gpu_nvidia(spoof, monkeypatch):
    seen = spoof(3)
    def boom(*a, **k): raise AssertionError("child probe launched off AMD")
    monkeypatch.setattr(subprocess, "run", boom)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert seen == [0, 1, 2]


def test_real_child_sees_only_its_card(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE",
        "import os, signal, sys\n"
        "assert sys.argv[1] == '0' and os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['HIP_VISIBLE_DEVICES']\n"
        "if os.environ['HIP_VISIBLE_DEVICES'] == '2': os.kill(os.getpid(), signal.SIGSEGV)\n")
    assert moe_utils._grouped_mm_survives_out_of_process(torch.device("cuda", 0)) is False
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", None)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6")
    assert moe_utils._grouped_mm_survives_out_of_process(torch.device("cuda", 0)) is True
