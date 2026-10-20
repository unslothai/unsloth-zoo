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

"""grouped_mm and ROCm crash probes answer per GPU model, probing only the cards experts actually run on."""
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
        monkeypatch.setattr(moe_utils, "_GROUPED_MM_HOME", None, raising = False)
        for name in ("_GROUPED_MM_FOREIGN", "_TORCH_GROUPED_MM_SUPPORTED_BY_KIND", "_GROUPED_MM_SURVIVES_BY_KIND"):
            monkeypatch.setattr(moe_utils, name, {}, raising = False)
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


def cuda(i): return torch.device("cuda", i)


@pytest.mark.parametrize("count, failing, expected", [
    (1, (), True), (1, (0,), False), (2, (), True), (2, (1,), True), (3, (2,), True), (3, (0,), False),
])
def test_default_probe_touches_only_the_current_card(spoof, count, failing, expected):
    seen = spoof(count, failing)
    assert moe_utils._probe_torch_grouped_mm_supported() is expected
    assert moe_utils._check_torch_grouped_mm_supported() is expected
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is expected
    assert seen == [0]
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is expected


def test_mixed_host_each_card_answers_for_itself(spoof):
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is True   # unused 4090 never disables the H100
    assert seen == [0]
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False  # experts spilled onto the 4090
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is True
    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False
    assert seen == [0, 1]


def test_mixed_host_current_card_unsupported(spoof):
    seen = spoof(2, failing = (0,), models = ["RTX4090", "H100"])
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is True
    assert seen == [1]
    assert moe_utils._check_torch_grouped_mm_supported() is False
    assert seen == [1, 0]


def test_a_current_device_switch_does_not_borrow_the_first_cards_answer(spoof, monkeypatch):
    # Model parallel moves the current device; cuda:1 must not reuse the answer probed on cuda:0.
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is False
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    assert seen == [0, 1]


@pytest.mark.parametrize("count", [2, 3, 8])
def test_homogeneous_host_keeps_one_probe(spoof, count):
    seen = spoof(count, models = ["MI300X"] * count)
    for i in range(count):
        assert moe_utils._check_torch_grouped_mm_supported(cuda(i)) is True
    assert seen == [0]


def test_same_model_cards_share_one_probe(spoof):
    seen = spoof(4, failing = (3,), models = ["H100", "H100", "A100", "A100"])
    assert [moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(4)] == [True] * 4
    assert seen == [0, 2]


def test_unreadable_properties_probe_the_card_when_used(spoof, monkeypatch):
    seen = spoof(2, failing = (1,), models = ["MI300X"] * 2)
    def props(i):
        if i == 1:
            raise RuntimeError("no properties")
        return types.SimpleNamespace(name = "MI300X", major = 9, minor = 4)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert seen == [0]
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False
    assert seen == [0, 1]


def test_call_site_gate_routes_same_model_cards_to_the_no_arg_check(spoof, monkeypatch):
    seen = spoof(3, failing = (2,), models = ["H100", "H100", "RTX4090"])
    assert [moe_utils._grouped_mm_supported_on(cuda(i)) for i in range(3)] == [True, True, False]
    assert seen == [0, 2]
    monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: "patched")   # zero-arg patches keep working
    assert moe_utils._grouped_mm_supported_on(cuda(1)) == "patched"
    assert moe_utils._grouped_mm_supported_on(torch.device("cpu")) == "patched"


def test_cpu_or_indexless_device_uses_the_default_answer(spoof):
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._check_torch_grouped_mm_supported(torch.device("cpu")) is True
    assert moe_utils._check_torch_grouped_mm_supported(torch.device("cuda")) is True
    assert seen == [0]


def test_grouped_mm_eager_asks_for_the_input_device(spoof, monkeypatch):
    spoof(2)
    asked = []
    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", lambda device: asked.append(device) or False)
    monkeypatch.setattr(moe_utils, "_manual_grouped_mm", lambda *a: "manual")
    x = torch.zeros(2, 4)
    assert moe_utils._grouped_mm_eager(x, torch.zeros(1, 4, 4), torch.tensor([2])) == "manual"
    assert asked == [x.device]


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


def test_rocm_crash_probe_one_child_per_used_model(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert children == [("0", ("0", "0"))]   # cards 1 and 2 unused: no child for them
    assert moe_utils._check_torch_grouped_mm_supported(cuda(2)) is True
    assert children == [("0", ("0", "0")), ("0", ("2", "2"))]
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is True
    assert len(children) == 2


def test_rocm_crash_on_unused_card_keeps_grouped_mm(spoof, monkeypatch):
    seen = spoof(2, hip = "7.1.0", models = ["gfx1100", "gfx1036"])
    children = _record_children(monkeypatch, crash_masks = ("1",))
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False
    assert [c[1] for c in children] == [("0", "0"), ("1", "1")]
    assert seen == [0]   # the crashing card's kernel never ran in-process


@pytest.mark.parametrize("var", ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_rocm_crash_probe_indexes_into_inherited_mask(spoof, monkeypatch, var):
    spoof(3, hip = "7.1.0")
    monkeypatch.setenv(var, "5, 2,7")
    children = _record_children(monkeypatch)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(3))
    assert [c[1] for c in children] == [("5", "5"), ("2", "2"), ("7", "7")]


def test_crash_probe_never_spawns_on_multi_gpu_nvidia(spoof, monkeypatch):
    seen = spoof(3)
    def boom(*a, **k): raise AssertionError("child probe launched off AMD")
    monkeypatch.setattr(subprocess, "run", boom)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(3))
    assert seen == [0, 1, 2]


def test_real_child_sees_only_its_card(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE",
        "import os, signal, sys\n"
        "assert sys.argv[1] == '0' and os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['HIP_VISIBLE_DEVICES']\n"
        "if os.environ['HIP_VISIBLE_DEVICES'] == '2': os.kill(os.getpid(), signal.SIGSEGV)\n")
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(0)) is True
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is False
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES_BY_KIND", {})
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6")
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is True


def test_routed_grouped_gate_asks_for_the_hidden_states_device():
    import inspect
    from unsloth_zoo.temporary_patches import moe_routed
    assert "_grouped_mm_supported_on(hidden_states.device)" in inspect.getsource(moe_routed)
