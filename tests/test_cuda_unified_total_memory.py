# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""``props.total_memory`` on a unified-memory NVIDIA host can be the dedicated carve-out while
the runtime's own total spans the shared pool, so ``cuda_total_memory`` applies the rule the ROCm
path already follows: compare, and adopt only a LARGER driver total. The tests below pin every
branch of it, above all the ones that must NOT change anything, since a discrete card is the
common case and its two figures always agree.
"""

from __future__ import annotations

import pytest

from unsloth_zoo import integrated_device


GIB = 1024 ** 3


class _Props:
    """Only the fields the probe reads. ``is_integrated`` is omitted when None, so a wheel that
    does not publish the field is covered too."""

    def __init__(self, total_memory, is_integrated = None):
        self.name = "stub"
        self.total_memory = total_memory
        if is_integrated is not None:
            self.is_integrated = is_integrated


def _install(monkeypatch, props, driver_total = None, hip = None):
    """``driver_total`` of None makes ``mem_get_info`` an assertion failure, which is how the
    tests prove a path never pays for the primary context that call pins."""
    torch = pytest.importorskip("torch")
    calls = []

    def _mem_get_info(index = 0):
        calls.append(index)
        if driver_total is None:
            raise AssertionError("mem_get_info must not be called on this path")
        if isinstance(driver_total, BaseException):
            raise driver_total
        return (driver_total // 2, driver_total)

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info, raising = False)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    return calls


class TestIntegratedDetection:
    def test_is_integrated_true(self, monkeypatch):
        monkeypatch.setattr(pytest.importorskip("torch").version, "hip", None, raising = False)
        assert integrated_device.cuda_props_are_integrated(_Props(8 * GIB, 1)) is True

    def test_is_integrated_false(self, monkeypatch):
        monkeypatch.setattr(pytest.importorskip("torch").version, "hip", None, raising = False)
        assert integrated_device.cuda_props_are_integrated(_Props(80 * GIB, 0)) is False

    def test_missing_field_reads_as_not_integrated(self, monkeypatch):
        # Absent is not True: an unknown part must keep the properties figure.
        monkeypatch.setattr(pytest.importorskip("torch").version, "hip", None, raising = False)
        assert integrated_device.cuda_props_are_integrated(_Props(80 * GIB)) is False

    def test_legacy_integrated_field_is_read(self, monkeypatch):
        monkeypatch.setattr(pytest.importorskip("torch").version, "hip", None, raising = False)
        props = _Props(8 * GIB)
        props.integrated = 1
        assert integrated_device.cuda_props_are_integrated(props) is True

    def test_hip_is_excluded_by_name(self, monkeypatch):
        # clr left the field unassigned before ROCm 6.2, and the ROCm total has its own
        # handling; this probe must never reach an AMD part, flag set or not.
        monkeypatch.setattr(pytest.importorskip("torch").version, "hip", "6.4.0", raising = False)
        assert integrated_device.cuda_props_are_integrated(_Props(8 * GIB, 1)) is False


class TestTotalMemoryRule:
    def test_driver_total_larger_is_adopted(self, monkeypatch):
        # The real N1X shape: a carve-out sized properties total, a shared-pool driver total.
        _install(monkeypatch, _Props(8128 * 1024 ** 2, 1), driver_total = 46477 * 1024 ** 2)
        assert integrated_device.cuda_total_memory(0) == 46477 * 1024 ** 2

    def test_driver_total_equal_changes_nothing(self, monkeypatch):
        _install(monkeypatch, _Props(45 * GIB, 1), driver_total = 45 * GIB)
        assert integrated_device.cuda_total_memory(0) == 45 * GIB

    def test_driver_total_smaller_never_shrinks(self, monkeypatch):
        # Only ever adopt a LARGER total. A driver total below properties is discarded.
        _install(monkeypatch, _Props(45 * GIB, 1), driver_total = 8 * GIB)
        assert integrated_device.cuda_total_memory(0) == 45 * GIB

    def test_driver_total_zero_is_discarded(self, monkeypatch):
        _install(monkeypatch, _Props(45 * GIB, 1), driver_total = 0)
        assert integrated_device.cuda_total_memory(0) == 45 * GIB

    def test_mem_get_info_raising_degrades_to_properties(self, monkeypatch):
        _install(
            monkeypatch, _Props(45 * GIB, 1),
            driver_total = RuntimeError("Torch not compiled with CUDA enabled"),
        )
        assert integrated_device.cuda_total_memory(0) == 45 * GIB

    def test_mem_get_info_returning_junk_degrades_to_properties(self, monkeypatch):
        # mem_get_info semantics have shifted across torch versions; a non-number answer
        # must be dropped, not propagated as a TypeError out of a capacity probe.
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.version, "hip", None, raising = False)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Props(45 * GIB, 1))
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (None, None), raising = False)
        assert integrated_device.cuda_total_memory(0) == 45 * GIB

    def test_the_device_index_is_forwarded(self, monkeypatch):
        calls = _install(monkeypatch, _Props(8 * GIB, 1), driver_total = 45 * GIB)
        integrated_device.cuda_total_memory(3)
        assert calls == [3]

    def test_caller_supplied_props_are_used(self, monkeypatch):
        # gpt_oss and flex_attention already hold props; re-reading them is wasted work.
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.version, "hip", None, raising = False)

        def _boom(*args, **kwargs):
            raise AssertionError("get_device_properties must not be called again")

        monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (0, 45 * GIB), raising = False)
        assert integrated_device.cuda_total_memory(0, props = _Props(8 * GIB, 1)) == 45 * GIB


class TestDiscreteIsUntouched:
    """Equal figures, and no driver call at all."""

    def test_equal_totals_are_returned_unchanged(self, monkeypatch):
        # A discrete card: properties and driver agree, and the value must be untouched.
        _install(monkeypatch, _Props(80 * GIB, 0), driver_total = 80 * GIB)
        assert integrated_device.cuda_total_memory(0) == 80 * GIB

    def test_discrete_never_calls_mem_get_info(self, monkeypatch):
        # driver_total None makes the call an AssertionError: a discrete card must not
        # pay for the primary context mem_get_info pins for the life of the process.
        calls = _install(monkeypatch, _Props(80 * GIB, 0), driver_total = None)
        assert integrated_device.cuda_total_memory(0) == 80 * GIB
        assert calls == []

    def test_wheel_without_the_field_never_calls_mem_get_info(self, monkeypatch):
        calls = _install(monkeypatch, _Props(80 * GIB), driver_total = None)
        assert integrated_device.cuda_total_memory(0) == 80 * GIB
        assert calls == []

    def test_hip_never_calls_mem_get_info(self, monkeypatch):
        # The ROCm path keeps whatever it did before, untouched by this probe.
        calls = _install(monkeypatch, _Props(96 * GIB, 1), driver_total = None, hip = "6.4.0")
        assert integrated_device.cuda_total_memory(0) == 96 * GIB
        assert calls == []


class TestUnreadableDevice:
    def test_properties_raising_answers_zero(self, monkeypatch):
        torch = pytest.importorskip("torch")

        def _boom(*args, **kwargs):
            raise RuntimeError("no CUDA-capable device is detected")

        monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)
        assert integrated_device.cuda_total_memory(0) == 0

    def test_properties_without_total_memory_answers_zero(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.version, "hip", None, raising = False)

        class _Bare:
            name = "stub"

        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: _Bare())
        assert integrated_device.cuda_total_memory(0) == 0


class TestFlexAttentionCallSite:
    """The kernel-option probe must keep raising for a device it cannot describe.

    `cuda_total_memory` answers 0 there, which is right for a caller comparing budgets and wrong
    for this one: 0 GiB reads as "16GB or less", so flex attention would stay ENABLED with 32x32
    kernel options where the probe previously turned it off. The real assignment is lifted out of
    the module source and evaluated here, so this tests the shipped expression, not a copy.
    """

    @staticmethod
    def _vram_expression():
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "unsloth_zoo" / "flex_attention" / "utils.py"
        ).read_text(encoding = "utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "vram_of_gpu":
                    return ast.get_source_segment(source, node.value)
        raise AssertionError("vram_of_gpu is no longer assigned in flex_attention/utils.py")

    @staticmethod
    def _namespace(device_count, properties):
        import types

        cuda = types.SimpleNamespace(
            device_count = lambda: device_count,
            get_device_properties = properties,
        )
        return {
            "torch": types.SimpleNamespace(cuda = cuda),
            "cuda_total_memory": integrated_device.cuda_total_memory,
        }

    def test_an_unreadable_device_still_raises(self):
        def _boom(index):
            raise RuntimeError("no CUDA-capable device is detected")

        with pytest.raises(RuntimeError):
            eval(self._vram_expression(), self._namespace(1, _boom))

    def test_no_devices_still_raises(self):
        # What sends a GPU-less host into the except that sets HAS_FLEX_ATTENTION = False.
        with pytest.raises(ValueError):
            eval(self._vram_expression(), self._namespace(0, lambda index: None))

    def test_a_readable_discrete_device_reports_its_properties_total(self, monkeypatch):
        monkeypatch.setattr(integrated_device, "_is_hip_build", lambda: False)
        props = _Props(80 * GIB, 0)
        assert eval(
            self._vram_expression(), self._namespace(1, lambda index: props)
        ) == pytest.approx(80.0)
