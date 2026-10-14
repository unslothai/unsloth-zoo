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

"""Standby (sleep mode) allocator routing per device.

Drives the real `patch_vllm_enable_sleep_mode` against fake
`vllm.device_allocator.{cumem,xpumem}` modules: XPU must patch vLLM's
XpuMemAllocator (the one its worker uses) and copy via `_xpu_memcpy_sync`,
CUDA keeps CuMemAllocator + cudaMemcpy, other devices keep the old CUDA route.
"""

import dataclasses
import sys
import types

import pytest


@pytest.fixture(scope = "module")
def vllm_utils():
    try:
        import unsloth_zoo.vllm_utils as m
    except Exception as e:  # no accelerator on this host
        pytest.skip(f"unsloth_zoo.vllm_utils unavailable: {e}")
    return m


@dataclasses.dataclass
class _AllocationData:
    handle: tuple
    tag: str
    cpu_backup_tensor: object = None
    is_asleep: bool = False


def _install_fake_vllm(vllm_utils, monkeypatch, device_type, with_xpumem = True):
    calls = []
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        monkeypatch.delenv(key, raising = False)

    class _Generate:
        def generate(self, *args, **kwargs):
            return None

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = type("LLM", (_Generate,), {})
    fake_vllm.AsyncLLMEngine = type("AsyncLLMEngine", (_Generate,), {})
    allocator_pkg = types.ModuleType("vllm.device_allocator")
    utils = types.ModuleType("vllm.utils")
    utils.is_pin_memory_available = lambda: False

    cumem = types.ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = type("CuMemAllocator", (), {"default_tag": "default"})
    cumem.AllocationData = _AllocationData
    cumem.unmap_and_release = lambda h: calls.append(("unmap", h[2]))
    cumem.create_and_map = lambda h: calls.append(("map", h[2]))
    cumem.libcudart = types.SimpleNamespace(
        cudaMemcpy = lambda dst, src, n: calls.append(("cudaMemcpy", dst, src, n)),
    )

    xpumem = types.ModuleType("vllm.device_allocator.xpumem")
    xpumem.XpuMemAllocator = type("XpuMemAllocator", (), {"default_tag": "default"})
    xpumem.AllocationData = _AllocationData
    xpumem.unmap_and_release = lambda h: calls.append(("unmap", h[2]))
    xpumem.create_and_allocate = lambda h: calls.append(("map", h[2]))
    xpumem.MEMCPY_HOST_TO_DEVICE = 0
    xpumem.MEMCPY_DEVICE_TO_HOST = 1
    xpumem._xpu_memcpy_sync = lambda dst, src, n, kind, dev: calls.append(
        ("xpu_memcpy", dst, src, n, kind, dev)
    )

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator", allocator_pkg)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem)
    monkeypatch.setitem(
        sys.modules, "vllm.device_allocator.xpumem", xpumem if with_xpumem else None
    )
    monkeypatch.setattr(vllm_utils, "vllm", fake_vllm, raising = False)
    monkeypatch.setattr(vllm_utils, "DEVICE_TYPE", device_type)
    monkeypatch.setattr(vllm_utils, "_device_empty_cache", lambda: calls.append("empty_cache"))
    return calls, cumem.CuMemAllocator, xpumem.XpuMemAllocator


def _sleep_and_wake(allocator_cls, calls):
    alloc = allocator_cls()
    alloc.pointer_to_data = {
        100: _AllocationData((3, 64, 100, 7), "weights"),
        200: _AllocationData((3, 32, 200, 8), "kv_cache"),
        300: _AllocationData((3, 16, 300, 9), "default"),
    }
    alloc.sleep(offload_tags = ("kv_cache",))
    cpu_ptr = alloc.pointer_to_data[200].cpu_backup_tensor.data_ptr()
    sleep_calls = [c for c in calls if c != "empty_cache"]
    del calls[:]
    alloc.wake_up()
    wake_calls = [c for c in calls if c != "empty_cache"]
    assert alloc.pointer_to_data[200].cpu_backup_tensor is None
    return cpu_ptr, sleep_calls, wake_calls


def test_xpu_patches_xpu_allocator_and_uses_xpu_memcpy(vllm_utils, monkeypatch):
    calls, cumem_cls, xpu_cls = _install_fake_vllm(vllm_utils, monkeypatch, "xpu")
    vllm_utils.patch_vllm_enable_sleep_mode()

    assert "sleep" in vars(xpu_cls) and "wake_up" in vars(xpu_cls)
    assert "sleep" not in vars(cumem_cls)
    cpu_ptr, sleep_calls, wake_calls = _sleep_and_wake(xpu_cls, calls)
    # weights are Unsloth-managed: never copied, unmapped or remapped.
    assert sleep_calls == [
        ("xpu_memcpy", cpu_ptr, 200, 32, 1, 3),
        ("unmap", 200),
        ("unmap", 300),
    ]
    assert wake_calls == [
        ("map", 200),
        ("xpu_memcpy", 200, cpu_ptr, 32, 0, 3),
        ("map", 300),
    ]


@pytest.mark.parametrize("device_type", ["cuda", "hip"])
def test_cuda_keeps_cumem_allocator_and_cudamemcpy(vllm_utils, monkeypatch, device_type):
    calls, cumem_cls, xpu_cls = _install_fake_vllm(vllm_utils, monkeypatch, device_type)
    vllm_utils.patch_vllm_enable_sleep_mode()

    assert "sleep" in vars(cumem_cls) and "sleep" not in vars(xpu_cls)
    cpu_ptr, sleep_calls, wake_calls = _sleep_and_wake(cumem_cls, calls)
    assert sleep_calls == [
        ("cudaMemcpy", cpu_ptr, 200, 32),
        ("unmap", 200),
        ("unmap", 300),
    ]
    assert wake_calls == [
        ("map", 200),
        ("cudaMemcpy", 200, cpu_ptr, 32),
        ("map", 300),
    ]


def test_other_devices_keep_the_cumem_route(vllm_utils, monkeypatch):
    # Before XPU support every device patched CuMemAllocator; Ascend NPU
    # (vllm-ascend) with UNSLOTH_VLLM_STANDBY must still load, not raise.
    calls, cumem_cls, xpu_cls = _install_fake_vllm(vllm_utils, monkeypatch, "npu")
    vllm_utils.patch_vllm_enable_sleep_mode()
    assert "sleep" in vars(cumem_cls) and "sleep" not in vars(xpu_cls)


def test_xpu_without_xpumem_names_the_vllm_floor(vllm_utils, monkeypatch):
    _install_fake_vllm(vllm_utils, monkeypatch, "xpu", with_xpumem = False)
    with pytest.raises(RuntimeError, match = r"vLLM >= 0\.23\.0"):
        vllm_utils.patch_vllm_enable_sleep_mode()
