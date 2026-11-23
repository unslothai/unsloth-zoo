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

"""Standby's patched CuMemAllocator.sleep must take both call shapes: vLLM <= 0.33 calls sleep(offload_tags) and
vLLM >= 0.34 (CuMemBackend.suspend) calls sleep(offload_tags, discardable). On 0.34 AllocationData also carries
is_asleep, which vLLM's own discard/_trim/free paths read, so the patch must keep it in sync."""

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
class _OldAllocationData:  # vLLM <= 0.33
    handle: tuple
    tag: str
    cpu_backup_tensor: object = None


@dataclasses.dataclass
class _NewAllocationData:  # vLLM >= 0.34
    handle: tuple
    tag: str
    cpu_backup_tensor: object = None
    is_asleep: bool = False


def _install_fake_cumem(vllm_utils, monkeypatch, data_cls):
    calls = []
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        monkeypatch.delenv(key, raising = False)

    class _Generate:
        def generate(self, *args, **kwargs):
            return None

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = type("LLM", (_Generate,), {})
    fake_vllm.AsyncLLMEngine = type("AsyncLLMEngine", (_Generate,), {})
    utils = types.ModuleType("vllm.utils")
    utils.is_pin_memory_available = lambda: False
    cumem = types.ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = type("CuMemAllocator", (), {"default_tag": "default"})
    cumem.AllocationData = data_cls
    cumem.unmap_and_release = lambda h: calls.append(("unmap", h[2]))
    cumem.create_and_map = lambda h: calls.append(("map", h[2]))
    cumem.libcudart = types.SimpleNamespace(
        cudaMemcpy = lambda dst, src, n: calls.append(("cudaMemcpy", n)),
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator", types.ModuleType("vllm.device_allocator"))
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem)
    monkeypatch.setattr(vllm_utils, "vllm", fake_vllm, raising = False)
    monkeypatch.setattr(vllm_utils, "vllm_version", "0.34.0", raising = False)
    monkeypatch.setattr(vllm_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(vllm_utils, "_device_empty_cache", lambda: None)
    vllm_utils.patch_vllm_enable_sleep_mode()
    alloc = cumem.CuMemAllocator()
    alloc.pointer_to_data = {
        100: data_cls((0, 64, 100, 7), "weights"),
        200: data_cls((0, 32, 200, 8), "kv_cache"),
        300: data_cls((0, 16, 300, 9), "runtime"),
        400: data_cls((0, 8, 400, 10), "cudagraph"),
    }
    return calls, alloc


def _expected_sleep_calls():
    # weights untouched; runtime + cudagraph backed up then released; kv_cache released without a copy.
    return [("unmap", 200), ("cudaMemcpy", 16), ("unmap", 300), ("cudaMemcpy", 8), ("unmap", 400)]


def test_patched_module_is_this_tree(vllm_utils):
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(vllm_utils.__file__).resolve().is_relative_to(root), vllm_utils.__file__


@pytest.mark.parametrize("data_cls", [_OldAllocationData, _NewAllocationData])
def test_sleep_accepts_vllm_034_positional_discardable(vllm_utils, monkeypatch, data_cls):
    calls, alloc = _install_fake_cumem(vllm_utils, monkeypatch, data_cls)
    # Exactly how vLLM 0.34 CuMemBackend.suspend calls it.
    alloc.sleep(("weights", "runtime", "cudagraph"), {100})
    assert calls == _expected_sleep_calls()
    del calls[:]
    alloc.wake_up()
    assert calls == [("map", 200), ("map", 300), ("cudaMemcpy", 16), ("map", 400), ("cudaMemcpy", 8)]


@pytest.mark.parametrize("data_cls", [_OldAllocationData, _NewAllocationData])
def test_sleep_keeps_old_call_shapes(vllm_utils, monkeypatch, data_cls):
    calls, alloc = _install_fake_cumem(vllm_utils, monkeypatch, data_cls)
    alloc.sleep(offload_tags = ("weights", "runtime", "cudagraph"))
    assert calls == _expected_sleep_calls()
    _, alloc = _install_fake_cumem(vllm_utils, monkeypatch, data_cls)
    alloc.sleep(("weights", "runtime", "cudagraph"), discardable = set())
    _, alloc = _install_fake_cumem(vllm_utils, monkeypatch, data_cls)
    alloc.sleep()


def test_is_asleep_tracked_and_no_double_release(vllm_utils, monkeypatch):
    calls, alloc = _install_fake_cumem(vllm_utils, monkeypatch, _NewAllocationData)
    alloc.sleep(("weights", "runtime", "cudagraph"), set())
    assert [d.is_asleep for d in alloc.pointer_to_data.values()] == [False, True, True, True]
    del calls[:]
    # A second sleep must not release handles that are already released.
    alloc.sleep(("weights", "runtime", "cudagraph"), set())
    assert [c for c in calls if c[0] == "unmap"] == []
    alloc.wake_up()
    assert [d.is_asleep for d in alloc.pointer_to_data.values()] == [False] * 4
    del calls[:]
    # A second wake_up must not map still-mapped handles again.
    alloc.wake_up()
    assert calls == []
