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

"""Standby wake_up without vLLM's refused FABRIC cuMemCreate (vLLM >= 0.16), against a fake CUDA driver."""

import ctypes
import inspect

import pytest

import unsloth_zoo.vllm_utils as vu

NOT_PERMITTED = 800
FABRIC, POSIX_FD = 8, 1
FAB_ATTR, RDMA_VMM_ATTR, RDMA_ATTR = 128, 110, 116


class FakeDriver:
    def __init__(self, fabric_attr = 1, fabric_create = NOT_PERMITTED, map_ret = 0, ctx = 1):
        self.fabric_attr, self.fabric_create, self.map_ret, self.ctx = fabric_attr, fabric_create, map_ret, ctx
        self.calls = []
        self.next_handle = 0xABC0

    def cuDeviceGetAttribute(self, out, attr, device):
        # 116 (non-VMM RDMA) answers 0 so a wrong attribute would leave the RDMA flag unset
        out._obj.value = {FAB_ATTR: self.fabric_attr, RDMA_VMM_ATTR: 1, RDMA_ATTR: 0}.get(attr, 0)
        return 0

    def cuMemGetAllocationGranularity(self, out, prop, option):
        out._obj.value = 2 << 20
        return 0

    def cuMemCreate(self, out, size, prop, flags):
        p = prop._obj
        self.calls.append(("create", p.requestedHandleTypes, p.allocFlags.gpuDirectRDMACapable, int(size.value)))
        if p.requestedHandleTypes == FABRIC and self.fabric_create:
            return self.fabric_create
        self.next_handle += 1
        out._obj.value = self.next_handle
        return 0

    def cuMemMap(self, ptr, size, offset, handle, flags):
        self.calls.append(("map", ptr.value, size.value, handle.value))
        return self.map_ret

    def cuMemSetAccess(self, ptr, size, desc, count):
        self.calls.append(("access", ptr.value, desc._obj.flags))
        return 0

    def cuMemRelease(self, handle):
        self.calls.append(("release", handle.value))
        return 0

    def cuMemUnmap(self, ptr, size):
        self.calls.append(("unmap", ptr.value))
        return 0

    def cuCtxGetCurrent(self, out):
        out._obj.value = self.ctx
        return 0


@pytest.fixture
def env(monkeypatch):
    def setup(vllm = "0.30.0", **driver_kw):
        drv = FakeDriver(**driver_kw)
        monkeypatch.setattr(vu, "_CUMEM_FAST_STATE", {"lib": drv})
        monkeypatch.setattr(vu, "vllm_version", vllm, raising = False)
        monkeypatch.setattr(vu.torch.version, "hip", None, raising = False)
        monkeypatch.delenv("UNSLOTH_VLLM_STANDBY_FAST_WAKE", raising = False)
        slot = ctypes.c_ulonglong(0x1111)
        vllm_calls = []
        handle = (0, 16 << 30, 0x7000_0000, ctypes.addressof(slot))
        return drv, slot, vllm_calls, handle
    return setup


def wake(handle, vllm_calls):
    return vu._cumem_create_and_map(handle, lambda h: vllm_calls.append(h))


def test_refused_fabric_is_probed_once_then_mapped_directly(env):
    drv, slot, vllm_calls, handle = env()
    wake(handle, vllm_calls)
    wake(handle, vllm_calls)
    assert vllm_calls == []
    creates = [c for c in drv.calls if c[0] == "create"]
    # one small FABRIC probe, then one POSIX fd create per wake with vLLM's RDMA flag and the full size
    assert [c[1] for c in creates] == [FABRIC, POSIX_FD, POSIX_FD]
    assert all(c[2] == 1 for c in creates) and creates[1][3] == 16 << 30
    assert [c for c in drv.calls if c[0] == "map"][0][1] == 0x7000_0000
    assert all(c[2] == 3 for c in drv.calls if c[0] == "access")
    assert slot.value == drv.next_handle  # vLLM's unmap_and_release reads this handle
    assert vu._CUMEM_FAST_STATE["fast_maps"] == 2


@pytest.mark.parametrize("case", [
    dict(vllm = "0.15.1"),                   # vLLM before the FABRIC attempt: its own path, untouched
    dict(fabric_create = 0),                 # fabric handles work here (IMEX): keep vLLM's FABRIC handle
    dict(fabric_attr = 0),                   # device without fabric support: vLLM never asks for it
    dict(ctx = 0),                           # no current context: let vLLM's ensure_context handle it
])
def test_other_machines_keep_vllms_create_and_map(env, case):
    drv, slot, vllm_calls, handle = env(**case)
    wake(handle, vllm_calls)
    assert vllm_calls == [handle]
    assert slot.value == 0x1111
    assert not [c for c in drv.calls if c[0] in ("map", "access")]


def test_opt_out_and_rocm_keep_vllms_path(env, monkeypatch):
    drv, slot, vllm_calls, handle = env()
    monkeypatch.setenv("UNSLOTH_VLLM_STANDBY_FAST_WAKE", "0")
    wake(handle, vllm_calls)
    monkeypatch.delenv("UNSLOTH_VLLM_STANDBY_FAST_WAKE")
    monkeypatch.setattr(vu.torch.version, "hip", "6.4", raising = False)
    wake(handle, vllm_calls)
    rocm_handle = (0, 1 << 30, 0x7000_0000, [1, 2], [1, 2], 2)  # ROCm passes chunk lists
    monkeypatch.setattr(vu.torch.version, "hip", None, raising = False)
    wake(rocm_handle, vllm_calls)
    assert vllm_calls == [handle, handle, rocm_handle]
    assert drv.calls == []


def test_a_failed_map_releases_and_falls_back(env):
    drv, slot, vllm_calls, handle = env(map_ret = 1)
    wake(handle, vllm_calls)
    assert vllm_calls == [handle]
    assert slot.value == 0x1111
    assert ("release", drv.next_handle) in drv.calls


def test_standby_wake_up_routes_through_the_fast_path():
    src = inspect.getsource(vu.patch_vllm_enable_sleep_mode)
    assert "_cumem_create_and_map(handle, create_and_map)" in src
