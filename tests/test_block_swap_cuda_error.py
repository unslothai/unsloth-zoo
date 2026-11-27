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

"""A refused cudaHostRegister / cudaHostUnregister must not leave its error pending for the thread's next
kernel launch (block swap host arena and disk tier buffers)."""

import importlib.util
import os
import sys
import types

import pytest
import torch
from _torch_version_tag import drop_rocm_tag  # noqa: E402

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dspec = importlib.util.spec_from_file_location(
    "unsloth_zoo.block_swap_disk", os.path.join(_HERE, "unsloth_zoo", "block_swap_disk.py"))
_dmod = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_dmod)
sys.modules.setdefault("unsloth_zoo.block_swap_disk", _dmod)
_spec = importlib.util.spec_from_file_location(
    "block_swap_cuda_error_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_disk = _mod._disk()

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA / ROCm GPU")


def _fake_cudart(monkeypatch, register_rc, unregister_rc, events):
    fake = types.SimpleNamespace(
        cudaHostRegister = lambda ptr, n, flags: events.append("register") or register_rc,
        cudaHostUnregister = lambda ptr: events.append("unregister") or unregister_rc,
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_disk, "consume_cuda_error", lambda: events.append("consumed"))


def test_consume_never_raises_without_a_cuda_runtime(monkeypatch):
    import ctypes

    def _no_runtime(*_a, **_k):
        raise OSError("no libcudart")

    monkeypatch.setattr(_disk, "_GET_LAST_ERROR", [])
    monkeypatch.setattr(ctypes, "CDLL", _no_runtime)
    _disk.consume_cuda_error()
    assert _disk._GET_LAST_ERROR == []


def test_consume_calls_the_resolved_get_last_error(monkeypatch):
    calls = []
    monkeypatch.setattr(_disk, "_GET_LAST_ERROR", [lambda: calls.append(1) or 0])
    _disk.consume_cuda_error()
    assert calls == [1]


@pytest.mark.parametrize("hip, want", [(None, "cudaGetLastError"), ("6.4", "hipGetLastError")])
def test_the_resolver_asks_the_runtime_torch_runs_on(monkeypatch, hip, want):
    import ctypes
    opened = []

    class _Lib:
        def __init__(self, path):
            opened.append(path)

        def __getattr__(self, name):
            return name

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    if not hip:
        drop_rocm_tag(monkeypatch, torch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", _Lib)
    assert _disk._resolve_get_last_error() == want
    assert opened == [torch._C.__file__]


def test_a_refused_arena_registration_consumes_before_raising(monkeypatch):
    events = []
    _fake_cudart(monkeypatch, 712, 0, events)
    with pytest.raises(RuntimeError, match = "cudaHostRegister failed"):
        _mod._Registered(torch.empty(1024, dtype = torch.uint8), 1024)
    assert events == ["register", "consumed"]


def test_a_refused_arena_unregistration_consumes(monkeypatch):
    events = []
    _fake_cudart(monkeypatch, 0, 1, events)
    reg = _mod._Registered(torch.empty(1024, dtype = torch.uint8), 1024)
    reg.__del__()
    assert events == ["register", "unregister", "consumed"] and reg.ptr is None


def test_an_accepted_registration_consumes_nothing(monkeypatch):
    events = []
    _fake_cudart(monkeypatch, 0, 0, events)
    reg = _mod._Registered(torch.empty(1024, dtype = torch.uint8), 1024)
    buf = _disk._HostBuf(4096, pin = True)
    assert buf.pinned
    reg.__del__()
    buf.__del__()
    assert events == ["register", "register", "unregister", "unregister"]


def test_a_refused_disk_buffer_registration_consumes_and_stays_pageable(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)
    events = []
    _fake_cudart(monkeypatch, 712, 0, events)
    buf = _disk._HostBuf(4096, pin = True)
    assert events == ["register", "consumed"] and not buf.pinned


def test_a_refused_disk_buffer_unregistration_consumes(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)
    events = []
    _fake_cudart(monkeypatch, 0, 1, events)
    buf = _disk._HostBuf(4096, pin = True)
    buf.__del__()
    assert events == ["register", "unregister", "consumed"] and not buf.pinned


def _launch():
    x = torch.ones(16, device = "cuda")
    (x * 2).sum().item()


_hip = bool(getattr(torch.version, "hip", None))
# Refusals both runtimes give: a zero-byte range (invalid value). CUDA also refuses a range registered twice
# (cudaErrorHostMemoryAlreadyRegistered); HIP accepts that one, so it is CUDA only.
_REFUSALS = ["zero_bytes", pytest.param("already_registered", marks = pytest.mark.skipif(
    _hip, reason = "HIP accepts a second registration of the same range"))]


@_cuda
@pytest.mark.parametrize("refusal", _REFUSALS)
def test_the_next_kernel_after_a_refused_arena_registration_runs(refusal):
    buf = torch.empty(1 << 20, dtype = torch.uint8)
    first = _mod._Registered(buf, buf.numel()) if refusal == "already_registered" else None
    with pytest.raises(RuntimeError, match = "cudaHostRegister failed"):
        _mod._Registered(buf, 0 if refusal == "zero_bytes" else buf.numel())
    _launch()
    del first


@_cuda
@pytest.mark.parametrize("refusal", _REFUSALS)
def test_the_next_kernel_after_a_refused_disk_buffer_registration_runs(monkeypatch, refusal):
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)
    first = _disk._HostBuf(1 << 20, pin = True) if refusal == "already_registered" else None
    real = torch.cuda.cudart()
    taken = first.ptr if first is not None else None
    # The buffer's registration reaches the runtime with arguments it refuses for real.
    redirect = types.SimpleNamespace(
        cudaHostRegister = lambda ptr, n, flags: (real.cudaHostRegister(ptr, 0, flags) if taken is None
                                                  else real.cudaHostRegister(taken, n, flags)),
        cudaHostUnregister = real.cudaHostUnregister,
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: redirect)
    second = _disk._HostBuf(1 << 20, pin = True)
    assert not second.pinned
    monkeypatch.undo()
    _launch()
    del first, second


@_cuda
def test_the_next_kernel_after_a_refused_unregistration_runs():
    buf = torch.empty(1 << 20, dtype = torch.uint8)
    reg = _mod._Registered(buf, buf.numel())
    assert int(torch.cuda.cudart().cudaHostUnregister(reg.ptr)) == 0
    # The range is no longer registered: the destructor's own unregister is refused.
    reg.__del__()
    _launch()
