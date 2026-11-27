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

"""An out-of-memory error in a capability probe or a compiled call records no verdict: the exact-merge self-check,
the MXFP4 hardware-convert probe, chunked_logprob's compiled kernels and the CPU loader's compiled decoder layers
must not latch "unsupported" / eager for the process after one transient OOM. CPU only, with mocks."""

import contextlib

import pytest
import torch


def _oom():
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")


class _CompileError(Exception):
    """What Dynamo / Inductor raise around an error hit while compiling (BackendCompilerFailed, InductorError)."""

    def __init__(self, inner):
        super().__init__(f"backend compiler failed: {type(inner).__name__}: {inner}")
        self.inner_exception = inner


def test_is_out_of_memory_sees_wrapped_and_host_errors():
    from unsloth_zoo.utils import _is_out_of_memory

    assert _is_out_of_memory(_oom())
    assert _is_out_of_memory(MemoryError())
    assert _is_out_of_memory(_CompileError(_oom()))
    try:
        try:
            raise _oom()
        except Exception as inner:
            raise RuntimeError("lowering failed") from inner
    except Exception as error:
        assert _is_out_of_memory(error)
    assert not _is_out_of_memory(RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED"))
    assert not _is_out_of_memory(None)


def test_exact_merge_self_check_reprobes_after_oom(monkeypatch):
    from unsloth_zoo import merged_rollout_exact as M

    monkeypatch.setattr(M, "triton", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_SELF_CHECK", {})
    results = [_oom(), True]

    def self_check(device, dtype):
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(M, "_self_check", self_check)
    device = torch.device("cuda", 0)
    assert M.exact_merge_available(device) is False
    assert M._SELF_CHECK == {}
    assert M.exact_merge_failure(device) is None
    assert M.exact_merge_available(device) is True
    # A real failure still latches.
    monkeypatch.setattr(M, "_SELF_CHECK", {})
    monkeypatch.setattr(M, "_self_check", lambda device, dtype: (_ for _ in ()).throw(RuntimeError("bad ptx")))
    assert M.exact_merge_available(device) is False
    assert "bad ptx" in M.exact_merge_failure(device)


@pytest.mark.parametrize("error, verdict", [(_oom(), None), (RuntimeError("ptxas: unsupported"), False)])
def test_mxfp4_convert_probe_gives_no_verdict_on_oom(monkeypatch, error, verdict):
    from unsloth_zoo import mxfp4_gemm as G

    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.delenv("UNSLOTH_MXFP4_CVT", raising = False)

    def arange(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "arange", arange)
    assert G._cvt_ok(0, eager = True) is verdict


def test_chunked_logprob_does_not_latch_eager_on_a_compile_oom(monkeypatch):
    from unsloth_zoo.fused_losses import chunked_logprob as C

    monkeypatch.setattr(C, "_COMPILE", True)
    monkeypatch.setattr(C, "_eager", set())
    calls = []

    def compiled(x):
        calls.append(1)
        if len(calls) == 1:
            raise _CompileError(_oom())
        return x + 1

    monkeypatch.setattr(C, "_compiled", {"probe": compiled})
    x = torch.zeros(1)
    with pytest.raises(_CompileError):
        C._run("probe", lambda x: x + 2, x)
    assert "probe" not in C._eager
    assert C._run("probe", lambda x: x + 2, x).item() == 1  # compiled again, not the eager fn


def test_cpu_loader_compiled_layer_survives_one_oom(monkeypatch, capsys):
    from unsloth_zoo.cpu import loader

    layer = torch.nn.Linear(4, 4)
    calls = []

    def compiled(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError()
        return torch.full((2, 4), 7.0)

    layer._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([layer])
    x = torch.ones(2, 4)
    with torch.no_grad():
        eager = layer(x)
        assert torch.equal(eager, torch.nn.functional.linear(x, layer.weight, layer.bias))
        assert layer._compiled_call_impl is not None, "one out-of-memory call dropped the layer to eager"
        assert torch.equal(layer(x), torch.full((2, 4), 7.0))
    assert len(calls) == 2
    assert "training eagerly" not in capsys.readouterr().out


def test_cpu_loader_out_of_memory_in_training_restarts_eagerly_once(monkeypatch, capsys):
    from unsloth_zoo.cpu import loader

    layer = torch.nn.Linear(4, 4)
    calls = []

    def compiled(x):
        calls.append(1)
        if len(calls) == 1:
            raise _CompileError(_oom())
        return torch.nn.functional.linear(x, layer.weight, layer.bias)

    layer._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([layer])
    x = torch.ones(2, 4, requires_grad = True)
    for _ in range(2):
        layer.zero_grad()
        layer(x).square().sum().backward()
        assert layer.weight.grad is not None
    assert len(calls) == 2  # the OOM call reran eagerly once, the second step compiled
    assert "_unsloth_eager_once" not in layer.__dict__
    assert layer._compiled_call_impl is not None
    assert "training eagerly" not in capsys.readouterr().out
