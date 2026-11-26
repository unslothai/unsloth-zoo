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

"""One OOM near the memory ceiling must not turn a fast path off for the rest of the process: inject it once,
then the next call takes the fast path again. CPU only."""
import inspect
import math
import os
import typing

import pytest
import torch


def _oom():
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _compile_oom():
    # What Dynamo raises when the OOM happens while Inductor compiles or autotunes the graph.
    from torch._dynamo.exc import BackendCompilerFailed

    try:
        return BackendCompilerFailed(lambda *a: None, _oom(), None)
    except TypeError:  # older signature without first_useful_frame
        return BackendCompilerFailed(lambda *a: None, _oom())


class _FailOnce:
    """Stands in for a compiled callable: raises `error` on the first call, then runs `fn`."""

    def __init__(self, fn, error):
        self.fn, self.error, self.calls, self.ok = fn, error, 0, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        out = self.fn(*args, **kwargs)
        self.ok += 1
        return out


# Fused CE chunk 0 probe.


@pytest.mark.usefixtures("compile_enabled")
@pytest.mark.parametrize("make_error", [_oom, _compile_oom])
def test_fused_ce_probe_oom_keeps_compile_and_compiled_graphs(monkeypatch, make_error):
    import unsloth_zoo.fused_losses.cross_entropy_loss as ce

    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_SUPPORTED", None)
    wrappers = []

    def fake_compile(fn, *args, **kwargs):
        # One compiled chunk shared across calls, like Dynamo's cache: only the first run fails.
        if not wrappers:
            wrappers.append(_FailOnce(fn, make_error()))
        wrappers[0].fn = fn
        return wrappers[0]

    resets = []
    monkeypatch.setattr(ce.torch, "compile", fake_compile)
    monkeypatch.setattr(ce.torch._dynamo, "reset", lambda: resets.append(1))

    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 37, 16, generator = gen)
    weight = torch.randn(64, 16, generator = gen) * 0.5
    labels = torch.randint(0, 64, (2, 37), generator = gen)

    def loss():
        h = hidden.clone().requires_grad_(True)
        out = ce.unsloth_fused_ce_loss(None, h, weight, None, labels, torch_compile = True, n_chunks = 3)
        out.backward()
        return out.detach(), h.grad

    reference = ce.unsloth_fused_ce_loss(
        None, hidden.clone().requires_grad_(True), weight, None, labels, torch_compile = False, n_chunks = 3
    ).detach()
    first, first_grad = loss()
    # The OOM call fell back to eager for itself only: same loss, no latch, no dynamo.reset().
    torch.testing.assert_close(first, reference)
    assert ce._FUSED_CE_COMPILE_SUPPORTED is None
    assert resets == []
    ok_before = wrappers[0].ok
    second, second_grad = loss()
    assert wrappers[0].ok > ok_before, "the next call did not take the compiled chunk"
    assert ce._FUSED_CE_COMPILE_SUPPORTED is True
    torch.testing.assert_close(second, reference)
    torch.testing.assert_close(second_grad, first_grad)


@pytest.mark.usefixtures("compile_enabled")
def test_fused_ce_probe_real_failure_still_latches(monkeypatch):
    import unsloth_zoo.fused_losses.cross_entropy_loss as ce

    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_SUPPORTED", None)
    monkeypatch.setattr(ce.torch, "compile", lambda fn, *a, **k: _FailOnce(fn, RuntimeError("inductor bug")))
    resets = []
    monkeypatch.setattr(ce.torch._dynamo, "reset", lambda: resets.append(1))
    gen = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, 9, 8, generator = gen, requires_grad = True)
    weight = torch.randn(32, 8, generator = gen)
    labels = torch.randint(0, 32, (1, 9), generator = gen)
    ce.unsloth_fused_ce_loss(None, hidden, weight, None, labels, torch_compile = True, n_chunks = 2).backward()
    assert ce._FUSED_CE_COMPILE_SUPPORTED is False
    assert resets == [1]


# GRPO recompute run().


def _recompute_standalone(monkeypatch, make_error):
    import unsloth_zoo.rl_replacements as rr
    import unsloth_zoo.temporary_patches.common as common

    kernels = []

    def compile_function(**kwargs):
        def wrap(fn):
            kernels.append(_FailOnce(fn, make_error()))
            return kernels[-1]

        return wrap

    monkeypatch.setattr(common, "_maybe_compile", compile_function)
    namespace = {"torch": torch, "os": os, "math": math, "Optional": typing.Optional}
    source = inspect.getsource(rr.chunked_hidden_states_selective_log_softmax_recompute)
    exec(compile(source, "<recompute>", "exec"), namespace)
    return namespace["chunked_hidden_states_selective_log_softmax_recompute"], kernels


@pytest.mark.parametrize("make_error", [_oom, _compile_oom])
def test_recompute_oom_runs_eager_once_then_compiled(monkeypatch, make_error):
    function, kernels = _recompute_standalone(monkeypatch, make_error)
    gen = torch.Generator().manual_seed(2)
    head = torch.randn(37, 6, generator = gen, dtype = torch.float64)
    states = torch.randn(1, 10, 6, generator = gen, dtype = torch.float64, requires_grad = True)
    index = torch.randint(0, 37, (1, 10), generator = gen)

    def call():
        return function(states, head, index, 3, fused_linear = False)

    reference = torch.log_softmax(states @ head.t(), -1).gather(-1, index.unsqueeze(-1)).squeeze(-1).detach()
    first = call()
    torch.testing.assert_close(first.double(), reference, rtol = 1e-5, atol = 1e-6)
    assert getattr(function, "_use_compiled", True), "an OOM latched the compiled recompute off"
    forward = kernels[0]
    ok_before = forward.ok
    second = call()
    assert forward.ok > ok_before, "the next call did not take the compiled kernel"
    torch.testing.assert_close(second.double(), reference, rtol = 1e-5, atol = 1e-6)


def test_recompute_real_failure_still_latches(monkeypatch):
    function, _ = _recompute_standalone(monkeypatch, lambda: RuntimeError("inductor bug"))
    gen = torch.Generator().manual_seed(3)
    head = torch.randn(11, 4, generator = gen, dtype = torch.float64)
    states = torch.randn(1, 5, 4, generator = gen, dtype = torch.float64)
    index = torch.randint(0, 11, (1, 5), generator = gen)
    function(states, head, index, 2, fused_linear = False)
    assert function._use_compiled is False


# MXFP4 dense probe.


def test_mxfp4_dense_probe_oom_is_not_cached(monkeypatch):
    import unsloth_zoo.mxfp4_gemm as mg

    monkeypatch.setattr(mg, "_AVAILABLE_DTYPE", {})
    monkeypatch.setattr(mg, "_TRACE_AVAILABLE", {})
    monkeypatch.setattr(mg, "mxfp4_grouped_mm_available", lambda device = None: True)
    probe = _FailOnce(lambda device, dtype: True, _oom())
    monkeypatch.setattr(mg, "_probe_dense", probe)
    monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", "auto")
    assert mg.mxfp4_grouped_gemv_available("cpu", torch.bfloat16) is False
    assert ("cpu", torch.bfloat16) not in mg._AVAILABLE_DTYPE
    assert mg.mxfp4_grouped_gemv_available("cpu", torch.bfloat16) is True
    assert probe.ok == 1


# triton_kernels matmul_ogs self check.


def test_matmul_ogs_self_check_oom_is_not_cached(monkeypatch):
    import unsloth_zoo.triton_kernels_compat as tkc

    monkeypatch.setattr(tkc, "_DEVICE_OK", {})
    monkeypatch.setattr(tkc, "native_mxfp4_matmul_ok", lambda device, backend: True)
    monkeypatch.setattr(tkc, "get_triton_kernels_backend", lambda: None)
    check = _FailOnce(lambda device: True, _oom())
    monkeypatch.setattr(tkc, "_self_check", check)
    assert tkc.matmul_ogs_available("cpu") is False
    assert "cpu" not in tkc._DEVICE_OK
    assert tkc.matmul_ogs_available("cpu") is True
    assert check.ok == 1
