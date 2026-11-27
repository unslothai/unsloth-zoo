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
        layer(x * 1).square().sum().backward()  # an activation, freed with the graph, as between real steps
        assert layer.weight.grad is not None
    assert len(calls) == 2  # the OOM call reran eagerly once, the second step compiled
    assert "_unsloth_eager_once" not in layer.__dict__
    assert layer._compiled_call_impl is not None
    assert "training eagerly" not in capsys.readouterr().out


@pytest.mark.parametrize("reentrant", [False, True])
def test_cpu_loader_out_of_memory_retry_recomputes_eagerly_in_its_checkpoint(monkeypatch, capsys, reentrant):
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader

    torch.manual_seed(0)
    block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
    # Stands in for the compiled forward without Dynamo: same values, but an extra op first saves different tensors,
    # so a checkpoint that recomputes in the other mode raises CheckpointError.
    scale = torch.ones(())
    compiled_forward = lambda x: block._call_impl(x * scale)
    calls = []

    def compiled(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        if len(calls) == 1:
            raise MemoryError()
        return compiled_forward(*args, **kwargs)

    block._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([block])
    x = torch.randn(4, 8)
    grads = []
    for _ in range(2):  # the out-of-memory step (eager forward and recompute), then a compiled one
        block.zero_grad()
        h = x.clone().requires_grad_(True)
        checkpoint.checkpoint(block, h, use_reentrant = reentrant).square().sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in block.parameters()]))
    torch.testing.assert_close(grads[0], grads[1])
    assert block._compiled_call_impl is not None
    assert "training eagerly" not in capsys.readouterr().out


@pytest.mark.parametrize("eager_first", [True, False])
def test_cpu_loader_checkpoints_sharing_an_input_after_an_oom_retry(monkeypatch, capsys, eager_first):
    # Two outstanding checkpoints of one layer on the same input, the first retried eagerly after an OOM: neither may
    # recompute in the other's mode, whichever is backpropagated first.
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader

    torch.manual_seed(0)
    block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
    # Stands in for the compiled forward without Dynamo: same values, but an extra op first saves different tensors,
    # so a checkpoint that recomputes in the other mode raises CheckpointError.
    scale = torch.ones(())
    compiled_forward = lambda x: block._call_impl(x * scale)
    calls = []

    def compiled(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError()
        return compiled_forward(*args, **kwargs)

    block._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([block])
    x = torch.randn(4, 8, requires_grad = True)
    h = x.detach().clone().requires_grad_(True)  # this step's input, freed with it
    first = checkpoint.checkpoint(block, h, use_reentrant = False)
    second = checkpoint.checkpoint(block, h, use_reentrant = False)
    for out in (first, second) if eager_first else (second, first):
        out.square().sum().backward()
    del h, first, second, out
    # The step's graph is gone: the layer compiles again on the next step.
    block.zero_grad()
    checkpoint.checkpoint(block, x.detach().clone().requires_grad_(True), use_reentrant = False).square().sum().backward()
    assert len(calls) == 3  # the OOM, the next step's compiled forward and its compiled recompute
    assert "training eagerly" not in capsys.readouterr().out


class _KeywordBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up, self.down = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)

    def forward(self, input):
        return self.down(torch.relu(self.up(input)))


def _oom_once_then(block, failures = 1):
    scale = torch.ones(())
    calls = []

    def compiled(*args, **kwargs):
        calls.append(1)
        if len(calls) <= failures:
            raise MemoryError()
        # Stand-in compiled forward: same values, different saved tensors (see above).
        if args:
            return block._call_impl(args[0] * scale)
        return block._call_impl(input = kwargs["input"] * scale)

    block._compiled_call_impl = compiled
    return calls


@pytest.mark.parametrize("keyword", [False, True])
def test_cpu_loader_out_of_memory_retry_replays_eagerly_with_retain_graph_and_keyword_inputs(capsys, keyword):
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader

    torch.manual_seed(0)
    block = _KeywordBlock()
    calls = _oom_once_then(block)
    loader._fall_back_to_eager_on_compile_error([block])
    h = torch.randn(4, 8, requires_grad = True)
    out = checkpoint.checkpoint(block, input = h, use_reentrant = False) if keyword else \
        checkpoint.checkpoint(block, h, use_reentrant = False)
    loss = out.square().sum()
    loss.backward(retain_graph = True)
    first = [p.grad.clone() for p in block.parameters()]
    block.zero_grad()
    loss.backward()  # the retained graph recomputes again: still eagerly, as the forward ran
    for a, p in zip(first, block.parameters()):
        torch.testing.assert_close(a, p.grad)
    assert len(calls) == 1
    del h, out, loss
    block.zero_grad()
    x = torch.randn(4, 8, requires_grad = True)
    out = checkpoint.checkpoint(block, input = x, use_reentrant = False) if keyword else \
        checkpoint.checkpoint(block, x, use_reentrant = False)
    out.square().sum().backward()
    assert len(calls) == 3  # compiled again: the forward and its recompute
    assert "training eagerly" not in capsys.readouterr().out


class _DictBlock(_KeywordBlock):
    def forward(self, inputs):
        return super().forward(inputs["input"])


def test_cpu_loader_out_of_memory_retry_with_nested_inputs_and_many_checkpoints(capsys):
    # A tensor inside a dict argument, and more outstanding checkpoints than the compiled queue keeps (64): each must
    # still recompute eagerly after the out-of-memory retry.
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader

    torch.manual_seed(0)
    block = _DictBlock()
    scale = torch.ones(())
    calls = []

    def compiled(inputs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError()
        return block._call_impl({"input": inputs["input"] * scale})

    block._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([block])
    xs = [torch.randn(4, 8, requires_grad = True) for _ in range(70)]
    outs = [checkpoint.checkpoint(block, {"input": x}, use_reentrant = False) for x in xs]
    torch.stack(outs).square().sum().backward()
    assert all(x.grad is not None for x in xs)
    assert len(calls) == 1
    assert "training eagerly" not in capsys.readouterr().out


def test_cpu_loader_reused_input_does_not_hold_the_eager_window_open(capsys):
    # A plain (not checkpointed) training layer called again and again on one long-lived input after an OOM: the
    # window closes with each step's output, so compile resumes and nothing accumulates.
    from unsloth_zoo.cpu import loader

    layer = torch.nn.Linear(4, 4)
    calls = []

    def compiled(x):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError()
        return torch.nn.functional.linear(x, layer.weight, layer.bias)

    layer._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([layer])
    x = torch.ones(2, 4, requires_grad = True)
    for _ in range(100):
        layer.zero_grad()
        layer(x).square().sum().backward()
    assert len(calls) == 100  # one OOM call rerun eagerly, every later step compiled
    assert "training eagerly" not in capsys.readouterr().out


def test_cpu_loader_retained_checkpoint_replays_eagerly_after_a_later_forward(capsys):
    # A retained eager checkpoint, then another forward on the same input: the retained graph can still recompute, so
    # the layer stays eager until it is freed, and its second backward replays eagerly.
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader

    torch.manual_seed(0)
    block = _KeywordBlock()
    calls = _oom_once_then(block)
    loader._fall_back_to_eager_on_compile_error([block])
    h = torch.randn(4, 8, requires_grad = True)
    loss = checkpoint.checkpoint(block, h, use_reentrant = False).square().sum()
    loss.backward(retain_graph = True)
    checkpoint.checkpoint(block, h, use_reentrant = False).square().sum().backward()
    loss.backward()
    assert len(calls) == 1  # nothing compiled while the retained graph lived
    del loss
    checkpoint.checkpoint(block, h, use_reentrant = False).square().sum().backward()
    assert len(calls) == 3  # freed: compiled again, forward and recompute
    assert "training eagerly" not in capsys.readouterr().out


class _TwoOutputBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.a(x), self.b(torch.relu(x))


def test_cpu_loader_eager_retry_window_covers_every_output():
    # Two outputs on independent branches: dropping the first must not reopen compile while the second can still be
    # backpropagated through the eagerly recorded checkpoint.
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader

    torch.manual_seed(0)
    block = _TwoOutputBlock()
    scale = torch.ones(())
    calls = []

    def compiled(x):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError()
        return block._call_impl(x * scale)

    block._compiled_call_impl = compiled
    loader._fall_back_to_eager_on_compile_error([block])
    h = torch.randn(4, 8, requires_grad = True)
    second = checkpoint.checkpoint(block, h, use_reentrant = False)[1]
    second.square().sum().backward()
    assert len(calls) == 1
