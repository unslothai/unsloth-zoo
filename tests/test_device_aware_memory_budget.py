"""The automatic tile / fused CE memory budget must be read from the operand's device.

`_default_target_gb` used to ask `mem_get_info(0)`, so under torchrun every rank sized its
chunks from GPU 0's free memory and opened a CUDA context on GPU 0. The fused CE multiplier
was also `functools.cache`d with the measurement inside, so the first reading (taken before
optimizer states exist) was reused for the rest of training.

CPU only: the accelerator namespace is mocked with one free-memory figure per device index
and a current device that is not 0.
"""
import importlib
import types

import pytest

GIB = 1024 ** 3


@pytest.fixture(autouse = True)
def _allow_cpu_import(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")


def _modules():
    try:
        ce = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
        tiled = importlib.import_module("unsloth_zoo.tiled_mlp")
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    return ce, tiled


class _FakeCuda:
    """Two GPUs with different free memory; the process lives on GPU 1 (rank 1)."""

    def __init__(self, free_by_index, current = 1):
        self.free_by_index = dict(free_by_index)
        self.current = current
        self.queried = []

    def is_available(self):
        return True

    def current_device(self):
        return self.current

    def mem_get_info(self, device = None):
        # torch resolves None to the current device, as torch.cuda.mem_get_info does.
        if device is None:
            index = self.current
        elif isinstance(device, int):
            index = device
        else:
            index = device.index if device.index is not None else self.current
        self.queried.append(index)
        free = self.free_by_index[index]
        return (free, 80 * GIB)


def _mock(monkeypatch, free_by_index, current = 1):
    ce, tiled = _modules()
    fake = _FakeCuda(free_by_index, current)
    monkeypatch.setattr(tiled, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(tiled.torch, "cuda", fake, raising = False)
    ce._get_chunk_multiplier.cache_clear()
    return ce, tiled, fake


def test_default_budget_uses_current_device_not_gpu0(monkeypatch):
    _, tiled, fake = _mock(monkeypatch, {0: 2 * GIB, 1: 20 * GIB}, current = 1)
    assert tiled._default_target_gb() == pytest.approx(10.0)
    assert 0 not in fake.queried, fake.queried


def test_default_budget_uses_the_given_device(monkeypatch):
    _, tiled, fake = _mock(monkeypatch, {0: 2 * GIB, 1: 20 * GIB}, current = 0)
    import torch as real_torch
    assert tiled._default_target_gb(real_torch.device("cuda", 1)) == pytest.approx(10.0)
    assert fake.queried == [1], fake.queried


def test_cpu_operand_budgets_from_host_ram(monkeypatch):
    _, tiled, fake = _mock(monkeypatch, {0: 2 * GIB, 1: 20 * GIB})
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available = 6 * GIB))
    import torch
    assert tiled._default_target_gb(torch.device("cpu")) == pytest.approx(3.0)
    assert fake.queried == []


def test_fused_ce_budget_is_remeasured_each_call(monkeypatch):
    """Free memory changes between steps; the chunk count must follow it."""
    ce, tiled, fake = _mock(monkeypatch, {0: 80 * GIB, 1: 8 * GIB}, current = 1)
    roomy = ce.get_chunk_size(1, 16_384, 151_936)
    fake.free_by_index[1] = 1 * GIB  # optimizer states / activations now hold most of it
    tight = ce.get_chunk_size(1, 16_384, 151_936)
    assert tight > roomy, (roomy, tight)
    assert 0 not in fake.queried, fake.queried


def test_fused_ce_explicit_budget_stays_cached(monkeypatch):
    ce, _, fake = _mock(monkeypatch, {0: 80 * GIB, 1: 8 * GIB})
    a = ce.get_chunk_size(1, 16_384, 151_936, target_gb = 2.0)
    b = ce.get_chunk_size(1, 16_384, 151_936, target_gb = 2.0)
    assert a == b
    assert ce._get_explicit_chunk_multiplier.cache_info().hits >= 1
    assert fake.queried == []


def test_fused_ce_passes_the_hidden_states_device(monkeypatch):
    ce, _ = _modules()
    import torch
    seen = []
    sentinel = object()

    def record(device = sentinel):
        seen.append(device)
        return 4.0

    monkeypatch.setattr(ce, "_free_target_gb", record)
    ce._get_chunk_multiplier.cache_clear()
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 16, requires_grad = True)
    weight = torch.randn(64, 16, requires_grad = True)
    labels = torch.randint(0, 64, (2, 6))
    loss = ce.unsloth_fused_ce_loss(
        trainer = None, hidden_states = hidden, lm_head_weight = weight,
        lm_head_bias = None, labels = labels,
    )
    loss.backward()
    assert seen and all(isinstance(d, torch.device) and d == hidden.device for d in seen), seen


def test_tiled_mlp_budget_follows_the_input_device(monkeypatch):
    _, tiled = _modules()
    import torch
    seen = []

    def record(device = None):
        seen.append(device)
        return 0.001

    monkeypatch.setattr(tiled, "_default_target_gb", record)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(16, 32)
            self.down = torch.nn.Linear(32, 16)

        def forward(self, x):
            return self.down(torch.nn.functional.gelu(self.up(x)))

    mlp = MLP()
    tiled.patch_mlp(mlp, target_arctic = False, target_gb = None, padded_length = 8)
    x = torch.randn(2, 64, 16, requires_grad = True)
    mlp(x).sum().backward()
    mlp(x).sum().backward()
    # Measured once per device, with the device the input lives on.
    assert seen == [x.device], seen
