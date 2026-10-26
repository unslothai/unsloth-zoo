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

"""Tile / fused CE memory budgets must be read from the operand's device, not GPU 0."""
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


def test_fused_ce_budget_is_measured_once_per_device(monkeypatch):
    """One host query per device, then cached: re-measuring every forward cost a query each step."""
    import torch
    ce, tiled, fake = _mock(monkeypatch, {0: 80 * GIB, 1: 8 * GIB}, current = 1)
    first = ce.get_chunk_size(1, 16_384, 151_936)
    fake.free_by_index[1] = 1 * GIB
    again = ce.get_chunk_size(1, 16_384, 151_936)
    assert again == first
    assert fake.queried == [1], fake.queried
    ce.get_chunk_size(1, 16_384, 151_936, device = torch.device("cuda", 0))
    assert fake.queried == [1, 0], fake.queried



def test_fused_ce_budget_is_read_once_per_device_across_shapes(monkeypatch):
    """New vocab sizes and fixed_gb values must reuse the device reading, not query again."""
    ce, tiled, fake = _mock(monkeypatch, {0: 80 * GIB, 1: 8 * GIB}, current = 1)
    ce.get_chunk_size(1, 16_384, 151_936)
    ce.get_chunk_size(1, 8_192, 151_936, fixed_gb = 0.5)
    ce.get_chunk_size(2, 4_096, 262_144, fixed_gb = 1.25)
    assert fake.queried == [1], fake.queried

def test_fused_ce_explicit_budget_stays_cached(monkeypatch):
    ce, _, fake = _mock(monkeypatch, {0: 80 * GIB, 1: 8 * GIB})
    a = ce.get_chunk_size(1, 16_384, 151_936, target_gb = 2.0)
    b = ce.get_chunk_size(1, 16_384, 151_936, target_gb = 2.0)
    assert a == b
    assert ce._get_explicit_chunk_multiplier.cache_info().hits >= 1
    assert fake.queried == []


@pytest.mark.parametrize("route", ["torch_func", "fused_linear"])
def test_fused_ce_budgets_the_lm_head_device(monkeypatch, route):
    ce, tiled = _modules()
    import torch
    seen = []
    sentinel = object()

    def record(device = sentinel):
        seen.append(device)
        return 4.0

    if route == "torch_func":
        monkeypatch.setattr(ce, "_CHUNKED_CE", False)
        monkeypatch.setattr(ce, "_free_target_gb", record)
        ce._get_chunk_multiplier.cache_clear()
    else:
        fl = importlib.import_module("unsloth_zoo.fused_losses.fused_linear")
        monkeypatch.setattr(tiled, "_default_target_gb", record)
        fl._budget_bytes.cache_clear()
        monkeypatch.setattr(fl, "_budget_bytes", fl._budget_bytes)
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 16, requires_grad = True)
    weight = torch.randn(64, 16, requires_grad = True)
    labels = torch.randint(0, 64, (2, 6))
    loss = ce.unsloth_fused_ce_loss(
        trainer = None, hidden_states = hidden, lm_head_weight = weight,
        lm_head_bias = None, labels = labels,
    )
    loss.backward()
    if route == "fused_linear":
        fl._budget_bytes.cache_clear()
    assert seen and all(isinstance(d, torch.device) and d == weight.device for d in seen), seen


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
    assert seen == [x.device], seen
