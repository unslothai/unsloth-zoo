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

"""bitsandbytes Optimizer8bit.step without the per-parameter device sync (non-paged only)."""

import inspect

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from unsloth_zoo.temporary_patches import bitsandbytes_optimizer_sync as S

KILL = S._KILL_SWITCH


def test_installed_step_shape_is_known():
    import bitsandbytes.optim.optimizer as m
    if not S._bnb_version_in_window():
        pytest.skip(f"bitsandbytes {bnb.__version__} outside the patch window")
    step = m.Optimizer8bit.step
    step = getattr(step, "_unsloth_original_step", step)
    shape = S._step_shape(inspect.getsource(step))
    assert shape in (S._step_shape(S._STEP_TORCH_SYNC), S._step_shape(S._STEP_SYNC_GPU))


def test_unknown_shape_is_rejected():
    changed = S._STEP_SYNC_GPU.replace("self.prefetch_state(p)\n", "self.prefetch_state(p)\n            sync_gpu(p)\n")
    assert S._step_shape(changed) != S._step_shape(S._STEP_SYNC_GPU)


if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)


@pytest.fixture(scope = "module", autouse = True)
def _patched():
    S.patch_bitsandbytes_optimizer_step_sync()
    import bitsandbytes.optim.optimizer as m
    if not getattr(m.Optimizer8bit.step, S._PATCHED_FLAG, False):
        if S._bnb_version_in_window():
            pytest.fail(f"patch not applied on bitsandbytes {bnb.__version__}")
        pytest.skip("bitsandbytes outside the patch window")
    yield


OPTIMIZERS = {
    "AdamW8bit":          lambda ps: bnb.optim.AdamW8bit(ps, lr = 1e-3, weight_decay = 0.01),
    # What transformers builds for optim="adamw_8bit" / "adamw_bnb_8bit".
    "AdamW_optim_bits_8": lambda ps: bnb.optim.AdamW(ps, lr = 1e-3, optim_bits = 8, is_paged = False),
    "AdamW32bit":         lambda ps: bnb.optim.AdamW32bit(ps, lr = 1e-3, weight_decay = 0.01),
    "Lion8bit":           lambda ps: bnb.optim.Lion8bit(ps, lr = 1e-4),
    "PagedAdamW8bit":     lambda ps: bnb.optim.PagedAdamW8bit(ps, lr = 1e-3, weight_decay = 0.01),
    "PagedAdamW32bit":    lambda ps: bnb.optim.PagedAdamW32bit(ps, lr = 1e-3, weight_decay = 0.01),
}
PAGED = {"PagedAdamW8bit", "PagedAdamW32bit"}
# Below min_8bit_size (4096) keeps 32-bit state, above 1e5 elements gets paged buffers.
SHAPES = [(64,), (32, 48), (256, 512), (512, 512), (1024, 160)]


def _params(dtype):
    g = torch.Generator(device = "cuda").manual_seed(0)
    return [
        torch.nn.Parameter(torch.randn(s, device = "cuda", dtype = dtype, generator = g) * 0.02)
        for s in SHAPES
    ]


def _run(name, dtype, steps, monkeypatch, kill):
    if kill: monkeypatch.setenv(KILL, "1")
    else:    monkeypatch.delenv(KILL, raising = False)
    params = _params(dtype)
    opt = OPTIMIZERS[name](params)
    g = torch.Generator(device = "cuda").manual_seed(1)
    for i in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, device = "cuda", dtype = dtype, generator = g)
        if i % 7 == 3:
            params[1].grad = None  # skipped parameter path
        opt.step()
    torch.cuda.synchronize()
    states = [
        {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state[p].items()}
        for p in params
    ]
    return [p.detach().clone() for p in params], states


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_bitwise_identical_50_steps(name, dtype, monkeypatch):
    p0, s0 = _run(name, dtype, 50, monkeypatch, kill = True)
    p1, s1 = _run(name, dtype, 50, monkeypatch, kill = False)
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    for a, b in zip(s0, s1):
        assert a.keys() == b.keys()
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k]), k
            else:
                assert a[k] == b[k], k


def _count_syncs(monkeypatch, name, kill, stream = None):
    if kill: monkeypatch.setenv(KILL, "1")
    else:    monkeypatch.delenv(KILL, raising = False)
    params = _params(torch.bfloat16)
    opt = OPTIMIZERS[name](params)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()  # state init outside the count
    torch.cuda.synchronize()
    calls = []
    real = torch.cuda.synchronize
    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(torch.cuda, "synchronize", counting)
    try:
        if stream is None:
            opt.step()
        else:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                opt.step()
            torch.cuda.current_stream().wait_stream(stream)
    finally:
        monkeypatch.setattr(torch.cuda, "synchronize", real)
    real()
    return len(calls)


@pytest.mark.parametrize("name", [n for n in OPTIMIZERS if n not in PAGED])
def test_non_paged_step_has_no_device_sync(name, monkeypatch):
    assert _count_syncs(monkeypatch, name, kill = True) == len(SHAPES)
    assert _count_syncs(monkeypatch, name, kill = False) == 0


@pytest.mark.parametrize("name", sorted(PAGED))
def test_paged_step_keeps_every_sync(name, monkeypatch):
    base = _count_syncs(monkeypatch, name, kill = True)
    assert base >= len(SHAPES) + 1
    assert _count_syncs(monkeypatch, name, kill = False) == base


def test_side_stream_keeps_per_param_sync(monkeypatch):
    s = torch.cuda.Stream()
    assert _count_syncs(monkeypatch, "AdamW8bit", kill = False, stream = s) == len(SHAPES)


def test_idempotent():
    import bitsandbytes.optim.optimizer as m
    before = m.Optimizer8bit.step
    S.patch_bitsandbytes_optimizer_step_sync()
    assert m.Optimizer8bit.step is before
    assert not getattr(before._unsloth_original_step, S._PATCHED_FLAG, False)
