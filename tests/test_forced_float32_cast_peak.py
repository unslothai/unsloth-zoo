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

"""The forced-float32 pass must not double a multi-GiB parameter.

`module.to(dtype)` builds the destination while the source is still live, so one
parameter momentarily costs twice its size. This pass runs AFTER the device map
has placed the weights to its own budget, so the planner cannot foresee it.
gemma-4 E4B's `embed_tokens_per_layer` is [262144, 10752], 5.25 GiB, and on two
T4s holding a student and a teacher that doubling is the difference between
fitting and an OOM.

These measure the peak rather than inspecting the code, because the whole claim
is about allocator behaviour.
"""
import os

import pytest
import torch

# Bound at import, so it is the REAL constant even while the fixture below
# patches the module attribute. Only the guard at the bottom reads it.
from unsloth_zoo.patching_utils import _FORCED_FLOAT32_STAGE_BYTES

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

# The real threshold is 1 GiB, so a parameter over it plus the full extra copy
# the negative control deliberately makes needs roughly 2 GiB of free VRAM.
# That makes the suite a function of whichever card the runner happens to have,
# failing with an OOM instead of skipping. The staging branch reads a
# module-level constant at call time, so a small stand-in exercises exactly the
# same path for a few MiB.
_TEST_STAGE_BYTES = 4 * 1024 ** 2


@pytest.fixture(autouse = True)
def _small_staging_threshold(monkeypatch):
    from unsloth_zoo import patching_utils
    monkeypatch.setattr(
        patching_utils, "_FORCED_FLOAT32_STAGE_BYTES", _TEST_STAGE_BYTES,
    )
    monkeypatch.setattr(
        patching_utils, "_FORCED_FLOAT32_CHUNK_BYTES", _TEST_STAGE_BYTES // 8,
        raising = False,
    )


def _param_over_the_threshold():
    """An embedding comfortably above the patched staging threshold."""
    rows = int(_TEST_STAGE_BYTES // (2 * 1024)) + 1024
    return torch.nn.Embedding(rows, 1024).cuda().to(torch.bfloat16)


def _cast_module(module, dtype):
    # Imported lazily: it is a closure inside the patching function, so reach it
    # through a tiny module that mirrors the call the pass makes.
    from unsloth_zoo import patching_utils
    return patching_utils._stage_cast_for_test(module, dtype)


def _peak_for(module, dtype):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _cast_module(module, dtype)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


@cuda
def test_a_large_parameter_is_not_doubled():
    big = _param_over_the_threshold()
    size = big.weight.numel() * big.weight.element_size()
    peak = _peak_for(big, torch.float16)
    assert big.weight.dtype == torch.float16
    # Measured above a baseline that already contains the original, so the naive
    # path costs a full extra copy (see the negative control below) and the
    # staged path should cost almost nothing: the device original is released
    # before the replacement is allocated.
    assert peak < 0.25 * size, (
        f"staged cast needed {peak/2**30:.2f} GiB extra for a "
        f"{size/2**30:.2f} GiB parameter")


@cuda
def test_the_naive_cast_really_does_double_it():
    """Negative control. If `module.to()` did not need a full extra copy, the
    test above would pass no matter what the staged path did."""
    big = _param_over_the_threshold()
    size = big.weight.numel() * big.weight.element_size()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    big.to(torch.float16)          # the unstaged path, deliberately
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    # The baseline already holds the original, so the extra is one full copy.
    assert peak > 0.9 * size, (
        f"naive cast needed only {peak/2**30:.2f} GiB extra for a "
        f"{size/2**30:.2f} GiB parameter, so the staged test measures nothing")


@cuda
def test_a_small_parameter_still_casts():
    small = torch.nn.Linear(256, 256).cuda().to(torch.bfloat16)
    _cast_module(small, torch.float16)
    assert small.weight.dtype == torch.float16
    assert small.bias.dtype == torch.float16


@cuda
def test_a_parameter_already_in_the_target_dtype_is_left_alone():
    m = torch.nn.Linear(256, 256).cuda().to(torch.float16)
    before = m.weight.data_ptr()
    _cast_module(m, torch.float16)
    assert m.weight.dtype == torch.float16
    assert m.weight.data_ptr() == before, "a no-op cast reallocated"


@cuda
def test_the_staged_cast_preserves_values():
    big = _param_over_the_threshold()
    with torch.no_grad():
        big.weight[:4].copy_(torch.arange(4 * 1024, dtype = torch.bfloat16).reshape(4, 1024).cuda())
    expected = big.weight[:4].detach().to(torch.float16).clone()
    _cast_module(big, torch.float16)
    assert torch.equal(big.weight[:4].detach(), expected)


def _values_with_extremes(rows):
    """Includes values float16 overflows and underflows."""
    w = torch.randn(rows, 1024, dtype = torch.float32)
    w[0, :4] = torch.tensor([1e5, -1e5, 1e-9, -1e-9])
    return w.to(torch.bfloat16).cuda()


@cuda
def test_same_size_cast_rewrites_the_storage_in_place():
    big = _param_over_the_threshold()
    with torch.no_grad():
        big.weight.copy_(_values_with_extremes(big.weight.shape[0]))
    expected = big.weight.detach().to(torch.float16)
    before = big.weight.data_ptr()
    _cast_module(big, torch.float16)
    assert big.weight.dtype == torch.float16
    assert big.weight.data_ptr() == before, "no new device or host tensor should be needed"
    assert torch.equal(big.weight.detach().view(torch.int16), expected.view(torch.int16))


@cuda
def test_an_aliased_storage_is_not_rewritten(monkeypatch):
    """Rewriting bytes an alias still reads as bfloat16 would corrupt it."""
    big = _param_over_the_threshold()
    alias = big.weight.detach()
    original = alias.clone()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (0, 0))
    _cast_module(big, torch.float16)
    assert big.weight.dtype == torch.float16
    assert alias.dtype == torch.bfloat16 and torch.equal(alias, original)
    assert torch.equal(big.weight.detach(), original.to(torch.float16))


@cuda
def test_without_the_use_count_api_nothing_is_rewritten_in_place(monkeypatch):
    from unsloth_zoo import patching_utils
    monkeypatch.delattr(torch._C, "_storage_Use_Count", raising = False)
    big = _param_over_the_threshold()
    assert not patching_utils._storage_is_private(big.weight)
    expected = big.weight.detach().to(torch.float16)
    _cast_module(big, torch.float16)
    assert torch.equal(big.weight.detach(), expected)


@cuda
@pytest.mark.parametrize("src, dst", [(torch.float32, torch.float16), (torch.bfloat16, torch.float32)])
def test_a_size_changing_cast_through_the_host_is_exact(monkeypatch, src, dst):
    from unsloth_zoo import patching_utils
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (0, 0))
    staged = []
    real = patching_utils._cast_via_host_chunked
    monkeypatch.setattr(patching_utils, "_cast_via_host_chunked",
                        lambda p, d: (staged.append(d), real(p, d))[1])
    big = _param_over_the_threshold().to(src)
    expected = big.weight.detach().to(dst)
    _cast_module(big, dst)
    assert staged == [dst]
    assert big.weight.dtype == dst and big.weight.device.type == "cuda"
    assert torch.equal(big.weight.detach(), expected)


@cuda
def test_a_tied_parameter_is_cast_once_for_both_modules():
    emb = torch.nn.Embedding(int(_TEST_STAGE_BYTES // 2048) + 1024, 1024).cuda().to(torch.bfloat16)
    head = torch.nn.Linear(1024, emb.num_embeddings, bias = False).cuda()
    head.weight = emb.weight
    expected = emb.weight.detach().to(torch.float16)
    _cast_module(emb, torch.float16)
    assert head.weight is emb.weight and head.weight.dtype == torch.float16
    assert torch.equal(head.weight.detach(), expected)


def _rss_anon():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    return None


@cuda
@pytest.mark.skipif(
    not os.path.exists("/proc/self/status") or not hasattr(torch._C, "_storage_Use_Count"),
    reason = "needs /proc RSS accounting and in-place casting",
)
def test_a_large_cast_does_not_stage_through_host_memory():
    """Guards the gemma-4 E4B load being OOM-killed on a Colab T4 VM."""
    import threading, time
    big = torch.nn.Embedding(128 * 1024, 1024).cuda().to(torch.bfloat16)   # 256 MiB
    size = big.weight.numel() * big.weight.element_size()
    torch.cuda.synchronize()
    base = _rss_anon()
    peak, done = [base], threading.Event()
    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], _rss_anon()); time.sleep(0.0005)
    t = threading.Thread(target = sample); t.start()
    try:
        _cast_module(big, torch.float16)
        torch.cuda.synchronize()
        time.sleep(0.05)
    finally:
        done.set(); t.join()
    assert big.weight.dtype == torch.float16
    assert peak[0] - base < 0.25 * size, (
        f"cast used {(peak[0] - base)/2**20:.0f} MiB of host memory for a "
        f"{size/2**20:.0f} MiB parameter")


def test_the_threshold_is_above_ordinary_projections():
    """A guard on the constant: if it ever drops low enough to catch every
    Linear, the host round trip stops being rare and starts costing load time."""
    assert _FORCED_FLOAT32_STAGE_BYTES >= 256 * 1024 ** 2
