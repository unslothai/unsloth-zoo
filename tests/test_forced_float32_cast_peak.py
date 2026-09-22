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


def test_the_threshold_is_above_ordinary_projections():
    """A guard on the constant: if it ever drops low enough to catch every
    Linear, the host round trip stops being rare and starts costing load time."""
    assert _FORCED_FLOAT32_STAGE_BYTES >= 256 * 1024 ** 2
