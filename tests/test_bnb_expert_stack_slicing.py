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

"""Quantizing an expert stack that bitsandbytes cannot count in 32 bits.

bitsandbytes' 4-bit quantize kernel indexes with a signed 32-bit int, so a
tensor of 2**31 elements or more aborts the process rather than raising. The
slicing path has to produce the same weights as a single call would, so the
tests below lower the threshold and compare the two directly instead of
allocating a 6.4e9 element stack.
"""
from pathlib import Path

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from unsloth_zoo.temporary_patches import moe_utils as MU  # noqa: E402
from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def small_threshold(monkeypatch):
    """Force slicing for a stack of a few experts, not a few billion elements."""
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", 3 * 64 * 128 + 1)
    return M


def _stack(experts=8, out=64, inp=128):
    torch.manual_seed(0)
    return torch.empty(experts, out, inp, dtype=torch.bfloat16, device="cuda").normal_()


def test_sliced_quantization_matches_one_call(small_threshold):
    value = _stack()
    sliced = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        compress_statistics=False,
    )
    whole = M.Params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        compress_statistics=False,
    ).to(value.device)

    a = bnb.functional.dequantize_4bit(sliced.data, sliced.quant_state)
    b = bnb.functional.dequantize_4bit(whole.data, whole.quant_state)
    assert tuple(sliced.quant_state.shape) == tuple(value.shape)
    # Same block boundaries and the same absmax per block, so this is equality,
    # not a tolerance.
    assert torch.equal(a.reshape(value.shape), b.reshape(value.shape))


def test_sliced_quantization_round_trips(small_threshold):
    value = _stack()
    sliced = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    back = bnb.functional.dequantize_4bit(
        sliced.data, sliced.quant_state
    ).reshape(value.shape).to(value.dtype)
    assert back.shape == value.shape
    # NF4 on unit-normal data lands around 0.073 mean absolute error whether or
    # not the call is sliced, so this only catches a mangled layout, not drift.
    assert (back.float() - value.float()).abs().mean() < 0.1


def test_small_stacks_take_the_unsliced_path(small_threshold):
    """Below the threshold nothing changes, including double quantization,
    which the sliced path deliberately turns off."""
    value = _stack(experts=1)
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        compress_statistics=True,
    )
    assert param.quant_state.nested


def test_slicing_declines_when_one_expert_is_already_too_large(monkeypatch):
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", 8)
    assert M._quantize_expert_stack_in_slices(
        _stack(experts=2), blocksize=64, quant_type="nf4",
    ) is None


def test_slicing_declines_on_a_partial_block(monkeypatch):
    """An expert that is not a whole number of blocks would put a block astride
    two slices, where byte concatenation silently rescales the tail."""
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", 2)
    value = torch.empty(4, 3, 7, dtype=torch.bfloat16, device="cuda").normal_()
    assert M._quantize_expert_stack_in_slices(
        value, blocksize=64, quant_type="nf4",
    ) is None


def test_sliced_dequant_matches_one_call(small_threshold):
    """dequantize_4bit counts in 32 bits too, and unlike quantize it runs on
    every forward, so the sliced read has to agree exactly."""
    value = _stack()
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape
    sliced = M._dequantize_4bit_in_slices(param)
    assert sliced is not None, "threshold did not force the sliced read"
    whole = bnb.functional.dequantize_4bit(
        param.data, param.quant_state
    ).reshape(value.shape)
    assert torch.equal(sliced, whole)


def test_dequant_slicing_declines_below_the_threshold():
    value = _stack(experts=2)
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape
    assert M._dequantize_4bit_in_slices(param) is None


def test_expert_dequant_helper_returns_the_logical_shape(small_threshold):
    value = _stack()
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape
    out = M._dequantize_bnb4bit_expert_weights(param, torch.bfloat16)
    assert out.shape == value.shape and out.dtype == torch.bfloat16


def test_ragged_slice_at_the_end(small_threshold):
    """Expert count not a multiple of the slice size, so the last call is short."""
    value = _stack(experts=7)
    sliced = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    assert sliced.data.numel() * 2 == value.numel()
    assert tuple(sliced.quant_state.shape) == (7, 64, 128)


@pytest.mark.parametrize(
    "quant_storage", [torch.uint8, torch.float16, torch.bfloat16, torch.float32]
)
def test_the_configured_quant_storage_survives_slicing(small_threshold, quant_storage):
    """FSDP asks for a floating point parameter storage dtype via
    bnb_4bit_quant_storage. Handing back uint8 silently breaks the wrap it
    asked for, and the packed tensor is then indexed in bytes while its
    elements are two or four bytes wide, which reads off the end of it."""
    value = _stack()
    sliced = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        quant_storage=quant_storage, compress_statistics=False,
    )
    whole = M.Params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
        quant_storage=quant_storage, compress_statistics=False,
    ).to(value.device)

    assert sliced.data.dtype == whole.data.dtype == quant_storage
    assert sliced.quant_storage == quant_storage
    assert torch.equal(
        sliced.data.reshape(-1).view(torch.uint8),
        whole.data.reshape(-1).view(torch.uint8),
    )

    sliced._original_shape = value.shape
    read = M._dequantize_4bit_in_slices(sliced)
    assert read is not None, "threshold did not force the sliced read"
    reference = bnb.functional.dequantize_4bit(
        whole.data, whole.quant_state
    ).reshape(value.shape)
    assert torch.equal(read, reference)


def test_the_forward_read_slices_too(small_threshold, monkeypatch):
    """The recompute and grouped-mm providers read the stack through
    moe_utils._get_base_weight on every forward, and again on every backward
    recomputation. That read is a single dequantize call, which aborts on the
    same element count the load-time one does."""
    from unsloth_zoo.temporary_patches import moe_utils as MU

    value = _stack()
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape

    seen = []
    real = bnb.functional.dequantize_4bit

    def _spy(data, *args, **kwargs):
        seen.append(data.numel() * data.element_size() * 2)
        return real(data, *args, **kwargs)

    monkeypatch.setattr(bnb.functional, "dequantize_4bit", _spy)
    out = MU._get_base_weight(param, torch.bfloat16)

    assert tuple(out.shape) == tuple(value.shape) and out.dtype == torch.bfloat16
    assert seen, "_get_base_weight did not dequantize at all"
    too_big = [n for n in seen if n >= M._BNB_MAX_QUANTIZE_NUMEL]
    assert not too_big, (
        f"_get_base_weight issued {len(too_big)} whole-stack dequantize call(s) of "
        f"{too_big} weights with the threshold at {M._BNB_MAX_QUANTIZE_NUMEL}; "
        f"bitsandbytes aborts the process on those."
    )


def test_get_base_weight_is_unchanged_below_the_threshold(monkeypatch):
    """No slicing, and still exactly one call, for every ordinary expert."""
    from unsloth_zoo.temporary_patches import moe_utils as MU

    value = _stack(experts=2)
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape

    seen = []
    real = bnb.functional.dequantize_4bit
    monkeypatch.setattr(
        bnb.functional, "dequantize_4bit",
        lambda d, *a, **k: (seen.append(d.numel()), real(d, *a, **k))[1],
    )
    out = MU._get_base_weight(param, torch.bfloat16)
    assert len(seen) == 1, f"expected one whole-stack call, got {len(seen)}"
    assert tuple(out.shape) == tuple(value.shape)


def test_sliced_dequant_does_not_hold_every_slice_at_once(small_threshold, monkeypatch):
    """Collecting the slices and calling torch.cat keeps every dense slice live
    alongside the concatenated result, so the peak is two full stacks. Measured
    on Inkling-Small's (256, 6144, 4096) projection that is 24.00 GiB against
    15.98 GiB written into a preallocated output, on every forward and every
    backward recomputation.

    Asserted on torch.cat not being called, because peak memory at fixture size
    is dominated by allocator reuse rather than by this.
    """
    value = _stack()
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape

    cats = []
    real = torch.cat
    monkeypatch.setattr(
        torch, "cat", lambda ts, *a, **k: (cats.append(len(ts)), real(ts, *a, **k))[1]
    )
    out = M._dequantize_4bit_in_slices(param)
    assert out is not None, "threshold did not force the sliced read"
    assert not cats, (
        f"the slices were concatenated ({cats}), which holds all of them plus "
        f"the result at once"
    )

    monkeypatch.undo()
    reference = bnb.functional.dequantize_4bit(
        param.data, param.quant_state
    ).reshape(value.shape)
    assert torch.equal(out, reference)


def test_the_forward_read_still_slices_from_the_cached_top_level_module(
    small_threshold, monkeypatch, tmp_path
):
    """compiler.py emits `from moe_utils import ...`, so unsloth_compiled_cache
    holds moe_utils as a TOP-LEVEL module with no package context. A relative
    import of a sibling raises there, and because the import is guarded the
    oversized stack would go quietly back onto the single call that aborts.

    Loaded exactly that way here. With a relative import this made one
    whole-stack call; the absolute import makes the sliced ones.
    """
    import importlib.util
    import shutil
    import sys

    source = Path(MU.__file__)
    cached = tmp_path / "moe_utils.py"
    shutil.copy(source, cached)

    spec = importlib.util.spec_from_file_location("moe_utils", cached)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "moe_utils", module)
    spec.loader.exec_module(module)
    assert module.__package__ == "", "fixture did not reproduce the no-package load"

    value = _stack()
    param = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    param._original_shape = value.shape

    seen = []
    real = bnb.functional.dequantize_4bit
    monkeypatch.setattr(
        bnb.functional, "dequantize_4bit",
        lambda d, *a, **k: (seen.append(d.numel() * d.element_size() * 2),
                            real(d, *a, **k))[1],
    )
    out = module._get_base_weight(param, torch.bfloat16)

    assert seen, "the cached module did not dequantize at all"
    too_big = [n for n in seen if n >= M._BNB_MAX_QUANTIZE_NUMEL]
    assert not too_big, (
        f"the cached top-level module issued {len(too_big)} whole-stack call(s) of "
        f"{too_big} weights; the sibling import did not resolve there"
    )
    assert tuple(out.shape) == tuple(value.shape)


@pytest.mark.parametrize("op", [torch.chunk, torch.split])
def test_a_sliced_stack_is_as_movable_and_shardable_as_an_unsliced_one(
    small_threshold, op
):
    """Params4bit.__torch_function__ rebuilds the parameter from its own
    attributes on torch.chunk and torch.split, reading .module among them, so a
    manually built stack that never defines it raises AttributeError where the
    constructor path succeeds."""
    value = _stack()
    sliced = M._make_expert_params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    )
    whole = M.Params4bit(
        value, requires_grad=False, blocksize=64, quant_type="nf4",
    ).to(value.device)

    assert sliced.module == whole.module
    for param in (whole, sliced):
        parts = op(param, 2) if op is torch.chunk else op(param, param.shape[0] // 2)
        assert len(parts) == 2
    # moving is the other thing a manually built parameter tends to lose
    assert sliced.to("cpu").data.device.type == "cpu"
    assert sliced.to(value.device).data.device.type == "cuda"
