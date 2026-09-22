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

"""The expert-stack slicing arithmetic, without CUDA and without bitsandbytes.

test_bnb_expert_stack_slicing.py needs both, so on a runner with neither it
reports "skipped" for every test, and a per-OS CI leg made of nothing but skips
is green while answering nothing. What the fix actually consists of is
arithmetic -- how many experts fit under the element cap, where the packed
bytes and the absmax blocks are cut, which layouts must be declined rather than
split -- and none of that needs a GPU.

So this drives the same two functions against a bitsandbytes stand-in that
models the 4-bit contract exactly: whole blocks, one absmax per block, two
weights per byte of the storage dtype. Two properties make it a real test
rather than a re-implementation. The stand-in RECORDS the element count of
every call, so "no single call went over the cap" is measured instead of
assumed; and the sliced result is compared byte for byte against one unsliced
call, so any mistake in the splice shows up as inequality rather than as
plausible-looking numbers.

The real-hardware equivalents live in test_bnb_expert_stack_slicing.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch

# Lowered cap. The real one is 2**31, and a stack that crosses it is 12 GiB of
# bfloat16, so every test here scales the threshold down instead and exercises
# the identical integer arithmetic.
SMALL_CAP = 4096
BLOCKSIZE = 64


# ---------------------------------------------------------------------------
# A bitsandbytes stand-in that keeps the parts of the contract the fix relies on
# ---------------------------------------------------------------------------

class _QuantState:
    def __init__(self, absmax, shape, code, blocksize, quant_type, dtype, **kwargs):
        self.absmax = absmax
        self.shape = torch.Size(tuple(shape))
        self.code = code
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.dtype = dtype
        # Double quantization nests absmax; the sliced path turns it off because
        # nested states from separate calls do not concatenate.
        self.nested = bool(kwargs.get("nested", False))


class _Params4bit(torch.Tensor):
    """Enough of Params4bit for torch.Tensor._make_subclass and the attributes
    the sliced path sets. `module` defaults to None exactly as the real one
    does, so a test asserting it was set cannot pass by accident."""

    module = None
    quant_state = None

    def __new__(cls, data = None, requires_grad = False, **kwargs):
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_state = kwargs.get("quant_state")
        return self


class _Recorder:
    """Every element count handed to the quantize and dequantize kernels.

    This is the whole point: bitsandbytes does not raise on an oversized call,
    it aborts the process inside the CUDA kernel, so the only thing a test can
    assert without a GPU is that no such call is ever made.
    """

    def __init__(self):
        self.quantize = []
        self.dequantize = []

    def clear(self):
        self.quantize.clear()
        self.dequantize.clear()


RECORDER = _Recorder()
# Stands in for bitsandbytes' NF4 lookup table. Only its identity matters here:
# the sliced path must carry ONE code object through to the spliced QuantState.
_CODE = torch.arange(16, dtype = torch.float32)


def _quantize_4bit(tensor, blocksize = 64, quant_type = "nf4",
                   quant_storage = torch.uint8, compress_statistics = False, **_):
    RECORDER.quantize.append(tensor.numel())
    flat = tensor.reshape(-1).float()
    if flat.numel() % blocksize != 0:
        raise ValueError(f"{flat.numel()} is not a whole number of {blocksize}-blocks")
    blocks = flat.view(-1, blocksize)
    absmax = blocks.abs().amax(dim = 1)
    safe = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    codes = (torch.round(blocks / safe[:, None] * 7.0) + 8).clamp(0, 15)
    codes = codes.reshape(-1).to(torch.uint8)
    # Two 4-bit weights per byte, high nibble first.
    packed = ((codes[0::2] << 4) | codes[1::2]).contiguous()
    if quant_storage != torch.uint8:
        packed = packed.view(quant_storage)
    state = _QuantState(
        absmax = absmax, shape = tensor.shape, code = _CODE, blocksize = blocksize,
        quant_type = quant_type, dtype = tensor.dtype,
        nested = bool(compress_statistics),
    )
    return packed.reshape(-1, 1), state


def _dequantize_4bit(packed, quant_state = None, **_):
    state = quant_state
    data = packed.reshape(-1)
    RECORDER.dequantize.append(data.numel() * 2 * data.element_size())
    if data.dtype != torch.uint8:
        data = data.view(torch.uint8)
    nibbles = torch.stack([(data >> 4).long(), (data & 0xF).long()], dim = 1).reshape(-1)
    values = (nibbles.float() - 8.0) / 7.0
    blocks = values.view(-1, state.blocksize)
    absmax = state.absmax.float()
    if blocks.shape[0] != absmax.numel():
        raise ValueError(
            f"{blocks.shape[0]} blocks but {absmax.numel()} absmax entries: the "
            f"packed bytes and the absmax were not cut at the same boundary"
        )
    return (blocks * absmax[:, None]).reshape(state.shape).to(state.dtype)


def _install_fake_bitsandbytes(monkeypatch):
    functional = types.ModuleType("bitsandbytes.functional")
    functional.QuantState = _QuantState
    functional.quantize_4bit = _quantize_4bit
    functional.dequantize_4bit = _dequantize_4bit

    nn_module = types.ModuleType("bitsandbytes.nn")
    nn_module.Params4bit = _Params4bit

    root = types.ModuleType("bitsandbytes")
    root.functional = functional
    root.nn = nn_module
    root.__version__ = "0.0.0-test-double"

    for name, module in (("bitsandbytes", root), ("bitsandbytes.functional", functional),
                         ("bitsandbytes.nn", nn_module)):
        monkeypatch.setitem(sys.modules, name, module)
    return root


@pytest.fixture
def M(monkeypatch):
    """A private copy of moe_utils_bnb4bit bound to the stand-in.

    Loaded under its own module name rather than reloaded in place: the real
    module may already be imported and bound to the real bitsandbytes, and
    reloading it would hand a test double to every other test in the session.
    The name keeps the package prefix so the module's relative imports
    (`from .common import ...`) still resolve.
    """
    _install_fake_bitsandbytes(monkeypatch)
    real = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")

    name = "unsloth_zoo.temporary_patches._moe_utils_bnb4bit_cpu_double"
    spec = importlib.util.spec_from_file_location(name, real.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    assert module.HAS_BNB, "the stand-in was not picked up; the copy bound something else"
    assert module.Params4bit is _Params4bit
    monkeypatch.setattr(module, "_BNB_MAX_QUANTIZE_NUMEL", SMALL_CAP)
    RECORDER.clear()
    return module


def _stack(experts = 8, out = 16, inp = 64, dtype = torch.bfloat16):
    torch.manual_seed(0)
    return torch.empty(experts, out, inp, dtype = dtype).normal_()


# ---------------------------------------------------------------------------
# The property the fix exists for
# ---------------------------------------------------------------------------

def test_no_single_quantize_call_exceeds_the_cap(M):
    """bitsandbytes aborts the process rather than raising, so the only thing
    worth asserting is that it is never asked."""
    value = _stack(experts = 8)           # 8192 elements, twice SMALL_CAP
    assert value.numel() > SMALL_CAP

    param = M._quantize_expert_stack_in_slices(
        value, blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    assert param is not None, "a stack over the cap was not sliced"
    assert len(RECORDER.quantize) > 1, "it went through in one call, so nothing was sliced"
    assert max(RECORDER.quantize) < SMALL_CAP, (
        f"a slice of {max(RECORDER.quantize)} elements still crosses the "
        f"{SMALL_CAP} cap, which is the abort this change exists to avoid"
    )


def test_the_slice_is_the_largest_that_fits(M):
    """Correctness alone is satisfied by one expert per call; that would be
    hundreds of kernel launches at load. experts_per_slice must be the largest
    count whose element total stays under the cap."""
    value = _stack(experts = 8)
    per_expert = value[0].numel()
    expected_per_slice = (SMALL_CAP - 1) // per_expert
    expected_calls = -(-value.shape[0] // expected_per_slice)

    M._quantize_expert_stack_in_slices(value, blocksize = BLOCKSIZE, quant_type = "nf4")
    assert len(RECORDER.quantize) == expected_calls
    assert RECORDER.quantize[0] == expected_per_slice * per_expert


def test_sliced_quantization_is_byte_identical_to_one_call(M):
    """The splice is exact or it is wrong; there is no tolerance to hide in."""
    value = _stack(experts = 8)
    sliced = M._quantize_expert_stack_in_slices(
        value, blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    whole, whole_state = _quantize_4bit(
        value, blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    assert torch.equal(sliced.data.reshape(-1), whole.reshape(-1))
    assert torch.equal(sliced.quant_state.absmax, whole_state.absmax)
    assert tuple(sliced.quant_state.shape) == tuple(value.shape)
    assert sliced.quant_state.code is _CODE
    assert sliced.quant_state.dtype == value.dtype


def test_sliced_quantization_does_not_nest_absmax(M):
    """Double quantization nests absmax per call and nested states from
    separate calls do not concatenate, so the sliced path must keep them flat
    whatever the caller asked for."""
    param = M._quantize_expert_stack_in_slices(
        _stack(experts = 8), blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    assert param.quant_state.nested is False
    assert param.compress_statistics is False


def test_module_is_set_on_the_sliced_param(M):
    """Params4bit.__torch_function__ reads .module on torch.chunk/torch.split,
    so an unset one makes a sliced stack raise where an unsliced one shards."""
    owner = object()
    param = M._quantize_expert_stack_in_slices(
        _stack(experts = 8), blocksize = BLOCKSIZE, quant_type = "nf4", module = owner,
    )
    assert param.module is owner


# ---------------------------------------------------------------------------
# Declines: one test per condition that must fall back rather than split
# ---------------------------------------------------------------------------

def test_declines_a_one_dimensional_tensor(M):
    assert M._quantize_expert_stack_in_slices(
        torch.zeros(SMALL_CAP * 2, dtype = torch.bfloat16),
        blocksize = BLOCKSIZE, quant_type = "nf4",
    ) is None


def test_declines_an_expert_that_is_not_whole_blocks(M):
    """A block astride two slices would be rescaled by the wrong absmax when
    the packed bytes are concatenated.

    2 elements per expert, deliberately: it is an even number, so it satisfies
    the two-weights-per-byte condition below and the ONLY thing that can
    decline it is the block check. The obvious choice of an odd per-expert
    count declines either way, so it passes whether or not this condition
    exists, which is how a dropped block check went unnoticed.
    """
    value = torch.zeros(8, 1, 2, dtype = torch.bfloat16)
    per_expert = value[0].numel()
    assert per_expert % BLOCKSIZE != 0 and per_expert % 2 == 0
    assert M._quantize_expert_stack_in_slices(
        value, blocksize = BLOCKSIZE, quant_type = "nf4",
    ) is None
    assert RECORDER.quantize == [], "it declined, but only after calling the kernel"


def test_declines_an_expert_with_an_odd_element_count(M):
    """Odd means a byte straddles two experts, so the packed pieces cannot be
    concatenated whatever the block size."""
    value = torch.zeros(8, 3, 7, dtype = torch.bfloat16)      # 21 per expert
    assert value[0].numel() % 2 == 1
    assert M._quantize_expert_stack_in_slices(
        value, blocksize = 1, quant_type = "nf4",
    ) is None
    assert RECORDER.quantize == []


def test_declines_when_one_expert_alone_is_over_the_cap(M, monkeypatch):
    """Slicing by expert cannot help, so the caller keeps the single call."""
    monkeypatch.setattr(M, "_BNB_MAX_QUANTIZE_NUMEL", BLOCKSIZE)
    assert M._quantize_expert_stack_in_slices(
        _stack(experts = 4), blocksize = BLOCKSIZE, quant_type = "nf4",
    ) is None


def test_declines_a_storage_dtype_the_slice_does_not_divide(M):
    """quant_storage may be float32, which is 8 weights per element. A slice
    that is not a whole number of storage elements cannot be concatenated."""
    # 4 per expert: a whole number of bytes (2 weights each) but not of float32
    # elements (8 weights each).
    value = torch.zeros(8, 2, 2, dtype = torch.bfloat16)
    assert M._quantize_expert_stack_in_slices(
        value, blocksize = 4, quant_type = "nf4", quant_storage = torch.float32,
    ) is None


def test_a_dividing_storage_dtype_is_honoured(M):
    """FSDP asks for a floating point parameter storage dtype and silently
    handing back uint8 breaks the wrap it asked for."""
    value = _stack(experts = 8)
    param = M._quantize_expert_stack_in_slices(
        value, blocksize = BLOCKSIZE, quant_type = "nf4", quant_storage = torch.float32,
    )
    assert param is not None
    assert param.data.dtype == torch.float32
    assert param.quant_storage == torch.float32


# ---------------------------------------------------------------------------
# The read side, which runs on every forward rather than once at load
# ---------------------------------------------------------------------------

def _quantized(M, value, quant_storage = torch.uint8):
    param = M._quantize_expert_stack_in_slices(
        value, blocksize = BLOCKSIZE, quant_type = "nf4", quant_storage = quant_storage,
    )
    param._original_shape = value.shape
    return param


def test_sliced_dequant_matches_one_call(M):
    value = _stack(experts = 8)
    param = _quantized(M, value)
    RECORDER.clear()

    sliced = M._dequantize_4bit_in_slices(param)
    assert sliced is not None, "a stack over the cap was read in one call"
    assert len(RECORDER.dequantize) > 1
    whole = _dequantize_4bit(param.data, param.quant_state)
    assert torch.equal(sliced, whole.reshape(value.shape))


def test_no_single_dequantize_call_exceeds_the_cap(M):
    param = _quantized(M, _stack(experts = 8))
    RECORDER.clear()
    M._dequantize_4bit_in_slices(param)
    assert max(RECORDER.dequantize) < SMALL_CAP


def test_dequant_slices_in_storage_elements_not_bytes(M):
    """The trap this guards: with a float32 quant_storage one packed element
    covers eight weights, so a slice offset computed in bytes starts four times
    too far in and every slice after the first reads the wrong data. uint8
    hides it because the two units coincide."""
    value = _stack(experts = 8)
    param = _quantized(M, value, quant_storage = torch.float32)
    assert param.data.element_size() == 4

    sliced = M._dequantize_4bit_in_slices(param)
    whole = _dequantize_4bit(param.data, param.quant_state)
    assert torch.equal(sliced, whole.reshape(value.shape))


def test_dequant_declines_below_the_cap(M):
    """Under the cap the single call is untouched, so the sliced reader must
    hand the decision back rather than take a path nothing needs."""
    value = _stack(experts = 1)
    assert value.numel() < SMALL_CAP
    param = _quantized(M, value)
    assert M._dequantize_4bit_in_slices(param) is None


def test_dequant_declines_on_a_partial_block(M):
    """1026 per expert: over the cap in total, an even number so the packed
    bytes would divide, and not a whole number of 64-blocks. Only the block
    check can decline it, which is what makes this test about that check."""
    param = _quantized(M, _stack(experts = 8))
    param._original_shape = torch.Size((8, 1, 1026))
    per_expert = 1026
    assert 8 * per_expert > SMALL_CAP
    assert per_expert % 2 == 0 and per_expert % BLOCKSIZE != 0
    assert M._dequantize_4bit_in_slices(param) is None
    assert RECORDER.dequantize == [], "it declined, but only after reading"


def test_dequant_preserves_the_logical_shape_and_dtype(M):
    value = _stack(experts = 8)
    param = _quantized(M, value)
    sliced = M._dequantize_4bit_in_slices(param)
    assert tuple(sliced.shape) == tuple(value.shape)
    assert sliced.dtype == value.dtype
    assert sliced.device == value.device


def test_round_trip_through_both_sliced_paths(M):
    """Quantize sliced, read back sliced. NF4 on unit-normal data lands well
    under 0.1 mean absolute error, so this catches a mangled layout rather than
    quantization drift."""
    value = _stack(experts = 8)
    param = _quantized(M, value)
    back = M._dequantize_4bit_in_slices(param).float()
    assert (back - value.float()).abs().mean() < 0.1
