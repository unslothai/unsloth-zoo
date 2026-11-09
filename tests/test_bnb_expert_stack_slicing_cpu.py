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

"""Expert-stack slicing arithmetic without CUDA or bitsandbytes, against a stand-in for the 4-bit contract (whole
blocks, one absmax per block, two weights per byte); compares sliced output byte for byte with one unsliced call and
records each call's element count."""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch

# Lowered cap: the real 2**31 needs a 12 GiB bfloat16 stack, and the integer arithmetic under test is identical either way.
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
        # Double quantization nests absmax; the sliced path turns it off because nested states from separate calls do not concatenate.
        self.nested = bool(kwargs.get("nested", False))


class _Params4bit(torch.Tensor):
    """Minimal Params4bit stand-in for torch.Tensor._make_subclass; module defaults to None as the real one does, so asserting it was set cannot pass by accident."""

    module = None
    quant_state = None

    def __new__(cls, data = None, requires_grad = False, **kwargs):
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_state = kwargs.get("quant_state")
        return self


class _Recorder:
    """Records every element count handed to the kernels; bitsandbytes aborts rather than raising, so never being asked is the only assertion possible without a GPU."""

    def __init__(self):
        self.quantize = []
        self.dequantize = []

    def clear(self):
        self.quantize.clear()
        self.dequantize.clear()


RECORDER = _Recorder()
# Stands in for bitsandbytes' NF4 lookup table; only its identity matters, since the sliced path must carry one code object into the spliced QuantState.
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
    """Private copy of moe_utils_bnb4bit bound to the stand-in, loaded under its own name so reloading does not leak the
    double into other tests; package prefix keeps from .common import resolving."""
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
    """bitsandbytes aborts rather than raising, so "never asked" is the test."""
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
    """One expert per call is also correct, and hundreds of launches at load."""
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
    """Nested absmax states from separate calls do not concatenate, so the sliced path keeps them flat regardless of the caller's request."""
    param = M._quantize_expert_stack_in_slices(
        _stack(experts = 8), blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    assert param.quant_state.nested is False
    assert param.compress_statistics is False


def test_module_is_set_on_the_sliced_param(M):
    """Params4bit.__torch_function__ reads .module on torch.chunk/split."""
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
    """A block straddling two slices is rescaled by the wrong absmax when packed bytes are concatenated; 2 elements per
    expert is even, so only the block check can decline it, while an odd count would hide a dropped check."""
    value = torch.zeros(8, 1, 2, dtype = torch.bfloat16)
    per_expert = value[0].numel()
    assert per_expert % BLOCKSIZE != 0 and per_expert % 2 == 0
    assert M._quantize_expert_stack_in_slices(
        value, blocksize = BLOCKSIZE, quant_type = "nf4",
    ) is None
    assert RECORDER.quantize == [], "it declined, but only after calling the kernel"


def test_declines_an_expert_with_an_odd_element_count(M):
    """An odd element count per expert means a byte straddles two experts, so packed pieces cannot be concatenated whatever the block size."""
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
    """float32 storage holds 8 weights per element, so a slice that is not a whole number of elements cannot be concatenated."""
    # 4 per expert is a whole number of bytes (2 weights each) but not of float32 elements (8 weights each), so the slice must decline.
    value = torch.zeros(8, 2, 2, dtype = torch.bfloat16)
    assert M._quantize_expert_stack_in_slices(
        value, blocksize = 4, quant_type = "nf4", quant_storage = torch.float32,
    ) is None


def test_a_dividing_storage_dtype_is_honoured(M):
    """FSDP asks for a float storage dtype; uint8 breaks the wrap it asked for."""
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
    """A byte-computed offset starts four times too far in under float32 storage; uint8 hides this because bytes and elements coincide."""
    value = _stack(experts = 8)
    param = _quantized(M, value, quant_storage = torch.float32)
    assert param.data.element_size() == 4

    sliced = M._dequantize_4bit_in_slices(param)
    whole = _dequantize_4bit(param.data, param.quant_state)
    assert torch.equal(sliced, whole.reshape(value.shape))


def test_dequant_declines_below_the_cap(M):
    """Under the cap the single call must stay untouched."""
    value = _stack(experts = 1)
    assert value.numel() < SMALL_CAP
    param = _quantized(M, value)
    assert M._dequantize_4bit_in_slices(param) is None


def test_dequant_declines_on_a_partial_block(M):
    """1026 per expert is even, so bytes divide, and over the cap in total, but not a whole number of 64-blocks; only the block check can decline it."""
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
    """NF4 on unit-normal data lands well under 0.1 mean absolute error, so the bound catches a mangled layout rather than quantization drift."""
    value = _stack(experts = 8)
    param = _quantized(M, value)
    back = M._dequantize_4bit_in_slices(param).float()
    assert (back - value.float()).abs().mean() < 0.1


# ---------------------------------------------------------------------------
# The boundary itself, which the shipped shapes sit exactly on
# ---------------------------------------------------------------------------

class _PretendCuda(torch.Tensor):
    """CPU tensor whose is_cuda returns True, so the gate in _make_expert_params4bit runs without a GPU; only the gate
    reads it, the rest is the same integer arithmetic."""

    @property
    def is_cuda(self):
        return True


def test_the_cap_is_the_measured_bitsandbytes_limit():
    """bitsandbytes 0.50.2: quantize_4bit takes 2**31 - 64 and aborts at 2**31 (csrc/ops.cu line 74); dequantize_4bit
    takes 2146435072 and aborts at 2**31 (line 93); raising re-enables the abort, lowering slices stacks needlessly"""
    real = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    assert real._BNB_MAX_QUANTIZE_NUMEL == 2 ** 31


def test_every_threshold_comparison_is_inclusive_at_the_cap():
    """Inkling-Small's w2_weight (256, 4096, 2048) is exactly 2**31 elements, so the cap is inclusive: using > instead
    of >= at any gate leaves that stack aborting; checked over the source for future gates."""
    import ast

    real = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    tree = ast.parse(open(real.__file__).read())

    seen = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left] + list(node.comparators)
        if not any(isinstance(o, ast.Name) and o.id == "_BNB_MAX_QUANTIZE_NUMEL"
                   for o in operands):
            continue
        seen.append((node.lineno, ast.unparse(node), [type(o).__name__ for o in node.ops]))

    assert len(seen) >= 4, (
        f"expected the load, forward, merge and repack gates, found {len(seen)}: "
        f"{seen}. A rename would empty this test rather than fail it."
    )
    for lineno, text, ops in seen:
        # numel >= CAP slices at the cap and numel < CAP declines below it; > or <= would let a stack of exactly CAP reach the aborting call.
        assert ops in (["GtE"], ["Lt"]), (
            f"line {lineno}: `{text}` is exclusive at the cap, so a stack of "
            f"exactly {2 ** 31} elements reaches the bitsandbytes call that aborts"
        )


def test_a_stack_of_exactly_the_cap_is_sliced(M):
    """The load-path gate, driven end to end rather than read out of the source."""
    value = _stack(experts = 4, out = 16, inp = 64).as_subclass(_PretendCuda)
    assert value.numel() == SMALL_CAP, "this test is only about the boundary"

    param = M._make_expert_params4bit(
        value, requires_grad = False, blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    assert len(RECORDER.quantize) > 1, (
        "a stack of exactly the cap went to the unsliced path, which is the "
        "call that aborts inside bitsandbytes"
    )
    assert max(RECORDER.quantize) < SMALL_CAP
    assert tuple(param.quant_state.shape) == tuple(value.shape)


def test_a_stack_one_element_under_the_cap_is_left_alone(M):
    """The other side of the boundary: the gate must not slice stacks bitsandbytes can still count."""
    value = _stack(experts = 4, out = 16, inp = 64).as_subclass(_PretendCuda)
    M._BNB_MAX_QUANTIZE_NUMEL = value.numel() + 1
    M._make_expert_params4bit(
        value, requires_grad = False, blocksize = BLOCKSIZE, quant_type = "nf4",
    )
    assert RECORDER.quantize == [], "it was sliced despite fitting in one call"


def test_dequant_slices_a_stack_of_exactly_the_cap(M):
    """The read side has the same boundary, and runs on every forward."""
    value = _stack(experts = 4, out = 16, inp = 64)
    assert value.numel() == SMALL_CAP
    param = _quantized(M, value)

    RECORDER.clear()
    out = M._dequantize_4bit_in_slices(param)
    assert out is not None, (
        "a stack of exactly the cap was handed to the single dequantize call "
        "that aborts at ops.cu line 93"
    )
    assert len(RECORDER.dequantize) > 1
    assert max(RECORDER.dequantize) < SMALL_CAP
    assert torch.equal(out, M._dequantize_4bit_in_slices(param))


def test_a_plain_tuple_original_shape_counts_the_same(M):
    """The size check uses torch.Size.numel() when available and multiplies dims otherwise, so a plain tuple
    _original_shape must count the same; the load path always sets a torch.Size, and other tests all take that route,
    so a broken fallback would go unnoticed."""
    value = _stack(experts = 8)
    assert value.numel() > SMALL_CAP

    param = _quantized(M, value)
    param._original_shape = torch.Size(tuple(value.shape))
    as_size = M._dequantize_4bit_in_slices(param)

    param._original_shape = tuple(value.shape)      # a plain tuple, not a Size
    assert not isinstance(param._original_shape, torch.Size)
    as_tuple = M._dequantize_4bit_in_slices(param)

    assert as_tuple is not None, (
        "a stack over the cap declined because its shape was a tuple, which "
        "puts it back on the single call that aborts"
    )
    assert torch.equal(as_size, as_tuple)
