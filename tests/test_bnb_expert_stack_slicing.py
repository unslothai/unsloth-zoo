"""Quantizing an expert stack that bitsandbytes cannot count in 32 bits.

bitsandbytes' 4-bit quantize kernel indexes with a signed 32-bit int, so a
tensor of 2**31 elements or more aborts the process rather than raising. The
slicing path has to produce the same weights as a single call would, so the
tests below lower the threshold and compare the two directly instead of
allocating a 6.4e9 element stack.
"""
import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M

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
