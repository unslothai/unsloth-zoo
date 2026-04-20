import pytest
import torch

from unsloth_zoo.temporary_patches.qwen3_moe import _make_qwen_moe_lora_extractor


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_extractor_preserves_dtype(dtype):
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 4, 64, 128
    wA = torch.randn(E * R, in_dim, dtype=dtype)
    wB = torch.randn(out_dim, E * R, dtype=dtype)
    first, second, _, _ = ext(None, wA, wB, 1.0, E)
    assert first.dtype == dtype
    assert second.dtype == dtype


def test_extractor_passes_tensor_scaling_unchanged():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 4, 64, 128
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    scaling = torch.tensor(0.125)
    _, _, returned, _ = ext(None, wA, wB, scaling, E)
    assert returned is scaling


def test_shared_factory_usable_by_qwen3_5_and_next():
    import unsloth_zoo.temporary_patches.qwen3_5_moe as q35
    import unsloth_zoo.temporary_patches.qwen3_next_moe as qnx
    assert q35._make_qwen_moe_lora_extractor is _make_qwen_moe_lora_extractor
    assert qnx._make_qwen_moe_lora_extractor is _make_qwen_moe_lora_extractor
    ext = _make_qwen_moe_lora_extractor()
    assert callable(ext)


def test_extractor_does_not_mutate_inputs():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 4, 64, 128
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    wA_before = wA.clone()
    wB_before = wB.clone()
    ext(None, wA, wB, 1.0, E)
    torch.testing.assert_close(wA, wA_before)
    torch.testing.assert_close(wB, wB_before)


def test_extractor_output_writable_without_aliasing_input():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 2, 2, 8, 16
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    first, second, _, _ = ext(None, wA, wB, 1.0, E)
    wA.zero_()
    wB.zero_()
    assert first.abs().sum().item() > 0
    assert second.abs().sum().item() > 0


def test_extractor_rank_1():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 1, 64, 32
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    first, second, _, _ = ext(None, wA, wB, 1.0, E)
    assert first.shape == (E, in_dim, R)
    assert second.shape == (E, R, out_dim)
