import pytest
import torch

from unsloth_zoo.temporary_patches.qwen3_moe import _make_qwen_moe_lora_extractor


class _Wrapper:
    def __init__(self, parameter_name, shape):
        self.parameter_name = parameter_name
        self.base_layer = type("_Base", (), {})()
        if parameter_name == "gate_up_proj":
            self.base_layer.hidden_dim = shape[2]
            self.base_layer.intermediate_dim = shape[1] // 2
        elif parameter_name == "down_proj":
            self.base_layer.hidden_dim = shape[1]
            self.base_layer.intermediate_dim = shape[2]

    def get_base_layer(self):
        return self.base_layer


def _make_layout_inputs(layout, E, R, in_dim, out_dim):
    if layout == "canonical":
        weight_A = torch.randn(E * R, in_dim)
        weight_B = torch.randn(out_dim, E * R)
    elif layout == "reversed":
        weight_A = torch.randn(E * R, out_dim)
        weight_B = torch.randn(in_dim, E * R)
    else:
        raise AssertionError(f"unknown layout: {layout}")
    return weight_A, weight_B


def _assert_layout_equivalence(layout, first, second, weight_A, weight_B, E, R, in_dim):
    x = torch.randn(6, in_dim)
    for e in range(E):
        Ae = weight_A[e * R : (e + 1) * R]
        Be = weight_B[:, e * R : (e + 1) * R]
        if layout == "canonical":
            naive = x @ Ae.T @ Be.T
        else:
            naive = x @ Be @ Ae
        via = (x @ first[e]) @ second[e]
        torch.testing.assert_close(via, naive, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "E,R,in_dim,out_dim",
    [
        (4, 4, 256, 128),
        (4, 4, 128, 256),
        (8, 2, 512, 1024),
        (16, 8, 2048, 512),
        (2, 1, 64, 32),
    ],
)
def test_extractor_shapes(E, R, in_dim, out_dim):
    ext = _make_qwen_moe_lora_extractor()
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    first, second, scaling, num_experts = ext(None, wA, wB, 2.5, E)
    assert first.shape == (E, in_dim, R)
    assert second.shape == (E, R, out_dim)
    assert scaling == 2.5
    assert num_experts == E
    assert first.is_contiguous()
    assert second.is_contiguous()


@pytest.mark.parametrize(
    "parameter_name,base_shape,in_dim,out_dim,layout",
    [
        ("gate_up_proj", (4, 48, 16), 16, 48, "canonical"),
        ("gate_up_proj", (4, 48, 16), 16, 48, "reversed"),
        ("down_proj", (4, 16, 24), 24, 16, "canonical"),
        ("down_proj", (4, 16, 24), 24, 16, "reversed"),
    ],
)
def test_extractor_qwen_moe_layout(parameter_name, base_shape, in_dim, out_dim, layout):
    ext = _make_qwen_moe_lora_extractor()
    E, R = base_shape[0], 3
    torch.manual_seed(0)
    wrapper = _Wrapper(parameter_name, base_shape)
    wA, wB = _make_layout_inputs(layout, E, R, in_dim, out_dim)
    first, second, scaling, num_experts = ext(wrapper, wA, wB, 2.5, E)
    assert first.shape == (E, in_dim, R)
    assert second.shape == (E, R, out_dim)
    assert scaling == 2.5
    assert num_experts == E
    assert first.is_contiguous()
    assert second.is_contiguous()
    _assert_layout_equivalence(layout, first, second, wA, wB, E, R, in_dim)


def test_extractor_fallback_numerical_equivalence_per_expert():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 4, 64, 128
    torch.manual_seed(0)
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    first, second, _, _ = ext(None, wA, wB, 1.0, E)
    x = torch.randn(6, in_dim)
    for e in range(E):
        Ae = wA[e * R : (e + 1) * R]
        Be = wB[:, e * R : (e + 1) * R]
        naive = x @ Ae.T @ Be.T
        via = (x @ first[e]) @ second[e]
        torch.testing.assert_close(via, naive, atol=1e-4, rtol=1e-4)


def test_extractor_ignores_wrapper_attributes():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 4, 64, 128
    torch.manual_seed(2)
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)

    class _Bogus:
        parameter_name = "down_proj"
        base_layer = None

    first_none, second_none, _, _ = ext(None, wA, wB, 1.0, E)
    first_bogus, second_bogus, _, _ = ext(_Bogus(), wA, wB, 1.0, E)
    torch.testing.assert_close(first_none, first_bogus)
    torch.testing.assert_close(second_none, second_bogus)


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


def test_shared_factory_used_by_qwen3_5_and_next():
    import unsloth_zoo.temporary_patches.qwen3_5_moe as q35
    import unsloth_zoo.temporary_patches.qwen3_next_moe as qnx
    assert q35._make_qwen_moe_lora_extractor is _make_qwen_moe_lora_extractor
    assert qnx._make_qwen_moe_lora_extractor is _make_qwen_moe_lora_extractor


def test_extractor_output_not_aliasing_input():
    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 2, 2, 8, 16
    wA = torch.randn(E * R, in_dim)
    wB = torch.randn(out_dim, E * R)
    first, second, _, _ = ext(None, wA, wB, 1.0, E)
    wA.zero_()
    wB.zero_()
    assert first.abs().sum().item() > 0
    assert second.abs().sum().item() > 0
