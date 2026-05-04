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


class _SquareWrapper:
    """Wrapper for the input_dim == output_dim ambiguity test.

    Sets `_did_swap_in_out_features` to mimic what PEFT 0.19 sets after the
    swap, so the extractor can disambiguate.
    """
    def __init__(self, parameter_name, dim, did_swap):
        self.parameter_name = parameter_name
        self.base_layer = type("_Base", (), {})()
        if parameter_name == "down_proj":
            # down_proj: (E, hidden_dim, intermediate_dim). Force in == out by
            # setting intermediate_dim == hidden_dim.
            self.base_layer.intermediate_dim = dim
            self.base_layer.hidden_dim = dim
        elif parameter_name == "gate_up_proj":
            # gate_up_proj: input_dim = hidden_dim, output_dim = 2*intermediate_dim.
            # Make square by intermediate_dim = hidden_dim/2.
            self.base_layer.hidden_dim = dim
            self.base_layer.intermediate_dim = dim // 2
        self._did_swap_in_out_features = did_swap

    def get_base_layer(self):
        return self.base_layer


@pytest.mark.parametrize("did_swap", [False, True], ids=["peft018", "peft019_swapped"])
def test_extractor_disambiguates_square_dims_via_did_swap(did_swap):
    """When input_dim == output_dim both branch predicates match. The extractor
    must use `_did_swap_in_out_features` to pick the correct branch.
    """
    ext = _make_qwen_moe_lora_extractor()
    E, R, dim = 4, 3, 16
    torch.manual_seed(7)
    wA = torch.randn(E * R, dim)
    wB = torch.randn(dim, E * R)
    wrapper = _SquareWrapper("down_proj", dim, did_swap=did_swap)
    first, second, scaling, num_experts = ext(wrapper, wA, wB, 1.0, E)
    assert first.shape == (E, dim, R)
    assert second.shape == (E, R, dim)

    # Reference: under PEFT 0.18 the reshape is the canonical permutation;
    # under PEFT 0.19 swapped, weight_A and weight_B roles are flipped.
    x = torch.randn(5, dim)
    for e in range(E):
        Ae = wA[e * R : (e + 1) * R]
        Be = wB[:, e * R : (e + 1) * R]
        if did_swap:
            naive = x @ Be @ Ae   # PEFT 0.19 reversed
        else:
            naive = x @ Ae.T @ Be.T  # PEFT 0.18 canonical
        via = (x @ first[e]) @ second[e]
        torch.testing.assert_close(via, naive, atol=1e-4, rtol=1e-4)


def test_extractor_fallback_warns_when_dims_mismatch(caplog):
    """When neither branch matches, the extractor must warn before falling
    through to the canonical permutation. Guards against the silent PR #601
    failure mode resurfacing.
    """
    import logging

    ext = _make_qwen_moe_lora_extractor()
    E, R, in_dim, out_dim = 4, 3, 16, 24
    torch.manual_seed(11)

    # Build a wrapper that reports plausible dims but supplies LoRA factors
    # whose shapes match neither branch (simulates a future PEFT layout).
    wrapper = _Wrapper("gate_up_proj", (E, 2 * 16, in_dim))
    weird_dim = 99
    wA = torch.randn(E * R, weird_dim)
    wB = torch.randn(weird_dim, E * R)

    # The warning is gated on UNSLOTH_ENABLE_LOGGING; force-enable for the test.
    import unsloth_zoo.temporary_patches.qwen3_moe as qwen3_moe_mod
    prev = qwen3_moe_mod.UNSLOTH_ENABLE_LOGGING
    try:
        qwen3_moe_mod.UNSLOTH_ENABLE_LOGGING = True
        with caplog.at_level(logging.WARNING):
            first, second, *_ = ext(wrapper, wA, wB, 1.0, E)
    finally:
        qwen3_moe_mod.UNSLOTH_ENABLE_LOGGING = prev

    assert first.shape == (E, weird_dim, R)
    assert second.shape == (E, R, weird_dim)
    # At least one warning record mentions the extractor.
    assert any(
        "Qwen MoE LoRA extractor could not match either layout" in rec.getMessage()
        for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]


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
