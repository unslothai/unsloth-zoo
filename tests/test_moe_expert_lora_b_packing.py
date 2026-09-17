"""A fused MoE expert lora_B must be read the way PEFT packed it.

PEFT stores a fused expert LoRA as two ordinary Linear weights and does NOT flatten them
the same way: `lora_A` is `(num_experts * rank, in)` with the expert index slowest, while
`lora_B` is `(out, num_experts * rank)` with the expert index FASTEST. Unsloth's separated
MoE forward used to read `lora_B` expert-slowest, so every expert was paired with the
wrong rank columns. Both readings produce identically shaped tensors and neither ever
raises, so the failure was silent: the delta's norm stayed right while its values were
scrambled.

These tests pin the reading against PEFT itself, and against the two degenerate cases
where the readings coincide, so the fixture cannot go vacuous.
"""
import itertools
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches.moe_utils import (  # noqa: E402
    LORA_B_LAYOUT_GROUPED_BY_EXPERT,
    LORA_B_LAYOUT_RANK_MAJOR,
    _canonical_lora_weights_for_grouped_mm,
    _reversed_lora_weights_for_grouped_mm,
    moe_lora_b_expert_columns,
    moe_lora_b_layout,
    unflatten_moe_lora_b,
)

gpu_available = (
    (hasattr(torch, "cuda") and torch.cuda.is_available())
    or (hasattr(torch, "xpu") and torch.xpu.is_available())
)


def _factors(num_experts, rank, dim_in, dim_out, seed = 0):
    g = torch.Generator().manual_seed(seed)
    weight_A = torch.randn(num_experts * rank, dim_in, generator = g, dtype = torch.float64)
    weight_B = torch.randn(dim_out, num_experts * rank, generator = g, dtype = torch.float64)
    return weight_A, weight_B


def _peft_lora_b(weight_B, num_experts):
    """peft/tuners/lora/layer.py ParamWrapper.get_delta_factors, verbatim."""
    return weight_B.reshape(weight_B.shape[0], -1, num_experts).permute(2, 0, 1)


def _vllm_lora_b(weight_B, num_experts):
    """vllm/lora/model_manager.py _stack_moe_lora_weights, verbatim."""
    lora_b = weight_B.reshape(weight_B.shape[0], -1, num_experts)
    return lora_b.permute(2, 0, 1)


@pytest.mark.parametrize("num_experts, rank", [(4, 6), (6, 6), (8, 4), (256, 8)])
def test_rank_major_is_bitwise_peft(num_experts, rank):
    """The default reading is PEFT's, bit for bit. A canary: if a PEFT release ever
    changes its packing this fails, and the default here has to follow rather than be
    left inverting a correct reconstruction."""
    _, weight_B = _factors(num_experts, rank, 16, 12)
    ours = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0],
        layout = LORA_B_LAYOUT_RANK_MAJOR,
    )
    assert torch.equal(ours, _peft_lora_b(weight_B, num_experts))


@pytest.mark.parametrize("num_experts, rank", [(4, 6), (256, 8)])
def test_rank_major_is_bitwise_vllm(num_experts, rank):
    """vLLM serves the adapter with its own copy of the same reshape. Pinning it here is
    what says a saved adapter is served as the function it was trained as."""
    _, weight_B = _factors(num_experts, rank, 16, 12)
    ours = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0],
        layout = LORA_B_LAYOUT_RANK_MAJOR,
    )
    assert torch.equal(ours, _vllm_lora_b(weight_B, num_experts))


@pytest.mark.parametrize("num_experts, rank", [(4, 6), (6, 6), (2, 2), (8, 4)])
def test_the_two_layouts_really_differ(num_experts, rank):
    """Without this the tests above could pass against a fixture where the two readings
    happen to coincide, and would then prove nothing."""
    _, weight_B = _factors(num_experts, rank, 16, 12)
    rank_major = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0], layout = LORA_B_LAYOUT_RANK_MAJOR,
    )
    grouped = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0],
        layout = LORA_B_LAYOUT_GROUPED_BY_EXPERT,
    )
    assert not torch.equal(rank_major, grouped)


@pytest.mark.parametrize("num_experts, rank", [(1, 8), (4, 1), (1, 1)])
def test_the_two_layouts_coincide_only_when_degenerate(num_experts, rank):
    """One expert, or one rank column per expert, leaves nothing to pack. These are the
    only cases where the old reading was accidentally right, which is why a smoke test on
    a toy model could never have caught this."""
    _, weight_B = _factors(num_experts, rank, 16, 12)
    rank_major = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0], layout = LORA_B_LAYOUT_RANK_MAJOR,
    )
    grouped = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0],
        layout = LORA_B_LAYOUT_GROUPED_BY_EXPERT,
    )
    assert torch.equal(rank_major, grouped)


def test_neither_layout_ever_raises():
    """Both readings are shape compatible for every (num_experts, rank), because
    num_experts * rank == rank * num_experts. So the wrong one cannot be caught by a
    shape check anywhere, and this is documented rather than assumed."""
    for num_experts, rank in itertools.product(range(1, 9), range(1, 9)):
        _, weight_B = _factors(num_experts, rank, 16, 12)
        for layout in (LORA_B_LAYOUT_RANK_MAJOR, LORA_B_LAYOUT_GROUPED_BY_EXPERT):
            got = unflatten_moe_lora_b(
                weight_B, num_experts, rank, weight_B.shape[0], layout = layout,
            )
            assert got.shape == (num_experts, 12, rank)


@pytest.mark.parametrize(
    "layout", [LORA_B_LAYOUT_RANK_MAJOR, LORA_B_LAYOUT_GROUPED_BY_EXPERT],
)
@pytest.mark.parametrize("num_experts, rank", [(4, 6), (6, 6), (8, 4)])
def test_expert_columns_agree_with_the_unflatten(layout, num_experts, rank):
    """The merge paths walk one expert at a time and the forward reshapes the whole
    tensor. They have to select the same columns or a merged checkpoint stops being the
    model the adapter was."""
    _, weight_B = _factors(num_experts, rank, 16, 12)
    stacked = unflatten_moe_lora_b(
        weight_B, num_experts, rank, weight_B.shape[0], layout = layout,
    )
    for expert_idx in range(num_experts):
        columns = moe_lora_b_expert_columns(expert_idx, num_experts, rank, layout = layout)
        assert torch.equal(weight_B[:, columns], stacked[expert_idx])


def test_default_layout_and_the_legacy_override(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising = False)
    assert moe_lora_b_layout() == LORA_B_LAYOUT_RANK_MAJOR

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", LORA_B_LAYOUT_GROUPED_BY_EXPERT)
    assert moe_lora_b_layout() == LORA_B_LAYOUT_GROUPED_BY_EXPERT

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", "expert_major")
    with pytest.raises(ValueError):
        moe_lora_b_layout()


@pytest.mark.parametrize("num_experts, rank", [(4, 6), (8, 4)])
def test_grouped_mm_factors_reconstruct_peft_delta(num_experts, rank):
    """The end the whole change exists for: the delta the separated forward applies is
    the delta PEFT's own merge would produce. Checked in float64 so the residual is not
    rounding."""
    dim_in, dim_out = 16, 12
    weight_A, weight_B = _factors(num_experts, rank, dim_in, dim_out)

    first, second = _canonical_lora_weights_for_grouped_mm(
        weight_A, weight_B, num_experts, rank, dim_in, dim_out,
    )
    # X @ first @ second, so the per-expert delta is (in, out); PEFT's einsum below
    # produces the same orientation.
    ours = torch.bmm(first, second)

    peft_A = weight_A.reshape(num_experts, rank, dim_in)
    peft_B = weight_B.reshape(dim_out, rank, num_experts)
    peft = torch.einsum("o r e, e r i -> e i o", peft_B, peft_A)
    assert torch.allclose(ours, peft, rtol = 0, atol = 1e-12)


@pytest.mark.parametrize("num_experts, rank", [(4, 6), (8, 4)])
def test_reversed_factors_reconstruct_peft_delta(num_experts, rank):
    """The transposed-parameter families take the other branch, and it moved too."""
    dim_in, dim_out = 16, 12
    weight_A, weight_B = _factors(num_experts, rank, dim_in, dim_out)

    first, second = _reversed_lora_weights_for_grouped_mm(
        weight_A, weight_B, num_experts, rank, dim_in, dim_out,
    )
    ours = torch.bmm(first, second)

    peft_A = weight_A.reshape(num_experts, rank, dim_in)
    peft_B = weight_B.reshape(dim_out, rank, num_experts).permute(2, 0, 1)
    assert torch.allclose(ours, torch.bmm(peft_B, peft_A), rtol = 0, atol = 1e-12)


@pytest.mark.parametrize(
    "layout", [LORA_B_LAYOUT_RANK_MAJOR, LORA_B_LAYOUT_GROUPED_BY_EXPERT],
)
def test_fused_expert_merge_loop_matches_the_forward(layout, monkeypatch):
    """saving_utils' 4-bit and fused expert merge slices one expert at a time. It has to
    select what the forward reshapes, under whichever layout is in force."""
    from unsloth_zoo.saving_utils import _apply_fused_expert_lora_delta

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", layout)
    num_experts, rank, dim_in, dim_out = 4, 6, 16, 12
    weight_A, weight_B = _factors(num_experts, rank, dim_in, dim_out)
    weight_A, weight_B = weight_A.float(), weight_B.float()

    expected = torch.bmm(
        unflatten_moe_lora_b(weight_B, num_experts, rank, dim_out, layout = layout),
        weight_A.reshape(num_experts, rank, dim_in),
    ) * 2.0

    merged = torch.zeros(num_experts, dim_out, dim_in)
    _apply_fused_expert_lora_delta(
        merged, weight_A, weight_B, num_experts, rank, dim_in, dim_out,
        alpha = 2.0, use_transpose = False,
    )
    assert torch.allclose(merged, expected, rtol = 0, atol = 1e-5)


@pytest.mark.skipif(not gpu_available, reason = "needs an accelerator for torch._grouped_mm")
def test_separated_forward_matches_stock_peft_forward():
    """The whole chain, on a real transformers experts module: Unsloth's separated MoE
    forward and PEFT's own forward must compute the same function from the same saved
    factors. They did not before this change, by more than the adapter's entire effect."""
    import types

    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import ParamWrapper
    from unsloth_zoo.temporary_patches import moe_utils

    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeConfig, Qwen3MoeSparseMoeBlock,
        )
    except ImportError:
        pytest.skip("this transformers has no stacked Qwen3MoeExperts")

    device = "cuda" if torch.cuda.is_available() else "xpu"
    num_experts, hidden, inter, rank = 8, 128, 48, 4
    config = Qwen3MoeConfig(
        hidden_size = hidden, intermediate_size = 4 * hidden, moe_intermediate_size = inter,
        num_experts = num_experts, num_experts_per_tok = 2, num_hidden_layers = 1,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 64,
    )
    # Built from a config rather than from_pretrained, so the experts dispatcher has no
    # implementation resolved and would otherwise fail to dispatch at all.
    config._experts_implementation = "eager"
    torch.manual_seed(0)
    block = Qwen3MoeSparseMoeBlock(config).to(device, torch.float32)
    for parameter in block.parameters():
        if parameter.dim() >= 2:
            torch.nn.init.normal_(parameter, std = 0.05)
    model = get_peft_model(block, LoraConfig(
        r = rank, lora_alpha = 2 * rank, lora_dropout = 0.0, bias = "none",
        target_modules = [],
        target_parameters = ["experts.gate_up_proj", "experts.down_proj"],
    ))
    # PEFT zero initialises lora_B, which would make every arm identical.
    torch.manual_seed(1)
    for name, parameter in model.named_parameters():
        if ".lora_B." in name:
            torch.nn.init.normal_(parameter, std = 0.05)

    moe_utils.patch_param_wrapper_for_moe()
    # PEFT's own forward, from whichever module actually installed the patch: importing
    # unsloth_zoo may already have patched it, so reading ParamWrapper.forward before
    # calling the patcher can hand back the patched one and make this test vacuous.
    import sys
    stock_forward = None
    for module in list(sys.modules.values()):
        original = getattr(module, "_original_param_wrapper_forward", None)
        if original is not None:
            stock_forward = original
            break
    patched_forward = ParamWrapper.forward
    if stock_forward is None or stock_forward is patched_forward:
        pytest.skip("this PEFT is not patched by patch_param_wrapper_for_moe")

    experts = None
    for module in model.modules():
        if type(module).__name__.endswith("Experts") and hasattr(module, "gate_up_proj"):
            experts = module
    experts.forward = types.MethodType(moe_utils.forward_native_grouped_mm, experts)

    # Call the wrapper chain directly rather than the sparse block, so the comparison
    # does not depend on transformers resolving an experts implementation for a config
    # that was never loaded from a checkpoint.
    torch.manual_seed(2)
    tokens = 16
    x = torch.randn(tokens, hidden, device = device, dtype = torch.float32)
    top_k_index = torch.randint(0, num_experts, (tokens, 2), device = device)
    top_k_weights = torch.softmax(
        torch.randn(tokens, 2, device = device, dtype = torch.float32), dim = -1,
    )
    wrapper = model.base_model.model.experts

    def run():
        with torch.no_grad():
            return wrapper(x, top_k_index, top_k_weights)

    # The reference arm first. If PEFT's own forward cannot run in this environment the
    # comparison has no ground truth, and skipping for that reason cannot hide a
    # regression in the separated path, which is the only thing under test here. Under
    # pytest, transformers can resolve an experts implementation that is not callable for
    # a config that was never loaded from a checkpoint. The same comparison outside the
    # harness is scripts/verify_lora_b_packing_fix.py.
    ParamWrapper.forward = stock_forward
    try:
        reference = run()
        with model.disable_adapter():
            base = run()
    except TypeError as exception:
        pytest.skip(f"stock experts dispatch is unavailable here: {exception}")
    finally:
        ParamWrapper.forward = patched_forward

    separated = run()

    effect = (reference - base).abs().max().item()
    assert effect > 1e-3, "the adapter does not reach the forward, so this proves nothing"
    assert (separated - reference).abs().max().item() < effect * 1e-3
