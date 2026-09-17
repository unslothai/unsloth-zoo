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
    MoELoRABLayoutError,
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


def _stub_wrapper(num_experts, rank, dim_in, dim_out):
    """The smallest thing `_extract_lora_from_wrapper` accepts: two Linear-shaped weight
    holders, a scaling and an expert count."""
    class _Weight:
        def __init__(self, tensor):
            self.weight = tensor

    class _Wrapper:
        def __init__(self):
            weight_A, weight_B = _factors(num_experts, rank, dim_in, dim_out)
            self.lora_A = {"default": _Weight(weight_A.float())}
            self.lora_B = {"default": _Weight(weight_B.float())}
            self.scaling = {"default": 1.0}
            self.num_experts = num_experts

        def get_base_layer(self):
            return None

    return _Wrapper()


def test_a_misspelled_layout_raises_instead_of_dropping_the_adapter(monkeypatch):
    """`_extract_lora_from_wrapper` turns every exception into "no LoRA here", so a typo in
    `UNSLOTH_MOE_LORA_B_LAYOUT` used to train the base model in silence. The one error that
    means "your configuration is wrong" has to escape that fallback."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        MoELoRABLayoutError,
        _extract_lora_from_wrapper,
    )

    wrapper = _stub_wrapper(4, 6, 16, 12)

    monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising = False)
    assert _extract_lora_from_wrapper(wrapper) is not None

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", "grouped-by-expert")
    with pytest.raises(MoELoRABLayoutError):
        _extract_lora_from_wrapper(wrapper)

    assert issubclass(MoELoRABLayoutError, ValueError)


@pytest.mark.parametrize(
    "layout", [LORA_B_LAYOUT_RANK_MAJOR, LORA_B_LAYOUT_GROUPED_BY_EXPERT],
)
def test_the_saving_utils_fallback_agrees_with_the_real_helpers(layout):
    """`saving_utils` keeps its own copy of the two readings for an install with no
    `temporary_patches`. Nothing else pins it, so it is free to drift away from the forward
    that trained the adapter."""
    from unsloth_zoo.saving_utils import (
        _fallback_moe_lora_b_expert_columns,
        _fallback_unflatten_moe_lora_b,
    )

    num_experts, rank, dim_out = 4, 6, 12
    _, weight_B = _factors(num_experts, rank, 16, dim_out)

    assert torch.equal(
        _fallback_unflatten_moe_lora_b(weight_B, num_experts, rank, dim_out, layout = layout),
        unflatten_moe_lora_b(weight_B, num_experts, rank, dim_out, layout = layout),
    )
    for expert_idx in range(num_experts):
        assert (
            _fallback_moe_lora_b_expert_columns(expert_idx, num_experts, rank, layout = layout)
            == moe_lora_b_expert_columns(expert_idx, num_experts, rank, layout = layout)
        )


def test_the_saving_utils_fallback_rejects_a_misspelled_layout(monkeypatch):
    """It resolves the same environment variable as `moe_lora_b_layout`, so it has to reject
    the same typos. Defaulting to rank_major on a typo would merge a legacy adapter with the
    wrong reading and bake the scramble into the checkpoint."""
    from unsloth_zoo.saving_utils import _fallback_layout

    monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising = False)
    assert _fallback_layout(None) == LORA_B_LAYOUT_RANK_MAJOR

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", LORA_B_LAYOUT_GROUPED_BY_EXPERT)
    assert _fallback_layout(None) == LORA_B_LAYOUT_GROUPED_BY_EXPERT

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", "grouped-by-expert")
    with pytest.raises(ValueError):
        _fallback_layout(None)
    with pytest.raises(ValueError):
        _fallback_layout("expert_major")


def test_an_explicit_layout_is_validated_not_assumed_grouped():
    """Both readers branch on "is this rank_major" and fall through to grouped_by_expert,
    so an unvalidated typo in an explicit argument is not a no-op: it permutes the columns
    of a standard adapter. A converter passing a layout in from a marker or a command line
    is the likeliest source of one, and it is exactly the damage moe_lora_b_layout()
    validates the environment variable to prevent."""
    weight_B = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    for bad in ("rank-major", "grouped", "RANK_MAJOR", "", "expert_major"):
        with pytest.raises(MoELoRABLayoutError):
            unflatten_moe_lora_b(weight_B, 2, 2, 6, layout=bad)
        with pytest.raises(MoELoRABLayoutError):
            moe_lora_b_expert_columns(0, 2, 2, layout=bad)


def test_both_valid_layouts_still_pass_through_explicitly():
    """The validator must not reject the two real values, including when they are named
    explicitly rather than resolved from the environment."""
    weight_B = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    for good in (LORA_B_LAYOUT_RANK_MAJOR, LORA_B_LAYOUT_GROUPED_BY_EXPERT):
        assert unflatten_moe_lora_b(weight_B, 2, 2, 6, layout=good).shape == (2, 6, 2)
        assert isinstance(moe_lora_b_expert_columns(0, 2, 2, layout=good), slice)
    # and the two disagree, so the argument is load-bearing rather than decorative
    assert not torch.equal(
        unflatten_moe_lora_b(weight_B, 2, 2, 6, layout=LORA_B_LAYOUT_RANK_MAJOR),
        unflatten_moe_lora_b(weight_B, 2, 2, 6, layout=LORA_B_LAYOUT_GROUPED_BY_EXPERT),
    )


@pytest.mark.parametrize("layout", [LORA_B_LAYOUT_RANK_MAJOR,
                                    LORA_B_LAYOUT_GROUPED_BY_EXPERT])
def test_the_grouped_mm_spelling_equals_the_unflatten_spelling(layout, monkeypatch):
    """The grouped-GEMM path builds (E, rank, out) in one copy instead of unflattening to
    (E, out, rank) and transposing, because the second copy is an extra saved tensor and
    breaks non-reentrant gradient checkpointing. One copy and two must still be the same
    tensor, or the forward silently changes meaning while the tests watch the other one."""
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", layout)
    E, r, out, in_dim = 4, 3, 7, 5
    torch.manual_seed(0)
    weight_B = torch.randn(out, E * r)
    weight_A = torch.randn(E * r, in_dim)

    _, second = _canonical_lora_weights_for_grouped_mm(weight_A, weight_B, E, r, in_dim, out)
    expected = unflatten_moe_lora_b(weight_B, E, r, out, layout=layout).transpose(1, 2)
    assert second.shape == (E, r, out)
    assert torch.equal(second, expected)


def test_the_grouped_mm_second_weight_is_materialised_once(monkeypatch):
    """Pin the copy count itself, since equality alone would still pass if the second
    copy came back. A one-copy result is contiguous and owns its storage at exactly the
    element count of the operand."""
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", LORA_B_LAYOUT_RANK_MAJOR)
    E, r, out, in_dim = 4, 3, 7, 5
    weight_B = torch.randn(out, E * r)
    weight_A = torch.randn(E * r, in_dim)
    _, second = _canonical_lora_weights_for_grouped_mm(weight_A, weight_B, E, r, in_dim, out)
    assert second.is_contiguous()
    assert second.untyped_storage().size() // second.element_size() == E * r * out


@pytest.mark.parametrize("num_experts, rank, out", [(4, 3, 7), (2, 2, 5), (1, 5, 4), (6, 1, 3)])
def test_the_column_reorder_matches_index_select_in_both_directions(num_experts, rank, out):
    """The reorder has a hand written backward, because index_select saves its index
    tensor and torch.utils.checkpoint(use_reentrant=False) counts saved tensors. A wrong
    backward here would be silent, so it is pinned against the spelling it replaced."""
    from unsloth_zoo.temporary_patches.moe_utils import _ExpertMajorColumns

    torch.manual_seed(0)
    weight = torch.randn(out, num_experts * rank, dtype=torch.double, requires_grad=True)
    reference_input = weight.detach().clone().requires_grad_(True)
    columns = torch.arange(num_experts * rank).view(rank, num_experts).t().reshape(-1)

    reference = reference_input.index_select(1, columns)
    ours = _ExpertMajorColumns.apply(weight, num_experts, rank)
    assert torch.equal(ours, reference)

    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    ours.backward(upstream)
    assert torch.equal(weight.grad, reference_input.grad)
    assert torch.autograd.gradcheck(
        lambda w: _ExpertMajorColumns.apply(w, num_experts, rank),
        (torch.randn(out, num_experts * rank, dtype=torch.double, requires_grad=True),),
    )


def test_the_column_reorder_saves_nothing_for_backward():
    """The whole reason for the custom Function: a saved tensor here is what made the
    measuring call and its recompute disagree under non-reentrant checkpointing."""
    from unsloth_zoo.temporary_patches.moe_utils import _ExpertMajorColumns

    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t), t)[1], lambda t: t,
    ):
        weight = torch.randn(7, 4 * 3, requires_grad=True)
        _ExpertMajorColumns.apply(weight, 4, 3).sum().backward()
    assert saved == [], f"the reorder saved {len(saved)} tensor(s) for backward"


def test_a_misspelled_layout_aborts_the_merge_before_anything_is_written(monkeypatch):
    """A typo in UNSLOTH_MOE_LORA_B_LAYOUT is a configuration mistake, not an expert the
    merge may skip.

    The per-expert merge helpers wrap their body in `except Exception`, record a fallback
    and return the base weight unchanged. `merge_and_overwrite_lora` writes, and can
    upload, every shard before it consults `_MOE_MERGE_STATE["fallback"]`, so swallowing
    this would publish a checkpoint with the expert deltas silently missing and only then
    report failure. It has to escape instead."""
    import unsloth_zoo.saving_utils as SU

    num_experts, rank, I, H = 4, 2, 3, 5

    _A = torch.zeros(num_experts * rank, H)
    _B = torch.zeros(2 * I, num_experts * rank)

    class _Stats:
        lora_A = _A
        lora_B = _B
        alpha = 1.0
        rank = 2
        module = None

    recorded = []
    monkeypatch.setattr(
        SU, "_record_moe_merge_fallback",
        lambda *a, **k: recorded.append(a), raising=False,
    )
    W = torch.zeros(I, H)

    # Sanity: with a VALID layout these shapes reach the merge rather than bailing out
    # early, so the abort below is really the layout check and not a shape refusal.
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", LORA_B_LAYOUT_RANK_MAJOR)
    SU._merge_moe_gate_expert(W, _Stats(), 0, num_experts, torch.float32)
    assert recorded == [], f"the fixture bailed out before the layout call: {recorded}"

    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", "rank-major")  # note the hyphen
    for helper in (SU._merge_moe_gate_expert, SU._merge_moe_up_expert):
        with pytest.raises(ValueError):
            helper(W, _Stats(), 0, num_experts, torch.float32)
    assert recorded == [], (
        "a layout typo was recorded as a merged-unchanged expert instead of aborting: "
        f"{recorded}"
    )


def test_every_merge_helper_lets_a_layout_error_escape(monkeypatch):
    """All four helpers, not just the two per-expert ones.

    `_merge_moe_experts_file` writes whatever the fused helpers return, so a broad
    `except Exception` there has the same consequence: a checkpoint written, and possibly
    uploaded, with its expert deltas missing. The error is injected at a call inside each
    helper's own `try`, which is precisely the ordering under test: the typed handler has
    to come before the broad one.
    """
    import unsloth_zoo.saving_utils as SU

    def _boom(*args, **kwargs):
        raise SU._MoELoRABLayoutError("Unsloth: UNSLOTH_MOE_LORA_B_LAYOUT is not a layout")

    recorded = []
    monkeypatch.setattr(
        SU, "_record_moe_merge_fallback",
        lambda *a, **k: recorded.append(a), raising=False,
    )
    monkeypatch.setattr(SU, "_refuse_dora_on_moe", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(SU, "_detect_moe_lora_layout", _boom, raising=False)
    monkeypatch.setattr(SU, "_apply_fused_expert_lora_delta", _boom, raising=False)

    class _Stats:
        lora_A = torch.zeros(8, 5)
        lora_B = torch.zeros(6, 8)
        alpha = 1.0
        rank = 2
        module = None

    per_expert = torch.zeros(3, 5)
    fused = torch.zeros(4, 6, 5)
    cases = (
        (SU._merge_moe_gate_expert, (per_expert, _Stats(), 0, 4, torch.float32)),
        (SU._merge_moe_up_expert, (per_expert, _Stats(), 0, 4, torch.float32)),
        (SU._merge_moe_down_proj_expert, (per_expert, _Stats(), 0, 4, torch.float32)),
        (SU._merge_moe_fused_gate_up_expert, (fused, _Stats(), torch.float32)),
        (SU._merge_moe_fused_down_proj_expert, (fused, _Stats(), torch.float32)),
    )
    for helper, args in cases:
        with pytest.raises(SU._MoELoRABLayoutError):
            helper(*args)
    assert recorded == [], (
        f"a layout error was recorded as a merged-unchanged expert: {recorded}"
    )


def test_an_invalid_layout_is_refused_at_import_not_at_first_use():
    """The switch is a process-wide mode set before any MoE work, so a typo has to fail
    before training starts rather than at whichever call happens to read it first.

    Under `torch.compile` that distinction is the whole point. Dynamo installs a guard on
    an `os.environ.get` only when the key is already set at trace time, so on the default
    path a per-call read is never re-evaluated inside a captured graph: a typo introduced
    after the first compiled call is silently ignored for the life of that graph. Reading
    and validating once at import cannot be skipped that way.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import unsloth_zoo.temporary_patches.moe_utils"],
        env={**os.environ, "UNSLOTH_MOE_LORA_B_LAYOUT": "typo-here",
             "UNSLOTH_IS_PRESENT": "1"},
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode != 0, "a misspelled layout imported cleanly"
    assert "MoELoRABLayoutError" in result.stderr
    assert "typo-here" in result.stderr


def test_a_valid_override_set_before_import_survives_compilation():
    """The supported usage: set it before the process does any MoE work. That must reach
    a compiled graph, or the escape hatch is decorative for exactly the MoE training runs
    it exists to serve."""
    import subprocess
    import sys

    program = (
        "import torch, unsloth_zoo.temporary_patches.moe_utils as MU\n"
        "def f(x):\n"
        "    return x + (1.0 if MU.moe_lora_b_layout() == MU.LORA_B_LAYOUT_RANK_MAJOR else 2.0)\n"
        "print(torch.compile(f, fullgraph=False)(torch.zeros(1)).tolist()[0])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "UNSLOTH_MOE_LORA_B_LAYOUT": "grouped_by_expert",
             "UNSLOTH_IS_PRESENT": "1"},
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "2.0", (
        f"the override did not reach the compiled graph: {result.stdout[-500:]}"
    )
