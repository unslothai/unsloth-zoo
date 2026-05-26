"""Regression tests for forward_native_moe_loop LoRA application.

These tests cover the gate_up_proj and down_proj transpose-on-mismatch checks
plus the once-per-forward dtype pre-cast that PR #618 introduces. They are
PEFT-version-agnostic: they construct LoRA factors by hand in the same shape
the extractor would produce, then verify that

    forward_native_moe_loop(experts, x, top_k_idx, top_k_weights)

produces the same per-expert output as the naive reference

    base_out + (X @ first[e]) @ second[e] * scaling

for both the canonical (E, 2*I, H) / (E, H, I) Qwen3-MoE storage AND the
transposed (E, H, 2*I) / (E, I, H) Qwen3-VL-MoE storage.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.temporary_patches.moe_utils import forward_native_moe_loop


def _build_experts(num_experts, hidden, intermediate, transposed_storage):
    """Build a minimal `experts` module that exposes the surface area
    `forward_native_moe_loop` reads."""
    experts = nn.Module()
    experts.num_experts = num_experts
    if transposed_storage:
        # Qwen3-VL-MoE layout: (E, in, out) ready for grouped_mm
        experts.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden, 2 * intermediate, dtype=torch.float32)
        )
        experts.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate, hidden, dtype=torch.float32)
        )
    else:
        # Qwen3-MoE / Qwen3.5 / Qwen3-Next standard F.linear layout: (E, out, in)
        experts.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.float32)
        )
        experts.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate, dtype=torch.float32)
        )
    experts.act_fn = F.silu
    return experts


def _build_lora_for(experts, rank, scaling):
    """Build (first_weight, second_weight, scaling) tuples in the layout
    that `_make_qwen_moe_lora_extractor` returns: first=(E, in, R), second=(E, R, out).
    """
    E = experts.num_experts
    if experts.gate_up_proj.shape[1] == 2 * (experts.down_proj.shape[1] if experts.down_proj.shape[1] != experts.gate_up_proj.shape[2] else experts.down_proj.shape[2]):
        # Heuristic that holds for both layouts in this test
        pass
    # Derive in/out from the actual base shape to stay layout-agnostic
    if experts.gate_up_proj.shape[1] > experts.gate_up_proj.shape[2]:
        # standard (E, 2*I, H)
        gate_up_in = experts.gate_up_proj.shape[2]
        gate_up_out = experts.gate_up_proj.shape[1]
    else:
        # transposed (E, H, 2*I)
        gate_up_in = experts.gate_up_proj.shape[1]
        gate_up_out = experts.gate_up_proj.shape[2]
    gate_up = (
        torch.randn(E, gate_up_in, rank, dtype=torch.float32),
        torch.randn(E, rank, gate_up_out, dtype=torch.float32),
        scaling,
    )

    if experts.down_proj.shape[1] > experts.down_proj.shape[2]:
        # standard (E, H, I)
        down_in = experts.down_proj.shape[2]
        down_out = experts.down_proj.shape[1]
    else:
        # transposed (E, I, H)
        down_in = experts.down_proj.shape[1]
        down_out = experts.down_proj.shape[2]
    down = (
        torch.randn(E, down_in, rank, dtype=torch.float32),
        torch.randn(E, rank, down_out, dtype=torch.float32),
        scaling,
    )
    return gate_up, down


def _naive_forward(experts, hidden_states, top_k_index, top_k_weights, gate_up_lora, down_lora):
    """Reference implementation that mirrors forward_native_moe_loop without
    the under-test transpose / pre-cast logic. Used as ground truth.
    """
    E = experts.num_experts
    out = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for ei_t in expert_hit:
        ei = ei_t.item()
        top_k_pos, token_idx = torch.where(expert_mask[ei])
        x = hidden_states[token_idx]

        gu_w = experts.gate_up_proj[ei]
        if gu_w.shape[-1] != x.shape[-1]:
            gu_w = gu_w.T
        gu = F.linear(x, gu_w)
        if gate_up_lora is not None:
            f, s, sc = gate_up_lora
            gu = gu + ((x @ f[ei]) @ s[ei]) * sc

        gate, up = gu.chunk(2, dim=-1)
        h = experts.act_fn(gate) * up

        d_w = experts.down_proj[ei]
        if d_w.shape[-1] != h.shape[-1]:
            d_w = d_w.T
        d = F.linear(h, d_w)
        if down_lora is not None:
            f, s, sc = down_lora
            d = d + ((h @ f[ei]) @ s[ei]) * sc

        d = d * top_k_weights[token_idx, top_k_pos, None]
        out.index_add_(0, token_idx, d.to(out.dtype))
    return out


@pytest.mark.parametrize("transposed_storage", [False, True],
                         ids=["canonical_storage", "transposed_storage"])
def test_forward_native_moe_loop_with_lora_matches_naive(transposed_storage):
    torch.manual_seed(0)
    num_experts = 4
    hidden = 32
    intermediate = 24
    rank = 4
    num_tokens = 7
    top_k = 2

    experts = _build_experts(num_experts, hidden, intermediate, transposed_storage)
    gate_up_lora, down_lora = _build_lora_for(experts, rank, scaling=2.0)

    # Stash the LoRA tensors where forward_native_moe_loop expects them.
    experts._unsloth_lora_gate_up_proj = gate_up_lora
    experts._unsloth_lora_down_proj = down_lora

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.float32)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    out = forward_native_moe_loop(experts, hidden_states, top_k_index, top_k_weights)
    ref = _naive_forward(experts, hidden_states, top_k_index, top_k_weights,
                         gate_up_lora, down_lora)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("transposed_storage", [False, True],
                         ids=["canonical_storage", "transposed_storage"])
def test_forward_native_moe_loop_no_lora_matches_naive(transposed_storage):
    """Same but without LoRA. Confirms the down_proj transpose-on-mismatch
    check (gemini HIGH) does not break the LoRA-disabled path.
    """
    torch.manual_seed(1)
    num_experts = 3
    hidden = 16
    intermediate = 12
    num_tokens = 5
    top_k = 2

    experts = _build_experts(num_experts, hidden, intermediate, transposed_storage)
    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.float32)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    out = forward_native_moe_loop(experts, hidden_states, top_k_index, top_k_weights)
    ref = _naive_forward(experts, hidden_states, top_k_index, top_k_weights,
                         None, None)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_forward_native_moe_loop_square_dim_uses_grouped_mm_flag():
    """When `intermediate_dim == hidden_dim`, the shape-based transpose check
    cannot tell which orientation the per-expert weight is stored in. The
    explicit `_unsloth_grouped_mm_format` flag must take precedence so that
    transposed storage is detected even at square dims.
    """
    torch.manual_seed(13)
    num_experts = 3
    dim = 16  # hidden == intermediate, so 2*intermediate == 2*hidden
    rank = 2
    num_tokens = 5
    top_k = 2

    # Use intermediate = dim so that down_proj is (E, dim, dim) — square.
    # gate_up_proj is (E, 2*dim, dim) which is non-square so easy.
    intermediate = dim
    hidden = dim

    experts = nn.Module()
    experts.num_experts = num_experts
    # Transposed (grouped_mm) storage:
    experts.gate_up_proj = nn.Parameter(
        torch.randn(num_experts, hidden, 2 * intermediate, dtype=torch.float32)
    )
    experts.down_proj = nn.Parameter(
        torch.randn(num_experts, intermediate, hidden, dtype=torch.float32)
    )
    experts.act_fn = F.silu
    experts._unsloth_grouped_mm_format = True

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.float32)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    out = forward_native_moe_loop(experts, hidden_states, top_k_index, top_k_weights)

    # Reference manually transposes both weights regardless of shape.
    ref = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for ei_t in expert_hit:
        ei = ei_t.item()
        top_k_pos, token_idx = torch.where(expert_mask[ei])
        x = hidden_states[token_idx]
        gu = F.linear(x, experts.gate_up_proj[ei].T)
        g, u = gu.chunk(2, dim=-1)
        h = experts.act_fn(g) * u
        d = F.linear(h, experts.down_proj[ei].T)
        d = d * top_k_weights[token_idx, top_k_pos, None]
        ref.index_add_(0, token_idx, d.to(ref.dtype))

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16],
                         ids=["fp32", "bf16", "fp16"])
def test_forward_native_moe_loop_lora_dtype_precast_no_loop_alloc(dtype):
    """Ensure the new pre-cast (LoRA factors cast to hidden_states.dtype once
    before the per-expert loop) produces correct output across dtypes."""
    torch.manual_seed(2)
    num_experts = 3
    hidden = 16
    intermediate = 12
    rank = 2
    num_tokens = 6
    top_k = 2

    experts = _build_experts(num_experts, hidden, intermediate, False)
    gate_up_lora, down_lora = _build_lora_for(experts, rank, scaling=1.5)
    # Cast everything (params + lora) to the target dtype.
    experts.gate_up_proj.data = experts.gate_up_proj.data.to(dtype)
    experts.down_proj.data = experts.down_proj.data.to(dtype)
    experts._unsloth_lora_gate_up_proj = (
        gate_up_lora[0].to(dtype), gate_up_lora[1].to(dtype), gate_up_lora[2],
    )
    experts._unsloth_lora_down_proj = (
        down_lora[0].to(dtype), down_lora[1].to(dtype), down_lora[2],
    )

    hidden_states = torch.randn(num_tokens, hidden, dtype=dtype)
    top_k_index = torch.randint(0, num_experts, (num_tokens, top_k))
    top_k_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1).to(dtype)

    out = forward_native_moe_loop(experts, hidden_states, top_k_index, top_k_weights)
    ref = _naive_forward(experts, hidden_states, top_k_index, top_k_weights,
                         experts._unsloth_lora_gate_up_proj,
                         experts._unsloth_lora_down_proj)

    # Looser tolerance for low-precision dtypes
    atol = 1e-4 if dtype == torch.float32 else 1e-1
    rtol = 1e-4 if dtype == torch.float32 else 1e-1
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    assert out.dtype == dtype
