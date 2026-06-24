"""The batched baddbmm path for fused MoE expert LoRA must be numerically identical to the
per-expert loop it replaces, for both the standard and transposed (GPT-OSS) layouts, with the
LoRA scaling (alpha) applied exactly once per expert."""
import pytest
import torch

from unsloth_zoo.saving_utils import _apply_fused_expert_lora_delta


def _loop_reference(merged, lora_A, lora_B, num_experts, rank, alpha, use_transpose):
    out = merged.clone()
    for e in range(num_experts):
        s, t = e * rank, (e + 1) * rank
        delta = lora_B[:, s:t] @ lora_A[s:t, :]
        out[e] = out[e].add(delta.T if use_transpose else delta, alpha=alpha)
    return out


def _make(num_experts, rank, dim_A, dim_B, use_transpose, dtype, device):
    torch.manual_seed(0)
    lora_A = torch.randn(num_experts * rank, dim_A, dtype=dtype, device=device)
    lora_B = torch.randn(dim_B, num_experts * rank, dtype=dtype, device=device)
    # delta_e is (dim_B, dim_A); merged[e] adds delta (std) or delta.T (transposed).
    shape = (num_experts, dim_A, dim_B) if use_transpose else (num_experts, dim_B, dim_A)
    merged = torch.randn(*shape, dtype=dtype, device=device)
    return merged, lora_A, lora_B


@pytest.mark.parametrize("use_transpose", [False, True])
def test_cpu_path_matches_loop(use_transpose):
    E, rank, dim_A, dim_B, alpha = 8, 4, 16, 24, 2.0
    merged, A, B = _make(E, rank, dim_A, dim_B, use_transpose, torch.float32, "cpu")
    ref = _loop_reference(merged, A, B, E, rank, alpha, use_transpose)
    got = _apply_fused_expert_lora_delta(merged.clone(), A, B, E, rank, dim_A, dim_B, alpha, use_transpose)
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the baddbmm path")
@pytest.mark.parametrize("use_transpose", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cuda_baddbmm_matches_loop(use_transpose, dtype):
    # 128 experts to mirror a real fused MoE; compare baddbmm vs the per-expert loop on the SAME
    # device so the only difference is batching (must be bitwise-identical).
    E, rank, dim_A, dim_B, alpha = 128, 16, 64, 96, 1.7
    merged, A, B = _make(E, rank, dim_A, dim_B, use_transpose, dtype, "cuda")
    ref = _loop_reference(merged, A, B, E, rank, alpha, use_transpose)
    got = _apply_fused_expert_lora_delta(merged.clone(), A, B, E, rank, dim_A, dim_B, alpha, use_transpose)
    assert torch.equal(got, ref), (got - ref).abs().max().item()
