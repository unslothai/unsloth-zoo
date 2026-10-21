# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bf16 full finetuning keeps norm weights in float32, so Qwen3-MoE / Mixtral RMSNorm hands the
experts float32 hidden states. Autocast casts F.linear (native loop) and forward_native_grouped_mm
casts explicitly, but the Triton grouped GEMM is a custom op autocast never sees: every expert GEMM
ran float32 x bf16 (3-10x slower). It now casts to the bf16 / fp16 expert weight dtype, as
torch._grouped_mm and plain transformers bf16 training do; float32 weights keep float32."""

import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


def _run(monkeypatch, x_dtype, w_dtype):
    tuned, gemm_inputs = [], []

    def fake_autotune(**kwargs):
        tuned.append(kwargs["dtype"])
        return ("fwd", "dX", "dW")

    def fake_grouped_gemm(X, W, m_sizes, topk, gather_indices, permute_x, permute_y, **kwargs):
        gemm_inputs.append((X.dtype, W.dtype))
        rows = gather_indices.shape[0]
        return torch.zeros(rows, W.shape[1], dtype = X.dtype)

    autotune = types.ModuleType("unsloth.kernels.moe.autotune_cache")
    autotune.get_or_autotune_moe_kernels = fake_autotune
    interface = types.ModuleType("unsloth.kernels.moe.grouped_gemm.interface")
    interface.grouped_gemm = fake_grouped_gemm
    monkeypatch.setitem(sys.modules, autotune.__name__, autotune)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setattr(M, "_TRITON_ROUTING", None, raising = False)

    E, I, H, T, K = 4, 16, 32, 6, 2
    experts = torch.nn.Module()
    experts.num_experts = E
    experts.gate_up_proj = torch.nn.Parameter(torch.zeros(E, 2 * I, H, dtype = w_dtype))
    experts.down_proj = torch.nn.Parameter(torch.zeros(E, H, I, dtype = w_dtype))
    experts.act_fn = torch.nn.functional.silu
    top_k_index = torch.arange(T * K).view(T, K) % E
    out = M.forward_triton_grouped_gemm(
        experts, torch.randn(1, T, H, dtype = x_dtype), top_k_index, torch.rand(T, K)
    )
    assert out.shape == (1, T, H)
    return tuned, gemm_inputs


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float16])
def test_float32_activations_run_the_gemms_in_the_weight_dtype(monkeypatch, w_dtype):
    tuned, gemm_inputs = _run(monkeypatch, torch.float32, w_dtype)
    assert gemm_inputs == [(w_dtype, w_dtype)] * 2
    # The kernels are tuned for the dtype they then run in.
    assert tuned == [w_dtype] * 2


@pytest.mark.parametrize(
    "x_dtype,w_dtype",
    [
        (torch.float32, torch.float32),  # float32 training: unchanged
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
    ],
)
def test_matching_dtypes_are_untouched(monkeypatch, x_dtype, w_dtype):
    tuned, gemm_inputs = _run(monkeypatch, x_dtype, w_dtype)
    assert gemm_inputs == [(x_dtype, w_dtype)] * 2
    assert tuned == [x_dtype] * 2
