# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Half-input GEMM dtype: X follows _triton_moe_compute_dtype into every Triton GEMM (autocast cannot see the custom op,
else gate_up ran float32 x bf16); float16 weights and FP32 flags keep float32."""

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


def test_float32_activations_run_the_gemms_in_bf16(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_TRITON_FP32", raising = False)
    tuned, gemm_inputs = _run(monkeypatch, torch.float32, torch.bfloat16)
    assert gemm_inputs == [(torch.bfloat16, torch.bfloat16)] * 2
    # The kernels are tuned for the dtype they then run in.
    assert tuned == [torch.bfloat16] * 2


def test_float16_weights_keep_float32_activations(monkeypatch):
    # float16 has no bf16 range: the intermediate gradient can overflow, so X stays float32.
    tuned, gemm_inputs = _run(monkeypatch, torch.float32, torch.float16)
    assert gemm_inputs[0] == (torch.float32, torch.float16)


@pytest.mark.parametrize("flag", ["UNSLOTH_FORCE_FLOAT32", "UNSLOTH_MOE_TRITON_FP32"])
def test_fp32_flags_keep_float32_activations(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    tuned, gemm_inputs = _run(monkeypatch, torch.float32, torch.bfloat16)
    assert gemm_inputs[0] == (torch.float32, torch.bfloat16)


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
