# SPDX-License-Identifier: AGPL-3.0-only
"""Step-3.7-Flash (transformers step3p7) routed experts through Unsloth's MoE backend.

Step3p7Experts runs its own per-expert Python loop. Under 4-bit QLoRA that loop matmuls the
packed bnb Params4bit ("size mismatch, got input (N), mat (N x H), vec (1)"), and in 16-bit it
runs every expert one at a time. patch_step3p7_moe routes the class through forward_moe_backend
(dequantize + grouped_mm, expert LoRA folded in) and keeps the class's own clamped swiglu gate,
which Step-3.7 uses on the routed experts of its last two layers (swiglu_limits).
"""
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

modeling = pytest.importorskip("transformers.models.step3p7.modeling_step3p7")
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig

from unsloth_zoo.temporary_patches import moe_utils as mu
from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb
from unsloth_zoo.temporary_patches.step3p7_moe import patch_step3p7_moe

E, H, I, K = 8, 64, 32, 2


def _config():
    return Step3p7TextConfig(
        hidden_size = H, moe_intermediate_size = I, n_routed_experts = E, num_experts_per_tok = K,
        num_hidden_layers = 1, num_attention_heads = 2, vocab_size = 32,
    )


def _experts(limit, seed = 0, scale = 1.5):
    module = modeling.Step3p7Experts(_config(), swiglu_limit = limit)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # Large enough that a limit of 1.0 clamps many gate and up entries.
        module.gate_up_proj.copy_(torch.randn(E, 2 * I, H, generator = g) * scale)
        module.down_proj.copy_(torch.randn(E, H, I, generator = g) * 0.05)
    return module


def _routing(n_tokens, device, seed = 1):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n_tokens, H, generator = g)
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(n_tokens)])
    top_k_weights = torch.rand(n_tokens, K, generator = g)
    return hidden.to(device), top_k_index.to(device), top_k_weights.to(device)


def _reference(module, gate_up, down, hidden, top_k_index, top_k_weights):
    """transformers' Step3p7Experts.forward loop, on explicit weights."""
    final = torch.zeros_like(hidden)
    mask = F.one_hot(top_k_index, num_classes = E).permute(2, 1, 0)
    for expert_idx in range(E):
        top_k_pos, token_idx = torch.where(mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        current = module._apply_gate(F.linear(hidden[token_idx], gate_up[expert_idx]))
        current = F.linear(current, down[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final


def test_the_reference_is_transformers_own_loop():
    # Guards the transcription above against transformers' own loop, stashed by patch_function
    # when the class is already patched in this process.
    klass = modeling.Step3p7Experts
    candidates = [klass.forward] + [v for v in vars(klass).values() if callable(v)]
    originals = [f for f in candidates if getattr(f, "__qualname__", "") == "Step3p7Experts.forward"]
    if not originals:
        pytest.skip("transformers' Step3p7Experts.forward is not reachable in this process")
    original = originals[0]
    module = _experts(limit = 1.0).float()
    hidden, top_k_index, top_k_weights = _routing(32, "cpu")
    expected = original(module, hidden, top_k_index, top_k_weights)
    got = _reference(module, module.gate_up_proj, module.down_proj, hidden, top_k_index, top_k_weights)
    torch.testing.assert_close(got, expected)


def test_patch_routes_the_forward_and_keeps_the_class_gate():
    patch_step3p7_moe()
    klass = modeling.Step3p7Experts
    assert klass._unsloth_already_patched is True
    assert klass._unsloth_own_apply_gate is True
    assert callable(klass._unsloth_lora_extractor_fn)
    assert klass.forward.__name__ == mu.get_forward_moe_backend().__name__
    forward = klass.forward
    patch_step3p7_moe()  # idempotent
    assert klass.forward is forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the MoE backends need CUDA")
@pytest.mark.parametrize("limit", [None, 1.0])
@pytest.mark.parametrize("backend", ["grouped_mm", "unsloth_triton", "native_torch"])
def test_every_backend_matches_the_native_loop(backend, limit, monkeypatch):
    if backend == "grouped_mm" and not getattr(mu, "_check_torch_grouped_mm_supported", lambda: True)():
        pytest.skip("torch._grouped_mm unavailable")
    if backend == "unsloth_triton" and not getattr(mu, "_check_grouped_gemm_available", lambda: True)():
        pytest.skip("Triton grouped GEMM unavailable")
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", backend)
    patch_step3p7_moe()
    module = _experts(limit).to("cuda", torch.bfloat16)
    hidden, top_k_index, top_k_weights = _routing(64, "cuda")
    hidden = (hidden * 2).to(torch.bfloat16)
    top_k_weights = top_k_weights.to(torch.bfloat16)
    expected = _reference(module, module.gate_up_proj, module.down_proj, hidden, top_k_index, top_k_weights)
    out = module(hidden, top_k_index, top_k_weights)
    torch.testing.assert_close(out.float(), expected.float(), rtol = 2e-2, atol = 2e-2)
    if limit is not None:
        # The clamp is really exercised: dropping it changes the answer.
        unclamped = _experts(None).to("cuda", torch.bfloat16)
        free = _reference(unclamped, module.gate_up_proj, module.down_proj, hidden, top_k_index, top_k_weights)
        assert (free.float() - expected.float()).abs().max() > 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "bnb 4-bit needs CUDA")
def test_bnb4bit_experts_match_the_dequantized_native_loop():
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.nn import Params4bit
    from transformers import BitsAndBytesConfig

    patch_step3p7_moe()
    model = nn.Module()
    model.experts = _experts(limit = 1.0)
    dense = {name: getattr(model.experts, name).detach().clone() for name in ("gate_up_proj", "down_proj")}
    mb.replace_expert_params_with_bnb_params(
        model,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.bfloat16
        ),
    )
    for name, value in dense.items():
        param = Params4bit(value.to(torch.bfloat16), requires_grad = False, quant_type = "nf4").to("cuda")
        param._original_shape = value.shape
        setattr(model.experts, name, param)
    experts = model.experts
    assert mb._moe_uses_bnb4bit_expert_weights(experts)

    gate_up = mb._dequantize_bnb4bit_expert_weights(experts.gate_up_proj, torch.bfloat16)
    down = mb._dequantize_bnb4bit_expert_weights(experts.down_proj, torch.bfloat16)
    hidden, top_k_index, top_k_weights = _routing(64, "cuda")
    hidden = (hidden * 2).to(torch.bfloat16)
    top_k_weights = top_k_weights.to(torch.bfloat16)
    expected = _reference(experts, gate_up, down, hidden, top_k_index, top_k_weights)
    out = experts(hidden, top_k_index, top_k_weights)
    assert out.dtype == torch.bfloat16 and out.shape == hidden.shape
    torch.testing.assert_close(out.float(), expected.float(), rtol = 2e-2, atol = 2e-2)

    hidden = hidden.clone().requires_grad_(True)
    experts(hidden, top_k_index, top_k_weights).float().pow(2).sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
