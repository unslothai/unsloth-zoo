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


# ---- Step-3.7-Flash-FP8: transformers swaps Step3p7Experts for FP8Experts at load ----

FB = 128  # the checkpoint's 128x128 weight blocks


def _fp8_model(limits = (0.0, 1.0), device = "meta"):
    """Two Step3p7SparseMoeBlocks (layer 0 unclamped, layer 1 clamped), then the FP8 quantizer's swap."""
    finegrained_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from transformers import FineGrainedFP8Config
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import patch_fp8_experts_interface

    patch_fp8_experts_interface()
    patch_step3p7_moe()
    config = Step3p7TextConfig(
        hidden_size = FB, moe_intermediate_size = FB, n_routed_experts = E, num_experts_per_tok = K,
        num_hidden_layers = 2, num_attention_heads = 2, vocab_size = 32, share_expert_dim = FB,
        swiglu_limits = list(limits), swiglu_limits_shared = [0.0, 0.0],
    )
    model = nn.Module()
    model.config = config  # replace_with_fp8_linear reads model.config.get_text_config()
    with torch.device("meta"):
        model.blocks = nn.ModuleList(modeling.Step3p7SparseMoeBlock(config, i) for i in range(2))
    skip = [f"blocks.{i}.{m}" for i in range(2) for m in ("gate", "shared_experts")]
    finegrained_fp8.replace_with_fp8_linear(
        model, modules_to_not_convert = skip,
        quantization_config = FineGrainedFP8Config(weight_block_size = [FB, FB]),
    )
    if device != "meta":
        model.to_empty(device = device)
    return model, config


def test_fp8_swap_keeps_the_layer_clamp_and_the_lora_aware_dispatch():
    model, config = _fp8_model()
    free, clamped = model.blocks[0].experts, model.blocks[1].experts
    assert type(free).__name__ == type(clamped).__name__ == "FP8Experts"
    # FP8Experts reads config.swiglu_limit, which step3p7 does not have: layer 1's bound must survive.
    assert clamped.limit == 1.0 and clamped._unsloth_own_apply_gate is True
    assert not getattr(free, "_unsloth_own_apply_gate", False)
    # step3p7 is not a @use_experts_implementation model: "eager" would run FP8Experts' own
    # per-expert fp8_linear loop, which never reads the expert LoRA.
    assert config._experts_implementation == "grouped_mm"

    # The restored gate is Step3p7Experts' (clamp after the activation), not FP8Experts' (before it).
    native = modeling.Step3p7Experts(Step3p7TextConfig(
        hidden_size = FB, moe_intermediate_size = FB, n_routed_experts = E, num_attention_heads = 2,
    ), swiglu_limit = 1.0)
    gate_up = torch.randn(16, 2 * FB) * 3
    torch.testing.assert_close(clamped._apply_gate(gate_up), native._apply_gate(gate_up))
    assert (free._apply_gate(gate_up) - native._apply_gate(gate_up)).abs().max() > 0.5


def _quantize_blocks(w):
    """(E, N, K) -> fp8 weight and (E, N/FB, K/FB) weight_scale_inv, as in the FP8 checkpoint."""
    fmax = torch.finfo(torch.float8_e4m3fn).max
    e, n, k = w.shape
    blk = w.float().reshape(e, n // FB, FB, k // FB, FB)
    scale = blk.abs().amax(dim = (2, 4)).clamp(min = 1e-12) / fmax
    q = (blk / scale[:, :, None, :, None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn).reshape(e, n, k)
    deq = (q.float().reshape(e, n // FB, FB, k // FB, FB) * scale[:, :, None, :, None]).reshape(e, n, k)
    return q, scale, deq


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the FP8 MoE backend needs CUDA")
def test_fp8_experts_run_the_clamped_gate_through_the_unsloth_backend():
    model, _ = _fp8_model(device = "cuda")
    experts = model.blocks[1].experts
    g = torch.Generator().manual_seed(0)
    dense = {}
    with torch.no_grad():
        for name, shape, scale in (("gate_up_proj", (E, 2 * FB, FB), 0.3), ("down_proj", (E, FB, FB), 0.05)):
            q, s, deq = _quantize_blocks(torch.randn(*shape, generator = g) * scale)
            getattr(experts, name).copy_(q.cuda())
            getattr(experts, name + "_scale_inv").copy_(s.cuda())
            dense[name] = deq.cuda()
    g = torch.Generator().manual_seed(1)
    hidden = (torch.randn(64, FB, generator = g) * 2).to("cuda", torch.bfloat16)
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(64)]).cuda()
    top_k_weights = torch.rand(64, K, generator = g).to("cuda", torch.bfloat16)

    def reference(gate):
        final = torch.zeros(64, FB, device = "cuda")
        for t in range(64):
            for j in range(K):
                e = int(top_k_index[t, j])
                h = gate(F.linear(hidden[t].float(), dense["gate_up_proj"][e]))
                final[t] += F.linear(h, dense["down_proj"][e]) * top_k_weights[t, j].float()
        return final

    native = modeling.Step3p7Experts(Step3p7TextConfig(
        hidden_size = FB, moe_intermediate_size = FB, n_routed_experts = E, num_attention_heads = 2,
    ), swiglu_limit = 1.0)
    expected = reference(native._apply_gate)
    unclamped = reference(lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
    out = experts(hidden, top_k_index, top_k_weights).float()
    err = (out - expected).abs().max().item()
    gap = (unclamped - expected).abs().max().item()
    # FP8 activations may be quantized on the fast path; the clamp's effect must dwarf that error.
    assert gap > 0.1 and err < gap / 5, (err, gap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the FP8 MoE backend needs CUDA")
def test_stock_fp8_experts_keep_the_configured_swiglu_through_the_unsloth_backend():
    # FP8Experts reads swiglu_alpha / swiglu_limit from the config (HY-V4, GLM-5-Next,
    # MiniMax-M3-VL); routing it through the Unsloth FP8 backend must keep that gate.
    finegrained_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from unsloth_zoo.temporary_patches.moe_utils_fp8 import forward_moe_backend_fp8

    config = Step3p7TextConfig(
        hidden_size = FB, moe_intermediate_size = FB, n_routed_experts = E, num_attention_heads = 2,
    )
    config.swiglu_alpha = 1.702
    config.swiglu_limit = 1.0
    config.hidden_act = "silu"
    experts = finegrained_fp8.FP8Experts(config, block_size = (FB, FB)).cuda()
    assert not getattr(experts, "_unsloth_own_apply_gate", False)
    g = torch.Generator().manual_seed(0)
    dense = {}
    with torch.no_grad():
        for name, shape, scale in (("gate_up_proj", (E, 2 * FB, FB), 0.3), ("down_proj", (E, FB, FB), 0.05)):
            q, s, deq = _quantize_blocks(torch.randn(*shape, generator = g) * scale)
            getattr(experts, name).copy_(q.cuda())
            getattr(experts, name + "_scale_inv").copy_(s.cuda())
            dense[name] = deq.cuda()
    g = torch.Generator().manual_seed(1)
    hidden = (torch.randn(64, FB, generator = g) * 2).to("cuda", torch.bfloat16)
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(64)]).cuda()
    top_k_weights = torch.rand(64, K, generator = g).to("cuda", torch.bfloat16)

    def reference(gate):
        final = torch.zeros(64, FB, device = "cuda")
        for t in range(64):
            for j in range(K):
                e = int(top_k_index[t, j])
                h = gate(F.linear(hidden[t].float(), dense["gate_up_proj"][e]))
                final[t] += F.linear(h, dense["down_proj"][e]) * top_k_weights[t, j].float()
        return final

    expected = reference(experts._apply_gate)
    plain = reference(lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
    out = forward_moe_backend_fp8(experts, hidden, top_k_index, top_k_weights).float()
    err = (out - expected).abs().max().item()
    gap = (plain - expected).abs().max().item()
    assert gap > 0.1 and err < gap / 5, (err, gap)
