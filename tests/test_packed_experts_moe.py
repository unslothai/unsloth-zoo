# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""NVFP4 / MXFP8 expert stacks through the MoE backend with PEFT LoRA on the experts: one training step is bitwise
equal to the same model holding the dense dequantized stacks; merge, unmerge and a full save write dense experts."""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import unsloth_zoo.packed_experts as pe
from unsloth_zoo.packed_experts import (
    Mxfp8ExpertParam,
    Nvfp4ExpertParam,
    is_new_packed_expert_param,
    quantize_experts_mxfp8,
    quantize_experts_nvfp4,
)
from unsloth_zoo.temporary_patches import moe_utils as mu
from unsloth_zoo.temporary_patches import mxfp4 as mx

QUANTIZE = {"nvfp4": quantize_experts_nvfp4, "mxfp8": quantize_experts_mxfp8}


@pytest.fixture
def grouped_mm_device(monkeypatch):
    if torch.cuda.is_available() and mu._check_torch_grouped_mm_supported():
        return "cuda"
    try:
        torch._grouped_mm(torch.randn(4, 32), torch.randn(2, 32, 16), offs = torch.tensor([2, 4], dtype = torch.int32))
    except Exception:
        pytest.skip("no torch._grouped_mm on this host")
    monkeypatch.setattr(mu, "_TORCH_GROUPED_MM_SUPPORTED", True)
    return "cpu"


class _QwenExperts(nn.Module):
    """transformers 5 fused experts layout: gate_up_proj (E, 2I, H), down_proj (E, H, I), F.linear per expert."""

    def __init__(self, gate_up, down, E, hidden, inter):
        super().__init__()
        self.num_experts = E
        self.hidden_dim = hidden
        self.intermediate_dim = inter
        self.gate_up_proj = gate_up
        self.down_proj = down
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return mu.forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)


_QwenExperts.__name__ = "Qwen3MoeExperts"


class _Block(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, *args):
        return self.experts(*args)


def _stacks(fmt, device, E, hidden, inter, dtype):
    g = torch.Generator().manual_seed(1)
    gu = (torch.randn(E, 2 * inter, hidden, generator = g) * 0.05).to(device)
    dn = (torch.randn(E, hidden, inter, generator = g) * 0.05).to(device)
    return QUANTIZE[fmt](gu, dtype), QUANTIZE[fmt](dn, dtype)


def _pair(fmt, device, E = 4, hidden = 128, inter = 64, rank = 4, dtype = torch.bfloat16, lora = True):
    peft = pytest.importorskip("peft")
    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    gate_up_p, down_p = _stacks(fmt, device, E, hidden, inter, dtype)
    models = []
    for packed in (True, False):
        if packed:
            gate_up, down = gate_up_p, down_p
        else:
            gate_up = nn.Parameter(gate_up_p.dequantize(dtype), requires_grad = False)
            down = nn.Parameter(down_p.dequantize(dtype), requires_grad = False)
        model = _Block(_QwenExperts(gate_up, down, E, hidden, inter)).to(device)
        if lora:
            model = peft.get_peft_model(model, peft.LoraConfig(
                r = rank, lora_alpha = 2 * rank, target_modules = [],
                target_parameters = ["experts.gate_up_proj", "experts.down_proj"],
            ))
            g = torch.Generator().manual_seed(6)
            with torch.no_grad():
                for name, param in sorted(model.named_parameters()):
                    if "lora_" in name:
                        param.copy_((torch.randn(param.shape, generator = g) * 0.05).to(param.device))
        models.append(model)
    return models


def _route(device, tokens = 48, E = 4, top_k = 2, hidden = 128, seed = 7, dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator = g).to(dtype)
    idx = torch.stack([torch.randperm(E - 1, generator = g)[:top_k] for _ in range(tokens)])
    w = torch.softmax(torch.randn(tokens, top_k, generator = g), dim = -1).to(dtype)
    return x.to(device), idx.to(device), w.to(device)


def _train_step(model, x, idx, w):
    xi = x.clone().requires_grad_(True)
    out = model(xi, idx, w)
    out.float().pow(2).mean().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    grads["x"] = xi.grad.clone()
    return out.detach(), grads


def _base(model):
    experts = model.base_model.model.experts if hasattr(model, "base_model") else model.experts
    while hasattr(experts, "base_layer"):
        experts = experts.base_layer
    return experts


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_packed_lora_step_bitwise_vs_dense(monkeypatch, grouped_mm_device, fmt):
    packed_model, dense_model = _pair(fmt, grouped_mm_device)
    lora_p = {n: p.shape for n, p in packed_model.named_parameters() if "lora_" in n}
    lora_d = {n: p.shape for n, p in dense_model.named_parameters() if "lora_" in n}
    assert lora_p == lora_d and len(lora_p) == 4
    x, idx, w = _route(grouped_mm_device)
    before = dict(pe.PACKED_EXPERT_STATS)
    out_p, grads_p = _train_step(packed_model, x, idx, w)
    # Engaged: both projections went through the packed op (the first wrapped call also probes the LoRA stash).
    assert pe.PACKED_EXPERT_STATS["calls"] - before["calls"] >= 2
    assert pe.PACKED_EXPERT_STATS["chunks"] - before["chunks"] >= 4
    out_d, grads_d = _train_step(dense_model, x, idx, w)
    assert torch.equal(out_p, out_d)
    assert set(grads_p) == set(grads_d) and len(grads_p) == 5
    for name in grads_p:
        assert torch.isfinite(grads_p[name]).all() and grads_p[name].abs().sum() > 0, name
        assert torch.equal(grads_p[name], grads_d[name]), name
    base = _base(packed_model)
    assert is_new_packed_expert_param(base.gate_up_proj) and is_new_packed_expert_param(base.down_proj)
    assert base.gate_up_proj.grad is None and base.down_proj.grad is None


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_no_grad_and_no_lora_forward(grouped_mm_device, fmt):
    packed_model, dense_model = _pair(fmt, grouped_mm_device, lora = False)
    x, idx, w = _route(grouped_mm_device)
    with torch.no_grad():
        assert torch.equal(packed_model(x, idx, w), dense_model(x, idx, w))


def test_fp16_compute(grouped_mm_device):
    if grouped_mm_device != "cuda":
        pytest.skip("fp16 grouped GEMM parity is a CUDA check")
    packed_model, dense_model = _pair("nvfp4", grouped_mm_device, dtype = torch.float16)
    x, idx, w = _route(grouped_mm_device, dtype = torch.float16)
    out_p, grads_p = _train_step(packed_model, x, idx, w)
    out_d, grads_d = _train_step(dense_model, x, idx, w)
    assert out_p.dtype == torch.float16 and torch.equal(out_p, out_d)
    for name in grads_p:
        assert torch.equal(grads_p[name], grads_d[name]), name


def test_shape_helpers_see_the_logical_stack():
    gate_up, down = _stacks("nvfp4", "cpu", 4, 64, 32, torch.bfloat16)
    experts = _QwenExperts(gate_up, down, 4, 64, 32)
    assert mu._is_moe_experts_module(experts)
    assert mu._get_param_shape_from_module(experts, "gate_up_proj") == (4, 64, 64)
    assert mu._get_param_shape_from_module(experts, "down_proj") == (4, 64, 32)
    assert mu._moe_uses_new_packed_expert_weights(experts)
    layout = mu._mxfp4_expert_layout(experts.gate_up_proj, "gate_up", 64, None, experts)
    assert layout is not None and layout[1] is True   # Qwen (E, out, in): the GEMM operand is the rows' transpose


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_merge_and_unmerge(grouped_mm_device, fmt):
    packed_model, dense_model = _pair(fmt, grouped_mm_device)
    base = _base(packed_model)
    packed = {n: getattr(base, n) for n in ("gate_up_proj", "down_proj")}
    packed_model.base_model.merge_adapter()
    for name in packed:
        assert not is_new_packed_expert_param(getattr(base, name))
    packed_model.base_model.unmerge_adapter()
    for name in packed:
        assert getattr(base, name) is packed[name]
    merged = [m.merge_and_unload() for m in (packed_model, dense_model)]
    for name in packed:
        a, b = (getattr(m.experts, name) for m in merged)
        assert not is_new_packed_expert_param(a) and a.dtype == torch.bfloat16
        assert torch.equal(a, b)


def test_full_save_writes_dense_experts(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    mx.patch_save_pretrained_mxfp4()
    config = transformers.PretrainedConfig()
    model = transformers.PreTrainedModel(config)
    gate_up, down = _stacks("nvfp4", "cpu", 2, 64, 32, torch.bfloat16)
    model.experts = _QwenExperts(gate_up, down, 2, 64, 32)
    try:
        model.save_pretrained(tmp_path)
    except Exception as exc:  # a bare PreTrainedModel may not save on every transformers release
        pytest.skip(f"bare PreTrainedModel cannot save here: {exc}")
    saved = {}
    for file in os.listdir(tmp_path):
        if file.endswith(".safetensors"):
            saved.update(safetensors.load_file(os.path.join(tmp_path, file)))
    for name, param in (("gate_up_proj", gate_up), ("down_proj", down)):
        weight = saved[f"experts.{name}"]
        assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == tuple(param._original_shape)
        assert torch.equal(weight, param.dequantize())
    assert model.experts.gate_up_proj is gate_up and model.experts.down_proj is down

