# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Remote-code MoE experts (w1 / w2 / w3 Linears, Kimi-K3) kept as one packed MXFP4 stack per
projection. The stacked bytes must decode to exactly the per-expert compressed-tensors weights,
the grouped forward must equal the same grouped GEMM on the dense decode bit for bit, dX must
come from a recompute (no 16-bit weight saved for backward), chunking and skipped experts must
not change a value, and expert LoRA from the stash must add the dense delta."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam, mxfp4_dequantize_torch
from unsloth_zoo.mxfp4_stacked_experts import (
    Mxfp4StackedExperts,
    stack_packed_expert_blocks,
    stack_packed_expert_scales,
)
from unsloth_zoo.temporary_patches import moe_utils as mu
from unsloth_zoo.temporary_patches import mxfp4 as mx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E, H, I = 8, 128, 64


class _Situ(nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        gate, up = x[..., :d].float(), x[..., d:].float()
        return (torch.tanh(gate) * torch.sigmoid(gate) * up).to(x.dtype)


def _checkpoint_bytes(seed = 0):
    """Per-expert compressed-tensors tensors: packed (out, in / 2) and scale (out, in / 32)."""
    g = torch.Generator().manual_seed(seed)

    def one(out_f, in_f):
        packed = torch.randint(0, 256, (out_f, in_f // 2), dtype = torch.uint8, generator = g)
        scale = torch.randint(118, 125, (out_f, in_f // 32), dtype = torch.uint8, generator = g)
        return packed, scale

    return {
        w: [one(*((H, I) if w == "w2" else (I, H))) for _ in range(E)] for w in ("w1", "w2", "w3")
    }


def _ct_decompress(packed, scale):
    """The checkpoint's own decode: compressed-tensors when installed, else the same LUT."""
    try:
        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    except Exception:
        return mxfp4_dequantize_torch(
            packed.reshape(packed.shape[0], -1, 16), scale, dtype = torch.float32,
        )
    scheme = QuantizationScheme(
        targets = ["Linear"],
        weights = QuantizationArgs(
            num_bits = 4, type = "float", strategy = "group", group_size = 32,
            symmetric = True, scale_dtype = torch.uint8,
        ),
        format = "mxfp4-pack-quantized",
    )
    comp = BaseCompressor.get_value_from_registry("mxfp4-pack-quantized")
    return comp.decompress({"weight_packed": packed, "weight_scale": scale}, scheme)["weight"].float()


def _experts(fused = True, seed = 0):
    ckpt = _checkpoint_bytes(seed)
    module = Mxfp4StackedExperts(E, H, I, _Situ() if fused else nn.SiLU(), fused, device = "cpu")
    module.gate_up_blocks.data = stack_packed_expert_blocks(
        [[p for p, _ in ckpt["w1"]], [p for p, _ in ckpt["w3"]]]
    )
    module.gate_up_scales.data = stack_packed_expert_scales(
        [[s for _, s in ckpt["w1"]], [s for _, s in ckpt["w3"]]]
    )
    module.down_blocks.data = stack_packed_expert_blocks([[p for p, _ in ckpt["w2"]]])
    module.down_scales.data = stack_packed_expert_scales([[s for _, s in ckpt["w2"]]])
    return module.finalize().to(DEVICE), ckpt


def _dense(module):
    return (
        module.gate_up_proj.dequantize(torch.bfloat16),
        module.down_proj.dequantize(torch.bfloat16),
    )


def _routing(tokens = 48, top_k = 2, experts = None, seed = 1):
    g = torch.Generator().manual_seed(seed)
    pool = torch.tensor(experts if experts is not None else range(E))
    idx = pool[torch.stack([torch.randperm(len(pool), generator = g)[:top_k] for _ in range(tokens)])]
    weight = torch.rand(tokens, top_k, generator = g)
    x = torch.randn(tokens, H, generator = g).to(torch.bfloat16)
    return x.to(DEVICE), idx.to(DEVICE), weight.to(DEVICE)


def _grouped_reference(module, x, idx, weight, gate_up_w, down_w, lora = None):
    """The same dispatch on dense (E, in, out) stacks through the zoo grouped GEMM."""
    T, k = idx.shape
    order = torch.argsort(idx.reshape(-1), stable = True)
    rows = x[order // k]
    offsets = torch.cumsum(torch.bincount(idx.reshape(-1), minlength = E), 0, dtype = torch.int32)
    gate_up = mu._grouped_mm_with_backward_fix(rows, gate_up_w, offsets)
    if lora is not None:
        first, second, scaling = lora
        gate_up = gate_up + mu._apply_lora_grouped_mm(rows, first, second, offsets, scaling)
    out = mu._grouped_mm_with_backward_fix(module._activate(gate_up).to(rows.dtype), down_w, offsets)
    unsorted = torch.empty_like(out)
    unsorted[order] = out
    return (unsorted.view(T, k, -1).float() * weight.float().unsqueeze(-1)).sum(1).to(x.dtype)


def test_stacked_bytes_decode_to_the_checkpoint_weights():
    module, ckpt = _experts()
    gate_up, down = (w.float().cpu() for w in _dense(module))
    for e in range(E):
        w1 = _ct_decompress(*ckpt["w1"][e])
        w3 = _ct_decompress(*ckpt["w3"][e])
        w2 = _ct_decompress(*ckpt["w2"][e])
        # (E, in, out): gate_up[e] is [w1; w3]^T, down[e] is w2^T.
        assert torch.equal(gate_up[e], torch.cat([w1, w3], 0).t())
        assert torch.equal(down[e], w2.t())


@pytest.mark.parametrize("fused", [True, False])
def test_forward_matches_the_dense_grouped_gemm_exactly(fused):
    module, _ = _experts(fused = fused)
    x, idx, weight = _routing()
    with torch.no_grad():
        got = module(x, idx, weight)
        want = _grouped_reference(module, x, idx, weight, *_dense(module))
    assert torch.equal(got, want)


def test_forward_matches_the_per_expert_linear_loop():
    module, ckpt = _experts()
    x, idx, weight = _routing()
    act = module.act_fn
    out = torch.zeros(x.shape[0], H, dtype = torch.float32, device = DEVICE)
    for e in range(E):
        w1, w3, w2 = (_ct_decompress(*ckpt[w][e]).to(DEVICE, torch.bfloat16) for w in ("w1", "w3", "w2"))
        for slot in range(idx.shape[1]):
            rows = (idx[:, slot] == e).nonzero().flatten()
            if rows.numel():
                h = act(torch.cat([F.linear(x[rows], w1), F.linear(x[rows], w3)], -1))
                out[rows] += F.linear(h, w2).float() * weight[rows, slot, None]
    with torch.no_grad():
        got = module(x, idx, weight).float()
    torch.testing.assert_close(got, out, atol = 2e-2, rtol = 2e-2)


def test_input_grad_is_recomputed_and_no_weight_is_saved():
    module, _ = _experts()
    x, idx, weight = _routing()
    x.requires_grad_(True)
    saved = []

    def pack(t):
        saved.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        got = module(x, idx, weight)
    got.float().square().sum().backward()
    stack = min(E * H * 2 * I, E * I * H)
    assert max(saved) < stack, (max(saved), stack)

    x_ref = x.detach().clone().requires_grad_(True)
    ref = _grouped_reference(module, x_ref, idx, weight, *_dense(module))
    ref.float().square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, atol = 0, rtol = 0)


def test_chunked_and_skipped_experts_change_nothing(monkeypatch):
    module, _ = _experts()
    x, idx, weight = _routing(experts = [0, 3, 5, 7])
    with torch.no_grad():
        whole = module(x, idx, weight)
    # One expert's gate_up is 128 * 128 * 2 bytes: a 1 MB budget still packs many, so force 1.
    monkeypatch.setattr("unsloth_zoo.mxfp4_stacked_experts._chunk_bytes", lambda: 1)
    x.requires_grad_(True)
    chunked = module(x, idx, weight)
    assert torch.equal(chunked.detach(), whole)
    chunked.float().sum().backward()
    x_ref = x.detach().clone().requires_grad_(True)
    monkeypatch.undo()
    module(x_ref, idx, weight).float().sum().backward()
    assert torch.equal(x.grad, x_ref.grad)


def test_indexing_an_expert_gives_its_linear():
    module, _ = _experts()
    x, idx, weight = _routing()
    gate_up, down = _dense(module)
    with torch.no_grad():
        got = module[3](x)
        want = module._activate(x @ gate_up[3]) @ down[3]
    assert len(module) == E
    assert torch.equal(got, want)


def test_stashed_lora_adds_the_dense_delta_and_trains():
    module, _ = _experts()
    x, idx, weight = _routing()
    r = 16
    g = torch.Generator().manual_seed(5)
    first = (torch.randn(E, H, r, generator = g) * 0.1).to(DEVICE, torch.bfloat16).requires_grad_(True)
    second = (torch.randn(E, r, 2 * I, generator = g) * 0.1).to(DEVICE, torch.bfloat16).requires_grad_(True)
    setattr(module, mu.moe_lora_stash_name("gate_up_proj"), (first, second, 2.0, E))
    got = module(x, idx, weight)
    want = _grouped_reference(module, x, idx, weight, *_dense(module), lora = (first, second, 2.0))
    torch.testing.assert_close(got, want, atol = 0, rtol = 0)
    got.float().sum().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    assert second.grad is not None and second.grad.abs().sum() > 0


def test_an_unloaded_stack_is_reported():
    module = Mxfp4StackedExperts(E, H, I, _Situ(), True, device = "meta")
    with pytest.raises(RuntimeError, match = "never loaded"):
        module.finalize()


class _Block(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, *args):
        return self.experts(*args)


class _DenseStackedExperts(nn.Module):
    """The same dispatch on dense stacks: the bf16 decompressed reference."""

    _unsloth_grouped_mm_format = True

    def __init__(self, packed):
        super().__init__()
        self.num_experts = packed.num_experts
        self.act_fn, self.fused_gate_up_act = packed.act_fn, packed.fused_gate_up_act
        gate_up, down = _dense(packed)
        self.gate_up_proj = nn.Parameter(gate_up, requires_grad = False)
        self.down_proj = nn.Parameter(down, requires_grad = False)

    _activate = Mxfp4StackedExperts._activate

    def forward(self, x, idx, weight):
        lora = mu.take_moe_lora_stash(self, "gate_up_proj")
        lora_down = mu.take_moe_lora_stash(self, "down_proj")
        T, k = idx.shape
        order = torch.argsort(idx.reshape(-1), stable = True)
        rows = x[order // k]
        offsets = torch.cumsum(torch.bincount(idx.reshape(-1), minlength = E), 0, dtype = torch.int32)
        gate_up = mu._grouped_mm_with_backward_fix(rows, self.gate_up_proj, offsets)
        if lora is not None:
            gate_up = gate_up + mu._apply_lora_grouped_mm(rows.to(lora[0].dtype), lora[0], lora[1], offsets, lora[2]).to(rows.dtype)
        hidden = self._activate(gate_up).to(rows.dtype)
        out = mu._grouped_mm_with_backward_fix(hidden, self.down_proj, offsets)
        if lora_down is not None:
            out = out + mu._apply_lora_grouped_mm(
                hidden.to(lora_down[0].dtype), lora_down[0], lora_down[1], offsets, lora_down[2]
            ).to(hidden.dtype)
        unsorted = torch.empty_like(out)
        unsorted[order] = out
        return (unsorted.view(T, k, -1).float() * weight.float().unsqueeze(-1)).sum(1).to(x.dtype)


def _peft_pair(rank = 4):
    peft = pytest.importorskip("peft")
    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    packed, _ = _experts()
    models = []
    for experts in (packed, _DenseStackedExperts(packed).to(DEVICE)):
        model = peft.get_peft_model(_Block(experts), peft.LoraConfig(
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


@pytest.mark.parametrize("checkpoint", [False, True])
def test_peft_expert_lora_trains_like_the_dense_reference(checkpoint):
    if checkpoint and DEVICE != "cuda":
        pytest.skip("Unsloth's offloaded checkpoint needs CUDA")
    packed_model, dense_model = _peft_pair()
    x, idx, weight = _routing()
    results = []
    for model in (packed_model, dense_model):
        xi = x.clone().requires_grad_(True)

        def block(h):
            return model(h, idx, weight)

        if checkpoint:
            from unsloth_zoo.gradient_checkpointing import unsloth_offloaded_gradient_checkpoint
            out = unsloth_offloaded_gradient_checkpoint(block, xi)
        else:
            out = block(xi)
        out.float().pow(2).mean().backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        results.append((out.detach(), xi.grad, grads))
    (out_a, dx_a, g_a), (out_b, dx_b, g_b) = results
    assert torch.equal(out_a, out_b)
    assert torch.equal(dx_a, dx_b)
    assert g_a.keys() == g_b.keys() and len(g_a) == 4
    for name in g_a:
        assert torch.equal(g_a[name], g_b[name]), name


def test_peft_merge_is_dense_and_unmerge_restores_the_packed_stack():
    packed_model, dense_model = _peft_pair()
    x, idx, weight = _routing()
    base = packed_model.base_model.model.experts
    while hasattr(base, "base_layer"):
        base = base.base_layer
    packed = {n: getattr(base, n) for n in ("gate_up_proj", "down_proj")}
    with torch.no_grad():
        unmerged = packed_model(x, idx, weight)
        packed_model.base_model.merge_adapter()
        assert not any(isinstance(getattr(base, n), Mxfp4ExpertParam) for n in packed)
        merged = packed_model(x, idx, weight)
        packed_model.base_model.unmerge_adapter()
    assert all(getattr(base, n) is packed[n] for n in packed)
    torch.testing.assert_close(merged.float(), unmerged.float(), atol = 3e-2, rtol = 3e-2)
