# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Remote-code MoE experts (w1 / w2 / w3 Linears, Kimi-K3) kept as one packed MXFP4 stack per
projection. The stacked bytes must decode to exactly the per-expert compressed-tensors weights,
the grouped forward must equal the same grouped GEMM on the dense decode bit for bit, dX must
come from a recompute (no 16-bit weight saved for backward), chunking and skipped experts must
not change a value, and expert LoRA from the stash must add the dense delta."""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam, mxfp4_dequantize_torch
from unsloth_zoo.mxfp4_stacked_experts import Mxfp4StackedExperts, stack_packed_experts
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
    module.gate_up_blocks.data = stack_packed_experts(
        [[p for p, _ in ckpt["w1"]], [p for p, _ in ckpt["w3"]]]
    )
    module.gate_up_scales.data = stack_packed_experts(
        [[s for _, s in ckpt["w1"]], [s for _, s in ckpt["w3"]]], blocks = False
    )
    module.down_blocks.data = stack_packed_experts([[p for p, _ in ckpt["w2"]]])
    module.down_scales.data = stack_packed_experts([[s for _, s in ckpt["w2"]]], blocks = False)
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
    # The failed call leaves the stack as it was, so every later call says so too.
    assert sorted(module._parameters) == sorted(
        ["gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales"]
    )
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


def test_dense_expert_modules_are_the_checkpoint_linears():
    from unsloth_zoo.mxfp4_stacked_experts import dense_expert_modules

    module, ckpt = _experts()
    experts = dense_expert_modules(module)
    assert len(experts) == E
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            linear = getattr(experts[e], w)
            assert linear.weight.device.type == "cpu" and linear.bias is None
            want = _ct_decompress(*ckpt[w][e]).to(torch.bfloat16)
            assert torch.equal(linear.weight, want), (e, w)
    # A merged (dense) stack is split the same way.
    gate_up, _ = _dense(module)
    module.gate_up_proj = nn.Parameter(gate_up + 1, requires_grad = False)
    experts = dense_expert_modules(module)
    assert torch.equal(experts[2].w3.weight, (gate_up[2, :, I:] + 1).t().cpu())


def _write_expert_shard(root, ckpt, prefix = "model.experts"):
    from safetensors.torch import save_file

    tensors = {"model.other.weight": torch.ones(4, 4, dtype = torch.bfloat16)}
    for w in ("w1", "w2", "w3"):
        for e, (packed, scale) in enumerate(ckpt[w]):
            tensors[f"{prefix}.{e}.{w}.weight_packed"] = packed
            tensors[f"{prefix}.{e}.{w}.weight_scale"] = scale
    save_file(tensors, os.path.join(root, "model.safetensors"), metadata = {"format": "pt"})


def test_merged_16bit_rewrite_decodes_mxfp4_and_folds_in_the_expert_lora(tmp_path):
    from safetensors.torch import load_file
    from unsloth_zoo.saving_utils import _dequantize_compressed_mxfp4_shards

    packed_model, _ = _peft_pair()
    ckpt = _checkpoint_bytes(0)
    _write_expert_shard(str(tmp_path), ckpt)
    wrappers = [n for n, m in packed_model.base_model.model.named_modules() if hasattr(m, "parameter_name")]
    assert sorted(wrappers) == ["experts", "experts.base_layer"]
    lora_weights = {"experts": 1, "experts.base_layer": 2, "model.other": 3}
    _dequantize_compressed_mxfp4_shards(str(tmp_path), ["model.safetensors"], lora_weights, packed_model)
    assert lora_weights == {"model.other": 3}  # the stacks' LoRA is merged here, not again later
    state = load_file(str(tmp_path / "model.safetensors"))
    assert not any(k.endswith(("weight_packed", "weight_scale")) for k in state)
    assert torch.equal(state["model.other.weight"], torch.ones(4, 4, dtype = torch.bfloat16))

    deltas = {}
    holder = packed_model.base_model.model.experts
    while hasattr(holder, "base_layer"):
        deltas[holder.parameter_name] = holder.get_delta_weight("default").float().cpu()
        holder = holder.base_layer
    with torch.no_grad():
        packed_model.base_model.merge_adapter()
        merged_gate_up = holder.gate_up_proj.detach().float().cpu()
        merged_down = holder.down_proj.detach().float().cpu()
        packed_model.base_model.unmerge_adapter()
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            decode = _ct_decompress(*ckpt[w][e])
            if w == "w2":
                delta, merged = deltas["down_proj"][e].t(), merged_down[e].t()
            else:
                cols = slice(0, I) if w == "w1" else slice(I, 2 * I)
                delta = deltas["gate_up_proj"][e, :, cols].t()
                merged = merged_gate_up[e, :, cols].t()
            got = state[f"model.experts.{e}.{w}.weight"]
            assert delta.abs().sum() > 0
            assert torch.equal(got, (decode + delta).to(torch.bfloat16)), (e, w)
            # The same weight PEFT's in-memory merge of the packed stack gives, to bf16 rounding.
            torch.testing.assert_close(got.float(), merged, atol = 1e-2, rtol = 1e-2)


def test_merged_16bit_rewrite_refuses_what_it_cannot_place(tmp_path):
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import _dequantize_compressed_mxfp4_shards

    packed_model, _ = _peft_pair()
    ckpt = _checkpoint_bytes(0)
    # The expert LoRA has no per-expert weights to land on.
    _write_expert_shard(str(tmp_path), ckpt, prefix = "model.elsewhere")
    with pytest.raises(RuntimeError, match = "would drop it"):
        _dequantize_compressed_mxfp4_shards(str(tmp_path), ["model.safetensors"], {}, packed_model)
    # A packed tensor that is not MXFP4 (int32 words).
    save_file(
        {"a.weight_packed": torch.zeros(4, 8, dtype = torch.int32), "a.weight_scale": torch.zeros(4, 1)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(RuntimeError, match = "not an MXFP4 weight"):
        _dequantize_compressed_mxfp4_shards(str(tmp_path), ["model.safetensors"], {}, nn.Linear(2, 2))


def test_compressed_packed_format_detection(tmp_path):
    import json
    from unsloth_zoo.saving_utils import _compressed_packed_format

    def write(quant):
        (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant} if quant else {}))
        return _compressed_packed_format(str(tmp_path))

    group = lambda fmt = None: {"format": fmt, "weights": {"num_bits": 4}}  # noqa: E731
    assert write({"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
                  "config_groups": {"g": group()}}) == "mxfp4-pack-quantized"
    assert write({"quant_method": "compressed-tensors", "format": "pack-quantized",
                  "config_groups": {"g": group()}}) == "pack-quantized"
    assert write({"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
                  "config_groups": {"g": group(), "h": group("pack-quantized")}}) == "mxfp4-pack-quantized,pack-quantized"
    assert write({"quant_method": "compressed-tensors", "format": "float-quantized",
                  "config_groups": {"g": group()}}) is None
    assert write({"quant_method": "bitsandbytes", "load_in_4bit": True}) is None
    assert write(None) is None
    # Kimi-K3 keeps it under text_config only; legacy llm-compressor nests it one level down.
    mxfp4 = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
             "config_groups": {"g": group()}}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"quantization_config": mxfp4}}))
    assert _compressed_packed_format(str(tmp_path)) == "mxfp4-pack-quantized"
    legacy = {"quant_method": "compressed-tensors", "quantization_config": dict(mxfp4)}
    assert write(legacy) == "mxfp4-pack-quantized"


def test_merged_exports_still_run_under_inference_mode():
    """The compressed-tensors helpers sit above `merge_and_overwrite_lora`: its decorator must
    stay on it, not move onto the first helper."""
    import inspect
    from unsloth_zoo import saving_utils

    assert inspect.unwrap(saving_utils.merge_and_overwrite_lora) is not saving_utils.merge_and_overwrite_lora
    assert inspect.unwrap(saving_utils._compressed_packed_format) is saving_utils._compressed_packed_format


def test_indexing_an_expert_after_a_merge_uses_the_merged_weights():
    packed_model, _ = _peft_pair()
    x, _, _ = _routing()
    base = packed_model.base_model.model.experts
    while hasattr(base, "base_layer"):
        base = base.base_layer
    with torch.no_grad():
        packed_model.base_model.merge_adapter()
        gate_up, down = base.gate_up_proj, base.down_proj
        assert not isinstance(gate_up, Mxfp4ExpertParam)
        got = base[3](x)
        want = base._activate(x @ gate_up[3].to(x.dtype)) @ down[3].to(x.dtype)
        packed_model.base_model.unmerge_adapter()
    torch.testing.assert_close(got, want, atol = 0, rtol = 0)


@pytest.mark.parametrize("lora", [False, True])
def test_no_routed_tokens_give_an_empty_output(lora):
    module, _ = _experts()
    x, idx, weight = _routing()
    x, idx, weight = x[:0].clone().requires_grad_(True), idx[:0], weight[:0]
    if lora:
        first = torch.zeros(E, H, 4, device = DEVICE, dtype = torch.bfloat16, requires_grad = True)
        second = torch.zeros(E, 4, 2 * I, device = DEVICE, dtype = torch.bfloat16, requires_grad = True)
        setattr(module, mu.moe_lora_stash_name("gate_up_proj"), (first, second, 2.0, E))
    out = module(x, idx, weight)
    assert out.shape == (0, H) and out.dtype == x.dtype
    out.float().sum().backward()
    assert x.grad is not None and x.grad.shape == (0, H)


class _Wrapper(nn.Module):
    """Stands in for a PEFT ParamWrapper around the stack."""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer


def _tiny_pretrained():
    from transformers import PretrainedConfig, PreTrainedModel

    class _TinyStacked(PreTrainedModel):
        config_class = PretrainedConfig
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.ModuleList([_Block(_experts(seed = i)[0]) for i in range(2)])

        def _init_weights(self, module):
            pass

    mx.patch_save_pretrained_mxfp4()
    return _TinyStacked(PretrainedConfig())


def _saved(path):
    from safetensors.torch import load_file

    return load_file(os.path.join(path, "model.safetensors"))


def test_a_refused_or_failed_full_save_leaves_the_model_as_it_was(tmp_path, monkeypatch):
    import unsloth_zoo.mxfp4_stacked_experts as mse

    model = _tiny_pretrained()
    stacks = [layer.experts for layer in model.layers]
    # A later stack still under LoRA: refused, and the earlier stack is not left swapped.
    model.layers[1].experts = _Wrapper(stacks[1])
    with pytest.raises(RuntimeError, match = "merge_and_unload"):
        model.save_pretrained(str(tmp_path / "wrapped"))
    assert model.layers[0].experts is stacks[0] and model.layers[1].experts.base_layer is stacks[1]
    model.layers[1].experts = stacks[1]
    # A failure while the dense modules are built (out of memory) puts the swapped ones back.
    calls, original = [], mse.dense_expert_modules

    def fail_second(experts, *args, **kwargs):
        calls.append(experts)
        if len(calls) == 2:
            raise torch.OutOfMemoryError("simulated")
        return original(experts, *args, **kwargs)

    monkeypatch.setattr(mse, "dense_expert_modules", fail_second)
    with pytest.raises(torch.OutOfMemoryError):
        model.save_pretrained(str(tmp_path / "oom"))
    assert [layer.experts for layer in model.layers] == stacks


def test_an_explicit_state_dict_is_saved_under_the_checkpoint_names(tmp_path):
    model = _tiny_pretrained()
    model.save_pretrained(str(tmp_path / "implicit"))
    model.save_pretrained(str(tmp_path / "explicit"), state_dict = model.state_dict())
    implicit, explicit = _saved(str(tmp_path / "implicit")), _saved(str(tmp_path / "explicit"))
    assert "layers.0.experts.3.w1.weight" in implicit
    assert explicit.keys() == implicit.keys()
    assert all(torch.equal(explicit[k], implicit[k]) for k in implicit)


def _write_expert_shards(root, ckpt, files, prefix = "model.experts"):
    """``files``: {filename: [checkpoint keys]}; every key of ``ckpt`` must be placed once."""
    from safetensors.torch import save_file

    tensors = {"model.other.weight": torch.ones(4, 4, dtype = torch.bfloat16)}
    for w in ("w1", "w2", "w3"):
        for e, (packed, scale) in enumerate(ckpt[w]):
            tensors[f"{prefix}.{e}.{w}.weight_packed"] = packed
            tensors[f"{prefix}.{e}.{w}.weight_scale"] = scale
    placed = [k for keys in files.values() for k in keys]
    assert sorted(placed) == sorted(tensors), set(tensors) ^ set(placed)
    for filename, keys in files.items():
        save_file({k: tensors[k] for k in keys}, os.path.join(root, filename), metadata = {"format": "pt"})
    return list(files)


def _stack_merge_reference(packed_model, ckpt):
    """{checkpoint key: fp32 decode + the stack's PEFT delta} for every expert weight."""
    deltas = {}
    holder = packed_model.base_model.model.experts
    while hasattr(holder, "base_layer"):
        deltas[holder.parameter_name] = holder.get_delta_weight("default").float().cpu()
        holder = holder.base_layer
    want = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            if w == "w2":
                delta = deltas["down_proj"][e].t()
            else:
                delta = deltas["gate_up_proj"][e, :, slice(0, I) if w == "w1" else slice(I, 2 * I)].t()
            want[f"model.experts.{e}.{w}.weight"] = (_ct_decompress(*ckpt[w][e]) + delta).to(torch.bfloat16)
    return want


def _read_shards(root, filenames):
    from safetensors.torch import load_file

    state = {}
    for filename in filenames:
        part = load_file(os.path.join(root, filename))
        assert not state.keys() & part.keys()
        state.update(part)
    return state


def _all_expert_keys(prefix = "model.experts"):
    return [f"{prefix}.{e}.{w}.weight_{k}" for e in range(E) for w in ("w1", "w2", "w3") for k in ("packed", "scale")]


def test_merged_16bit_rewrite_reads_scales_from_any_shard(tmp_path):
    """A scale stored in another shard than its packed bytes (or in a shard with no packed
    bytes at all) is still found and dropped, whatever order the shards are rewritten in."""
    from unsloth_zoo.saving_utils import _dequantize_compressed_mxfp4_shards

    packed_model, _ = _peft_pair()
    ckpt = _checkpoint_bytes(0)
    key = lambda e, w, kind: f"model.experts.{e}.{w}.weight_{kind}"  # noqa: E731
    everything = lambda e: [key(e, w, k) for w in ("w1", "w2", "w3") for k in ("packed", "scale")]  # noqa: E731
    files = {
        "a.safetensors": ["model.other.weight", *everything(0), *everything(1), key(2, "w1", "scale")],
        "b.safetensors": [k for k in everything(2) if k != key(2, "w1", "scale")] + [
            k for e in range(3, E) for k in everything(e) if k != key(7, "w2", "scale")
        ],
        "c.safetensors": [key(7, "w2", "scale")],
    }
    filenames = _write_expert_shards(str(tmp_path), ckpt, files)
    want = _stack_merge_reference(packed_model, ckpt)
    _dequantize_compressed_mxfp4_shards(str(tmp_path), filenames, {}, packed_model)
    state = _read_shards(str(tmp_path), filenames)
    assert not any(k.endswith(("weight_packed", "weight_scale")) for k in state)
    for k, v in want.items():
        assert torch.equal(state[k], v), k


def test_merged_16bit_rewrite_refuses_before_writing_a_shard(tmp_path):
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import _dequantize_compressed_mxfp4_shards

    packed_model, _ = _peft_pair()
    ckpt = _checkpoint_bytes(0)
    all_keys = _all_expert_keys()
    # Expert 7's w3 is missing from the checkpoint: its LoRA has nowhere to go.
    files = {"a.safetensors": ["model.other.weight"] + all_keys[: len(all_keys) // 2],
             "b.safetensors": all_keys[len(all_keys) // 2 :]}
    filenames = _write_expert_shards(str(tmp_path), ckpt, files)
    save_file(
        {k: v for k, v in _read_shards(str(tmp_path), ["b.safetensors"]).items() if ".7.w3." not in k},
        str(tmp_path / "b.safetensors"), metadata = {"format": "pt"},
    )
    before = {f: (tmp_path / f).read_bytes() for f in filenames}
    with pytest.raises(RuntimeError, match = "would drop it"):
        _dequantize_compressed_mxfp4_shards(str(tmp_path), filenames, {}, packed_model)
    assert {f: (tmp_path / f).read_bytes() for f in filenames} == before
    # A later shard that is not MXFP4 refuses before the first one is rewritten either.
    files = {"a.safetensors": ["model.other.weight"] + all_keys, "b.safetensors": []}
    filenames = _write_expert_shards(str(tmp_path), ckpt, files)
    save_file({"z.weight_packed": torch.zeros(4, 8, dtype = torch.int32),
               "z.weight_scale": torch.zeros(4, 1)}, str(tmp_path / "b.safetensors"))
    before = {f: (tmp_path / f).read_bytes() for f in filenames}
    with pytest.raises(RuntimeError, match = "not an MXFP4 weight"):
        _dequantize_compressed_mxfp4_shards(str(tmp_path), filenames, {}, packed_model)
    assert {f: (tmp_path / f).read_bytes() for f in filenames} == before


def test_merged_16bit_rewrite_keeps_an_adapter_merged_in_memory(tmp_path):
    """After merge_adapter() the stacks' LoRA is in the in-memory weights only; the export rebuilds
    the weights from the checkpoint, so it must fold that adapter in all the same."""
    from unsloth_zoo.saving_utils import _dequantize_compressed_mxfp4_shards

    packed_model, _ = _peft_pair()
    ckpt = _checkpoint_bytes(0)
    want = _stack_merge_reference(packed_model, ckpt)
    filenames = _write_expert_shards(
        str(tmp_path), ckpt, {"model.safetensors": ["model.other.weight"] + _all_expert_keys()}
    )
    with torch.no_grad():
        packed_model.base_model.merge_adapter()
    try:
        _dequantize_compressed_mxfp4_shards(str(tmp_path), filenames, {}, packed_model)
    finally:
        packed_model.base_model.unmerge_adapter()
    state = _read_shards(str(tmp_path), filenames)
    for k, v in want.items():
        assert torch.equal(state[k], v), k


def test_merged_16bit_rewrite_computes_each_stacks_lora_per_shard(tmp_path, monkeypatch):
    """The stacks' LoRA deltas are built as the shards that need them are rewritten, not all
    up front (a real MoE's fp32 deltas do not fit in memory at once)."""
    peft = pytest.importorskip("peft")
    import unsloth_zoo.saving_utils as saving

    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    body = nn.Module()
    body.layers = nn.ModuleList([_Block(_experts(seed = i)[0]) for i in range(2)])
    model = peft.get_peft_model(body, peft.LoraConfig(
        r = 4, lora_alpha = 8, target_modules = [], target_parameters = ["experts.gate_up_proj"],
    ))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    events = []
    real_replace = os.replace

    def replace(a, b):
        events.append(("write", os.path.basename(b)))
        return real_replace(a, b)

    monkeypatch.setattr(saving.os, "replace", replace)
    for layer in (0, 1):
        wrapper = model.base_model.model.layers[layer].experts
        real = wrapper.get_delta_weight

        def delta(adapter, real = real, layer = layer):
            events.append(("delta", layer))
            return real(adapter)

        monkeypatch.setattr(wrapper, "get_delta_weight", delta)
        prefix = f"model.layers.{layer}.experts"
        _write_expert_shards(
            str(tmp_path), _checkpoint_bytes(layer),
            {"other.safetensors": ["model.other.weight"], f"l{layer}.safetensors": _all_expert_keys(prefix)},
            prefix = prefix,
        )
    saving._dequantize_compressed_mxfp4_shards(str(tmp_path), ["l0.safetensors", "l1.safetensors"], {}, model)
    assert events == [("delta", 0), ("write", "l0.safetensors"), ("delta", 1), ("write", "l1.safetensors")]


def test_a_full_save_describes_linears_a_merge_made_dense(tmp_path):
    """Unsloth's per-Linear packed module becomes a plain dense nn.Linear on a PEFT merge (its
    packed bytes kept aside in `_unsloth_mxfp4_packed_state`). A full save must not describe it
    as packed: a bitsandbytes config skips it, an MXFP4 compressed-tensors config is dropped,
    even when no packed module is left to swap."""
    import json
    from transformers import BitsAndBytesConfig

    model = _tiny_pretrained()
    model.layers[0].proj = nn.Linear(H, H, bias = False)
    model.layers[0].proj.__dict__["_unsloth_mxfp4_packed_state"] = ("packed", "scale", torch.bfloat16)
    model.config.quantization_config = BitsAndBytesConfig(load_in_4bit = True, llm_int8_skip_modules = ["lm_head"])
    model.save_pretrained(str(tmp_path / "bnb"))
    skip = json.load(open(tmp_path / "bnb" / "config.json"))["quantization_config"]["llm_int8_skip_modules"]
    assert "layers.0.proj" in skip and "layers.0.experts" in skip
    assert model.config.quantization_config.llm_int8_skip_modules == ["lm_head"]
    # Nothing packed left at all (every stack merged away): still described as dense.
    for layer in model.layers:
        layer.experts = nn.Identity()
    model.config.quantization_config = {
        "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
        "config_groups": {"g": {"format": "mxfp4-pack-quantized", "weights": {"num_bits": 4}}},
    }
    model.save_pretrained(str(tmp_path / "ct"))
    assert "quantization_config" not in json.load(open(tmp_path / "ct" / "config.json"))
    assert model.config.quantization_config["format"] == "mxfp4-pack-quantized"
