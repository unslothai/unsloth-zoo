# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

import copy
import os
import pickle
import subprocess
import sys
import textwrap
from importlib.metadata import version as importlib_version

import pytest
import torch
import torch.nn as nn

import unsloth_zoo.mxfp4_dequant as mxd
from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam, mxfp4_dequantize, mxfp4_dequantize_torch
from unsloth_zoo.temporary_patches import moe_utils as mu
from unsloth_zoo.temporary_patches import mxfp4 as mx
from unsloth_zoo.utils import Version

TRANSFORMERS_5 = Version(importlib_version("transformers")) >= Version("5.0.0")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
needs_triton = pytest.mark.skipif(not mxd._HAS_TRITON, reason = "needs Triton")

_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _reference(blocks, scales, dtype = torch.bfloat16):
    blocks, scales = blocks.cpu(), scales.cpu()
    lut = torch.tensor(_FP4, dtype = dtype)
    *prefix, G, _ = blocks.shape
    out = torch.empty(*prefix, G, 32, dtype = dtype)
    out[..., 0::2] = lut[(blocks & 0x0F).long()]
    out[..., 1::2] = lut[(blocks >> 4).long()]
    out = torch.ldexp(out, (scales.to(torch.int32) - 127).unsqueeze(-1)).to(dtype)
    return out.reshape(*prefix, G * 32).transpose(-2, -1).contiguous()


def _random_mxfp4(E, N, K, device = "cpu", seed = 0, low = 118, high = 125):
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype = torch.uint8, generator = g)
    scales = torch.randint(low, high, (E, N, K // 32), dtype = torch.uint8, generator = g)
    return blocks.to(device), scales.to(device)


def _packed(E, N, K, device = "cpu", seed = 0):
    blocks, scales = _random_mxfp4(E, N, K, device = device, seed = seed)
    return Mxfp4ExpertParam(blocks, mxfp4_scales = scales)


def _bits(t):
    return t.contiguous().view(torch.uint8).cpu()


def test_torch_reference_matches_transformers():
    mxfp4 = pytest.importorskip("transformers.integrations.mxfp4")
    convert = getattr(mxfp4, "_convert_moe_packed_tensors", None)
    if convert is None:
        pytest.skip("this transformers has no _convert_moe_packed_tensors")
    blocks, scales = _random_mxfp4(2, 8, 64, low = 0, high = 255)
    scales.view(-1)[:3] = torch.tensor([0, 127, 254], dtype = torch.uint8)
    assert torch.equal(_bits(mxfp4_dequantize_torch(blocks, scales, transpose = True)), _bits(convert(blocks, scales)))
    assert torch.equal(_bits(mxfp4_dequantize_torch(blocks, scales, transpose = True)), _bits(_reference(blocks, scales)))



@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("chunk_bytes", [1, 5 * 64 * 16 * 8, 1 << 30])
def test_torch_fallback_chunks_are_bit_identical(monkeypatch, transpose, chunk_bytes):
    # Chunk sizes incl. one not dividing N (cut at an expert edge) all match the unchunked result.
    monkeypatch.setattr(mxd, "_TORCH_CHUNK_BYTES", chunk_bytes)
    blocks, scales = _random_mxfp4(3, 7, 64, low = 0, high = 255)
    want = _reference(blocks, scales)
    if not transpose:
        want = want.transpose(-2, -1).contiguous()
    assert torch.equal(_bits(mxfp4_dequantize_torch(blocks, scales, transpose = transpose)), _bits(want))
    out = torch.full_like(want, float("nan"))
    assert mxfp4_dequantize_torch(blocks, scales, transpose = transpose, out = out) is out
    assert torch.equal(_bits(out), _bits(want))
    strided = torch.empty(want.shape[:-1] + (want.shape[-1] * 2,), dtype = want.dtype)[..., ::2]
    mxfp4_dequantize_torch(blocks, scales, transpose = transpose, out = strided)
    assert torch.equal(_bits(strided), _bits(want))
    two_d = mxfp4_dequantize_torch(blocks[0], scales[0], transpose = transpose)
    assert torch.equal(_bits(two_d), _bits(want[0]))


@needs_cuda
def test_torch_fallback_never_materialises_a_whole_stack_of_indices():
    # One 64 MB packed stack: unchunked, its int64 indices alone are 512 MB.
    blocks, scales = _random_mxfp4(8, 1024, 8192, device = "cuda")
    out = torch.empty(8, 8192, 1024, dtype = torch.bfloat16, device = "cuda")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    mxfp4_dequantize_torch(blocks, scales, transpose = True, out = out)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base < 4 * mxd._TORCH_CHUNK_BYTES

@needs_cuda
@needs_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("transpose", [False, True])
def test_fused_kernel_is_bit_identical(dtype, transpose):
    E, N, G = 3, 131, 9
    blocks = torch.randint(0, 256, (E, N, G, 16), dtype = torch.uint8, device = "cuda")
    blocks.view(-1)[:256] = torch.arange(256, dtype = torch.uint8)
    if dtype == torch.float16:
        scales = torch.randint(110, 140, (E, N, G), dtype = torch.uint8, device = "cuda")
    else:
        scales = (torch.arange(E * N * G, device = "cuda") % 256).to(torch.uint8).reshape(E, N, G)
    # Every byte under scale 255 (2^128): ldexp keeps a zero nibble zero rather than 0 * inf = NaN.
    scales.view(-1)[:16] = 255
    want = mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose)
    got = mxd._kernel_dequantize(blocks, scales, dtype, transpose, None, None, None)
    assert got.shape == want.shape and got.dtype == dtype
    assert torch.equal(_bits(got), _bits(want))
    assert not got.isnan().any()
    assert mxd.mxfp4_kernel_available()


@needs_cuda
@needs_triton
@pytest.mark.parametrize("transpose", [False, True])
def test_only_routed_experts_are_written(transpose):
    blocks, scales = _random_mxfp4(4, 64, 128, device = "cuda")
    want = mxfp4_dequantize_torch(blocks, scales, transpose = transpose)
    counts = torch.tensor([3, 0, 1, 0], dtype = torch.int32, device = "cuda")
    for kwargs in (dict(token_counts = counts), dict(experts = torch.tensor([0, 2], device = "cuda"))):
        out = torch.full(want.shape, 7.0, dtype = torch.bfloat16, device = "cuda")
        mxfp4_dequantize(blocks, scales, transpose = transpose, out = out, **kwargs)
        assert torch.equal(out[[0, 2]], want[[0, 2]])
        assert (out[[1, 3]] == 7.0).all()


@needs_cuda
@needs_triton
def test_a_kernel_that_disagrees_with_the_reference_is_not_used(monkeypatch):
    monkeypatch.setattr(mxd, "_KERNEL_VERIFIED", {})
    real = mxd._kernel_dequantize
    monkeypatch.setattr(mxd, "_kernel_dequantize", lambda *a: real(*a).add_(1))
    blocks, scales = _random_mxfp4(2, 64, 64, device = "cuda")
    with pytest.warns(UserWarning, match = "does not match"):
        assert not mxd.mxfp4_kernel_available()
    got = mxfp4_dequantize(blocks, scales, transpose = True)
    assert torch.equal(_bits(got), _bits(_reference(blocks, scales)))


def test_no_triton_imports_and_falls_back(monkeypatch):
    code = textwrap.dedent(f"""
        import sys, importlib.util
        sys.modules["triton"] = None
        spec = importlib.util.spec_from_file_location("m", {mxd.__file__!r})
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        import torch
        b = torch.randint(0, 256, (2, 8, 2, 16), dtype = torch.uint8)
        s = torch.randint(120, 130, (2, 8, 2), dtype = torch.uint8)
        assert not m._HAS_TRITON and not m.mxfp4_kernel_available()
        p = m.Mxfp4ExpertParam(b, mxfp4_scales = s)
        assert torch.equal(p.dequantize(), m.mxfp4_dequantize_torch(b, s, transpose = True))
        print("ok")
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 300)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]
    import transformers.models.gpt_oss.modeling_gpt_oss as modeling_gpt_oss
    monkeypatch.setattr(mx, "mxfp4_kernel_available", lambda *a, **k: False)
    monkeypatch.setattr(mu, "select_moe_backend", lambda: "grouped_mm")
    monkeypatch.setattr(modeling_gpt_oss.GptOssExperts, "_unsloth_lora_patched", True, raising = False)
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-20b")
    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED", raising = False)
    monkeypatch.delenv("UNSLOTH_ENABLE_FULL_FINETUNING", raising = False)
    assert mx.keep_mxfp4_experts_packed() is TRANSFORMERS_5


def test_packed_param_survives_module_casts_and_copies():
    blocks, scales = _random_mxfp4(2, 64, 32)
    param = Mxfp4ExpertParam(blocks, mxfp4_scales = scales)
    assert param._original_shape == (2, 32, 64) and not param.requires_grad
    module = nn.Module()
    module.w = param
    module.to(torch.bfloat16)
    module.half()
    assert module.w.dtype == torch.uint8 and isinstance(module.w, Mxfp4ExpertParam)
    assert torch.equal(copy.deepcopy(param).dequantize(), _reference(blocks, scales))
    rebuilt = type(param)(param.data, requires_grad = False, **param.__dict__)
    assert isinstance(rebuilt, Mxfp4ExpertParam) and rebuilt._original_shape == param._original_shape
    if torch.cuda.is_available():
        module.cuda()
        out = module.w.dequantize()
        assert out.is_cuda and torch.equal(out.cpu(), _reference(blocks, scales))


def _gate_env(monkeypatch, lora_patched = True):
    import transformers.models.gpt_oss.modeling_gpt_oss as modeling_gpt_oss
    monkeypatch.setattr(mu, "select_moe_backend", lambda: "grouped_mm")
    monkeypatch.setattr(mx, "mxfp4_kernel_available", lambda *a, **k: True)
    monkeypatch.setattr(modeling_gpt_oss.GptOssExperts, "_unsloth_lora_patched", lora_patched, raising = False)
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-20b")
    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED", raising = False)
    monkeypatch.delenv("UNSLOTH_ENABLE_FULL_FINETUNING", raising = False)
    monkeypatch.setattr(torch.version, "hip", None)


def test_keep_packed_switch(monkeypatch):
    _gate_env(monkeypatch)
    assert mx.keep_mxfp4_experts_packed() is TRANSFORMERS_5
    if not TRANSFORMERS_5:
        return
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    assert mx.keep_mxfp4_experts_packed() is False
    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED")
    for name in ("unsloth/qwen3-30b-a3b", "unsloth/gpt-oss-20b-unsloth_load_in_4bit_"):
        monkeypatch.setenv("UNSLOTH_MODEL_NAME", name)
        assert mx.keep_mxfp4_experts_packed() is False
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-20b")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "1")
    assert mx.keep_mxfp4_experts_packed() is False
    monkeypatch.delenv("UNSLOTH_ENABLE_FULL_FINETUNING")
    monkeypatch.setattr(mu, "select_moe_backend", lambda: "native_torch")
    assert mx.keep_mxfp4_experts_packed() is False
    monkeypatch.setattr(mu, "select_moe_backend", lambda: "grouped_mm")
    monkeypatch.setattr(mx, "mxfp4_kernel_available", lambda *a, **k: False)
    assert mx.keep_mxfp4_experts_packed() is True
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert mx.keep_mxfp4_experts_packed() is True
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    assert mx.keep_mxfp4_experts_packed() is False
    _gate_env(monkeypatch, lora_patched = False)
    assert mx.keep_mxfp4_experts_packed() is False


@pytest.mark.skipif(not TRANSFORMERS_5, reason = "dequantize_convertops is transformers 5")
def test_dequantize_convertops_keeps_or_dequantizes(monkeypatch):
    import transformers.integrations.mxfp4 as mxfp4
    mx.patch_convert_moe_packed_tensors()
    blocks, scales = _random_mxfp4(2, 64, 96, device = "cuda" if torch.cuda.is_available() else "cpu")
    params = tuple(__import__("inspect").signature(mxfp4.dequantize_convertops).parameters)
    extra = ("cpu",) if params == ("blocks", "scales", "target_device") else ()
    monkeypatch.setattr(mx, "keep_mxfp4_experts_packed", lambda: False)
    dense = mxfp4.dequantize_convertops(blocks, scales, *extra)
    assert not isinstance(dense, Mxfp4ExpertParam) and dense.dtype == torch.bfloat16
    assert torch.equal(_bits(dense), _bits(_reference(blocks, scales)))
    monkeypatch.setattr(mx, "keep_mxfp4_experts_packed", lambda: True)
    packed = mxfp4.dequantize_convertops(blocks, scales, *extra)
    assert isinstance(packed, Mxfp4ExpertParam) and tuple(packed._original_shape) == tuple(dense.shape)
    assert torch.equal(_bits(packed.dequantize()), _bits(dense))


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


class _GptOssExperts(nn.Module):
    def __init__(self, gate_up, down, E, hidden, inter):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.num_experts = E
        self.hidden_size = hidden
        self.alpha = 1.702
        self.limit = 7.0
        self.gate_up_proj = gate_up
        self.down_proj = down
        self.gate_up_proj_bias = nn.Parameter(torch.randn(E, 2 * inter, generator = g).to(torch.bfloat16) * 0.1, requires_grad = False)
        self.down_proj_bias = nn.Parameter(torch.randn(E, hidden, generator = g).to(torch.bfloat16) * 0.1, requires_grad = False)
        self._unsloth_grouped_mm_format = True

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return mu.forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)


# The GPT-OSS activation keys on the class name.
_GptOssExperts.__name__ = "GptOssExperts"


class _Block(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, *args):
        return self.experts(*args)


def _peft_pair(device, E = 4, hidden = 128, inter = 64, rank = 4):
    peft = pytest.importorskip("peft")
    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    gb, gs = _random_mxfp4(E, 2 * inter, hidden, device = device, seed = 1)
    db, ds = _random_mxfp4(E, hidden, inter, device = device, seed = 2)
    models = []
    for packed in (True, False):
        if packed:
            gate_up, down = Mxfp4ExpertParam(gb, mxfp4_scales = gs), Mxfp4ExpertParam(db, mxfp4_scales = ds)
        else:
            gate_up = nn.Parameter(_reference(gb, gs).to(device), requires_grad = False)
            down = nn.Parameter(_reference(db, ds).to(device), requires_grad = False)
        model = _Block(_GptOssExperts(gate_up, down, E, hidden, inter)).to(device)
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


def _route(device, tokens = 48, E = 4, top_k = 2, hidden = 128, seed = 7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator = g).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(E - 1, generator = g)[:top_k] for _ in range(tokens)])
    w = torch.softmax(torch.randn(tokens, top_k, generator = g), dim = -1).to(torch.bfloat16)
    return x.to(device), idx.to(device), w.to(device)


def _train_step(model, x, idx, w, checkpoint):
    xi = x.clone().requires_grad_(True)

    def block(h):
        return model(h, idx, w)

    if checkpoint:
        from unsloth_zoo.gradient_checkpointing import unsloth_offloaded_gradient_checkpoint
        out = unsloth_offloaded_gradient_checkpoint(block, xi)
    else:
        out = block(xi)
    out.float().pow(2).mean().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    grads["x"] = xi.grad.clone()
    return out.detach(), grads


@pytest.mark.parametrize("checkpoint", [False, True])
def test_packed_experts_train_like_the_load_time_dequant(monkeypatch, grouped_mm_device, checkpoint):
    if checkpoint and grouped_mm_device != "cuda":
        pytest.skip("Unsloth gradient checkpointing offloads to a CUDA device")
    packed_model, dense_model = _peft_pair(grouped_mm_device)
    dense_shapes = {n: p.shape for n, p in dense_model.named_parameters() if "lora_" in n}
    lora = {n: p.shape for n, p in packed_model.named_parameters() if "lora_" in n}
    assert lora == dense_shapes and len(lora) == 4
    # The packed stack must take the separated LoRA path, never PEFT's own `param + delta`.
    peft_calls = []
    original = mu._original_param_wrapper_forward
    monkeypatch.setattr(mu, "_original_param_wrapper_forward", lambda *a, **k: peft_calls.append(1) or original(*a, **k))
    x, idx, w = _route(grouped_mm_device)
    out_p, grads_p = _train_step(packed_model, x, idx, w, checkpoint)
    out_d, grads_d = _train_step(dense_model, x, idx, w, checkpoint)
    assert peft_calls == []
    assert torch.equal(out_p, out_d)
    assert set(grads_p) == set(grads_d) and len(grads_p) == 5
    for name in grads_p:
        assert torch.isfinite(grads_p[name]).all() and grads_p[name].abs().sum() > 0, name
        assert torch.equal(grads_p[name], grads_d[name]), name
    experts = packed_model.base_model.model.experts
    while hasattr(experts, "base_layer"):
        experts = experts.base_layer
    assert isinstance(experts.gate_up_proj, Mxfp4ExpertParam)


def test_a_4d_packed_stack_is_an_experts_module():
    # Regression: a 4-D uint8 stack made PEFT's ParamWrapper add a bf16 delta to the packed blocks.
    blocks, scales = _random_mxfp4(4, 128, 64)
    experts = _GptOssExperts(Mxfp4ExpertParam(blocks, mxfp4_scales = scales), _packed(4, 64, 64), 4, 64, 64)
    assert mu._is_moe_experts_module(experts)
    assert not mu._can_fold_moe_lora_through_peft(experts, "gate_up_proj")
    assert mu._get_param_shape_from_module(experts, "gate_up_proj") == (4, 64, 128)
    assert mu._base_is_recomputable(experts.gate_up_proj) == (mu._should_use_separated_lora() and mu._check_torch_grouped_mm_supported())


@needs_cuda
@pytest.mark.parametrize("override, per_projection", [(None, 2), ("1", 3)])
def test_checkpoint_recompute_reuses_its_dequant_in_backward(monkeypatch, override, per_projection):
    """The backward reuses the recompute's dequant; UNSLOTH_MOE_RECOMPUTE=1 rebuilds it instead."""
    if not mu._check_torch_grouped_mm_supported():
        pytest.skip("no torch._grouped_mm on this device")
    if override is None:
        monkeypatch.delenv("UNSLOTH_MOE_RECOMPUTE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_MOE_RECOMPUTE", override)
    from unsloth_zoo.gradient_checkpointing import in_gradient_checkpoint_recompute
    packed_model, _ = _peft_pair("cuda")
    x, idx, w = _route("cuda")
    _train_step(packed_model, x, idx, w, checkpoint = True)
    calls = []
    original = Mxfp4ExpertParam.dequantize

    def counted(self, *args, **kwargs):
        if in_gradient_checkpoint_recompute():
            calls.append("recompute")
        elif torch._C._current_graph_task_id() != -1:
            calls.append("backward")
        else:
            calls.append("forward")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Mxfp4ExpertParam, "dequantize", counted)
    _train_step(packed_model, x, idx, w, checkpoint = True)
    assert len(calls) == 2 * per_projection
    assert calls.count("forward") == 2 and calls.count("recompute") == 2
    assert calls.count("backward") == 2 * (per_projection - 2)


def test_merge_and_unmerge(grouped_mm_device):
    packed_model, dense_model = _peft_pair(grouped_mm_device)
    wrapper = packed_model.base_model.model.experts
    base = wrapper
    while hasattr(base, "base_layer"):
        base = base.base_layer
    packed = {n: getattr(base, n) for n in ("gate_up_proj", "down_proj")}
    packed_model.base_model.merge_adapter()
    for name in packed:
        assert not isinstance(getattr(base, name), Mxfp4ExpertParam)
    packed_model.base_model.unmerge_adapter()
    for name in packed:
        assert getattr(base, name) is packed[name]
    merged = [m.merge_and_unload() for m in (packed_model, dense_model)]
    for name in packed:
        a, b = (getattr(m.experts, name) for m in merged)
        assert not isinstance(a, Mxfp4ExpertParam) and a.dtype == torch.bfloat16
        assert torch.equal(a, b)


def test_unmerge_after_a_model_move_restores_on_the_current_device(grouped_mm_device):
    packed_model, _ = _peft_pair(grouped_mm_device)
    base = packed_model.base_model.model.experts
    while hasattr(base, "base_layer"):
        base = base.base_layer
    want = {n: getattr(base, n).dequantize() for n in ("gate_up_proj", "down_proj")}
    packed_model.base_model.merge_adapter()
    packed_model.to("meta")
    packed_model.base_model.unmerge_adapter()
    for name in want:
        param = getattr(base, name)
        assert isinstance(param, Mxfp4ExpertParam)
        assert param.is_meta and param.mxfp4_scales.is_meta
    if grouped_mm_device == "cuda":
        packed_model, _ = _peft_pair(grouped_mm_device)
        base = packed_model.base_model.model.experts
        while hasattr(base, "base_layer"):
            base = base.base_layer
        packed_model.base_model.merge_adapter()
        packed_model.to("cpu")
        packed_model.base_model.unmerge_adapter()
        for name in want:
            param = getattr(base, name)
            assert param.device.type == "cpu"
            assert torch.equal(param.dequantize(), want[name].cpu())


def _tiny_gpt_oss(device):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GptOssConfig"):
        pytest.skip("no GptOss in this transformers")
    config = transformers.GptOssConfig(
        vocab_size = 128, hidden_size = 128, intermediate_size = 64, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 64, num_local_experts = 4,
        num_experts_per_tok = 2, layer_types = ["full_attention"], max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    model = transformers.GptOssForCausalLM(config).to(torch.bfloat16).to(device)
    experts = model.model.layers[0].mlp.experts
    E, H, I = 4, 128, 64
    gate_up = _packed(E, 2 * I, H, device = device, seed = 11)
    down = _packed(E, H, I, device = device, seed = 12)
    experts.gate_up_proj, experts.down_proj = gate_up, down
    return model, experts


@pytest.mark.parametrize("explicit_state_dict", [False, True, "positional"])
def test_full_save_writes_dequantized_experts(tmp_path, explicit_state_dict):
    safetensors = pytest.importorskip("safetensors.torch")
    mx.patch_save_pretrained_mxfp4()
    model, experts = _tiny_gpt_oss("cpu")
    packed = experts.gate_up_proj, experts.down_proj
    if explicit_state_dict == "positional":
        model.save_pretrained(tmp_path, True, model.state_dict())
    else:
        kwargs = {}
        if explicit_state_dict:
            kwargs["state_dict"] = model.state_dict()
        model.save_pretrained(tmp_path, **kwargs)
    saved = {}
    for file in os.listdir(tmp_path):
        if file.endswith(".safetensors"):
            saved.update(safetensors.load_file(os.path.join(tmp_path, file)))
    for name, param in zip(("gate_up_proj", "down_proj"), packed):
        weight = saved[f"model.layers.0.mlp.experts.{name}"]
        assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == tuple(param._original_shape)
        assert torch.equal(weight, param.dequantize())
    assert experts.gate_up_proj is packed[0] and experts.down_proj is packed[1]


@pytest.mark.skipif(not TRANSFORMERS_5, reason = "weight conversions are transformers 5")
def test_full_save_of_a_loaded_model_writes_experts_under_their_own_names(tmp_path):
    """Reversed converters: 5.4 raised NotImplementedError, 5.17 wrote "gate_up_proj$"."""
    safetensors = pytest.importorskip("safetensors.torch")
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.mxfp4 import Mxfp4Dequantize
    mx.patch_save_pretrained_mxfp4()
    model, experts = _tiny_gpt_oss("cpu")
    conversions = [
        WeightConverter(
            source_patterns = [f"{proj}_blocks", f"{proj}_scales"], target_patterns = rf"{proj}$",
            operations = [Mxfp4Dequantize(None)],
        )
        for proj in ("down_proj", "gate_up_proj")
    ]
    model._weight_conversions = conversions   # as a load of a pre-quantized checkpoint leaves it
    packed = experts.gate_up_proj, experts.down_proj
    model.save_pretrained(tmp_path)
    saved = safetensors.load_file(os.path.join(tmp_path, "model.safetensors"))
    assert not any(key.endswith("$") for key in saved)
    for name, param in zip(("gate_up_proj", "down_proj"), packed):
        assert torch.equal(saved[f"model.layers.0.mlp.experts.{name}"], param.dequantize())
    assert "quantization_config" not in open(os.path.join(tmp_path, "config.json")).read()
    assert model._weight_conversions is conversions
    assert experts.gate_up_proj is packed[0] and experts.down_proj is packed[1]


def test_adapter_save_never_dequantizes(tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    mx.patch_peft_param_wrapper_mxfp4()
    mx.patch_save_pretrained_mxfp4()
    model, _ = _tiny_gpt_oss("cpu")
    model = peft.get_peft_model(model, peft.LoraConfig(
        r = 4, target_modules = ["q_proj"], target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    ))
    calls = []
    monkeypatch.setattr(Mxfp4ExpertParam, "dequantize", lambda self, *a, **k: calls.append(1))
    model.save_pretrained(tmp_path)
    from safetensors.torch import load_file
    saved = load_file(os.path.join(tmp_path, "adapter_model.safetensors"))
    assert saved and all("lora_" in key for key in saved)
    assert any("experts" in key for key in saved)
    assert calls == []


@needs_cuda
def test_bf16_inference_path_dequantizes_packed_experts():
    from unsloth_zoo.temporary_patches import gpt_oss
    model, experts = _tiny_gpt_oss("cuda")
    mlp = model.model.layers[0].mlp
    g = torch.Generator(device = "cuda").manual_seed(13)
    inputs = [torch.randn(1, 1, 128, device = "cuda", dtype = torch.bfloat16, generator = g) for _ in range(4)]
    inputs.append(torch.randn(1, 5, 128, device = "cuda", dtype = torch.bfloat16, generator = g))
    with torch.no_grad():
        got = [gpt_oss.moe_forward_inference_bf16(mlp, h).clone() for h in inputs]
        packed = experts.gate_up_proj, experts.down_proj
        experts.gate_up_proj = nn.Parameter(packed[0].dequantize(), requires_grad = False)
        experts.down_proj = nn.Parameter(packed[1].dequantize(), requires_grad = False)
        want = [gpt_oss.moe_forward_inference_bf16(mlp, h).clone() for h in inputs]
    for a, b in zip(got, want):
        assert torch.isfinite(a).all() and torch.equal(a, b)


def test_decode_stacks_are_shared_and_freed_with_their_model():
    import gc
    import weakref
    from unsloth_zoo.temporary_patches import gpt_oss

    first, second = _packed(4, 128, 64, seed = 1), _packed(4, 128, 64, seed = 2)
    counts = torch.ones(4, dtype = torch.int32)
    stack = gpt_oss._mxfp4_decode_stack(first, torch.bfloat16, counts)
    assert gpt_oss._mxfp4_decode_stack(second, torch.bfloat16, counts) is stack
    assert torch.equal(stack, second.dequantize(torch.bfloat16))
    alive = weakref.ref(stack)
    del stack, first
    gc.collect()
    assert alive() is not None
    del second
    gc.collect()
    assert alive() is None


@pytest.mark.skipif(not TRANSFORMERS_5, reason = "packing is transformers 5 only")
def test_offloaded_loads_keep_the_load_time_dequant(monkeypatch):
    import transformers.modeling_utils as modeling_utils

    _gate_env(monkeypatch)
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "1")
    monkeypatch.setattr(mx, "_LOAD_OFFLOADS", [False])
    assert mx.keep_mxfp4_experts_packed() is True
    for device_map, offloads in (({"model.layers.0": 0, "model.layers.1": "cpu"}, True),
                                 ({"model.layers.0": 0, "lm_head": "disk"}, True),
                                 ({"": 0}, False)):
        monkeypatch.setattr(modeling_utils, "_get_device_map", lambda *a, _m = device_map, **k: _m)
        mx.patch_mxfp4_offload_guard()
        assert modeling_utils._get_device_map(None, "auto", None, None) == device_map
        assert mx._LOAD_OFFLOADS[0] is offloads
        assert mx.keep_mxfp4_experts_packed() is (not offloads)


def test_the_decode_lock_is_held_until_the_kernel_is_enqueued(monkeypatch):
    from unsloth_zoo.temporary_patches import gpt_oss

    model, experts = _tiny_gpt_oss("cpu")
    mlp = model.model.layers[0].mlp
    seen = []

    def kernel(hidden_states, routing_weights, gate_up, gate_up_bias, down, *rest):
        slots = [s for s in gpt_oss._MXFP4_DECODE_SLOTS.values() if s.stack is gate_up or s.stack is down]
        seen.append((len(slots), all(s.lock.locked() for s in slots)))
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(gpt_oss, "_moe_forward_inference_bf16_kernel", kernel)
    with torch.no_grad():
        gpt_oss.moe_forward_inference_bf16(mlp, torch.randn(1, 1, 128, dtype = torch.bfloat16))
    assert seen == [(2, True)]
    assert not any(s.lock.locked() for s in gpt_oss._MXFP4_DECODE_SLOTS.values())


def test_stream_events_cover_every_asynchronous_backend(monkeypatch):
    from unsloth_zoo.temporary_patches import gpt_oss

    calls = []

    class FakeEvent:
        def record(self, stream):
            calls.append(("record", self, stream))

    class FakeStream:
        def wait_event(self, event):
            calls.append(("wait", event, self))

    stream = FakeStream()

    class FakeApi:
        Event = FakeEvent

        @staticmethod
        def current_stream(device = None):
            return stream

    assert gpt_oss._device_stream_api(torch.device("cpu")) is None
    assert gpt_oss._device_stream_api(torch.device("cuda")) is torch.cuda
    assert gpt_oss._device_stream_api(torch.device("xpu")) is torch.xpu

    model, experts = _tiny_gpt_oss("cpu")
    mlp = model.model.layers[0].mlp
    monkeypatch.setattr(gpt_oss, "_device_stream_api", lambda device: FakeApi)
    monkeypatch.setattr(gpt_oss, "_moe_forward_inference_bf16_kernel", lambda h, *rest: torch.zeros_like(h))
    h = torch.randn(1, 1, 128, dtype = torch.bfloat16)
    with torch.no_grad():
        gpt_oss.moe_forward_inference_bf16(mlp, h)
        recorded = [c[1] for c in calls if c[0] == "record"]
        assert len(recorded) == 2 and all(c[2] is stream for c in calls)
        calls.clear()
        gpt_oss.moe_forward_inference_bf16(mlp, h)
    waited = [c[1] for c in calls if c[0] == "wait"]
    assert sorted(map(id, waited)) == sorted(map(id, recorded))


def test_projections_of_the_same_shape_do_not_share_a_stack():
    from unsloth_zoo.temporary_patches import gpt_oss

    a, b = _packed(4, 64, 64, seed = 1), _packed(4, 64, 64, seed = 2)
    counts = torch.ones(4, dtype = torch.int32)
    gate_up = gpt_oss._mxfp4_decode_stack(a, torch.bfloat16, counts, role = "gate_up")
    down = gpt_oss._mxfp4_decode_stack(b, torch.bfloat16, counts, role = "down")
    assert gate_up is not down
    assert torch.equal(gate_up, a.dequantize(torch.bfloat16))


@needs_cuda
def test_another_stream_waits_for_the_last_kernel_on_the_shared_stack():
    from unsloth_zoo.temporary_patches import gpt_oss

    model, experts = _tiny_gpt_oss("cuda")
    mlp = model.model.layers[0].mlp
    h = torch.randn(1, 1, 128, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        want = gpt_oss.moe_forward_inference_bf16(mlp, h).clone()
        slots = list(gpt_oss._MXFP4_DECODE_SLOTS.values())
        assert slots and all(s.event is not None for s in slots)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            got = gpt_oss.moe_forward_inference_bf16(mlp, h)
        side.synchronize()
    assert sorted(map(id, gpt_oss._MXFP4_DECODE_SLOTS.values())) == sorted(map(id, slots))
    assert torch.equal(got, want)


def test_module_moves_to_and_from_meta_keep_the_packed_param():
    blocks, scales = _random_mxfp4(2, 64, 64)
    module = nn.Module()
    module.w = Mxfp4ExpertParam(blocks, mxfp4_scales = scales)
    want = module.w.dequantize()
    module.to("meta")
    assert isinstance(module.w, Mxfp4ExpertParam) and module.w.is_meta
    assert module.w.mxfp4_scales.is_meta and tuple(module.w._original_shape) == tuple(want.shape)
    detached = Mxfp4ExpertParam(blocks, mxfp4_scales = scales).detach()
    assert isinstance(detached, Mxfp4ExpertParam) and torch.equal(detached.dequantize(), want)


@pytest.mark.skipif(not TRANSFORMERS_5, reason = "packing is transformers 5 only")
def test_a_later_load_without_a_device_map_clears_the_offload_flag(monkeypatch):
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

    monkeypatch.setattr(Mxfp4HfQuantizer, "validate_environment", lambda self, *a, **k: None)
    mx.patch_mxfp4_offload_guard()
    monkeypatch.setattr(mx, "_LOAD_OFFLOADS", [True])
    Mxfp4HfQuantizer.validate_environment(object.__new__(Mxfp4HfQuantizer), device_map = None)
    assert mx._LOAD_OFFLOADS[0] is False


def test_load_state_dict_carries_the_scales():
    """load_state_dict takes scales with the blocks and refuses bare uint8 blocks."""
    a, b = nn.Module(), nn.Module()
    a.w, b.w = _packed(4, 64, 64, seed = 1), _packed(4, 64, 64, seed = 2)
    assert not torch.equal(a.w.mxfp4_scales, b.w.mxfp4_scales)
    target = b.w
    b.load_state_dict(a.state_dict())
    assert b.w is target and isinstance(b.w, Mxfp4ExpertParam)
    assert torch.equal(b.w.dequantize(), a.w.dequantize())
    assert b.w.mxfp4_scales.data_ptr() != a.w.mxfp4_scales.data_ptr()   # copied, not aliased
    c = nn.Module()
    c.w = _packed(4, 64, 64, seed = 3)
    c.load_state_dict(pickle.loads(pickle.dumps(a.state_dict())))
    assert torch.equal(c.w.dequantize(), a.w.dequantize())
    with pytest.raises(RuntimeError, match = "bare MXFP4 blocks"):
        c.load_state_dict({"w": _packed(4, 64, 64, seed = 4).data.clone()})
    assert torch.equal(c.w.dequantize(), a.w.dequantize())


def test_load_state_dict_of_a_gpt_oss_model_takes_the_other_models_experts():
    a, experts_a = _tiny_gpt_oss("cpu")
    b, experts_b = _tiny_gpt_oss("cpu")
    experts_b.gate_up_proj = _packed(4, 128, 128, seed = 21)
    experts_b.down_proj = _packed(4, 128, 64, seed = 22)
    b.load_state_dict(a.state_dict())
    for name in ("gate_up_proj", "down_proj"):
        assert torch.equal(getattr(experts_b, name).dequantize(), getattr(experts_a, name).dequantize())
