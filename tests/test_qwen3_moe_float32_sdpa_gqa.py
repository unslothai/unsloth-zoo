# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Forced-float32 Qwen3-MoE attention must keep GQA off the SDPA math kernel (CUDA only)."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level = True)

modeling = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
from torch.nn.attention import SDPBackend, sdpa_kernel

from unsloth_zoo.temporary_patches import qwen3_moe_float32 as patches
from unsloth_zoo.temporary_patches import utils as patch_utils

HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 16, 4, 64, 512
NEG = torch.finfo(torch.float32).min
_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _efficient_takes_gqa():
    q = torch.empty(1, HEADS, 8, HEAD_DIM, device = "cuda", requires_grad = True)
    kv = torch.empty(1, KV_HEADS, 8, HEAD_DIM, device = "cuda", requires_grad = True)
    try:
        params = torch.backends.cuda.SDPAParams(q, kv, kv, None, 0.0, True, True)
        return torch.backends.cuda.can_use_efficient_attention(params, False) is True
    except Exception:
        return False


def _flash_takes_fp16_gqa():
    q = torch.empty(1, HEADS, 8, HEAD_DIM, device = "cuda", dtype = torch.float16, requires_grad = True)
    kv = torch.empty(1, KV_HEADS, 8, HEAD_DIM, device = "cuda", dtype = torch.float16, requires_grad = True)
    try:
        params = torch.backends.cuda.SDPAParams(q, kv, kv, None, 0.0, True, True)
        return torch.backends.cuda.can_use_flash_attention(params, False) is True
    except Exception:
        return False


@pytest.fixture
def attn(monkeypatch):
    calls = []

    def counting_sdpa(query, key, value, *args, enable_gqa = False, **kwargs):
        calls.append(dict(enable_gqa = bool(enable_gqa), kv_heads = key.shape[1], q_heads = query.shape[1]))
        return _ORIGINAL_SDPA(query, key, value, *args, enable_gqa = enable_gqa, **kwargs)

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    monkeypatch.setattr(modeling.Qwen3MoeAttention, "forward", modeling.Qwen3MoeAttention.forward)
    # The patch binds SDPA when it is installed, so the counter goes in first.
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", counting_sdpa)
    patches.patch_Qwen3MoeAttention_float32()
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _ORIGINAL_SDPA)

    config = modeling.Qwen3MoeConfig(
        vocab_size = 128, hidden_size = HIDDEN, intermediate_size = 256, moe_intermediate_size = 64,
        num_hidden_layers = 1, num_attention_heads = HEADS, num_key_value_heads = KV_HEADS, head_dim = HEAD_DIM,
        num_experts = 4, num_experts_per_tok = 2, attention_dropout = 0.0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    module = modeling.Qwen3MoeAttention(config, layer_idx = 0).cuda().half().train()
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.05) if p.dim() > 1 else p.fill_(1.0)
    rope = modeling.Qwen3MoeRotaryEmbedding(config).cuda()
    return module, rope, calls


def _masks(batch, seq):
    i = torch.arange(seq, device = "cuda")
    causal = i[None, :] <= i[:, None]
    keep = torch.ones(batch, seq, dtype = torch.bool, device = "cuda")
    keep[-1, : seq // 8] = False
    padded = causal[None, None] & keep[:, None, None, :]
    padded = padded | ~padded.any(-1, keepdim = True)  # transformers unmasks fully padded rows for SDPA
    doc = torch.repeat_interleave(torch.arange(3, device = "cuda"), torch.tensor([seq // 4, seq // 2, seq - 3 * seq // 4], device = "cuda"))
    packed = (causal & (doc[None, :] == doc[:, None]))[None, None].expand(batch, 1, seq, seq)
    sliding = (causal & (i[None, :] > i[:, None] - seq // 4))[None, None].expand(batch, 1, seq, seq)
    to_float = lambda m: torch.zeros(m.shape, device = "cuda").masked_fill_(~m, NEG)
    return {
        "none": None, "causal_bool": causal[None, None].expand(batch, 1, seq, seq),
        "padded_bool": padded, "padded_float": to_float(padded),
        "packed_bool": packed, "packed_float": to_float(packed), "sliding_bool": sliding,
    }


def _run(module, rope, hidden, mask):
    module.zero_grad(set_to_none = True)
    x = hidden.clone().requires_grad_()
    position_embeddings = rope(x, torch.arange(x.shape[1], device = "cuda")[None].expand(x.shape[0], -1))
    out, _ = module(x, position_embeddings = position_embeddings, attention_mask = mask)
    grad = torch.randn(out.shape, device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(1))
    out.float().backward(grad)
    grads = {"x": x.grad.float()} | {n: p.grad.float() for n, p in module.named_parameters()}
    return out.float().detach(), grads


@pytest.mark.parametrize("mask_name", list(_masks(2, 256)))
def test_efficient_only_matches_math_reference(attn, mask_name):
    module, rope, calls = attn
    mask = _masks(2, 256)[mask_name]
    hidden = torch.randn(2, 256, HIDDEN, device = "cuda", dtype = torch.float16, generator = torch.Generator(device = "cuda").manual_seed(2))
    with sdpa_kernel([SDPBackend.MATH]):
        ref_out, ref_grads = _run(module, rope, hidden, mask)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out, grads = _run(module, rope, hidden, mask)
    assert torch.isfinite(out).all()
    assert _rel(out, ref_out) < 1e-3
    assert len(grads) == 7
    for name in grads:
        assert torch.isfinite(grads[name]).all(), name
        assert _rel(grads[name], ref_grads[name]) < 2e-3, name


def test_gqa_not_forwarded_when_efficient_rejects_it(attn):
    if _efficient_takes_gqa():
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    module, rope, calls = attn
    hidden = torch.randn(1, 128, HIDDEN, device = "cuda", dtype = torch.float16)
    calls.clear()
    _run(module, rope, hidden, None)
    assert calls and all(not c["enable_gqa"] and c["kv_heads"] == c["q_heads"] for c in calls), calls


def test_six_argument_sdpa_params_still_repeats_kv(attn, monkeypatch):
    # torch 2.4: SDPAParams takes no enable_gqa argument.
    real = torch.backends.cuda.SDPAParams

    def six_args(query, key, value, attn_mask, dropout_p, is_causal, *extra):
        if extra:
            raise TypeError("SDPAParams takes 6 arguments")
        return real(query, key, value, attn_mask, dropout_p, is_causal, False)

    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", six_args)
    module, rope, calls = attn
    hidden = torch.randn(1, 128, HIDDEN, device = "cuda", dtype = torch.float16)
    calls.clear()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        _run(module, rope, hidden, None)
    assert calls and all(not c["enable_gqa"] and c["kv_heads"] == c["q_heads"] for c in calls), calls


def test_gqa_kept_when_flash_takes_it_under_autocast(attn):
    if not torch.backends.cuda.flash_sdp_enabled() or not _flash_takes_fp16_gqa():
        pytest.skip("flash does not take fp16 GQA here")
    module, rope, calls = attn
    hidden = torch.randn(1, 128, HIDDEN, device = "cuda", dtype = torch.float16)
    calls.clear()
    with torch.autocast("cuda", dtype = torch.float16):
        _run(module, rope, hidden, None)
    assert calls and all(c["enable_gqa"] and c["kv_heads"] == KV_HEADS for c in calls), calls


def test_gqa_kept_when_efficient_takes_it(attn):
    if not _efficient_takes_gqa():
        pytest.skip("the mem-efficient kernel rejects enable_gqa on this torch")
    module, rope, calls = attn
    hidden = torch.randn(1, 128, HIDDEN, device = "cuda", dtype = torch.float16)
    calls.clear()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        _run(module, rope, hidden, None)
    assert calls and all(c["enable_gqa"] and c["kv_heads"] == KV_HEADS for c in calls), calls


def _peak_mib(module, rope, hidden, backends):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(backends):
        _run(module, rope, hidden, None)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def test_peak_memory_without_flash(attn):
    if _efficient_takes_gqa():
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    module, rope, calls = attn
    hidden = torch.randn(1, 2048, HIDDEN, device = "cuda", dtype = torch.float16)
    _peak_mib(module, rope, hidden, [SDPBackend.MATH])
    math_peak = _peak_mib(module, rope, hidden, [SDPBackend.MATH])
    head_peak = _peak_mib(module, rope, hidden, [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    assert head_peak < 0.4 * math_peak, (head_peak, math_peak)


def _helper():
    forward = modeling.Qwen3MoeAttention.forward
    cells = dict(zip(forward.__code__.co_freevars, (c.cell_contents for c in forward.__closure__ or ())))
    return cells.get("float32_sdpa")


@pytest.mark.parametrize("mask_name", ["none", "padded_float", "packed_bool", "sliding_bool"])
def test_attention_op_matches_fp64(attn, mask_name):
    helper = _helper()
    mask = _masks(1, 256)[mask_name]
    mask = mask if mask is None or mask.dtype == torch.bool else mask.float()
    causal, scale = mask is None, HEAD_DIM**-0.5
    q, k, v = (torch.randn(1, h, 256, HEAD_DIM, device = "cuda", requires_grad = True) for h in (HEADS, KV_HEADS, KV_HEADS))
    if helper is not None:
        out = helper(q, k, v, mask, 0.0, causal, scale, HEADS // KV_HEADS)
    else:
        out = _ORIGINAL_SDPA(q, k, v, attn_mask = mask, is_causal = causal, scale = scale, enable_gqa = True)
    grad = torch.randn(out.shape, device = "cuda", dtype = torch.float64, generator = torch.Generator(device = "cuda").manual_seed(3))
    out.backward(grad.float())
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    mask64 = mask.double() if mask is not None and mask.dtype != torch.bool else mask
    with sdpa_kernel([SDPBackend.MATH]):
        ref = _ORIGINAL_SDPA(q64, k64, v64, attn_mask = mask64, is_causal = causal, scale = scale, enable_gqa = True)
    ref.backward(grad)
    assert _rel(out, ref) < 1e-5
    for name, mine, theirs in (("q", q.grad, q64.grad), ("k", k.grad, k64.grad), ("v", v.grad, v64.grad)):
        assert _rel(mine, theirs) < 1e-5, name
