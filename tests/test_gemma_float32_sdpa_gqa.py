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

"""Gemma3 / Gemma4 patched attention must keep GQA off the SDPA math kernel (CUDA only)."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level = True)

from torch.nn.attention import SDPBackend, sdpa_kernel

from unsloth_zoo.temporary_patches import utils as patch_utils

HEAD_DIM, SEQ = 256, 256
NEG = torch.finfo(torch.float32).min
_ORIGINAL_SDPA = F.scaled_dot_product_attention
SITES = ["gemma3_forced", "gemma3_generic", "gemma4_forced"]


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _probe(dtype, heads, kv_heads, flash):
    q = torch.empty(1, heads, 8, HEAD_DIM, device = "cuda", dtype = dtype, requires_grad = True)
    kv = torch.empty(1, kv_heads, 8, HEAD_DIM, device = "cuda", dtype = dtype, requires_grad = True)
    try:
        params = torch.backends.cuda.SDPAParams(q, kv, kv, None, 0.0, True, True)
        check = torch.backends.cuda.can_use_flash_attention if flash else torch.backends.cuda.can_use_efficient_attention
        return check(params, False) is True
    except Exception:
        return False


def _efficient_takes_gqa():
    return _probe(torch.float32, 4, 2, False)


def _build(site, monkeypatch):
    calls = []

    def counting_sdpa(query, key, value, *args, enable_gqa = False, **kwargs):
        calls.append(dict(
            enable_gqa = bool(enable_gqa), kv_heads = key.shape[1], q_heads = query.shape[1],
            head_dim = query.shape[-1], masked = kwargs.get("attn_mask") is not None,
        ))
        return _ORIGINAL_SDPA(query, key, value, *args, enable_gqa = enable_gqa, **kwargs)

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0" if site == "gemma3_generic" else "1")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    if site.startswith("gemma3"):
        modeling = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
        from unsloth_zoo.temporary_patches import gemma as patches
        cls = modeling.Gemma3Attention
        patch = patches.patch_Gemma3Attention if site == "gemma3_forced" else patches.patch_Gemma3Attention_generic
        config = modeling.Gemma3TextConfig(
            vocab_size = 128, hidden_size = 256, intermediate_size = 128, num_hidden_layers = 2,
            num_attention_heads = 4, num_key_value_heads = 2, head_dim = HEAD_DIM, sliding_window = 64,
            layer_types = ["sliding_attention", "full_attention"], query_pre_attn_scalar = HEAD_DIM,
        )
        model_cls = modeling.Gemma3ForCausalLM
    else:
        modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        from unsloth_zoo.temporary_patches import gemma4_float32 as patches
        cls = modeling.Gemma4TextAttention
        patch = patches.patch_Gemma4TextAttention
        config = modeling.Gemma4TextConfig(
            vocab_size = 128, vocab_size_per_layer_input = 128, hidden_size = 256, intermediate_size = 128,
            num_hidden_layers = 4, num_attention_heads = 4, num_key_value_heads = 2, head_dim = HEAD_DIM,
            global_head_dim = 2 * HEAD_DIM, sliding_window = 64, layer_types = ["sliding_attention", "full_attention"] * 2,
            num_kv_shared_layers = 2, hidden_size_per_layer_input = 16,
        )
        model_cls = modeling.Gemma4ForCausalLM
    monkeypatch.setattr(cls, "forward", cls.forward)
    # The patch binds SDPA when it is installed, so the counter goes in first.
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", counting_sdpa)
    patch()
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _ORIGINAL_SDPA)
    config._attn_implementation = "sdpa"
    dtype = torch.bfloat16 if site == "gemma3_generic" else torch.float16
    torch.manual_seed(0)
    model = model_cls(config).to("cuda", dtype).train()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.05)
    return model, calls


@pytest.fixture(params = SITES)
def site(request, monkeypatch):
    model, calls = _build(request.param, monkeypatch)
    return request.param, model, calls


def _capture(model, batch, seq, padded):
    captured = []
    hooks = [
        layer.self_attn.register_forward_pre_hook(lambda m, a, k: captured.append((m, a, k)), with_kwargs = True)
        for layer in model.model.layers
    ]
    ids = torch.randint(0, 128, (batch, seq), device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(4))
    keep = torch.ones(batch, seq, dtype = torch.long, device = "cuda")
    if padded:
        keep[-1, : seq // 8] = 0
    try:
        with torch.no_grad():
            model(input_ids = ids, attention_mask = keep, use_cache = False)
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def _run(module, args, kwargs):
    module.zero_grad(set_to_none = True)
    kwargs = dict(kwargs)
    if kwargs.get("shared_kv_states") is not None:
        # Gemma4 producer layers write graph-carrying k/v here; keep each run's graph separate.
        kwargs["shared_kv_states"] = {k: tuple(t.detach() for t in v) for k, v in kwargs["shared_kv_states"].items()}
    hidden = (args[0] if args else kwargs.pop("hidden_states")).detach().clone().requires_grad_()
    out = module(hidden, *args[1:], **kwargs)[0]
    grad = torch.randn(out.shape, device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(1))
    out.float().backward(grad)
    grads = {"x": hidden.grad.float()} | {n: p.grad.float() for n, p in module.named_parameters() if p.grad is not None}
    return out.float().detach(), grads


@pytest.mark.parametrize("padded", [False, True])
def test_efficient_only_matches_math_reference(site, padded):
    name, model, calls = site
    for module, args, kwargs in _capture(model, 2, SEQ, padded):
        with sdpa_kernel([SDPBackend.MATH]):
            ref_out, ref_grads = _run(module, args, kwargs)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out, grads = _run(module, args, kwargs)
        tol = 2e-2 if name == "gemma3_generic" else 2e-3
        assert torch.isfinite(out).all()
        assert _rel(out, ref_out) < tol
        assert "x" in grads and len(grads) >= 2
        for key in grads:
            assert torch.isfinite(grads[key]).all(), key
            assert _rel(grads[key], ref_grads[key]) < tol, key


def test_gqa_not_forwarded_when_efficient_rejects_it(site):
    if _efficient_takes_gqa():
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    name, model, calls = site
    for module, args, kwargs in _capture(model, 2, SEQ, True):
        calls.clear()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            _run(module, args, kwargs)
        assert calls and all(not c["enable_gqa"] and c["kv_heads"] == c["q_heads"] for c in calls), calls


def test_six_argument_sdpa_params_still_repeats_kv(site, monkeypatch):
    # torch 2.4: SDPAParams takes no enable_gqa argument.
    real = torch.backends.cuda.SDPAParams

    def six_args(query, key, value, attn_mask, dropout_p, is_causal, *extra):
        if extra:
            raise TypeError("SDPAParams takes 6 arguments")
        return real(query, key, value, attn_mask, dropout_p, is_causal, False)

    name, model, calls = site
    captured = _capture(model, 2, SEQ, True)
    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", six_args)
    for module, args, kwargs in captured:
        calls.clear()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            _run(module, args, kwargs)
        assert calls and all(not c["enable_gqa"] and c["kv_heads"] == c["q_heads"] for c in calls), calls


def test_gqa_kept_when_flash_takes_it(site):
    name, model, calls = site
    dtype = torch.bfloat16 if name == "gemma3_generic" else torch.float16
    if not torch.backends.cuda.flash_sdp_enabled() or not _probe(dtype, 4, 2, True):
        pytest.skip("flash does not take half precision GQA here")
    # Inside the sliding window nothing needs a mask, and flash cannot take one (nor head_dim 512).
    captured = _capture(model, 1, 48, False)
    calls.clear()
    for module, args, kwargs in captured:
        if name == "gemma3_generic":
            _run(module, args, kwargs)
        else:
            with torch.autocast("cuda", dtype = torch.float16):
                _run(module, args, kwargs)
    flash_calls = [c for c in calls if not c["masked"] and c["head_dim"] == HEAD_DIM]
    assert flash_calls and all(c["enable_gqa"] and c["kv_heads"] < c["q_heads"] for c in flash_calls), calls


def test_gqa_kept_when_efficient_takes_it(site):
    if not _efficient_takes_gqa():
        pytest.skip("the mem-efficient kernel rejects enable_gqa on this torch")
    name, model, calls = site
    for module, args, kwargs in _capture(model, 2, SEQ, True):
        calls.clear()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            _run(module, args, kwargs)
        assert calls and all(c["enable_gqa"] and c["kv_heads"] < c["q_heads"] for c in calls), calls


def _peak_mib(module, args, kwargs, backends):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(backends):
        _run(module, args, kwargs)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def test_peak_memory_without_flash(site):
    if _efficient_takes_gqa():
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    name, model, calls = site
    module, args, kwargs = _capture(model, 1, 4096, False)[1]
    _peak_mib(module, args, kwargs, [SDPBackend.MATH])
    math_peak = _peak_mib(module, args, kwargs, [SDPBackend.MATH])
    head_peak = _peak_mib(module, args, kwargs, [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    assert head_peak < 0.5 * math_peak, (head_peak, math_peak)


def _masks(seq):
    i = torch.arange(seq, device = "cuda")
    causal = i[None, :] <= i[:, None]
    keep = torch.ones(seq, dtype = torch.bool, device = "cuda")
    keep[: seq // 8] = False
    padded = causal & keep[None, :]
    padded = padded | ~padded.any(-1, keepdim = True)
    sliding = causal & (i[None, :] > i[:, None] - seq // 4)
    return {
        "none": None,
        "padded_float": torch.zeros(seq, seq, device = "cuda").masked_fill_(~padded, NEG)[None, None],
        "sliding_bool": sliding[None, None],
    }


@pytest.mark.parametrize("head_dim", [HEAD_DIM, 2 * HEAD_DIM])
@pytest.mark.parametrize("mask_name", list(_masks(8)))
def test_helper_matches_fp64(mask_name, head_dim):
    from unsloth_zoo.temporary_patches.sdpa_gqa import make_gqa_sdpa
    helper = make_gqa_sdpa(_ORIGINAL_SDPA)
    mask = _masks(SEQ)[mask_name]
    causal = mask is None
    q, k, v = (torch.randn(1, h, SEQ, head_dim, device = "cuda", requires_grad = True) for h in (8, 2, 2))
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = helper(q, k, v, mask, 0.0, causal, 1.0, 4)
    grad = torch.randn(out.shape, device = "cuda", dtype = torch.float64, generator = torch.Generator(device = "cuda").manual_seed(3))
    out.backward(grad.float())
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    mask64 = mask.double() if mask is not None and mask.dtype != torch.bool else mask
    with sdpa_kernel([SDPBackend.MATH]):
        ref = _ORIGINAL_SDPA(q64, k64, v64, attn_mask = mask64, is_causal = causal, scale = 1.0, enable_gqa = True)
    ref.backward(grad)
    assert _rel(out, ref) < 1e-5
    for name, mine, theirs in (("q", q.grad, q64.grad), ("k", k.grad, k64.grad), ("v", v.grad, v64.grad)):
        assert _rel(mine, theirs) < 1e-5, name
