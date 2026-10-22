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

"""make_gqa_sdpa must not repeat k/v when SDPA would run the enable_gqa call on cuDNN (CUDA only)."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level = True)

from torch.nn.attention import SDPBackend, sdpa_kernel

from unsloth_zoo.temporary_patches import sdpa_gqa

B, H, H_KV, SEQ, HEAD_DIM, WINDOW = 2, 8, 2, 512, 128, 128
_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _inputs(dtype = torch.bfloat16):
    g = torch.Generator(device = "cuda").manual_seed(0)
    q = torch.randn(B, H, SEQ, HEAD_DIM, device = "cuda", dtype = dtype, generator = g).requires_grad_()
    k = torch.randn(B, H_KV, SEQ, HEAD_DIM, device = "cuda", dtype = dtype, generator = g).requires_grad_()
    v = torch.randn(B, H_KV, SEQ, HEAD_DIM, device = "cuda", dtype = dtype, generator = g).requires_grad_()
    i = torch.arange(SEQ, device = "cuda")
    allowed = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < WINDOW)
    mask = torch.zeros(SEQ, SEQ, device = "cuda", dtype = dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
    return q, k, v, mask[None, None].expand(B, 1, SEQ, SEQ).contiguous()


def _cudnn_takes_native_gqa():
    q, k, v, mask = _inputs()
    try:
        choice = torch._fused_sdp_choice(q, k, v, mask, 0.0, False, scale = None, enable_gqa = True)
    except Exception:
        return False
    return choice == int(SDPBackend.CUDNN_ATTENTION)


def _counting_helper():
    calls = []

    def counting_sdpa(query, key, value, *args, enable_gqa = False, **kwargs):
        calls.append(dict(enable_gqa = bool(enable_gqa), kv_heads = key.shape[1], q_heads = query.shape[1]))
        return _ORIGINAL_SDPA(query, key, value, *args, enable_gqa = enable_gqa, **kwargs)

    return sdpa_gqa.make_gqa_sdpa(counting_sdpa), calls


def _repeat(t):
    b, h_kv, s, d = t.shape
    return t[:, :, None].expand(b, h_kv, H // H_KV, s, d).reshape(b, H, s, d)


def _run(fn, q, k, v):
    for t in (q, k, v):
        t.grad = None
    out = fn(q, k, v)
    out.backward(torch.ones_like(out))
    return out.detach().clone(), q.grad.clone(), k.grad.clone(), v.grad.clone()


@pytest.fixture
def cudnn_sdp_on():
    # Importing vLLM turns cuDNN SDPA off process wide, so turn it on here and restore it after.
    was = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    yield
    torch.backends.cuda.enable_cudnn_sdp(was)


@pytest.fixture
def cudnn_gqa(cudnn_sdp_on):
    if not _cudnn_takes_native_gqa():
        pytest.skip("SDPA does not pick cuDNN for masked bf16 GQA on this torch / GPU / cuDNN")


def test_cudnn_gqa_call_is_not_repeated(cudnn_gqa):
    helper, calls = _counting_helper()
    q, k, v, mask = _inputs()
    helper(q, k, v, mask, 0.0, False, None, H // H_KV)
    assert calls == [dict(enable_gqa = True, kv_heads = H_KV, q_heads = H)], calls


def test_cudnn_gqa_matches_repeated_cudnn(cudnn_gqa):
    helper, _ = _counting_helper()
    q, k, v, mask = _inputs()
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        ref = _run(lambda a, b, c: _ORIGINAL_SDPA(a, _repeat(b), _repeat(c), attn_mask = mask), q, k, v)
    out = _run(lambda a, b, c: helper(a, b, c, mask, 0.0, False, None, H // H_KV), q, k, v)
    # Same cuDNN kernel on the same values: forward and dq are bitwise equal.
    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[1], ref[1])
    # dk / dv: cuDNN sums the group in its kernel, autograd sums the repeat in bf16. Check both against fp64.
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    with sdpa_kernel([SDPBackend.MATH]):
        gold = _run(lambda a, b, c: _ORIGINAL_SDPA(a, b, c, attn_mask = mask.double(), enable_gqa = True), q64, k64, v64)
    for i in (2, 3):
        err_out = (out[i].double() - gold[i]).norm() / gold[i].norm()
        err_ref = (ref[i].double() - gold[i]).norm() / gold[i].norm()
        assert err_out < 1e-2 and err_out <= 1.5 * err_ref, (i, float(err_out), float(err_ref))


def test_repeat_kept_when_cudnn_disabled():
    if sdpa_gqa._efficient_takes(*_inputs()[:3], _inputs()[3], 0.0, False, True):
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    helper, calls = _counting_helper()
    q, k, v, mask = _inputs()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        helper(q, k, v, mask, 0.0, False, None, H // H_KV)
    assert calls == [dict(enable_gqa = False, kv_heads = H, q_heads = H)], calls


def test_repeat_kept_when_cudnn_gate_does_not_check_gqa(monkeypatch, cudnn_sdp_on):
    # torch 2.5 - 2.7: the cuDNN gate does not check head counts, so keep repeating there.
    if sdpa_gqa._efficient_takes(*_inputs()[:3], _inputs()[3], 0.0, False, True):
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    monkeypatch.setattr(sdpa_gqa, "_CUDNN_GATE_CHECKS_GQA", False, raising = False)
    helper, calls = _counting_helper()
    q, k, v, mask = _inputs()
    helper(q, k, v, mask, 0.0, False, None, H // H_KV)
    assert calls == [dict(enable_gqa = False, kv_heads = H, q_heads = H)], calls


def test_float32_still_repeats(cudnn_sdp_on):
    # cuDNN rejects fp32, so the #1942 float32 path keeps its repeat.
    if sdpa_gqa._efficient_takes(*_inputs(torch.float32)[:3], _inputs(torch.float32)[3], 0.0, False, True):
        pytest.skip("the mem-efficient kernel takes enable_gqa on this torch")
    helper, calls = _counting_helper()
    q, k, v, mask = _inputs(torch.float32)
    helper(q, k, v, mask, 0.0, False, None, H // H_KV)
    assert calls == [dict(enable_gqa = False, kv_heads = H, q_heads = H)], calls


def _saved_bytes(fn, q, k, v):
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(q, k, v)
    return sum(saved), out


def test_cudnn_gqa_retains_no_repeated_kv(cudnn_gqa):
    # Autograd must not keep [B, H, S, D] copies of k / v alive until backward.
    helper, _ = _counting_helper()
    q, k, v, mask = _inputs()
    native, _ = _saved_bytes(lambda a, b, c: _ORIGINAL_SDPA(a, b, c, attn_mask = mask, enable_gqa = True), q, k, v)
    mine, _ = _saved_bytes(lambda a, b, c: helper(a, b, c, mask, 0.0, False, None, H // H_KV), q, k, v)
    assert mine == native, (mine, native, 2 * B * H * SEQ * HEAD_DIM * 2)


def test_cudnn_probe_runs_on_the_operands_device(monkeypatch, cudnn_sdp_on):
    # The cuDNN choice reads the current device, so it must run under the operands' device.
    seen = []
    real_device = torch.cuda.device

    class recording_device(real_device):
        def __init__(self, device):
            seen.append(torch.device(device))
            super().__init__(device)

    monkeypatch.setattr(torch.cuda, "device", recording_device)
    q, k, v, mask = _inputs()
    sdpa_gqa._cudnn_takes_gqa(q, k, v, mask, 0.0, False, None)
    assert seen.count(q.device) >= 1, seen
