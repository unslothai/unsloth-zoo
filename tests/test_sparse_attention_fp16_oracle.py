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

"""fp64-oracle precision gate for the sparse MQA Triton backward in fp16, Triton forced (so it also runs on sm75,
where dispatch never picks it). The backward follows FlashAttention's rounding design: delta = sum(dO * O) from the
saved fp16 output, P and dS rounded to fp16 before the dots, fp32 accumulation. Each gradient's max and mean error
vs fp64 is bounded against

  * the same design written in torch (_designed_grads): any extra rounding in the kernel (an fp16 accumulator, a
    lost fp32 dS, a wrong delta) raises the kernel's error above it;
  * the fp32-math torch reference (FlashAttention's own test bound is 2x the PyTorch error; we allow 2.5x).

Measured on T4 (sm75), L4 (sm89) and B200 (sm100), 5 seeds x 3 lengths x head_dim 64 / 128 / 256 / 512: kernel /
design error is 0.84-1.09 (max) for dq and dkv, dq / fp32-torch 1.0-2.0 (max), dkv is 4-10x below the fp32-torch
arm (which rounds every gathered row's gradient to fp16), dsink relative error <= 8e-4. The mean ratio is
0.9989-1.0006 over these cases on T4 and B200, so its bound is 1.01: fp16 logits (1.015-1.025) and an fp16
accumulator fail every case; delta rounded to fp16 (1.001-1.012) is below what the gate resolves.
"""

import pytest
import torch

from unsloth_zoo import sparse_attention as sa

# NVIDIA CUDA only: the forced index kernel skips its own HIP exclusion (sparse_mqa_supported), and a CPU wheel under
# the consolidated is_available() spoof has no torch.version.cuda.
_CUDA = torch.cuda.is_available() and torch.version.cuda is not None and torch.version.hip is None
_F16_EPS = torch.finfo(torch.float16).eps


def _inputs(b, s, h, d, n, k, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(b, s, h, d, generator = g) * 0.5).cuda().half()
    kv = (torch.randn(b, n, d, generator = g) * 0.5).cuda().half()
    sink = torch.randn(h, generator = g).cuda()
    idxs = torch.stack([torch.stack([torch.randperm(n, generator = g)[:k] for _ in range(s)]) for _ in range(b)])
    idxs = torch.where(torch.rand(b, s, k, generator = g) < 0.2, -1, idxs)
    idxs[0, 0] = -1   # an empty row
    do = torch.randn(b, s, h, d, generator = g).cuda().half()
    return q, kv, sink, idxs.to(torch.int32).cuda(), do


def _grads(fn, q, kv, sink, do):
    """(dq, dkv, dsink), and the forward output those gradients were taken through."""
    q, kv, sink = (t.detach().clone().requires_grad_() for t in (q, kv, sink))
    out = fn(q, kv, sink)
    return torch.autograd.grad(out, (q, kv, sink), do), out.detach()


def _designed_grads(q, kv, sink, idxs, scale, do, out16):
    """The kernel's backward rounding design in torch: fp32 logits / lse, delta from the fp16 output, P and dS
    rounded to fp16 before the dots, fp32 accumulation, one final cast."""
    b, s, h, d = q.shape
    n = kv.shape[1]
    kv_pad = torch.cat([kv, kv.new_zeros((b, 1, d))], 1)
    valid = (idxs >= 0) & (idxs < n)
    safe = torch.where(valid, idxs, n).long()
    bi = torch.arange(b, device = q.device).view(b, 1, 1).expand_as(safe)
    slots = kv_pad[bi, safe].float()
    logits = torch.einsum("bshd,bskd->bshk", q.float(), slots) * scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(torch.cat([logits, sink.float().view(1, 1, h, 1).expand(b, s, h, 1)], -1), -1, keepdim = True)
    lse = torch.where(~valid.any(-1).view(b, s, 1, 1), float("inf"), lse)
    p = torch.exp(logits - lse).masked_fill(~valid.unsqueeze(2), 0.0)
    delta = (do.float() * out16.float()).sum(-1, keepdim = True)
    ds = (p * (torch.einsum("bshd,bskd->bshk", do.float(), slots) - delta)).half().float()
    dq = (torch.einsum("bshk,bskd->bshd", ds, slots) * scale).half()
    dslots = torch.einsum("bshk,bshd->bskd", p.half().float(), do.float()) + torch.einsum("bshk,bshd->bskd", ds, q.float()) * scale
    dkv = torch.zeros(b, n + 1, d, device = q.device).index_put_((bi, safe), dslots, accumulate = True)[:, :n].half()
    dsink = -(torch.exp(sink.float().view(1, 1, h, 1) - lse) * delta).sum((0, 1)).view(h)
    return dq, dkv, dsink


def _err(x, r):
    e = (x.double() - r).abs()
    return e.max().item(), e.mean().item()


def _kernel_module():
    pytest.importorskip("triton")
    from unsloth_zoo import sparse_mqa_kernel as mk
    return mk


@pytest.mark.skipif(not _CUDA, reason = "NVIDIA CUDA Triton kernel")
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shape", [(1, 128, 16, 64, 192, 48), (1, 128, 16, 128, 192, 48), (1, 128, 16, 256, 192, 48),
                                   (1, 131, 8, 256, 140, 33), (1, 128, 16, 512, 192, 48), (1, 512, 16, 256, 768, 192)])
def test_triton_fp16_backward_precision_vs_fp64(monkeypatch, shape, seed):
    _check_backward_precision(monkeypatch, _kernel_module(), shape, seed)


@pytest.mark.skipif(not _CUDA, reason = "NVIDIA CUDA Triton kernel")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_triton_fp16_backward_precision_vs_fp64_split_d(monkeypatch, seed):
    # Parts with >= 96 KB of shared memory launch head_dim 512 unsplit; force the 64 KB (T4) split-D backward here.
    mk = _kernel_module()
    configs = mk._configs
    monkeypatch.setattr(mk, "_configs", lambda device, d, backward: configs(device, d, backward)[-1:])
    monkeypatch.setattr(mk, "_CHOSEN", {})
    assert configs(torch.device("cuda"), 512, True)[-1][3] == 256
    _check_backward_precision(monkeypatch, mk, (1, 128, 16, 512, 192, 48), seed)


def _check_backward_precision(monkeypatch, mk, shape, seed):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "triton")
    # Both torch arms are fp32 / fp64 einsums: TF32 left on by another test would move the bounds. Saved and
    # restored by hand: monkeypatch cannot undo an attribute of torch's cuBLASModule.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _check_backward_precision_fp32(mk, shape, seed)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _check_backward_precision_fp32(mk, shape, seed):
    b, s, h, d, n, k = shape
    q, kv, sink, idxs, do = _inputs(b, s, h, d, n, k, 1000 * seed + d + s)
    scale = d ** -0.5
    before, calls = dict(sa.SPARSE_ATTN_STATS), mk.SPARSE_MQA_STATS["bwd"]
    got, out16 = _grads(lambda q, kv, sk: sa.sparse_attn(q, kv, sk, idxs, scale), q, kv, sink, do)
    torch.cuda.synchronize()
    # The Triton backward ran (no silent fallback to the torch path).
    assert sa.SPARSE_ATTN_STATS["triton"] == before["triton"] + 1 and sa.SPARSE_ATTN_STATS["fallback"] == before["fallback"]
    assert mk.SPARSE_MQA_STATS["bwd"] == calls + 1
    ref, _ = _grads(lambda q, kv, sk: sa.sparse_attn_reference(q, kv, sk, idxs, scale), q.double(), kv.double(), sink.double(), do.double())
    fp32_math, _ = _grads(lambda q, kv, sk: sa.sparse_attn_reference(q, kv, sk, idxs, scale), q, kv, sink, do)
    designed = _designed_grads(q, kv, sink, idxs, scale, do, out16)
    for name, x, r, des, f32 in zip(("dq", "dkv", "dsink"), got, ref, designed, fp32_math):
        e_max, e_mean = _err(x, r)
        d_max, d_mean = _err(des, r)
        t_max, _ = _err(f32, r)
        scale_r = r.abs().max().item()
        msg = (f"d{d} s{s} seed{seed} {name}: kernel max {e_max:.3e} mean {e_mean:.3e} | designed max {d_max:.3e} "
               f"mean {d_mean:.3e} | fp32-torch max {t_max:.3e} | |ref| {scale_r:.3e}")
        if name == "dsink":
            # dsink = -sum p_sink * delta is reduced in torch; delta comes from the fp16 output by design, so its
            # error is fp16-level relative to |dsink| (measured <= 8e-4 relative).
            assert e_max <= 2 * _F16_EPS * max(scale_r, 1.0), msg
            continue
        assert e_max <= 1.25 * d_max, msg
        # A whole-tensor mean is stable to ~1e-3 (module docstring), so 1.01 catches fp16 logits, which 1.05 let through.
        assert e_mean <= 1.01 * d_mean, msg
        assert e_max <= 2.5 * t_max, msg
