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

"""Exact fold / unfold kernel (merged_rollout_exact), CUDA only."""

import pytest
import torch

from unsloth_zoo import merged_rollout_exact as ex

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or ex.triton is None, reason = "needs CUDA + Triton")
DEV = "cuda"


def _bits(t):
    return t.view(torch.int16)


def _dense(N, K, R, rel, dtype = torch.bfloat16, seed = 0):
    g = torch.Generator(device = DEV).manual_seed(seed)
    W = (torch.randn(N, K, device = DEV, generator = g) * 0.02).to(dtype)
    A = torch.randn(R, K, device = DEV, generator = g).to(dtype)
    B = torch.randn(N, R, device = DEV, generator = g).to(dtype)
    d = (B.float() @ A.float())
    scale = rel * 0.02 * 0.8 / d.abs().mean().item()
    return W, A, B, scale


def _target(W, A, B, scale):
    return ex.ExactFoldTarget(W.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), scale)


def _oracle(W, A, B, scale):
    # One fp32 sum of the fp32 delta, rounded once to W's dtype (what the kernel promises).
    d = (B.double() @ A.double() * scale).float()
    return (W.float() + d).to(W.dtype)


@pytest.mark.parametrize("rel", [1e-4, 1e-3, 1e-2, 1e-1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fold_value_and_bit_exact_restore(rel, dtype):
    W, A, B, scale = _dense(256, 1024, 16, rel, dtype)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    stats = ex.exact_fold([t])
    want = _oracle(W0, A, B, scale)
    # fp32 rank-16 sum vs fp64 reference: equal except rare double-rounding ties, never more than 1 ulp.
    diff = (_bits(W).int() - _bits(want).int()).abs()
    assert diff.max().item() <= 1 and (diff != 0).float().mean().item() < 1e-3
    assert 0 <= stats["bad"] < W.numel()
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0))


def test_many_cycles_with_growing_adapter_stay_bit_exact():
    W, A, B, scale = _dense(512, 2048, 16, 1e-3)
    W0 = W.clone()
    stream = torch.cuda.Stream()
    for step in range(200):
        B.add_(torch.randn_like(B) * 0.05)   # optimizer step stand-in between rollouts
        t = _target(W, A, B, scale * (1 + step / 50))
        ex.exact_fold([t], stream = stream)
        ex.exact_unfold([t], stream = stream)
    assert torch.equal(_bits(W), _bits(W0))


def test_fused_qkv_row_views_leave_other_rows_alone():
    qkv = (torch.randn(96 + 2 * 32, 256, device = DEV) * 0.02).bfloat16()
    before = qkv.clone()
    q = qkv[:96]
    _, A, B, scale = _dense(96, 256, 8, 1e-2)
    t = _target(q, A, B, scale)
    ex.exact_fold([t])
    assert torch.equal(qkv[96:], before[96:]) and not torch.equal(qkv[:96], before[:96])
    ex.exact_unfold([t])
    assert torch.equal(_bits(qkv), _bits(before))


def test_stacked_experts_layout():
    E, N, K, R = 8, 64, 128, 4
    g = torch.Generator(device = DEV).manual_seed(1)
    W = (torch.randn(E, N, K, device = DEV, generator = g) * 0.02).bfloat16()
    A = torch.randn(E, R, K, device = DEV, generator = g).bfloat16()
    B = torch.randn(E, N, R, device = DEV, generator = g).bfloat16()
    scale = 0.01
    W0 = W.clone()
    t = ex.ExactFoldTarget(W, A, B, scale)
    ex.exact_fold([t])
    for e in range(E):
        want = _oracle(W0[e], A[e], B[e], scale)
        assert (_bits(W[e]).int() - _bits(want).int()).abs().max().item() <= 1
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0))


def test_transposed_expert_view():
    # PEFT "e i o" layouts: the kernel's N axis is W's last dim, via strides.
    E, IN, OUT, R = 4, 128, 96, 4
    W = (torch.randn(E, IN, OUT, device = DEV) * 0.02).bfloat16()
    A = torch.randn(E, R, IN, device = DEV).bfloat16()
    B = torch.randn(E, OUT, R, device = DEV).bfloat16()
    W0 = W.clone()
    t = ex.ExactFoldTarget(W.transpose(1, 2), A, B, 0.01)
    ex.exact_fold([t])
    for e in range(E):
        want = _oracle(W0[e].T, A[e], B[e], 0.01)
        assert (_bits(W[e].T.contiguous()).int() - _bits(want).int()).abs().max().item() <= 1
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0))


def test_signed_zero_and_nan_restore_bit_exact():
    W, A, B, scale = _dense(64, 256, 8, 1e-2)
    W[0, :8] = -0.0
    B[0] = 0   # d == +0 on row 0, so the fold turns -0.0 into +0.0, equal by value but not by bits
    W[1, :4] = float("nan")
    W0 = W.clone()
    t = _target(W, A, B, scale)
    ex.exact_fold([t])
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0))


def test_fixup_store_is_small_for_small_deltas():
    W, A, B, scale = _dense(1024, 4096, 16, 1e-3)
    stats = ex.exact_fold([_target(W, A, B, scale)])
    assert stats["bad"] / stats["elements"] < 0.01


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_store_needs_no_positions(dtype):
    # Trained-size adapter (|d| about 10% of |W|): the old store took 6 B per inexact element; the stream takes a few bits.
    # The per-weight choice keeps the stream for it on its own.
    W, A, B, scale = _dense(1024, 4096, 16, 1e-1, dtype)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    stats = ex.exact_fold([t])
    report = ex.fixup_report([t], stats)
    assert not t.store.indexed[t.slot]
    assert stats["bad"] > 0.01 * stats["elements"]
    assert report["host_store_bytes"] == stats["store_bytes"] == t.host_store_bytes()
    assert report["host_store_bytes"] < 0.25 * 6 * stats["bad"]
    # Escapes are what no code can name from (new, d): exponent-gap-0 ties (|c| = 1 while r = 0, so unfold cannot tell
    # which of them moved) and gaps over 7 (tiny weights next to a large delta, |c| up to ~2^15). Measured on this
    # adapter: bf16 1.3-1.4% of the inexact elements (bf16's 8-bit mantissa ties far more often), fp16 0.2%.
    # Widening the code to gap 8 or 15 only removes 17% / 31% of the bf16 escapes, so the codes stay at gap 7.
    assert report["escapes"] < (0.02 if dtype == torch.bfloat16 else 0.005) * stats["bad"]
    # Escapes stay a small part of the store (6 B each against the stream's few bits per inexact element).
    assert report["escapes"] * 6 < 0.2 * report["host_store_bytes"]
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0))


def test_corrupted_fixup_store_is_caught(monkeypatch):
    monkeypatch.setattr(ex, "_INDEXED_PER_TILE", -1)   # the stream form, whatever the per-weight choice would be
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    t = _target(W, A, B, scale)
    stats = ex.exact_fold([t])
    assert stats["bad"] > 0
    nz = (t.host_words != 0).nonzero()
    t.host_words[nz[0, 0]] ^= 1 << 2   # flip: a wrong correction must not pass silently
    with pytest.raises(RuntimeError, match = "checksum mismatch"):
        ex.exact_unfold([t])


@pytest.mark.parametrize("rel, indexed", [(1e-4, True), (1e-3, True), (1e-1, False)])
def test_small_adapters_take_the_indexed_form(rel, indexed):
    # Few inexact elements per coded tile (GRPO-step adapters): the indexed form, cheaper to fold and unfold than the
    # stream there; trained-size adapters keep the stream. Both restore bit for bit, in one call.
    W, A, B, scale = _dense(512, 2048, 16, rel, torch.float16)
    W2, A2, B2, scale2 = _dense(256, 1024, 16, 1e-1, torch.float16, seed = 5)
    W0, W20 = W.clone(), W2.clone()
    t, t2 = _target(W, A, B, scale), _target(W2, A2, B2, scale2)
    stats = ex.exact_fold([t, t2])
    assert t.store.indexed[t.slot] == indexed and not t2.store.indexed[t2.slot]
    if indexed:
        assert t.host_words.numel() == 0 and t.host_idx.numel() == t.bad > 0
    ex.exact_unfold([t2, t])
    assert torch.equal(_bits(W), _bits(W0)) and torch.equal(_bits(W2), _bits(W20))
    assert stats["bad"] == t.bad + t2.bad


def test_changed_adapter_between_fold_and_unfold_is_caught():
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    t = _target(W, A, B, scale)
    ex.exact_fold([t])
    B.add_(0.5)   # flip: d differs at unfold, so the round trip is wrong and must be detected
    with pytest.raises(RuntimeError, match = "checksum mismatch"):
        ex.exact_unfold([t])


def test_self_check_passes_and_is_cached():
    assert ex.exact_merge_available(torch.device("cuda"), torch.bfloat16)
    assert ex.exact_merge_available(torch.device("cuda"), torch.float16)
    assert ex.exact_merge_failure(torch.device("cuda"), torch.bfloat16) is None


def test_failed_self_check_disables_exact(monkeypatch):
    monkeypatch.setattr(ex, "_SELF_CHECK", {})
    monkeypatch.setattr(ex, "_self_check", lambda device, dtype: "unfold is not bit exact")
    assert not ex.exact_merge_available(torch.device("cuda"), torch.bfloat16)
    assert ex.exact_merge_failure(torch.device("cuda"), torch.bfloat16) == "unfold is not bit exact"


def test_disable_pinned_memory_uses_pageable_stores(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    ex.exact_fold([t], stream = torch.cuda.Stream())
    assert t.host_words.numel() > 0 and not t.host_words.is_pinned() and not t.host_wcnt.is_pinned()
    ex.exact_unfold([t], stream = torch.cuda.Stream())
    assert torch.equal(_bits(W), _bits(W0))


def test_pinning_failure_falls_back_to_pageable(monkeypatch):
    # WSL2 caps page-locked memory: the pinned allocation raises, the fold must still restore.
    real_empty = torch.empty
    def empty(*args, pin_memory = False, **kwargs):
        if pin_memory: raise RuntimeError("CUDA error: out of memory")
        return real_empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", empty)
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    ex.exact_fold([t], stream = torch.cuda.Stream())
    assert not t.host_words.is_pinned()
    ex.exact_unfold([t], stream = torch.cuda.Stream())
    assert torch.equal(_bits(W), _bits(W0))


def test_host_allocation_failure_leaves_weights_untouched(monkeypatch):
    def fail(targets, sizes): raise MemoryError("no host RAM")
    monkeypatch.setattr(ex, "_host_buffers", fail)
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    with pytest.raises(MemoryError):
        ex.exact_fold([t])
    assert torch.equal(_bits(W), _bits(W0))
    ex.exact_unfold([t])   # nothing was folded: a no-op, not a checksum failure
    assert torch.equal(_bits(W), _bits(W0))


def test_offload_failure_after_fold_restores_from_device_stores(monkeypatch):
    def fail(targets, stream): raise RuntimeError("copy failed")
    monkeypatch.setattr(ex, "_offload", fail)
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    with pytest.raises(RuntimeError, match = "copy failed"):
        ex.exact_fold([t], stream = torch.cuda.Stream())
    assert not torch.equal(_bits(W), _bits(W0))
    ex.exact_unfold([t], stream = torch.cuda.Stream())
    assert torch.equal(_bits(W), _bits(W0))


def test_unfold_on_another_stream_waits_for_the_offload():
    # The offload's device-to-host copies are held back on their stream; an unfold on the default stream must not read the host stores before they land.
    W, A, B, scale = _dense(1024, 4096, 16, 1e-2)
    W0 = W.clone()
    t = _target(W, A, B, scale)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)
    ex.exact_fold([t], stream = side)
    ex.exact_unfold([t])
    assert torch.equal(_bits(W), _bits(W0))
