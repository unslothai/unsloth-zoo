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


def test_corrupted_fixup_store_is_caught():
    W, A, B, scale = _dense(256, 1024, 16, 1e-2)
    t = _target(W, A, B, scale)
    stats = ex.exact_fold([t])
    assert stats["bad"] > 0
    t.host_val[0] = t.host_val[0] + 1   # flip: a wrong original value must not pass silently
    with pytest.raises(RuntimeError, match = "checksum mismatch"):
        ex.exact_unfold([t])


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
