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

"""The BF16 dequantized FP8 expert banks are released between forward and backward, except when the
forward is a gradient checkpointing recompute (it runs inside backward, and this layer's backward follows
at once), where releasing would only re-dequantize the banks for nothing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.utils.checkpoint as tuc

import unsloth_zoo.temporary_patches.moe_utils_fp8 as M

E, N, K = 2, 8, 4


@pytest.fixture
def bank(monkeypatch):
    refills = []

    def fake_dequant(base, quant, dtype, quant_kind = None):
        refills.append(base.shape[0])
        return base.to(dtype) * quant
    monkeypatch.setattr(M, "_dequantize_full_expert_weights", fake_dequant)
    torch.manual_seed(0)
    base = torch.randn(E, N, K).to(torch.float8_e4m3fn)
    quant = torch.tensor(0.5)
    return base, quant, refills


def _layer(x, base, quant, nbytes_after_release):
    dequantized = (base.to(torch.bfloat16) * quant).contiguous()
    out = torch.einsum("tk,enk->tn", x, dequantized)
    M._release_dequantized_banks_until_backward(out, x, ((dequantized, base, quant, None),))
    nbytes_after_release.append(dequantized.untyped_storage().nbytes())
    return out


def test_plain_forward_releases_and_refills_for_backward(bank):
    base, quant, refills = bank
    x = torch.randn(3, K, dtype = torch.bfloat16, requires_grad = True)
    seen = []
    out = _layer(x, base, quant, seen)
    assert seen == [0]  # released after forward
    out.float().square().sum().backward()
    assert sum(refills) == E  # the whole bank refilled once for this backward
    ref_x = x.detach().clone().requires_grad_(True)
    torch.einsum("tk,enk->tn", ref_x, (base.to(torch.bfloat16) * quant)).float().square().sum().backward()
    assert torch.equal(x.grad, ref_x.grad)


def test_reentrant_checkpoint_recompute_keeps_the_bank(bank):
    base, quant, refills = bank
    x = torch.randn(3, K, dtype = torch.bfloat16, requires_grad = True)
    seen = []
    out = tuc.checkpoint(lambda h: _layer(h, base, quant, seen), x, use_reentrant = True)
    out.float().square().sum().backward()
    # First forward under no_grad, recompute inside backward: neither releases, nothing is re-dequantized.
    assert len(seen) == 2 and all(nbytes > 0 for nbytes in seen), seen
    assert refills == []
    ref_x = x.detach().clone().requires_grad_(True)
    torch.einsum("tk,enk->tn", ref_x, (base.to(torch.bfloat16) * quant)).float().square().sum().backward()
    assert torch.equal(x.grad, ref_x.grad)


def test_refill_keeps_the_block_size_of_a_non_aligned_bank():
    # H = 192 is not a multiple of 128: a sliced bank without block_size would infer 96-wide blocks.
    torch.manual_seed(0)
    experts, rows, hidden = 2, 256, 192
    base = torch.randn(experts, rows, hidden).to(torch.float8_e4m3fn)
    scale = torch.rand(experts, 2, 2) + 0.5
    base.block_size = scale.block_size = (128, 128)
    dequantized = M._dequantize_full_expert_weights(base, scale, torch.float32, quant_kind = "weight_scale_inv")
    reference = dequantized.clone()
    x = torch.randn(3, hidden, requires_grad = True)
    # F.linear saves the bank's own storage (a view), so the input gradient reads the refilled values.
    out = torch.nn.functional.linear(x, dequantized.view(-1, hidden))
    M._release_dequantized_banks_until_backward(out, x, ((dequantized, base, scale, "weight_scale_inv"),))
    out.square().sum().backward()
    ref_x = x.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(ref_x, reference.view(-1, hidden)).square().sum().backward()
    assert torch.equal(x.grad, ref_x.grad)


def test_refilled_bank_is_released_when_the_expert_inputs_are_frozen(bank):
    base, quant, refills = bank
    x = torch.randn(3, K, dtype = torch.bfloat16)  # frozen inputs: no input hook to release on
    scale = torch.ones((), dtype = torch.bfloat16, requires_grad = True)
    dequantized = (base.to(torch.bfloat16) * quant).contiguous()
    out = torch.nn.functional.linear(x * scale, dequantized.view(-1, K))
    M._release_dequantized_banks_until_backward(out, x, ((dequantized, base, quant, None),))
    out.float().square().sum().backward()
    assert sum(refills) == E and scale.grad is not None
    assert dequantized.untyped_storage().nbytes() == 0  # freed at the end of backward, not kept with the graph


def test_non_reentrant_checkpoint_does_not_refill_the_discarded_bank(bank):
    base, quant, refills = bank
    x = torch.randn(3, K, dtype = torch.bfloat16, requires_grad = True)
    seen = []
    out = tuc.checkpoint(lambda h: _layer(h, base, quant, seen), x, use_reentrant = False)
    out.float().square().sum().backward()
    # The first forward under checkpoint hooks keeps its bank (the recompute may stop before this hook).
    assert seen and all(nbytes > 0 for nbytes in seen), seen
    assert refills == []
    ref_x = x.detach().clone().requires_grad_(True)
    torch.einsum("tk,enk->tn", ref_x, (base.to(torch.bfloat16) * quant)).float().square().sum().backward()
    assert torch.equal(x.grad, ref_x.grad)


def test_bank_is_kept_when_checkpoint_hooks_cannot_be_inspected(bank, monkeypatch):
    # torch 2.6 has no _top_saved_tensors_default_hooks: a non-reentrant checkpoint would be invisible,
    # so the bank must stay (no release, no refill) rather than risk holding two.
    base, quant, refills = bank
    monkeypatch.delattr(torch._C._autograd, "_top_saved_tensors_default_hooks", raising = False)
    x = torch.randn(3, K, dtype = torch.bfloat16, requires_grad = True)
    seen = []
    out = _layer(x, base, quant, seen)
    assert seen == [E * N * K * 2]  # bf16 bank still allocated
    out.float().square().sum().backward()
    assert refills == []


@pytest.mark.parametrize("scale_shape", [(16, 1), (16, N)])
def test_refill_slices_per_expert_2d_scales(scale_shape):
    # Real dequantizer, 16 experts so the refill runs in 8 chunks; (E, 1) / (E, M) scales must be
    # chunked with the weight or the dequantizer sees a short weight with all E scale rows.
    torch.manual_seed(0)
    base = torch.randn(16, N, K).to(torch.float8_e4m3fn)
    scale = torch.rand(*scale_shape) + 0.5
    want = M._dequantize_full_expert_weights(base, scale, torch.bfloat16)
    assert want is not None
    dequantized = want.clone().contiguous()
    x = torch.randn(3, K, dtype = torch.bfloat16, requires_grad = True)
    out = torch.einsum("tk,enk->tn", x, dequantized)
    M._release_dequantized_banks_until_backward(out, x, ((dequantized, base, scale, None),))
    assert dequantized.untyped_storage().nbytes() == 0
    out.float().square().sum().backward()  # the refill runs here
    ref_x = x.detach().clone().requires_grad_(True)
    torch.einsum("tk,enk->tn", ref_x, want).float().square().sum().backward()
    assert torch.equal(x.grad, ref_x.grad)


def test_refilled_bank_is_released_when_autograd_prunes_the_input_gradient(bank):
    # The output node also depends on routing weights r. Asking only for r's gradient runs that node
    # (so the bank is refilled) but prunes the x branch, so x's release hook never fires.
    base, quant, refills = bank
    x = torch.randn(3, K, dtype = torch.bfloat16, requires_grad = True)
    r = torch.rand(3, N, dtype = torch.bfloat16, requires_grad = True)
    dequantized = (base.to(torch.bfloat16) * quant).contiguous()
    out = torch.einsum("tk,enk->tn", x, dequantized) * r
    M._release_dequantized_banks_until_backward(out, x, ((dequantized, base, quant, None),))
    (gr,) = torch.autograd.grad(out.float().square().sum(), [r])
    assert sum(refills) == E and gr is not None
    assert dequantized.untyped_storage().nbytes() == 0  # released at the end of that backward
