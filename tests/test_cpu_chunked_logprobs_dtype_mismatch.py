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

"""CPU chunked log-probs with autocast off and an LM head whose dtype differs from the hidden states (fp32
trainable lm_head under bf16 hidden, or bf16=False eval of a bf16 head): computed in the hidden dtype, as
TRL's fused-head kernel does, instead of failing the matmul on mixed dtypes."""

import pytest
import torch


def _reference(hidden, weight, targets):
    logits = hidden.to(hidden.dtype) @ weight.to(hidden.dtype).t()
    return torch.log_softmax(logits.double(), dim = -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("hidden_dtype, weight_dtype", [
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float32),
])
@pytest.mark.parametrize("hidden_grad, head_grad", [(False, False), (True, False), (True, True)])
def test_mixed_hidden_weight_dtype(hidden_dtype, weight_dtype, hidden_grad, head_grad):
    from unsloth_zoo.cpu.ops import cpu_chunked_log_probs
    torch.manual_seed(0)
    n, h, v = 6, 16, 32
    hidden = torch.randn(n, h).to(hidden_dtype).requires_grad_(hidden_grad)
    weight = torch.randn(v, h).to(weight_dtype).requires_grad_(head_grad)
    bias = torch.randn(v).to(weight_dtype)
    targets = torch.randint(0, v, (n,))
    out = cpu_chunked_log_probs(hidden, weight, None, targets, 1.0, None)
    log_probs, entropy = out[0], out[1]
    assert log_probs.dtype == torch.float32 and log_probs.shape == (n,)
    assert entropy is not None and torch.isfinite(entropy).all()
    torch.testing.assert_close(log_probs.double(), _reference(hidden.detach(), weight.detach(), targets),
                               atol = 3e-2, rtol = 0)
    if hidden_grad or head_grad:
        log_probs.sum().backward()
        if hidden_grad:
            assert hidden.grad is not None and hidden.grad.dtype == hidden_dtype
            assert torch.isfinite(hidden.grad.float()).all()
        if head_grad:
            assert weight.grad is not None and weight.grad.dtype == weight_dtype
            assert torch.isfinite(weight.grad.float()).all()
    # A bias in the head's dtype is cast the same way.
    with torch.no_grad():
        out_b = cpu_chunked_log_probs(hidden, weight, bias, targets, 1.0, None)[0]
        ref_b = torch.log_softmax(
            (hidden @ weight.to(hidden_dtype).t()).double() + bias.to(hidden_dtype).double(), dim = -1
        ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(out_b.double(), ref_b, atol = 3e-2, rtol = 0)
