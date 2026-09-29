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

import pytest
import torch

from unsloth_zoo import compiler

# The forced-float32 template before the fix: it rounded `result` to float16 first.
OLD_FORCED_FLOAT32 = """
torch_addmm = torch.addmm
torch_add   = torch.add
torch_float16 = torch.float16
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    xA = dropout(x.to(torch_float16)) @ lora_A.weight.to(torch_float16).t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]).to(torch_float16),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.to(torch_float16).t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)
    bias = lora_B.bias
    if bias is not None:
        output = torch_add(output, bias.to(torch_float16), alpha = scaling)
    return output
"""


def _load(source):
    namespace = {"torch": torch}
    exec(source, namespace)
    return namespace["lora_forward"]


def _adapter(in_features, out_features, rank, bias):
    torch.manual_seed(3407)
    lora_A = torch.nn.Linear(in_features, rank, bias = False)
    lora_B = torch.nn.Linear(rank, out_features, bias = bias)
    with torch.no_grad():
        lora_A.weight.mul_(0.1)
        lora_B.weight.normal_(0, 0.01)
        if bias:
            lora_B.bias.normal_(0, 0.01)
    return lora_A, lora_B


def _half_matmul_supported():
    try:
        torch.ones(2, 2, dtype = torch.float16) @ torch.ones(2, 2, dtype = torch.float16)
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _half_matmul_supported(), reason = "float16 matmul unsupported on this CPU build")


@pytest.mark.parametrize("bias", [False, True])
def test_a_float32_base_result_past_float16_max_stays_finite(bias):
    # gpt-oss-20b bnb-4bit: an expert down_proj computes in float32 and measured 66794.9
    # for one token, past float16's 65504.
    lora_forward = _load(compiler.COMPILED_LORA_FORWARD_forced_float32)
    lora_A, lora_B = _adapter(64, 32, 8, bias)
    x = torch.randn(5, 64, dtype = torch.float32)
    result = torch.randn(5, 32, dtype = torch.float32)
    result[2, 7] = 66794.890625

    out = lora_forward(result, lora_A, lora_B, torch.nn.Identity(), x, 2.0)

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    xA = x.half().float() @ lora_A.weight.half().float().t()
    expected = result + 2.0 * (xA.half().float() @ lora_B.weight.t())
    if bias:
        expected = expected + 2.0 * lora_B.bias
    torch.testing.assert_close(out, expected, rtol = 1e-3, atol = 1e-3)
    # The old template overflowed this exact element.
    old = _load(OLD_FORCED_FLOAT32)(result, lora_A, lora_B, torch.nn.Identity(), x, 2.0).to(result.dtype)
    assert torch.isinf(old[2, 7])


@pytest.mark.parametrize("bias", [False, True])
def test_a_float16_base_result_is_bit_identical_to_before(bias):
    new = _load(compiler.COMPILED_LORA_FORWARD_forced_float32)
    old = _load(OLD_FORCED_FLOAT32)
    lora_A, lora_B = _adapter(64, 32, 8, bias)
    x = torch.randn(5, 64, dtype = torch.float16)
    result = torch.randn(5, 32, dtype = torch.float16)

    a = new(result, lora_A, lora_B, torch.nn.Identity(), x, 2.0)
    b = old(result, lora_A, lora_B, torch.nn.Identity(), x, 2.0)

    assert a.dtype == b.dtype == torch.float16
    assert torch.equal(a, b)
