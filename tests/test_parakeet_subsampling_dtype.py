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

"""The FastConformer subsampling (Parakeet, Canary) must not run its frozen conv stack on float32
activations under Unsloth's conv dtype wrapper: same outputs, half the activation memory."""

import pytest
import torch

modeling_parakeet = pytest.importorskip("transformers.models.parakeet.modeling_parakeet")

from unsloth_zoo.compiler import _dtype_safe_forward
from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


def _subsampling():
    config = modeling_parakeet.ParakeetEncoderConfig(
        hidden_size = 64, num_mel_bins = 80, subsampling_conv_channels = 16, subsampling_factor = 8,
        num_hidden_layers = 1, num_attention_heads = 4, intermediate_size = 128,
    )
    torch.manual_seed(0)
    return modeling_parakeet.ParakeetEncoderSubsamplingConv2D(config).to(torch.bfloat16)


class _Holder(torch.nn.Module):
    def __init__(self, subsampling):
        super().__init__()
        self.subsampling = subsampling


def _hook(module):
    for param in module.parameters():  # frozen, as under LoRA
        param.requires_grad_(False)
    requires_grad_for_gradient_checkpointing(_Holder(module))


def _run(module, features, mask):
    dtypes = []
    hooks = [
        layer.register_forward_hook(lambda m, args, out: dtypes.append(out.dtype))
        for layer in module.layers if isinstance(layer, torch.nn.Conv2d)
    ]
    try:
        # training runs under autocast, which casts the subsampling's final Linear input
        with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
            out = module(features, mask)
    finally:
        for hook in hooks:
            hook.remove()
    return out, dtypes


@pytest.fixture
def unsloth_conv_wrapper(monkeypatch):
    # what _patch_torch_dtype_modules installs on torch.nn.Conv2d
    monkeypatch.setattr(
        torch.nn.Conv2d, "forward", _dtype_safe_forward(torch.nn.Conv2d.forward, True, False)
    )


def test_subsampling_runs_16_bit_and_matches_bitwise(unsloth_conv_wrapper):
    module = _subsampling()
    features = torch.randn(2, 64, 80)
    mask = torch.ones(2, 64, dtype = torch.bool)
    mask[1, 40:] = False
    expected, before = _run(module, features, mask)
    _hook(module)
    _hook(module)  # idempotent: one hook, not two
    assert len(module._forward_pre_hooks) == 1
    got, after = _run(module, features, mask)
    assert set(before) == {torch.float32}
    assert set(after) == {torch.bfloat16}
    assert got.dtype == expected.dtype
    assert torch.equal(got, expected)
    # keyword call path
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
        assert torch.equal(module(input_features = features, attention_mask = mask), expected)


def _first_conv_input_dtype(module):
    seen = []
    module.layers[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].dtype))
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
        module(torch.randn(1, 32, 80), None)
    return seen


def test_float32_weights_are_left_alone():
    module = _subsampling().float()
    _hook(module)
    assert _first_conv_input_dtype(module) == [torch.float32]


def test_quantized_closing_linear_keeps_float32():
    # bitsandbytes' Linear4bit computes in its input's dtype: a float32 input there must stay float32
    module = _subsampling()
    module.linear.weight.quant_state = object()
    _hook(module)
    assert _first_conv_input_dtype(module) == [torch.float32]
