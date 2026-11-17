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

"""Fused CE on a TorchAO QAT lm_head must match stock lm_head.forward() loss and grads (CPU)."""

import inspect
import os
import textwrap

import pytest
import torch

qat = pytest.importorskip("torchao.quantization.qat")
gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
from torchao.quantization import quantize_  # noqa: E402
from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402
from unsloth_zoo import compiler  # noqa: E402
from unsloth_zoo.fused_losses import unsloth_fused_ce_loss, unsloth_fused_lm_head_loss  # noqa: E402
from unsloth_zoo.fused_losses.forward_install import _can_fuse_loss  # noqa: E402

V, H = 64, 32


def _base_config(scheme):
    from torchao.quantization import Int4WeightOnlyConfig, Int8DynamicActivationIntxWeightConfig
    from torchao.quantization.granularity import PerGroup

    if scheme == "int8-int4":
        return Int8DynamicActivationIntxWeightConfig(weight_dtype = torch.int4, weight_granularity = PerGroup(32))
    return Int4WeightOnlyConfig(group_size = 32)


def _prepare_head(model, scheme):
    quantize_(model, qat.QATConfig(_base_config(scheme), step = "prepare"), filter_fn = lambda m, fqn: fqn == "lm_head")
    assert type(model.lm_head).__name__ == "FakeQuantizedLinear"
    if scheme == "int8-int4":
        assert model.lm_head.activation_fake_quantizer is not None
    assert model.lm_head.weight_fake_quantizer is not None


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(V, H)
        self.proj = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias = False)

    def hidden(self, ids):
        return self.proj(self.embed(ids))


def _labels():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (2, 9), generator = g)
    return ids


@pytest.mark.parametrize("scheme", ["int8-int4", "int4"])
@pytest.mark.parametrize("train_head", [False, True])
def test_adapter_matches_qat_forward(scheme, train_head, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "1")
    torch.manual_seed(0)
    model = Toy()
    _prepare_head(model, scheme)
    model.lm_head.weight.requires_grad_(train_head)
    ids = _labels()
    params = [p for p in model.parameters() if p.requires_grad]

    expected = ForCausalLMLoss(model.lm_head(model.hidden(ids)).float(), ids, vocab_size = V)
    expected_grads = torch.autograd.grad(expected, params)
    actual = unsloth_fused_lm_head_loss(model.hidden(ids), model.lm_head, ids, vocab_size = V, torch_compile = False)
    actual_grads = torch.autograd.grad(actual, params)

    torch.testing.assert_close(actual, expected)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol = 1e-4, atol = 1e-5)


def _fused_gemma3_forward():
    source = inspect.getsource(gemma3.Gemma3ForCausalLM.forward)
    source = compiler.fixup_fused_lm_head(source)
    source, _ = compiler.apply_fused_lm_head(source, "Gemma3ForCausalLM")
    assert "unsloth_fused_ce_loss" in source
    from _compiled_module_names import fused_loss_header_names

    scope = dict(vars(gemma3))
    scope.update(fused_loss_header_names())
    scope.update(
        os = os,
        UNSLOTH_ENABLE_CCE = False,
        HAS_CUT_CROSS_ENTROPY = False,
        UNSLOTH_COMPILE_DISABLE = True,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss,
        _can_fuse_loss = _can_fuse_loss,
        unsloth_training_stance = lambda: None,
        EMPTY_LOGITS = None,
    )
    source = textwrap.dedent(source)
    exec(source[source.index("def forward"):], scope)
    return scope["forward"]


@pytest.mark.parametrize("scheme", ["int8-int4", "int4"])
def test_compiled_forward_matches_qat_forward(scheme, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    config = gemma3.Gemma3TextConfig(
        vocab_size = V, hidden_size = H, intermediate_size = 64, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
        final_logit_softcapping = None, attn_implementation = "eager",
    )
    torch.manual_seed(0)
    model = gemma3.Gemma3ForCausalLM(config).float().train()
    model.loss_function = ForCausalLMLoss
    _prepare_head(model, scheme)
    ids = _labels()
    params = [p for p in model.parameters() if p.requires_grad]

    expected = model(input_ids = ids, labels = ids).loss
    expected_grads = torch.autograd.grad(expected, params)
    actual = _fused_gemma3_forward()(model, input_ids = ids, labels = ids).loss
    actual_grads = torch.autograd.grad(actual, params)

    torch.testing.assert_close(actual, expected)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol = 1e-4, atol = 1e-5)
