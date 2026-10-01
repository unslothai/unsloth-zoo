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

"""The fused lm_head rewrite of a loss_function forward must use caller shift_labels like stock (CPU)."""

import inspect
import os
import textwrap

import pytest
import torch

gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402
from unsloth_zoo import compiler  # noqa: E402
from unsloth_zoo.fused_losses import unsloth_fused_ce_loss  # noqa: E402


def _fused_forward():
    source = inspect.getsource(gemma3.Gemma3ForCausalLM.forward)
    source = compiler.fixup_fused_lm_head(source)
    source, _ = compiler.apply_fused_lm_head(source, "Gemma3ForCausalLM")
    assert "unsloth_fused_ce_loss" in source
    scope = dict(vars(gemma3))
    scope.update(
        os = os,
        UNSLOTH_ENABLE_CCE = False,
        HAS_CUT_CROSS_ENTROPY = False,
        UNSLOTH_COMPILE_DISABLE = True,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss,
        EMPTY_LOGITS = None,
    )
    source = textwrap.dedent(source)
    source = source[source.index("def forward"):]  # drop decorators
    exec(source, scope)
    return scope["forward"]


@pytest.fixture
def model():
    config = gemma3.Gemma3TextConfig(
        vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 8,
        final_logit_softcapping = None, attn_implementation = "eager",
    )
    torch.manual_seed(0)
    model = gemma3.Gemma3ForCausalLM(config).float().train()
    model.loss_function = ForCausalLMLoss
    return model


@pytest.mark.parametrize("explicit", [False, True])
def test_fused_forward_matches_stock_loss_and_grads(model, explicit, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 64, (2, 9), generator = g)
    kwargs = {}
    if explicit:
        shift = torch.randint(0, 64, (2, 9), generator = g)
        shift[:, -1] = -100
        shift[0, :3] = -100
        kwargs["shift_labels"] = shift
    fused = _fused_forward()
    params = [p for p in model.parameters() if p.requires_grad]
    expected = model(input_ids = ids, labels = ids, **kwargs).loss
    expected_grads = torch.autograd.grad(expected, params)
    actual = fused(model, input_ids = ids, labels = ids, **kwargs).loss
    actual_grads = torch.autograd.grad(actual, params)
    torch.testing.assert_close(actual, expected)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol = 1e-4, atol = 1e-5)
